"""AgentOrchestrator — ties together identity, memory, and tools into a Strands Agent.

Creates and manages the agent lifecycle: identity loading, memory hydration,
system prompt composition, tool registration, message handling with Bedrock
throttling retry, and session teardown.

Requirements: 7.1, 7.2, 7.3, 12.1, 12.2, 12.3
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from agent.identity import IdentityManager
from agent.memory import MemoryManager, SessionHandle
from agent.model_router import ModelRouter
from agent.models import IdentityBundle, MemoryContext
from agent.tools import ToolRegistry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Retry constants (Req 12.1)
# ---------------------------------------------------------------------------
MAX_THROTTLE_RETRIES = 5
BASE_DELAYS = [1, 2, 4, 8, 16]  # seconds
MAX_JITTER = 1.0  # seconds


# ---------------------------------------------------------------------------
# AgentHandle — holds per-agent session state
# ---------------------------------------------------------------------------
@dataclass
class AgentHandle:
    """Holds all state for a running agent session."""

    agent_id: str
    user_id: str
    identity_bundle: IdentityBundle
    memory_handle: SessionHandle
    system_prompt: str
    tools: list[Callable[..., Any]]
    _strands_agent: Any = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# ThrottlingError — raised when Bedrock returns HTTP 429
# ---------------------------------------------------------------------------
class ThrottlingError(Exception):
    """Raised when Bedrock returns a throttling error (HTTP 429)."""


# ---------------------------------------------------------------------------
# AgentFactory protocol — injectable for testing
# ---------------------------------------------------------------------------
class AgentFactory(Protocol):
    """Protocol for creating a Strands Agent instance."""

    def __call__(
        self,
        *,
        system_prompt: str,
        tools: list[Callable[..., Any]],
        model_id: str,
    ) -> Any:
        """Create and return a Strands Agent (or compatible mock)."""
        ...


# ---------------------------------------------------------------------------
# AgentOrchestrator
# ---------------------------------------------------------------------------
class AgentOrchestrator:
    """Orchestrates agent creation, message handling, and session lifecycle.

    Parameters
    ----------
    identity_manager:
        Manages identity file loading and updates.
    memory_manager:
        Manages AgentCore Memory sessions.
    tool_registry:
        Provides the list of tool callables.
    model_router:
        Provides the effective Bedrock model ID.
    agent_factory:
        Callable that creates a Strands Agent instance.  Injected for
        testability — in production this wraps ``strands.Agent``.
    """

    def __init__(
        self,
        identity_manager: IdentityManager,
        memory_manager: MemoryManager,
        tool_registry: ToolRegistry,
        model_router: ModelRouter,
        agent_factory: AgentFactory,
    ) -> None:
        self._identity = identity_manager
        self._memory = memory_manager
        self._tools = tool_registry
        self._model_router = model_router
        self._agent_factory = agent_factory

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_agent(self, agent_id: str, user_id: str) -> AgentHandle:
        """Create a fully initialised agent for an interactive session.

        1. Load identity files from S3.
        2. Run bootstrap if needed.
        3. Initialise memory session.
        4. Retrieve LTM context.
        5. Build system prompt from identity + memory.
        6. Create Strands Agent via the injected factory.

        Requirements: 7.1
        """
        # Identity loading
        self._identity.run_bootstrap(agent_id)
        bundle = self._identity.load_identity(agent_id)

        # Memory hydration
        import uuid

        session_id = str(uuid.uuid4())
        memory_handle = self._memory.init_session(agent_id, user_id, session_id)
        memory_context = self._memory.retrieve_context(memory_handle)

        # System prompt composition
        system_prompt = self._identity.build_system_prompt(bundle, memory_context)

        # Tool registration
        tools = self._tools.get_tools()

        # Create Strands Agent
        model_id = self._model_router.get_effective_model_id()
        strands_agent = self._agent_factory(
            system_prompt=system_prompt,
            tools=tools,
            model_id=model_id,
        )

        handle = AgentHandle(
            agent_id=agent_id,
            user_id=user_id,
            identity_bundle=bundle,
            memory_handle=memory_handle,
            system_prompt=system_prompt,
            tools=tools,
            _strands_agent=strands_agent,
        )
        return handle

    def handle_message(self, agent: AgentHandle, message: str) -> str:
        """Pass a user message to the agent and return the response.

        Implements Bedrock throttling retry with exponential backoff and
        jitter (up to 5 attempts: 1s, 2s, 4s, 8s, 16s + jitter).

        Requirements: 7.2, 12.1, 12.2, 12.3
        """
        # Store user turn
        self._memory.store_turn(agent.memory_handle, "user", message)

        # Invoke LLM with throttling retry
        response = self._invoke_with_retry(agent, message)

        # Store assistant turn
        self._memory.store_turn(agent.memory_handle, "assistant", response)

        return response

    def end_session(self, agent: AgentHandle) -> None:
        """End the agent session: flush LTM and persist identity files.

        Requirements: 7.3
        """
        # Trigger LTM strategy execution
        self._memory.flush_to_ltm(agent.memory_handle)

        # Persist updated identity files
        bundle = agent.identity_bundle
        self._identity.update_file(agent.agent_id, "identity", bundle.identity)
        self._identity.update_file(agent.agent_id, "user_profile", bundle.user_profile)
        self._identity.update_file(agent.agent_id, "durable_memory", bundle.durable_memory)

        logger.info("Session ended for agent %s", agent.agent_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _invoke_with_retry(self, agent: AgentHandle, message: str) -> str:
        """Invoke the Strands Agent with Bedrock throttling retry.

        On throttling (HTTP 429), retries up to 5 times with exponential
        backoff (1s, 2s, 4s, 8s, 16s) plus random jitter.

        Requirements: 12.1, 12.2, 12.3
        """
        last_error: Exception | None = None

        for attempt in range(MAX_THROTTLE_RETRIES):
            try:
                result = agent._strands_agent(message)
                # Extract text from the agent response
                if isinstance(result, str):
                    return result
                if hasattr(result, "message"):
                    return str(result.message)
                if hasattr(result, "text"):
                    return str(result.text)
                if hasattr(result, "content"):
                    content = result.content
                    if isinstance(content, list):
                        # Extract text from content blocks
                        parts = []
                        for block in content:
                            if isinstance(block, dict) and "text" in block:
                                parts.append(block["text"])
                            elif isinstance(block, str):
                                parts.append(block)
                            elif hasattr(block, "text"):
                                parts.append(str(block.text))
                            else:
                                parts.append(str(block))
                        return "\n".join(parts)
                    return str(content)
                return str(result)

            except ThrottlingError as exc:
                last_error = exc
                if attempt < MAX_THROTTLE_RETRIES - 1:
                    delay = BASE_DELAYS[attempt] + random.uniform(0, MAX_JITTER)
                    logger.info(
                        "Bedrock throttled (attempt %d/%d), retrying in %.1fs",
                        attempt + 1,
                        MAX_THROTTLE_RETRIES,
                        delay,
                    )
                    time.sleep(delay)
                else:
                    logger.warning(
                        "All %d Bedrock retry attempts exhausted",
                        MAX_THROTTLE_RETRIES,
                    )

            except Exception as exc:
                # Check if this is a Bedrock throttling error wrapped in
                # another exception (e.g., botocore ClientError with 429)
                if _is_throttling_error(exc):
                    last_error = exc
                    if attempt < MAX_THROTTLE_RETRIES - 1:
                        delay = BASE_DELAYS[attempt] + random.uniform(0, MAX_JITTER)
                        logger.info(
                            "Bedrock throttled (attempt %d/%d), retrying in %.1fs",
                            attempt + 1,
                            MAX_THROTTLE_RETRIES,
                            delay,
                        )
                        time.sleep(delay)
                    else:
                        logger.warning(
                            "All %d Bedrock retry attempts exhausted",
                            MAX_THROTTLE_RETRIES,
                        )
                else:
                    # Non-throttling error — don't retry
                    logger.error("Agent invocation failed: %s", exc)
                    return (
                        "I'm sorry, I encountered an unexpected error. "
                        "Please try again."
                    )

        # All retries exhausted (Req 12.2)
        return (
            "I'm experiencing high demand right now and couldn't process "
            "your message after several attempts. Please try again in a "
            "moment."
        )


def _is_throttling_error(exc: Exception) -> bool:
    """Check if an exception represents a Bedrock throttling error (HTTP 429)."""
    # botocore ClientError
    if hasattr(exc, "response"):
        status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")  # type: ignore[union-attr]
        if status == 429:
            return True
        error_code = exc.response.get("Error", {}).get("Code", "")  # type: ignore[union-attr]
        if error_code in ("ThrottlingException", "TooManyRequestsException"):
            return True

    # Check string representation as fallback
    exc_str = str(exc).lower()
    if "throttl" in exc_str or "429" in exc_str or "too many requests" in exc_str:
        return True

    return False
