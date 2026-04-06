"""Memory management for the OpenClaw AWS Agent.

Wraps AgentCore Memory to provide OpenClaw-compatible memory semantics.
Maps OpenClaw's two-tier memory (ephemeral + durable) onto AgentCore's
STM and LTM strategies.

When AgentCore Memory is unreachable, falls back to in-memory conversation
history with no persistence.

Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 6.1, 6.2, 6.3, 6.4
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol

from agent.models import MemoryContext, Turn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_SUMMARIES = 5
MAX_FACTS = 20
MAX_RETRY_ATTEMPTS = 3
INITIAL_BACKOFF_SECONDS = 1.0


# ---------------------------------------------------------------------------
# SessionHandle — holds per-session state
# ---------------------------------------------------------------------------
@dataclass
class SessionHandle:
    """Holds session state for a single memory session."""

    agent_id: str
    user_id: str
    session_id: str
    turns: list[Turn] = field(default_factory=list)
    is_degraded: bool = False
    _remote_session: Any = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# MemoryClient protocol — injectable for testing
# ---------------------------------------------------------------------------

class MemoryClient(Protocol):
    """Protocol for the AgentCore Memory client.

    This abstraction allows injecting a mock for testing since
    AgentCore Memory is a managed service that cannot be mocked
    with moto.
    """

    def init_session(
        self, agent_id: str, user_id: str, session_id: str
    ) -> Any:
        """Initialize a remote memory session and return a session handle."""
        ...

    def store_turn(
        self, remote_session: Any, role: str, content: str
    ) -> None:
        """Store a conversation turn in STM."""
        ...

    def retrieve_summaries(
        self, remote_session: Any, max_count: int
    ) -> list[str]:
        """Retrieve recent session summaries from LTM."""
        ...

    def retrieve_preferences(self, remote_session: Any) -> list[str]:
        """Retrieve user preferences from LTM."""
        ...

    def retrieve_facts(
        self, remote_session: Any, max_count: int
    ) -> list[str]:
        """Retrieve known facts from LTM."""
        ...

    def flush_to_ltm(self, remote_session: Any) -> None:
        """Trigger LTM strategy execution (summary, preference, semantic)."""
        ...

    def search(self, agent_id: str, query: str) -> list[dict[str, Any]]:
        """Search across all stored memories for a given agent."""
        ...


# ---------------------------------------------------------------------------
# MemoryManager
# ---------------------------------------------------------------------------
class MemoryManager:
    """Manages memory sessions with AgentCore Memory, falling back to
    in-memory storage when the service is unreachable.

    Parameters
    ----------
    memory_client:
        An object implementing the ``MemoryClient`` protocol.  Pass
        ``None`` to start in degraded mode immediately (useful for
        local development without AgentCore Memory).
    """

    def __init__(self, memory_client: MemoryClient | None = None) -> None:
        self._client = memory_client
        self._is_available: bool = memory_client is not None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def init_session(
        self, agent_id: str, user_id: str, session_id: str
    ) -> SessionHandle:
        """Initialize a memory session.

        Attempts to connect to AgentCore Memory with exponential backoff.
        If all retries fail, falls back to in-memory conversation history.

        Requirements: 5.1, 6.1, 6.3, 6.4
        """
        handle = SessionHandle(
            agent_id=agent_id,
            user_id=user_id,
            session_id=session_id,
        )

        if self._client is None:
            handle.is_degraded = True
            logger.warning(
                "Memory features are degraded: no memory client configured"
            )
            return handle

        # Attempt reconnection on new session start (Req 6.4)
        remote_session = self._retry_with_backoff(
            lambda: self._client.init_session(agent_id, user_id, session_id),
            operation="init_session",
        )

        if remote_session is None:
            handle.is_degraded = True
            self._is_available = False
            logger.warning(
                "Memory features are degraded: AgentCore Memory is "
                "unreachable after %d attempts",
                MAX_RETRY_ATTEMPTS,
            )
        else:
            handle._remote_session = remote_session
            handle.is_degraded = False
            self._is_available = True

        return handle

    def store_turn(
        self, handle: SessionHandle, role: str, content: str
    ) -> None:
        """Store a conversation turn in STM with chronological ordering.

        Requirements: 5.2, 5.3
        """
        turn = Turn(
            role=role,
            content=content,
            timestamp=datetime.now(timezone.utc),
        )
        # Always store locally for chronological ordering
        handle.turns.append(turn)

        if not handle.is_degraded and handle._remote_session is not None:
            try:
                self._client.store_turn(
                    handle._remote_session, role, content
                )
            except Exception:
                logger.warning(
                    "Failed to store turn in AgentCore Memory, "
                    "continuing with in-memory storage"
                )

    def retrieve_context(self, handle: SessionHandle) -> MemoryContext:
        """Retrieve LTM context for the current session.

        Returns at most 5 summaries, all preferences, and at most 20 facts.

        Requirements: 5.4
        """
        if handle.is_degraded or handle._remote_session is None:
            return MemoryContext()

        summaries: list[str] = []
        preferences: list[str] = []
        facts: list[str] = []

        try:
            summaries = self._client.retrieve_summaries(
                handle._remote_session, MAX_SUMMARIES
            )
            # Enforce limit even if client returns more
            summaries = summaries[:MAX_SUMMARIES]
        except Exception:
            logger.warning("Failed to retrieve summaries from LTM")

        try:
            preferences = self._client.retrieve_preferences(
                handle._remote_session
            )
        except Exception:
            logger.warning("Failed to retrieve preferences from LTM")

        try:
            facts = self._client.retrieve_facts(
                handle._remote_session, MAX_FACTS
            )
            # Enforce limit even if client returns more
            facts = facts[:MAX_FACTS]
        except Exception:
            logger.warning("Failed to retrieve facts from LTM")

        return MemoryContext(
            summaries=summaries,
            preferences=preferences,
            facts=facts,
        )

    def flush_to_ltm(self, handle: SessionHandle) -> None:
        """Trigger LTM strategy execution (summary, preference, semantic).

        Requirements: 5.5
        """
        if handle.is_degraded or handle._remote_session is None:
            logger.info(
                "Skipping LTM flush: memory is degraded or no remote session"
            )
            return

        try:
            self._client.flush_to_ltm(handle._remote_session)
        except Exception:
            logger.warning("Failed to flush to LTM")

    def search_memory(
        self, agent_id: str, query: str
    ) -> list[dict[str, Any]]:
        """Search across all stored memories for a given agent.

        Requirements: 5.6
        """
        if self._client is None or not self._is_available:
            logger.warning(
                "Memory search unavailable: AgentCore Memory is not connected"
            )
            return []

        try:
            return self._client.search(agent_id, query)
        except Exception:
            logger.warning("Memory search failed")
            return []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _retry_with_backoff(
        self,
        operation_fn: Any,
        operation: str = "operation",
    ) -> Any | None:
        """Retry an operation with exponential backoff.

        Attempts up to MAX_RETRY_ATTEMPTS times with delays of
        1s, 2s, 4s (exponential backoff).

        Requirements: 6.3
        """
        backoff = INITIAL_BACKOFF_SECONDS

        for attempt in range(1, MAX_RETRY_ATTEMPTS + 1):
            try:
                return operation_fn()
            except Exception as exc:
                if attempt < MAX_RETRY_ATTEMPTS:
                    logger.info(
                        "Retry %d/%d for %s failed (%s), "
                        "retrying in %.1fs",
                        attempt,
                        MAX_RETRY_ATTEMPTS,
                        operation,
                        exc,
                        backoff,
                    )
                    time.sleep(backoff)
                    backoff *= 2
                else:
                    logger.warning(
                        "All %d retry attempts for %s exhausted: %s",
                        MAX_RETRY_ATTEMPTS,
                        operation,
                        exc,
                    )

        return None
