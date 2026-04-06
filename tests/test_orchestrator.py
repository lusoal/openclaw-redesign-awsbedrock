"""Tests for the AgentOrchestrator.

Covers agent creation, message handling with throttling retry,
session teardown, and error handling.

Uses unittest.mock since Strands Agent and Bedrock cannot be mocked with moto.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, call, patch

import pytest

from agent.memory import MemoryManager, SessionHandle
from agent.models import IdentityBundle, MemoryContext
from agent.orchestrator import (
    MAX_THROTTLE_RETRIES,
    AgentHandle,
    AgentOrchestrator,
    ThrottlingError,
    _is_throttling_error,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_identity_manager() -> MagicMock:
    """Create a mock IdentityManager with sensible defaults."""
    mgr = MagicMock()
    mgr.run_bootstrap.return_value = None
    mgr.load_identity.return_value = IdentityBundle(
        agent_id="test-agent",
        soul="I am a test soul.",
        agents="Test agent instructions.",
        identity="Test identity.",
        user_profile="Test user profile.",
        durable_memory="Test memory.",
        loaded_at="2025-01-01T00:00:00Z",
    )
    mgr.build_system_prompt.return_value = "## SOUL\n\nI am a test soul."
    mgr.update_file.return_value = None
    return mgr


def _make_memory_manager() -> MagicMock:
    """Create a mock MemoryManager."""
    mgr = MagicMock(spec=MemoryManager)
    handle = SessionHandle(
        agent_id="test-agent",
        user_id="user-1",
        session_id="sess-1",
    )
    mgr.init_session.return_value = handle
    mgr.retrieve_context.return_value = MemoryContext(
        summaries=["summary-1"],
        preferences=["pref-1"],
        facts=["fact-1"],
    )
    mgr.store_turn.return_value = None
    mgr.flush_to_ltm.return_value = None
    return mgr


def _make_tool_registry() -> MagicMock:
    """Create a mock ToolRegistry."""
    registry = MagicMock()
    registry.get_tools.return_value = [lambda: "tool1", lambda: "tool2"]
    return registry


def _make_model_router() -> MagicMock:
    """Create a mock ModelRouter."""
    router = MagicMock()
    router.get_effective_model_id.return_value = "anthropic.claude-sonnet-4-20250514"
    return router


def _make_agent_factory() -> MagicMock:
    """Create a mock agent factory that returns a callable agent."""
    mock_agent = MagicMock(return_value="Hello! I'm your assistant.")
    factory = MagicMock(return_value=mock_agent)
    return factory


def _build_orchestrator(**overrides):
    """Build an AgentOrchestrator with mock dependencies."""
    kwargs = {
        "identity_manager": _make_identity_manager(),
        "memory_manager": _make_memory_manager(),
        "tool_registry": _make_tool_registry(),
        "model_router": _make_model_router(),
        "agent_factory": _make_agent_factory(),
    }
    kwargs.update(overrides)
    return AgentOrchestrator(**kwargs), kwargs


# ===================================================================
# Agent creation
# ===================================================================

class TestCreateAgent:
    def test_create_agent_returns_handle(self):
        orch, deps = _build_orchestrator()

        handle = orch.create_agent("test-agent", "user-1")

        assert isinstance(handle, AgentHandle)
        assert handle.agent_id == "test-agent"
        assert handle.user_id == "user-1"
        assert handle.system_prompt == "## SOUL\n\nI am a test soul."
        assert handle._strands_agent is not None

    def test_create_agent_runs_bootstrap(self):
        orch, deps = _build_orchestrator()

        orch.create_agent("test-agent", "user-1")

        deps["identity_manager"].run_bootstrap.assert_called_once_with("test-agent")

    def test_create_agent_loads_identity(self):
        orch, deps = _build_orchestrator()

        orch.create_agent("test-agent", "user-1")

        deps["identity_manager"].load_identity.assert_called_once_with("test-agent")

    def test_create_agent_inits_memory_session(self):
        orch, deps = _build_orchestrator()

        orch.create_agent("test-agent", "user-1")

        deps["memory_manager"].init_session.assert_called_once()
        args = deps["memory_manager"].init_session.call_args
        assert args[0][0] == "test-agent"
        assert args[0][1] == "user-1"
        # session_id is a UUID string
        assert len(args[0][2]) > 0

    def test_create_agent_retrieves_ltm_context(self):
        orch, deps = _build_orchestrator()

        orch.create_agent("test-agent", "user-1")

        deps["memory_manager"].retrieve_context.assert_called_once()

    def test_create_agent_builds_system_prompt(self):
        orch, deps = _build_orchestrator()

        orch.create_agent("test-agent", "user-1")

        deps["identity_manager"].build_system_prompt.assert_called_once()

    def test_create_agent_gets_tools(self):
        orch, deps = _build_orchestrator()

        handle = orch.create_agent("test-agent", "user-1")

        deps["tool_registry"].get_tools.assert_called_once()
        assert len(handle.tools) == 2

    def test_create_agent_calls_factory_with_correct_args(self):
        orch, deps = _build_orchestrator()

        orch.create_agent("test-agent", "user-1")

        deps["agent_factory"].assert_called_once()
        call_kwargs = deps["agent_factory"].call_args[1]
        assert call_kwargs["system_prompt"] == "## SOUL\n\nI am a test soul."
        assert call_kwargs["model_id"] == "anthropic.claude-sonnet-4-20250514"
        assert len(call_kwargs["tools"]) == 2


# ===================================================================
# Message handling
# ===================================================================

class TestHandleMessage:
    def test_handle_message_returns_response(self):
        orch, deps = _build_orchestrator()
        handle = orch.create_agent("test-agent", "user-1")

        response = orch.handle_message(handle, "Hello")

        assert response == "Hello! I'm your assistant."

    def test_handle_message_stores_user_turn(self):
        orch, deps = _build_orchestrator()
        handle = orch.create_agent("test-agent", "user-1")

        orch.handle_message(handle, "Hello")

        # First call is user turn, second is assistant turn
        store_calls = deps["memory_manager"].store_turn.call_args_list
        assert store_calls[0] == call(handle.memory_handle, "user", "Hello")

    def test_handle_message_stores_assistant_turn(self):
        orch, deps = _build_orchestrator()
        handle = orch.create_agent("test-agent", "user-1")

        orch.handle_message(handle, "Hello")

        store_calls = deps["memory_manager"].store_turn.call_args_list
        assert store_calls[1] == call(
            handle.memory_handle, "assistant", "Hello! I'm your assistant."
        )

    def test_handle_message_with_object_response(self):
        """Agent returns an object with .text attribute."""
        factory = MagicMock()
        agent_obj = MagicMock()
        result_obj = MagicMock(spec=[])
        result_obj.text = "Response from object"
        agent_obj.return_value = result_obj
        factory.return_value = agent_obj

        orch, deps = _build_orchestrator(agent_factory=factory)
        handle = orch.create_agent("test-agent", "user-1")

        response = orch.handle_message(handle, "Hello")

        assert response == "Response from object"


# ===================================================================
# Throttling retry
# ===================================================================

class TestThrottlingRetry:
    @patch("agent.orchestrator.time.sleep")
    def test_retry_on_throttling_error(self, mock_sleep):
        factory = MagicMock()
        agent_obj = MagicMock()
        # Fail twice with throttling, then succeed
        agent_obj.side_effect = [
            ThrottlingError("429"),
            ThrottlingError("429"),
            "Success after retry",
        ]
        factory.return_value = agent_obj

        orch, deps = _build_orchestrator(agent_factory=factory)
        handle = orch.create_agent("test-agent", "user-1")

        response = orch.handle_message(handle, "Hello")

        assert response == "Success after retry"
        assert mock_sleep.call_count == 2
        assert agent_obj.call_count == 3

    @patch("agent.orchestrator.time.sleep")
    def test_all_retries_exhausted_returns_friendly_message(self, mock_sleep):
        factory = MagicMock()
        agent_obj = MagicMock()
        agent_obj.side_effect = ThrottlingError("429")
        factory.return_value = agent_obj

        orch, deps = _build_orchestrator(agent_factory=factory)
        handle = orch.create_agent("test-agent", "user-1")

        response = orch.handle_message(handle, "Hello")

        assert "high demand" in response.lower()
        assert "try again" in response.lower()
        assert agent_obj.call_count == MAX_THROTTLE_RETRIES
        assert mock_sleep.call_count == MAX_THROTTLE_RETRIES - 1

    @patch("agent.orchestrator.random.uniform", return_value=0.5)
    @patch("agent.orchestrator.time.sleep")
    def test_exponential_backoff_with_jitter(self, mock_sleep, mock_uniform):
        factory = MagicMock()
        agent_obj = MagicMock()
        agent_obj.side_effect = [
            ThrottlingError("429"),
            ThrottlingError("429"),
            ThrottlingError("429"),
            "Success",
        ]
        factory.return_value = agent_obj

        orch, deps = _build_orchestrator(agent_factory=factory)
        handle = orch.create_agent("test-agent", "user-1")

        orch.handle_message(handle, "Hello")

        delays = [c.args[0] for c in mock_sleep.call_args_list]
        # 1 + 0.5, 2 + 0.5, 4 + 0.5
        assert delays == [1.5, 2.5, 4.5]

    @patch("agent.orchestrator.time.sleep")
    def test_retry_on_wrapped_throttling_error(self, mock_sleep):
        """Handles botocore-style ClientError with 429 status."""
        factory = MagicMock()
        agent_obj = MagicMock()

        # Simulate a botocore ClientError with 429
        client_error = Exception("Throttling")
        client_error.response = {  # type: ignore[attr-defined]
            "ResponseMetadata": {"HTTPStatusCode": 429},
            "Error": {"Code": "ThrottlingException"},
        }
        agent_obj.side_effect = [client_error, "Success"]
        factory.return_value = agent_obj

        orch, deps = _build_orchestrator(agent_factory=factory)
        handle = orch.create_agent("test-agent", "user-1")

        response = orch.handle_message(handle, "Hello")

        assert response == "Success"
        assert mock_sleep.call_count == 1

    def test_non_throttling_error_returns_error_message(self):
        factory = MagicMock()
        agent_obj = MagicMock()
        agent_obj.side_effect = RuntimeError("Something broke")
        factory.return_value = agent_obj

        orch, deps = _build_orchestrator(agent_factory=factory)
        handle = orch.create_agent("test-agent", "user-1")

        response = orch.handle_message(handle, "Hello")

        assert "unexpected error" in response.lower()

    @patch("agent.orchestrator.time.sleep")
    def test_retry_logs_delay_info(self, mock_sleep, caplog):
        factory = MagicMock()
        agent_obj = MagicMock()
        agent_obj.side_effect = [ThrottlingError("429"), "OK"]
        factory.return_value = agent_obj

        orch, deps = _build_orchestrator(agent_factory=factory)
        handle = orch.create_agent("test-agent", "user-1")

        with caplog.at_level(logging.INFO):
            orch.handle_message(handle, "Hello")

        assert any("throttled" in r.message.lower() for r in caplog.records)


# ===================================================================
# Session end
# ===================================================================

class TestEndSession:
    def test_end_session_flushes_ltm(self):
        orch, deps = _build_orchestrator()
        handle = orch.create_agent("test-agent", "user-1")

        orch.end_session(handle)

        deps["memory_manager"].flush_to_ltm.assert_called_once_with(
            handle.memory_handle
        )

    def test_end_session_persists_identity_files(self):
        orch, deps = _build_orchestrator()
        handle = orch.create_agent("test-agent", "user-1")

        orch.end_session(handle)

        update_calls = deps["identity_manager"].update_file.call_args_list
        assert len(update_calls) == 3
        # Check all three dynamic files are persisted
        file_types = {c[0][1] for c in update_calls}
        assert file_types == {"identity", "user_profile", "durable_memory"}

    def test_end_session_logs_completion(self, caplog):
        orch, deps = _build_orchestrator()
        handle = orch.create_agent("test-agent", "user-1")

        with caplog.at_level(logging.INFO):
            orch.end_session(handle)

        assert any("session ended" in r.message.lower() for r in caplog.records)


# ===================================================================
# _is_throttling_error helper
# ===================================================================

class TestIsThrottlingError:
    def test_throttling_exception_code(self):
        exc = Exception("throttle")
        exc.response = {  # type: ignore[attr-defined]
            "ResponseMetadata": {"HTTPStatusCode": 429},
            "Error": {"Code": "ThrottlingException"},
        }
        assert _is_throttling_error(exc) is True

    def test_too_many_requests_code(self):
        exc = Exception("too many")
        exc.response = {  # type: ignore[attr-defined]
            "ResponseMetadata": {"HTTPStatusCode": 200},
            "Error": {"Code": "TooManyRequestsException"},
        }
        assert _is_throttling_error(exc) is True

    def test_http_429_status(self):
        exc = Exception("error")
        exc.response = {  # type: ignore[attr-defined]
            "ResponseMetadata": {"HTTPStatusCode": 429},
            "Error": {"Code": "SomeOtherCode"},
        }
        assert _is_throttling_error(exc) is True

    def test_string_fallback_throttle(self):
        exc = Exception("Request was throttled by Bedrock")
        assert _is_throttling_error(exc) is True

    def test_non_throttling_error(self):
        exc = RuntimeError("Something else broke")
        assert _is_throttling_error(exc) is False

    def test_plain_exception_no_response(self):
        exc = ValueError("bad value")
        assert _is_throttling_error(exc) is False
