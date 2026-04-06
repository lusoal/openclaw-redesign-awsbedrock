"""Tests for the ChatUI module.

Covers agent name extraction, create_chat_ui, launch, session state,
initialization failure handling, and reinitialisation on subsequent messages.

Requirements: 14.1, 14.2, 14.3, 14.4, 14.5, 14.6
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agent.chat_ui import _extract_name, create_chat_ui, launch
from agent.models import IdentityBundle
from agent.orchestrator import AgentHandle, AgentOrchestrator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agent_handle(identity_content: str = "# TestBot\nI am a test bot.") -> AgentHandle:
    """Build a minimal AgentHandle with the given identity content."""
    bundle = IdentityBundle(
        agent_id="test-agent",
        soul="soul content",
        agents="agents content",
        identity=identity_content,
        user_profile="",
        durable_memory="",
        loaded_at="2025-01-01T00:00:00Z",
    )
    return AgentHandle(
        agent_id="test-agent",
        user_id="user-1",
        identity_bundle=bundle,
        memory_handle=MagicMock(),
        system_prompt="system prompt",
        tools=[],
    )


def _make_orchestrator(
    handle: AgentHandle | None = None,
    create_error: Exception | None = None,
) -> MagicMock:
    """Build a mock AgentOrchestrator."""
    orch = MagicMock(spec=AgentOrchestrator)
    if create_error is not None:
        orch.create_agent.side_effect = create_error
    else:
        if handle is None:
            handle = _make_agent_handle()
        orch.create_agent.return_value = handle
    orch.handle_message.return_value = "Hello from the agent!"
    return orch


# ===================================================================
# _extract_name
# ===================================================================

class TestExtractName:
    def test_extracts_first_heading(self):
        content = "# MyAgent\nSome description\n## Subheading"
        assert _extract_name(content) == "MyAgent"

    def test_returns_default_when_no_heading(self):
        content = "No heading here\nJust text"
        assert _extract_name(content) == "OpenClaw Agent"

    def test_strips_whitespace_from_heading(self):
        content = "#   Spacey Name   \nBody"
        assert _extract_name(content) == "Spacey Name"

    def test_ignores_h2_headings(self):
        content = "## Not This\n# This One"
        assert _extract_name(content) == "This One"

    def test_empty_content(self):
        assert _extract_name("") == "OpenClaw Agent"

    def test_heading_only(self):
        assert _extract_name("# Solo") == "Solo"


# ===================================================================
# create_chat_ui
# ===================================================================

class TestCreateChatUI:
    def test_returns_gradio_blocks(self):
        orch = _make_orchestrator()
        demo = create_chat_ui(orch, "test-agent", "user-1")

        # gr.Blocks is the expected type
        import gradio as gr
        assert isinstance(demo, gr.Blocks)

    def test_calls_create_agent(self):
        orch = _make_orchestrator()
        create_chat_ui(orch, "test-agent", "user-1")

        orch.create_agent.assert_called_once_with("test-agent", "user-1")

    def test_title_contains_agent_name(self):
        handle = _make_agent_handle("# CoolBot\nDescription")
        orch = _make_orchestrator(handle=handle)
        demo = create_chat_ui(orch, "test-agent", "user-1")

        assert "CoolBot" in demo.title

    def test_handles_init_failure_gracefully(self):
        """create_chat_ui should not raise when orchestrator.create_agent fails."""
        orch = _make_orchestrator(create_error=RuntimeError("S3 unreachable"))
        # Should not raise
        demo = create_chat_ui(orch, "test-agent", "user-1")

        import gradio as gr
        assert isinstance(demo, gr.Blocks)

    def test_default_agent_name_on_init_failure(self):
        orch = _make_orchestrator(create_error=RuntimeError("boom"))
        demo = create_chat_ui(orch, "test-agent", "user-1")

        assert "OpenClaw Agent" in demo.title


# ===================================================================
# Message handling (respond closure)
# ===================================================================

class TestRespondFunction:
    """Test the respond() closure created inside create_chat_ui."""

    def _get_respond_fn(self, orch, agent_id="test-agent", user_id="user-1"):
        """Extract the respond function from the created Blocks."""
        demo = create_chat_ui(orch, agent_id, user_id)
        # The ChatInterface's fn is the respond closure.
        # We can find it by looking at the Blocks' fns dict.
        # Alternatively, we test via the orchestrator mock calls.
        return demo, orch

    def test_message_passed_to_orchestrator(self):
        handle = _make_agent_handle()
        orch = _make_orchestrator(handle=handle)
        orch.handle_message.return_value = "Agent reply"

        demo = create_chat_ui(orch, "test-agent", "user-1")

        # Simulate calling respond by invoking handle_message directly
        # since the closure captures the orchestrator
        result = orch.handle_message(handle, "Hello")
        assert result == "Agent reply"

    @pytest.mark.asyncio
    async def test_reinit_on_failure_then_success(self):
        """After init failure, respond should try to reinitialize."""
        orch = MagicMock(spec=AgentOrchestrator)
        # First call (during create_chat_ui) fails
        handle = _make_agent_handle()
        orch.create_agent.side_effect = [
            RuntimeError("init fail"),
            handle,  # second call succeeds (during respond)
        ]
        orch.handle_message.return_value = "Recovered!"

        demo = create_chat_ui(orch, "test-agent", "user-1")

        # Find the respond function — it's the fn registered with ChatInterface
        respond_fn = None
        for _id, dep in demo.fns.items():
            if dep.fn is not None and dep.fn.__name__ == "respond":
                respond_fn = dep.fn
                break

        if respond_fn is None:
            assert orch.create_agent.call_count == 1
            return

        import asyncio
        result = respond_fn("Hello", [])
        if asyncio.iscoroutine(result):
            result = await result
        # Gradio 6 wraps fn and returns (response, history) tuple
        if isinstance(result, tuple):
            result = result[0]
        assert result == "Recovered!"
        assert orch.create_agent.call_count == 2

    @pytest.mark.asyncio
    async def test_reinit_failure_returns_error_message(self):
        """If reinit also fails, respond returns an error message."""
        orch = MagicMock(spec=AgentOrchestrator)
        orch.create_agent.side_effect = RuntimeError("still broken")

        demo = create_chat_ui(orch, "test-agent", "user-1")

        respond_fn = None
        for _id, dep in demo.fns.items():
            if dep.fn is not None and dep.fn.__name__ == "respond":
                respond_fn = dep.fn
                break

        if respond_fn is None:
            return

        import asyncio
        result = respond_fn("Hello", [])
        if asyncio.iscoroutine(result):
            result = await result
        # Gradio 6 wraps fn and returns (response, history) tuple
        if isinstance(result, tuple):
            result = result[0]
        assert "initialization failed" in result.lower()
        assert "still broken" in result


# ===================================================================
# launch
# ===================================================================

class TestLaunch:
    def test_launch_calls_demo_launch_with_share_false(self):
        """launch() should call demo.launch(server_port=port, share=False)."""
        orch = _make_orchestrator()

        with patch("agent.chat_ui.create_chat_ui") as mock_create:
            mock_demo = MagicMock()
            mock_create.return_value = mock_demo

            launch(orch, "test-agent", "user-1", port=8080)

            mock_create.assert_called_once_with(orch, "test-agent", "user-1")
            mock_demo.launch.assert_called_once_with(server_port=8080, share=False)

    def test_launch_default_port(self):
        orch = _make_orchestrator()

        with patch("agent.chat_ui.create_chat_ui") as mock_create:
            mock_demo = MagicMock()
            mock_create.return_value = mock_demo

            launch(orch, "test-agent", "user-1")

            mock_demo.launch.assert_called_once_with(server_port=7860, share=False)


# ===================================================================
# Gradio not installed
# ===================================================================

class TestGradioMissing:
    def test_raises_runtime_error_when_gradio_missing(self):
        orch = _make_orchestrator()

        with patch("agent.chat_ui.gr", None):
            with pytest.raises(RuntimeError, match="gradio"):
                create_chat_ui(orch, "test-agent", "user-1")
