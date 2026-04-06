"""Tests for agent/main.py — entry point, component wiring, and CLI.

Covers handle_invocation routing (heartbeat vs interactive),
_build_components wiring, and the main() CLI entry point.

Requirements: 7.1, 7.2, 10.1, 13.4
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agent.main import _build_components, handle_invocation


# ---------------------------------------------------------------------------
# handle_invocation — routing
# ---------------------------------------------------------------------------

class TestHandleInvocation:
    """Test that handle_invocation routes heartbeat vs interactive correctly."""

    @patch("agent.main._build_components")
    def test_missing_agent_id_returns_error(self, mock_build):
        result = handle_invocation({})

        assert result["status"] == "error"
        assert "agent_id" in result["error_message"].lower()
        mock_build.assert_not_called()

    @patch("agent.main._build_components")
    def test_heartbeat_routed_to_handler(self, mock_build):
        mock_orch = MagicMock()
        mock_hb = MagicMock()
        mock_hb.handle_heartbeat.return_value = {"status": "heartbeat_ok"}
        mock_build.return_value = (MagicMock(), mock_orch, mock_hb)

        event = {
            "type": "heartbeat",
            "agent_id": "test-agent",
            "user_id": "user-1",
            "task": "Do something",
        }
        result = handle_invocation(event)

        mock_hb.handle_heartbeat.assert_called_once_with(event)
        assert result["status"] == "heartbeat_ok"

    @patch("agent.main._build_components")
    def test_interactive_session_returns_response(self, mock_build):
        mock_orch = MagicMock()
        mock_handle = MagicMock()
        mock_orch.create_agent.return_value = mock_handle
        mock_orch.handle_message.return_value = "Hello!"
        mock_build.return_value = (MagicMock(), mock_orch, MagicMock())

        event = {
            "agent_id": "test-agent",
            "user_id": "user-1",
            "message": "Hi there",
        }
        result = handle_invocation(event)

        assert result["status"] == "completed"
        assert result["output"] == "Hello!"
        mock_orch.create_agent.assert_called_once_with("test-agent", "user-1")
        mock_orch.handle_message.assert_called_once_with(mock_handle, "Hi there")
        mock_orch.end_session.assert_called_once_with(mock_handle)

    @patch("agent.main._build_components")
    def test_interactive_missing_message_returns_error(self, mock_build):
        mock_build.return_value = (MagicMock(), MagicMock(), MagicMock())

        event = {"agent_id": "test-agent", "user_id": "user-1"}
        result = handle_invocation(event)

        assert result["status"] == "error"
        assert "message" in result["error_message"].lower()

    @patch("agent.main._build_components")
    def test_interactive_exception_returns_error(self, mock_build):
        mock_orch = MagicMock()
        mock_orch.create_agent.side_effect = RuntimeError("S3 down")
        mock_build.return_value = (MagicMock(), mock_orch, MagicMock())

        event = {
            "agent_id": "test-agent",
            "user_id": "user-1",
            "message": "Hello",
        }
        result = handle_invocation(event)

        assert result["status"] == "error"
        assert "S3 down" in result["error_message"]

    @patch("agent.main._build_components")
    def test_interactive_default_user_id(self, mock_build):
        mock_orch = MagicMock()
        mock_handle = MagicMock()
        mock_orch.create_agent.return_value = mock_handle
        mock_orch.handle_message.return_value = "Hi anon"
        mock_build.return_value = (MagicMock(), mock_orch, MagicMock())

        event = {"agent_id": "test-agent", "message": "Hello"}
        result = handle_invocation(event)

        mock_orch.create_agent.assert_called_once_with("test-agent", "anonymous")
        assert result["status"] == "completed"


# ---------------------------------------------------------------------------
# _build_components — wiring
# ---------------------------------------------------------------------------

class TestBuildComponents:
    """Test that _build_components wires dependencies correctly."""

    @patch("agent.main.boto3")
    @patch("agent.main.DeploymentConfig")
    def test_returns_three_components(self, mock_dc_cls, mock_boto3):
        from agent.models import RuntimeConfig

        mock_config = RuntimeConfig(
            agent_id="test-agent",
            s3_bucket="my-bucket",
            bedrock_model_id="anthropic.claude-sonnet-4-20250514",
            bedrock_region="us-east-1",
            memory_id="mem-123",
        )
        mock_dc = MagicMock()
        mock_dc.configure_from_stack.return_value = mock_config
        mock_dc_cls.return_value = mock_dc

        mock_s3 = MagicMock()
        mock_boto3.client.return_value = mock_s3

        config, orch, hb = _build_components("test-agent")

        assert config.agent_id == "test-agent"
        assert config.s3_bucket == "my-bucket"
        # Orchestrator and heartbeat handler are created
        from agent.orchestrator import AgentOrchestrator
        from agent.heartbeat import HeartbeatHandler

        assert isinstance(orch, AgentOrchestrator)
        assert isinstance(hb, HeartbeatHandler)

    @patch("agent.main.boto3")
    @patch("agent.main.DeploymentConfig")
    def test_scheduler_client_created_when_group_set(self, mock_dc_cls, mock_boto3):
        from agent.models import RuntimeConfig

        mock_config = RuntimeConfig(
            agent_id="test-agent",
            s3_bucket="my-bucket",
            bedrock_model_id="anthropic.claude-sonnet-4-20250514",
            bedrock_region="us-east-1",
            memory_id="mem-123",
            schedule_group_name="my-group",
            scheduler_role_arn="arn:aws:iam::123456789012:role/sched",
        )
        mock_dc = MagicMock()
        mock_dc.configure_from_stack.return_value = mock_config
        mock_dc_cls.return_value = mock_dc

        mock_boto3.client.return_value = MagicMock()

        _build_components("test-agent")

        # boto3.client should be called for both s3 and scheduler
        client_calls = [c[0][0] for c in mock_boto3.client.call_args_list]
        assert "s3" in client_calls
        assert "scheduler" in client_calls


# ---------------------------------------------------------------------------
# chat_ui.py launch script
# ---------------------------------------------------------------------------

class TestChatUIScript:
    """Test the root-level chat_ui.py launch script."""

    @patch("agent.main._build_components")
    @patch("agent.chat_ui.launch")
    def test_chat_ui_script_wires_correctly(self, mock_launch, mock_build):
        """Verify the chat_ui.py script calls launch with correct args."""
        mock_orch = MagicMock()
        mock_build.return_value = (MagicMock(), mock_orch, MagicMock())

        # Import and call main from the root chat_ui module
        import importlib
        import chat_ui as chat_ui_script

        with patch("sys.argv", ["chat_ui.py", "--agent-id", "my-agent", "--user-id", "user-42"]):
            importlib.reload(chat_ui_script)
            chat_ui_script.main()

        mock_build.assert_called_once_with("my-agent")
        mock_launch.assert_called_once_with(
            mock_orch, "my-agent", "user-42", port=7860
        )
