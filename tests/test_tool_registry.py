"""Tests for ToolRegistry, IdentityTools, MemoryTools, and utility tools."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from agent.identity import IdentityManager
from agent.memory import MemoryManager
from agent.tools import ToolRegistry
from agent.tools.identity_tools import IdentityTools
from agent.tools.memory_tools import MemoryTools
from agent.tools.utils import get_current_date


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_s3_client(bucket: str = "test-bucket") -> MagicMock:
    """Create a mock S3 client with NoSuchKey exception support."""
    client = MagicMock()
    client.exceptions = MagicMock()
    client.exceptions.NoSuchKey = type("NoSuchKey", (Exception,), {})
    return client


def _make_identity_manager() -> MagicMock:
    """Create a mock IdentityManager."""
    mgr = MagicMock(spec=IdentityManager)
    return mgr


def _make_memory_manager() -> MagicMock:
    """Create a mock MemoryManager."""
    mgr = MagicMock(spec=MemoryManager)
    return mgr


# ---------------------------------------------------------------------------
# IdentityTools tests
# ---------------------------------------------------------------------------

class TestIdentityTools:
    def test_update_identity_calls_manager(self):
        mgr = _make_identity_manager()
        tools = IdentityTools(identity_manager=mgr)
        result = tools.update_identity(agent_id="agent-1", content="new identity")
        mgr.update_file.assert_called_once_with("agent-1", "identity", "new identity")
        assert "IDENTITY.md" in result

    def test_update_user_profile_calls_manager(self):
        mgr = _make_identity_manager()
        tools = IdentityTools(identity_manager=mgr)
        result = tools.update_user_profile(agent_id="agent-1", content="new profile")
        mgr.update_file.assert_called_once_with("agent-1", "user_profile", "new profile")
        assert "USER.md" in result

    def test_save_to_memory_calls_manager(self):
        mgr = _make_identity_manager()
        tools = IdentityTools(identity_manager=mgr)
        result = tools.save_to_memory(agent_id="agent-1", content="remember this")
        mgr.update_file.assert_called_once_with("agent-1", "durable_memory", "remember this")
        assert "MEMORY.md" in result


# ---------------------------------------------------------------------------
# MemoryTools tests
# ---------------------------------------------------------------------------

class TestMemoryTools:
    def test_search_memory_returns_results(self):
        mgr = _make_memory_manager()
        mgr.search_memory.return_value = [{"content": "found it"}]
        tools = MemoryTools(memory_manager=mgr)
        result = tools.search_memory(agent_id="agent-1", query="test query")
        mgr.search_memory.assert_called_once_with("agent-1", "test query")
        assert "found it" in result

    def test_search_memory_no_results(self):
        mgr = _make_memory_manager()
        mgr.search_memory.return_value = []
        tools = MemoryTools(memory_manager=mgr)
        result = tools.search_memory(agent_id="agent-1", query="nothing")
        assert "No memories found" in result


# ---------------------------------------------------------------------------
# get_current_date tests
# ---------------------------------------------------------------------------

class TestGetCurrentDate:
    def test_returns_iso_string(self):
        result = get_current_date()
        # Should be parseable as an ISO datetime
        parsed = datetime.fromisoformat(result)
        assert parsed.tzinfo is not None

    def test_returns_utc(self):
        result = get_current_date()
        parsed = datetime.fromisoformat(result)
        assert parsed.tzinfo == timezone.utc


# ---------------------------------------------------------------------------
# ToolRegistry tests
# ---------------------------------------------------------------------------

class TestToolRegistry:
    def test_get_tools_without_scheduler(self):
        registry = ToolRegistry(
            identity_manager=_make_identity_manager(),
            memory_manager=_make_memory_manager(),
            s3_client=_make_s3_client(),
            bucket="test-bucket",
        )
        tools = registry.get_tools()
        # Should have 6 tools: manage_tasks, update_identity, update_user_profile,
        # save_to_memory, search_memory, get_current_date
        assert len(tools) == 6
        names = [t.__name__ for t in tools]
        assert "manage_tasks" in names
        assert "update_identity" in names
        assert "update_user_profile" in names
        assert "save_to_memory" in names
        assert "search_memory" in names
        assert "get_current_date" in names

    def test_get_tools_with_scheduler(self):
        registry = ToolRegistry(
            identity_manager=_make_identity_manager(),
            memory_manager=_make_memory_manager(),
            s3_client=_make_s3_client(),
            bucket="test-bucket",
            scheduler_client=MagicMock(),
            schedule_group="test-group",
            agent_runtime_arn="arn:aws:test",
            scheduler_role_arn="arn:aws:role",
        )
        tools = registry.get_tools()
        # Should have 7 tools (6 + schedule_task)
        assert len(tools) == 7
        names = [t.__name__ for t in tools]
        assert "schedule_task" in names

    def test_all_tools_are_callable(self):
        registry = ToolRegistry(
            identity_manager=_make_identity_manager(),
            memory_manager=_make_memory_manager(),
            s3_client=_make_s3_client(),
            bucket="test-bucket",
            scheduler_client=MagicMock(),
            schedule_group="test-group",
            agent_runtime_arn="arn:aws:test",
            scheduler_role_arn="arn:aws:role",
        )
        for t in registry.get_tools():
            assert callable(t)

    def test_all_tools_have_tool_marker(self):
        registry = ToolRegistry(
            identity_manager=_make_identity_manager(),
            memory_manager=_make_memory_manager(),
            s3_client=_make_s3_client(),
            bucket="test-bucket",
            scheduler_client=MagicMock(),
            schedule_group="test-group",
            agent_runtime_arn="arn:aws:test",
            scheduler_role_arn="arn:aws:role",
        )
        for t in registry.get_tools():
            assert getattr(t, "_is_tool", False), f"{t.__name__} missing _is_tool marker"
