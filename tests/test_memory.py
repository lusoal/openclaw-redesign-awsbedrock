"""Tests for the MemoryManager.

Covers session initialization, turn storage, LTM context retrieval,
graceful degradation, warning logging, and exponential backoff retry.

Uses unittest.mock since AgentCore Memory cannot be mocked with moto.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from agent.memory import (
    INITIAL_BACKOFF_SECONDS,
    MAX_FACTS,
    MAX_RETRY_ATTEMPTS,
    MAX_SUMMARIES,
    MemoryManager,
    SessionHandle,
)
from agent.models import MemoryContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_client(**overrides) -> MagicMock:
    """Create a mock MemoryClient with sensible defaults."""
    client = MagicMock()
    client.init_session.return_value = "remote-session-token"
    client.store_turn.return_value = None
    client.retrieve_summaries.return_value = ["summary-1"]
    client.retrieve_preferences.return_value = ["pref-1"]
    client.retrieve_facts.return_value = ["fact-1"]
    client.flush_to_ltm.return_value = None
    client.search.return_value = [{"id": "1", "content": "result"}]
    for k, v in overrides.items():
        setattr(client, k, v)
    return client


# ===================================================================
# Session initialization
# ===================================================================

class TestInitSession:
    def test_successful_init(self):
        client = _make_client()
        mgr = MemoryManager(memory_client=client)

        handle = mgr.init_session("agent-1", "user-1", "sess-1")

        assert handle.agent_id == "agent-1"
        assert handle.user_id == "user-1"
        assert handle.session_id == "sess-1"
        assert handle.is_degraded is False
        assert handle._remote_session == "remote-session-token"
        client.init_session.assert_called_once_with(
            "agent-1", "user-1", "sess-1"
        )

    def test_no_client_starts_degraded(self, caplog):
        mgr = MemoryManager(memory_client=None)

        with caplog.at_level(logging.WARNING):
            handle = mgr.init_session("agent-1", "user-1", "sess-1")

        assert handle.is_degraded is True
        assert handle._remote_session is None
        assert "degraded" in caplog.text.lower()


    @patch("agent.memory.time.sleep")
    def test_init_falls_back_after_retries(self, mock_sleep, caplog):
        client = _make_client()
        client.init_session.side_effect = ConnectionError("unreachable")
        mgr = MemoryManager(memory_client=client)

        with caplog.at_level(logging.WARNING):
            handle = mgr.init_session("agent-1", "user-1", "sess-1")

        assert handle.is_degraded is True
        assert handle._remote_session is None
        assert client.init_session.call_count == MAX_RETRY_ATTEMPTS
        assert "degraded" in caplog.text.lower()

    @patch("agent.memory.time.sleep")
    def test_exponential_backoff_delays(self, mock_sleep):
        client = _make_client()
        client.init_session.side_effect = ConnectionError("unreachable")
        mgr = MemoryManager(memory_client=client)

        mgr.init_session("agent-1", "user-1", "sess-1")

        # Backoff: 1s, 2s (no sleep after last attempt)
        assert mock_sleep.call_count == MAX_RETRY_ATTEMPTS - 1
        delays = [call.args[0] for call in mock_sleep.call_args_list]
        expected = [
            INITIAL_BACKOFF_SECONDS * (2 ** i)
            for i in range(MAX_RETRY_ATTEMPTS - 1)
        ]
        assert delays == expected

    @patch("agent.memory.time.sleep")
    def test_reconnection_on_new_session_after_outage(self, mock_sleep):
        """After a failed session, a new session should attempt reconnection."""
        client = _make_client()
        # First session: all retries fail
        client.init_session.side_effect = ConnectionError("unreachable")
        mgr = MemoryManager(memory_client=client)
        handle1 = mgr.init_session("agent-1", "user-1", "sess-1")
        assert handle1.is_degraded is True

        # Second session: service is back
        client.init_session.side_effect = None
        client.init_session.return_value = "new-remote-token"
        handle2 = mgr.init_session("agent-1", "user-1", "sess-2")
        assert handle2.is_degraded is False
        assert handle2._remote_session == "new-remote-token"


# ===================================================================
# Turn storage and retrieval
# ===================================================================

class TestStoreTurn:
    def test_store_turn_appends_locally(self):
        client = _make_client()
        mgr = MemoryManager(memory_client=client)
        handle = mgr.init_session("agent-1", "user-1", "sess-1")

        mgr.store_turn(handle, "user", "Hello")
        mgr.store_turn(handle, "assistant", "Hi there")

        assert len(handle.turns) == 2
        assert handle.turns[0].role == "user"
        assert handle.turns[0].content == "Hello"
        assert handle.turns[1].role == "assistant"
        assert handle.turns[1].content == "Hi there"

    def test_turns_maintain_chronological_order(self):
        client = _make_client()
        mgr = MemoryManager(memory_client=client)
        handle = mgr.init_session("agent-1", "user-1", "sess-1")

        messages = [
            ("user", "First"),
            ("assistant", "Second"),
            ("user", "Third"),
            ("assistant", "Fourth"),
        ]
        for role, content in messages:
            mgr.store_turn(handle, role, content)

        for i in range(len(handle.turns) - 1):
            assert handle.turns[i].timestamp <= handle.turns[i + 1].timestamp

        contents = [t.content for t in handle.turns]
        assert contents == ["First", "Second", "Third", "Fourth"]

    def test_store_turn_calls_remote(self):
        client = _make_client()
        mgr = MemoryManager(memory_client=client)
        handle = mgr.init_session("agent-1", "user-1", "sess-1")

        mgr.store_turn(handle, "user", "Hello")

        client.store_turn.assert_called_once_with(
            "remote-session-token", "user", "Hello"
        )

    def test_store_turn_degraded_mode_stores_locally(self):
        mgr = MemoryManager(memory_client=None)
        handle = mgr.init_session("agent-1", "user-1", "sess-1")

        mgr.store_turn(handle, "user", "Hello")

        assert len(handle.turns) == 1
        assert handle.turns[0].content == "Hello"

    def test_store_turn_remote_failure_continues_locally(self, caplog):
        client = _make_client()
        client.store_turn.side_effect = ConnectionError("fail")
        mgr = MemoryManager(memory_client=client)
        handle = mgr.init_session("agent-1", "user-1", "sess-1")

        with caplog.at_level(logging.WARNING):
            mgr.store_turn(handle, "user", "Hello")

        assert len(handle.turns) == 1
        assert "Failed to store turn" in caplog.text


# ===================================================================
# LTM context retrieval
# ===================================================================

class TestRetrieveContext:
    def test_retrieve_context_returns_memory_context(self):
        client = _make_client()
        mgr = MemoryManager(memory_client=client)
        handle = mgr.init_session("agent-1", "user-1", "sess-1")

        ctx = mgr.retrieve_context(handle)

        assert isinstance(ctx, MemoryContext)
        assert ctx.summaries == ["summary-1"]
        assert ctx.preferences == ["pref-1"]
        assert ctx.facts == ["fact-1"]

    def test_retrieve_context_enforces_summary_limit(self):
        client = _make_client()
        client.retrieve_summaries.return_value = [
            f"summary-{i}" for i in range(10)
        ]
        mgr = MemoryManager(memory_client=client)
        handle = mgr.init_session("agent-1", "user-1", "sess-1")

        ctx = mgr.retrieve_context(handle)

        assert len(ctx.summaries) <= MAX_SUMMARIES

    def test_retrieve_context_enforces_facts_limit(self):
        client = _make_client()
        client.retrieve_facts.return_value = [
            f"fact-{i}" for i in range(30)
        ]
        mgr = MemoryManager(memory_client=client)
        handle = mgr.init_session("agent-1", "user-1", "sess-1")

        ctx = mgr.retrieve_context(handle)

        assert len(ctx.facts) <= MAX_FACTS

    def test_retrieve_context_degraded_returns_empty(self):
        mgr = MemoryManager(memory_client=None)
        handle = mgr.init_session("agent-1", "user-1", "sess-1")

        ctx = mgr.retrieve_context(handle)

        assert ctx.summaries == []
        assert ctx.preferences == []
        assert ctx.facts == []

    def test_retrieve_context_partial_failure(self, caplog):
        """If one LTM retrieval fails, others still succeed."""
        client = _make_client()
        client.retrieve_summaries.side_effect = ConnectionError("fail")
        mgr = MemoryManager(memory_client=client)
        handle = mgr.init_session("agent-1", "user-1", "sess-1")

        with caplog.at_level(logging.WARNING):
            ctx = mgr.retrieve_context(handle)

        assert ctx.summaries == []
        assert ctx.preferences == ["pref-1"]
        assert ctx.facts == ["fact-1"]
        assert "Failed to retrieve summaries" in caplog.text


# ===================================================================
# Flush to LTM
# ===================================================================

class TestFlushToLtm:
    def test_flush_calls_remote(self):
        client = _make_client()
        mgr = MemoryManager(memory_client=client)
        handle = mgr.init_session("agent-1", "user-1", "sess-1")

        mgr.flush_to_ltm(handle)

        client.flush_to_ltm.assert_called_once_with("remote-session-token")

    def test_flush_degraded_skips(self, caplog):
        mgr = MemoryManager(memory_client=None)
        handle = mgr.init_session("agent-1", "user-1", "sess-1")

        with caplog.at_level(logging.INFO):
            mgr.flush_to_ltm(handle)

        assert "Skipping LTM flush" in caplog.text

    def test_flush_failure_logs_warning(self, caplog):
        client = _make_client()
        client.flush_to_ltm.side_effect = ConnectionError("fail")
        mgr = MemoryManager(memory_client=client)
        handle = mgr.init_session("agent-1", "user-1", "sess-1")

        with caplog.at_level(logging.WARNING):
            mgr.flush_to_ltm(handle)

        assert "Failed to flush to LTM" in caplog.text


# ===================================================================
# Search memory
# ===================================================================

class TestSearchMemory:
    def test_search_returns_results(self):
        client = _make_client()
        mgr = MemoryManager(memory_client=client)
        mgr.init_session("agent-1", "user-1", "sess-1")

        results = mgr.search_memory("agent-1", "test query")

        assert len(results) == 1
        client.search.assert_called_once_with("agent-1", "test query")

    def test_search_no_client_returns_empty(self, caplog):
        mgr = MemoryManager(memory_client=None)

        with caplog.at_level(logging.WARNING):
            results = mgr.search_memory("agent-1", "test query")

        assert results == []
        assert "unavailable" in caplog.text.lower()

    def test_search_failure_returns_empty(self, caplog):
        client = _make_client()
        client.search.side_effect = ConnectionError("fail")
        mgr = MemoryManager(memory_client=client)
        mgr.init_session("agent-1", "user-1", "sess-1")

        with caplog.at_level(logging.WARNING):
            results = mgr.search_memory("agent-1", "test query")

        assert results == []
        assert "search failed" in caplog.text.lower()


# ===================================================================
# Graceful degradation — warning logging
# ===================================================================

class TestGracefulDegradation:
    def test_degraded_warning_logged_on_no_client(self, caplog):
        mgr = MemoryManager(memory_client=None)

        with caplog.at_level(logging.WARNING):
            mgr.init_session("agent-1", "user-1", "sess-1")

        warning_messages = [
            r.message for r in caplog.records if r.levelno == logging.WARNING
        ]
        assert any("degraded" in msg.lower() for msg in warning_messages)

    @patch("agent.memory.time.sleep")
    def test_degraded_warning_logged_on_connection_failure(
        self, mock_sleep, caplog
    ):
        client = _make_client()
        client.init_session.side_effect = ConnectionError("unreachable")
        mgr = MemoryManager(memory_client=client)

        with caplog.at_level(logging.WARNING):
            mgr.init_session("agent-1", "user-1", "sess-1")

        warning_messages = [
            r.message for r in caplog.records if r.levelno == logging.WARNING
        ]
        assert any("degraded" in msg.lower() for msg in warning_messages)

    def test_all_operations_work_in_degraded_mode(self):
        """Verify the full lifecycle works without a memory client."""
        mgr = MemoryManager(memory_client=None)
        handle = mgr.init_session("agent-1", "user-1", "sess-1")

        mgr.store_turn(handle, "user", "Hello")
        mgr.store_turn(handle, "assistant", "Hi")

        ctx = mgr.retrieve_context(handle)
        assert ctx == MemoryContext()

        mgr.flush_to_ltm(handle)

        results = mgr.search_memory("agent-1", "query")
        assert results == []

        # Turns are still stored locally
        assert len(handle.turns) == 2
