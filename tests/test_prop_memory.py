"""Property-based tests for MemoryManager (Properties 3 and 4).

Property 3: STM turn ordering — turns retrieved in same chronological order
as stored, with non-decreasing timestamps.
**Validates: Requirement 5.3**

Property 4: LTM retrieval limits — summaries ≤5, facts ≤20.
**Validates: Requirement 5.4**
"""

from __future__ import annotations

from unittest.mock import MagicMock

from hypothesis import given, settings
from hypothesis import strategies as st

from agent.memory import MAX_FACTS, MAX_SUMMARIES, MemoryManager


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# A conversation turn: (role, content) pair
role_st = st.sampled_from(["user", "assistant", "system"])
content_st = st.text(min_size=1, max_size=200).filter(lambda s: s.strip())
turn_pair_st = st.tuples(role_st, content_st)

# List of turns to store (1–30 turns)
turn_list_st = st.lists(turn_pair_st, min_size=1, max_size=30)

# Non-empty string for LTM entries
ltm_entry_st = st.text(min_size=1, max_size=100).filter(lambda s: s.strip())

# Summaries list: 0–20 items (can exceed the limit to test clamping)
summaries_st = st.lists(ltm_entry_st, min_size=0, max_size=20)

# Facts list: 0–50 items (can exceed the limit to test clamping)
facts_st = st.lists(ltm_entry_st, min_size=0, max_size=50)

# Preferences list (no limit enforced, but generate some)
preferences_st = st.lists(ltm_entry_st, min_size=0, max_size=10)


def _make_mock_client() -> MagicMock:
    """Create a mock MemoryClient that succeeds on all operations."""
    client = MagicMock()
    client.init_session.return_value = "remote-session"
    client.store_turn.return_value = None
    return client


# ---------------------------------------------------------------------------
# Property 3: STM turn ordering
# ---------------------------------------------------------------------------


class TestSTMTurnOrdering:
    """Turns retrieved from the session handle are in the same chronological
    order as stored, with non-decreasing timestamps.

    **Validates: Requirement 5.3**
    """

    @given(turns=turn_list_st)
    @settings(max_examples=50)
    def test_turns_preserve_insertion_order(self, turns: list[tuple[str, str]]):
        """**Validates: Requirements 5.3**

        For any sequence of (role, content) pairs stored via store_turn,
        the handle's turns list must contain them in the same order.
        """
        client = _make_mock_client()
        mgr = MemoryManager(memory_client=client)
        handle = mgr.init_session("agent-1", "user-1", "sess-1")

        for role, content in turns:
            mgr.store_turn(handle, role, content)

        # Same number of turns stored
        assert len(handle.turns) == len(turns)

        # Order preserved: roles and contents match input order
        for i, (role, content) in enumerate(turns):
            assert handle.turns[i].role == role
            assert handle.turns[i].content == content

    @given(turns=turn_list_st)
    @settings(max_examples=50)
    def test_timestamps_are_non_decreasing(self, turns: list[tuple[str, str]]):
        """**Validates: Requirements 5.3**

        Timestamps assigned to stored turns must be non-decreasing.
        """
        client = _make_mock_client()
        mgr = MemoryManager(memory_client=client)
        handle = mgr.init_session("agent-1", "user-1", "sess-1")

        for role, content in turns:
            mgr.store_turn(handle, role, content)

        for i in range(len(handle.turns) - 1):
            assert handle.turns[i].timestamp <= handle.turns[i + 1].timestamp


# ---------------------------------------------------------------------------
# Property 4: LTM retrieval limits
# ---------------------------------------------------------------------------


class TestLTMRetrievalLimits:
    """retrieve_context enforces summaries ≤ MAX_SUMMARIES and facts ≤ MAX_FACTS.

    **Validates: Requirement 5.4**
    """

    @given(summaries=summaries_st, facts=facts_st, preferences=preferences_st)
    @settings(max_examples=50)
    def test_summaries_and_facts_clamped(
        self,
        summaries: list[str],
        facts: list[str],
        preferences: list[str],
    ):
        """**Validates: Requirements 5.4**

        No matter how many summaries or facts the memory client returns,
        retrieve_context must return at most MAX_SUMMARIES summaries
        and at most MAX_FACTS facts.
        """
        client = _make_mock_client()
        client.retrieve_summaries.return_value = summaries
        client.retrieve_preferences.return_value = preferences
        client.retrieve_facts.return_value = facts

        mgr = MemoryManager(memory_client=client)
        handle = mgr.init_session("agent-1", "user-1", "sess-1")

        ctx = mgr.retrieve_context(handle)

        assert len(ctx.summaries) <= MAX_SUMMARIES
        assert len(ctx.facts) <= MAX_FACTS

        # The returned items are a prefix of the input (order preserved)
        expected_summaries = summaries[:MAX_SUMMARIES]
        expected_facts = facts[:MAX_FACTS]
        assert ctx.summaries == expected_summaries
        assert ctx.facts == expected_facts
