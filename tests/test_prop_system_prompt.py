"""Property-based tests for system prompt composition (Properties 1 and 2).

Property 1: System prompt determinism — same IdentityBundle and MemoryContext
produce identical output.
**Validates: Requirement 3.2**

Property 2: System prompt completeness — composed prompt contains all non-empty
identity fields and memory context entries.
**Validates: Requirement 3.1**
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

from hypothesis import given, settings
from hypothesis import strategies as st

from agent.identity import IdentityManager
from agent.models import IdentityBundle, MemoryContext

NOW = datetime.now(timezone.utc)

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Valid agent_id: 1-50 chars, only [a-zA-Z0-9-]
valid_agent_id_st = st.from_regex(r"[a-zA-Z0-9\-]{1,50}", fullmatch=True)

# Non-empty text (at least one non-whitespace character)
non_empty_text_st = st.text(min_size=1, max_size=200).filter(lambda s: s.strip())

# Any text including empty
any_text_st = st.text(min_size=0, max_size=200)

# Non-empty list of non-empty strings for memory context fields
non_empty_str_list_st = st.lists(non_empty_text_st, min_size=1, max_size=5)

# Possibly-empty list of strings for memory context fields
any_str_list_st = st.lists(any_text_st, min_size=0, max_size=5)


@st.composite
def identity_bundle_st(draw: st.DrawFn) -> IdentityBundle:
    """Generate a valid IdentityBundle with random field values."""
    return IdentityBundle(
        agent_id=draw(valid_agent_id_st),
        soul=draw(non_empty_text_st),
        agents=draw(non_empty_text_st),
        identity=draw(any_text_st),
        user_profile=draw(any_text_st),
        durable_memory=draw(any_text_st),
        loaded_at=NOW,
    )


@st.composite
def identity_bundle_all_non_empty_st(draw: st.DrawFn) -> IdentityBundle:
    """Generate an IdentityBundle where all identity fields are non-empty."""
    return IdentityBundle(
        agent_id=draw(valid_agent_id_st),
        soul=draw(non_empty_text_st),
        agents=draw(non_empty_text_st),
        identity=draw(non_empty_text_st),
        user_profile=draw(non_empty_text_st),
        durable_memory=draw(non_empty_text_st),
        loaded_at=NOW,
    )


@st.composite
def memory_context_st(draw: st.DrawFn) -> MemoryContext:
    """Generate a MemoryContext with random lists."""
    return MemoryContext(
        summaries=draw(any_str_list_st),
        preferences=draw(any_str_list_st),
        facts=draw(any_str_list_st),
    )


@st.composite
def memory_context_non_empty_st(draw: st.DrawFn) -> MemoryContext:
    """Generate a MemoryContext where all lists have at least one non-empty entry."""
    return MemoryContext(
        summaries=draw(non_empty_str_list_st),
        preferences=draw(non_empty_str_list_st),
        facts=draw(non_empty_str_list_st),
    )


def _make_manager() -> IdentityManager:
    """Create an IdentityManager with a mock S3 client (not used by build_system_prompt)."""
    return IdentityManager(s3_client=MagicMock(), bucket="test-bucket")


# ---------------------------------------------------------------------------
# Property 1: System prompt determinism
# ---------------------------------------------------------------------------


class TestSystemPromptDeterminism:
    """Same IdentityBundle and MemoryContext produce identical output."""

    @given(bundle=identity_bundle_st(), ctx=memory_context_st())
    @settings(max_examples=50)
    def test_same_inputs_produce_identical_output(
        self, bundle: IdentityBundle, ctx: MemoryContext
    ):
        """**Validates: Requirement 3.2**"""
        mgr = _make_manager()
        first = mgr.build_system_prompt(bundle, ctx)
        second = mgr.build_system_prompt(bundle, ctx)
        assert first == second


# ---------------------------------------------------------------------------
# Property 2: System prompt completeness
# ---------------------------------------------------------------------------


class TestSystemPromptCompleteness:
    """Composed prompt contains all non-empty identity fields and memory entries."""

    @given(bundle=identity_bundle_all_non_empty_st(), ctx=memory_context_non_empty_st())
    @settings(max_examples=50)
    def test_all_identity_fields_present(
        self, bundle: IdentityBundle, ctx: MemoryContext
    ):
        """**Validates: Requirement 3.1**"""
        mgr = _make_manager()
        prompt = mgr.build_system_prompt(bundle, ctx)

        # Every non-empty identity field value must appear in the prompt
        assert bundle.soul in prompt
        assert bundle.agents in prompt
        assert bundle.identity in prompt
        assert bundle.user_profile in prompt
        assert bundle.durable_memory in prompt

        # Every memory context entry must appear in the prompt
        for summary in ctx.summaries:
            assert summary in prompt
        for pref in ctx.preferences:
            assert pref in prompt
        for fact in ctx.facts:
            assert fact in prompt
