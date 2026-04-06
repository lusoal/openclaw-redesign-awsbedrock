"""Property-based tests for IdentityBundle required fields (Property 21).

**Validates: Requirement 16.1**

For any IdentityBundle, the soul and agents fields shall be non-empty strings.
Construction with empty soul or agents shall be rejected. Optional fields
(identity, user_profile, durable_memory) can be empty strings.
"""

from datetime import datetime, timezone

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from agent.models import IdentityBundle

NOW = datetime.now(timezone.utc)

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Valid agent_id: 1-50 chars, only [a-zA-Z0-9-]
valid_agent_id_st = st.from_regex(r"[a-zA-Z0-9\-]{1,50}", fullmatch=True)

# Non-empty text (at least one non-whitespace character)
non_empty_text_st = st.text(min_size=1, max_size=200).filter(lambda s: s.strip())

# Whitespace-only strings
whitespace_only_st = st.from_regex(r"[ \t\n\r]+", fullmatch=True)

# Any text including empty
any_text_st = st.text(min_size=0, max_size=200)


# ---------------------------------------------------------------------------
# Property 21: Valid IdentityBundle construction
# ---------------------------------------------------------------------------


class TestIdentityBundleValidConstruction:
    """Valid IdentityBundle with non-empty soul and agents is accepted."""

    @given(
        agent_id=valid_agent_id_st,
        soul=non_empty_text_st,
        agents=non_empty_text_st,
        identity=any_text_st,
        user_profile=any_text_st,
        durable_memory=any_text_st,
    )
    @settings(max_examples=50)
    def test_valid_construction(
        self,
        agent_id: str,
        soul: str,
        agents: str,
        identity: str,
        user_profile: str,
        durable_memory: str,
    ):
        """**Validates: Requirement 16.1**"""
        bundle = IdentityBundle(
            agent_id=agent_id,
            soul=soul,
            agents=agents,
            identity=identity,
            user_profile=user_profile,
            durable_memory=durable_memory,
            loaded_at=NOW,
        )
        assert bundle.soul == soul
        assert bundle.agents == agents
        assert bundle.identity == identity
        assert bundle.user_profile == user_profile
        assert bundle.durable_memory == durable_memory


# ---------------------------------------------------------------------------
# Property 21: Empty soul is rejected
# ---------------------------------------------------------------------------


class TestIdentityBundleEmptySoul:
    """Empty soul string is rejected."""

    def test_empty_soul_rejected(self):
        """**Validates: Requirement 16.1**"""
        with pytest.raises(ValidationError, match="soul"):
            IdentityBundle(
                agent_id="agent-1",
                soul="",
                agents="valid agents",
                loaded_at=NOW,
            )


# ---------------------------------------------------------------------------
# Property 21: Whitespace-only soul is rejected
# ---------------------------------------------------------------------------


class TestIdentityBundleWhitespaceSoul:
    """Whitespace-only soul string is rejected."""

    @given(soul=whitespace_only_st)
    @settings(max_examples=20)
    def test_whitespace_soul_rejected(self, soul: str):
        """**Validates: Requirement 16.1**"""
        with pytest.raises(ValidationError, match="soul"):
            IdentityBundle(
                agent_id="agent-1",
                soul=soul,
                agents="valid agents",
                loaded_at=NOW,
            )


# ---------------------------------------------------------------------------
# Property 21: Empty agents is rejected
# ---------------------------------------------------------------------------


class TestIdentityBundleEmptyAgents:
    """Empty agents string is rejected."""

    def test_empty_agents_rejected(self):
        """**Validates: Requirement 16.1**"""
        with pytest.raises(ValidationError, match="agents"):
            IdentityBundle(
                agent_id="agent-1",
                soul="valid soul",
                agents="",
                loaded_at=NOW,
            )


# ---------------------------------------------------------------------------
# Property 21: Whitespace-only agents is rejected
# ---------------------------------------------------------------------------


class TestIdentityBundleWhitespaceAgents:
    """Whitespace-only agents string is rejected."""

    @given(agents=whitespace_only_st)
    @settings(max_examples=20)
    def test_whitespace_agents_rejected(self, agents: str):
        """**Validates: Requirement 16.1**"""
        with pytest.raises(ValidationError, match="agents"):
            IdentityBundle(
                agent_id="agent-1",
                soul="valid soul",
                agents=agents,
                loaded_at=NOW,
            )


# ---------------------------------------------------------------------------
# Property 21: Optional fields can be empty strings
# ---------------------------------------------------------------------------


class TestIdentityBundleOptionalFieldsEmpty:
    """Optional fields (identity, user_profile, durable_memory) can be empty."""

    @given(agent_id=valid_agent_id_st, soul=non_empty_text_st, agents=non_empty_text_st)
    @settings(max_examples=30)
    def test_optional_fields_default_empty(self, agent_id: str, soul: str, agents: str):
        """**Validates: Requirement 16.1**"""
        bundle = IdentityBundle(
            agent_id=agent_id,
            soul=soul,
            agents=agents,
            loaded_at=NOW,
        )
        assert bundle.identity == ""
        assert bundle.user_profile == ""
        assert bundle.durable_memory == ""
