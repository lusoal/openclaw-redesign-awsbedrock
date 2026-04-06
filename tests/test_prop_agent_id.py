"""Property-based tests for agent_id validation (Property 20).

**Validates: Requirements 1.5, 15.8, 16.7**

Tests that validate_agent_id accepts only non-empty strings of alphanumeric
characters and hyphens, up to 50 characters long. Also verifies that models
using agent_id (IdentityBundle, HeartbeatPayload, RuntimeConfig, InfraStackProps)
enforce the same validation.
"""

from datetime import datetime, timezone

import pytest
from hypothesis import given, assume, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from agent.models import (
    IdentityBundle,
    HeartbeatPayload,
    InfraStackProps,
    RuntimeConfig,
    validate_agent_id,
)

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Valid agent_id: 1-50 chars, only [a-zA-Z0-9-]
valid_agent_id_st = st.from_regex(r"[a-zA-Z0-9\-]{1,50}", fullmatch=True)

# Invalid: empty string
empty_st = st.just("")

# Invalid: longer than 50 chars, but otherwise valid characters
too_long_st = st.from_regex(r"[a-zA-Z0-9\-]{51,80}", fullmatch=True)

# Invalid: contains at least one character outside [a-zA-Z0-9-]
# We generate a string that has at least one bad char mixed in
_bad_char_st = st.text(
    alphabet=st.sampled_from("!@#$%^&*()_+=[]{}|;:',.<>?/~ \t\n\"\\"),
    min_size=1,
    max_size=1,
)
_alnum_prefix = st.from_regex(r"[a-zA-Z0-9]{0,10}", fullmatch=True)
_alnum_suffix = st.from_regex(r"[a-zA-Z0-9]{0,10}", fullmatch=True)

invalid_chars_st = st.builds(
    lambda prefix, bad, suffix: prefix + bad + suffix,
    _alnum_prefix,
    _bad_char_st,
    _alnum_suffix,
)


# ---------------------------------------------------------------------------
# Property 20: Agent ID validation
# ---------------------------------------------------------------------------


class TestValidateAgentIdAcceptsValid:
    """Valid agent_id strings (^[a-zA-Z0-9-]+$ with length 1-50) are accepted."""

    @given(agent_id=valid_agent_id_st)
    def test_valid_agent_id_accepted(self, agent_id: str):
        """**Validates: Requirements 1.5, 15.8, 16.7**"""
        result = validate_agent_id(agent_id)
        assert result == agent_id


class TestValidateAgentIdRejectsEmpty:
    """Empty strings are rejected."""

    def test_empty_string_rejected(self):
        """**Validates: Requirements 1.5, 15.8, 16.7**"""
        with pytest.raises(ValueError, match="non-empty"):
            validate_agent_id("")

    @given(whitespace=st.from_regex(r"[ \t\n\r]+", fullmatch=True))
    @settings(max_examples=20)
    def test_whitespace_only_rejected(self, whitespace: str):
        """**Validates: Requirements 1.5, 15.8, 16.7**"""
        with pytest.raises(ValueError):
            validate_agent_id(whitespace)


class TestValidateAgentIdRejectsTooLong:
    """Strings longer than 50 characters are rejected."""

    @given(agent_id=too_long_st)
    def test_too_long_rejected(self, agent_id: str):
        """**Validates: Requirements 1.5, 15.8, 16.7**"""
        assert len(agent_id) > 50
        with pytest.raises(ValueError, match="at most 50"):
            validate_agent_id(agent_id)


class TestValidateAgentIdRejectsInvalidChars:
    """Strings with characters outside [a-zA-Z0-9-] are rejected."""

    @given(agent_id=invalid_chars_st)
    def test_invalid_chars_rejected(self, agent_id: str):
        """**Validates: Requirements 1.5, 15.8, 16.7**"""
        # Ensure the string is non-empty and within length so we isolate the char check
        assume(len(agent_id) <= 50)
        assume(len(agent_id) > 0)
        with pytest.raises(ValueError):
            validate_agent_id(agent_id)


# ---------------------------------------------------------------------------
# Agent ID validation propagates through models
# ---------------------------------------------------------------------------

NOW = datetime.now(timezone.utc)


class TestIdentityBundleAgentId:
    """IdentityBundle enforces agent_id validation via validate_agent_id."""

    @given(agent_id=valid_agent_id_st)
    @settings(max_examples=20)
    def test_valid_agent_id_accepted(self, agent_id: str):
        """**Validates: Requirements 1.5, 15.8, 16.7**"""
        bundle = IdentityBundle(
            agent_id=agent_id, soul="soul", agents="agents", loaded_at=NOW
        )
        assert bundle.agent_id == agent_id

    def test_invalid_agent_id_rejected(self):
        """**Validates: Requirements 1.5, 15.8, 16.7**"""
        with pytest.raises(ValidationError):
            IdentityBundle(
                agent_id="bad agent!", soul="soul", agents="agents", loaded_at=NOW
            )


class TestHeartbeatPayloadAgentId:
    """HeartbeatPayload enforces agent_id validation via validate_agent_id."""

    @given(agent_id=valid_agent_id_st)
    @settings(max_examples=20)
    def test_valid_agent_id_accepted(self, agent_id: str):
        """**Validates: Requirements 1.5, 15.8, 16.7**"""
        hp = HeartbeatPayload(
            type="heartbeat", task="do-it", agent_id=agent_id, user_id="user1"
        )
        assert hp.agent_id == agent_id

    def test_invalid_agent_id_rejected(self):
        """**Validates: Requirements 1.5, 15.8, 16.7**"""
        with pytest.raises(ValidationError):
            HeartbeatPayload(
                type="heartbeat", task="do-it", agent_id="", user_id="user1"
            )


class TestRuntimeConfigAgentId:
    """RuntimeConfig enforces agent_id validation via validate_agent_id."""

    @given(agent_id=valid_agent_id_st)
    @settings(max_examples=20)
    def test_valid_agent_id_accepted(self, agent_id: str):
        """**Validates: Requirements 1.5, 15.8, 16.7**"""
        rc = RuntimeConfig(
            agent_id=agent_id,
            s3_bucket="bucket",
            bedrock_model_id="model",
            bedrock_region="us-east-1",
            memory_id="mem",
        )
        assert rc.agent_id == agent_id

    def test_invalid_agent_id_rejected(self):
        """**Validates: Requirements 1.5, 15.8, 16.7**"""
        with pytest.raises(ValidationError):
            RuntimeConfig(
                agent_id="bad agent!",
                s3_bucket="bucket",
                bedrock_model_id="model",
                bedrock_region="us-east-1",
                memory_id="mem",
            )


class TestInfraStackPropsAgentId:
    """InfraStackProps enforces agent_id validation via validate_agent_id."""

    @given(agent_id=valid_agent_id_st)
    @settings(max_examples=20)
    def test_valid_agent_id_accepted(self, agent_id: str):
        """**Validates: Requirements 1.5, 15.8, 16.7**"""
        props = InfraStackProps(
            agent_id=agent_id,
            bedrock_model_id="model",
            bedrock_region="us-east-1",
            memory_id="mem",
            schedule_group_name="sg",
        )
        assert props.agent_id == agent_id

    def test_invalid_agent_id_rejected(self):
        """**Validates: Requirements 1.5, 15.8, 16.7**"""
        with pytest.raises(ValidationError):
            InfraStackProps(
                agent_id="a" * 51,
                bedrock_model_id="model",
                bedrock_region="us-east-1",
                memory_id="mem",
                schedule_group_name="sg",
            )
