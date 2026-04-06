"""Property-based tests for BedrockModelConfig and RuntimeConfig validation.

**Property 19: Model configuration validation**
For any BedrockModelConfig where routing_enabled is true, router_arn shall be
non-empty. For any BedrockModelConfig where routing_enabled is false, model_id
shall be non-empty. For all configurations, fallback_model_id shall be non-empty.

**Property 22: RuntimeConfig required fields**
For any RuntimeConfig, the fields agent_id, s3_bucket, bedrock_model_id,
bedrock_region, and memory_id shall all be non-empty strings. Construction with
any empty required field shall be rejected.

**Validates: Requirements 11.4, 11.5, 16.5, 16.6**
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from agent.models import BedrockModelConfig, RuntimeConfig

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Non-empty text (at least one non-whitespace character)
non_empty_text_st = st.text(min_size=1, max_size=100).filter(lambda s: s.strip())

# Whitespace-only strings
whitespace_only_st = st.from_regex(r"[ \t\n\r]+", fullmatch=True)

# Valid agent_id: 1-50 chars, only [a-zA-Z0-9-]
valid_agent_id_st = st.from_regex(r"[a-zA-Z0-9\-]{1,50}", fullmatch=True)


# ===========================================================================
# Property 19: BedrockModelConfig — routing enabled requires router_arn
# ===========================================================================


class TestBedrockModelConfigRoutingEnabled:
    """When routing_enabled=True, router_arn must be non-empty."""

    @given(router_arn=non_empty_text_st, fallback=non_empty_text_st)
    @settings(max_examples=50)
    def test_valid_routing_enabled(self, router_arn: str, fallback: str):
        """**Validates: Requirements 11.4, 16.5**"""
        cfg = BedrockModelConfig(
            routing_enabled=True,
            router_arn=router_arn,
            fallback_model_id=fallback,
        )
        assert cfg.routing_enabled is True
        assert cfg.router_arn == router_arn
        assert cfg.fallback_model_id == fallback

    def test_empty_router_arn_rejected(self):
        """**Validates: Requirements 11.4, 16.5**"""
        with pytest.raises(ValidationError, match="router_arn"):
            BedrockModelConfig(
                routing_enabled=True,
                router_arn="",
                fallback_model_id="fallback-model",
            )

    def test_missing_router_arn_rejected(self):
        """**Validates: Requirements 11.4, 16.5**"""
        with pytest.raises(ValidationError, match="router_arn"):
            BedrockModelConfig(
                routing_enabled=True,
                fallback_model_id="fallback-model",
            )

    @given(ws=whitespace_only_st)
    @settings(max_examples=20)
    def test_whitespace_router_arn_rejected(self, ws: str):
        """**Validates: Requirements 11.4, 16.5**"""
        with pytest.raises(ValidationError, match="router_arn"):
            BedrockModelConfig(
                routing_enabled=True,
                router_arn=ws,
                fallback_model_id="fallback-model",
            )


# ===========================================================================
# Property 19: BedrockModelConfig — routing disabled requires model_id
# ===========================================================================


class TestBedrockModelConfigRoutingDisabled:
    """When routing_enabled=False, model_id must be non-empty."""

    @given(model_id=non_empty_text_st, fallback=non_empty_text_st)
    @settings(max_examples=50)
    def test_valid_routing_disabled(self, model_id: str, fallback: str):
        """**Validates: Requirements 11.4, 16.5**"""
        cfg = BedrockModelConfig(
            routing_enabled=False,
            model_id=model_id,
            fallback_model_id=fallback,
        )
        assert cfg.routing_enabled is False
        assert cfg.model_id == model_id
        assert cfg.fallback_model_id == fallback

    def test_empty_model_id_rejected(self):
        """**Validates: Requirements 11.4, 16.5**"""
        with pytest.raises(ValidationError, match="model_id"):
            BedrockModelConfig(
                routing_enabled=False,
                model_id="",
                fallback_model_id="fallback-model",
            )

    def test_missing_model_id_rejected(self):
        """**Validates: Requirements 11.4, 16.5**"""
        with pytest.raises(ValidationError, match="model_id"):
            BedrockModelConfig(
                routing_enabled=False,
                fallback_model_id="fallback-model",
            )

    @given(ws=whitespace_only_st)
    @settings(max_examples=20)
    def test_whitespace_model_id_rejected(self, ws: str):
        """**Validates: Requirements 11.4, 16.5**"""
        with pytest.raises(ValidationError, match="model_id"):
            BedrockModelConfig(
                routing_enabled=False,
                model_id=ws,
                fallback_model_id="fallback-model",
            )


# ===========================================================================
# Property 19: BedrockModelConfig — fallback_model_id always required
# ===========================================================================


class TestBedrockModelConfigFallback:
    """fallback_model_id must always be non-empty regardless of routing_enabled."""

    def test_empty_fallback_rejected_routing_enabled(self):
        """**Validates: Requirements 11.5, 16.5**"""
        with pytest.raises(ValidationError, match="fallback_model_id"):
            BedrockModelConfig(
                routing_enabled=True,
                router_arn="arn:aws:bedrock:us-east-1:123456789012:router/test",
                fallback_model_id="",
            )

    def test_empty_fallback_rejected_routing_disabled(self):
        """**Validates: Requirements 11.5, 16.5**"""
        with pytest.raises(ValidationError, match="fallback_model_id"):
            BedrockModelConfig(
                routing_enabled=False,
                model_id="anthropic.claude-sonnet-4-20250514",
                fallback_model_id="",
            )

    @given(fallback=non_empty_text_st)
    @settings(max_examples=30)
    def test_non_empty_fallback_accepted(self, fallback: str):
        """**Validates: Requirements 11.5, 16.5**"""
        cfg = BedrockModelConfig(
            routing_enabled=False,
            model_id="some-model",
            fallback_model_id=fallback,
        )
        assert cfg.fallback_model_id == fallback


# ===========================================================================
# Property 22: RuntimeConfig — all required fields non-empty
# ===========================================================================


class TestRuntimeConfigValidConstruction:
    """Valid RuntimeConfig with all required fields non-empty is accepted."""

    @given(
        agent_id=valid_agent_id_st,
        s3_bucket=non_empty_text_st,
        bedrock_model_id=non_empty_text_st,
        bedrock_region=non_empty_text_st,
        memory_id=non_empty_text_st,
    )
    @settings(max_examples=50)
    def test_valid_construction(
        self,
        agent_id: str,
        s3_bucket: str,
        bedrock_model_id: str,
        bedrock_region: str,
        memory_id: str,
    ):
        """**Validates: Requirement 16.6**"""
        rc = RuntimeConfig(
            agent_id=agent_id,
            s3_bucket=s3_bucket,
            bedrock_model_id=bedrock_model_id,
            bedrock_region=bedrock_region,
            memory_id=memory_id,
        )
        assert rc.agent_id == agent_id
        assert rc.s3_bucket == s3_bucket
        assert rc.bedrock_model_id == bedrock_model_id
        assert rc.bedrock_region == bedrock_region
        assert rc.memory_id == memory_id


class TestRuntimeConfigEmptyFieldsRejected:
    """Each required field individually empty is rejected."""

    _VALID_DEFAULTS = {
        "agent_id": "agent-1",
        "s3_bucket": "my-bucket",
        "bedrock_model_id": "anthropic.claude-sonnet-4-20250514",
        "bedrock_region": "us-east-1",
        "memory_id": "mem-123",
    }

    @pytest.mark.parametrize("field", [
        "agent_id",
        "s3_bucket",
        "bedrock_model_id",
        "bedrock_region",
        "memory_id",
    ])
    def test_empty_field_rejected(self, field: str):
        """**Validates: Requirement 16.6**"""
        kwargs = {**self._VALID_DEFAULTS, field: ""}
        with pytest.raises(ValidationError):
            RuntimeConfig(**kwargs)
