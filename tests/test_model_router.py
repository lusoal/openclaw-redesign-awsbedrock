"""Tests for ModelRouter (task 2.2).

Validates Requirements 11.1, 11.2, 11.3, 11.4, 11.5.
"""

from __future__ import annotations

import logging

import pytest

from agent.model_router import ModelRouter
from agent.models import BedrockModelConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VALID_ROUTER_ARN = (
    "arn:aws:bedrock:us-east-1:123456789012:default-prompt-router/anthropic.claude:1"
)
VALID_MODEL_ID = "anthropic.claude-sonnet-4-20250514"
FALLBACK_MODEL_ID = "anthropic.claude-haiku-3"


# ---------------------------------------------------------------------------
# Tests: Routing enabled returns router_arn (Req 11.1)
# ---------------------------------------------------------------------------

class TestRoutingEnabled:
    """When routing is enabled, the router ARN is used as the model_id."""

    def test_get_effective_model_id_returns_router_arn(self):
        router = ModelRouter(
            routing_enabled=True,
            router_arn=VALID_ROUTER_ARN,
            fallback_model_id=FALLBACK_MODEL_ID,
        )
        assert router.get_effective_model_id() == VALID_ROUTER_ARN

    def test_get_model_config_has_routing_enabled(self):
        router = ModelRouter(
            routing_enabled=True,
            router_arn=VALID_ROUTER_ARN,
            fallback_model_id=FALLBACK_MODEL_ID,
        )
        config = router.get_model_config()
        assert isinstance(config, BedrockModelConfig)
        assert config.routing_enabled is True
        assert config.router_arn == VALID_ROUTER_ARN
        assert config.fallback_model_id == FALLBACK_MODEL_ID

    def test_effective_model_id_with_fallback_when_available(self):
        router = ModelRouter(
            routing_enabled=True,
            router_arn=VALID_ROUTER_ARN,
            fallback_model_id=FALLBACK_MODEL_ID,
        )
        assert router.get_effective_model_id_with_fallback(router_available=True) == VALID_ROUTER_ARN


# ---------------------------------------------------------------------------
# Tests: Routing disabled returns model_id (Req 11.2)
# ---------------------------------------------------------------------------

class TestRoutingDisabled:
    """When routing is disabled, the specific model_id is used."""

    def test_get_effective_model_id_returns_model_id(self):
        router = ModelRouter(
            routing_enabled=False,
            model_id=VALID_MODEL_ID,
            fallback_model_id=FALLBACK_MODEL_ID,
        )
        assert router.get_effective_model_id() == VALID_MODEL_ID

    def test_get_model_config_has_routing_disabled(self):
        router = ModelRouter(
            routing_enabled=False,
            model_id=VALID_MODEL_ID,
            fallback_model_id=FALLBACK_MODEL_ID,
        )
        config = router.get_model_config()
        assert isinstance(config, BedrockModelConfig)
        assert config.routing_enabled is False
        assert config.model_id == VALID_MODEL_ID
        assert config.fallback_model_id == FALLBACK_MODEL_ID

    def test_effective_model_id_with_fallback_ignores_availability(self):
        """When routing is disabled, router_available has no effect."""
        router = ModelRouter(
            routing_enabled=False,
            model_id=VALID_MODEL_ID,
            fallback_model_id=FALLBACK_MODEL_ID,
        )
        assert router.get_effective_model_id_with_fallback(router_available=False) == VALID_MODEL_ID


# ---------------------------------------------------------------------------
# Tests: Fallback when router unavailable (Req 11.3)
# ---------------------------------------------------------------------------

class TestFallback:
    """When the Prompt Router is unavailable, fallback_model_id is used."""

    def test_falls_back_when_router_unavailable(self):
        router = ModelRouter(
            routing_enabled=True,
            router_arn=VALID_ROUTER_ARN,
            fallback_model_id=FALLBACK_MODEL_ID,
        )
        result = router.get_effective_model_id_with_fallback(router_available=False)
        assert result == FALLBACK_MODEL_ID

    def test_logs_warning_on_fallback(self, caplog):
        router = ModelRouter(
            routing_enabled=True,
            router_arn=VALID_ROUTER_ARN,
            fallback_model_id=FALLBACK_MODEL_ID,
        )
        with caplog.at_level(logging.WARNING, logger="agent.model_router"):
            router.get_effective_model_id_with_fallback(router_available=False)

        assert any("Prompt Router is unavailable" in r.message for r in caplog.records)
        assert any(FALLBACK_MODEL_ID in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Tests: ARN format validation (Req 11.4)
# ---------------------------------------------------------------------------

class TestARNValidation:
    """router_arn must match the expected Bedrock Prompt Router ARN format."""

    def test_valid_default_prompt_router_arn(self):
        # Should not raise
        ModelRouter(
            routing_enabled=True,
            router_arn="arn:aws:bedrock:us-east-1:123456789012:default-prompt-router/anthropic.claude:1",
            fallback_model_id=FALLBACK_MODEL_ID,
        )

    def test_valid_prompt_router_arn(self):
        # Should not raise
        ModelRouter(
            routing_enabled=True,
            router_arn="arn:aws:bedrock:us-west-2:987654321098:prompt-router/my-router",
            fallback_model_id=FALLBACK_MODEL_ID,
        )

    def test_rejects_invalid_arn_format(self):
        with pytest.raises(ValueError, match="does not match expected"):
            ModelRouter(
                routing_enabled=True,
                router_arn="not-an-arn",
                fallback_model_id=FALLBACK_MODEL_ID,
            )

    def test_rejects_non_bedrock_arn(self):
        with pytest.raises(ValueError, match="does not match expected"):
            ModelRouter(
                routing_enabled=True,
                router_arn="arn:aws:s3:::my-bucket",
                fallback_model_id=FALLBACK_MODEL_ID,
            )

    def test_rejects_empty_router_arn_when_enabled(self):
        with pytest.raises(ValueError, match="router_arn must be provided"):
            ModelRouter(
                routing_enabled=True,
                router_arn="",
                fallback_model_id=FALLBACK_MODEL_ID,
            )

    def test_rejects_none_router_arn_when_enabled(self):
        with pytest.raises(ValueError, match="router_arn must be provided"):
            ModelRouter(
                routing_enabled=True,
                router_arn=None,
                fallback_model_id=FALLBACK_MODEL_ID,
            )

    def test_arn_validation_skipped_when_routing_disabled(self):
        # Invalid ARN should be fine when routing is disabled
        ModelRouter(
            routing_enabled=False,
            model_id=VALID_MODEL_ID,
            router_arn="garbage",
            fallback_model_id=FALLBACK_MODEL_ID,
        )


# ---------------------------------------------------------------------------
# Tests: fallback_model_id always required (Req 11.5)
# ---------------------------------------------------------------------------

class TestFallbackModelIdRequired:
    """fallback_model_id must always be set regardless of routing config."""

    def test_rejects_empty_fallback_model_id(self):
        with pytest.raises(ValueError, match="fallback_model_id must be non-empty"):
            ModelRouter(
                routing_enabled=True,
                router_arn=VALID_ROUTER_ARN,
                fallback_model_id="",
            )

    def test_rejects_whitespace_fallback_model_id(self):
        with pytest.raises(ValueError, match="fallback_model_id must be non-empty"):
            ModelRouter(
                routing_enabled=False,
                model_id=VALID_MODEL_ID,
                fallback_model_id="   ",
            )

    def test_rejects_missing_model_id_when_disabled(self):
        with pytest.raises(ValueError, match="model_id must be provided"):
            ModelRouter(
                routing_enabled=False,
                model_id="",
                fallback_model_id=FALLBACK_MODEL_ID,
            )
