"""Tests for DeploymentConfig (task 2.1).

Validates Requirements 13.1, 13.2, 13.3, 13.4.
"""

from __future__ import annotations

import logging
import os
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

from agent.config import DeploymentConfig
from agent.models import RuntimeConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ssm_response(agent_id: str, values: dict[str, str]) -> dict:
    """Build a mock get_parameters response."""
    params = [
        {"Name": f"/{agent_id}/config/{key}", "Value": val}
        for key, val in values.items()
    ]
    all_keys = {
        "bucket-name", "memory-id", "bedrock-model-id",
        "prompt-router-arn", "schedule-group-name", "scheduler-role-arn",
    }
    invalid = [
        f"/{agent_id}/config/{k}" for k in all_keys - set(values.keys())
    ]
    return {"Parameters": params, "InvalidParameters": invalid}


# ---------------------------------------------------------------------------
# Tests: SSM-based configuration (Req 13.1)
# ---------------------------------------------------------------------------

class TestConfigureFromStackSSM:
    """When SSM parameters are available, they populate RuntimeConfig."""

    def test_reads_all_ssm_parameters(self):
        ssm = MagicMock()
        ssm.get_parameters.return_value = _make_ssm_response("my-agent", {
            "bucket-name": "my-bucket",
            "memory-id": "mem-123",
            "bedrock-model-id": "anthropic.claude-sonnet-4-20250514",
            "prompt-router-arn": "arn:aws:bedrock:us-east-1:123456789012:default-prompt-router/anthropic.claude:1",
            "schedule-group-name": "agent-schedules",
            "scheduler-role-arn": "arn:aws:iam::123456789012:role/sched",
        })

        cfg = DeploymentConfig(ssm_client=ssm)
        result = cfg.configure_from_stack("my-agent")

        assert isinstance(result, RuntimeConfig)
        assert result.agent_id == "my-agent"
        assert result.s3_bucket == "my-bucket"
        assert result.memory_id == "mem-123"
        assert result.bedrock_model_id == "anthropic.claude-sonnet-4-20250514"
        assert result.prompt_router_arn == "arn:aws:bedrock:us-east-1:123456789012:default-prompt-router/anthropic.claude:1"
        assert result.schedule_group_name == "agent-schedules"
        assert result.scheduler_role_arn == "arn:aws:iam::123456789012:role/sched"

    def test_ssm_parameter_path_uses_agent_id(self):
        ssm = MagicMock()
        ssm.get_parameters.return_value = _make_ssm_response("test-agent", {
            "bucket-name": "b",
            "memory-id": "m",
            "bedrock-model-id": "model",
        })

        cfg = DeploymentConfig(ssm_client=ssm)
        cfg.configure_from_stack("test-agent")

        call_args = ssm.get_parameters.call_args
        names = call_args[1].get("Names") or call_args[0][0] if call_args[0] else call_args[1]["Names"]
        assert all(n.startswith("/test-agent/config/") for n in names)


# ---------------------------------------------------------------------------
# Tests: Environment variable fallback (Req 13.2)
# ---------------------------------------------------------------------------

class TestConfigureFromStackEnvFallback:
    """When SSM is unavailable, environment variables are used."""

    def test_falls_back_to_env_vars_on_ssm_error(self):
        ssm = MagicMock()
        ssm.get_parameters.side_effect = ClientError(
            {"Error": {"Code": "AccessDeniedException", "Message": "no access"}},
            "GetParameters",
        )

        env = {
            "IDENTITY_BUCKET": "env-bucket",
            "MEMORY_ID": "env-mem",
            "BEDROCK_MODEL_ID": "env-model",
            "BEDROCK_REGION": "us-west-2",
            "PROMPT_ROUTER_ARN": "arn:aws:bedrock:us-west-2:111111111111:default-prompt-router/x:1",
            "SCHEDULE_GROUP_NAME": "env-group",
            "SCHEDULER_ROLE_ARN": "arn:aws:iam::111111111111:role/r",
        }

        with patch.dict(os.environ, env, clear=False):
            cfg = DeploymentConfig(ssm_client=ssm)
            result = cfg.configure_from_stack("my-agent")

        assert result.s3_bucket == "env-bucket"
        assert result.memory_id == "env-mem"
        assert result.bedrock_model_id == "env-model"
        assert result.bedrock_region == "us-west-2"
        assert result.prompt_router_arn == env["PROMPT_ROUTER_ARN"]
        assert result.schedule_group_name == "env-group"
        assert result.scheduler_role_arn == env["SCHEDULER_ROLE_ARN"]

    def test_ssm_values_take_precedence_over_env(self):
        ssm = MagicMock()
        ssm.get_parameters.return_value = _make_ssm_response("ag", {
            "bucket-name": "ssm-bucket",
            "memory-id": "ssm-mem",
            "bedrock-model-id": "ssm-model",
        })

        env = {
            "IDENTITY_BUCKET": "env-bucket",
            "MEMORY_ID": "env-mem",
            "BEDROCK_MODEL_ID": "env-model",
            "BEDROCK_REGION": "eu-west-1",
        }

        with patch.dict(os.environ, env, clear=False):
            cfg = DeploymentConfig(ssm_client=ssm)
            result = cfg.configure_from_stack("ag")

        # SSM wins
        assert result.s3_bucket == "ssm-bucket"
        assert result.memory_id == "ssm-mem"
        assert result.bedrock_model_id == "ssm-model"
        # Region comes from env
        assert result.bedrock_region == "eu-west-1"

    def test_partial_ssm_fills_gaps_from_env(self):
        """SSM provides some params, env provides the rest."""
        ssm = MagicMock()
        ssm.get_parameters.return_value = _make_ssm_response("ag", {
            "bucket-name": "ssm-bucket",
        })

        env = {
            "MEMORY_ID": "env-mem",
            "BEDROCK_MODEL_ID": "env-model",
            "BEDROCK_REGION": "us-east-1",
        }

        with patch.dict(os.environ, env, clear=False):
            cfg = DeploymentConfig(ssm_client=ssm)
            result = cfg.configure_from_stack("ag")

        assert result.s3_bucket == "ssm-bucket"
        assert result.memory_id == "env-mem"
        assert result.bedrock_model_id == "env-model"


# ---------------------------------------------------------------------------
# Tests: Warning logging (Req 13.3)
# ---------------------------------------------------------------------------

class TestConfigureFromStackLogging:
    """Warnings are logged when SSM is unavailable."""

    def test_logs_warning_on_ssm_client_error(self, caplog):
        ssm = MagicMock()
        ssm.get_parameters.side_effect = ClientError(
            {"Error": {"Code": "InternalError", "Message": "boom"}},
            "GetParameters",
        )

        env = {
            "IDENTITY_BUCKET": "b",
            "MEMORY_ID": "m",
            "BEDROCK_MODEL_ID": "model",
            "BEDROCK_REGION": "us-east-1",
        }

        with patch.dict(os.environ, env, clear=False):
            cfg = DeploymentConfig(ssm_client=ssm)
            with caplog.at_level(logging.WARNING, logger="agent.config"):
                cfg.configure_from_stack("ag")

        assert any("SSM-based configuration is unavailable" in r.message for r in caplog.records)

    def test_logs_warning_on_unexpected_error(self, caplog):
        ssm = MagicMock()
        ssm.get_parameters.side_effect = RuntimeError("network down")

        env = {
            "IDENTITY_BUCKET": "b",
            "MEMORY_ID": "m",
            "BEDROCK_MODEL_ID": "model",
            "BEDROCK_REGION": "us-east-1",
        }

        with patch.dict(os.environ, env, clear=False):
            cfg = DeploymentConfig(ssm_client=ssm)
            with caplog.at_level(logging.WARNING, logger="agent.config"):
                cfg.configure_from_stack("ag")

        assert any("Unexpected error reading SSM parameters" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Tests: Default region and lazy client (Req 13.4)
# ---------------------------------------------------------------------------

class TestDeploymentConfigMisc:
    """Miscellaneous behaviour: default region, lazy client creation."""

    def test_default_bedrock_region(self):
        ssm = MagicMock()
        ssm.get_parameters.return_value = _make_ssm_response("ag", {
            "bucket-name": "b",
            "memory-id": "m",
            "bedrock-model-id": "model",
        })

        # Ensure BEDROCK_REGION is not set
        env_clean = {k: v for k, v in os.environ.items() if k != "BEDROCK_REGION"}
        with patch.dict(os.environ, env_clean, clear=True):
            cfg = DeploymentConfig(ssm_client=ssm)
            result = cfg.configure_from_stack("ag")

        assert result.bedrock_region == "us-east-1"

    def test_returns_runtime_config_type(self):
        ssm = MagicMock()
        ssm.get_parameters.return_value = _make_ssm_response("ag", {
            "bucket-name": "b",
            "memory-id": "m",
            "bedrock-model-id": "model",
        })

        cfg = DeploymentConfig(ssm_client=ssm)
        result = cfg.configure_from_stack("ag")
        assert isinstance(result, RuntimeConfig)

    def test_optional_fields_none_when_not_provided(self):
        ssm = MagicMock()
        ssm.get_parameters.return_value = _make_ssm_response("ag", {
            "bucket-name": "b",
            "memory-id": "m",
            "bedrock-model-id": "model",
        })

        # Clear optional env vars
        env_clean = {
            k: v for k, v in os.environ.items()
            if k not in ("PROMPT_ROUTER_ARN", "SCHEDULE_GROUP_NAME", "SCHEDULER_ROLE_ARN")
        }
        with patch.dict(os.environ, env_clean, clear=True):
            cfg = DeploymentConfig(ssm_client=ssm)
            result = cfg.configure_from_stack("ag")

        assert result.prompt_router_arn is None
        assert result.schedule_group_name is None
        assert result.scheduler_role_arn is None
