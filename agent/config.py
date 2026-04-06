"""Deployment configuration for the OpenClaw AWS Agent.

Manages AgentCore Runtime deployment configuration, reading from
SSM Parameters or falling back to environment variables.

Requirements: 13.1, 13.2, 13.3, 13.4
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from agent.models import RuntimeConfig

logger = logging.getLogger(__name__)

# SSM parameter keys under /{agent_id}/config/
_SSM_KEYS = {
    "bucket-name": "s3_bucket",
    "memory-id": "memory_id",
    "bedrock-model-id": "bedrock_model_id",
    "prompt-router-arn": "prompt_router_arn",
    "schedule-group-name": "schedule_group_name",
    "scheduler-role-arn": "scheduler_role_arn",
    "agent-runtime-arn": "agent_runtime_arn",
    "scheduler-target-arn": "scheduler_target_arn",
}

# Environment variable fallback mapping
_ENV_VARS = {
    "s3_bucket": "IDENTITY_BUCKET",
    "memory_id": "MEMORY_ID",
    "bedrock_model_id": "BEDROCK_MODEL_ID",
    "prompt_router_arn": "PROMPT_ROUTER_ARN",
    "schedule_group_name": "SCHEDULE_GROUP_NAME",
    "scheduler_role_arn": "SCHEDULER_ROLE_ARN",
    "agent_runtime_arn": "AGENT_RUNTIME_ARN",
    "scheduler_target_arn": "SCHEDULER_TARGET_ARN",
}


class DeploymentConfig:
    """Reads runtime configuration from SSM Parameters or environment variables.

    Supports ``agentcore create``, ``agentcore dev``, and ``agentcore launch``
    workflows.  When running locally (``agentcore dev``), SSM may not be
    reachable — the class falls back to environment variables and logs a
    warning so the developer knows SSM-based configuration is unavailable.
    """

    def __init__(self, ssm_client: Optional[object] = None) -> None:
        self._ssm = ssm_client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def configure_from_stack(self, agent_id: str) -> RuntimeConfig:
        """Build a ``RuntimeConfig`` by reading SSM Parameters at
        ``/{agent_id}/config/*``, falling back to environment variables
        when SSM is unavailable.

        Parameters
        ----------
        agent_id:
            The agent identifier used as the SSM parameter path prefix
            and included in the returned config.

        Returns
        -------
        RuntimeConfig
            Validated runtime configuration.
        """
        ssm_values = self._read_ssm_parameters(agent_id)

        config_values: dict[str, str | None] = {}
        for field_name, env_var in _ENV_VARS.items():
            # Prefer SSM value, fall back to env var
            ssm_val = ssm_values.get(field_name)
            if ssm_val is not None:
                config_values[field_name] = ssm_val
            else:
                config_values[field_name] = os.environ.get(env_var)

        bedrock_region = os.environ.get("BEDROCK_REGION", "us-east-1")

        return RuntimeConfig(
            agent_id=agent_id,
            s3_bucket=config_values.get("s3_bucket") or "",
            bedrock_model_id=config_values.get("bedrock_model_id") or "",
            bedrock_region=bedrock_region,
            memory_id=config_values.get("memory_id") or "",
            prompt_router_arn=config_values.get("prompt_router_arn"),
            schedule_group_name=config_values.get("schedule_group_name"),
            scheduler_role_arn=config_values.get("scheduler_role_arn"),
            agent_runtime_arn=config_values.get("agent_runtime_arn"),
            scheduler_target_arn=config_values.get("scheduler_target_arn"),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_ssm_client(self):
        """Lazily create the SSM client if one was not injected."""
        if self._ssm is None:
            region = os.environ.get("BEDROCK_REGION") or os.environ.get("AWS_REGION", "us-east-1")
            self._ssm = boto3.client("ssm", region_name=region)
        return self._ssm

    def _read_ssm_parameters(self, agent_id: str) -> dict[str, str]:
        """Attempt to read all config parameters from SSM.

        Returns a dict mapping RuntimeConfig field names to their SSM
        values.  If SSM is unreachable, returns an empty dict and logs a
        warning (Req 13.3).
        """
        result: dict[str, str] = {}
        prefix = f"/{agent_id}/config"
        param_names = [f"{prefix}/{key}" for key in _SSM_KEYS]

        try:
            ssm = self._get_ssm_client()
            # GetParameters allows fetching up to 10 params in one call
            response = ssm.get_parameters(Names=param_names)

            for param in response.get("Parameters", []):
                # Extract the trailing key from the full name
                key = param["Name"].rsplit("/", 1)[-1]
                field_name = _SSM_KEYS.get(key)
                if field_name:
                    result[field_name] = param["Value"]

            invalid = response.get("InvalidParameters", [])
            if invalid:
                logger.info(
                    "SSM parameters not found (may not be provisioned yet): %s",
                    invalid,
                )

        except (BotoCoreError, ClientError) as exc:
            logger.warning(
                "SSM-based configuration is unavailable, falling back to "
                "environment variables: %s",
                exc,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Unexpected error reading SSM parameters, falling back to "
                "environment variables: %s",
                exc,
            )

        return result
