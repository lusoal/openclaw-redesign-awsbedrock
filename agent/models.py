"""Core data models for the OpenClaw AWS Agent.

All models use Pydantic v2 for validation. Implements validation rules
from Requirements 16.1–16.7, 1.5, 15.8.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Annotated, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Reusable agent_id validator (Req 1.5, 15.8)
# ---------------------------------------------------------------------------
_AGENT_ID_PATTERN = re.compile(r"^[a-zA-Z0-9-]+\Z")


def validate_agent_id(value: str) -> str:
    """Validate agent_id: non-empty, alphanumeric + hyphens, max 50 chars."""
    if not value or not value.strip():
        raise ValueError("agent_id must be non-empty")
    if len(value) > 50:
        raise ValueError("agent_id must be at most 50 characters")
    if not _AGENT_ID_PATTERN.match(value):
        raise ValueError(
            "agent_id must contain only alphanumeric characters and hyphens"
        )
    return value


# ---------------------------------------------------------------------------
# Kebab-case pattern for schedule names
# ---------------------------------------------------------------------------
_KEBAB_CASE_PATTERN = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")

# ---------------------------------------------------------------------------
# EventBridge cron/rate expression patterns
# ---------------------------------------------------------------------------
_CRON_PATTERN = re.compile(r"^cron\(.+\)$")
_RATE_PATTERN = re.compile(r"^rate\(\d+\s+(minute|minutes|hour|hours|day|days)\)$")
_AT_PATTERN = re.compile(r"^at\(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\)$")

# ---------------------------------------------------------------------------
# ARN pattern
# ---------------------------------------------------------------------------
_ARN_PATTERN = re.compile(r"^arn:aws:[a-zA-Z0-9-]+:[a-z0-9-]*:\d{12}:.+$")


# ---------------------------------------------------------------------------
# IdentityBundle (Req 16.1)
# ---------------------------------------------------------------------------
class IdentityBundle(BaseModel):
    """Complete set of identity files loaded for an agent session."""

    agent_id: str
    soul: str
    agents: str
    identity: str = ""
    user_profile: str = ""
    durable_memory: str = ""
    bootstrap: Optional[str] = None
    loaded_at: datetime

    @field_validator("agent_id")
    @classmethod
    def _validate_agent_id(cls, v: str) -> str:
        return validate_agent_id(v)

    @field_validator("soul")
    @classmethod
    def _soul_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("soul must be non-empty")
        return v

    @field_validator("agents")
    @classmethod
    def _agents_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("agents must be non-empty")
        return v


# ---------------------------------------------------------------------------
# Turn
# ---------------------------------------------------------------------------
class Turn(BaseModel):
    """A single conversation turn."""

    role: str
    content: str
    timestamp: datetime


# ---------------------------------------------------------------------------
# MemoryContext
# ---------------------------------------------------------------------------
class MemoryContext(BaseModel):
    """Memory state retrieved from AgentCore Memory at session start."""

    summaries: list[str] = Field(default_factory=list)
    preferences: list[str] = Field(default_factory=list)
    facts: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# TaskItem (Req 16.2)
# ---------------------------------------------------------------------------
class TaskItem(BaseModel):
    """A single task in the user's personal task list."""

    id: UUID
    title: Annotated[str, Field(min_length=1, max_length=500)]
    status: Literal["pending", "completed"]
    priority: Literal["low", "medium", "high"] = "medium"
    created_at: datetime
    completed_at: Optional[datetime] = None

    @model_validator(mode="after")
    def _completed_at_consistency(self) -> TaskItem:
        if self.status == "pending" and self.completed_at is not None:
            raise ValueError("completed_at must be None when status is 'pending'")
        return self


# ---------------------------------------------------------------------------
# HeartbeatPayload (Req 16.4)
# ---------------------------------------------------------------------------
class HeartbeatPayload(BaseModel):
    """Payload sent by EventBridge Scheduler when invoking the agent."""

    type: Literal["heartbeat"]
    task: Annotated[str, Field(min_length=1)]
    agent_id: str
    user_id: str

    @field_validator("agent_id")
    @classmethod
    def _validate_agent_id(cls, v: str) -> str:
        return validate_agent_id(v)

    @field_validator("user_id")
    @classmethod
    def _user_id_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("user_id must be non-empty")
        return v


# ---------------------------------------------------------------------------
# ScheduleItem (Req 16.3)
# ---------------------------------------------------------------------------
class ScheduleItem(BaseModel):
    """A scheduled agent invocation."""

    id: UUID
    name: Annotated[str, Field(min_length=1, max_length=100)]
    cron_expression: str
    schedule_type: Literal["recurring", "one-time"] = "recurring"
    description: str = ""
    payload: HeartbeatPayload
    eventbridge_rule_arn: Optional[str] = None
    status: Literal["active", "paused"] = "active"
    created_at: datetime
    last_triggered_at: Optional[datetime] = None
    next_trigger_at: Optional[datetime] = None

    @field_validator("name")
    @classmethod
    def _name_kebab_case(cls, v: str) -> str:
        if not _KEBAB_CASE_PATTERN.match(v):
            raise ValueError(
                "name must be kebab-case (lowercase alphanumeric separated by hyphens)"
            )
        return v

    @field_validator("cron_expression")
    @classmethod
    def _valid_cron_expression(cls, v: str) -> str:
        if not _CRON_PATTERN.match(v) and not _RATE_PATTERN.match(v) and not _AT_PATTERN.match(v):
            raise ValueError(
                "cron_expression must be a valid EventBridge cron(), rate(), or at() expression"
            )
        return v


# ---------------------------------------------------------------------------
# HeartbeatResult
# ---------------------------------------------------------------------------
class HeartbeatResult(BaseModel):
    """Result of processing a heartbeat invocation."""

    status: str
    output: Optional[str] = None
    error_message: Optional[str] = None


# ---------------------------------------------------------------------------
# BedrockModelConfig (Req 16.5)
# ---------------------------------------------------------------------------
class BedrockModelConfig(BaseModel):
    """Configuration for Bedrock model with optional Intelligent Prompt Routing."""

    routing_enabled: bool
    model_id: Optional[str] = None
    router_arn: Optional[str] = None
    fallback_model_id: Annotated[str, Field(min_length=1)]

    @model_validator(mode="after")
    def _routing_consistency(self) -> BedrockModelConfig:
        if self.routing_enabled:
            if not self.router_arn or not self.router_arn.strip():
                raise ValueError(
                    "router_arn must be non-empty when routing_enabled is True"
                )
        else:
            if not self.model_id or not self.model_id.strip():
                raise ValueError(
                    "model_id must be non-empty when routing_enabled is False"
                )
        return self


# ---------------------------------------------------------------------------
# RuntimeConfig (Req 16.6)
# ---------------------------------------------------------------------------
class RuntimeConfig(BaseModel):
    """Configuration for AgentCore Runtime deployment."""

    agent_id: Annotated[str, Field(min_length=1)]
    s3_bucket: Annotated[str, Field(min_length=1)]
    bedrock_model_id: Annotated[str, Field(min_length=1)]
    bedrock_region: Annotated[str, Field(min_length=1)]
    memory_id: Annotated[str, Field(min_length=1)]
    prompt_router_arn: Optional[str] = None
    schedule_group_name: Optional[str] = None
    scheduler_role_arn: Optional[str] = None
    agent_runtime_arn: Optional[str] = None
    scheduler_target_arn: Optional[str] = None
    heartbeat_queue_url: Optional[str] = None
    gateway_url: Optional[str] = None
    gateway_id: Optional[str] = None
    custom_tool_role_arn: Optional[str] = None

    @field_validator("agent_id")
    @classmethod
    def _validate_agent_id(cls, v: str) -> str:
        return validate_agent_id(v)


# ---------------------------------------------------------------------------
# BootstrapResult
# ---------------------------------------------------------------------------
class BootstrapResult(BaseModel):
    """Result of executing the bootstrap ritual."""

    files_initialized: list[str] = Field(default_factory=list)
    bootstrap_deleted: bool = False


# ---------------------------------------------------------------------------
# InfraStackProps (Req 16.7)
# ---------------------------------------------------------------------------
class InfraStackProps(BaseModel):
    """Configuration properties for the CDK InfrastructureStack."""

    agent_id: str
    bedrock_model_id: str
    bedrock_region: str
    enable_prompt_routing: bool = False
    prompt_router_arn: Optional[str] = None
    memory_id: Optional[str] = None
    schedule_group_name: Annotated[str, Field(min_length=1)]

    @field_validator("agent_id")
    @classmethod
    def _validate_agent_id(cls, v: str) -> str:
        return validate_agent_id(v)

    @model_validator(mode="after")
    def _prompt_router_required_when_enabled(self) -> InfraStackProps:
        if self.enable_prompt_routing:
            if not self.prompt_router_arn or not self.prompt_router_arn.strip():
                raise ValueError(
                    "prompt_router_arn must be provided when enable_prompt_routing is True"
                )
            if not _ARN_PATTERN.match(self.prompt_router_arn):
                raise ValueError(
                    "prompt_router_arn must be a valid ARN format"
                )
        return self
