"""Smoke tests for agent/models.py to verify all validation rules."""

import pytest
from datetime import datetime, timezone
from uuid import uuid4

from agent.models import (
    IdentityBundle,
    MemoryContext,
    Turn,
    TaskItem,
    ScheduleItem,
    HeartbeatPayload,
    HeartbeatResult,
    BedrockModelConfig,
    RuntimeConfig,
    BootstrapResult,
    InfraStackProps,
    validate_agent_id,
)

NOW = datetime.now(timezone.utc)


# --- validate_agent_id ---

def test_valid_agent_id():
    assert validate_agent_id("my-agent-1") == "my-agent-1"


def test_agent_id_empty():
    with pytest.raises(ValueError, match="non-empty"):
        validate_agent_id("")


def test_agent_id_too_long():
    with pytest.raises(ValueError, match="at most 50"):
        validate_agent_id("a" * 51)


def test_agent_id_invalid_chars():
    with pytest.raises(ValueError, match="alphanumeric"):
        validate_agent_id("bad agent!")


# --- IdentityBundle ---

def test_identity_bundle_valid():
    ib = IdentityBundle(agent_id="a1", soul="s", agents="a", loaded_at=NOW)
    assert ib.soul == "s"


def test_identity_bundle_empty_soul():
    with pytest.raises(Exception):
        IdentityBundle(agent_id="a1", soul="", agents="a", loaded_at=NOW)


def test_identity_bundle_empty_agents():
    with pytest.raises(Exception):
        IdentityBundle(agent_id="a1", soul="s", agents="  ", loaded_at=NOW)


# --- TaskItem ---

def test_task_item_valid():
    t = TaskItem(id=uuid4(), title="Do stuff", status="pending", created_at=NOW)
    assert t.priority == "medium"


def test_task_item_pending_with_completed_at():
    with pytest.raises(Exception):
        TaskItem(id=uuid4(), title="X", status="pending", created_at=NOW, completed_at=NOW)


def test_task_item_title_too_long():
    with pytest.raises(Exception):
        TaskItem(id=uuid4(), title="x" * 501, status="pending", created_at=NOW)


def test_task_item_title_empty():
    with pytest.raises(Exception):
        TaskItem(id=uuid4(), title="", status="pending", created_at=NOW)


# --- ScheduleItem ---

def test_schedule_item_valid_cron():
    hp = HeartbeatPayload(type="heartbeat", task="t", agent_id="a1", user_id="u1")
    si = ScheduleItem(id=uuid4(), name="daily-check", cron_expression="cron(0 9 * * ? *)", payload=hp, created_at=NOW)
    assert si.status == "active"


def test_schedule_item_valid_rate():
    hp = HeartbeatPayload(type="heartbeat", task="t", agent_id="a1", user_id="u1")
    si = ScheduleItem(id=uuid4(), name="every-hour", cron_expression="rate(1 hour)", payload=hp, created_at=NOW)
    assert si.cron_expression == "rate(1 hour)"


def test_schedule_item_bad_name():
    hp = HeartbeatPayload(type="heartbeat", task="t", agent_id="a1", user_id="u1")
    with pytest.raises(Exception):
        ScheduleItem(id=uuid4(), name="Not Kebab", cron_expression="cron(0 9 * * ? *)", payload=hp, created_at=NOW)


def test_schedule_item_bad_cron():
    hp = HeartbeatPayload(type="heartbeat", task="t", agent_id="a1", user_id="u1")
    with pytest.raises(Exception):
        ScheduleItem(id=uuid4(), name="test", cron_expression="every 5 min", payload=hp, created_at=NOW)


# --- HeartbeatPayload ---

def test_heartbeat_payload_valid():
    hp = HeartbeatPayload(type="heartbeat", task="do it", agent_id="a1", user_id="u1")
    assert hp.type == "heartbeat"


def test_heartbeat_payload_empty_task():
    with pytest.raises(Exception):
        HeartbeatPayload(type="heartbeat", task="", agent_id="a1", user_id="u1")


def test_heartbeat_payload_empty_user_id():
    with pytest.raises(Exception):
        HeartbeatPayload(type="heartbeat", task="t", agent_id="a1", user_id="")


# --- BedrockModelConfig ---

def test_bedrock_routing_enabled_valid():
    bmc = BedrockModelConfig(routing_enabled=True, router_arn="arn:something", fallback_model_id="fb")
    assert bmc.routing_enabled


def test_bedrock_routing_enabled_no_arn():
    with pytest.raises(Exception):
        BedrockModelConfig(routing_enabled=True, fallback_model_id="fb")


def test_bedrock_routing_disabled_no_model():
    with pytest.raises(Exception):
        BedrockModelConfig(routing_enabled=False, fallback_model_id="fb")


def test_bedrock_routing_disabled_valid():
    bmc = BedrockModelConfig(routing_enabled=False, model_id="claude", fallback_model_id="fb")
    assert bmc.model_id == "claude"


# --- RuntimeConfig ---

def test_runtime_config_valid():
    rc = RuntimeConfig(agent_id="a1", s3_bucket="b", bedrock_model_id="m", bedrock_region="us-east-1", memory_id="mem")
    assert rc.agent_id == "a1"


def test_runtime_config_empty_bucket():
    with pytest.raises(Exception):
        RuntimeConfig(agent_id="a1", s3_bucket="", bedrock_model_id="m", bedrock_region="us-east-1", memory_id="mem")


# --- InfraStackProps ---

def test_infra_props_valid():
    isp = InfraStackProps(agent_id="a1", bedrock_model_id="m", bedrock_region="us-east-1", schedule_group_name="sg")
    assert isp.enable_prompt_routing is False
    assert isp.memory_id is None


def test_infra_props_with_memory_id_override():
    isp = InfraStackProps(agent_id="a1", bedrock_model_id="m", bedrock_region="us-east-1", memory_id="mem-override", schedule_group_name="sg")
    assert isp.memory_id == "mem-override"


def test_infra_props_routing_no_arn():
    with pytest.raises(Exception):
        InfraStackProps(agent_id="a1", bedrock_model_id="m", bedrock_region="us-east-1", enable_prompt_routing=True, schedule_group_name="sg")


def test_infra_props_routing_invalid_arn():
    with pytest.raises(Exception):
        InfraStackProps(agent_id="a1", bedrock_model_id="m", bedrock_region="us-east-1", enable_prompt_routing=True, prompt_router_arn="not-an-arn", schedule_group_name="sg")


def test_infra_props_routing_valid_arn():
    isp = InfraStackProps(
        agent_id="a1", bedrock_model_id="m", bedrock_region="us-east-1",
        enable_prompt_routing=True,
        prompt_router_arn="arn:aws:bedrock:us-east-1:123456789012:default-prompt-router/anthropic.claude:1",
        schedule_group_name="sg",
    )
    assert isp.enable_prompt_routing


# --- Simple models ---

def test_memory_context_defaults():
    mc = MemoryContext()
    assert mc.summaries == []
    assert mc.preferences == []
    assert mc.facts == []


def test_turn():
    t = Turn(role="user", content="hi", timestamp=NOW)
    assert t.role == "user"


def test_heartbeat_result():
    hr = HeartbeatResult(status="completed", output="done")
    assert hr.output == "done"


def test_bootstrap_result():
    br = BootstrapResult(files_initialized=["IDENTITY.md"], bootstrap_deleted=True)
    assert br.bootstrap_deleted
