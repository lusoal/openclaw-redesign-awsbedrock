"""Property-based tests for the schedule_task tool (Properties 11, 12, 13, 14).

Uses moto-mocked S3 and MagicMock for the EventBridge Scheduler client.
Hypothesis strategies generate random kebab-case names, cron expressions, etc.

Property 11: Schedule serialization round-trip
Property 12: Schedule name prefixing
Property 13: Schedule deletion consistency
Property 14: Schedule list completeness
"""

from __future__ import annotations

import json
import uuid
from unittest.mock import MagicMock

import boto3
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from moto import mock_aws

from agent.tools.schedules import ScheduleManager

BUCKET = "prop-sched-bucket"
AGENT_ID_PREFIX = "agent"
USER_ID = "prop-user"
SCHEDULE_GROUP = "agent-schedules"
AGENT_RUNTIME_ARN = "arn:aws:agentcore:us-east-1:123456789012:runtime/test"
SCHEDULER_ROLE_ARN = "arn:aws:iam::123456789012:role/scheduler-role"
SCHEDULE_ARN_TEMPLATE = "arn:aws:scheduler:us-east-1:123456789012:schedule/agent-schedules/{name}"

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Kebab-case segment: 1-10 lowercase alphanumeric chars
_segment = st.from_regex(r"[a-z0-9]{1,10}", fullmatch=True)

# Kebab-case name: 1-4 segments joined by hyphens (max ~44 chars, well under 100)
kebab_name_st = st.lists(_segment, min_size=1, max_size=4).map(lambda segs: "-".join(segs))

# Valid agent_id: lowercase alphanumeric + hyphens, 1-20 chars
agent_id_st = st.from_regex(r"[a-z][a-z0-9-]{0,19}", fullmatch=True)

# Valid cron expressions
cron_st = st.sampled_from([
    "cron(0 9 * * ? *)",
    "cron(30 12 * * ? *)",
    "cron(0 0 1 * ? *)",
    "rate(5 minutes)",
    "rate(1 hour)",
    "rate(2 days)",
])

# Non-empty description
description_st = st.text(min_size=1, max_size=100).filter(lambda s: s.strip())

# Non-empty task prompt
task_prompt_st = st.text(min_size=1, max_size=200).filter(lambda s: s.strip())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_scheduler_mock() -> MagicMock:
    """Return a MagicMock for the EventBridge Scheduler client."""
    mock = MagicMock()
    mock.create_schedule.return_value = {
        "ScheduleArn": SCHEDULE_ARN_TEMPLATE.format(name="mock-schedule"),
    }
    mock.delete_schedule.return_value = {}
    return mock


def _read_s3_schedules(client, agent_id: str) -> list[dict]:
    """Read the raw schedule list from S3."""
    key = f"agents/{agent_id}/schedules.json"
    obj = client.get_object(Bucket=BUCKET, Key=key)
    return json.loads(obj["Body"].read()).get("schedules", [])



# ---------------------------------------------------------------------------
# Property 12: Schedule name prefixing
# **Validates: Requirement 9.8**
# ---------------------------------------------------------------------------


class TestScheduleNamePrefixing:
    """For any agent_id and schedule name, the EventBridge schedule name
    created by the SchedulerManager shall be prefixed with the agent_id."""

    @given(
        agent_id=agent_id_st,
        name=kebab_name_st,
        cron=cron_st,
        task_prompt=task_prompt_st,
    )
    @settings(max_examples=50, deadline=None)
    def test_eventbridge_name_prefixed_with_agent_id(
        self, agent_id: str, name: str, cron: str, task_prompt: str
    ):
        """**Validates: Requirement 9.8**"""
        with mock_aws():
            client = boto3.client("s3", region_name="us-east-1")
            client.create_bucket(Bucket=BUCKET)
            scheduler = _make_scheduler_mock()

            mgr = ScheduleManager(
                s3_client=client,
                scheduler_client=scheduler,
                bucket=BUCKET,
                schedule_group=SCHEDULE_GROUP,
                agent_runtime_arn=AGENT_RUNTIME_ARN,
                scheduler_role_arn=SCHEDULER_ROLE_ARN,
            )

            result = mgr.schedule_task(
                "create", agent_id, USER_ID,
                name=name,
                description="prop test",
                cron_expression=cron,
                task_prompt=task_prompt,
            )

            assert "created" in result.lower()

            # Verify the EventBridge schedule name is prefixed with agent_id
            scheduler.create_schedule.assert_called_once()
            call_kwargs = scheduler.create_schedule.call_args[1]
            eb_name = call_kwargs["Name"]

            assert eb_name == f"{agent_id}-{name}"
            assert eb_name.startswith(f"{agent_id}-")


# ---------------------------------------------------------------------------
# Property 11: Schedule serialization round-trip
# **Validates: Requirement 17.6**
# ---------------------------------------------------------------------------


class TestScheduleSerializationRoundTrip:
    """For any valid list of ScheduleItems, serializing to JSON then
    deserializing shall produce an equivalent list with identical field values."""

    @given(
        names=st.lists(kebab_name_st, min_size=1, max_size=5, unique=True),
        crons=st.data(),
        descriptions=st.data(),
        task_prompts=st.data(),
    )
    @settings(max_examples=50, deadline=None)
    def test_round_trip_preserves_all_fields(
        self,
        names: list[str],
        crons,
        descriptions,
        task_prompts,
    ):
        """**Validates: Requirement 17.6**"""
        agent_id = "roundtrip-agent"

        with mock_aws():
            client = boto3.client("s3", region_name="us-east-1")
            client.create_bucket(Bucket=BUCKET)
            scheduler = _make_scheduler_mock()

            mgr = ScheduleManager(
                s3_client=client,
                scheduler_client=scheduler,
                bucket=BUCKET,
                schedule_group=SCHEDULE_GROUP,
                agent_runtime_arn=AGENT_RUNTIME_ARN,
                scheduler_role_arn=SCHEDULER_ROLE_ARN,
            )

            drawn_crons = [crons.draw(cron_st) for _ in names]
            drawn_descs = [descriptions.draw(description_st) for _ in names]
            drawn_prompts = [task_prompts.draw(task_prompt_st) for _ in names]

            # Create all schedules
            for name, cron, desc, prompt in zip(names, drawn_crons, drawn_descs, drawn_prompts):
                result = mgr.schedule_task(
                    "create", agent_id, USER_ID,
                    name=name,
                    description=desc,
                    cron_expression=cron,
                    task_prompt=prompt,
                )
                assert "created" in result.lower()

            # Read raw JSON from S3
            raw = _read_s3_schedules(client, agent_id)
            assert len(raw) == len(names)

            # Re-serialize and re-deserialize (round-trip)
            key = f"agents/{agent_id}/schedules.json"
            payload = json.dumps({"schedules": raw}, indent=2)
            client.put_object(
                Bucket=BUCKET, Key=key,
                Body=payload, ContentType="application/json",
            )
            round_tripped = _read_s3_schedules(client, agent_id)

            # Every field must survive the round-trip
            assert len(round_tripped) == len(raw)
            for original, restored in zip(raw, round_tripped):
                assert original["id"] == restored["id"]
                assert original["name"] == restored["name"]
                assert original["description"] == restored["description"]
                assert original["cron_expression"] == restored["cron_expression"]
                assert original["status"] == restored["status"]
                assert original["eventbridge_rule_arn"] == restored["eventbridge_rule_arn"]
                assert original["created_at"] == restored["created_at"]
                assert original["last_triggered_at"] == restored["last_triggered_at"]
                assert original["payload"] == restored["payload"]


# ---------------------------------------------------------------------------
# Property 14: Schedule list completeness
# **Validates: Requirement 9.3**
# ---------------------------------------------------------------------------


class TestScheduleListCompleteness:
    """For any set of active schedules persisted in S3, calling schedule_task
    with action 'list' shall return every active schedule with its cron
    expression, name, description, and last triggered time."""

    @given(
        names=st.lists(kebab_name_st, min_size=1, max_size=5, unique=True),
        crons=st.data(),
        descriptions=st.data(),
        task_prompts=st.data(),
    )
    @settings(max_examples=50, deadline=None)
    def test_list_returns_every_active_schedule(
        self,
        names: list[str],
        crons,
        descriptions,
        task_prompts,
    ):
        """**Validates: Requirement 9.3**"""
        agent_id = "list-agent"

        with mock_aws():
            client = boto3.client("s3", region_name="us-east-1")
            client.create_bucket(Bucket=BUCKET)
            scheduler = _make_scheduler_mock()

            mgr = ScheduleManager(
                s3_client=client,
                scheduler_client=scheduler,
                bucket=BUCKET,
                schedule_group=SCHEDULE_GROUP,
                agent_runtime_arn=AGENT_RUNTIME_ARN,
                scheduler_role_arn=SCHEDULER_ROLE_ARN,
            )

            drawn_crons = [crons.draw(cron_st) for _ in names]
            drawn_descs = [descriptions.draw(description_st) for _ in names]
            drawn_prompts = [task_prompts.draw(task_prompt_st) for _ in names]

            # Create all schedules
            for name, cron, desc, prompt in zip(names, drawn_crons, drawn_descs, drawn_prompts):
                mgr.schedule_task(
                    "create", agent_id, USER_ID,
                    name=name,
                    description=desc,
                    cron_expression=cron,
                    task_prompt=prompt,
                )

            # List schedules
            result = mgr.schedule_task("list", agent_id, USER_ID)

            # Every schedule's name, cron, and description must appear
            for name, cron, desc in zip(names, drawn_crons, drawn_descs):
                assert name in result, f"Schedule name '{name}' not found in list output"
                assert cron in result, f"Cron '{cron}' not found in list output"
                assert desc in result, f"Description '{desc}' not found in list output"

            # "last run" indicator must appear for each schedule
            assert result.count("last run:") == len(names)

            # Count header shows correct number
            assert f"Active schedules ({len(names)}):" in result


# ---------------------------------------------------------------------------
# Property 13: Schedule deletion consistency
# **Validates: Requirement 9.4**
# ---------------------------------------------------------------------------


class TestScheduleDeletionConsistency:
    """For any existing schedule, calling schedule_task with action 'delete'
    shall remove both the EventBridge Scheduler rule and the corresponding
    entry from schedules.json in S3."""

    @given(
        names=st.lists(kebab_name_st, min_size=1, max_size=5, unique=True),
        target_index=st.data(),
        crons=st.data(),
    )
    @settings(max_examples=50, deadline=None)
    def test_delete_removes_eventbridge_rule_and_s3_entry(
        self,
        names: list[str],
        target_index,
        crons,
    ):
        """**Validates: Requirement 9.4**"""
        agent_id = "delete-agent"

        with mock_aws():
            client = boto3.client("s3", region_name="us-east-1")
            client.create_bucket(Bucket=BUCKET)
            scheduler = _make_scheduler_mock()

            mgr = ScheduleManager(
                s3_client=client,
                scheduler_client=scheduler,
                bucket=BUCKET,
                schedule_group=SCHEDULE_GROUP,
                agent_runtime_arn=AGENT_RUNTIME_ARN,
                scheduler_role_arn=SCHEDULER_ROLE_ARN,
            )

            drawn_crons = [crons.draw(cron_st) for _ in names]

            # Create all schedules
            for name, cron in zip(names, drawn_crons):
                mgr.schedule_task(
                    "create", agent_id, USER_ID,
                    name=name,
                    description=f"desc-{name}",
                    cron_expression=cron,
                    task_prompt=f"task-{name}",
                )

            # Pick a random schedule to delete
            idx = target_index.draw(st.integers(min_value=0, max_value=len(names) - 1))
            raw_before = _read_s3_schedules(client, agent_id)
            target_id = raw_before[idx]["id"]
            target_name = raw_before[idx]["name"]

            # Reset mock call tracking so we only see the delete call
            scheduler.reset_mock()

            # Delete the target schedule
            result = mgr.schedule_task(
                "delete", agent_id, USER_ID, schedule_id=target_id[:8]
            )
            assert "deleted" in result.lower()

            # 1) EventBridge delete_schedule was called with the correct prefixed name
            scheduler.delete_schedule.assert_called_once_with(
                Name=f"{agent_id}-{target_name}",
                GroupName=SCHEDULE_GROUP,
            )

            # 2) S3 entry is removed
            raw_after = _read_s3_schedules(client, agent_id)
            remaining_ids = [s["id"] for s in raw_after]
            assert target_id not in remaining_ids

            # 3) List size reduced by exactly one
            assert len(raw_after) == len(names) - 1

            # 4) All other schedules are still present
            for s in raw_before:
                if s["id"] != target_id:
                    assert s["id"] in remaining_ids
