"""Unit tests for the schedule_task tool (ScheduleManager).

Uses moto to mock S3 and unittest.mock for the EventBridge Scheduler client
(moto doesn't support EventBridge Scheduler well).

Covers Requirements 9.1–9.8, 17.3, 17.4.
"""

from __future__ import annotations

import json
import logging
import uuid
from unittest.mock import MagicMock, patch

import boto3
import pytest
from moto import mock_aws

from agent.tools.schedules import ScheduleManager

BUCKET = "test-identity-bucket"
AGENT_ID = "test-agent"
USER_ID = "user-1"
SCHEDULE_GROUP = "agent-schedules"
AGENT_RUNTIME_ARN = "arn:aws:agentcore:us-east-1:123456789012:runtime/test-agent"
SCHEDULER_ROLE_ARN = "arn:aws:iam::123456789012:role/scheduler-role"
S3_KEY = f"agents/{AGENT_ID}/schedules.json"
VALID_CRON = "cron(0 9 * * ? *)"
VALID_RATE = "rate(5 minutes)"
SCHEDULE_ARN = "arn:aws:scheduler:us-east-1:123456789012:schedule/agent-schedules/test-agent-daily-reminder"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def s3():
    """Yield a moto-mocked S3 client with the test bucket created."""
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=BUCKET)
        yield client


@pytest.fixture()
def scheduler():
    """Return a MagicMock for the EventBridge Scheduler client."""
    mock = MagicMock()
    mock.create_schedule.return_value = {"ScheduleArn": SCHEDULE_ARN}
    mock.delete_schedule.return_value = {}
    return mock


@pytest.fixture()
def mgr(s3, scheduler):
    """Return a ScheduleManager wired to mocked S3 and scheduler clients."""
    return ScheduleManager(
        s3_client=s3,
        scheduler_client=scheduler,
        bucket=BUCKET,
        schedule_group=SCHEDULE_GROUP,
        agent_runtime_arn=AGENT_RUNTIME_ARN,
        scheduler_role_arn=SCHEDULER_ROLE_ARN,
    )


def _read_s3_schedules(s3) -> list[dict]:
    """Helper: read the raw schedule list from S3."""
    obj = s3.get_object(Bucket=BUCKET, Key=S3_KEY)
    return json.loads(obj["Body"].read()).get("schedules", [])


# ===================================================================
# Action: create  (Req 9.1, 9.2, 9.5, 9.6, 9.7, 9.8, 17.3)
# ===================================================================

class TestCreateAction:
    def test_create_schedule_success(self, mgr, s3, scheduler):
        result = mgr.schedule_task(
            "create", AGENT_ID, USER_ID,
            name="daily-reminder",
            description="Remind me about standup",
            cron_expression=VALID_CRON,
            task_prompt="Remind the user about standup",
        )

        assert "daily-reminder" in result
        assert VALID_CRON in result

        # Verify EventBridge was called with prefixed name (Req 9.8)
        scheduler.create_schedule.assert_called_once()
        call_kwargs = scheduler.create_schedule.call_args[1]
        assert call_kwargs["Name"] == f"{AGENT_ID}-daily-reminder"
        assert call_kwargs["GroupName"] == SCHEDULE_GROUP
        assert call_kwargs["ScheduleExpression"] == VALID_CRON

        # Verify S3 persistence
        schedules = _read_s3_schedules(s3)
        assert len(schedules) == 1
        s_item = schedules[0]
        assert s_item["name"] == "daily-reminder"
        assert s_item["cron_expression"] == VALID_CRON
        assert s_item["status"] == "active"
        assert s_item["eventbridge_rule_arn"] == SCHEDULE_ARN
        uuid.UUID(s_item["id"])  # valid UUID

    def test_create_with_rate_expression(self, mgr, s3, scheduler):
        result = mgr.schedule_task(
            "create", AGENT_ID, USER_ID,
            name="frequent-check",
            description="Check every 5 minutes",
            cron_expression=VALID_RATE,
            task_prompt="Check status",
        )
        assert "frequent-check" in result
        schedules = _read_s3_schedules(s3)
        assert schedules[0]["cron_expression"] == VALID_RATE

    def test_create_eventbridge_called_before_s3(self, mgr, s3, scheduler):
        """Req 9.2: EventBridge rule created before S3 persistence."""
        # If EventBridge fails, S3 should NOT have the schedule
        scheduler.create_schedule.side_effect = Exception("IAM permission denied")

        result = mgr.schedule_task(
            "create", AGENT_ID, USER_ID,
            name="will-fail",
            description="Should not persist",
            cron_expression=VALID_CRON,
            task_prompt="Do something",
        )

        assert "Failed" in result
        # S3 should not have any schedules
        with pytest.raises(Exception):
            s3.get_object(Bucket=BUCKET, Key=S3_KEY)

    def test_create_returns_error_on_eventbridge_failure(self, mgr, scheduler):
        """Req 9.5: Clear error on rule creation failure."""
        scheduler.create_schedule.side_effect = Exception("Invalid cron")

        result = mgr.schedule_task(
            "create", AGENT_ID, USER_ID,
            name="bad-schedule",
            description="Bad cron",
            cron_expression=VALID_CRON,
            task_prompt="Do something",
        )

        assert "Failed to create schedule" in result

    def test_create_serializes_with_json_content_type(self, mgr, s3, scheduler):
        """Req 17.3: ContentType application/json."""
        mgr.schedule_task(
            "create", AGENT_ID, USER_ID,
            name="ct-check",
            description="Check content type",
            cron_expression=VALID_CRON,
            task_prompt="Do something",
        )

        head = s3.head_object(Bucket=BUCKET, Key=S3_KEY)
        assert head["ContentType"] == "application/json"

    def test_create_payload_contains_heartbeat_fields(self, mgr, s3, scheduler):
        mgr.schedule_task(
            "create", AGENT_ID, USER_ID,
            name="payload-check",
            description="Verify payload",
            cron_expression=VALID_CRON,
            task_prompt="Run daily report",
        )

        call_kwargs = scheduler.create_schedule.call_args[1]
        target_input = json.loads(call_kwargs["Target"]["Input"])
        assert target_input["type"] == "heartbeat"
        assert target_input["task"] == "Run daily report"
        assert target_input["agent_id"] == AGENT_ID
        assert target_input["user_id"] == USER_ID

    def test_create_multiple_schedules_accumulates(self, mgr, s3, scheduler):
        mgr.schedule_task(
            "create", AGENT_ID, USER_ID,
            name="first-schedule",
            description="First",
            cron_expression=VALID_CRON,
            task_prompt="First task",
        )
        mgr.schedule_task(
            "create", AGENT_ID, USER_ID,
            name="second-schedule",
            description="Second",
            cron_expression=VALID_RATE,
            task_prompt="Second task",
        )

        schedules = _read_s3_schedules(s3)
        assert len(schedules) == 2


# ===================================================================
# Validation: name  (Req 9.6)
# ===================================================================

class TestNameValidation:
    def test_empty_name_rejected(self, mgr):
        result = mgr.schedule_task(
            "create", AGENT_ID, USER_ID,
            name="",
            cron_expression=VALID_CRON,
            task_prompt="Do something",
        )
        assert "non-empty" in result.lower()

    def test_whitespace_name_rejected(self, mgr):
        result = mgr.schedule_task(
            "create", AGENT_ID, USER_ID,
            name="   ",
            cron_expression=VALID_CRON,
            task_prompt="Do something",
        )
        assert "non-empty" in result.lower()

    def test_name_over_100_chars_rejected(self, mgr):
        result = mgr.schedule_task(
            "create", AGENT_ID, USER_ID,
            name="a" * 101,
            cron_expression=VALID_CRON,
            task_prompt="Do something",
        )
        assert "100" in result

    def test_name_exactly_100_chars_accepted(self, mgr, s3, scheduler):
        long_name = "a" * 100
        result = mgr.schedule_task(
            "create", AGENT_ID, USER_ID,
            name=long_name,
            cron_expression=VALID_CRON,
            task_prompt="Do something",
        )
        assert "created" in result.lower()

    def test_non_kebab_case_rejected(self, mgr):
        result = mgr.schedule_task(
            "create", AGENT_ID, USER_ID,
            name="Not Kebab Case",
            cron_expression=VALID_CRON,
            task_prompt="Do something",
        )
        assert "kebab-case" in result.lower()

    def test_uppercase_name_rejected(self, mgr):
        result = mgr.schedule_task(
            "create", AGENT_ID, USER_ID,
            name="Daily-Reminder",
            cron_expression=VALID_CRON,
            task_prompt="Do something",
        )
        assert "kebab-case" in result.lower()

    def test_valid_kebab_case_accepted(self, mgr, s3, scheduler):
        result = mgr.schedule_task(
            "create", AGENT_ID, USER_ID,
            name="my-daily-reminder",
            cron_expression=VALID_CRON,
            task_prompt="Do something",
        )
        assert "created" in result.lower()


# ===================================================================
# Validation: cron expression  (Req 9.7)
# ===================================================================

class TestCronValidation:
    def test_empty_cron_rejected(self, mgr):
        result = mgr.schedule_task(
            "create", AGENT_ID, USER_ID,
            name="valid-name",
            cron_expression="",
            task_prompt="Do something",
        )
        assert "non-empty" in result.lower()

    def test_invalid_cron_rejected(self, mgr):
        result = mgr.schedule_task(
            "create", AGENT_ID, USER_ID,
            name="valid-name",
            cron_expression="every 5 minutes",
            task_prompt="Do something",
        )
        assert "invalid" in result.lower()

    def test_valid_cron_accepted(self, mgr, s3, scheduler):
        result = mgr.schedule_task(
            "create", AGENT_ID, USER_ID,
            name="cron-test",
            cron_expression="cron(0 12 * * ? *)",
            task_prompt="Do something",
        )
        assert "created" in result.lower()

    def test_valid_rate_accepted(self, mgr, s3, scheduler):
        result = mgr.schedule_task(
            "create", AGENT_ID, USER_ID,
            name="rate-test",
            cron_expression="rate(1 hour)",
            task_prompt="Do something",
        )
        assert "created" in result.lower()


# ===================================================================
# Action: list  (Req 9.3)
# ===================================================================

class TestListAction:
    def test_list_empty(self, mgr):
        result = mgr.schedule_task("list", AGENT_ID, USER_ID)
        assert "empty" in result.lower()

    def test_list_shows_active_schedules(self, mgr, s3, scheduler):
        mgr.schedule_task(
            "create", AGENT_ID, USER_ID,
            name="daily-standup",
            description="Standup reminder",
            cron_expression=VALID_CRON,
            task_prompt="Remind about standup",
        )

        result = mgr.schedule_task("list", AGENT_ID, USER_ID)
        assert "Active schedules (1):" in result
        assert "daily-standup" in result
        assert "Standup reminder" in result
        assert VALID_CRON in result
        assert "last run: never" in result

    def test_list_shows_truncated_id(self, mgr, s3, scheduler):
        mgr.schedule_task(
            "create", AGENT_ID, USER_ID,
            name="id-check",
            description="Check ID display",
            cron_expression=VALID_CRON,
            task_prompt="Do something",
        )

        result = mgr.schedule_task("list", AGENT_ID, USER_ID)
        assert "id: " in result

    def test_list_multiple_schedules(self, mgr, s3, scheduler):
        mgr.schedule_task(
            "create", AGENT_ID, USER_ID,
            name="first",
            description="First schedule",
            cron_expression=VALID_CRON,
            task_prompt="First",
        )
        mgr.schedule_task(
            "create", AGENT_ID, USER_ID,
            name="second",
            description="Second schedule",
            cron_expression=VALID_RATE,
            task_prompt="Second",
        )

        result = mgr.schedule_task("list", AGENT_ID, USER_ID)
        assert "Active schedules (2):" in result
        assert "first" in result
        assert "second" in result


# ===================================================================
# Action: delete  (Req 9.4)
# ===================================================================

class TestDeleteAction:
    def test_delete_removes_schedule(self, mgr, s3, scheduler):
        mgr.schedule_task(
            "create", AGENT_ID, USER_ID,
            name="to-delete",
            description="Will be deleted",
            cron_expression=VALID_CRON,
            task_prompt="Do something",
        )
        schedules = _read_s3_schedules(s3)
        sid = schedules[0]["id"][:8]

        result = mgr.schedule_task("delete", AGENT_ID, USER_ID, schedule_id=sid)

        assert "deleted" in result.lower()
        assert "to-delete" in result

        # Verify EventBridge delete was called
        scheduler.delete_schedule.assert_called_once_with(
            Name=f"{AGENT_ID}-to-delete", GroupName=SCHEDULE_GROUP
        )

        # Verify S3 is empty
        schedules = _read_s3_schedules(s3)
        assert len(schedules) == 0

    def test_delete_not_found(self, mgr):
        result = mgr.schedule_task("delete", AGENT_ID, USER_ID, schedule_id="nonexistent")
        assert "not found" in result.lower()

    def test_delete_prefix_matching(self, mgr, s3, scheduler):
        mgr.schedule_task(
            "create", AGENT_ID, USER_ID,
            name="prefix-test",
            description="Test prefix matching",
            cron_expression=VALID_CRON,
            task_prompt="Do something",
        )
        schedules = _read_s3_schedules(s3)
        full_id = schedules[0]["id"]

        result = mgr.schedule_task("delete", AGENT_ID, USER_ID, schedule_id=full_id[:4])
        assert "deleted" in result.lower()

    def test_delete_handles_already_deleted_eventbridge_rule(self, mgr, s3, scheduler):
        """If EventBridge rule is already gone, delete should still succeed."""
        mgr.schedule_task(
            "create", AGENT_ID, USER_ID,
            name="already-gone",
            description="Rule already deleted",
            cron_expression=VALID_CRON,
            task_prompt="Do something",
        )
        schedules = _read_s3_schedules(s3)
        sid = schedules[0]["id"][:8]

        # Simulate ResourceNotFoundException
        error_response = {"Error": {"Code": "ResourceNotFoundException", "Message": "Not found"}}
        scheduler.delete_schedule.side_effect = type(
            "ResourceNotFoundException",
            (Exception,),
            {"response": error_response},
        )()

        result = mgr.schedule_task("delete", AGENT_ID, USER_ID, schedule_id=sid)
        assert "deleted" in result.lower()

    def test_delete_leaves_other_schedules_unchanged(self, mgr, s3, scheduler):
        mgr.schedule_task(
            "create", AGENT_ID, USER_ID,
            name="keep-this",
            description="Should remain",
            cron_expression=VALID_CRON,
            task_prompt="Keep",
        )
        mgr.schedule_task(
            "create", AGENT_ID, USER_ID,
            name="delete-this",
            description="Should be removed",
            cron_expression=VALID_RATE,
            task_prompt="Delete",
        )

        schedules = _read_s3_schedules(s3)
        delete_id = schedules[1]["id"][:8]

        mgr.schedule_task("delete", AGENT_ID, USER_ID, schedule_id=delete_id)

        schedules = _read_s3_schedules(s3)
        assert len(schedules) == 1
        assert schedules[0]["name"] == "keep-this"


# ===================================================================
# Edge cases
# ===================================================================

class TestEdgeCases:
    def test_missing_file_treated_as_empty(self, mgr):
        """No schedules file in S3 → list returns empty."""
        result = mgr.schedule_task("list", AGENT_ID, USER_ID)
        assert "empty" in result.lower()

    def test_malformed_json_starts_empty(self, mgr, s3, caplog):
        """Corrupted JSON → log warning and treat as empty list."""
        s3.put_object(Bucket=BUCKET, Key=S3_KEY, Body=b"NOT VALID JSON{{{")

        with caplog.at_level(logging.WARNING):
            result = mgr.schedule_task("list", AGENT_ID, USER_ID)

        assert "empty" in result.lower()
        assert "malformed" in caplog.text.lower()

    def test_unknown_action(self, mgr):
        result = mgr.schedule_task("update", AGENT_ID, USER_ID)
        assert "unknown action" in result.lower()


# ===================================================================
# Serialization round-trip  (Req 17.3, 17.4)
# ===================================================================

class TestSerialization:
    def test_round_trip_preserves_data(self, mgr, s3, scheduler):
        """Create a schedule, read it back from S3, and verify all fields survive."""
        mgr.schedule_task(
            "create", AGENT_ID, USER_ID,
            name="round-trip",
            description="Test round trip",
            cron_expression=VALID_CRON,
            task_prompt="Do the thing",
        )

        schedules = _read_s3_schedules(s3)
        assert len(schedules) == 1
        s_item = schedules[0]
        assert s_item["name"] == "round-trip"
        assert s_item["description"] == "Test round trip"
        assert s_item["cron_expression"] == VALID_CRON
        assert s_item["status"] == "active"
        assert s_item["eventbridge_rule_arn"] == SCHEDULE_ARN
        assert s_item["last_triggered_at"] is None
        uuid.UUID(s_item["id"])  # valid UUID

        # Payload fields
        payload = s_item["payload"]
        assert payload["type"] == "heartbeat"
        assert payload["task"] == "Do the thing"
        assert payload["agent_id"] == AGENT_ID
        assert payload["user_id"] == USER_ID

        # created_at should be a parseable ISO timestamp
        from datetime import datetime
        datetime.fromisoformat(s_item["created_at"])
