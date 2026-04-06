"""Property-based tests for HeartbeatHandler (Properties 16, 17, 18).

Uses hypothesis strategies with mock S3 client and agent factory to validate
heartbeat handling across random inputs.

Property 16: Heartbeat payload agent_id validation
Property 17: Heartbeat timestamp update
Property 18: Heartbeat response format
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from io import BytesIO
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from agent.heartbeat import HeartbeatHandler
from agent.models import HeartbeatPayload

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Valid agent_id: 1-50 chars, alphanumeric + hyphens, non-empty after strip
valid_agent_id_st = st.from_regex(r"[a-zA-Z0-9][a-zA-Z0-9\-]{0,49}", fullmatch=True)

# Valid user_id: non-empty printable string
valid_user_id_st = st.text(min_size=1, max_size=50).filter(lambda s: s.strip())

# Valid task prompt: non-empty string
valid_task_st = st.text(min_size=1, max_size=200).filter(lambda s: s.strip())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_s3(schedules: list[dict] | None = None, missing: bool = False) -> MagicMock:
    """Create a mock S3 client with optional schedule data."""
    client = MagicMock()
    no_such_key = type("NoSuchKey", (Exception,), {})
    client.exceptions = MagicMock()
    client.exceptions.NoSuchKey = no_such_key

    if missing:
        error = no_such_key("NoSuchKey")
        error.response = {"Error": {"Code": "NoSuchKey"}}
        client.get_object.side_effect = error
    elif schedules is not None:
        body = json.dumps({"schedules": schedules}).encode()
        client.get_object.return_value = {"Body": BytesIO(body)}
    else:
        body = json.dumps({"schedules": []}).encode()
        client.get_object.return_value = {"Body": BytesIO(body)}

    client.put_object.return_value = None
    return client


def _make_schedule(
    task_prompt: str,
    agent_id: str,
    user_id: str,
    status: str = "active",
) -> dict:
    """Create a schedule dict matching the S3 JSON format."""
    return {
        "id": str(uuid4()),
        "name": "test-schedule",
        "description": "Test schedule",
        "cron_expression": "cron(0 9 * * ? *)",
        "payload": {
            "type": "heartbeat",
            "task": task_prompt,
            "agent_id": agent_id,
            "user_id": user_id,
        },
        "eventbridge_rule_arn": f"arn:aws:scheduler:us-east-1:123456789012:schedule/agent-schedules/{agent_id}-test-schedule",
        "status": status,
        "created_at": "2025-01-15T10:30:00Z",
        "last_triggered_at": None,
        "next_trigger_at": None,
    }


# ---------------------------------------------------------------------------
# Property 16: Heartbeat payload agent_id validation
# **Validates: Requirements 10.6, 16.4**
# ---------------------------------------------------------------------------


class TestHeartbeatPayloadValidation:
    """For any HeartbeatPayload where the agent_id does not match the deployed
    agent's id, the heartbeat handler shall reject the payload. Type must be
    the literal string 'heartbeat', task must be non-empty, and agent_id and
    user_id must be valid identifiers."""

    @given(
        deployed_id=valid_agent_id_st,
        payload_id=valid_agent_id_st,
        task=valid_task_st,
        user_id=valid_user_id_st,
    )
    @settings(max_examples=50, deadline=None)
    def test_mismatched_agent_id_rejected(
        self, deployed_id: str, payload_id: str, task: str, user_id: str
    ):
        """**Validates: Requirements 10.6, 16.4**

        Mismatched agent_id must produce an error response.
        """
        assume(deployed_id != payload_id)

        s3 = _make_mock_s3()
        factory = MagicMock(return_value="output")
        handler = HeartbeatHandler(
            s3_client=s3,
            bucket="test-bucket",
            deployed_agent_id=deployed_id,
            agent_factory=factory,
        )

        result = handler.handle_heartbeat({
            "type": "heartbeat",
            "task": task,
            "agent_id": payload_id,
            "user_id": user_id,
        })

        assert result["status"] == "error"
        assert "does not match" in result["error_message"]
        factory.assert_not_called()

    @given(
        agent_id=valid_agent_id_st,
        task=valid_task_st,
        user_id=valid_user_id_st,
        bad_type=st.text(min_size=1, max_size=50).filter(lambda s: s != "heartbeat"),
    )
    @settings(max_examples=50, deadline=None)
    def test_type_must_be_heartbeat(
        self, agent_id: str, task: str, user_id: str, bad_type: str
    ):
        """**Validates: Requirements 10.6, 16.4**

        Any type value other than 'heartbeat' must be rejected.
        """
        s3 = _make_mock_s3()
        factory = MagicMock(return_value="output")
        handler = HeartbeatHandler(
            s3_client=s3,
            bucket="test-bucket",
            deployed_agent_id=agent_id,
            agent_factory=factory,
        )

        result = handler.handle_heartbeat({
            "type": bad_type,
            "task": task,
            "agent_id": agent_id,
            "user_id": user_id,
        })

        assert result["status"] == "error"
        factory.assert_not_called()

    @given(
        agent_id=valid_agent_id_st,
        user_id=valid_user_id_st,
    )
    @settings(max_examples=50, deadline=None)
    def test_empty_task_rejected(self, agent_id: str, user_id: str):
        """**Validates: Requirements 10.6, 16.4**

        An empty task field must be rejected.
        """
        s3 = _make_mock_s3()
        factory = MagicMock(return_value="output")
        handler = HeartbeatHandler(
            s3_client=s3,
            bucket="test-bucket",
            deployed_agent_id=agent_id,
            agent_factory=factory,
        )

        result = handler.handle_heartbeat({
            "type": "heartbeat",
            "task": "",
            "agent_id": agent_id,
            "user_id": user_id,
        })

        assert result["status"] == "error"
        factory.assert_not_called()

    @given(
        agent_id=valid_agent_id_st,
        task=valid_task_st,
        user_id=valid_user_id_st,
    )
    @settings(max_examples=50, deadline=None)
    def test_matching_agent_id_accepted(
        self, agent_id: str, task: str, user_id: str
    ):
        """**Validates: Requirements 10.6, 16.4**

        When agent_id matches the deployed agent, the payload is accepted
        (no validation error).
        """
        schedule = _make_schedule(task, agent_id, user_id)
        s3 = _make_mock_s3(schedules=[schedule])
        factory = MagicMock(return_value="task done")
        handler = HeartbeatHandler(
            s3_client=s3,
            bucket="test-bucket",
            deployed_agent_id=agent_id,
            agent_factory=factory,
        )

        result = handler.handle_heartbeat({
            "type": "heartbeat",
            "task": task,
            "agent_id": agent_id,
            "user_id": user_id,
        })

        # Should NOT be a validation error
        assert result["status"] != "error"


# ---------------------------------------------------------------------------
# Property 17: Heartbeat timestamp update
# **Validates: Requirement 10.5**
# ---------------------------------------------------------------------------


class TestHeartbeatTimestampUpdate:
    """For any heartbeat execution that processes a scheduled task, the
    corresponding schedule's last_triggered_at timestamp in schedules.json
    shall be updated to a value no earlier than the execution start time."""

    @given(
        agent_id=valid_agent_id_st,
        task=valid_task_st,
        user_id=valid_user_id_st,
    )
    @settings(max_examples=50, deadline=None)
    def test_last_triggered_at_updated_no_earlier_than_start(
        self, agent_id: str, task: str, user_id: str
    ):
        """**Validates: Requirement 10.5**

        After heartbeat execution, last_triggered_at is set to a value
        no earlier than the time just before execution started.
        """
        schedule = _make_schedule(task, agent_id, user_id)
        s3 = _make_mock_s3(schedules=[schedule])
        factory = MagicMock(return_value="result output")
        handler = HeartbeatHandler(
            s3_client=s3,
            bucket="test-bucket",
            deployed_agent_id=agent_id,
            agent_factory=factory,
        )

        before_exec = datetime.now(timezone.utc)

        handler.handle_heartbeat({
            "type": "heartbeat",
            "task": task,
            "agent_id": agent_id,
            "user_id": user_id,
        })

        # Verify put_object was called to persist updated schedules
        s3.put_object.assert_called_once()
        call_kwargs = s3.put_object.call_args[1]
        persisted = json.loads(call_kwargs["Body"])
        updated_schedule = persisted["schedules"][0]

        assert updated_schedule["last_triggered_at"] is not None

        ts = datetime.fromisoformat(updated_schedule["last_triggered_at"])
        # Timestamp must be no earlier than execution start
        assert ts >= before_exec

    @given(
        agent_id=valid_agent_id_st,
        task_a=valid_task_st,
        task_b=valid_task_st,
        user_id=valid_user_id_st,
    )
    @settings(max_examples=50, deadline=None)
    def test_only_matching_schedule_timestamp_updated(
        self, agent_id: str, task_a: str, task_b: str, user_id: str
    ):
        """**Validates: Requirement 10.5**

        Only the schedule matching the heartbeat task gets its timestamp
        updated; other schedules remain unchanged.
        """
        assume(task_a != task_b)

        schedule_a = _make_schedule(task_a, agent_id, user_id)
        schedule_b = _make_schedule(task_b, agent_id, user_id)
        s3 = _make_mock_s3(schedules=[schedule_a, schedule_b])
        factory = MagicMock(return_value="done")
        handler = HeartbeatHandler(
            s3_client=s3,
            bucket="test-bucket",
            deployed_agent_id=agent_id,
            agent_factory=factory,
        )

        handler.handle_heartbeat({
            "type": "heartbeat",
            "task": task_a,
            "agent_id": agent_id,
            "user_id": user_id,
        })

        persisted = json.loads(s3.put_object.call_args[1]["Body"])
        schedules = persisted["schedules"]

        # Schedule A (matched) should have timestamp updated
        assert schedules[0]["last_triggered_at"] is not None
        # Schedule B (not matched) should remain None
        assert schedules[1]["last_triggered_at"] is None


# ---------------------------------------------------------------------------
# Property 18: Heartbeat response format
# **Validates: Requirements 10.3, 10.4**
# ---------------------------------------------------------------------------


class TestHeartbeatResponseFormat:
    """For any heartbeat invocation that produces output, the response shall
    contain status 'completed' and a non-empty output field. For any heartbeat
    invocation with no pending tasks, the response shall contain status
    'heartbeat_ok'."""

    @given(
        agent_id=valid_agent_id_st,
        task=valid_task_st,
        user_id=valid_user_id_st,
        output=st.text(min_size=1, max_size=200).filter(lambda s: s.strip()),
    )
    @settings(max_examples=50, deadline=None)
    def test_completed_status_with_non_empty_output(
        self, agent_id: str, task: str, user_id: str, output: str
    ):
        """**Validates: Requirements 10.3, 10.4**

        When the agent factory returns non-empty output, the response
        has status 'completed' and a non-empty output field.
        """
        schedule = _make_schedule(task, agent_id, user_id)
        s3 = _make_mock_s3(schedules=[schedule])
        factory = MagicMock(return_value=output)
        handler = HeartbeatHandler(
            s3_client=s3,
            bucket="test-bucket",
            deployed_agent_id=agent_id,
            agent_factory=factory,
        )

        result = handler.handle_heartbeat({
            "type": "heartbeat",
            "task": task,
            "agent_id": agent_id,
            "user_id": user_id,
        })

        assert result["status"] == "completed"
        assert result["output"] is not None
        assert result["output"].strip() != ""

    @given(
        agent_id=valid_agent_id_st,
        task=valid_task_st,
        user_id=valid_user_id_st,
    )
    @settings(max_examples=50, deadline=None)
    def test_heartbeat_ok_when_no_pending_tasks(
        self, agent_id: str, task: str, user_id: str
    ):
        """**Validates: Requirements 10.3, 10.4**

        When no schedule matches the heartbeat task, the response has
        status 'heartbeat_ok'.
        """
        # No schedules at all
        s3 = _make_mock_s3(schedules=[])
        factory = MagicMock(return_value="should not be called")
        handler = HeartbeatHandler(
            s3_client=s3,
            bucket="test-bucket",
            deployed_agent_id=agent_id,
            agent_factory=factory,
        )

        result = handler.handle_heartbeat({
            "type": "heartbeat",
            "task": task,
            "agent_id": agent_id,
            "user_id": user_id,
        })

        assert result["status"] == "heartbeat_ok"
        factory.assert_not_called()

    @given(
        agent_id=valid_agent_id_st,
        task=valid_task_st,
        user_id=valid_user_id_st,
    )
    @settings(max_examples=50, deadline=None)
    def test_heartbeat_ok_when_missing_schedules_json(
        self, agent_id: str, task: str, user_id: str
    ):
        """**Validates: Requirements 10.3, 10.4**

        When schedules.json is missing from S3, the response has
        status 'heartbeat_ok' with no pending tasks.
        """
        s3 = _make_mock_s3(missing=True)
        factory = MagicMock(return_value="should not be called")
        handler = HeartbeatHandler(
            s3_client=s3,
            bucket="test-bucket",
            deployed_agent_id=agent_id,
            agent_factory=factory,
        )

        result = handler.handle_heartbeat({
            "type": "heartbeat",
            "task": task,
            "agent_id": agent_id,
            "user_id": user_id,
        })

        assert result["status"] == "heartbeat_ok"
        factory.assert_not_called()
