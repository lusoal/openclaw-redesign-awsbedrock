"""Tests for the HeartbeatHandler.

Covers payload validation, schedule loading, task execution,
timestamp updates, error handling, and missing/corrupted schedules.

Uses unittest.mock for S3 and agent factory.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from io import BytesIO
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from agent.heartbeat import HeartbeatHandler
from agent.models import HeartbeatResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_s3_client(schedules: list[dict] | None = None, missing: bool = False) -> MagicMock:
    """Create a mock S3 client.

    Parameters
    ----------
    schedules:
        If provided, get_object returns this schedule list.
    missing:
        If True, get_object raises NoSuchKey.
    """
    client = MagicMock()

    # Set up NoSuchKey exception class
    no_such_key = type("NoSuchKey", (Exception,), {})
    client.exceptions = MagicMock()
    client.exceptions.NoSuchKey = no_such_key

    if missing:
        error = no_such_key("NoSuchKey")
        error.response = {"Error": {"Code": "NoSuchKey"}}
        client.get_object.side_effect = error
    elif schedules is not None:
        body = json.dumps({"schedules": schedules}).encode()
        client.get_object.return_value = {
            "Body": BytesIO(body),
        }
    else:
        body = json.dumps({"schedules": []}).encode()
        client.get_object.return_value = {
            "Body": BytesIO(body),
        }

    client.put_object.return_value = None
    return client


def _make_schedule(
    task_prompt: str = "Remind user about standup",
    agent_id: str = "test-agent",
    user_id: str = "user-1",
    status: str = "active",
) -> dict:
    """Create a schedule dict matching the S3 JSON format."""
    return {
        "id": str(uuid4()),
        "name": "daily-standup",
        "description": "Daily standup reminder",
        "cron_expression": "cron(0 9 * * ? *)",
        "payload": {
            "type": "heartbeat",
            "task": task_prompt,
            "agent_id": agent_id,
            "user_id": user_id,
        },
        "eventbridge_rule_arn": "arn:aws:scheduler:us-east-1:123456789012:schedule/agent-schedules/test-agent-daily-standup",
        "status": status,
        "created_at": "2025-01-15T10:30:00Z",
        "last_triggered_at": None,
        "next_trigger_at": None,
    }


def _make_heartbeat_event(
    task: str = "Remind user about standup",
    agent_id: str = "test-agent",
    user_id: str = "user-1",
) -> dict:
    """Create a heartbeat event payload."""
    return {
        "type": "heartbeat",
        "task": task,
        "agent_id": agent_id,
        "user_id": user_id,
    }


def _make_agent_factory(response: str = "Standup reminder sent!") -> MagicMock:
    """Create a mock agent factory."""
    return MagicMock(return_value=response)


# ===================================================================
# Heartbeat with matching schedule
# ===================================================================

class TestHeartbeatWithTask:
    def test_completed_status_with_output(self):
        schedule = _make_schedule()
        s3 = _make_s3_client(schedules=[schedule])
        factory = _make_agent_factory("Reminder: standup in 5 minutes!")
        handler = HeartbeatHandler(
            s3_client=s3,
            bucket="test-bucket",
            deployed_agent_id="test-agent",
            agent_factory=factory,
        )

        result = handler.handle_heartbeat(_make_heartbeat_event())

        assert result["status"] == "completed"
        assert result["output"] == "Reminder: standup in 5 minutes!"

    def test_agent_factory_called_with_correct_args(self):
        schedule = _make_schedule()
        s3 = _make_s3_client(schedules=[schedule])
        factory = _make_agent_factory()
        handler = HeartbeatHandler(
            s3_client=s3,
            bucket="test-bucket",
            deployed_agent_id="test-agent",
            agent_factory=factory,
        )

        handler.handle_heartbeat(_make_heartbeat_event())

        factory.assert_called_once_with(
            "test-agent", "user-1", "Remind user about standup"
        )

    def test_empty_response_returns_heartbeat_ok(self):
        schedule = _make_schedule()
        s3 = _make_s3_client(schedules=[schedule])
        factory = _make_agent_factory("   ")
        handler = HeartbeatHandler(
            s3_client=s3,
            bucket="test-bucket",
            deployed_agent_id="test-agent",
            agent_factory=factory,
        )

        result = handler.handle_heartbeat(_make_heartbeat_event())

        assert result["status"] == "heartbeat_ok"


# ===================================================================
# Heartbeat with no matching schedule
# ===================================================================

class TestHeartbeatNoTask:
    def test_no_schedules_returns_heartbeat_ok(self):
        s3 = _make_s3_client(schedules=[])
        factory = _make_agent_factory()
        handler = HeartbeatHandler(
            s3_client=s3,
            bucket="test-bucket",
            deployed_agent_id="test-agent",
            agent_factory=factory,
        )

        result = handler.handle_heartbeat(_make_heartbeat_event())

        assert result["status"] == "heartbeat_ok"
        factory.assert_not_called()

    def test_no_matching_task_returns_heartbeat_ok(self):
        schedule = _make_schedule(task_prompt="Different task")
        s3 = _make_s3_client(schedules=[schedule])
        factory = _make_agent_factory()
        handler = HeartbeatHandler(
            s3_client=s3,
            bucket="test-bucket",
            deployed_agent_id="test-agent",
            agent_factory=factory,
        )

        result = handler.handle_heartbeat(
            _make_heartbeat_event(task="Remind user about standup")
        )

        assert result["status"] == "heartbeat_ok"
        factory.assert_not_called()

    def test_paused_schedule_not_matched(self):
        schedule = _make_schedule(status="paused")
        s3 = _make_s3_client(schedules=[schedule])
        factory = _make_agent_factory()
        handler = HeartbeatHandler(
            s3_client=s3,
            bucket="test-bucket",
            deployed_agent_id="test-agent",
            agent_factory=factory,
        )

        result = handler.handle_heartbeat(_make_heartbeat_event())

        assert result["status"] == "heartbeat_ok"
        factory.assert_not_called()


# ===================================================================
# Payload validation
# ===================================================================

class TestPayloadValidation:
    def test_mismatched_agent_id_returns_error(self):
        s3 = _make_s3_client()
        factory = _make_agent_factory()
        handler = HeartbeatHandler(
            s3_client=s3,
            bucket="test-bucket",
            deployed_agent_id="test-agent",
            agent_factory=factory,
        )

        result = handler.handle_heartbeat(
            _make_heartbeat_event(agent_id="wrong-agent")
        )

        assert result["status"] == "error"
        assert "does not match" in result["error_message"]

    def test_missing_type_field_returns_error(self):
        s3 = _make_s3_client()
        factory = _make_agent_factory()
        handler = HeartbeatHandler(
            s3_client=s3,
            bucket="test-bucket",
            deployed_agent_id="test-agent",
            agent_factory=factory,
        )

        result = handler.handle_heartbeat(
            {"task": "Do something", "agent_id": "test-agent", "user_id": "user-1"}
        )

        assert result["status"] == "error"

    def test_wrong_type_value_returns_error(self):
        s3 = _make_s3_client()
        factory = _make_agent_factory()
        handler = HeartbeatHandler(
            s3_client=s3,
            bucket="test-bucket",
            deployed_agent_id="test-agent",
            agent_factory=factory,
        )

        result = handler.handle_heartbeat(
            {"type": "interactive", "task": "Do something",
             "agent_id": "test-agent", "user_id": "user-1"}
        )

        assert result["status"] == "error"

    def test_empty_task_returns_error(self):
        s3 = _make_s3_client()
        factory = _make_agent_factory()
        handler = HeartbeatHandler(
            s3_client=s3,
            bucket="test-bucket",
            deployed_agent_id="test-agent",
            agent_factory=factory,
        )

        result = handler.handle_heartbeat(
            {"type": "heartbeat", "task": "",
             "agent_id": "test-agent", "user_id": "user-1"}
        )

        assert result["status"] == "error"

    def test_empty_user_id_returns_error(self):
        s3 = _make_s3_client()
        factory = _make_agent_factory()
        handler = HeartbeatHandler(
            s3_client=s3,
            bucket="test-bucket",
            deployed_agent_id="test-agent",
            agent_factory=factory,
        )

        result = handler.handle_heartbeat(
            {"type": "heartbeat", "task": "Do something",
             "agent_id": "test-agent", "user_id": ""}
        )

        assert result["status"] == "error"


# ===================================================================
# Timestamp update
# ===================================================================

class TestTimestampUpdate:
    def test_last_triggered_at_updated_after_execution(self):
        schedule = _make_schedule()
        s3 = _make_s3_client(schedules=[schedule])
        factory = _make_agent_factory("Done!")
        handler = HeartbeatHandler(
            s3_client=s3,
            bucket="test-bucket",
            deployed_agent_id="test-agent",
            agent_factory=factory,
        )

        handler.handle_heartbeat(_make_heartbeat_event())

        # Verify put_object was called to persist updated schedules
        s3.put_object.assert_called_once()
        call_kwargs = s3.put_object.call_args[1]
        assert call_kwargs["Key"] == "agents/test-agent/schedules.json"
        assert call_kwargs["ContentType"] == "application/json"

        # Parse the persisted data and check timestamp
        persisted = json.loads(call_kwargs["Body"])
        updated_schedule = persisted["schedules"][0]
        assert updated_schedule["last_triggered_at"] is not None

        # Verify the timestamp is a valid ISO format
        ts = datetime.fromisoformat(updated_schedule["last_triggered_at"])
        assert ts.tzinfo is not None

    def test_only_matching_schedule_timestamp_updated(self):
        schedule1 = _make_schedule(task_prompt="Task A")
        schedule2 = _make_schedule(task_prompt="Task B")
        s3 = _make_s3_client(schedules=[schedule1, schedule2])
        factory = _make_agent_factory("Done!")
        handler = HeartbeatHandler(
            s3_client=s3,
            bucket="test-bucket",
            deployed_agent_id="test-agent",
            agent_factory=factory,
        )

        handler.handle_heartbeat(_make_heartbeat_event(task="Task A"))

        persisted = json.loads(s3.put_object.call_args[1]["Body"])
        schedules = persisted["schedules"]
        # First schedule (Task A) should have timestamp updated
        assert schedules[0]["last_triggered_at"] is not None
        # Second schedule (Task B) should remain None
        assert schedules[1]["last_triggered_at"] is None


# ===================================================================
# Missing/corrupted schedules.json
# ===================================================================

class TestMissingSchedules:
    def test_missing_schedules_json_returns_heartbeat_ok(self):
        s3 = _make_s3_client(missing=True)
        factory = _make_agent_factory()
        handler = HeartbeatHandler(
            s3_client=s3,
            bucket="test-bucket",
            deployed_agent_id="test-agent",
            agent_factory=factory,
        )

        result = handler.handle_heartbeat(_make_heartbeat_event())

        assert result["status"] == "heartbeat_ok"
        factory.assert_not_called()

    def test_corrupted_schedules_json_returns_heartbeat_ok(self, caplog):
        s3 = MagicMock()
        s3.exceptions = MagicMock()
        s3.exceptions.NoSuchKey = type("NoSuchKey", (Exception,), {})
        # Return invalid JSON
        s3.get_object.return_value = {
            "Body": BytesIO(b"not valid json{{{"),
        }
        factory = _make_agent_factory()
        handler = HeartbeatHandler(
            s3_client=s3,
            bucket="test-bucket",
            deployed_agent_id="test-agent",
            agent_factory=factory,
        )

        with caplog.at_level(logging.WARNING):
            result = handler.handle_heartbeat(_make_heartbeat_event())

        assert result["status"] == "heartbeat_ok"
        factory.assert_not_called()


# ===================================================================
# Error handling — no crash
# ===================================================================

class TestErrorHandling:
    def test_agent_factory_exception_returns_error(self):
        schedule = _make_schedule()
        s3 = _make_s3_client(schedules=[schedule])
        factory = MagicMock(side_effect=RuntimeError("LLM exploded"))
        handler = HeartbeatHandler(
            s3_client=s3,
            bucket="test-bucket",
            deployed_agent_id="test-agent",
            agent_factory=factory,
        )

        result = handler.handle_heartbeat(_make_heartbeat_event())

        assert result["status"] == "error"
        assert "LLM exploded" in result["error_message"]

    def test_s3_read_error_returns_heartbeat_ok(self):
        """S3 read failure treated as empty schedule list."""
        s3 = MagicMock()
        s3.exceptions = MagicMock()
        s3.exceptions.NoSuchKey = type("NoSuchKey", (Exception,), {})
        s3.get_object.side_effect = RuntimeError("S3 is down")
        factory = _make_agent_factory()
        handler = HeartbeatHandler(
            s3_client=s3,
            bucket="test-bucket",
            deployed_agent_id="test-agent",
            agent_factory=factory,
        )

        result = handler.handle_heartbeat(_make_heartbeat_event())

        assert result["status"] == "heartbeat_ok"

    def test_timestamp_update_failure_does_not_crash(self, caplog):
        schedule = _make_schedule()
        s3 = _make_s3_client(schedules=[schedule])
        s3.put_object.side_effect = RuntimeError("S3 write failed")
        factory = _make_agent_factory("Done!")
        handler = HeartbeatHandler(
            s3_client=s3,
            bucket="test-bucket",
            deployed_agent_id="test-agent",
            agent_factory=factory,
        )

        with caplog.at_level(logging.WARNING):
            result = handler.handle_heartbeat(_make_heartbeat_event())

        # Should still return completed since the task ran
        assert result["status"] == "completed"
        assert result["output"] == "Done!"
