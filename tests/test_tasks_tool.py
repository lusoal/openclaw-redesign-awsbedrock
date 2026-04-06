"""Unit tests for the manage_tasks tool (TaskManager).

Uses moto to mock S3.  Covers Requirements 8.1–8.9, 17.1, 17.2.
"""

from __future__ import annotations

import json
import logging
import uuid

import boto3
import pytest
from moto import mock_aws

from agent.tools.tasks import TaskManager

BUCKET = "test-identity-bucket"
AGENT_ID = "test-agent"
USER_ID = "user-1"
S3_KEY = f"agents/{AGENT_ID}/tasks/{USER_ID}.json"


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
def mgr(s3):
    """Return a TaskManager wired to the mocked S3 client."""
    return TaskManager(s3_client=s3, bucket=BUCKET)


def _read_s3_tasks(s3) -> list[dict]:
    """Helper: read the raw task list from S3."""
    obj = s3.get_object(Bucket=BUCKET, Key=S3_KEY)
    return json.loads(obj["Body"].read()).get("tasks", [])


# ===================================================================
# Action: add  (Req 8.1, 8.8, 8.9, 17.1)
# ===================================================================

class TestAddAction:
    def test_add_creates_task_with_defaults(self, mgr, s3):
        result = mgr.manage_tasks("add", AGENT_ID, USER_ID, title="Buy milk")

        assert "Buy milk" in result
        assert "medium" in result

        tasks = _read_s3_tasks(s3)
        assert len(tasks) == 1
        t = tasks[0]
        assert t["title"] == "Buy milk"
        assert t["status"] == "pending"
        assert t["priority"] == "medium"
        assert t["completed_at"] is None
        # id should be a valid UUID
        uuid.UUID(t["id"])

    def test_add_respects_priority(self, mgr, s3):
        mgr.manage_tasks("add", AGENT_ID, USER_ID, title="Urgent", priority="high")

        tasks = _read_s3_tasks(s3)
        assert tasks[0]["priority"] == "high"

    def test_add_invalid_priority_defaults_to_medium(self, mgr, s3):
        mgr.manage_tasks("add", AGENT_ID, USER_ID, title="Task", priority="critical")

        tasks = _read_s3_tasks(s3)
        assert tasks[0]["priority"] == "medium"

    def test_add_empty_title_rejected(self, mgr):
        result = mgr.manage_tasks("add", AGENT_ID, USER_ID, title="")
        assert "non-empty" in result.lower()

    def test_add_whitespace_title_rejected(self, mgr):
        result = mgr.manage_tasks("add", AGENT_ID, USER_ID, title="   ")
        assert "non-empty" in result.lower()

    def test_add_title_over_500_chars_rejected(self, mgr):
        result = mgr.manage_tasks("add", AGENT_ID, USER_ID, title="x" * 501)
        assert "500" in result

    def test_add_title_exactly_500_chars_accepted(self, mgr, s3):
        result = mgr.manage_tasks("add", AGENT_ID, USER_ID, title="x" * 500)
        assert "added" in result.lower()

    def test_add_serializes_with_json_content_type(self, mgr, s3):
        mgr.manage_tasks("add", AGENT_ID, USER_ID, title="Check CT")

        head = s3.head_object(Bucket=BUCKET, Key=S3_KEY)
        assert head["ContentType"] == "application/json"

    def test_add_multiple_tasks_accumulates(self, mgr, s3):
        mgr.manage_tasks("add", AGENT_ID, USER_ID, title="First")
        mgr.manage_tasks("add", AGENT_ID, USER_ID, title="Second")

        tasks = _read_s3_tasks(s3)
        assert len(tasks) == 2


# ===================================================================
# Action: list  (Req 8.2)
# ===================================================================

class TestListAction:
    def test_list_empty(self, mgr):
        result = mgr.manage_tasks("list", AGENT_ID, USER_ID)
        assert "empty" in result.lower()

    def test_list_groups_by_status(self, mgr, s3):
        mgr.manage_tasks("add", AGENT_ID, USER_ID, title="Pending task", priority="low")
        mgr.manage_tasks("add", AGENT_ID, USER_ID, title="Done task", priority="high")

        # Complete the second task
        tasks = _read_s3_tasks(s3)
        tid = tasks[1]["id"][:8]
        mgr.manage_tasks("complete", AGENT_ID, USER_ID, task_id=tid)

        result = mgr.manage_tasks("list", AGENT_ID, USER_ID)
        assert "Pending (1):" in result
        assert "Completed (1):" in result
        assert "[low]" in result
        assert "✓" in result

    def test_list_shows_truncated_id(self, mgr):
        mgr.manage_tasks("add", AGENT_ID, USER_ID, title="Task A")
        result = mgr.manage_tasks("list", AGENT_ID, USER_ID)
        # The id shown should be 8 chars
        assert "(id: " in result


# ===================================================================
# Action: complete  (Req 8.3, 8.5)
# ===================================================================

class TestCompleteAction:
    def test_complete_sets_status_and_timestamp(self, mgr, s3):
        mgr.manage_tasks("add", AGENT_ID, USER_ID, title="Review PR")
        tasks = _read_s3_tasks(s3)
        tid = tasks[0]["id"][:8]

        result = mgr.manage_tasks("complete", AGENT_ID, USER_ID, task_id=tid)

        assert "completed" in result.lower()
        tasks = _read_s3_tasks(s3)
        assert tasks[0]["status"] == "completed"
        assert tasks[0]["completed_at"] is not None

    def test_complete_prefix_matching(self, mgr, s3):
        mgr.manage_tasks("add", AGENT_ID, USER_ID, title="Task")
        tasks = _read_s3_tasks(s3)
        full_id = tasks[0]["id"]

        # Use first 4 chars as prefix
        result = mgr.manage_tasks("complete", AGENT_ID, USER_ID, task_id=full_id[:4])
        assert "completed" in result.lower()

    def test_complete_not_found(self, mgr):
        result = mgr.manage_tasks("complete", AGENT_ID, USER_ID, task_id="nonexistent")
        assert "not found" in result.lower()

    def test_complete_leaves_other_tasks_unchanged(self, mgr, s3):
        mgr.manage_tasks("add", AGENT_ID, USER_ID, title="Task A")
        mgr.manage_tasks("add", AGENT_ID, USER_ID, title="Task B")
        tasks = _read_s3_tasks(s3)
        tid_a = tasks[0]["id"][:8]

        mgr.manage_tasks("complete", AGENT_ID, USER_ID, task_id=tid_a)

        tasks = _read_s3_tasks(s3)
        assert tasks[0]["status"] == "completed"
        assert tasks[1]["status"] == "pending"


# ===================================================================
# Action: delete  (Req 8.4, 8.5)
# ===================================================================

class TestDeleteAction:
    def test_delete_removes_task(self, mgr, s3):
        mgr.manage_tasks("add", AGENT_ID, USER_ID, title="Temp task")
        tasks = _read_s3_tasks(s3)
        tid = tasks[0]["id"][:8]

        result = mgr.manage_tasks("delete", AGENT_ID, USER_ID, task_id=tid)

        assert "deleted" in result.lower()
        tasks = _read_s3_tasks(s3)
        assert len(tasks) == 0

    def test_delete_not_found(self, mgr):
        result = mgr.manage_tasks("delete", AGENT_ID, USER_ID, task_id="nonexistent")
        assert "not found" in result.lower()

    def test_delete_prefix_matching(self, mgr, s3):
        mgr.manage_tasks("add", AGENT_ID, USER_ID, title="Task")
        tasks = _read_s3_tasks(s3)
        full_id = tasks[0]["id"]

        result = mgr.manage_tasks("delete", AGENT_ID, USER_ID, task_id=full_id[:4])
        assert "deleted" in result.lower()


# ===================================================================
# Malformed / missing JSON  (Req 8.6, 8.7)
# ===================================================================

class TestEdgeCases:
    def test_missing_file_treated_as_empty(self, mgr):
        """No tasks file in S3 → list returns empty."""
        result = mgr.manage_tasks("list", AGENT_ID, USER_ID)
        assert "empty" in result.lower()

    def test_malformed_json_starts_empty(self, mgr, s3, caplog):
        """Corrupted JSON → log warning and treat as empty list."""
        s3.put_object(Bucket=BUCKET, Key=S3_KEY, Body=b"NOT VALID JSON{{{")

        with caplog.at_level(logging.WARNING):
            result = mgr.manage_tasks("list", AGENT_ID, USER_ID)

        assert "empty" in result.lower()
        assert "malformed" in caplog.text.lower()

    def test_unknown_action(self, mgr):
        result = mgr.manage_tasks("archive", AGENT_ID, USER_ID)
        assert "unknown action" in result.lower()


# ===================================================================
# Serialization round-trip  (Req 17.1, 17.2)
# ===================================================================

class TestSerialization:
    def test_round_trip_preserves_data(self, mgr, s3):
        """Add a task, read it back from S3, and verify all fields survive."""
        mgr.manage_tasks("add", AGENT_ID, USER_ID, title="Round trip", priority="high")

        tasks = _read_s3_tasks(s3)
        assert len(tasks) == 1
        t = tasks[0]
        assert t["title"] == "Round trip"
        assert t["priority"] == "high"
        assert t["status"] == "pending"
        assert t["completed_at"] is None
        uuid.UUID(t["id"])  # valid UUID
        # created_at should be a parseable ISO timestamp
        from datetime import datetime
        datetime.fromisoformat(t["created_at"])
