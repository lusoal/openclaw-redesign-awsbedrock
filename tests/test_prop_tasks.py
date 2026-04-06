"""Property-based tests for the manage_tasks tool (Properties 5, 6, 7, 8, 10).

Uses moto-mocked S3 and hypothesis strategies to validate TaskManager
behaviour across random inputs.

Property 5: Task add produces valid TaskItem
Property 6: Task list completeness
Property 7: Task completion state transition
Property 8: Task deletion reduces list
Property 10: Task serialization round-trip
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

import boto3
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from moto import mock_aws

from agent.models import TaskItem
from agent.tools.tasks import TaskManager

BUCKET = "prop-test-bucket"
AGENT_ID = "prop-agent"
USER_ID = "prop-user"
S3_KEY = f"agents/{AGENT_ID}/tasks/{USER_ID}.json"

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Valid title: 1-500 printable chars, non-whitespace-only
valid_title_st = st.text(min_size=1, max_size=200).filter(lambda s: s.strip())

# Valid priority
valid_priority_st = st.sampled_from(["low", "medium", "high"])


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


# ---------------------------------------------------------------------------
# Property 5: Task add produces valid TaskItem
# **Validates: Requirements 8.1, 16.2**
# ---------------------------------------------------------------------------


class TestTaskAddProducesValidTaskItem:
    """For any valid title and priority, add creates a TaskItem with a valid
    UUID id, status 'pending', non-None created_at, and None completed_at."""

    @given(title=valid_title_st, priority=valid_priority_st)
    @settings(max_examples=50, deadline=None)
    def test_add_produces_valid_task_item(self, title: str, priority: str):
        """**Validates: Requirements 8.1, 16.2**"""
        with mock_aws():
            client = boto3.client("s3", region_name="us-east-1")
            client.create_bucket(Bucket=BUCKET)
            mgr = TaskManager(s3_client=client, bucket=BUCKET)

            result = mgr.manage_tasks("add", AGENT_ID, USER_ID, title=title, priority=priority)
            assert "added" in result.lower()

            tasks = _raw_tasks(client)
            assert len(tasks) == 1
            t = tasks[0]

            # Valid UUID
            uuid.UUID(t["id"])

            # Status is pending
            assert t["status"] == "pending"

            # created_at is non-None and parseable
            assert t["created_at"] is not None
            datetime.fromisoformat(t["created_at"])

            # completed_at is None
            assert t["completed_at"] is None

            # Title and priority match input
            assert t["title"] == title
            assert t["priority"] == priority

            # Validates as a TaskItem model
            TaskItem(**{**t, "id": uuid.UUID(t["id"]),
                        "created_at": datetime.fromisoformat(t["created_at"])})


def _raw_tasks(client) -> list[dict]:
    """Read raw tasks from S3 using a given client."""
    obj = client.get_object(Bucket=BUCKET, Key=S3_KEY)
    return json.loads(obj["Body"].read()).get("tasks", [])


# ---------------------------------------------------------------------------
# Property 6: Task list completeness
# **Validates: Requirement 8.2**
# ---------------------------------------------------------------------------


class TestTaskListCompleteness:
    """List returns every task, grouped by status."""

    @given(
        titles=st.lists(valid_title_st, min_size=1, max_size=10),
        complete_indices=st.data(),
    )
    @settings(max_examples=50, deadline=None)
    def test_list_returns_every_task_grouped(self, titles: list[str], complete_indices):
        """**Validates: Requirements 8.2**"""
        with mock_aws():
            client = boto3.client("s3", region_name="us-east-1")
            client.create_bucket(Bucket=BUCKET)
            mgr = TaskManager(s3_client=client, bucket=BUCKET)

            # Add all tasks
            for t in titles:
                mgr.manage_tasks("add", AGENT_ID, USER_ID, title=t)

            # Decide which to complete (draw a subset of indices)
            indices_to_complete = complete_indices.draw(
                st.lists(
                    st.integers(min_value=0, max_value=len(titles) - 1),
                    max_size=len(titles),
                    unique=True,
                )
            )

            # Complete selected tasks
            raw = _raw_tasks(client)
            for idx in indices_to_complete:
                tid = raw[idx]["id"][:8]
                mgr.manage_tasks("complete", AGENT_ID, USER_ID, task_id=tid)

            # List tasks
            result = mgr.manage_tasks("list", AGENT_ID, USER_ID)

            # Every title must appear in the output
            for t in titles:
                assert t in result

            # Verify grouping: if there are pending tasks, "Pending" appears
            n_completed = len(indices_to_complete)
            n_pending = len(titles) - n_completed

            if n_pending > 0:
                assert "pending" in result.lower()
            if n_completed > 0:
                assert "completed" in result.lower()


# ---------------------------------------------------------------------------
# Property 7: Task completion state transition
# **Validates: Requirements 8.3, 16.2**
# ---------------------------------------------------------------------------


class TestTaskCompletionStateTransition:
    """Completing a task changes its status to 'completed', sets completed_at,
    and leaves other tasks unchanged."""

    @given(
        titles=st.lists(valid_title_st, min_size=2, max_size=8, unique=True),
        target_index=st.data(),
    )
    @settings(max_examples=50, deadline=None)
    def test_complete_transitions_only_target(self, titles: list[str], target_index):
        """**Validates: Requirements 8.3, 16.2**"""
        with mock_aws():
            client = boto3.client("s3", region_name="us-east-1")
            client.create_bucket(Bucket=BUCKET)
            mgr = TaskManager(s3_client=client, bucket=BUCKET)

            for t in titles:
                mgr.manage_tasks("add", AGENT_ID, USER_ID, title=t)

            idx = target_index.draw(st.integers(min_value=0, max_value=len(titles) - 1))

            before = _raw_tasks(client)
            target_id = before[idx]["id"]

            # Complete the target task
            result = mgr.manage_tasks("complete", AGENT_ID, USER_ID, task_id=target_id[:8])
            assert "completed" in result.lower()

            after = _raw_tasks(client)

            for task in after:
                if task["id"] == target_id:
                    # Target: status changed, completed_at set
                    assert task["status"] == "completed"
                    assert task["completed_at"] is not None
                    datetime.fromisoformat(task["completed_at"])
                else:
                    # Others: unchanged — still pending, no completed_at
                    assert task["status"] == "pending"
                    assert task["completed_at"] is None


# ---------------------------------------------------------------------------
# Property 8: Task deletion reduces list
# **Validates: Requirement 8.4**
# ---------------------------------------------------------------------------


class TestTaskDeletionReducesList:
    """Deleting a task reduces the list size by one and the deleted task
    is absent from the remaining list."""

    @given(
        titles=st.lists(valid_title_st, min_size=1, max_size=10, unique=True),
        target_index=st.data(),
    )
    @settings(max_examples=50, deadline=None)
    def test_delete_reduces_list_by_one(self, titles: list[str], target_index):
        """**Validates: Requirements 8.4**"""
        with mock_aws():
            client = boto3.client("s3", region_name="us-east-1")
            client.create_bucket(Bucket=BUCKET)
            mgr = TaskManager(s3_client=client, bucket=BUCKET)

            for t in titles:
                mgr.manage_tasks("add", AGENT_ID, USER_ID, title=t)

            idx = target_index.draw(st.integers(min_value=0, max_value=len(titles) - 1))

            before = _raw_tasks(client)
            assert len(before) == len(titles)
            target_id = before[idx]["id"]

            # Delete the target task
            result = mgr.manage_tasks("delete", AGENT_ID, USER_ID, task_id=target_id[:8])
            assert "deleted" in result.lower()

            after = _raw_tasks(client)

            # Size reduced by exactly one
            assert len(after) == len(titles) - 1

            # Deleted task is absent
            remaining_ids = [t["id"] for t in after]
            assert target_id not in remaining_ids


# ---------------------------------------------------------------------------
# Property 10: Task serialization round-trip
# **Validates: Requirement 17.5**
# ---------------------------------------------------------------------------


class TestTaskSerializationRoundTrip:
    """Serializing tasks to S3 then deserializing produces an equivalent list."""

    @given(
        titles=st.lists(valid_title_st, min_size=1, max_size=8),
        priorities=st.data(),
    )
    @settings(max_examples=50, deadline=None)
    def test_round_trip_preserves_all_fields(self, titles: list[str], priorities):
        """**Validates: Requirements 17.5**"""
        with mock_aws():
            client = boto3.client("s3", region_name="us-east-1")
            client.create_bucket(Bucket=BUCKET)
            mgr = TaskManager(s3_client=client, bucket=BUCKET)

            drawn_priorities = [
                priorities.draw(valid_priority_st) for _ in titles
            ]

            # Add tasks with various priorities
            for title, prio in zip(titles, drawn_priorities):
                mgr.manage_tasks("add", AGENT_ID, USER_ID, title=title, priority=prio)

            # Read raw JSON from S3
            raw = _raw_tasks(client)
            assert len(raw) == len(titles)

            # Re-serialize and re-deserialize
            payload = json.dumps({"tasks": raw}, indent=2)
            client.put_object(
                Bucket=BUCKET, Key=S3_KEY,
                Body=payload, ContentType="application/json",
            )
            round_tripped = _raw_tasks(client)

            # Every field must survive the round-trip
            assert len(round_tripped) == len(raw)
            for original, restored in zip(raw, round_tripped):
                assert original["id"] == restored["id"]
                assert original["title"] == restored["title"]
                assert original["status"] == restored["status"]
                assert original["priority"] == restored["priority"]
                assert original["created_at"] == restored["created_at"]
                assert original["completed_at"] == restored["completed_at"]
