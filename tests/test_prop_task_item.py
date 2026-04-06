"""Property-based tests for TaskItem validation (Property 9).

**Validates: Requirements 8.8, 8.9, 16.2**

For any string, the TaskItem validator shall accept it as a title if and only if
it is non-empty and at most 500 characters. For any string, the TaskItem validator
shall accept it as a priority if and only if it is one of "low", "medium", or "high".
For any TaskItem with status "pending", completed_at shall be None.
"""

from datetime import datetime, timezone
from uuid import uuid4

import pytest
from hypothesis import given, assume, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from agent.models import TaskItem

NOW = datetime.now(timezone.utc)

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Valid title: 1-500 characters of printable text
valid_title_st = st.text(min_size=1, max_size=500).filter(lambda s: len(s.strip()) > 0)

# Valid priority values
valid_priority_st = st.sampled_from(["low", "medium", "high"])

# Valid status values
valid_status_st = st.sampled_from(["pending", "completed"])

# Invalid priority: any string NOT in {low, medium, high}
invalid_priority_st = st.text(min_size=1, max_size=20).filter(
    lambda s: s not in {"low", "medium", "high"}
)

# Title that is too long: 501-600 chars
too_long_title_st = st.text(min_size=501, max_size=600)


# ---------------------------------------------------------------------------
# Property 9: TaskItem validation — valid construction
# ---------------------------------------------------------------------------


class TestTaskItemValidConstruction:
    """Valid TaskItem construction with title 1-500 chars, valid priority, valid status."""

    @given(title=valid_title_st, priority=valid_priority_st, status=valid_status_st)
    @settings(max_examples=50)
    def test_valid_task_item_accepted(self, title: str, priority: str, status: str):
        """**Validates: Requirements 8.8, 8.9, 16.2**"""
        completed_at = NOW if status == "completed" else None
        task = TaskItem(
            id=uuid4(),
            title=title,
            status=status,
            priority=priority,
            created_at=NOW,
            completed_at=completed_at,
        )
        assert task.title == title
        assert task.priority == priority
        assert task.status == status


# ---------------------------------------------------------------------------
# Property 9: Title must be non-empty
# ---------------------------------------------------------------------------


class TestTaskItemTitleNonEmpty:
    """Empty title strings are rejected."""

    def test_empty_title_rejected(self):
        """**Validates: Requirements 8.8, 16.2**"""
        with pytest.raises(ValidationError):
            TaskItem(
                id=uuid4(),
                title="",
                status="pending",
                created_at=NOW,
            )


# ---------------------------------------------------------------------------
# Property 9: Title must be at most 500 characters
# ---------------------------------------------------------------------------


class TestTaskItemTitleMaxLength:
    """Titles longer than 500 characters are rejected."""

    @given(title=too_long_title_st)
    @settings(max_examples=30)
    def test_title_over_500_rejected(self, title: str):
        """**Validates: Requirements 8.8, 16.2**"""
        assert len(title) > 500
        with pytest.raises(ValidationError):
            TaskItem(
                id=uuid4(),
                title=title,
                status="pending",
                created_at=NOW,
            )


# ---------------------------------------------------------------------------
# Property 9: Priority must be one of {low, medium, high}
# ---------------------------------------------------------------------------


class TestTaskItemPriorityValidation:
    """Priority must be one of low, medium, high."""

    @given(priority=invalid_priority_st)
    @settings(max_examples=30)
    def test_invalid_priority_rejected(self, priority: str):
        """**Validates: Requirements 8.9, 16.2**"""
        with pytest.raises(ValidationError):
            TaskItem(
                id=uuid4(),
                title="Valid title",
                status="pending",
                priority=priority,
                created_at=NOW,
            )


# ---------------------------------------------------------------------------
# Property 9: completed_at must be None when status is "pending"
# ---------------------------------------------------------------------------


class TestTaskItemCompletedAtConsistency:
    """completed_at must be None when status is pending."""

    @given(data=st.data())
    @settings(max_examples=30)
    def test_pending_with_completed_at_rejected(self, data):
        """**Validates: Requirements 16.2**"""
        completed_at = data.draw(
            st.datetimes(
                min_value=datetime(2020, 1, 1),
                max_value=datetime(2030, 1, 1),
                timezones=st.just(timezone.utc),
            )
        )
        with pytest.raises(ValidationError, match="completed_at must be None"):
            TaskItem(
                id=uuid4(),
                title="Some task",
                status="pending",
                created_at=NOW,
                completed_at=completed_at,
            )


# ---------------------------------------------------------------------------
# Property 9: completed_at can be set when status is "completed"
# ---------------------------------------------------------------------------


class TestTaskItemCompletedStatus:
    """completed_at can be set when status is completed."""

    @given(
        completed_at=st.datetimes(
            min_value=datetime(2020, 1, 1),
            max_value=datetime(2030, 1, 1),
            timezones=st.just(timezone.utc),
        )
    )
    @settings(max_examples=30)
    def test_completed_with_completed_at_accepted(self, completed_at: datetime):
        """**Validates: Requirements 16.2**"""
        task = TaskItem(
            id=uuid4(),
            title="Done task",
            status="completed",
            created_at=NOW,
            completed_at=completed_at,
        )
        assert task.completed_at == completed_at
        assert task.status == "completed"


# ---------------------------------------------------------------------------
# Property 9: Default priority is "medium"
# ---------------------------------------------------------------------------


class TestTaskItemDefaultPriority:
    """Default priority is medium when not specified."""

    @given(title=valid_title_st)
    @settings(max_examples=20)
    def test_default_priority_is_medium(self, title: str):
        """**Validates: Requirements 8.9**"""
        task = TaskItem(
            id=uuid4(),
            title=title,
            status="pending",
            created_at=NOW,
        )
        assert task.priority == "medium"
