"""Property-based tests for ScheduleItem validation (Property 15).

**Validates: Requirements 9.6, 9.7, 16.3**

For any ScheduleItem, the validator shall accept it if and only if id is a valid
UUID, name is non-empty kebab-case at most 100 characters, and cron_expression is
a valid EventBridge cron or rate expression.
"""

from datetime import datetime, timezone
from uuid import uuid4

import pytest
from hypothesis import given, assume, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from agent.models import HeartbeatPayload, ScheduleItem

NOW = datetime.now(timezone.utc)


def _make_payload(agent_id: str = "test-agent") -> HeartbeatPayload:
    """Helper to create a valid HeartbeatPayload for ScheduleItem construction."""
    return HeartbeatPayload(
        type="heartbeat",
        task="do-something",
        agent_id=agent_id,
        user_id="user-1",
    )


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Valid kebab-case name: lowercase alphanumeric segments separated by hyphens, 1-100 chars
_kebab_segment = st.from_regex(r"[a-z0-9]{1,15}", fullmatch=True)
valid_kebab_name_st = st.lists(_kebab_segment, min_size=1, max_size=6).map(
    lambda segs: "-".join(segs)
).filter(lambda s: 1 <= len(s) <= 100)

# Valid cron expressions
valid_cron_st = st.sampled_from([
    "cron(0 9 * * ? *)",
    "cron(30 14 ? * MON-FRI *)",
    "cron(0 0 1 * ? *)",
    "cron(15 10 * * ? 2025)",
    "cron(0/5 * * * ? *)",
])

# Valid rate expressions
valid_rate_st = st.sampled_from([
    "rate(1 minute)",
    "rate(5 minutes)",
    "rate(1 hour)",
    "rate(12 hours)",
    "rate(1 day)",
    "rate(7 days)",
])

# Combined valid EventBridge expressions
valid_expression_st = st.one_of(valid_cron_st, valid_rate_st)

# Non-kebab-case names: contain uppercase, spaces, or underscores
non_kebab_names_st = st.sampled_from([
    "MySchedule",
    "my schedule",
    "my_schedule",
    "ALLCAPS",
    "has Space",
    "under_score",
    "camelCase",
    "Mixed-Case",
    "trailing-",
    "-leading",
    "double--dash",
    "has.dot",
    "has@symbol",
])

# Names longer than 100 chars (valid kebab chars but too long)
too_long_name_st = st.from_regex(r"[a-z]{101,120}", fullmatch=True)

# Invalid cron expressions: don't match cron() or rate() format
invalid_cron_st = st.sampled_from([
    "0 9 * * ? *",
    "cron 0 9 * * ? *",
    "rate(5 seconds)",
    "every 5 minutes",
    "* * * * *",
    "schedule(0 9 * * ?)",
    "rate(minutes 5)",
    "rate(1 week)",
    "cron()",
    "",
])


# ---------------------------------------------------------------------------
# Property 15: Valid ScheduleItem construction
# ---------------------------------------------------------------------------


class TestScheduleItemValidConstruction:
    """Valid ScheduleItem with UUID id, kebab-case name, valid expression."""

    @given(name=valid_kebab_name_st, expr=valid_expression_st)
    @settings(max_examples=50)
    def test_valid_schedule_item_accepted(self, name: str, expr: str):
        """**Validates: Requirements 9.6, 9.7, 16.3**"""
        item = ScheduleItem(
            id=uuid4(),
            name=name,
            cron_expression=expr,
            payload=_make_payload(),
            created_at=NOW,
        )
        assert item.name == name
        assert item.cron_expression == expr
        assert item.status == "active"


# ---------------------------------------------------------------------------
# Property 15: Non-kebab-case names are rejected
# ---------------------------------------------------------------------------


class TestScheduleItemNameKebabCase:
    """Names that are not kebab-case are rejected."""

    @given(name=non_kebab_names_st)
    @settings(max_examples=30)
    def test_non_kebab_name_rejected(self, name: str):
        """**Validates: Requirements 9.6, 16.3**"""
        with pytest.raises(ValidationError, match="kebab-case"):
            ScheduleItem(
                id=uuid4(),
                name=name,
                cron_expression="cron(0 9 * * ? *)",
                payload=_make_payload(),
                created_at=NOW,
            )


# ---------------------------------------------------------------------------
# Property 15: Names longer than 100 chars are rejected
# ---------------------------------------------------------------------------


class TestScheduleItemNameMaxLength:
    """Names longer than 100 characters are rejected."""

    @given(name=too_long_name_st)
    @settings(max_examples=30)
    def test_name_over_100_rejected(self, name: str):
        """**Validates: Requirements 9.6, 16.3**"""
        assert len(name) > 100
        with pytest.raises(ValidationError):
            ScheduleItem(
                id=uuid4(),
                name=name,
                cron_expression="cron(0 9 * * ? *)",
                payload=_make_payload(),
                created_at=NOW,
            )


# ---------------------------------------------------------------------------
# Property 15: Invalid cron expressions are rejected
# ---------------------------------------------------------------------------


class TestScheduleItemInvalidCronExpression:
    """Expressions not matching cron() or rate() format are rejected."""

    @given(expr=invalid_cron_st)
    @settings(max_examples=30)
    def test_invalid_cron_rejected(self, expr: str):
        """**Validates: Requirements 9.7, 16.3**"""
        with pytest.raises(ValidationError):
            ScheduleItem(
                id=uuid4(),
                name="valid-name",
                cron_expression=expr,
                payload=_make_payload(),
                created_at=NOW,
            )


# ---------------------------------------------------------------------------
# Property 15: Valid cron() expressions are accepted
# ---------------------------------------------------------------------------


class TestScheduleItemValidCronExpressions:
    """Valid cron() expressions are accepted."""

    @given(expr=valid_cron_st)
    @settings(max_examples=20)
    def test_valid_cron_accepted(self, expr: str):
        """**Validates: Requirements 9.7, 16.3**"""
        item = ScheduleItem(
            id=uuid4(),
            name="my-cron-job",
            cron_expression=expr,
            payload=_make_payload(),
            created_at=NOW,
        )
        assert item.cron_expression == expr


# ---------------------------------------------------------------------------
# Property 15: Valid rate() expressions are accepted
# ---------------------------------------------------------------------------


class TestScheduleItemValidRateExpressions:
    """Valid rate() expressions are accepted."""

    @given(expr=valid_rate_st)
    @settings(max_examples=20)
    def test_valid_rate_accepted(self, expr: str):
        """**Validates: Requirements 9.7, 16.3**"""
        item = ScheduleItem(
            id=uuid4(),
            name="my-rate-job",
            cron_expression=expr,
            payload=_make_payload(),
            created_at=NOW,
        )
        assert item.cron_expression == expr
