"""schedule_task tool — scheduled agent invocations backed by EventBridge + S3.

Implements Requirements 9.1–9.8, 17.3, 17.4.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any

from agent.models import HeartbeatPayload, ScheduleItem
from agent.tools._decorator import tool

logger = logging.getLogger(__name__)

_KEBAB_CASE_PATTERN = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
_CRON_PATTERN = re.compile(r"^cron\(.+\)$")
_RATE_PATTERN = re.compile(r"^rate\(\d+\s+(minute|minutes|hour|hours|day|days)\)$")


class ScheduleManager:
    """Encapsulates schedule CRUD operations with EventBridge Scheduler + S3.

    Parameters
    ----------
    s3_client:
        A boto3 S3 client (or compatible mock).
    scheduler_client:
        A boto3 EventBridge Scheduler client (or compatible mock).
    bucket:
        The S3 bucket name used for storage.
    schedule_group:
        The EventBridge schedule group name.
    agent_runtime_arn:
        The ARN of the AgentCore Runtime target for EventBridge.
    scheduler_role_arn:
        The IAM role ARN assumed by EventBridge Scheduler.
    """

    def __init__(
        self,
        s3_client: Any,
        scheduler_client: Any,
        bucket: str,
        schedule_group: str,
        agent_runtime_arn: str,
        scheduler_role_arn: str,
    ) -> None:
        self._s3 = s3_client
        self._scheduler = scheduler_client
        self._bucket = bucket
        self._schedule_group = schedule_group
        self._agent_runtime_arn = agent_runtime_arn
        self._scheduler_role_arn = scheduler_role_arn

    # ------------------------------------------------------------------
    # Public tool entry-point
    # ------------------------------------------------------------------

    @tool
    def schedule_task(
        self,
        action: str,
        agent_id: str,
        user_id: str,
        name: str = "",
        description: str = "",
        cron_expression: str = "",
        task_prompt: str = "",
        schedule_id: str = "",
    ) -> str:
        """Manage scheduled agent invocations. Actions: create, list, delete.

        Args:
            action: One of 'create', 'list', 'delete'
            agent_id: The agent's identifier
            user_id: The user's identifier
            name: Schedule name (required for 'create', kebab-case)
            description: Human-readable description (required for 'create')
            cron_expression: EventBridge cron expression (required for 'create')
            task_prompt: What the agent should do when triggered (required for 'create')
            schedule_id: Schedule ID prefix (required for 'delete')
        """
        key = f"agents/{agent_id}/schedules.json"
        schedules = self._load_schedules(key)

        if action == "create":
            return self._create(key, schedules, agent_id, user_id, name, description, cron_expression, task_prompt)
        if action == "list":
            return self._list(schedules)
        if action == "delete":
            return self._delete(key, schedules, agent_id, schedule_id)

        return f"Unknown action: {action}. Use create, list, or delete."

    # ------------------------------------------------------------------
    # Action helpers
    # ------------------------------------------------------------------

    def _create(
        self,
        key: str,
        schedules: list[dict],
        agent_id: str,
        user_id: str,
        name: str,
        description: str,
        cron_expression: str,
        task_prompt: str,
    ) -> str:
        # Validate name (Req 9.6)
        if not name or not name.strip():
            return "Schedule name must be non-empty."
        if len(name) > 100:
            return "Schedule name must be at most 100 characters."
        if not _KEBAB_CASE_PATTERN.match(name):
            return "Schedule name must be kebab-case (lowercase alphanumeric separated by hyphens)."

        # Validate cron expression (Req 9.7)
        if not cron_expression or not cron_expression.strip():
            return "Cron expression must be non-empty."
        if not _CRON_PATTERN.match(cron_expression) and not _RATE_PATTERN.match(cron_expression):
            return "Invalid cron expression. Must be a valid EventBridge cron() or rate() expression."

        now = datetime.now(timezone.utc)
        schedule_name = f"{agent_id}-{name}"  # Req 9.8: prefix with agent_id

        payload = {
            "type": "heartbeat",
            "task": task_prompt,
            "agent_id": agent_id,
            "user_id": user_id,
        }

        # Req 9.2: Create EventBridge rule FIRST, before persisting to S3
        try:
            response = self._scheduler.create_schedule(
                Name=schedule_name,
                GroupName=self._schedule_group,
                ScheduleExpression=cron_expression,
                FlexibleTimeWindow={"Mode": "OFF"},
                Target={
                    "Arn": self._agent_runtime_arn,
                    "RoleArn": self._scheduler_role_arn,
                    "Input": json.dumps(payload),
                },
                State="ENABLED",
            )
            rule_arn = response["ScheduleArn"]
        except Exception as exc:
            # Req 9.5: Return clear error without persisting to S3
            logger.error("Failed to create EventBridge schedule '%s': %s", schedule_name, exc)
            return f"Failed to create schedule: {exc}"

        new_schedule = ScheduleItem(
            id=uuid.uuid4(),
            name=name,
            description=description,
            cron_expression=cron_expression,
            payload=HeartbeatPayload(**payload),
            eventbridge_rule_arn=rule_arn,
            status="active",
            created_at=now,
            last_triggered_at=None,
            next_trigger_at=None,
        )

        schedules.append(new_schedule.model_dump(mode="json"))
        self._save_schedules(key, schedules)
        return f"Schedule created: '{name}' ({cron_expression}) — {description}"

    @staticmethod
    def _list(schedules: list[dict]) -> str:
        if not schedules:
            return "No scheduled tasks. Your schedule is empty."

        active = [s for s in schedules if s.get("status") == "active"]
        if not active:
            return "No active scheduled tasks."

        lines: list[str] = [f"Active schedules ({len(active)}):"]
        for s in active:
            last = s.get("last_triggered_at") or "never"
            lines.append(
                f"  [{s.get('cron_expression', '')}] {s.get('name', '')}: "
                f"{s.get('description', '')} "
                f"(last run: {last}, id: {str(s.get('id', ''))[:8]})"
            )
        return "\n".join(lines)

    def _delete(
        self, key: str, schedules: list[dict], agent_id: str, schedule_id: str
    ) -> str:
        target = None
        remaining: list[dict] = []
        for s in schedules:
            if str(s.get("id", "")).startswith(schedule_id):
                target = s
            else:
                remaining.append(s)

        if target is None:
            return f"Schedule not found with id starting with '{schedule_id}'"

        # Req 9.4: Delete EventBridge rule first, then remove from S3
        schedule_name = f"{agent_id}-{target.get('name', '')}"
        try:
            self._scheduler.delete_schedule(
                Name=schedule_name, GroupName=self._schedule_group
            )
        except Exception as exc:
            # If the rule is already gone, that's fine
            err_code = ""
            try:
                err_code = exc.response.get("Error", {}).get("Code", "")  # type: ignore[union-attr]
            except Exception:
                pass
            if err_code != "ResourceNotFoundException":
                logger.warning(
                    "Failed to delete EventBridge schedule '%s': %s", schedule_name, exc
                )

        self._save_schedules(key, remaining)
        return f"Schedule deleted: '{target.get('name', '')}'"

    # ------------------------------------------------------------------
    # S3 persistence helpers
    # ------------------------------------------------------------------

    def _load_schedules(self, key: str) -> list[dict]:
        """Load schedules from S3. Returns [] on missing file or malformed JSON."""
        try:
            obj = self._s3.get_object(Bucket=self._bucket, Key=key)
            raw = obj["Body"].read()
            data = json.loads(raw)
            return data.get("schedules", [])
        except self._s3.exceptions.NoSuchKey:
            return []
        except json.JSONDecodeError:
            logger.warning(
                "Malformed schedules JSON at s3://%s/%s — starting with empty list",
                self._bucket,
                key,
            )
            return []
        except Exception as exc:
            # Handle ClientError for missing key in some SDK versions
            err_code = ""
            try:
                err_code = exc.response.get("Error", {}).get("Code", "")  # type: ignore[union-attr]
            except Exception:
                pass
            if err_code in ("NoSuchKey", "404"):
                return []
            raise

    def _save_schedules(self, key: str, schedules: list[dict]) -> None:
        """Serialize schedules to S3 with ContentType application/json (Req 17.3)."""
        self._s3.put_object(
            Bucket=self._bucket,
            Key=key,
            Body=json.dumps({"schedules": schedules}, indent=2),
            ContentType="application/json",
        )
