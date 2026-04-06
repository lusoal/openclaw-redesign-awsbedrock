"""Heartbeat handler — processes scheduled EventBridge invocations.

Routes heartbeat payloads to the appropriate handler, loads schedules
from S3, executes task prompts via the LLM, and updates timestamps.

Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Callable

from agent.models import HeartbeatPayload, HeartbeatResult

logger = logging.getLogger(__name__)


class HeartbeatHandler:
    """Handles heartbeat invocations from EventBridge Scheduler.

    Parameters
    ----------
    s3_client:
        A boto3 S3 client (or compatible mock).
    bucket:
        The S3 bucket name where schedules are stored.
    deployed_agent_id:
        The agent_id of the deployed agent, used to validate incoming
        heartbeat payloads.
    agent_factory:
        Callable that accepts ``(agent_id, user_id, task_prompt)`` and
        returns the LLM response as a string.  Injected for testability.
    """

    def __init__(
        self,
        s3_client: Any,
        bucket: str,
        deployed_agent_id: str,
        agent_factory: Callable[[str, str, str], str],
    ) -> None:
        self._s3 = s3_client
        self._bucket = bucket
        self._deployed_agent_id = deployed_agent_id
        self._agent_factory = agent_factory

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def handle_heartbeat(self, event: dict[str, Any]) -> dict[str, Any]:
        """Process a heartbeat invocation.

        Returns a dict matching the HeartbeatResult schema:
        - ``{"status": "completed", "output": "..."}`` when a task produces output
        - ``{"status": "heartbeat_ok"}`` when no pending tasks
        - ``{"status": "error", "error_message": "..."}`` on failure

        Requirements: 10.1–10.8
        """
        try:
            return self._process(event)
        except Exception as exc:
            # Req 10.7: Return error without crashing
            logger.error("Heartbeat processing failed: %s", exc)
            return HeartbeatResult(
                status="error",
                error_message=str(exc),
            ).model_dump()

    # ------------------------------------------------------------------
    # Internal processing
    # ------------------------------------------------------------------

    def _process(self, event: dict[str, Any]) -> dict[str, Any]:
        """Core heartbeat processing logic."""
        # Validate payload (Req 10.6, 16.4)
        payload = self._validate_payload(event)

        # Load schedules from S3 (Req 10.2, 10.8)
        schedules = self._load_schedules(payload.agent_id)

        # Find matching schedule for this task
        matching = self._find_matching_schedule(schedules, payload.task)

        if matching is None:
            # No pending tasks (Req 10.4)
            return HeartbeatResult(status="heartbeat_ok").model_dump()

        # Execute task prompt via LLM (Req 10.2)
        result = self._agent_factory(
            payload.agent_id,
            payload.user_id,
            payload.task,
        )

        # Update last_triggered_at (Req 10.5)
        self._update_trigger_time(
            payload.agent_id, schedules, matching, payload.task
        )

        # Req 10.3: Return completed with output
        if result and result.strip():
            return HeartbeatResult(
                status="completed",
                output=result,
            ).model_dump()

        return HeartbeatResult(status="heartbeat_ok").model_dump()

    def _validate_payload(self, event: dict[str, Any]) -> HeartbeatPayload:
        """Validate and parse the heartbeat payload.

        Raises ValueError if the payload is invalid or the agent_id
        does not match the deployed agent.

        Requirements: 10.6, 16.4
        """
        # Parse with Pydantic validation
        payload = HeartbeatPayload(**event)

        # Validate agent_id matches deployed agent (Req 10.6)
        if payload.agent_id != self._deployed_agent_id:
            raise ValueError(
                f"Heartbeat agent_id '{payload.agent_id}' does not match "
                f"deployed agent '{self._deployed_agent_id}'"
            )

        return payload

    def _load_schedules(self, agent_id: str) -> list[dict[str, Any]]:
        """Load schedules.json from S3.

        Returns an empty list if the file is missing or corrupted (Req 10.8).
        """
        key = f"agents/{agent_id}/schedules.json"
        try:
            obj = self._s3.get_object(Bucket=self._bucket, Key=key)
            raw = obj["Body"].read()
            data = json.loads(raw)
            return data.get("schedules", [])
        except Exception as exc:
            # Req 10.8: Handle missing/corrupted as empty schedule list
            err_code = ""
            try:
                err_code = getattr(exc, "response", {}).get("Error", {}).get("Code", "")
            except Exception:
                pass

            if err_code in ("NoSuchKey", "404"):
                logger.info(
                    "No schedules.json found for agent %s — treating as empty",
                    agent_id,
                )
            else:
                logger.warning(
                    "Failed to load schedules.json for agent %s: %s — "
                    "treating as empty schedule list",
                    agent_id,
                    exc,
                )
            return []

    @staticmethod
    def _find_matching_schedule(
        schedules: list[dict[str, Any]], task_prompt: str
    ) -> dict[str, Any] | None:
        """Find the schedule whose payload task matches the given prompt."""
        for schedule in schedules:
            if schedule.get("status") != "active":
                continue
            payload = schedule.get("payload", {})
            if payload.get("task") == task_prompt:
                return schedule
        return None

    def _update_trigger_time(
        self,
        agent_id: str,
        schedules: list[dict[str, Any]],
        matched: dict[str, Any],
        task_prompt: str,
    ) -> None:
        """Update last_triggered_at for the matched schedule and persist.

        Requirements: 10.5
        """
        now = datetime.now(timezone.utc).isoformat()
        matched_id = matched.get("id")

        for schedule in schedules:
            if schedule.get("id") == matched_id:
                schedule["last_triggered_at"] = now
                break

        key = f"agents/{agent_id}/schedules.json"
        try:
            self._s3.put_object(
                Bucket=self._bucket,
                Key=key,
                Body=json.dumps({"schedules": schedules}, indent=2),
                ContentType="application/json",
            )
        except Exception as exc:
            logger.warning(
                "Failed to update last_triggered_at for agent %s: %s",
                agent_id,
                exc,
            )
