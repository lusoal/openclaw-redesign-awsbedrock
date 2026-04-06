"""manage_tasks tool — personal task management backed by S3.

Implements Requirements 8.1–8.9, 17.1, 17.2.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from agent.models import TaskItem
from agent.tools._decorator import tool

logger = logging.getLogger(__name__)


class TaskManager:
    """Encapsulates task CRUD operations with S3 persistence.

    Parameters
    ----------
    s3_client:
        A boto3 S3 client (or compatible mock).
    bucket:
        The S3 bucket name used for storage.
    """

    def __init__(self, s3_client: Any, bucket: str) -> None:
        self._s3 = s3_client
        self._bucket = bucket

    # ------------------------------------------------------------------
    # Public tool entry-point
    # ------------------------------------------------------------------

    @tool
    def manage_tasks(
        self,
        action: str,
        agent_id: str,
        user_id: str,
        title: str = "",
        task_id: str = "",
        priority: str = "medium",
    ) -> str:
        """Manage personal tasks for the user. Actions: add, list, complete, delete.

        Args:
            action: One of 'add', 'list', 'complete', 'delete'
            agent_id: The agent's identifier
            user_id: The user's identifier
            title: Task title (required for 'add')
            task_id: Task ID prefix (required for 'complete' and 'delete')
            priority: Priority level — 'low', 'medium', or 'high' (used with 'add')
        """
        key = f"agents/{agent_id}/tasks/{user_id}.json"
        tasks = self._load_tasks(key)

        if action == "add":
            return self._add(key, tasks, title, priority)
        if action == "list":
            return self._list(tasks)
        if action == "complete":
            return self._complete(key, tasks, task_id)
        if action == "delete":
            return self._delete(key, tasks, task_id)

        return f"Unknown action: {action}. Use add, list, complete, or delete."

    # ------------------------------------------------------------------
    # Action helpers
    # ------------------------------------------------------------------

    def _add(
        self, key: str, tasks: list[dict], title: str, priority: str
    ) -> str:
        # Validate title (Req 8.8)
        if not title or not title.strip():
            return "Title must be non-empty."
        if len(title) > 500:
            return "Title must be at most 500 characters."

        # Validate priority (Req 8.9)
        if priority not in ("low", "medium", "high"):
            priority = "medium"

        now = datetime.now(timezone.utc)
        new_task = TaskItem(
            id=uuid.uuid4(),
            title=title,
            status="pending",
            priority=priority,
            created_at=now,
            completed_at=None,
        )
        tasks.append(new_task.model_dump(mode="json"))
        self._save_tasks(key, tasks)
        return f"Task added: '{title}' (priority: {priority})"

    @staticmethod
    def _list(tasks: list[dict]) -> str:
        if not tasks:
            return "No tasks found. Your task list is empty."

        pending = [t for t in tasks if t.get("status") == "pending"]
        done = [t for t in tasks if t.get("status") == "completed"]
        lines: list[str] = []

        if pending:
            lines.append(f"Pending ({len(pending)}):")
            for t in pending:
                tid = str(t.get("id", ""))[:8]
                lines.append(f"  [{t.get('priority', 'medium')}] {t.get('title', '')} (id: {tid})")

        if done:
            lines.append(f"Completed ({len(done)}):")
            for t in done:
                lines.append(f"  ✓ {t.get('title', '')}")

        return "\n".join(lines)

    def _complete(self, key: str, tasks: list[dict], task_id: str) -> str:
        for t in tasks:
            if str(t.get("id", "")).startswith(task_id):
                t["status"] = "completed"
                t["completed_at"] = datetime.now(timezone.utc).isoformat()
                self._save_tasks(key, tasks)
                return f"Task completed: '{t.get('title', '')}'"
        return f"Task not found with id starting with '{task_id}'"

    def _delete(self, key: str, tasks: list[dict], task_id: str) -> str:
        original_len = len(tasks)
        remaining = [t for t in tasks if not str(t.get("id", "")).startswith(task_id)]
        if len(remaining) < original_len:
            self._save_tasks(key, remaining)
            return "Task deleted."
        return f"Task not found with id starting with '{task_id}'"

    # ------------------------------------------------------------------
    # S3 persistence helpers
    # ------------------------------------------------------------------

    def _load_tasks(self, key: str) -> list[dict]:
        """Load tasks from S3. Returns [] on missing file or malformed JSON."""
        try:
            obj = self._s3.get_object(Bucket=self._bucket, Key=key)
            raw = obj["Body"].read()
            data = json.loads(raw)
            return data.get("tasks", [])
        except self._s3.exceptions.NoSuchKey:
            return []
        except json.JSONDecodeError:
            logger.warning("Malformed tasks JSON at s3://%s/%s — starting with empty list", self._bucket, key)
            return []
        except Exception:
            # ClientError for missing key in some SDK versions
            try:
                code = ""
                import botocore.exceptions  # noqa: F811
                # re-raise if it's not a missing-key error
                raise
            except Exception as exc:
                err_code = getattr(getattr(exc, "response", {}), "get", lambda *a: None)
                if callable(err_code):
                    try:
                        err_code = exc.response.get("Error", {}).get("Code", "")  # type: ignore[union-attr]
                    except Exception:
                        err_code = ""
                if err_code in ("NoSuchKey", "404"):
                    return []
                raise

    def _save_tasks(self, key: str, tasks: list[dict]) -> None:
        """Serialize tasks to S3 with ContentType application/json (Req 17.1)."""
        self._s3.put_object(
            Bucket=self._bucket,
            Key=key,
            Body=json.dumps({"tasks": tasks}, indent=2),
            ContentType="application/json",
        )
