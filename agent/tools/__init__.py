"""Agent tools subpackage — custom tools registered via Strands @tool decorator.

Provides a :class:`ToolRegistry` that creates all tool instances and
exposes them as a flat list of callables for registration with a
Strands Agent.

Implements Requirement 7.1.
"""

from __future__ import annotations

from typing import Any, Callable

from agent.identity import IdentityManager
from agent.memory import MemoryManager
from agent.tools.identity_tools import IdentityTools
from agent.tools.memory_tools import MemoryTools
from agent.tools.schedules import ScheduleManager
from agent.tools.tasks import TaskManager
from agent.tools.utils import get_current_date


class ToolRegistry:
    """Creates and holds references to all tool instances.

    Parameters
    ----------
    identity_manager:
        The :class:`IdentityManager` for identity file operations.
    memory_manager:
        The :class:`MemoryManager` for memory operations.
    s3_client:
        A boto3 S3 client (or compatible mock).
    bucket:
        The S3 bucket name used for storage.
    scheduler_client:
        A boto3 EventBridge Scheduler client (or compatible mock).
        May be ``None`` if scheduling is not configured.
    schedule_group:
        The EventBridge schedule group name.
    agent_runtime_arn:
        The ARN of the AgentCore Runtime target for EventBridge.
    scheduler_role_arn:
        The IAM role ARN assumed by EventBridge Scheduler.
    """

    def __init__(
        self,
        identity_manager: IdentityManager,
        memory_manager: MemoryManager,
        s3_client: Any,
        bucket: str,
        scheduler_client: Any = None,
        schedule_group: str = "",
        agent_runtime_arn: str = "",
        scheduler_role_arn: str = "",
    ) -> None:
        self._task_manager = TaskManager(s3_client=s3_client, bucket=bucket)
        self._identity_tools = IdentityTools(identity_manager=identity_manager)
        self._memory_tools = MemoryTools(memory_manager=memory_manager)

        self._schedule_manager: ScheduleManager | None = None
        if scheduler_client is not None:
            self._schedule_manager = ScheduleManager(
                s3_client=s3_client,
                scheduler_client=scheduler_client,
                bucket=bucket,
                schedule_group=schedule_group,
                agent_runtime_arn=agent_runtime_arn,
                scheduler_role_arn=scheduler_role_arn,
            )

    def get_tools(self) -> list[Callable[..., Any]]:
        """Return a list of all tool callables for registration with a Strands Agent."""
        tools: list[Callable[..., Any]] = [
            self._task_manager.manage_tasks,
            self._identity_tools.update_identity,
            self._identity_tools.update_user_profile,
            self._identity_tools.save_to_memory,
            self._memory_tools.search_memory,
            get_current_date,
        ]
        if self._schedule_manager is not None:
            tools.append(self._schedule_manager.schedule_task)
        return tools
