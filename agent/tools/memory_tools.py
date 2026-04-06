"""Memory tools — search across stored memories via the agent.

Wraps MemoryManager.search_memory for use as a Strands @tool callable.

Implements Requirement 7.1 (ToolRegistry registers memory tools).
"""

from __future__ import annotations

import json

from agent.memory import MemoryManager
from agent.tools._decorator import tool


class MemoryTools:
    """Provides @tool-decorated methods for memory search.

    Parameters
    ----------
    memory_manager:
        The :class:`MemoryManager` instance used for memory operations.
    """

    def __init__(self, memory_manager: MemoryManager) -> None:
        self._memory_manager = memory_manager

    @tool
    def search_memory(self, agent_id: str, query: str) -> str:
        """Search across all stored memories for a given agent.

        Args:
            agent_id: The agent's identifier.
            query: The search query string.
        """
        results = self._memory_manager.search_memory(agent_id, query)
        if not results:
            return "No memories found matching your query."
        return json.dumps(results, indent=2, default=str)
