"""Identity tools — update dynamic identity files via the agent.

Wraps IdentityManager.update_file for use as Strands @tool callables.

Implements Requirement 7.1 (ToolRegistry registers identity tools).
"""

from __future__ import annotations

from agent.identity import IdentityManager
from agent.tools._decorator import tool


class IdentityTools:
    """Provides @tool-decorated methods for updating identity files.

    Parameters
    ----------
    identity_manager:
        The :class:`IdentityManager` instance used for S3 persistence.
    """

    def __init__(self, identity_manager: IdentityManager) -> None:
        self._identity_manager = identity_manager

    @tool
    def update_identity(self, agent_id: str, content: str) -> str:
        """Update the agent's IDENTITY.md file with new content.

        Args:
            agent_id: The agent's identifier.
            content: The new content for IDENTITY.md.
        """
        self._identity_manager.update_file(agent_id, "identity", content)
        return "IDENTITY.md updated successfully."

    @tool
    def update_user_profile(self, agent_id: str, content: str) -> str:
        """Update the user's USER.md profile file with new content.

        Args:
            agent_id: The agent's identifier.
            content: The new content for USER.md.
        """
        self._identity_manager.update_file(agent_id, "user_profile", content)
        return "USER.md updated successfully."

    @tool
    def save_to_memory(self, agent_id: str, content: str) -> str:
        """Save important information to the agent's MEMORY.md durable memory file.

        Args:
            agent_id: The agent's identifier.
            content: The new content for MEMORY.md.
        """
        self._identity_manager.update_file(agent_id, "durable_memory", content)
        return "MEMORY.md updated successfully."
