"""Utility tools — general-purpose helpers for the agent.

Implements Requirement 7.1 (ToolRegistry registers utility tools).
"""

from __future__ import annotations

from datetime import datetime, timezone

from agent.tools._decorator import tool


@tool
def get_current_date() -> str:
    """Get the current UTC date and time as an ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()
