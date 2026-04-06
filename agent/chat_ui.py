"""ChatUI — Gradio-based local chat interface for the OpenClaw AWS Agent.

Provides a polished web UI that connects directly to the AgentOrchestrator.
Displays the agent's name from IDENTITY.md in the header and maintains
session state across the conversation.

Requirements: 14.1, 14.2, 14.3, 14.4, 14.5, 14.6
"""

from __future__ import annotations

import logging
from typing import Any

try:
    import gradio as gr
except ImportError:
    gr = None  # type: ignore[assignment]

from agent.orchestrator import AgentHandle, AgentOrchestrator

logger = logging.getLogger(__name__)


def _extract_name(identity_content: str) -> str:
    """Extract the agent's name from IDENTITY.md content (first ``# `` heading).

    Returns ``"OpenClaw Agent"`` when no heading is found.
    """
    for line in identity_content.split("\n"):
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return "OpenClaw Agent"


def create_chat_ui(
    orchestrator: AgentOrchestrator,
    agent_id: str,
    user_id: str,
) -> Any:
    """Create and return a Gradio Blocks chat interface.

    Parameters
    ----------
    orchestrator:
        A fully configured :class:`AgentOrchestrator`.
    agent_id:
        The agent identifier used for identity loading and session creation.
    user_id:
        The user identifier for the chat session.

    Returns
    -------
    gr.Blocks
        The Gradio Blocks demo object (not yet launched).

    Raises
    ------
    RuntimeError
        If the ``gradio`` package is not installed.
    """
    if gr is None:
        raise RuntimeError(
            "The 'gradio' package is required for the chat UI. "
            "Install it with: pip install 'gradio>=4.0.0'"
        )

    # --- Session state (mutable, shared across messages) ---
    # We store the AgentHandle and any init error in a simple dict so that
    # the respond() closure can mutate them across calls.
    session: dict[str, Any] = {
        "agent": None,
        "init_error": None,
    }

    # --- Attempt initial agent creation ---
    agent_name = "OpenClaw Agent"
    try:
        handle: AgentHandle = orchestrator.create_agent(agent_id, user_id)
        session["agent"] = handle
        agent_name = _extract_name(handle.identity_bundle.identity)
    except Exception as exc:
        session["init_error"] = str(exc)
        logger.error("AgentOrchestrator initialization failed: %s", exc)

    # --- Message handler ---
    def respond(message: str, history: list[dict[str, str]]) -> str:
        # If we have no agent yet, try to reinitialize (Req 14.6)
        if session["agent"] is None:
            try:
                handle = orchestrator.create_agent(agent_id, user_id)
                session["agent"] = handle
                session["init_error"] = None
                # Update agent name on successful reinit
                nonlocal agent_name
                agent_name = _extract_name(handle.identity_bundle.identity)
            except Exception as exc:
                session["init_error"] = str(exc)
                logger.error("Agent reinitialisation failed: %s", exc)
                return (
                    f"⚠️ Agent initialization failed: {exc}\n\n"
                    "I'll try again on your next message."
                )

        return orchestrator.handle_message(session["agent"], message)

    # --- Build Gradio UI ---
    with gr.Blocks(title=f"{agent_name} — OpenClaw Agent") as demo:
        gr.Markdown(f"## {agent_name}")
        gr.Markdown("_Powered by OpenClaw on AWS_")

        # Show init error banner if agent failed to start (Req 14.5)
        if session["init_error"] is not None:
            gr.Markdown(
                f"⚠️ **Initialization error:** {session['init_error']}\n\n"
                "The agent will attempt to reinitialize when you send a message."
            )

        gr.ChatInterface(
            fn=respond,
            examples=[
                "What's on my task list?",
                "Remind me to review PRs every morning at 9am",
                "Tell me about yourself",
            ],
        )

    return demo


def launch(
    orchestrator: AgentOrchestrator,
    agent_id: str,
    user_id: str,
    port: int = 7860,
) -> None:
    """Create and launch the Gradio chat UI on localhost.

    Parameters
    ----------
    orchestrator:
        A fully configured :class:`AgentOrchestrator`.
    agent_id:
        The agent identifier.
    user_id:
        The user identifier.
    port:
        The local port to serve on (default ``7860``).
    """
    demo = create_chat_ui(orchestrator, agent_id, user_id)
    demo.launch(server_port=port, share=False)
