"""AgentCore Runtime entrypoint for the OpenClaw AWS Agent."""
import json

from bedrock_agentcore.runtime import BedrockAgentCoreApp

from agent.main import _build_components

app = BedrockAgentCoreApp()


@app.entrypoint
def invoke(payload):
    """Handle AgentCore Runtime invocations."""
    agent_id = payload.get("agent_id", "")
    if not agent_id:
        return {"status": "error", "error_message": "Missing agent_id in event"}

    _config, orchestrator, heartbeat_handler = _build_components(agent_id)

    # Route heartbeat vs interactive
    if payload.get("type") == "heartbeat":
        return heartbeat_handler.handle_heartbeat(payload)

    # Interactive session
    user_id = payload.get("user_id", "anonymous")
    message = payload.get("message", "")
    if not message:
        return {"status": "error", "error_message": "Missing message in event"}

    try:
        handle = orchestrator.create_agent(agent_id, user_id)
        response = orchestrator.handle_message(handle, message)
        orchestrator.end_session(handle)
        # Ensure output is always a plain string
        if not isinstance(response, str):
            if isinstance(response, dict):
                content = response.get("content", response)
                if isinstance(content, list):
                    response = "\n".join(
                        b.get("text", str(b)) if isinstance(b, dict) else str(b)
                        for b in content
                    )
                elif isinstance(content, str):
                    response = content
                else:
                    response = json.dumps(response)
            else:
                response = str(response)
        return {"status": "completed", "output": response}
    except Exception as exc:
        return {"status": "error", "error_message": str(exc)}


if __name__ == "__main__":
    app.run()
