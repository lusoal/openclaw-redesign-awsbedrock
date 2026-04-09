"""AgentCore Runtime entrypoint for the OpenClaw AWS Agent."""
import json
import logging
import threading

from bedrock_agentcore.runtime import BedrockAgentCoreApp

from agent.main import _build_components

logger = logging.getLogger(__name__)

app = BedrockAgentCoreApp()

# Cache components per agent (expensive to build)
_components_cache = {}
_components_lock = threading.Lock()

# Cache agent handles per user session (preserves conversation history)
_agent_cache = {}
_agent_lock = threading.Lock()


def _get_components(agent_id):
    with _components_lock:
        if agent_id not in _components_cache:
            _components_cache[agent_id] = _build_components(agent_id)
        return _components_cache[agent_id]


def _get_or_create_agent(agent_id, user_id):
    """Get cached agent handle or create a new one."""
    cache_key = f"{agent_id}:{user_id}"
    with _agent_lock:
        if cache_key in _agent_cache:
            return _agent_cache[cache_key]

    # Create outside lock (slow operation)
    _config, orchestrator, _hb = _get_components(agent_id)
    handle = orchestrator.create_agent(agent_id, user_id)

    with _agent_lock:
        # Double-check after lock
        if cache_key not in _agent_cache:
            _agent_cache[cache_key] = (orchestrator, handle)
        return _agent_cache[cache_key]


@app.entrypoint
def invoke(payload):
    """Handle AgentCore Runtime invocations."""
    agent_id = payload.get("agent_id", "")
    if not agent_id:
        return {"status": "error", "error_message": "Missing agent_id in event"}

    # Route heartbeat
    if payload.get("type") == "heartbeat":
        _config, _orch, heartbeat_handler = _get_components(agent_id)
        return heartbeat_handler.handle_heartbeat(payload)

    # Interactive session
    user_id = payload.get("user_id", "anonymous")
    message = payload.get("message", "")
    if not message:
        return {"status": "error", "error_message": "Missing message in event"}

    try:
        orchestrator, handle = _get_or_create_agent(agent_id, user_id)
        # Lock the agent during invocation to prevent concurrent access
        cache_key = f"{agent_id}:{user_id}"
        with _agent_lock:
            response = orchestrator.handle_message(handle, message)

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
        logger.error("Invocation failed: %s", exc)
        # Clear cached agent on error so next request rebuilds
        cache_key = f"{agent_id}:{user_id}"
        with _agent_lock:
            _agent_cache.pop(cache_key, None)
        return {"status": "error", "error_message": str(exc)}


if __name__ == "__main__":
    app.run()
