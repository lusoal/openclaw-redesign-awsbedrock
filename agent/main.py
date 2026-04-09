"""Agent entry point — handles AgentCore Runtime invocations and local CLI.

Routes heartbeat payloads to the heartbeat handler and interactive sessions
to the AgentOrchestrator.  Wires all components together from DeploymentConfig.

Provides a CLI entry point for ``agentcore dev`` local testing.

Requirements: 7.1, 7.2, 10.1, 13.4
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any

import boto3

from agent.config import DeploymentConfig
from agent.heartbeat import HeartbeatHandler
from agent.identity import IdentityManager
from agent.memory import MemoryManager
from agent.model_router import ModelRouter
from agent.models import RuntimeConfig
from agent.orchestrator import AgentOrchestrator
from agent.tools import ToolRegistry

logger = logging.getLogger(__name__)


def _build_components(
    agent_id: str,
    ssm_client: Any = None,
) -> tuple[RuntimeConfig, AgentOrchestrator, HeartbeatHandler]:
    """Create all components from config for a given agent_id.

    Wires: DeploymentConfig -> ModelRouter -> IdentityManager ->
           MemoryManager -> ToolRegistry -> AgentOrchestrator

    Also creates a HeartbeatHandler for heartbeat routing.

    Parameters
    ----------
    agent_id:
        The agent identifier.
    ssm_client:
        Optional SSM client for testing.  When ``None``, a real client
        is created lazily by DeploymentConfig.

    Returns
    -------
    tuple
        (RuntimeConfig, AgentOrchestrator, HeartbeatHandler)
    """
    # 1. DeploymentConfig — read SSM / env vars
    deployment = DeploymentConfig(ssm_client=ssm_client)
    config = deployment.configure_from_stack(agent_id)

    # 2. ModelRouter
    routing_enabled = bool(config.prompt_router_arn)
    model_router = ModelRouter(
        routing_enabled=routing_enabled,
        router_arn=config.prompt_router_arn if routing_enabled else None,
        model_id=config.bedrock_model_id if not routing_enabled else None,
        fallback_model_id=config.bedrock_model_id,
    )

    # 3. S3 client + IdentityManager
    s3_client = boto3.client("s3", region_name=config.bedrock_region)
    identity_manager = IdentityManager(s3_client=s3_client, bucket=config.s3_bucket)

    # 4. MemoryManager — use AgentCore Memory if memory_id is available
    memory_manager = MemoryManager(memory_client=None)
    _memory_config = None
    memory_id = config.memory_id or ""
    if memory_id and memory_id.strip():
        try:
            from bedrock_agentcore.memory.integrations.strands.config import AgentCoreMemoryConfig
            _memory_config = {
                "memory_id": memory_id,
                "region": config.bedrock_region,
            }
            logger.info("AgentCore Memory configured: %s", memory_id)
        except Exception as exc:
            logger.warning("Failed to configure AgentCore Memory: %s", exc)
            _memory_config = None

    # 5. Scheduler client (optional)
    scheduler_client = None
    schedule_group = config.schedule_group_name or ""
    scheduler_role_arn = config.scheduler_role_arn or ""
    if schedule_group:
        scheduler_client = boto3.client(
            "scheduler", region_name=config.bedrock_region
        )

    # 6. ToolRegistry
    gateway_id = config.gateway_id or ""
    agentcore_client = None
    lambda_client = None
    if gateway_id:
        agentcore_client = boto3.client(
            "bedrock-agentcore-control", region_name=config.bedrock_region
        )
        lambda_client = boto3.client(
            "lambda", region_name=config.bedrock_region
        )

    tool_registry = ToolRegistry(
        identity_manager=identity_manager,
        memory_manager=memory_manager,
        s3_client=s3_client,
        bucket=config.s3_bucket,
        scheduler_client=scheduler_client,
        schedule_group=schedule_group,
        agent_runtime_arn=config.scheduler_target_arn or config.agent_runtime_arn or "",
        scheduler_role_arn=scheduler_role_arn,
        agentcore_client=agentcore_client,
        lambda_client=lambda_client,
        gateway_id=gateway_id,
        agent_id=agent_id,
        lambda_role_arn=config.custom_tool_role_arn or "",
    )

    # 7. Gateway MCP tools (if configured)
    gateway_url = config.gateway_url or ""
    _mcp_client = None
    if gateway_url:
        try:
            from strands.tools.mcp.mcp_client import MCPClient
            from mcp.client.streamable_http import streamablehttp_client
            import httpx
            from botocore.auth import SigV4Auth
            from botocore.awsrequest import AWSRequest
            import botocore.session as botocore_session

            logger.info("Connecting to AgentCore Gateway: %s", gateway_url)

            # Custom httpx auth that signs every request with SigV4
            _bsession = botocore_session.get_session()
            _creds = _bsession.get_credentials()
            _region = config.bedrock_region

            class SigV4AuthFlow(httpx.Auth):
                def auth_flow(self, request):
                    aws_req = AWSRequest(
                        method=request.method,
                        url=str(request.url),
                        data=request.content,
                        headers=dict(request.headers),
                    )
                    SigV4Auth(_creds, "bedrock-agentcore", _region).add_auth(aws_req)
                    for k, v in aws_req.headers.items():
                        request.headers[k] = v
                    yield request

            _mcp_client = MCPClient(
                lambda: streamablehttp_client(
                    gateway_url,
                    auth=SigV4AuthFlow(),
                )
            )
            _mcp_client.__enter__()
            logger.info("Gateway MCP client connected")
        except Exception as exc:
            logger.warning("Failed to connect to Gateway MCP: %s", exc)
            _mcp_client = None

    # 8. Agent factory — creates a Strands Agent
    def _agent_factory(
        *, system_prompt: str, tools: list, model_id: str
    ) -> Any:
        try:
            from strands import Agent
            from strands.models.bedrock import BedrockModel

            model = BedrockModel(
                model_id=model_id,
                region_name=config.bedrock_region,
            )

            # Combine built-in tools with Gateway MCP tools (refresh each invocation)
            all_tools = list(tools)
            if _mcp_client is not None:
                try:
                    mcp_tools = _mcp_client.list_tools_sync()
                    all_tools.extend(mcp_tools)
                    logger.info("Loaded %d MCP tools from Gateway", len(mcp_tools))
                except Exception as exc:
                    logger.warning("Failed to load MCP tools: %s", exc)
            elif gateway_url:
                # Try reconnecting if initial connection failed
                try:
                    from strands.tools.mcp.mcp_client import MCPClient
                    from mcp.client.streamable_http import streamablehttp_client
                    import httpx
                    from botocore.auth import SigV4Auth
                    from botocore.awsrequest import AWSRequest
                    import botocore.session as botocore_session

                    _bs = botocore_session.get_session()
                    _cr = _bs.get_credentials()
                    _rg = config.bedrock_region

                    class _SigV4(httpx.Auth):
                        def auth_flow(self, request):
                            aws_req = AWSRequest(method=request.method, url=str(request.url), data=request.content, headers=dict(request.headers))
                            SigV4Auth(_cr, "bedrock-agentcore", _rg).add_auth(aws_req)
                            for k, v in aws_req.headers.items():
                                request.headers[k] = v
                            yield request

                    mc = MCPClient(lambda: streamablehttp_client(gateway_url, auth=_SigV4()))
                    with mc:
                        mcp_tools = mc.list_tools_sync()
                        all_tools.extend(mcp_tools)
                        logger.info("Reconnected to Gateway, loaded %d MCP tools", len(mcp_tools))
                except Exception as exc:
                    logger.warning("Failed to reconnect to Gateway: %s", exc)

            # Create session manager per agent invocation (unique session)
            _session_manager = None
            if _memory_config:
                try:
                    from bedrock_agentcore.memory.integrations.strands.config import AgentCoreMemoryConfig
                    from bedrock_agentcore.memory.integrations.strands.session_manager import AgentCoreMemorySessionManager
                    import uuid as _muuid

                    _session_manager = AgentCoreMemorySessionManager(
                        agentcore_memory_config=AgentCoreMemoryConfig(
                            memory_id=_memory_config["memory_id"],
                            session_id=str(_muuid.uuid4()),
                            actor_id="openclaw-agent",
                        ),
                        region_name=_memory_config["region"],
                    )
                except Exception as exc:
                    logger.warning("Failed to create memory session: %s", exc)

            return Agent(
                model=model,
                tools=all_tools,
                system_prompt=system_prompt,
                session_manager=_session_manager,
            )
        except ImportError:
            logger.warning(
                "strands-agents not installed; using passthrough agent"
            )
            # Fallback for environments without strands installed
            def _passthrough(message: str) -> str:
                return f"[echo] {message}"

            return _passthrough

    # 8. AgentOrchestrator
    orchestrator = AgentOrchestrator(
        identity_manager=identity_manager,
        memory_manager=memory_manager,
        tool_registry=tool_registry,
        model_router=model_router,
        agent_factory=_agent_factory,
    )

    # 9. HeartbeatHandler
    def _heartbeat_agent_factory(
        hb_agent_id: str, hb_user_id: str, task_prompt: str
    ) -> str:
        handle = orchestrator.create_agent(hb_agent_id, hb_user_id)
        return orchestrator.handle_message(handle, task_prompt)

    heartbeat_handler = HeartbeatHandler(
        s3_client=s3_client,
        bucket=config.s3_bucket,
        deployed_agent_id=agent_id,
        agent_factory=_heartbeat_agent_factory,
    )

    return config, orchestrator, heartbeat_handler


def handle_invocation(event: dict[str, Any]) -> dict[str, Any]:
    """Entry point for AgentCore Runtime invocations.

    Routes heartbeat payloads (``type: "heartbeat"``) to the heartbeat
    handler.  All other payloads are treated as interactive sessions.

    Requirements: 7.1, 7.2, 10.1
    """
    agent_id = event.get("agent_id", "")
    if not agent_id:
        return {"status": "error", "error_message": "Missing agent_id in event"}

    _config, orchestrator, heartbeat_handler = _build_components(agent_id)

    # Route heartbeat vs interactive (Req 10.1)
    if event.get("type") == "heartbeat":
        return heartbeat_handler.handle_heartbeat(event)

    # Interactive session
    user_id = event.get("user_id", "anonymous")
    message = event.get("message", "")
    if not message:
        return {"status": "error", "error_message": "Missing message in event"}

    try:
        handle = orchestrator.create_agent(agent_id, user_id)
        response = orchestrator.handle_message(handle, message)
        orchestrator.end_session(handle)
        return {"status": "completed", "output": response}
    except Exception as exc:
        logger.error("Interactive session failed: %s", exc)
        return {"status": "error", "error_message": str(exc)}


def main() -> None:
    """CLI entry point for ``agentcore dev`` local testing.

    Usage::

        python -m agent.main --agent-id my-agent [--user-id user-123]

    Starts an interactive REPL that sends messages to the agent.

    Requirements: 13.4
    """
    parser = argparse.ArgumentParser(
        description="OpenClaw AWS Agent — local dev CLI"
    )
    parser.add_argument(
        "--agent-id",
        required=True,
        help="The agent identifier",
    )
    parser.add_argument(
        "--user-id",
        default="local-user",
        help="The user identifier (default: local-user)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    logger.info("Building components for agent '%s'...", args.agent_id)
    _config, orchestrator, _heartbeat = _build_components(args.agent_id)

    logger.info("Creating agent session...")
    handle = orchestrator.create_agent(args.agent_id, args.user_id)

    print(f"\n🐾 OpenClaw Agent [{args.agent_id}] ready. Type 'quit' to exit.\n")

    try:
        while True:
            try:
                user_input = input("You: ").strip()
            except EOFError:
                break

            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit"):
                break

            response = orchestrator.handle_message(handle, user_input)
            print(f"\nAgent: {response}\n")
    except KeyboardInterrupt:
        print("\n")
    finally:
        orchestrator.end_session(handle)
        logger.info("Session ended.")


if __name__ == "__main__":
    main()
