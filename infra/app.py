"""CDK app entry point for the OpenClaw AWS Agent infrastructure stack."""

import sys
import os

# Add the project root to the path so we can import agent.models
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import aws_cdk as cdk
from agent.models import InfraStackProps
from stacks.agent_stack import InfrastructureStack

app = cdk.App()

# Read configuration from CDK context values
props = InfraStackProps(
    agent_id=app.node.try_get_context("agent_id") or "my-agent",
    bedrock_model_id=app.node.try_get_context("bedrock_model_id")
    or "global.anthropic.claude-sonnet-4-20250514-v1:0",
    bedrock_region=app.node.try_get_context("bedrock_region") or "us-east-1",
    enable_prompt_routing=app.node.try_get_context("enable_prompt_routing") if app.node.try_get_context("enable_prompt_routing") is not None else True,
    prompt_router_arn=app.node.try_get_context("prompt_router_arn"),
    memory_id=app.node.try_get_context("memory_id"),
    schedule_group_name=app.node.try_get_context("schedule_group_name")
    or "agent-schedules",
)

InfrastructureStack(
    app,
    f"{props.agent_id}-infra",
    props=props,
    env=cdk.Environment(region=props.bedrock_region),
)

app.synth()
