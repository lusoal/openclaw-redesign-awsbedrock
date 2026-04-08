# OpenClaw AWS Agent

An AI agent inspired by [OpenClaw](https://github.com/kustomzone/OpenClaw)'s identity-driven, memory-rich architecture — rebuilt entirely on AWS managed services. No self-hosted databases, no custom embedding pipelines, no local file systems. Just S3, Bedrock, AgentCore, and EventBridge.

The agent loads its personality from S3-stored identity files, maintains context via AgentCore Memory, manages tasks and schedules through natural conversation, and can proactively remind you via EventBridge Scheduler.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Client Layer                            │
│   Gradio Chat UI ──► AgentCore Runtime (deployed)           │
│                      ◄── SQS (scheduled notifications)      │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│              AgentCore Runtime (Container)                   │
│   BedrockAgentCoreApp  ·  AgentOrchestrator                 │
│   Strands Agent  ·  System Prompt  ·  Tool Registry         │
└──────┬──────────────┬───────────────┬───────────────────────┘
       │              │               │
       ▼              ▼               ▼
┌────────────┐ ┌─────────────┐ ┌──────────────────────────────┐
│ S3 Bucket  │ │  AgentCore  │ │  EventBridge Scheduler       │
│ Identity   │ │   Memory    │ │       │                      │
│ Tasks JSON │ │  STM + LTM  │ │       ▼                      │
│ Schedules  │ │             │ │  Lambda ──► SQS ──► UI       │
└────────────┘ └─────────────┘ └──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│              Amazon Bedrock (LLM Inference)                 │
│   Claude Sonnet 4  ·  Intelligent Prompt Routing (optional) │
└─────────────────────────────────────────────────────────────┘
```

## Prerequisites

- Python 3.11+
- AWS account with Bedrock model access (Claude Sonnet 4)
- AWS CDK CLI (`npm install -g aws-cdk`)
- Docker (for container image builds)
- AWS credentials configured

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/lusoal/openclaw-redesign-awsbedrock.git
cd openclaw-redesign-awsbedrock
python -m venv venv && source venv/bin/activate
pip install -e ".[dev,infra]"

# 2. Deploy everything
cd infra && pip install -r requirements.txt
cdk bootstrap
cdk deploy

# 3. Upload identity files
BUCKET=$(aws cloudformation describe-stacks \
  --stack-name my-agent-infra \
  --query 'Stacks[0].Outputs[?OutputKey==`IdentityBucketName`].OutputValue' \
  --output text)
aws s3 cp identity_files/SOUL.md s3://$BUCKET/agents/my-agent/SOUL.md
aws s3 cp identity_files/AGENTS.md s3://$BUCKET/agents/my-agent/AGENTS.md

# 4. Launch the UI (connects to deployed AgentCore Runtime)
cd ..
python chat_ui.py --agent-id my-agent
```

## What Gets Deployed

A single `cdk deploy` provisions everything:

| Resource | Purpose |
|---|---|
| S3 Bucket | Identity files, tasks, schedules (encrypted, versioned, SSL-only) |
| AgentCore Memory | Summarization, user preference, and semantic memory strategies |
| AgentCore Runtime | Containerized agent with BedrockAgentCoreApp wrapper |
| Lambda (scheduler-invoker) | Bridges EventBridge → SQS for schedule notifications |
| SQS Queue | Delivers scheduled reminder results to the UI |
| EventBridge Schedule Group | Isolates agent schedules |
| SSM Parameters | Runtime config at `/{agent_id}/config/*` |
| IAM Roles | Least-privilege for runtime, scheduler, and Lambda |

## Identity Files

The agent's personality lives in S3 at `agents/{agent_id}/`:

| File | Required | Purpose |
|---|---|---|
| `SOUL.md` | Yes | Core personality and behavioral directives |
| `AGENTS.md` | Yes | Tool documentation and usage instructions |
| `IDENTITY.md` | No | Self-conception (agent can update this) |
| `USER.md` | No | User profile (agent learns about you) |
| `MEMORY.md` | No | Curated long-term knowledge |

## Tools

The agent has 7 tools registered via Strands `@tool` decorator:

| Tool | What it does |
|---|---|
| `manage_tasks` | CRUD for personal tasks in S3 (add/list/complete/delete) |
| `schedule_task` | Create/list/delete EventBridge cron schedules |
| `update_identity` | Update the agent's IDENTITY.md |
| `update_user_profile` | Update USER.md with info about you |
| `save_to_memory` | Persist important info to MEMORY.md |
| `search_memory` | Search across AgentCore Memory |
| `get_current_date` | Current UTC timestamp |

## Configuration

Config is read from SSM Parameters (deployed) with environment variable fallback (local dev):

| SSM Parameter | Env Var | Purpose |
|---|---|---|
| `/{id}/config/bucket-name` | `IDENTITY_BUCKET` | S3 bucket |
| `/{id}/config/bedrock-model-id` | `BEDROCK_MODEL_ID` | Model ID |
| `/{id}/config/memory-id` | `MEMORY_ID` | AgentCore Memory ID |
| `/{id}/config/schedule-group-name` | `SCHEDULE_GROUP_NAME` | EventBridge group |
| `/{id}/config/scheduler-role-arn` | `SCHEDULER_ROLE_ARN` | Scheduler IAM role |
| `/{id}/config/scheduler-target-arn` | `SCHEDULER_TARGET_ARN` | Lambda target ARN |
| `/{id}/config/agent-runtime-arn` | `AGENT_RUNTIME_ARN` | Runtime ARN |

## Testing

```bash
pytest tests/ -v              # All tests
pytest tests/ -k "test_prop"  # Property-based tests only
```

## Project Structure

```
├── main.py                    # AgentCore Runtime entrypoint (BedrockAgentCoreApp)
├── Dockerfile                 # Container image for AgentCore deployment
├── requirements.txt           # Runtime dependencies
├── chat_ui.py                 # Gradio UI (calls deployed AgentCore Runtime)
├── agent/                     # Agent source code
│   ├── main.py                # _build_components + handle_invocation + CLI
│   ├── orchestrator.py        # Ties identity, memory, tools, and LLM together
│   ├── identity.py            # S3 identity file management
│   ├── memory.py              # AgentCore Memory wrapper with degraded mode
│   ├── model_router.py        # Bedrock model selection + prompt routing
│   ├── config.py              # SSM / env var configuration
│   ├── heartbeat.py           # EventBridge heartbeat handler
│   ├── models.py              # Pydantic data models
│   └── tools/                 # Strands @tool implementations
├── identity_files/            # Sample identity files for upload to S3
├── infra/                     # CDK infrastructure
│   ├── stacks/agent_stack.py  # All AWS resources in one stack
│   └── lambda/                # Lambda function for scheduler bridge
└── tests/                     # Unit + property-based tests
```
