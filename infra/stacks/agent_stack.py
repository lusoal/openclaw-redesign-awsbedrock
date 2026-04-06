"""CDK InfrastructureStack for the OpenClaw AWS Agent.

Provisions S3 bucket, IAM roles, EventBridge Schedule Group, SSM Parameters,
AgentCore Memory, and AgentCore Runtime required by the agent.

Requirements: 15.1, 15.2, 15.3, 15.4, 15.5, 15.6, 15.7, 15.8, 15.9
"""

from __future__ import annotations

import os

from aws_cdk import (
    CfnOutput,
    Duration,
    RemovalPolicy,
    Stack,
    aws_iam as iam,
    aws_lambda as _lambda,
    aws_s3 as s3,
    aws_scheduler as scheduler,
    aws_sqs as sqs,
    aws_ssm as ssm,
)
from aws_cdk.aws_bedrock_agentcore_alpha import (
    AgentCoreRuntime,
    AgentRuntimeArtifact,
    Memory,
    MemoryStrategy,
    Runtime,
)
from aws_cdk.aws_ecr_assets import Platform
import aws_cdk.aws_bedrock_alpha as bedrock
from constructs import Construct

from agent.models import InfraStackProps, validate_agent_id


class InfrastructureStack(Stack):
    """Provisions all AWS resources required by the OpenClaw agent."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        props: InfraStackProps,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # Validate agent_id (Req 15.8) — InfraStackProps already validates
        # via Pydantic, but belt-and-suspenders for direct Stack usage.
        validate_agent_id(props.agent_id)

        # --- S3 Bucket (Req 15.1) ---
        identity_bucket = s3.Bucket(
            self,
            "IdentityBucket",
            bucket_name=f"{props.agent_id}-identity-{self.account}",
            encryption=s3.BucketEncryption.S3_MANAGED,
            versioned=True,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            removal_policy=RemovalPolicy.RETAIN,
            enforce_ssl=True,
        )

        # --- EventBridge Schedule Group (Req 15.4) ---
        schedule_group = scheduler.CfnScheduleGroup(
            self,
            "ScheduleGroup",
            name=props.schedule_group_name,
        )

        # --- IAM Role: EventBridge Scheduler execution role (Req 15.3) ---
        scheduler_role = iam.Role(
            self,
            "SchedulerExecutionRole",
            assumed_by=iam.ServicePrincipal("scheduler.amazonaws.com"),
            description=(
                "Role assumed by EventBridge Scheduler to invoke "
                "AgentCore Runtime"
            ),
        )

        # --- SQS Queue for heartbeat results ---
        heartbeat_queue = sqs.Queue(
            self,
            "HeartbeatQueue",
            queue_name=f"{props.agent_id}-heartbeat-results",
            retention_period=Duration.days(1),
            visibility_timeout=Duration.seconds(30),
        )

        # --- AgentCore Memory ---
        memory = Memory(
            self,
            "AgentMemory",
            memory_name=f"{props.agent_id.replace('-', '_')}_memory",
            description=f"Memory for agent {props.agent_id}",
            expiration_duration=Duration.days(90),
            memory_strategies=[
                MemoryStrategy.using_built_in_summarization(),
                MemoryStrategy.using_built_in_user_preference(),
                MemoryStrategy.using_built_in_semantic(),
            ],
        )

        # Resolve memory_id: use override from props if provided, else the
        # construct-generated ID.
        resolved_memory_id = props.memory_id if props.memory_id else memory.memory_id

        # --- AgentCore Runtime ---
        project_root = os.path.join(os.path.dirname(__file__), "..", "..")
        agent_runtime_artifact = AgentRuntimeArtifact.from_asset(
            directory=project_root,
            platform=Platform.LINUX_ARM64,
            exclude=[
                "cdk.out",
                "infra/cdk.out",
                "infra",
                ".git",
                ".kiro",
                ".hypothesis",
                "__pycache__",
                "*.pyc",
                "node_modules",
                "venv",
                "*.egg-info",
                ".pytest_cache",
                "tests",
            ],
        )

        runtime = Runtime(
            self,
            "AgentRuntime",
            runtime_name=f"{props.agent_id.replace('-', '_')}_runtime",
            agent_runtime_artifact=agent_runtime_artifact,
            environment_variables={
                "IDENTITY_BUCKET": identity_bucket.bucket_name,
                "MEMORY_ID": resolved_memory_id,
                "BEDROCK_MODEL_ID": props.bedrock_model_id,
                "BEDROCK_REGION": props.bedrock_region,
                "SCHEDULE_GROUP_NAME": props.schedule_group_name,
            },
        )

        # --- Grant Bedrock model invoke to the runtime ---
        model = bedrock.BedrockFoundationModel.ANTHROPIC_CLAUDE_SONNET_4_V1_0
        model.grant_invoke(runtime)

        # Grant access to the global inference profile as well
        runtime.add_to_role_policy(
            iam.PolicyStatement(
                actions=[
                    "bedrock:InvokeModel",
                    "bedrock:InvokeModelWithResponseStream",
                ],
                resources=[
                    f"arn:aws:bedrock:us-east-1:{self.account}:inference-profile/global.anthropic.claude-sonnet-4-20250514-v1:0",
                    "arn:aws:bedrock:*::foundation-model/anthropic.claude-sonnet-4-20250514-v1:0",
                ],
            )
        )

        # --- Lambda: Scheduler -> AgentCore Runtime bridge ---
        scheduler_invoker = _lambda.Function(
            self,
            "SchedulerInvoker",
            function_name=f"{props.agent_id}-scheduler-invoker",
            runtime=_lambda.Runtime.PYTHON_3_12,
            handler="scheduler_invoker.handler",
            code=_lambda.Code.from_asset(
                os.path.join(os.path.dirname(__file__), "..", "lambda")
            ),
            timeout=Duration.minutes(5),
            environment={
                "AGENT_RUNTIME_ARN": runtime.agent_runtime_arn,
                "HEARTBEAT_QUEUE_URL": heartbeat_queue.queue_url,
            },
        )

        # Grant the Lambda permission to invoke AgentCore Runtime
        scheduler_invoker.add_to_role_policy(
            iam.PolicyStatement(
                actions=[
                    "bedrock-agentcore:InvokeAgentRuntime",
                    "bedrock-agentcore:CreateRuntimeSession",
                    "bedrock-agentcore:GetAgentRuntimeEndpoint",
                ],
                resources=[
                    runtime.agent_runtime_arn,
                    f"{runtime.agent_runtime_arn}/*",
                ],
            )
        )

        # Grant Lambda permission to send messages to the heartbeat queue
        heartbeat_queue.grant_send_messages(scheduler_invoker)

        # Grant EventBridge Scheduler permission to invoke the Lambda
        scheduler_invoker.grant_invoke(scheduler_role)

        # --- Grant S3 access to the runtime ---
        identity_bucket.grant_read_write(runtime)

        # --- EventBridge Scheduler permissions for the runtime ---
        runtime.add_to_role_policy(
            iam.PolicyStatement(
                actions=[
                    "scheduler:CreateSchedule",
                    "scheduler:DeleteSchedule",
                    "scheduler:GetSchedule",
                    "scheduler:ListSchedules",
                ],
                resources=[
                    (
                        f"arn:aws:scheduler:{self.region}:{self.account}"
                        f":schedule/{props.schedule_group_name}/*"
                    )
                ],
            )
        )

        # PassRole for scheduler execution role
        runtime.add_to_role_policy(
            iam.PolicyStatement(
                actions=["iam:PassRole"],
                resources=[scheduler_role.role_arn],
                conditions={
                    "StringEquals": {
                        "iam:PassedToService": "scheduler.amazonaws.com",
                    }
                },
            )
        )

        # SSM read permissions — scoped to agent config parameters
        runtime.add_to_role_policy(
            iam.PolicyStatement(
                actions=["ssm:GetParameter", "ssm:GetParameters"],
                resources=[
                    (
                        f"arn:aws:ssm:{self.region}:{self.account}"
                        f":parameter/{props.agent_id}/config/*"
                    )
                ],
            )
        )

        # --- SSM Parameters (Req 15.5) ---
        ssm.StringParameter(
            self,
            "ParamBucketName",
            parameter_name=f"/{props.agent_id}/config/bucket-name",
            string_value=identity_bucket.bucket_name,
        )
        ssm.StringParameter(
            self,
            "ParamMemoryId",
            parameter_name=f"/{props.agent_id}/config/memory-id",
            string_value=resolved_memory_id,
        )
        ssm.StringParameter(
            self,
            "ParamBedrockModelId",
            parameter_name=f"/{props.agent_id}/config/bedrock-model-id",
            string_value=props.bedrock_model_id,
        )
        ssm.StringParameter(
            self,
            "ParamScheduleGroupName",
            parameter_name=f"/{props.agent_id}/config/schedule-group-name",
            string_value=props.schedule_group_name,
        )
        ssm.StringParameter(
            self,
            "ParamSchedulerRoleArn",
            parameter_name=f"/{props.agent_id}/config/scheduler-role-arn",
            string_value=scheduler_role.role_arn,
        )

        if props.enable_prompt_routing and props.prompt_router_arn:
            ssm.StringParameter(
                self,
                "ParamPromptRouterArn",
                parameter_name=f"/{props.agent_id}/config/prompt-router-arn",
                string_value=props.prompt_router_arn,
            )

        ssm.StringParameter(
            self,
            "ParamAgentRuntimeArn",
            parameter_name=f"/{props.agent_id}/config/agent-runtime-arn",
            string_value=runtime.agent_runtime_arn,
        )

        ssm.StringParameter(
            self,
            "ParamSchedulerTargetArn",
            parameter_name=f"/{props.agent_id}/config/scheduler-target-arn",
            string_value=scheduler_invoker.function_arn,
        )

        ssm.StringParameter(
            self,
            "ParamHeartbeatQueueUrl",
            parameter_name=f"/{props.agent_id}/config/heartbeat-queue-url",
            string_value=heartbeat_queue.queue_url,
        )

        # --- Stack Outputs (Req 15.6) ---
        CfnOutput(
            self, "IdentityBucketName", value=identity_bucket.bucket_name
        )
        CfnOutput(
            self, "IdentityBucketArn", value=identity_bucket.bucket_arn
        )
        CfnOutput(
            self,
            "SchedulerExecutionRoleArn",
            value=scheduler_role.role_arn,
        )
        CfnOutput(
            self, "ScheduleGroupName", value=schedule_group.name
        )
        CfnOutput(
            self, "RuntimeId", value=runtime.agent_runtime_id
        )
        CfnOutput(
            self, "RuntimeArn", value=runtime.agent_runtime_arn
        )
        CfnOutput(
            self, "MemoryId", value=memory.memory_id
        )
        CfnOutput(
            self, "MemoryArn", value=memory.memory_arn
        )
        CfnOutput(
            self, "SchedulerInvokerArn", value=scheduler_invoker.function_arn
        )
        CfnOutput(
            self, "HeartbeatQueueUrl", value=heartbeat_queue.queue_url
        )
