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
    aws_sns as sns,
    aws_sns_subscriptions as subs,
    aws_sqs as sqs,
    aws_ssm as ssm,
)
from aws_cdk.aws_bedrock_agentcore_alpha import (
    AgentCoreRuntime,
    AgentRuntimeArtifact,
    Gateway,
    GatewayAuthorizer,
    GatewayTarget,
    IamAuthorizer,
    InlineToolSchema,
    LambdaTargetConfiguration,
    McpProtocolConfiguration,
    MCPProtocolVersion,
    Memory,
    MemoryStrategy,
    Runtime,
    SchemaDefinition,
    SchemaDefinitionType,
    ToolDefinition,
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

        # --- SQS Queue for heartbeat results (UI polls this) ---
        heartbeat_queue = sqs.Queue(
            self,
            "HeartbeatQueue",
            queue_name=f"{props.agent_id}-heartbeat-results",
            retention_period=Duration.days(1),
            visibility_timeout=Duration.seconds(30),
        )

        # --- SNS Topic for notifications (fan-out to multiple channels) ---
        notification_topic = sns.Topic(
            self,
            "NotificationTopic",
            topic_name=f"{props.agent_id}-notifications",
            display_name=f"OpenClaw Agent Notifications ({props.agent_id})",
        )

        # Subscribe SQS to SNS so the Gradio UI keeps working
        notification_topic.add_subscription(
            subs.SqsSubscription(heartbeat_queue, raw_message_delivery=True)
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

        # --- Intelligent Prompt Router (auto model selection) ---
        # Using Anthropic Claude router. The Meta Llama router does not
        # support tool use in streaming mode and is incompatible.
        prompt_router = bedrock.PromptRouter(
            bedrock.PromptRouterProps(
                prompt_router_id=bedrock.DefaultPromptRouterIdentifier.ANTHROPIC_CLAUDE_V1.prompt_router_id,
                routing_models=bedrock.DefaultPromptRouterIdentifier.ANTHROPIC_CLAUDE_V1.routing_models,
            ),
            props.bedrock_region,
        )

        # Grant the runtime permission to use the prompt router
        runtime.add_to_role_policy(
            iam.PolicyStatement(
                actions=[
                    "bedrock:InvokeModel",
                    "bedrock:InvokeModelWithResponseStream",
                    "bedrock:GetFoundationModel",
                ],
                resources=[
                    prompt_router.prompt_router_arn,
                    f"arn:aws:bedrock:{props.bedrock_region}:{self.account}:default-prompt-router/*",
                    f"arn:aws:bedrock:*::foundation-model/*",
                    f"arn:aws:bedrock:{props.bedrock_region}:{self.account}:inference-profile/*",
                ],
            )
        )

        # Use prompt router ARN if routing is enabled, otherwise use direct model
        effective_prompt_router_arn = (
            prompt_router.prompt_router_arn if props.enable_prompt_routing else None
        )

        # --- Lambda: Scheduler -> SQS bridge with auto-cleanup ---
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
                "NOTIFICATION_TOPIC_ARN": notification_topic.topic_arn,
                "IDENTITY_BUCKET": identity_bucket.bucket_name,
                "SCHEDULE_GROUP_NAME": props.schedule_group_name,
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

        # Grant Lambda permission to publish to the notification topic
        notification_topic.grant_publish(scheduler_invoker)

        # Grant Lambda S3 access for auto-cleanup of one-time schedules
        identity_bucket.grant_read_write(scheduler_invoker)

        # Grant Lambda permission to delete EventBridge schedules
        scheduler_invoker.add_to_role_policy(
            iam.PolicyStatement(
                actions=["scheduler:DeleteSchedule"],
                resources=[
                    (
                        f"arn:aws:scheduler:{self.region}:{self.account}"
                        f":schedule/{props.schedule_group_name}/*"
                    )
                ],
            )
        )

        # Grant EventBridge Scheduler permission to invoke the Lambda
        scheduler_invoker.grant_invoke(scheduler_role)

        # --- Lambda: Agent Invoker (subscribes to SNS for agent actions) ---
        agent_invoker = _lambda.Function(
            self,
            "AgentInvoker",
            function_name=f"{props.agent_id}-agent-invoker",
            runtime=_lambda.Runtime.PYTHON_3_12,
            handler="agent_invoker.handler",
            code=_lambda.Code.from_asset(
                os.path.join(os.path.dirname(__file__), "..", "lambda")
            ),
            timeout=Duration.minutes(10),
            environment={
                "AGENT_RUNTIME_ARN": runtime.agent_runtime_arn,
                "NOTIFICATION_TOPIC_ARN": notification_topic.topic_arn,
            },
        )

        # Grant agent-invoker permission to invoke AgentCore Runtime
        agent_invoker.add_to_role_policy(
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

        # Grant agent-invoker permission to publish results back to SNS
        notification_topic.grant_publish(agent_invoker)

        # Subscribe agent-invoker to SNS with filter: only agent_action=true
        notification_topic.add_subscription(
            subs.LambdaSubscription(
                agent_invoker,
                filter_policy={
                    "agent_action": sns.SubscriptionFilter.string_filter(
                        allowlist=["true"],
                    ),
                },
            )
        )

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

        # AgentCore Memory permissions
        runtime.add_to_role_policy(
            iam.PolicyStatement(
                actions=["bedrock-agentcore:*"],
                resources=[
                    memory.memory_arn,
                    f"{memory.memory_arn}/*",
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

        if props.enable_prompt_routing and effective_prompt_router_arn:
            ssm.StringParameter(
                self,
                "ParamPromptRouterArn",
                parameter_name=f"/{props.agent_id}/config/prompt-router-arn",
                string_value=effective_prompt_router_arn,
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

        ssm.StringParameter(
            self,
            "ParamNotificationTopicArn",
            parameter_name=f"/{props.agent_id}/config/notification-topic-arn",
            string_value=notification_topic.topic_arn,
        )

        # --- AgentCore Gateway (MCP) ---
        gateway = Gateway(
            self,
            "AgentGateway",
            gateway_name=f"{props.agent_id}-gateway",
            description=f"MCP Gateway for agent {props.agent_id}",
            authorizer_configuration=IamAuthorizer(),
            protocol_configuration=McpProtocolConfiguration(
                supported_versions=[MCPProtocolVersion.MCP_2025_06_18],
            ),
        )

        # Grant the Gateway role broad AWS access for dynamically added services.
        # Change this to a more restrictive policy for production use.
        gateway.role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name("ReadOnlyAccess")
        )

        # Lambda for Gateway tools
        gateway_tools_fn = _lambda.Function(
            self,
            "GatewayToolsFunction",
            function_name=f"{props.agent_id}-gateway-tools",
            runtime=_lambda.Runtime.PYTHON_3_12,
            handler="gateway_tools.lambda_handler",
            code=_lambda.Code.from_asset(
                os.path.join(os.path.dirname(__file__), "..", "lambda")
            ),
            timeout=Duration.seconds(30),
        )

        # Grant the Lambda read-only AWS access for the tools
        gateway_tools_fn.add_to_role_policy(
            iam.PolicyStatement(
                actions=[
                    "s3:ListAllMyBuckets",
                    "sts:GetCallerIdentity",
                    "lambda:ListFunctions",
                ],
                resources=["*"],
            )
        )

        # Define tool schemas
        empty_object_schema = SchemaDefinition(
            type=SchemaDefinitionType.OBJECT,
        )

        tool_definitions = [
            ToolDefinition(
                name="list_s3_buckets",
                description="List all S3 buckets in the AWS account",
                input_schema=empty_object_schema,
            ),
            ToolDefinition(
                name="describe_account",
                description="Get AWS account ID, region, and caller identity",
                input_schema=empty_object_schema,
            ),
            ToolDefinition(
                name="list_lambda_functions",
                description="List Lambda functions in the AWS account (up to 20)",
                input_schema=empty_object_schema,
            ),
        ]

        # Add Lambda target to Gateway
        gateway_target = GatewayTarget(
            self,
            "GatewayToolsTarget",
            gateway=gateway,
            gateway_target_name=f"{props.agent_id}-aws-tools",
            description="AWS account tools (S3, Lambda, STS)",
            target_configuration=LambdaTargetConfiguration(
                lambda_function=gateway_tools_fn,
                tool_schema=InlineToolSchema(schema=tool_definitions),
            ),
        )

        # --- Lambda execution role for custom tools created by the agent ---
        custom_tool_role = iam.Role(
            self,
            "CustomToolLambdaRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            description="Execution role for agent-created custom Lambda tools",
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AWSLambdaBasicExecutionRole"
                ),
                # Change to "ReadOnlyAccess" to restrict write operations
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "AdministratorAccess"
                ),
            ],
        )

        # Grant the runtime permission to pass this role to Lambda
        runtime.add_to_role_policy(
            iam.PolicyStatement(
                actions=["iam:PassRole"],
                resources=[custom_tool_role.role_arn],
                conditions={
                    "StringEquals": {
                        "iam:PassedToService": "lambda.amazonaws.com",
                    }
                },
            )
        )

        # Grant the Gateway role permission to invoke custom Lambdas
        gateway.role.add_to_policy(
            iam.PolicyStatement(
                actions=["lambda:InvokeFunction"],
                resources=[
                    f"arn:aws:lambda:{self.region}:{self.account}:function:{props.agent_id}-custom-*"
                ],
            )
        )

        ssm.StringParameter(
            self,
            "ParamCustomToolRoleArn",
            parameter_name=f"/{props.agent_id}/config/custom-tool-role-arn",
            string_value=custom_tool_role.role_arn,
        )

        # Store Gateway URL in SSM for the agent to discover
        ssm.StringParameter(
            self,
            "ParamGatewayUrl",
            parameter_name=f"/{props.agent_id}/config/gateway-url",
            string_value=gateway.gateway_url,
        )

        ssm.StringParameter(
            self,
            "ParamGatewayId",
            parameter_name=f"/{props.agent_id}/config/gateway-id",
            string_value=gateway.gateway_id,
        )

        # Grant the runtime permission to invoke the Gateway
        runtime.add_to_role_policy(
            iam.PolicyStatement(
                actions=["bedrock-agentcore:*"],
                resources=[
                    gateway.gateway_arn,
                    f"{gateway.gateway_arn}/*",
                ],
            )
        )

        # Grant the runtime permission to manage Gateway targets
        runtime.add_to_role_policy(
            iam.PolicyStatement(
                actions=[
                    "bedrock-agentcore:CreateGatewayTarget",
                    "bedrock-agentcore:DeleteGatewayTarget",
                    "bedrock-agentcore:ListGatewayTargets",
                    "bedrock-agentcore:GetGatewayTarget",
                ],
                resources=[
                    gateway.gateway_arn,
                    f"{gateway.gateway_arn}/*",
                ],
            )
        )

        # Grant the runtime permission to create Lambda functions for custom tools
        runtime.add_to_role_policy(
            iam.PolicyStatement(
                actions=[
                    "lambda:CreateFunction",
                    "lambda:UpdateFunctionCode",
                    "lambda:GetFunction",
                    "lambda:DeleteFunction",
                    "lambda:AddPermission",
                    "lambda:RemovePermission",
                ],
                resources=[
                    f"arn:aws:lambda:{self.region}:{self.account}:function:{props.agent_id}-custom-*"
                ],
            )
        )

        # Grant PassRole for Lambda execution role
        runtime.add_to_role_policy(
            iam.PolicyStatement(
                actions=["iam:PassRole"],
                resources=[gateway.role.role_arn],
                conditions={
                    "StringEquals": {
                        "iam:PassedToService": "lambda.amazonaws.com",
                    }
                },
            )
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
        CfnOutput(
            self, "NotificationTopicArn", value=notification_topic.topic_arn
        )
        CfnOutput(
            self, "GatewayUrl", value=gateway.gateway_url
        )
        CfnOutput(
            self, "GatewayArn", value=gateway.gateway_arn
        )
