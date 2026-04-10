"""Lambda: subscribes to SNS, invokes AgentCore Runtime for agent actions.

Only triggers on messages with agent_action=true attribute.
Invokes the agent with the task prompt and publishes the result back to SNS.
"""

import json
import logging
import os

import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

RUNTIME_ARN = os.environ["AGENT_RUNTIME_ARN"]
TOPIC_ARN = os.environ["NOTIFICATION_TOPIC_ARN"]
REGION = os.environ.get("AWS_REGION", "us-east-1")

agentcore = boto3.client("bedrock-agentcore", region_name=REGION)
sns = boto3.client("sns", region_name=REGION)


def handler(event, context):
    """Invoke AgentCore Runtime and publish the result."""
    logger.info("Received SNS event")

    for record in event.get("Records", []):
        try:
            body = json.loads(record["Sns"]["Message"])
        except (json.JSONDecodeError, KeyError):
            logger.warning("Invalid SNS message format")
            continue

        agent_id = body.get("agent_id", "unknown")
        user_id = body.get("user_id", "unknown")
        task = body.get("task", "")

        if not task:
            continue

        logger.info("Invoking agent for task: %s", task[:100])

        # Invoke AgentCore Runtime
        try:
            payload = json.dumps({
                "agent_id": agent_id,
                "user_id": user_id,
                "message": task,
            })
            resp = agentcore.invoke_agent_runtime(
                agentRuntimeArn=RUNTIME_ARN,
                runtimeSessionId=f"heartbeat-{agent_id}-{context.aws_request_id}"[:64] + "0" * max(0, 33 - len(f"heartbeat-{agent_id}-{context.aws_request_id}"[:64])),
                payload=payload,
                qualifier="DEFAULT",
            )
            response_body = resp["response"].read().decode()
            result = json.loads(response_body) if response_body else {}
            output = result.get("output", str(result))
        except Exception as exc:
            logger.error("Failed to invoke agent: %s", exc)
            output = f"Agent invocation failed: {exc}"

        # Publish agent's response back to SNS (without agent_action to avoid loop)
        try:
            sns.publish(
                TopicArn=TOPIC_ARN,
                Message=json.dumps({
                    "type": "agent_action_result",
                    "agent_id": agent_id,
                    "user_id": user_id,
                    "task": task,
                    "result": {"status": "completed", "output": output},
                }),
                Subject=f"Agent completed: {task[:80]}",
                MessageAttributes={
                    "agent_action": {
                        "DataType": "String",
                        "StringValue": "false",
                    }
                },
            )
            logger.info("Published agent result to SNS")
        except Exception as exc:
            logger.error("Failed to publish result: %s", exc)

    return {"statusCode": 200}
