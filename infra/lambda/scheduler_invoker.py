"""Lambda: bridges EventBridge Scheduler -> SQS notification.

Receives a heartbeat payload and sends the task prompt directly
to SQS for the UI to display as a notification.
"""

import json
import logging
import os

import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

QUEUE_URL = os.environ["HEARTBEAT_QUEUE_URL"]
REGION = os.environ.get("AWS_REGION", "us-east-1")

sqs = boto3.client("sqs", region_name=REGION)


def handler(event, context):
    """Forward the heartbeat task prompt to SQS."""
    logger.info("Received event: %s", json.dumps(event))

    message = {
        "type": "heartbeat_result",
        "agent_id": event.get("agent_id", "unknown"),
        "user_id": event.get("user_id", "unknown"),
        "task": event.get("task", ""),
        "result": {"status": "completed", "output": event.get("task", "Scheduled reminder")},
    }

    try:
        sqs.send_message(QueueUrl=QUEUE_URL, MessageBody=json.dumps(message))
        logger.info("Sent notification to SQS")
        return {"statusCode": 200, "body": "ok"}
    except Exception as exc:
        logger.error("Failed to send to SQS: %s", exc)
        return {"statusCode": 500, "body": str(exc)}
