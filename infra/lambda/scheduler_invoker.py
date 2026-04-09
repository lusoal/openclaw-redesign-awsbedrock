"""Lambda: bridges EventBridge Scheduler -> SNS notification.

Receives a heartbeat payload and publishes the task prompt to SNS.
SNS fans out to SQS (for UI), email, SMS, Slack, etc.
For one-time schedules (at() expressions), auto-deletes the schedule.
"""

import json
import logging
import os

import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

TOPIC_ARN = os.environ.get("NOTIFICATION_TOPIC_ARN", "")
QUEUE_URL = os.environ.get("HEARTBEAT_QUEUE_URL", "")
BUCKET = os.environ.get("IDENTITY_BUCKET", "")
SCHEDULE_GROUP = os.environ.get("SCHEDULE_GROUP_NAME", "agent-schedules")
REGION = os.environ.get("AWS_REGION", "us-east-1")

sns = boto3.client("sns", region_name=REGION)
sqs = boto3.client("sqs", region_name=REGION)
scheduler = boto3.client("scheduler", region_name=REGION)
s3 = boto3.client("s3", region_name=REGION)


def handler(event, context):
    """Publish notification to SNS, auto-cleanup one-time schedules."""
    logger.info("Received event: %s", json.dumps(event))

    agent_id = event.get("agent_id", "unknown")
    user_id = event.get("user_id", "unknown")
    task = event.get("task", "")

    message = {
        "type": "heartbeat_result",
        "agent_id": agent_id,
        "user_id": user_id,
        "task": task,
        "result": {"status": "completed", "output": task or "Scheduled reminder"},
    }

    # Publish to SNS (fans out to all subscribers)
    if TOPIC_ARN:
        try:
            sns.publish(
                TopicArn=TOPIC_ARN,
                Message=json.dumps(message),
                Subject=f"Agent Reminder: {task[:80]}" if task else "Agent Reminder",
            )
            logger.info("Published to SNS topic")
        except Exception as exc:
            logger.error("Failed to publish to SNS: %s", exc)
            # Fallback to direct SQS if SNS fails
            if QUEUE_URL:
                try:
                    sqs.send_message(QueueUrl=QUEUE_URL, MessageBody=json.dumps(message))
                    logger.info("Fallback: sent directly to SQS")
                except Exception as sqs_exc:
                    logger.error("Fallback SQS also failed: %s", sqs_exc)
    elif QUEUE_URL:
        # No SNS configured, send directly to SQS
        try:
            sqs.send_message(QueueUrl=QUEUE_URL, MessageBody=json.dumps(message))
        except Exception as exc:
            logger.error("Failed to send to SQS: %s", exc)

    # Auto-cleanup one-time schedules
    _auto_cleanup_one_time(agent_id, task)

    return {"statusCode": 200, "body": "ok"}


def _auto_cleanup_one_time(agent_id, task):
    """Delete one-time schedules after they fire."""
    if not BUCKET:
        return

    key = f"agents/{agent_id}/schedules.json"
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=key)
        data = json.loads(obj["Body"].read())
        schedules = data.get("schedules", [])
    except Exception:
        return

    remaining = []
    deleted_name = None
    for s in schedules:
        payload = s.get("payload", {})
        if (
            payload.get("task") == task
            and s.get("schedule_type") == "one-time"
            and s.get("status") == "active"
        ):
            deleted_name = s.get("name")
            logger.info("Auto-deleting one-time schedule: %s", deleted_name)
        else:
            remaining.append(s)

    if deleted_name is None:
        return

    schedule_name = f"{agent_id}-{deleted_name}"
    try:
        scheduler.delete_schedule(Name=schedule_name, GroupName=SCHEDULE_GROUP)
        logger.info("Deleted EventBridge schedule: %s", schedule_name)
    except Exception as exc:
        logger.warning("Failed to delete EventBridge schedule: %s", exc)

    try:
        s3.put_object(
            Bucket=BUCKET,
            Key=key,
            Body=json.dumps({"schedules": remaining}, indent=2),
            ContentType="application/json",
        )
    except Exception as exc:
        logger.warning("Failed to update schedules.json: %s", exc)
