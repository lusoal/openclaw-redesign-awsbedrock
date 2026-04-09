#!/usr/bin/env python3
"""ChatUI — calls the deployed AgentCore Runtime and polls SQS for notifications.

Usage::

    python chat_ui.py --agent-id my-agent --user-id user-123
"""

from __future__ import annotations

import argparse
import json
import logging
import threading
import time
import uuid

import boto3
import gradio as gr

logger = logging.getLogger(__name__)


def _extract_text(value) -> str:
    """Recursively extract plain text from any agent response format."""
    if isinstance(value, str):
        # Try parsing as JSON first
        try:
            parsed = json.loads(value)
            return _extract_text(parsed)
        except (json.JSONDecodeError, TypeError):
            pass
        # Try parsing Python repr format (single quotes)
        try:
            import ast
            parsed = ast.literal_eval(value)
            return _extract_text(parsed)
        except (ValueError, SyntaxError):
            pass
        return value
    if isinstance(value, dict):
        for key in ("text", "output", "message"):
            if key in value and isinstance(value[key], str):
                return _extract_text(value[key])
        if "content" in value:
            return _extract_text(value["content"])
        return str(value)
    if isinstance(value, list):
        parts = [_extract_text(item) for item in value]
        return "\n".join(p for p in parts if p)
    return str(value)


class AgentCoreClient:
    def __init__(self, runtime_arn: str, region: str) -> None:
        self._arn = runtime_arn
        self._client = boto3.client("bedrock-agentcore", region_name=region)
        # Stable session ID per UI session (maintains conversation context)
        self._session_id = f"ui-{uuid.uuid4().hex}00000000000"

    def invoke(self, agent_id: str, user_id: str, message: str) -> str:
        payload = json.dumps({
            "agent_id": agent_id,
            "user_id": user_id,
            "message": message,
        })
        try:
            resp = self._client.invoke_agent_runtime(
                agentRuntimeArn=self._arn,
                runtimeSessionId=self._session_id,
                payload=payload,
                qualifier="DEFAULT",
            )
            body = resp["response"].read().decode()
            data = json.loads(body) if body else {}
            if data.get("status") == "error":
                return f"Error: {data.get('error_message', 'unknown')}"
            output = data.get("output", body)
            # The output might be a JSON string itself
            if isinstance(output, str):
                try:
                    output = json.loads(output)
                except (json.JSONDecodeError, TypeError):
                    pass
            return _extract_text(output)
        except Exception as exc:
            return f"Error: {exc}"


class SQSPoller:
    def __init__(self, queue_url: str, region: str) -> None:
        self._queue_url = queue_url
        self._sqs = boto3.client("sqs", region_name=region)
        self._messages: list[str] = []
        self._lock = threading.Lock()
        self._running = False

    def start(self) -> None:
        self._running = True
        threading.Thread(target=self._poll_loop, daemon=True).start()

    def get_messages(self) -> list[str]:
        with self._lock:
            msgs = list(self._messages)
            self._messages.clear()
            return msgs

    def _poll_loop(self) -> None:
        while self._running:
            try:
                resp = self._sqs.receive_message(
                    QueueUrl=self._queue_url,
                    MaxNumberOfMessages=10,
                    WaitTimeSeconds=5,
                )
                for msg in resp.get("Messages", []):
                    body = json.loads(msg["Body"])
                    task = body.get("task", "")
                    result = body.get("result", {})
                    output = result.get("output") or task or "Reminder"
                    with self._lock:
                        self._messages.append(f"🔔 {output}")
                    self._sqs.delete_message(
                        QueueUrl=self._queue_url,
                        ReceiptHandle=msg["ReceiptHandle"],
                    )
            except Exception:
                time.sleep(5)


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenClaw Agent — Remote UI")
    parser.add_argument("--agent-id", required=True)
    parser.add_argument("--user-id", default="local-user")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--region", default="us-east-1")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    ssm = boto3.client("ssm", region_name=args.region)
    params = ssm.get_parameters(Names=[
        f"/{args.agent_id}/config/agent-runtime-arn",
        f"/{args.agent_id}/config/heartbeat-queue-url",
    ])
    config = {p["Name"].rsplit("/", 1)[-1]: p["Value"] for p in params.get("Parameters", [])}

    runtime_arn = config.get("agent-runtime-arn", "")
    queue_url = config.get("heartbeat-queue-url", "")
    if not runtime_arn:
        logger.error("agent-runtime-arn not found in SSM")
        return

    logger.info("Runtime: %s", runtime_arn)
    ac_client = AgentCoreClient(runtime_arn, args.region)

    poller = None
    if queue_url:
        poller = SQSPoller(queue_url, args.region)
        poller.start()
        logger.info("SQS poller started: %s", queue_url)

    def respond(message: str, history):
        return ac_client.invoke(args.agent_id, args.user_id, message)

    def check_notifications():
        if not poller:
            return gr.update(value="No notifications", visible=True)
        msgs = poller.get_messages()
        if msgs:
            return gr.update(value="\n\n".join(msgs), visible=True)
        return gr.update(visible=True)

    with gr.Blocks(title="OpenClaw Agent") as demo:
        gr.Markdown("## 🐾 OpenClaw Agent")
        gr.Markdown("_Connected to AgentCore Runtime_")

        with gr.Row():
            with gr.Column(scale=3):
                gr.ChatInterface(
                    fn=respond,
                    examples=[
                        "What can you do?",
                        "Add a task to review PRs",
                        "Remind me every 2 minutes to stretch",
                        "List my schedules",
                        "List my tasks",
                    ],
                )
            with gr.Column(scale=1):
                gr.Markdown("### 🔔 Notifications")
                notif_box = gr.Markdown(value="No notifications yet")
                timer = gr.Timer(value=5)
                timer.tick(fn=check_notifications, outputs=[notif_box])

    demo.launch(server_port=args.port, share=False)


if __name__ == "__main__":
    main()
