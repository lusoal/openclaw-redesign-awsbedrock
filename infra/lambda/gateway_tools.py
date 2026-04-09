"""Sample Gateway Lambda target — provides AWS account tools.

These tools demonstrate how AgentCore Gateway exposes Lambda functions
as MCP tools that the agent can discover and call automatically.
"""

import json
import logging
import os

import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

REGION = os.environ.get("AWS_REGION", "us-east-1")


def lambda_handler(event, context):
    """Route tool calls based on the tool name from Gateway."""
    logger.info("Received event: %s", json.dumps(event))

    # Gateway passes the tool name via client context
    tool_name = ""
    if context and hasattr(context, "client_context") and context.client_context:
        tool_name = context.client_context.custom.get("bedrockAgentCoreToolName", "")

    if "list_s3_buckets" in tool_name:
        return _list_s3_buckets()
    elif "describe_account" in tool_name:
        return _describe_account()
    elif "list_lambda_functions" in tool_name:
        return _list_lambda_functions()

    return {"statusCode": 400, "body": json.dumps({"error": f"Unknown tool: {tool_name}"})}


def _list_s3_buckets():
    """List all S3 buckets in the account."""
    s3 = boto3.client("s3", region_name=REGION)
    resp = s3.list_buckets()
    buckets = [b["Name"] for b in resp.get("Buckets", [])]
    return {"statusCode": 200, "body": json.dumps({"buckets": buckets, "count": len(buckets)})}


def _describe_account():
    """Get basic account information."""
    sts = boto3.client("sts", region_name=REGION)
    identity = sts.get_caller_identity()
    return {
        "statusCode": 200,
        "body": json.dumps({
            "account_id": identity["Account"],
            "region": REGION,
            "caller_arn": identity["Arn"],
        }),
    }


def _list_lambda_functions():
    """List Lambda functions in the account."""
    lmb = boto3.client("lambda", region_name=REGION)
    resp = lmb.list_functions(MaxItems=20)
    functions = [
        {"name": f["FunctionName"], "runtime": f.get("Runtime", "N/A")}
        for f in resp.get("Functions", [])
    ]
    return {"statusCode": 200, "body": json.dumps({"functions": functions, "count": len(functions)})}
