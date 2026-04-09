"""manage_gateway_tools — dynamically add/remove Gateway targets.

Allows the agent to extend its own capabilities by adding AWS service
access (Smithy models) or custom Lambda tools to the AgentCore Gateway.
"""

from __future__ import annotations

import io
import json
import logging
import zipfile
from typing import Any

from agent.tools._decorator import tool

logger = logging.getLogger(__name__)

# Maven coordinates for AWS Smithy models
_SMITHY_MAVEN_BASE = "https://repo1.maven.org/maven2/software/amazon/api/models"


class GatewayToolManager:
    """Manages dynamic Gateway target creation and deletion.

    Parameters
    ----------
    agentcore_client:
        boto3 client for bedrock-agentcore-control.
    lambda_client:
        boto3 client for Lambda.
    s3_client:
        boto3 client for S3.
    bucket:
        S3 bucket for storing Smithy models and Lambda code.
    gateway_id:
        The AgentCore Gateway identifier.
    agent_id:
        The agent identifier (used for naming).
    """

    def __init__(
        self,
        agentcore_client: Any,
        lambda_client: Any,
        s3_client: Any,
        bucket: str,
        gateway_id: str,
        agent_id: str,
        lambda_role_arn: str = "",
    ) -> None:
        self._ac = agentcore_client
        self._lambda = lambda_client
        self._s3 = s3_client
        self._bucket = bucket
        self._gateway_id = gateway_id
        self._agent_id = agent_id
        self._lambda_role_arn = lambda_role_arn

    @tool
    def manage_gateway_tools(
        self,
        action: str,
        service_name: str = "",
        tool_name: str = "",
        description: str = "",
        python_code: str = "",
    ) -> str:
        """Add or remove tools from the agent's Gateway. Actions: add-aws-service, add-lambda, list, remove.

        Args:
            action: One of 'add-aws-service', 'add-lambda', 'list', 'remove'
            service_name: AWS service name for add-aws-service (e.g. 'eks', 'dynamodb', 'cloudwatch', 'ec2')
            tool_name: Name for the tool/target (required for add-lambda and remove)
            description: Description of the tool (for add-lambda)
            python_code: Python code for the Lambda function (for add-lambda)
        """
        if not self._gateway_id:
            return "Gateway not configured. Cannot manage tools."

        if action == "add-aws-service":
            return self._add_smithy_target(service_name)
        elif action == "add-lambda":
            return self._add_lambda_target(tool_name, description, python_code)
        elif action == "list":
            return self._list_targets()
        elif action == "remove":
            return self._remove_target(tool_name)

        return f"Unknown action: {action}. Use add-aws-service, add-lambda, list, or remove."

    def _add_smithy_target(self, service_name: str) -> str:
        """Download a Smithy model from Maven and register as a Gateway target."""
        if not service_name:
            return "service_name is required. Examples: eks, dynamodb, cloudwatch, ec2, s3, lambda, iam"

        service_name = service_name.lower().strip()

        # Download Smithy model from Maven
        try:
            import urllib.request
            # Try to get the latest version index
            maven_url = f"{_SMITHY_MAVEN_BASE}/{service_name}/maven-metadata.xml"
            with urllib.request.urlopen(maven_url, timeout=10) as resp:
                metadata = resp.read().decode()

            # Extract latest version
            import re
            match = re.search(r"<latest>([^<]+)</latest>", metadata)
            if not match:
                match = re.search(r"<version>([^<]+)</version>", metadata)
            if not match:
                return f"Could not find Smithy model version for '{service_name}' on Maven."
            version = match.group(1)

            # Download the JAR (contains the JSON model)
            jar_url = f"{_SMITHY_MAVEN_BASE}/{service_name}/{version}/{service_name}-{version}.jar"
            logger.info("Downloading Smithy model: %s", jar_url)
            with urllib.request.urlopen(jar_url, timeout=30) as resp:
                jar_bytes = resp.read()

        except Exception as exc:
            return f"Failed to download Smithy model for '{service_name}': {exc}"

        # Extract the JSON model from the JAR
        try:
            model_json = None
            with zipfile.ZipFile(io.BytesIO(jar_bytes)) as zf:
                for name in zf.namelist():
                    if name.endswith(".json") and "manifest" not in name.lower():
                        model_json = zf.read(name).decode()
                        break

            if not model_json:
                return f"No JSON model found in Smithy JAR for '{service_name}'."

        except Exception as exc:
            return f"Failed to extract Smithy model: {exc}"

        # Upload to S3
        s3_key = f"agents/{self._agent_id}/smithy-models/{service_name}.json"
        try:
            self._s3.put_object(
                Bucket=self._bucket,
                Key=s3_key,
                Body=model_json,
                ContentType="application/json",
            )
        except Exception as exc:
            return f"Failed to upload Smithy model to S3: {exc}"

        # Register as Gateway target
        target_name = f"{self._agent_id}-{service_name}"
        try:
            self._ac.create_gateway_target(
                gatewayIdentifier=self._gateway_id,
                name=target_name,
                description=f"AWS {service_name.upper()} service access via Smithy model",
                targetConfiguration={
                    "mcp": {
                        "smithyModel": {
                            "s3": {
                                "uri": f"s3://{self._bucket}/{s3_key}",
                            }
                        }
                    }
                },
                credentialProviderConfigurations=[
                    {"credentialProviderType": "GATEWAY_IAM_ROLE"}
                ],
            )
        except Exception as exc:
            err_msg = str(exc)
            if "already exists" in err_msg.lower() or "conflict" in err_msg.lower():
                return f"AWS {service_name.upper()} is already registered as a Gateway target."
            return f"Failed to register Gateway target: {exc}"

        # Clear caches so next request picks up the new tool
        try:
            from main import clear_caches
            clear_caches()
        except Exception:
            pass

        return f"AWS {service_name.upper()} added as a Gateway tool. The agent can now call {service_name.upper()} APIs."

    def _add_lambda_target(self, tool_name: str, description: str, python_code: str) -> str:
        """Create a Lambda function and register it as a Gateway target."""
        if not tool_name:
            return "tool_name is required."
        if not python_code:
            return "python_code is required."

        function_name = f"{self._agent_id}-custom-{tool_name}"

        # Create zip package
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("lambda_function.py", python_code)
        zip_bytes = zip_buffer.getvalue()

        # Create or update Lambda
        try:
            self._lambda.get_function(FunctionName=function_name)
            # Update existing
            self._lambda.update_function_code(
                FunctionName=function_name,
                ZipFile=zip_bytes,
            )
            logger.info("Updated Lambda: %s", function_name)
        except self._lambda.exceptions.ResourceNotFoundException:
            # Create new
            if not self._lambda_role_arn:
                return "Cannot create Lambda: no execution role configured."
            self._lambda.create_function(
                FunctionName=function_name,
                Runtime="python3.12",
                Role=self._lambda_role_arn,
                Handler="lambda_function.lambda_handler",
                Code={"ZipFile": zip_bytes},
                Description=description or f"Custom tool: {tool_name}",
                Timeout=30,
            )
            logger.info("Created Lambda: %s", function_name)
        except Exception as exc:
            return f"Failed to create Lambda: {exc}"

        # Get Lambda ARN
        try:
            fn_info = self._lambda.get_function(FunctionName=function_name)
            lambda_arn = fn_info["Configuration"]["FunctionArn"]
        except Exception as exc:
            return f"Failed to get Lambda ARN: {exc}"

        # Register as Gateway target
        target_name = f"{self._agent_id}-{tool_name}"
        try:
            self._ac.create_gateway_target(
                gatewayIdentifier=self._gateway_id,
                name=target_name,
                description=description or f"Custom tool: {tool_name}",
                targetConfiguration={
                    "mcp": {
                        "lambda": {
                            "lambdaArn": lambda_arn,
                            "toolSchema": {
                                "inlinePayload": [
                                    {
                                        "name": tool_name,
                                        "description": description or tool_name,
                                        "inputSchema": {
                                            "type": "object",
                                            "properties": {},
                                        },
                                    }
                                ]
                            },
                        }
                    }
                },
                credentialProviderConfigurations=[
                    {"credentialProviderType": "GATEWAY_IAM_ROLE"}
                ],
            )
        except Exception as exc:
            return f"Lambda created but failed to register Gateway target: {exc}"

        # Clear caches so next request picks up the new tool
        try:
            from main import clear_caches
            clear_caches()
        except Exception:
            pass

        return f"Custom tool '{tool_name}' created and registered. It's now available as a Gateway tool."

    def _list_targets(self) -> str:
        """List all Gateway targets."""
        try:
            resp = self._ac.list_gateway_targets(gatewayIdentifier=self._gateway_id)
            targets = resp.get("items", []) or resp.get("targets", [])
            if not targets:
                return "No Gateway targets registered."

            lines = [f"Gateway targets ({len(targets)}):"]
            for t in targets:
                status = t.get("status", "unknown")
                name = t.get("name", "unnamed")
                desc = t.get("description", "")
                lines.append(f"  [{status}] {name}: {desc}")
            return "\n".join(lines)
        except Exception as exc:
            return f"Failed to list targets: {exc}"

    def _remove_target(self, tool_name: str) -> str:
        """Remove a Gateway target by name."""
        if not tool_name:
            return "tool_name is required."

        # Find the target ID by name
        try:
            resp = self._ac.list_gateway_targets(gatewayIdentifier=self._gateway_id)
            target_id = None
            for t in resp.get("items", []) or resp.get("targets", []):
                if t.get("name") == tool_name or t.get("name") == f"{self._agent_id}-{tool_name}":
                    target_id = t.get("targetId")
                    break

            if not target_id:
                return f"Target '{tool_name}' not found."

            self._ac.delete_gateway_target(
                gatewayIdentifier=self._gateway_id,
                targetId=target_id,
            )

            # Clear caches so next request reflects the removal
            try:
                from main import clear_caches
                clear_caches()
            except Exception:
                pass

            return f"Target '{tool_name}' removed from Gateway."
        except Exception as exc:
            return f"Failed to remove target: {exc}"
