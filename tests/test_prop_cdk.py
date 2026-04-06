"""Property-based tests for CDK InfrastructureStack.

**Property 23: CDK S3 bucket security**
For any valid InfraStackProps, the synthesized CloudFormation template shall
contain an S3 bucket with encryption enabled, versioning enabled, public access
blocked, and SSL enforced.

**Validates: Requirement 15.1**

**Property 24: CDK IAM policy scoping**
For any valid InfraStackProps, all IAM policy resource ARNs in the synthesized
CloudFormation template shall be scoped to specific resources (containing the
agent_id, bucket name, memory_id, or schedule group name) with no wildcard-only
resource statements.

**Validates: Requirement 15.2**
"""

from __future__ import annotations

import json

import aws_cdk as cdk
from hypothesis import given, settings
from hypothesis import strategies as st

from agent.models import InfraStackProps
from infra.stacks.agent_stack import InfrastructureStack

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Valid agent_id: 1-20 chars (keep short to avoid CDK naming limits)
valid_agent_id_st = st.from_regex(r"[a-z][a-z0-9\-]{0,14}", fullmatch=True)

# Valid bedrock model id
bedrock_model_id_st = st.just("anthropic.claude-sonnet-4-20250514")

# Valid bedrock region
bedrock_region_st = st.sampled_from(["us-east-1", "us-west-2", "eu-west-1"])

# Valid memory id
memory_id_st = st.from_regex(r"mem-[a-z0-9]{6,10}", fullmatch=True)

# Valid schedule group name (kebab-case)
schedule_group_st = st.from_regex(r"[a-z][a-z0-9\-]{2,15}", fullmatch=True)


def _synth_template(props: InfraStackProps) -> dict:
    """Synthesize the CDK stack and return the CloudFormation template dict."""
    app = cdk.App()
    stack = InfrastructureStack(
        app,
        "TestStack",
        props=props,
        env=cdk.Environment(account="123456789012", region=props.bedrock_region),
    )
    assembly = app.synth()
    template = assembly.get_stack_by_name("TestStack").template
    return template


# ===========================================================================
# Property 23: CDK S3 bucket security
# ===========================================================================


class TestCdkS3BucketSecurity:
    """Synthesized template has encryption, versioning, public access blocked,
    and SSL enforced on the S3 bucket."""

    @given(
        agent_id=valid_agent_id_st,
        bedrock_region=bedrock_region_st,
        schedule_group=schedule_group_st,
    )
    @settings(max_examples=10, deadline=30000)
    def test_s3_bucket_security_properties(
        self,
        agent_id: str,
        bedrock_region: str,
        schedule_group: str,
    ):
        """**Validates: Requirement 15.1**"""
        props = InfraStackProps(
            agent_id=agent_id,
            bedrock_model_id="anthropic.claude-sonnet-4-20250514",
            bedrock_region=bedrock_region,
            schedule_group_name=schedule_group,
            enable_prompt_routing=False,
        )
        template = _synth_template(props)
        resources = template["Resources"]

        # Find the S3 bucket resource
        buckets = {
            k: v for k, v in resources.items()
            if v["Type"] == "AWS::S3::Bucket"
        }
        assert len(buckets) >= 1, "Expected at least one S3 bucket"

        for _logical_id, bucket in buckets.items():
            bucket_props = bucket["Properties"]

            # Encryption: SSE-S3 (AES256)
            enc = bucket_props.get("BucketEncryption", {})
            rules = enc.get("ServerSideEncryptionConfiguration", [])
            assert len(rules) > 0, "S3 bucket must have encryption configured"
            algo = rules[0]["ServerSideEncryptionByDefault"]["SSEAlgorithm"]
            assert algo == "aws:kms" or algo == "AES256", (
                f"Expected SSE-S3 (AES256) or aws:kms, got {algo}"
            )

            # Versioning enabled
            versioning = bucket_props.get("VersioningConfiguration", {})
            assert versioning.get("Status") == "Enabled", (
                "S3 bucket must have versioning enabled"
            )

            # Public access blocked (all four flags true)
            pab = bucket_props.get("PublicAccessBlockConfiguration", {})
            assert pab.get("BlockPublicAcls") is True
            assert pab.get("BlockPublicPolicy") is True
            assert pab.get("IgnorePublicAcls") is True
            assert pab.get("RestrictPublicBuckets") is True

        # SSL enforced: bucket policy with aws:SecureTransport condition
        bucket_policies = {
            k: v for k, v in resources.items()
            if v["Type"] == "AWS::S3::BucketPolicy"
        }
        assert len(bucket_policies) >= 1, (
            "Expected a bucket policy for SSL enforcement"
        )

        ssl_enforced = False
        for _logical_id, policy_resource in bucket_policies.items():
            statements = (
                policy_resource["Properties"]["PolicyDocument"]["Statement"]
            )
            for stmt in statements:
                condition = stmt.get("Condition", {})
                # CDK enforce_ssl creates a Deny with StringEquals
                # aws:SecureTransport = "false"
                bool_cond = condition.get("Bool", {})
                str_eq_cond = condition.get("StringEquals", {})
                if "aws:SecureTransport" in bool_cond:
                    ssl_enforced = True
                elif "aws:SecureTransport" in str_eq_cond:
                    ssl_enforced = True

        assert ssl_enforced, (
            "S3 bucket policy must enforce SSL (aws:SecureTransport condition)"
        )


# ===========================================================================
# Property 24: CDK IAM policy scoping
# ===========================================================================


class TestCdkIamPolicyScoping:
    """All IAM policy resource ARNs scoped to specific resources, no
    wildcard-only statements."""

    @given(
        agent_id=valid_agent_id_st,
        bedrock_region=bedrock_region_st,
        schedule_group=schedule_group_st,
    )
    @settings(max_examples=10, deadline=30000)
    def test_no_wildcard_only_iam_resources(
        self,
        agent_id: str,
        bedrock_region: str,
        schedule_group: str,
    ):
        """**Validates: Requirement 15.2**"""
        props = InfraStackProps(
            agent_id=agent_id,
            bedrock_model_id="anthropic.claude-sonnet-4-20250514",
            bedrock_region=bedrock_region,
            schedule_group_name=schedule_group,
            enable_prompt_routing=False,
        )
        template = _synth_template(props)
        resources = template["Resources"]

        # Collect all IAM policy documents from inline policies on roles
        iam_policies = {
            k: v for k, v in resources.items()
            if v["Type"] == "AWS::IAM::Policy"
        }

        for logical_id, policy_resource in iam_policies.items():
            doc = policy_resource["Properties"]["PolicyDocument"]
            statements = doc.get("Statement", [])
            for stmt in statements:
                raw_resources = stmt.get("Resource", [])
                # Normalize to list
                if isinstance(raw_resources, str):
                    raw_resources = [raw_resources]
                elif isinstance(raw_resources, dict):
                    # CloudFormation intrinsic (Fn::Join, Ref, etc.) — scoped
                    continue

                # X-Ray telemetry actions require Resource: "*" by design
                actions = stmt.get("Action", [])
                if isinstance(actions, str):
                    actions = [actions]
                xray_actions = {
                    "xray:PutTraceSegments",
                    "xray:PutTelemetryRecords",
                    "xray:GetSamplingRules",
                    "xray:GetSamplingTargets",
                }
                if set(actions) <= xray_actions:
                    continue

                for res in raw_resources:
                    if isinstance(res, str):
                        assert res != "*", (
                            f"IAM policy {logical_id} has wildcard-only "
                            f"resource '*' — all resources must be scoped. "
                            f"Statement: {json.dumps(stmt, default=str)}"
                        )
                    # dicts are CloudFormation intrinsics (Fn::Join, Ref,
                    # Fn::GetAtt, etc.) which are scoped by construction
