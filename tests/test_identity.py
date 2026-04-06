"""Unit tests for agent/identity.py — IdentityManager.

Covers Requirements 1.1–1.5: loading, caching, required/optional files, validation.
"""

import boto3
import pytest
from moto import mock_aws

from agent.identity import IdentityManager

BUCKET = "test-identity-bucket"
AGENT_ID = "my-agent-1"
PREFIX = f"agents/{AGENT_ID}/"


def _put(s3, key: str, body: str) -> None:
    s3.put_object(Bucket=BUCKET, Key=key, Body=body.encode())


@pytest.fixture()
def s3():
    """Yield a moto-mocked S3 client with the test bucket created."""
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=BUCKET)
        yield client


@pytest.fixture()
def full_s3(s3):
    """S3 bucket pre-populated with all five identity files."""
    _put(s3, f"{PREFIX}SOUL.md", "You are a helpful agent.")
    _put(s3, f"{PREFIX}AGENTS.md", "Agent instructions here.")
    _put(s3, f"{PREFIX}IDENTITY.md", "Name: TestBot")
    _put(s3, f"{PREFIX}USER.md", "User likes Python.")
    _put(s3, f"{PREFIX}MEMORY.md", "Remembered fact.")
    return s3


# ------------------------------------------------------------------
# Successful loading
# ------------------------------------------------------------------

def test_load_all_files(full_s3):
    mgr = IdentityManager(full_s3, BUCKET)
    bundle = mgr.load_identity(AGENT_ID)

    assert bundle.agent_id == AGENT_ID
    assert bundle.soul == "You are a helpful agent."
    assert bundle.agents == "Agent instructions here."
    assert bundle.identity == "Name: TestBot"
    assert bundle.user_profile == "User likes Python."
    assert bundle.durable_memory == "Remembered fact."
    assert bundle.loaded_at is not None


# ------------------------------------------------------------------
# Missing required files
# ------------------------------------------------------------------

def test_missing_soul_raises(s3):
    _put(s3, f"{PREFIX}AGENTS.md", "agents content")
    mgr = IdentityManager(s3, BUCKET)

    with pytest.raises(RuntimeError, match="SOUL.md"):
        mgr.load_identity(AGENT_ID)


def test_missing_agents_raises(s3):
    _put(s3, f"{PREFIX}SOUL.md", "soul content")
    mgr = IdentityManager(s3, BUCKET)

    with pytest.raises(RuntimeError, match="AGENTS.md"):
        mgr.load_identity(AGENT_ID)


def test_missing_both_required_raises(s3):
    mgr = IdentityManager(s3, BUCKET)

    with pytest.raises(RuntimeError, match="SOUL.md") as exc_info:
        mgr.load_identity(AGENT_ID)
    assert "AGENTS.md" in str(exc_info.value)


# ------------------------------------------------------------------
# Optional files treated as empty strings
# ------------------------------------------------------------------

def test_missing_optional_files_default_empty(s3):
    _put(s3, f"{PREFIX}SOUL.md", "soul")
    _put(s3, f"{PREFIX}AGENTS.md", "agents")
    mgr = IdentityManager(s3, BUCKET)

    bundle = mgr.load_identity(AGENT_ID)

    assert bundle.identity == ""
    assert bundle.user_profile == ""
    assert bundle.durable_memory == ""


# ------------------------------------------------------------------
# Caching behaviour
# ------------------------------------------------------------------

def test_caching_returns_same_bundle(full_s3):
    mgr = IdentityManager(full_s3, BUCKET)
    first = mgr.load_identity(AGENT_ID)
    second = mgr.load_identity(AGENT_ID)

    assert first is second  # exact same object — no second S3 fetch


def test_caching_does_not_hit_s3_twice(full_s3):
    """After the first load, delete the files from S3 — second call must still succeed."""
    mgr = IdentityManager(full_s3, BUCKET)
    mgr.load_identity(AGENT_ID)

    # Remove all files from S3
    for filename in ("SOUL.md", "AGENTS.md", "IDENTITY.md", "USER.md", "MEMORY.md"):
        full_s3.delete_object(Bucket=BUCKET, Key=f"{PREFIX}{filename}")

    # Should still return the cached bundle without error
    bundle = mgr.load_identity(AGENT_ID)
    assert bundle.soul == "You are a helpful agent."


# ------------------------------------------------------------------
# agent_id validation
# ------------------------------------------------------------------

def test_invalid_agent_id_empty(s3):
    mgr = IdentityManager(s3, BUCKET)
    with pytest.raises(ValueError, match="non-empty"):
        mgr.load_identity("")


def test_invalid_agent_id_special_chars(s3):
    mgr = IdentityManager(s3, BUCKET)
    with pytest.raises(ValueError, match="alphanumeric"):
        mgr.load_identity("bad agent!")


def test_invalid_agent_id_too_long(s3):
    mgr = IdentityManager(s3, BUCKET)
    with pytest.raises(ValueError, match="at most 50"):
        mgr.load_identity("a" * 51)


# ------------------------------------------------------------------
# Bootstrap execution (Requirements 2.1–2.5)
# ------------------------------------------------------------------

def test_bootstrap_with_bootstrap_md_present(s3):
    """BOOTSTRAP.md exists → dynamic files initialised, BOOTSTRAP.md deleted."""
    _put(s3, f"{PREFIX}SOUL.md", "soul")
    _put(s3, f"{PREFIX}AGENTS.md", "agents")
    _put(s3, f"{PREFIX}BOOTSTRAP.md", "Initialise the agent identity files.")

    mgr = IdentityManager(s3, BUCKET)
    result = mgr.run_bootstrap(AGENT_ID)

    assert result.bootstrap_deleted is True
    assert set(result.files_initialized) == {"IDENTITY.md", "USER.md", "MEMORY.md"}

    # Dynamic files should now exist (empty content)
    for filename in ("IDENTITY.md", "USER.md", "MEMORY.md"):
        obj = s3.get_object(Bucket=BUCKET, Key=f"{PREFIX}{filename}")
        assert obj["Body"].read().decode() == ""

    # BOOTSTRAP.md should be deleted
    with pytest.raises(s3.exceptions.NoSuchKey):
        s3.get_object(Bucket=BUCKET, Key=f"{PREFIX}BOOTSTRAP.md")


def test_bootstrap_no_bootstrap_md(s3):
    """No BOOTSTRAP.md → empty result, nothing changed."""
    mgr = IdentityManager(s3, BUCKET)
    result = mgr.run_bootstrap(AGENT_ID)

    assert result.files_initialized == []
    assert result.bootstrap_deleted is False


def test_bootstrap_failure_retains_bootstrap_md(s3, monkeypatch):
    """On failure, BOOTSTRAP.md is retained and empty result returned."""
    _put(s3, f"{PREFIX}BOOTSTRAP.md", "bootstrap content")

    mgr = IdentityManager(s3, BUCKET)

    # Force put_object to raise so the bootstrap fails after reading BOOTSTRAP.md
    def _failing_put(key, body):
        raise RuntimeError("simulated S3 write failure")

    monkeypatch.setattr(mgr, "_put_object", _failing_put)

    result = mgr.run_bootstrap(AGENT_ID)

    assert result.files_initialized == []
    assert result.bootstrap_deleted is False

    # BOOTSTRAP.md should still exist
    obj = s3.get_object(Bucket=BUCKET, Key=f"{PREFIX}BOOTSTRAP.md")
    assert obj["Body"].read().decode() == "bootstrap content"


# ------------------------------------------------------------------
# System prompt composition (Requirements 3.1, 3.2)
# ------------------------------------------------------------------

def test_build_system_prompt_deterministic(full_s3):
    """Same inputs must always produce the same output (Req 3.2)."""
    from agent.models import MemoryContext

    mgr = IdentityManager(full_s3, BUCKET)
    bundle = mgr.load_identity(AGENT_ID)
    ctx = MemoryContext(
        summaries=["Had a productive session."],
        preferences=["Prefers concise answers."],
        facts=["User is a Python developer."],
    )

    first = mgr.build_system_prompt(bundle, ctx)
    second = mgr.build_system_prompt(bundle, ctx)

    assert first == second


def test_build_system_prompt_includes_all_non_empty_fields(full_s3):
    """Composed prompt must contain every non-empty identity field and memory entry (Req 3.1)."""
    from agent.models import MemoryContext

    mgr = IdentityManager(full_s3, BUCKET)
    bundle = mgr.load_identity(AGENT_ID)
    ctx = MemoryContext(
        summaries=["summary-one"],
        preferences=["pref-one"],
        facts=["fact-one"],
    )

    prompt = mgr.build_system_prompt(bundle, ctx)

    # Identity fields
    assert bundle.soul in prompt
    assert bundle.agents in prompt
    assert bundle.identity in prompt
    assert bundle.user_profile in prompt
    assert bundle.durable_memory in prompt

    # Memory context entries
    assert "summary-one" in prompt
    assert "pref-one" in prompt
    assert "fact-one" in prompt


def test_build_system_prompt_skips_empty_sections(s3):
    """Empty identity fields and empty memory lists should not appear in the prompt."""
    from agent.models import MemoryContext

    _put(s3, f"{PREFIX}SOUL.md", "soul content")
    _put(s3, f"{PREFIX}AGENTS.md", "agents content")

    mgr = IdentityManager(s3, BUCKET)
    bundle = mgr.load_identity(AGENT_ID)
    ctx = MemoryContext()  # all lists empty

    prompt = mgr.build_system_prompt(bundle, ctx)

    assert "IDENTITY" not in prompt
    assert "USER" not in prompt
    assert "MEMORY" not in prompt
    assert "RECENT SUMMARIES" not in prompt
    assert "USER PREFERENCES" not in prompt
    assert "KNOWN FACTS" not in prompt

    # Required sections should still be present
    assert "soul content" in prompt
    assert "agents content" in prompt


# ------------------------------------------------------------------
# update_file (Requirements 4.1, 4.2, 4.3)
# ------------------------------------------------------------------

def test_update_file_writes_to_s3(full_s3):
    """update_file should persist content to the correct S3 key (Req 4.1)."""
    mgr = IdentityManager(full_s3, BUCKET)
    mgr.load_identity(AGENT_ID)  # populate cache

    mgr.update_file(AGENT_ID, "identity", "Updated identity content")

    obj = full_s3.get_object(Bucket=BUCKET, Key=f"{PREFIX}IDENTITY.md")
    assert obj["Body"].read().decode() == "Updated identity content"


def test_update_file_updates_cache(full_s3):
    """After update_file, the cached bundle should reflect the new content."""
    mgr = IdentityManager(full_s3, BUCKET)
    mgr.load_identity(AGENT_ID)

    mgr.update_file(AGENT_ID, "user_profile", "New user profile")

    bundle = mgr.load_identity(AGENT_ID)  # returns cached
    assert bundle.user_profile == "New user profile"


def test_update_file_invalid_file_type(full_s3):
    """update_file should reject unknown file types."""
    mgr = IdentityManager(full_s3, BUCKET)

    with pytest.raises(ValueError, match="Unknown file_type"):
        mgr.update_file(AGENT_ID, "soul", "nope")


def test_update_file_conflict_detection(full_s3, monkeypatch):
    """update_file should detect ETag conflicts and retry (Req 4.2, 4.3)."""
    from unittest.mock import patch, MagicMock
    from botocore.exceptions import ClientError

    mgr = IdentityManager(full_s3, BUCKET)
    mgr.load_identity(AGENT_ID)

    call_count = 0
    original_put = full_s3.put_object

    def mock_put_object(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1 and "IfMatch" in kwargs:
            # Simulate ETag mismatch on first attempt
            raise ClientError(
                {"Error": {"Code": "PreconditionFailed", "Message": "ETag mismatch"}},
                "PutObject",
            )
        # Remove IfMatch for the retry since we can't guarantee the ETag
        kwargs.pop("IfMatch", None)
        return original_put(**kwargs)

    monkeypatch.setattr(full_s3, "put_object", mock_put_object)

    mgr.update_file(AGENT_ID, "durable_memory", "Conflict-resolved content")

    # Should have retried — call_count >= 2
    assert call_count >= 2

    # Cache should be updated
    bundle = mgr.load_identity(AGENT_ID)
    assert bundle.durable_memory == "Conflict-resolved content"
