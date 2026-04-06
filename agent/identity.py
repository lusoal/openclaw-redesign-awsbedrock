"""IdentityManager — loads, caches, and manages Sacred Text identity files from S3.

Implements Requirements 1.1–1.5: identity file loading, caching, and validation.
Implements Requirements 2.1–2.5: bootstrap execution.
Implements Requirements 3.1–3.3: system prompt composition.
Implements Requirements 4.1–4.3: identity file update and conflict handling.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from botocore.exceptions import ClientError

from agent.models import BootstrapResult, IdentityBundle, MemoryContext, validate_agent_id

logger = logging.getLogger(__name__)

# Files that MUST exist in S3 — missing any of these is a fatal error.
_REQUIRED_FILES = ("SOUL.md", "AGENTS.md")

# Files that are optional on first run — treated as empty strings if missing.
_OPTIONAL_FILES = ("IDENTITY.md", "USER.md", "MEMORY.md")

# Mapping from filename to IdentityBundle field name.
_FILE_FIELD_MAP: dict[str, str] = {
    "SOUL.md": "soul",
    "AGENTS.md": "agents",
    "IDENTITY.md": "identity",
    "USER.md": "user_profile",
    "MEMORY.md": "durable_memory",
}

# Mapping from file_type argument to S3 filename.
_FILE_TYPE_TO_FILENAME: dict[str, str] = {
    "identity": "IDENTITY.md",
    "user_profile": "USER.md",
    "durable_memory": "MEMORY.md",
}


class IdentityManager:
    """Manages loading and caching of identity files stored in S3.

    Parameters
    ----------
    s3_client:
        A boto3 S3 client (or compatible mock).
    bucket:
        The S3 bucket name where identity files are stored.
    """

    def __init__(self, s3_client: Any, bucket: str) -> None:
        self._s3 = s3_client
        self._bucket = bucket
        self._cache: dict[str, IdentityBundle] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_identity(self, agent_id: str) -> IdentityBundle:
        """Load identity files for *agent_id* from S3, returning a cached bundle on repeat calls.

        Raises
        ------
        ValueError
            If *agent_id* fails format validation.
        RuntimeError
            If a required file (SOUL.md or AGENTS.md) is missing from S3.
        """
        validate_agent_id(agent_id)

        if agent_id in self._cache:
            return self._cache[agent_id]

        prefix = f"agents/{agent_id}/"
        contents: dict[str, str] = {}

        # Load required files — raise on missing.
        missing: list[str] = []
        for filename in _REQUIRED_FILES:
            key = f"{prefix}{filename}"
            body = self._get_object_or_none(key)
            if body is None:
                missing.append(filename)
            else:
                contents[filename] = body

        if missing:
            paths = ", ".join(f"s3://{self._bucket}/{prefix}{f}" for f in missing)
            raise RuntimeError(
                f"Required identity file(s) missing: {', '.join(missing)}. "
                f"Expected at: {paths}"
            )

        # Load optional files — default to empty string.
        for filename in _OPTIONAL_FILES:
            key = f"{prefix}{filename}"
            body = self._get_object_or_none(key)
            contents[filename] = body if body is not None else ""

        bundle = IdentityBundle(
            agent_id=agent_id,
            soul=contents["SOUL.md"],
            agents=contents["AGENTS.md"],
            identity=contents.get("IDENTITY.md", ""),
            user_profile=contents.get("USER.md", ""),
            durable_memory=contents.get("MEMORY.md", ""),
            loaded_at=datetime.now(timezone.utc),
        )

        self._cache[agent_id] = bundle
        return bundle

    def run_bootstrap(self, agent_id: str) -> BootstrapResult:
        """Execute the one-time bootstrap ritual for *agent_id*.

        If ``BOOTSTRAP.md`` exists at ``agents/{agent_id}/BOOTSTRAP.md``, load it,
        initialise the dynamic identity files (IDENTITY.md, USER.md, MEMORY.md)
        with default empty content, then delete BOOTSTRAP.md.

        On failure the bootstrap file is retained for retry and the agent starts
        with default empty dynamic files.

        Implements Requirements 2.1–2.5.
        """
        validate_agent_id(agent_id)

        bootstrap_key = f"agents/{agent_id}/BOOTSTRAP.md"
        content = self._get_object_or_none(bootstrap_key)

        if content is None:
            return BootstrapResult(files_initialized=[], bootstrap_deleted=False)

        dynamic_files = ("IDENTITY.md", "USER.md", "MEMORY.md")
        prefix = f"agents/{agent_id}/"

        try:
            for filename in dynamic_files:
                self._put_object(f"{prefix}{filename}", "")

            self._s3.delete_object(Bucket=self._bucket, Key=bootstrap_key)

            logger.info(
                "Bootstrap completed for agent %s — initialised %s, deleted BOOTSTRAP.md",
                agent_id,
                ", ".join(dynamic_files),
            )
            return BootstrapResult(
                files_initialized=list(dynamic_files),
                bootstrap_deleted=True,
            )
        except Exception:
            logger.exception(
                "Bootstrap failed for agent %s — BOOTSTRAP.md retained for retry",
                agent_id,
            )
            return BootstrapResult(files_initialized=[], bootstrap_deleted=False)

    def build_system_prompt(
        self, bundle: IdentityBundle, memory_context: MemoryContext
    ) -> str:
        """Compose a system prompt from identity files and memory context.

        The output is deterministic: the same *bundle* and *memory_context*
        always produce the identical string.

        Implements Requirements 3.1, 3.2.
        """
        sections: list[str] = []

        # Identity sections — deterministic order.
        _section_pairs = [
            ("SOUL", bundle.soul),
            ("AGENTS", bundle.agents),
            ("IDENTITY", bundle.identity),
            ("USER", bundle.user_profile),
            ("MEMORY", bundle.durable_memory),
        ]
        for heading, content in _section_pairs:
            if content and content.strip():
                sections.append(f"## {heading}\n\n{content}")

        # Memory context sections.
        if memory_context.summaries:
            body = "\n".join(f"- {s}" for s in memory_context.summaries)
            sections.append(f"## RECENT SUMMARIES\n\n{body}")

        if memory_context.preferences:
            body = "\n".join(f"- {p}" for p in memory_context.preferences)
            sections.append(f"## USER PREFERENCES\n\n{body}")

        if memory_context.facts:
            body = "\n".join(f"- {f}" for f in memory_context.facts)
            sections.append(f"## KNOWN FACTS\n\n{body}")

        return "\n\n".join(sections)

    def update_file(
        self, agent_id: str, file_type: str, content: str
    ) -> None:
        """Persist a dynamic identity file update to S3.

        Uses S3 conditional writes (ETag comparison) to detect conflicts.
        On conflict the file is reloaded, the conflict is logged, and the
        write is retried once.

        Parameters
        ----------
        agent_id:
            The agent whose file is being updated.
        file_type:
            One of ``"identity"``, ``"user_profile"``, ``"durable_memory"``.
        content:
            The new file content.

        Raises
        ------
        ValueError
            If *file_type* is not a recognised dynamic file type.

        Implements Requirements 4.1, 4.2, 4.3.
        """
        validate_agent_id(agent_id)

        filename = _FILE_TYPE_TO_FILENAME.get(file_type)
        if filename is None:
            raise ValueError(
                f"Unknown file_type '{file_type}'. "
                f"Must be one of: {', '.join(sorted(_FILE_TYPE_TO_FILENAME))}"
            )

        key = f"agents/{agent_id}/{filename}"
        field_name = _FILE_FIELD_MAP[filename]

        self._conditional_put(agent_id, key, field_name, content, retry=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _conditional_put(
        self,
        agent_id: str,
        key: str,
        field_name: str,
        content: str,
        *,
        retry: bool,
    ) -> None:
        """Write *content* to S3 with ETag-based conflict detection.

        If the current ETag does not match (another writer modified the
        object), reload the file, log the conflict, and retry once when
        *retry* is ``True``.
        """
        etag = self._get_etag(key)

        try:
            put_kwargs: dict[str, Any] = {
                "Bucket": self._bucket,
                "Key": key,
                "Body": content.encode("utf-8"),
            }
            if etag is not None:
                put_kwargs["IfMatch"] = etag

            self._s3.put_object(**put_kwargs)
        except ClientError as exc:
            error_code = exc.response.get("Error", {}).get("Code", "")
            if error_code in ("PreconditionFailed", "412") and retry:
                logger.warning(
                    "Conflict detected writing %s for agent %s — reloading and retrying",
                    key,
                    agent_id,
                )
                # Reload the latest version so the cache stays fresh.
                latest = self._get_object_or_none(key)
                if latest is not None and agent_id in self._cache:
                    self._cache[agent_id] = self._cache[agent_id].model_copy(
                        update={field_name: latest}
                    )
                # Retry once without further retries.
                self._conditional_put(
                    agent_id, key, field_name, content, retry=False
                )
                return
            raise

        # Update the cached bundle if one exists.
        if agent_id in self._cache:
            self._cache[agent_id] = self._cache[agent_id].model_copy(
                update={field_name: content}
            )

    def _get_etag(self, key: str) -> str | None:
        """Return the ETag of an S3 object, or ``None`` if it does not exist."""
        try:
            response = self._s3.head_object(Bucket=self._bucket, Key=key)
            return response.get("ETag")
        except ClientError:
            return None

    def _get_object_or_none(self, key: str) -> str | None:
        """Return the UTF-8 body of an S3 object, or ``None`` if the key does not exist."""
        try:
            response = self._s3.get_object(Bucket=self._bucket, Key=key)
            return response["Body"].read().decode("utf-8")
        except self._s3.exceptions.NoSuchKey:
            return None

    def _put_object(self, key: str, body: str) -> None:
        """Write a UTF-8 string to an S3 object."""
        self._s3.put_object(Bucket=self._bucket, Key=key, Body=body.encode("utf-8"))
