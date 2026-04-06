"""Model routing configuration for Amazon Bedrock Intelligent Prompt Routing.

Provides the ModelRouter class that selects between a Prompt Router ARN
(for automatic model selection) and a specific model_id, with fallback
support when the router is unavailable.

Requirements: 11.1, 11.2, 11.3, 11.4, 11.5
"""

from __future__ import annotations

import logging
import re

from agent.models import BedrockModelConfig

logger = logging.getLogger(__name__)

# ARN pattern for Bedrock Prompt Router resources
_ROUTER_ARN_PATTERN = re.compile(
    r"^arn:aws:bedrock:[a-z0-9-]+:\d{12}:(default-prompt-router|prompt-router)/.+$"
)


class ModelRouter:
    """Configures Bedrock model selection with optional Intelligent Prompt Routing.

    When routing is enabled, the Prompt Router ARN is used as the model_id
    passed to Strands ``BedrockModel``.  When routing is disabled, a specific
    ``model_id`` is used instead.  If the Prompt Router is unavailable at
    runtime, the router falls back to ``fallback_model_id``.

    Parameters
    ----------
    routing_enabled:
        Whether to use Bedrock Intelligent Prompt Routing.
    router_arn:
        The Prompt Router ARN.  Required when *routing_enabled* is True.
    model_id:
        A specific Bedrock model identifier.  Used when routing is disabled.
    fallback_model_id:
        Model to use when the Prompt Router is unavailable.  Must always
        be set regardless of routing configuration.
    """

    def __init__(
        self,
        *,
        routing_enabled: bool,
        router_arn: str | None = None,
        model_id: str | None = None,
        fallback_model_id: str,
    ) -> None:
        if not fallback_model_id or not fallback_model_id.strip():
            raise ValueError("fallback_model_id must be non-empty")

        if routing_enabled:
            if not router_arn or not router_arn.strip():
                raise ValueError(
                    "router_arn must be provided when routing is enabled"
                )
            if not _ROUTER_ARN_PATTERN.match(router_arn):
                raise ValueError(
                    f"router_arn does not match expected Bedrock Prompt Router "
                    f"ARN format: {router_arn}"
                )
        else:
            if not model_id or not model_id.strip():
                raise ValueError(
                    "model_id must be provided when routing is disabled"
                )

        self._routing_enabled = routing_enabled
        self._router_arn = router_arn
        self._model_id = model_id
        self._fallback_model_id = fallback_model_id

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_model_config(self) -> BedrockModelConfig:
        """Return a ``BedrockModelConfig`` reflecting the current routing state."""
        return BedrockModelConfig(
            routing_enabled=self._routing_enabled,
            router_arn=self._router_arn,
            model_id=self._model_id,
            fallback_model_id=self._fallback_model_id,
        )

    def get_effective_model_id(self) -> str:
        """Return the model_id string to pass to Strands ``BedrockModel``.

        * If routing is enabled → returns the router ARN.
        * If routing is disabled → returns the specific model_id.
        """
        if self._routing_enabled:
            return self._router_arn  # type: ignore[return-value]
        return self._model_id  # type: ignore[return-value]

    def get_effective_model_id_with_fallback(
        self,
        router_available: bool = True,
    ) -> str:
        """Return the effective model_id, falling back when the router is down.

        Parameters
        ----------
        router_available:
            Set to ``False`` when the Prompt Router returned an error or
            is otherwise unreachable.  The method will fall back to
            ``fallback_model_id`` and log a warning.

        Returns
        -------
        str
            The model identifier to use for the current invocation.
        """
        if self._routing_enabled and not router_available:
            logger.warning(
                "Prompt Router is unavailable (ARN: %s); falling back to %s",
                self._router_arn,
                self._fallback_model_id,
            )
            return self._fallback_model_id

        return self.get_effective_model_id()
