# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0
"""InferFlux self-hosted model policy over Sandhi's typed runtime."""

import httpx

from victor.providers.openai_compat_model_policy import get_openai_compat_provider_spec
from victor.providers.sandhi_openai_compat_policy import SandhiOpenAICompatPolicy

_SPEC = get_openai_compat_provider_spec("inferflux")
DEFAULT_BASE_URL = _SPEC.base_url
INFERFLUX_MODELS = {model: dict(metadata) for model, metadata in _SPEC.models.items()}


class InferfluxProvider(SandhiOpenAICompatPolicy):
    """Thin typed policy for the self-hosted InferFlux inference server.

    Transport, wire facts, and base URL come from Sandhi's catalog descriptor
    (OpenAI Chat Completions dialect, default ``http://127.0.0.1:8080/v1`` —
    Sandhi ADR-0008). Model ids are OPERATOR config on the InferFlux side
    (its ``registry.yaml``), so the model lineup lives in victor's
    ``openai_compat_model_policy.yaml`` tier, not the catalog.
    """

    CONFIG_KEY = "inferflux"

    async def get_parallel_capacity(self, model: str) -> int:
        """Return declared sequence capacity; never infer it from model context size.

        Operators may pass max_parallel_sequences from the serving configuration.
        Without that declaration, require the admin model response to expose it.
        Older ROCm servers do not expose this field and fail explicitly.
        """
        capacity = self.extra_config.get("max_parallel_sequences")
        if capacity is None:
            base = (self.base_url or DEFAULT_BASE_URL).rstrip("/")
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(
                    f"{base}/admin/models",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
                response.raise_for_status()
                payload = response.json()
            matches = [item for item in payload.get("models", []) if item.get("id") == model]
            if len(matches) != 1:
                raise ValueError("InferFlux capacity lookup requires one matching model")
            capacity = matches[0].get("max_parallel_sequences")
        if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity < 1:
            raise ValueError(
                "InferFlux does not declare a positive max_parallel_sequences; "
                "supply the verified serving-config value as a provider option"
            )
        return capacity


__all__ = ["DEFAULT_BASE_URL", "INFERFLUX_MODELS", "InferfluxProvider"]
