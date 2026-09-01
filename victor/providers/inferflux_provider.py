# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0
"""InferFlux self-hosted model policy over Sandhi's typed runtime."""

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


__all__ = ["DEFAULT_BASE_URL", "INFERFLUX_MODELS", "InferfluxProvider"]
