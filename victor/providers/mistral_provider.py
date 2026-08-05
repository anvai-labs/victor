# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0
"""Mistral model policy over Sandhi's typed runtime."""

from victor.providers.openai_compat_model_policy import get_openai_compat_provider_spec
from victor.providers.sandhi_openai_compat_policy import SandhiOpenAICompatPolicy

_SPEC = get_openai_compat_provider_spec("mistral")
DEFAULT_BASE_URL = _SPEC.base_url
MISTRAL_MODELS = {model: dict(metadata) for model, metadata in _SPEC.models.items()}


class MistralProvider(SandhiOpenAICompatPolicy):
    CONFIG_KEY = "mistral"


__all__ = ["DEFAULT_BASE_URL", "MISTRAL_MODELS", "MistralProvider"]
