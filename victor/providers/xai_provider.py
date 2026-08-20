# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0
"""xAI model policy over Sandhi's typed runtime."""

from typing import Optional

from victor.providers.openai_compat_model_policy import get_openai_compat_provider_spec
from victor.providers.sandhi_openai_compat_policy import SandhiOpenAICompatPolicy

_SPEC = get_openai_compat_provider_spec("xai")
DEFAULT_BASE_URL = _SPEC.base_url
XAI_MODELS = {model: dict(metadata) for model, metadata in _SPEC.models.items()}


class XAIProvider(SandhiOpenAICompatPolicy):
    CONFIG_KEY = "xai"

    def _clean_model_name(self, model: str) -> str:
        return {
            "grok-4.1-fast": "grok-4-1-fast",
            "grok-4.1-fast-reasoning": "grok-4-1-fast-reasoning",
            "grok-4.1-fast-non-reasoning": "grok-4-1-fast-non-reasoning",
        }.get(model, model)

    def get_context_window(self, model: str) -> int:
        return self.context_window(self._clean_model_name(model))

    def context_window(self, model: Optional[str] = None) -> int:
        return super().context_window(self._clean_model_name(model) if model else model)


__all__ = ["DEFAULT_BASE_URL", "XAI_MODELS", "XAIProvider"]
