# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0
"""Z.AI model and endpoint-selection policy over Sandhi's typed runtime."""

from __future__ import annotations

from typing import Any, Dict, Optional

from victor.providers.openai_compat_model_policy import get_openai_compat_provider_spec
from victor.providers.sandhi_openai_compat_policy import SandhiOpenAICompatPolicy

_SPEC = get_openai_compat_provider_spec("zai")
ZAI_BASE_URLS = {name: f"{url}/" for name, url in _SPEC.endpoint_options.items()}
ZAI_MODELS = {model: dict(metadata) for model, metadata in _SPEC.models.items()}


class ZAIProvider(SandhiOpenAICompatPolicy):
    """Victor selects a declared Z.AI plan/region; Sandhi owns those endpoint facts."""

    CONFIG_KEY = "zai"
    DEFAULT_TIMEOUT = 300

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = DEFAULT_TIMEOUT,
        max_retries: int = 3,
        non_interactive: Optional[bool] = None,
        coding_plan: bool = False,
        endpoint: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        suffix_endpoint: Optional[str] = None
        if model and ":" in model:
            candidate = model.rsplit(":", 1)[1]
            if candidate == "anthropic":
                raise ValueError(
                    "Z.AI's Anthropic endpoint is a different protocol and is not admitted "
                    "through the OpenAI-compatible provider"
                )
            if candidate in self.provider_spec().endpoint_options:
                suffix_endpoint = candidate
        selected = endpoint or ("coding" if coding_plan else None) or suffix_endpoint or "standard"
        if selected == "anthropic":
            raise ValueError(
                "Z.AI's Anthropic endpoint is a different protocol and is not admitted "
                "through the OpenAI-compatible provider"
            )
        if selected not in self.provider_spec().endpoint_options:
            available = ", ".join(sorted(self.provider_spec().endpoint_options))
            raise ValueError(f"unknown Z.AI endpoint {selected!r}; expected one of {available}")
        resolved_base = base_url or self.provider_spec().endpoint_options[selected]
        super().__init__(
            api_key=api_key,
            base_url=resolved_base,
            timeout=timeout,
            max_retries=max_retries,
            non_interactive=non_interactive,
            **kwargs,
        )

    def _clean_model_name(self, model: str) -> str:
        if model and ":" in model:
            name, suffix = model.rsplit(":", 1)
            if suffix in self.provider_spec().endpoint_options:
                return name
        return model

    def _get_provider_params(
        self,
        model: str,
        temperature: float,
        max_tokens: int,
        thinking: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {"temperature": temperature, "max_tokens": max_tokens}
        if thinking:
            params["thinking"] = {"type": "enabled"}
        return params

    def _extract_response_metadata(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        reasoning = message.get("reasoning_content")
        return {"reasoning_content": reasoning} if reasoning else None

    def _extract_stream_metadata(self, delta: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        reasoning = delta.get("reasoning_content")
        return {"reasoning_content": reasoning, "thinking_mode": True} if reasoning else None

    def context_window(self, model: Optional[str] = None) -> int:
        return super().context_window(self._clean_model_name(model or ""))

    def get_context_window(self, model: str) -> int:
        return self.context_window(model)


__all__ = ["ZAI_BASE_URLS", "ZAI_MODELS", "ZAIProvider"]
