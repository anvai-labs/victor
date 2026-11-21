"""Moonshot/Kimi model policy over Sandhi's typed OpenAI-compatible runtime."""

from __future__ import annotations

from typing import Any

import sandhi_gateway

from victor.providers.openai_compat_model_policy import get_openai_compat_provider_spec
from victor.providers.sandhi_openai_compat_policy import SandhiOpenAICompatPolicy

_SPEC = get_openai_compat_provider_spec("moonshot")
DEFAULT_BASE_URL = _SPEC.base_url
KIMI_K3_BASE_URL = sandhi_gateway.provider_spec("moonshot", "kimi-k3")["base_url"]
KIMI_K3_REASONING_EFFORTS = frozenset({"low", "high", "max"})
KIMI_K3_MODELS = {
    model: dict(metadata) for model, metadata in _SPEC.models.items() if model.startswith("kimi-k3")
}
KIMI_K2_MODELS = {
    model: dict(metadata) for model, metadata in _SPEC.models.items() if model.startswith("kimi-k2")
}


class MoonshotProvider(SandhiOpenAICompatPolicy):
    """Victor-owned Kimi model policy; endpoint routing and K3 constraints live in Sandhi."""

    CONFIG_KEY = "moonshot"

    @classmethod
    def resolve_base_url_for_model(cls, model: str) -> str:
        return str(sandhi_gateway.provider_spec("moonshot", model)["base_url"])

    def _build_request_payload(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Build compatibility payload; Sandhi enforces K3 wire constraints authoritatively."""
        payload = super()._build_request_payload(*args, **kwargs)
        model = str(payload.get("model", ""))
        if model.startswith("kimi-k3"):
            payload["temperature"] = 1.0
            effort = payload.get("reasoning_effort")
            if effort is not None and effort not in KIMI_K3_REASONING_EFFORTS:
                allowed = ", ".join(sorted(KIMI_K3_REASONING_EFFORTS))
                raise ValueError(f"reasoning_effort must be one of {allowed}")
        return payload


__all__ = [
    "DEFAULT_BASE_URL",
    "KIMI_K2_MODELS",
    "KIMI_K3_BASE_URL",
    "KIMI_K3_MODELS",
    "KIMI_K3_REASONING_EFFORTS",
    "MoonshotProvider",
]
