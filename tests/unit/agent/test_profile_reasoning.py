from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from victor.agent.orchestrator_creation import create_orchestrator_from_settings
from victor.config.settings import ProfileConfig
from victor.providers.openai_provider import OpenAIProvider


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "effort", [None, "none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"]
)
async def test_profile_reasoning_reaches_orchestrator(monkeypatch, effort):
    profile = ProfileConfig(provider="openai", model="gpt-6-sol", reasoning_effort=effort)
    settings = SimpleNamespace(
        load_profiles=lambda: {"default": profile},
        get_provider_settings=lambda *a, **k: {},
        enable_observability_logging=False,
    )
    monkeypatch.setattr(
        "victor.providers.registry.ProviderRegistry.create", lambda *a, **k: object()
    )
    monkeypatch.setattr(
        "victor.agent.tool_calling.capabilities.ModelCapabilityLoader.get_capabilities",
        lambda *a: None,
    )
    factory = MagicMock()
    await create_orchestrator_from_settings(factory, settings)
    assert factory.call_args.kwargs["reasoning_effort"] == effort


def test_unknown_reasoning_label_is_rejected():
    with pytest.raises(ValidationError):
        ProfileConfig(provider="openai", model="gpt-6-sol", reasoning_effort="typo")


@pytest.mark.parametrize("model", ["gpt-6-sol", "gpt-6-astra", "gpt-6-luna", "gpt-5.6-sol"])
def test_openai_recognizes_reasoning_models(model):
    provider = object.__new__(OpenAIProvider)
    assert provider.supports_reasoning_effort(model)
