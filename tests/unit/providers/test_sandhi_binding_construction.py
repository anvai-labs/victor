"""Construction conformance against the REAL sandhi binding (TD-0008 rule 5).

Victor's seam tests mock the runtime, so a validity rule that lives only inside
the binding's dispatcher is invisible to CI — that is exactly how gateway mode
shipped broken for every openai-family provider (auth_scheme rejection,
victor#678). This tier constructs typed handles for every provider family, in
both direct and gateway mode, against the INSTALLED binding: construction-time
validation runs for real, and no network is involved.
"""

from __future__ import annotations

import pytest

sg = pytest.importorskip("sandhi_gateway")

from victor.providers.anthropic_provider import AnthropicProvider
from victor.providers.deepseek_provider import DeepSeekProvider
from victor.providers.google_provider import GoogleProvider
from victor.providers.moonshot_provider import MoonshotProvider
from victor.providers.openai_provider import OpenAIProvider
from victor.providers.sandhi_transport import resolve_transport_class

GATEWAY = {"url": "http://127.0.0.1:9/v1", "virtual_key": "vk_conformance"}

CASES = [
    ("openai", OpenAIProvider, "gpt-test"),
    ("moonshot", MoonshotProvider, "kimi-k3"),
    ("deepseek", DeepSeekProvider, "deepseek-chat"),
    ("anthropic", AnthropicProvider, "claude-test"),
    ("google", GoogleProvider, "gemini-test"),
]


def _construct(name, native_cls, extra_kwargs):
    cls = resolve_transport_class(name, native_cls, {})
    return cls(api_key="sk-conformance-fake", **extra_kwargs)


@pytest.mark.parametrize("name,native_cls,model", CASES, ids=[c[0] for c in CASES])
def test_direct_mode_handle_constructs_on_real_binding(name, native_cls, model):
    provider = _construct(name, native_cls, {})
    handle = provider._typed_provider(model)
    assert handle is not None


@pytest.mark.parametrize("name,native_cls,model", CASES, ids=[c[0] for c in CASES])
def test_gateway_mode_handle_constructs_on_real_binding(name, native_cls, model):
    """The victor#678 regression class: gateway construction must satisfy the
    binding's parameter validation for EVERY family, not just the mocked view."""
    provider = _construct(name, native_cls, {"gateway": dict(GATEWAY)})
    handle = provider._typed_provider(model)
    assert handle is not None
