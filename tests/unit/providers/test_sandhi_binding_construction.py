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
from victor.providers.inferflux_provider import InferfluxProvider
from victor.providers.moonshot_provider import MoonshotProvider
from victor.providers.openai_provider import OpenAIProvider
from victor.providers.sandhi_transport import resolve_transport_class
from victor.providers.zai_provider import ZAIProvider

GATEWAY = {"url": "http://127.0.0.1:9/v1", "virtual_key": "vk_conformance"}

CASES = [
    ("openai", OpenAIProvider, "gpt-test"),
    ("moonshot", MoonshotProvider, "kimi-k3"),
    ("deepseek", DeepSeekProvider, "deepseek-chat"),
    ("anthropic", AnthropicProvider, "claude-test"),
    ("google", GoogleProvider, "gemini-test"),
    ("inferflux", InferfluxProvider, "qwen3-coder-30b"),
    ("zai", ZAIProvider, "glm-5.3"),
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


def test_gateway_mode_constructs_with_symlinked_ca_on_real_binding(tmp_path):
    """Victor and Sandhi must agree that a symlink to a regular CA file is valid."""
    import datetime

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "gateway-ca-conformance")])
    now = datetime.datetime.now(datetime.timezone.utc)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=1))
        .not_valid_after(now + datetime.timedelta(hours=1))
        .sign(key, hashes.SHA256())
    )
    certificate_file = tmp_path / "gateway-ca.der"
    certificate_file.write_bytes(certificate.public_bytes(serialization.Encoding.DER))
    ca_file = tmp_path / "gateway-ca-link.der"
    try:
        ca_file.symlink_to(certificate_file)
    except OSError as error:
        pytest.skip(f"temporary symlink support unavailable: {error}")
    provider = _construct(
        "openai", OpenAIProvider, {"gateway": {**GATEWAY, "ca_file": str(ca_file)}}
    )
    assert provider._typed_provider("gpt-test") is not None


@pytest.mark.parametrize(
    "protocol", ["chat_completions", "anthropic_messages", "responses", "gemini_generate_content"]
)
@pytest.mark.parametrize("name,native_cls,model", CASES, ids=[c[0] for c in CASES])
def test_gateway_explicit_client_dialect_is_independent_of_provider_family(
    name, native_cls, model, protocol
):
    provider = _construct(name, native_cls, {"gateway": {**GATEWAY, "protocol": protocol}})
    provider._sandhi_endpoint_profile = "openai_chat_terminal"
    assert provider._typed_provider(model) is not None


@pytest.mark.parametrize(
    "name, native_cls, qualified, unqualified",
    [
        ("anthropic", AnthropicProvider, "claude-sonnet-4-6", "claude-unqualified-fixture"),
        ("google", GoogleProvider, "gemini-3-flash-preview", "gemini-3-flash"),
    ],
)
def test_direct_discovery_exposes_only_qualified_thinking_models(
    name, native_cls, qualified, unqualified
):
    provider = _construct(name, native_cls, {})
    assert provider.supports_reasoning_effort(qualified)
    assert not provider.supports_reasoning_effort(unqualified)
