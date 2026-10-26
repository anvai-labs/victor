import pytest
from pydantic import ValidationError
from victor.config.settings import ProviderConfig
from victor.providers.sandhi_transport import SandhiTypedProviderMixin


def test_stream_setup_config_survives_profile_serialization():
    assert (
        ProviderConfig(stream_setup_timeout_secs=300).model_dump()["stream_setup_timeout_secs"]
        == 300
    )


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), 601, True])
def test_invalid_stream_setup_config_is_rejected(value):
    with pytest.raises(ValidationError):
        ProviderConfig(stream_setup_timeout_secs=value)


def test_stream_setup_runtime_is_explicit_and_bounded():
    provider = SandhiTypedProviderMixin()
    provider.extra_config = {}
    # The 90s default matches the Sandhi library default: measured cache-miss
    # prefills of ~50K-token prompts exceed the old 30s time-to-first-byte bound.
    assert provider._sandhi_stream_setup_timeout() == 90
    provider.extra_config = {"stream_setup_timeout_secs": 45}
    assert provider._sandhi_stream_setup_timeout() == 45
    provider.extra_config = {"stream_setup_timeout_secs": float("inf")}
    with pytest.raises(ValueError):
        provider._sandhi_stream_setup_timeout()
    provider.extra_config = {"stream_setup_timeout_secs": True}
    with pytest.raises(ValueError):
        provider._sandhi_stream_setup_timeout()


@pytest.mark.asyncio
async def test_stream_setup_profile_reaches_ffi(monkeypatch):
    from types import SimpleNamespace
    from victor.providers import sandhi_transport as st
    from victor.providers.inferflux_provider import InferfluxProvider

    calls = []
    runtime = SimpleNamespace(provider=lambda *args, **kwargs: calls.append(kwargs) or object())
    monkeypatch.setattr(st, "_sg", SimpleNamespace(ProviderRuntime=lambda: runtime))
    monkeypatch.setattr(st, "_verify_wire_contract", lambda: None)
    cfg = ProviderConfig(stream_setup_timeout_secs=45).model_dump(exclude_none=True)
    provider = InferfluxProvider(api_key="synthetic", **cfg)
    try:
        provider._typed_provider("qwen3-coder-30b")
        assert calls[0]["stream_setup_timeout_secs"] == 45
    finally:
        await provider.close()
