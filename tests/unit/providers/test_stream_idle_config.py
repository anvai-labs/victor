import pytest
from pydantic import ValidationError
from victor.config.settings import ProviderConfig
from victor.providers.sandhi_transport import SandhiTypedProviderMixin


def test_stream_idle_config_survives_profile_serialization():
    assert (
        ProviderConfig(stream_idle_timeout_secs=300).model_dump()["stream_idle_timeout_secs"] == 300
    )


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), 601, True])
def test_invalid_stream_idle_config_is_rejected(value):
    with pytest.raises(ValidationError):
        ProviderConfig(stream_idle_timeout_secs=value)


def test_stream_idle_runtime_is_explicit_and_bounded():
    provider = SandhiTypedProviderMixin()
    provider.extra_config = {}
    assert provider._sandhi_stream_idle_timeout() == 90
    provider.extra_config = {"stream_idle_timeout_secs": 300}
    assert provider._sandhi_stream_idle_timeout() == 300
    provider.extra_config = {"stream_idle_timeout_secs": float("inf")}
    with pytest.raises(ValueError):
        provider._sandhi_stream_idle_timeout()


@pytest.mark.asyncio
async def test_stream_idle_profile_reaches_ffi(monkeypatch):
    from types import SimpleNamespace
    from victor.providers import sandhi_transport as st
    from victor.providers.inferflux_provider import InferfluxProvider

    calls = []
    runtime = SimpleNamespace(provider=lambda *args, **kwargs: calls.append(kwargs) or object())
    monkeypatch.setattr(st, "_sg", SimpleNamespace(ProviderRuntime=lambda: runtime))
    monkeypatch.setattr(st, "_verify_wire_contract", lambda: None)
    cfg = ProviderConfig(stream_idle_timeout_secs=300).model_dump(exclude_none=True)
    provider = InferfluxProvider(api_key="synthetic", **cfg)
    try:
        provider._typed_provider("qwen3-coder-30b")
        assert calls[0]["stream_idle_timeout_secs"] == 300
    finally:
        await provider.close()
