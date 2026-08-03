"""Contract tests for Victor's direct typed Sandhi provider boundary."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import sys

import pytest

import victor.providers.sandhi_transport as st
from victor.providers.base import (
    Message,
    ProviderConnectionError,
    ProviderRateLimitError,
    ToolDefinition,
)
from victor.providers.deepseek_provider import DeepSeekProvider
from victor.providers.google_provider import GoogleProvider
from victor.providers.llamacpp_provider import LlamaCppProvider
from victor.providers.lmstudio_provider import LMStudioProvider
from victor.providers.moonshot_provider import MoonshotProvider
from victor.providers.ollama_provider import OllamaProvider
from victor.providers.openai_provider import OpenAIProvider
from victor.providers.qwen_provider import QwenProvider
from victor.providers.vllm_provider import VLLMProvider


class FakeTypedProvider:
    def __init__(self, *, complete_error: BaseException | None = None) -> None:
        self.requests: list[dict] = []
        self.complete_error = complete_error

    async def complete_json(self, request_json: str) -> str:
        self.requests.append(json.loads(request_json))
        if self.complete_error:
            raise self.complete_error
        return json.dumps(
            {
                "schema_version": "1",
                "id": "r1",
                "model": "deepseek-chat",
                "output": {
                    "content": "hello",
                    "tool_calls": [{"id": "c1", "name": "lookup", "arguments": '{"q":1}'}],
                },
                "finish_reason": "tool_calls",
                "usage": {
                    "tokens_in": 6,
                    "tokens_out": 5,
                    "cache_creation_tokens": 0,
                    "cache_read_tokens": 4,
                    "completeness": "final",
                    "attempts": 1,
                },
                "extensions": {
                    "openai": {
                        "id": "r1",
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 5,
                            "total_tokens": 15,
                        },
                    }
                },
            }
        )

    def stream_json(self, request_json: str):
        self.requests.append(json.loads(request_json))

        async def events():
            for event in (
                {"event": "response_start", "id": "r2", "model": "deepseek-chat"},
                {"event": "text_delta", "delta": "he"},
                {"event": "reasoning_delta", "delta": "think"},
                {"event": "tool_call_start", "index": 0, "id": "c1", "name": "lookup"},
                {"event": "tool_call_arguments_delta", "index": 0, "delta": '{"q":'},
                {"event": "tool_call_arguments_delta", "index": 0, "delta": "1}"},
                {"event": "tool_call_end", "index": 0},
                {"event": "finish", "reason": "tool_calls"},
                {
                    "event": "usage",
                    "usage": {
                        "tokens_in": 6,
                        "tokens_out": 5,
                        "cache_creation_tokens": 0,
                        "cache_read_tokens": 4,
                    },
                },
            ):
                yield json.dumps(event)

        return events()


class FakeRuntime:
    def __init__(self, handle: FakeTypedProvider) -> None:
        self.handle = handle
        self.calls: list[tuple] = []

    def provider(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.handle


def install_runtime(monkeypatch, handle: FakeTypedProvider | None = None) -> FakeRuntime:
    runtime = FakeRuntime(handle or FakeTypedProvider())
    monkeypatch.setattr(st, "_sg", SimpleNamespace(ProviderRuntime=lambda: runtime))
    return runtime


def make_provider() -> DeepSeekProvider:
    return DeepSeekProvider(api_key="k", base_url="https://api.deepseek.com/v1")


def test_resolver_always_uses_sandhi_for_admitted_provider(monkeypatch):
    install_runtime(monkeypatch)
    assert st.resolve_transport_class("deepseek", DeepSeekProvider, {}) is DeepSeekProvider


@pytest.mark.parametrize(
    ("name", "provider_cls", "expected"),
    (
        ("openai", OpenAIProvider, st.SandhiOpenAIProvider),
        ("google", GoogleProvider, st.SandhiGoogleProvider),
        ("ollama", OllamaProvider, st.SandhiOllamaProvider),
        ("qwen", QwenProvider, QwenProvider),
        ("lmstudio", LMStudioProvider, st.SandhiLMStudioProvider),
        ("vllm", VLLMProvider, st.SandhiVLLMProvider),
        ("llama.cpp", LlamaCppProvider, st.SandhiLlamaCppProvider),
    ),
)
def test_native_families_resolve_to_typed_sandhi_handles(
    name: str, provider_cls: type, expected: type, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_runtime(monkeypatch)
    assert st.resolve_transport_class(name, provider_cls, {}) is expected


@pytest.mark.asyncio
async def test_catalog_default_is_omitted_so_sandhi_owns_model_endpoint_routing(monkeypatch):
    runtime = FakeRuntime(FakeTypedProvider())
    monkeypatch.setattr(
        st,
        "_sg",
        SimpleNamespace(
            ProviderRuntime=lambda: runtime,
            provider_spec=lambda provider: {
                "slug": "moonshot",
                "base_url": "https://api.moonshot.cn/v1",
            },
        ),
    )
    resolved = st.resolve_transport_class("moonshot", MoonshotProvider, {})
    provider = resolved(api_key="k")
    await provider.chat([Message(role="user", content="hi")], model="kimi-k3")

    args, kwargs = runtime.calls[0]
    assert args[:3] == ("moonshot", "kimi-k3", "k")
    assert kwargs["base_url"] is None


@pytest.mark.asyncio
async def test_openai_oauth_explicitly_selects_responses_and_refreshes_before_handle(monkeypatch):
    runtime = install_runtime(monkeypatch)
    with patch("victor.providers.openai_provider.OAuthTokenManager") as manager_cls:
        manager = MagicMock()
        manager._load_cached.return_value = SimpleNamespace(
            access_token="cached-oauth", is_expired=False
        )
        manager.get_valid_token = AsyncMock(return_value="fresh-oauth")
        manager.get_chatgpt_account_id.return_value = "workspace_123"
        manager_cls.return_value = manager
        provider = st.SandhiOpenAIProvider(auth_mode="oauth")

    await provider.chat(
        [Message(role="developer", content="policy"), Message(role="user", content="hi")],
        model="o3",
        reasoning_effort="high",
    )

    args, kwargs = runtime.calls[0]
    assert args[:3] == ("openai", "o3", "fresh-oauth")
    assert kwargs["protocol"] == "chatgpt_responses"
    assert kwargs["base_url"] == "https://chatgpt.com/backend-api/codex"
    assert json.loads(kwargs["headers_json"])["originator"] == "victor"
    assert json.loads(kwargs["headers_json"])["ChatGPT-Account-ID"] == "workspace_123"
    request = runtime.handle.requests[0]
    assert "temperature" not in request
    # At sandhi's >= 0.1.5 floor (contract minor >= 4) reasoning_effort rides the
    # typed ChatRequestV1 field; sandhi's Responses codec maps it to reasoning.effort
    # in the native body (openai_responses_typed.rs), so victor no longer dual-writes
    # it into the extensions bucket.
    assert request["reasoning_effort"] == "high"
    assert request["extensions"] == {"openai_responses": {}}


def test_resolver_fails_closed_when_binding_is_missing(monkeypatch):
    monkeypatch.setattr(st, "_sg", None)
    with pytest.raises(ProviderConnectionError):
        st.resolve_transport_class("deepseek", DeepSeekProvider, {})


def test_unknown_non_admitted_provider_is_unchanged(monkeypatch):
    install_runtime(monkeypatch)

    class Other(BaseException):
        pass

    assert st.resolve_transport_class("other", Other, {}) is Other


@pytest.mark.parametrize("name", sorted(st.VICTOR_NATIVE_ONLY_PROVIDER_ALIASES))
def test_explicit_native_only_boundary_is_preserved(name, monkeypatch):
    install_runtime(monkeypatch)

    class NativeOnly:
        pass

    assert st.resolve_transport_class(name, NativeOnly, {}) is NativeOnly


def test_unclassified_victor_owned_provider_fails_closed(monkeypatch):
    install_runtime(monkeypatch)
    provider_cls = type(
        "FutureProvider",
        (),
        {"__module__": "victor.providers.future_provider"},
    )

    with pytest.raises(ProviderConnectionError, match="not classified"):
        st.resolve_transport_class("future", provider_cls, {})


@pytest.mark.asyncio
async def test_complete_consumes_typed_response_and_reuses_handle(monkeypatch):
    runtime = install_runtime(monkeypatch)
    provider = make_provider()
    messages = [Message(role="developer", content="policy"), Message(role="user", content="hi")]
    tools = [ToolDefinition(name="lookup", description="Lookup", parameters={"type": "object"})]

    first = await provider.chat(messages, model="deepseek-chat", tools=tools)
    second = await provider.chat(messages, model="deepseek-chat", tools=tools)

    assert first.content == "hello"
    assert first.tool_calls == [{"id": "c1", "name": "lookup", "arguments": {"q": 1}}]
    assert first.usage == {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
        "cache_read_input_tokens": 4,
    }
    assert len(runtime.calls) == 1, "the typed provider handle must be persistent"
    assert len(runtime.handle.requests) == 2
    assert runtime.handle.requests[0]["messages"][0]["role"] == "developer"
    assert runtime.handle.requests[0]["tools"][0]["name"] == "lookup"
    assert second.raw_response == first.raw_response


@pytest.mark.asyncio
async def test_stream_consumes_typed_events_without_sse_round_trip(monkeypatch):
    install_runtime(monkeypatch)
    provider = make_provider()

    chunks = [
        chunk
        async for chunk in provider.stream(
            [Message(role="user", content="hi")], model="deepseek-chat"
        )
    ]

    assert chunks[0].content == "he"
    assert chunks[1].metadata == {"reasoning_content": "think"}
    assert chunks[-1].is_final
    assert chunks[-1].stop_reason == "tool_calls"
    assert chunks[-1].tool_calls == [{"id": "c1", "name": "lookup", "arguments": {"q": 1}}]
    assert chunks[-1].usage == {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
        "cache_read_input_tokens": 4,
    }


@pytest.mark.asyncio
async def test_binding_failure_is_mapped_and_never_replayed(monkeypatch):
    error = RuntimeError(
        json.dumps(
            {
                "code": "rate_limited",
                "message": "slow down",
                "retryable": True,
                "http_status": 429,
            }
        )
    )
    install_runtime(monkeypatch, FakeTypedProvider(complete_error=error))
    provider = make_provider()

    with pytest.raises(ProviderRateLimitError):
        await provider.chat([Message(role="user", content="hi")], model="deepseek-chat")


def test_pilot_and_raw_bridge_symbols_are_gone():
    for obsolete in (
        "SandhiTransportUnavailable",
        "set_sandhi_transport_providers",
        "configure_from_settings",
        "sse_lines",
        "_binding_complete",
    ):
        assert not hasattr(st, obsolete)


def test_non_routine_usage_state_survives_victor_compatibility_mapping():
    assert st._usage_diagnostics(
        {
            "attempts": 3,
            "completeness": "final",
            "outcome": "success",
            "upstream_request_id": "up_1",
        }
    ) == {
        "attempts": 3,
        "completeness": "final",
        "outcome": "success",
        "upstream_request_id": "up_1",
    }
    assert (
        st._usage_diagnostics({"attempts": 1, "completeness": "final", "outcome": "success"})
        is None
    )


# =============================================================================
# Gateway mode (TD-0003 P3) — point the FFI handle at the Sandhi proxy with a vk.
# =============================================================================


def make_gateway_provider() -> DeepSeekProvider:
    return DeepSeekProvider(
        api_key="real-upstream-key",
        base_url="https://api.deepseek.com/v1",
        gateway={"url": "http://localhost:8600", "virtual_key": "vk_test_123"},
    )


@pytest.mark.asyncio
async def test_gateway_mode_points_ffi_handle_at_proxy_with_virtual_key(monkeypatch):
    runtime = install_runtime(monkeypatch)
    provider = make_gateway_provider()

    await provider.chat([Message(role="user", content="hi")], model="deepseek-chat")

    args, kwargs = runtime.calls[0]
    # The slug is preserved so the proxy still speaks the openai-compat dialect;
    # the virtual key replaces the credential; the proxy URL replaces the endpoint.
    assert args[:3] == ("deepseek", "deepseek-chat", "vk_test_123")
    assert kwargs["base_url"] == "http://localhost:8600"
    # sandhi >= 0.1.5 (victor's floor) accepts "bearer" family-wide as a no-op,
    # so gateway mode presents it unconditionally for the virtual key.
    assert kwargs["auth_scheme"] == "bearer"


@pytest.mark.asyncio
async def test_gateway_mode_preserves_protocol_alongside_overrides(monkeypatch):
    """A gateway-mode provider still selects its wire protocol (e.g. responses)."""
    runtime = install_runtime(monkeypatch)
    provider = make_gateway_provider()
    # An OAuth/responses provider carries _sandhi_protocol; gateway mode must not drop it.
    provider._sandhi_protocol = "chatgpt_responses"

    await provider.chat([Message(role="user", content="hi")], model="deepseek-chat")

    _, kwargs = runtime.calls[0]
    assert kwargs["protocol"] == "chatgpt_responses"
    assert kwargs["base_url"] == "http://localhost:8600"
    assert kwargs["auth_scheme"] == "bearer"


@pytest.mark.asyncio
async def test_gateway_mode_requests_bearer_regardless_of_auth_scheme_family(monkeypatch):
    """sandhi >= 0.1.5 accepts "bearer" family-wide, so gateway mode presents it
    unconditionally — with or without a per-family _sandhi_auth_scheme marker —
    and the virtual key rides Authorization."""
    runtime = install_runtime(monkeypatch)
    provider = make_gateway_provider()
    provider._sandhi_auth_scheme = "api_key"  # marks an auth-scheme family

    await provider.chat([Message(role="user", content="hi")], model="deepseek-chat")

    _, kwargs = runtime.calls[0]
    assert kwargs["auth_scheme"] == "bearer"


@pytest.mark.asyncio
async def test_gateway_mode_reuses_handle_across_calls(monkeypatch):
    runtime = install_runtime(monkeypatch)
    provider = make_gateway_provider()

    await provider.chat([Message(role="user", content="hi")], model="deepseek-chat")
    await provider.chat([Message(role="user", content="again")], model="deepseek-chat")

    # The gateway handle is cached just like a direct-mode handle (one FFI build).
    assert len(runtime.calls) == 1


@pytest.mark.asyncio
async def test_gateway_mode_missing_virtual_key_fails_closed(monkeypatch):
    install_runtime(monkeypatch)
    monkeypatch.delenv("SANDHI_GATEWAY_VIRTUAL_KEY_DEEPSEEK", raising=False)
    monkeypatch.delenv("SANDHI_GATEWAY_VIRTUAL_KEY", raising=False)
    provider = DeepSeekProvider(
        api_key="real-upstream-key",
        base_url="https://api.deepseek.com/v1",
        gateway={"url": "http://localhost:8600"},
    )

    with pytest.raises(ProviderConnectionError, match="virtual_key"):
        await provider.chat([Message(role="user", content="hi")], model="deepseek-chat")


@pytest.mark.asyncio
async def test_direct_mode_is_unchanged_when_gateway_not_configured(monkeypatch):
    """Regression: absent gateway leaves the provider in direct FFI mode."""
    runtime = install_runtime(monkeypatch)
    provider = DeepSeekProvider(api_key="k", base_url="https://api.deepseek.com/v1")

    await provider.chat([Message(role="user", content="hi")], model="deepseek-chat")

    args, kwargs = runtime.calls[0]
    assert args[:3] == ("deepseek", "deepseek-chat", "k")
    # No gateway override: auth_scheme is not forced to bearer.
    assert kwargs.get("auth_scheme") in (None, "api_key", "")
    assert kwargs["base_url"] == "https://api.deepseek.com/v1"


def test_resolve_provider_gateway_normalizes_block_and_unwraps_secret():
    from pydantic import SecretStr

    from victor.config.provider_config_registry import resolve_provider_gateway

    base: dict = {"gateway": {"url": "http://localhost:8600", "virtual_key": SecretStr("vk_s")}}
    resolve_provider_gateway(base, "deepseek")
    assert base["gateway"] == {"url": "http://localhost:8600", "virtual_key": "vk_s"}


def test_resolve_provider_gateway_env_fallback_per_provider_then_global(monkeypatch):
    from victor.config.provider_config_registry import resolve_provider_gateway

    base: dict = {"gateway": {"url": "http://localhost:8600"}}
    monkeypatch.setenv("SANDHI_GATEWAY_VIRTUAL_KEY_DEEPSEEK", "vk_per_provider")
    monkeypatch.setenv("SANDHI_GATEWAY_VIRTUAL_KEY", "vk_global")
    resolve_provider_gateway(base, "deepseek")
    assert base["gateway"]["virtual_key"] == "vk_per_provider"

    base = {"gateway": {"url": "http://localhost:8600"}}
    monkeypatch.delenv("SANDHI_GATEWAY_VIRTUAL_KEY_DEEPSEEK", raising=False)
    resolve_provider_gateway(base, "deepseek")
    assert base["gateway"]["virtual_key"] == "vk_global"


def test_resolve_provider_gateway_drops_block_without_url():
    from victor.config.provider_config_registry import resolve_provider_gateway

    base: dict = {"gateway": {"virtual_key": "vk_orphan"}, "api_key": "k"}
    resolve_provider_gateway(base, "deepseek")
    assert "gateway" not in base
    assert base["api_key"] == "k"


class TestWireContractHandshake:
    """One-time fail-soft handshake against the installed binding."""

    def setup_method(self):
        st._wire_contract_checked = False

    def teardown_method(self):
        st._wire_contract_checked = False

    def test_mismatch_warns_once(self, monkeypatch, caplog):
        fake_sg = SimpleNamespace(wire_contract_version=lambda: "2")
        import sys

        monkeypatch.setitem(sys.modules, "sandhi_gateway", fake_sg)
        with caplog.at_level("WARNING"):
            st._verify_wire_contract()
            st._verify_wire_contract()  # second call must be a no-op
        warnings = [r for r in caplog.records if "wire-contract mismatch" in r.getMessage()]
        assert len(warnings) == 1

    def test_matching_version_is_silent(self, monkeypatch, caplog):
        fake_sg = SimpleNamespace(wire_contract_version=lambda: "1")
        import sys

        monkeypatch.setitem(sys.modules, "sandhi_gateway", fake_sg)
        with caplog.at_level("WARNING"):
            st._verify_wire_contract()
        assert not any("wire-contract" in r.getMessage() for r in caplog.records)

    def test_old_binding_without_surface_is_silent(self, monkeypatch, caplog):
        fake_sg = SimpleNamespace()  # predates wire_contract_version
        import sys

        monkeypatch.setitem(sys.modules, "sandhi_gateway", fake_sg)
        with caplog.at_level("WARNING"):
            st._verify_wire_contract()
        assert not any("wire-contract" in r.getMessage() for r in caplog.records)


class TestUpstreamBodySurfacing:
    """details.upstream_body from ProviderErrorV1 must reach the surfaced message."""

    def _typed_error(self, details=None):
        import json as _json

        payload = {
            "code": "upstream_error",
            "message": "upstream status 400",
            "retryable": False,
            "http_status": 400,
            "provider": "moonshot",
        }
        if details is not None:
            payload["details"] = details
        return RuntimeError(_json.dumps(payload))

    def test_upstream_body_appended_to_message(self):
        body = '{"error":{"message":"tool call id call_9 not found"}}'
        err = st.map_sandhi_error(self._typed_error({"upstream_body": body}), "moonshot", 30.0)
        assert "tool call id call_9 not found" in str(err)

    def test_no_details_keeps_prior_message(self):
        err = st.map_sandhi_error(self._typed_error(), "moonshot", 30.0)
        assert "upstream status 400" in str(err)
        assert "upstream body" not in str(err)

    def test_body_already_in_message_not_duplicated(self):
        body = "duplicate snippet content that is already present"
        payload_err = self._typed_error({"upstream_body": body})
        import json as _json

        parsed = _json.loads(str(payload_err))
        parsed["message"] = f"upstream status 400: {body}"
        err = st.map_sandhi_error(RuntimeError(_json.dumps(parsed)), "moonshot", 30.0)
        assert str(err).count(body) == 1


# =============================================================================
# Native body optional (foundations strategy F1) — the typed path must behave
# identically when `extensions` is absent; native-only usage fields surface
# through metadata["sandhi_usage"], never through the body.
# =============================================================================


def _typed_payload(**overrides):
    payload = {
        "schema_version": "1",
        "id": "r9",
        "model": "deepseek-chat",
        "output": {"content": "hello", "tool_calls": []},
        "finish_reason": "stop",
        "usage": {
            "tokens_in": 6,
            "tokens_out": 5,
            "cache_creation_tokens": 0,
            "cache_read_tokens": 4,
            "completeness": "final",
            "attempts": 1,
            "outcome": "success",
        },
    }
    payload.update(overrides)
    return payload


def test_extensions_absent_is_behavior_identical(monkeypatch):
    """With no native body, usage/metadata must match the with-body result."""
    install_runtime(monkeypatch)
    provider = make_provider()

    with_native = provider._completion_from_typed(
        _typed_payload(
            extensions={
                "openai": {
                    "id": "r9",
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                }
            }
        ),
        "deepseek-chat",
    )
    without_native = provider._completion_from_typed(_typed_payload(), "deepseek-chat")

    assert without_native.usage == with_native.usage
    assert without_native.usage == {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
        "cache_read_input_tokens": 4,
    }
    assert without_native.metadata == with_native.metadata is None
    assert without_native.content == "hello"
    # Debug fallback: with no native body, raw_response is the typed document.
    assert without_native.raw_response["schema_version"] == "1"


def test_native_only_usage_fields_surface_as_diagnostics(monkeypatch):
    """cache-miss/cost exist only in native bodies; the transport boundary
    extracts them into metadata['sandhi_usage'] so the runtime never reads the
    body (unblocks sandhi G8 native-body gating)."""
    install_runtime(monkeypatch)
    provider = make_provider()

    response = provider._completion_from_typed(
        _typed_payload(
            extensions={
                "openai": {
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15,
                        "prompt_cache_miss_tokens": 7,
                        "cost_in_usd_ticks": 123,
                    }
                }
            }
        ),
        "deepseek-chat",
    )

    assert response.metadata["sandhi_usage"] == {
        "cache_miss_tokens": 7,
        "cost_in_usd_ticks": 123,
    }


def test_native_only_extractor_ignores_junk():
    assert st._native_only_usage(None) == {}
    assert st._native_only_usage("usage") == {}
    assert st._native_only_usage({"prompt_cache_miss_tokens": "x", "cost_in_usd_ticks": None}) == {}
    assert st._native_only_usage({"prompt_cache_miss_tokens": 0, "cost_in_usd_ticks": 0}) == {}


class TestTypedErrorClassFastPath:
    """sandhi>=0.1.3 SandhiProviderError: classification without parse dependence."""

    def test_unparseable_typed_instance_stays_provider_error(self, monkeypatch):
        class FakeSandhiProviderError(RuntimeError):
            pass

        monkeypatch.setattr(st, "_SANDHI_PROVIDER_ERROR_CLS", FakeSandhiProviderError)
        err = st.map_sandhi_error(
            FakeSandhiProviderError("truncated payload not json"), "moonshot", 30.0
        )
        from victor.providers.base import ProviderConnectionError, ProviderError

        assert isinstance(err, ProviderError)
        assert not isinstance(err, ProviderConnectionError)
        assert "truncated payload not json" in str(err)

    def test_plain_unparseable_runtime_error_stays_binding_failure(self, monkeypatch):
        class FakeSandhiProviderError(RuntimeError):
            pass

        monkeypatch.setattr(st, "_SANDHI_PROVIDER_ERROR_CLS", FakeSandhiProviderError)
        err = st.map_sandhi_error(RuntimeError("segfault in binding"), "moonshot", 30.0)
        from victor.providers.base import ProviderConnectionError

        assert isinstance(err, ProviderConnectionError)
        assert "binding failure" in str(err)


# =============================================================================
# W3a soak (sandhi#90): Victor opts out of the native-body echo by default.
# =============================================================================


def test_typed_request_opts_out_of_native_body_by_default():
    request = st._typed_request_from_openai_payload(
        {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    )
    assert request["include_native_response"] is False


def test_typed_request_honors_native_body_opt_in(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(
        "victor.config.settings.get_settings",
        lambda: SimpleNamespace(provider=SimpleNamespace(sandhi_include_native_response=True)),
    )
    request = st._typed_request_from_openai_payload(
        {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    )
    # Opt-in restores sandhi's default (include): the field stays off the wire.
    assert "include_native_response" not in request


# =============================================================================
# W3b: wire-truth latency flows from the typed usage surface into diagnostics.
# =============================================================================


def test_latency_fields_extracted_from_neutral_usage():
    assert st._latency_fields({"duration_ms": 120, "time_to_first_token_ms": 45}) == {
        "duration_ms": 120,
        "time_to_first_token_ms": 45,
    }
    # Tolerant-absent: pre-W3b runtimes carry neither field.
    assert st._latency_fields({"tokens_in": 1}) == {}
    assert st._latency_fields(None) == {}
    assert st._latency_fields({"duration_ms": "x", "time_to_first_token_ms": -1}) == {}


def test_completion_surfaces_wire_latency_in_diagnostics(monkeypatch):
    install_runtime(monkeypatch)
    provider = make_provider()
    payload = _typed_payload()
    payload["usage"]["duration_ms"] = 120
    response = provider._completion_from_typed(payload, "deepseek-chat")
    assert response.metadata["sandhi_usage"]["duration_ms"] == 120


# =============================================================================
# W3c: minor-version handshake — victor reads the installed contract minor.
# =============================================================================


def test_installed_minor_defaults_to_zero_for_old_bindings(monkeypatch):
    import types

    fake_sg = types.SimpleNamespace(wire_contract_version=lambda: "1")
    monkeypatch.setitem(sys.modules, "sandhi_gateway", fake_sg)
    monkeypatch.setattr(st, "_wire_contract_checked", False)
    monkeypatch.setattr(st, "_installed_contract_minor", 0)
    assert st.installed_chat_contract_minor() == 0


def test_installed_minor_read_from_binding(monkeypatch):
    import types

    fake_sg = types.SimpleNamespace(
        wire_contract_version=lambda: "1", chat_contract_minor=lambda: 3
    )
    monkeypatch.setitem(sys.modules, "sandhi_gateway", fake_sg)
    monkeypatch.setattr(st, "_wire_contract_checked", False)
    monkeypatch.setattr(st, "_installed_contract_minor", 0)
    assert st.installed_chat_contract_minor() == 3


def test_handshake_accepts_current_known_minor(monkeypatch, caplog):
    """The 0.1.5 floor speaks contract minor 6 (UsageV2.basis + run cost tree);
    the handshake reads it exactly and does not warn it is 'ahead'."""
    import types

    assert st.KNOWN_CONTRACT_MINOR == 6
    fake_sg = types.SimpleNamespace(
        wire_contract_version=lambda: "1", chat_contract_minor=lambda: 6
    )
    monkeypatch.setitem(sys.modules, "sandhi_gateway", fake_sg)
    monkeypatch.setattr(st, "_wire_contract_checked", False)
    monkeypatch.setattr(st, "_installed_contract_minor", 0)
    with caplog.at_level("INFO"):
        assert st.installed_chat_contract_minor() == 6
    assert not any("ahead of victor" in r.getMessage() for r in caplog.records)


def test_handshake_tolerates_newer_minor_forward_compat(monkeypatch, caplog):
    """installed_minor > KNOWN stays valid forward-compat: victor reads the
    newer minor, logs an informational 'ahead' note, and keeps transporting
    (newer additive fields are simply ignored until victor catches up)."""
    import types

    fake_sg = types.SimpleNamespace(
        wire_contract_version=lambda: "1",
        chat_contract_minor=lambda: st.KNOWN_CONTRACT_MINOR + 1,
    )
    monkeypatch.setitem(sys.modules, "sandhi_gateway", fake_sg)
    monkeypatch.setattr(st, "_wire_contract_checked", False)
    monkeypatch.setattr(st, "_installed_contract_minor", 0)
    with caplog.at_level("INFO"):
        assert st.installed_chat_contract_minor() == st.KNOWN_CONTRACT_MINOR + 1
    assert any("ahead of victor" in r.getMessage() for r in caplog.records)


def test_installed_binding_meets_victor_floor_when_export_exists():
    """G-ledger floor pin: once the installed sandhi-gateway exports
    chat_contract_minor, it must be >= victor's known minor (conditional so
    the pinned pre-W3c binding keeps passing until the next pin bump)."""
    sg = pytest.importorskip("sandhi_gateway")
    minor_fn = getattr(sg, "chat_contract_minor", None)
    if not callable(minor_fn):
        pytest.skip("installed sandhi-gateway predates chat_contract_minor")
    assert int(minor_fn()) >= st.KNOWN_CONTRACT_MINOR


# =============================================================================
# W3d/G7: codec purity — promoted typed fields, family-gated bucket, no leak.
# =============================================================================


def _openai_payload(**extra):
    payload = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    payload.update(extra)
    return payload


def test_reasoning_effort_promoted_to_typed_field(monkeypatch):
    monkeypatch.setattr(st, "_wire_contract_checked", True)
    monkeypatch.setattr(st, "_installed_contract_minor", 4)
    request = st._typed_request_from_openai_payload(_openai_payload(reasoning_effort="high"))
    assert request["reasoning_effort"] == "high"
    # At minor >= 4 the extensions copy is dropped (no dual-write).
    assert "reasoning_effort" not in request.get("extensions", {}).get("openai", {})


def test_thinking_normalized_from_victor_shape(monkeypatch):
    monkeypatch.setattr(st, "_wire_contract_checked", True)
    monkeypatch.setattr(st, "_installed_contract_minor", 4)
    request = st._typed_request_from_openai_payload(
        _openai_payload(thinking={"type": "enabled", "budget_tokens": 2048})
    )
    assert request["thinking"] == {"enabled": True, "budget_tokens": 2048}


def test_dual_write_keeps_extensions_copy_below_minor_4(monkeypatch):
    monkeypatch.setattr(st, "_wire_contract_checked", True)
    monkeypatch.setattr(st, "_installed_contract_minor", 3)  # pinned pre-W3d runtime
    request = st._typed_request_from_openai_payload(_openai_payload(reasoning_effort="high"))
    assert request["reasoning_effort"] == "high"  # typed field always emitted
    # ...and the extensions copy is retained so the old runtime still sees it.
    assert request["extensions"]["openai"]["reasoning_effort"] == "high"


def test_internal_kwargs_never_reach_extensions(monkeypatch):
    monkeypatch.setattr(st, "_wire_contract_checked", True)
    monkeypatch.setattr(st, "_installed_contract_minor", 4)
    request = st._typed_request_from_openai_payload(
        _openai_payload(execution_mode="fast", topology_action="escalate", top_p=0.9)
    )
    native = request.get("extensions", {}).get("openai", {})
    assert "execution_mode" not in native
    assert "topology_action" not in native
    # A real (non-internal) passthrough param still rides extensions.
    assert native.get("top_p") == 0.9


def test_normalize_thinking_shapes():
    assert st._normalize_thinking(True) == {"enabled": True}
    assert st._normalize_thinking({"type": "disabled"}) == {"enabled": False}
    assert st._normalize_thinking({"enabled": True, "budget_tokens": 100}) == {
        "enabled": True,
        "budget_tokens": 100,
    }
    assert st._normalize_thinking("nonsense") is None


def test_neutral_mixin_drops_bucket_for_native_encoder_families(monkeypatch):
    """W3d/G7 D3: gemini/cohere/ollama encoders clone extensions[<slug>] as the
    base body, so an OpenAI-shaped bucket must NOT be re-labeled to their key."""
    from victor.providers.base import Message

    monkeypatch.setattr(st, "_wire_contract_checked", True)
    monkeypatch.setattr(st, "_installed_contract_minor", 4)

    class _Stub(st.SandhiNeutralProviderMixin):
        def __init__(self, slug):
            self._slug = slug

        def _sandhi_slug(self):
            return self._slug

    msgs = [Message(role="user", content="hi")]
    # Gemini: native encoder → bucket dropped even with a passthrough param.
    gemini_req = _Stub("gemini")._neutral_request(msgs, "g", 0.7, 100, None, top_p=0.9)
    assert "extensions" not in gemini_req
    # An openai-compat local (lmstudio): bucket re-labeled, not dropped.
    local_req = _Stub("lmstudio")._neutral_request(msgs, "m", 0.7, 100, None, top_p=0.9)
    assert local_req.get("extensions", {}).get("lmstudio", {}).get("top_p") == 0.9
