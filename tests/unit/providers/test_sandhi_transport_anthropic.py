"""Anthropic coverage for the direct typed Sandhi path."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import victor.providers.sandhi_transport as st
from victor.providers.anthropic_provider import AnthropicProvider
from victor.providers.base import Message, ToolDefinition


class FakeAnthropicHandle:
    def __init__(self) -> None:
        self.requests: list[dict] = []

    async def complete_json(self, request_json: str, wire_headers_json=None) -> str:
        self.requests.append(json.loads(request_json))
        return json.dumps(
            {
                "schema_version": "1",
                "id": "msg_1",
                "model": "claude-test",
                "output": {
                    "content": "answer",
                    "tool_calls": [{"id": "t1", "name": "lookup", "arguments": '{"q":1}'}],
                },
                "finish_reason": "tool_calls",
                "usage": {
                    "tokens_in": 10,
                    "tokens_out": 5,
                    "cache_creation_tokens": 3,
                    "cache_read_tokens": 7,
                    "completeness": "final",
                    "attempts": 1,
                },
                "extensions": {
                    "reasoning": "considered",
                    "anthropic": {"id": "msg_1", "usage": {"input_tokens": 10}},
                },
            }
        )

    def stream_json(self, request_json: str, wire_headers_json=None):
        self.requests.append(json.loads(request_json))

        async def events():
            yield json.dumps({"event": "response_start", "id": "msg_2", "model": "claude-test"})
            yield json.dumps({"event": "reasoning_delta", "delta": "think"})
            yield json.dumps({"event": "text_delta", "delta": "answer"})
            yield json.dumps({"event": "finish", "reason": "stop"})
            yield json.dumps(
                {
                    "event": "usage",
                    "usage": {
                        "tokens_in": 10,
                        "tokens_out": 5,
                        "cache_creation_tokens": 3,
                        "cache_read_tokens": 7,
                    },
                }
            )

        return events()


class FakeRuntime:
    def __init__(self) -> None:
        self.handle = FakeAnthropicHandle()
        self.calls: list[tuple] = []

    def provider(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.handle


@pytest.fixture
def runtime(monkeypatch) -> FakeRuntime:
    value = FakeRuntime()
    monkeypatch.setattr(st, "_sg", SimpleNamespace(ProviderRuntime=lambda: value))
    return value


def make_provider() -> st.SandhiAnthropicProvider:
    return st.SandhiAnthropicProvider(
        api_key="k", base_url="https://api.anthropic.test", max_retries=2
    )


def test_resolver_uses_typed_anthropic(runtime):
    assert (
        st.resolve_transport_class("anthropic", AnthropicProvider, {}) is st.SandhiAnthropicProvider
    )


@pytest.mark.asyncio
async def test_complete_preserves_anthropic_policy_and_consumes_typed_response(runtime):
    provider = make_provider()
    tools = [
        ToolDefinition(
            name="lookup",
            description="Lookup",
            parameters={"type": "object"},
            schema_level="full",
        )
    ]
    response = await provider.chat(
        [Message(role="system", content="policy"), Message(role="user", content="hello")],
        model="claude-test",
        tools=tools,
    )

    assert response.content == "answer"
    assert response.metadata == {"reasoning_content": "considered"}
    assert response.tool_calls == [{"id": "t1", "name": "lookup", "arguments": {"q": 1}}]
    assert response.usage == {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
        "cache_creation_input_tokens": 3,
        "cache_read_input_tokens": 7,
    }

    request = runtime.handle.requests[0]
    assert request["messages"][0] == {"role": "system", "content": "policy"}
    native = request["extensions"]["anthropic"]
    assert native["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert native["tools"][0]["cache_control"] == {"type": "ephemeral"}
    assert runtime.calls[0][1]["base_url"] == "https://api.anthropic.test"
    assert runtime.calls[0][1]["auth_scheme"] == "api_key"


@pytest.mark.asyncio
async def test_oauth_selects_bearer_auth_before_constructing_handle(runtime):
    with patch("victor.providers.anthropic_provider.OAuthTokenManager") as manager_cls:
        manager = MagicMock()
        manager._load_cached.return_value = SimpleNamespace(
            access_token="cached-oauth", is_expired=False
        )
        manager.get_valid_token = AsyncMock(return_value="fresh-oauth")
        manager_cls.return_value = manager
        provider = st.SandhiAnthropicProvider(auth_mode="oauth")

    await provider.chat([Message(role="user", content="hello")], model="claude-test")

    args, kwargs = runtime.calls[0]
    assert args[:3] == ("anthropic", "claude-test", "fresh-oauth")
    assert kwargs["auth_scheme"] == "bearer"


@pytest.mark.asyncio
async def test_stream_uses_typed_anthropic_events(runtime):
    provider = make_provider()
    chunks = [
        chunk
        async for chunk in provider.stream(
            [Message(role="user", content="hello")], model="claude-test"
        )
    ]
    assert chunks[0].metadata == {"reasoning_content": "think"}
    assert chunks[1].content == "answer"
    assert chunks[-1].is_final and chunks[-1].stop_reason == "stop"
    assert chunks[-1].usage == {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
        "cache_creation_input_tokens": 3,
        "cache_read_input_tokens": 7,
    }


def test_anthropic_request_attaches_cache_markers_to_typed_extensions():
    """FEP-0011 prompt-cache markers ride ONLY in extensions["anthropic"].

    Sandhi's anthropic encoder re-encodes messages/tools from the typed fields
    and merges ``cache_control`` exclusively from the native bucket (system
    blocks by index, tools by name). The typed request must therefore carry the
    full native body, markers intact, on every call.
    """
    provider = make_provider()
    request = provider._anthropic_request(
        [Message(role="system", content="policy"), Message(role="user", content="hello")],
        model="claude-test",
        temperature=0.3,
        max_tokens=1024,
        tools=[
            ToolDefinition(
                name="lookup",
                description="Lookup",
                parameters={"type": "object"},
                schema_level="full",
            )
        ],
    )
    native = request["extensions"]["anthropic"]
    assert native["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert native["tools"][0]["cache_control"] == {"type": "ephemeral"}
    # The typed channel stays consistent with the native policy decision.
    assert request["temperature"] == 0.3
    assert request["messages"][0]["role"] == "system"


def test_anthropic_request_mirrors_native_temperature_policy_under_thinking():
    """Temperature+thinking policy is decided once, in _build_request_params.

    The typed request must mirror the native body's temperature decision
    instead of recomputing the thinking shape — the two checks could otherwise
    drift apart and resurrect the 400 the guard exists to prevent.
    """
    provider = make_provider()
    request = provider._anthropic_request(
        [Message(role="user", content="hello")],
        model="claude-test",
        temperature=0.7,
        max_tokens=1024,
        tools=None,
        thinking={"type": "enabled", "budget_tokens": 2048},
    )
    native = request["extensions"]["anthropic"]
    assert "temperature" not in native
    assert "temperature" not in request
    assert native["thinking"] == {"type": "enabled", "budget_tokens": 2048}


def test_anthropic_request_guard_fires_when_native_drops_cache_markers(monkeypatch):
    """A _build_request_params regression must fail loudly, not silently decache."""

    def markerless(messages, *, model, temperature, max_tokens, tools=None, **kwargs):
        return {
            "model": model,
            "messages": [],
            "max_tokens": max_tokens,
            "temperature": temperature,
            # system/tools present but WITHOUT cache_control — the regression
            "system": [{"type": "text", "text": "policy"}],
            "tools": [{"name": "lookup", "description": "d", "input_schema": {}}],
        }

    provider = make_provider()
    monkeypatch.setattr(provider, "_build_request_params", markerless)
    with pytest.raises(RuntimeError, match="prompt-cache markers missing"):
        provider._anthropic_request(
            [Message(role="system", content="policy"), Message(role="user", content="hi")],
            model="claude-test",
            temperature=0.3,
            max_tokens=1024,
            tools=[ToolDefinition(name="lookup", description="d", parameters={"type": "object"})],
        )


def test_cache_channel_guard_fires_when_extensions_severed():
    """Markers in the native body but not on the typed request = severed channel."""
    native = {
        "model": "claude-test",
        "messages": [],
        "max_tokens": 100,
        "system": [{"type": "text", "text": "policy", "cache_control": {"type": "ephemeral"}}],
    }
    severed = {"schema_version": "1", "model": "claude-test", "messages": [], "extensions": {}}
    with pytest.raises(RuntimeError, match="prompt-cache markers severed"):
        st._assert_anthropic_cache_channel(native, severed, supports_caching=True)


def test_cache_channel_guard_passes_when_caching_unsupported_or_nothing_cacheable():
    native_plain = {"model": "m", "messages": [], "max_tokens": 100}
    request_plain = {"schema_version": "1", "model": "m", "messages": []}
    # No system block / no tools: nothing to cache, no markers required.
    st._assert_anthropic_cache_channel(native_plain, request_plain, supports_caching=True)
    # Provider opts out of caching: markers never required.
    markerless = {
        "model": "m",
        "messages": [],
        "max_tokens": 100,
        "system": [{"type": "text", "text": "policy"}],
    }
    st._assert_anthropic_cache_channel(markerless, request_plain, supports_caching=False)


def test_anthropic_request_guard_rejects_temperature_alongside_thinking(monkeypatch):
    """Belt-and-suspenders mirror of sandhi's encoder guard, at the policy layer."""

    def contradictory(messages, *, model, temperature, max_tokens, tools=None, **kwargs):
        return {
            "model": model,
            "messages": [],
            "max_tokens": max_tokens,
            "temperature": temperature,  # BOTH present — the regression
            "thinking": {"type": "enabled", "budget_tokens": 2048},
        }

    provider = make_provider()
    monkeypatch.setattr(provider, "_build_request_params", contradictory)
    with pytest.raises(RuntimeError, match="temperature.*thinking|thinking.*temperature"):
        provider._anthropic_request(
            [Message(role="user", content="hello")],
            model="claude-test",
            temperature=0.7,
            max_tokens=1024,
            tools=None,
            thinking={"type": "enabled", "budget_tokens": 2048},
        )


@pytest.mark.asyncio
async def test_stream_assembles_parallel_tool_calls(runtime):
    """Streaming tool-call events assemble into the final chunk's tool_calls.

    Relocated from test_anthropic_stream_tool_use, which exercised the deleted
    native SDK stream. The Sandhi typed stream owns tool assembly now.
    """

    async def tool_events():
        yield json.dumps(
            {"event": "tool_call_start", "index": 0, "id": "call_1", "name": "dummy_tool"}
        )
        yield json.dumps(
            {"event": "tool_call_arguments_delta", "index": 0, "delta": '{"echo": "hi"}'}
        )
        yield json.dumps({"event": "tool_call_start", "index": 1, "id": "call_2", "name": "other"})
        yield json.dumps({"event": "tool_call_arguments_delta", "index": 1, "delta": '{"x": 2}'})
        yield json.dumps({"event": "finish", "reason": "tool_calls"})

    runtime.handle.stream_json = lambda request_json, wire_headers_json=None: tool_events()
    provider = make_provider()
    chunks = [
        chunk
        async for chunk in provider.stream(
            [Message(role="user", content="hi")],
            model="claude-test",
            tools=[
                ToolDefinition(name="dummy_tool", description="d", parameters={"type": "object"})
            ],
        )
    ]

    final = chunks[-1]
    assert final.is_final
    assert final.stop_reason == "tool_calls"
    assert final.tool_calls == [
        {"id": "call_1", "name": "dummy_tool", "arguments": {"echo": "hi"}},
        {"id": "call_2", "name": "other", "arguments": {"x": 2}},
    ]
