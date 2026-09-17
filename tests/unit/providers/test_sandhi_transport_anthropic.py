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
