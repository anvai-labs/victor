"""Golden typed-runtime determinism battery using the real Sandhi binding.

Two independent handles hit equivalent localhost fixture servers with identical inputs; the
contract is byte-level request determinism and semantic response parity. Fixture bodies mirror
the sandhi recorded corpus (commit 3102dd8) plus tool-call shapes the corpus lacks.
"""

from __future__ import annotations

import json
from copy import deepcopy

import pytest

pytest.importorskip("sandhi_gateway", reason="requires the victor[sandhi] extra")

from victor.providers.base import (
    CompletionResponse,
    Message,
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)

pytestmark = pytest.mark.integration

COMPLETE_BODY = {
    "id": "chatcmpl-corpus",
    "object": "chat.completion",
    "model": "deepseek-chat",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello, world"},
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 1000,
        "completion_tokens": 250,
        "total_tokens": 1250,
        "prompt_tokens_details": {"cached_tokens": 800},
    },
}

TOOL_CALL_BODY = {
    "id": "chatcmpl-tools",
    "object": "chat.completion",
    "model": "deepseek-chat",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Paris"}',
                        },
                    }
                ],
            },
            "finish_reason": "tool_calls",
        }
    ],
    "usage": {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70},
}

STREAM_SSE = (
    'data: {"id":"c","object":"chat.completion.chunk","model":"deepseek-chat",'
    '"choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}],"usage":null}\n\n'
    'data: {"id":"c","object":"chat.completion.chunk","model":"deepseek-chat",'
    '"choices":[{"index":0,"delta":{"content":", world"},"finish_reason":null}],"usage":null}\n\n'
    'data: {"id":"c","object":"chat.completion.chunk","model":"deepseek-chat",'
    '"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],'
    '"usage":{"prompt_tokens":1000,"completion_tokens":250,"total_tokens":1250,'
    '"prompt_tokens_details":{"cached_tokens":800}}}\n\n'
    "data: [DONE]\n\n"
).encode()

MESSAGES = [Message(role="user", content="hi there")]


def semantic_response(response):
    """Compare all response fields except independently measured latency values.

    Preserve latency field presence and validate its shape. Only these two
    diagnostic locations are normalized; token accounting, other metadata,
    tool arguments and request bodies must still match exactly.
    """
    result = deepcopy(response.model_dump())
    for container, field in (("raw_response", "usage"), ("metadata", "sandhi_usage")):
        usage = (result.get(container) or {}).get(field) or {}
        for key in ("duration_ms", "time_to_first_token_ms"):
            if key in usage:
                assert type(usage[key]) is int and usage[key] >= 0
                usage[key] = 0
    return result


class TestSemanticResponseComparison:
    def test_different_timings_match_without_mutating_responses(self):
        first = CompletionResponse(
            content="same",
            raw_response={"usage": {"duration_ms": 1, "time_to_first_token_ms": 0}},
            metadata={"sandhi_usage": {"duration_ms": 1, "time_to_first_token_ms": 0}},
        )
        second = first.model_copy(deep=True)
        second.raw_response["usage"].update(duration_ms=7, time_to_first_token_ms=2)
        second.metadata["sandhi_usage"].update(duration_ms=7, time_to_first_token_ms=2)
        before = second.model_dump()
        assert first.model_dump() != before
        assert semantic_response(first) == semantic_response(second)
        assert second.model_dump() == before

    @pytest.mark.parametrize(
        "field,value",
        [
            ("content", "different"),
            ("usage", {"prompt_tokens": 9}),
            ("tool_calls", [{"name": "changed_tool", "arguments": {"duration_ms": 9}}]),
            ("raw_response", {"usage": {"tokens_in": 9}}),
            ("metadata", {"sandhi_usage": {"attempts": 2}}),
            ("metadata", {"sandhi_usage": {"duration_ms": 0}}),
        ],
    )
    def test_semantic_or_latency_presence_changes_still_differ(self, field, value):
        first = CompletionResponse(content="same")
        second = first.model_copy(update={field: value}, deep=True)
        assert semantic_response(first) != semantic_response(second)

    @pytest.mark.parametrize("value", [-1, True, "1", None])
    @pytest.mark.parametrize(
        "container,field", [("raw_response", "usage"), ("metadata", "sandhi_usage")]
    )
    def test_invalid_latency_is_rejected(self, value, container, field):
        response = CompletionResponse(
            content="same", **{container: {field: {"duration_ms": value}}}
        )
        with pytest.raises(AssertionError):
            semantic_response(response)


async def run_chat(provider):
    return await provider.chat(MESSAGES, model="deepseek-chat", temperature=0.2, max_tokens=64)


async def run_stream(provider):
    return [
        chunk
        async for chunk in provider.stream(
            MESSAGES, model="deepseek-chat", temperature=0.2, max_tokens=64
        )
    ]


class TestCompletionParity:
    async def test_completion_response_and_request_parity(self, fixture_server, make_pair):
        native_srv = fixture_server(body=json.dumps(COMPLETE_BODY).encode())
        sandhi_srv = fixture_server(body=json.dumps(COMPLETE_BODY).encode(), delay_secs=0.025)
        native, _ = make_pair(native_srv.url)
        _, sandhi = make_pair(sandhi_srv.url)

        native_resp = await run_chat(native)
        sandhi_resp = await run_chat(sandhi)

        # (a) semantic response parity — wall-clock latency is independently measured.
        assert semantic_response(native_resp) == semantic_response(sandhi_resp)
        assert sandhi_resp.content == "Hello, world"
        assert sandhi_resp.usage["prompt_tokens"] == 1000
        assert sandhi_resp.usage["cache_read_input_tokens"] == 800

        # (c) request-side parity: identical JSON body and auth header
        assert len(native_srv.requests) == 1 and len(sandhi_srv.requests) == 1
        native_req, sandhi_req = native_srv.requests[0], sandhi_srv.requests[0]
        assert json.loads(native_req.body) == json.loads(sandhi_req.body)
        assert native_req.headers.get("authorization") == sandhi_req.headers.get("authorization")
        assert native_req.path == sandhi_req.path == "/v1/chat/completions"

    async def test_tool_call_parity(self, fixture_server, make_pair):
        native_srv = fixture_server(body=json.dumps(TOOL_CALL_BODY).encode())
        sandhi_srv = fixture_server(body=json.dumps(TOOL_CALL_BODY).encode(), delay_secs=0.025)
        native, _ = make_pair(native_srv.url)
        _, sandhi = make_pair(sandhi_srv.url)

        native_resp = await run_chat(native)
        sandhi_resp = await run_chat(sandhi)
        assert semantic_response(native_resp) == semantic_response(sandhi_resp)
        assert sandhi_resp.tool_calls == [
            {"id": "call_1", "name": "get_weather", "arguments": {"city": "Paris"}}
        ]

    async def test_no_double_request(self, fixture_server, make_pair):
        # Retries are explicitly disabled by this fixture, so there is exactly one
        # upstream POST per chat.
        srv = fixture_server(body=json.dumps(COMPLETE_BODY).encode())
        _, sandhi = make_pair(srv.url)
        await run_chat(sandhi)
        assert len(srv.requests) == 1


class TestStreamParity:
    async def test_stream_chunk_sequence_parity(self, fixture_server, make_pair):
        native_srv = fixture_server(body=STREAM_SSE, content_type="text/event-stream")
        sandhi_srv = fixture_server(body=STREAM_SSE, content_type="text/event-stream")
        native, _ = make_pair(native_srv.url)
        _, sandhi = make_pair(sandhi_srv.url)

        native_chunks = await run_stream(native)
        sandhi_chunks = await run_stream(sandhi)

        assert [semantic_response(c) for c in native_chunks] == [
            semantic_response(c) for c in sandhi_chunks
        ]
        assert "".join(c.content for c in sandhi_chunks) == "Hello, world"
        final = sandhi_chunks[-1]
        assert final.is_final
        # Terminal usage is parsed at the Sandhi wire boundary.
        usage_chunks = [c for c in sandhi_chunks if c.usage]
        assert usage_chunks and usage_chunks[-1].usage["cache_read_input_tokens"] == 800

        # Both handles use Sandhi, which injects include_usage at the wire boundary.
        native_body = json.loads(native_srv.requests[0].body)
        sandhi_body = json.loads(sandhi_srv.requests[0].body)
        assert native_body.pop("stream_options", None) == {"include_usage": True}
        assert sandhi_body.pop("stream_options", None) == {"include_usage": True}
        assert native_body == sandhi_body


class TestErrorParity:
    @pytest.mark.parametrize(
        "status,expected",
        [
            (401, ProviderAuthError),
            (429, ProviderRateLimitError),
            (500, ProviderError),
        ],
    )
    async def test_error_class_parity(self, fixture_server, make_pair, status, expected):
        body = json.dumps({"error": {"message": "boom"}}).encode()
        sandhi_srv = fixture_server(status=status, body=body)
        _, sandhi = make_pair(sandhi_srv.url)

        with pytest.raises(expected):
            await run_chat(sandhi)
        assert not hasattr(sandhi, "_sandhi_demoted")
        assert len(sandhi_srv.requests) == 1

    async def test_timeout_surfaces_within_bound(self, fixture_server, make_pair):
        srv = fixture_server(body=json.dumps(COMPLETE_BODY).encode(), delay_secs=8.0)
        _, sandhi = make_pair(srv.url, timeout=1)
        import time

        start = time.monotonic()
        with pytest.raises(ProviderTimeoutError):
            await run_chat(sandhi)
        assert time.monotonic() - start < 7.0, "timeout must fire well before the 8s delay"
        assert not hasattr(sandhi, "_sandhi_demoted")
