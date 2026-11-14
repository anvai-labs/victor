"""Anthropic transport battery against the real sandhi binding (T6).

Native Anthropic SDK transport has been removed (TD-0002): Sandhi owns transport, so
there is no native arm to compare against. These tests exercise the Sandhi typed variant
(``SandhiAnthropicProvider``) end-to-end against a localhost fixture server, asserting the
request shape (path, auth header, JSON body) and the semantic response the binding returns.
The complete body mirrors the sandhi recorded corpus
(``tests/unit/providers/fixtures/sandhi_usage/anthropic/complete_cache_split.json``)
plus a crafted tool_use shape the corpus lacks.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("sandhi_gateway", reason="requires the victor[sandhi] extra")

from victor.providers.base import (
    Message,
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
    ToolDefinition,
)

pytestmark = pytest.mark.integration

_CORPUS_PATH = (
    Path(__file__).parents[3]
    / "unit"
    / "providers"
    / "fixtures"
    / "sandhi_usage"
    / "anthropic"
    / "complete_cache_split.json"
)
COMPLETE_BODY = json.loads(_CORPUS_PATH.read_text())

TOOL_USE_BODY = {
    "id": "msg_01Tools",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-4-6",
    "content": [
        {
            "type": "tool_use",
            "id": "toolu_01",
            "name": "get_weather",
            "input": {"city": "Paris"},
        }
    ],
    "stop_reason": "tool_use",
    "stop_sequence": None,
    "usage": {
        "input_tokens": 50,
        "output_tokens": 20,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    },
}

MESSAGES = [
    Message(role="system", content="You are terse."),
    Message(role="user", content="hi there"),
]

TOOLS = [
    ToolDefinition(
        name="get_weather",
        description="Get the weather for a city",
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    )
]


@pytest.fixture
def make_anthropic_pair():
    """Factory fixture: (native, sandhi-backed) Anthropic providers for a fixture server.

    The binding derives ``{base_url}/v1/messages`` for slug ``anthropic``; the native SDK
    resolves the same path from its ``base_url`` — both point at the bare server root.
    ``max_retries=0`` disables retries in both transports so each fixture response maps
    to exactly one request in the wire-parity assertions.
    """

    def _make(server_url: str, timeout: int = 30):
        from victor.providers.anthropic_provider import AnthropicProvider
        from victor.providers.sandhi_transport import SandhiAnthropicProvider

        kwargs = {
            "api_key": "parity-key",
            "base_url": server_url,
            "timeout": timeout,
            "max_retries": 0,
        }
        return AnthropicProvider(**kwargs), SandhiAnthropicProvider(**kwargs)

    return _make


async def run_chat(provider, tools=None):
    return await provider.chat(
        MESSAGES, model="claude-sonnet-4-6", temperature=0.2, max_tokens=64, tools=tools
    )


class TestCompletionParity:
    async def test_completion_response_and_request_shape(self, fixture_server, make_anthropic_pair):
        sandhi_srv = fixture_server(body=json.dumps(COMPLETE_BODY).encode())
        _, sandhi = make_anthropic_pair(sandhi_srv.url)

        sandhi_resp = await run_chat(sandhi)

        # semantic response the binding returns
        assert sandhi_resp.content == "Hello, world"
        assert sandhi_resp.stop_reason == "stop"
        assert sandhi_resp.usage["prompt_tokens"] == 1024
        assert sandhi_resp.usage["completion_tokens"] == 256
        assert sandhi_resp.usage["cache_creation_input_tokens"] == 2048
        assert sandhi_resp.usage["cache_read_input_tokens"] == 4096

        # request-side: path, auth header, and JSON body
        assert len(sandhi_srv.requests) == 1
        sandhi_req = sandhi_srv.requests[0]
        assert sandhi_req.path == "/v1/messages"
        assert sandhi_req.headers.get("x-api-key") == "parity-key"
        assert sandhi_req.headers.get("anthropic-version") == "2023-06-01"
        # Sandhi's anthropic adapter injects an explicit non-streaming marker at the
        # source (documented binding behavior). Everything else is the shared
        # _build_request_params output.
        sandhi_body = json.loads(sandhi_req.body)
        assert sandhi_body.pop("stream", None) is False
        assert sandhi_body["model"] == "claude-sonnet-4-6"
        assert sandhi_body["messages"] == [
            {"role": "user", "content": [{"type": "text", "text": "hi there"}]}
        ]
        expected_body = sandhi._build_request_params(
            MESSAGES, model="claude-sonnet-4-6", temperature=0.2, max_tokens=64, tools=None
        )
        assert sandhi_body["system"] == expected_body["system"]

        assert not hasattr(sandhi, "_sandhi_demoted")

    async def test_tool_use_shape(self, fixture_server, make_anthropic_pair):
        sandhi_srv = fixture_server(body=json.dumps(TOOL_USE_BODY).encode())
        _, sandhi = make_anthropic_pair(sandhi_srv.url)

        sandhi_resp = await run_chat(sandhi, tools=TOOLS)

        assert sandhi_resp.tool_calls == [
            {"id": "toolu_01", "name": "get_weather", "arguments": {"city": "Paris"}}
        ]
        assert sandhi_resp.stop_reason == "tool_calls"
        # tools (with the cache boundary marker) reach the wire, modulo sandhi's
        # injected non-streaming marker
        sandhi_body = json.loads(sandhi_srv.requests[0].body)
        assert sandhi_body.pop("stream", None) is False
        assert "tools" in sandhi_body and len(sandhi_body["tools"]) == 1
        assert sandhi_body["messages"] == [
            {"role": "user", "content": [{"type": "text", "text": "hi there"}]}
        ]

    async def test_no_double_request(self, fixture_server, make_anthropic_pair):
        # Retries are explicitly disabled by this fixture, so there is exactly one
        # upstream POST per chat.
        srv = fixture_server(body=json.dumps(COMPLETE_BODY).encode())
        _, sandhi = make_anthropic_pair(srv.url)
        await run_chat(sandhi)
        assert len(srv.requests) == 1


class TestErrorParity:
    @pytest.mark.parametrize(
        "status,expected",
        [
            (401, ProviderAuthError),
            (429, ProviderRateLimitError),
            (500, ProviderError),
        ],
    )
    async def test_error_class_parity(self, fixture_server, make_anthropic_pair, status, expected):
        body = json.dumps(
            {"type": "error", "error": {"type": "api_error", "message": "boom"}}
        ).encode()
        sandhi_srv = fixture_server(status=status, body=body)
        _, sandhi = make_anthropic_pair(sandhi_srv.url)

        with pytest.raises(expected):
            await run_chat(sandhi)
        assert not hasattr(sandhi, "_sandhi_demoted")
        assert len(sandhi_srv.requests) == 1
