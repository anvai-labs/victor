"""Tests for VertexAIProvider streaming over the shared SSE helper."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from victor.providers.base import Message
from victor.providers.logging import ProviderLogger
from victor.providers.vertex_provider import VertexAIProvider


class _Response:
    def __init__(self, lines):
        self._lines = lines

    def raise_for_status(self) -> None:
        return None

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class _Client:
    def __init__(self, lines):
        self._lines = lines
        self.calls = []

    @asynccontextmanager
    async def stream(self, method, url, json=None, headers=None):
        self.calls.append((method, url))
        yield _Response(self._lines)


def _provider(lines) -> VertexAIProvider:
    provider = object.__new__(VertexAIProvider)
    provider._provider_logger = ProviderLogger("vertex", "tests.vertex")
    provider.client = _Client(lines)
    provider.base_url = "https://vertex.example/models"

    async def _headers():
        return {}

    provider._get_auth_headers = _headers  # type: ignore[method-assign]
    provider._build_request_payload = lambda *a, **k: {}  # type: ignore[method-assign]
    return provider


async def test_stream_yields_text_and_function_calls(caplog) -> None:
    provider = _provider(
        [
            'data: {"candidates":[{"content":{"parts":[{"text":"hi"}]}}]}',
            "data: not-json",
            'data: {"candidates":[{"content":{"parts":[{"functionCall":'
            '{"name":"read","args":{"p":1}}}]},"finishReason":"STOP"}]}',
        ]
    )
    with caplog.at_level(logging.WARNING):
        chunks = [
            c async for c in provider.stream([Message(role="user", content="x")], model="gemini")
        ]

    assert [c.content for c in chunks] == ["hi", ""]
    assert chunks[-1].is_final and chunks[-1].stop_reason == "stop"
    assert chunks[-1].tool_calls == [{"id": "read", "name": "read", "arguments": {"p": 1}}]
    assert "Vertex AI JSON decode error on line: data: not-json" in caplog.text
    assert provider.client.calls == [
        ("POST", "https://vertex.example/models/gemini:streamGenerateContent?alt=sse")
    ]


async def test_stream_done_marker_emits_final_chunk() -> None:
    provider = _provider(["data: [DONE]"])
    chunks = [c async for c in provider.stream([], model="gemini")]
    assert len(chunks) == 1
    assert chunks[0].is_final and chunks[0].tool_calls is None
