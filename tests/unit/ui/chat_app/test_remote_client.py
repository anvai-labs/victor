# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Remote-mode transport tests: SSE parsing and the served-API surface mapping.

These run without chainlit and without a live server — ``httpx.MockTransport``
stands in for ``victor serve`` and the parser is exercised against records split
across arbitrary chunk boundaries (the streaming case that naive line parsers miss).
"""

from __future__ import annotations

import json

import httpx
import pytest

from victor.ui.chat_app.remote_client import RemoteChatClient, _decode_record, _parse_sse


def _sse(dicts: list[dict]) -> bytes:
    return b"".join(f"data: {json.dumps(d)}\n\n".encode() for d in dicts)


async def _collect(aiter) -> list:
    return [item async for item in aiter]


async def _as_stream(chunks: list[bytes]):
    for chunk in chunks:
        yield chunk


class TestSseParsing:
    async def test_records_split_across_chunk_boundaries(self):
        payload = _sse(
            [
                {"event": "content", "content": "hel"},
                {"event": "content", "content": "lo"},
                {"event": "stream_end"},
            ]
        )
        # Split mid-record, mid-line and at odd offsets.
        chunks = [payload[i : i + 7] for i in range(0, len(payload), 7)]
        events = await _collect(_parse_sse(_as_stream(chunks)))
        assert [e["event"] for e in events] == ["content", "content", "stream_end"]
        assert events[0]["content"] == "hel"

    async def test_comments_and_keepalives_are_ignored(self):
        raw = b': keepalive\n\ndata: {"event": "content", "content": "x"}\n\n'
        events = await _collect(_parse_sse(_as_stream([raw])))
        assert events == [{"event": "content", "content": "x"}]

    async def test_trailing_record_without_blank_line_is_flushed(self):
        raw = b'data: {"event": "stream_end"}'
        events = await _collect(_parse_sse(_as_stream([raw])))
        assert events == [{"event": "stream_end"}]

    def test_non_json_record_is_skipped_not_fatal(self):
        assert _decode_record(b"data: [DONE]") is None
        assert _decode_record(b"") is None
        assert _decode_record(b"data: not-json") is None

    async def test_crlf_framing_is_parsed_not_silently_dropped(self):
        raw = (
            b'data: {"event": "content", "content": "a"}\r\n\r\n'
            b'data: {"event": "content", "content": "b"}\r\n\r\n'
            b'data: {"event": "stream_end"}\r\n\r\n'
        )
        events = await _collect(_parse_sse(_as_stream([raw])))
        assert [e["event"] for e in events] == ["content", "content", "stream_end"]

    async def test_crlf_split_across_chunks_is_framed_once(self):
        part1 = b'data: {"event": "content", "content": "x"}\r'
        part2 = b'\n\r\ndata: {"event": "stream_end"}\r\n\r\n'
        events = await _collect(_parse_sse(_as_stream([part1, part2])))
        assert [e["event"] for e in events] == ["content", "stream_end"]

    async def test_clean_eof_with_partial_record_surfaces_error(self):
        raw = b'data: {"event": "content", "content": "trunca'
        events = await _collect(_parse_sse(_as_stream([raw])))
        assert len(events) == 1
        assert events[0]["event"] == "error"
        assert "mid-record" in events[0]["message"]

    def test_multiline_data_records_are_joined(self):
        record = b'data: {"event":\ndata:  "content"}'
        assert _decode_record(record) == {"event": "content"}


class TestRemoteChatClient:
    def _client(self, handler) -> RemoteChatClient:
        return RemoteChatClient("http://test", "sk-test", transport=httpx.MockTransport(handler))

    async def test_stream_yields_wire_events_with_auth_header(self):
        seen_auth = []
        body = _sse(
            [
                {"event": "thinking", "content": "hmm"},
                {"event": "content", "content": "hi"},
                {"event": "stream_end"},
            ]
        )

        def handler(request: httpx.Request) -> httpx.Response:
            seen_auth.append(request.headers.get("Authorization"))
            assert request.url.path == "/chat/wire-stream"
            assert "text/event-stream" in request.headers.get("Accept", "")
            return httpx.Response(200, content=body, headers={"Content-Type": "text/event-stream"})

        client = self._client(handler)
        events = await _collect(client.stream("hello"))
        assert [e["event"] for e in events] == ["thinking", "content", "stream_end"]
        assert seen_auth == ["Bearer sk-test"]
        await client.close()

    async def test_transport_failure_surfaces_as_wire_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(503, text="unavailable")

        client = self._client(handler)
        events = await _collect(client.stream("hello"))
        assert len(events) == 1
        assert events[0]["event"] == "error"
        assert "remote transport failure" in events[0]["message"]
        await client.close()

    async def test_get_available_providers_maps_dict_shape(self):
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/providers"
            return httpx.Response(200, json={"providers": ["zai", "inferflux"]})

        client = self._client(handler)
        assert await client.get_available_providers() == ["zai", "inferflux"]
        assert client.get_last_turn_cost() is None
        assert client.provider_name == "remote"

    async def test_provider_listing_degrades_to_empty(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="boom")

        client = self._client(handler)
        assert await client.get_available_providers() == []

    async def test_history_fetch_is_async_with_limit(self):
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.params["limit"] == "50"
            return httpx.Response(200, json={"messages": [{"role": "user", "content": "hi"}]})

        client = self._client(handler)
        rows = await client.get_messages(limit=50)
        assert rows == [{"role": "user", "content": "hi"}]

    def test_unsupported_session_surfaces_raise_actionable_errors(self):
        client = self._client(lambda request: httpx.Response(200, json={}))
        for call in (client.list_recent_sessions, lambda: client.resume_session("s")):
            with pytest.raises(RuntimeError, match="not available in remote mode"):
                call()
