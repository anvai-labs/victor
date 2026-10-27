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

"""Remote-mode chat transport for the Chainlit app (VAS-14 "remote mode uses API").

Speaks the served agent-service API — ``POST /chat/wire-stream`` (the v1 wire event
contract, SSE-framed), ``GET /providers``, ``GET /history`` — and yields the raw wire
dicts that :func:`victor.ui.chat_app.event_mapping.map_wire_event` renders, so the
presentation layer is byte-for-byte the same code path as the TUI wire timeline.

This is deliberately a duck-typed subset of the ``VictorClient`` surface the chat app
consumes, not a subclass: remote mode cannot share in-process state. Known v1 gaps
(captured as VAS-10/VAS-14 evidence in the PR): no session listing/resume over HTTP,
no mid-stream HITL approvals, no per-turn cost object on the wire. Each degrades
loudly (empty footer, hidden resume picker, approvals off) rather than silently.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncIterator, Optional

import httpx

logger = logging.getLogger(__name__)

_WIRE_STREAM_PATH = "/chat/wire-stream"


class RemoteChatClient:
    """HTTP client for a running ``victor serve`` instance.

    ``stream()`` yields v1 wire-event dicts (``{"event": "content", ...}``); the chat
    app renders them with ``map_wire_event``. Transport failures surface as a wire
    ``error`` event so a turn always renders a visible failure instead of hanging.
    """

    is_remote = True
    """Marker the chat app checks to switch render mapping and degrade unsupported features."""

    @property
    def provider_name(self) -> str:
        """Cosmetic identity for error hints; the serving profile owns the real provider."""
        return "remote"

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        *,
        timeout: float = 600.0,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ) -> None:
        headers = {"Accept": "text/event-stream"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        if (
            base_url.startswith("http://")
            and "localhost" not in base_url
            and "127.0.0.1" not in base_url
        ):
            logger.warning(
                "VICTOR_UI_REMOTE_URL uses cleartext http:// to a non-local host; "
                "the bearer key travels unencrypted"
            )
        base = base_url.rstrip("/")
        self._http = httpx.AsyncClient(
            base_url=base, headers=headers, timeout=timeout, transport=transport
        )
        # Small request/response calls (providers, history) run on worker threads
        # under a short deadline — unlike the streaming client's long turn timeout,
        # a wedged server must not stall chat start for minutes.
        self._sync = httpx.Client(
            base_url=base,
            headers=headers,
            timeout=httpx.Timeout(10.0, connect=5.0),
            transport=transport,
        )

    async def initialize(self) -> None:  # parity no-op: VictorClient.initialize()
        """Remain parity with the embedded client's lifecycle; HTTP needs no warmup."""

    def set_approval_handler(self, handler: Any) -> None:
        """Accept the handler for signature parity and refuse it loudly.

        Mid-stream HITL approvals need the durable FEP-0029 resume flow over HTTP;
        until the server exposes it to remote UIs, remote mode builds its session
        with approvals disabled so this is never called in practice.
        """
        logger.warning("remote mode ignores approval handlers; approvals are disabled")

    async def get_available_providers(self) -> list[str]:
        """Fetch the server's provider list (best-effort, off the event loop)."""
        try:
            response = await asyncio.to_thread(self._sync.get, "/providers")
            response.raise_for_status()
            payload = response.json()
        except Exception:
            logger.warning("remote provider listing failed", exc_info=True)
            return []
        if isinstance(payload, list):
            return [str(item) for item in payload]
        if isinstance(payload, dict):
            providers = payload.get("providers")
            if isinstance(providers, list):
                return [str(item) for item in providers]
        return []

    def get_last_turn_cost(self) -> None:
        """Wire v1 carries no per-turn cost object; the app skips the cost footer."""
        return None

    def list_recent_sessions(self, limit: int = 10):  # noqa: ARG002 - parity surface
        """Session listing is not served over HTTP yet (VAS-10 gap)."""
        raise RuntimeError(
            "session resume is not available in remote mode; restart a conversation instead"
        )

    def resume_session(self, session_id: str):  # noqa: ARG002 - parity surface
        """Session resume is not served over HTTP yet (VAS-10 gap)."""
        raise RuntimeError(
            "session resume is not available in remote mode; restart a conversation instead"
        )

    async def get_messages(self, limit: int = 100) -> list[Any]:
        """Fetch conversation history rows from ``GET /history`` (best-effort)."""
        try:
            response = await asyncio.to_thread(self._sync.get, "/history", params={"limit": limit})
            response.raise_for_status()
            payload = response.json()
        except Exception:
            logger.warning("remote history fetch failed", exc_info=True)
            return []
        rows = payload.get("messages") if isinstance(payload, dict) else payload
        return rows if isinstance(rows, list) else []

    async def stream(self, message: str) -> AsyncIterator[dict]:
        """Stream one turn as v1 wire-event dicts from ``POST /chat/wire-stream``."""
        try:
            async with self._http.stream(
                "POST",
                _WIRE_STREAM_PATH,
                json={"messages": [{"role": "user", "content": message}]},
            ) as response:
                response.raise_for_status()
                async for wire in _parse_sse(response.aiter_bytes()):
                    yield wire
        except httpx.HTTPError as exc:
            yield {"event": "error", "message": f"remote transport failure: {exc}"}

    async def close(self) -> None:
        await self._http.aclose()
        await asyncio.to_thread(self._sync.close)


async def _parse_sse(chunks: AsyncIterator[bytes]) -> AsyncIterator[dict]:
    """Incrementally decode SSE ``data:`` records into JSON dicts.

    Tolerates record framing split across chunk boundaries and ignores comment/keepalive
    lines. Non-JSON payloads (never emitted by the wire endpoint, but cheap to guard)
    are skipped with a debug log rather than killing the turn.
    """
    buffer = b""
    async for chunk in chunks:
        buffer += chunk
        # SSE permits CR, LF and CRLF terminators; normalize before framing. A
        # trailing lone CR is held back — its LF partner may arrive next chunk.
        buffer = buffer.replace(b"\r\n", b"\n")
        if buffer.endswith(b"\r"):
            buffer = buffer[:-1].replace(b"\r", b"\n") + b"\r"
        else:
            buffer = buffer.replace(b"\r", b"\n")
        while b"\n\n" in buffer:
            record, buffer = buffer.split(b"\n\n", 1)
            payload = _decode_record(record)
            if payload is not None:
                yield payload
    if buffer.strip():
        # Clean EOF with an unterminated record: a complete JSON payload (server
        # died between record and framing newline) still delivers; anything else
        # is a truncated stream and surfaces as a wire error, never silence.
        tail = _decode_record(buffer)
        if tail is not None:
            yield tail
        else:
            yield {"event": "error", "message": "remote stream ended mid-record"}


def _decode_record(record: bytes) -> Optional[dict]:
    data_lines = [
        line[len("data: ") :].decode("utf-8", errors="replace")
        for line in record.split(b"\n")
        if line.startswith(b"data: ")
    ]
    if not data_lines:
        return None
    try:
        payload = json.loads("\n".join(data_lines))
    except json.JSONDecodeError:
        logger.debug("skipping non-JSON SSE record: %r", data_lines[:1])
        return None
    return payload if isinstance(payload, dict) else None
