# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
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

"""Versioned JSON wire contract for agent events (UX P1).

One schema for every surface: the framework's typed events serialize to small,
versioned JSON documents that the web (SSE/WebSocket), CLI, and TUI all render
from — the agent loop stays legible everywhere without per-surface reshaping.

Wire shape (``v`` is the contract version; additive-only within a version):

    {"v": 1, "event": "thinking",    "content": "..."}
    {"v": 1, "event": "tool_call",   "tool": "read", "arguments": {...},
             "call_id": "..."}                                   # call_id optional
    {"v": 1, "event": "tool_result", "tool": "read", "success": true, "result": "...",
             "call_id": "...", "elapsed_ms": 120, "truncated": true}  # last 3 optional
    {"v": 1, "event": "content",     "content": "..."}
    {"v": 1, "event": "error",       "message": "..."}
    {"v": 1, "event": "stream_end"}

Design notes:
- ``to_wire_event`` returns ``None`` for event types outside the v1 contract
  (tool_progress, stage_change, …) — they can be admitted additively later.
- ``TOOL_ERROR`` maps onto ``tool_result`` with ``success: false`` — the wire
  contract's six types stay closed while losing no fidelity.
- Tool results are made JSON-safe and size-bounded: UIs render previews;
  full results belong to the conversation, not the event stream.
- ``tool_result.result`` is always the human-readable output string, never the
  internal tool-pipeline payload dict; ``call_id`` correlates parallel tool
  calls with their results, ``elapsed_ms`` carries the measured duration, and
  ``truncated`` flags output bounded upstream or on the wire (all additive,
  UX P3).
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, Optional

WIRE_VERSION = 1

# The v1 contract: exactly these event types cross the wire.
WIRE_EVENT_TYPES = frozenset(
    {"thinking", "tool_call", "tool_result", "content", "error", "stream_end"}
)

# Tool results larger than this are truncated on the wire (UIs show previews;
# the full result lives in the conversation, not the event stream).
MAX_RESULT_CHARS = 16_384


def _json_safe(value: Any, *, max_chars: int = MAX_RESULT_CHARS) -> Any:
    """Coerce an arbitrary tool result into a JSON-safe, size-bounded value."""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if not isinstance(value, str):
        try:
            json.dumps(value)
        except (TypeError, ValueError):
            value = str(value)
        else:
            return value
    if len(value) > max_chars:
        return value[:max_chars] + f"… [truncated {len(value) - max_chars} chars]"
    return value


def to_wire_event(event: Any) -> Optional[Dict[str, Any]]:
    """Serialize a client stream event to the versioned wire shape.

    Accepts anything matching the ``VictorClient.stream()`` event surface
    (``event_type``/``content``/``tool_name``/``arguments``/``result``/
    ``success``). Returns ``None`` for event types outside the v1 contract.
    """
    event_type = str(getattr(event, "event_type", "") or "")

    if event_type in ("content", "thinking"):
        content = getattr(event, "content", None)
        if not content:
            return None
        return {"v": WIRE_VERSION, "event": event_type, "content": content}

    if event_type == "tool_call":
        wire: Dict[str, Any] = {
            "v": WIRE_VERSION,
            "event": "tool_call",
            "tool": getattr(event, "tool_name", None) or "tool",
        }
        arguments = getattr(event, "arguments", None)
        if isinstance(arguments, dict) and arguments:
            wire["arguments"] = _json_safe_arguments(arguments)
        call_id = _extract_call_id(event)
        if call_id:
            wire["call_id"] = call_id
        return wire

    if event_type in ("tool_result", "tool_error"):
        success = bool(getattr(event, "success", True)) and event_type != "tool_error"
        wire = {
            "v": WIRE_VERSION,
            "event": "tool_result",
            "tool": getattr(event, "tool_name", None) or "tool",
            "success": success,
        }
        raw_result = getattr(event, "result", None)
        if isinstance(raw_result, dict) and (
            "result" in raw_result or "original_result" in raw_result
        ):
            # The client surface flattens the internal tool-pipeline payload
            # onto ``result`` (original_result, elapsed, follow_up_suggestions,
            # was_pruned, …). Only contract fields cross the wire: the
            # human-readable output string plus measured duration/truncation.
            display = raw_result.get("original_result") or raw_result.get("result")
            wire["result"] = _json_safe(display)
            wire["success"] = success and bool(raw_result.get("success", True))
            elapsed = raw_result.get("elapsed")
            if isinstance(elapsed, (int, float)) and elapsed > 0:
                wire["elapsed_ms"] = int(elapsed * 1000)
            truncated = bool(raw_result.get("was_pruned")) or _was_bounded(display, wire["result"])
            if truncated:
                wire["truncated"] = True
        else:
            wire["result"] = _json_safe(raw_result)
            if _was_bounded(raw_result, wire["result"]):
                wire["truncated"] = True
        call_id = _extract_call_id(event)
        if call_id:
            wire["call_id"] = call_id
        return wire

    if event_type == "error":
        return {
            "v": WIRE_VERSION,
            "event": "error",
            "message": str(getattr(event, "content", None) or "unknown error"),
        }

    if event_type == "stream_end":
        return {"v": WIRE_VERSION, "event": "stream_end"}

    return None  # outside the v1 contract (tool_progress, stage_change, ...)


def _json_safe_arguments(arguments: Dict[str, Any]) -> Dict[str, Any]:
    return {str(key): _json_safe(value, max_chars=2_048) for key, value in arguments.items()}


def _extract_call_id(event: Any) -> Optional[str]:
    """Best-effort tool call id, correlating parallel tool_call/tool_result pairs.

    Mirrors the chat UI's extraction: the id may ride on the event itself or in
    its metadata/payload under a few historical key names.
    """
    for attr in ("tool_call_id", "call_id"):
        value = getattr(event, attr, None)
        if value:
            return str(value)
    for container in (getattr(event, "metadata", None), getattr(event, "result", None)):
        if isinstance(container, dict):
            for key in ("tool_call_id", "id", "call_id"):
                value = container.get(key)
                if value:
                    return str(value)
    return None


def _was_bounded(original: Any, on_wire: Any) -> bool:
    """True when the wire value is a truncation of the original string."""
    return isinstance(original, str) and isinstance(on_wire, str) and len(on_wire) != len(original)


def encode_sse(wire_event: Dict[str, Any]) -> str:
    """Frame one wire event as a Server-Sent Events ``data:`` record."""
    return f"data: {json.dumps(wire_event, ensure_ascii=False)}\n\n"


async def stream_wire_events(client: Any, message: str) -> AsyncIterator[Dict[str, Any]]:
    """Drive ``client.stream(message)`` and yield v1 wire events.

    Guarantees: every stream terminates with exactly one ``stream_end``, and
    an in-stream exception surfaces as an ``error`` event before it — a wire
    consumer never sees a silently truncated stream.
    """
    ended = False
    try:
        async for event in client.stream(message):
            wire = to_wire_event(event)
            if wire is None:
                continue
            if wire["event"] == "stream_end":
                if ended:
                    continue
                ended = True
            yield wire
    except Exception as exc:  # surface, then terminate — never truncate silently
        yield {"v": WIRE_VERSION, "event": "error", "message": str(exc)}
    if not ended:
        yield {"v": WIRE_VERSION, "event": "stream_end"}


async def stream_sse(client: Any, message: str) -> AsyncIterator[str]:
    """SSE-framed variant of :func:`stream_wire_events` for HTTP surfaces."""
    async for wire in stream_wire_events(client, message):
        yield encode_sse(wire)
