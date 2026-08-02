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

"""Map Victor stream events to UI-agnostic render actions.

The chat UI (``app.py``) drives Chainlit from these actions, but this module imports
**nothing** from Chainlit so the mapping is unit-testable without the optional ``chat-ui``
extra installed.

Two parallel translators produce the same :class:`RenderAction` vocabulary:

- :func:`map_event` — the in-process path, over the public surface of
  ``VictorClient.stream()`` events (``event_type``, ``content``, ``tool_name``,
  ``result``, ``metadata``; see ``victor/framework/client.py``). Keeps the rich
  in-process fields (full output, follow-up hints).
- :func:`map_wire_event` — the cross-surface path, over v1 wire-event dicts
  (``victor/framework/wire_events.py``), for renderers fed by the versioned
  contract (SSE consumers, TUI parity). Bounded by design.

Their agreement over the same stream is pinned by the renderer replay contract
test (``tests/unit/ui/chat_app/test_wire_render_parity.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional


class RenderKind(str, Enum):
    """What the UI should do with a single stream event."""

    TOKEN = "token"  # append a content token to the assistant message
    THINKING = "thinking"  # reasoning text -> render as a collapsed step
    TOOL_START = "tool_start"  # a tool call began (carries arguments)
    TOOL_END = "tool_end"  # a tool produced a result -> render a tool step
    MEMBER_START = "member_start"  # a team member began (carries member_id) -> lane marker
    MEMBER_END = "member_end"  # a team member finished (member_id + success) -> lane marker
    MEMBER_AWAITING = "member_awaiting"  # a team member paused awaiting approval (ADR-023 2b)
    ERROR = "error"  # surface an error to the user
    IGNORE = "ignore"  # lifecycle/no-op events (stream_start, stream_end, ...)


@dataclass
class RenderAction:
    """A UI-agnostic instruction derived from one Victor stream event."""

    kind: RenderKind
    text: str = ""
    tool_name: Optional[str] = None
    call_id: Optional[str] = None  # correlates TOOL_START/TOOL_END for parallel calls
    success: bool = True
    elapsed: float = 0.0
    was_pruned: bool = False
    follow_up_suggestions: Optional[list[dict[str, Any]]] = None
    member_id: Optional[str] = None  # set for MEMBER_START/MEMBER_END lane markers (ADR-023)
    metadata: Dict[str, Any] = field(default_factory=dict)


def _extract_call_id(event: Any, metadata: Dict[str, Any]) -> Optional[str]:
    """Best-effort tool call id (correlates a tool_call to its tool_result)."""
    for key in ("tool_call_id", "id", "call_id"):
        value = metadata.get(key)
        if value:
            return str(value)
    for attr in ("tool_call_id", "call_id"):
        value = getattr(event, attr, None)
        if value:
            return str(value)
    return None


def _normalize_event_type(event_type: Any) -> str:
    """Reduce a str / ``EventType`` enum / ``"EventType.CONTENT"`` repr to a bare token.

    ``VictorClient.stream()`` yields events whose ``event_type`` is usually a lowercase
    string ("content", "tool_call", ...) but may be an ``EventType`` member; normalize all
    forms to the bare lowercase name ("content", "tool_call", "error").
    """
    value = getattr(event_type, "value", event_type)
    text = str(value).strip().lower()
    if "." in text:  # e.g. "eventtype.content" -> "content"
        text = text.rsplit(".", 1)[-1]
    return text


def _tool_result_payload(event: Any) -> Dict[str, Any]:
    """Best-effort extraction of the flat tool-result dict from an event.

    ``VictorClient._to_stream_event`` flattens the payload onto ``event.result``;
    fall back to a nested ``metadata['tool_result']`` (or bare ``metadata``) so the
    mapping still works if an event arrives un-flattened.
    """
    result = getattr(event, "result", None)
    if isinstance(result, dict):
        return result
    metadata = getattr(event, "metadata", None)
    if isinstance(metadata, dict):
        nested = metadata.get("tool_result")
        if isinstance(nested, dict):
            return nested
        return metadata
    return {}


def map_event(event: Any) -> RenderAction:
    """Translate one ``VictorClient.stream()`` event into a :class:`RenderAction`.

    Unknown / lifecycle events map to :attr:`RenderKind.IGNORE` so callers can render with a
    single ``match`` and never crash on a new event type.
    """
    event_type = _normalize_event_type(getattr(event, "event_type", ""))
    content = getattr(event, "content", None) or ""
    metadata = getattr(event, "metadata", None) or {}

    if event_type == "content":
        return RenderAction(RenderKind.TOKEN, text=content)

    if event_type == "thinking":
        # Reasoning content arrives on `content` or `metadata['reasoning_content']`.
        text = content or str(metadata.get("reasoning_content", ""))
        return RenderAction(RenderKind.THINKING, text=text, metadata=dict(metadata))

    if event_type == "tool_call":
        return RenderAction(
            RenderKind.TOOL_START,
            tool_name=getattr(event, "tool_name", None) or "tool",
            call_id=_extract_call_id(event, metadata),
            metadata={"arguments": metadata.get("arguments", {})},
        )

    if event_type in ("tool_result", "tool_error"):
        payload = _tool_result_payload(event)
        # Prefer the full output for direct display; the bare ``result`` is a CLI
        # "/expand" placeholder, so it is the last resort behind the real output.
        text = payload.get("original_result") or content or payload.get("result") or ""
        return RenderAction(
            RenderKind.TOOL_END,
            text=str(text),
            tool_name=getattr(event, "tool_name", None) or "tool",
            call_id=_extract_call_id(event, payload) or _extract_call_id(event, metadata),
            success=event_type != "tool_error" and bool(payload.get("success", True)),
            elapsed=float(payload.get("elapsed", 0.0) or 0.0),
            was_pruned=bool(payload.get("was_pruned", False)),
            follow_up_suggestions=payload.get("follow_up_suggestions") or None,
            metadata={"arguments": payload.get("arguments", {})},
        )

    if event_type == "error":
        return RenderAction(
            RenderKind.ERROR,
            text=content or str(metadata.get("error", "Unknown streaming error")),
        )

    if event_type == "awaiting_approval":
        # FEP-0029: a single-agent turn durably paused on a policy ASK. Render it on the SAME paused
        # lane as a team member pause (reuse RenderKind.MEMBER_AWAITING) — labelled "agent", with the
        # gated tool + the resume run_id so the user knows how to continue (`session resume <id>`).
        req = metadata.get("approval_request") or {}
        title = str(req.get("title") or req.get("tool_name") or "")
        run_id = metadata.get("run_id")
        detail = f"{title} · run {run_id}" if run_id else title
        return RenderAction(
            RenderKind.MEMBER_AWAITING,
            text=detail,
            member_id="agent",
            metadata=dict(metadata),
        )

    if event_type == "custom":
        # Team member lifecycle events (ADR-023) ride on CUSTOM with a member_* sub-type.
        # Every other custom event (milestones, etc.) falls through to IGNORE unchanged.
        custom_type = str(metadata.get("custom_type", ""))
        if custom_type == "member_start":
            return RenderAction(
                RenderKind.MEMBER_START,
                text=content,
                member_id=getattr(event, "member_id", None) or metadata.get("member_id"),
                metadata=dict(metadata),
            )
        if custom_type in ("member_completed", "member_error"):
            return RenderAction(
                RenderKind.MEMBER_END,
                text=content,
                success=custom_type != "member_error",
                member_id=getattr(event, "member_id", None) or metadata.get("member_id"),
                metadata=dict(metadata),
            )
        if custom_type == "member_awaiting_approval":
            # ADR-023 pillar 2b: the member durably paused awaiting human approval.
            return RenderAction(
                RenderKind.MEMBER_AWAITING,
                text=content,
                member_id=getattr(event, "member_id", None) or metadata.get("member_id"),
                metadata=dict(metadata),
            )

    return RenderAction(RenderKind.IGNORE)


def map_wire_event(wire: Any) -> RenderAction:
    """Translate one v1 wire event (``victor.framework.wire_events``) into a RenderAction.

    The wire contract is the cross-surface schema (web SSE, TUI, remote UIs);
    this mapper gives every Python surface the same render semantics for it
    that the in-process chat app has for raw client events. Unknown event
    types map to :attr:`RenderKind.IGNORE` — additive contract growth never
    crashes a renderer.

    Parity with :func:`map_event` over the same underlying stream is pinned by
    the renderer replay contract test; the wire path differs only where the
    contract intentionally bounds payloads (result size, no follow-up hints).
    """
    if not isinstance(wire, dict):
        return RenderAction(RenderKind.IGNORE)

    event = str(wire.get("event", ""))

    if event == "content":
        return RenderAction(RenderKind.TOKEN, text=str(wire.get("content", "") or ""))

    if event == "thinking":
        return RenderAction(RenderKind.THINKING, text=str(wire.get("content", "") or ""))

    if event == "tool_call":
        return RenderAction(
            RenderKind.TOOL_START,
            tool_name=str(wire.get("tool", "") or "tool"),
            call_id=str(wire["call_id"]) if wire.get("call_id") else None,
            metadata={"arguments": wire.get("arguments", {}) or {}},
        )

    if event == "tool_result":
        result = wire.get("result")
        elapsed_ms = wire.get("elapsed_ms")
        return RenderAction(
            RenderKind.TOOL_END,
            text="" if result is None else str(result),
            tool_name=str(wire.get("tool", "") or "tool"),
            call_id=str(wire["call_id"]) if wire.get("call_id") else None,
            success=bool(wire.get("success", True)),
            elapsed=(float(elapsed_ms) / 1000.0) if isinstance(elapsed_ms, (int, float)) else 0.0,
            was_pruned=bool(wire.get("truncated", False)),
        )

    if event == "error":
        return RenderAction(
            RenderKind.ERROR,
            text=str(wire.get("message", "") or "Unknown streaming error"),
        )

    if event == "member_start":
        return RenderAction(
            RenderKind.MEMBER_START,
            text=str(wire.get("content", "") or ""),
            member_id=wire.get("member_id"),
        )

    if event in ("member_completed", "member_error"):
        return RenderAction(
            RenderKind.MEMBER_END,
            text=str(wire.get("content", "") or ""),
            success=event != "member_error",
            member_id=wire.get("member_id"),
        )

    if event == "member_awaiting_approval":
        return RenderAction(
            RenderKind.MEMBER_AWAITING,
            text=str(wire.get("content", "") or ""),
            member_id=wire.get("member_id"),
        )

    return RenderAction(RenderKind.IGNORE)  # stream_end + future additive types


def segment_turn(kinds: Iterable["RenderKind"]) -> List[str]:
    """Return the ordered render-segment types for a turn — the natural-flow contract.

    A turn's events interleave per agent iteration: [text][tool_call/result…][text]…. To
    render that like the terminal (instead of all tool steps piling at the end), the UI emits
    a NEW text message segment whenever text resumes after a tool run, and groups each
    iteration's tool calls into one tool segment. This pure helper encodes that contract so it
    can be unit-tested; ``app.py`` mirrors it online while streaming.

    Returns a list like ``["text", "tools", "text", "tools", "text"]``. THINKING/IGNORE do not
    open a segment (reasoning renders in its own step; lifecycle events are no-ops).
    """
    segments: List[str] = []
    phase: Optional[str] = None
    for kind in kinds:
        if kind in (RenderKind.TOKEN, RenderKind.ERROR):
            if phase != "text":
                segments.append("text")
                phase = "text"
        elif kind in (RenderKind.TOOL_START, RenderKind.TOOL_END):
            if phase != "tools":
                segments.append("tools")
                phase = "tools"
    return segments


def provider_switch_hint(current: Optional[str], available: Iterable[str]) -> str:
    """One-line hint suggesting other providers to switch to after a turn fails (Chainlit-free).

    Lists available providers other than the current one (order-preserving, deduped), so a
    failure card can nudge the user toward an alternative. Returns ``""`` when there is no
    alternative to suggest. ``app.py`` renders this; kept pure here so it is unit-testable.
    """
    others = [p for p in dict.fromkeys(available) if p and p != current]
    if not others:
        return ""
    return "_Try another provider:_ " + ", ".join(others)


def history_messages(messages: Iterable[Any]) -> List[tuple[str, str]]:
    """Normalize ``VictorClient.get_messages()`` objects into ``(author, content)`` pairs.

    Used to replay a reconnected session's visible turns. Only user/assistant messages with
    content are kept (system/tool messages are internal orchestration); the author is ``"You"``
    for user turns and ``"Victor"`` for assistant turns. Accepts message objects (``.role`` /
    ``.content``, where ``role`` may be a str or an enum with ``.value``) or plain dicts. Pure +
    Chainlit-free for unit testing.
    """
    pairs: List[tuple[str, str]] = []
    for msg in messages or []:
        raw_role = getattr(msg, "role", None)
        if raw_role is None and isinstance(msg, dict):
            raw_role = msg.get("role")
        role = str(getattr(raw_role, "value", raw_role) or "").lower()

        content = getattr(msg, "content", None)
        if content is None and isinstance(msg, dict):
            content = msg.get("content")
        content = str(content or "").strip()

        if not content or role not in ("user", "assistant"):
            continue
        pairs.append(("You" if role == "user" else "Victor", content))
    return pairs
