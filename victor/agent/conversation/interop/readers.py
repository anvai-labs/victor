# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
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

"""Transcript readers: external formats → victor ConversationSession IR.

Every reader streams its input line-by-line for JSONL sources (no whole-file
``json.load``), applies the shared resource guards, and accounts every loss in
a ConversionReport. Tool traffic is preserved structurally (role=tool_call /
tool with tool_calls / tool_call_id), not flattened to text; text traces are
only emitted for orphan tool results whose matching call is unavailable.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from victor.agent.conversation.interop.formats import (
    MAX_FILE_BYTES,
    MAX_LINE_BYTES,
    MAX_MESSAGE_CHARS,
    MAX_MESSAGES,
    DetectionResult,
    TranscriptFormat,
    UnsupportedFormatError,
    detect_format,
)
from victor.agent.conversation.interop.report import ConversionReport
from victor.agent.conversation.session_model import ConversationSession
from victor.agent.conversation.types import ConversationMessage

_MICROSECOND = timedelta(microseconds=1)


@dataclass
class ImportedSession:
    """Result of reading a transcript into the victor IR."""

    session: ConversationSession
    report: ConversionReport
    source_path: Path


def _deterministic_session_id(path: Path) -> str:
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return f"imp_{digest[:12]}"


def _parse_ts(raw: Any) -> Optional[datetime]:
    """Parse RFC3339/ISO timestamps (Z suffix included); None on failure."""
    if not raw or not isinstance(raw, str):
        return None
    text = raw.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _ensure_strictly_increasing(
    stamp: Optional[datetime], previous: Optional[datetime]
) -> datetime:
    """Guarantee strictly increasing timestamps so order survives storage."""
    base = stamp or previous or datetime.now(timezone.utc)
    if previous is not None and base <= previous:
        base = previous + _MICROSECOND
    return base


def _clamp_content(content: str, report: ConversionReport) -> str:
    if len(content) > MAX_MESSAGE_CHARS:
        report.truncations += 1
        return content[:MAX_MESSAGE_CHARS] + " …[import-truncated]"
    return content


def _iter_jsonl(path: Path, report: ConversionReport):
    """Yield (ordinal, parsed-record) pairs; malformed lines are skipped+noted."""
    with open(path, "rb") as f:
        for ordinal, raw in enumerate(f):
            raw = raw.strip()
            if not raw:
                continue
            if len(raw) > MAX_LINE_BYTES:
                report.note_skip(ordinal, "line", "line exceeds guard")
                continue
            try:
                yield ordinal, json.loads(raw)
            except json.JSONDecodeError as exc:
                report.note_skip(ordinal, "line", f"malformed JSON: {exc.msg}")


class _TurnBuilder:
    """Accumulates ConversationMessages with monotonic timestamps."""

    def __init__(self) -> None:
        self.messages: List[ConversationMessage] = []
        self._previous: Optional[datetime] = None
        self.first_ts: Optional[datetime] = None
        self.last_ts: Optional[datetime] = None

    def add(
        self,
        role: str,
        content: str,
        *,
        stamp: Optional[datetime] = None,
        msg_id: Optional[str] = None,
        tool_name: Optional[str] = None,
        tool_call_id: Optional[str] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        normalized = _ensure_strictly_increasing(stamp, self._previous)
        self._previous = normalized
        self.first_ts = self.first_ts or normalized
        self.last_ts = normalized
        self.messages.append(
            ConversationMessage(
                id=msg_id,
                role=role,
                content=content,
                timestamp=normalized,
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                tool_calls=tool_calls,
                metadata=metadata or {},
            )
        )

    def cap_reached(self) -> bool:
        return len(self.messages) >= MAX_MESSAGES


# ---------------------------------------------------------------------------
# Claude Code


def _read_claude(path: Path, report: ConversionReport) -> ImportedSession:
    builder = _TurnBuilder()
    cwd: Optional[str] = None
    model: Optional[str] = None
    source_session: Optional[str] = None
    pending_tool_calls: Dict[str, str] = {}  # tool_use id -> name (for orphan check)
    stopped = False

    for ordinal, entry in _iter_jsonl(path, report):
        if stopped:
            break
        report.entries_read += 1
        if not isinstance(entry, dict):
            report.note_skip(ordinal, "entry", "not an object")
            continue
        entry_type = entry.get("type")
        cwd = cwd or entry.get("cwd")
        source_session = source_session or entry.get("sessionId")
        if entry.get("isSidechain") or entry.get("isMeta"):
            report.note_skip(ordinal, entry_type or "entry", "sidechain/meta")
            continue
        if entry_type not in ("user", "assistant"):
            continue  # bookkeeping entries (file-history, queue ops, …) — not losses
        message = entry.get("message") or {}
        stamp = _parse_ts(entry.get("timestamp"))
        role = message.get("role", "user" if entry_type == "user" else "assistant")
        content = message.get("content")

        if entry_type == "assistant" and not model:
            model = message.get("model")

        if isinstance(content, str):
            text = content.strip()
            if text.startswith(("<system-reminder", "<local-command", "Caveat:")):
                report.note_skip(ordinal, entry_type, "harness-injected preamble")
                continue
            if text:
                builder.add(
                    role, _clamp_content(text, report), stamp=stamp, msg_id=entry.get("uuid")
                )
            continue

        if not isinstance(content, list):
            report.note_skip(ordinal, entry_type, "unrecognized content shape")
            continue

        for block in content:
            if builder.cap_reached():
                report.note_warning(f"stopped at {MAX_MESSAGES} messages (guard)")
                stopped = True
                break
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                text = (block.get("text") or "").strip()
                if text:
                    builder.add(role, _clamp_content(text, report), stamp=stamp)
            elif btype == "thinking":
                report.thinking_blocks_dropped += 1  # signatures don't transfer
            elif btype == "tool_use":
                call_id = block.get("id") or f"call_{len(builder.messages)}"
                name = block.get("name") or "tool"
                pending_tool_calls[call_id] = name
                arguments = block.get("input")
                builder.add(
                    "tool_call",
                    "",
                    stamp=stamp,
                    tool_name=name,
                    tool_call_id=call_id,
                    tool_calls=[
                        {
                            "id": call_id,
                            "name": name,
                            "arguments": (
                                _clamp_content(json.dumps(arguments, default=str), report)
                                if arguments is not None
                                else "{}"
                            ),
                        }
                    ],
                )
                report.tool_calls_preserved += 1
            elif btype == "tool_result":
                call_id = block.get("tool_use_id")
                inner = block.get("content")
                if isinstance(inner, list):
                    inner = "\n".join(p.get("text", "") for p in inner if isinstance(p, dict))
                body = _clamp_content(str(inner or ""), report)
                if call_id in pending_tool_calls:
                    builder.add(
                        "tool",
                        body,
                        stamp=stamp,
                        tool_call_id=call_id,
                        tool_name=pending_tool_calls[call_id],
                    )
                else:
                    # Orphan result (call lived in a skipped branch): keep as a
                    # user-visible trace so context is not silently lost.
                    report.note_warning("orphan tool_result kept as text trace")
                    builder.add("user", f"[tool_result: {body}]", stamp=stamp)

    first_user = next((m.content for m in builder.messages if m.role == "user" and m.content), None)
    session = ConversationSession(
        session_id=_deterministic_session_id(path),
        title=(first_user[:50] if first_user else "Claude Code import"),
        messages=builder.messages,
        model=model,
        provider="claude-code",
        created_at=builder.first_ts or datetime.now(timezone.utc),
    )
    _annotate(report, builder, session, TranscriptFormat.CLAUDE_CODE)
    report.session_id = session.session_id
    return ImportedSession(session=session, report=report, source_path=path)


# ---------------------------------------------------------------------------
# Codex rollout


def _read_codex(path: Path, report: ConversionReport) -> ImportedSession:
    builder = _TurnBuilder()
    cwd: Optional[str] = None
    model: Optional[str] = None
    call_names: Dict[str, str] = {}
    stopped = False

    for ordinal, entry in _iter_jsonl(path, report):
        if stopped:
            break
        report.entries_read += 1
        if not isinstance(entry, dict):
            report.note_skip(ordinal, "entry", "not an object")
            continue
        entry_type = entry.get("type")
        stamp = _parse_ts(entry.get("timestamp"))
        payload = entry.get("payload") or {}

        if entry_type == "session_meta":
            cwd = cwd or payload.get("cwd")
            continue
        if entry_type == "turn_context":
            model = model or payload.get("model")
            continue
        if entry_type == "compacted":
            # Post-compaction history is encrypted; it cannot transfer.
            report.encrypted_items_skipped += 1
            continue
        if entry_type != "response_item":
            # event_msg / token_usage_record / world_state …
            continue

        ptype = payload.get("type")
        if ptype == "compacted" or "encrypted_content" in payload:
            report.encrypted_items_skipped += 1
            continue
        if ptype == "reasoning":
            report.thinking_blocks_dropped += 1  # reasoning never transfers
            continue
        if ptype in ("additional_tools", "tool_approval_request", "web_search_call"):
            continue  # bookkeeping
        if ptype == "message":
            role = payload.get("role")
            if role not in ("user", "assistant"):
                continue  # developer/system preamble stays out of the chat flow
            texts: List[str] = []
            for item in payload.get("content", []):
                if isinstance(item, dict) and item.get("type") in (
                    "input_text",
                    "output_text",
                ):
                    texts.append(item.get("text") or "")
            text = "\n".join(t for t in texts if t.strip()).strip()
            if not text:
                continue
            if role == "user" and (
                text.startswith("# AGENTS.md")
                or text.lstrip().startswith(
                    ("<environment_context", "<user_instructions>", "<ENVIRONMENT")
                )
            ):
                report.note_skip(ordinal, "user", "environment preamble")
                continue
            if builder.cap_reached():
                report.note_warning(f"stopped at {MAX_MESSAGES} messages (guard)")
                stopped = True
                continue
            builder.add(role, _clamp_content(text, report), stamp=stamp)
        elif ptype == "function_call":
            if builder.cap_reached():
                stopped = True
                continue
            call_id = payload.get("call_id") or payload.get("id") or "call_0"
            name = payload.get("name") or "tool"
            call_names[call_id] = name
            builder.add(
                "tool_call",
                "",
                stamp=stamp,
                tool_name=name,
                tool_call_id=call_id,
                tool_calls=[
                    {
                        "id": call_id,
                        "name": name,
                        "arguments": _clamp_content(str(payload.get("arguments") or "{}"), report),
                    }
                ],
            )
            report.tool_calls_preserved += 1
        elif ptype == "function_call_output":
            if builder.cap_reached():
                stopped = True
                continue
            call_id = payload.get("call_id") or ""
            output = payload.get("output")
            if isinstance(output, dict):
                output = json.dumps(output, default=str)
            body = _clamp_content(str(output or ""), report)
            builder.add(
                "tool",
                body,
                stamp=stamp,
                tool_call_id=call_id or None,
                tool_name=call_names.get(call_id),
            )

    first_user = next((m.content for m in builder.messages if m.role == "user" and m.content), None)
    session = ConversationSession(
        session_id=_deterministic_session_id(path),
        title=(first_user[:50] if first_user else "Codex import"),
        messages=builder.messages,
        model=model,
        provider="codex",
        created_at=builder.first_ts or datetime.now(timezone.utc),
    )
    _annotate(report, builder, session, TranscriptFormat.CODEX_ROLLOUT)
    report.session_id = session.session_id
    return ImportedSession(session=session, report=report, source_path=path)


# ---------------------------------------------------------------------------
# ChatGPT export


def _deepest_leaf(mapping: Dict[str, Any]) -> Optional[str]:
    """Deepest root→leaf node id; cycle-safe (visited guard)."""
    depth: Dict[str, int] = {}
    children: Dict[str, List[str]] = {}
    for node_id, node in mapping.items():
        parent = node.get("parent") if isinstance(node, dict) else None
        if parent is None:
            depth[node_id] = 0
        else:
            children.setdefault(parent, []).append(node_id)
    visited = set(depth)
    queue = list(depth)
    while queue:
        next_queue = []
        for nid in queue:
            for child in children.get(nid, []):
                if child in visited:
                    continue
                visited.add(child)
                depth[child] = depth[nid] + 1
                next_queue.append(child)
        queue = next_queue
    if not depth:
        return None
    return max(depth, key=lambda nid: (depth[nid], nid))


def _read_chatgpt(data: Dict[str, Any], path: Path, report: ConversionReport) -> ImportedSession:
    builder = _TurnBuilder()
    mapping = data.get("mapping") or {}
    node_id = data.get("current_node")
    if not node_id or node_id not in mapping:
        node_id = _deepest_leaf(mapping)
        if node_id:
            report.note_warning("current_node missing; used deepest branch")
    chain: List[Tuple[str, Dict[str, Any]]] = []
    while node_id and node_id in mapping:
        node = mapping[node_id]
        message = node.get("message")
        if message and message.get("author", {}).get("role") in ("user", "assistant"):
            chain.append((node_id, message))
        node_id = node.get("parent")

    for node_id, message in reversed(chain):
        if builder.cap_reached():
            report.note_warning(f"stopped at {MAX_MESSAGES} messages (guard)")
            break
        content = message.get("content", {})
        if content.get("content_type") != "text":
            report.note_skip(
                0,
                message.get("author", {}).get("role", "?"),
                f"content_type={content.get('content_type')}",
            )
            continue
        parts = content.get("parts", [])
        text = "\n".join(p if isinstance(p, str) else str(p) for p in parts if p).strip()
        if not text:
            continue
        create_time = message.get("create_time")
        stamp = datetime.fromtimestamp(create_time, tz=timezone.utc) if create_time else None
        builder.add(
            message["author"]["role"],
            _clamp_content(text, report),
            stamp=stamp,
            msg_id=node_id,
        )
        report.entries_read += 1

    session = ConversationSession(
        session_id=_deterministic_session_id(path),
        title=data.get("title") or "ChatGPT import",
        messages=builder.messages,
        model=data.get("default_model_slug"),
        provider="openai",
        created_at=builder.first_ts or datetime.now(timezone.utc),
    )
    _annotate(report, builder, session, TranscriptFormat.CHATGPT_EXPORT)
    report.session_id = session.session_id
    return ImportedSession(session=session, report=report, source_path=path)


# ---------------------------------------------------------------------------
# Victor / Anthropic / generic JSON documents


def _read_json_document(path: Path, report: ConversionReport) -> ImportedSession:
    raw = path.read_text(encoding="utf-8")
    doc = json.loads(raw)  # detection already validated the shape prefix
    if isinstance(doc, list):
        # Legacy `victor session export` array — convert the first session and
        # note the rest.
        if len(doc) > 1:
            report.note_warning(f"export array held {len(doc)} sessions; imported the first")
        doc = doc[0] if doc else {}
        report.source_format = TranscriptFormat.VICTOR_JSON.value

    builder = _TurnBuilder()
    session_id: Optional[str] = None
    title: Optional[str] = None
    model: Optional[str] = None
    provider: Optional[str] = None
    created_at: Optional[datetime] = None

    if isinstance(doc.get("conversation"), dict):
        # Legacy export payload: {metadata, conversation}
        metadata = doc.get("metadata") or {}
        session_id = metadata.get("session_id")
        title = metadata.get("title")
        model = metadata.get("model")
        provider = metadata.get("provider")
        created_at = _parse_ts(metadata.get("created_at"))
        message_dicts = doc["conversation"].get("messages", [])
    else:
        session_id = doc.get("session_id")
        title = doc.get("title")
        model = doc.get("model")
        provider = doc.get("provider")
        created_at = _parse_ts(doc.get("created_at"))
        message_dicts = doc.get("messages", [])

    for msg in message_dicts:
        if not isinstance(msg, dict):
            continue
        if builder.cap_reached():
            report.note_warning(f"stopped at {MAX_MESSAGES} messages (guard)")
            break
        role = msg.get("role", "user")
        if role not in ("user", "assistant", "tool", "tool_call", "system"):
            report.note_skip(0, role, "unknown role")
            continue
        content = msg.get("content")
        if isinstance(content, list):
            content = "\n".join(b.get("text", "") for b in content if isinstance(b, dict))
        text = _clamp_content(str(content or ""), report)
        if not text and not msg.get("tool_calls"):
            continue
        builder.add(
            role,
            text,
            stamp=_parse_ts(msg.get("timestamp")),
            msg_id=msg.get("id"),
            tool_name=msg.get("tool_name"),
            tool_call_id=msg.get("tool_call_id"),
            tool_calls=msg.get("tool_calls"),
        )
        if msg.get("tool_calls"):
            report.tool_calls_preserved += 1
        report.entries_read += 1

    first_user = next((m.content for m in builder.messages if m.role == "user" and m.content), None)
    session = ConversationSession(
        session_id=session_id or _deterministic_session_id(path),
        title=title or (first_user[:50] if first_user else "Imported session"),
        messages=builder.messages,
        model=model,
        provider=provider or "unknown",
        created_at=created_at or builder.first_ts or datetime.now(timezone.utc),
    )
    _annotate(report, builder, session, TranscriptFormat.VICTOR_JSON)
    report.session_id = session.session_id
    return ImportedSession(session=session, report=report, source_path=path)


def _annotate(
    report: ConversionReport,
    builder: _TurnBuilder,
    session: ConversationSession,
    source: TranscriptFormat,
) -> None:
    report.source_format = source.value
    for message in builder.messages:
        report.count_role(message.role if isinstance(message.role, str) else message.role.value)
    if builder.first_ts and builder.last_ts:
        report.time_span = f"{builder.first_ts.isoformat()} → {builder.last_ts.isoformat()}"
    report.model = session.model


# ---------------------------------------------------------------------------
# public API


def import_transcript(path: Path | str, fmt: Optional[TranscriptFormat] = None) -> ImportedSession:
    """Read any supported transcript into the victor IR.

    Args:
        path: transcript file path.
        fmt: expected format; None auto-detects. An explicit flag that does
            not match the file content raises UnsupportedFormatError.

    Raises:
        UnsupportedFormatError: unknown format, size guard, or flag mismatch.
    """
    path = Path(path).expanduser()
    if not path.is_file():
        raise UnsupportedFormatError(f"File not found: {path}")
    if path.stat().st_size > MAX_FILE_BYTES:
        raise UnsupportedFormatError(f"File exceeds {MAX_FILE_BYTES // (1024 * 1024)} MB guard")
    detection: DetectionResult = detect_format(path)
    resolved = fmt or detection.format
    if fmt is not None and fmt != detection.format and detection.confidence >= 0.7:
        raise UnsupportedFormatError(
            f"--format {fmt.value} does not match detected "
            f"{detection.format.value} ({'; '.join(detection.reasons) or 'no signature'})",
            detection=detection,
        )
    report = ConversionReport(source_format=resolved.value, target_format="victor")

    if resolved is TranscriptFormat.CLAUDE_CODE:
        return _read_claude(path, report)
    if resolved is TranscriptFormat.CODEX_ROLLOUT:
        return _read_codex(path, report)

    # Document formats need full-file JSON; detection validated the prefix.
    doc = json.loads(path.read_text(encoding="utf-8"))
    if resolved is TranscriptFormat.CHATGPT_EXPORT:
        if isinstance(doc, list):
            if len(doc) > 1:
                report.note_warning(f"conversations.json held {len(doc)} chats; imported the first")
            doc = doc[0]
        return _read_chatgpt(doc, path, report)
    return _read_json_document(path, report)
