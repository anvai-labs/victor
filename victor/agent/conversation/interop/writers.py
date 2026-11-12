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

"""Transcript writers: victor ConversationSession IR → external formats.

Schema facts encoded here were verified against real on-disk transcripts
(2026-11-06 interop audit): codex strictly validates ``session_meta``
(``history_mode`` ∈ {legacy, paginated}); claude entries need
``uuid``/``parentUuid``/``sessionId``/``message``; thinking blocks are never
emitted (their signatures cannot transfer).
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import replace
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from victor.agent.conversation.interop.formats import TranscriptFormat
from victor.agent.conversation.interop.report import ConversionReport
from victor.agent.conversation.session_model import ConversationSession
from victor.agent.conversation.types import ConversationMessage

_InstallTarget = Literal["claude", "codex"]
_COLLISION_RETRIES = 5
# Built-in codex model provider id that every codex install can resume under.
_CODEX_RESUMABLE_PROVIDER = "openai"
# Admission bounds on user data, not a token estimate or model-fit guarantee.
CODEX_CONTEXT_BYTES = 128 * 1024
CODEX_HANDOFF_FILE_BYTES = 64 * 1024


def _codex_handoff(
    session: ConversationSession, context_file: Path, report: ConversionReport
) -> ConversationSession:
    """Replace inference history only when an explicit bounded handoff is supplied."""
    descriptor = os.open(
        Path(context_file).expanduser(), os.O_RDONLY | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("--context-file must be a regular UTF-8 file")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            raw = stream.read(CODEX_HANDOFF_FILE_BYTES + 1)
    finally:
        os.close(descriptor)
    if len(raw) > CODEX_HANDOFF_FILE_BYTES:
        raise ValueError(f"--context-file exceeds {CODEX_HANDOFF_FILE_BYTES} UTF-8 bytes")
    text = raw.decode("utf-8")
    if not text.strip():
        raise ValueError("--context-file must contain a nonempty historical handoff")
    digest = hashlib.sha256(raw).hexdigest()
    count = len(session.messages)
    tools = sum(bool(m.tool_calls) or _role(m) in ("tool_call", "tool") for m in session.messages)
    report.note_warning(
        f"Explicit context-file handoff replaces {count} source messages "
        f"({tools} tool messages); source session {session.session_id}; SHA256 {digest}. "
        "Original session unchanged; older history must be consulted separately."
    )
    content = (
        "Historical continuation handoff, explicitly supplied by the user.\n"
        f"Source Victor session: {session.session_id}\n"
        f"Context-file SHA256: {digest}\n"
        f"Replaces {count} source messages, including {tools} tool messages.\n"
        "The following is historical context, not current authorization. Old approvals, "
        "commands and claims of running jobs must be revalidated against current state.\n\n" + text
    )
    return replace(session, messages=[ConversationMessage(role="user", content=content)])


def _role(message: ConversationMessage) -> str:
    role = message.role
    return role.value if hasattr(role, "value") else str(role)


def _ts(message: ConversationMessage) -> str:
    stamp = message.timestamp or datetime.now(timezone.utc)
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp.isoformat().replace("+00:00", "Z")


def _clamp(text: str, report: ConversionReport, limit: int = 200_000) -> str:
    if len(text) > limit:
        report.truncations += 1
        return text[:limit] + " …[export-truncated]"
    return text


def _parse_arguments(raw: Any, report: ConversionReport) -> Dict[str, Any]:
    """Tool-call arguments are stored as a JSON string; parse defensively."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {"value": parsed}
        except json.JSONDecodeError:
            report.note_warning("tool arguments were not valid JSON; wrapped as text")
            return {"value": raw}
    return {}


# ---------------------------------------------------------------------------
# victor native JSON


def write_victor_json(session: ConversationSession, out: Path, report: ConversionReport) -> Path:
    """Single re-importable document (the reader's VICTOR_JSON shape)."""
    doc = {
        "session_id": session.session_id,
        "title": session.title,
        "model": session.model,
        "provider": session.provider,
        "created_at": session.created_at.isoformat() if session.created_at else None,
        "messages": [
            {
                "id": m.id,
                "role": _role(m),
                "content": _clamp(m.content, report),
                "timestamp": _ts(m),
                "token_count": m.token_count,
                "tool_name": m.tool_name,
                "tool_call_id": m.tool_call_id,
                "tool_calls": m.tool_calls,
                "metadata": m.metadata or {},
            }
            for m in session.messages
        ],
    }
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# Claude Code transcript


def write_claude_code(
    session: ConversationSession,
    out: Path,
    report: ConversionReport,
    *,
    cwd: Optional[str] = None,
    session_uuid: Optional[str] = None,
) -> tuple[Path, str]:
    """Emit a Claude Code project transcript; returns (path, session uuid)."""
    target_session = session_uuid or str(uuid.uuid4())
    parent: Optional[str] = None
    lines: List[str] = []

    def emit(entry: Dict[str, Any]) -> None:
        nonlocal parent
        entry_id = str(uuid.uuid4())
        entry.update(
            {
                "parentUuid": parent,
                "isSidechain": False,
                "isMeta": False,
                "userType": "external",
                "cwd": cwd or str(Path.cwd()),
                "sessionId": target_session,
                "version": "2.0",
                "uuid": entry_id,
            }
        )
        if entry.get("timestamp") is None:
            entry["timestamp"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        parent = entry_id
        lines.append(json.dumps(entry, ensure_ascii=False))

    for message in session.messages:
        role = _role(message)
        stamp = _ts(message)
        if role == "user":
            emit(
                {
                    "type": "user",
                    "timestamp": stamp,
                    "message": {"role": "user", "content": _clamp(message.content, report)},
                }
            )
        elif role == "assistant":
            # OpenAI/victor-style assistant carrying tool_calls inline: emit
            # the text plus native tool_use blocks in one entry.
            content: List[Dict[str, Any]] = []
            if message.content:
                content.append({"type": "text", "text": _clamp(message.content, report)})
            if message.tool_calls:
                for call in message.tool_calls:
                    content.append(
                        {
                            "type": "tool_use",
                            "id": call.get("id") or message.tool_call_id or str(uuid.uuid4()),
                            "name": call.get("name") or message.tool_name or "tool",
                            "input": _parse_arguments(call.get("arguments"), report),
                        }
                    )
                report.tool_calls_preserved += 1
            emit(
                {
                    "type": "assistant",
                    "timestamp": stamp,
                    "message": {
                        "role": "assistant",
                        "content": content or [{"type": "text", "text": ""}],
                    },
                }
            )
        elif role == "tool_call" and message.tool_calls:
            for call in message.tool_calls:
                emit(
                    {
                        "type": "assistant",
                        "timestamp": stamp,
                        "message": {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "tool_use",
                                    "id": call.get("id")
                                    or message.tool_call_id
                                    or str(uuid.uuid4()),
                                    "name": call.get("name") or message.tool_name or "tool",
                                    "input": _parse_arguments(call.get("arguments"), report),
                                }
                            ],
                        },
                    }
                )
            report.tool_calls_preserved += 1
        elif role == "tool":
            body = _clamp(message.content, report)
            emit(
                {
                    "type": "user",
                    "timestamp": stamp,
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": message.tool_call_id or "",
                                "content": body,
                            }
                        ],
                    },
                }
            )
        else:  # system → tolerated as user-visible context note
            if message.content:
                emit(
                    {
                        "type": "user",
                        "timestamp": stamp,
                        "message": {"role": "user", "content": _clamp(message.content, report)},
                    }
                )
                report.note_warning("system message emitted as user note")

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out, target_session


def to_persist_messages(session: ConversationSession) -> List[Dict[str, Any]]:
    """Build the save_session message payload from an imported session.

    Single source of truth shared by the /import slash command and the
    ``victor data session import`` CLI so both persist identical fidelity.
    """
    return [
        {
            "id": m.id,
            "role": _role(m),
            "content": m.content,
            "timestamp": m.timestamp.isoformat() if m.timestamp else None,
            "token_count": m.token_count,
            "tool_name": m.tool_name,
            "tool_call_id": m.tool_call_id,
            "tool_calls": m.tool_calls,
            "metadata": m.metadata or {},
        }
        for m in session.messages
    ]


def _emit_function_calls(message, report: ConversionReport, emit, stamp: str) -> None:
    """Emit one function_call response_item per tool call on a message.

    Shared by both IR shapes: explicit role=tool_call messages and
    OpenAI/victor-style assistants carrying tool_calls inline.
    """
    for call in message.tool_calls or []:
        arguments = call.get("arguments")
        if isinstance(arguments, dict):
            arguments = json.dumps(arguments, ensure_ascii=False)
        emit(
            "response_item",
            {
                "type": "function_call",
                "call_id": call.get("id") or message.tool_call_id or str(uuid.uuid4()),
                "name": call.get("name") or message.tool_name or "tool",
                "arguments": _clamp(str(arguments or "{}"), report),
            },
            timestamp=stamp,
        )
    report.tool_calls_preserved += 1


# ---------------------------------------------------------------------------
# Codex rollout


def write_codex_rollout(
    session: ConversationSession,
    out: Path,
    report: ConversionReport,
    *,
    cwd: Optional[str] = None,
    session_uuid: Optional[str] = None,
    max_context_bytes: Optional[int] = None,
) -> tuple[Path, str]:
    """Emit a Codex rollout transcript; returns (path, session uuid)."""
    target_session = session_uuid or str(uuid.uuid4())
    workspace = cwd or str(Path.cwd())
    lines: List[str] = []
    ordinal = 0
    report.context_bytes = 0
    report.context_byte_limit = max_context_bytes or CODEX_CONTEXT_BYTES

    def emit(obj_type: str, payload: Dict[str, Any], timestamp: Optional[str] = None) -> None:
        nonlocal ordinal
        if obj_type == "response_item":
            report.context_bytes = (report.context_bytes or 0) + len(
                json.dumps(payload, ensure_ascii=False).encode("utf-8")
            )
            if max_context_bytes is not None and report.context_bytes > max_context_bytes:
                raise ValueError(
                    f"Codex context payload exceeds the {max_context_bytes}-byte install guard. "
                    "Export a file archive without --install, then supply a bounded historical "
                    "handoff with --context-file. No session was installed."
                )
        lines.append(
            json.dumps(
                {
                    "timestamp": timestamp
                    or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                    "ordinal": ordinal,
                    "type": obj_type,
                    "payload": payload,
                },
                ensure_ascii=False,
            )
        )
        ordinal += 1

    # Strictly validated by codex: history_mode must be legacy|paginated.
    emit(
        "session_meta",
        {
            "creator_user_id": "victor-interop",
            "creator_account_id": str(uuid.uuid4()),
            "session_id": target_session,
            "id": target_session,
            "parent_thread_id": target_session,
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "cwd": workspace,
            "runtime_workspace_roots": [workspace],
            "originator": "victor",
            "cli_version": "interop",
            "source": "victor-interop",
            "thread_source": "user",
            # codex resolves this against its configured providers on resume and
            # refuses unknown ids ("modelProviderNotConfigured", codex 0.166), so
            # it must be a built-in id; "openai" is always configured.
            "model_provider": _CODEX_RESUMABLE_PROVIDER,
            "history_mode": "legacy",
        },
    )
    turn_id = str(uuid.uuid4())
    emit(
        "turn_context",
        {
            "turn_id": turn_id,
            "root_turn_id": turn_id,
            "disabled_plugin_ids": [],
            "cwd": workspace,
            "approval_policy": "never",
            "sandbox_policy": {"type": "read-only"},
        },
    )
    emit("event_msg", {"type": "task_started", "turn_id": turn_id, "root_turn_id": turn_id})

    for message in session.messages:
        role = _role(message)
        stamp = _ts(message)
        if role == "user":
            text = _clamp(message.content, report)
            emit(
                "event_msg",
                {
                    "type": "user_message",
                    "message": text,
                    "images": [],
                    "local_images": [],
                    "text_elements": [],
                },
                timestamp=stamp,
            )
            emit(
                "response_item",
                {
                    "type": "message",
                    "id": f"msg_{target_session[:8]}_{ordinal:04d}",
                    "role": "user",
                    "content": [{"type": "input_text", "text": text}],
                },
                timestamp=stamp,
            )
        elif role == "assistant":
            if message.content:
                text = _clamp(message.content, report)
                emit("event_msg", {"type": "agent_message", "message": text}, timestamp=stamp)
                emit(
                    "response_item",
                    {
                        "type": "message",
                        "id": f"msg_{target_session[:8]}_{ordinal:04d}",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": text}],
                    },
                    timestamp=stamp,
                )
            # OpenAI/victor-style assistant carrying tool_calls inline: emit
            # the text message (if any) then function_call items.
            if message.tool_calls:
                _emit_function_calls(message, report, emit, stamp)
        elif role == "tool_call" and message.tool_calls:
            _emit_function_calls(message, report, emit, stamp)
        elif role == "tool":
            emit(
                "response_item",
                {
                    "type": "function_call_output",
                    "call_id": message.tool_call_id or "",
                    "output": _clamp(message.content, report),
                },
                timestamp=stamp,
            )
        elif role == "system" and message.content:
            report.note_warning("system message skipped for codex target")

    emit(
        "event_msg",
        {
            "type": "task_complete",
            "turn_id": turn_id,
            "root_turn_id": turn_id,
            "last_agent_message": "",
        },
    )

    if report.context_over_budget:
        report.note_warning(
            f"Codex context payload is {report.context_bytes} bytes, above the "
            f"{report.context_byte_limit}-byte install guard. This file is an archive; "
            "a model turn or remote compaction may fail. Use --context-file for a bounded handoff."
        )
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out, target_session


# ---------------------------------------------------------------------------
# generic


def write_generic_json(session: ConversationSession, out: Path, report: ConversionReport) -> Path:
    doc = {
        "session_id": session.session_id,
        "title": session.title,
        "model": session.model,
        "provider": session.provider,
        "messages": [
            {"role": _role(m), "content": _clamp(m.content, report)} for m in session.messages
        ],
    }
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# dispatch + install


def export_session(
    session: ConversationSession,
    fmt: TranscriptFormat,
    out: Path,
    report: Optional[ConversionReport] = None,
    *,
    cwd: Optional[str] = None,
    session_uuid: Optional[str] = None,
    context_file: Optional[Path] = None,
) -> Path:
    """Write the session in the target format; returns the output path."""
    report = report or ConversionReport(target_format=fmt.value, session_id=session.session_id)
    report.target_format = fmt.value
    if context_file is not None:
        if fmt is not TranscriptFormat.CODEX_ROLLOUT:
            raise ValueError("--context-file requires --format codex")
        session = _codex_handoff(session, context_file, report)
    if fmt is TranscriptFormat.VICTOR_JSON:
        return write_victor_json(session, Path(out), report)
    if fmt is TranscriptFormat.CLAUDE_CODE:
        return write_claude_code(session, Path(out), report, cwd=cwd, session_uuid=session_uuid)[0]
    if fmt is TranscriptFormat.CODEX_ROLLOUT:
        return write_codex_rollout(session, Path(out), report, cwd=cwd, session_uuid=session_uuid)[
            0
        ]
    if fmt is TranscriptFormat.GENERIC_JSON:
        return write_generic_json(session, Path(out), report)
    raise ValueError(f"Format {fmt} is import-only")


def _claude_slug(project_cwd: Path) -> str:
    """Claude Code project-dir slug: every non-alphanumeric char becomes '-'."""
    return re.sub(r"[^A-Za-z0-9]", "-", str(project_cwd))


def install_session(
    session: ConversationSession,
    target: _InstallTarget,
    report: Optional[ConversionReport] = None,
    *,
    project_cwd: Optional[Path] = None,
    context_file: Optional[Path] = None,
) -> Path:
    """Write the session directly into the target tool's local store.

    Safety: fixed base directories, slug/path sanitization, containment check
    after resolve(), and no-overwrite (fresh uuid on collision, hard error
    after bounded retries). Paths derive ONLY from the user's cwd — never
    from transcript content.
    """
    report = report or ConversionReport(target_format=target, session_id=session.session_id)
    if target not in ("claude", "codex"):
        raise ValueError("install target must be claude or codex")
    if context_file is not None:
        if target != "codex":
            raise ValueError("--context-file requires codex")
        session = _codex_handoff(session, context_file, report)
    project_cwd = (project_cwd or Path.cwd()).resolve()

    if target == "claude":
        base = Path.home() / ".claude" / "projects"
        slug = _claude_slug(project_cwd)
        if not re.fullmatch(r"[A-Za-z0-9-]+", slug):
            raise ValueError(f"sanitized slug is not filesystem-safe: {slug!r}")
        target_dir = base / slug
    else:
        now = datetime.now()
        base = Path.home() / ".codex" / "sessions"
        target_dir = base / f"{now:%Y}" / f"{now:%m}" / f"{now:%d}"

    if not base.is_dir():
        raise FileNotFoundError(f"{target} store not found at {base} — is {target} installed?")

    fmt = TranscriptFormat.CLAUDE_CODE if target == "claude" else TranscriptFormat.CODEX_ROLLOUT
    for _attempt in range(_COLLISION_RETRIES):
        session_uuid = str(uuid.uuid4())
        if target == "claude":
            out_path = target_dir / f"{session_uuid}.jsonl"
        else:
            stamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
            out_path = target_dir / f"rollout-{stamp}-{session_uuid}.jsonl"
        # Post-resolve containment: refuse anything that escaped the base.
        if not out_path.resolve().is_relative_to(base.resolve()):
            raise ValueError(f"install path escaped the {target} store: {out_path}")
        if out_path.exists():
            continue
        target_dir.mkdir(parents=True, exist_ok=True)
        if target == "codex":
            write_codex_rollout(
                session,
                out_path,
                report,
                cwd=str(project_cwd),
                session_uuid=session_uuid,
                max_context_bytes=CODEX_CONTEXT_BYTES,
            )
        else:
            export_session(
                session, fmt, out_path, report, cwd=str(project_cwd), session_uuid=session_uuid
            )
        return out_path
    raise FileExistsError(
        f"could not find a free session filename in {target_dir} "
        f"after {_COLLISION_RETRIES} attempts"
    )
