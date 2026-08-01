"""Utilities for loading and sanitizing interactive input history."""

from __future__ import annotations

import datetime as _dt
import json
import sqlite3
from pathlib import Path

from victor.agent.conversation.history_metadata import (
    build_internal_history_metadata,
    is_hidden_from_interactive_history,
)

_INTERNAL_HISTORY_PREFIXES = {
    # FEP-0026 envelope. Hiding normally works off the message's metadata; this is
    # the fallback for rows that reach the store without it, which is not
    # hypothetical — two rebuild paths were found dropping metadata (#715).
    "<system-reminder": "system_reminder",
    # Legacy prefix, retained for conversations stored before the envelope.
    "[SYSTEM-REMINDER:": "system_reminder",
    "[GROUNDING-FEEDBACK:": "grounding_feedback",
    "[SYSTEM:": "system_message",
    "[TASK-HINT:": "task_hint",
}

_INTERNAL_HISTORY_MESSAGE_PREFIXES = {
    "Continue. Use appropriate tools if needed.": "continuation_prompt",
    "Continue your analysis. Use tools like ": "continuation_prompt",
    "Continue with the implementation. Use tools like ": "continuation_prompt",
    "Please provide a summary of your findings/work so far.": "request_summary",
    "CRITICAL: Provide your FINAL ANALYSIS NOW.": "force_completion",
    "You mentioned using ": "tool_call_recovery",
    "You are unable to make tool calls.": "tool_call_recovery",
    "The previous action did not complete.": "tool_call_recovery",
}

_INTERNAL_HISTORY_SUBSTRINGS = {
    "Call tools sequentially, waiting for results": "tool_rules",
    "GROUNDING:": "grounding_prompt",
    "Use ONLY content in <TOOL_OUTPUT>": "tool_output_prompt",
    "TOOL RULES:": "tool_rules",
}


def classify_internal_history_entry(content: str) -> str | None:
    """Classify obvious internal control-plane prompts that should stay hidden."""
    text = content.strip()
    if not text:
        return None

    for prefix, kind in _INTERNAL_HISTORY_PREFIXES.items():
        if text.startswith(prefix):
            return kind

    for prefix, kind in _INTERNAL_HISTORY_MESSAGE_PREFIXES.items():
        if text.startswith(prefix):
            return kind

    for fragment, kind in _INTERNAL_HISTORY_SUBSTRINGS.items():
        if fragment in text:
            return kind

    return None


def is_interactive_history_entry(content: str, max_length: int = 4000) -> bool:
    """Return True when content should appear in user-facing input history."""
    text = content.strip()
    if not text or len(text) > max_length:
        return False

    if text.startswith("<TOOL_OUTPUT") or text.startswith("<") or text.startswith("{"):
        return False

    if classify_internal_history_entry(text) is not None:
        return False

    return True


def _is_interactive_history_metadata(metadata_json: str | None) -> bool:
    """Return True when stored metadata allows a message in input history."""
    if not metadata_json:
        return True

    try:
        metadata = json.loads(metadata_json)
    except (TypeError, ValueError, json.JSONDecodeError):
        return True

    if not isinstance(metadata, dict):
        return True

    return not is_hidden_from_interactive_history(metadata)


def load_prompt_toolkit_history_entries(history_file: Path) -> list[str]:
    """Load prompt_toolkit FileHistory entries in file order (oldest first)."""
    if not history_file.exists():
        return []

    entries: list[str] = []
    lines: list[str] = []
    with history_file.open(encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            if raw_line.startswith("+"):
                lines.append(raw_line[1:])
                continue

            if lines:
                entries.append("".join(lines)[:-1])
                lines = []

        if lines:
            entries.append("".join(lines)[:-1])

    return entries


def write_prompt_toolkit_history_entries(history_file: Path, entries: list[str]) -> None:
    """Rewrite a prompt_toolkit FileHistory file from normalized entries."""
    history_file.parent.mkdir(parents=True, exist_ok=True)

    with history_file.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(f"\n# {_dt.datetime.now()}\n")
            for line in entry.split("\n"):
                handle.write(f"+{line}\n")


def sanitize_prompt_toolkit_history_file(history_file: Path, max_entries: int = 250) -> int:
    """Remove invalid history items and trim to the newest complete entries."""
    entries = load_prompt_toolkit_history_entries(history_file)
    if not entries:
        return 0

    filtered = [entry for entry in entries if is_interactive_history_entry(entry)]
    if max_entries > 0:
        filtered = filtered[-max_entries:]

    if filtered != entries:
        write_prompt_toolkit_history_entries(history_file, filtered)

    return len(filtered)


def count_prompt_toolkit_history_entries(history_file: Path) -> int:
    """Return the number of complete prompt_toolkit history entries."""
    return len(load_prompt_toolkit_history_entries(history_file))


def backfill_internal_history_metadata(db_path: Path, limit: int = 1000) -> int:
    """Mark obvious legacy internal prompts so future history loads can skip them cheaply."""
    if limit <= 0 or not db_path.exists():
        return 0

    updated = 0

    with sqlite3.connect(db_path) as conn:
        table_check = conn.execute("""
            SELECT 1
            FROM sqlite_master
            WHERE type = 'table'
              AND name = 'messages'
            """).fetchone()
        if table_check is None:
            return 0

        rows = conn.execute(
            """
            SELECT rowid, content, metadata
            FROM messages
            WHERE role = 'user'
              AND content IS NOT NULL
              AND content != ''
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

        for rowid, content, metadata_json in rows:
            kind = classify_internal_history_entry(content)
            if kind is None:
                continue

            metadata: dict[str, object]
            try:
                parsed = json.loads(metadata_json) if metadata_json else {}
            except (TypeError, ValueError, json.JSONDecodeError):
                parsed = {}

            metadata = parsed if isinstance(parsed, dict) else {}
            if is_hidden_from_interactive_history(metadata):
                continue

            metadata.update(build_internal_history_metadata(kind))
            conn.execute(
                "UPDATE messages SET metadata = ? WHERE rowid = ?",
                (json.dumps(metadata), rowid),
            )
            updated += 1

    return updated


def load_input_history_from_db(
    db_path: Path,
    limit: int = 100,
    repair_legacy_metadata: bool = True,
) -> list[str]:
    """Load recent unique user input from the conversation database.

    Optionally backfills metadata onto obvious legacy internal prompts so future
    loads and history exports do not rely on content heuristics alone.
    """
    if limit <= 0 or not db_path.exists():
        return []

    fetch_limit = max(limit * 10, 200)
    messages: list[str] = []
    seen: set[str] = set()

    if repair_legacy_metadata:
        backfill_internal_history_metadata(db_path, limit=fetch_limit)

    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute(
            """
            SELECT content, metadata
            FROM messages
            WHERE role = 'user'
              AND content IS NOT NULL
              AND content != ''
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (fetch_limit,),
        )

        for content, metadata_json in cursor.fetchall():
            if content in seen:
                continue
            if not _is_interactive_history_metadata(metadata_json):
                continue
            if not is_interactive_history_entry(content):
                continue
            seen.add(content)
            messages.append(content)
            if len(messages) >= limit:
                break

    messages.reverse()
    return messages
