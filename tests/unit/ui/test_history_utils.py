"""Tests for interactive history filtering and sanitization."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from victor.ui.history_utils import (
    backfill_internal_history_metadata,
    classify_internal_history_entry,
    is_interactive_history_entry,
    load_input_history_from_db,
    load_prompt_toolkit_history_entries,
    sanitize_prompt_toolkit_history_file,
    write_prompt_toolkit_history_entries,
)


def test_is_interactive_history_entry_filters_internal_prompts():
    assert is_interactive_history_entry("fix the failing test")
    assert not is_interactive_history_entry("[SYSTEM-REMINDER: use tools]")
    assert not is_interactive_history_entry("[GROUNDING-FEEDBACK: correction required]")
    assert not is_interactive_history_entry("Continue. Use appropriate tools if needed.")
    assert not is_interactive_history_entry(
        "- Call tools sequentially, waiting for results\n\nGROUNDING:\n- Use ONLY content in <TOOL_OUTPUT> markers."
    )
    assert not is_interactive_history_entry(
        "Please provide a summary of your findings/work so far. Conclude your response."
    )


def test_classify_internal_history_entry_assigns_prompt_kinds():
    assert classify_internal_history_entry("[SYSTEM-REMINDER: use tools]") == "system_reminder"
    assert (
        classify_internal_history_entry("Continue. Use appropriate tools if needed.")
        == "continuation_prompt"
    )
    assert classify_internal_history_entry("normal user prompt") is None


def test_load_input_history_from_db_filters_internal_and_keeps_recent_unique(
    tmp_path: Path,
):
    db_path = tmp_path / "project.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE messages (
                role TEXT,
                content TEXT,
                timestamp TEXT,
                metadata TEXT
            )
            """)
        conn.executemany(
            "INSERT INTO messages(role, content, timestamp, metadata) VALUES (?, ?, ?, ?)",
            [
                ("user", "first real prompt", "2026-04-26 10:00:00", None),
                ("user", "[SYSTEM-REMINDER: use tools]", "2026-04-26 10:01:00", None),
                ("user", "first real prompt", "2026-04-26 10:02:00", None),
                (
                    "user",
                    "Continue. Use appropriate tools if needed.",
                    "2026-04-26 10:03:00",
                    None,
                ),
                ("user", "second real prompt", "2026-04-26 10:04:00", None),
                (
                    "user",
                    "plain-looking hidden prompt",
                    "2026-04-26 10:04:30",
                    '{"interactive_history": false, "internal_prompt_kind": "prompt_tool_call"}',
                ),
                (
                    "user",
                    '<TOOL_OUTPUT tool="read">x</TOOL_OUTPUT>',
                    "2026-04-26 10:05:00",
                    None,
                ),
                ("assistant", "ignored assistant", "2026-04-26 10:06:00", None),
                ("user", "multiline\nreal prompt", "2026-04-26 10:07:00", None),
            ],
        )

    assert load_input_history_from_db(db_path, limit=10) == [
        "first real prompt",
        "second real prompt",
        "multiline\nreal prompt",
    ]


def test_load_input_history_from_db_backfills_legacy_internal_metadata(tmp_path: Path):
    db_path = tmp_path / "project.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE messages (
                role TEXT,
                content TEXT,
                timestamp TEXT,
                metadata TEXT
            )
            """)
        conn.executemany(
            "INSERT INTO messages(role, content, timestamp, metadata) VALUES (?, ?, ?, ?)",
            [
                ("user", "[SYSTEM-REMINDER: use tools]", "2026-04-26 10:01:00", None),
                ("user", "visible prompt", "2026-04-26 10:02:00", None),
            ],
        )

    assert load_input_history_from_db(db_path, limit=10) == ["visible prompt"]

    with sqlite3.connect(db_path) as conn:
        metadata_json = conn.execute(
            """
            SELECT metadata
            FROM messages
            WHERE content = ?
            """,
            ("[SYSTEM-REMINDER: use tools]",),
        ).fetchone()[0]

    metadata = json.loads(metadata_json)
    assert metadata["interactive_history"] is False
    assert metadata["internal_prompt_kind"] == "system_reminder"


def test_backfill_internal_history_metadata_preserves_existing_fields(tmp_path: Path):
    db_path = tmp_path / "project.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE messages (
                role TEXT,
                content TEXT,
                timestamp TEXT,
                metadata TEXT
            )
            """)
        conn.execute(
            "INSERT INTO messages(role, content, timestamp, metadata) VALUES (?, ?, ?, ?)",
            (
                "user",
                "Please provide a summary of your findings/work so far.",
                "2026-04-26 10:01:00",
                json.dumps({"source": "legacy"}),
            ),
        )

    assert backfill_internal_history_metadata(db_path, limit=10) == 1

    with sqlite3.connect(db_path) as conn:
        metadata_json = conn.execute("SELECT metadata FROM messages").fetchone()[0]

    metadata = json.loads(metadata_json)
    assert metadata["source"] == "legacy"
    assert metadata["interactive_history"] is False
    assert metadata["internal_prompt_kind"] == "request_summary"


def test_sanitize_prompt_toolkit_history_file_preserves_complete_multiline_entries(
    tmp_path: Path,
):
    history_file = tmp_path / "chat_history"
    history_file.write_text(
        "+[SYSTEM-REMINDER: leaked orphan entry]\n"
        "\n# 2026-04-26 10:00:00\n"
        "+first real prompt\n"
        "\n# 2026-04-26 10:01:00\n"
        "+Continue. Use appropriate tools if needed.\n"
        "\n# 2026-04-26 10:02:00\n"
        "+multi line 1\n"
        "+multi line 2\n",
        encoding="utf-8",
    )

    remaining = sanitize_prompt_toolkit_history_file(history_file, max_entries=10)

    assert remaining == 2
    assert load_prompt_toolkit_history_entries(history_file) == [
        "first real prompt",
        "multi line 1\nmulti line 2",
    ]


def test_sanitize_prompt_toolkit_history_file_trims_by_entries_not_lines(
    tmp_path: Path,
):
    history_file = tmp_path / "chat_history"
    write_prompt_toolkit_history_entries(
        history_file,
        [
            "one line",
            "two line a\ntwo line b",
            "three line a\nthree line b\nthree line c",
        ],
    )

    sanitize_prompt_toolkit_history_file(history_file, max_entries=2)

    assert load_prompt_toolkit_history_entries(history_file) == [
        "two line a\ntwo line b",
        "three line a\nthree line b\nthree line c",
    ]


class TestEnvelopedGuidanceStaysOutOfInputHistory:
    """Enveloped runtime guidance must not surface when the user presses ↑.

    Hiding normally works off ``build_internal_history_metadata``. But metadata is
    not guaranteed to survive: two rebuild paths were found silently dropping it
    (fixed in #715), and the content-prefix table exists precisely as the fallback
    for rows that reach the store without it.

    That table knew the legacy ``[SYSTEM-REMINDER:`` prefix. Once FEP-0026 replaced
    that prefix with the authenticated envelope, the fallback stopped matching, so
    a metadata-less row would have leaked runtime guidance into the user's history.
    """

    def test_enveloped_guidance_is_classified_as_internal(self):
        from victor.agent.control_plane import mint_channel_nonce, wrap_guidance
        from victor.ui.history_utils import classify_internal_history_entry

        enveloped = wrap_guidance("You are an expert coding assistant.", mint_channel_nonce())

        assert classify_internal_history_entry(enveloped) == "system_reminder"

    def test_legacy_prefix_still_classified_for_stored_conversations(self):
        from victor.ui.history_utils import classify_internal_history_entry

        assert classify_internal_history_entry("[SYSTEM-REMINDER: old row]") == "system_reminder"

    def test_ordinary_user_text_is_not_hidden(self):
        from victor.ui.history_utils import classify_internal_history_entry

        assert classify_internal_history_entry("implement the metrics registry") is None
