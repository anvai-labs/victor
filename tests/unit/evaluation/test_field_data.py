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

"""Unit tests for the ADR-036 field-data collection harness.

Covers the graduation-evidence contract: opt-in (default OFF writes nothing),
anonymized records (no conversation content, no raw session ids — grep-pinned),
size-capped daily JSONL rotation, and the aggregate/evidence-table the ADR-036
queue reads.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace, SimpleNamespace as NS

import pytest

from victor.evaluation.field_data import (
    DEFAULT_FIELD_DATA_MAX_FILE_BYTES,
    FIELD_DATA_ENV,
    FIELD_DATA_FLAGS,
    FieldDataRecorder,
    aggregate_field_data,
    anonymize_session_id,
    field_data_wiring_audit,
    format_evidence_table,
    read_store,
    resolve_field_data_enabled,
    resolve_policy_flags,
)


def _settings(enabled: bool = True, **overrides) -> NS:
    group = NS(
        field_data_enabled=enabled,
        field_data_store_dir=None,
        field_data_max_file_bytes=DEFAULT_FIELD_DATA_MAX_FILE_BYTES,
    )
    for k, v in overrides.items():
        setattr(group, k, v)
    return NS(evaluation=group)


def _store_files(store_dir: Path) -> list[Path]:
    return sorted(store_dir.glob("field_data-*.jsonl"))


# ── resolver: opt-in, default off ─────────────────────────────────────────


def test_resolver_defaults_off_without_settings_or_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(FIELD_DATA_ENV, raising=False)
    assert resolve_field_data_enabled(None) is False
    assert resolve_field_data_enabled(SimpleNamespace()) is False
    assert resolve_field_data_enabled(SimpleNamespace(evaluation=None)) is False


def test_resolver_reads_settings_group_and_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(FIELD_DATA_ENV, raising=False)
    assert resolve_field_data_enabled(_settings(enabled=True)) is True
    assert resolve_field_data_enabled(_settings(enabled=False)) is False
    monkeypatch.setenv(FIELD_DATA_ENV, "true")
    assert resolve_field_data_enabled(_settings(enabled=False)) is True
    monkeypatch.setenv(FIELD_DATA_ENV, "false")
    assert resolve_field_data_enabled(_settings(enabled=True)) is False


# ── recorder: writes when enabled, nothing when disabled ──────────────────


def test_disabled_recorder_writes_nothing(tmp_path: Path) -> None:
    recorder = FieldDataRecorder(settings=_settings(enabled=False), store_dir=tmp_path)
    assert recorder.active is False
    assert (
        recorder.record_session_start(session_id="s1", mode="auto", provider="p", model="m") is None
    )
    assert recorder.record_session_end("s1", completion_status="completed") is None
    assert _store_files(tmp_path) == []


def test_enabled_recorder_writes_paired_start_and_end_records(tmp_path: Path) -> None:
    recorder = FieldDataRecorder(settings=_settings(enabled=True), store_dir=tmp_path)
    start = recorder.record_session_start(
        session_id="session-abc",
        mode="auto",
        provider="ollama",
        model="qwen3-coder-tools:30b",
        flags={"per_turn_auditor": True, "codebase_intent_indexing_enabled": False},
    )
    end = recorder.record_session_end(
        "session-abc",
        completion_status="completed",
        turn_count=4,
        tool_call_count=11,
    )
    assert start is not None and end is not None
    files = _store_files(tmp_path)
    assert len(files) == 1
    records = [json.loads(line) for line in files[0].read_text().splitlines() if line.strip()]
    assert [r["event"] for r in records] == ["session_start", "session_end"]

    assert start["schema_version"] == 1
    assert start["mode"] == "auto"
    assert start["provider"] == "ollama"
    assert start["model"] == "qwen3-coder-tools:30b"
    assert start["flags"] == {
        "per_turn_auditor": True,
        "codebase_intent_indexing_enabled": False,
    }
    # The stored identifier is the anonymized hash — never the raw id.
    assert start["session_hash"] == anonymize_session_id("session-abc")
    assert "session-abc" not in files[0].read_text()

    assert end["completion_status"] == "completed"
    assert end["turn_count"] == 4
    assert end["tool_call_count"] == 11
    assert end["duration_seconds"] is not None and end["duration_seconds"] >= 0.0


def test_session_hash_is_anonymized_and_stable() -> None:
    h = anonymize_session_id("some-internal-session-id")
    assert re.fullmatch(r"[0-9a-f]{16}", h)
    assert "some-internal-session-id" not in h
    assert anonymize_session_id("some-internal-session-id") == h
    assert anonymize_session_id("other") != h


def test_recorder_resolves_flags_through_production_seams(tmp_path: Path) -> None:
    settings = _settings(enabled=True)
    settings.agent = SimpleNamespace(per_turn_auditor=True, effect_gated_completion=False)
    settings.search = SimpleNamespace(graph=SimpleNamespace(codebase_intent_indexing_enabled=True))
    recorder = FieldDataRecorder(settings=settings, store_dir=tmp_path)
    start = recorder.record_session_start(session_id="s", mode="auto")
    assert start["flags"] == {
        "per_turn_auditor": True,
        "codebase_intent_indexing_enabled": True,
    }


def test_resolve_policy_flags_defaults_to_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VICTOR_PER_TURN_AUDITOR", raising=False)
    flags = resolve_policy_flags(None)
    assert flags == {"per_turn_auditor": False, "codebase_intent_indexing_enabled": False}


def test_resolve_policy_flags_reads_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VICTOR_PER_TURN_AUDITOR", "true")
    flags = resolve_policy_flags(None)
    assert flags["per_turn_auditor"] is True
    assert flags["codebase_intent_indexing_enabled"] is False


# ── rotation at the size cap ──────────────────────────────────────────────


def test_rotation_creates_capped_segments(tmp_path: Path) -> None:
    recorder = FieldDataRecorder(
        settings=_settings(enabled=True),
        store_dir=tmp_path,
        max_file_bytes=400,
    )
    for i in range(40):
        recorder.record_session_start(
            session_id=f"s{i}",
            mode="auto",
            provider="ollama",
            model="m:1b",
            flags={"per_turn_auditor": False, "codebase_intent_indexing_enabled": False},
        )
    files = _store_files(tmp_path)
    assert len(files) > 1, "size cap must rotate onto a new segment"
    for f in files[:-1]:
        assert f.stat().st_size <= 400, f.name
    records = read_store(tmp_path)
    assert len(records) == 40
    assert [r["session_hash"] for r in records][:1] == [anonymize_session_id("s0")]


def test_default_rotation_cap_is_documented() -> None:
    assert DEFAULT_FIELD_DATA_MAX_FILE_BYTES >= 1024 * 1024


# ── aggregation + evidence table ──────────────────────────────────────────


def _fixture_records() -> list[dict]:
    records: list[dict] = []

    def start(h: str, ts: float, auditor: bool) -> None:
        records.append(
            {
                "schema_version": 1,
                "event": "session_start",
                "ts": ts,
                "session_hash": h,
                "mode": "auto",
                "provider": "ollama",
                "model": "m:1b",
                "flags": {
                    "per_turn_auditor": auditor,
                    "codebase_intent_indexing_enabled": False,
                },
            }
        )

    def end(h: str, ts: float, status: str, turns: int, tools: int, ok: bool) -> None:
        records.append(
            {
                "schema_version": 1,
                "event": "session_end",
                "ts": ts,
                "session_hash": h,
                "completion_status": status,
                "success": ok,
                "turn_count": turns,
                "duration_seconds": round(ts % 100, 3),
                "tool_call_count": tools,
                "error_count": 1 if status == "error" else 0,
                "error_kind": "ProviderError" if status == "error" else None,
            }
        )

    start("a" * 16, 10.0, True)
    end("a" * 16, 20.0, "completed", 6, 12, True)
    start("b" * 16, 11.0, True)
    end("b" * 16, 25.0, "error", 3, 2, False)
    start("c" * 16, 12.0, False)
    end("c" * 16, 30.0, "cancelled", 2, 1, False)
    start("d" * 16, 13.0, False)
    end("d" * 16, 40.0, "completed", 8, 20, True)
    start("e" * 16, 14.0, True)  # no end record → no_end
    return records


def test_aggregate_joins_start_and_end_and_counts_outcomes() -> None:
    agg = aggregate_field_data(_fixture_records())
    assert agg["session_count"] == 5
    assert agg["status_counts"] == {
        "completed": 2,
        "error": 1,
        "cancelled": 1,
        "no_end": 1,
    }
    on = agg["flags"]["per_turn_auditor"]["on"]
    assert on["sessions"] == 3
    assert on["succeeded"] == 1
    assert on["completed"] == 1
    assert on["error"] == 1
    assert on["no_end"] == 1
    assert on["mean_turns"] == pytest.approx((6 + 3) / 2)
    off = agg["flags"]["per_turn_auditor"]["off"]
    assert off["sessions"] == 2
    assert off["succeeded"] == 1
    assert off["completed"] == 1
    assert off["cancelled"] == 1
    assert agg["flags"]["codebase_intent_indexing_enabled"]["on"]["sessions"] == 0


def test_aggregate_reads_a_real_store_directory(tmp_path: Path) -> None:
    settings = _settings(enabled=True)
    settings.agent = SimpleNamespace(per_turn_auditor=True)
    settings.search = SimpleNamespace(graph=SimpleNamespace(codebase_intent_indexing_enabled=False))
    recorder = FieldDataRecorder(settings=settings, store_dir=tmp_path)
    recorder.record_session_start(session_id="x", mode="auto", provider="p", model="m")
    recorder.record_session_end("x", completion_status="completed", turn_count=2, success=True)
    agg = aggregate_field_data(read_store(tmp_path))
    assert agg["session_count"] == 1
    assert agg["flags"]["per_turn_auditor"]["on"]["succeeded"] == 1


def test_evidence_table_lists_both_flag_arms() -> None:
    table = format_evidence_table(aggregate_field_data(_fixture_records()))
    for flag in FIELD_DATA_FLAGS:
        assert flag in table
    for status in ("succeeded", "completed", "cancelled", "error", "no_end"):
        assert status in table
    assert "sessions" in table
    # Markdown table rows: one per flag × arm.
    rows = [line for line in table.splitlines() if line.startswith("| per_turn_auditor")]
    assert len(rows) == 2
    on_row = next(r for r in rows if "| on |" in r)
    off_row = next(r for r in rows if "| off |" in r)
    assert "| 3 |" in on_row
    assert "| 2 |" in off_row


def test_aggregate_ignores_orphan_end_records() -> None:
    records = [
        {
            "schema_version": 1,
            "event": "session_end",
            "ts": 5.0,
            "session_hash": "z" * 16,
            "completion_status": "completed",
            "turn_count": 1,
            "duration_seconds": 1.0,
            "tool_call_count": 0,
            "error_count": 0,
            "error_kind": None,
        }
    ]
    agg = aggregate_field_data(records)
    assert agg["session_count"] == 0
    assert agg["status_counts"] == {}


# ── privacy: grep-pinned ──────────────────────────────────────────────────


def test_store_never_contains_conversation_content_or_user_identifiers(
    tmp_path: Path,
) -> None:
    """Grep-pin: the store holds booleans/counters/durations only.

    A plausible conversation transcript, a raw session id, and a code snippet
    are planted around the API surface; none may appear in the JSONL store.
    """
    secrets = [
        "SECRET-PROMPT-CONTENT-the-user-typed",
        "def this_function_must_never_leak():",
        "raw-plaintext-session-id-xyz",
        "sk-live-user-api-key",
    ]
    recorder = FieldDataRecorder(settings=_settings(enabled=True), store_dir=tmp_path)
    recorder.record_session_start(
        session_id=secrets[2],
        mode="auto",
        provider="ollama",
        model="m:1b",
        flags={"per_turn_auditor": True, "codebase_intent_indexing_enabled": False},
    )
    recorder.record_session_end(
        "raw-plaintext-session-id-xyz",
        completion_status="completed",
        turn_count=3,
        tool_call_count=4,
    )
    dumped = "\n".join(f.read_text() for f in _store_files(tmp_path))
    for secret in secrets:
        assert secret not in dumped, f"privacy leak: {secret}"
    for record in (json.loads(line) for line in dumped.splitlines() if line.strip()):
        # No free-text keys: every stored value is a bool, number, short
        # category word, or the anonymized hash.
        for key, value in record.items():
            if isinstance(value, str):
                assert len(value) <= 64, (key, value)
                assert key in {
                    "event",
                    "session_hash",
                    "mode",
                    "provider",
                    "model",
                    "completion_status",
                    "error_kind",
                }, key


# ── placement audit (dormancy guard) ──────────────────────────────────────


def test_field_data_wiring_audit_pins_the_real_session_path() -> None:
    """The recorder must be wired where the session actually runs.

    Dormancy lesson (FEP-0041): a flag whose resolver exists but whose build
    site never consults it is dead. run_unified is the SOLE live streaming
    entry point, so the audit pins the recorder's presence there.
    """
    audit = field_data_wiring_audit()
    assert audit == {
        "streaming_path_consults_field_data_switch": True,
        "streaming_path_resolves_field_data_flag": True,
        "streaming_path_records_session_start": True,
        "streaming_path_records_session_end": True,
    }
