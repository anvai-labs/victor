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

"""ADR-036 field-data collection harness (opt-in, privacy-preserving).

ADR-036 gates every default-on flag flip on measured evidence. The graduation
batteries (``flag_ab`` / ``auditor_ab`` / ``memory_ab``) cover the deterministic
and live-run arms; what was missing is the **field** arm: what do the flags
actually do in real sessions once 0.32.0 is out? This module is that recorder.

**What is recorded, per session run** (one ``run_unified`` invocation = one
user message through the agentic loop):

- ``session_start`` — the ADR-036 queue's flag states as resolved through the
  PRODUCTION resolvers (``per_turn_auditor``,
  ``codebase_intent_indexing_enabled``; ``effect_gated_completion`` is
  deliberately NOT tracked — its graduation is LOCKED OUT by the 2026-08-03
  negative A/B), plus session metadata (executor mode, provider, model).
- ``session_end`` — outcome markers only: completion status
  (completed/cancelled/error), turn count, duration, tool-call count, error
  count and the exception CLASS name (never the message).

**What is never recorded**: conversation content, code, prompts, tool
arguments/results, token or cost figures, and raw session identifiers — the
session id is reduced to a truncated SHA-256 hash before it reaches the store.

**Store**: append-only JSONL under the Victor state directory's ``evaluation/``
subfolder (``~/.victor/evaluation`` by default; override
``evaluation.field_data_store_dir``), one file per day
(``field_data-YYYYMMDD.jsonl``), size-capped with numbered rotation segments.
Records are written best-effort: a recorder failure must never break a session.

**Strictly opt-in**: ``evaluation.field_data_enabled`` (env
``VICTOR_FIELD_DATA_ENABLED``) defaults OFF; a disabled recorder is a complete
no-op — nothing is written, nothing is resolved.

Aggregate into the ADR-036 evidence table with:

    victor evaluation field-data summary [--store-dir DIR] [--json]
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

#: Env override for the master switch (mirrors the VICTOR_* flag machinery).
FIELD_DATA_ENV = "VICTOR_FIELD_DATA_ENABLED"

#: The ADR-036 queue flags whose exposure is recorded. ``effect_gated_completion``
#: is intentionally absent: its graduation is LOCKED OUT by the negative A/B
#: (FOU-S4, #884) — there is no field question left to answer for it.
FIELD_DATA_FLAGS = ("per_turn_auditor", "codebase_intent_indexing_enabled")

FIELD_DATA_SCHEMA_VERSION = 1
SESSION_START_EVENT = "session_start"
SESSION_END_EVENT = "session_end"
STATUS_COMPLETED = "completed"
STATUS_CANCELLED = "cancelled"
STATUS_ERROR = "error"
STATUS_NO_END = "no_end"

#: Default per-segment size cap (5 MiB) before a day file rotates.
DEFAULT_FIELD_DATA_MAX_FILE_BYTES = 5 * 1024 * 1024


def resolve_field_data_enabled(settings: Any) -> bool:
    """Resolve the field-data master switch: env → evaluation group → default off.

    Mirrors ``resolve_per_turn_auditor_enabled``. Env override
    ``VICTOR_FIELD_DATA_ENABLED`` wins; otherwise
    ``settings.evaluation.field_data_enabled`` (default False).
    """
    env = os.environ.get(FIELD_DATA_ENV)
    if env is not None:
        return env.strip().lower() in ("1", "true", "yes", "on")
    return bool(getattr(getattr(settings, "evaluation", None), "field_data_enabled", False))


def default_store_dir() -> Path:
    """The Victor state dir ``evaluation/`` subfolder (``~/.victor/evaluation``)."""
    from victor.config.settings import GLOBAL_VICTOR_DIR

    return Path(GLOBAL_VICTOR_DIR) / "evaluation"


def anonymize_session_id(session_id: Any) -> str:
    """Reduce a session identifier to a truncated SHA-256 hash (16 hex chars).

    Session ids are random UUIDs in practice, so the truncated hash is both
    stable (joins start↔end records) and non-identifying.
    """
    return hashlib.sha256(str(session_id).encode("utf-8")).hexdigest()[:16]


def resolve_policy_flags(settings: Any) -> Dict[str, bool]:
    """Resolve the recorded flags through their PRODUCTION seams.

    ``per_turn_auditor`` via the framework resolver (env → AgentSettings → off),
    ``codebase_intent_indexing_enabled`` via ``graph_flag`` (the ONE effective
    intent switch, ``search.graph.…``). Returns every flag with a boolean so the
    store schema is stable regardless of what is enabled.
    """
    flags: Dict[str, bool] = {}
    try:
        from victor.framework.per_turn_auditor import resolve_per_turn_auditor_enabled

        flags["per_turn_auditor"] = bool(resolve_per_turn_auditor_enabled(settings))
    except Exception:  # noqa: BLE001 — a resolver gap must not break recording
        logger.debug("per_turn_auditor flag resolution failed", exc_info=True)
        flags["per_turn_auditor"] = False
    try:
        from victor.config.settings import graph_flag

        flags["codebase_intent_indexing_enabled"] = bool(
            graph_flag(settings, "codebase_intent_indexing_enabled", False)
        )
    except Exception:  # noqa: BLE001
        logger.debug("codebase_intent_indexing_enabled flag resolution failed", exc_info=True)
        flags["codebase_intent_indexing_enabled"] = False
    return flags


def field_data_wiring_audit() -> Dict[str, bool]:
    """Static placement audit: the recorder must be wired on the real session path.

    Guards the placement-not-existence class (FEP-0041 dormancy lesson): a
    recorder that exists but is never consulted by the session entry point is
    dead. ``StreamingChatExecutor.run_unified`` is the SOLE live streaming
    entry point, so the audit pins the opt-in switch consultation (env + group
    hint guarding the heavy victor.evaluation import), the production resolver
    call, and both record calls there.
    """
    from victor.agent.services import chat_stream_executor

    src = inspect.getsource(chat_stream_executor)
    return {
        "streaming_path_consults_field_data_switch": (
            "VICTOR_FIELD_DATA_ENABLED" in src and "field_data_enabled" in src
        ),
        "streaming_path_resolves_field_data_flag": "resolve_field_data_enabled" in src,
        "streaming_path_records_session_start": "record_session_start" in src,
        "streaming_path_records_session_end": "record_session_end" in src,
    }


class FieldDataRecorder:
    """Opt-in per-session recorder writing append-only JSONL to the state dir.

    Construct with ``settings=`` (the master switch resolves through
    :func:`resolve_field_data_enabled`), or pass ``enabled=`` directly for
    callers that manage the switch themselves. Every record method is a strict
    no-op while disabled; every write is best-effort (logged, never raised).
    """

    def __init__(
        self,
        *,
        settings: Any = None,
        store_dir: Optional[Path] = None,
        max_file_bytes: Optional[int] = None,
        enabled: Optional[bool] = None,
        clock: Any = time.time,
    ) -> None:
        group = getattr(settings, "evaluation", None)
        self.active: bool = (
            resolve_field_data_enabled(settings) if enabled is None else bool(enabled)
        )
        if store_dir is not None:
            self._store_dir = Path(store_dir)
        else:
            override = getattr(group, "field_data_store_dir", None)
            self._store_dir = Path(override) if override else default_store_dir()
        self._max_file_bytes = int(
            max_file_bytes
            if max_file_bytes is not None
            else (
                getattr(group, "field_data_max_file_bytes", None)
                or DEFAULT_FIELD_DATA_MAX_FILE_BYTES
            )
        )
        self._clock = clock
        self._settings = settings
        self._started: Dict[str, float] = {}

    @classmethod
    def from_settings(cls, settings: Any) -> "FieldDataRecorder":
        """Build from a Settings object (group + env resolved; no-op unless enabled)."""
        return cls(settings=settings)

    # ── recording ─────────────────────────────────────────────────────────

    def record_session_start(
        self,
        *,
        session_id: Any,
        mode: str = "",
        provider: str = "",
        model: str = "",
        settings: Any = None,
        flags: Optional[Dict[str, bool]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Record one session run's flag exposure + metadata. No-op when disabled."""
        if not self.active:
            return None
        session_hash = anonymize_session_id(session_id)
        ts = self._clock()
        self._started[session_hash] = ts
        if flags is None:
            flags = resolve_policy_flags(settings if settings is not None else self._settings)
        record = {
            "schema_version": FIELD_DATA_SCHEMA_VERSION,
            "event": SESSION_START_EVENT,
            "ts": round(float(ts), 3),
            "session_hash": session_hash,
            "mode": str(mode)[:64],
            "provider": str(provider)[:64],
            "model": str(model)[:64],
            "flags": {flag: bool(flags.get(flag, False)) for flag in FIELD_DATA_FLAGS},
        }
        self._append(record)
        return record

    def record_session_end(
        self,
        session_id: Any,
        *,
        completion_status: str,
        turn_count: int = 0,
        duration_seconds: Optional[float] = None,
        tool_call_count: int = 0,
        error_count: int = 0,
        error_kind: Optional[str] = None,
        success: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Record one session run's outcome markers. No-op when disabled.

        ``success`` is the loop's terminal verdict
        (``agentic_loop_success``): a ``completed`` status with a False
        verdict is a failed run, not a cancelled one — the ON-vs-OFF skew of
        that signal is exactly the demotion evidence ADR-036 reads.
        """
        if not self.active:
            return None
        session_hash = anonymize_session_id(session_id)
        ts = self._clock()
        start_ts = self._started.pop(session_hash, None)
        if duration_seconds is None and start_ts is not None:
            duration_seconds = max(0.0, float(ts - start_ts))
        record = {
            "schema_version": FIELD_DATA_SCHEMA_VERSION,
            "event": SESSION_END_EVENT,
            "ts": round(float(ts), 3),
            "session_hash": session_hash,
            "success": bool(success),
            "completion_status": str(completion_status)[:32],
            "turn_count": max(0, int(turn_count)),
            "duration_seconds": (
                round(float(duration_seconds), 3) if duration_seconds is not None else None
            ),
            "tool_call_count": max(0, int(tool_call_count)),
            "error_count": max(0, int(error_count)),
            "error_kind": str(error_kind)[:64] if error_kind else None,
        }
        self._append(record)
        return record

    # ── storage ───────────────────────────────────────────────────────────

    def _store_file(self, pending_bytes: int = 0) -> Path:
        """The day's JSONL segment, rotating onto the next number at the cap.

        Segments are uniformly numbered (``field_data-YYYYMMDD.NNN.jsonl``) so
        lexicographic file order equals chronological append order — the store
        reads back in sequence without a timestamp re-sort. Rotates when
        appending ``pending_bytes`` would push the current segment past the cap
        (so a segment only exceeds the cap when a single record alone is
        larger than it).
        """
        day = datetime.fromtimestamp(self._clock()).strftime("%Y%m%d")
        segment = 0
        candidate = self._store_dir / f"field_data-{day}.{segment:03d}.jsonl"
        while (
            candidate.exists() and candidate.stat().st_size + pending_bytes > self._max_file_bytes
        ):
            segment += 1
            candidate = self._store_dir / f"field_data-{day}.{segment:03d}.jsonl"
        return candidate

    def _append(self, record: Dict[str, Any]) -> None:
        """Append one JSONL line, best-effort — never raise into the session."""
        try:
            self._store_dir.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(record, sort_keys=True) + "\n"
            path = self._store_file(len(payload.encode("utf-8")))
            with path.open("a", encoding="utf-8") as fh:
                fh.write(payload)
        except Exception:  # noqa: BLE001 — field data must never break a session
            logger.debug("field-data append failed", exc_info=True)


# ── store reading + aggregation (the ADR-036 evidence table) ──────────────


def read_store(store_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Read every record from the store (all day files + rotation segments)."""
    root = Path(store_dir) if store_dir is not None else default_store_dir()
    records: List[Dict[str, Any]] = []
    if not root.exists():
        return records
    for path in sorted(root.glob("field_data-*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                logger.debug("skipping malformed field-data line in %s", path)
    # Append order IS chronological: day files and rotation segments sort
    # lexicographically into time order, so no timestamp re-sort is needed
    # (sub-second ties would make a ts sort unstable anyway).
    return records


def aggregate_field_data(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Join start/end records into per-session rows and aggregate per flag arm.

    Returns the dict ``format_evidence_table`` renders: overall session count,
    status distribution, and per flag an ``on``/``off`` exposure arm with
    outcome counts and mean duration/turns/tool calls. End records without a
    start (and vice versa) count as ``no_end`` / are ignored, never crash.
    """
    starts: Dict[str, Dict[str, Any]] = {}
    ends: Dict[str, Dict[str, Any]] = {}
    for record in records:
        event = record.get("event")
        session_hash = record.get("session_hash")
        if not isinstance(session_hash, str):
            continue
        if event == SESSION_START_EVENT:
            starts[session_hash] = record
        elif event == SESSION_END_EVENT:
            ends[session_hash] = record

    sessions: List[Dict[str, Any]] = []
    for session_hash, start in starts.items():
        end = ends.get(session_hash)
        sessions.append(
            {
                "flags": start.get("flags", {}),
                "status": (
                    end.get("completion_status", STATUS_NO_END)
                    if end is not None
                    else STATUS_NO_END
                ),
                "duration_seconds": end.get("duration_seconds") if end is not None else None,
                "turn_count": end.get("turn_count", 0) if end is not None else 0,
                "tool_call_count": end.get("tool_call_count", 0) if end is not None else 0,
                "success": bool(end.get("success", False)) if end is not None else False,
            }
        )

    def arm_stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Means run over sessions WITH an end record only — a no-end session
        # (crash/kill) has no outcome to average; its count is reported as-is.
        ended = [r for r in rows if r["status"] != STATUS_NO_END]
        durations = [r["duration_seconds"] for r in ended if r["duration_seconds"] is not None]
        return {
            "sessions": len(rows),
            "succeeded": sum(1 for r in rows if r["success"]),
            "completed": sum(1 for r in rows if r["status"] == STATUS_COMPLETED),
            "cancelled": sum(1 for r in rows if r["status"] == STATUS_CANCELLED),
            "error": sum(1 for r in rows if r["status"] == STATUS_ERROR),
            "no_end": sum(1 for r in rows if r["status"] == STATUS_NO_END),
            "mean_duration_seconds": (
                round(sum(durations) / len(durations), 3) if durations else None
            ),
            "mean_turns": (
                round(sum(r["turn_count"] for r in ended) / len(ended), 3) if ended else None
            ),
            "mean_tool_calls": (
                round(sum(r["tool_call_count"] for r in ended) / len(ended), 3) if ended else None
            ),
        }

    flags_agg: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for flag in FIELD_DATA_FLAGS:
        on_rows = [s for s in sessions if s["flags"].get(flag)]
        off_rows = [s for s in sessions if not s["flags"].get(flag)]
        flags_agg[flag] = {"on": arm_stats(on_rows), "off": arm_stats(off_rows)}

    status_counts: Dict[str, int] = {}
    for s in sessions:
        status_counts[s["status"]] = status_counts.get(s["status"], 0) + 1

    return {
        "schema_version": FIELD_DATA_SCHEMA_VERSION,
        "session_count": len(sessions),
        "status_counts": status_counts,
        "flags": flags_agg,
    }


def format_evidence_table(aggregated: Dict[str, Any]) -> str:
    """Render the aggregated store as the ADR-036 evidence table (markdown).

    One row per flag × exposure arm: sessions exposed, outcome distribution,
    and the mean duration/turns/tool-call profile the graduation decisions read.
    """
    session_count = aggregated.get("session_count", 0)
    status_counts = aggregated.get("status_counts", {})
    lines = [
        "# ADR-036 field-data evidence",
        "",
        f"sessions: {session_count} | statuses: "
        + ", ".join(f"{k}={v}" for k, v in sorted(status_counts.items())),
        "",
        "| flag | arm | sessions | succeeded | completed | cancelled | error | no_end"
        " | mean_duration_s | mean_turns | mean_tool_calls |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    def num(value: Any) -> str:
        return "—" if value is None else str(value)

    for flag in FIELD_DATA_FLAGS:
        arms = aggregated.get("flags", {}).get(flag, {})
        for arm in ("on", "off"):
            stats = arms.get(arm, {})
            lines.append(
                f"| {flag} | {arm} | {stats.get('sessions', 0)} | {stats.get('succeeded', 0)} "
                f"| {stats.get('completed', 0)} "
                f"| {stats.get('cancelled', 0)} | {stats.get('error', 0)} | {stats.get('no_end', 0)} "
                f"| {num(stats.get('mean_duration_seconds'))} | {num(stats.get('mean_turns'))} "
                f"| {num(stats.get('mean_tool_calls'))} |"
            )
    lines.append("")
    lines.append(
        "Read per ADR-036: a default-on flip needs the ON arm matching-or-beating OFF across "
        "outcomes and durations; a regression (or an error/no-end skew) is demotion evidence."
    )
    return "\n".join(lines)
