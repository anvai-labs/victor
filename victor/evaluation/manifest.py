# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License").

"""Per-task execution manifest — the closed loop's joinable artifact.

Every benchmark run emits ``eval_manifest_<run_id>.jsonl`` (one record per
task) alongside the lean results JSON. Each record joins three things that
were previously disconnected:

1. **Outcome (reward)** — did the patch pass the task's tests? Ground-truth,
   no LLM judge.
2. **Trace** — the bounded execution trace (messages, tool calls, edits) the
   adapter captured, now persisted on ``TaskResult.trace``.
3. **Decisions** — every decision the agent logged during the task
   (``victor.agent.decisions.chain.log_decision``), filtered out of the
   global ``~/.victor/logs/decisions.jsonl`` by the per-task ``session_id``
   the adapter stamps on the correlation spine.

This single artifact is simultaneously the observability record (what did the
agent actually do — real ``code search``/``grep``/``graph`` usage), and the
classifier's training data (decisions labeled by outcome).
``victor.ml.mining`` projects it into ``training_rows.jsonl``.

Design notes
------------
- Decisions live in one global JSONL (cheap append, no per-run path
  management). We read it once and bucket by ``session_id``.
- The manifest is bounded: traces were already capped by the adapter (50
  messages / 100 tool calls / 20 edits); decisions are emitted as-is (they
  are small per-task).
- Emission is best-effort: a failure here must never break the benchmark.
"""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)

# Global decision log written by victor.agent.decisions.chain.log_decision.
_DECISIONS_LOG = Path.home() / ".victor" / "logs" / "decisions.jsonl"


def _status_value(status: Any) -> str:
    """Serialize a TaskStatus (enum or raw) to a string."""
    return getattr(status, "value", str(status)) if status is not None else ""


def _reward(task_result: Any) -> str:
    """Derive a 3-valued reward from the task outcome.

    ``pass``    — every test passed (or the harness marked it PASSED).
    ``partial`` — some but not all tests passed.
    ``fail``    — no tests passed / failed status.

    The miner maps these to classifier labels per DecisionType (e.g. a
    ``task_completion`` decision is a positive example only under ``pass``).
    """
    status = _status_value(getattr(task_result, "status", None)).upper()
    if status == "PASSED":
        return "pass"
    total = int(getattr(task_result, "tests_total", 0) or 0)
    passed = int(getattr(task_result, "tests_passed", 0) or 0)
    if total > 0 and passed == total:
        return "pass"
    if passed > 0:
        return "partial"
    return "fail"


def load_decisions_by_session(session_ids: Iterable[str]) -> dict[str, list[dict]]:
    """Bucket decision records from the global log by ``session_id``.

    Reads ``~/.victor/logs/decisions.jsonl`` once and keeps only records whose
    ``session_id`` is in ``session_ids``. Robust to malformed lines and a
    missing log file (returns an empty mapping).
    """
    wanted = {sid for sid in session_ids if sid}
    if not wanted:
        return {}
    bucketed: dict[str, list[dict]] = {sid: [] for sid in wanted}
    if not _DECISIONS_LOG.exists():
        logger.debug("Execution manifest: decisions log not found at %s", _DECISIONS_LOG)
        return bucketed
    try:
        with _DECISIONS_LOG.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sid = rec.get("session_id")
                if sid in wanted:
                    bucketed[sid].append(rec)
    except OSError as exc:
        logger.warning("Execution manifest: could not read decisions log: %s", exc)
    return bucketed


def build_manifest_records(task_results: Iterable[Any]) -> list[dict[str, Any]]:
    """Build one manifest record per task, joining outcome + trace + decisions."""
    task_results = list(task_results)
    session_ids = {
        getattr(tr, "session_id", "") for tr in task_results if getattr(tr, "session_id", "")
    }
    decisions_by_session = load_decisions_by_session(session_ids)

    records: list[dict[str, Any]] = []
    for tr in task_results:
        sid = getattr(tr, "session_id", "") or ""
        trace = getattr(tr, "trace", {}) or {}
        records.append(
            {
                "session_id": sid,
                "task_id": getattr(tr, "task_id", ""),
                "status": _status_value(getattr(tr, "status", None)),
                "reward": _reward(tr),
                "tests_passed": int(getattr(tr, "tests_passed", 0) or 0),
                "tests_total": int(getattr(tr, "tests_total", 0) or 0),
                "tool_calls": int(getattr(tr, "tool_calls", 0) or 0),
                "code_search_calls": int(getattr(tr, "code_search_calls", 0) or 0),
                "graph_calls": int(getattr(tr, "graph_calls", 0) or 0),
                "turns": int(getattr(tr, "turns", 0) or 0),
                # Bounded trace captured by the adapter (messages, tool calls,
                # edits). May be empty for runners that don't instrument it.
                "trace": trace,
                # Every decision logged during this task (join by session_id).
                "decisions": decisions_by_session.get(sid, []),
            }
        )
    return records


def emit_execution_manifest(
    eval_result: Any,
    output_dir: Optional[Path] = None,
    *,
    run_id: Optional[str] = None,
) -> Optional[Path]:
    """Write ``eval_manifest_<run_id>.jsonl`` and return its path.

    Best-effort: any failure is logged and ``None`` returned so a manifest
    problem can never break the benchmark run.

    Args:
        eval_result: An object exposing ``task_results`` (list of TaskResult).
        output_dir: Directory to write into. Defaults to
            ``~/.victor/evaluations``.
        run_id: Stable id for the manifest filename. Generated if omitted.
    """
    try:
        task_results = list(getattr(eval_result, "task_results", []) or [])
        if not task_results:
            logger.debug("Execution manifest: no task_results, skipping emission")
            return None

        if output_dir is None:
            output_dir = Path.home() / ".victor" / "evaluations"
        output_dir.mkdir(parents=True, exist_ok=True)
        run_id = run_id or uuid.uuid4().hex[:12]
        manifest_path = output_dir / f"eval_manifest_{run_id}.jsonl"

        records = build_manifest_records(task_results)
        with manifest_path.open("w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, default=str) + "\n")

        n_with_decisions = sum(1 for r in records if r["decisions"])
        n_with_trace = sum(1 for r in records if r["trace"])
        logger.info(
            "Execution manifest emitted: %s (%d tasks, %d with trace, %d with decisions)",
            manifest_path,
            len(records),
            n_with_trace,
            n_with_decisions,
        )
        return manifest_path
    except Exception as exc:  # never break the benchmark
        logger.warning("Execution manifest emission failed: %s", exc)
        return None
