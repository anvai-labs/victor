"""Usage trace scanning with native acceleration.

Scans usage.jsonl files (plain and gzip-compressed) and aggregates
per-session statistics for GEPA prompt optimization.
Uses Rust implementation when available for 5-8x speedup.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from victor.processing.native._base import _NATIVE_AVAILABLE, _native


@dataclass
class SessionStats:
    """Per-session aggregated statistics from usage.jsonl scanning."""

    session_id: str
    tool_calls: int
    tool_failures: int
    task_type: str
    tokens: int
    completion_score: float


def scan_usage_file(file_path: str) -> List[SessionStats]:
    """Scan a single JSONL file and aggregate per-session stats.

    Uses Rust implementation when available. Falls back to Python.
    """
    if _NATIVE_AVAILABLE and hasattr(_native, "scan_usage_file"):
        try:
            rust_results = _native.scan_usage_file(file_path)
            return [
                SessionStats(
                    session_id=r.session_id,
                    tool_calls=r.tool_calls,
                    tool_failures=r.tool_failures,
                    task_type=r.task_type,
                    tokens=r.tokens,
                    completion_score=r.completion_score,
                )
                for r in rust_results
            ]
        except Exception:
            pass

    return _scan_usage_file_python(file_path)


def scan_usage_files(file_paths: List[str]) -> List[SessionStats]:
    """Scan multiple JSONL files and merge results.

    Uses Rust implementation when available. Falls back to Python.
    """
    if _NATIVE_AVAILABLE and hasattr(_native, "scan_usage_files"):
        try:
            rust_results = _native.scan_usage_files(file_paths)
            return [
                SessionStats(
                    session_id=r.session_id,
                    tool_calls=r.tool_calls,
                    tool_failures=r.tool_failures,
                    task_type=r.task_type,
                    tokens=r.tokens,
                    completion_score=r.completion_score,
                )
                for r in rust_results
            ]
        except Exception:
            pass

    # Python fallback: aggregate per session ACROSS the whole file list,
    # mirroring the Rust batch semantics — a session appearing in several
    # files contributes all of its events once, not once per file.
    sessions: Dict[str, Dict[str, Any]] = {}
    for path in file_paths:
        _accumulate_usage_file(path, sessions)
    return _finalize_session_stats(sessions)


def _scan_usage_file_python(file_path: str) -> List[SessionStats]:
    """Pure Python JSONL trace scanner — reference implementation."""
    sessions: Dict[str, Dict[str, Any]] = {}
    _accumulate_usage_file(file_path, sessions)
    return _finalize_session_stats(sessions)


def _accumulate_usage_file(file_path: str, sessions: Dict[str, Dict[str, Any]]) -> None:
    """Fold one usage.jsonl file's events into the shared session map."""
    path = Path(file_path)
    if not path.exists():
        return

    try:
        opener = gzip.open if path.suffix == ".gz" else open
        mode = "rt" if path.suffix == ".gz" else "r"
        with opener(path, mode) as f:
            for line in f:
                try:
                    event = json.loads(line.strip())
                    sid = event.get("session_id", "")
                    etype = event.get("event_type", "")
                    data = event.get("data", {})

                    if sid not in sessions:
                        sessions[sid] = {
                            "tool_calls": 0,
                            "tool_failures": 0,
                            "task_type": "default",
                            "tokens": 0,
                        }

                    if etype == "tool_call":
                        sessions[sid]["tool_calls"] += 1
                    elif etype == "tool_result":
                        if not data.get("success", True):
                            sessions[sid]["tool_failures"] += 1
                    elif etype == "task_classification":
                        sessions[sid]["task_type"] = data.get("task_type", "default")
                except (json.JSONDecodeError, KeyError):
                    continue
    except Exception:
        pass


def _finalize_session_stats(sessions: Dict[str, Dict[str, Any]]) -> List[SessionStats]:
    """Shared scoring/filter/sort tail — mirrors the Rust
    ``finalize_session_stats`` so both engines' semantics stay aligned."""
    results = []
    for sid, s in sessions.items():
        if s["tool_calls"] < 2:
            continue
        failure_rate = s["tool_failures"] / max(s["tool_calls"], 1)
        completion_score = max(0.0, 1.0 - failure_rate * 1.5)
        results.append(
            SessionStats(
                session_id=sid,
                tool_calls=s["tool_calls"],
                tool_failures=s["tool_failures"],
                task_type=s["task_type"],
                tokens=s["tokens"],
                completion_score=completion_score,
            )
        )

    results.sort(key=lambda s: -s.completion_score)
    return results
