# Copyright 2026 Vijaykumar Singh <vijaykumar@anvaiops.com>
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

"""Native/Python usage-trace scanner differential parity (FOU-C3).

The GEPA trace scanner (``victor.processing.native.trace_scanner``) has a
Rust path (``scan_usage_file``/``scan_usage_files``, 5-8x on multi-MB logs)
and a pure-Python reference (``_scan_usage_file_python``). These tests run
synthetic usage.jsonl corpora (plain, gzip, malformed lines, filtered
sessions, task classifications) through both paths — including a session id
appearing in MULTIPLE files, where both engines aggregate per session across
the whole batch (the formerly-pinned concatenation divergence in the Python
fallback was fixed to the native semantics).

Selected by the native-parity CI job (``pytest -m native_parity``).
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

import victor.processing.native.trace_scanner as ts
from victor.processing.native._base import _NATIVE_AVAILABLE, _native

pytestmark = pytest.mark.native_parity

_requires_native = pytest.mark.skipif(
    not (_NATIVE_AVAILABLE and hasattr(_native, "scan_usage_file")),
    reason="native wheel missing trace_scanner ops — rebuild with maturin",
)


def _write_events(path: Path, events) -> Path:
    lines = [e if isinstance(e, str) else json.dumps(e) for e in events]
    path.write_text("\n".join(lines) + "\n")
    return path


def _sample_events():
    return [
        {"session_id": "s1", "event_type": "tool_call", "data": {}},
        {"session_id": "s1", "event_type": "tool_call", "data": {}},
        {"session_id": "s1", "event_type": "tool_result", "data": {"success": True}},
        {"session_id": "s1", "event_type": "task_classification", "data": {"task_type": "coding"}},
        # All failures clamp the completion score at 0.
        {"session_id": "s2", "event_type": "tool_call", "data": {}},
        {"session_id": "s2", "event_type": "tool_result", "data": {"success": False}},
        {"session_id": "s2", "event_type": "tool_result", "data": {"success": False}},
        # Below the >= 2 tool_calls threshold: filtered by both engines.
        {"session_id": "s3", "event_type": "tool_call", "data": {}},
        {"session_id": "s3", "event_type": "tool_result", "data": {"success": True}},
        # Malformed lines are skipped, not fatal.
        "not json at all",
        '{"session_id": "s4", "event_type": "tool_call"}',
        '{"session_id": "s4", "event_type": "tool_call"}',
        '{"session_id": "s4", "event_type": "tool_result", "data": {"success": false}}',
        {"session_id": "s5", "event_type": "unknown_event", "data": {"ignored": True}},
    ]


def _as_mapping(stats):
    """Both paths' results as session-keyed tuples (order-independent compare)."""
    return {s.session_id: (s.tool_calls, s.tool_failures, s.task_type, s.tokens) for s in stats}, {
        s.session_id: s.completion_score for s in stats
    }


def _forced_python(fn, *args):
    saved = ts._NATIVE_AVAILABLE
    ts._NATIVE_AVAILABLE = False
    try:
        return fn(*args)
    finally:
        ts._NATIVE_AVAILABLE = saved


class TestSingleFileParity:
    @_requires_native
    def test_plain_jsonl_matches_reference(self, tmp_path):
        path = _write_events(tmp_path / "usage.jsonl", _sample_events())
        native = ts.scan_usage_file(str(path))
        fallback = _forced_python(ts.scan_usage_file, str(path))

        native_map, native_scores = _as_mapping(native)
        fallback_map, fallback_scores = _as_mapping(fallback)
        assert native_map == fallback_map
        for sid, score in native_scores.items():
            # f32 (Rust) vs f64 (Python) scoring — sub-1e-6 tolerance.
            assert score == pytest.approx(fallback_scores[sid], abs=1e-6)
        # Both filter the below-threshold session and apply classifications.
        assert "s3" not in native_map
        assert native_map["s1"][2] == "coding"
        # Highest completion score first in both paths.
        assert native[0].session_id == fallback[0].session_id

    @_requires_native
    def test_gzip_matches_reference(self, tmp_path):
        path = tmp_path / "usage.jsonl.gz"
        with gzip.open(path, "wt") as f:
            f.write("\n".join(json.dumps(e) for e in _sample_events()) + "\n")
        native = ts.scan_usage_file(str(path))
        fallback = _forced_python(ts.scan_usage_file, str(path))
        native_map, native_scores = _as_mapping(native)
        fallback_map, fallback_scores = _as_mapping(fallback)
        assert native_map == fallback_map
        for sid, score in native_scores.items():
            assert score == pytest.approx(fallback_scores[sid], abs=1e-6)

    @_requires_native
    def test_empty_and_missing_files_match(self, tmp_path):
        empty = tmp_path / "empty.jsonl"
        empty.write_text("")
        missing = str(tmp_path / "nope.jsonl")
        for path in (str(empty), missing):
            assert ts.scan_usage_file(path) == []
            assert _forced_python(ts.scan_usage_file, path) == []

    @_requires_native
    def test_no_failure_clamps_at_zero_both_paths(self, tmp_path):
        path = _write_events(
            tmp_path / "usage.jsonl",
            [
                {"session_id": "bad", "event_type": "tool_call", "data": {}},
                {"session_id": "bad", "event_type": "tool_call", "data": {}},
                {"session_id": "bad", "event_type": "tool_result", "data": {"success": False}},
                {"session_id": "bad", "event_type": "tool_result", "data": {"success": False}},
            ],
        )
        native = ts.scan_usage_file(str(path))
        fallback = _forced_python(ts.scan_usage_file, str(path))
        assert native[0].completion_score == fallback[0].completion_score == 0.0


class TestMultiFileParity:
    @_requires_native
    def test_disjoint_sessions_match(self, tmp_path):
        """The real deployment shape: sessions partition across files."""
        first = _write_events(
            tmp_path / "a.jsonl",
            [
                {"session_id": "s1", "event_type": "tool_call", "data": {}},
                {"session_id": "s1", "event_type": "tool_call", "data": {}},
                {"session_id": "s1", "event_type": "tool_result", "data": {"success": True}},
            ],
        )
        second = _write_events(
            tmp_path / "b.jsonl",
            [
                {"session_id": "s2", "event_type": "tool_call", "data": {}},
                {"session_id": "s2", "event_type": "tool_result", "data": {"success": False}},
                {"session_id": "s2", "event_type": "tool_result", "data": {"success": False}},
            ],
        )
        native = ts.scan_usage_files([str(first), str(second)])
        fallback = _forced_python(ts.scan_usage_files, [str(first), str(second)])
        native_map, native_scores = _as_mapping(native)
        fallback_map, fallback_scores = _as_mapping(fallback)
        assert native_map == fallback_map
        for sid, score in native_scores.items():
            assert score == pytest.approx(fallback_scores[sid], abs=1e-6)
        # Descending completion_score in both paths.
        assert native[0].session_id == fallback[0].session_id == "s1"

    @_requires_native
    def test_duplicate_session_across_files(self, tmp_path):
        """The same session id in two files aggregates across the batch.

        The Rust batch folds events per session across the file list; the
        Python fallback shares one session accumulator across files (fixed
        from per-file concatenation, which emitted the session once per
        file). Both engines must agree exactly — counts, failures,
        classification and score included.
        """
        events = [
            {"session_id": "s1", "event_type": "tool_call", "data": {}},
            {"session_id": "s1", "event_type": "tool_call", "data": {}},
            {"session_id": "s1", "event_type": "tool_result", "data": {"success": True}},
            {
                "session_id": "s1",
                "event_type": "task_classification",
                "data": {"task_type": "coding"},
            },
            # The second file contributes MORE events for the same session:
            # both engines must merge them into one entry.
            {"session_id": "s1", "event_type": "tool_call", "data": {}},
            {"session_id": "s1", "event_type": "tool_result", "data": {"success": False}},
            # A disjoint session in the second file stays independent.
            {"session_id": "s2", "event_type": "tool_call", "data": {}},
            {"session_id": "s2", "event_type": "tool_call", "data": {}},
            {"session_id": "s2", "event_type": "tool_result", "data": {"success": True}},
        ]
        first = _write_events(tmp_path / "a.jsonl", events[:4])
        second = _write_events(tmp_path / "b.jsonl", events[4:])

        native = ts.scan_usage_files([str(first), str(second)])
        fallback = _forced_python(ts.scan_usage_files, [str(first), str(second)])

        assert len(native) == len(fallback) == 2, "exactly one entry per session id"
        native_map, native_scores = _as_mapping(native)
        fallback_map, fallback_scores = _as_mapping(fallback)
        assert native_map == fallback_map
        for sid, score in native_scores.items():
            assert score == pytest.approx(fallback_scores[sid], abs=1e-6)
        # s1 spans both files: 3 tool_calls (2 + 1), 1 failure — merged, not
        # duplicated (concatenation would double-count the first file's
        # events).
        assert native_map["s1"][:2] == (3, 1)
        assert native_map["s1"][2] == "coding"
        assert fallback_map["s1"][:2] == (3, 1)
