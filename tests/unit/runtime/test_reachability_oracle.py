# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Tests for the FEP-0022 Phase 2 oracle (accumulation + candidate-dead diff).

Pure-function tests over the sidecar/baseline/exempt machinery. Sidecars are
produced by the real Phase 1 recorder (flush), so these also cover the
sidecar-format <-> accumulator contract.
"""

from __future__ import annotations

from pathlib import Path

from victor.runtime.reachability import (
    activate,
    candidate_dead,
    load_baseline,
    load_exempt,
    merge_sidecar_paths,
    record,
    write_baseline,
)


def _flush_sidecar(path: Path, keys: list[str]) -> Path:
    """Produce a real Phase 1 sidecar by arming the recorder and flushing."""
    with activate(run_id=path.stem, out_path=path):
        for key in keys:
            record("di", key)
    return path


# --- merge ------------------------------------------------------------------


def test_merge_unions_across_sidecars_and_dedups(tmp_path):
    s1 = _flush_sidecar(tmp_path / "r1.jsonl", ["m:A", "m:B"])
    s2 = _flush_sidecar(tmp_path / "r2.jsonl", ["m:B", "m:C"])
    merged = merge_sidecar_paths([s1, s2])
    assert merged == {"di": {"m:A", "m:B", "m:C"}}


def test_merge_skips_header_line(tmp_path):
    # The recorder writes a {run_id, ts, count} header on line 1; the merger must
    # not treat it as a witness (it has no kind/key).
    s = _flush_sidecar(tmp_path / "r.jsonl", ["m:A"])
    merged = merge_sidecar_paths([s])
    assert merged == {"di": {"m:A"}}


# --- baseline round-trip ----------------------------------------------------


def test_baseline_round_trip_is_deterministic(tmp_path):
    merged = {"di": {"m:B", "m:A"}, "flag": {"z", "a"}}
    out = write_baseline(merged, tmp_path / "baseline.json")
    text = out.read_text()
    # values sorted within each kind
    assert text.index("m:A") < text.index("m:B")
    assert text.index('"a"') < text.index('"z"')
    assert load_baseline(out) == {"di": {"m:A", "m:B"}, "flag": {"a", "z"}}


# --- exempt loading ---------------------------------------------------------


def test_load_exempt_ignores_comments_and_blanks(tmp_path):
    ex = tmp_path / "exempt.txt"
    ex.write_text("# header\nm:A   # trailing reason\n\nm:B\n# another comment\n")
    assert load_exempt(ex) == {"m:A", "m:B"}


# --- candidate-dead oracle --------------------------------------------------


def test_candidate_dead_diffs_and_subtracts_exempt():
    registered = ["m:A", "m:B", "m:C", "m:D"]
    observed = ["m:A", "m:B"]
    exempt = ["m:C"]
    assert candidate_dead(registered, observed, exempt) == ["m:D"]


def test_candidate_dead_is_sorted():
    assert candidate_dead(["z", "a", "m"], [], []) == ["a", "m", "z"]


def test_candidate_dead_empty_when_all_observed_or_exempt():
    assert candidate_dead(["m:A"], ["m:A"], []) == []
    assert candidate_dead(["m:A"], [], ["m:A"]) == []


# --- end-to-end: sidecars -> baseline -> oracle -----------------------------


def test_end_to_end_accumulate_then_oracle(tmp_path):
    s1 = _flush_sidecar(tmp_path / "r1.jsonl", ["victor.mod:UsedA", "victor.mod:UsedB"])
    s2 = _flush_sidecar(tmp_path / "r2.jsonl", ["victor.mod:UsedB"])
    baseline_path = write_baseline(merge_sidecar_paths([s1, s2]), tmp_path / "baseline.json")
    observed = load_baseline(baseline_path)["di"]

    registered = {"victor.mod:UsedA", "victor.mod:UsedB", "victor.mod:DeadC", "victor.mod:CondD"}
    exempt_path = tmp_path / "exempt.txt"
    exempt_path.write_text("victor.mod:CondD  # legitimately conditional\n")
    exempt = load_exempt(exempt_path)

    assert candidate_dead(registered, observed, exempt) == ["victor.mod:DeadC"]
