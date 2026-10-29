# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Baseline-vs-current pytest-benchmark regression check for the performance workflow."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _load():
    module_path = (
        Path(__file__).resolve().parents[3] / "scripts" / "ci" / "benchmark_regressions.py"
    )
    spec = importlib.util.spec_from_file_location("benchmark_regressions", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolve the defining module through sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_mod = _load()
compare = _mod.compare
main = _mod.main

FAST = "tests/performance/test_registry.py::test_lookup"
SLOW = "tests/performance/test_graph.py::test_compile[small]"
TINY = "tests/performance/test_hash.py::test_hash_key"


def _bench(fullname: str, mean: float, stddev: float) -> dict:
    """A realistic pytest-benchmark 5.x ``benchmarks[]`` entry."""
    return {
        "group": None,
        "name": fullname.split("::")[-1],
        "fullname": fullname,
        "params": None,
        "param": None,
        "extra_info": {},
        "options": {
            "disable_gc": False,
            "timer": "perf_counter",
            "min_rounds": 5,
            "max_time": 1.0,
            "min_time": 5e-06,
            "warmup": False,
        },
        "stats": {
            "min": mean - stddev,
            "max": mean + 3 * stddev,
            "mean": mean,
            "stddev": stddev,
            "rounds": 200,
            "median": mean,
            "iqr": stddev,
            "q1": mean - stddev / 2,
            "q3": mean + stddev / 2,
            "iqr_outliers": 3,
            "stddev_outliers": 5,
            "outliers": "5;3",
            "ld15iqr": mean - stddev,
            "hd15iqr": mean + stddev,
            "ops": 1.0 / mean,
            "total": mean * 200,
            "iterations": 1,
        },
    }


def _report(*benches: dict) -> dict:
    return {
        "machine_info": {"node": "runner", "processor": "x86_64", "python_version": "3.12.7"},
        "commit_info": {"id": "abc123", "branch": "develop", "dirty": False},
        "benchmarks": list(benches),
        "datetime": "2026-10-29T00:00:00.000000+00:00",
        "version": "5.2.3",
    }


BASELINE = _report(_bench(FAST, 1.0e-4, 2.0e-6), _bench(SLOW, 5.0e-3, 1.0e-4))


def _write(tmp_path: Path, baseline: dict, current: dict) -> tuple[str, str]:
    b, c = tmp_path / "baseline.json", tmp_path / "output.json"
    b.write_text(json.dumps(baseline))
    c.write_text(json.dumps(current))
    return str(b), str(c)


# --- positive ---------------------------------------------------------------------------


def test_real_regression_is_reported_by_fullname():
    # FAST is 50% slower with a tight spread: far outside the noise band.
    current = _report(_bench(FAST, 1.5e-4, 2.0e-6), _bench(SLOW, 5.0e-3, 1.0e-4))
    result = compare(BASELINE, current)
    assert [c.fullname for c in result.regressions] == [FAST]
    assert result.regressions[0].change_pct == pytest.approx(50.0)
    assert result.only_in_baseline == [] and result.only_in_current == []


def test_warn_mode_annotates_and_exits_zero(tmp_path, capsys, monkeypatch):
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    current = _report(_bench(FAST, 1.5e-4, 2.0e-6), _bench(SLOW, 5.0e-3, 1.0e-4))
    assert main(list(_write(tmp_path, BASELINE, current))) == 0
    out = capsys.readouterr().out
    assert f"::warning::Performance regression {FAST}" in out
    assert "| Benchmark |" in summary.read_text()
    assert FAST in summary.read_text()


def test_fail_mode_exits_one_on_regression(tmp_path, capsys):
    current = _report(_bench(FAST, 1.5e-4, 2.0e-6), _bench(SLOW, 5.0e-3, 1.0e-4))
    assert main([*_write(tmp_path, BASELINE, current), "--mode", "fail"]) == 1
    assert f"::error::Performance regression {FAST}" in capsys.readouterr().out


def test_threshold_flag_is_honoured(tmp_path):
    # 15% slower, tight spread: clean under the default 20%, a regression at 10%.
    current = _report(_bench(FAST, 1.15e-4, 1.0e-6), _bench(SLOW, 5.0e-3, 1.0e-4))
    paths = _write(tmp_path, BASELINE, current)
    assert main([*paths, "--mode", "fail"]) == 0
    assert main([*paths, "--mode", "fail", "--threshold", "10"]) == 1


# --- negative ---------------------------------------------------------------------------


def test_slowdown_within_noise_is_not_flagged():
    # +30% but stddev is huge relative to the delta (delta 3e-5 < 2 * hypot(4e-5, 4e-5)).
    baseline = _report(_bench(FAST, 1.0e-4, 4.0e-5))
    current = _report(_bench(FAST, 1.3e-4, 4.0e-5))
    result = compare(baseline, current)
    assert result.regressions == []
    assert result.comparisons[0].reason == "within noise"


def test_improvement_is_not_flagged():
    current = _report(_bench(FAST, 0.5e-4, 1.0e-6), _bench(SLOW, 2.0e-3, 1.0e-4))
    result = compare(BASELINE, current)
    assert result.regressions == []
    assert all(c.change_pct < 0 for c in result.comparisons)


def test_sub_microsecond_benchmark_is_ignored():
    baseline = _report(_bench(TINY, 2.0e-7, 1.0e-9))
    current = _report(_bench(TINY, 4.0e-7, 1.0e-9))  # +100%, but 0.2us -> 0.4us
    result = compare(baseline, current)
    assert result.regressions == []
    assert result.comparisons[0].reason == "below min mean"
    assert compare(baseline, current, min_mean_us=0.0).regressions != []


def test_malformed_json_exits_two(tmp_path, capsys):
    b = tmp_path / "baseline.json"
    c = tmp_path / "output.json"
    b.write_text("{not json")
    c.write_text(json.dumps(BASELINE))
    assert main([str(b), str(c)]) == 2
    assert "::error::" in capsys.readouterr().out


def test_missing_file_exits_two(tmp_path):
    c = tmp_path / "output.json"
    c.write_text(json.dumps(BASELINE))
    assert main([str(tmp_path / "absent.json"), str(c)]) == 2


@pytest.mark.parametrize(
    "bad",
    [
        {"benchmarks": {FAST: {"stats": {"mean": 1.0}}}},  # dict instead of list
        {"no_benchmarks": []},
        {"benchmarks": [{"name": "x", "stats": {"mean": 1.0}}]},  # no fullname
        {"benchmarks": [{"fullname": FAST, "stats": {"stddev": 1.0}}]},  # no mean
        {"benchmarks": [{"fullname": FAST, "stats": {"mean": "fast"}}]},
        [],
    ],
)
def test_malformed_shape_exits_two(tmp_path, bad):
    assert main(list(_write(tmp_path, bad, BASELINE))) == 2


def test_zero_overlap_warns_loudly(tmp_path, capsys):
    current = _report(_bench("tests/performance/test_other.py::test_new", 1.0e-4, 1.0e-6))
    assert main(list(_write(tmp_path, BASELINE, current))) == 0
    out = capsys.readouterr().out
    assert "::warning::No benchmarks matched" in out
    assert "Only in baseline (2)" in out
    assert main([*_write(tmp_path, BASELINE, current), "--mode", "fail"]) == 1


# --- regression: the old inline workflow check -----------------------------------------


def _old_inline_check(baseline: dict, current: dict) -> list:
    """Verbatim logic of the previous ``Check for regressions`` step."""
    regressions = []
    for test in baseline["benchmarks"]:
        if test in current["benchmarks"]:
            baseline_mean = baseline["benchmarks"][test]["stats"]["mean"]
            current_mean = current["benchmarks"][test]["stats"]["mean"]
            change_pct = ((current_mean - baseline_mean) / baseline_mean) * 100
            if change_pct > 20:
                regressions.append((test, change_pct))
    return regressions


def test_old_dict_membership_check_found_nothing_on_same_input():
    # Real runs never reproduce identical stats, so no whole-dict membership test matches.
    current = _report(_bench(FAST, 3.0e-4, 2.0e-6), _bench(SLOW, 5.01e-3, 1.0e-4))
    assert _old_inline_check(copy.deepcopy(BASELINE), copy.deepcopy(current)) == []
    assert [c.fullname for c in compare(BASELINE, current).regressions] == [FAST]
