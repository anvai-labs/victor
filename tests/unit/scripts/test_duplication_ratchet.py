"""Duplication ratchet (scripts/ci/duplication_ratchet.py) — enforcement contracts.

The jscpd RUN is never invoked here (no node dependency in unit tests): every
case drives the parse/enforce/write surface against synthetic jscpd JSON
reports or module functions directly.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


def load_ratchet_module():
    module_path = Path(__file__).resolve().parents[3] / "scripts" / "ci" / "duplication_ratchet.py"
    spec = importlib.util.spec_from_file_location("duplication_ratchet", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ratchet = load_ratchet_module()


def _report(path: Path, duplicated_lines: int, total_lines: int = 10_000, clones: int = 5):
    payload = {
        "statistics": {
            "total": {
                "lines": total_lines,
                "duplicatedLines": duplicated_lines,
                "clones": clones,
                "percentage": 100.0 * duplicated_lines / total_lines,
            }
        }
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _baseline(path: Path, duplicated_lines: int, total_lines: int = 10_000):
    path.write_text(
        json.dumps(
            {
                "jscpd": "5.4.2",
                "updated": "2026-10-21",
                "total_lines": total_lines,
                "duplicated_lines": duplicated_lines,
                "clones": 5,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_equal_baseline_passes(tmp_path, capsys):
    report = _report(tmp_path / "r.json", 100)
    _baseline(tmp_path / "baseline.json", 100)
    baseline = ratchet.load_baseline(tmp_path / "baseline.json")
    assert ratchet.enforce(baseline, ratchet.parse_report(report)) == 0
    assert "delta +0 lines" in capsys.readouterr().out


def test_one_line_increase_fails_with_guidance(tmp_path, capsys):
    report = _report(tmp_path / "r.json", 101)
    _baseline(tmp_path / "baseline.json", 100)
    baseline = ratchet.load_baseline(tmp_path / "baseline.json")
    assert ratchet.enforce(baseline, ratchet.parse_report(report)) == 1
    out = capsys.readouterr().out
    assert "delta +1 lines" in out
    # The failure must teach the fix: how to reproduce, and that raises are
    # never silent.
    assert "consoleFull" in out and "never silent" in out


def test_decrease_passes_and_suggests_locking_it_in(tmp_path, capsys):
    report = _report(tmp_path / "r.json", 90)
    _baseline(tmp_path / "baseline.json", 100)
    baseline = ratchet.load_baseline(tmp_path / "baseline.json")
    assert ratchet.enforce(baseline, ratchet.parse_report(report)) == 0
    assert "--update-baseline" in capsys.readouterr().out


def test_main_report_mode_enforces(tmp_path):
    report = _report(tmp_path / "r.json", 100)
    real_baseline = ratchet.BASELINE_PATH
    try:
        ratchet.BASELINE_PATH = _baseline(tmp_path / "baseline.json", 100)
        assert ratchet.main(["--report", str(report)]) == 0
        ratchet.BASELINE_PATH = _baseline(tmp_path / "baseline.json", 99)
        assert ratchet.main(["--report", str(report)]) == 1
    finally:
        ratchet.BASELINE_PATH = real_baseline


def test_malformed_report_is_a_clean_error(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text('{"statistics": {}}', encoding="utf-8")
    with pytest.raises(SystemExit, match="not a jscpd JSON report"):
        ratchet.parse_report(bad)


def test_missing_baseline_is_a_clean_error(tmp_path):
    with pytest.raises(SystemExit, match="malformed"):
        ratchet.load_baseline(tmp_path / "absent.json")


def test_write_baseline_round_trips(tmp_path):
    metrics = ratchet.parse_report(_report(tmp_path / "r.json", 120, total_lines=6_000))
    target = tmp_path / "new-baseline.json"
    ratchet.write_baseline(target, "5.4.2", metrics)
    reloaded = ratchet.load_baseline(target)
    assert reloaded["duplicated_lines"] == 120
    assert reloaded["jscpd"] == "5.4.2"
    # percentages are recomputed at compare time; the baseline stores counts.
    assert "percentage" not in json.loads(target.read_text())
