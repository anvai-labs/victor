#!/usr/bin/env python3
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

"""Duplication ratchet: the code corpus's duplicated-line count may only fall.

Runs jscpd over the scan config in ``.jscpd.json`` (code formats only —
python/typescript/tsx/javascript across ``victor/``, ``verticals/``,
``vscode-victor/src/``), then compares ``statistics.total.duplicatedLines``
against the committed baseline in ``scripts/ci/duplication_baseline.json``.

- current > baseline  -> FAIL. Copy-paste growth must not merge silently.
- current == baseline -> pass.
- current < baseline  -> pass; lower the baseline in the SAME dedup PR via
  ``--update-baseline`` so the win is locked in (ratchets only tighten).

The jscpd version is pinned BY the baseline file and the run always invokes
``npx --yes jscpd@<pin>`` — a version bump regenerates the baseline
deliberately (tokenization differs across major versions), never by drift.

Usage::

    python scripts/ci/duplication_ratchet.py                 # run + enforce
    python scripts/ci/duplication_ratchet.py --report R.json # parse-only
    python scripts/ci/duplication_ratchet.py --update-baseline

``--report`` parses an existing jscpd JSON report instead of running the
scan (unit tests and local debugging; no node/npx needed).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
BASELINE_PATH = ROOT / "scripts" / "ci" / "duplication_baseline.json"
SCAN_TIMEOUT_SECONDS = 300  # measured ~2-3s locally; generous CI ceiling
# Only used to BOOTSTRAP a missing baseline under --update-baseline; after the
# first commit the pinned version always comes from the baseline file itself.
_BOOTSTRAP_PIN = "5.4.2"


def load_baseline(path: Path = BASELINE_PATH) -> dict[str, Any]:
    """Read the committed baseline; a missing/malformed file is a hard error."""

    try:
        baseline = json.loads(path.read_text(encoding="utf-8"))
        return {
            "jscpd": str(baseline["jscpd"]),
            "updated": str(baseline.get("updated", "")),
            "total_lines": int(baseline["total_lines"]),
            "duplicated_lines": int(baseline["duplicated_lines"]),
            "clones": int(baseline["clones"]),
        }
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SystemExit(
            f"duplication-ratchet: baseline {path} is missing or malformed ({exc}); "
            "regenerate it with `python scripts/ci/duplication_ratchet.py --update-baseline` "
            "and commit it with the change that justifies the number"
        ) from exc


def run_jscpd(version: str, output_dir: Path) -> Path:
    """Run the pinned jscpd over the repo scan config; return the report path.

    One retry: the run fetches jscpd via npx, and this gates merges — a
    transient registry fetch must not false-red a PR. A real scan failure
    fails on the second attempt like any other check.
    """

    command = [
        "npx",
        "--yes",
        f"jscpd@{version}",
        "--silent",
        "--reporters",
        "json",
        "--output",
        str(output_dir),
    ]
    last_output = ""
    for attempt in (1, 2):
        result = subprocess.run(  # noqa: S603 - fixed argv, pinned package
            command,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=SCAN_TIMEOUT_SECONDS,
            check=False,
        )
        report = output_dir / "jscpd-report.json"
        if result.returncode == 0 and report.exists():
            return report
        last_output = (result.stdout or "") + (result.stderr or "")
        if attempt == 1:
            print(f"jscpd attempt 1 failed (exit {result.returncode}); retrying once")
    sys.stderr.write(last_output)
    raise SystemExit(f"duplication-ratchet: jscpd@{version} failed twice; see output above")


def parse_report(path: Path) -> dict[str, Any]:
    """Extract the ratchet metrics from a jscpd JSON report."""

    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        total = report["statistics"]["total"]
        return {
            "total_lines": int(total["lines"]),
            "duplicated_lines": int(total["duplicatedLines"]),
            "clones": int(total["clones"]),
            "percentage": float(total["percentage"]),
        }
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SystemExit(
            f"duplication-ratchet: report {path} is not a jscpd JSON report ({exc})"
        ) from exc


def write_baseline(path: Path, version: str, metrics: dict[str, Any]) -> None:
    """Persist current metrics as the new baseline (committed, reviewable)."""

    payload = {
        "jscpd": version,
        "updated": date.today().isoformat(),
        "total_lines": metrics["total_lines"],
        "duplicated_lines": metrics["duplicated_lines"],
        "clones": metrics["clones"],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def enforce(baseline: dict[str, Any], metrics: dict[str, Any]) -> int:
    """Compare and report; 0 on pass (equal or improved), 1 on regression."""

    base_lines = baseline["duplicated_lines"]
    current_lines = metrics["duplicated_lines"]
    delta = current_lines - base_lines
    base_pct = 100.0 * base_lines / max(baseline["total_lines"], 1)
    print(
        f"duplication-ratchet: {current_lines} duplicated lines "
        f"({metrics['clones']} clones, {metrics['percentage']:.2f}%) "
        f"vs baseline {base_lines} ({base_pct:.2f}%) -> delta {delta:+d} lines"
    )
    if delta > 0:
        print(
            "FAIL: the duplicated-line count rose. Inspect locally with "
            f"`npx --yes jscpd@{baseline['jscpd']} --reporters consoleFull` "
            "(repo root) and refactor the new copy-paste into a shared helper. "
            "An intentional increase requires explicit justification in the PR "
            "plus `--update-baseline` — never silent."
        )
        return 1
    if delta < 0:
        print(
            "improvement: lock it in by running "
            "`python scripts/ci/duplication_ratchet.py --update-baseline` "
            "and committing the lowered baseline in this PR."
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--report",
        type=Path,
        help="parse this existing jscpd JSON report instead of running a scan",
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="write the current metrics as the new baseline (commit the result)",
    )
    args = parser.parse_args(argv)

    try:
        # Resolve the path at call time (BASELINE_PATH module global) so tests
        # can point the CLI surface at a temporary baseline.
        baseline = load_baseline(BASELINE_PATH)
    except SystemExit:
        if not args.update_baseline:
            raise
        baseline = {
            "jscpd": _BOOTSTRAP_PIN,
            "updated": "",
            "total_lines": 0,
            "duplicated_lines": 0,
            "clones": 0,
        }
    if args.report is not None:
        report_path = args.report
    else:
        tmp = tempfile.mkdtemp(prefix="jscpd-ratchet-")
        report_path = run_jscpd(baseline["jscpd"], Path(tmp))
    metrics = parse_report(report_path)

    if args.update_baseline:
        if metrics["duplicated_lines"] > baseline["duplicated_lines"]:
            print(
                "WARNING: this RAISES the baseline "
                f"({baseline['duplicated_lines']} -> {metrics['duplicated_lines']}); "
                "the PR must justify why the duplication is intentional."
            )
        write_baseline(BASELINE_PATH, baseline["jscpd"], metrics)
        print(f"baseline written to {BASELINE_PATH}")
        return 0
    return enforce(baseline, metrics)


if __name__ == "__main__":
    raise SystemExit(main())
