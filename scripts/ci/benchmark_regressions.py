#!/usr/bin/env python3
# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Compare two pytest-benchmark JSON reports and flag performance regressions.

pytest-benchmark stores ``benchmarks`` as a LIST of dicts (``fullname``, ``name``,
``stats`` ...). Benchmarks are matched by ``fullname`` (the pytest node id), which is
stable across runs; the previous inline workflow check tested dict membership in that
list, which never matched, so it could not detect anything.

Noise policy. Baseline and current run in different jobs on different shared runners,
so a raw mean ratio alone is mostly cross-machine noise. A benchmark is a regression
only when ALL of the following hold:

1. ``change_pct = (current_mean - baseline_mean) / baseline_mean * 100`` exceeds
   ``--threshold`` (default 20%).
2. The absolute slowdown exceeds ``--noise-k`` (default 2.0) times the combined
   standard deviation ``sqrt(baseline_stddev**2 + current_stddev**2)``. With k=2 the
   slowdown has to sit about two "sigmas" of the per-round spread outside the observed
   run-to-run variability of both measurements, so jittery benchmarks need a larger
   effect before they are flagged, while tight benchmarks are flagged on the
   percentage threshold alone. (The per-round stddev is deliberately used instead of the
   standard error of the mean: SEM shrinks with round count and would ignore the
   machine-to-machine variance that dominates here.)
3. The baseline mean is at least ``--min-mean-us`` microseconds (default 1.0).
   Sub-microsecond benchmarks measure little more than harness/call overhead, timer
   granularity and CPU frequency or cache state, which differ between heterogeneous
   runners; a 20% swing on them is not actionable.

Modes: ``--mode warn`` (default) reports regressions as GitHub ``::warning::``
annotations and exits 0; ``--mode fail`` exits 1 when any regression is found.
Missing or malformed reports always exit 2. Zero matched benchmarks emits a loud
warning (and fails in ``fail`` mode): an empty comparison is exactly the silent-pass
failure this script replaces. When ``GITHUB_STEP_SUMMARY`` is set, a markdown table is
appended to it.

Usage: ``benchmark_regressions.py baseline.json current.json [--mode warn|fail]``
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

EXIT_OK = 0
EXIT_REGRESSION = 1
EXIT_BAD_INPUT = 2

DEFAULT_THRESHOLD_PCT = 20.0
DEFAULT_NOISE_K = 2.0
DEFAULT_MIN_MEAN_US = 1.0


class BenchmarkReportError(ValueError):
    """Raised when a pytest-benchmark report is missing or malformed."""


@dataclass(frozen=True)
class Comparison:
    """One benchmark matched by ``fullname`` across baseline and current."""

    fullname: str
    baseline_mean: float
    current_mean: float
    baseline_stddev: float
    current_stddev: float
    change_pct: float
    noise_band: float
    regression: bool
    reason: str


@dataclass
class ComparisonResult:
    """All comparisons plus benchmarks present on only one side."""

    comparisons: list[Comparison] = field(default_factory=list)
    only_in_baseline: list[str] = field(default_factory=list)
    only_in_current: list[str] = field(default_factory=list)

    @property
    def regressions(self) -> list[Comparison]:
        return [c for c in self.comparisons if c.regression]


def _index(report: Any, label: str) -> dict[str, dict[str, float]]:
    """Return ``{fullname: stats}`` for a parsed report, validating its shape."""
    if not isinstance(report, dict):
        raise BenchmarkReportError(f"{label}: report is not a JSON object")
    benchmarks = report.get("benchmarks")
    if not isinstance(benchmarks, list):
        raise BenchmarkReportError(f"{label}: 'benchmarks' must be a list")
    indexed: dict[str, dict[str, float]] = {}
    for i, bench in enumerate(benchmarks):
        if not isinstance(bench, dict):
            raise BenchmarkReportError(f"{label}: benchmarks[{i}] is not an object")
        fullname = bench.get("fullname")
        stats = bench.get("stats")
        if not isinstance(fullname, str) or not fullname:
            raise BenchmarkReportError(f"{label}: benchmarks[{i}] has no 'fullname'")
        if not isinstance(stats, dict):
            raise BenchmarkReportError(f"{label}: {fullname} has no 'stats' object")
        try:
            mean = float(stats["mean"])
            stddev = float(stats.get("stddev", 0.0) or 0.0)
        except (KeyError, TypeError, ValueError) as exc:
            raise BenchmarkReportError(f"{label}: {fullname} has invalid stats: {exc}") from exc
        if not math.isfinite(mean) or mean < 0 or not math.isfinite(stddev) or stddev < 0:
            raise BenchmarkReportError(f"{label}: {fullname} has non-finite/negative stats")
        if fullname in indexed:
            raise BenchmarkReportError(f"{label}: duplicate benchmark fullname {fullname}")
        indexed[fullname] = {"mean": mean, "stddev": stddev}
    return indexed


def compare(
    baseline: Any,
    current: Any,
    *,
    threshold_pct: float = DEFAULT_THRESHOLD_PCT,
    noise_k: float = DEFAULT_NOISE_K,
    min_mean_us: float = DEFAULT_MIN_MEAN_US,
) -> ComparisonResult:
    """Compare two parsed pytest-benchmark reports, matching benchmarks by ``fullname``."""
    base_idx = _index(baseline, "baseline")
    cur_idx = _index(current, "current")
    result = ComparisonResult(
        only_in_baseline=sorted(set(base_idx) - set(cur_idx)),
        only_in_current=sorted(set(cur_idx) - set(base_idx)),
    )
    min_mean_s = min_mean_us / 1e6
    for name in sorted(set(base_idx) & set(cur_idx)):
        b, c = base_idx[name], cur_idx[name]
        delta = c["mean"] - b["mean"]
        change_pct = (delta / b["mean"] * 100.0) if b["mean"] > 0 else 0.0
        noise_band = noise_k * math.hypot(b["stddev"], c["stddev"])
        if b["mean"] <= 0 or b["mean"] < min_mean_s:
            regression, reason = False, "below min mean"
        elif change_pct <= threshold_pct:
            regression, reason = False, "ok" if change_pct <= 0 else "under threshold"
        elif delta <= noise_band:
            regression, reason = False, "within noise"
        else:
            regression, reason = True, "REGRESSION"
        result.comparisons.append(
            Comparison(
                fullname=name,
                baseline_mean=b["mean"],
                current_mean=c["mean"],
                baseline_stddev=b["stddev"],
                current_stddev=c["stddev"],
                change_pct=change_pct,
                noise_band=noise_band,
                regression=regression,
                reason=reason,
            )
        )
    return result


def _fmt_s(seconds: float) -> str:
    if seconds >= 1:
        return f"{seconds:.3f} s"
    if seconds >= 1e-3:
        return f"{seconds * 1e3:.3f} ms"
    return f"{seconds * 1e6:.3f} us"


def render_markdown(result: ComparisonResult, threshold_pct: float, noise_k: float) -> str:
    """Render a markdown summary (used for both stdout and the step summary)."""
    lines = [
        "## Benchmark baseline comparison",
        "",
        f"Matched {len(result.comparisons)} benchmark(s); "
        f"{len(result.regressions)} regression(s) "
        f"(>{threshold_pct:g}% slower and > {noise_k:g}x combined stddev).",
        "",
    ]
    if result.comparisons:
        lines += [
            "| Benchmark | Baseline mean | Current mean | Change | Noise band | Status |",
            "|---|---:|---:|---:|---:|---|",
        ]
        ordered = sorted(result.comparisons, key=lambda c: (-c.regression, -c.change_pct))
        for c in ordered:
            lines.append(
                f"| `{c.fullname}` | {_fmt_s(c.baseline_mean)} | {_fmt_s(c.current_mean)} "
                f"| {c.change_pct:+.1f}% | {_fmt_s(c.noise_band)} | {c.reason} |"
            )
        lines.append("")
    if result.only_in_baseline:
        lines.append(f"Only in baseline ({len(result.only_in_baseline)}):")
        lines += [f"- `{n}`" for n in result.only_in_baseline]
        lines.append("")
    if result.only_in_current:
        lines.append(f"Only in current ({len(result.only_in_current)}):")
        lines += [f"- `{n}`" for n in result.only_in_current]
        lines.append("")
    return "\n".join(lines)


def _load(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise BenchmarkReportError(f"{label}: file not found: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkReportError(f"{label}: cannot parse {path}: {exc}") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("baseline", type=Path, help="pytest-benchmark JSON for the base branch")
    parser.add_argument("current", type=Path, help="pytest-benchmark JSON for the PR head")
    parser.add_argument("--mode", choices=("warn", "fail"), default="warn")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD_PCT)
    parser.add_argument("--noise-k", type=float, default=DEFAULT_NOISE_K)
    parser.add_argument("--min-mean-us", type=float, default=DEFAULT_MIN_MEAN_US)
    args = parser.parse_args(argv)

    try:
        result = compare(
            _load(args.baseline, "baseline"),
            _load(args.current, "current"),
            threshold_pct=args.threshold,
            noise_k=args.noise_k,
            min_mean_us=args.min_mean_us,
        )
    except BenchmarkReportError as exc:
        print(f"::error::benchmark comparison input invalid: {exc}")
        return EXIT_BAD_INPUT

    summary = render_markdown(result, args.threshold, args.noise_k)
    print(summary)
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as fh:
            fh.write(summary + "\n")

    level = "error" if args.mode == "fail" else "warning"
    failed = False
    if not result.comparisons:
        print(
            f"::{level}::No benchmarks matched by fullname between baseline and current "
            f"({len(result.only_in_baseline)} baseline-only, "
            f"{len(result.only_in_current)} current-only); nothing was compared."
        )
        failed = True
    for c in result.regressions:
        print(
            f"::{level}::Performance regression {c.fullname}: "
            f"{_fmt_s(c.baseline_mean)} -> {_fmt_s(c.current_mean)} ({c.change_pct:+.1f}%)"
        )
        failed = True
    if failed and args.mode == "fail":
        return EXIT_REGRESSION
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
