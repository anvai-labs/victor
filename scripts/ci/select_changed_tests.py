#!/usr/bin/env python3
# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Select unit-test targets for a set of changed files (fast develop gate).

Maps each changed file to the unit tests worth running on a develop PR, leveraging the repo's
mirrored layout (``tests/unit`` mirrors ``victor``):

- A changed **test** file (``tests/unit/**/test_*.py``) -> run it directly.
- A changed **source** file (``victor/<rel>/<name>.py``) -> run its mirror tests
  ``tests/unit/<rel>/test_<name>.py`` and ``tests/unit/<rel>/test_<name>_*.py`` (when they exist).

Prints existing pytest targets (one per line, deduped, sorted). Empty output is valid only when
no core Python source changed. A changed ``victor/**.py`` file without a mirror test is an error:
silently deferring it to the later develop-to-main promotion leaves develop untested.

Usage: ``select_changed_tests.py <changed_file> [<changed_file> ...]``
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# A change that maps past this many targets is a *sweeping* edit — a repo-wide
# rename, license/email-header pass, or formatting run — not a proportional
# feature change. Selecting its (near-full-suite) target set blows the fast
# gate's wall-clock budget (observed: a 3,265-file email sweep mapped to
# thousands of targets and the 25-min job was cancelled at ~5%). Past the cap we
# select NOTHING and lean on the documented safety net — the sharded full suite
# at develop->main — exactly as an unmapped change already does. Set well above
# any real feature PR's footprint so it only trips on mechanical sweeps.
MAX_SELECTED_TARGETS = 200


class SelectionError(ValueError):
    """The changed core source cannot be covered safely by the fast gate."""


def select(changed: list[str]) -> list[str]:
    targets: set[str] = set()
    unmapped_sources: list[str] = []
    for raw in changed:
        p = raw.strip()
        if not p.endswith(".py"):
            continue
        path = Path(p)
        name = path.name
        # Changed test file -> run it directly.
        if p.startswith("tests/") and name.startswith("test_"):
            if (ROOT / p).exists():
                targets.add(p)
            continue
        # Changed source file -> map to its mirror tests.
        if p.startswith("victor/"):
            rel = path.relative_to("victor")
            test_dir = ROOT / "tests" / "unit" / rel.parent
            candidates: list[Path] = []
            if test_dir.is_dir():
                candidates = list(test_dir.glob(f"test_{rel.stem}.py")) + list(
                    test_dir.glob(f"test_{rel.stem}_*.py")
                )
            # Some older suites flatten one source directory level (for
            # example benchmarks/deep_research.py is covered by
            # evaluation/test_deep_research_benchmark.py). Preserve those
            # explicit stem matches before declaring the source untested.
            if not candidates:
                unit_root = ROOT / "tests" / "unit"
                candidates = list(unit_root.rglob(f"test_{rel.stem}.py")) + list(
                    unit_root.rglob(f"test_{rel.stem}_*.py")
                )
            for cand in candidates:
                targets.add(str(cand.relative_to(ROOT)))
            if not candidates:
                unmapped_sources.append(p)
    if unmapped_sources:
        joined = "\n  - ".join(unmapped_sources)
        raise SelectionError(
            "changed core source has no mirrored unit test; add a test target before merging:\n"
            f"  - {joined}"
        )
    result = sorted(targets)
    if len(result) > MAX_SELECTED_TARGETS:
        raise SelectionError(
            f"select_changed_tests: {len(result)} targets exceed the "
            f"{MAX_SELECTED_TARGETS} cap — treating as a sweeping/mechanical change; "
            "split the change or run it through an explicitly sharded full-suite gate."
        )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*")
    parser.add_argument(
        "--stdin",
        action="store_true",
        help="read one changed path per line from stdin (safe for spaces and leading dashes)",
    )
    args = parser.parse_args()
    changed = [line.rstrip("\n") for line in sys.stdin] if args.stdin else args.files
    try:
        for target in select(changed):
            print(target)
    except SelectionError as exc:
        print(f"select_changed_tests: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
