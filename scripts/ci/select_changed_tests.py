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

Prints existing pytest targets (one per line, deduped, sorted). Empty output means "nothing
relevant to run" — the caller should treat that as a pass (the full sharded suite at
develop->main is the safety net). This keeps the per-PR gate proportional to the change instead
of running the ~7h single-process suite.

Usage: ``select_changed_tests.py <changed_file> [<changed_file> ...]``
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def select(changed: list[str]) -> list[str]:
    targets: set[str] = set()
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
            if not test_dir.is_dir():
                continue
            for cand in list(test_dir.glob(f"test_{rel.stem}.py")) + list(
                test_dir.glob(f"test_{rel.stem}_*.py")
            ):
                targets.add(str(cand.relative_to(ROOT)))
    return sorted(targets)


if __name__ == "__main__":
    for target in select(sys.argv[1:]):
        print(target)
