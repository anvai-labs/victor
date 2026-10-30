# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Import smoke for source files whose diff only removes imports (ci-fast).

``select_changed_tests.py`` exempts import-only changes from mirror tests;
this is what guards them instead. Every such module is imported. A failure
counts against the change only if it is new: the module's merge-base source
is executed the same way, and an identical failure (type and message) there
is reported as pre-existing rather than failing the job.

Usage: ``import_smoke.py --base <ref> < paths.txt``
"""

from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
import types
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[2]

Failure = tuple[str, str]


def _module_name(path: str) -> str:
    return path[: -len(".py")].replace("/", ".").removesuffix(".__init__")


def _describe(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _run_source(name: str, source: str, path: str) -> str | None:
    """Execute ``source`` as module ``name``; return its failure, if any."""
    module = types.ModuleType(name)
    module.__file__ = str(ROOT / path)
    is_package = path.endswith("__init__.py")
    module.__package__ = name if is_package else name.rpartition(".")[0]
    try:
        # nosemgrep: python.lang.security.audit.exec-detected.exec-detected
        exec(compile(source, path, "exec"), module.__dict__)  # noqa: S102 - trusted repo source
    except Exception as exc:
        return _describe(exc)
    return None


def smoke(
    paths: list[str],
    read_base: Callable[[str], str | None],
    root: Path = ROOT,
) -> tuple[list[Failure], list[Failure]]:
    """Import each path; return ``(new_failures, pre_existing_failures)``."""
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    new: list[Failure] = []
    pre_existing: list[Failure] = []
    for path in paths:
        name = _module_name(path)
        try:
            importlib.import_module(name)  # nosemgrep - repo module names
            continue
        except Exception as exc:
            failure = _describe(exc)
        base_source = read_base(path)
        if base_source is not None and _run_source(name, base_source, path) == failure:
            pre_existing.append((name, failure))
        else:
            new.append((name, failure))
    return new, pre_existing


def _git_reader(ref: str) -> Callable[[str], str | None]:
    def read(path: str) -> str | None:
        result = subprocess.run(
            ["git", "show", f"{ref}:{path}"], cwd=ROOT, capture_output=True, text=True
        )
        return result.stdout if result.returncode == 0 else None

    return read


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="merge-base ref of the change")
    args = parser.parse_args()
    paths = sys.stdin.read().split()
    new, pre_existing = smoke(paths, _git_reader(args.base))
    for name, failure in pre_existing:
        print(f"::warning::pre-existing import failure (unchanged by this PR): {name}: {failure}")
    for name, failure in new:
        print(f"::error::import-only change broke {name}: {failure}")
    print(f"import_smoke: {len(paths)} module(s), {len(new)} new failure(s)")
    raise SystemExit(1 if new else 0)
