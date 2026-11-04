"""Select actual-Windows qualification from a known diff, failing closed."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
import os
from pathlib import Path
import re
import subprocess

_EXACT = {
    ".github/workflows/ci-fast.yml",
    ".github/workflows/ci-graph-windows.yml",
    "scripts/ci/windows-source-coverage.ini",
    "scripts/ci/windows_source_required.py",
    "scripts/ci/gate_results.py",
    "scripts/ci/select_changed_tests.py",
    "scripts/benchmark_windows_source.py",
    "pyproject.toml",
    "constraints.txt",
}
_PREFIXES = (
    "victor/core/graph_rag/",
    "tests/unit/core/graph_rag/",
    "victor-contracts/",
    "tests/unit/scripts/test_windows_source_required",
    "tests/unit/scripts/test_promotion_gates",
    "tests/unit/scripts/test_benchmark_windows_source",
)


def paths_require_windows(paths: Iterable[str]) -> bool:
    """Require qualification for source, runtime dependency and gate changes."""
    return any(
        path in _EXACT
        or path.startswith(_PREFIXES)
        or (path.startswith("requirements") and path.endswith(".txt"))
        for path in paths
    )


def required(*, event: str, base: str, cwd: Path | None = None) -> bool:
    """Only a successful known diff may return False; manual runs qualify all."""
    if event == "workflow_dispatch":
        return True
    if event not in {"pull_request", "push"}:
        raise ValueError("Unknown Windows qualification event")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", base):
        raise ValueError("Windows qualification requires a known full base SHA")
    if base == "0" * 40:
        if event == "push":
            return True
        raise ValueError("Pull request qualification requires a real base SHA")
    result = subprocess.run(
        ["git", "diff", "--name-only", "--no-renames", "-z", base, "HEAD"],
        cwd=cwd,
        check=True,
        capture_output=True,
    ).stdout
    if result and not result.endswith(b"\0"):
        raise ValueError("Truncated Windows source change inventory")
    return paths_require_windows(
        path for path in result.decode("utf-8", errors="strict").split("\0") if path
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", required=True)
    parser.add_argument("--base", default="")
    args = parser.parse_args()
    value = "true" if required(event=args.event, base=args.base) else "false"
    # GitHub output is written only after successful complete selection.
    with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as stream:
        stream.write(f"required={value}\n")
    print(f"Windows source qualification required: {value}")


if __name__ == "__main__":
    main()
