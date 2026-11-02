#!/usr/bin/env python3
"""Resolve an ancestor baseline for changed-file tests."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import json
from pathlib import Path
import re
import subprocess

_SHA = re.compile(r"[0-9a-f]{40}")


def choose_base(
    runs: Sequence[Mapping[str, object]],
    *,
    head_sha: str,
    branch: str,
    repository: str,
    push_before: str,
    is_ancestor: Callable[[str], bool],
) -> str:
    """Prefer the newest green push from this branch and repository's history.

    Promotion PRs deliberately skip quick tests and cannot establish a green
    baseline. Manual dispatches, foreign repositories, unrelated history and
    the current run's own head likewise provide no such evidence.
    """
    eligible: list[tuple[int, str]] = []
    for run in runs:
        if not isinstance(run, Mapping):
            continue
        sha = run.get("head_sha")
        number = run.get("run_number")
        head_repo = run.get("head_repository")
        if (
            run.get("event") != "push"
            or branch == "main"  # Protected main pushes delegate tests to promotion.
            or run.get("status") != "completed"
            or run.get("conclusion") != "success"
            or run.get("head_branch") != branch
            or not isinstance(head_repo, Mapping)
            or head_repo.get("full_name") != repository
            or not isinstance(sha, str)
            or _SHA.fullmatch(sha) is None
            or sha == head_sha
            or not isinstance(number, int)
            or isinstance(number, bool)
        ):
            continue
        eligible.append((number, sha))
    for _, sha in sorted(eligible, reverse=True):
        if is_ancestor(sha):
            return sha
    if _SHA.fullmatch(push_before) and push_before != "0" * 40 and is_ancestor(push_before):
        return push_before
    if is_ancestor("HEAD~1"):
        return "HEAD~1"
    raise ValueError("No valid ancestor baseline for changed-file tests")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--before", default="")
    args = parser.parse_args()

    def is_ancestor(ref: str) -> bool:
        return (
            subprocess.run(
                ["git", "merge-base", "--is-ancestor", ref, args.head],
                check=False,
                capture_output=True,
            ).returncode
            == 0
        )

    try:
        payload = json.loads(args.runs.read_text())
        if not isinstance(payload, dict) or not isinstance(payload.get("workflow_runs"), list):
            raise ValueError("Workflow run response has no run list")
        print(
            choose_base(
                payload["workflow_runs"],
                head_sha=args.head,
                branch=args.branch,
                repository=args.repository,
                push_before=args.before,
                is_ancestor=is_ancestor,
            )
        )
    except (OSError, ValueError, TypeError) as exc:
        parser.exit(1, f"Changed-test baseline failed: {exc}\n")


if __name__ == "__main__":
    main()
