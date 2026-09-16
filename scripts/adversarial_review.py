#!/usr/bin/env python3
"""Record and verify local, commit-bound adversarial-review attestations."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

SCHEMA_VERSION = 1
ZERO_SHA = "0" * 40


class ReviewGateError(RuntimeError):
    """Raised when the local review gate cannot validate an attestation."""


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip() or "git command failed"
        raise ReviewGateError(detail)
    return result.stdout.strip()


def _repo_root(repo: Path) -> Path:
    return Path(_git(repo, "rev-parse", "--show-toplevel")).resolve()


def _resolve_commit(repo: Path, ref: str) -> str:
    sha = _git(repo, "rev-parse", "--verify", f"{ref}^{{commit}}")
    if len(sha) != 40:
        raise ReviewGateError(f"Could not resolve a full commit SHA for {ref!r}")
    return sha


def _attestation_dir(repo: Path) -> Path:
    root = _repo_root(repo)
    common_dir = Path(_git(root, "rev-parse", "--git-common-dir"))
    if not common_dir.is_absolute():
        common_dir = root / common_dir
    return common_dir.resolve() / "victor" / "adversarial-reviews"


def _attestation_path(repo: Path, commit_sha: str) -> Path:
    return _attestation_dir(repo) / f"{commit_sha}.json"


def record_review(
    repo: Path,
    *,
    commit_ref: str,
    base_ref: str,
    reviewer: str,
    summary: str,
) -> Path:
    """Record a clean review for one immutable commit."""
    reviewer = reviewer.strip()
    summary = summary.strip()
    if not reviewer:
        raise ReviewGateError("Reviewer identity must not be empty")
    if not summary:
        raise ReviewGateError("Review summary must not be empty")

    commit_sha = _resolve_commit(repo, commit_ref)
    base_sha = _resolve_commit(repo, base_ref)
    tree_sha = _git(repo, "show", "-s", "--format=%T", commit_sha)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "verdict": "clean",
        "commit_sha": commit_sha,
        "tree_sha": tree_sha,
        "base_ref": base_ref,
        "base_sha": base_sha,
        "reviewer": reviewer,
        "summary": summary,
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
    }

    path = _attestation_path(repo, commit_sha)
    try:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.chmod(0o600)
        temporary.replace(path)
    except OSError as exc:
        raise ReviewGateError(f"Could not write local review attestation: {exc}") from exc
    return path


def _load_attestation(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ReviewGateError("No clean adversarial review is recorded for this commit") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ReviewGateError(f"Review attestation is unreadable: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReviewGateError("Review attestation must contain a JSON object")
    return payload


def verify_review(repo: Path, commit_ref: str) -> dict[str, Any]:
    """Verify that one commit has a well-formed, clean local attestation."""
    commit_sha = _resolve_commit(repo, commit_ref)
    payload = _load_attestation(_attestation_path(repo, commit_sha))
    expected_tree = _git(repo, "show", "-s", "--format=%T", commit_sha)

    expected = {
        "schema_version": SCHEMA_VERSION,
        "verdict": "clean",
        "commit_sha": commit_sha,
        "tree_sha": expected_tree,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ReviewGateError(f"Review attestation has invalid {key!r}")
    for key in ("base_ref", "base_sha", "reviewer", "summary", "reviewed_at"):
        if not isinstance(payload.get(key), str) or not payload[key].strip():
            raise ReviewGateError(f"Review attestation is missing {key!r}")
    if len(payload["base_sha"]) != 40 or any(
        character not in "0123456789abcdef" for character in payload["base_sha"].lower()
    ):
        raise ReviewGateError("Review attestation has invalid 'base_sha'")
    return payload


def _commit_from_environment(repo: Path) -> str | None:
    remote_branch = os.environ.get("PRE_COMMIT_REMOTE_BRANCH", "")
    if remote_branch and not remote_branch.startswith("refs/heads/"):
        return None

    candidate = os.environ.get("PRE_COMMIT_TO_REF", "").strip()
    if candidate == ZERO_SHA:
        return None
    return candidate or _resolve_commit(repo, "HEAD")


def _record_command(commit_sha: str) -> str:
    return (
        "python scripts/adversarial_review.py record "
        f"--commit {commit_sha} --reviewer <reviewer-or-session> "
        '--summary "<findings and disposition>"'
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd(), help=argparse.SUPPRESS)
    subparsers = parser.add_subparsers(dest="command", required=True)

    record = subparsers.add_parser("record", help="record a clean review for a commit")
    record.add_argument("--commit", default="HEAD", help="commit reviewed (default: HEAD)")
    record.add_argument("--base", default="origin/develop", help="review base ref")
    record.add_argument("--reviewer", required=True, help="independent reviewer or session")
    record.add_argument("--summary", required=True, help="findings and their disposition")

    check = subparsers.add_parser("check", help="verify the commit being pushed")
    check.add_argument("--commit", help="commit to verify (default: pre-push ref or HEAD)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    commit_ref = getattr(args, "commit", None) or "HEAD"
    try:
        if args.command == "record":
            path = record_review(
                args.repo,
                commit_ref=args.commit,
                base_ref=args.base,
                reviewer=args.reviewer,
                summary=args.summary,
            )
            commit_sha = _resolve_commit(args.repo, args.commit)
            print(f"Recorded clean adversarial review for {commit_sha}")
            print(f"Local attestation: {path}")
            return 0

        commit_ref = args.commit or _commit_from_environment(args.repo)
        if commit_ref is None:
            print("Adversarial review gate: no branch commit to verify")
            return 0
        payload = verify_review(args.repo, commit_ref)
        print(
            "Adversarial review gate passed for " f"{payload['commit_sha']} ({payload['reviewer']})"
        )
        return 0
    except ReviewGateError as exc:
        print(f"Adversarial review gate failed: {exc}", file=sys.stderr)
        try:
            commit_sha = _resolve_commit(args.repo, commit_ref or "HEAD")
        except ReviewGateError:
            commit_sha = "HEAD"
        print(
            f"After an independent review, record it with:\n  {_record_command(commit_sha)}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
