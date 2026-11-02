#!/usr/bin/env python3
"""Fail closed on the complete CI dependency contract for the current event."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping

CORE_JOBS = (
    "windows-source-changes",
    "windows-source",
    "format",
    "lint",
    "repo-hygiene",
    "docs-drift",
    "typecheck",
    "strict-typecheck",
    "imports",
    "imports-with-verticals",
    "strict-fallback-guards",
    "definition-import-boundaries",
    "version-sync",
    "guards",
    "hotspot-size-guard",
    "security-quick",
    "quick-tests",
    "collect-check",
    "attribution",
    "lint-title",
    "vertical-boundaries",
    "fep-validate",
    "secret-scan",
    "code-security",
    "dependency-audit",
    "rust-packages",
    "native-parity",
    "duplication-ratchet",
)
PROMOTION_JOBS = (
    "promotion-tests",
    "promotion-integration",
    "promotion-build",
    "promotion-security",
    "promotion-validation",
    "promotion-performance",
    "promotion-contracts",
    "promotion-codegraph",
    "promotion-verticals",
    "promotion-vertical-packages",
)
MAIN_DELEGATED_JOBS = frozenset(
    {"security-quick", "secret-scan", "code-security", "dependency-audit"}
)


def validate_results(needs: Mapping[str, object], *, event: str, base: str, ref: str = "") -> None:
    """Require all event-appropriate jobs, rejecting absent or unknown results.

    Main pull requests delegate duplicate fast security scans to their complete
    reusable security workflow. Other pull requests keep the lightweight gate.
    Intentional skips are specific to the event and job; a skipped promotion
    surface can never authorize a main merge.
    """
    if not isinstance(needs, Mapping):
        raise ValueError("CI dependency results must be an object")
    if event not in {"pull_request", "push", "workflow_dispatch"}:
        raise ValueError(f"Unknown CI event: {event!r}")
    expected = set(CORE_JOBS) | set(PROMOTION_JOBS)
    missing, extra = expected - needs.keys(), needs.keys() - expected
    if missing or extra:
        raise ValueError(
            f"CI dependency mismatch: missing={sorted(missing)}, unexpected={sorted(extra)}"
        )
    selector = needs["windows-source-changes"]
    outputs = selector.get("outputs") if isinstance(selector, Mapping) else None
    windows_required = outputs.get("required") if isinstance(outputs, Mapping) else None
    if not isinstance(windows_required, str) or windows_required not in {"true", "false"}:
        raise ValueError("Windows source qualification flag is missing or unknown")
    promotion = event == "pull_request" and base == "main"
    allowed_skips = set(MAIN_DELEGATED_JOBS) | {"quick-tests"} if promotion else set()
    if event != "pull_request":
        allowed_skips.add("lint-title")
    if event == "push" and ref == "refs/heads/main":
        allowed_skips.add("quick-tests")
    failures = []
    for job in sorted(expected):
        record = needs[job]
        result = record.get("result") if isinstance(record, Mapping) else None
        if job in PROMOTION_JOBS:
            allowed = {"success"} if promotion else {"skipped"}
        elif job == "windows-source":
            allowed = {"success"} if windows_required == "true" else {"skipped"}
        else:
            allowed = {"success", "skipped"} if job in allowed_skips else {"success"}
        if not isinstance(result, str) or result not in allowed:
            failures.append(f"{job}={result!r} (expected {sorted(allowed)})")
    if failures:
        raise ValueError("CI prerequisites did not pass: " + "; ".join(failures))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--needs", required=True)
    parser.add_argument("--event", required=True)
    parser.add_argument("--base", default="")
    parser.add_argument("--ref", default="")
    args = parser.parse_args()
    try:
        validate_results(json.loads(args.needs), event=args.event, base=args.base, ref=args.ref)
    except (ValueError, TypeError) as exc:
        parser.exit(1, f"CI gate failed: {exc}\n")
    print("All event-appropriate CI prerequisites passed")


if __name__ == "__main__":
    main()
