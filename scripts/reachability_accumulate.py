#!/usr/bin/env python3
# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""FEP-0022 Phase 2 — accumulate reachability sidecars + run the offline oracle.

Offline, triage-required (no CI gate; that is Phase 3). Two subcommands:

  accumulate <sidecar.jsonl ...> -o baseline.json
      Merge one or more Phase-1 sidecars into the ever-observed baseline.

  report baseline.json [--exempt FILE] [--registered FILE|bootstrap]
      Diff the baseline against live DI registrations (minus the exempt list)
      and print the candidate-dead set.

The "registered" set is environment-specific (it depends on the full agent
bootstrap, which only the trajectory harness exercises completely).
``--registered bootstrap`` reads whatever the runtime container has registered;
if that is empty in your context, dump the keys once and pass
``--registered file.txt`` (one ``module:QualName`` per line).

Examples:
  python scripts/reachability_accumulate.py accumulate reachability-*.jsonl -o baseline.json
  python scripts/reachability_accumulate.py report baseline.json --exempt exempt.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from victor.runtime.reachability import (
    candidate_dead,
    load_baseline,
    load_exempt,
    merge_sidecar_paths,
    write_baseline,
)


def _registered_from_bootstrap() -> set[str]:
    """Best-effort: read registered service keys from the runtime container.

    Returns whatever is currently registered. In a bare process the container
    may be empty — run this through the trajectory harness, or dump the keys
    once and pass ``--registered file``.
    """
    from victor.core import get_container

    container = get_container()
    return {f"{t.__module__}:{t.__qualname__}" for t in container.get_registered_types()}


def _registered_from_file(path: Path) -> set[str]:
    keys: set[str] = set()
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        entry = raw.split("#", 1)[0].strip()
        if entry:
            keys.add(entry)
    return keys


def _resolve_registered(arg: str) -> set[str]:
    if arg == "bootstrap":
        return _registered_from_bootstrap()
    return _registered_from_file(Path(arg))


def cmd_accumulate(args: argparse.Namespace) -> int:
    merged = merge_sidecar_paths(args.sidecars)
    out = write_baseline(merged, args.output)
    total = sum(len(v) for v in merged.values())
    print(
        f"wrote {out}: {total} unique witness(es) across "
        f"{len(args.sidecars)} sidecar(s), kinds={sorted(merged)}"
    )
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    baseline = load_baseline(args.baseline)
    observed = baseline.get("di", set())
    exempt = load_exempt(args.exempt) if args.exempt else set()
    registered = _resolve_registered(args.registered)

    candidates = candidate_dead(registered, observed, exempt)
    print(f"registered: {len(registered)}")
    print(f"observed:   {len(observed)} (di)")
    print(f"exempt:     {len(exempt)}")
    print(f"candidates: {len(candidates)}  (registered, never observed, not exempt)")
    for key in candidates:
        print(f"  {key}")
    if not registered:
        print(
            "\nnote: registered set is empty -- bootstrap didn't populate the "
            "container.\n      run via the trajectory harness or pass "
            "--registered <keys.txt>.",
            file=sys.stderr,
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_acc = sub.add_parser("accumulate", help="merge sidecars -> baseline.json")
    p_acc.add_argument("sidecars", nargs="+", type=Path)
    p_acc.add_argument("-o", "--output", required=True, type=Path)
    p_acc.set_defaults(func=cmd_accumulate)

    p_rep = sub.add_parser("report", help="diff baseline vs registered -> candidates")
    p_rep.add_argument("baseline", type=Path)
    p_rep.add_argument("--exempt", type=Path)
    p_rep.add_argument(
        "--registered",
        default="bootstrap",
        help="'bootstrap' (default) or a path to a keys file",
    )
    p_rep.set_defaults(func=cmd_report)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
