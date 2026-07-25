#!/usr/bin/env python3
# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
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
# See the License for the specific language governing permissions and
# limitations under the License.

"""Classify and archive noise in ``~/.victor/evaluations``.

The corpus mixes three very different things, and counting them together made
the earlier prompt-evolution audit hard to read (see
``docs/analysis/2026-07-25-prompt-evolution-audit.md``):

* **synthetic** — ``model`` is ``test``/``test-model``: fixtures written by the
  test suite, never a real run.
* **inert** — a real model, but no task did any tool call: nothing an agent
  actually did, so nothing to learn from.
* **signal** — a real model with real agent activity. This is the corpus that
  matters for Pareto seeding.

Nothing is deleted. Files move to ``evaluations/archive/<class>/`` so counts
stop lying while the data stays recoverable; ``--undo`` moves them back.

    prune_eval_corpus.py report                 # classify, change nothing
    prune_eval_corpus.py archive                # dry run
    prune_eval_corpus.py archive --apply
    prune_eval_corpus.py undo --apply           # restore everything archived
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

DEFAULT_DIR = Path(os.environ.get("VICTOR_EVAL_DIR", Path.home() / ".victor" / "evaluations"))
ARCHIVE_NAME = "archive"
SYNTHETIC_MODELS = {"test", "test-model", ""}

SYNTHETIC = "synthetic"
INERT = "inert"
SIGNAL = "signal"
UNREADABLE = "unreadable"


def _load(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path) as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def classify(payload: Optional[Dict[str, Any]]) -> Tuple[str, str]:
    """Return ``(class, reason)`` for one artifact."""
    if payload is None:
        return UNREADABLE, "not readable as a JSON object"

    config = payload.get("config") if isinstance(payload.get("config"), dict) else {}
    model = str(payload.get("model") or config.get("model") or "").strip()
    if model in SYNTHETIC_MODELS:
        return SYNTHETIC, f"model={model or '(absent)'}"

    tasks = [t for t in (payload.get("tasks") or []) if isinstance(t, dict)]
    if not tasks:
        # Session-shaped artifacts carry a single task inline rather than a list.
        if payload.get("task_id"):
            tasks = [payload]
        else:
            return INERT, "no tasks recorded"

    active = sum(1 for t in tasks if (t.get("tool_calls") or 0) > 0)
    if active == 0:
        return INERT, f"{len(tasks)} task(s), none with a tool call"
    return SIGNAL, f"{active}/{len(tasks)} task(s) with agent activity"


def _artifacts(eval_dir: Path) -> List[Path]:
    archive = eval_dir / ARCHIVE_NAME
    return sorted(
        p
        for p in eval_dir.glob("eval_*.json")
        if archive not in p.parents  # never re-scan what we already archived
    )


def _classified(eval_dir: Path) -> Iterable[Tuple[Path, str, str]]:
    for path in _artifacts(eval_dir):
        kind, reason = classify(_load(path))
        yield path, kind, reason


def _joinable_counts(eval_dir: Path) -> Tuple[int, int]:
    """(tasks in signal artifacts, of which carry a session_id)."""
    total = joinable = 0
    for path in _artifacts(eval_dir):
        payload = _load(path)
        kind, _ = classify(payload)
        if kind != SIGNAL or payload is None:
            continue
        for task in payload.get("tasks") or []:
            if not isinstance(task, dict):
                continue
            total += 1
            if task.get("session_id"):
                joinable += 1
    return total, joinable


def cmd_report(args) -> int:
    counts: collections.Counter = collections.Counter()
    examples: Dict[str, str] = {}
    for path, kind, reason in _classified(args.dir):
        counts[kind] += 1
        examples.setdefault(kind, f"{path.name} — {reason}")

    total = sum(counts.values())
    if not total:
        print(f"No artifacts in {args.dir}.")
        return 0

    print(f"{total} artifact(s) in {args.dir}\n")
    for kind in (SIGNAL, SYNTHETIC, INERT, UNREADABLE):
        if not counts[kind]:
            continue
        print(f"  {kind:<11} {counts[kind]:>5}  ({100 * counts[kind] / total:.0f}%)")
        print(f"              e.g. {examples[kind]}")

    tasks, joinable = _joinable_counts(args.dir)
    if tasks:
        print(f"\nTasks in signal artifacts: {tasks}")
        print(
            f"  with a session_id (joinable to a trace): {joinable} "
            f"({100 * joinable / tasks:.0f}%)"
        )
        if joinable < tasks:
            print(
                "  Artifacts written before session_id was serialized cannot be "
                "joined retroactively; only new runs carry it."
            )
    return 0


def _archive_plan(eval_dir: Path, kinds: set) -> List[Tuple[Path, Path]]:
    plan = []
    for path, kind, _ in _classified(eval_dir):
        if kind in kinds:
            plan.append((path, eval_dir / ARCHIVE_NAME / kind / path.name))
    return plan


def cmd_archive(args) -> int:
    kinds = {SYNTHETIC, INERT} | ({UNREADABLE} if args.include_unreadable else set())
    plan = _archive_plan(args.dir, kinds)
    if not plan:
        print("Nothing to archive.")
        return 0

    by_kind: collections.Counter = collections.Counter(dest.parent.name for _, dest in plan)
    for kind, n in by_kind.most_common():
        print(f"  {kind}: {n}")

    if not args.apply:
        print(f"\nDry run: {len(plan)} artifact(s) would move under {args.dir / ARCHIVE_NAME}/.")
        print("Nothing is deleted — re-run with --apply, and `undo --apply` reverses it.")
        return 0

    moved = 0
    for src, dest in plan:
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dest))
            moved += 1
        except OSError as exc:
            print(f"  skipped {src.name}: {exc}", file=sys.stderr)
    print(f"\nArchived {moved} artifact(s) under {args.dir / ARCHIVE_NAME}/.")
    return 0


def cmd_undo(args) -> int:
    archive = args.dir / ARCHIVE_NAME
    staged = list(archive.glob("*/eval_*.json")) if archive.is_dir() else []
    if not staged:
        print("Nothing archived to restore.")
        return 0
    if not args.apply:
        print(f"Dry run: {len(staged)} artifact(s) would return to {args.dir}.")
        return 0
    restored = 0
    for path in staged:
        try:
            shutil.move(str(path), str(args.dir / path.name))
            restored += 1
        except OSError as exc:
            print(f"  skipped {path.name}: {exc}", file=sys.stderr)
    print(f"Restored {restored} artifact(s) to {args.dir}.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR, help=f"default: {DEFAULT_DIR}")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("report", help="classify the corpus, change nothing")

    archive = sub.add_parser("archive", help="move synthetic/inert artifacts aside")
    archive.add_argument("--apply", action="store_true", help="actually move; default is a dry run")
    archive.add_argument(
        "--include-unreadable",
        action="store_true",
        help="also archive files that will not parse",
    )

    undo = sub.add_parser("undo", help="restore everything previously archived")
    undo.add_argument("--apply", action="store_true", help="actually move; default is a dry run")

    args = parser.parse_args()
    return {"report": cmd_report, "archive": cmd_archive, "undo": cmd_undo}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
