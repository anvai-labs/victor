#!/usr/bin/env python3
# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
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

"""Audit, export, and purge GEPA prompt candidates in ``~/.victor/victor.db``.

Evolution is *local*: candidates accumulate in one operator's SQLite file and
never reach anyone else. Shipping an improvement means promoting its text into
``victor/agent/prompt_section_texts.py``, which is version-controlled, reviewed,
and installed with the package. This script is the bridge:

    prompt_candidates.py audit                 # verdict per candidate
    prompt_candidates.py show <hash>           # full text + diff vs shipped baseline
    prompt_candidates.py export <hash>         # paste-ready Python literal
    prompt_candidates.py propose SECTION        # register a hand-written candidate
    prompt_candidates.py promote <hash>        # write it into the shipped section module
    prompt_candidates.py purge --apply         # drop rejected candidates (backs up first)

``audit`` classifies each candidate against the shipped baseline; only
``PROMOTE`` candidates are worth moving into source. Nothing is deleted without
``--apply``, and ``purge`` always copies the rows to a timestamped backup table
first.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import os
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_DB = Path(os.environ.get("VICTOR_DB", Path.home() / ".victor" / "victor.db"))
TABLE = "agent_prompt_candidate"

# A candidate must beat the shipped baseline on evidence, not just differ from it.
MIN_BENCHMARK_RUNS = 3

# The promotion codegen (marker re-templatizing + section replacement) lives in
# the package so it is tested and reusable; these aliases keep the historical
# ``_helper`` names this script and its tests call.
from victor.framework.rl.prompt_promotion import (  # noqa: E402
    build_promoted_source,
    is_fstring_section as _is_fstring_section,
    marker_placeholders as _marker_placeholders,
    replace_section as _replace_section,
    retemplatize as _retemplatize,
)


@dataclass
class Candidate:
    section_name: str
    provider: str
    text_hash: str
    parent_hash: str
    text: str
    generation: int
    sample_count: int
    is_active: int
    requires_benchmark: int
    benchmark_passed: int
    benchmark_runs: int
    benchmark_score: float
    strategy_chain: str
    created_at: str


def _hygiene():
    """The same corruption checks the runtime persist gate applies."""
    from victor.framework.rl.prompt_hygiene import (
        find_redundant_additions,
        has_truncated_tail,
    )

    return find_redundant_additions, has_truncated_tail


def _baselines() -> Dict[str, str]:
    from victor.agent import prompt_section_texts as pst

    return {name: getattr(pst, name) for name in pst.__all__}


def _md5(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:12]


def _load(db: Path) -> List[Candidate]:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = con.execute(
            f"SELECT section_name, provider, text_hash, COALESCE(parent_hash, ''), text, "
            f"generation, sample_count, is_active, requires_benchmark, benchmark_passed, "
            f"benchmark_runs, benchmark_score, COALESCE(strategy_chain, ''), created_at "
            f"FROM {TABLE} ORDER BY created_at"
        ).fetchall()
    finally:
        con.close()
    return [Candidate(*row) for row in rows]


def _verdict(cand: Candidate, baselines: Dict[str, str]) -> tuple:
    """Return ``(verdict, reasons)`` for one candidate."""
    reasons: List[str] = []
    baseline = baselines.get(cand.section_name)
    find_redundant_additions, has_truncated_tail = _hygiene()

    if baseline is None:
        return "ORPHAN", [f"section '{cand.section_name}' is no longer shipped"]

    if cand.parent_hash and cand.parent_hash != _md5(baseline):
        reasons.append(
            f"parent {cand.parent_hash} != shipped baseline {_md5(baseline)} "
            "(evolved from a stale seed; diff is not against what ships)"
        )

    if has_truncated_tail(cand.text):
        reasons.append("ends mid-sentence — the final instruction is a fragment")

    redundant = find_redundant_additions(baseline, cand.text)
    if redundant:
        reasons.append(f"restates existing guidance: {redundant[0]!r}")

    servable = not cand.requires_benchmark or cand.benchmark_passed
    if cand.benchmark_runs and not cand.benchmark_passed:
        reasons.append(
            f"failed benchmark ({cand.benchmark_score:.2f} over {cand.benchmark_runs} runs)"
        )
    elif cand.benchmark_runs < MIN_BENCHMARK_RUNS:
        reasons.append(
            f"unproven: {cand.benchmark_runs} benchmark runs, {cand.sample_count} live samples"
        )

    # Severity counts defects only. The servable note below is an advisory —
    # it tells the operator this candidate can already be injected, which is a
    # reason to look sooner, not evidence against the text. Counted as a defect
    # it pushed every servable-but-unproven candidate from HOLD to REJECT, and
    # purge deletes REJECTs: one --apply would have destroyed the best evolved
    # candidate we had, whose only fault was 1 benchmark run instead of 3.
    defects = list(reasons)

    if servable and reasons:
        reasons.append("SERVABLE despite the above — Thompson sampling can inject it today")

    if defects:
        return ("REJECT" if len(defects) > 1 or has_truncated_tail(cand.text) else "HOLD"), reasons
    return "PROMOTE", ["clean diff against the shipped baseline with benchmark evidence"]


def _fmt_flags(cand: Candidate) -> str:
    servable = "servable" if (not cand.requires_benchmark or cand.benchmark_passed) else "inert"
    return (
        f"gen={cand.generation} samples={cand.sample_count} active={cand.is_active} "
        f"bench={cand.benchmark_score:.2f}/{cand.benchmark_runs} {servable}"
    )


def cmd_audit(args) -> int:
    baselines = _baselines()
    candidates = _load(args.db)
    if not candidates:
        print(f"No candidates in {args.db}.")
        return 0
    print(f"{len(candidates)} candidate(s) in {args.db}\n")
    for cand in candidates:
        verdict, reasons = _verdict(cand, baselines)
        base = baselines.get(cand.section_name, "")
        delta = len(cand.text) - len(base) if base else 0
        print(f"[{verdict}] {cand.section_name} / {cand.provider} / {cand.text_hash}")
        print(
            f"        {_fmt_flags(cand)} chars={len(cand.text)} ({delta:+d}) "
            f"strategy={cand.strategy_chain or 'n/a'} created={cand.created_at}"
        )
        for reason in reasons:
            print(f"        - {reason}")
        print()
    return 0


def _find(candidates: List[Candidate], text_hash: str) -> Optional[Candidate]:
    matches = [c for c in candidates if c.text_hash.startswith(text_hash)]
    if len(matches) != 1:
        return None
    return matches[0]


def cmd_show(args) -> int:
    candidates = _load(args.db)
    cand = _find(candidates, args.hash)
    if cand is None:
        print(f"No unique candidate matching {args.hash!r}.", file=sys.stderr)
        return 1
    baseline = _baselines().get(cand.section_name, "")
    verdict, reasons = _verdict(cand, _baselines())
    print(f"[{verdict}] {cand.section_name} / {cand.provider} / {cand.text_hash}")
    for reason in reasons:
        print(f"  - {reason}")
    print(f"\n--- diff vs shipped {cand.section_name} ---")
    for line in difflib.unified_diff(
        baseline.splitlines(), cand.text.splitlines(), "shipped", "candidate", lineterm=""
    ):
        print(line)
    return 0


def cmd_export(args) -> int:
    """Emit a paste-ready assignment for prompt_section_texts.py."""
    candidates = _load(args.db)
    cand = _find(candidates, args.hash)
    if cand is None:
        print(f"No unique candidate matching {args.hash!r}.", file=sys.stderr)
        return 1
    verdict, _ = _verdict(cand, _baselines())
    if verdict != "PROMOTE" and not args.force:
        print(
            f"Refusing to export a {verdict} candidate. Re-run with --force if you "
            "have reviewed it by hand.",
            file=sys.stderr,
        )
        return 1
    print(
        f"# promoted from victor.db candidate {cand.text_hash} "
        f"({cand.provider}, gen {cand.generation}, {cand.created_at})"
    )
    print(f'{cand.section_name} = """')
    print(cand.text)
    print('""".strip()')
    return 0


SECTION_TEXTS_PATH = REPO_ROOT / "victor" / "agent" / "prompt_section_texts.py"


def cmd_promote(args) -> int:
    """Write a candidate's text into the shipped section module."""
    candidates = _load(args.db)
    cand = _find(candidates, args.hash)
    if cand is None:
        print(f"No unique candidate matching {args.hash!r}.", file=sys.stderr)
        return 1

    verdict, reasons = _verdict(cand, _baselines())
    if verdict != "PROMOTE" and not args.force:
        print(f"Refusing to promote a {verdict} candidate:", file=sys.stderr)
        for reason in reasons:
            print(f"  - {reason}", file=sys.stderr)
        print(
            "\nA candidate that has not beaten the shipped prompt on a paired benchmark "
            "run is a proposal, not an improvement. Re-run with --force only if you have "
            "reviewed the diff by hand.",
            file=sys.stderr,
        )
        return 1

    source = SECTION_TEXTS_PATH.read_text()

    provenance = (
        f"# evolved candidate {cand.text_hash} ({cand.provider}, gen {cand.generation}, "
        f"{cand.created_at}); benchmark {cand.benchmark_score:.2f} over "
        f"{cand.benchmark_runs} run(s), {cand.sample_count} live samples"
    )
    updated, error = build_promoted_source(source, cand.section_name, cand.text, provenance)
    if error is not None:
        print(f"Cannot promote {cand.text_hash}: {error}.", file=sys.stderr)
        return 1

    diff = "\n".join(
        difflib.unified_diff(
            source.splitlines(),
            updated.splitlines(),
            fromfile=f"a/{SECTION_TEXTS_PATH.name}",
            tofile=f"b/{SECTION_TEXTS_PATH.name}",
            lineterm="",
        )
    )
    print(diff)

    if not args.apply:
        print(
            f"\nDry run: {cand.section_name} would be replaced. Re-run with --apply to write, "
            "then review, test, and open the PR yourself.",
            file=sys.stderr,
        )
        return 0

    SECTION_TEXTS_PATH.write_text(updated)
    print(f"\nWrote {cand.section_name} to {SECTION_TEXTS_PATH}.", file=sys.stderr)
    print(
        "Verify before committing: python -c 'import victor.agent.prompt_section_texts' "
        "and run the prompt tests.",
        file=sys.stderr,
    )
    return 0


def cmd_propose(args) -> int:
    """Register a hand-written candidate so it can be measured, not just trusted.

    Evolution is not the only source of a good prompt. A human reading failure
    traces often sees the fix first — the largest failure class in one mbpp run
    was the agent renaming functions away from the identifiers the tests call,
    which reflection did not surface at all.

    Without this, a human-proposed improvement can only be applied on faith:
    edit the section, ship it, hope. Registering it as a candidate puts it
    through the identical paired benchmark and McNemar gate an evolved candidate
    faces, so "I think this helps" becomes "+6/24 against the seed, p=0.03".
    """
    section = args.section
    baselines = _baselines()
    if section not in baselines:
        print(
            f"Unknown section {section!r}. Known: {', '.join(sorted(baselines))}",
            file=sys.stderr,
        )
        return 1

    text = Path(args.file).read_text().rstrip("\n") if args.file else sys.stdin.read().rstrip("\n")
    if not text.strip():
        print("Refusing to register an empty candidate.", file=sys.stderr)
        return 1

    seed = baselines[section]
    if text == seed:
        print("Candidate is byte-identical to the shipped section; nothing to measure.")
        return 1

    text_hash = _md5(text)
    parent_hash = _md5(seed)

    # Same corruption checks the runtime persist gate applies. A hand-written
    # candidate is not exempt: a truncated tail or a restated line is the same
    # defect whoever wrote it.
    find_redundant_additions, has_truncated_tail = _hygiene()
    problems = []
    if has_truncated_tail(text):
        problems.append("ends mid-sentence — the final instruction is a fragment")
    for line in find_redundant_additions(seed, text):
        problems.append(f"restates existing guidance: {line!r}")
    if problems and not args.force:
        print("Refusing to register:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        print("Re-run with --force to register anyway.", file=sys.stderr)
        return 1

    con = sqlite3.connect(args.db)
    try:
        with con:
            con.execute(
                f"INSERT OR REPLACE INTO {TABLE} "
                "(section_name, provider, text_hash, text, generation, parent_hash, "
                " char_length, strategy_name, strategy_chain, requires_benchmark, is_active) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 'human', 'human', 1, 0)",
                (section, args.provider, text_hash, text, 0, parent_hash, len(text)),
            )
    finally:
        con.close()

    delta = len(text) - len(seed)
    print(
        f"Registered {text_hash} for {section} / {args.provider} ({len(text)} chars, {delta:+d})."
    )
    print(
        "\nIt is inert until measured — requires_benchmark=1 and is_active=0, so nothing\n"
        "serves it. Measure it against the shipped prompt:\n\n"
        f"  victor benchmark run-prompt-suite mbpp --prompt-section {section} \\\n"
        f"    --candidate-hash {text_hash} --include-baseline -n 24 \\\n"
        "    --record-benchmark-results --profile kimi\n"
    )
    return 0


def cmd_purge(args) -> int:
    baselines = _baselines()
    candidates = _load(args.db)
    doomed = [c for c in candidates if _verdict(c, baselines)[0] in {"REJECT", "ORPHAN"}]
    if not doomed:
        print("Nothing to purge.")
        return 0

    for cand in doomed:
        verdict, reasons = _verdict(cand, baselines)
        print(f"[{verdict}] {cand.section_name} / {cand.provider} / {cand.text_hash}")
        for reason in reasons:
            print(f"        - {reason}")

    if not args.apply:
        print(f"\nDry run: {len(doomed)} candidate(s) would be purged. Re-run with --apply.")
        return 0

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup = f"{TABLE}_backup_{stamp}"
    con = sqlite3.connect(args.db)
    try:
        with con:
            con.execute(f"CREATE TABLE {backup} AS SELECT * FROM {TABLE}")
            con.executemany(
                f"DELETE FROM {TABLE} WHERE text_hash = ? AND section_name = ? AND provider = ?",
                [(c.text_hash, c.section_name, c.provider) for c in doomed],
            )
        remaining = con.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()[0]
    finally:
        con.close()
    print(f"\nPurged {len(doomed)} candidate(s). Backup table: {backup}. Remaining: {remaining}.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help=f"default: {DEFAULT_DB}")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("audit", help="classify every candidate against the shipped baseline")

    show = sub.add_parser("show", help="full diff for one candidate")
    show.add_argument("hash")

    export = sub.add_parser("export", help="paste-ready literal for prompt_section_texts.py")
    export.add_argument("hash")
    export.add_argument("--force", action="store_true", help="export a non-PROMOTE candidate")

    propose = sub.add_parser(
        "propose", help="register a hand-written candidate so it can be benchmarked"
    )
    propose.add_argument("section", help="section name, e.g. GROUNDING_RULES")
    propose.add_argument("--file", help="file holding the candidate text (default: stdin)")
    propose.add_argument("--provider", default="default", help="provider scope for the candidate")
    propose.add_argument("--force", action="store_true", help="register despite hygiene problems")

    promote = sub.add_parser(
        "promote", help="write a candidate into prompt_section_texts.py (dry run by default)"
    )
    promote.add_argument("hash", help="candidate text_hash (prefix is fine)")
    promote.add_argument("--apply", action="store_true", help="write the file instead of diffing")
    promote.add_argument(
        "--force", action="store_true", help="promote a candidate the audit did not clear"
    )

    purge = sub.add_parser("purge", help="drop REJECT/ORPHAN candidates (backs up first)")
    purge.add_argument("--apply", action="store_true", help="actually delete; default is a dry run")

    args = parser.parse_args()
    return {
        "audit": cmd_audit,
        "show": cmd_show,
        "export": cmd_export,
        "propose": cmd_propose,
        "promote": cmd_promote,
        "purge": cmd_purge,
    }[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
