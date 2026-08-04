#!/usr/bin/env python3
# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0
"""Re-gate LLM completion judges on a REAL SWE-bench stratum (FEP-0030 / EVR-3).

Runs an Ollama-hosted LLM completion judge over the blinded views of a
SWE-bench stratum (or a raw ``eval_manifest_*.jsonl``) and scores Krippendorff's
α / Cohen's κ vs the in-container FAIL_TO_PASS gold — the same gate metric as
``run_offline_calibration.py``, but on the real-distribution stratum produced by
``victor.evaluation.swe_bench_stratum``.

This is the runner behind FINDINGS.md's SWE-bench real-distribution re-gate: it
showed judges that pass on the (positive-heavy) calibration corpus over-credit
on the (negative-heavy) real distribution.

Example::

    python gate_swe_bench_stratum.py \\
        ~/.victor/evaluations/eval_manifest_<id>.jsonl \\
        --judge llama3.3:70b --endpoint http://192.168.1.20:11434 \\
        --out labels/swe-bench-lite-30/llama.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from victor.evaluation.calibration_rubric_judge import (
    CALIBRATION_TASK_FAMILY,
    AsyncRubricCompletionEvaluator,
    LLMRubricJudge,
    _project,
    make_provider_complete_fn,
)
from victor.evaluation.swe_bench_stratum import (
    StratumExample,
    gate_stratum,
    manifest_to_stratum,
)
from victor.providers.registry import ProviderRegistry


def _load_examples(path: Path) -> list[StratumExample]:
    """Accept either a stratum jsonl (text/label) or a raw eval manifest."""
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if rows and "text" in rows[0] and "label" in rows[0]:
        return [
            StratumExample(
                task_id=r.get("task_id", "?"),
                family=r.get("family", "swe-bench"),
                text=r["text"],
                label=int(r["label"]),
            )
            for r in rows
        ]
    return manifest_to_stratum(path)


async def _amain() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stratum", type=Path, help="stratum jsonl or eval_manifest jsonl")
    ap.add_argument("--judge", default="llama3.3:70b", help="Ollama model name")
    ap.add_argument("--endpoint", default="http://localhost:11434", help="Ollama base_url")
    ap.add_argument("--threshold", type=float, default=0.5, help="completion score → verdict cut")
    ap.add_argument("--out", type=Path, default=None, help="write result JSON here")
    args = ap.parse_args()

    examples = _load_examples(args.stratum)
    provider = ProviderRegistry.create("ollama", base_url=args.endpoint)
    complete_fn = make_provider_complete_fn(provider, args.judge, max_tokens=512)
    evaluator = AsyncRubricCompletionEvaluator(LLMRubricJudge(complete_fn))

    # The LLM call is async; gate_stratum wants a sync judge(text)->0/1. Score
    # every view first (caching the verdict by its exact view text), then hand
    # gate_stratum a pure lookup over that cache.
    by_text: dict[str, int] = {}
    print(f"judge={args.judge} endpoint={args.endpoint} n={len(examples)}")
    for ex in examples:
        result = await evaluator.aevaluate(task_family=CALIBRATION_TASK_FAMILY, content=ex.text)
        score = _project(result, "complete")
        verdict = 1 if score >= args.threshold else 0
        by_text[ex.text] = verdict
        print(f"  {ex.task_id:30} gold={ex.label} judge={verdict} (score={score:.2f})")

    res = gate_stratum(examples, lambda text: by_text[text], judge_name=args.judge)
    print(
        f"\nRESULT judge={args.judge}: n={res.n} pos={res.n_pos} neg={res.n_neg} "
        f"agree={res.agree}/{res.n} TP={res.true_pos} FP={res.false_pos} "
        f"TN={res.true_neg} FN={res.false_neg} alpha={res.krippendorff_alpha} "
        f"kappa={res.cohens_kappa}"
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(res.to_dict(), indent=2) + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    asyncio.run(_amain())
