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

"""FOU-S3d injection-screen false-positive battery (ADR-037 decision 5; ADR-036 evidence).

Runs the S3c injection screen (``victor/security/injection_screen.py``) over
two deterministic corpora — a REALISTIC benign corpus (source code, tool
documentation quoting ``<tool_call>`` syntax, logs/config dumps, shell
transcripts, review comments discussing prompt injection itself) and an
adversarial injection corpus (S3c base + obfuscated variants) — and reports:

- **Per-family TP/FP/FN**: for each injection family, detection of the
  family's ground-truth samples (TP), benign samples wrongly attributed to
  the family (FP), and family samples that escaped detection (FN).
- **Overall annotation-level FP rate**: fraction of benign samples the
  screen flags at all (the number that gates hard-fail graduation).
- **Per-pattern attribution**: for every FP/FN, which regex fired (family +
  pattern index + span + excerpt) or which patterns were tried and missed.

Verdict semantics per ADR-036/ADR-037 decision 5 (the battery REPORTS; it
does not flip any production flag — hard-fail wiring stays absent):

- ``GRADUATE_HARD_FAIL_CANDIDATE`` — FP rate == 0 on the benign corpus AND
  FN rate == 0 on the injection corpus.
- ``CONDITIONAL`` — FP == 0, some FN: patterns may be tightened/added in a
  follow-up; hard-fail stays gated.
- ``HOLD`` — any FP: hard-fail stays gated; the offending patterns are
  named in ``offending_patterns``.

**Why not ``flag_ab.py``?** Same rationale as ``memory_ab.py``: that harness
drives the real agent against a live provider (minutes per task,
nondeterministic), while this gate is about the SCREEN's classification
behavior — pure, deterministic, offline. This module mirrors memory_ab's
reporting conventions instead (JSON snapshots, a verdict dict, a small CLI).

    python -m victor.evaluation.injection_fp_battery --out-dir output/injection_fp_battery
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from victor.evaluation.injection_fp_corpus import (
    InjectionFPSample,
    benign_corpus,
    injection_corpus,
)
from victor.security.injection_screen import FAMILIES, iter_pattern_matches

logger = logging.getLogger(__name__)

BENIGN_SNAPSHOT = "injection_fp_benign.json"
INJECTION_SNAPSHOT = "injection_fp_injection.json"

#: Tightenings applied in the FOU-S3d pass, recorded in every battery report
#: (each keeps its benign counterpart pinned in the corpus/tests).
TIGHTENINGS: Tuple[Dict[str, str], ...] = (
    {
        "family": "instruction_impersonation",
        "change": "quoted-mention guard + imperative-continuation lookahead",
        "benign_counterpart": (
            "py-020/docs-004/review-001 quoted payloads; py-007/ts-016/rust-003 "
            "dev notes ('ignore previous instructions in migrate_v1.py')"
        ),
    },
    {
        "family": "instruction_impersonation",
        "change": "'(new|updated|revised) instructions:' requires a model-directed verb",
        "benign_counterpart": "py-018/ts-009/docs-006 human-directed task notes",
    },
    {
        "family": "instruction_impersonation",
        "change": "'you are now a X' requires a constraint-removal cue nearby",
        "benign_counterpart": "ts-013/rust-006/policy-013 UI copy and fixture comments",
    },
    {
        "family": "instruction_impersonation",
        "change": "line-start override headers require an imperative continuation",
        "benign_counterpart": "docs-005/docs-014/log-005/policy-005 policy prose and syslog",
    },
    {
        "family": "tool_call_forgery",
        "change": "'<tool_call>{' requires a suspicious body (command/secret/URL tokens)",
        "benign_counterpart": "docs-001/rust-004 documentation quoting read-only examples",
    },
    {
        "family": "tool_call_forgery",
        "change": "'<function_results>' requires letter-start content",
        "benign_counterpart": "docs-001 prose quoting the bare tag in backticks",
    },
    {
        "family": "tool_call_forgery",
        "change": "routing tokens require the dotted method form",
        "benign_counterpart": "log-009 gateway prose; docs-003/review-012 quoted tokens",
    },
    {
        "family": "tool_call_forgery",
        "change": "'executed/ran the command' requires a HIGH-RISK payload "
        "(pipe-to-shell, metadata IP, credential files, root deletion)",
        "benign_counterpart": "log-003/log-008/ts-007 build and deploy logs",
    },
    {
        "family": "exfil_solicitation",
        "change": "'repeat everything above' line-anchored with direction-suffix lookahead",
        "benign_counterpart": "py-009/ts-005 echo-service comments; docs-013 UI prose",
    },
    {
        "family": "exfil_solicitation",
        "change": "added (tell|give|show|share|print|output|reveal) + 'the/all' form",
        "benign_counterpart": "no FP on corpus; closes inj-028/inj-032/inj-033 FNs",
    },
)


def screen_sample(sample: InjectionFPSample) -> Dict[str, Any]:
    """Screen one corpus sample and score it against its ground truth.

    Returns a per-sample report with per-pattern attribution: every hit
    carries (family, pattern_index, span, excerpt); a missed injection
    sample reports the patterns that were tried and found nothing.
    """
    hits = iter_pattern_matches(sample.text)
    hit_reports = [
        {
            "family": family,
            "pattern_index": index,
            "start": start,
            "end": end,
            "excerpt": sample.text[start:end][:80],
        }
        for family, index, start, end in hits
    ]
    is_fp = sample.expected_family is None and bool(hits)
    matched_expected = sample.expected_family is not None and any(
        family == sample.expected_family for family, _index, _start, _end in hits
    )
    is_fn = sample.expected_family is not None and not matched_expected
    return {
        "sample_id": sample.sample_id,
        "category": sample.category,
        "expected_family": sample.expected_family,
        "note": sample.note,
        "flagged": bool(hits),
        "families": sorted({family for family, _i, _s, _e in hits}),
        "hits": hit_reports,
        "is_false_positive": is_fp,
        "is_false_negative": is_fn,
        # FN attribution: which patterns of the expected family were tried.
        "patterns_tried": (_pattern_count(sample.expected_family) if is_fn else 0),
    }


def _pattern_count(family: Optional[str]) -> int:
    for entry in FAMILIES:
        if entry.name == family:
            return len(entry.patterns)
    return 0


@dataclass
class InjectionFPArmResult:
    """Aggregated screen outcome over one corpus half (mirrors memory_ab arms)."""

    side: str  # "benign" or "injection"
    samples: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def false_positives(self) -> int:
        return sum(1 for s in self.samples if s["is_false_positive"])

    @property
    def false_negatives(self) -> int:
        return sum(1 for s in self.samples if s["is_false_negative"])

    @property
    def true_positives(self) -> int:
        return sum(
            1
            for s in self.samples
            if s["expected_family"] is not None and not s["is_false_negative"]
        )

    def per_family(self) -> Dict[str, Dict[str, int]]:
        """Per-family TP/FP/FN table."""
        table: Dict[str, Dict[str, int]] = {}
        for s in self.samples:
            family = s["expected_family"] or "benign"
            row = table.setdefault(family, {"tp": 0, "fp": 0, "fn": 0, "total": 0})
            row["total"] += 1
            if s["is_false_positive"]:
                row["fp"] += 1
            elif s["is_false_negative"]:
                row["fn"] += 1
            elif s["expected_family"] is not None:
                row["tp"] += 1
        return table

    @property
    def fp_rate(self) -> float:
        benign = [s for s in self.samples if s["expected_family"] is None]
        if not benign:
            return 0.0
        return self.false_positives / len(benign)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "side": self.side,
            "sample_count": len(self.samples),
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "annotation_fp_rate": round(self.fp_rate, 6),
            "per_family": self.per_family(),
            "samples": self.samples,
        }

    @classmethod
    def from_samples(cls, side: str, reports: Sequence[Dict[str, Any]]) -> "InjectionFPArmResult":
        return cls(side=side, samples=list(reports))


@dataclass(frozen=True)
class InjectionFPVerdict:
    """ADR-036 evidence verdict for hard-fail graduation of the screen.

    ``should_gate_hard_fail`` is the machine-readable flip of the rule for
    the ADR-036 queue and the FOU ledger: True only for
    GRADUATE_HARD_FAIL_CANDIDATE. The battery itself never flips anything.
    """

    verdict: str
    should_gate_hard_fail: bool
    reasons: List[str] = field(default_factory=list)
    offending_patterns: List[Tuple[str, int]] = field(default_factory=list)
    missed_samples: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict,
            "should_gate_hard_fail": self.should_gate_hard_fail,
            "reasons": list(self.reasons),
            "offending_patterns": [
                {"family": family, "pattern_index": index}
                for family, index in self.offending_patterns
            ],
            "missed_samples": list(self.missed_samples),
        }


def assess_injection_fp_battery(
    benign: InjectionFPArmResult,
    injection: InjectionFPArmResult,
) -> InjectionFPVerdict:
    """Verdict per ADR-036: evidence-only; hard-fail graduation gate."""
    reasons: List[str] = []
    fp = benign.false_positives
    fn = injection.false_negatives

    offending: List[Tuple[str, int]] = []
    for s in benign.samples:
        if s["is_false_positive"]:
            for hit in s["hits"]:
                offending.append((hit["family"], hit["pattern_index"]))
    missed = [s["sample_id"] for s in injection.samples if s["is_false_negative"]]

    if fp == 0:
        reasons.append(
            f"zero false positives across {len(benign.samples)} realistic benign samples "
            f"(annotation-level FP rate 0.0)"
        )
    else:
        reasons.append(
            f"{fp} false positive(s) on the benign corpus — hard-fail stays gated; "
            f"offending patterns named"
        )

    if fn == 0:
        reasons.append("zero false negatives across the adversarial injection corpus")
    elif fp == 0:
        reasons.append(
            f"{fn} documented false negative(s) on the injection corpus "
            f"({', '.join(missed)}) — patterns may be tightened in a follow-up; "
            f"hard-fail stays gated"
        )
    else:
        reasons.append(f"{fn} false negative(s) on the injection corpus")

    if fp == 0 and fn == 0:
        return InjectionFPVerdict("GRADUATE_HARD_FAIL_CANDIDATE", True, reasons)
    if fp == 0:
        return InjectionFPVerdict("CONDITIONAL", False, reasons, offending, missed)
    return InjectionFPVerdict("HOLD", False, reasons, offending, missed)


def run_injection_fp_battery(
    *,
    out_dir: str = ".",
    benign_provider: Optional[Callable[[], Sequence[InjectionFPSample]]] = None,
    injection_provider: Optional[Callable[[], Sequence[InjectionFPSample]]] = None,
) -> Dict[str, Any]:
    """Screen both corpus halves and emit snapshots + the ADR-036 verdict.

    Mirrors ``memory_ab.run_memory_ab``'s output contract: two JSON
    snapshots (``injection_fp_benign.json`` / ``injection_fp_injection.json``),
    both arms' dicts inline, and the verdict. Deterministic and offline —
    pure pattern scan, no provider, no agent loop, unit-testable runtime.

    Returns:
        Dict with snapshot paths, both arm dicts, the ADR-036 verdict, the
        recorded tightenings, and ``hard_fail_wired=False`` (the battery
        REPORTS; it never flips a production flag).
    """
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    benign = InjectionFPArmResult.from_samples(
        "benign",
        [screen_sample(s) for s in (benign_provider or benign_corpus)()],
    )
    injection = InjectionFPArmResult.from_samples(
        "injection",
        [screen_sample(s) for s in (injection_provider or injection_corpus)()],
    )

    benign_path = out / BENIGN_SNAPSHOT
    inj_path = out / INJECTION_SNAPSHOT
    benign_path.write_text(json.dumps(benign.to_dict(), indent=2) + "\n", encoding="utf-8")
    inj_path.write_text(json.dumps(injection.to_dict(), indent=2) + "\n", encoding="utf-8")

    verdict = assess_injection_fp_battery(benign, injection)

    return {
        "battery": "injection_fp",
        "benign_path": str(benign_path),
        "injection_path": str(inj_path),
        "benign_battery": benign.to_dict(),
        "injection_battery": injection.to_dict(),
        "verdict": verdict.to_dict(),
        "tightenings": [dict(t) for t in TIGHTENINGS],
        # Proof-of-posture: this battery gates nothing by itself — hard-fail
        # wiring stays absent until a clean verdict lands in a future PR.
        "hard_fail_wired": False,
    }


def _main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m victor.evaluation.injection_fp_battery",
        description="Deterministic false-positive battery for the FOU-S3c injection "
        "screen (benign FP rate + adversarial FN rate -> ADR-036 verdict).",
    )
    parser.add_argument(
        "--out-dir",
        default=".",
        help="Where to write the two snapshot JSONs",
    )
    args = parser.parse_args(argv)

    result = run_injection_fp_battery(out_dir=args.out_dir)
    print(json.dumps(result, indent=2))
    # Exit policy: clean annotation level (no FP) -> 0; any FP (HOLD) -> 1.
    # CONDITIONAL (FN-only) still exits 0: the battery ran clean at the
    # annotation level, misses are reported for follow-up.
    return 0 if result["benign_battery"]["false_positives"] == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(_main())
