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

"""Human-label overlay for judge-calibration runs (EVR-2, ADR-011).

Recomputes agreement from a *saved* calibration run plus a human-labels file —
no executor, no LLM calls. Three numbers matter, in this order:

1. **human↔verifier κ** — validates the programmatic gold every prior run
   relied on. Below its threshold the run is a STOP-THE-LINE failure: the
   verifier-gold conclusions (FINDINGS runs 1-11) are void until the corpus
   verifiers are fixed.
2. **human↔judge α** — the ADR-011 letter, overall and per family.
3. **human↔secondary-annotator κ** — annotation quality control when a second
   (LLM) annotator's labels file is supplied.

Pre-registered thresholds live in
``docs/architecture/evr2-human-validation-protocol.md`` and are defaulted
here; changing them after labels exist defeats pre-registration — don't.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

from victor.evaluation.judge_calibration import (
    JudgeReliability,
    evaluate_judge_agreement,
)
from victor.evaluation.judge_calibration_harness import binary_categorize

# Pre-registered thresholds (protocol doc §Thresholds). See module docstring.
HUMAN_JUDGE_ALPHA_THRESHOLD = 0.7
HUMAN_VERIFIER_KAPPA_THRESHOLD = 0.8
MIN_FAMILY_N = 16  # per-family α is claimed only at n ≥ 16 (FINDINGS lesson 2)


class LabelsError(ValueError):
    """Raised for malformed or incomplete human-label files."""


def load_labels(path: Path) -> dict[str, float]:
    """Load ``{"task_id", "label"}`` JSONL; reject null/invalid labels loudly.

    An unlabeled template line (``label: null``) is an error, not a skip — a
    silently partial label set would quietly shrink n and flatter per-family α.
    """
    labels: dict[str, float] = {}
    unlabeled: list[str] = []
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise LabelsError(f"{path}:{lineno}: not valid JSON: {exc}") from exc
        task_id = row.get("task_id")
        if not task_id:
            raise LabelsError(f"{path}:{lineno}: missing task_id")
        if task_id in labels:
            raise LabelsError(f"{path}:{lineno}: duplicate task_id {task_id!r}")
        label = row.get("label")
        if label is None:
            unlabeled.append(task_id)
            continue
        if label not in (0, 1, 0.0, 1.0):
            raise LabelsError(f"{path}:{lineno}: label must be 0 or 1, got {label!r}")
        labels[task_id] = float(label)
    if unlabeled:
        raise LabelsError(
            f"{path}: {len(unlabeled)} task(s) left unlabeled (label=null): "
            f"{', '.join(unlabeled[:5])}{'…' if len(unlabeled) > 5 else ''}"
        )
    if not labels:
        raise LabelsError(f"{path}: no labels found")
    return labels


def load_report_samples(path: Path) -> list[dict[str, Any]]:
    """The ``samples`` rows of a saved :class:`CalibrationReport` JSON."""
    data = json.loads(path.read_text())
    samples = data.get("samples")
    if not isinstance(samples, list) or not samples:
        raise LabelsError(f"{path}: no samples in report")
    return samples


@dataclass(frozen=True)
class OverlayVerdict:
    """One thresholded check of the overlay."""

    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class HumanOverlayReport:
    """Agreement of human gold against verifier gold and every judge."""

    n: int
    human_vs_verifier: JudgeReliability
    human_vs_judge: dict[str, JudgeReliability]
    human_vs_judge_per_family: dict[str, dict[str, JudgeReliability]]
    human_vs_secondary: Optional[JudgeReliability]
    verdicts: tuple[OverlayVerdict, ...]
    thresholds: dict[str, float] = field(
        default_factory=lambda: {
            "human_judge_alpha": HUMAN_JUDGE_ALPHA_THRESHOLD,
            "human_verifier_kappa": HUMAN_VERIFIER_KAPPA_THRESHOLD,
            "min_family_n": MIN_FAMILY_N,
        }
    )

    @property
    def stop_the_line(self) -> bool:
        """True when verifier gold failed human validation — runs 1-11 are void."""
        return any(v.name == "human_vs_verifier_kappa" and not v.passed for v in self.verdicts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "thresholds": self.thresholds,
            "stop_the_line": self.stop_the_line,
            "human_vs_verifier": self.human_vs_verifier.to_dict(),
            "human_vs_judge": {name: rel.to_dict() for name, rel in self.human_vs_judge.items()},
            "human_vs_judge_per_family": {
                name: {family: rel.to_dict() for family, rel in families.items()}
                for name, families in self.human_vs_judge_per_family.items()
            },
            "human_vs_secondary": (
                self.human_vs_secondary.to_dict() if self.human_vs_secondary else None
            ),
            "verdicts": [
                {"name": v.name, "passed": v.passed, "detail": v.detail} for v in self.verdicts
            ],
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")


def _agreement(
    gold: list[float], other: list[float], *, level: str = "nominal"
) -> JudgeReliability:
    return evaluate_judge_agreement(gold, other, level=level, categorize=binary_categorize)


def _alpha_ok(rel: JudgeReliability, threshold: float) -> bool:
    alpha = rel.krippendorff_alpha
    return alpha is not None and alpha == alpha and alpha >= threshold


def _fmt(value: Optional[float]) -> str:
    if value is None or value != value:
        return "n/a"
    return f"{value:.4f}"


def compute_overlay(
    judge_reports: Mapping[str, list[dict[str, Any]]],
    human_labels: Mapping[str, float],
    *,
    secondary_labels: Optional[Mapping[str, float]] = None,
    alpha_threshold: float = HUMAN_JUDGE_ALPHA_THRESHOLD,
    verifier_kappa_threshold: float = HUMAN_VERIFIER_KAPPA_THRESHOLD,
    min_family_n: int = MIN_FAMILY_N,
) -> HumanOverlayReport:
    """Join saved run samples with human labels and compute every pairwise agreement.

    ``judge_reports`` maps judge name → sample rows (``load_report_samples``).
    All reports from one run share (task_id, gold), so verifier agreement is
    computed once from the first report. Tasks without a human label are an
    error upstream (``load_labels``); human labels for unknown task_ids are
    ignored with the mismatch surfaced in the verdict detail.
    """
    if not judge_reports:
        raise LabelsError("no judge reports supplied")

    first = next(iter(judge_reports.values()))
    run_rows = {row["task_id"]: row for row in first}
    matched_ids = [tid for tid in run_rows if tid in human_labels]
    if not matched_ids:
        raise LabelsError("human labels share no task_ids with the run reports")
    unmatched_labels = sorted(set(human_labels) - set(run_rows))

    human = [human_labels[tid] for tid in matched_ids]
    verifier = [float(run_rows[tid]["gold"]) for tid in matched_ids]
    human_vs_verifier = _agreement(human, verifier)

    verdicts: list[OverlayVerdict] = []
    kappa = human_vs_verifier.cohens_kappa
    kappa_ok = kappa is not None and kappa == kappa and kappa >= verifier_kappa_threshold
    verdicts.append(
        OverlayVerdict(
            name="human_vs_verifier_kappa",
            passed=kappa_ok,
            detail=(
                f"κ={_fmt(kappa)} vs ≥ {verifier_kappa_threshold}"
                f" on n={len(matched_ids)}"
                + (
                    f"; {len(unmatched_labels)} label(s) had no matching run row"
                    if unmatched_labels
                    else ""
                )
                + ("" if kappa_ok else " — STOP THE LINE: verifier gold failed human validation")
            ),
        )
    )

    human_vs_judge: dict[str, JudgeReliability] = {}
    per_family_all: dict[str, dict[str, JudgeReliability]] = {}
    for name, samples in judge_reports.items():
        rows = {row["task_id"]: row for row in samples}
        ids = [tid for tid in matched_ids if tid in rows]
        judged = [float(rows[tid]["judged"]) for tid in ids]
        gold_h = [human_labels[tid] for tid in ids]
        overall = _agreement(gold_h, judged)
        human_vs_judge[name] = overall

        families = sorted({rows[tid]["family"] for tid in ids})
        per_family: dict[str, JudgeReliability] = {}
        family_failures: list[str] = []
        thin_families: list[str] = []
        for family in families:
            fam_ids = [tid for tid in ids if rows[tid]["family"] == family]
            rel = _agreement(
                [human_labels[tid] for tid in fam_ids],
                [float(rows[tid]["judged"]) for tid in fam_ids],
            )
            per_family[family] = rel
            if rel.n < min_family_n:
                thin_families.append(f"{family}(n={rel.n})")
            elif not _alpha_ok(rel, alpha_threshold):
                family_failures.append(f"{family}(α={_fmt(rel.krippendorff_alpha)})")
        per_family_all[name] = per_family

        overall_ok = _alpha_ok(overall, alpha_threshold)
        passed = overall_ok and not family_failures
        detail = (
            f"overall α={_fmt(overall.krippendorff_alpha)} vs ≥ {alpha_threshold} on n={overall.n}"
        )
        if family_failures:
            detail += f"; failing families: {', '.join(family_failures)}"
        if thin_families:
            detail += (
                f"; families below n={min_family_n} (directional only, not claimed): "
                f"{', '.join(thin_families)}"
            )
        verdicts.append(
            OverlayVerdict(name=f"human_vs_judge_alpha[{name}]", passed=passed, detail=detail)
        )

    human_vs_secondary: Optional[JudgeReliability] = None
    if secondary_labels:
        both = [tid for tid in matched_ids if tid in secondary_labels]
        if both:
            human_vs_secondary = _agreement(
                [human_labels[tid] for tid in both],
                [secondary_labels[tid] for tid in both],
            )
            verdicts.append(
                OverlayVerdict(
                    name="human_vs_secondary_kappa",
                    passed=True,  # QC signal, not a gate — reported, never gating
                    detail=(
                        f"κ={_fmt(human_vs_secondary.cohens_kappa)} on n={human_vs_secondary.n} "
                        "(annotation quality control; audit disagreements per protocol)"
                    ),
                )
            )

    return HumanOverlayReport(
        n=len(matched_ids),
        human_vs_verifier=human_vs_verifier,
        human_vs_judge=human_vs_judge,
        human_vs_judge_per_family=per_family_all,
        human_vs_secondary=human_vs_secondary,
        verdicts=tuple(verdicts),
    )
