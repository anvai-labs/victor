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

"""LLM-judge reliability gating (EVR-2, ADR-011, FEP-0008 Phase D).

No LLM-as-judge (rubric completion scoring, trajectory grading) is *trusted in production* until it
is measured against a human-labeled set: substring/keyword judging agrees with humans at chance
(κ≈0.05; AgentProp-Bench 2604.16706). This module provides the measurement and the gate:

- inter-rater agreement metrics — **Cohen's κ** (categorical) and **Krippendorff's α** (interval or
  nominal, multi-rater, missing-data tolerant) — implemented in pure Python (no scipy/sklearn);
- :func:`evaluate_judge_agreement` to compare a judge's labels against the gold set;
- :class:`JudgeReliabilityGate` — enable a judge only at ``α ≥ threshold`` (default 0.7), else fall
  back to algorithmic scoring;
- :class:`OrderSwapEnsembleJudge` — run a list-scoring judge under multiple orderings and average per
  candidate, to defuse position/verbosity bias (a documented, training-free κ improvement).

Rejection (catch bad input) and recovery (fix after accepting) are *statistically independent*
capabilities (AgentProp-Bench) and are tracked separately at the trajectory layer (see
``trajectory_eval`` Recovery dimension) — not folded into one score here.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Hashable, Optional, Sequence, TypeVar

T = TypeVar("T")

# ----------------------------------------------------------------------------------------------------
# Agreement metrics (pure Python)
# ----------------------------------------------------------------------------------------------------


def cohens_kappa(rater_a: Sequence[Hashable], rater_b: Sequence[Hashable]) -> float:
    """Cohen's κ — chance-corrected agreement between two raters over categorical labels.

    ``κ = (p_o - p_e) / (1 - p_e)``. Returns 1.0 when both raters are constant and agree, 0.0 when
    expected agreement is already perfect but they disagree. Raises on length mismatch / empty input.
    """
    if len(rater_a) != len(rater_b):
        raise ValueError("raters must have equal length")
    n = len(rater_a)
    if n == 0:
        raise ValueError("need at least one item")
    agree = sum(1 for a, b in zip(rater_a, rater_b) if a == b)
    p_o = agree / n
    count_a = Counter(rater_a)
    count_b = Counter(rater_b)
    p_e = sum((count_a[c] / n) * (count_b[c] / n) for c in set(count_a) | set(count_b))
    if p_e >= 1.0:
        return 1.0 if p_o >= 1.0 else 0.0
    return (p_o - p_e) / (1.0 - p_e)


def krippendorff_alpha(
    units: Sequence[Sequence[Optional[float]]], level: str = "interval"
) -> float:
    """Krippendorff's α over ``units`` (one list of rater values per item; ``None`` = missing).

    Coincidence-matrix formulation: ``α = 1 - D_o / D_e`` with difference function δ² (nominal: 0/1;
    interval: squared difference). Multi-rater and missing-data tolerant. Returns ``nan`` when there
    are no pairable values, and 1.0 when there is no observed disagreement.
    """
    o: dict[tuple[Any, Any], float] = defaultdict(float)
    n = 0.0
    for vals in units:
        present = [v for v in vals if v is not None]
        m = len(present)
        if m < 2:
            continue
        n += m
        weight = 1.0 / (m - 1)
        for i in range(m):
            for j in range(m):
                if i != j:
                    o[(present[i], present[j])] += weight
    if n == 0:
        return float("nan")

    values = sorted({c for c, _ in o} | {k for _, k in o})
    n_c = {c: sum(o.get((c, k), 0.0) for k in values) for c in values}

    def delta2(c: Any, k: Any) -> float:
        if level == "nominal":
            return 0.0 if c == k else 1.0
        return (float(c) - float(k)) ** 2

    d_o = 0.0
    d_e = 0.0
    for idx, c in enumerate(values):
        for k in values[idx + 1 :]:  # unordered pairs c < k
            d = delta2(c, k)
            d_o += o.get((c, k), 0.0) * d
            d_e += n_c[c] * n_c[k] * d
    d_o /= n
    if n > 1:
        d_e /= n * (n - 1)
    if d_e == 0.0:
        return 1.0
    return 1.0 - d_o / d_e


# ----------------------------------------------------------------------------------------------------
# Reliability report + gate
# ----------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class JudgeReliability:
    """Measured agreement of a judge against a human-labeled set."""

    n: int
    krippendorff_alpha: Optional[float]
    cohens_kappa: Optional[float] = None
    level: str = "interval"

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "krippendorff_alpha": _round(self.krippendorff_alpha),
            "cohens_kappa": _round(self.cohens_kappa),
            "level": self.level,
        }


def _round(value: Optional[float]) -> Optional[float]:
    return None if value is None else round(value, 4)


def evaluate_judge_agreement(
    gold: Sequence[float],
    judged: Sequence[float],
    *,
    level: str = "interval",
    categorize: Optional[Callable[[float], Hashable]] = None,
) -> JudgeReliability:
    """Compare a judge's scores against the gold human labels.

    Computes Krippendorff's α over the (gold, judged) pairs. When ``categorize`` is supplied (e.g. a
    5-bucket mapping), also computes Cohen's κ on the categorized labels.
    """
    if len(gold) != len(judged):
        raise ValueError("gold and judged must have equal length")
    units = [[g, j] for g, j in zip(gold, judged)]
    alpha = krippendorff_alpha(units, level=level)
    kappa = None
    if categorize is not None:
        kappa = cohens_kappa([categorize(g) for g in gold], [categorize(j) for j in judged])
    return JudgeReliability(n=len(gold), krippendorff_alpha=alpha, cohens_kappa=kappa, level=level)


@dataclass(frozen=True)
class GateDecision:
    """Outcome of the reliability gate."""

    trusted: bool
    reason: str


@dataclass(frozen=True)
class JudgeReliabilityGate:
    """Trust an LLM judge only when its measured α clears the threshold.

    A judge that fails the gate must fall back to algorithmic scoring — this is the precondition that
    makes rubric/effect LLM judging (ADR-009/010) safe to default on.
    """

    alpha_threshold: float = 0.7

    def decide(self, reliability: JudgeReliability) -> GateDecision:
        alpha = reliability.krippendorff_alpha
        if alpha is None or alpha != alpha:  # None or NaN
            return GateDecision(False, "no measurable agreement (insufficient labeled data)")
        if alpha >= self.alpha_threshold:
            return GateDecision(True, f"α={alpha:.3f} ≥ {self.alpha_threshold:.2f}")
        return GateDecision(
            False,
            f"α={alpha:.3f} < {self.alpha_threshold:.2f} — falling back to algorithmic scoring",
        )

    def is_trusted(self, reliability: JudgeReliability) -> bool:
        return self.decide(reliability).trusted


# ----------------------------------------------------------------------------------------------------
# Order-swap ensemble (position-bias mitigation)
# ----------------------------------------------------------------------------------------------------


def _orderings(n: int, swaps: int) -> list[list[int]]:
    """Deterministic index orderings: identity, reverse, then rotations — deduplicated."""
    if n <= 1:
        return [list(range(n))]
    seen: list[list[int]] = []
    candidates = [list(range(n)), list(range(n - 1, -1, -1))]
    for shift in range(1, n):
        candidates.append([(i + shift) % n for i in range(n)])
    for order in candidates:
        if order not in seen:
            seen.append(order)
        if len(seen) >= max(2, swaps):
            break
    return seen


class OrderSwapEnsembleJudge:
    """Wrap a list-scoring judge and average its scores across multiple input orderings.

    ``judge(candidates) -> scores`` must return one score per candidate, aligned to the input order.
    Running it under several orderings and mapping back to original positions cancels position /
    verbosity bias (AgentProp-Bench: a 3-LLM order-swapped ensemble lifts κ from chance to moderate).
    Deterministic (fixed orderings) for reproducibility.
    """

    def __init__(self, judge: Callable[[Sequence[T]], Sequence[float]], *, swaps: int = 2) -> None:
        self._judge = judge
        self._swaps = max(2, swaps)

    def score(self, candidates: Sequence[T]) -> list[float]:
        n = len(candidates)
        if n == 0:
            return []
        acc = [0.0] * n
        cnt = [0] * n
        for order in _orderings(n, self._swaps):
            permuted = [candidates[i] for i in order]
            scores = list(self._judge(permuted))
            if len(scores) != n:
                raise ValueError("judge must return one score per candidate")
            for pos, original_index in enumerate(order):
                acc[original_index] += scores[pos]
                cnt[original_index] += 1
        return [acc[i] / cnt[i] if cnt[i] else 0.0 for i in range(n)]
