"""Caller-declared completion criteria gate (FEP-0043).

A bounded post-filter on the EVALUATE seam: when the caller supplies
``criteria`` in the run context, a COMPLETE decision is downgraded to
RETRY while any declared minimum is unmet — for at most
``max_downgrades`` rounds, after which the COMPLETE stands annotated
with the unmet minimums (best-effort retention, mirroring EffectGate's
``max_downgrades`` convention). The loop's budget-exhaustion marker
(``state["_criteria_budget_exhausted"]``) also lets COMPLETE stand.

Victor stays generic: criteria are opaque caller key/values evaluated
against generic state-key entry counts (the same ``len(...)`` semantics
:class:`AnalysisFulfillment` uses for ``min_findings``). The framework
never learns what a criterion means.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from victor.framework.evaluation_nodes import EvaluationDecision, EvaluationResult


@dataclass
class CriteriaGateConfig:
    """Configuration for :class:`CriteriaGate`. Disabled by default (flag-graduation policy)."""

    enabled: bool = False
    max_downgrades: int = 2


class CriteriaGate:
    """Downgrade an ungrounded COMPLETE while caller criteria are unmet,
    bounded; annotate the final decision with any unmet minimums.

    Reads ``state["criteria"]`` (seeded from ``run(query, context=...)``).
    Each criterion key names a state key: the gate counts that state key's
    entries (list or dict length) toward the declared minimum. Criteria
    are opaque caller key/values — the framework never interprets what a
    state key collects.
    """

    def __init__(self, config: CriteriaGateConfig) -> None:
        self.config = config
        self.reset()

    def reset(self) -> None:
        self._downgrades = 0
        self.last_unmet: List[str] = []

    @staticmethod
    def _count_state_entries(state: Dict[str, Any], key: str) -> int:
        value = state.get(key)
        if isinstance(value, list):
            return len(value)
        if isinstance(value, dict):
            return len(value)
        return 0

    def unmet_criteria(self, state: Dict[str, Any]) -> List[Tuple[str, int, int]]:
        """Return [(name, count, minimum)] for every unmet declared minimum.

        Non-positive-int minimums are silently skipped: a JSON round-trip
        that widens ``3`` to ``3.0`` must not silently disable the gate's
        counterpart criteria, and a zero minimum is vacuous."""
        criteria = state.get("criteria")
        if not isinstance(criteria, dict):
            return []
        unmet: List[Tuple[str, int, int]] = []
        for name, minimum in criteria.items():
            if not isinstance(minimum, int) or isinstance(minimum, bool) or minimum <= 0:
                continue
            count = self._count_state_entries(state, str(name))
            if count < minimum:
                unmet.append((str(name), count, minimum))
        return unmet

    async def apply(
        self,
        evaluation: EvaluationResult,
        action_result: Any,
        state: Dict[str, Any],
    ) -> EvaluationResult:
        """Downgrade an ungrounded COMPLETE while caller criteria are unmet,
        bounded by ``max_downgrades``; annotate the final decision with any
        unmet minimums so the terminal report is honest. Passes everything
        else through, and yields (COMPLETE stands) when the caller's run
        context marks the iteration budget exhausted."""
        if not self.config.enabled:
            return evaluation
        if evaluation.decision != EvaluationDecision.COMPLETE:
            return evaluation
        unmet = self.unmet_criteria(state)
        budget_exhausted = bool(state.get("_criteria_budget_exhausted"))
        if not unmet or budget_exhausted or self._downgrades >= self.config.max_downgrades:
            if unmet:
                annotated = EvaluationResult(
                    decision=evaluation.decision,
                    score=evaluation.score,
                    reason=evaluation.reason,
                    metrics=evaluation.metrics,
                    metadata={
                        **(evaluation.metadata or {}),
                        "criteria_gate": {
                            "unmet": [name for name, _, _ in unmet],
                            "downgrades_exhausted": self._downgrades >= self.config.max_downgrades,
                            "budget_exhausted": budget_exhausted,
                        },
                    },
                )
                self.last_unmet = [name for name, _, _ in unmet]
                return annotated
            return evaluation
        self._downgrades += 1
        self.last_unmet = [name for name, _, _ in unmet]
        detail = ", ".join(f"{name}: {count}/{minimum}" for name, count, minimum in unmet)
        return EvaluationResult(
            decision=EvaluationDecision.RETRY,
            score=evaluation.score,
            reason=f"Caller completion criteria unmet (repair {self._downgrades}"
            f"/{self.config.max_downgrades}): {detail}. "
            "Continue collecting results toward the declared minimums.",
            metrics=evaluation.metrics,
            metadata={
                **(evaluation.metadata or {}),
                "criteria_gate": {"downgrade": self._downgrades,
                                  "unmet": [name for name, _, _ in unmet]},
            },
        )
