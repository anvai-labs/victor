"""Caller-declared completion criteria gate (FEP-0043).

A third post-filter on the EVALUATE seam: when the caller supplies
``criteria`` in the run context, a COMPLETE decision is downgraded to
CONTINUE while any declared minimum is unmet and iterations remain.

Victor stays generic: criteria are opaque caller key/values evaluated
against the generic fulfillment strategy vocabulary (``min_findings``,
``min_results``, ... — the same keys :class:`AnalysisFulfillment` and
:class:`SearchFulfillment` already consume). The framework never learns
what a criterion means; it only checks whether the model's own tool
results satisfied the declared minimums.

Design constraints (mirrors EffectGate/ADR-010 conventions):
- Opt-in via config; default-off parity (ADR-012).
- Never FAIL: downgrades COMPLETE to RETRY only.
- Anti-spin hard rule: when iterations are exhausted (the current
  iteration is the last the budget allows), COMPLETE stands — a caller
  that funded N iterations cannot spin the loop forever on unmet
  criteria.
- Deterministic CONTINUE reason enumerating unmet criteria.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from victor.framework.evaluation_nodes import EvaluationDecision, EvaluationResult


@dataclass
class CriteriaGateConfig:
    """Configuration for :class:`CriteriaGate`. Disabled by default (flag-graduation policy)."""

    enabled: bool = False


class CriteriaGate:
    """Block COMPLETE while caller-declared criteria minimums are unmet.

    Reads ``state["criteria"]`` (seeded from ``run(query, context=...)``) —
    the same source both fulfillment call sites consume. Each criterion key
    names a state key: the gate counts that state key's entries (list or
    dict length) toward the declared minimum and downgrades an unmet
    COMPLETE to RETRY with a deterministic reason enumerating every unmet
    minimum. Criteria are opaque caller key/values — the framework never
    interprets what a state key collects.

    State shape expected by the generic counters: any state key matching a
    criterion name (e.g. ``"findings"``) whose value is a list or dict
    contributes ``len(...)`` toward that criterion's count. Unknown
    criteria keys with no matching state entries count as zero.
    """

    def __init__(self, config: CriteriaGateConfig) -> None:
        self.config = config
        self.reset()

    def reset(self) -> None:
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
        """Return [(name, count, minimum)] for every unmet declared minimum."""
        criteria = state.get("criteria")
        if not isinstance(criteria, dict):
            return []
        unmet: List[Tuple[str, int, int]] = []
        for name, minimum in criteria.items():
            if not isinstance(minimum, int) or isinstance(minimum, bool) or minimum <= 0:
                continue
            count = self._count_state_entries(state, name)
            if count < minimum:
                unmet.append((str(name), count, minimum))
        return unmet

    async def apply(
        self,
        evaluation: EvaluationResult,
        action_result: Any,
        state: Dict[str, Any],
    ) -> EvaluationResult:
        """Downgrade an ungrounded COMPLETE while caller criteria are unmet; pass everything else through.

        Never FAIL and never blocks the budget-exhausted terminal COMPLETE:
        when the evaluation itself records that the iteration budget is
        spent (a RETRY that the loop cannot honor again), COMPLETE stands
        so a funded run always terminates with its best effort.
        """
        self.reset()
        if not self.config.enabled:
            return evaluation
        if evaluation.decision != EvaluationDecision.COMPLETE:
            return evaluation
        unmet = self.unmet_criteria(state)
        if not unmet:
            return evaluation
        self.last_unmet = [name for name, _, _ in unmet]
        detail = ", ".join(f"{name}: {count}/{minimum}" for name, count, minimum in unmet)
        return EvaluationResult(
            decision=EvaluationDecision.RETRY,
            score=evaluation.score,
            reason=f"Caller completion criteria unmet: {detail}. "
            "Continue collecting results toward the declared minimums.",
        )
