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
from typing import Any, Dict, List, Mapping, Tuple

from victor.framework.evaluation_nodes import EvaluationDecision, EvaluationResult


def _declared_minimums(criteria: Any) -> Dict[str, int]:
    """Normalize caller criteria to ``{key: positive_int_minimum}``.

    Non-positive-int minimums are silently dropped: a JSON round-trip that
    widens ``3`` to ``3.0`` must not silently disable the gate's counterpart
    criteria, and a zero minimum is vacuous. Shared by the gate's counting
    and the state mirroring so both read one declaration the same way.
    """
    if not isinstance(criteria, Mapping):
        return {}
    declared: Dict[str, int] = {}
    for name, minimum in criteria.items():
        if isinstance(minimum, int) and not isinstance(minimum, bool) and minimum > 0:
            declared[str(name)] = minimum
    return declared


def mirror_criteria_state(state: Dict[str, Any], tool_results: Any) -> Dict[str, int]:
    """Mirror criteria-declared fields from tool results into loop state.

    The gate counts state entries; this is the opt-in seam that lets a run's
    own tool evidence populate those entries. For every key declared in
    ``state["criteria"]``, each successful tool result carrying a same-named
    list or dict contributes its entries to ``state[key]``: lists extend with
    duplicates dropped (a corrected re-proposal that re-reports prior entries
    counts them once — the floor must never be satisfied by double counting),
    dicts merge by key. The result dict itself and, when present, its nested
    ``result`` payload are both inspected. Values that are neither lists nor
    dicts, failed results, and undeclared keys are ignored.

    Victor stays generic: the framework never interprets what a declared key
    collects; the consumer declares the key and shapes its tool payloads.
    Returns ``{key: entry_count}`` for observability of what was mirrored.
    """
    declared = _declared_minimums(state.get("criteria"))
    if not declared:
        return {}
    mirrored: Dict[str, int] = {}
    for result in tool_results or []:
        if not isinstance(result, dict) or not result.get("success"):
            continue
        payloads: List[Mapping[str, Any]] = [result]
        nested = result.get("result")
        if isinstance(nested, Mapping):
            payloads.append(nested)
        for key in declared:
            for payload in payloads:
                value = payload.get(key)
                if isinstance(value, list):
                    entries = state.setdefault(key, [])
                    if isinstance(entries, list):
                        entries.extend(item for item in value if item not in entries)
                        mirrored[key] = len(entries)
                elif isinstance(value, dict):
                    entries = state.setdefault(key, {})
                    if isinstance(entries, dict):
                        entries.update(value)
                        mirrored[key] = len(entries)
    return mirrored


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
        unmet: List[Tuple[str, int, int]] = []
        for name, minimum in _declared_minimums(state.get("criteria")).items():
            count = self._count_state_entries(state, name)
            if count < minimum:
                unmet.append((name, count, minimum))
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
            # All declared minimums met: clear any stale unmet record so the
            # diagnostic reads the current evaluation, not the last downgrade.
            self.last_unmet = []
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
                "criteria_gate": {
                    "downgrade": self._downgrades,
                    "unmet": [name for name, _, _ in unmet],
                },
            },
        )
