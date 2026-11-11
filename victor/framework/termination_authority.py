# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
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

"""TerminationAuthority — one decision object for the loop-terminating
heuristics (seam-duplication audit Cluster 3, consolidation PR 5).

Before this module, #1725's criteria-guided suppression ("while the caller
declared completion objectives that remain unmet and iteration budget
remains, degrade the termination to guidance") was re-implemented at three
evaluation sites (the legacy spin branch and the adaptive-plateau check in
``agentic_loop.py``, and ``_check_spin_detection`` in
``enhanced_completion_evaluation.py``), and the loop's
``_criteria_budget_exhausted`` marker was written inline at six sites. The
streaming blocked-streak seam had no guided suppression at all.

``TerminationAuthority`` owns that one decision:

- ``spin_decision`` — the spin heuristic fired (SpinState.TERMINATED);
  returns the legacy FAIL bytes or the guided-CONTINUE bytes.
- ``plateau_decision`` — the adaptive-plateau check; returns ``decide``
  when the heuristic does not fire, ``terminate`` for the legacy PLATEAU
  exit, ``guide`` while suppression applies.
- ``observe_iteration`` / ``reset_budget_marker`` — the single
  per-iteration fold of the criteria budget marker.
- ``guided_suppression_check`` — the shared predicate for seams that only
  need the suppression test (streaming blocked-streak tracking).

Semantics: while ``criteria_check()`` is True the heuristics return
``guide`` (the caller injects the guided CONTINUE / logs the suppression);
without a declaration — or once the budget marker flips — every decision
is byte-identical to the pre-consolidation behavior. Termination stays
guaranteed regardless: the iteration cap, the existing bounded
criteria-gate yields, and the LoopGuards wall-clock are not gated on this
object.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Mapping, Optional

# Action values for TerminationDecision.action.
ACTION_TERMINATE = "terminate"
"""The heuristic fires exactly as it did pre-consolidation."""
ACTION_GUIDE = "guide"
"""The heuristic fired; criteria-guided suppression applies."""
ACTION_DECIDE = "decide"
"""The heuristic did not fire; normal evaluation, unchanged."""

GUIDED_SUPPRESSION_SUFFIX = (
    "; caller-declared completion criteria are unmet with iteration budget "
    "remaining — continuing with guidance instead of failing"
)
"""Appended to a spin reason when guided suppression degrades FAIL → CONTINUE."""

PLATEAU_SUPPRESSED_LOG = (
    "Adaptive plateau suppressed: caller-declared completion criteria are "
    "unmet with iteration budget remaining"
)
"""The plateau suppression log/reason line (byte-identical to pre-consolidation)."""


@dataclass
class TerminationDecision:
    """Outcome of one termination-heuristic consultation.

    Attributes:
        action: ``'terminate'`` (heuristic fires, legacy bytes), ``'guide'``
            (heuristic fired but criteria-guided suppression applies — the
            caller injects the guided CONTINUE / logs the suppression), or
            ``'decide'`` (heuristic did not fire; normal evaluation,
            unchanged).
        reason: The exact legacy reason string for the chosen action.
        score: The legacy score (0.1/0.2 for spin outcomes; informational
            for plateau, whose caller owns its own scores).
    """

    action: str
    reason: str
    score: float = 0.1


@dataclass
class TerminationAuthority:
    """Owns the criteria-guided suppression decision for loop-terminating
    heuristics.

    Args:
        criteria_check: Zero-argument predicate; True while the caller
            declared completion objectives remain unmet with iteration
            budget remaining. ``TerminationAuthority.for_state(state)``
            binds it to ``criteria_gate.criteria_unmet_with_budget``.
        max_consecutive_blocked: The fully-blocked-batch threshold the
            authority's consumer uses for spin detection (mirrors
            ``SpinDetectorConfig.max_consecutive_all_blocked``); carried so
            the authority is the single place that names the threshold.
    """

    criteria_check: Callable[[], bool]
    max_consecutive_blocked: int = 4

    @classmethod
    def for_state(cls, state: Any) -> "TerminationAuthority":
        """Build an authority bound to a loop-state mapping.

        ``state`` may be ``None`` (no declaration → unguided, the
        default-off parity path).
        """
        from victor.framework.criteria_gate import criteria_unmet_with_budget

        return cls(criteria_check=lambda: criteria_unmet_with_budget(state))

    @property
    def guided(self) -> bool:
        """True while criteria-guided suppression applies this iteration."""
        return bool(self.criteria_check())

    def spin_decision(
        self,
        consecutive_all_blocked: int = 0,
        repetition_count: int = 0,
        consecutive_no_tool_turns: int = 0,
        repetition_threshold: int = 3,
    ) -> TerminationDecision:
        """Decide the spin outcome, mirroring the legacy branch precedence:
        fully-blocked batches, then repetition (fires at
        ``repetition_threshold - 1``), then stuck-without-tools.

        Unguided: the legacy FAIL reason and score (0.1 blocked/repetition,
        0.2 stuck). Guided: the same reason plus
        ``GUIDED_SUPPRESSION_SUFFIX`` at CONTINUE score 0.2.
        """
        if consecutive_all_blocked > 0:
            base = (
                f"Spin detected: {consecutive_all_blocked} consecutive fully-blocked tool batches"
            )
            fail_score = 0.1
        elif repetition_count >= repetition_threshold - 1:
            base = (
                f"Repetitive loop detected: the model repeated the same tool "
                f"calls for {repetition_count} consecutive turns"
            )
            fail_score = 0.1
        else:
            base = f"Agent stuck: {consecutive_no_tool_turns} turns without tool calls"
            fail_score = 0.2
        if self.guided:
            return TerminationDecision(
                action=ACTION_GUIDE,
                reason=base + GUIDED_SUPPRESSION_SUFFIX,
                score=0.2,
            )
        return TerminationDecision(
            action=ACTION_TERMINATE,
            reason=base,
            score=fail_score,
        )

    def plateau_decision(
        self,
        progress_scores: List[float],
        iteration: int,
        plateau_window: int,
        plateau_tolerance: float,
        min_plateau_iteration: int,
    ) -> TerminationDecision:
        """Decide the adaptive-plateau outcome over ``progress_scores``.

        ``decide`` when the plateau heuristic does not fire (not enough
        history, iteration below the deferred minimum, or the recent window
        still improves); ``terminate`` for the legacy PLATEAU exit;
        ``guide`` while suppression applies (the caller logs
        ``PLATEAU_SUPPRESSED_LOG`` and continues — the iteration cap still
        bounds the loop).
        """
        scores = list(progress_scores)
        if len(scores) < plateau_window or iteration < min_plateau_iteration:
            return TerminationDecision(action=ACTION_DECIDE, reason="", score=0.0)
        recent = scores[-plateau_window:]
        improvement = max(recent) - min(recent)
        if improvement >= plateau_tolerance:
            return TerminationDecision(action=ACTION_DECIDE, reason="", score=0.0)
        if self.guided:
            return TerminationDecision(
                action=ACTION_GUIDE,
                reason=PLATEAU_SUPPRESSED_LOG,
                score=0.0,
            )
        return TerminationDecision(action=ACTION_TERMINATE, reason="", score=0.0)

    def reset_budget_marker(self, state: Any) -> None:
        """Loop-init fold of the criteria budget marker (ungated start)."""
        state["_criteria_budget_exhausted"] = False

    def observe_iteration(self, state: Any, iteration: int, effective_max: int) -> None:
        """Single per-iteration fold of the criteria budget marker.

        Marks the final funded iteration (``iteration >= effective_max``) so
        the criteria gate yields (COMPLETE stands) there instead of forcing
        a RETRY, and so guided suppression switches off — from that
        iteration on, the bounded yields and the iteration cap own
        termination.
        """
        state["_criteria_budget_exhausted"] = iteration >= effective_max

    @classmethod
    def guided_suppression_check(cls, state: Any) -> Optional[Callable[[], bool]]:
        """Shared guided-suppression predicate for seams that only need the
        suppression test (streaming blocked-streak tracking).

        Returns ``None`` when nothing is declared (the seam keeps its
        ungated legacy behavior — default-off parity), otherwise a
        zero-argument callable answering "is guided suppression active".
        """
        if not isinstance(state, Mapping):
            return None
        if not (state.get("criteria") or state.get("complete_on_key")):
            return None
        authority = cls.for_state(state)
        return lambda: authority.guided
