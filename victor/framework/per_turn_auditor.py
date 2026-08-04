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

"""Online per-turn auditor (EVR-6, FEP-0008 Phase C) — MVP.

A continue/alarm check applied to each turn at the EVALUATE seam, mirroring the effect gate
(ADR-010 / EVR-4): it wraps the completion cascade as a post-filter and, when it *alarms*,
downgrades a ``COMPLETE`` to ``RETRY`` (never ``FAIL``), bounded by ``max_alarms`` then
annotate-and-allow — so a too-eager auditor can never trap the loop.

**Two-tier signal.** A **deterministic** turn-health heuristic runs first and always — it flags a
*degenerate* completion: a turn that claims done while producing neither assistant content nor any
successful tool call (the "declare success, do nothing" failure mode), with no LLM/network/latency.
If the heuristic does not alarm, an optional **edge-model judge** (a :class:`TurnJudge` injected by
the runtime — see ``victor/agent/edge_turn_judge.py``, backed by the local Ollama edge model) gets
the final say; it returns ``None`` (no opinion) when the edge model is unavailable, leaving the
deterministic verdict intact. The framework stays free of any edge-model dependency (the judge is
injected, not imported).

Opt-in, default off per the flag-graduation policy; a strict no-op when disabled (ADR-012 parity).
The framework side is pure and unit-testable standalone (inject a fake judge).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Protocol, runtime_checkable

from victor.framework.evaluation_nodes import EvaluationDecision, EvaluationResult

logger = logging.getLogger(__name__)


@runtime_checkable
class TurnJudge(Protocol):
    """A pluggable per-turn judge (EVR-6). Kept abstract so the framework does not depend on the
    runtime edge-model service — the agent side injects an implementation.

    ``judge`` returns an :class:`AuditSignal` (a continue/alarm opinion) or ``None`` when it has no
    opinion (e.g. the edge model was unavailable or fell back to a heuristic), in which case the
    auditor keeps its deterministic verdict.
    """

    def judge(
        self, action_result: Any, state: Optional[Dict[str, Any]]
    ) -> "Optional[AuditSignal]": ...


class AuditVerdict(str, Enum):
    """A per-turn auditor decision."""

    CONTINUE = "continue"  # the turn looks healthy — pass through
    ALARM = "alarm"  # the turn looks wrong — downgrade a COMPLETE to RETRY


@dataclass(frozen=True)
class AuditSignal:
    """The auditor's verdict for one turn, with a human-readable reason."""

    verdict: AuditVerdict
    reason: str = ""

    @property
    def is_alarm(self) -> bool:
        return self.verdict is AuditVerdict.ALARM


@dataclass
class PerTurnAuditorConfig:
    """Config for :class:`PerTurnAuditor`. Disabled by default (flag-graduation policy)."""

    enabled: bool = False
    # Bounded downgrades per run, mirroring the effect gate — then annotate-and-allow so the
    # auditor can never trap the loop in a RETRY cycle.
    max_alarms: int = 2


def resolve_per_turn_auditor_enabled(settings: Any) -> bool:
    """Resolve the EVR-6 auditor flag: env → AgentSettings → default off.

    Mirrors :func:`victor.framework.effect_gate.resolve_effect_gate_enabled`. Env override
    ``VICTOR_PER_TURN_AUDITOR`` wins; otherwise ``settings.agent.per_turn_auditor`` (default False).
    """
    env = os.environ.get("VICTOR_PER_TURN_AUDITOR")
    if env is not None:
        return env.strip().lower() in ("1", "true", "yes", "on")
    return bool(getattr(getattr(settings, "agent", None), "per_turn_auditor", False))


class PerTurnAuditor:
    """Post-filter on the EVALUATE seam that gates COMPLETE on a per-turn continue/alarm check."""

    def __init__(
        self,
        config: Optional[PerTurnAuditorConfig] = None,
        *,
        judge: Optional[TurnJudge] = None,
    ) -> None:
        self.config = config or PerTurnAuditorConfig()
        # Optional pluggable judge (EVR-6): an edge-model prefix judge injected by the runtime.
        # None → deterministic heuristic only. Runs only when the heuristic did not already alarm.
        self._judge = judge
        self._alarms = 0
        self._turn_index = 0

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def reset(self) -> None:
        """Reset per-run counters (call at the start of each loop run, like the effect gate)."""
        self._alarms = 0
        self._turn_index = 0

    def audit_turn(self, action_result: Any, state: Optional[Dict[str, Any]] = None) -> AuditSignal:
        """Return the continue/alarm signal for one turn's ``TurnResult``-shaped result.

        Deterministic heuristic — a *degenerate* completion (no assistant content and no successful
        tool call) — runs first and always. If it does not alarm and an edge-model
        :class:`TurnJudge` is injected, the judge gets the final say (returning ``None`` leaves the
        heuristic's CONTINUE verdict intact, e.g. when the edge model is unavailable).
        """
        content = str(getattr(action_result, "content", "") or "").strip()
        tool_results = getattr(action_result, "tool_results", None) or []
        any_success = any(isinstance(r, Mapping) and r.get("success") for r in tool_results)
        if not content and not any_success:
            return AuditSignal(
                AuditVerdict.ALARM,
                "degenerate turn: COMPLETE with no assistant content and no successful tool call",
            )
        if self._judge is not None:
            try:
                judged = self._judge.judge(action_result, state)
            except Exception:  # noqa: BLE001 — a judge must never break the loop
                logger.debug(
                    "[PerTurnAuditor] judge raised; keeping heuristic verdict", exc_info=True
                )
                judged = None
            if judged is not None:
                return judged
        return AuditSignal(AuditVerdict.CONTINUE)

    def apply(
        self,
        evaluation: EvaluationResult,
        action_result: Any,
        state: Optional[Dict[str, Any]] = None,
    ) -> EvaluationResult:
        """Gate a COMPLETE evaluation on the per-turn signal; pass everything else through.

        Strict no-op when disabled or when the decision is not COMPLETE. On alarm, downgrades to
        RETRY (bounded by ``max_alarms``, then annotate-and-allow).
        """
        if not self.config.enabled:
            return evaluation
        self._turn_index += 1
        if not evaluation.should_complete:
            return evaluation

        signal = self.audit_turn(action_result, state)
        if not signal.is_alarm:
            evaluation.metadata["per_turn_auditor"] = {"verdict": "continue"}
            return evaluation

        if self._alarms >= max(0, int(self.config.max_alarms)):
            logger.info(
                "[PerTurnAuditor] alarm but budget exhausted (%d/%d) — allowing COMPLETE",
                self._alarms,
                self.config.max_alarms,
            )
            evaluation.metadata["per_turn_auditor_exhausted"] = True
            evaluation.metadata["per_turn_auditor"] = {
                "verdict": "alarm",
                "reason": signal.reason,
                "allowed": True,
            }
            return evaluation

        self._alarms += 1
        logger.info(
            "[PerTurnAuditor] COMPLETE downgraded to RETRY: %s (alarm %d/%d)",
            signal.reason,
            self._alarms,
            self.config.max_alarms,
        )
        return EvaluationResult(
            decision=EvaluationDecision.RETRY,
            score=min(evaluation.score, 0.4),
            reason=f"per-turn auditor alarm: {signal.reason} (EVR-6)",
            metrics=dict(evaluation.metrics),
            metadata={
                **evaluation.metadata,
                "per_turn_auditor": {"verdict": "alarm", "reason": signal.reason},
            },
        )
