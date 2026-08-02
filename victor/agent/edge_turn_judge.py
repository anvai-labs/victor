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

"""Edge-model per-turn judge (EVR-6, FEP-0008 Phase C).

The runtime-side implementation of the framework's ``TurnJudge`` seam: it judges an agent turn
with the local **edge model** (``victor/agent/edge_model.py`` → Ollama, token-free, ~sub-second)
via the standard decision service (``DecisionType.TURN_AUDIT``), returning an
:class:`~victor.framework.per_turn_auditor.AuditSignal` — or ``None`` when the edge model has no
usable opinion (unavailable / fell back to heuristic / timed out), so the auditor keeps its
deterministic verdict.

Lives on the *agent* side so the framework ``PerTurnAuditor`` stays free of any edge-model
dependency; the runtime injects an instance (agent → framework imports are the allowed direction).
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Optional

from victor.framework.per_turn_auditor import AuditSignal, AuditVerdict

logger = logging.getLogger(__name__)

#: How much of the turn's output tail the judge sees — small, to keep the edge call cheap.
_TAIL_CHARS = 500


class EdgeTurnJudge:
    """A :class:`TurnJudge` backed by the edge-model decision service."""

    def __init__(self, service: Any) -> None:
        self._service = service

    def judge(self, action_result: Any, state: Optional[dict]) -> Optional[AuditSignal]:
        if self._service is None:
            return None
        from victor.agent.decisions.schemas import DecisionType

        content = str(getattr(action_result, "content", "") or "")
        tool_results = getattr(action_result, "tool_results", None) or []
        has_tools = bool(tool_results)
        all_failed = has_tools and not any(
            isinstance(r, Mapping) and r.get("success") for r in tool_results
        )
        try:
            decision = self._service.decide_sync(
                DecisionType.TURN_AUDIT,
                {
                    "turn_tail": content[-_TAIL_CHARS:],
                    "has_tool_calls": has_tools,
                    "all_tools_failed": all_failed,
                },
                heuristic_confidence=0.0,
            )
        except Exception:  # noqa: BLE001 — a judge must never break the loop
            logger.debug("[EdgeTurnJudge] decide_sync raised", exc_info=True)
            return None

        # Only trust an actual LLM verdict; heuristic/cache-fallback/timeout → no opinion.
        if decision is None or getattr(decision, "source", None) != "llm":
            return None
        result = getattr(decision, "result", None)
        verdict = getattr(result, "verdict", None)
        if verdict == "alarm":
            reason = (
                str(getattr(result, "reason", "") or "").strip() or "edge judge flagged the turn"
            )
            return AuditSignal(AuditVerdict.ALARM, f"edge judge: {reason}")
        if verdict == "continue":
            return AuditSignal(AuditVerdict.CONTINUE)
        return None


def build_edge_turn_judge(*, timeout_ms: int = 2000) -> Optional[EdgeTurnJudge]:
    """Build an :class:`EdgeTurnJudge`, or ``None`` when no edge model is available.

    Cheap-to-call and safe: :func:`create_edge_decision_service` returns ``None`` when Ollama /
    the edge model is unavailable, so the auditor degrades to its deterministic heuristic.
    """
    from victor.agent.edge_model import EdgeModelConfig, create_edge_decision_service

    service = create_edge_decision_service(EdgeModelConfig(timeout_ms=timeout_ms))
    if service is None:
        return None
    return EdgeTurnJudge(service)
