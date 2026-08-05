# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License").

"""Always-on, logging-only decision service (FEP-0012 closed loop).

The problem this solves: ``log_decision`` is called from exactly one place —
inside ``LLMDecisionService.decide_sync`` — and only on the LLM-escalation
path. So when **no** decision backend is registered (``--no-edge-model`` + no
trained classifier artifact → AUTO resolves to ``None``), **zero** decisions
are logged. The benchmark then produces outcomes but no decisions, so
``victor ml mine``/``train`` have nothing to train on.

This service is the fix. It implements ``LLMDecisionServiceProtocol`` as a
heuristic pass-through — it returns the caller's heuristic result unchanged
(so decision **outcomes** are identical to the no-service case) but **logs
every call** via ``log_decision``. It is registered as the ``HEURISTIC``
backend and the ``AUTO`` fallback when no classifier/edge/LLM backend is
available, guaranteeing decisions are always captured for RL/training data.

Design notes
------------
- It never calls an LLM/edge model — there is nothing to "decide" beyond the
  heuristic the caller already computed; the value is the **telemetry**.
- ``is_healthy()`` is always True so AUTO adopts it as the fallback.
- Logging is best-effort (``log_decision`` already swallows errors).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict

from victor.agent.decisions.chain import log_decision
from victor.agent.services.protocols.decision_service import DecisionResult

logger = logging.getLogger(__name__)


class LoggingDecisionService:
    """Heuristic pass-through decision service that logs every decision.

    Implements ``LLMDecisionServiceProtocol``. Returns the caller-supplied
    heuristic result (no LLM call) and records the decision via
    :func:`log_decision` so it joins the closed-loop training data.
    """

    def __init__(self) -> None:
        self._calls = 0

    def is_healthy(self) -> bool:
        """Always healthy — this service has no dependencies to fail on."""
        return True

    def _make_result(
        self,
        decision_type: Any,
        heuristic_result: Any,
        heuristic_confidence: float,
        latency_ms: float,
    ) -> DecisionResult:
        return DecisionResult(
            decision_type=decision_type,
            result=heuristic_result,
            source="heuristic",
            confidence=heuristic_confidence,
            latency_ms=latency_ms,
        )

    def _log(
        self,
        decision_type: Any,
        context: Dict[str, Any],
        result: DecisionResult,
    ) -> None:
        try:
            log_decision(
                decision_type=getattr(decision_type, "value", str(decision_type)),
                context=context,
                result=str(getattr(result.result, "__dict__", result.result)),
                source=result.source,
                confidence=result.confidence,
            )
        except Exception as exc:  # never break a decision on logging
            logger.debug("LoggingDecisionService: log_decision failed: %s", exc)

    def decide_sync(
        self,
        decision_type: Any,
        context: Dict[str, Any],
        *,
        heuristic_result: Any = None,
        heuristic_confidence: float = 0.0,
    ) -> DecisionResult:
        start = time.monotonic()
        self._calls += 1
        result = self._make_result(
            decision_type, heuristic_result, heuristic_confidence, _elapsed_ms(start)
        )
        self._log(decision_type, context, result)
        return result

    async def decide(
        self,
        decision_type: Any,
        context: Dict[str, Any],
        *,
        heuristic_result: Any = None,
        heuristic_confidence: float = 0.0,
    ) -> DecisionResult:
        return self.decide_sync(
            decision_type,
            context,
            heuristic_result=heuristic_result,
            heuristic_confidence=heuristic_confidence,
        )

    async def decide_async(
        self,
        decision_type: Any,
        context: Dict[str, Any],
        *,
        heuristic_result: Any = None,
        heuristic_confidence: float = 0.0,
    ) -> DecisionResult:
        return self.decide_sync(
            decision_type,
            context,
            heuristic_result=heuristic_result,
            heuristic_confidence=heuristic_confidence,
        )


def _elapsed_ms(start: float) -> float:
    return (time.monotonic() - start) * 1000.0
