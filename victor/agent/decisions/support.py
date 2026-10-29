"""Shared decision-routing support for heuristic classifiers.

Several runtime classifiers (context compaction, continuation, error
classification, task completion, thinking-loop detection and task-type
classification) augment their heuristics with bounded LLM decisions. They all
route those decisions the same way: prefer the canonical runtime-intelligence
service, fall back to a legacy decision service, and otherwise decline.
"""

from __future__ import annotations

from typing import Any, Dict

__all__ = ["DecisionSupportMixin"]


class DecisionSupportMixin:
    """Route bounded decisions through runtime intelligence or a decision service.

    Classes using this mixin must set ``_runtime_intelligence`` and
    ``_decision_service`` (either may be ``None``) in ``__init__``.
    """

    _runtime_intelligence: Any
    _decision_service: Any

    def _has_decision_support(self) -> bool:
        """Return whether LLM-backed decisions are available."""
        if self._runtime_intelligence is not None:
            return True
        return self._decision_service is not None

    def _decide_sync(
        self,
        decision_type: Any,
        context: Dict[str, Any],
        *,
        heuristic_result: Any = None,
        heuristic_confidence: float = 0.0,
    ) -> Any:
        """Delegate a decision, preferring the canonical runtime-intelligence service.

        Returns:
            The decision result, or ``None`` when no decision backend is configured.
        """
        if self._runtime_intelligence is not None:
            return self._runtime_intelligence.decide_sync(
                decision_type,
                context,
                heuristic_result=heuristic_result,
                heuristic_confidence=heuristic_confidence,
            )
        if self._decision_service is not None:
            return self._decision_service.decide_sync(
                decision_type,
                context,
                heuristic_result=heuristic_result,
                heuristic_confidence=heuristic_confidence,
            )
        return None
