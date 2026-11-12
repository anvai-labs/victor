"""Run-scoped tool budget ledger (seam-duplication audit Cluster 1).

Victor's ~10 independent tool-budget owners/envelopes (see
``docs/analysis/2026-11-09-seam-duplication-audit.md``, Cluster 1) each kept
their own (ceiling, used) pair, and their counters diverged: a caller-funded
134-call run was refused at ~55-60 calls because cumulative session counters
were subtracted from the per-run funding. This module collapses the owners
that enforce tool-call admission onto ONE object per run:

- ``ToolBudgetLedger`` owns (ceiling, used, scope). ``retarget`` is the
  caller re-funding seam (the ceiling moves, usage persists); ``reset``
  preserves the interactive-turn replenish semantics.
- Enforcement sites (pipeline batch admission, single-call admission,
  ``ToolService`` budget runtime) read remaining/used THROUGH the ledger
  when one is bound. Unbound, every site keeps its legacy
  ``config.tool_budget`` / cumulative-counter path unchanged.
- ``runtime_context_overrides["tool_budget"]`` is the only funding write:
  the first funding binds a fresh ledger to the run (prior session usage is
  NOT subtracted), later fundings retarget the existing one, and the funding
  is deliberately NOT rolled back at turn end (replacing the per-turn
  budget apply/restore).

Read-through helpers live here so call sites stay one-liners.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple


class ToolBudgetLedger:
    """Own (ceiling, used, scope) for one run's tool-call budget.

    The ledger never raises on over-spend: consuming past the ceiling clamps
    ``remaining`` at zero and reports exhaustion, matching the legacy
    cumulative-counter behavior at every enforcement site.
    """

    def __init__(self, ceiling: int, scope: str = "run") -> None:
        ceiling = int(ceiling)
        if ceiling < 0:
            raise ValueError(f"Budget ceiling must be non-negative: {ceiling}")
        self._ceiling = ceiling
        self._used = 0
        self._scope = scope

    @property
    def ceiling(self) -> int:
        """Funded ceiling for this run (moved by :meth:`retarget`)."""
        return self._ceiling

    @property
    def used(self) -> int:
        """Tool calls consumed against this run's ledger."""
        return self._used

    @property
    def remaining(self) -> int:
        """Funded calls still admittable (clamped at zero)."""
        return max(0, self._ceiling - self._used)

    @property
    def is_exhausted(self) -> bool:
        """Whether no funded calls remain."""
        return self._used >= self._ceiling

    @property
    def scope(self) -> str:
        """Accounting scope label (default ``"run"``)."""
        return self._scope

    def consume(self, amount: int = 1) -> None:
        """Record tool usage against this run.

        Raises:
            ValueError: If ``amount`` is negative.
        """
        if amount < 0:
            raise ValueError(f"Cannot consume negative budget: {amount}")
        self._used += int(amount)

    def retarget(self, new_ceiling: int) -> None:
        """Re-fund this run: move the ceiling, keep usage.

        This is the fix for the measured divergence: cumulative counters plus
        a funded override used to silently collapse the run budget to
        ``new_ceiling - prior usage``. Retargeting keeps ``used`` so the
        funded ceiling is the run TOTAL.

        Raises:
            ValueError: If ``new_ceiling`` is negative.
        """
        new_ceiling = int(new_ceiling)
        if new_ceiling < 0:
            raise ValueError(f"Budget ceiling must be non-negative: {new_ceiling}")
        self._ceiling = new_ceiling

    def reset(self) -> None:
        """Clear usage while keeping the ceiling (interactive-turn replenish)."""
        self._used = 0

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"ToolBudgetLedger(ceiling={self._ceiling}, used={self._used}, "
            f"scope={self._scope!r})"
        )


def resolve_budget_pipeline(orchestrator: Any, fallback: Any) -> Any:
    """Find the pipeline that can host the run ledger, if any."""
    pipeline = getattr(orchestrator, "_tool_pipeline", None)
    if pipeline is None:
        pipeline = getattr(fallback, "_tool_pipeline", None)
    return pipeline


def _bound_ledger(pipeline: Any) -> Optional[ToolBudgetLedger]:
    ledger = getattr(pipeline, "_budget_ledger", None)
    return ledger if isinstance(ledger, ToolBudgetLedger) else None


def resolved_budget_view(orchestrator: Any, fallback: Any, explicit: Any = None) -> Tuple[int, int]:
    """Return (effective ceiling, used) for budget reads at admission seams.

    ``explicit`` (a raw override value, coerced here) wins as the ceiling.
    Ledger-bound: both values otherwise come from the run ledger. Unbound:
    the legacy mirror values (``tool_budget`` / ``tool_calls_used``) pass
    through unchanged, so unbound runs behave exactly as before.
    """
    try:
        explicit_ceiling = None if explicit is None else int(explicit)
    except (TypeError, ValueError):
        explicit_ceiling = None
    ledger = _bound_ledger(resolve_budget_pipeline(orchestrator, fallback))
    if ledger is not None:
        ceiling = ledger.ceiling
        used = ledger.used
    else:
        ceiling = getattr(fallback, "tool_budget", None)
        used = getattr(fallback, "tool_calls_used", None)
        if ceiling is None:
            ceiling = 0
        if used is None:
            used = 0
    if explicit_ceiling is not None:
        ceiling = explicit_ceiling
    return ceiling, used


__all__ = ["ToolBudgetLedger", "resolve_budget_pipeline", "resolved_budget_view"]
