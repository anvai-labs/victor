"""Judge-identity pinning for rubric completion gating (ADR-011).

FINDINGS graduation-checklist item 3 (benchmarks/judge_calibration/FINDINGS.md):
`completion_strategy=rubric` may only gate completion when the judge model is
one that passed calibration. The rubric judge is the session's own provider
model (see ``_build_rubric_complete_fn``), so an uncalibrated session model
must downgrade rubric/hybrid to ``enhanced`` — loudly — rather than let an
unmeasured judge (or the heuristic fallback, measured at α=−0.092) gate alone.

Calibrated set defaults to the FINDINGS gate-passers and is overridable via
``agent.rubric_judge_calibrated_models`` in settings. Escape hatch for judge
experimentation: ``VICTOR_RUBRIC_JUDGE_ALLOW_UNCALIBRATED=1``.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

# FINDINGS runs 10-11: llama3.3:70b (α=1.000, n=96 scripted, gating-grade) and
# gemma4:31b (α=0.865 on real agent trajectories, zero false completions).
DEFAULT_CALIBRATED_JUDGE_MODELS: frozenset[str] = frozenset({"gemma4:31b", "llama3.3:70b"})

_ALLOW_UNCALIBRATED_ENV = "VICTOR_RUBRIC_JUDGE_ALLOW_UNCALIBRATED"

# Warn once per (model) per process — the strategy is resolved on every turn.
_warned_models: set[str] = set()


def _calibrated_models(settings: Any) -> frozenset[str]:
    configured = getattr(getattr(settings, "agent", None), "rubric_judge_calibrated_models", None)
    if configured:
        return frozenset(str(m).strip().lower() for m in configured)
    return DEFAULT_CALIBRATED_JUDGE_MODELS


def resolve_completion_strategy(settings: Any, judge_model: Optional[str]) -> str:
    """Full strategy resolution: env override → settings default → judge pinning.

    The single entry point both loop-construction sites use; keeps the
    resolution logic (and its 100-line-budget) out of the capped hotspot
    ``turn_execution_runtime``.
    """
    strategy = os.environ.get("VICTOR_COMPLETION_STRATEGY") or getattr(
        getattr(settings, "agent", None), "completion_strategy", "enhanced"
    )
    return resolve_gated_completion_strategy(strategy, judge_model, settings)


def resolve_gated_completion_strategy(
    strategy: str,
    judge_model: Optional[str],
    settings: Any = None,
) -> str:
    """Return the effective completion strategy after judge-identity pinning.

    ``rubric``/``hybrid`` pass through only when ``judge_model`` is in the
    calibrated set (or the escape-hatch env var is set); otherwise they
    downgrade to ``enhanced`` with a warning. All other strategies pass
    through unchanged.
    """
    if strategy not in ("rubric", "hybrid"):
        return strategy
    if os.environ.get(_ALLOW_UNCALIBRATED_ENV, "").lower() in ("1", "true", "yes"):
        return strategy

    normalized = (judge_model or "").strip().lower()
    if normalized and normalized in _calibrated_models(settings):
        return strategy

    key = normalized or "<no-model>"
    if key not in _warned_models:
        _warned_models.add(key)
        logger.warning(
            "completion_strategy=%s requested but judge model %r is not in the calibrated set "
            "(ADR-011 judge-identity pinning); falling back to 'enhanced'. Add the model to "
            "agent.rubric_judge_calibrated_models after it passes calibration "
            "(benchmarks/judge_calibration/), or set %s=1 to experiment.",
            strategy,
            judge_model,
            _ALLOW_UNCALIBRATED_ENV,
        )
    return "enhanced"
