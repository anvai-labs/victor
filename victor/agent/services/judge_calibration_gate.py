"""Judge-identity pinning + judge-backend resolution for rubric completion (ADR-011, FEP-0030).

FINDINGS graduation-checklist item 3 (benchmarks/judge_calibration/FINDINGS.md):
`completion_strategy=rubric` may only gate completion when the judge model is
one that passed calibration. Historically the rubric judge was hardwired to the
session's own provider model (see ``_build_rubric_complete_fn``), which made it
a self-judge and confined rubric to sessions whose *chat* model is calibrated.

FEP-0030 Phase 1 introduces the ``agent.completion_judge`` seam: the judge model
is resolved from config, INDEPENDENT of the session model, via
``resolve_completion_judge_model``. The default ``"session-model"`` reproduces
the historical behavior exactly (judge = session model); ``"enhanced"`` forces
the algorithmic evaluator (rubric/hybrid downgrade). Decoupled backends
(``llm:…``, ``classifier:…``) are Phase 2. The calibration pin then applies to
the *resolved* judge, not the session model.

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
# llama3.3:70b: scripted α=1.000 (run 10) + real-trajectory α=0.878 with refactor 1.000
# (run 13, vs verifier AND annotator gold). gemma4:31b was demoted in run 12 (α=0.694 at
# n=96, refactor 0.300). See docs/architecture/judge-independence-experiments.md.
DEFAULT_CALIBRATED_JUDGE_MODELS: frozenset[str] = frozenset({"llama3.3:70b"})

_ALLOW_UNCALIBRATED_ENV = "VICTOR_RUBRIC_JUDGE_ALLOW_UNCALIBRATED"

# Warn once per (model) per process — the strategy is resolved on every turn.
_warned_models: set[str] = set()


def _calibrated_models(settings: Any) -> frozenset[str]:
    configured = getattr(getattr(settings, "agent", None), "rubric_judge_calibrated_models", None)
    if configured:
        return frozenset(str(m).strip().lower() for m in configured)
    return DEFAULT_CALIBRATED_JUDGE_MODELS


# FEP-0030 completion-judge backends recognized in Phase 1. Decoupled backends
# ("llm:<model>", "classifier:<path>") are defined but not yet resolvable here —
# Phase 2 wires them; until then they resolve to the session model with a note,
# so config written ahead of Phase 2 never breaks a session.
_JUDGE_BACKEND_ENV = "VICTOR_COMPLETION_JUDGE"
_warned_backends: set[str] = set()


def resolve_completion_judge_model(settings: Any, session_model: Optional[str]) -> Optional[str]:
    """Resolve the judge model from ``agent.completion_judge`` (FEP-0030 seam).

    Independent of the session model by design. Returns the model name the
    calibration pin should check, or ``None`` to force the algorithmic
    ``enhanced`` path.

    - ``"session-model"`` (default): the session's own model — the historical
      behavior, preserved exactly.
    - ``"enhanced"``: ``None`` — rubric/hybrid downgrade to enhanced.
    - ``"llm:<model>"`` / ``"classifier:<path>"``: Phase 2 backends; for now
      treated as ``"session-model"`` with a one-time note (forward-compatible).
    """
    backend = os.environ.get(_JUDGE_BACKEND_ENV) or getattr(
        getattr(settings, "agent", None), "completion_judge", "session-model"
    )
    backend = (backend or "session-model").strip()

    if backend == "session-model":
        return session_model
    if backend == "enhanced":
        return None
    if backend.startswith(("llm:", "classifier:")):
        if backend not in _warned_backends:
            _warned_backends.add(backend)
            logger.info(
                "agent.completion_judge=%r is a FEP-0030 Phase 2 decoupled backend, not yet "
                "wired; using the session model as the judge for now.",
                backend,
            )
        return session_model
    if backend not in _warned_backends:
        _warned_backends.add(backend)
        logger.warning(
            "agent.completion_judge=%r is not recognized; using the session model as the judge. "
            "Valid: 'session-model' (default), 'enhanced'.",
            backend,
        )
    return session_model


def resolve_completion_strategy(settings: Any, session_model: Optional[str]) -> str:
    """Full strategy resolution: env override → settings default → judge backend → pinning.

    The single entry point both loop-construction sites use; keeps the
    resolution logic (and its 100-line-budget) out of the capped hotspot
    ``turn_execution_runtime``. The judge model is resolved from the
    ``agent.completion_judge`` backend (FEP-0030), independent of the session
    model; the default ``"session-model"`` preserves historical behavior.
    """
    strategy = os.environ.get("VICTOR_COMPLETION_STRATEGY") or getattr(
        getattr(settings, "agent", None), "completion_strategy", "enhanced"
    )
    judge_model = resolve_completion_judge_model(settings, session_model)
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
