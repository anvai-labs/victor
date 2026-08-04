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


# FEP-0030 completion-judge backends. Phase 1 shipped the seam ("session-model",
# "enhanced"); Phase 2 wires the decoupled "llm:<model>@<endpoint>" backend (a
# calibrated LLM judge on a side endpoint, independent of the session model).
# "classifier:<path>" (resident ModernBERT) is Phase 2b — until then it behaves
# as "session-model" with a note, so config written ahead never breaks a session.
_JUDGE_BACKEND_ENV = "VICTOR_COMPLETION_JUDGE"
_warned_backends: set[str] = set()


def _configured_backend(settings: Any) -> str:
    backend = os.environ.get(_JUDGE_BACKEND_ENV) or getattr(
        getattr(settings, "agent", None), "completion_judge", "session-model"
    )
    # Non-string (unset, mocked settings, misconfig) → the safe default. Never
    # let a stray value reach parse_completion_judge / provider construction.
    if not isinstance(backend, str) or not backend.strip():
        return "session-model"
    return backend.strip()


def parse_completion_judge(backend: str) -> tuple[str, Optional[str], Optional[str]]:
    """Parse an ``agent.completion_judge`` value into ``(kind, model, endpoint)``.

    - ``"session-model"`` → ``("session-model", None, None)``
    - ``"enhanced"`` → ``("enhanced", None, None)``
    - ``"llm:<model>[@<base_url>]"`` → ``("llm", model, base_url|None)``. The
      model may itself contain colons (``llama3.3:70b``); the endpoint is
      everything after the LAST ``@``.
    - ``"classifier:<path>"`` → ``("classifier", path, None)``
    - anything else → ``("unknown", None, None)``
    """
    if backend in ("session-model", "enhanced"):
        return backend, None, None
    if backend.startswith("llm:"):
        rest = backend[len("llm:") :]
        model, sep, endpoint = rest.rpartition("@")
        if not sep:  # no '@' → whole remainder is the model, default endpoint
            return "llm", rest or None, None
        return "llm", model or None, endpoint or None
    if backend.startswith("classifier:"):
        return "classifier", backend[len("classifier:") :] or None, None
    return "unknown", None, None


def resolve_completion_judge_model(settings: Any, session_model: Optional[str]) -> Optional[str]:
    """Resolve the judge model the calibration pin should check (FEP-0030 seam).

    Independent of the session model by design. Returns the model name the pin
    checks, or ``None`` to force the algorithmic ``enhanced`` path.

    - ``"session-model"`` (default): the session's own model (historical behavior).
    - ``"enhanced"``: ``None`` — rubric/hybrid downgrade to enhanced.
    - ``"llm:<model>@<endpoint>"``: the decoupled judge *model* — so a session on
      an uncalibrated chat model can still gate rubric with a calibrated side
      judge (e.g. ``llm:llama3.3:70b@http://host:11434``).
    - ``"classifier:<path>"``: Phase 2b; treated as session-model with a note.
    """
    backend = _configured_backend(settings)
    kind, model, _endpoint = parse_completion_judge(backend)

    if kind == "session-model":
        return session_model
    if kind == "enhanced":
        return None
    if kind == "llm":
        return model
    if kind == "classifier":
        if backend not in _warned_backends:
            _warned_backends.add(backend)
            logger.info(
                "agent.completion_judge=%r (classifier) is FEP-0030 Phase 2b, not yet wired; "
                "using the session model as the judge for now.",
                backend,
            )
        return session_model
    if backend not in _warned_backends:
        _warned_backends.add(backend)
        logger.warning(
            "agent.completion_judge=%r is not recognized; using the session model as the judge. "
            "Valid: 'session-model' (default), 'enhanced', 'llm:<model>@<endpoint>'.",
            backend,
        )
    return session_model


def build_judge_complete_fn(settings: Any, provider_context: Any) -> Optional[Any]:
    """Build the async ``complete_fn(prompt)->text`` for the resolved LLM judge backend.

    Centralizes judge construction (FEP-0030) so the capped ``turn_execution_runtime``
    hotspot only delegates. Returns ``None`` when no LLM judge applies (``enhanced``,
    or no provider) — the loop then uses the heuristic/algorithmic path.

    - ``session-model`` / ``classifier`` (Phase 2b) / unknown: the session provider+model.
    - ``llm:<model>@<endpoint>``: a side Ollama provider at ``<endpoint>`` (default
      the provider's own base URL) with ``<model>`` — decoupled from the session.
    """
    from victor.providers.base import Message

    backend = _configured_backend(settings)
    kind, model, endpoint = parse_completion_judge(backend)

    if kind == "enhanced":
        return None

    if kind == "llm" and model:
        from victor.providers.registry import ProviderRegistry

        kwargs = {"base_url": endpoint} if endpoint else {}
        try:
            provider = ProviderRegistry.create("ollama", **kwargs)
        except Exception as exc:  # unavailable side endpoint → fall back to heuristic
            logger.warning(
                "completion judge backend %r could not create its provider (%s); "
                "falling back to the heuristic judge.",
                backend,
                exc,
            )
            return None
        judge_model = model
    else:
        # session-model, classifier (Phase 2b forward-compat), unknown.
        provider = getattr(provider_context, "provider", None)
        judge_model = getattr(provider_context, "model", None)
        if provider is None:
            return None

    async def complete(prompt: str) -> str:
        resp = await provider.chat(
            [Message(role="user", content=prompt)],
            model=judge_model,
            temperature=0.0,
            max_tokens=400,
        )
        return getattr(resp, "content", "") or ""

    return complete


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
