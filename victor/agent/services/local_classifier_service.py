# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License").

"""Shipped local-classifier decision service (FEP-0012 Phase 5).

A drop-in alternative to the Ollama edge model: implements
``LLMDecisionServiceProtocol`` and serves micro-decisions from a bundled
classic-ML artifact (``victor.ml.model.EdgeClassifierModel``) via pure-numpy
inference — no Ollama, no server, no new runtime dependency.

Behavior:
- **Heuristic fast-path:** if the caller's heuristic confidence is already high,
  return it unchanged (same contract as the edge model).
- **Classify** the DecisionTypes this service owns (those with a head in the
  artifact AND a label->Pydantic mapper). Map the predicted label to the typed
  decision object.
- **Defer** to the heuristic when: no artifact is loaded, the DecisionType has
  no head or no mapper, the classifier is unconfident (below τ), or the label
  does not map to a valid enum value. Deferring keeps the system correct even
  for types/inputs the classifier doesn't yet cover.

The service is selected via the ``decision_backend`` enum (FEP-0012):
``local_classifier`` forces it; ``auto`` (default) adopts it when a healthy
artifact is present, else falls back to the legacy edge/LLM selection. See
``victor/core/bootstrap_services.py``.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from victor.agent.decisions.schemas import (
    DecisionType,
    IntentDecision,
    IntentType,
    TaskCompletionDecision,
    TaskTypeDecision,
    TaskCategoryType,
    ToolNecessityDecision,
)
from victor.agent.services.protocols.decision_service import (
    DecisionMetrics,
    DecisionResult,
)

logger = logging.getLogger(__name__)

# Bundled artifact location (relative to the package root). Resolved lazily.
_DEFAULT_ARTIFACT = (
    Path(__file__).resolve().parents[3] / "victor" / "models" / "edge_classifier_v1.npz"
)

# Heuristic confidence at/above which we skip classification entirely.
_DEFAULT_HEURISTIC_THRESHOLD = 0.6

# Per-project RL delta cache TTL (FEP-0012 Phase 6). The delta table is tiny
# and top-K bounded; reload at most this often so predict stays sub-ms.
_DELTA_CACHE_TTL_S = 60.0


# --------------------------------------------------------------------------
# label -> typed decision object mappers
# --------------------------------------------------------------------------


def _map_enum(
    label: str, conf: float, model_cls: type, field: str, enum_cls: type
) -> Optional[Any]:
    """Map a label to a Pydantic decision with an enum field (by value then name)."""
    try:
        value = enum_cls(label)  # match by value
    except ValueError:
        try:
            value = enum_cls[label]  # match by member name
        except KeyError:
            return None  # label not a valid enum member -> defer
    return model_cls(**{field: value, "confidence": float(conf)})


def _map_bool(
    label: str, conf: float, model_cls: type, field: str, true_labels: Tuple[str, ...]
) -> Any:
    value = label.strip().lower() in true_labels
    return model_cls(**{field: value, "confidence": float(conf)})


# task_completion head labels ARE reward buckets (pass/partial/fail), so a plain
# bool map is wrong: a "pass" prediction means "this looks like a successful
# completion" -> is_complete=True; "fail" -> False; "partial"/unknown -> defer
# (return None so the heuristic path handles it). Also accepts the legacy
# complete/incomplete convention for forward-compat with a differently-trained head.
_COMPLETION_TRUE_LABELS = ("pass", "complete", "true", "yes", "done", "finished")
_COMPLETION_FALSE_LABELS = ("fail", "incomplete", "false", "no", "unfinished")
_COMPLETION_DEFER_LABELS = ("partial",)


def _map_completion(label: str, conf: float) -> Optional[Any]:
    norm = label.strip().lower()
    if norm in _COMPLETION_DEFER_LABELS:
        return None
    if norm in _COMPLETION_TRUE_LABELS:
        return TaskCompletionDecision(is_complete=True, confidence=float(conf))
    if norm in _COMPLETION_FALSE_LABELS:
        return TaskCompletionDecision(is_complete=False, confidence=float(conf))
    return None  # unknown label -> defer to heuristic


# Per-DecisionType mapper: (label, confidence) -> Pydantic decision object (or None to defer).
_MAPPERS: Dict[DecisionType, Callable[[str, float], Optional[Any]]] = {
    DecisionType.TASK_TYPE_CLASSIFICATION: lambda label, conf: _map_enum(
        label, conf, TaskTypeDecision, "task_type", TaskCategoryType
    ),
    DecisionType.TASK_COMPLETION: _map_completion,
    DecisionType.TOOL_NECESSITY: lambda label, conf: _map_bool(
        label,
        conf,
        ToolNecessityDecision,
        "requires_tools",
        ("requires_tools", "true", "yes"),
    ),
    DecisionType.INTENT_CLASSIFICATION: lambda label, conf: _map_enum(
        label, conf, IntentDecision, "intent", IntentType
    ),
}

# Which context field(s) to featurize per DecisionType (first non-empty wins).
_TEXT_FIELDS: Dict[DecisionType, Tuple[str, ...]] = {
    DecisionType.TASK_TYPE_CLASSIFICATION: ("message_excerpt", "message", "query"),
    DecisionType.TASK_COMPLETION: ("response_tail", "content_tail"),
    DecisionType.TOOL_NECESSITY: ("message", "query", "text"),
    DecisionType.INTENT_CLASSIFICATION: ("text_tail", "response_tail"),
}


def _extract_text(decision_type: DecisionType, context: Dict[str, Any]) -> str:
    for field in _TEXT_FIELDS.get(decision_type, ()):
        value = context.get(field)
        if value:
            return str(value)
    # Fallback: join the string-valued context entries.
    return " ".join(str(v) for v in context.values() if isinstance(v, str))


class LocalClassifierDecisionService:
    """Decision service backed by the shipped edge-classifier artifact."""

    def __init__(
        self,
        model: Optional[Any] = None,
        *,
        heuristic_threshold: float = _DEFAULT_HEURISTIC_THRESHOLD,
    ) -> None:
        self._model = model
        self._heuristic_threshold = heuristic_threshold
        self._metrics = DecisionMetrics()
        # FEP-0012 Phase 6: per-project RL delta, cached per decision type with
        # a TTL so a fresh project DB read doesn't happen on every micro-decision.
        # {decision_type: (loaded_at_perf, delta_dict)}; None entry = disabled.
        self._delta_cache: Dict[str, Tuple[float, Optional[Dict[int, Any]]]] = {}

    # --------------------------------------------------------------- factory
    @classmethod
    def from_artifact(
        cls, artifact_path: Optional[str] = None, **kwargs: Any
    ) -> "LocalClassifierDecisionService":
        """Build a service, loading the artifact if present (else defers all)."""
        path = Path(artifact_path) if artifact_path else _DEFAULT_ARTIFACT
        env_override = os.environ.get("VICTOR_EDGE_CLASSIFIER_PATH")
        if env_override:
            path = Path(env_override)
        model = None
        if path.exists():
            try:
                from victor.ml.model import EdgeClassifierModel

                model = EdgeClassifierModel.load(str(path))
                logger.info(
                    "LocalClassifierDecisionService loaded artifact: %s (heads=%s)",
                    path,
                    sorted(model.heads),
                )
            except Exception as e:  # corrupt / wrong spec version
                logger.warning("Edge-classifier artifact load failed (%s): %s", path, e)
        else:
            logger.debug("No edge-classifier artifact at %s; service will defer all", path)
        return cls(model=model, **kwargs)

    # --------------------------------------------------------- protocol impl
    @property
    def budget_remaining(self) -> int:
        return 10_000  # local inference is effectively unbounded

    def reset_budget(self) -> None:
        """No per-turn budget for local inference (no-op)."""

    def is_healthy(self) -> bool:
        return self._model is not None

    def get_metrics(self) -> DecisionMetrics:
        return self._metrics

    async def decide(
        self,
        decision_type: DecisionType,
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
        decision_type: DecisionType,
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

    def decide_sync(
        self,
        decision_type: DecisionType,
        context: Dict[str, Any],
        *,
        heuristic_result: Any = None,
        heuristic_confidence: float = 0.0,
    ) -> DecisionResult:
        start = time.perf_counter()
        self._metrics.total_calls += 1

        # Fast path: heuristic already confident enough.
        if heuristic_confidence >= self._heuristic_threshold:
            return self._result(
                decision_type,
                heuristic_result,
                "heuristic",
                heuristic_confidence,
                start,
            )

        # No artifact / unsupported type -> defer to heuristic.
        mapper = _MAPPERS.get(decision_type)
        if self._model is None or mapper is None:
            return self._result(
                decision_type,
                heuristic_result,
                "heuristic",
                heuristic_confidence,
                start,
            )

        text = _extract_text(decision_type, context)
        head = self._model.heads.get(decision_type.value)
        delta = (
            self._load_delta_cached(decision_type.value, head.labels) if head is not None else None
        )
        label, confidence = self._model.predict(decision_type.value, text, delta=delta)
        if label is None:
            # Unconfident (below τ) -> defer to heuristic.
            return self._result(
                decision_type,
                heuristic_result,
                "heuristic",
                heuristic_confidence,
                start,
            )

        decision = mapper(label, confidence)
        if decision is None:
            # Predicted label doesn't map to a valid enum value -> defer.
            return self._result(
                decision_type,
                heuristic_result,
                "heuristic",
                heuristic_confidence,
                start,
            )

        self._metrics.llm_calls += 1  # reuse field as "classifier decisions served"
        return self._result(decision_type, decision, "local_classifier", confidence, start)

    def _result(
        self,
        decision_type: DecisionType,
        result: Any,
        source: str,
        confidence: float,
        start: float,
    ) -> DecisionResult:
        latency_ms = (time.perf_counter() - start) * 1000
        self._metrics._latency_sum += latency_ms  # type: ignore[attr-defined]
        n = max(self._metrics.total_calls, 1)
        self._metrics.avg_latency_ms = self._metrics._latency_sum / n  # type: ignore[attr-defined]
        return DecisionResult(
            decision_type=decision_type,
            result=result,
            source=source,
            confidence=confidence,
            latency_ms=latency_ms,
        )

    # ----------------------------------------------- FEP-0012 Phase 6 delta
    def _load_delta_cached(self, decision_type: str, labels: List[str]) -> Optional[Dict[int, Any]]:
        """Return the per-project RL delta for ``decision_type`` (cached, gated).

        Returns ``None`` (→ no blend) when local learning is disabled, the
        artifact has no head for this type, or the project DB is empty/unreadable.
        The delta is reloaded at most every ``_DELTA_CACHE_TTL_S`` seconds.
        """
        if not labels:
            return None
        try:
            from victor.config.decision_settings import DecisionServiceSettings

            if not DecisionServiceSettings().local_learning_enabled:
                return None
        except Exception:
            return None

        now = time.perf_counter()
        cached = self._delta_cache.get(decision_type)
        if cached is not None and (now - cached[0]) < _DELTA_CACHE_TTL_S:
            return cached[1]

        delta: Optional[Dict[int, Any]] = None
        try:
            from victor.agent.decisions.local_delta import load_delta

            loaded = load_delta(decision_type, list(labels))
            delta = loaded if loaded else None
        except Exception as exc:  # degrade gracefully — never block a decision
            logger.debug("delta load failed for %s: %s", decision_type, exc)
        self._delta_cache[decision_type] = (now, delta)
        return delta


def create_local_classifier_decision_service(
    artifact_path: Optional[str] = None,
) -> Optional[LocalClassifierDecisionService]:
    """Factory used by bootstrap. Returns a service (always); health = artifact loaded."""
    try:
        return LocalClassifierDecisionService.from_artifact(artifact_path)
    except Exception as e:
        logger.debug("Local classifier decision service unavailable: %s", e)
        return None
