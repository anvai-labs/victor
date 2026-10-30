from __future__ import annotations

"""Degradation and recovery feedback summarization helpers."""

from collections import Counter
from typing import Any, Iterable, Mapping, Optional

from victor.evaluation.summary_mapping import as_event_mapping as _as_event_mapping
from victor.evaluation.summary_mapping import coerce_optional_float as _coerce_float
from victor.evaluation.summary_mapping import coerce_optional_int as _coerce_int
from victor.evaluation.summary_mapping import coerce_optional_text as _coerce_optional_text
from victor.evaluation.summary_mapping import extract_sequence as _extract_sequence
from victor.evaluation.summary_mapping import extract_value as _extract_value


def _normalize_event(event: Any) -> Optional[dict[str, Any]]:
    event = _as_event_mapping(event)
    if event is None:
        return None

    source = _coerce_optional_text(event.get("source"))
    kind = _coerce_optional_text(event.get("kind"))
    if source is None and kind is None:
        return None

    reasons = [
        str(reason).strip()
        for reason in list(event.get("degradation_reasons") or event.get("reasons") or [])
        if str(reason).strip()
    ]
    return {
        "source": source,
        "kind": kind,
        "failure_type": _coerce_optional_text(event.get("failure_type")),
        "provider": _coerce_optional_text(event.get("provider")),
        "model": _coerce_optional_text(event.get("model")),
        "task_type": _coerce_optional_text(event.get("task_type")),
        "pre_degraded": bool(event.get("pre_degraded", False)),
        "post_degraded": bool(event.get("post_degraded", event.get("degraded", False))),
        "recovered": bool(event.get("recovered", False)),
        "adaptation_cost": _coerce_float(event.get("adaptation_cost")),
        "time_to_recover_seconds": _coerce_float(event.get("time_to_recover_seconds")),
        "confidence": _coerce_float(event.get("confidence")),
        "iteration": _coerce_int(event.get("iteration")),
        "reasons": reasons,
    }


def _normalize_recovery_event(event: Any) -> Optional[dict[str, Any]]:
    event = _as_event_mapping(event)
    if event is None:
        return None

    source = _coerce_optional_text(event.get("source")) or "streaming_recovery"
    kind = _coerce_optional_text(event.get("kind")) or "recovery_action"
    action = _coerce_optional_text(event.get("action"))
    strategy_name = _coerce_optional_text(event.get("strategy_name"))
    reason = _coerce_optional_text(event.get("reason"))
    failure_type = _coerce_optional_text(event.get("failure_type")) or "STREAMING_RECOVERY"

    reasons = []
    for value in (action, strategy_name, reason):
        if value and value not in reasons:
            reasons.append(value)

    return {
        "source": source,
        "kind": kind,
        "failure_type": failure_type,
        "provider": _coerce_optional_text(event.get("provider")),
        "model": _coerce_optional_text(event.get("model")),
        "task_type": _coerce_optional_text(event.get("task_type")),
        "pre_degraded": bool(event.get("pre_degraded", True)),
        "post_degraded": bool(event.get("post_degraded", event.get("degraded", False))),
        "recovered": bool(event.get("recovered", False)),
        "adaptation_cost": _coerce_float(event.get("adaptation_cost", 1.0)),
        "time_to_recover_seconds": _coerce_float(event.get("time_to_recover_seconds")),
        "confidence": _coerce_float(event.get("confidence")),
        "iteration": _coerce_int(event.get("iteration")),
        "reasons": reasons,
    }


def _is_confidence_degradation(event: Mapping[str, Any]) -> bool:
    kind = _coerce_optional_text(event.get("kind")) or ""
    source = _coerce_optional_text(event.get("source")) or ""
    return kind == "confidence_early_stop" or source == "streaming_confidence"


def _is_drift_event(event: Mapping[str, Any]) -> bool:
    kind = _coerce_optional_text(event.get("kind")) or ""
    if kind in {
        "persistent_provider_degradation",
        "content_repetition",
        "confidence_early_stop",
    }:
        return True
    if bool(event.get("post_degraded")):
        return True
    reasons = {
        str(reason).strip().lower()
        for reason in list(event.get("reasons") or [])
        if str(reason).strip()
    }
    return bool(
        reasons.intersection(
            {
                "failure_streak",
                "low_recent_success_rate",
                "latency_trend",
                "content_repetition",
                "confidence_threshold_reached",
            }
        )
    )


def _is_intervention_event(event: Mapping[str, Any]) -> bool:
    kind = _coerce_optional_text(event.get("kind")) or ""
    source = _coerce_optional_text(event.get("source")) or ""
    if kind == "recovery_action" or source == "streaming_recovery":
        return True
    adaptation_cost = _coerce_float(event.get("adaptation_cost"))
    return adaptation_cost is not None and adaptation_cost > 0.0


def _variance(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / len(values)


def extract_degradation_events(value: Any) -> list[dict[str, Any]]:
    """Extract normalized degradation events from task, trace, or payload objects."""
    events: list[dict[str, Any]] = []
    for item in _extract_sequence(value, "degradation_events"):
        normalized = _normalize_event(item)
        if normalized is not None:
            events.append(normalized)
    for item in _extract_sequence(value, "recovery_events"):
        normalized = _normalize_recovery_event(item)
        if normalized is not None:
            events.append(normalized)

    for container_name in ("metadata", "failure_details", "completion_signals"):
        container = _extract_value(value, container_name)
        if isinstance(container, Mapping):
            for item in _extract_sequence(container, "degradation_events"):
                normalized = _normalize_event(item)
                if normalized is not None:
                    events.append(normalized)
            for item in _extract_sequence(container, "recovery_events"):
                normalized = _normalize_recovery_event(item)
                if normalized is not None:
                    events.append(normalized)

    trace = _extract_value(value, "trace")
    if trace is not None and trace is not value:
        events.extend(extract_degradation_events(trace))

    deduped: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for event in events:
        key = (
            event.get("source"),
            event.get("kind"),
            event.get("failure_type"),
            event.get("provider"),
            event.get("model"),
            event.get("task_type"),
            event.get("pre_degraded"),
            event.get("post_degraded"),
            event.get("recovered"),
            event.get("adaptation_cost"),
            event.get("time_to_recover_seconds"),
            event.get("iteration"),
            tuple(event.get("reasons") or []),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(event)
    return deduped


def summarize_degradation_feedback(value: Any) -> Optional[dict[str, Any]]:
    """Return a task-level degradation summary suitable for benchmark artifacts."""
    events = extract_degradation_events(value)
    if not events:
        return None

    source_counts = Counter(event["source"] for event in events if event.get("source"))
    kind_counts = Counter(event["kind"] for event in events if event.get("kind"))
    failure_types = Counter(event["failure_type"] for event in events if event.get("failure_type"))
    provider_counts = Counter(event["provider"] for event in events if event.get("provider"))
    reason_counts = Counter(
        reason for event in events for reason in list(event.get("reasons") or []) if reason
    )
    degraded_event_count = sum(
        1 for event in events if event.get("pre_degraded") or event.get("post_degraded")
    )
    recovered_event_count = sum(1 for event in events if event.get("recovered"))
    drift_event_count = sum(1 for event in events if _is_drift_event(event))
    intervention_event_count = sum(1 for event in events if _is_intervention_event(event))
    confidence_degradation_detected = any(_is_confidence_degradation(event) for event in events)
    persistent_provider_degradation_detected = any(
        event.get("kind") == "persistent_provider_degradation" for event in events
    )
    adaptation_costs = [
        float(event["adaptation_cost"])
        for event in events
        if event.get("adaptation_cost") is not None
    ]
    recovery_times = [
        float(event["time_to_recover_seconds"])
        for event in events
        if event.get("time_to_recover_seconds") is not None
    ]
    confidence_values = [
        float(event["confidence"]) for event in events if event.get("confidence") is not None
    ]
    avg_adaptation_cost = (
        round(sum(adaptation_costs) / max(1, len(adaptation_costs)), 4) if adaptation_costs else 0.0
    )
    avg_time_to_recover = (
        round(sum(recovery_times) / max(1, len(recovery_times)), 4) if recovery_times else 0.0
    )
    adaptation_cost_variance = round(_variance(adaptation_costs), 4) if adaptation_costs else 0.0
    recovery_time_variance = round(_variance(recovery_times), 4) if recovery_times else 0.0
    avg_confidence_at_degradation = (
        round(sum(confidence_values) / max(1, len(confidence_values)), 4)
        if confidence_values
        else 0.0
    )
    high_adaptation_cost_detected = any(cost >= 2.0 for cost in adaptation_costs)
    adaptation_pressure = min(1.0, avg_adaptation_cost / 4.0)
    drift_density = drift_event_count / max(1, len(events))
    drift_score = round(
        max(
            0.0,
            min(
                1.0,
                (drift_density * 0.6)
                + (adaptation_pressure * 0.2)
                + (0.1 if confidence_degradation_detected else 0.0)
                + (0.1 if persistent_provider_degradation_detected else 0.0),
            ),
        ),
        4,
    )

    return {
        "event_count": len(events),
        "degraded_event_count": degraded_event_count,
        "recovered_event_count": recovered_event_count,
        "drift_event_count": drift_event_count,
        "intervention_event_count": intervention_event_count,
        "drift_detected": drift_event_count > 0,
        "content_degradation_detected": any(
            event.get("kind") == "content_repetition" for event in events
        ),
        "confidence_degradation_detected": confidence_degradation_detected,
        "provider_degradation_detected": any(
            event.get("source") == "provider_performance" for event in events
        ),
        "persistent_provider_degradation_detected": persistent_provider_degradation_detected,
        "high_adaptation_cost_detected": high_adaptation_cost_detected,
        "avg_adaptation_cost": avg_adaptation_cost,
        "avg_time_to_recover_seconds": avg_time_to_recover,
        "adaptation_cost_variance": adaptation_cost_variance,
        "recovery_time_variance": recovery_time_variance,
        "avg_confidence_at_degradation": avg_confidence_at_degradation,
        "drift_score": drift_score,
        "sources": dict(source_counts),
        "kinds": dict(kind_counts),
        "failure_types": dict(failure_types),
        "providers": dict(provider_counts),
        "reasons": dict(reason_counts),
    }


def aggregate_degradation_feedback(
    values: Iterable[Any],
    *,
    total_tasks: Optional[int] = None,
) -> dict[str, Any]:
    """Aggregate degradation summaries across benchmark task results."""
    summaries = [summary for value in values if (summary := summarize_degradation_feedback(value))]
    if not summaries:
        return {
            "tasks_with_degradation_feedback": 0,
            "degradation_feedback_coverage": 0.0,
            "degradation_event_count": 0,
            "degraded_task_count": 0,
            "recovered_task_count": 0,
            "degradation_recovery_rate": 0.0,
            "avg_degradation_adaptation_cost": 0.0,
            "avg_degradation_time_to_recover_seconds": 0.0,
            "avg_degradation_cost_variance": 0.0,
            "avg_degradation_recovery_time_variance": 0.0,
            "avg_degradation_intervention_count": 0.0,
            "avg_degradation_confidence": 0.0,
            "avg_degradation_drift_score": 0.0,
            "content_degradation_task_count": 0,
            "confidence_degradation_task_count": 0,
            "provider_degradation_task_count": 0,
            "persistent_degradation_task_count": 0,
            "drift_task_count": 0,
            "degradation_drift_rate": 0.0,
            "degradation_intervention_task_count": 0,
            "degradation_intervention_rate": 0.0,
            "high_adaptation_cost_task_count": 0,
            "degradation_high_cost_rate": 0.0,
            "degradation_confidence_rate": 0.0,
            "degradation_stability_score": 1.0,
            "degradation_sources": {},
            "degradation_kinds": {},
            "degradation_failure_types": {},
            "degradation_providers": {},
            "degradation_reasons": {},
        }

    task_count = total_tasks if total_tasks is not None else len(summaries)
    source_counts = Counter()
    kind_counts = Counter()
    failure_types = Counter()
    providers = Counter()
    reasons = Counter()
    degraded_task_count = 0
    recovered_task_count = 0
    content_degradation_task_count = 0
    provider_degradation_task_count = 0
    confidence_degradation_task_count = 0
    persistent_degradation_task_count = 0
    drift_task_count = 0
    intervention_task_count = 0
    high_adaptation_cost_task_count = 0

    for summary in summaries:
        source_counts.update(dict(summary.get("sources") or {}))
        kind_counts.update(dict(summary.get("kinds") or {}))
        failure_types.update(dict(summary.get("failure_types") or {}))
        providers.update(dict(summary.get("providers") or {}))
        reasons.update(dict(summary.get("reasons") or {}))
        if int(summary.get("degraded_event_count", 0) or 0) > 0:
            degraded_task_count += 1
        if int(summary.get("recovered_event_count", 0) or 0) > 0:
            recovered_task_count += 1
        if bool(summary.get("content_degradation_detected")):
            content_degradation_task_count += 1
        if bool(summary.get("confidence_degradation_detected")):
            confidence_degradation_task_count += 1
        if bool(summary.get("provider_degradation_detected")):
            provider_degradation_task_count += 1
        if bool(summary.get("persistent_provider_degradation_detected")):
            persistent_degradation_task_count += 1
        if bool(summary.get("drift_detected")):
            drift_task_count += 1
        if int(summary.get("intervention_event_count", 0) or 0) > 0:
            intervention_task_count += 1
        if bool(summary.get("high_adaptation_cost_detected")):
            high_adaptation_cost_task_count += 1

    avg_adaptation_cost = round(
        sum(float(summary.get("avg_adaptation_cost", 0.0) or 0.0) for summary in summaries)
        / max(1, len(summaries)),
        4,
    )
    avg_recovery_time = round(
        sum(float(summary.get("avg_time_to_recover_seconds", 0.0) or 0.0) for summary in summaries)
        / max(1, len(summaries)),
        4,
    )
    avg_cost_variance = round(
        sum(float(summary.get("adaptation_cost_variance", 0.0) or 0.0) for summary in summaries)
        / max(1, len(summaries)),
        4,
    )
    avg_recovery_variance = round(
        sum(float(summary.get("recovery_time_variance", 0.0) or 0.0) for summary in summaries)
        / max(1, len(summaries)),
        4,
    )
    avg_intervention_count = round(
        sum(float(summary.get("intervention_event_count", 0.0) or 0.0) for summary in summaries)
        / max(1, len(summaries)),
        4,
    )
    avg_confidence = round(
        sum(
            float(summary.get("avg_confidence_at_degradation", 0.0) or 0.0) for summary in summaries
        )
        / max(1, len(summaries)),
        4,
    )
    avg_drift_score = round(
        sum(float(summary.get("drift_score", 0.0) or 0.0) for summary in summaries)
        / max(1, len(summaries)),
        4,
    )
    drift_rate = round(drift_task_count / max(1, len(summaries)), 4)
    intervention_rate = round(intervention_task_count / max(1, len(summaries)), 4)
    confidence_rate = round(confidence_degradation_task_count / max(1, len(summaries)), 4)
    high_cost_rate = round(high_adaptation_cost_task_count / max(1, len(summaries)), 4)
    stability_score = round(
        max(
            0.0,
            min(
                1.0,
                1.0
                - (drift_rate * 0.35)
                - (intervention_rate * 0.15)
                - (high_cost_rate * 0.15)
                - (min(1.0, avg_adaptation_cost / 4.0) * 0.15)
                - (min(1.0, avg_cost_variance / 4.0) * 0.1)
                - (confidence_rate * 0.05)
                + (round(recovered_task_count / max(1, degraded_task_count), 4) * 0.1),
            ),
        ),
        4,
    )

    return {
        "tasks_with_degradation_feedback": len(summaries),
        "degradation_feedback_coverage": round(len(summaries) / max(1, task_count), 4),
        "degradation_event_count": sum(
            int(summary.get("event_count", 0) or 0) for summary in summaries
        ),
        "degraded_task_count": degraded_task_count,
        "recovered_task_count": recovered_task_count,
        "degradation_recovery_rate": round(recovered_task_count / max(1, degraded_task_count), 4),
        "avg_degradation_adaptation_cost": avg_adaptation_cost,
        "avg_degradation_time_to_recover_seconds": avg_recovery_time,
        "avg_degradation_cost_variance": avg_cost_variance,
        "avg_degradation_recovery_time_variance": avg_recovery_variance,
        "avg_degradation_intervention_count": avg_intervention_count,
        "avg_degradation_confidence": avg_confidence,
        "avg_degradation_drift_score": avg_drift_score,
        "content_degradation_task_count": content_degradation_task_count,
        "confidence_degradation_task_count": confidence_degradation_task_count,
        "provider_degradation_task_count": provider_degradation_task_count,
        "persistent_degradation_task_count": persistent_degradation_task_count,
        "drift_task_count": drift_task_count,
        "degradation_drift_rate": drift_rate,
        "degradation_intervention_task_count": intervention_task_count,
        "degradation_intervention_rate": intervention_rate,
        "high_adaptation_cost_task_count": high_adaptation_cost_task_count,
        "degradation_high_cost_rate": high_cost_rate,
        "degradation_confidence_rate": confidence_rate,
        "degradation_stability_score": stability_score,
        "degradation_sources": dict(source_counts),
        "degradation_kinds": dict(kind_counts),
        "degradation_failure_types": dict(failure_types),
        "degradation_providers": dict(providers),
        "degradation_reasons": dict(reasons),
    }


__all__ = [
    "aggregate_degradation_feedback",
    "extract_degradation_events",
    "summarize_degradation_feedback",
]
