# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Topology-feedback summarization helpers for benchmark and runtime evaluation."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping, Optional

_TOPOLOGY_OVERHEAD = {
    "direct": 0.04,
    "single_agent": 0.10,
    "parallel_exploration": 0.18,
    "team": 0.22,
    "escalated_single_agent": 0.16,
    "safe_stop": 0.08,
}
_FORMATION_OVERHEAD = {
    "parallel": 0.03,
    "hierarchical": 0.04,
    "mesh": 0.06,
}
_SUCCESS_STATUSES = {"passed", "complete", "completed", "resolved", "success"}
_FAILURE_STATUSES = {"failed", "error", "timeout", "cancelled"}


from victor.core.utils import clamp as _clamp


def _coerce_optional_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(getattr(value, "value", value)).strip()
    return text or None


def _coerce_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _extract_value(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _extract_mapping(value: Any, key: str) -> dict[str, Any]:
    raw_value = _extract_value(value, key, {})
    return dict(raw_value) if isinstance(raw_value, Mapping) else {}


def _extract_sequence(value: Any, key: str) -> list[Any]:
    raw_value = _extract_value(value, key, [])
    if raw_value is None:
        return []
    if isinstance(raw_value, (list, tuple)):
        return list(raw_value)
    return []


def _normalize_event(event: Any) -> Optional[dict[str, Any]]:
    if event is None:
        return None
    if hasattr(event, "to_dict"):
        event = event.to_dict()
    if not isinstance(event, Mapping):
        return None
    action = _coerce_optional_text(event.get("action"))
    topology = _coerce_optional_text(event.get("topology"))
    if action is None and topology is None:
        return None
    return {
        "action": action,
        "topology": topology,
        "execution_mode": _coerce_optional_text(event.get("execution_mode")),
        "provider": _coerce_optional_text(event.get("provider")),
        "formation": _coerce_optional_text(event.get("formation")),
        "selection_policy": _coerce_optional_text(event.get("selection_policy"))
        or _coerce_optional_text(_extract_mapping(event, "telemetry_tags").get("selection_policy")),
        "fallback_action": _coerce_optional_text(event.get("fallback_action")),
        "confidence": _coerce_float(event.get("confidence")) or 0.0,
        "outcome": (
            dict(event.get("outcome") or {}) if isinstance(event.get("outcome"), Mapping) else {}
        ),
    }


def _extract_topology_summary_mapping(value: Any) -> Optional[dict[str, Any]]:
    if isinstance(value, Mapping):
        summary = value.get("topology_summary")
        if isinstance(summary, Mapping):
            return dict(summary)
    else:
        summary = getattr(value, "topology_summary", None)
        if isinstance(summary, Mapping):
            return dict(summary)
    for container_name in ("metadata", "failure_details", "completion_signals"):
        container = _extract_value(value, container_name)
        if isinstance(container, Mapping):
            summary = container.get("topology_summary")
            if isinstance(summary, Mapping):
                return dict(summary)
    trace = _extract_value(value, "trace")
    if trace is not None and trace is not value:
        return _extract_topology_summary_mapping(trace)
    return None


def _extract_optimization_summary_mapping(value: Any) -> Optional[dict[str, Any]]:
    if isinstance(value, Mapping):
        summary = value.get("optimization_summary")
        if isinstance(summary, Mapping):
            return dict(summary)
    else:
        summary = getattr(value, "optimization_summary", None)
        if isinstance(summary, Mapping):
            return dict(summary)
    for container_name in ("metadata", "failure_details", "completion_signals"):
        container = _extract_value(value, container_name)
        if isinstance(container, Mapping):
            summary = container.get("optimization_summary")
            if isinstance(summary, Mapping):
                return dict(summary)
    trace = _extract_value(value, "trace")
    if trace is not None and trace is not value:
        return _extract_optimization_summary_mapping(trace)
    return None


def extract_topology_events(value: Any) -> list[dict[str, Any]]:
    """Extract normalized topology events from task, trace, or payload objects."""
    events: list[dict[str, Any]] = []
    for item in _extract_sequence(value, "topology_events"):
        normalized = _normalize_event(item)
        if normalized is not None:
            events.append(normalized)

    for container_name in ("metadata", "failure_details", "completion_signals"):
        container = _extract_value(value, container_name)
        if isinstance(container, Mapping):
            for item in _extract_sequence(container, "topology_events"):
                normalized = _normalize_event(item)
                if normalized is not None:
                    events.append(normalized)

    trace = _extract_value(value, "trace")
    if trace is not None and trace is not value:
        events.extend(extract_topology_events(trace))

    deduped: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for event in events:
        key = (
            event.get("action"),
            event.get("topology"),
            event.get("execution_mode"),
            event.get("provider"),
            event.get("formation"),
            event.get("selection_policy"),
            event.get("fallback_action"),
            event.get("confidence"),
            tuple(sorted((event.get("outcome") or {}).items())),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(event)
    return deduped


def summarize_topology_feedback(value: Any) -> Optional[dict[str, Any]]:
    """Return a task-level topology summary suitable for benchmark artifacts."""
    existing_summary = _extract_topology_summary_mapping(value)
    if existing_summary:
        return existing_summary

    events = extract_topology_events(value)
    if not events:
        return None

    status = (_coerce_optional_text(_extract_value(value, "status")) or "").lower()
    completion_score = _coerce_float(_extract_value(value, "completion_score"))
    if completion_score is None:
        completion_score = _coerce_float(_extract_value(value, "overall_score"))
    tool_calls = _coerce_int(_extract_value(value, "tool_calls"))
    turns = _coerce_int(_extract_value(value, "turns"))

    trace = _extract_value(value, "trace")
    if trace is not None:
        tool_calls = (
            tool_calls
            if tool_calls is not None
            else _coerce_int(_extract_value(trace, "total_tool_calls"))
        )
        turns = turns if turns is not None else _coerce_int(_extract_value(trace, "turns"))

    if completion_score is None:
        if status in _SUCCESS_STATUSES:
            completion_score = 1.0
        elif status in _FAILURE_STATUSES:
            completion_score = 0.0
        else:
            completion_score = 0.5

    first_event = events[0]
    last_event = events[-1]
    action_counts = Counter(event["action"] for event in events if event.get("action"))
    topology_counts = Counter(event["topology"] for event in events if event.get("topology"))
    execution_counts = Counter(
        event["execution_mode"] for event in events if event.get("execution_mode")
    )
    provider_counts = Counter(event["provider"] for event in events if event.get("provider"))
    formation_counts = Counter(event["formation"] for event in events if event.get("formation"))
    selection_policy_counts = Counter(
        event["selection_policy"] for event in events if event.get("selection_policy")
    )

    average_confidence = sum(event["confidence"] for event in events) / len(events)
    max_confidence = max(event["confidence"] for event in events)
    fallback_count = sum(1 for event in events if event.get("fallback_action"))

    dominant_topology = last_event.get("topology") or first_event.get("topology") or "single_agent"
    dominant_formation = last_event.get("formation") or first_event.get("formation")
    base_overhead = _TOPOLOGY_OVERHEAD.get(dominant_topology, 0.12)
    formation_overhead = _FORMATION_OVERHEAD.get(dominant_formation or "", 0.0)
    event_overhead = min(0.12, max(0, len(events) - 1) * 0.04)
    tool_overhead = min(0.12, ((tool_calls or 0) / 30.0) * 0.12)
    turn_overhead = min(0.10, ((turns or 0) / 10.0) * 0.10)
    coordination_overhead = _clamp(
        base_overhead + formation_overhead + event_overhead + tool_overhead + turn_overhead,
        0.0,
        0.95,
    )

    calibration_penalty = 0.0
    if completion_score < 0.5 and average_confidence > 0.65:
        calibration_penalty = min(0.25, (average_confidence - 0.65) * 0.5)
    elif completion_score > 0.8 and average_confidence < 0.4:
        calibration_penalty = min(0.08, (0.4 - average_confidence) * 0.2)

    efficiency_component = 1.0 - coordination_overhead
    topology_reward = _clamp(
        (0.7 * completion_score) + (0.3 * efficiency_component) - calibration_penalty,
        0.0,
        1.0,
    )

    return {
        "event_count": len(events),
        "decision_transitions": max(0, len(events) - 1),
        "selected_action": first_event.get("action"),
        "final_action": last_event.get("action"),
        "selected_topology": first_event.get("topology"),
        "final_topology": last_event.get("topology"),
        "selected_execution_mode": first_event.get("execution_mode"),
        "final_execution_mode": last_event.get("execution_mode"),
        "selected_provider": first_event.get("provider"),
        "final_provider": last_event.get("provider"),
        "selected_formation": first_event.get("formation"),
        "final_formation": last_event.get("formation"),
        "selected_selection_policy": first_event.get("selection_policy"),
        "final_selection_policy": last_event.get("selection_policy"),
        "avg_confidence": round(average_confidence, 4),
        "max_confidence": round(max_confidence, 4),
        "fallback_count": fallback_count,
        "fallback_rate": round(fallback_count / len(events), 4),
        "coordination_overhead": round(coordination_overhead, 4),
        "quality_component": round(completion_score, 4),
        "efficiency_component": round(efficiency_component, 4),
        "calibration_penalty": round(calibration_penalty, 4),
        "topology_reward": round(topology_reward, 4),
        "status": status or None,
        "tool_calls": tool_calls,
        "turns": turns,
        "actions": dict(action_counts),
        "topologies": dict(topology_counts),
        "execution_modes": dict(execution_counts),
        "providers": dict(provider_counts),
        "formations": dict(formation_counts),
        "selection_policies": dict(selection_policy_counts),
    }


def summarize_optimization_feedback(value: Any) -> Optional[dict[str, Any]]:
    """Return a task-level optimization summary when present on task artifacts."""
    existing_summary = _extract_optimization_summary_mapping(value)
    if not existing_summary:
        return None

    feasible = bool(existing_summary.get("feasible", False))
    reward = _coerce_float(existing_summary.get("reward"))
    if reward is None:
        return None

    reward_components = {
        str(key): float(component_value)
        for key, component_value in dict(existing_summary.get("reward_components") or {}).items()
        if _coerce_float(component_value) is not None
    }
    feasibility_failures = [
        str(failure)
        for failure in list(existing_summary.get("feasibility_failures") or [])
        if str(failure).strip()
    ]
    return {
        "feasible": feasible,
        "reward": round(reward, 4),
        "reward_components": reward_components,
        "feasibility_failures": feasibility_failures,
    }


def aggregate_topology_feedback(
    values: Iterable[Any],
    *,
    total_tasks: Optional[int] = None,
) -> dict[str, Any]:
    """Aggregate topology summaries across benchmark task results."""
    entries = [
        (summary, summarize_optimization_feedback(value))
        for value in values
        if (summary := summarize_topology_feedback(value))
    ]
    summaries = [summary for summary, _ in entries]
    if not entries:
        count = total_tasks or 0
        return {
            "tasks_with_topology_feedback": 0,
            "topology_feedback_coverage": 0.0 if count else 0.0,
            "avg_topology_reward": 0.0,
            "avg_topology_confidence": 0.0,
            "avg_coordination_overhead": 0.0,
            "topology_fallback_rate": 0.0,
            "topology_actions": {},
            "topology_final_actions": {},
            "topology_kinds": {},
            "topology_final_kinds": {},
            "topology_execution_modes": {},
            "topology_providers": {},
            "topology_formations": {},
            "topology_selection_policies": {},
            "topology_selection_policy_reward_totals": {},
            "avg_topology_reward_by_selection_policy": {},
            "topology_learned_override_reward_delta": None,
            "topology_selection_policy_optimization_counts": {},
            "topology_selection_policy_optimization_reward_totals": {},
            "avg_topology_optimization_reward_by_selection_policy": {},
            "topology_selection_policy_feasible_counts": {},
            "topology_selection_policy_feasibility_rates": {},
            "topology_learned_override_optimization_reward_delta": None,
            "topology_learned_override_feasibility_delta": None,
        }

    selected_actions = Counter(
        summary["selected_action"] for summary in summaries if summary.get("selected_action")
    )
    final_actions = Counter(
        summary["final_action"] for summary in summaries if summary.get("final_action")
    )
    selected_topologies = Counter(
        summary["selected_topology"] for summary in summaries if summary.get("selected_topology")
    )
    final_topologies = Counter(
        summary["final_topology"] for summary in summaries if summary.get("final_topology")
    )
    execution_modes = Counter(
        summary["final_execution_mode"]
        for summary in summaries
        if summary.get("final_execution_mode")
    )
    providers = Counter(
        summary["final_provider"] for summary in summaries if summary.get("final_provider")
    )
    formations = Counter(
        summary["final_formation"] for summary in summaries if summary.get("final_formation")
    )
    selection_policies = Counter(
        summary["final_selection_policy"]
        for summary in summaries
        if summary.get("final_selection_policy")
    )
    task_count = total_tasks if total_tasks is not None else len(summaries)
    selection_policy_reward_totals: dict[str, float] = {}
    selection_policy_optimization_counts: dict[str, int] = {}
    selection_policy_optimization_reward_totals: dict[str, float] = {}
    selection_policy_feasible_counts: dict[str, int] = {}
    for summary in summaries:
        selection_policy = summary.get("final_selection_policy")
        if not selection_policy:
            continue
        selection_policy_reward_totals[selection_policy] = round(
            selection_policy_reward_totals.get(selection_policy, 0.0) + summary["topology_reward"],
            4,
        )
    for summary, optimization_summary in entries:
        selection_policy = summary.get("final_selection_policy")
        if not selection_policy or optimization_summary is None:
            continue
        selection_policy_optimization_counts[selection_policy] = (
            selection_policy_optimization_counts.get(selection_policy, 0) + 1
        )
        selection_policy_optimization_reward_totals[selection_policy] = round(
            selection_policy_optimization_reward_totals.get(selection_policy, 0.0)
            + float(optimization_summary["reward"]),
            4,
        )
        if optimization_summary.get("feasible"):
            selection_policy_feasible_counts[selection_policy] = (
                selection_policy_feasible_counts.get(selection_policy, 0) + 1
            )
    avg_reward_by_selection_policy = {
        policy: round(
            selection_policy_reward_totals[policy] / max(1, count),
            4,
        )
        for policy, count in selection_policies.items()
        if count > 0 and policy in selection_policy_reward_totals
    }
    avg_optimization_reward_by_selection_policy = {
        policy: round(
            selection_policy_optimization_reward_totals[policy]
            / max(1, selection_policy_optimization_counts[policy]),
            4,
        )
        for policy in selection_policy_optimization_reward_totals
        if selection_policy_optimization_counts.get(policy, 0) > 0
    }
    selection_policy_feasibility_rates = {
        policy: round(
            selection_policy_feasible_counts.get(policy, 0)
            / max(1, selection_policy_optimization_counts[policy]),
            4,
        )
        for policy in selection_policy_optimization_counts
        if selection_policy_optimization_counts[policy] > 0
    }
    learned_override_reward_delta: Optional[float] = None
    if (
        "learned_close_override" in avg_reward_by_selection_policy
        and "heuristic" in avg_reward_by_selection_policy
    ):
        learned_override_reward_delta = round(
            avg_reward_by_selection_policy["learned_close_override"]
            - avg_reward_by_selection_policy["heuristic"],
            4,
        )
    learned_override_optimization_reward_delta: Optional[float] = None
    if (
        "learned_close_override" in avg_optimization_reward_by_selection_policy
        and "heuristic" in avg_optimization_reward_by_selection_policy
    ):
        learned_override_optimization_reward_delta = round(
            avg_optimization_reward_by_selection_policy["learned_close_override"]
            - avg_optimization_reward_by_selection_policy["heuristic"],
            4,
        )
    learned_override_feasibility_delta: Optional[float] = None
    if (
        "learned_close_override" in selection_policy_feasibility_rates
        and "heuristic" in selection_policy_feasibility_rates
    ):
        learned_override_feasibility_delta = round(
            selection_policy_feasibility_rates["learned_close_override"]
            - selection_policy_feasibility_rates["heuristic"],
            4,
        )

    return {
        "tasks_with_topology_feedback": len(summaries),
        "topology_feedback_coverage": round(len(summaries) / max(1, task_count), 4),
        "avg_topology_reward": round(
            sum(summary["topology_reward"] for summary in summaries) / len(summaries), 4
        ),
        "avg_topology_confidence": round(
            sum(summary["avg_confidence"] for summary in summaries) / len(summaries), 4
        ),
        "avg_coordination_overhead": round(
            sum(summary["coordination_overhead"] for summary in summaries) / len(summaries),
            4,
        ),
        "topology_fallback_rate": round(
            sum(summary["fallback_rate"] for summary in summaries) / len(summaries), 4
        ),
        "topology_actions": dict(selected_actions),
        "topology_final_actions": dict(final_actions),
        "topology_kinds": dict(selected_topologies),
        "topology_final_kinds": dict(final_topologies),
        "topology_execution_modes": dict(execution_modes),
        "topology_providers": dict(providers),
        "topology_formations": dict(formations),
        "topology_selection_policies": dict(selection_policies),
        "topology_selection_policy_reward_totals": dict(selection_policy_reward_totals),
        "avg_topology_reward_by_selection_policy": dict(avg_reward_by_selection_policy),
        "topology_learned_override_reward_delta": learned_override_reward_delta,
        "topology_selection_policy_optimization_counts": dict(selection_policy_optimization_counts),
        "topology_selection_policy_optimization_reward_totals": dict(
            selection_policy_optimization_reward_totals
        ),
        "avg_topology_optimization_reward_by_selection_policy": dict(
            avg_optimization_reward_by_selection_policy
        ),
        "topology_selection_policy_feasible_counts": dict(selection_policy_feasible_counts),
        "topology_selection_policy_feasibility_rates": dict(selection_policy_feasibility_rates),
        "topology_learned_override_optimization_reward_delta": (
            learned_override_optimization_reward_delta
        ),
        "topology_learned_override_feasibility_delta": learned_override_feasibility_delta,
    }


__all__ = [
    "aggregate_topology_feedback",
    "extract_topology_events",
    "summarize_optimization_feedback",
    "summarize_topology_feedback",
]
