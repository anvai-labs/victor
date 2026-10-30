from __future__ import annotations

"""Planning-feedback summarization helpers for benchmark and runtime evaluation."""

from collections import Counter
from typing import Any, Iterable, Mapping, Optional

from victor.evaluation.summary_mapping import as_event_mapping as _as_event_mapping
from victor.evaluation.summary_mapping import coerce_optional_float as _coerce_float
from victor.evaluation.summary_mapping import coerce_optional_text as _coerce_optional_text
from victor.evaluation.summary_mapping import extract_sequence as _extract_sequence
from victor.evaluation.summary_mapping import extract_value as _extract_value

_SUCCESS_STATUSES = {"passed", "complete", "completed", "resolved", "success"}
_FAILURE_STATUSES = {"failed", "error", "timeout", "cancelled"}


def _normalize_event(event: Any) -> Optional[dict[str, Any]]:
    event = _as_event_mapping(event)
    if event is None:
        return None

    selection_policy = _coerce_optional_text(event.get("selection_policy"))
    used_llm_planning_raw = event.get("used_llm_planning")
    if selection_policy is None and used_llm_planning_raw is None:
        return None

    constraint_tags = [
        str(tag).strip() for tag in list(event.get("constraint_tags") or []) if str(tag).strip()
    ]
    return {
        "selection_policy": selection_policy,
        "used_llm_planning": (
            bool(used_llm_planning_raw) if used_llm_planning_raw is not None else None
        ),
        "task_type": _coerce_optional_text(event.get("task_type")),
        "skip_reason": _coerce_optional_text(event.get("skip_reason")),
        "force_reason": _coerce_optional_text(event.get("force_reason")),
        "forced_by_runtime_feedback": bool(event.get("forced_by_runtime_feedback", False)),
        "constraint_tags": constraint_tags,
        "experiment_support": _coerce_float(event.get("experiment_support")) or 0.0,
    }


def extract_planning_events(value: Any) -> list[dict[str, Any]]:
    """Extract normalized planning events from task, trace, or payload objects."""
    events: list[dict[str, Any]] = []
    for item in _extract_sequence(value, "planning_events"):
        normalized = _normalize_event(item)
        if normalized is not None:
            events.append(normalized)

    for container_name in ("metadata", "failure_details", "completion_signals"):
        container = _extract_value(value, container_name)
        if isinstance(container, Mapping):
            for item in _extract_sequence(container, "planning_events"):
                normalized = _normalize_event(item)
                if normalized is not None:
                    events.append(normalized)

    trace = _extract_value(value, "trace")
    if trace is not None and trace is not value:
        events.extend(extract_planning_events(trace))

    deduped: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for event in events:
        key = (
            event.get("selection_policy"),
            event.get("used_llm_planning"),
            event.get("task_type"),
            event.get("skip_reason"),
            event.get("force_reason"),
            event.get("forced_by_runtime_feedback"),
            tuple(event.get("constraint_tags") or []),
            event.get("experiment_support"),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(event)
    return deduped


def summarize_planning_feedback(value: Any) -> Optional[dict[str, Any]]:
    """Return a task-level planning summary suitable for benchmark artifacts."""
    events = extract_planning_events(value)
    if not events:
        return None

    status = (_coerce_optional_text(_extract_value(value, "status")) or "").lower()
    completion_score = _coerce_float(_extract_value(value, "completion_score"))
    if completion_score is None:
        completion_score = _coerce_float(_extract_value(value, "overall_score"))
    if completion_score is None:
        if status in _SUCCESS_STATUSES:
            completion_score = 1.0
        elif status in _FAILURE_STATUSES:
            completion_score = 0.0
        else:
            completion_score = 0.5

    first_event = events[0]
    last_event = events[-1]
    selection_policies = Counter(
        event["selection_policy"] for event in events if event.get("selection_policy")
    )
    force_reasons = Counter(event["force_reason"] for event in events if event.get("force_reason"))
    constraint_counts = Counter(
        tag for event in events for tag in list(event.get("constraint_tags") or [])
    )
    used_llm_count = sum(1 for event in events if event.get("used_llm_planning") is True)
    fast_path_count = sum(1 for event in events if event.get("used_llm_planning") is False)
    avg_support = sum(event.get("experiment_support", 0.0) for event in events) / len(events)

    return {
        "event_count": len(events),
        "selected_policy": first_event.get("selection_policy"),
        "final_policy": last_event.get("selection_policy"),
        "selected_used_llm_planning": first_event.get("used_llm_planning"),
        "final_used_llm_planning": last_event.get("used_llm_planning"),
        "forced_by_runtime_feedback": any(
            bool(event.get("forced_by_runtime_feedback")) for event in events
        ),
        "force_reason": last_event.get("force_reason") or first_event.get("force_reason"),
        "avg_experiment_support": round(avg_support, 4),
        "used_llm_count": used_llm_count,
        "fast_path_count": fast_path_count,
        "completion_score": round(float(completion_score), 4),
        "status": status or None,
        "selection_policies": dict(selection_policies),
        "force_reasons": dict(force_reasons),
        "constraint_tags": dict(constraint_counts),
    }


def aggregate_planning_feedback(
    values: Iterable[Any],
    *,
    total_tasks: Optional[int] = None,
) -> dict[str, Any]:
    """Aggregate planning summaries across benchmark task results."""
    summaries = [summary for value in values if (summary := summarize_planning_feedback(value))]
    if not summaries:
        return {
            "tasks_with_planning_feedback": 0,
            "planning_feedback_coverage": 0.0,
            "planning_policy_counts": {},
            "planning_final_policy_counts": {},
            "planning_force_reasons": {},
            "planning_constraint_tags": {},
            "planning_used_llm_rate": 0.0,
            "planning_fast_path_rate": 0.0,
            "planning_forced_slow_path_count": 0,
            "planning_policy_completion_totals": {},
            "avg_completion_by_planning_policy": {},
            "planning_policy_pass_rates": {},
            "planning_forced_slow_path_completion_delta": None,
            "avg_planning_experiment_support": 0.0,
        }

    task_count = total_tasks if total_tasks is not None else len(summaries)
    selected_policies = Counter(
        summary["selected_policy"] for summary in summaries if summary.get("selected_policy")
    )
    final_policies = Counter(
        summary["final_policy"] for summary in summaries if summary.get("final_policy")
    )
    force_reasons = Counter(
        summary["force_reason"] for summary in summaries if summary.get("force_reason")
    )
    constraint_tags = Counter(
        tag
        for summary in summaries
        for tag, count in dict(summary.get("constraint_tags") or {}).items()
        for _ in range(int(count))
    )
    policy_completion_totals: dict[str, float] = {}
    policy_pass_counts: dict[str, int] = {}
    for summary in summaries:
        final_policy = summary.get("final_policy")
        if not final_policy:
            continue
        policy_completion_totals[final_policy] = round(
            policy_completion_totals.get(final_policy, 0.0) + float(summary["completion_score"]),
            4,
        )
        if str(summary.get("status") or "").lower() in _SUCCESS_STATUSES:
            policy_pass_counts[final_policy] = policy_pass_counts.get(final_policy, 0) + 1

    avg_completion_by_policy = {
        policy: round(policy_completion_totals[policy] / max(1, count), 4)
        for policy, count in final_policies.items()
        if count > 0 and policy in policy_completion_totals
    }
    pass_rates_by_policy = {
        policy: round(policy_pass_counts.get(policy, 0) / max(1, count), 4)
        for policy, count in final_policies.items()
        if count > 0
    }
    forced_delta: Optional[float] = None
    if (
        "experiment_forced_slow_path" in avg_completion_by_policy
        and "heuristic_fast_path" in avg_completion_by_policy
    ):
        forced_delta = round(
            avg_completion_by_policy["experiment_forced_slow_path"]
            - avg_completion_by_policy["heuristic_fast_path"],
            4,
        )

    return {
        "tasks_with_planning_feedback": len(summaries),
        "planning_feedback_coverage": round(len(summaries) / max(1, task_count), 4),
        "planning_policy_counts": dict(selected_policies),
        "planning_final_policy_counts": dict(final_policies),
        "planning_force_reasons": dict(force_reasons),
        "planning_constraint_tags": dict(constraint_tags),
        "planning_used_llm_rate": round(
            sum(summary["used_llm_count"] for summary in summaries)
            / max(1, sum(summary["event_count"] for summary in summaries)),
            4,
        ),
        "planning_fast_path_rate": round(
            sum(summary["fast_path_count"] for summary in summaries)
            / max(1, sum(summary["event_count"] for summary in summaries)),
            4,
        ),
        "planning_forced_slow_path_count": sum(
            final_policies.get("experiment_forced_slow_path", 0) for _ in [0]
        ),
        "planning_policy_completion_totals": dict(policy_completion_totals),
        "avg_completion_by_planning_policy": dict(avg_completion_by_policy),
        "planning_policy_pass_rates": dict(pass_rates_by_policy),
        "planning_forced_slow_path_completion_delta": forced_delta,
        "avg_planning_experiment_support": round(
            sum(summary["avg_experiment_support"] for summary in summaries) / len(summaries),
            4,
        ),
    }


__all__ = [
    "aggregate_planning_feedback",
    "extract_planning_events",
    "summarize_planning_feedback",
]
