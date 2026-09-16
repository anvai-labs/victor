from __future__ import annotations

"""Distill evaluation results into reusable structured experiment memory."""

import hashlib
import time
from pathlib import Path
from typing import Any, Mapping, Optional

from victor.evaluation.experiment_memory import (
    ExperimentInsight,
    ExperimentMemoryRecord,
    ExperimentScope,
    ExperimentTaskSummary,
)
from victor.evaluation.protocol import EvaluationResult
from victor.evaluation.topology_feedback import (
    summarize_optimization_feedback,
    summarize_topology_feedback,
)


def _coerce_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(getattr(value, "value", value)).strip()
    return text or None


def _normalize_summary(
    summary: Optional[Mapping[str, Any]],
    result: EvaluationResult,
) -> dict[str, Any]:
    if summary is None:
        return dict(result.get_metrics())
    return dict(summary)


def _build_scope(result: EvaluationResult) -> ExperimentScope:
    dataset_metadata = dict(result.config.dataset_metadata or {})
    dataset_name = None
    for key in ("source_name", "dataset_name", "dataset", "name"):
        candidate = _coerce_text(dataset_metadata.get(key))
        if candidate:
            dataset_name = candidate
            break

    tags = {
        result.config.benchmark.value,
        _coerce_text(result.config.provider) or "",
        _coerce_text(result.config.prompt_section_name) or "",
        dataset_name or "",
    }
    return ExperimentScope(
        benchmark=result.config.benchmark.value,
        provider=_coerce_text(result.config.provider),
        model=_coerce_text(result.config.model),
        prompt_candidate_hash=_coerce_text(result.config.prompt_candidate_hash),
        section_name=_coerce_text(result.config.prompt_section_name),
        dataset_name=dataset_name,
        tags=tuple(sorted(tag for tag in tags if tag)),
    )


def _build_task_summaries(result: EvaluationResult) -> list[ExperimentTaskSummary]:
    summaries: list[ExperimentTaskSummary] = []
    for task_result in result.task_results:
        summaries.append(
            ExperimentTaskSummary(
                task_id=task_result.task_id,
                status=task_result.status.value,
                completion_score=float(task_result.completion_score or 0.0),
                failure_category=_coerce_text(task_result.failure_category),
                failure_taxonomy=_coerce_text(task_result.failure_taxonomy_path),
                topology=summarize_topology_feedback(task_result),
                optimization=summarize_optimization_feedback(task_result),
            )
        )
    return summaries


def _append_gate_failure_constraints(
    insights: list[ExperimentInsight],
    keywords: set[str],
    summary_metrics: Mapping[str, Any],
) -> None:
    gate_failures = dict(summary_metrics.get("optimization_gate_failures") or {})
    if gate_failures:
        sorted_failures = sorted(gate_failures.items(), key=lambda item: (-int(item[1]), item[0]))
        for gate_name, count in sorted_failures[:3]:
            keywords.add(str(gate_name))
            insights.append(
                ExperimentInsight(
                    kind="environment_constraint",
                    summary=f"Repeated hard constraint failure: {gate_name}.",
                    confidence=min(0.95, 0.45 + (0.1 * int(count))),
                    evidence={"gate_failure": gate_name, "count": int(count)},
                )
            )
        return

    failure_categories = dict(summary_metrics.get("failure_categories") or {})
    if not failure_categories:
        return

    dominant_category, count = max(
        failure_categories.items(),
        key=lambda item: (int(item[1]), item[0]),
    )
    keywords.add(str(dominant_category))
    insights.append(
        ExperimentInsight(
            kind="environment_constraint",
            summary=f"Dominant failure mode remained {dominant_category}.",
            confidence=min(0.9, 0.4 + (0.08 * int(count))),
            evidence={"failure_category": dominant_category, "count": int(count)},
        )
    )


def _append_policy_insights(
    insights: list[ExperimentInsight],
    keywords: set[str],
    summary_metrics: Mapping[str, Any],
) -> None:
    selection_policy_counts = dict(
        summary_metrics.get("topology_selection_policy_optimization_counts") or {}
    )
    delta = _coerce_float(
        summary_metrics.get("topology_learned_override_optimization_reward_delta")
    )
    feasibility_delta = _coerce_float(
        summary_metrics.get("topology_learned_override_feasibility_delta")
    )
    learned_count = int(selection_policy_counts.get("learned_close_override", 0) or 0)
    heuristic_count = int(selection_policy_counts.get("heuristic", 0) or 0)

    if delta is not None and delta <= -0.05:
        keywords.update({"learned_close_override", "heuristic"})
        insights.append(
            ExperimentInsight(
                kind="failed_hypothesis",
                summary="Learned close override underperformed heuristic routing for this scope.",
                confidence=0.9 if learned_count and heuristic_count else 0.75,
                evidence={
                    "optimization_reward_delta": round(delta, 4),
                    "learned_count": learned_count,
                    "heuristic_count": heuristic_count,
                },
            )
        )
    elif delta is not None and delta >= 0.05:
        keywords.update({"learned_close_override", "heuristic"})
        insights.append(
            ExperimentInsight(
                kind="successful_transformation",
                summary="Learned close override improved optimization reward over heuristic routing.",
                confidence=0.9 if learned_count and heuristic_count else 0.75,
                evidence={
                    "optimization_reward_delta": round(delta, 4),
                    "learned_count": learned_count,
                    "heuristic_count": heuristic_count,
                },
            )
        )

    if feasibility_delta is not None and feasibility_delta <= -0.1:
        keywords.add("feasibility")
        insights.append(
            ExperimentInsight(
                kind="failed_hypothesis",
                summary="Learned close override reduced feasibility relative to heuristic routing.",
                confidence=0.85 if learned_count and heuristic_count else 0.7,
                evidence={
                    "feasibility_delta": round(feasibility_delta, 4),
                    "learned_count": learned_count,
                    "heuristic_count": heuristic_count,
                },
            )
        )


def _append_planning_insights(
    insights: list[ExperimentInsight],
    keywords: set[str],
    summary_metrics: Mapping[str, Any],
) -> None:
    planning_constraints = dict(summary_metrics.get("planning_constraint_tags") or {})
    gate_failures = set(dict(summary_metrics.get("optimization_gate_failures") or {}))
    if planning_constraints:
        for constraint_name, count in sorted(
            planning_constraints.items(),
            key=lambda item: (-int(item[1]), item[0]),
        )[:2]:
            keywords.update({"planning", "fast_path", str(constraint_name)})
            if str(constraint_name) in gate_failures:
                continue
            insights.append(
                ExperimentInsight(
                    kind="environment_constraint",
                    summary=(
                        f"Constraint-sensitive planning scope: {constraint_name} required "
                        "deliberate planning."
                    ),
                    confidence=min(0.92, 0.45 + (0.08 * int(count))),
                    evidence={
                        "gate_failure": constraint_name,
                        "planning_sensitive": True,
                        "count": int(count),
                    },
                )
            )

    policy_counts = dict(summary_metrics.get("planning_policy_counts") or {})
    forced_count = int(policy_counts.get("experiment_forced_slow_path", 0) or 0)
    heuristic_count = int(policy_counts.get("heuristic_fast_path", 0) or 0)
    forced_delta = _coerce_float(summary_metrics.get("planning_forced_slow_path_completion_delta"))
    if forced_delta is not None and forced_delta >= 0.1:
        keywords.update({"planning", "fast_path", "llm_planning"})
        insights.append(
            ExperimentInsight(
                kind="successful_transformation",
                summary="Forced LLM planning outperformed heuristic fast-path for this scope.",
                confidence=0.88 if forced_count and heuristic_count else 0.72,
                evidence={
                    "completion_delta": round(forced_delta, 4),
                    "forced_count": forced_count,
                    "heuristic_count": heuristic_count,
                },
            )
        )
        dominant_constraint = next(iter(planning_constraints), None)
        if dominant_constraint:
            insights.append(
                ExperimentInsight(
                    kind="next_candidate",
                    summary=(
                        f"Keep LLM planning enabled when {dominant_constraint} "
                        "constraints are present."
                    ),
                    confidence=0.78,
                    evidence={
                        "constraint_tag": dominant_constraint,
                        "completion_delta": round(forced_delta, 4),
                    },
                )
            )
    elif forced_delta is not None and forced_delta <= -0.1:
        keywords.update({"planning", "fast_path", "llm_planning"})
        insights.append(
            ExperimentInsight(
                kind="failed_hypothesis",
                summary="Forced LLM planning underperformed heuristic fast-path for this scope.",
                confidence=0.88 if forced_count and heuristic_count else 0.72,
                evidence={
                    "completion_delta": round(forced_delta, 4),
                    "forced_count": forced_count,
                    "heuristic_count": heuristic_count,
                },
            )
        )


def _append_degradation_insights(
    insights: list[ExperimentInsight],
    keywords: set[str],
    summary_metrics: Mapping[str, Any],
) -> None:
    provider_degradation_tasks = int(summary_metrics.get("provider_degradation_task_count", 0) or 0)
    content_degradation_tasks = int(summary_metrics.get("content_degradation_task_count", 0) or 0)
    recovery_rate = _coerce_float(summary_metrics.get("degradation_recovery_rate"))
    reasons = dict(summary_metrics.get("degradation_reasons") or {})

    if provider_degradation_tasks > 0:
        keywords.update({"provider_degradation", "recovery"})
        insights.append(
            ExperimentInsight(
                kind="environment_constraint",
                summary="Provider instability affected this scope during execution.",
                confidence=min(0.92, 0.45 + (0.08 * provider_degradation_tasks)),
                evidence={
                    "provider_degradation_task_count": provider_degradation_tasks,
                    "recovery_rate": recovery_rate,
                },
            )
        )

    if content_degradation_tasks > 0:
        keywords.update({"stuck_loop", "content_repetition"})
        insights.append(
            ExperimentInsight(
                kind="failed_hypothesis",
                summary="Loop content repetition indicates unresolved convergence pressure.",
                confidence=min(0.9, 0.42 + (0.08 * content_degradation_tasks)),
                evidence={"content_degradation_task_count": content_degradation_tasks},
            )
        )

    if reasons:
        dominant_reason = sorted(reasons.items(), key=lambda item: (-int(item[1]), item[0]))[0]
        keywords.add(str(dominant_reason[0]))
        insights.append(
            ExperimentInsight(
                kind="next_candidate",
                summary=f"Harden recovery against the dominant degradation cause: {dominant_reason[0]}.",
                confidence=0.74,
                evidence={
                    "reason": dominant_reason[0],
                    "count": int(dominant_reason[1]),
                },
            )
        )


def _append_team_insights(
    insights: list[ExperimentInsight],
    keywords: set[str],
    summary_metrics: Mapping[str, Any],
) -> None:
    materialized_count = int(summary_metrics.get("team_worktree_materialized_count", 0) or 0)
    low_risk_count = int(summary_metrics.get("team_low_risk_task_count", 0) or 0)
    medium_risk_count = int(summary_metrics.get("team_medium_risk_task_count", 0) or 0)
    high_risk_count = int(summary_metrics.get("team_high_risk_task_count", 0) or 0)
    merge_conflict_tasks = int(summary_metrics.get("team_merge_conflict_task_count", 0) or 0)
    out_of_scope_tasks = int(summary_metrics.get("team_out_of_scope_write_task_count", 0) or 0)
    readonly_violation_tasks = int(
        summary_metrics.get("team_readonly_violation_task_count", 0) or 0
    )
    cleanup_error_tasks = int(summary_metrics.get("team_cleanup_error_task_count", 0) or 0)

    if high_risk_count > 0 or merge_conflict_tasks > 0:
        keywords.update({"team", "worktree", "merge"})
        insights.append(
            ExperimentInsight(
                kind="failed_hypothesis",
                summary="Worktree-isolated team execution still produced high merge risk for this scope.",
                confidence=min(0.92, 0.45 + (0.08 * max(high_risk_count, merge_conflict_tasks))),
                evidence={
                    "high_risk_task_count": high_risk_count,
                    "merge_conflict_task_count": merge_conflict_tasks,
                },
            )
        )

    if materialized_count > 0 and low_risk_count >= materialized_count and cleanup_error_tasks == 0:
        keywords.update({"team", "worktree", "merge_safe"})
        insights.append(
            ExperimentInsight(
                kind="successful_transformation",
                summary="Worktree-isolated team execution stayed merge-safe for this scope.",
                confidence=min(0.9, 0.5 + (0.07 * low_risk_count)),
                evidence={
                    "materialized_count": materialized_count,
                    "low_risk_task_count": low_risk_count,
                },
            )
        )

    if cleanup_error_tasks > 0 or out_of_scope_tasks > 0 or readonly_violation_tasks > 0:
        keywords.update({"team", "worktree", "scope_control"})
        insights.append(
            ExperimentInsight(
                kind="environment_constraint",
                summary="Isolated team execution needs tighter scope and cleanup enforcement.",
                confidence=min(
                    0.9,
                    0.46
                    + (
                        0.07
                        * max(
                            cleanup_error_tasks,
                            out_of_scope_tasks,
                            readonly_violation_tasks,
                        )
                    ),
                ),
                evidence={
                    "cleanup_error_task_count": cleanup_error_tasks,
                    "out_of_scope_write_task_count": out_of_scope_tasks,
                    "readonly_violation_task_count": readonly_violation_tasks,
                },
            )
        )

    if medium_risk_count > 0 or out_of_scope_tasks > 0:
        keywords.update({"team", "merge_order", "claimed_paths"})
        insights.append(
            ExperimentInsight(
                kind="next_candidate",
                summary="Narrow claimed_paths or enforce stricter merge ordering before widening team parallelism.",
                confidence=0.76,
                evidence={
                    "medium_risk_task_count": medium_risk_count,
                    "out_of_scope_write_task_count": out_of_scope_tasks,
                },
            )
        )


def _append_next_candidate(
    insights: list[ExperimentInsight],
    keywords: set[str],
    summary_metrics: Mapping[str, Any],
) -> None:
    reward_delta = _coerce_float(
        summary_metrics.get("topology_learned_override_optimization_reward_delta")
    )
    feasibility_delta = _coerce_float(
        summary_metrics.get("topology_learned_override_feasibility_delta")
    )
    feasibility_rate = _coerce_float(summary_metrics.get("optimization_feasibility_rate"))
    pass_rate = _coerce_float(summary_metrics.get("pass_rate"))
    gate_failures = dict(summary_metrics.get("optimization_gate_failures") or {})
    code_intelligence_coverage = _coerce_float(
        summary_metrics.get("code_intelligence_task_coverage")
    )

    if (
        reward_delta is not None
        and reward_delta <= -0.05
        or feasibility_delta is not None
        and feasibility_delta <= -0.1
    ):
        keywords.update({"heuristic", "routing"})
        insights.append(
            ExperimentInsight(
                kind="next_candidate",
                summary=(
                    "Tighten learned override thresholds for this scope or prefer "
                    "heuristic routing until feasibility recovers."
                ),
                confidence=0.86,
                evidence={
                    "optimization_reward_delta": reward_delta,
                    "feasibility_delta": feasibility_delta,
                },
            )
        )
        return

    if gate_failures and feasibility_rate is not None and feasibility_rate < 0.75:
        dominant_failure = sorted(gate_failures.items(), key=lambda item: (-int(item[1]), item[0]))[
            0
        ]
        keywords.add(str(dominant_failure[0]))
        insights.append(
            ExperimentInsight(
                kind="next_candidate",
                summary=f"Target the dominant hard constraint first: {dominant_failure[0]}.",
                confidence=0.8,
                evidence={
                    "gate_failure": dominant_failure[0],
                    "count": int(dominant_failure[1]),
                },
            )
        )
        return

    if pass_rate is not None and pass_rate < 0.6 and code_intelligence_coverage == 0.0:
        keywords.update({"code_intelligence", "coverage"})
        insights.append(
            ExperimentInsight(
                kind="next_candidate",
                summary="Increase code-intelligence coverage before widening topology complexity.",
                confidence=0.72,
                evidence={
                    "pass_rate": pass_rate,
                    "code_intelligence_coverage": code_intelligence_coverage,
                },
            )
        )
        return

    insights.append(
        ExperimentInsight(
            kind="next_candidate",
            summary="Collect more scope-matched experiment traces before broad rollout.",
            confidence=0.65,
            evidence={},
        )
    )


def _build_keywords(
    scope: ExperimentScope,
    task_summaries: list[ExperimentTaskSummary],
    insights: list[ExperimentInsight],
) -> list[str]:
    keywords: set[str] = set(scope.tags)
    for value in (
        scope.benchmark,
        scope.provider,
        scope.model,
        scope.prompt_candidate_hash,
        scope.section_name,
        scope.dataset_name,
    ):
        if value:
            keywords.add(value)
    for task_summary in task_summaries:
        if task_summary.failure_category:
            keywords.add(task_summary.failure_category)
        if task_summary.failure_taxonomy:
            keywords.add(task_summary.failure_taxonomy)
        if task_summary.topology:
            for key in (
                "selected_action",
                "final_action",
                "selected_topology",
                "final_topology",
                "selected_selection_policy",
                "final_selection_policy",
            ):
                value = task_summary.topology.get(key)
                if value:
                    keywords.add(str(value))
        if task_summary.optimization:
            for failure_name in task_summary.optimization.get("feasibility_failures") or []:
                keywords.add(str(failure_name))
    for insight in insights:
        keywords.update(word for word in insight.summary.replace(".", " ").split() if "_" in word)
    return sorted(keyword for keyword in keywords if keyword)


def analyze_evaluation_result(
    result: EvaluationResult,
    *,
    summary: Optional[Mapping[str, Any]] = None,
    runtime_feedback: Optional[Mapping[str, Any]] = None,
    source_result_path: Optional[Path] = None,
    created_at: Optional[float] = None,
) -> ExperimentMemoryRecord:
    """Analyze an evaluation result into a reusable structured memory record."""
    created_timestamp = float(created_at if created_at is not None else time.time())
    summary_metrics = _normalize_summary(summary, result)
    scope = _build_scope(result)
    task_summaries = _build_task_summaries(result)

    insights: list[ExperimentInsight] = []
    keyword_seed: set[str] = set()
    _append_policy_insights(insights, keyword_seed, summary_metrics)
    _append_planning_insights(insights, keyword_seed, summary_metrics)
    _append_degradation_insights(insights, keyword_seed, summary_metrics)
    _append_team_insights(insights, keyword_seed, summary_metrics)
    _append_gate_failure_constraints(insights, keyword_seed, summary_metrics)
    _append_next_candidate(insights, keyword_seed, summary_metrics)
    keywords = sorted(set(_build_keywords(scope, task_summaries, insights)) | keyword_seed)

    source_text = str(source_result_path) if source_result_path is not None else ""
    record_seed = "|".join(
        [
            scope.benchmark,
            scope.provider or "",
            scope.model or "",
            scope.prompt_candidate_hash or "",
            scope.section_name or "",
            source_text,
            ",".join(task.task_id for task in task_summaries),
        ]
    )
    digest = hashlib.sha1(record_seed.encode("utf-8"), usedforsecurity=False).hexdigest()[:12]
    metadata = {
        "dataset_metadata": dict(result.config.dataset_metadata or {}),
        "task_count": len(task_summaries),
    }
    if runtime_feedback is not None:
        metadata["runtime_feedback_metadata"] = dict(runtime_feedback.get("metadata") or {})

    return ExperimentMemoryRecord(
        record_id=f"{scope.benchmark}:{digest}",
        created_at=created_timestamp,
        scope=scope,
        summary_metrics=summary_metrics,
        task_summaries=task_summaries,
        insights=insights,
        keywords=keywords,
        source_result_path=source_text or None,
        metadata=metadata,
    )


__all__ = ["analyze_evaluation_result"]
