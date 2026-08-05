# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Persistence and derivation helpers for runtime calibration from validated evaluation truth."""

from __future__ import annotations

import json
import hashlib
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

from victor_contracts.runtime_evaluation import RuntimeEvaluationFeedback

RUNTIME_EVALUATION_FEEDBACK_FILENAME = "runtime_evaluation_feedback.json"
VALIDATED_RUNTIME_FEEDBACK_SOURCES = {
    "benchmark_truth_feedback",
    "validated_session_truth_feedback",
    "validated_evaluation_truth_feedback",
}
AGGREGATED_RUNTIME_FEEDBACK_SOURCE = "validated_evaluation_truth_aggregate"
SESSION_TOPOLOGY_RUNTIME_FEEDBACK_SOURCE = "session_topology_runtime_feedback"
AGGREGATED_SESSION_TOPOLOGY_RUNTIME_FEEDBACK_SOURCE = "session_topology_runtime_feedback_aggregate"
FRESHNESS_HALF_LIFE_DAYS = 14.0
SESSION_TOPOLOGY_FRESHNESS_HALF_LIFE_DAYS = 5.0
VALIDATED_RUNTIME_FEEDBACK_SOURCE_TRUST = {
    "benchmark_truth_feedback": 1.0,
    "validated_evaluation_truth_feedback": 0.95,
    "validated_session_truth_feedback": 0.85,
}
RUNTIME_FEEDBACK_SOURCE_TRUST = {
    **VALIDATED_RUNTIME_FEEDBACK_SOURCE_TRUST,
    AGGREGATED_RUNTIME_FEEDBACK_SOURCE: 0.95,
    SESSION_TOPOLOGY_RUNTIME_FEEDBACK_SOURCE: 0.35,
    AGGREGATED_SESSION_TOPOLOGY_RUNTIME_FEEDBACK_SOURCE: 0.35,
}
RUNTIME_TOPOLOGY_FEEDBACK_FILENAME_PREFIX = "runtime_topology_feedback"
TOPOLOGY_RUNTIME_FEEDBACK_SOURCES = {
    *VALIDATED_RUNTIME_FEEDBACK_SOURCES,
    AGGREGATED_RUNTIME_FEEDBACK_SOURCE,
    SESSION_TOPOLOGY_RUNTIME_FEEDBACK_SOURCE,
    AGGREGATED_SESSION_TOPOLOGY_RUNTIME_FEEDBACK_SOURCE,
}
TOPOLOGY_RUNTIME_METADATA_KEYS = (
    "topology_feedback_coverage",
    "avg_topology_reward",
    "avg_topology_confidence",
    "degradation_feedback_coverage",
    "degradation_event_count",
    "degraded_task_count",
    "recovered_task_count",
    "degradation_recovery_rate",
    "avg_degradation_adaptation_cost",
    "avg_degradation_time_to_recover_seconds",
    "avg_degradation_cost_variance",
    "avg_degradation_recovery_time_variance",
    "avg_degradation_intervention_count",
    "avg_degradation_confidence",
    "avg_degradation_drift_score",
    "content_degradation_task_count",
    "confidence_degradation_task_count",
    "provider_degradation_task_count",
    "persistent_degradation_task_count",
    "drift_task_count",
    "degradation_drift_rate",
    "degradation_intervention_task_count",
    "degradation_intervention_rate",
    "high_adaptation_cost_task_count",
    "degradation_high_cost_rate",
    "degradation_confidence_rate",
    "degradation_stability_score",
    "degradation_sources",
    "degradation_kinds",
    "degradation_failure_types",
    "degradation_providers",
    "degradation_reasons",
    "optimization_feasible_tasks",
    "optimization_infeasible_tasks",
    "optimization_feasibility_rate",
    "avg_optimization_reward",
    "avg_feasible_optimization_reward",
    "avg_infeasible_optimization_reward",
    "optimization_gate_failures",
    "topology_observation_count",
    "topology_actions",
    "topology_final_actions",
    "topology_kinds",
    "topology_final_kinds",
    "topology_execution_modes",
    "topology_providers",
    "topology_formations",
    "topology_selection_policies",
    "topology_selection_policy_reward_totals",
    "avg_topology_reward_by_selection_policy",
    "topology_learned_override_reward_delta",
    "topology_selection_policy_optimization_counts",
    "topology_selection_policy_optimization_reward_totals",
    "avg_topology_optimization_reward_by_selection_policy",
    "topology_selection_policy_feasible_counts",
    "topology_selection_policy_feasibility_rates",
    "topology_learned_override_optimization_reward_delta",
    "topology_learned_override_feasibility_delta",
    "topology_selection_policy_scope_metrics",
    "topology_action_agreement",
    "topology_kind_agreement",
    "topology_provider_agreement",
    "topology_formation_agreement",
    "topology_conflict_score",
    "tasks_with_team_feedback",
    "team_feedback_coverage",
    "team_formations",
    "team_merge_risk_levels",
    "team_worktree_plan_count",
    "team_worktree_materialized_count",
    "team_worktree_dry_run_count",
    "team_low_risk_task_count",
    "team_medium_risk_task_count",
    "team_high_risk_task_count",
    "team_merge_conflict_task_count",
    "team_merge_conflict_count",
    "team_merge_overlap_task_count",
    "team_out_of_scope_write_task_count",
    "team_out_of_scope_write_count",
    "team_readonly_violation_task_count",
    "team_readonly_violation_count",
    "team_cleanup_task_count",
    "team_cleanup_error_task_count",
    "team_cleanup_error_count",
    "avg_team_assignments",
    "avg_team_scoped_members",
    "avg_team_members_with_changes",
    "avg_team_changed_file_count",
    "team_materialized_assignment_total",
    "team_worktree_scope_metrics",
)
SCOPE_FIELD_WEIGHTS = {
    "project": 4.0,
    "model": 3.5,
    "task_type": 2.5,
    "provider": 2.0,
    "benchmark": 1.5,
    "vertical": 1.0,
    "workflow": 0.75,
}


@dataclass(frozen=True)
class RuntimeEvaluationFeedbackScope:
    """Explicit scope schema for validated evaluation-truth feedback."""

    project: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    task_type: Optional[str] = None
    benchmark: Optional[str] = None
    vertical: Optional[str] = None
    workflow: Optional[str] = None
    tags: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the canonical scope schema."""
        return {
            "project": self.project,
            "provider": self.provider,
            "model": self.model,
            "task_type": self.task_type,
            "benchmark": self.benchmark,
            "vertical": self.vertical,
            "workflow": self.workflow,
            "tags": list(self.tags),
        }

    def is_empty(self) -> bool:
        """Return whether the scope carries any discriminating fields."""
        return not any(
            (
                self.project,
                self.provider,
                self.model,
                self.task_type,
                self.benchmark,
                self.vertical,
                self.workflow,
                self.tags,
            )
        )

    @classmethod
    def from_value(cls, value: Any) -> "RuntimeEvaluationFeedbackScope":
        """Normalize mappings and existing scope objects into the typed schema."""
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            return cls()
        tags_value = value.get("tags") or ()
        if isinstance(tags_value, str):
            tags = (tags_value,)
        else:
            tags = tuple(str(tag) for tag in tags_value if str(tag).strip())
        return cls(
            project=_coerce_optional_text(value.get("project")),
            provider=_coerce_optional_text(value.get("provider")),
            model=_coerce_optional_text(value.get("model")),
            task_type=_coerce_optional_text(value.get("task_type")),
            benchmark=_coerce_optional_text(value.get("benchmark")),
            vertical=_coerce_optional_text(value.get("vertical")),
            workflow=_coerce_optional_text(value.get("workflow")),
            tags=tags,
        )


def get_runtime_evaluation_feedback_path(base_dir: Optional[Path] = None) -> Path:
    """Return the canonical persisted runtime-feedback file path."""
    if base_dir is not None:
        return Path(base_dir) / RUNTIME_EVALUATION_FEEDBACK_FILENAME

    try:
        from victor.config.secure_paths import get_victor_dir

        return get_victor_dir() / "evaluations" / RUNTIME_EVALUATION_FEEDBACK_FILENAME
    except ImportError:
        return Path.home() / ".victor" / "evaluations" / RUNTIME_EVALUATION_FEEDBACK_FILENAME


def _resolve_feedback_base_dir(target: Optional[Path]) -> Path:
    """Resolve the directory used for scoped runtime-feedback artifacts."""
    target_path = Path(target) if target is not None else get_runtime_evaluation_feedback_path()
    feedback_dir = _resolve_feedback_directory(target_path)
    if feedback_dir is not None:
        return feedback_dir
    return target_path.parent


def _coerce_optional_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalized_scope_token(value: Any) -> Optional[str]:
    text = _coerce_optional_text(value)
    return text.lower() if text is not None else None


from victor.core.utils import clamp as _clamp


def _parse_timestamp(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _format_timestamp(value: Optional[datetime]) -> Optional[str]:
    if value is None:
        return None
    normalized = value.astimezone(timezone.utc)
    return normalized.isoformat()


def _extract_task_entries(
    result_or_payload: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Normalize evaluation result objects and saved payloads into plain records."""
    if hasattr(result_or_payload, "task_results") and hasattr(result_or_payload, "get_metrics"):
        tasks: list[dict[str, Any]] = []
        for task in result_or_payload.task_results:
            assessment = task.get_confidence_assessment()
            tasks.append(
                {
                    "status": task.status.value,
                    "confidence_assessment": assessment.to_dict(),
                }
            )
        config = {
            "benchmark": result_or_payload.config.benchmark.value,
            "model": result_or_payload.config.model,
            "provider": result_or_payload.config.provider,
            "prompt_candidate_hash": result_or_payload.config.prompt_candidate_hash,
            "section_name": result_or_payload.config.prompt_section_name,
            "prompt_section_name": result_or_payload.config.prompt_section_name,
            "dataset_metadata": dict(result_or_payload.config.dataset_metadata),
        }
        return tasks, result_or_payload.get_metrics(), config

    payload = dict(result_or_payload or {})
    tasks = list(payload.get("tasks") or payload.get("task_results") or [])
    summary = dict(payload.get("summary") or {})
    for section_name in ("quality", "optimization", "topology", "degradation"):
        section_payload = payload.get(section_name)
        if isinstance(section_payload, Mapping):
            summary.update(dict(section_payload))
    config = dict(payload.get("config") or {})
    if not config:
        for field_name in ("benchmark", "model", "provider"):
            if payload.get(field_name) is not None:
                config[field_name] = payload.get(field_name)
    return tasks, summary, config


def derive_runtime_evaluation_feedback(
    result_or_payload: Any,
) -> RuntimeEvaluationFeedback:
    """Derive runtime calibration feedback from evaluation results or saved payloads."""
    tasks, summary, config = _extract_task_entries(result_or_payload)
    passed_confidences: list[float] = []
    failed_confidences: list[float] = []
    passed_evidence: list[float] = []

    for task in tasks:
        assessment_payload = task.get("confidence_assessment") or {}
        if not assessment_payload:
            continue
        confidence = float(assessment_payload.get("confidence_score", 0.0) or 0.0)
        evidence = float(assessment_payload.get("evidence_score", 0.0) or 0.0)
        status = str(task.get("status") or "").lower()
        if status == "passed":
            passed_confidences.append(confidence)
            passed_evidence.append(evidence)
        else:
            failed_confidences.append(confidence)

    overconfidence_rate = float(summary.get("overconfidence_rate", 0.0) or 0.0)
    underconfidence_rate = float(summary.get("underconfidence_rate", 0.0) or 0.0)
    truth_alignment_rate = float(summary.get("truth_alignment_rate", 0.0) or 0.0)

    threshold = 0.8
    threshold += min(0.15, overconfidence_rate * 0.25)
    threshold -= min(0.15, underconfidence_rate * 0.20)

    if passed_confidences:
        threshold = min(threshold, (sum(passed_confidences) / len(passed_confidences)) - 0.05)
    if failed_confidences:
        threshold = max(threshold, (sum(failed_confidences) / len(failed_confidences)) + 0.05)
    if truth_alignment_rate < 0.75:
        threshold += min(0.08, (0.75 - truth_alignment_rate) * 0.20)
    threshold = _clamp(threshold, 0.55, 0.92)

    progress_threshold = _clamp(threshold - 0.15, 0.35, threshold)
    if underconfidence_rate > overconfidence_rate:
        progress_threshold = _clamp(
            progress_threshold - min(0.05, (underconfidence_rate - overconfidence_rate) * 0.10),
            0.35,
            threshold,
        )
    elif overconfidence_rate > underconfidence_rate:
        progress_threshold = _clamp(
            progress_threshold + min(0.05, (overconfidence_rate - underconfidence_rate) * 0.10),
            0.35,
            threshold,
        )

    evidence_threshold = threshold + 0.05
    if passed_evidence:
        evidence_threshold = max(
            evidence_threshold, (sum(passed_evidence) / len(passed_evidence)) - 0.02
        )
    if overconfidence_rate > 0:
        evidence_threshold += min(0.05, overconfidence_rate * 0.10)
    evidence_threshold = _clamp(evidence_threshold, 0.55, 0.95)

    topology_selection_policy_optimization_counts = dict(
        summary.get("topology_selection_policy_optimization_counts") or {}
    )
    topology_selection_policy_optimization_reward_totals = dict(
        summary.get("topology_selection_policy_optimization_reward_totals") or {}
    )
    avg_topology_optimization_reward_by_selection_policy = dict(
        summary.get("avg_topology_optimization_reward_by_selection_policy") or {}
    )
    if (
        not avg_topology_optimization_reward_by_selection_policy
        and topology_selection_policy_optimization_counts
        and topology_selection_policy_optimization_reward_totals
    ):
        avg_topology_optimization_reward_by_selection_policy = _average_mapping_from_totals(
            topology_selection_policy_optimization_counts,
            topology_selection_policy_optimization_reward_totals,
        )
    topology_selection_policy_feasible_counts = dict(
        summary.get("topology_selection_policy_feasible_counts") or {}
    )
    topology_selection_policy_feasibility_rates = dict(
        summary.get("topology_selection_policy_feasibility_rates") or {}
    )
    if (
        not topology_selection_policy_feasibility_rates
        and topology_selection_policy_optimization_counts
    ):
        topology_selection_policy_feasibility_rates = {
            str(policy): round(
                float(topology_selection_policy_feasible_counts.get(policy, 0.0))
                / max(1.0, float(count_value)),
                4,
            )
            for policy, count_value in topology_selection_policy_optimization_counts.items()
            if float(count_value) > 0.0
        }

    return RuntimeEvaluationFeedback(
        completion_threshold=round(threshold, 4),
        enhanced_progress_threshold=round(progress_threshold, 4),
        minimum_supported_evidence_score=round(evidence_threshold, 4),
        metadata={
            "source": "benchmark_truth_feedback",
            "validated_evaluation_truth": True,
            "truth_validation_mode": "benchmark",
            "benchmark": config.get("benchmark"),
            "provider": config.get("provider"),
            "model": config.get("model"),
            "prompt_candidate_hash": config.get("prompt_candidate_hash"),
            "section_name": config.get("section_name") or config.get("prompt_section_name"),
            "scope": RuntimeEvaluationFeedbackScope(
                benchmark=_coerce_optional_text(config.get("benchmark")),
                provider=_coerce_optional_text(config.get("provider")),
                model=_coerce_optional_text(config.get("model")),
            ).to_dict(),
            "dataset_metadata": dict(config.get("dataset_metadata") or {}),
            "truth_alignment_rate": round(truth_alignment_rate, 4),
            "overconfidence_rate": round(overconfidence_rate, 4),
            "underconfidence_rate": round(underconfidence_rate, 4),
            "task_count": len(tasks),
            "topology_feedback_coverage": round(
                float(summary.get("topology_feedback_coverage", 0.0) or 0.0), 4
            ),
            "avg_topology_reward": round(float(summary.get("avg_topology_reward", 0.0) or 0.0), 4),
            "avg_topology_confidence": round(
                float(summary.get("avg_topology_confidence", 0.0) or 0.0), 4
            ),
            "degradation_feedback_coverage": round(
                float(summary.get("degradation_feedback_coverage", 0.0) or 0.0), 4
            ),
            "degradation_event_count": int(summary.get("degradation_event_count", 0) or 0),
            "degraded_task_count": int(summary.get("degraded_task_count", 0) or 0),
            "recovered_task_count": int(summary.get("recovered_task_count", 0) or 0),
            "degradation_recovery_rate": round(
                float(summary.get("degradation_recovery_rate", 0.0) or 0.0),
                4,
            ),
            "avg_degradation_adaptation_cost": round(
                float(summary.get("avg_degradation_adaptation_cost", 0.0) or 0.0),
                4,
            ),
            "avg_degradation_time_to_recover_seconds": round(
                float(summary.get("avg_degradation_time_to_recover_seconds", 0.0) or 0.0),
                4,
            ),
            "avg_degradation_cost_variance": round(
                float(summary.get("avg_degradation_cost_variance", 0.0) or 0.0),
                4,
            ),
            "avg_degradation_recovery_time_variance": round(
                float(summary.get("avg_degradation_recovery_time_variance", 0.0) or 0.0),
                4,
            ),
            "avg_degradation_intervention_count": round(
                float(summary.get("avg_degradation_intervention_count", 0.0) or 0.0),
                4,
            ),
            "avg_degradation_confidence": round(
                float(summary.get("avg_degradation_confidence", 0.0) or 0.0),
                4,
            ),
            "avg_degradation_drift_score": round(
                float(summary.get("avg_degradation_drift_score", 0.0) or 0.0),
                4,
            ),
            "content_degradation_task_count": int(
                summary.get("content_degradation_task_count", 0) or 0
            ),
            "confidence_degradation_task_count": int(
                summary.get("confidence_degradation_task_count", 0) or 0
            ),
            "provider_degradation_task_count": int(
                summary.get("provider_degradation_task_count", 0) or 0
            ),
            "persistent_degradation_task_count": int(
                summary.get("persistent_degradation_task_count", 0) or 0
            ),
            "drift_task_count": int(summary.get("drift_task_count", 0) or 0),
            "degradation_drift_rate": round(
                float(summary.get("degradation_drift_rate", 0.0) or 0.0),
                4,
            ),
            "degradation_intervention_task_count": int(
                summary.get("degradation_intervention_task_count", 0) or 0
            ),
            "degradation_intervention_rate": round(
                float(summary.get("degradation_intervention_rate", 0.0) or 0.0),
                4,
            ),
            "high_adaptation_cost_task_count": int(
                summary.get("high_adaptation_cost_task_count", 0) or 0
            ),
            "degradation_high_cost_rate": round(
                float(summary.get("degradation_high_cost_rate", 0.0) or 0.0),
                4,
            ),
            "degradation_confidence_rate": round(
                float(summary.get("degradation_confidence_rate", 0.0) or 0.0),
                4,
            ),
            "degradation_stability_score": round(
                float(summary.get("degradation_stability_score", 0.0) or 0.0),
                4,
            ),
            "degradation_sources": dict(summary.get("degradation_sources") or {}),
            "degradation_kinds": dict(summary.get("degradation_kinds") or {}),
            "degradation_failure_types": dict(summary.get("degradation_failure_types") or {}),
            "degradation_providers": dict(summary.get("degradation_providers") or {}),
            "degradation_reasons": dict(summary.get("degradation_reasons") or {}),
            "topology_actions": dict(summary.get("topology_actions") or {}),
            "topology_execution_modes": dict(summary.get("topology_execution_modes") or {}),
            "topology_selection_policies": dict(summary.get("topology_selection_policies") or {}),
            "topology_selection_policy_reward_totals": dict(
                summary.get("topology_selection_policy_reward_totals") or {}
            ),
            "avg_topology_reward_by_selection_policy": dict(
                summary.get("avg_topology_reward_by_selection_policy") or {}
            ),
            "topology_learned_override_reward_delta": _coerce_float(
                summary.get("topology_learned_override_reward_delta")
            ),
            "optimization_feasible_tasks": int(
                summary.get("optimization_feasible_tasks", summary.get("feasible_tasks", 0)) or 0
            ),
            "optimization_infeasible_tasks": int(
                summary.get(
                    "optimization_infeasible_tasks",
                    summary.get("infeasible_tasks", 0),
                )
                or 0
            ),
            "optimization_feasibility_rate": round(
                float(
                    summary.get(
                        "optimization_feasibility_rate",
                        summary.get("feasibility_rate", 0.0),
                    )
                    or 0.0
                ),
                4,
            ),
            "avg_optimization_reward": round(
                float(
                    summary.get("avg_optimization_reward", summary.get("avg_reward", 0.0)) or 0.0
                ),
                4,
            ),
            "avg_feasible_optimization_reward": round(
                float(
                    summary.get(
                        "avg_feasible_optimization_reward",
                        summary.get("avg_feasible_reward", 0.0),
                    )
                    or 0.0
                ),
                4,
            ),
            "avg_infeasible_optimization_reward": round(
                float(
                    summary.get(
                        "avg_infeasible_optimization_reward",
                        summary.get("avg_infeasible_reward", 0.0),
                    )
                    or 0.0
                ),
                4,
            ),
            "optimization_gate_failures": dict(
                summary.get("optimization_gate_failures", summary.get("gate_failures")) or {}
            ),
            "topology_selection_policy_optimization_counts": (
                topology_selection_policy_optimization_counts
            ),
            "topology_selection_policy_optimization_reward_totals": (
                topology_selection_policy_optimization_reward_totals
            ),
            "avg_topology_optimization_reward_by_selection_policy": (
                avg_topology_optimization_reward_by_selection_policy
            ),
            "topology_selection_policy_feasible_counts": (
                topology_selection_policy_feasible_counts
            ),
            "topology_selection_policy_feasibility_rates": (
                topology_selection_policy_feasibility_rates
            ),
            "topology_learned_override_optimization_reward_delta": (
                _coerce_float(summary.get("topology_learned_override_optimization_reward_delta"))
                or _selection_policy_reward_delta(
                    avg_topology_optimization_reward_by_selection_policy
                )
            ),
            "topology_learned_override_feasibility_delta": (
                _coerce_float(summary.get("topology_learned_override_feasibility_delta"))
                or _selection_policy_reward_delta(topology_selection_policy_feasibility_rates)
            ),
            "tasks_with_team_feedback": int(summary.get("tasks_with_team_feedback", 0) or 0),
            "team_feedback_coverage": round(
                float(summary.get("team_feedback_coverage", 0.0) or 0.0),
                4,
            ),
            "team_formations": dict(summary.get("team_formations") or {}),
            "team_merge_risk_levels": dict(summary.get("team_merge_risk_levels") or {}),
            "team_worktree_plan_count": int(summary.get("team_worktree_plan_count", 0) or 0),
            "team_worktree_materialized_count": int(
                summary.get("team_worktree_materialized_count", 0) or 0
            ),
            "team_worktree_dry_run_count": int(summary.get("team_worktree_dry_run_count", 0) or 0),
            "team_low_risk_task_count": int(summary.get("team_low_risk_task_count", 0) or 0),
            "team_medium_risk_task_count": int(summary.get("team_medium_risk_task_count", 0) or 0),
            "team_high_risk_task_count": int(summary.get("team_high_risk_task_count", 0) or 0),
            "team_merge_conflict_task_count": int(
                summary.get("team_merge_conflict_task_count", 0) or 0
            ),
            "team_merge_conflict_count": int(summary.get("team_merge_conflict_count", 0) or 0),
            "team_merge_overlap_task_count": int(
                summary.get("team_merge_overlap_task_count", 0) or 0
            ),
            "team_out_of_scope_write_task_count": int(
                summary.get("team_out_of_scope_write_task_count", 0) or 0
            ),
            "team_out_of_scope_write_count": int(
                summary.get("team_out_of_scope_write_count", 0) or 0
            ),
            "team_readonly_violation_task_count": int(
                summary.get("team_readonly_violation_task_count", 0) or 0
            ),
            "team_readonly_violation_count": int(
                summary.get("team_readonly_violation_count", 0) or 0
            ),
            "team_cleanup_task_count": int(summary.get("team_cleanup_task_count", 0) or 0),
            "team_cleanup_error_task_count": int(
                summary.get("team_cleanup_error_task_count", 0) or 0
            ),
            "team_cleanup_error_count": int(summary.get("team_cleanup_error_count", 0) or 0),
            "avg_team_assignments": round(
                float(summary.get("avg_team_assignments", 0.0) or 0.0), 4
            ),
            "avg_team_scoped_members": round(
                float(summary.get("avg_team_scoped_members", 0.0) or 0.0),
                4,
            ),
            "avg_team_members_with_changes": round(
                float(summary.get("avg_team_members_with_changes", 0.0) or 0.0),
                4,
            ),
            "avg_team_changed_file_count": round(
                float(summary.get("avg_team_changed_file_count", 0.0) or 0.0),
                4,
            ),
            "team_materialized_assignment_total": int(
                summary.get("team_materialized_assignment_total", 0) or 0
            ),
            "team_worktree_scope_metrics": dict(summary.get("team_worktree_scope_metrics") or {}),
        },
    )


def runtime_evaluation_feedback_scope_from_context(
    context: Optional[Mapping[str, Any]],
) -> RuntimeEvaluationFeedbackScope:
    """Build the canonical scope schema from runtime context/config mappings."""
    if not isinstance(context, Mapping):
        return RuntimeEvaluationFeedbackScope()
    if "scope" in context:
        explicit_scope = RuntimeEvaluationFeedbackScope.from_value(context.get("scope"))
        if not explicit_scope.is_empty():
            return explicit_scope
    return RuntimeEvaluationFeedbackScope(
        project=_coerce_optional_text(
            context.get("project")
            or context.get("project_name")
            or context.get("repository")
            or context.get("repo")
            or context.get("workspace_name")
        ),
        provider=_coerce_optional_text(context.get("provider") or context.get("provider_name")),
        model=_coerce_optional_text(context.get("model") or context.get("model_name")),
        task_type=_coerce_optional_text(context.get("task_type")),
        benchmark=_coerce_optional_text(context.get("benchmark")),
        vertical=_coerce_optional_text(context.get("vertical")),
        workflow=_coerce_optional_text(context.get("workflow") or context.get("workflow_name")),
        tags=tuple(str(tag) for tag in (context.get("tags") or ()) if str(tag).strip()),
    )


def build_runtime_evaluation_feedback_payload(
    feedback: RuntimeEvaluationFeedback,
    *,
    source_result_path: Optional[Path] = None,
    saved_at: Optional[datetime] = None,
) -> dict[str, Any]:
    """Serialize runtime feedback with normalized validation metadata."""
    payload = feedback.to_dict()
    metadata = dict(payload.get("metadata") or {})
    source = metadata.get("source")
    validated = bool(metadata.get("validated_evaluation_truth")) or (
        source in VALIDATED_RUNTIME_FEEDBACK_SOURCES
    )
    metadata["validated_evaluation_truth"] = validated
    if metadata.get("truth_validation_mode") is None:
        if source == "benchmark_truth_feedback":
            metadata["truth_validation_mode"] = "benchmark"
        elif validated:
            metadata["truth_validation_mode"] = "posthoc_validated"
    scope = RuntimeEvaluationFeedbackScope.from_value(metadata.get("scope"))
    if scope.is_empty():
        scope = runtime_evaluation_feedback_scope_from_context(metadata)
    metadata["scope"] = scope.to_dict()
    if metadata.get("benchmark") is None:
        metadata["benchmark"] = scope.benchmark
    if metadata.get("model") is None:
        metadata["model"] = scope.model
    resolved_saved_at = (
        saved_at or _parse_timestamp(metadata.get("saved_at")) or datetime.now(timezone.utc)
    )
    metadata["saved_at"] = _format_timestamp(resolved_saved_at)
    if source_result_path is not None:
        metadata["source_result_path"] = str(source_result_path)
    else:
        metadata["source_result_path"] = metadata.get("source_result_path")
    payload["metadata"] = metadata
    return payload


def build_validated_session_feedback_payload(
    feedback: RuntimeEvaluationFeedback,
    *,
    scope: RuntimeEvaluationFeedbackScope,
    validation_label: str = "posthoc_validated",
    metadata: Optional[Mapping[str, Any]] = None,
    source: str = "validated_session_truth_feedback",
    source_result_path: Optional[Path] = None,
    saved_at: Optional[datetime] = None,
) -> dict[str, Any]:
    """Build an explicit validated-session truth payload using the canonical schema."""
    resolved_scope = RuntimeEvaluationFeedbackScope.from_value(scope)
    payload_metadata = dict(metadata or {})
    payload_metadata.update(
        {
            "source": source,
            "validated_evaluation_truth": True,
            "truth_validation_mode": validation_label,
            "scope": resolved_scope.to_dict(),
        }
    )
    return build_runtime_evaluation_feedback_payload(
        RuntimeEvaluationFeedback(
            completion_threshold=feedback.completion_threshold,
            enhanced_progress_threshold=feedback.enhanced_progress_threshold,
            minimum_supported_evidence_score=feedback.minimum_supported_evidence_score,
            metadata=payload_metadata,
        ),
        source_result_path=source_result_path,
        saved_at=saved_at,
    )


def _runtime_feedback_scope_storage_key(scope: Any) -> str:
    """Return a stable storage key for a scoped runtime-feedback artifact."""
    resolved_scope = RuntimeEvaluationFeedbackScope.from_value(scope)
    if resolved_scope.is_empty():
        return "global"
    encoded = json.dumps(
        resolved_scope.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:16]


def get_session_topology_runtime_feedback_path(
    *,
    base_path: Optional[Path] = None,
    scope: Optional[Any] = None,
) -> Path:
    """Return the scoped persistence path for live topology runtime feedback."""
    base_dir = _resolve_feedback_base_dir(base_path)
    scope_key = _runtime_feedback_scope_storage_key(scope)
    return base_dir / f"{RUNTIME_TOPOLOGY_FEEDBACK_FILENAME_PREFIX}.{scope_key}.json"


def build_session_topology_runtime_feedback_payload(
    feedback: RuntimeEvaluationFeedback,
    *,
    scope: Optional[Any],
    metadata: Optional[Mapping[str, Any]] = None,
    source: str = SESSION_TOPOLOGY_RUNTIME_FEEDBACK_SOURCE,
    source_result_path: Optional[Path] = None,
    saved_at: Optional[datetime] = None,
) -> dict[str, Any]:
    """Build a scoped live-topology feedback payload for cross-session reuse."""
    resolved_scope = RuntimeEvaluationFeedbackScope.from_value(scope)
    payload_metadata = dict(feedback.metadata or {})
    payload_metadata.update(dict(metadata or {}))
    payload_metadata.update(
        {
            "source": source,
            "validated_evaluation_truth": False,
            "truth_validation_mode": "live_session_topology",
            "scope": resolved_scope.to_dict(),
            "topology_feedback_only": True,
        }
    )
    return build_runtime_evaluation_feedback_payload(
        RuntimeEvaluationFeedback(
            completion_threshold=feedback.completion_threshold,
            enhanced_progress_threshold=feedback.enhanced_progress_threshold,
            minimum_supported_evidence_score=feedback.minimum_supported_evidence_score,
            metadata=payload_metadata,
        ),
        source_result_path=source_result_path,
        saved_at=saved_at,
    )


def save_session_topology_runtime_feedback(
    feedback: RuntimeEvaluationFeedback,
    *,
    base_path: Optional[Path] = None,
    scope: Optional[Any] = None,
    source_result_path: Optional[Path] = None,
) -> Path:
    """Persist scoped live-topology runtime feedback without changing thresholds."""
    target_path = get_session_topology_runtime_feedback_path(
        base_path=base_path,
        scope=scope,
    )
    target_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_session_topology_runtime_feedback_payload(
        feedback,
        scope=scope,
        source_result_path=source_result_path,
    )
    target_path.write_text(json.dumps(payload, indent=2))
    return target_path


def build_swe_bench_validated_session_feedback_payload(
    validation_result: Any,
    *,
    score: Optional[Any] = None,
    source_result_path: Optional[Path] = None,
    saved_at: Optional[datetime] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    """Build validated session-truth payload from objective SWE-bench validation."""
    baseline = getattr(validation_result, "baseline", None)
    if baseline is None:
        return None

    metadata_payload = dict(metadata or {})
    provider = _coerce_optional_text(metadata_payload.get("provider"))
    model = _coerce_optional_text(metadata_payload.get("model"))
    prompt_candidate_hash = _coerce_optional_text(metadata_payload.get("prompt_candidate_hash"))
    section_name = _coerce_optional_text(
        metadata_payload.get("section_name") or metadata_payload.get("prompt_section_name")
    )

    baseline_status = str(getattr(getattr(baseline, "status", None), "value", "") or "").lower()
    fail_to_pass = list(getattr(baseline, "fail_to_pass", []) or [])
    pass_to_pass = list(getattr(baseline, "pass_to_pass", []) or [])
    total_fail_to_pass = len(fail_to_pass)
    total_pass_to_pass = len(pass_to_pass)
    total_validated_tests = total_fail_to_pass + total_pass_to_pass
    if baseline_status != "valid" or total_validated_tests == 0:
        return None

    fixed_tests = list(getattr(validation_result, "fail_to_pass_fixed", []) or [])
    broken_tests = list(getattr(validation_result, "pass_to_pass_broken", []) or [])
    fail_to_pass_score = (
        float(getattr(score, "fail_to_pass_score", 0.0) or 0.0)
        if score is not None
        else len(fixed_tests) / max(1, total_fail_to_pass)
    )
    pass_to_pass_score = (
        float(getattr(score, "pass_to_pass_score", 1.0) or 1.0)
        if score is not None
        else (
            1.0 - (len(broken_tests) / max(1, total_pass_to_pass))
            if total_pass_to_pass > 0
            else 1.0
        )
    )
    overall_score = (
        float(getattr(score, "overall_score", getattr(validation_result, "score", 0.0)) or 0.0)
        if score is not None
        else float(getattr(validation_result, "score", 0.0) or 0.0)
    )
    resolved = bool(getattr(score, "resolved", getattr(validation_result, "success", False)))
    partial = bool(getattr(score, "partial", getattr(validation_result, "partial_success", False)))
    unresolved_gap = _clamp(1.0 - overall_score, 0.0, 1.0)
    regression_rate = len(broken_tests) / max(1, total_pass_to_pass)

    completion_threshold = _clamp(
        0.72 + (unresolved_gap * 0.16) + (regression_rate * 0.12) - (0.05 if resolved else 0.0),
        0.58,
        0.92,
    )
    progress_threshold = _clamp(
        completion_threshold - 0.18 - (0.03 if partial and not resolved else 0.0),
        0.35,
        completion_threshold,
    )
    evidence_threshold = _clamp(
        0.76 + (unresolved_gap * 0.10) + (regression_rate * 0.14),
        0.65,
        0.95,
    )
    truth_alignment_rate = _clamp(0.55 + (overall_score * 0.4) - (regression_rate * 0.1), 0.4, 0.99)

    post_change_results = getattr(validation_result, "post_change_results", None)
    payload_metadata = dict(metadata_payload)
    payload_metadata.update(
        {
            "benchmark": "swe_bench",
            "provider": provider,
            "model": model,
            "prompt_candidate_hash": prompt_candidate_hash,
            "section_name": section_name,
            "prompt_section_name": section_name,
            "vertical": "coding",
            "truth_alignment_rate": round(truth_alignment_rate, 4),
            "task_count": total_validated_tests,
            "validation_summary": {
                "success": bool(getattr(validation_result, "success", False)),
                "partial_success": bool(getattr(validation_result, "partial_success", False)),
                "fail_to_pass_total": total_fail_to_pass,
                "fail_to_pass_fixed": len(fixed_tests),
                "pass_to_pass_total": total_pass_to_pass,
                "pass_to_pass_broken": len(broken_tests),
                "post_change_total": int(getattr(post_change_results, "total", 0) or 0),
                "post_change_passed": int(getattr(post_change_results, "passed", 0) or 0),
            },
            "score_summary": {
                "resolved": resolved,
                "partial": partial,
                "fail_to_pass_score": round(fail_to_pass_score, 4),
                "pass_to_pass_score": round(pass_to_pass_score, 4),
                "overall_score": round(overall_score, 4),
            },
        }
    )

    return build_validated_session_feedback_payload(
        RuntimeEvaluationFeedback(
            completion_threshold=round(completion_threshold, 4),
            enhanced_progress_threshold=round(progress_threshold, 4),
            minimum_supported_evidence_score=round(evidence_threshold, 4),
        ),
        scope=RuntimeEvaluationFeedbackScope(
            project=_coerce_optional_text(getattr(baseline, "repo", None)),
            provider=provider,
            model=model,
            task_type="edit",
            benchmark="swe_bench",
            vertical="coding",
            workflow="evaluation_orchestrator",
            tags=("agentic", "coding", "validated-session"),
        ),
        validation_label="swe_bench_posthoc_validation",
        metadata=payload_metadata,
        source_result_path=source_result_path,
        saved_at=saved_at,
    )


def _extract_value(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _extract_mapping(value: Any, key: str) -> dict[str, Any]:
    raw_value = _extract_value(value, key, {})
    return dict(raw_value) if isinstance(raw_value, Mapping) else {}


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


def _coerce_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        items = value
    else:
        items = [value]
    return [str(item).strip() for item in items if str(item).strip()]


def _normalized_status(value: Any) -> str:
    raw_status = _extract_value(value, "status", "")
    return str(getattr(raw_status, "value", raw_status or "")).strip().lower()


def _validated_requirement_count(*items: list[str]) -> int:
    return max(1, sum(len(item) for item in items))


def _sanitize_runtime_feedback_tag(tag: str) -> str:
    return re.sub(r"[^a-z0-9_.-]+", "-", tag.strip().lower()).strip("-")


def _build_coverage_validated_session_feedback_payload(
    evaluation_result: Any,
    *,
    validation_label: str,
    benchmark_default: str,
    vertical: str,
    task_type: str,
    workflow: str,
    primary_coverage_key: str,
    secondary_coverage_key: str,
    primary_matched_key: str,
    primary_missing_key: str,
    secondary_matched_key: str,
    secondary_missing_key: str,
    blocked_hits_key: str,
    primary_weight: float,
    secondary_weight: float,
    source_result_path: Optional[Path] = None,
    saved_at: Optional[datetime] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    """Build validated session-truth payloads for coverage-based evaluators."""
    failure_details = _extract_mapping(evaluation_result, "failure_details")
    completion_score = _coerce_float(_extract_value(evaluation_result, "completion_score"))
    primary_coverage = _coerce_float(failure_details.get(primary_coverage_key))
    secondary_coverage = _coerce_float(failure_details.get(secondary_coverage_key))

    if (
        primary_coverage is None
        and secondary_coverage is None
        and (completion_score is None or completion_score <= 0.0)
    ):
        return None

    if primary_coverage is None:
        primary_coverage = completion_score or 0.0
    if secondary_coverage is None:
        secondary_coverage = completion_score or 0.0

    combined_completion = (
        completion_score
        if completion_score is not None and completion_score > 0.0
        else (primary_coverage * primary_weight) + (secondary_coverage * secondary_weight)
    )
    combined_completion = _clamp(combined_completion, 0.0, 1.0)
    primary_gap = _clamp(1.0 - primary_coverage, 0.0, 1.0)
    secondary_gap = _clamp(1.0 - secondary_coverage, 0.0, 1.0)
    blocked_hits = _coerce_text_list(failure_details.get(blocked_hits_key))
    blocked_penalty = min(0.18, len(blocked_hits) * 0.08)
    status = _normalized_status(evaluation_result)
    passed = status == "passed"

    completion_threshold = _clamp(
        0.58
        + ((1.0 - combined_completion) * 0.18)
        + (primary_gap * 0.11)
        + (secondary_gap * 0.08)
        + blocked_penalty
        - (0.06 if passed else 0.0),
        0.5,
        0.92,
    )
    progress_threshold = _clamp(
        completion_threshold - 0.17 - (0.03 if not passed and combined_completion < 0.6 else 0.0),
        0.35,
        completion_threshold,
    )
    evidence_threshold = _clamp(
        0.68 + (primary_gap * 0.08) + (secondary_gap * 0.06) + blocked_penalty,
        0.6,
        0.95,
    )

    matched_primary = _coerce_text_list(failure_details.get(primary_matched_key))
    missing_primary = _coerce_text_list(failure_details.get(primary_missing_key))
    matched_secondary = _coerce_text_list(failure_details.get(secondary_matched_key))
    missing_secondary = _coerce_text_list(failure_details.get(secondary_missing_key))
    task_count = _validated_requirement_count(
        matched_primary,
        missing_primary,
        matched_secondary,
        missing_secondary,
    )

    total_tasks = _coerce_int(_extract_value(evaluation_result, "total_tasks"))
    passed_tasks = _coerce_int(_extract_value(evaluation_result, "passed_tasks"))
    failed_tasks = _coerce_int(_extract_value(evaluation_result, "failed_tasks"))
    if total_tasks and passed_tasks is not None:
        truth_alignment_rate = _clamp(
            (passed_tasks / max(1, total_tasks)) - (blocked_penalty * 0.15),
            0.4,
            0.99,
        )
    else:
        truth_alignment_rate = _clamp(
            0.58 + (combined_completion * 0.32) - (blocked_penalty * 0.25),
            0.4,
            0.99,
        )

    dataset_metadata = _extract_value(evaluation_result, "dataset_metadata")
    if not isinstance(dataset_metadata, Mapping):
        dataset_metadata = {}
    benchmark = (
        _coerce_optional_text(_extract_value(evaluation_result, "benchmark")) or benchmark_default
    )
    provider = _coerce_optional_text(_extract_value(evaluation_result, "provider"))
    model = _coerce_optional_text(_extract_value(evaluation_result, "model"))
    prompt_candidate_hash = _coerce_optional_text(
        _extract_value(evaluation_result, "prompt_candidate_hash")
    )
    section_name = _coerce_optional_text(
        _extract_value(evaluation_result, "section_name")
        or _extract_value(evaluation_result, "prompt_section_name")
    )
    project = _coerce_optional_text(_extract_value(evaluation_result, "project"))
    task_id = _coerce_optional_text(_extract_value(evaluation_result, "task_id"))
    failure_category = _extract_value(evaluation_result, "failure_category")
    failure_category_value = _coerce_optional_text(
        getattr(failure_category, "value", failure_category)
    )
    raw_tags = _coerce_text_list(_extract_value(evaluation_result, "tags"))
    scope_tags = tuple(
        tag
        for tag in dict.fromkeys(
            [
                "agentic",
                vertical,
                "validated-session",
                *(
                    _sanitize_runtime_feedback_tag(tag)
                    for tag in raw_tags
                    if _sanitize_runtime_feedback_tag(tag)
                ),
            ]
        )
        if tag
    )

    payload_metadata = dict(metadata or {})
    payload_metadata.update(
        {
            "benchmark": benchmark,
            "provider": provider,
            "model": model,
            "prompt_candidate_hash": prompt_candidate_hash,
            "section_name": section_name,
            "dataset_metadata": dict(dataset_metadata),
            "truth_alignment_rate": round(truth_alignment_rate, 4),
            "task_count": task_count,
            "task_id": task_id,
            "status": status or None,
            "failure_category": failure_category_value,
            "validation_summary": {
                primary_coverage_key: round(primary_coverage, 4),
                secondary_coverage_key: round(secondary_coverage, 4),
                primary_matched_key: matched_primary,
                primary_missing_key: missing_primary,
                secondary_matched_key: matched_secondary,
                secondary_missing_key: missing_secondary,
                blocked_hits_key: blocked_hits,
                "completion_score": round(combined_completion, 4),
                "status": status or None,
            },
        }
    )
    if total_tasks is not None:
        payload_metadata["suite_summary"] = {
            "total_tasks": total_tasks,
            "passed_tasks": passed_tasks or 0,
            "failed_tasks": failed_tasks or 0,
        }

    return build_validated_session_feedback_payload(
        RuntimeEvaluationFeedback(
            completion_threshold=round(completion_threshold, 4),
            enhanced_progress_threshold=round(progress_threshold, 4),
            minimum_supported_evidence_score=round(evidence_threshold, 4),
        ),
        scope=RuntimeEvaluationFeedbackScope(
            project=project,
            provider=provider,
            model=model,
            task_type=task_type,
            benchmark=benchmark,
            vertical=vertical,
            workflow=workflow,
            tags=scope_tags,
        ),
        validation_label=validation_label,
        metadata=payload_metadata,
        source_result_path=source_result_path,
        saved_at=saved_at,
    )


def build_browser_validated_session_feedback_payload(
    evaluation_result: Any,
    *,
    source_result_path: Optional[Path] = None,
    saved_at: Optional[datetime] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    """Build validated session-truth payloads from browser-task post-hoc validation."""
    return _build_coverage_validated_session_feedback_payload(
        evaluation_result,
        validation_label="browser_posthoc_validation",
        benchmark_default="browser_task",
        vertical="browser",
        task_type="interaction",
        workflow="evaluation_harness",
        primary_coverage_key="action_coverage",
        secondary_coverage_key="answer_coverage",
        primary_matched_key="matched_actions",
        primary_missing_key="missing_actions",
        secondary_matched_key="matched_answer_phrases",
        secondary_missing_key="missing_answer_phrases",
        blocked_hits_key="forbidden_action_hits",
        primary_weight=0.65,
        secondary_weight=0.35,
        source_result_path=source_result_path,
        saved_at=saved_at,
        metadata=metadata,
    )


def build_deep_research_validated_session_feedback_payload(
    evaluation_result: Any,
    *,
    source_result_path: Optional[Path] = None,
    saved_at: Optional[datetime] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    """Build validated session-truth payloads from deep-research post-hoc validation."""
    return _build_coverage_validated_session_feedback_payload(
        evaluation_result,
        validation_label="deep_research_posthoc_validation",
        benchmark_default="dr3_eval",
        vertical="research",
        task_type="analysis",
        workflow="evaluation_harness",
        primary_coverage_key="claim_coverage",
        secondary_coverage_key="citation_coverage",
        primary_matched_key="matched_claims",
        primary_missing_key="missing_claims",
        secondary_matched_key="matched_citations",
        secondary_missing_key="missing_citations",
        blocked_hits_key="forbidden_claim_hits",
        primary_weight=0.6,
        secondary_weight=0.4,
        source_result_path=source_result_path,
        saved_at=saved_at,
        metadata=metadata,
    )


def _extract_feedback_payload(value: Any) -> Optional[dict[str, Any]]:
    """Normalize direct feedback payloads and evaluation result JSON records."""
    if not isinstance(value, Mapping):
        return None
    if "runtime_evaluation_feedback" in value:
        nested = value.get("runtime_evaluation_feedback")
        return dict(nested) if isinstance(nested, Mapping) else None
    if any(
        key in value
        for key in (
            "completion_threshold",
            "enhanced_progress_threshold",
            "minimum_supported_evidence_score",
        )
    ):
        return dict(value)
    return None


def _load_feedback_payload_file(path: Path) -> Optional[dict[str, Any]]:
    """Load a normalized feedback payload from a result or direct-feedback file."""
    try:
        raw_payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    payload = _extract_feedback_payload(raw_payload)
    if payload is None:
        return None
    feedback = RuntimeEvaluationFeedback.from_dict(payload)
    source_result_path = payload.get("metadata", {}).get("source_result_path")
    if source_result_path is None and path.name.startswith("eval_"):
        source_result_path = path
    saved_at = _parse_timestamp(payload.get("metadata", {}).get("saved_at"))
    if saved_at is None:
        try:
            saved_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        except OSError:
            saved_at = None
    return build_runtime_evaluation_feedback_payload(
        feedback,
        source_result_path=Path(source_result_path) if source_result_path else None,
        saved_at=saved_at,
    )


def _is_validated_feedback_payload(payload: Mapping[str, Any]) -> bool:
    metadata = dict(payload.get("metadata") or {})
    if metadata.get("validated_evaluation_truth") is True:
        return True
    return metadata.get("source") in VALIDATED_RUNTIME_FEEDBACK_SOURCES


def _resolve_feedback_directory(target_path: Path) -> Optional[Path]:
    if target_path.exists() and target_path.is_dir():
        return target_path
    if target_path.name == RUNTIME_EVALUATION_FEEDBACK_FILENAME:
        return target_path.parent
    return None


def _load_feedback_payloads_from_directory(base_dir: Path) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for result_path in sorted(base_dir.glob("eval_*.json")):
        payload = _load_feedback_payload_file(result_path)
        if payload is not None:
            payloads.append(payload)
    if payloads:
        return payloads

    canonical_path = base_dir / RUNTIME_EVALUATION_FEEDBACK_FILENAME
    if canonical_path.exists():
        payload = _load_feedback_payload_file(canonical_path)
        if payload is not None:
            payloads.append(payload)
    return payloads


def _source_trust_weight(source: Optional[str]) -> float:
    return RUNTIME_FEEDBACK_SOURCE_TRUST.get(str(source or ""), 0.75)


def _scope_similarity_weight(
    candidate_scope: Any,
    target_scope: Any,
) -> float:
    selected_scope = RuntimeEvaluationFeedbackScope.from_value(target_scope)
    if selected_scope.is_empty():
        return 1.0

    artifact_scope = RuntimeEvaluationFeedbackScope.from_value(candidate_scope)
    considered_weight = 0.0
    score = 0.0

    for field_name, field_weight in SCOPE_FIELD_WEIGHTS.items():
        selected_value = _normalized_scope_token(getattr(selected_scope, field_name))
        if selected_value is None:
            continue
        considered_weight += field_weight
        artifact_value = _normalized_scope_token(getattr(artifact_scope, field_name))
        if artifact_value is None:
            score += field_weight * 0.45
        elif artifact_value == selected_value:
            score += field_weight
        elif field_name in {"project", "model", "task_type"}:
            score += field_weight * 0.05
        else:
            score += field_weight * 0.20

    selected_tags = {
        _normalized_scope_token(tag)
        for tag in selected_scope.tags
        if _normalized_scope_token(tag) is not None
    }
    if selected_tags:
        considered_weight += 1.0
        artifact_tags = {
            _normalized_scope_token(tag)
            for tag in artifact_scope.tags
            if _normalized_scope_token(tag) is not None
        }
        overlap = len(selected_tags & artifact_tags) / max(1, len(selected_tags))
        score += max(0.1, overlap)

    if considered_weight == 0:
        return 1.0
    return _clamp(0.2 + (score / considered_weight) * 1.25, 0.1, 1.5)


def _feedback_weight(
    payload: Mapping[str, Any],
    reference_time: datetime,
    *,
    target_scope: Optional[RuntimeEvaluationFeedbackScope] = None,
) -> float:
    metadata = dict(payload.get("metadata") or {})
    is_live_topology_feedback = (
        metadata.get("source") == SESSION_TOPOLOGY_RUNTIME_FEEDBACK_SOURCE
        or metadata.get("topology_feedback_only") is True
    )
    saved_at = _parse_timestamp(metadata.get("saved_at")) or reference_time
    age_days = max(0.0, (reference_time - saved_at).total_seconds() / 86400.0)
    half_life_days = (
        SESSION_TOPOLOGY_FRESHNESS_HALF_LIFE_DAYS
        if is_live_topology_feedback
        else FRESHNESS_HALF_LIFE_DAYS
    )
    recency_weight = math.pow(0.5, age_days / max(half_life_days, 1e-9))
    truth_alignment = _clamp(
        float(metadata.get("truth_alignment_rate", 0.75) or 0.75),
        0.25,
        1.0,
    )
    task_count = max(1.0, float(metadata.get("task_count", 1.0) or 1.0))
    coverage_weight = min(3.0, 0.75 + (math.sqrt(task_count) / 4.0))
    trust_weight = _source_trust_weight(metadata.get("source"))
    scope_weight = _scope_similarity_weight(metadata.get("scope"), target_scope)
    minimum_weight = 0.005 if is_live_topology_feedback else 0.05
    return max(
        minimum_weight,
        recency_weight * truth_alignment * coverage_weight * trust_weight * scope_weight,
    )


def _select_metadata(values: list[Any], *, fallback: Any = None) -> Any:
    filtered = [value for value in values if value not in (None, {}, [])]
    if not filtered:
        return fallback
    unique = []
    for value in filtered:
        if value not in unique:
            unique.append(value)
    return unique[0] if len(unique) == 1 else fallback


def _average_mapping_from_totals(
    counts: Mapping[str, Any],
    totals: Mapping[str, Any],
) -> dict[str, float]:
    averages: dict[str, float] = {}
    for key, count_value in counts.items():
        label = str(key).strip()
        if not label:
            continue
        try:
            count = float(count_value)
            total = float(totals.get(label, 0.0))
        except (TypeError, ValueError):
            continue
        if count <= 0.0:
            continue
        averages[label] = round(total / count, 4)
    return averages


def _selection_policy_reward_delta(averages: Mapping[str, Any]) -> Optional[float]:
    try:
        learned = float(averages["learned_close_override"])
        heuristic = float(averages["heuristic"])
    except (KeyError, TypeError, ValueError):
        return None
    return round(learned - heuristic, 4)


def _model_family_token(value: Any) -> Optional[str]:
    text = _coerce_optional_text(value)
    if text is None:
        return None
    normalized = text.strip().lower()
    parts = [part for part in re.split(r"[-_/:\s]+", normalized) if part]
    if not parts:
        return None
    return parts[0]


def _selection_policy_scope_label(
    metadata: Mapping[str, Any],
    *,
    dimension: str,
) -> Optional[str]:
    scope = RuntimeEvaluationFeedbackScope.from_value(metadata.get("scope"))
    if dimension == "task_type":
        return _normalized_scope_token(scope.task_type or metadata.get("task_type"))
    if dimension == "provider":
        return _normalized_scope_token(scope.provider or metadata.get("provider"))
    if dimension == "model_family":
        return _model_family_token(scope.model or metadata.get("model"))
    return None


def _merge_selection_policy_scope_bucket(
    scope_metrics: dict[str, dict[str, dict[str, Any]]],
    *,
    dimension: str,
    label: str,
    policy_counts: Mapping[str, Any],
    policy_reward_totals: Mapping[str, Any],
    policy_optimization_counts: Mapping[str, Any],
    policy_optimization_reward_totals: Mapping[str, Any],
    policy_feasible_counts: Mapping[str, Any],
) -> None:
    """Merge one scoped policy-metrics bucket into the aggregate structure."""
    bucket = scope_metrics[dimension].setdefault(
        label,
        {
            "policy_counts": {},
            "policy_reward_totals": {},
            "policy_optimization_counts": {},
            "policy_optimization_reward_totals": {},
            "policy_feasible_counts": {},
        },
    )
    for policy, count_value in policy_counts.items():
        policy_name = str(policy).strip()
        if not policy_name:
            continue
        try:
            count = float(count_value)
        except (TypeError, ValueError):
            continue
        if count <= 0.0:
            continue
        bucket["policy_counts"][policy_name] = round(
            bucket["policy_counts"].get(policy_name, 0.0) + count,
            4,
        )
    for policy, reward_value in policy_reward_totals.items():
        policy_name = str(policy).strip()
        if not policy_name:
            continue
        try:
            reward_total = float(reward_value)
        except (TypeError, ValueError):
            continue
        if reward_total <= 0.0:
            continue
        bucket["policy_reward_totals"][policy_name] = round(
            bucket["policy_reward_totals"].get(policy_name, 0.0) + reward_total,
            4,
        )
    for policy, count_value in policy_optimization_counts.items():
        policy_name = str(policy).strip()
        if not policy_name:
            continue
        try:
            count = float(count_value)
        except (TypeError, ValueError):
            continue
        if count <= 0.0:
            continue
        bucket["policy_optimization_counts"][policy_name] = round(
            bucket["policy_optimization_counts"].get(policy_name, 0.0) + count,
            4,
        )
    for policy, reward_value in policy_optimization_reward_totals.items():
        policy_name = str(policy).strip()
        if not policy_name:
            continue
        try:
            reward_total = float(reward_value)
        except (TypeError, ValueError):
            continue
        if reward_total <= 0.0:
            continue
        bucket["policy_optimization_reward_totals"][policy_name] = round(
            bucket["policy_optimization_reward_totals"].get(policy_name, 0.0) + reward_total,
            4,
        )
    for policy, count_value in policy_feasible_counts.items():
        policy_name = str(policy).strip()
        if not policy_name:
            continue
        try:
            count = float(count_value)
        except (TypeError, ValueError):
            continue
        if count <= 0.0:
            continue
        bucket["policy_feasible_counts"][policy_name] = round(
            bucket["policy_feasible_counts"].get(policy_name, 0.0) + count,
            4,
        )


def _build_selection_policy_scope_metrics(
    metadata_list: list[dict[str, Any]],
    _weights: list[float],
) -> dict[str, dict[str, dict[str, Any]]]:
    scope_metrics: dict[str, dict[str, dict[str, Any]]] = {
        "task_type": {},
        "provider": {},
        "model_family": {},
    }

    for metadata in metadata_list:
        explicit_bucket_keys: set[tuple[str, str]] = set()
        explicit_scope_metrics = metadata.get("topology_selection_policy_scope_metrics") or {}
        if isinstance(explicit_scope_metrics, Mapping):
            for dimension, entries in explicit_scope_metrics.items():
                if dimension not in scope_metrics or not isinstance(entries, Mapping):
                    continue
                for label, bucket in entries.items():
                    normalized_label = _normalized_scope_token(label)
                    if normalized_label is None or not isinstance(bucket, Mapping):
                        continue
                    explicit_bucket_keys.add((dimension, normalized_label))
                    _merge_selection_policy_scope_bucket(
                        scope_metrics,
                        dimension=dimension,
                        label=normalized_label,
                        policy_counts=dict(bucket.get("policy_counts") or {}),
                        policy_reward_totals=dict(bucket.get("policy_reward_totals") or {}),
                        policy_optimization_counts=dict(
                            bucket.get("policy_optimization_counts") or {}
                        ),
                        policy_optimization_reward_totals=dict(
                            bucket.get("policy_optimization_reward_totals") or {}
                        ),
                        policy_feasible_counts=dict(bucket.get("policy_feasible_counts") or {}),
                    )

        selection_policy_counts = metadata.get("topology_selection_policies") or {}
        selection_policy_reward_totals = (
            metadata.get("topology_selection_policy_reward_totals") or {}
        )
        selection_policy_optimization_counts = (
            metadata.get("topology_selection_policy_optimization_counts") or {}
        )
        selection_policy_optimization_reward_totals = (
            metadata.get("topology_selection_policy_optimization_reward_totals") or {}
        )
        selection_policy_feasible_counts = (
            metadata.get("topology_selection_policy_feasible_counts") or {}
        )
        if not isinstance(selection_policy_counts, Mapping):
            selection_policy_counts = {}
        if not isinstance(selection_policy_reward_totals, Mapping):
            selection_policy_reward_totals = {}
        if not isinstance(selection_policy_optimization_counts, Mapping):
            selection_policy_optimization_counts = {}
        if not isinstance(selection_policy_optimization_reward_totals, Mapping):
            selection_policy_optimization_reward_totals = {}
        if not isinstance(selection_policy_feasible_counts, Mapping):
            selection_policy_feasible_counts = {}
        if (
            not selection_policy_counts
            and not selection_policy_reward_totals
            and not selection_policy_optimization_counts
            and not selection_policy_optimization_reward_totals
            and not selection_policy_feasible_counts
        ):
            continue

        for dimension in scope_metrics:
            label = _selection_policy_scope_label(metadata, dimension=dimension)
            if label is None:
                continue
            if (dimension, label) in explicit_bucket_keys:
                continue
            _merge_selection_policy_scope_bucket(
                scope_metrics,
                dimension=dimension,
                label=label,
                policy_counts=selection_policy_counts,
                policy_reward_totals=selection_policy_reward_totals,
                policy_optimization_counts=selection_policy_optimization_counts,
                policy_optimization_reward_totals=selection_policy_optimization_reward_totals,
                policy_feasible_counts=selection_policy_feasible_counts,
            )

    for dimension, entries in scope_metrics.items():
        for label, bucket in entries.items():
            policy_counts = dict(bucket.get("policy_counts") or {})
            policy_reward_totals = dict(bucket.get("policy_reward_totals") or {})
            policy_optimization_counts = dict(bucket.get("policy_optimization_counts") or {})
            policy_optimization_reward_totals = dict(
                bucket.get("policy_optimization_reward_totals") or {}
            )
            policy_feasible_counts = dict(bucket.get("policy_feasible_counts") or {})
            avg_reward_by_policy = _average_mapping_from_totals(
                policy_counts,
                policy_reward_totals,
            )
            bucket["policy_counts"] = policy_counts
            bucket["policy_reward_totals"] = policy_reward_totals
            bucket["policy_optimization_counts"] = policy_optimization_counts
            bucket["policy_optimization_reward_totals"] = policy_optimization_reward_totals
            bucket["policy_feasible_counts"] = policy_feasible_counts
            bucket["avg_reward_by_policy"] = avg_reward_by_policy
            bucket["learned_override_reward_delta"] = _selection_policy_reward_delta(
                avg_reward_by_policy
            )
            bucket["avg_optimization_reward_by_policy"] = _average_mapping_from_totals(
                policy_optimization_counts,
                policy_optimization_reward_totals,
            )
            bucket["feasibility_rate_by_policy"] = {
                policy: round(
                    float(policy_feasible_counts.get(policy, 0.0)) / max(1.0, float(count_value)),
                    4,
                )
                for policy, count_value in policy_optimization_counts.items()
                if float(count_value) > 0.0
            }
            bucket["learned_override_optimization_reward_delta"] = _selection_policy_reward_delta(
                bucket["avg_optimization_reward_by_policy"]
            )
            bucket["learned_override_feasibility_delta"] = _selection_policy_reward_delta(
                bucket["feasibility_rate_by_policy"]
            )
    return scope_metrics


def _merge_team_worktree_scope_bucket(
    scope_metrics: dict[str, dict[str, dict[str, Any]]],
    *,
    dimension: str,
    label: str,
    task_count: Any,
    coverage: Any,
    worktree_plan_count: Any,
    worktree_materialized_count: Any,
    worktree_dry_run_count: Any,
    cleanup_task_count: Any,
    cleanup_error_task_count: Any,
    merge_conflict_task_count: Any,
    low_risk_task_count: Any,
    medium_risk_task_count: Any,
    high_risk_task_count: Any,
    avg_team_assignments: Any,
    avg_team_scoped_members: Any,
    avg_team_members_with_changes: Any,
    avg_team_changed_file_count: Any,
    formation_distribution: Mapping[str, Any],
    merge_risk_distribution: Mapping[str, Any],
    weight: float = 1.0,
) -> None:
    """Merge one scoped team/worktree bucket into the aggregate structure."""
    bucket = scope_metrics[dimension].setdefault(
        label,
        {
            "tasks_with_team_feedback": 0.0,
            "team_feedback_coverage": 0.0,
            "team_worktree_plan_count": 0.0,
            "team_worktree_materialized_count": 0.0,
            "team_worktree_dry_run_count": 0.0,
            "team_cleanup_task_count": 0.0,
            "team_cleanup_error_task_count": 0.0,
            "team_merge_conflict_task_count": 0.0,
            "team_low_risk_task_count": 0.0,
            "team_medium_risk_task_count": 0.0,
            "team_high_risk_task_count": 0.0,
            "avg_team_assignments": 0.0,
            "avg_team_scoped_members": 0.0,
            "avg_team_members_with_changes": 0.0,
            "avg_team_changed_file_count": 0.0,
            "team_formations": {},
            "team_merge_risk_levels": {},
            "_coverage_weight": 0.0,
            "_avg_weight": 0.0,
            "_avg_team_assignments_total": 0.0,
            "_avg_team_scoped_members_total": 0.0,
            "_avg_team_members_with_changes_total": 0.0,
            "_avg_team_changed_file_count_total": 0.0,
        },
    )

    try:
        resolved_weight = max(0.0, float(weight))
    except (TypeError, ValueError):
        resolved_weight = 0.0
    if resolved_weight <= 0.0:
        return

    def _coerce_non_negative(value: Any) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, numeric)

    resolved_task_count = _coerce_non_negative(task_count)
    count_weight = max(1.0, resolved_task_count) * resolved_weight

    bucket["tasks_with_team_feedback"] = round(
        bucket["tasks_with_team_feedback"] + (resolved_task_count * resolved_weight),
        4,
    )
    bucket["_coverage_weight"] = round(bucket["_coverage_weight"] + count_weight, 4)
    bucket["team_feedback_coverage"] = round(
        bucket["team_feedback_coverage"] + (_coerce_non_negative(coverage) * count_weight),
        4,
    )
    for field_name, field_value in (
        ("team_worktree_plan_count", worktree_plan_count),
        ("team_worktree_materialized_count", worktree_materialized_count),
        ("team_worktree_dry_run_count", worktree_dry_run_count),
        ("team_cleanup_task_count", cleanup_task_count),
        ("team_cleanup_error_task_count", cleanup_error_task_count),
        ("team_merge_conflict_task_count", merge_conflict_task_count),
        ("team_low_risk_task_count", low_risk_task_count),
        ("team_medium_risk_task_count", medium_risk_task_count),
        ("team_high_risk_task_count", high_risk_task_count),
    ):
        bucket[field_name] = round(
            bucket[field_name] + (_coerce_non_negative(field_value) * resolved_weight),
            4,
        )

    bucket["_avg_weight"] = round(bucket["_avg_weight"] + count_weight, 4)
    for total_field, avg_value in (
        ("_avg_team_assignments_total", avg_team_assignments),
        ("_avg_team_scoped_members_total", avg_team_scoped_members),
        ("_avg_team_members_with_changes_total", avg_team_members_with_changes),
        ("_avg_team_changed_file_count_total", avg_team_changed_file_count),
    ):
        bucket[total_field] = round(
            bucket[total_field] + (_coerce_non_negative(avg_value) * count_weight),
            4,
        )

    for field_name, distribution in (
        ("team_formations", formation_distribution),
        ("team_merge_risk_levels", merge_risk_distribution),
    ):
        if not isinstance(distribution, Mapping):
            continue
        target = dict(bucket.get(field_name) or {})
        for key, value in distribution.items():
            normalized_key = str(key).strip()
            if not normalized_key:
                continue
            numeric = _coerce_non_negative(value)
            if numeric <= 0.0:
                continue
            target[normalized_key] = round(
                float(target.get(normalized_key, 0.0)) + (numeric * resolved_weight),
                4,
            )
        bucket[field_name] = target


def _build_team_worktree_scope_metrics(
    metadata_list: list[dict[str, Any]],
    _weights: list[float],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Aggregate scoped team/worktree routing metrics for runtime policy reuse."""
    scope_metrics: dict[str, dict[str, dict[str, Any]]] = {
        "task_type": {},
        "provider": {},
        "model_family": {},
    }

    for metadata in metadata_list:
        explicit_bucket_keys: set[tuple[str, str]] = set()
        explicit_scope_metrics = metadata.get("team_worktree_scope_metrics") or {}
        if isinstance(explicit_scope_metrics, Mapping):
            for dimension, entries in explicit_scope_metrics.items():
                if dimension not in scope_metrics or not isinstance(entries, Mapping):
                    continue
                for label, bucket in entries.items():
                    normalized_label = _normalized_scope_token(label)
                    if normalized_label is None or not isinstance(bucket, Mapping):
                        continue
                    explicit_bucket_keys.add((dimension, normalized_label))
                    _merge_team_worktree_scope_bucket(
                        scope_metrics,
                        dimension=dimension,
                        label=normalized_label,
                        task_count=bucket.get("tasks_with_team_feedback"),
                        coverage=bucket.get("team_feedback_coverage"),
                        worktree_plan_count=bucket.get("team_worktree_plan_count"),
                        worktree_materialized_count=bucket.get("team_worktree_materialized_count"),
                        worktree_dry_run_count=bucket.get("team_worktree_dry_run_count"),
                        cleanup_task_count=bucket.get("team_cleanup_task_count"),
                        cleanup_error_task_count=bucket.get("team_cleanup_error_task_count"),
                        merge_conflict_task_count=bucket.get("team_merge_conflict_task_count"),
                        low_risk_task_count=bucket.get("team_low_risk_task_count"),
                        medium_risk_task_count=bucket.get("team_medium_risk_task_count"),
                        high_risk_task_count=bucket.get("team_high_risk_task_count"),
                        avg_team_assignments=bucket.get("avg_team_assignments"),
                        avg_team_scoped_members=bucket.get("avg_team_scoped_members"),
                        avg_team_members_with_changes=bucket.get("avg_team_members_with_changes"),
                        avg_team_changed_file_count=bucket.get("avg_team_changed_file_count"),
                        formation_distribution=dict(bucket.get("team_formations") or {}),
                        merge_risk_distribution=dict(bucket.get("team_merge_risk_levels") or {}),
                    )

        if not any(
            metadata.get(field_name) not in (None, {}, [])
            for field_name in (
                "tasks_with_team_feedback",
                "team_feedback_coverage",
                "team_worktree_plan_count",
                "team_formations",
                "team_merge_risk_levels",
            )
        ):
            continue

        for dimension in scope_metrics:
            label = _selection_policy_scope_label(metadata, dimension=dimension)
            if label is None or (dimension, label) in explicit_bucket_keys:
                continue
            _merge_team_worktree_scope_bucket(
                scope_metrics,
                dimension=dimension,
                label=label,
                task_count=metadata.get("tasks_with_team_feedback"),
                coverage=metadata.get("team_feedback_coverage"),
                worktree_plan_count=metadata.get("team_worktree_plan_count"),
                worktree_materialized_count=metadata.get("team_worktree_materialized_count"),
                worktree_dry_run_count=metadata.get("team_worktree_dry_run_count"),
                cleanup_task_count=metadata.get("team_cleanup_task_count"),
                cleanup_error_task_count=metadata.get("team_cleanup_error_task_count"),
                merge_conflict_task_count=metadata.get("team_merge_conflict_task_count"),
                low_risk_task_count=metadata.get("team_low_risk_task_count"),
                medium_risk_task_count=metadata.get("team_medium_risk_task_count"),
                high_risk_task_count=metadata.get("team_high_risk_task_count"),
                avg_team_assignments=metadata.get("avg_team_assignments"),
                avg_team_scoped_members=metadata.get("avg_team_scoped_members"),
                avg_team_members_with_changes=metadata.get("avg_team_members_with_changes"),
                avg_team_changed_file_count=metadata.get("avg_team_changed_file_count"),
                formation_distribution=dict(metadata.get("team_formations") or {}),
                merge_risk_distribution=dict(metadata.get("team_merge_risk_levels") or {}),
            )

    finalized: dict[str, dict[str, dict[str, Any]]] = {
        "task_type": {},
        "provider": {},
        "model_family": {},
    }
    for dimension, entries in scope_metrics.items():
        for label, bucket in entries.items():
            coverage_weight = max(float(bucket.get("_coverage_weight", 0.0) or 0.0), 1e-9)
            avg_weight = max(float(bucket.get("_avg_weight", 0.0) or 0.0), 1e-9)
            finalized[dimension][label] = {
                "tasks_with_team_feedback": round(
                    float(bucket.get("tasks_with_team_feedback", 0.0) or 0.0),
                    4,
                ),
                "team_feedback_coverage": round(
                    float(bucket.get("team_feedback_coverage", 0.0) or 0.0) / coverage_weight,
                    4,
                ),
                "team_worktree_plan_count": round(
                    float(bucket.get("team_worktree_plan_count", 0.0) or 0.0),
                    4,
                ),
                "team_worktree_materialized_count": round(
                    float(bucket.get("team_worktree_materialized_count", 0.0) or 0.0),
                    4,
                ),
                "team_worktree_dry_run_count": round(
                    float(bucket.get("team_worktree_dry_run_count", 0.0) or 0.0),
                    4,
                ),
                "team_cleanup_task_count": round(
                    float(bucket.get("team_cleanup_task_count", 0.0) or 0.0),
                    4,
                ),
                "team_cleanup_error_task_count": round(
                    float(bucket.get("team_cleanup_error_task_count", 0.0) or 0.0),
                    4,
                ),
                "team_merge_conflict_task_count": round(
                    float(bucket.get("team_merge_conflict_task_count", 0.0) or 0.0),
                    4,
                ),
                "team_low_risk_task_count": round(
                    float(bucket.get("team_low_risk_task_count", 0.0) or 0.0),
                    4,
                ),
                "team_medium_risk_task_count": round(
                    float(bucket.get("team_medium_risk_task_count", 0.0) or 0.0),
                    4,
                ),
                "team_high_risk_task_count": round(
                    float(bucket.get("team_high_risk_task_count", 0.0) or 0.0),
                    4,
                ),
                "avg_team_assignments": round(
                    float(bucket.get("_avg_team_assignments_total", 0.0) or 0.0) / avg_weight,
                    4,
                ),
                "avg_team_scoped_members": round(
                    float(bucket.get("_avg_team_scoped_members_total", 0.0) or 0.0) / avg_weight,
                    4,
                ),
                "avg_team_members_with_changes": round(
                    float(bucket.get("_avg_team_members_with_changes_total", 0.0) or 0.0)
                    / avg_weight,
                    4,
                ),
                "avg_team_changed_file_count": round(
                    float(bucket.get("_avg_team_changed_file_count_total", 0.0) or 0.0)
                    / avg_weight,
                    4,
                ),
                "team_formations": dict(bucket.get("team_formations") or {}),
                "team_merge_risk_levels": dict(bucket.get("team_merge_risk_levels") or {}),
            }
    return finalized


def _distribution_agreement(distribution: Any) -> Optional[float]:
    if not isinstance(distribution, Mapping):
        return None
    total = 0.0
    dominant = 0.0
    for value in distribution.values():
        try:
            count = float(value)
        except (TypeError, ValueError):
            continue
        if count <= 0.0:
            continue
        total += count
        dominant = max(dominant, count)
    if total <= 0.0:
        return None
    return round(dominant / total, 4)


def _topology_conflict_score(
    *,
    action_agreement: Optional[float],
    topology_agreement: Optional[float],
    provider_agreement: Optional[float],
    formation_agreement: Optional[float],
) -> Optional[float]:
    weighted_agreements = [
        (action_agreement, 0.35),
        (topology_agreement, 0.30),
        (provider_agreement, 0.20),
        (formation_agreement, 0.15),
    ]
    usable = [(value, weight) for value, weight in weighted_agreements if value is not None]
    if not usable:
        return None
    weighted_consensus = sum(value * weight for value, weight in usable) / sum(
        weight for _, weight in usable
    )
    return round(_clamp(1.0 - weighted_consensus, 0.0, 1.0), 4)


def _aggregate_feedback_payloads(
    payloads: list[dict[str, Any]],
    *,
    scope: Optional[Any] = None,
) -> Optional[RuntimeEvaluationFeedback]:
    if not payloads:
        return None

    validated_payloads = [
        payload for payload in payloads if _is_validated_feedback_payload(payload)
    ]
    excluded_count = len(payloads) - len(validated_payloads)
    if not validated_payloads:
        return None

    selected_scope = RuntimeEvaluationFeedbackScope.from_value(scope)
    saved_at_values = [
        _parse_timestamp(dict(payload.get("metadata") or {}).get("saved_at"))
        for payload in validated_payloads
    ]
    resolved_saved_at_values = [value for value in saved_at_values if value is not None]
    reference_time = max(resolved_saved_at_values, default=datetime.now(timezone.utc))
    weights = [
        _feedback_weight(payload, reference_time, target_scope=selected_scope)
        for payload in validated_payloads
    ]

    def weighted_average(field_name: str) -> Optional[float]:
        weighted_pairs = [
            (float(payload[field_name]), weight)
            for payload, weight in zip(validated_payloads, weights)
            if payload.get(field_name) is not None
        ]
        if not weighted_pairs:
            return None
        numerator = sum(value * weight for value, weight in weighted_pairs)
        denominator = sum(weight for _, weight in weighted_pairs)
        return round(numerator / max(denominator, 1e-9), 4)

    def weighted_metadata_average(field_name: str) -> Optional[float]:
        weighted_pairs = [
            (float(metadata[field_name]), weight)
            for metadata, weight in zip(metadata_list, weights)
            if metadata.get(field_name) is not None
        ]
        if not weighted_pairs:
            return None
        numerator = sum(value * weight for value, weight in weighted_pairs)
        denominator = sum(weight for _, weight in weighted_pairs)
        return round(numerator / max(denominator, 1e-9), 4)

    def weighted_distribution(field_name: str) -> dict[str, float]:
        aggregated: dict[str, float] = {}
        for metadata, weight in zip(metadata_list, weights):
            distribution = metadata.get(field_name) or {}
            if not isinstance(distribution, Mapping):
                continue
            for key, value in distribution.items():
                label = str(key).strip()
                if not label:
                    continue
                try:
                    count = float(value)
                except (TypeError, ValueError):
                    continue
                aggregated[label] = round(aggregated.get(label, 0.0) + (count * weight), 4)
        return aggregated

    metadata_list = [dict(payload.get("metadata") or {}) for payload in validated_payloads]
    selection_policy_counts = weighted_distribution("topology_selection_policies")
    selection_policy_reward_totals = weighted_distribution(
        "topology_selection_policy_reward_totals"
    )
    optimization_gate_failures = weighted_distribution("optimization_gate_failures")
    selection_policy_optimization_counts = weighted_distribution(
        "topology_selection_policy_optimization_counts"
    )
    selection_policy_optimization_reward_totals = weighted_distribution(
        "topology_selection_policy_optimization_reward_totals"
    )
    selection_policy_feasible_counts = weighted_distribution(
        "topology_selection_policy_feasible_counts"
    )
    avg_reward_by_selection_policy = _average_mapping_from_totals(
        selection_policy_counts,
        selection_policy_reward_totals,
    )
    avg_optimization_reward_by_selection_policy = _average_mapping_from_totals(
        selection_policy_optimization_counts,
        selection_policy_optimization_reward_totals,
    )
    selection_policy_feasibility_rates = {
        policy: round(
            float(selection_policy_feasible_counts.get(policy, 0.0)) / max(1.0, float(count_value)),
            4,
        )
        for policy, count_value in selection_policy_optimization_counts.items()
        if float(count_value) > 0.0
    }
    selection_policy_scope_metrics = _build_selection_policy_scope_metrics(
        metadata_list,
        weights,
    )
    team_worktree_scope_metrics = _build_team_worktree_scope_metrics(
        metadata_list,
        weights,
    )
    scope_scores = [
        _scope_similarity_weight(metadata.get("scope"), selected_scope)
        for metadata in metadata_list
    ]
    freshest_payload = max(
        validated_payloads,
        key=lambda payload: _parse_timestamp(dict(payload.get("metadata") or {}).get("saved_at"))
        or datetime.min.replace(tzinfo=timezone.utc),
    )
    freshest_metadata = dict(freshest_payload.get("metadata") or {})
    benchmarks = [metadata.get("benchmark") for metadata in metadata_list]
    models = [metadata.get("model") for metadata in metadata_list]
    datasets = [metadata.get("dataset_metadata") for metadata in metadata_list]
    task_counts = [int(metadata.get("task_count", 0) or 0) for metadata in metadata_list]
    freshest_saved_at = max(resolved_saved_at_values, default=None)
    oldest_saved_at = min(resolved_saved_at_values, default=None)

    return RuntimeEvaluationFeedback(
        completion_threshold=weighted_average("completion_threshold"),
        enhanced_progress_threshold=weighted_average("enhanced_progress_threshold"),
        minimum_supported_evidence_score=weighted_average("minimum_supported_evidence_score"),
        metadata={
            "source": AGGREGATED_RUNTIME_FEEDBACK_SOURCE,
            "validated_evaluation_truth": True,
            "truth_validation_mode": "aggregated_validated_evaluation_truth",
            "aggregation_strategy": "recency_reliability_weighted",
            "aggregated_artifact_count": len(validated_payloads),
            "excluded_artifact_count": excluded_count,
            "validation_sources": sorted(
                {
                    metadata.get("source")
                    for metadata in metadata_list
                    if metadata.get("source") is not None
                }
            ),
            "benchmark": _select_metadata(benchmarks, fallback=freshest_metadata.get("benchmark")),
            "model": _select_metadata(models, fallback=freshest_metadata.get("model")),
            "dataset_metadata": _select_metadata(
                datasets,
                fallback=dict(freshest_metadata.get("dataset_metadata") or {}),
            ),
            "truth_alignment_rate": round(
                sum(
                    _clamp(
                        float(metadata.get("truth_alignment_rate", 0.75) or 0.75),
                        0.0,
                        1.0,
                    )
                    * weight
                    for metadata, weight in zip(metadata_list, weights)
                )
                / max(sum(weights), 1e-9),
                4,
            ),
            "topology_feedback_coverage": weighted_metadata_average("topology_feedback_coverage"),
            "avg_topology_reward": weighted_metadata_average("avg_topology_reward"),
            "avg_topology_confidence": weighted_metadata_average("avg_topology_confidence"),
            "degradation_feedback_coverage": weighted_metadata_average(
                "degradation_feedback_coverage"
            ),
            "degradation_event_count": weighted_metadata_average("degradation_event_count"),
            "degraded_task_count": weighted_metadata_average("degraded_task_count"),
            "recovered_task_count": weighted_metadata_average("recovered_task_count"),
            "degradation_recovery_rate": weighted_metadata_average("degradation_recovery_rate"),
            "avg_degradation_adaptation_cost": weighted_metadata_average(
                "avg_degradation_adaptation_cost"
            ),
            "avg_degradation_time_to_recover_seconds": weighted_metadata_average(
                "avg_degradation_time_to_recover_seconds"
            ),
            "avg_degradation_cost_variance": weighted_metadata_average(
                "avg_degradation_cost_variance"
            ),
            "avg_degradation_recovery_time_variance": weighted_metadata_average(
                "avg_degradation_recovery_time_variance"
            ),
            "avg_degradation_intervention_count": weighted_metadata_average(
                "avg_degradation_intervention_count"
            ),
            "avg_degradation_confidence": weighted_metadata_average("avg_degradation_confidence"),
            "avg_degradation_drift_score": weighted_metadata_average("avg_degradation_drift_score"),
            "content_degradation_task_count": weighted_metadata_average(
                "content_degradation_task_count"
            ),
            "confidence_degradation_task_count": weighted_metadata_average(
                "confidence_degradation_task_count"
            ),
            "provider_degradation_task_count": weighted_metadata_average(
                "provider_degradation_task_count"
            ),
            "persistent_degradation_task_count": weighted_metadata_average(
                "persistent_degradation_task_count"
            ),
            "drift_task_count": weighted_metadata_average("drift_task_count"),
            "degradation_drift_rate": weighted_metadata_average("degradation_drift_rate"),
            "degradation_intervention_task_count": weighted_metadata_average(
                "degradation_intervention_task_count"
            ),
            "degradation_intervention_rate": weighted_metadata_average(
                "degradation_intervention_rate"
            ),
            "high_adaptation_cost_task_count": weighted_metadata_average(
                "high_adaptation_cost_task_count"
            ),
            "degradation_high_cost_rate": weighted_metadata_average("degradation_high_cost_rate"),
            "degradation_confidence_rate": weighted_metadata_average("degradation_confidence_rate"),
            "degradation_stability_score": weighted_metadata_average("degradation_stability_score"),
            "degradation_sources": weighted_distribution("degradation_sources"),
            "degradation_kinds": weighted_distribution("degradation_kinds"),
            "degradation_failure_types": weighted_distribution("degradation_failure_types"),
            "degradation_providers": weighted_distribution("degradation_providers"),
            "degradation_reasons": weighted_distribution("degradation_reasons"),
            "optimization_feasible_tasks": weighted_metadata_average("optimization_feasible_tasks"),
            "optimization_infeasible_tasks": weighted_metadata_average(
                "optimization_infeasible_tasks"
            ),
            "optimization_feasibility_rate": weighted_metadata_average(
                "optimization_feasibility_rate"
            ),
            "avg_optimization_reward": weighted_metadata_average("avg_optimization_reward"),
            "avg_feasible_optimization_reward": weighted_metadata_average(
                "avg_feasible_optimization_reward"
            ),
            "avg_infeasible_optimization_reward": weighted_metadata_average(
                "avg_infeasible_optimization_reward"
            ),
            "optimization_gate_failures": optimization_gate_failures,
            "topology_actions": weighted_distribution("topology_actions"),
            "topology_final_actions": weighted_distribution("topology_final_actions"),
            "topology_kinds": weighted_distribution("topology_kinds"),
            "topology_final_kinds": weighted_distribution("topology_final_kinds"),
            "topology_execution_modes": weighted_distribution("topology_execution_modes"),
            "topology_providers": weighted_distribution("topology_providers"),
            "topology_formations": weighted_distribution("topology_formations"),
            "topology_selection_policies": selection_policy_counts,
            "topology_selection_policy_reward_totals": selection_policy_reward_totals,
            "avg_topology_reward_by_selection_policy": avg_reward_by_selection_policy,
            "topology_learned_override_reward_delta": _selection_policy_reward_delta(
                avg_reward_by_selection_policy
            ),
            "topology_selection_policy_optimization_counts": (selection_policy_optimization_counts),
            "topology_selection_policy_optimization_reward_totals": (
                selection_policy_optimization_reward_totals
            ),
            "avg_topology_optimization_reward_by_selection_policy": (
                avg_optimization_reward_by_selection_policy
            ),
            "topology_selection_policy_feasible_counts": selection_policy_feasible_counts,
            "topology_selection_policy_feasibility_rates": selection_policy_feasibility_rates,
            "topology_learned_override_optimization_reward_delta": (
                _selection_policy_reward_delta(avg_optimization_reward_by_selection_policy)
            ),
            "topology_learned_override_feasibility_delta": _selection_policy_reward_delta(
                selection_policy_feasibility_rates
            ),
            "topology_selection_policy_scope_metrics": selection_policy_scope_metrics,
            "team_worktree_scope_metrics": team_worktree_scope_metrics,
            "task_count": sum(task_counts),
            "scope_selection_strategy": (
                "scoped_relevance_recency_reliability_weighted"
                if not selected_scope.is_empty()
                else "recency_reliability_weighted"
            ),
            "scope_target": (None if selected_scope.is_empty() else selected_scope.to_dict()),
            "best_scope_match_score": round(max(scope_scores, default=1.0), 4),
            "freshest_saved_at": _format_timestamp(freshest_saved_at),
            "oldest_saved_at": _format_timestamp(oldest_saved_at),
            "saved_at": _format_timestamp(freshest_saved_at or reference_time),
            "source_result_path": freshest_metadata.get("source_result_path"),
        },
    )


def _load_session_topology_feedback_payloads_from_directory(
    base_dir: Path,
) -> list[dict[str, Any]]:
    """Load persisted scoped live-topology artifacts from the feedback directory."""
    payloads: list[dict[str, Any]] = []
    pattern = f"{RUNTIME_TOPOLOGY_FEEDBACK_FILENAME_PREFIX}.*.json"
    for result_path in sorted(base_dir.glob(pattern)):
        payload = _load_feedback_payload_file(result_path)
        if payload is not None:
            payloads.append(payload)
    return payloads


def _has_topology_runtime_metadata(metadata: Mapping[str, Any]) -> bool:
    """Return whether metadata carries any topology-routing statistics."""
    return any(
        metadata.get(field_name) not in (None, {}, [])
        for field_name in TOPOLOGY_RUNTIME_METADATA_KEYS
    )


def _aggregate_topology_feedback_metadata(
    payloads: list[dict[str, Any]],
    *,
    scope: Optional[Any] = None,
) -> Optional[dict[str, Any]]:
    """Aggregate topology metadata across validated and live scoped feedback artifacts."""
    if not payloads:
        return None

    topology_payloads = []
    for payload in payloads:
        metadata = dict(payload.get("metadata") or {})
        if metadata.get("source") not in TOPOLOGY_RUNTIME_FEEDBACK_SOURCES:
            continue
        if not _has_topology_runtime_metadata(metadata):
            continue
        topology_payloads.append(payload)

    if not topology_payloads:
        return None

    selected_scope = RuntimeEvaluationFeedbackScope.from_value(scope)
    saved_at_values = [
        _parse_timestamp(dict(payload.get("metadata") or {}).get("saved_at"))
        for payload in topology_payloads
    ]
    resolved_saved_at_values = [value for value in saved_at_values if value is not None]
    reference_time = max(resolved_saved_at_values, default=datetime.now(timezone.utc))
    weights = [
        _feedback_weight(payload, reference_time, target_scope=selected_scope)
        for payload in topology_payloads
    ]
    metadata_list = [dict(payload.get("metadata") or {}) for payload in topology_payloads]

    def weighted_metadata_average(field_name: str) -> Optional[float]:
        weighted_pairs = [
            (float(metadata[field_name]), weight)
            for metadata, weight in zip(metadata_list, weights)
            if metadata.get(field_name) is not None
        ]
        if not weighted_pairs:
            return None
        numerator = sum(value * weight for value, weight in weighted_pairs)
        denominator = sum(weight for _, weight in weighted_pairs)
        return round(numerator / max(denominator, 1e-9), 4)

    def weighted_distribution(field_name: str) -> dict[str, float]:
        aggregated: dict[str, float] = {}
        for metadata, weight in zip(metadata_list, weights):
            distribution = metadata.get(field_name) or {}
            if not isinstance(distribution, Mapping):
                continue
            for key, value in distribution.items():
                label = str(key).strip()
                if not label:
                    continue
                try:
                    count = float(value)
                except (TypeError, ValueError):
                    continue
                aggregated[label] = round(aggregated.get(label, 0.0) + (count * weight), 4)
        return aggregated

    action_distribution = weighted_distribution("topology_actions")
    final_action_distribution = weighted_distribution("topology_final_actions")
    topology_distribution = weighted_distribution("topology_kinds")
    final_topology_distribution = weighted_distribution("topology_final_kinds")
    execution_mode_distribution = weighted_distribution("topology_execution_modes")
    provider_distribution = weighted_distribution("topology_providers")
    formation_distribution = weighted_distribution("topology_formations")
    selection_policy_distribution = weighted_distribution("topology_selection_policies")
    selection_policy_reward_totals = weighted_distribution(
        "topology_selection_policy_reward_totals"
    )
    optimization_gate_failures = weighted_distribution("optimization_gate_failures")
    selection_policy_optimization_counts = weighted_distribution(
        "topology_selection_policy_optimization_counts"
    )
    selection_policy_optimization_reward_totals = weighted_distribution(
        "topology_selection_policy_optimization_reward_totals"
    )
    selection_policy_feasible_counts = weighted_distribution(
        "topology_selection_policy_feasible_counts"
    )
    avg_reward_by_selection_policy = _average_mapping_from_totals(
        selection_policy_distribution,
        selection_policy_reward_totals,
    )
    avg_optimization_reward_by_selection_policy = _average_mapping_from_totals(
        selection_policy_optimization_counts,
        selection_policy_optimization_reward_totals,
    )
    selection_policy_feasibility_rates = {
        policy: round(
            float(selection_policy_feasible_counts.get(policy, 0.0)) / max(1.0, float(count_value)),
            4,
        )
        for policy, count_value in selection_policy_optimization_counts.items()
        if float(count_value) > 0.0
    }
    selection_policy_scope_metrics = _build_selection_policy_scope_metrics(
        metadata_list,
        weights,
    )
    team_worktree_scope_metrics = _build_team_worktree_scope_metrics(
        metadata_list,
        weights,
    )
    action_agreement = _distribution_agreement(final_action_distribution or action_distribution)
    topology_agreement = _distribution_agreement(
        final_topology_distribution or topology_distribution
    )
    provider_agreement = _distribution_agreement(provider_distribution)
    formation_agreement = _distribution_agreement(formation_distribution)
    conflict_score = _topology_conflict_score(
        action_agreement=action_agreement,
        topology_agreement=topology_agreement,
        provider_agreement=provider_agreement,
        formation_agreement=formation_agreement,
    )

    scope_scores = [
        _scope_similarity_weight(metadata.get("scope"), selected_scope)
        for metadata in metadata_list
    ]
    sources = [
        str(metadata.get("source"))
        for metadata in metadata_list
        if metadata.get("source") is not None
    ]
    freshest_saved_at = max(resolved_saved_at_values, default=None)
    oldest_saved_at = min(resolved_saved_at_values, default=None)
    live_count = sum(
        1
        for metadata in metadata_list
        if metadata.get("source") == SESSION_TOPOLOGY_RUNTIME_FEEDBACK_SOURCE
    )
    validated_count = sum(
        1
        for metadata in metadata_list
        if metadata.get("source") in VALIDATED_RUNTIME_FEEDBACK_SOURCES
        or metadata.get("source") == AGGREGATED_RUNTIME_FEEDBACK_SOURCE
    )

    return {
        "source": AGGREGATED_SESSION_TOPOLOGY_RUNTIME_FEEDBACK_SOURCE,
        "topology_feedback_sources": list(dict.fromkeys(sources)),
        "topology_feedback_coverage": weighted_metadata_average("topology_feedback_coverage"),
        "avg_topology_reward": weighted_metadata_average("avg_topology_reward"),
        "avg_topology_confidence": weighted_metadata_average("avg_topology_confidence"),
        "degradation_feedback_coverage": weighted_metadata_average("degradation_feedback_coverage"),
        "degradation_event_count": weighted_metadata_average("degradation_event_count"),
        "degraded_task_count": weighted_metadata_average("degraded_task_count"),
        "recovered_task_count": weighted_metadata_average("recovered_task_count"),
        "degradation_recovery_rate": weighted_metadata_average("degradation_recovery_rate"),
        "avg_degradation_adaptation_cost": weighted_metadata_average(
            "avg_degradation_adaptation_cost"
        ),
        "avg_degradation_time_to_recover_seconds": weighted_metadata_average(
            "avg_degradation_time_to_recover_seconds"
        ),
        "avg_degradation_cost_variance": weighted_metadata_average("avg_degradation_cost_variance"),
        "avg_degradation_recovery_time_variance": weighted_metadata_average(
            "avg_degradation_recovery_time_variance"
        ),
        "avg_degradation_intervention_count": weighted_metadata_average(
            "avg_degradation_intervention_count"
        ),
        "avg_degradation_confidence": weighted_metadata_average("avg_degradation_confidence"),
        "avg_degradation_drift_score": weighted_metadata_average("avg_degradation_drift_score"),
        "content_degradation_task_count": weighted_metadata_average(
            "content_degradation_task_count"
        ),
        "confidence_degradation_task_count": weighted_metadata_average(
            "confidence_degradation_task_count"
        ),
        "provider_degradation_task_count": weighted_metadata_average(
            "provider_degradation_task_count"
        ),
        "persistent_degradation_task_count": weighted_metadata_average(
            "persistent_degradation_task_count"
        ),
        "drift_task_count": weighted_metadata_average("drift_task_count"),
        "degradation_drift_rate": weighted_metadata_average("degradation_drift_rate"),
        "degradation_intervention_task_count": weighted_metadata_average(
            "degradation_intervention_task_count"
        ),
        "degradation_intervention_rate": weighted_metadata_average("degradation_intervention_rate"),
        "high_adaptation_cost_task_count": weighted_metadata_average(
            "high_adaptation_cost_task_count"
        ),
        "degradation_high_cost_rate": weighted_metadata_average("degradation_high_cost_rate"),
        "degradation_confidence_rate": weighted_metadata_average("degradation_confidence_rate"),
        "degradation_stability_score": weighted_metadata_average("degradation_stability_score"),
        "degradation_sources": weighted_distribution("degradation_sources"),
        "degradation_kinds": weighted_distribution("degradation_kinds"),
        "degradation_failure_types": weighted_distribution("degradation_failure_types"),
        "degradation_providers": weighted_distribution("degradation_providers"),
        "degradation_reasons": weighted_distribution("degradation_reasons"),
        "tasks_with_team_feedback": weighted_metadata_average("tasks_with_team_feedback"),
        "team_feedback_coverage": weighted_metadata_average("team_feedback_coverage"),
        "team_formations": weighted_distribution("team_formations"),
        "team_merge_risk_levels": weighted_distribution("team_merge_risk_levels"),
        "team_worktree_plan_count": weighted_metadata_average("team_worktree_plan_count"),
        "team_worktree_materialized_count": weighted_metadata_average(
            "team_worktree_materialized_count"
        ),
        "team_worktree_dry_run_count": weighted_metadata_average("team_worktree_dry_run_count"),
        "team_low_risk_task_count": weighted_metadata_average("team_low_risk_task_count"),
        "team_medium_risk_task_count": weighted_metadata_average("team_medium_risk_task_count"),
        "team_high_risk_task_count": weighted_metadata_average("team_high_risk_task_count"),
        "team_merge_conflict_task_count": weighted_metadata_average(
            "team_merge_conflict_task_count"
        ),
        "team_merge_conflict_count": weighted_metadata_average("team_merge_conflict_count"),
        "team_merge_overlap_task_count": weighted_metadata_average("team_merge_overlap_task_count"),
        "team_out_of_scope_write_task_count": weighted_metadata_average(
            "team_out_of_scope_write_task_count"
        ),
        "team_out_of_scope_write_count": weighted_metadata_average("team_out_of_scope_write_count"),
        "team_readonly_violation_task_count": weighted_metadata_average(
            "team_readonly_violation_task_count"
        ),
        "team_readonly_violation_count": weighted_metadata_average("team_readonly_violation_count"),
        "team_cleanup_task_count": weighted_metadata_average("team_cleanup_task_count"),
        "team_cleanup_error_task_count": weighted_metadata_average("team_cleanup_error_task_count"),
        "team_cleanup_error_count": weighted_metadata_average("team_cleanup_error_count"),
        "avg_team_assignments": weighted_metadata_average("avg_team_assignments"),
        "avg_team_scoped_members": weighted_metadata_average("avg_team_scoped_members"),
        "avg_team_members_with_changes": weighted_metadata_average("avg_team_members_with_changes"),
        "avg_team_changed_file_count": weighted_metadata_average("avg_team_changed_file_count"),
        "team_materialized_assignment_total": weighted_metadata_average(
            "team_materialized_assignment_total"
        ),
        "team_worktree_scope_metrics": team_worktree_scope_metrics,
        "optimization_feasible_tasks": weighted_metadata_average("optimization_feasible_tasks"),
        "optimization_infeasible_tasks": weighted_metadata_average("optimization_infeasible_tasks"),
        "optimization_feasibility_rate": weighted_metadata_average("optimization_feasibility_rate"),
        "avg_optimization_reward": weighted_metadata_average("avg_optimization_reward"),
        "avg_feasible_optimization_reward": weighted_metadata_average(
            "avg_feasible_optimization_reward"
        ),
        "avg_infeasible_optimization_reward": weighted_metadata_average(
            "avg_infeasible_optimization_reward"
        ),
        "optimization_gate_failures": optimization_gate_failures,
        "topology_observation_count": (
            weighted_metadata_average("topology_observation_count")
            or weighted_metadata_average("task_count")
        ),
        "topology_actions": action_distribution,
        "topology_final_actions": final_action_distribution,
        "topology_kinds": topology_distribution,
        "topology_final_kinds": final_topology_distribution,
        "topology_execution_modes": execution_mode_distribution,
        "topology_providers": provider_distribution,
        "topology_formations": formation_distribution,
        "topology_selection_policies": selection_policy_distribution,
        "topology_selection_policy_reward_totals": selection_policy_reward_totals,
        "avg_topology_reward_by_selection_policy": avg_reward_by_selection_policy,
        "topology_learned_override_reward_delta": _selection_policy_reward_delta(
            avg_reward_by_selection_policy
        ),
        "topology_selection_policy_optimization_counts": selection_policy_optimization_counts,
        "topology_selection_policy_optimization_reward_totals": (
            selection_policy_optimization_reward_totals
        ),
        "avg_topology_optimization_reward_by_selection_policy": (
            avg_optimization_reward_by_selection_policy
        ),
        "topology_selection_policy_feasible_counts": selection_policy_feasible_counts,
        "topology_selection_policy_feasibility_rates": selection_policy_feasibility_rates,
        "topology_learned_override_optimization_reward_delta": (
            _selection_policy_reward_delta(avg_optimization_reward_by_selection_policy)
        ),
        "topology_learned_override_feasibility_delta": _selection_policy_reward_delta(
            selection_policy_feasibility_rates
        ),
        "topology_selection_policy_scope_metrics": selection_policy_scope_metrics,
        "topology_action_agreement": action_agreement,
        "topology_kind_agreement": topology_agreement,
        "topology_provider_agreement": provider_agreement,
        "topology_formation_agreement": formation_agreement,
        "topology_conflict_score": conflict_score,
        "topology_feedback_artifact_count": len(topology_payloads),
        "topology_feedback_live_artifact_count": live_count,
        "topology_feedback_validated_artifact_count": validated_count,
        "topology_scope_selection_strategy": (
            "scoped_relevance_recency_reliability_weighted"
            if not selected_scope.is_empty()
            else "recency_reliability_weighted"
        ),
        "topology_scope_target": (None if selected_scope.is_empty() else selected_scope.to_dict()),
        "topology_best_scope_match_score": round(max(scope_scores, default=1.0), 4),
        "topology_freshest_saved_at": _format_timestamp(freshest_saved_at),
        "topology_oldest_saved_at": _format_timestamp(oldest_saved_at),
    }


def _merge_runtime_feedback_with_topology_metadata(
    feedback: RuntimeEvaluationFeedback,
    topology_metadata: Optional[Mapping[str, Any]],
) -> RuntimeEvaluationFeedback:
    """Overlay aggregated topology metadata without changing calibration thresholds."""
    if not topology_metadata:
        return feedback

    metadata = dict(feedback.metadata or {})
    for key, value in topology_metadata.items():
        if key == "source":
            continue
        if value in (None, "", {}, []):
            continue
        metadata[key] = value
    if metadata.get("source") is None and topology_metadata.get("source") is not None:
        metadata["source"] = topology_metadata.get("source")

    return RuntimeEvaluationFeedback(
        completion_threshold=feedback.completion_threshold,
        enhanced_progress_threshold=feedback.enhanced_progress_threshold,
        minimum_supported_evidence_score=feedback.minimum_supported_evidence_score,
        metadata=metadata,
    )


def save_runtime_evaluation_feedback(
    feedback: RuntimeEvaluationFeedback,
    *,
    path: Optional[Path] = None,
    source_result_path: Optional[Path] = None,
) -> Path:
    """Persist runtime calibration feedback to the canonical evaluation-feedback file."""
    target_path = Path(path) if path is not None else get_runtime_evaluation_feedback_path()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_runtime_evaluation_feedback_payload(
        feedback,
        source_result_path=source_result_path,
    )
    target_path.write_text(json.dumps(payload, indent=2))
    return target_path


def refresh_runtime_evaluation_feedback_aggregate(base_dir: Path) -> Optional[Path]:
    """Refresh the canonical aggregate from validated evaluation-truth artifacts."""
    aggregate = load_runtime_evaluation_feedback(
        Path(base_dir) / RUNTIME_EVALUATION_FEEDBACK_FILENAME
    )
    if aggregate is None:
        return None
    canonical_path = get_runtime_evaluation_feedback_path(base_dir)
    canonical_path.parent.mkdir(parents=True, exist_ok=True)
    canonical_path.write_text(
        json.dumps(
            build_runtime_evaluation_feedback_payload(
                aggregate,
                source_result_path=(
                    Path(aggregate.metadata["source_result_path"])
                    if aggregate.metadata.get("source_result_path")
                    else None
                ),
                saved_at=_parse_timestamp(aggregate.metadata.get("saved_at")),
            ),
            indent=2,
        )
    )
    return canonical_path


def load_runtime_evaluation_feedback(
    path: Optional[Path] = None,
    *,
    scope: Optional[Any] = None,
) -> Optional[RuntimeEvaluationFeedback]:
    """Load persisted runtime calibration feedback when available."""
    target_path = Path(path) if path is not None else get_runtime_evaluation_feedback_path()
    feedback_dir = _resolve_feedback_directory(target_path)
    if feedback_dir is not None:
        payloads = _load_feedback_payloads_from_directory(feedback_dir)
        topology_metadata = _aggregate_topology_feedback_metadata(
            payloads + _load_session_topology_feedback_payloads_from_directory(feedback_dir),
            scope=scope,
        )
        aggregate = _aggregate_feedback_payloads(
            payloads,
            scope=scope,
        )
        if aggregate is not None:
            return _merge_runtime_feedback_with_topology_metadata(
                aggregate,
                topology_metadata,
            )
        if target_path.exists():
            payload = _load_feedback_payload_file(target_path)
            if payload is not None:
                return _merge_runtime_feedback_with_topology_metadata(
                    RuntimeEvaluationFeedback.from_dict(payload),
                    topology_metadata,
                )
        if topology_metadata is not None:
            return RuntimeEvaluationFeedback(metadata=dict(topology_metadata))
        return None

    if not target_path.exists():
        return None
    payload = _load_feedback_payload_file(target_path)
    if payload is None:
        return None
    return RuntimeEvaluationFeedback.from_dict(payload)
