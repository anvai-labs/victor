from __future__ import annotations

# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Registry-backed emitters for validated session-truth feedback artifacts."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol

from victor.evaluation.protocol import (
    BenchmarkType,
    EvaluationConfig,
    EvaluationResult,
    TaskResult,
    is_browser_task_benchmark,
)
from victor.evaluation.runtime_feedback import (
    build_browser_validated_session_feedback_payload,
    build_deep_research_validated_session_feedback_payload,
    build_swe_bench_validated_session_feedback_payload,
)
from victor.evaluation.topology_feedback import summarize_topology_feedback
from victor.evaluation.validated_session_truth_naming import (
    ValidatedSessionTruthArtifactNamingPolicy,
    create_default_validated_session_truth_artifact_naming_policy,
)


@dataclass(frozen=True)
class ValidatedSessionTruthArtifact:
    """Serialized validated session-truth artifact ready to persist."""

    path: Path
    record: dict[str, Any]


@dataclass(frozen=True)
class ValidatedSessionTruthEmissionContext:
    """Canonical emission context shared by all benchmark-family emitters."""

    benchmark: BenchmarkType
    results_dir: Path
    task_id: str
    source_result_path: Optional[Path] = None
    task_result: Optional[TaskResult] = None
    config: Optional[EvaluationConfig] = None
    evaluation_result: Optional[EvaluationResult] = None
    summary: Mapping[str, Any] = field(default_factory=dict)
    validation_result: Any = None
    score: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class ValidatedSessionTruthEmitter(Protocol):
    """Strategy interface for benchmark-family validated session-truth emission."""

    def supports(self, benchmark: BenchmarkType) -> bool:
        """Return whether the emitter can handle the benchmark family."""

    def build_artifact(
        self,
        context: ValidatedSessionTruthEmissionContext,
    ) -> Optional[ValidatedSessionTruthArtifact]:
        """Build a persisted validated session-truth artifact when evidence is strong enough."""


DEFAULT_VALIDATED_SESSION_TRUTH_ARTIFACT_NAMING_POLICY = (
    create_default_validated_session_truth_artifact_naming_policy()
)


def _optional_metadata_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _session_feedback_input(
    context: ValidatedSessionTruthEmissionContext,
) -> Optional[dict[str, Any]]:
    task_result = context.task_result
    config = context.config
    if task_result is None or config is None:
        return None
    return {
        "task_id": task_result.task_id,
        "status": task_result.status.value,
        "completion_score": task_result.completion_score,
        "failure_category": (
            task_result.failure_category.value if task_result.failure_category is not None else None
        ),
        "failure_details": dict(task_result.failure_details),
        "benchmark": config.benchmark.value,
        "provider": config.provider,
        "model": config.model,
        "prompt_candidate_hash": config.prompt_candidate_hash,
        "section_name": config.prompt_section_name,
        "prompt_section_name": config.prompt_section_name,
        "dataset_metadata": dict(config.dataset_metadata),
        "total_tasks": context.summary.get("total_tasks"),
        "passed_tasks": context.summary.get("passed"),
        "failed_tasks": context.summary.get("failed"),
    }


def _artifact_record_from_task_result(
    context: ValidatedSessionTruthEmissionContext,
    *,
    payload: Mapping[str, Any],
) -> Optional[dict[str, Any]]:
    task_result = context.task_result
    config = context.config
    if task_result is None or config is None:
        return None
    topology_summary = summarize_topology_feedback(task_result)
    return {
        "task_id": task_result.task_id,
        "benchmark": config.benchmark.value,
        "provider": config.provider,
        "model": config.model,
        "prompt_candidate_hash": config.prompt_candidate_hash,
        "section_name": config.prompt_section_name,
        "prompt_section_name": config.prompt_section_name,
        "runtime_evaluation_feedback": dict(payload),
        "status": task_result.status.value,
        "completion_score": task_result.completion_score,
        "failure_category": (
            task_result.failure_category.value if task_result.failure_category is not None else None
        ),
        "failure_details": dict(task_result.failure_details),
        "topology_summary": topology_summary,
    }


class BrowserValidatedSessionTruthEmitter:
    """Validated session-truth emitter for browser-task benchmark families."""

    def __init__(
        self,
        naming_policy: ValidatedSessionTruthArtifactNamingPolicy = (
            DEFAULT_VALIDATED_SESSION_TRUTH_ARTIFACT_NAMING_POLICY
        ),
    ):
        self._naming_policy = naming_policy

    def supports(self, benchmark: BenchmarkType) -> bool:
        return is_browser_task_benchmark(benchmark)

    def build_artifact(
        self,
        context: ValidatedSessionTruthEmissionContext,
    ) -> Optional[ValidatedSessionTruthArtifact]:
        payload_input = _session_feedback_input(context)
        if payload_input is None:
            return None
        topology_summary = summarize_topology_feedback(context.task_result)

        artifact_path = self._naming_policy.path_for_evaluation_task(
            results_dir=context.results_dir,
            benchmark=context.benchmark,
            task_id=context.task_id,
            source_result_path=context.source_result_path,
        )
        payload = build_browser_validated_session_feedback_payload(
            payload_input,
            source_result_path=artifact_path,
            metadata={
                "source_evaluation_path": str(context.source_result_path),
                "topology_summary": topology_summary,
                **context.metadata,
            },
        )
        if payload is None:
            return None
        record = _artifact_record_from_task_result(context, payload=payload)
        if record is None:
            return None
        return ValidatedSessionTruthArtifact(path=artifact_path, record=record)


class DeepResearchValidatedSessionTruthEmitter:
    """Validated session-truth emitter for DR3-style deep-research benchmarks."""

    def __init__(
        self,
        naming_policy: ValidatedSessionTruthArtifactNamingPolicy = (
            DEFAULT_VALIDATED_SESSION_TRUTH_ARTIFACT_NAMING_POLICY
        ),
    ):
        self._naming_policy = naming_policy

    def supports(self, benchmark: BenchmarkType) -> bool:
        return benchmark == BenchmarkType.DR3_EVAL

    def build_artifact(
        self,
        context: ValidatedSessionTruthEmissionContext,
    ) -> Optional[ValidatedSessionTruthArtifact]:
        payload_input = _session_feedback_input(context)
        if payload_input is None:
            return None
        topology_summary = summarize_topology_feedback(context.task_result)

        artifact_path = self._naming_policy.path_for_evaluation_task(
            results_dir=context.results_dir,
            benchmark=context.benchmark,
            task_id=context.task_id,
            source_result_path=context.source_result_path,
        )
        payload = build_deep_research_validated_session_feedback_payload(
            payload_input,
            source_result_path=artifact_path,
            metadata={
                "source_evaluation_path": str(context.source_result_path),
                "topology_summary": topology_summary,
                **context.metadata,
            },
        )
        if payload is None:
            return None
        record = _artifact_record_from_task_result(context, payload=payload)
        if record is None:
            return None
        return ValidatedSessionTruthArtifact(path=artifact_path, record=record)


class SWEBenchValidatedSessionTruthEmitter:
    """Validated session-truth emitter for objective SWE-bench validation output."""

    def __init__(
        self,
        naming_policy: ValidatedSessionTruthArtifactNamingPolicy = (
            DEFAULT_VALIDATED_SESSION_TRUTH_ARTIFACT_NAMING_POLICY
        ),
    ):
        self._naming_policy = naming_policy

    def supports(self, benchmark: BenchmarkType) -> bool:
        return benchmark == BenchmarkType.SWE_BENCH

    def build_artifact(
        self,
        context: ValidatedSessionTruthEmissionContext,
    ) -> Optional[ValidatedSessionTruthArtifact]:
        validation_result = context.validation_result
        if validation_result is None:
            return None

        artifact_path = self._naming_policy.path_for_validation_task(
            results_dir=context.results_dir,
            benchmark=context.benchmark,
            task_id=context.task_id,
            source_result_path=context.source_result_path,
        )
        payload = build_swe_bench_validated_session_feedback_payload(
            validation_result,
            score=context.score,
            source_result_path=artifact_path,
            metadata=dict(context.metadata),
        )
        if payload is None:
            return None

        provider = _optional_metadata_text(context.metadata.get("provider"))
        model = _optional_metadata_text(context.metadata.get("model"))
        section_name = _optional_metadata_text(
            context.metadata.get("section_name") or context.metadata.get("prompt_section_name")
        )
        prompt_candidate_hash = _optional_metadata_text(
            context.metadata.get("prompt_candidate_hash")
        )

        return ValidatedSessionTruthArtifact(
            path=artifact_path,
            record={
                "instance_id": context.task_id,
                "repo": getattr(getattr(validation_result, "baseline", None), "repo", None),
                "provider": provider,
                "model": model,
                "prompt_candidate_hash": prompt_candidate_hash,
                "section_name": section_name,
                "prompt_section_name": section_name,
                "runtime_evaluation_feedback": payload,
                "validation_result": (
                    validation_result.to_dict()
                    if hasattr(validation_result, "to_dict")
                    else {
                        "success": getattr(validation_result, "success", False),
                        "partial_success": getattr(validation_result, "partial_success", False),
                        "score": getattr(validation_result, "score", 0.0),
                    }
                ),
                "score": (
                    context.score.to_dict()
                    if context.score is not None and hasattr(context.score, "to_dict")
                    else None
                ),
            },
        )


class ValidatedSessionTruthEmitterRegistry:
    """Registry for benchmark-family validated session-truth emitters."""

    def __init__(self, emitters: Optional[list[ValidatedSessionTruthEmitter]] = None):
        self._emitters = list(emitters or [])

    def register(self, emitter: ValidatedSessionTruthEmitter) -> None:
        self._emitters.append(emitter)

    def resolve(self, benchmark: BenchmarkType) -> Optional[ValidatedSessionTruthEmitter]:
        for emitter in self._emitters:
            if emitter.supports(benchmark):
                return emitter
        return None


def create_default_validated_session_truth_emitter_registry() -> (
    ValidatedSessionTruthEmitterRegistry
):
    """Return the canonical emitter registry used by evaluation runtimes."""
    return ValidatedSessionTruthEmitterRegistry(
        [
            BrowserValidatedSessionTruthEmitter(),
            DeepResearchValidatedSessionTruthEmitter(),
            SWEBenchValidatedSessionTruthEmitter(),
        ]
    )
