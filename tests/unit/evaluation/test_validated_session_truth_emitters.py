# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for registry-backed validated session-truth emitters."""

from victor.evaluation.baseline_validator import (
    BaselineStatus,
    BaselineValidationResult,
    TestBaseline,
)
from victor.evaluation.protocol import (
    BenchmarkFailureCategory,
    BenchmarkType,
    EvaluationConfig,
    EvaluationResult,
    TaskResult,
    TaskStatus,
)
from victor.evaluation.result_correlation import SWEBenchScore
from victor.evaluation.test_runners import TestRunResults
from victor.evaluation.validated_session_truth_emitters import (
    ValidatedSessionTruthEmissionContext,
    ValidatedSessionTruthEmitterRegistry,
    create_default_validated_session_truth_emitter_registry,
)


def test_default_registry_resolves_browser_benchmark():
    registry = create_default_validated_session_truth_emitter_registry()

    emitter = registry.resolve(BenchmarkType.GUIDE)

    assert emitter is not None
    assert emitter.supports(BenchmarkType.CLAW_BENCH) is True


def test_default_registry_resolves_deep_research_benchmark():
    registry = create_default_validated_session_truth_emitter_registry()

    emitter = registry.resolve(BenchmarkType.DR3_EVAL)

    assert emitter is not None
    assert emitter.supports(BenchmarkType.DR3_EVAL) is True


def test_registry_returns_none_for_unsupported_benchmark():
    registry = ValidatedSessionTruthEmitterRegistry()

    assert registry.resolve(BenchmarkType.HUMAN_EVAL) is None


def test_browser_emitter_builds_artifact_with_canonical_payload(tmp_path):
    registry = create_default_validated_session_truth_emitter_registry()
    emitter = registry.resolve(BenchmarkType.GUIDE)
    assert emitter is not None

    config = EvaluationConfig(
        benchmark=BenchmarkType.GUIDE,
        model="test",
        provider="openai",
        prompt_candidate_hash="cand-123",
        prompt_section_name="GROUNDING_RULES",
        dataset_metadata={"source_name": "GUIDE"},
    )
    task_result = TaskResult(
        task_id="guide-1",
        status=TaskStatus.FAILED,
        completion_score=0.45,
        failure_category=BenchmarkFailureCategory.TASK_COMPLETION,
        failure_details={
            "action_coverage": 0.5,
            "answer_coverage": 0.35,
            "matched_actions": ["open_url"],
            "missing_actions": ["click"],
            "matched_answer_phrases": [],
            "missing_answer_phrases": ["settings"],
            "forbidden_action_hits": [],
        },
        metadata={
            "topology_events": [
                {
                    "action": "team_plan",
                    "topology": "team",
                    "execution_mode": "team_execution",
                    "formation": "parallel",
                    "provider": "openai",
                    "confidence": 0.84,
                }
            ]
        },
    )
    evaluation_result = EvaluationResult(config=config, task_results=[task_result])

    artifact = emitter.build_artifact(
        ValidatedSessionTruthEmissionContext(
            benchmark=BenchmarkType.GUIDE,
            results_dir=tmp_path,
            task_id=task_result.task_id,
            source_result_path=tmp_path / "eval_guide_20260425_010101.json",
            task_result=task_result,
            config=config,
            evaluation_result=evaluation_result,
            summary={"total_tasks": 1, "passed": 0, "failed": 1},
        )
    )

    assert artifact is not None
    assert artifact.path.name.startswith("eval_session_guide_guide-1_")
    assert (
        artifact.record["runtime_evaluation_feedback"]["metadata"]["truth_validation_mode"]
        == "browser_posthoc_validation"
    )
    assert artifact.record["runtime_evaluation_feedback"]["metadata"]["prompt_candidate_hash"] == (
        "cand-123"
    )
    assert artifact.record["runtime_evaluation_feedback"]["metadata"]["section_name"] == (
        "GROUNDING_RULES"
    )
    assert artifact.record["runtime_evaluation_feedback"]["metadata"]["scope"]["provider"] == (
        "openai"
    )
    assert artifact.record["runtime_evaluation_feedback"]["metadata"]["scope"]["vertical"] == (
        "browser"
    )
    assert artifact.record["topology_summary"]["selected_action"] == "team_plan"
    assert (
        artifact.record["runtime_evaluation_feedback"]["metadata"]["topology_summary"][
            "final_execution_mode"
        ]
        == "team_execution"
    )


def test_deep_research_emitter_returns_none_without_posthoc_evidence(tmp_path):
    registry = create_default_validated_session_truth_emitter_registry()
    emitter = registry.resolve(BenchmarkType.DR3_EVAL)
    assert emitter is not None

    config = EvaluationConfig(benchmark=BenchmarkType.DR3_EVAL, model="test")
    task_result = TaskResult(task_id="dr3-empty", status=TaskStatus.FAILED, completion_score=0.0)
    evaluation_result = EvaluationResult(config=config, task_results=[task_result])

    artifact = emitter.build_artifact(
        ValidatedSessionTruthEmissionContext(
            benchmark=BenchmarkType.DR3_EVAL,
            results_dir=tmp_path,
            task_id=task_result.task_id,
            source_result_path=tmp_path / "eval_dr3_eval_20260425_010101.json",
            task_result=task_result,
            config=config,
            evaluation_result=evaluation_result,
            summary={"total_tasks": 1, "passed": 0, "failed": 1},
        )
    )

    assert artifact is None


def test_default_registry_resolves_swe_bench_emitter():
    registry = create_default_validated_session_truth_emitter_registry()

    emitter = registry.resolve(BenchmarkType.SWE_BENCH)

    assert emitter is not None
    assert emitter.supports(BenchmarkType.SWE_BENCH) is True


def test_swe_bench_emitter_builds_artifact_from_validation_outputs(tmp_path):
    registry = create_default_validated_session_truth_emitter_registry()
    emitter = registry.resolve(BenchmarkType.SWE_BENCH)
    assert emitter is not None

    validation_result = BaselineValidationResult(
        instance_id="django__123",
        baseline=TestBaseline(
            instance_id="django__123",
            repo="django/django",
            base_commit="abc123",
            fail_to_pass=["test_fix_a", "test_fix_b"],
            pass_to_pass=["test_keep_green"],
            status=BaselineStatus.VALID,
        ),
        post_change_results=TestRunResults(total=3, passed=2, failed=1, duration_seconds=8.0),
        fail_to_pass_fixed=["test_fix_a"],
        pass_to_pass_broken=[],
        success=False,
        partial_success=True,
        score=0.5,
    )
    score = SWEBenchScore(
        instance_id="django__123",
        resolved=False,
        partial=True,
        fail_to_pass_score=0.5,
        pass_to_pass_score=1.0,
        overall_score=0.7,
        tests_fixed=1,
        tests_broken=0,
        total_fail_to_pass=2,
        total_pass_to_pass=1,
    )

    artifact = emitter.build_artifact(
        ValidatedSessionTruthEmissionContext(
            benchmark=BenchmarkType.SWE_BENCH,
            results_dir=tmp_path,
            task_id="django__123",
            validation_result=validation_result,
            score=score,
            metadata={
                "provider": "anthropic",
                "model": "claude-sonnet",
                "prompt_candidate_hash": "cand-123",
                "section_name": "GROUNDING_RULES",
                "prompt_section_name": "GROUNDING_RULES",
            },
        )
    )

    assert artifact is not None
    assert artifact.path.name == "eval_session_django__123.json"
    assert artifact.record["provider"] == "anthropic"
    assert artifact.record["model"] == "claude-sonnet"
    assert artifact.record["prompt_candidate_hash"] == "cand-123"
    assert artifact.record["section_name"] == "GROUNDING_RULES"
    assert (
        artifact.record["runtime_evaluation_feedback"]["metadata"]["truth_validation_mode"]
        == "swe_bench_posthoc_validation"
    )
    assert artifact.record["runtime_evaluation_feedback"]["metadata"]["provider"] == "anthropic"
    assert artifact.record["runtime_evaluation_feedback"]["metadata"]["model"] == "claude-sonnet"
    assert (
        artifact.record["runtime_evaluation_feedback"]["metadata"]["prompt_candidate_hash"]
        == "cand-123"
    )
    assert (
        artifact.record["runtime_evaluation_feedback"]["metadata"]["section_name"]
        == "GROUNDING_RULES"
    )
    assert artifact.record["runtime_evaluation_feedback"]["metadata"]["scope"]["provider"] == (
        "anthropic"
    )
    assert artifact.record["runtime_evaluation_feedback"]["metadata"]["scope"]["model"] == (
        "claude-sonnet"
    )
