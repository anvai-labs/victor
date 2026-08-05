# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for validated session-truth service orchestration."""

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
    ValidatedSessionTruthArtifact,
    ValidatedSessionTruthEmitterRegistry,
)
from victor.evaluation.validated_session_truth_service import (
    ValidatedSessionTruthService,
    create_default_validated_session_truth_service,
)


def test_service_persists_evaluation_result_via_registry_and_persistence_helper(
    tmp_path, monkeypatch
):
    captured = {}

    class StubEmitter:
        def supports(self, benchmark):
            return benchmark == BenchmarkType.GUIDE

        def build_artifact(self, context):
            captured["context"] = context
            return ValidatedSessionTruthArtifact(
                path=tmp_path / "eval_session_guide_stub.json",
                record={"runtime_evaluation_feedback": {"metadata": {"source": "stub"}}},
            )

    def fake_persist(artifacts, *, refresh_dir, refresh_when_empty=False):
        captured["artifacts"] = list(artifacts)
        captured["refresh_dir"] = refresh_dir
        captured["refresh_when_empty"] = refresh_when_empty
        return [artifact.path for artifact in captured["artifacts"]]

    monkeypatch.setattr(
        "victor.evaluation.validated_session_truth_service.persist_validated_session_truth_artifacts",
        fake_persist,
    )
    service = ValidatedSessionTruthService(ValidatedSessionTruthEmitterRegistry([StubEmitter()]))
    result = EvaluationResult(
        config=EvaluationConfig(benchmark=BenchmarkType.GUIDE, model="test"),
        task_results=[
            TaskResult(
                task_id="guide-1",
                status=TaskStatus.FAILED,
                completion_score=0.4,
                failure_category=BenchmarkFailureCategory.TASK_COMPLETION,
            )
        ],
    )

    saved_paths = service.persist_evaluation_result(
        result,
        results_dir=tmp_path,
        source_result_path=tmp_path / "eval_guide_20260425_010101.json",
        summary={"total_tasks": 1, "passed": 0, "failed": 1},
    )

    assert saved_paths == [tmp_path / "eval_session_guide_stub.json"]
    assert captured["context"].benchmark == BenchmarkType.GUIDE
    assert captured["context"].task_id == "guide-1"
    assert captured["context"].source_result_path == tmp_path / "eval_guide_20260425_010101.json"
    assert captured["refresh_dir"] == tmp_path
    assert captured["refresh_when_empty"] is True


def test_service_persists_validation_result_via_registry_and_persistence_helper(
    tmp_path, monkeypatch
):
    captured = {}

    class StubEmitter:
        def supports(self, benchmark):
            return benchmark == BenchmarkType.SWE_BENCH

        def build_artifact(self, context):
            captured["context"] = context
            return ValidatedSessionTruthArtifact(
                path=tmp_path / "eval_session_swe_stub.json",
                record={"runtime_evaluation_feedback": {"metadata": {"source": "stub"}}},
            )

    def fake_persist(artifacts, *, refresh_dir, refresh_when_empty=False):
        captured["artifacts"] = list(artifacts)
        captured["refresh_dir"] = refresh_dir
        captured["refresh_when_empty"] = refresh_when_empty
        return [artifact.path for artifact in captured["artifacts"]]

    monkeypatch.setattr(
        "victor.evaluation.validated_session_truth_service.persist_validated_session_truth_artifacts",
        fake_persist,
    )
    service = ValidatedSessionTruthService(ValidatedSessionTruthEmitterRegistry([StubEmitter()]))

    validation_result = BaselineValidationResult(
        instance_id="django__123",
        baseline=TestBaseline(
            instance_id="django__123",
            repo="django/django",
            base_commit="abc123",
            fail_to_pass=["test_fix_a"],
            pass_to_pass=["test_keep_green"],
            status=BaselineStatus.VALID,
        ),
        post_change_results=TestRunResults(total=2, passed=1, failed=1, duration_seconds=2.0),
        fail_to_pass_fixed=["test_fix_a"],
        pass_to_pass_broken=[],
        success=False,
        partial_success=True,
        score=0.5,
    )
    score = SWEBenchScore(instance_id="django__123", overall_score=0.7)

    saved_path = service.persist_validation_result(
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
        },
    )

    assert saved_path == tmp_path / "eval_session_swe_stub.json"
    assert captured["context"].benchmark == BenchmarkType.SWE_BENCH
    assert captured["context"].validation_result is validation_result
    assert captured["context"].score is score
    assert captured["context"].metadata["provider"] == "anthropic"
    assert captured["context"].metadata["model"] == "claude-sonnet"
    assert captured["context"].metadata["prompt_candidate_hash"] == "cand-123"
    assert captured["context"].metadata["section_name"] == "GROUNDING_RULES"
    assert captured["refresh_dir"] == tmp_path
    assert captured["refresh_when_empty"] is False


def test_service_returns_none_when_validation_emitter_is_missing(tmp_path):
    service = ValidatedSessionTruthService(ValidatedSessionTruthEmitterRegistry())

    saved_path = service.persist_validation_result(
        benchmark=BenchmarkType.SWE_BENCH,
        results_dir=tmp_path,
        task_id="django__123",
        validation_result=object(),
    )

    assert saved_path is None


def test_create_default_validated_session_truth_service_uses_supplied_registry():
    registry = ValidatedSessionTruthEmitterRegistry()

    service = create_default_validated_session_truth_service(registry)

    assert isinstance(service, ValidatedSessionTruthService)
    assert service._emitters is registry


def test_service_creates_results_dir_before_persisting(tmp_path, monkeypatch):
    captured = {}

    class StubEmitter:
        def supports(self, benchmark):
            return benchmark == BenchmarkType.GUIDE

        def build_artifact(self, context):
            return ValidatedSessionTruthArtifact(
                path=context.results_dir / "eval_session_stub.json",
                record={"runtime_evaluation_feedback": {"metadata": {"source": "stub"}}},
            )

    def fake_persist(artifacts, *, refresh_dir, refresh_when_empty=False):
        captured["refresh_dir_exists"] = refresh_dir.exists()
        return [artifact.path for artifact in artifacts]

    monkeypatch.setattr(
        "victor.evaluation.validated_session_truth_service.persist_validated_session_truth_artifacts",
        fake_persist,
    )
    service = ValidatedSessionTruthService(ValidatedSessionTruthEmitterRegistry([StubEmitter()]))
    results_dir = tmp_path / "missing" / "evaluations"
    result = EvaluationResult(
        config=EvaluationConfig(benchmark=BenchmarkType.GUIDE, model="test"),
        task_results=[
            TaskResult(task_id="guide-1", status=TaskStatus.PASSED, completion_score=1.0)
        ],
    )

    service.persist_evaluation_result(
        result,
        results_dir=results_dir,
        source_result_path=results_dir / "eval_guide_20260425_010101.json",
        summary={"total_tasks": 1, "passed": 1, "failed": 0},
    )

    assert captured["refresh_dir_exists"] is True


def test_service_skips_task_when_emitter_raises_for_evaluation_result(tmp_path):
    class ExplodingEmitter:
        def supports(self, benchmark):
            return benchmark == BenchmarkType.GUIDE

        def build_artifact(self, context):
            raise RuntimeError(f"boom:{context.task_id}")

    service = ValidatedSessionTruthService(
        ValidatedSessionTruthEmitterRegistry([ExplodingEmitter()])
    )
    result = EvaluationResult(
        config=EvaluationConfig(benchmark=BenchmarkType.GUIDE, model="test"),
        task_results=[
            TaskResult(task_id="guide-1", status=TaskStatus.PASSED, completion_score=1.0),
            TaskResult(task_id="guide-2", status=TaskStatus.FAILED, completion_score=0.2),
        ],
    )

    saved_paths = service.persist_evaluation_result(
        result,
        results_dir=tmp_path,
        source_result_path=tmp_path / "eval_guide_20260425_010101.json",
        summary={"total_tasks": 2, "passed": 1, "failed": 1},
    )

    assert saved_paths == []


def test_service_returns_none_when_validation_persistence_raises(tmp_path, monkeypatch):
    class StubEmitter:
        def supports(self, benchmark):
            return benchmark == BenchmarkType.SWE_BENCH

        def build_artifact(self, context):
            return ValidatedSessionTruthArtifact(
                path=tmp_path / "eval_session_stub.json",
                record={"runtime_evaluation_feedback": {"metadata": {"source": "stub"}}},
            )

    def fake_persist(artifacts, *, refresh_dir, refresh_when_empty=False):
        raise OSError("disk full")

    monkeypatch.setattr(
        "victor.evaluation.validated_session_truth_service.persist_validated_session_truth_artifacts",
        fake_persist,
    )
    service = ValidatedSessionTruthService(ValidatedSessionTruthEmitterRegistry([StubEmitter()]))

    saved_path = service.persist_validation_result(
        benchmark=BenchmarkType.SWE_BENCH,
        results_dir=tmp_path / "evaluations",
        task_id="django__123",
        validation_result=object(),
    )

    assert saved_path is None
