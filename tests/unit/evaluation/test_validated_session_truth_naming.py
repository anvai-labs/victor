# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for validated session-truth artifact naming policy."""

from victor.evaluation.protocol import BenchmarkType
from victor.evaluation.validated_session_truth_naming import (
    create_default_validated_session_truth_artifact_naming_policy,
)


def test_naming_policy_preserves_evaluation_task_artifact_layout(tmp_path):
    policy = create_default_validated_session_truth_artifact_naming_policy()

    path = policy.path_for_evaluation_task(
        results_dir=tmp_path,
        benchmark=BenchmarkType.GUIDE,
        task_id="guide/1",
        source_result_path=tmp_path / "eval_guide_20260425_010101.json",
    )

    assert path == tmp_path / "eval_session_guide_guide_1_eval_guide_20260425_010101.json"


def test_naming_policy_preserves_swe_bench_validation_layout(tmp_path):
    policy = create_default_validated_session_truth_artifact_naming_policy()

    path = policy.path_for_validation_task(
        results_dir=tmp_path,
        benchmark=BenchmarkType.SWE_BENCH,
        task_id="django__123",
    )

    assert path == tmp_path / "eval_session_django__123.json"


def test_naming_policy_can_fallback_to_evaluation_layout_for_non_swe_validation(
    tmp_path,
):
    policy = create_default_validated_session_truth_artifact_naming_policy()

    path = policy.path_for_validation_task(
        results_dir=tmp_path,
        benchmark=BenchmarkType.DR3_EVAL,
        task_id="dr3 123",
        source_result_path=tmp_path / "eval_dr3_eval_20260425_010101.json",
    )

    assert path == tmp_path / "eval_session_dr3_eval_dr3_123_eval_dr3_eval_20260425_010101.json"
