# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for validated session-truth persistence helpers."""

import json
from unittest.mock import patch

from victor.evaluation.validated_session_truth_emitters import (
    ValidatedSessionTruthArtifact,
)
from victor.evaluation.validated_session_truth_persistence import (
    persist_validated_session_truth_artifacts,
)


def test_persist_validated_session_truth_artifacts_writes_all_and_refreshes_once(
    tmp_path,
):
    artifacts = [
        ValidatedSessionTruthArtifact(
            path=tmp_path / "eval_session_a.json",
            record={"runtime_evaluation_feedback": {"metadata": {"source": "test"}}},
        ),
        ValidatedSessionTruthArtifact(
            path=tmp_path / "nested" / "eval_session_b.json",
            record={"runtime_evaluation_feedback": {"metadata": {"source": "test"}}},
        ),
    ]

    with patch(
        "victor.evaluation.validated_session_truth_persistence.refresh_runtime_evaluation_feedback_aggregate"
    ) as refresh_aggregate:
        saved_paths = persist_validated_session_truth_artifacts(
            artifacts,
            refresh_dir=tmp_path,
        )

    assert saved_paths == [artifact.path for artifact in artifacts]
    assert (
        json.loads(artifacts[0].path.read_text())["runtime_evaluation_feedback"]["metadata"][
            "source"
        ]
        == "test"
    )
    assert artifacts[1].path.exists()
    refresh_aggregate.assert_called_once_with(tmp_path)


def test_persist_validated_session_truth_artifacts_can_refresh_without_artifacts(
    tmp_path,
):
    with patch(
        "victor.evaluation.validated_session_truth_persistence.refresh_runtime_evaluation_feedback_aggregate"
    ) as refresh_aggregate:
        saved_paths = persist_validated_session_truth_artifacts(
            [],
            refresh_dir=tmp_path,
            refresh_when_empty=True,
        )

    assert saved_paths == []
    refresh_aggregate.assert_called_once_with(tmp_path)


def test_persist_validated_session_truth_artifacts_skips_refresh_when_empty_by_default(
    tmp_path,
):
    with patch(
        "victor.evaluation.validated_session_truth_persistence.refresh_runtime_evaluation_feedback_aggregate"
    ) as refresh_aggregate:
        saved_paths = persist_validated_session_truth_artifacts(
            [],
            refresh_dir=tmp_path,
        )

    assert saved_paths == []
    refresh_aggregate.assert_not_called()
