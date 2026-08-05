from __future__ import annotations

# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Shared persistence helpers for validated session-truth artifacts."""

import json
from pathlib import Path
from typing import Iterable

from victor.evaluation.runtime_feedback import (
    refresh_runtime_evaluation_feedback_aggregate,
)
from victor.evaluation.validated_session_truth_emitters import (
    ValidatedSessionTruthArtifact,
)


def persist_validated_session_truth_artifacts(
    artifacts: Iterable[ValidatedSessionTruthArtifact],
    *,
    refresh_dir: Path,
    refresh_when_empty: bool = False,
) -> list[Path]:
    """Persist validated session-truth artifacts and refresh the aggregate once."""
    saved_paths: list[Path] = []
    for artifact in artifacts:
        artifact.path.parent.mkdir(parents=True, exist_ok=True)
        artifact.path.write_text(json.dumps(artifact.record, indent=2))
        saved_paths.append(artifact.path)

    if saved_paths or refresh_when_empty:
        refresh_runtime_evaluation_feedback_aggregate(refresh_dir)

    return saved_paths
