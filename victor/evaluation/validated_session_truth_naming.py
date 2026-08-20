from __future__ import annotations

# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Explicit artifact naming policy for validated session-truth artifacts."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from victor.evaluation.protocol import BenchmarkType


def _safe_task_id(task_id: str) -> str:
    return str(task_id).replace("/", "_").replace("\\", "_").replace(" ", "_")


@dataclass(frozen=True)
class ValidatedSessionTruthArtifactNamingPolicy:
    """Backward-compatible artifact naming policy for session-truth artifacts."""

    def path_for_evaluation_task(
        self,
        *,
        results_dir: Path,
        benchmark: BenchmarkType,
        task_id: str,
        source_result_path: Optional[Path],
    ) -> Path:
        source_stem = source_result_path.stem if source_result_path is not None else "session"
        return results_dir / (
            f"eval_session_{benchmark.value}_{_safe_task_id(task_id)}_{source_stem}.json"
        )

    def path_for_validation_task(
        self,
        *,
        results_dir: Path,
        benchmark: BenchmarkType,
        task_id: str,
        source_result_path: Optional[Path] = None,
    ) -> Path:
        if benchmark == BenchmarkType.SWE_BENCH and source_result_path is None:
            return results_dir / f"eval_session_{_safe_task_id(task_id)}.json"
        return self.path_for_evaluation_task(
            results_dir=results_dir,
            benchmark=benchmark,
            task_id=task_id,
            source_result_path=source_result_path,
        )


def create_default_validated_session_truth_artifact_naming_policy() -> (
    ValidatedSessionTruthArtifactNamingPolicy
):
    """Return the canonical validated session-truth artifact naming policy."""
    return ValidatedSessionTruthArtifactNamingPolicy()
