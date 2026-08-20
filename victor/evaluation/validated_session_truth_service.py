from __future__ import annotations

# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Canonical service for validated session-truth emission and persistence."""

import logging
from pathlib import Path
from typing import Any, Mapping, Optional

from victor.evaluation.protocol import BenchmarkType, EvaluationResult
from victor.evaluation.validated_session_truth_emitters import (
    ValidatedSessionTruthEmissionContext,
    ValidatedSessionTruthEmitterRegistry,
    create_default_validated_session_truth_emitter_registry,
)
from victor.evaluation.validated_session_truth_persistence import (
    persist_validated_session_truth_artifacts,
)

logger = logging.getLogger(__name__)


class ValidatedSessionTruthService:
    """Own emitter resolution, context assembly, and persistence for session truth."""

    def __init__(self, emitters: Optional[ValidatedSessionTruthEmitterRegistry] = None):
        self._emitters = emitters or create_default_validated_session_truth_emitter_registry()

    def persist_evaluation_result(
        self,
        result: EvaluationResult,
        *,
        results_dir: Path,
        source_result_path: Path,
        summary: Optional[Mapping[str, Any]] = None,
        refresh_when_empty: bool = True,
    ) -> list[Path]:
        """Persist per-task validated session-truth artifacts from an evaluation result."""
        results_dir.mkdir(parents=True, exist_ok=True)
        emitter = self._emitters.resolve(result.config.benchmark)
        if emitter is None:
            return []

        summary_payload = dict(summary or result.get_metrics())
        artifacts = []
        for task_result in result.task_results:
            try:
                artifact = emitter.build_artifact(
                    ValidatedSessionTruthEmissionContext(
                        benchmark=result.config.benchmark,
                        results_dir=results_dir,
                        task_id=task_result.task_id,
                        source_result_path=source_result_path,
                        task_result=task_result,
                        config=result.config,
                        evaluation_result=result,
                        summary=summary_payload,
                    )
                )
            except Exception as exc:
                logger.warning(
                    "Validated session-truth emission failed for %s/%s: %s",
                    result.config.benchmark.value,
                    task_result.task_id,
                    exc,
                )
                continue
            if artifact is not None:
                artifacts.append(artifact)

        try:
            return persist_validated_session_truth_artifacts(
                artifacts,
                refresh_dir=results_dir,
                refresh_when_empty=refresh_when_empty,
            )
        except Exception as exc:
            logger.warning(
                "Validated session-truth persistence failed for %s: %s",
                result.config.benchmark.value,
                exc,
            )
            return []

    def persist_validation_result(
        self,
        *,
        benchmark: BenchmarkType,
        results_dir: Path,
        task_id: str,
        validation_result: Any,
        score: Any = None,
        source_result_path: Optional[Path] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        refresh_when_empty: bool = False,
    ) -> Optional[Path]:
        """Persist validated session truth from benchmark-specific validation output."""
        results_dir.mkdir(parents=True, exist_ok=True)
        emitter = self._emitters.resolve(benchmark)
        if emitter is None:
            return None

        try:
            artifact = emitter.build_artifact(
                ValidatedSessionTruthEmissionContext(
                    benchmark=benchmark,
                    results_dir=results_dir,
                    task_id=task_id,
                    source_result_path=source_result_path,
                    validation_result=validation_result,
                    score=score,
                    metadata=dict(metadata or {}),
                )
            )
        except Exception as exc:
            logger.warning(
                "Validated session-truth emission failed for %s/%s: %s",
                benchmark.value,
                task_id,
                exc,
            )
            return None
        if artifact is None:
            return None

        try:
            saved_paths = persist_validated_session_truth_artifacts(
                [artifact],
                refresh_dir=results_dir,
                refresh_when_empty=refresh_when_empty,
            )
        except Exception as exc:
            logger.warning(
                "Validated session-truth persistence failed for %s/%s: %s",
                benchmark.value,
                task_id,
                exc,
            )
            return None
        return saved_paths[0] if saved_paths else None


def create_default_validated_session_truth_service(
    emitters: Optional[ValidatedSessionTruthEmitterRegistry] = None,
) -> ValidatedSessionTruthService:
    """Return the canonical validated session-truth service instance."""
    return ValidatedSessionTruthService(emitters)
