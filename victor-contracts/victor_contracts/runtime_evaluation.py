# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Shared runtime-evaluation feedback contract.

`RuntimeEvaluationFeedback` is the calibrated-threshold overlay produced by the
evaluation subsystem and consumed by the framework/agent runtime. It lives in
``victor_contracts`` (the definition layer everyone imports downward) so that
neither producer (``victor.evaluation``) nor consumer (``victor.framework`` /
``victor.agent``) has to import the other at module scope.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional


@dataclass(frozen=True)
class RuntimeEvaluationFeedback:
    """Calibrated runtime-threshold overlay produced by live or benchmark feedback."""

    completion_threshold: Optional[float] = None
    enhanced_progress_threshold: Optional[float] = None
    minimum_supported_evidence_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize feedback for persistence."""
        return {
            "completion_threshold": self.completion_threshold,
            "enhanced_progress_threshold": self.enhanced_progress_threshold,
            "minimum_supported_evidence_score": self.minimum_supported_evidence_score,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RuntimeEvaluationFeedback":
        """Reconstruct persisted runtime-evaluation feedback."""
        return cls(
            completion_threshold=(
                float(payload["completion_threshold"])
                if payload.get("completion_threshold") is not None
                else None
            ),
            enhanced_progress_threshold=(
                float(payload["enhanced_progress_threshold"])
                if payload.get("enhanced_progress_threshold") is not None
                else None
            ),
            minimum_supported_evidence_score=(
                float(payload["minimum_supported_evidence_score"])
                if payload.get("minimum_supported_evidence_score") is not None
                else None
            ),
            metadata=dict(payload.get("metadata") or {}),
        )


__all__ = ["RuntimeEvaluationFeedback"]
