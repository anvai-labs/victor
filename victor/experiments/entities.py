# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Core entity definitions for experiment tracking.

This module defines the primary data structures used throughout the experiment
tracking system, following the design from the experiment tracking design document.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field
from victor.core.constants import DEFAULT_VERTICAL


class ExperimentStatus(str, Enum):
    """Status of an experiment.

    Attributes:
        DRAFT: Experiment created but not yet started
        RUNNING: Actively collecting runs
        PAUSED: Temporarily halted
        COMPLETED: Finished successfully
        FAILED: Failed to complete
        ARCHIVED: No longer active
    """

    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    ARCHIVED = "archived"


class RunStatus(str, Enum):
    """Status of a single run.

    Attributes:
        QUEUED: Waiting to start
        RUNNING: Currently executing
        COMPLETED: Finished successfully
        FAILED: Failed with error
        KILLED: Terminated by user
    """

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    KILLED = "killed"


class ArtifactType(str, Enum):
    """Type of artifact.

    Attributes:
        WORKFLOW_CONFIG: Workflow definition file
        MODEL: Trained model checkpoint
        LOG: Debug or execution log
        VISUALIZATION: Chart or plot
        DATA: Input/output data
        CODE: Source code snapshot
        CUSTOM: Custom artifact type
    """

    WORKFLOW_CONFIG = "workflow_config"
    MODEL = "model"
    LOG = "log"
    VISUALIZATION = "visualization"
    DATA = "data"
    CODE = "code"
    CUSTOM = "custom"


@dataclass
class Metric:
    """A single metric data point.

    Attributes:
        run_id: ID of the run this metric belongs to
        key: Metric name (e.g., "quality_score")
        value: Metric value
        timestamp: When the metric was logged
        step: Optional step number for time-series metrics
    """

    run_id: str
    key: str
    value: float
    timestamp: datetime
    step: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "run_id": self.run_id,
            "key": self.key,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "step": self.step,
        }


@dataclass
class Experiment:
    """A collection of runs testing a hypothesis.

    Experiments group related runs together, typically testing a specific
    hypothesis or comparing different configurations.

    Attributes:
        experiment_id: Unique identifier (UUID)
        name: Human-readable name
        description: Detailed description of the experiment
        hypothesis: What we're testing
        tags: List of tags for categorization
        parameters: Hyperparameters and configuration
        created_at: When the experiment was created
        status: Current status of the experiment
        git_commit_sha: Git commit for reproducibility
        git_branch: Git branch name
        git_dirty: Whether there were uncommitted changes
    """

    experiment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    hypothesis: str = ""
    tags: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    status: ExperimentStatus = ExperimentStatus.DRAFT
    git_commit_sha: str = ""
    git_branch: str = ""
    git_dirty: bool = False

    # Optional fields for advanced use cases
    parent_id: Optional[str] = None
    group_id: Optional[str] = None
    workflow_name: Optional[str] = None
    vertical: str = DEFAULT_VERTICAL
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "description": self.description,
            "hypothesis": self.hypothesis,
            "tags": self.tags,
            "parameters": self.parameters,
            "created_at": self.created_at.isoformat(),
            "status": self.status.value,
            "git_commit_sha": self.git_commit_sha,
            "git_branch": self.git_branch,
            "git_dirty": 1 if self.git_dirty else 0,
            "parent_id": self.parent_id,
            "group_id": self.group_id,
            "workflow_name": self.workflow_name,
            "vertical": self.vertical,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": (self.completed_at.isoformat() if self.completed_at else None),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Experiment":
        """Create from dictionary storage."""
        return cls(
            experiment_id=data["experiment_id"],
            name=data["name"],
            description=data.get("description", ""),
            hypothesis=data.get("hypothesis", ""),
            tags=data.get("tags", []),
            parameters=data.get("parameters", {}),
            created_at=datetime.fromisoformat(data["created_at"]),
            status=ExperimentStatus(data.get("status", ExperimentStatus.DRAFT)),
            git_commit_sha=data.get("git_commit_sha", ""),
            git_branch=data.get("git_branch", ""),
            git_dirty=bool(data.get("git_dirty", 0)),
            parent_id=data.get("parent_id"),
            group_id=data.get("group_id"),
            workflow_name=data.get("workflow_name"),
            vertical=data.get("vertical", "coding"),
            started_at=(
                datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None
            ),
            completed_at=(
                datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None
            ),
        )


@dataclass
class Run:
    """A single execution within an experiment.

    Runs represent individual executions of a workflow or experiment,
    with their own parameters, metrics, and artifacts.

    Attributes:
        run_id: Unique identifier (UUID)
        experiment_id: Parent experiment ID
        name: Human-readable name for this run
        status: Current status
        started_at: When the run started
        completed_at: When the run completed (if finished)
        metrics_summary: Final metric values
        parameters: Run-specific parameters
        error_message: Error message if failed
    """

    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    experiment_id: str = ""
    name: str = ""
    status: RunStatus = RunStatus.QUEUED
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    metrics_summary: Dict[str, float] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None

    # Environment metadata (captured automatically)
    python_version: str = ""
    os_info: str = ""
    victor_version: str = ""
    dependencies: Dict[str, str] = field(default_factory=dict)

    # Provider/model info
    provider: str = ""
    model: str = ""
    task_type: str = ""

    # Artifacts
    artifact_count: int = 0
    artifact_size_bytes: int = 0

    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate run duration in seconds."""
        if self.completed_at and self.started_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "name": self.name,
            "status": self.status.value,
            "started_at": self.started_at.isoformat(),
            "completed_at": (self.completed_at.isoformat() if self.completed_at else None),
            "metrics_summary": self.metrics_summary,
            "parameters": self.parameters,
            "error_message": self.error_message,
            "python_version": self.python_version,
            "os_info": self.os_info,
            "victor_version": self.victor_version,
            "dependencies": self.dependencies,
            "provider": self.provider,
            "model": self.model,
            "task_type": self.task_type,
            "artifact_count": self.artifact_count,
            "artifact_size_bytes": self.artifact_size_bytes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Run":
        """Create from dictionary storage."""
        return cls(
            run_id=data["run_id"],
            experiment_id=data["experiment_id"],
            name=data["name"],
            status=RunStatus(data["status"]),
            started_at=datetime.fromisoformat(data["started_at"]),
            completed_at=(
                datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None
            ),
            metrics_summary=data.get("metrics_summary", {}),
            parameters=data.get("parameters", {}),
            error_message=data.get("error_message"),
            python_version=data.get("python_version", ""),
            os_info=data.get("os_info", ""),
            victor_version=data.get("victor_version", ""),
            dependencies=data.get("dependencies", {}),
            provider=data.get("provider", ""),
            model=data.get("model", ""),
            task_type=data.get("task_type", ""),
            artifact_count=data.get("artifact_count", 0),
            artifact_size_bytes=data.get("artifact_size_bytes", 0),
        )


@dataclass
class Artifact:
    """A file or data associated with a run.

    Artifacts are files generated during a run, such as workflow configs,
    model checkpoints, logs, visualizations, or data files.

    Attributes:
        artifact_id: Unique identifier (UUID)
        run_id: ID of the run this artifact belongs to
        artifact_type: Type of artifact
        filename: Name of the file
        file_path: Path to the file (local or remote)
        file_size_bytes: Size in bytes
        created_at: When the artifact was logged
        metadata: Custom metadata
    """

    artifact_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    run_id: str = ""
    artifact_type: ArtifactType = ArtifactType.CUSTOM
    filename: str = ""
    file_path: str = ""
    file_size_bytes: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "artifact_id": self.artifact_id,
            "run_id": self.run_id,
            "artifact_type": self.artifact_type.value,
            "filename": self.filename,
            "file_path": self.file_path,
            "file_size_bytes": self.file_size_bytes,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Artifact":
        """Create from dictionary storage."""
        return cls(
            artifact_id=data["artifact_id"],
            run_id=data["run_id"],
            artifact_type=ArtifactType(data["artifact_type"]),
            filename=data["filename"],
            file_path=data["file_path"],
            file_size_bytes=data["file_size_bytes"],
            created_at=datetime.fromisoformat(data["created_at"]),
            metadata=data.get("metadata", {}),
        )


# Query and comparison types


class ExperimentQuery(BaseModel):
    """Query for filtering experiments.

    Attributes:
        name_contains: Filter by name substring
        description_contains: Filter by description substring
        tags_any: Experiments with any of these tags
        tags_all: Experiments with all of these tags
        status: Filter by status
        created_after: Filter by creation date (after)
        created_before: Filter by creation date (before)
        metric_name: Filter by metric name
        metric_min: Minimum metric value
        metric_max: Maximum metric value
        parent_id: Filter by parent experiment
        group_id: Filter by experiment group
        provider: Filter by provider
        model: Filter by model
        limit: Maximum number of results
        offset: Offset for pagination
        sort_by: Field to sort by
        sort_order: Sort order (asc or desc)
    """

    name_contains: Optional[str] = None
    description_contains: Optional[str] = None
    tags_any: Optional[List[str]] = None
    tags_all: Optional[List[str]] = None
    status: Optional[ExperimentStatus] = None
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
    metric_name: Optional[str] = None
    metric_min: Optional[float] = None
    metric_max: Optional[float] = None
    parent_id: Optional[str] = None
    group_id: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    limit: int = Field(default=100, ge=1, le=1000)
    offset: int = Field(default=0, ge=0)
    sort_by: str = Field(default="created_at")
    sort_order: str = Field(default="desc", pattern="^(asc|desc)$")


@dataclass
class MetricDiff:
    """Difference in a metric between experiments.

    Attributes:
        metric_name: Name of the metric
        control_value: Value from control experiment
        treatment_value: Value from treatment experiment
        absolute_diff: Absolute difference
        relative_diff: Relative difference (percentage)
        p_value: Statistical significance p-value
        is_significant: Whether difference is statistically significant
        confidence_interval: 95% confidence interval
    """

    metric_name: str
    control_value: float
    treatment_value: float
    absolute_diff: float
    relative_diff: float
    p_value: float = 1.0
    is_significant: bool = False
    confidence_interval: Tuple[float, float] = (0.0, 0.0)


@dataclass
class ExperimentComparison:
    """Result of comparing two or more experiments.

    Attributes:
        experiment_ids: IDs of experiments being compared
        comparison_date: When comparison was performed
        metric_diffs: Metric-by-metric differences
        overall_winner: Experiment ID with best overall performance
        significant_metrics: List of metrics with significant differences
        confidence_level: Statistical confidence level (e.g., 0.95)
        recommendation: Text recommendation
        should_rollback: Whether to rollback to baseline
    """

    experiment_ids: List[str]
    comparison_date: datetime
    metric_diffs: Dict[str, MetricDiff] = field(default_factory=dict)
    overall_winner: Optional[str] = None
    significant_metrics: List[str] = field(default_factory=list)
    confidence_level: float = 0.95
    recommendation: str = ""
    should_rollback: bool = False
