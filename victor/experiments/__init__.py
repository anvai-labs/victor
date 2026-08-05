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

"""Experiment Tracking System for Victor.

This package provides MLflow-like experiment tracking for workflow optimization,
A/B testing, and hyperparameter tuning.

Key Features:
- Create and manage experiments
- Track runs with metrics and parameters
- Log and retrieve artifacts
- Query and compare experiments
- Automatic reproducibility metadata capture

Usage:
    from victor.experiments import ExperimentTracker

    tracker = ExperimentTracker()

    # Create experiment
    experiment = tracker.create_experiment(
        name="tool-selector-optimization",
        description="Optimize tool selection thresholds",
        tags=["optimization", "tool-selection"],
    )

    # Start a run
    with tracker.start_run(experiment.experiment_id, name="run-1") as run:
        run.log_metric("quality_score", 0.85)
        run.log_param("tool_budget", 10)
        run.log_artifact("config.json", "/path/to/config.json")

Architecture:
    ExperimentTracker (Public API)
    ├── ActiveRun (Context manager for single run)
    ├── IStorageBackend (Protocol for storage)
    ├── SQLiteStorage (Default storage implementation)
    ├── ArtifactManager (Artifact storage)
    └── MetricsAggregator (Metric statistics)
"""

from victor.experiments.entities import (
    # Core entities
    Experiment,
    Run,
    Metric,
    Artifact,
    # Enums
    ExperimentStatus,
    RunStatus,
    ArtifactType,
    # Query types
    ExperimentQuery,
    MetricDiff,
    ExperimentComparison,
)

from victor.experiments.tracking import (
    ExperimentTracker,
    ActiveRun,
    get_experiment_tracker,
)

from victor.experiments.storage import (
    IStorageBackend,
    StorageBackendError,
)

from victor.experiments.sqlite_store import (
    SQLiteStorage,
    get_default_storage,
)

from victor.experiments.artifacts import (
    ArtifactManager,
    get_artifact_manager,
)

from victor.experiments.metrics import (
    MetricsAggregator,
    MetricStatistics,
    get_metrics_history,
    get_best_run,
)

__all__ = [
    # Entities
    "Experiment",
    "Run",
    "Metric",
    "Artifact",
    # Enums
    "ExperimentStatus",
    "RunStatus",
    "ArtifactType",
    # Query types
    "ExperimentQuery",
    "MetricDiff",
    "ExperimentComparison",
    # Tracking
    "ExperimentTracker",
    "ActiveRun",
    "get_experiment_tracker",
    # Storage
    "IStorageBackend",
    "StorageBackendError",
    "SQLiteStorage",
    "get_default_storage",
    # Artifacts
    "ArtifactManager",
    "get_artifact_manager",
    # Metrics
    "MetricsAggregator",
    "MetricStatistics",
    "get_metrics_history",
    "get_best_run",
]

# Version info
__version__ = "0.1.0"
