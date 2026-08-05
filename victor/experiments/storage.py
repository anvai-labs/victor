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

"""Storage backend abstraction for experiment tracking.

This module defines the protocol for storage backends, allowing different
storage implementations (SQLite, PostgreSQL, etc.) to be used interchangeably.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol

from victor.experiments.entities import (
    Artifact,
    Experiment,
    ExperimentQuery,
    Metric,
    Run,
)


class StorageBackendError(Exception):
    """Base exception for storage backend errors."""

    pass


class IStorageBackend(Protocol):
    """Protocol for experiment storage backends.

    This protocol defines the interface that all storage backends must implement.
    It provides CRUD operations for experiments, runs, metrics, and artifacts.

    Implementations must be thread-safe for concurrent access.
    """

    # Experiment operations

    def create_experiment(self, experiment: Experiment) -> str:
        """Create a new experiment.

        Args:
            experiment: Experiment to create

        Returns:
            Experiment ID

        Raises:
            StorageBackendError: If creation fails
        """
        ...

    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Get an experiment by ID.

        Args:
            experiment_id: Experiment ID

        Returns:
            Experiment if found, None otherwise
        """
        ...

    def update_experiment(self, experiment_id: str, updates: Dict[str, Any]) -> bool:
        """Update an experiment.

        Args:
            experiment_id: Experiment ID
            updates: Dictionary of fields to update

        Returns:
            True if updated, False if not found

        Raises:
            StorageBackendError: If update fails
        """
        ...

    def delete_experiment(self, experiment_id: str) -> bool:
        """Delete an experiment and all its runs.

        Args:
            experiment_id: Experiment ID

        Returns:
            True if deleted, False if not found

        Raises:
            StorageBackendError: If deletion fails
        """
        ...

    def list_experiments(self, query: Optional[ExperimentQuery] = None) -> List[Experiment]:
        """List experiments with optional filtering.

        Args:
            query: Optional query for filtering

        Returns:
            List of experiments
        """
        ...

    # Run operations

    def create_run(self, run: Run) -> str:
        """Create a new run.

        Args:
            run: Run to create

        Returns:
            Run ID

        Raises:
            StorageBackendError: If creation fails
        """
        ...

    def get_run(self, run_id: str) -> Optional[Run]:
        """Get a run by ID.

        Args:
            run_id: Run ID

        Returns:
            Run if found, None otherwise
        """
        ...

    def update_run(self, run_id: str, updates: Dict[str, Any]) -> bool:
        """Update a run.

        Args:
            run_id: Run ID
            updates: Dictionary of fields to update

        Returns:
            True if updated, False if not found

        Raises:
            StorageBackendError: If update fails
        """
        ...

    def list_runs(self, experiment_id: str) -> List[Run]:
        """List all runs for an experiment.

        Args:
            experiment_id: Experiment ID

        Returns:
            List of runs
        """
        ...

    # Metric operations

    def log_metric(self, metric: Metric) -> None:
        """Log a metric for a run.

        Args:
            metric: Metric to log

        Raises:
            StorageBackendError: If logging fails
        """
        ...

    def get_metrics(self, run_id: str) -> List[Metric]:
        """Get all metrics for a run.

        Args:
            run_id: Run ID

        Returns:
            List of metrics
        """
        ...

    def get_metric_history(self, run_id: str, metric_key: str) -> List[Metric]:
        """Get history of a specific metric.

        Args:
            run_id: Run ID
            metric_key: Metric name

        Returns:
            List of metrics ordered by timestamp
        """
        ...

    # Artifact operations

    def log_artifact(self, artifact: Artifact) -> None:
        """Log an artifact for a run.

        Args:
            artifact: Artifact to log

        Raises:
            StorageBackendError: If logging fails
        """
        ...

    def get_artifacts(self, run_id: str) -> List[Artifact]:
        """Get all artifacts for a run.

        Args:
            run_id: Run ID

        Returns:
            List of artifacts
        """
        ...

    def get_artifact(self, artifact_id: str) -> Optional[Artifact]:
        """Get an artifact by ID.

        Args:
            artifact_id: Artifact ID

        Returns:
            Artifact if found, None otherwise
        """
        ...

    # Utility methods

    def close(self) -> None:
        """Close the storage backend and release resources.

        This method should be called when the backend is no longer needed.
        """
        ...

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
