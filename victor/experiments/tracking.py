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

"""Experiment tracking API for Victor.

This module provides the main ExperimentTracker API for creating experiments
and tracking runs, similar to MLflow's tracking API.
"""

from __future__ import annotations

import contextlib
import logging
import os
import platform
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

if TYPE_CHECKING:
    from victor.experiments.artifacts import ArtifactManager

import importlib.metadata

from victor.experiments.entities import (
    Artifact,
    ArtifactType,
    Experiment,
    ExperimentQuery,
    Metric,
    Run,
    RunStatus,
)
from victor.experiments.storage import IStorageBackend
from victor.experiments.sqlite_store import SQLiteStorage

logger = logging.getLogger(__name__)


class ActiveRun:
    """Context manager for an active run.

    This class provides a context manager for automatically managing the
    lifecycle of a run, including logging metrics, parameters, and artifacts.

    Example:
        with experiment.start_run(name="my-run") as run:
            run.log_metric("accuracy", 0.95)
            run.log_param("learning_rate", 0.001)
        # Run is automatically ended when exiting context
    """

    def __init__(
        self,
        run: Run,
        storage: IStorageBackend,
        artifact_manager: Optional["ArtifactManager"] = None,
    ):
        """Initialize active run.

        Args:
            run: Run entity
            storage: Storage backend
            artifact_manager: Optional artifact manager for logging artifacts
        """
        self._run = run
        self._storage = storage
        self._artifact_manager = artifact_manager
        self._ended = False

    @property
    def run_id(self) -> str:
        """Get run ID."""
        return self._run.run_id

    @property
    def experiment_id(self) -> str:
        """Get experiment ID."""
        return self._run.experiment_id

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """Log a metric for this run.

        Args:
            key: Metric name
            value: Metric value
            step: Optional step number for time-series metrics
        """
        if self._ended:
            raise RuntimeError("Cannot log metric: run has ended")

        metric = Metric(
            run_id=self._run.run_id,
            key=key,
            value=value,
            timestamp=datetime.now(timezone.utc),
            step=step,
        )

        self._storage.log_metric(metric)

        # Update metrics summary with latest value
        self._run.metrics_summary[key] = value
        self._storage.update_run(self._run.run_id, {"metrics_summary": self._run.metrics_summary})

        logger.debug(f"Logged metric {key}={value} for run {self._run.run_id}")

    def log_param(self, key: str, value: Any) -> None:
        """Log a parameter for this run.

        Args:
            key: Parameter name
            value: Parameter value (will be converted to string)
        """
        if self._ended:
            raise RuntimeError("Cannot log param: run has ended")

        self._run.parameters[key] = str(value)
        self._storage.update_run(self._run.run_id, {"parameters": self._run.parameters})

        logger.debug(f"Logged param {key}={value} for run {self._run.run_id}")

    def log_artifact(
        self,
        artifact_path: str,
        artifact_type: Union[ArtifactType, str] = ArtifactType.CUSTOM,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Artifact:
        """Log an artifact for this run.

        Args:
            artifact_path: Path to the artifact file
            artifact_type: Type of artifact
            metadata: Optional metadata dictionary

        Returns:
            Created artifact entity

        Raises:
            FileNotFoundError: If artifact file doesn't exist
            RuntimeError: If artifact manager is not available
        """
        if self._ended:
            raise RuntimeError("Cannot log artifact: run has ended")

        if self._artifact_manager is None:
            raise RuntimeError("Artifact manager not configured")

        file_path = Path(artifact_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Artifact file not found: {artifact_path}")

        # Log artifact via artifact manager
        artifact = self._artifact_manager.log_artifact(
            run_id=self._run.run_id,
            file_path=str(file_path),
            artifact_type=artifact_type,
            metadata=metadata or {},
        )

        # Update run artifact counts
        self._run.artifact_count += 1
        self._run.artifact_size_bytes += artifact.file_size_bytes
        self._storage.update_run(
            self._run.run_id,
            {
                "artifact_count": self._run.artifact_count,
                "artifact_size_bytes": self._run.artifact_size_bytes,
            },
        )

        logger.debug(f"Logged artifact {artifact_path} for run {self._run.run_id}")
        return artifact

    def set_status(self, status: Union[RunStatus, str]) -> None:
        """Set the status of this run.

        Args:
            status: New status
        """
        if self._ended:
            raise RuntimeError("Cannot set status: run has ended")

        if isinstance(status, str):
            status = RunStatus(status)

        self._run.status = status
        self._storage.update_run(self._run.run_id, {"status": status.value})

    def set_tag(self, key: str, value: str) -> None:
        """Set a tag for this run.

        Tags are stored in the parameters dictionary with a "tag." prefix.

        Args:
            key: Tag key
            value: Tag value
        """
        self.log_param(f"tag.{key}", value)

    def end_run(self, status: RunStatus = RunStatus.COMPLETED) -> None:
        """End the run manually.

        Args:
            status: Final status of the run
        """
        if self._ended:
            return

        self._ended = True
        self._run.status = status
        self._run.completed_at = datetime.now(timezone.utc)

        self._storage.update_run(
            self._run.run_id,
            {
                "status": status.value,
                "completed_at": self._run.completed_at.isoformat(),
            },
        )

        logger.info(f"Ended run {self._run.run_id} with status {status.value}")

    def __enter__(self) -> "ActiveRun":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager.

        Automatically ends the run when exiting the context.
        If an exception occurred, marks the run as failed.
        """
        if exc_type is not None:
            # Exception occurred - mark run as failed
            self.end_run(status=RunStatus.FAILED)
        else:
            # Normal completion
            self.end_run(status=RunStatus.COMPLETED)


class ExperimentTracker:
    """Main experiment tracking API.

    This class provides a high-level API for creating experiments and tracking
    runs, similar to MLflow's tracking API.

    Example:
        tracker = ExperimentTracker()

        # Create experiment
        experiment = tracker.create_experiment(
            name="my-experiment",
            description="Testing new algorithm",
            tags=["optimization"],
        )

        # Start a run
        with experiment.start_run(name="run-1") as run:
            run.log_metric("accuracy", 0.95)
            run.log_param("learning_rate", 0.001)
    """

    def __init__(self, storage: Optional[IStorageBackend] = None):
        """Initialize experiment tracker.

        Args:
            storage: Storage backend. If None, uses default SQLite storage.
        """
        self._storage = storage or SQLiteStorage()
        self._artifact_manager: Optional[Any] = None  # Will be set later
        self._active_runs_lock = threading.Lock()
        self._active_runs: Dict[str, ActiveRun] = {}

    def set_artifact_manager(self, artifact_manager: Any) -> None:
        """Set the artifact manager for logging artifacts.

        Args:
            artifact_manager: Artifact manager instance
        """
        self._artifact_manager = artifact_manager

    def create_experiment(
        self,
        name: str,
        description: str = "",
        hypothesis: str = "",
        tags: Optional[List[str]] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Experiment:
        """Create a new experiment.

        Args:
            name: Human-readable name
            description: Detailed description
            hypothesis: Hypothesis being tested
            tags: List of tags for categorization
            parameters: Experiment-level parameters

        Returns:
            Created experiment entity
        """
        # Capture reproducibility metadata
        git_info = self._capture_git_info()

        experiment = Experiment(
            name=name,
            description=description,
            hypothesis=hypothesis,
            tags=tags or [],
            parameters=parameters or {},
            git_commit_sha=git_info["commit_sha"],
            git_branch=git_info["branch"],
            git_dirty=git_info["dirty"],
        )

        self._storage.create_experiment(experiment)
        logger.info(f"Created experiment {experiment.experiment_id}: {name}")

        return experiment

    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Get an experiment by ID.

        Args:
            experiment_id: Experiment ID

        Returns:
            Experiment if found, None otherwise
        """
        return self._storage.get_experiment(experiment_id)

    def list_experiments(self, query: Optional[ExperimentQuery] = None) -> List[Experiment]:
        """List experiments with optional filtering.

        Args:
            query: Optional query for filtering

        Returns:
            List of experiments
        """
        return self._storage.list_experiments(query)

    def delete_experiment(self, experiment_id: str) -> bool:
        """Delete an experiment and all its runs.

        Args:
            experiment_id: Experiment ID

        Returns:
            True if deleted, False if not found
        """
        result = self._storage.delete_experiment(experiment_id)
        if result:
            logger.info(f"Deleted experiment {experiment_id}")
        return result

    def start_run(
        self,
        experiment_id: str,
        run_name: str = "",
        parameters: Optional[Dict[str, Any]] = None,
    ) -> ActiveRun:
        """Start a new run for an experiment.

        Args:
            experiment_id: Parent experiment ID
            run_name: Human-readable name for the run
            parameters: Run-specific parameters

        Returns:
            ActiveRun context manager

        Raises:
            ValueError: If experiment doesn't exist
        """
        experiment = self._storage.get_experiment(experiment_id)
        if experiment is None:
            raise ValueError(f"Experiment not found: {experiment_id}")

        # Capture environment metadata
        env_info = self._capture_environment()

        run = Run(
            experiment_id=experiment_id,
            name=run_name or f"run-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}",
            status=RunStatus.RUNNING,
            parameters=parameters or {},
            python_version=env_info["python_version"],
            os_info=env_info["os_info"],
            victor_version=env_info["victor_version"],
            dependencies=env_info["dependencies"],
        )

        self._storage.create_run(run)

        # Create active run context manager
        active_run = ActiveRun(run, self._storage, self._artifact_manager)

        # Track active run
        with self._active_runs_lock:
            self._active_runs[run.run_id] = active_run

        logger.info(f"Started run {run.run_id} for experiment {experiment_id}")
        return active_run

    def get_run(self, run_id: str) -> Optional[Run]:
        """Get a run by ID.

        Args:
            run_id: Run ID

        Returns:
            Run if found, None otherwise
        """
        return self._storage.get_run(run_id)

    def list_runs(self, experiment_id: str) -> List[Run]:
        """List all runs for an experiment.

        Args:
            experiment_id: Experiment ID

        Returns:
            List of runs
        """
        return self._storage.list_runs(experiment_id)

    def get_active_run(self, run_id: str) -> Optional[ActiveRun]:
        """Get an active run by ID.

        Args:
            run_id: Run ID

        Returns:
            ActiveRun if currently active, None otherwise
        """
        with self._active_runs_lock:
            return self._active_runs.get(run_id)

    # Private helper methods

    def _capture_git_info(self) -> Dict[str, Any]:
        """Capture git repository information.

        Returns:
            Dictionary with git metadata
        """
        git_info = {
            "commit_sha": "unknown",
            "branch": "unknown",
            "dirty": False,
        }

        try:
            # Get commit SHA
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                git_info["commit_sha"] = result.stdout.strip()

            # Get branch name
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                git_info["branch"] = result.stdout.strip()

            # Check if working directory is dirty
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=False,
            )
            git_info["dirty"] = len(result.stdout.strip()) > 0

        except (subprocess.SubprocessError, FileNotFoundError):
            # Git not available or not in a git repository
            pass

        return git_info

    def _capture_environment(self) -> Dict[str, Any]:
        """Capture environment information for reproducibility.

        Returns:
            Dictionary with environment metadata
        """
        env_info = {
            "python_version": sys.version,
            "os_info": platform.platform(),
            "victor_version": "unknown",
            "dependencies": {},
        }

        # Get Victor version
        try:
            env_info["victor_version"] = importlib.metadata.version("victor-ai")
        except importlib.metadata.PackageNotFoundError:
            pass

        # Get pip dependencies (optional, may fail in some environments)
        try:
            result = subprocess.run(
                ["pip", "freeze"],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
            if result.returncode == 0:
                # Parse pip freeze output
                for line in result.stdout.strip().split("\n"):
                    if "==" in line:
                        package, version = line.split("==", 1)
                        env_info["dependencies"][package] = version
        except (
            subprocess.SubprocessError,
            FileNotFoundError,
            subprocess.TimeoutExpired,
        ):
            # pip not available or timeout
            pass

        return env_info


# Singleton instance for global access
_tracker: Optional[ExperimentTracker] = None
_tracker_lock = threading.Lock()


def get_experiment_tracker() -> ExperimentTracker:
    """Get the global experiment tracker instance.

    Returns:
        Shared ExperimentTracker instance
    """
    global _tracker
    with _tracker_lock:
        if _tracker is None:
            _tracker = ExperimentTracker()
        return _tracker
