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

"""Integration hooks for experiment tracking.

This module provides integration points for automatic experiment tracking
with WorkflowEngine and ObservabilityBus.

Usage:
    # Enable auto-tracking for workflows
    from victor.experiments.integration import enable_workflow_tracking

    enable_workflow_tracking()

    # Now all workflow executions are automatically tracked
    result = await engine.execute_yaml("workflow.yaml", state)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from victor.experiments import ExperimentTracker, ActiveRun, get_experiment_tracker

logger = logging.getLogger(__name__)

# Global state for active tracking
_active_workflow_run: Optional[ActiveRun] = None


class WorkflowExperimentTracker:
    """Tracker for automatic workflow experiment tracking.

    This class integrates with WorkflowEngine to automatically create
    experiments and runs for workflow executions.

    Example:
        tracker = WorkflowExperimentTracker()
        tracker.enable()

        # Workflows are now automatically tracked
        result = await engine.execute_yaml("workflow.yaml", state)

        tracker.disable()
    """

    def __init__(self, tracker: Optional[ExperimentTracker] = None) -> None:
        """Initialize workflow experiment tracker.

        Args:
            tracker: Experiment tracker instance. If None, uses default.
        """
        self._tracker = tracker or get_experiment_tracker()
        self._enabled = False

    def enable(self) -> None:
        """Enable automatic workflow tracking."""
        if self._enabled:
            logger.warning("Workflow tracking already enabled")
            return

        self._enabled = True
        logger.info("Enabled automatic workflow experiment tracking")

    def disable(self) -> None:
        """Disable automatic workflow tracking."""
        if not self._enabled:
            return

        self._enabled = False
        logger.info("Disabled automatic workflow experiment tracking")

    def is_enabled(self) -> bool:
        """Check if tracking is enabled."""
        return self._enabled

    def on_workflow_start(
        self,
        workflow_name: str,
        workflow_path: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> ActiveRun:
        """Called when a workflow execution starts.

        Args:
            workflow_name: Name of the workflow
            workflow_path: Optional path to workflow file
            params: Optional workflow parameters

        Returns:
            ActiveRun context manager
        """
        if not self._enabled:
            raise RuntimeError("Workflow tracking is not enabled")

        # Create or get experiment
        experiment_name = f"workflow-{workflow_name}"
        experiments = self._tracker.list_experiments()

        # Find existing experiment
        experiment = None
        for exp in experiments:
            if exp.name == experiment_name:
                experiment = exp
                break

        # Create new experiment if needed
        if experiment is None:
            experiment = self._tracker.create_experiment(
                name=experiment_name,
                description=f"Automatic tracking for workflow: {workflow_name}",
                tags=["workflow", "auto-tracked"],
            )

        # Start a run
        run_name = f"{workflow_name}-{workflow_path or 'manual'}"
        run = self._tracker.start_run(
            experiment_id=experiment.experiment_id,
            run_name=run_name,
            params=params or {},
        )

        logger.info(f"Started tracking workflow '{workflow_name}' as run {run.run_id}")

        return run

    def on_workflow_complete(
        self,
        run: ActiveRun,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[Exception] = None,
    ) -> None:
        """Called when a workflow execution completes.

        Args:
            run: Active run from on_workflow_start
            result: Optional workflow result
            error: Optional error if workflow failed
        """
        if error:
            run.set_status("failed")
            if result:
                run.log_param("error_type", type(error).__name__)
                run.log_param("error_message", str(error))
        else:
            # Log result metrics if available
            if result:
                # Extract common metrics
                if "quality_score" in result:
                    run.log_metric("quality_score", result["quality_score"])
                if "success" in result:
                    run.log_metric("success", 1.0 if result["success"] else 0.0)
                if "duration" in result:
                    run.log_metric("duration_seconds", result["duration"])
                if "cost" in result:
                    run.log_metric("cost", result["cost"])

            run.set_status("completed")

        logger.info(f"Completed tracking run {run.run_id}")

    def log_metric(self, key: str, value: float) -> None:
        """Log a metric to the currently active run.

        Args:
            key: Metric name
            value: Metric value
        """
        global _active_workflow_run
        if _active_workflow_run:
            _active_workflow_run.log_metric(key, value)
        else:
            logger.warning(f"No active workflow run to log metric: {key}")

    def log_param(self, key: str, value: Any) -> None:
        """Log a parameter to the currently active run.

        Args:
            key: Parameter name
            value: Parameter value
        """
        global _active_workflow_run
        if _active_workflow_run:
            _active_workflow_run.log_param(key, value)
        else:
            logger.warning(f"No active workflow run to log param: {key}")


# Global tracker instance
_workflow_tracker: Optional[WorkflowExperimentTracker] = None


def get_workflow_tracker() -> WorkflowExperimentTracker:
    """Get the global workflow tracker instance.

    Returns:
        Shared WorkflowExperimentTracker instance
    """
    global _workflow_tracker
    if _workflow_tracker is None:
        _workflow_tracker = WorkflowExperimentTracker()
    return _workflow_tracker


def enable_workflow_tracking(
    tracker: Optional[ExperimentTracker] = None,
) -> WorkflowExperimentTracker:
    """Enable automatic workflow tracking.

    This is a convenience function that creates and enables a tracker.

    Args:
        tracker: Optional experiment tracker instance

    Returns:
        Enabled WorkflowExperimentTracker instance

    Example:
        tracker = enable_workflow_tracking()
        # Workflows are now automatically tracked
    """
    workflow_tracker = WorkflowExperimentTracker(tracker)
    workflow_tracker.enable()
    return workflow_tracker


def disable_workflow_tracking() -> None:
    """Disable automatic workflow tracking.

    Example:
        disable_workflow_tracking()
    """
    global _workflow_tracker
    if _workflow_tracker:
        _workflow_tracker.disable()
