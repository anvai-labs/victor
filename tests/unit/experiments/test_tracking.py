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

"""Unit tests for experiment tracking module."""

import pytest
from datetime import datetime, timezone
from pathlib import Path
from victor.experiments import (
    ExperimentTracker,
    Experiment,
    Run,
    RunStatus,
    ExperimentStatus,
    SQLiteStorage,
    ActiveRun,
)


@pytest.fixture
def temp_db(tmp_path: Path):
    """Create a temporary database for testing."""
    db_path = str(tmp_path / "test.db")
    storage = SQLiteStorage(db_path)
    return storage


@pytest.fixture
def tracker(temp_db: SQLiteStorage):
    """Create a tracker with temporary storage."""
    return ExperimentTracker(storage=temp_db)


def test_create_experiment(tracker: ExperimentTracker):
    """Test creating an experiment."""
    experiment = tracker.create_experiment(
        name="test-experiment",
        description="Test description",
        hypothesis="Test hypothesis",
        tags=["test", "unit"],
    )

    assert experiment.experiment_id
    assert experiment.name == "test-experiment"
    assert experiment.description == "Test description"
    assert experiment.hypothesis == "Test hypothesis"
    assert experiment.tags == ["test", "unit"]
    assert experiment.status == ExperimentStatus.DRAFT


def test_get_experiment(tracker: ExperimentTracker):
    """Test retrieving an experiment."""
    experiment = tracker.create_experiment(name="test-experiment")

    retrieved = tracker.get_experiment(experiment.experiment_id)

    assert retrieved is not None
    assert retrieved.experiment_id == experiment.experiment_id
    assert retrieved.name == experiment.name


def test_list_experiments(tracker: ExperimentTracker):
    """Test listing experiments."""
    # Create multiple experiments
    exp1 = tracker.create_experiment(name="exp-1", tags=["a"])
    exp2 = tracker.create_experiment(name="exp-2", tags=["b"])

    experiments = tracker.list_experiments()

    assert len(experiments) == 2
    assert any(e.experiment_id == exp1.experiment_id for e in experiments)
    assert any(e.experiment_id == exp2.experiment_id for e in experiments)


def test_delete_experiment(tracker: ExperimentTracker):
    """Test deleting an experiment."""
    experiment = tracker.create_experiment(name="test-experiment")

    # Verify it exists
    assert tracker.get_experiment(experiment.experiment_id) is not None

    # Delete it
    result = tracker.delete_experiment(experiment.experiment_id)
    assert result is True

    # Verify it's gone
    assert tracker.get_experiment(experiment.experiment_id) is None


def test_start_run(tracker: ExperimentTracker):
    """Test starting a run."""
    experiment = tracker.create_experiment(name="test-experiment")

    run = tracker.start_run(
        experiment_id=experiment.experiment_id,
        run_name="test-run",
        parameters={"param1": "value1"},
    )

    assert run.run_id
    assert run.experiment_id == experiment.experiment_id
    assert run._run.name == "test-run"
    assert run._run.parameters == {"param1": "value1"}
    assert run._run.status == RunStatus.RUNNING


def test_log_metric(tracker: ExperimentTracker):
    """Test logging metrics."""
    experiment = tracker.create_experiment(name="test-experiment")

    with tracker.start_run(experiment_id=experiment.experiment_id) as run:
        run.log_metric("accuracy", 0.95)
        run.log_metric("loss", 0.05)
        run.log_metric("accuracy", 0.96, step=2)

    # Retrieve run and check metrics
    runs = tracker.list_runs(experiment.experiment_id)
    assert len(runs) == 1

    retrieved_run = runs[0]
    assert "accuracy" in retrieved_run.metrics_summary
    assert retrieved_run.metrics_summary["accuracy"] == 0.96  # Last value


def test_log_param(tracker: ExperimentTracker):
    """Test logging parameters."""
    experiment = tracker.create_experiment(name="test-experiment")

    with tracker.start_run(experiment_id=experiment.experiment_id) as run:
        run.log_param("learning_rate", 0.001)
        run.log_param("batch_size", 32)

    # Retrieve run and check parameters
    runs = tracker.list_runs(experiment.experiment_id)
    assert len(runs) == 1

    retrieved_run = runs[0]
    assert retrieved_run.parameters["learning_rate"] == "0.001"
    assert retrieved_run.parameters["batch_size"] == "32"


def test_run_status_on_complete(tracker: ExperimentTracker):
    """Test that run status is set to completed on normal exit."""
    experiment = tracker.create_experiment(name="test-experiment")

    with tracker.start_run(experiment_id=experiment.experiment_id) as run:
        run.log_metric("test", 1.0)

    # Check run was completed
    runs = tracker.list_runs(experiment.experiment_id)
    assert len(runs) == 1
    assert runs[0].status == RunStatus.COMPLETED
    assert runs[0].completed_at is not None


def test_run_status_on_error(tracker: ExperimentTracker):
    """Test that run status is set to failed on exception."""
    experiment = tracker.create_experiment(name="test-experiment")

    try:
        with tracker.start_run(experiment_id=experiment.experiment_id) as run:
            run.log_metric("test", 1.0)
            raise ValueError("Test error")
    except ValueError:
        pass

    # Check run was failed
    runs = tracker.list_runs(experiment.experiment_id)
    assert len(runs) == 1
    assert runs[0].status == RunStatus.FAILED


def test_manual_run_end(tracker: ExperimentTracker):
    """Test manually ending a run."""
    experiment = tracker.create_experiment(name="test-experiment")

    run = tracker.start_run(experiment_id=experiment.experiment_id)
    run.log_metric("test", 1.0)
    run.end_run(status=RunStatus.COMPLETED)

    # Check run was ended
    runs = tracker.list_runs(experiment.experiment_id)
    assert len(runs) == 1
    assert runs[0].status == RunStatus.COMPLETED


def test_run_duration(tracker: ExperimentTracker):
    """Test that run duration is calculated correctly."""
    import time

    experiment = tracker.create_experiment(name="test-experiment")

    start = datetime.now(timezone.utc)
    run = tracker.start_run(experiment_id=experiment.experiment_id)
    time.sleep(0.1)  # Small delay
    run.end_run()

    # Check duration
    runs = tracker.list_runs(experiment.experiment_id)
    assert len(runs) == 1
    duration = runs[0].duration_seconds
    assert duration is not None
    assert duration >= 0.1  # At least 100ms


def test_experiment_git_info(tracker: ExperimentTracker, tmp_path: Path):
    """Test that git info is captured."""
    # Create experiment
    experiment = tracker.create_experiment(name="test-experiment")

    # Git info should be captured (may be "unknown" if not in git repo)
    assert experiment.git_commit_sha
    assert experiment.git_branch
    assert isinstance(experiment.git_dirty, bool)


def test_run_environment_info(tracker: ExperimentTracker):
    """Test that environment info is captured."""
    experiment = tracker.create_experiment(name="test-experiment")

    with tracker.start_run(experiment_id=experiment.experiment_id) as run:
        pass

    runs = tracker.list_runs(experiment.experiment_id)
    assert len(runs) == 1

    run_data = runs[0]
    assert run_data.python_version
    assert run_data.os_info
    assert run_data.victor_version
