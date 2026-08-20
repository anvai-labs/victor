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

"""Unit tests for SQLite storage backend."""

import pytest
from datetime import datetime, timezone
from pathlib import Path

from victor.experiments import (
    SQLiteStorage,
    Experiment,
    Run,
    Metric,
    Artifact,
    ArtifactType,
    RunStatus,
    ExperimentStatus,
    ExperimentQuery,
)


@pytest.fixture
def temp_db(tmp_path: Path):
    """Create a temporary database."""
    db_path = str(tmp_path / "test.db")
    storage = SQLiteStorage(db_path)
    return storage


def test_create_experiment(temp_db: SQLiteStorage):
    """Test creating an experiment."""
    experiment = Experiment(
        name="test-experiment",
        description="Test description",
        tags=["test", "unit"],
    )

    exp_id = temp_db.create_experiment(experiment)

    assert exp_id == experiment.experiment_id


def test_get_experiment(temp_db: SQLiteStorage):
    """Test retrieving an experiment."""
    experiment = Experiment(
        name="test-experiment",
        tags=["test"],
    )

    temp_db.create_experiment(experiment)

    retrieved = temp_db.get_experiment(experiment.experiment_id)

    assert retrieved is not None
    assert retrieved.experiment_id == experiment.experiment_id
    assert retrieved.name == experiment.name
    assert retrieved.tags == experiment.tags


def test_update_experiment(temp_db: SQLiteStorage):
    """Test updating an experiment."""
    experiment = Experiment(name="test-experiment")
    temp_db.create_experiment(experiment)

    # Update status
    result = temp_db.update_experiment(
        experiment.experiment_id, {"status": ExperimentStatus.RUNNING}
    )

    assert result is True

    # Verify update
    retrieved = temp_db.get_experiment(experiment.experiment_id)
    assert retrieved.status == ExperimentStatus.RUNNING


def test_delete_experiment(temp_db: SQLiteStorage):
    """Test deleting an experiment."""
    experiment = Experiment(name="test-experiment")
    temp_db.create_experiment(experiment)

    # Delete
    result = temp_db.delete_experiment(experiment.experiment_id)
    assert result is True

    # Verify deletion
    retrieved = temp_db.get_experiment(experiment.experiment_id)
    assert retrieved is None


def test_create_run(temp_db: SQLiteStorage):
    """Test creating a run."""
    # Create experiment first
    experiment = Experiment(name="test-experiment")
    temp_db.create_experiment(experiment)

    # Create run
    run = Run(
        experiment_id=experiment.experiment_id,
        name="test-run",
        status=RunStatus.RUNNING,
    )

    run_id = temp_db.create_run(run)

    assert run_id == run.run_id


def test_get_run(temp_db: SQLiteStorage):
    """Test retrieving a run."""
    # Create experiment
    experiment = Experiment(name="test-experiment")
    temp_db.create_experiment(experiment)

    # Create run
    run = Run(
        experiment_id=experiment.experiment_id,
        name="test-run",
    )

    temp_db.create_run(run)

    # Retrieve
    retrieved = temp_db.get_run(run.run_id)

    assert retrieved is not None
    assert retrieved.run_id == run.run_id
    assert retrieved.name == run.name


def test_update_run(temp_db: SQLiteStorage):
    """Test updating a run."""
    # Create experiment and run
    experiment = Experiment(name="test-experiment")
    temp_db.create_experiment(experiment)

    run = Run(
        experiment_id=experiment.experiment_id,
        name="test-run",
    )
    temp_db.create_run(run)

    # Update status
    result = temp_db.update_run(run.run_id, {"status": RunStatus.COMPLETED})

    assert result is True

    # Verify update
    retrieved = temp_db.get_run(run.run_id)
    assert retrieved.status == RunStatus.COMPLETED


def test_list_runs(temp_db: SQLiteStorage):
    """Test listing runs for an experiment."""
    # Create experiment
    experiment = Experiment(name="test-experiment")
    temp_db.create_experiment(experiment)

    # Create multiple runs
    run1 = Run(experiment_id=experiment.experiment_id, name="run-1")
    run2 = Run(experiment_id=experiment.experiment_id, name="run-2")

    temp_db.create_run(run1)
    temp_db.create_run(run2)

    # List runs
    runs = temp_db.list_runs(experiment.experiment_id)

    assert len(runs) == 2
    assert any(r.run_id == run1.run_id for r in runs)
    assert any(r.run_id == run2.run_id for r in runs)


def test_log_metric(temp_db: SQLiteStorage):
    """Test logging a metric."""
    # Create experiment and run
    experiment = Experiment(name="test-experiment")
    temp_db.create_experiment(experiment)

    run = Run(experiment_id=experiment.experiment_id, name="test-run")
    temp_db.create_run(run)

    # Log metric
    metric = Metric(
        run_id=run.run_id,
        key="accuracy",
        value=0.95,
        timestamp=datetime.now(timezone.utc),
    )

    temp_db.log_metric(metric)

    # Retrieve metrics
    metrics = temp_db.get_metrics(run.run_id)

    assert len(metrics) == 1
    assert metrics[0].key == "accuracy"
    assert metrics[0].value == 0.95


def test_get_metrics(temp_db: SQLiteStorage):
    """Test retrieving all metrics for a run."""
    # Create experiment and run
    experiment = Experiment(name="test-experiment")
    temp_db.create_experiment(experiment)

    run = Run(experiment_id=experiment.experiment_id, name="test-run")
    temp_db.create_run(run)

    # Log multiple metrics
    temp_db.log_metric(
        Metric(run_id=run.run_id, key="m1", value=1.0, timestamp=datetime.now(timezone.utc))
    )
    temp_db.log_metric(
        Metric(run_id=run.run_id, key="m2", value=2.0, timestamp=datetime.now(timezone.utc))
    )
    temp_db.log_metric(
        Metric(run_id=run.run_id, key="m1", value=1.5, timestamp=datetime.now(timezone.utc))
    )

    # Retrieve all metrics
    metrics = temp_db.get_metrics(run.run_id)

    assert len(metrics) == 3


def test_log_artifact(temp_db: SQLiteStorage, tmp_path: Path):
    """Test logging an artifact."""
    # Create experiment and run
    experiment = Experiment(name="test-experiment")
    temp_db.create_experiment(experiment)

    run = Run(experiment_id=experiment.experiment_id, name="test-run")
    temp_db.create_run(run)

    # Create a test file
    test_file = tmp_path / "test.txt"
    test_file.write_text("test content")

    # Create artifact
    artifact = Artifact(
        run_id=run.run_id,
        artifact_type=ArtifactType.CUSTOM,
        filename="test.txt",
        file_path=str(test_file),
        file_size_bytes=test_file.stat().st_size,
    )

    temp_db.log_artifact(artifact)

    # Retrieve artifacts
    artifacts = temp_db.get_artifacts(run.run_id)

    assert len(artifacts) == 1
    assert artifacts[0].filename == "test.txt"
    assert artifacts[0].artifact_type == ArtifactType.CUSTOM


def test_query_experiments(temp_db: SQLiteStorage):
    """Test querying experiments."""
    # Create experiments
    exp1 = Experiment(name="exp-1", tags=["a", "b"])
    exp2 = Experiment(name="exp-2", tags=["b", "c"])

    temp_db.create_experiment(exp1)
    temp_db.create_experiment(exp2)

    # Query with tag filter
    query = ExperimentQuery(tags_any=["a"])
    results = temp_db.list_experiments(query)

    assert len(results) == 1
    assert results[0].experiment_id == exp1.experiment_id


def test_experiment_with_parameters(temp_db: SQLiteStorage):
    """Test experiment with parameters."""
    import json

    experiment = Experiment(
        name="test-experiment",
        parameters={"lr": 0.001, "batch_size": 32},
    )

    temp_db.create_experiment(experiment)

    # Retrieve
    retrieved = temp_db.get_experiment(experiment.experiment_id)

    assert retrieved.parameters == {"lr": 0.001, "batch_size": 32}
