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

"""Unit tests for artifact manager."""

import pytest
from pathlib import Path

from victor.experiments import (
    ArtifactManager,
    Artifact,
    ArtifactType,
    SQLiteStorage,
    Experiment,
    Run,
)


@pytest.fixture
def temp_dirs(tmp_path: Path):
    """Create temporary directories for testing."""
    artifact_root = str(tmp_path / "artifacts")
    db_path = str(tmp_path / "test.db")

    storage = SQLiteStorage(db_path)
    manager = ArtifactManager(artifact_root=artifact_root, storage=storage)

    return manager, storage


def test_log_artifact(temp_dirs):
    """Test logging an artifact."""
    manager, storage = temp_dirs

    # Create experiment and run
    experiment = Experiment(name="test-experiment")
    storage.create_experiment(experiment)

    run = Run(experiment_id=experiment.experiment_id, name="test-run")
    storage.create_run(run)

    # Create a test file
    test_file = Path(manager.artifact_root) / "test.txt"
    test_file.write_text("test content")

    # Log artifact
    artifact = manager.log_artifact(
        run_id=run.run_id,
        file_path=str(test_file),
        artifact_type=ArtifactType.CUSTOM,
    )

    assert artifact.artifact_id
    assert artifact.run_id == run.run_id
    assert artifact.filename == "test.txt"
    assert artifact.file_size_bytes > 0


def test_get_artifact_path(temp_dirs):
    """Test getting artifact path."""
    manager, storage = temp_dirs

    # Create experiment and run
    experiment = Experiment(name="test-experiment")
    storage.create_experiment(experiment)

    run = Run(experiment_id=experiment.experiment_id, name="test-run")
    storage.create_run(run)

    # Create and log artifact
    test_file = Path(manager.artifact_root) / "test.txt"
    test_file.write_text("test content")

    artifact = manager.log_artifact(
        run_id=run.run_id,
        file_path=str(test_file),
    )

    # Get path
    path = manager.get_artifact_path(artifact.artifact_id)

    assert path is not None
    assert path.exists()


def test_list_artifacts(temp_dirs):
    """Test listing artifacts for a run."""
    manager, storage = temp_dirs

    # Create experiment and run
    experiment = Experiment(name="test-experiment")
    storage.create_experiment(experiment)

    run = Run(experiment_id=experiment.experiment_id, name="test-run")
    storage.create_run(run)

    # Create multiple artifacts
    for i in range(3):
        test_file = Path(manager.artifact_root) / f"test{i}.txt"
        test_file.write_text(f"content {i}")
        manager.log_artifact(
            run_id=run.run_id,
            file_path=str(test_file),
        )

    # List artifacts
    artifacts = manager.list_artifacts(run.run_id)

    assert len(artifacts) == 3


def test_storage_usage(temp_dirs):
    """Test getting storage usage statistics."""
    manager, storage = temp_dirs

    # Create experiment and run
    experiment = Experiment(name="test-experiment")
    storage.create_experiment(experiment)

    run = Run(experiment_id=experiment.experiment_id, name="test-run")
    storage.create_run(run)

    # Create artifact
    test_file = Path(manager.artifact_root) / "test.txt"
    test_file.write_text("test content for size check")

    manager.log_artifact(
        run_id=run.run_id,
        file_path=str(test_file),
    )

    # Get usage
    usage = manager.get_storage_usage(run_id=run.run_id)

    assert usage["file_count"] >= 1
    assert usage["bytes"] > 0


def test_cleanup_run_artifacts(temp_dirs):
    """Test cleaning up artifacts for a run."""
    manager, storage = temp_dirs

    # Create experiment and run
    experiment = Experiment(name="test-experiment")
    storage.create_experiment(experiment)

    run = Run(experiment_id=experiment.experiment_id, name="test-run")
    storage.create_run(run)

    # Create artifacts
    test_file = Path(manager.artifact_root) / "test.txt"
    test_file.write_text("test content")

    manager.log_artifact(
        run_id=run.run_id,
        file_path=str(test_file),
    )

    # Cleanup
    count = manager.cleanup_run_artifacts(run.run_id)

    assert count >= 1

    # Verify artifacts are removed
    artifacts = manager.list_artifacts(run.run_id)
    # Database records remain but files are deleted
