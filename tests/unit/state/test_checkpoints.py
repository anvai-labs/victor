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

"""Tests for the git-based checkpoint system."""

import subprocess
import tempfile
from pathlib import Path

import pytest

from victor.agent.checkpoints import (
    GitCheckpoint,
    GitCheckpointManager,
    CheckpointNotFoundError,
    NotAGitRepositoryError,
)


@pytest.fixture
def git_repo(tmp_path):
    """Create a temporary git repository for testing."""
    repo_path = tmp_path / "test_repo"
    repo_path.mkdir()

    # Initialize git repo
    subprocess.run(
        ["git", "init"],
        cwd=repo_path,
        capture_output=True,
        check=True,
    )

    # Configure git user (required for commits)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repo_path,
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=repo_path,
        capture_output=True,
        check=True,
    )

    # Create initial commit
    test_file = repo_path / "test.txt"
    test_file.write_text("initial content\n")

    subprocess.run(
        ["git", "add", "."],
        cwd=repo_path,
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=repo_path,
        capture_output=True,
        check=True,
    )

    return repo_path


class TestGitCheckpointManagerInit:
    """Tests for GitCheckpointManager initialization."""

    def test_init_in_git_repo(self, git_repo):
        """Test initialization in a git repository."""
        manager = GitCheckpointManager(str(git_repo))
        assert manager.repo_path == git_repo

    def test_init_not_git_repo(self, tmp_path):
        """Test initialization fails outside git repository."""
        with pytest.raises(NotAGitRepositoryError):
            GitCheckpointManager(str(tmp_path))


class TestGitCheckpointCreation:
    """Tests for checkpoint creation."""

    def test_create_checkpoint_no_changes(self, git_repo):
        """Test creating checkpoint with no changes."""
        manager = GitCheckpointManager(str(git_repo))
        checkpoint = manager.create("Clean state")

        assert checkpoint.id.startswith(manager.PREFIX)
        assert checkpoint.description == "No changes"
        assert checkpoint.stash_ref is None

    def test_create_checkpoint_with_changes(self, git_repo):
        """Test creating checkpoint with unstaged changes."""
        # Make changes
        test_file = git_repo / "test.txt"
        test_file.write_text("modified content\n")

        manager = GitCheckpointManager(str(git_repo))
        checkpoint = manager.create("After modifications")

        assert checkpoint.id.startswith(manager.PREFIX)
        assert checkpoint.description == "After modifications"
        assert checkpoint.stash_ref == "stash@{0}"

        # Verify working tree still has changes
        assert test_file.read_text() == "modified content\n"

    def test_create_multiple_checkpoints(self, git_repo):
        """Test creating multiple checkpoints."""
        manager = GitCheckpointManager(str(git_repo))
        test_file = git_repo / "test.txt"

        # Create first checkpoint
        test_file.write_text("version 1\n")
        cp1 = manager.create("Version 1")

        # Create second checkpoint
        test_file.write_text("version 2\n")
        cp2 = manager.create("Version 2")

        assert cp1.id != cp2.id
        checkpoints = manager.list_checkpoints()
        assert len(checkpoints) == 2


class TestGitCheckpointRollback:
    """Tests for checkpoint rollback."""

    def test_rollback_to_checkpoint(self, git_repo):
        """Test rolling back to a checkpoint."""
        manager = GitCheckpointManager(str(git_repo))
        test_file = git_repo / "test.txt"

        # Save original content
        original_content = test_file.read_text()

        # Make changes and create checkpoint
        test_file.write_text("checkpoint state\n")
        checkpoint = manager.create("GitCheckpoint state")

        # Make more changes
        test_file.write_text("latest changes\n")

        # Rollback to checkpoint
        manager.rollback(checkpoint.id)

        # Verify content is restored to checkpoint state
        assert test_file.read_text() == "checkpoint state\n"

    def test_rollback_nonexistent_checkpoint(self, git_repo):
        """Test rollback fails for non-existent checkpoint."""
        manager = GitCheckpointManager(str(git_repo))

        with pytest.raises(CheckpointNotFoundError):
            manager.rollback("victor_checkpoint_nonexistent")

    def test_rollback_with_drop(self, git_repo):
        """Test rollback with stash drop."""
        manager = GitCheckpointManager(str(git_repo))
        test_file = git_repo / "test.txt"

        # Create checkpoint
        test_file.write_text("checkpoint state\n")
        checkpoint = manager.create("To be dropped")

        # Make changes
        test_file.write_text("new changes\n")

        # Rollback and drop
        manager.rollback(checkpoint.id, drop_stash=True)

        # Verify checkpoint was removed
        checkpoints = manager.list_checkpoints()
        assert len(checkpoints) == 0


class TestGitCheckpointListing:
    """Tests for listing checkpoints."""

    def test_list_empty(self, git_repo):
        """Test listing checkpoints when none exist."""
        manager = GitCheckpointManager(str(git_repo))
        checkpoints = manager.list_checkpoints()
        assert checkpoints == []

    def test_list_multiple_checkpoints(self, git_repo):
        """Test listing multiple checkpoints."""
        manager = GitCheckpointManager(str(git_repo))
        test_file = git_repo / "test.txt"

        # Create checkpoints
        test_file.write_text("v1\n")
        cp1 = manager.create("GitCheckpoint 1")

        test_file.write_text("v2\n")
        cp2 = manager.create("GitCheckpoint 2")

        test_file.write_text("v3\n")
        cp3 = manager.create("GitCheckpoint 3")

        # List checkpoints
        checkpoints = manager.list_checkpoints()
        assert len(checkpoints) == 3

        # Verify all checkpoints are present (order may vary by git version)
        checkpoint_ids = {cp.id for cp in checkpoints}
        assert cp1.id in checkpoint_ids
        assert cp2.id in checkpoint_ids
        assert cp3.id in checkpoint_ids

    def test_get_specific_checkpoint(self, git_repo):
        """Test getting a specific checkpoint by ID."""
        manager = GitCheckpointManager(str(git_repo))
        test_file = git_repo / "test.txt"

        # Create checkpoint
        test_file.write_text("specific\n")
        checkpoint = manager.create("Specific checkpoint")

        # Retrieve it
        retrieved = manager.get_checkpoint(checkpoint.id)
        assert retrieved is not None
        assert retrieved.id == checkpoint.id
        assert retrieved.description == "Specific checkpoint"

    def test_get_nonexistent_checkpoint(self, git_repo):
        """Test getting non-existent checkpoint returns None."""
        manager = GitCheckpointManager(str(git_repo))
        retrieved = manager.get_checkpoint("victor_checkpoint_nonexistent")
        assert retrieved is None


class TestGitCheckpointCleanup:
    """Tests for checkpoint cleanup."""

    def test_cleanup_keeps_recent(self, git_repo):
        """Test cleanup keeps recent checkpoints."""
        manager = GitCheckpointManager(str(git_repo))
        test_file = git_repo / "test.txt"

        # Create many checkpoints
        for i in range(15):
            test_file.write_text(f"version {i}\n")
            manager.create(f"GitCheckpoint {i}")

        # Cleanup, keeping 10
        removed = manager.cleanup_old(keep_count=10)

        assert removed == 5
        checkpoints = manager.list_checkpoints()
        assert len(checkpoints) == 10

    def test_cleanup_no_action_when_few(self, git_repo):
        """Test cleanup does nothing when few checkpoints exist."""
        manager = GitCheckpointManager(str(git_repo))
        test_file = git_repo / "test.txt"

        # Create only 3 checkpoints
        for i in range(3):
            test_file.write_text(f"version {i}\n")
            manager.create(f"GitCheckpoint {i}")

        # Try to cleanup with keep_count=10
        removed = manager.cleanup_old(keep_count=10)

        assert removed == 0
        checkpoints = manager.list_checkpoints()
        assert len(checkpoints) == 3
