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

"""Tests for the auto-commit module."""

import subprocess
import tempfile
from pathlib import Path

import pytest

from victor.agent.auto_commit import (
    AutoCommitter,
    ChangeType,
    CommitResult,
    get_auto_committer,
)


@pytest.fixture
def temp_git_repo():
    """Create a temporary git repository."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo = Path(tmpdir)

        # Initialize git repo
        subprocess.run(["git", "init"], cwd=repo, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=repo,
            capture_output=True,
        )
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo, capture_output=True)

        # Create initial commit
        (repo / "README.md").write_text("# Test\n")
        subprocess.run(["git", "add", "."], cwd=repo, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=repo, capture_output=True)

        yield repo


@pytest.fixture
def committer(temp_git_repo):
    """Create an AutoCommitter with temp repo."""
    return AutoCommitter(workspace_root=temp_git_repo)


class TestChangeType:
    """Tests for ChangeType enum."""

    def test_all_types_exist(self):
        """All conventional commit types should exist."""
        assert ChangeType.FEAT.value == "feat"
        assert ChangeType.FIX.value == "fix"
        assert ChangeType.REFACTOR.value == "refactor"
        assert ChangeType.DOCS.value == "docs"
        assert ChangeType.TEST.value == "test"
        assert ChangeType.CHORE.value == "chore"


class TestCommitResult:
    """Tests for CommitResult dataclass."""

    def test_success_result(self):
        """CommitResult should store success info."""
        result = CommitResult(
            success=True,
            commit_hash="abc123",
            message="feat: add feature",
            files_committed=["src/api.py"],
        )

        assert result.success is True
        assert result.commit_hash == "abc123"
        assert result.error is None

    def test_failure_result(self):
        """CommitResult should store failure info."""
        result = CommitResult(
            success=False,
            error="Nothing to commit",
        )

        assert result.success is False
        assert result.error == "Nothing to commit"
        assert result.commit_hash is None


class TestAutoCommitter:
    """Tests for AutoCommitter class."""

    def test_is_git_repo(self, committer, temp_git_repo):
        """is_git_repo should detect git repository."""
        assert committer.is_git_repo() is True

        # Test non-git directory
        non_git = AutoCommitter(workspace_root=Path("/tmp"))
        # /tmp might be a git repo, so just test it doesn't crash
        result = non_git.is_git_repo()
        assert isinstance(result, bool)

    def test_has_changes_no_changes(self, committer):
        """has_changes should return False when no changes."""
        assert committer.has_changes() is False

    def test_has_changes_with_changes(self, committer, temp_git_repo):
        """has_changes should return True when files modified."""
        (temp_git_repo / "new_file.py").write_text("# new\n")
        assert committer.has_changes() is True

    def test_get_changed_files(self, committer, temp_git_repo):
        """get_changed_files should list modified files."""
        (temp_git_repo / "file1.py").write_text("# file1\n")
        (temp_git_repo / "file2.py").write_text("# file2\n")

        files = committer.get_changed_files()
        assert "file1.py" in files
        assert "file2.py" in files

    def test_stage_files(self, committer, temp_git_repo):
        """stage_files should add files to staging."""
        (temp_git_repo / "staged.py").write_text("# staged\n")

        result = committer.stage_files(["staged.py"])
        assert result is True

        # Check it's staged
        status = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            cwd=temp_git_repo,
            capture_output=True,
            text=True,
        )
        assert "staged.py" in status.stdout

    def test_generate_commit_message_conventional(self, committer):
        """generate_commit_message should use conventional format."""
        message = committer.generate_commit_message(
            description="Add user authentication",
            change_type="feat",
            scope="auth",
            files=["src/auth.py"],
        )

        assert message.startswith("feat(auth): Add user authentication")
        assert "Generated-by: victor-code-ai" in message
        assert (
            "Co-authored-by: victor-code-ai " "<296282796+victor-code-ai@users.noreply.github.com>"
        ) in message

    def test_generate_commit_message_auto_detect_feat(self, committer):
        """generate_commit_message should auto-detect feat type."""
        message = committer.generate_commit_message(
            description="Add new login feature",
            files=["src/login.py"],
        )

        assert message.startswith("feat:")

    def test_generate_commit_message_auto_detect_fix(self, committer):
        """generate_commit_message should auto-detect fix type."""
        message = committer.generate_commit_message(
            description="Fix bug in authentication",
            files=["src/auth.py"],
        )

        assert message.startswith("fix:")

    def test_generate_commit_message_auto_detect_test(self, committer):
        """generate_commit_message should auto-detect test type."""
        message = committer.generate_commit_message(
            description="Update validation",
            files=["tests/test_auth.py"],
        )

        assert message.startswith("test:")

    def test_generate_commit_message_truncates_long(self, committer):
        """generate_commit_message should truncate long descriptions."""
        long_desc = "A" * 100
        message = committer.generate_commit_message(description=long_desc)

        first_line = message.split("\n")[0]
        assert len(first_line) <= 72

    def test_commit_changes_success(self, committer, temp_git_repo):
        """commit_changes should create a commit."""
        (temp_git_repo / "new_feature.py").write_text("# feature\n")

        result = committer.commit_changes(
            files=["new_feature.py"],
            description="Add new feature",
            change_type="feat",
        )

        assert result.success is True
        assert result.commit_hash is not None
        assert len(result.files_committed) == 1

    def test_commit_changes_no_files(self, committer):
        """commit_changes should fail when no files to commit."""
        result = committer.commit_changes(
            files=[],
            description="Empty commit",
        )

        assert result.success is False
        assert "No files" in result.error

    def test_commit_changes_auto_stage(self, committer, temp_git_repo):
        """commit_changes should auto-stage files."""
        (temp_git_repo / "auto_staged.py").write_text("# auto\n")

        result = committer.commit_changes(
            files=["auto_staged.py"],
            description="Auto staged commit",
            auto_stage=True,
        )

        assert result.success is True

    def test_undo_last_commit(self, committer, temp_git_repo):
        """undo_last_commit should reset last commit."""
        # Create a commit
        (temp_git_repo / "to_undo.py").write_text("# undo\n")
        committer.commit_changes(
            files=["to_undo.py"],
            description="Will be undone",
        )

        # Get commit count before
        log_before = subprocess.run(
            ["git", "rev-list", "--count", "HEAD"],
            cwd=temp_git_repo,
            capture_output=True,
            text=True,
        )
        count_before = int(log_before.stdout.strip())

        # Undo
        result = committer.undo_last_commit(keep_changes=True)
        assert result is True

        # Verify commit count decreased
        log_after = subprocess.run(
            ["git", "rev-list", "--count", "HEAD"],
            cwd=temp_git_repo,
            capture_output=True,
            text=True,
        )
        count_after = int(log_after.stdout.strip())
        assert count_after == count_before - 1

    def test_get_last_commit_info(self, committer, temp_git_repo):
        """get_last_commit_info should return commit details."""
        info = committer.get_last_commit_info()

        assert info is not None
        assert "hash" in info
        assert "subject" in info
        assert "author_name" in info

    def test_is_last_commit_by_victor(self, committer, temp_git_repo):
        """is_last_commit_by_victor should detect Victor commits."""
        # Initial commit is not by Victor
        assert committer.is_last_commit_by_victor() is False

        # Make a Victor commit
        (temp_git_repo / "victor_file.py").write_text("# by victor\n")
        committer.commit_changes(
            files=["victor_file.py"],
            description="Victor change",
        )

        assert committer.is_last_commit_by_victor() is True


class TestGlobalAutoCommitter:
    """Tests for global auto-committer functions."""

    def test_get_auto_committer_singleton(self):
        """get_auto_committer should return same instance."""
        committer1 = get_auto_committer()
        committer2 = get_auto_committer()
        assert committer1 is committer2
