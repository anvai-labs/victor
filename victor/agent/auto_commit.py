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

"""Auto-commit integration for AI-assisted code changes.

This module provides:
- Automatic git commits after successful AI edits
- Intelligent commit message generation
- Co-authorship attribution for AI changes
- Configurable commit behavior

Usage:
    from victor.agent.auto_commit import AutoCommitter

    committer = AutoCommitter()

    # After successful edit
    result = committer.commit_changes(
        files=["src/api.py"],
        description="Add input validation",
        change_type="feat"
    )
"""

import logging
import re
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional

from victor.core.attribution import append_victor_commit_attribution

logger = logging.getLogger(__name__)


class ChangeType(Enum):
    """Conventional commit types."""

    FEAT = "feat"  # New feature
    FIX = "fix"  # Bug fix
    REFACTOR = "refactor"  # Code refactoring
    DOCS = "docs"  # Documentation
    TEST = "test"  # Tests
    CHORE = "chore"  # Maintenance
    STYLE = "style"  # Formatting
    PERF = "perf"  # Performance
    BUILD = "build"  # Build system
    CI = "ci"  # CI/CD


@dataclass
class CommitResult:
    """Result of a commit operation."""

    success: bool
    commit_hash: Optional[str] = None
    message: str = ""
    files_committed: Optional[List[str]] = None
    error: Optional[str] = None

    def __post_init__(self) -> None:
        if self.files_committed is None:
            self.files_committed = []


class AutoCommitter:
    """Handles automatic git commits for AI-assisted changes.

    Features:
    - Conventional commit format
    - AI co-authorship attribution
    - Smart commit message generation
    - Configurable auto-commit behavior
    """

    COMMIT_SIGNATURE = "Generated-by: victor-code-ai"
    LEGACY_COMMIT_SIGNATURE = "[Victor]"

    def __init__(
        self,
        workspace_root: Optional[Path] = None,
        auto_commit: bool = True,
        use_conventional_commits: bool = True,
    ):
        """Initialize auto-committer.

        Args:
            workspace_root: Root directory of workspace (default: cwd)
            auto_commit: Whether to automatically commit changes
            use_conventional_commits: Whether to use conventional commit format
        """
        self.workspace_root = workspace_root or Path.cwd()
        self.auto_commit = auto_commit
        self.use_conventional_commits = use_conventional_commits

    def _run_git(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        """Run a git command.

        Args:
            *args: Git command arguments
            check: Whether to raise on non-zero exit

        Returns:
            CompletedProcess result
        """
        return subprocess.run(
            ["git"] + list(args),
            capture_output=True,
            text=True,
            timeout=30,
            cwd=self.workspace_root,
            check=check,
        )

    def is_git_repo(self) -> bool:
        """Check if workspace is a git repository."""
        try:
            result = self._run_git("rev-parse", "--git-dir", check=False)
            return result.returncode == 0
        except Exception:
            return False

    def has_changes(self, files: Optional[List[str]] = None) -> bool:
        """Check if there are uncommitted changes.

        Args:
            files: Specific files to check (None = all)

        Returns:
            True if there are changes
        """
        try:
            cmd = ["status", "--porcelain"]
            if files:
                cmd.extend(files)
            result = self._run_git(*cmd, check=False)
            return bool(result.stdout.strip())
        except Exception:
            return False

    def get_changed_files(self) -> List[str]:
        """Get list of changed files."""
        try:
            result = self._run_git("status", "--porcelain", check=False)
            files = []
            for line in result.stdout.strip().split("\n"):
                if line:
                    # Format: "XY filename"
                    filename = line[3:].strip()
                    if " -> " in filename:
                        filename = filename.split(" -> ")[1]
                    if filename:
                        files.append(filename)
            return files
        except Exception:
            return []

    def stage_files(self, files: List[str]) -> bool:
        """Stage files for commit.

        Args:
            files: Files to stage

        Returns:
            True if successful
        """
        if not files:
            return False

        try:
            self._run_git("add", *files)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to stage files: {e.stderr}")
            return False

    def generate_commit_message(
        self,
        description: str,
        change_type: Optional[str] = None,
        scope: Optional[str] = None,
        files: Optional[List[str]] = None,
    ) -> str:
        """Generate a commit message.

        Args:
            description: Description of changes
            change_type: Type of change (feat, fix, etc.)
            scope: Optional scope (e.g., "api", "auth")
            files: Files being changed (for auto-detection)

        Returns:
            Formatted commit message
        """
        # Auto-detect change type if not provided
        if not change_type:
            change_type = self._detect_change_type(description, files)

        # Build message
        if self.use_conventional_commits:
            if scope:
                header = f"{change_type}({scope}): {description}"
            else:
                header = f"{change_type}: {description}"
        else:
            header = description

        # Truncate header to 72 chars
        if len(header) > 72:
            header = header[:69] + "..."

        # Build full message
        parts = [header, ""]

        if files:
            parts.append("Files changed:")
            for f in files[:10]:  # Limit to 10 files
                parts.append(f"  - {f}")
            if len(files) > 10:
                parts.append(f"  ... and {len(files) - 10} more")
            parts.append("")

        return append_victor_commit_attribution("\n".join(parts))

    def _detect_change_type(self, description: str, files: Optional[List[str]] = None) -> str:
        """Auto-detect change type from description and files.

        Args:
            description: Change description
            files: Files being changed

        Returns:
            Change type string
        """
        description_lower = description.lower()

        # Check description keywords
        if any(word in description_lower for word in ["fix", "bug", "error", "issue", "crash"]):
            return ChangeType.FIX.value
        if any(word in description_lower for word in ["add", "new", "feature", "implement"]):
            return ChangeType.FEAT.value
        if any(
            word in description_lower for word in ["refactor", "restructure", "reorganize", "clean"]
        ):
            return ChangeType.REFACTOR.value
        if any(word in description_lower for word in ["test", "spec", "coverage"]):
            return ChangeType.TEST.value
        if any(word in description_lower for word in ["doc", "readme", "comment"]):
            return ChangeType.DOCS.value
        if any(word in description_lower for word in ["style", "format", "lint"]):
            return ChangeType.STYLE.value
        if any(word in description_lower for word in ["perf", "optim", "speed", "fast"]):
            return ChangeType.PERF.value

        # Check file patterns
        if files:
            file_str = " ".join(files).lower()
            if "test" in file_str or "spec" in file_str:
                return ChangeType.TEST.value
            if "readme" in file_str or "doc" in file_str:
                return ChangeType.DOCS.value
            if ".github" in file_str or "ci" in file_str:
                return ChangeType.CI.value

        # Default to chore
        return ChangeType.CHORE.value

    def commit_changes(
        self,
        files: Optional[List[str]] = None,
        description: str = "AI-assisted changes",
        change_type: Optional[str] = None,
        scope: Optional[str] = None,
        auto_stage: bool = True,
    ) -> CommitResult:
        """Commit changes to git.

        Args:
            files: Files to commit (None = all staged/modified)
            description: Description for commit message
            change_type: Type of change (feat, fix, etc.)
            scope: Optional scope
            auto_stage: Whether to auto-stage files

        Returns:
            CommitResult with outcome
        """
        if not self.is_git_repo():
            return CommitResult(
                success=False,
                error="Not a git repository",
            )

        # Get files to commit
        if files is None:
            files = self.get_changed_files()

        if not files:
            return CommitResult(
                success=False,
                error="No files to commit",
            )

        # Stage files if requested
        if auto_stage:
            if not self.stage_files(files):
                return CommitResult(
                    success=False,
                    error="Failed to stage files",
                    files_committed=files,
                )

        # Generate commit message
        message = self.generate_commit_message(
            description=description,
            change_type=change_type,
            scope=scope,
            files=files,
        )

        # Commit
        try:
            result = self._run_git("commit", "-m", message)

            # Extract commit hash
            commit_hash = None
            hash_match = re.search(r"\[[\w-]+\s+([a-f0-9]+)\]", result.stdout)
            if hash_match:
                commit_hash = hash_match.group(1)

            logger.info(f"Committed {len(files)} files: {commit_hash}")

            return CommitResult(
                success=True,
                commit_hash=commit_hash,
                message=message,
                files_committed=files,
            )

        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.strip() if e.stderr else str(e)
            logger.error(f"Commit failed: {error_msg}")

            return CommitResult(
                success=False,
                error=error_msg,
                files_committed=files,
            )

    def undo_last_commit(self, keep_changes: bool = True) -> bool:
        """Undo the last commit.

        Args:
            keep_changes: Whether to keep file changes (soft reset)

        Returns:
            True if successful
        """
        try:
            if keep_changes:
                self._run_git("reset", "--soft", "HEAD~1")
            else:
                self._run_git("reset", "--hard", "HEAD~1")
            logger.info("Undid last commit")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to undo commit: {e.stderr}")
            return False

    def get_last_commit_info(self) -> Optional[dict]:
        """Get information about the last commit.

        Returns:
            Dict with commit info or None
        """
        try:
            # Get basic commit info
            result = self._run_git("log", "-1", "--format=%H|%s|%an|%ae|%ci", check=False)
            if result.returncode == 0 and result.stdout.strip():
                parts = result.stdout.strip().split("|")
                if len(parts) >= 5:
                    # Also get full commit message to check for signature
                    body_result = self._run_git("log", "-1", "--format=%B", check=False)
                    full_message = body_result.stdout if body_result.returncode == 0 else ""

                    return {
                        "hash": parts[0],
                        "subject": parts[1],
                        "author_name": parts[2],
                        "author_email": parts[3],
                        "date": parts[4],
                        "is_victor": (
                            self.COMMIT_SIGNATURE in full_message
                            or self.LEGACY_COMMIT_SIGNATURE in full_message
                            or "Co-authored-by: victor-code-ai" in full_message
                        ),
                    }
        except Exception as e:
            logger.debug("Failed to retrieve last commit info: %s", e)
        return None

    def is_last_commit_by_victor(self) -> bool:
        """Check if the last commit was made by Victor."""
        info = self.get_last_commit_info()
        return info.get("is_victor", False) if info else False


# Default singleton instance
_default_committer: Optional[AutoCommitter] = None


def get_auto_committer() -> AutoCommitter:
    """Get the default auto-committer instance."""
    global _default_committer
    if _default_committer is None:
        _default_committer = AutoCommitter()
    return _default_committer


def set_auto_committer(committer: AutoCommitter) -> None:
    """Set the default auto-committer instance."""
    global _default_committer
    _default_committer = committer
