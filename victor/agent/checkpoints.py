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

"""Git-based checkpoint system for safe rollback during refactoring operations.

This module provides checkpoint/rollback capability similar to Claude Code and Aider,
using git stash as the underlying mechanism for safe state preservation.

**Design Pattern**: Memento pattern using git as the storage backend.

**Key Features**:
1. Non-destructive: Uses git stash, doesn't create commits
2. Safe: Preserves both staged and unstaged changes
3. Transparent: Works with existing git workflows
4. Automatic cleanup: Old checkpoints are automatically removed

**Usage**:
    manager = GitCheckpointManager()

    # Before making risky changes
    checkpoint = manager.create("Before major refactoring")

    # ... make changes ...

    # If something goes wrong
    if error:
        manager.rollback(checkpoint.id)

    # Or list all checkpoints
    checkpoints = manager.list_checkpoints()

**Integration Points**:
- `orchestrator.py`: Create checkpoints before multi-file edits
- `tool_pipeline.py`: Create checkpoints before batch operations
- `slash_commands.py`: Add `/checkpoint`, `/rollback`, `/checkpoints` commands

**Safety Notes**:
- Checkpoints are git stashes with special naming convention
- Rolling back restores working tree state but doesn't affect git index
- Checkpoints persist across Victor sessions (stored in git)
- Use `cleanup_old()` to prevent stash accumulation
"""

import logging
import re
import subprocess
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class GitCheckpoint:
    """Git stash-based checkpoint for working tree snapshots.

    Renamed from Checkpoint to be semantically distinct:
    - GitCheckpoint (here): Git stash-based with stash_ref/stash_message
    - ExecutionCheckpoint (victor.agent.time_aware_executor): Progress tracking
    - WorkflowCheckpoint (victor.framework.graph): Workflow state persistence
    - HITLCheckpoint (victor.framework.hitl): Human-in-the-loop pause/resume

    Attributes:
        id: Unique checkpoint identifier (UUID)
        timestamp: When the checkpoint was created
        description: Human-readable description of the checkpoint
        stash_ref: Git stash reference (e.g., "stash@{0}")
        stash_message: Full stash message with metadata
    """

    id: str
    timestamp: datetime
    description: str
    stash_ref: Optional[str]
    stash_message: str


class CheckpointError(Exception):
    """Base exception for checkpoint operations."""

    pass


class NotAGitRepositoryError(CheckpointError):
    """Raised when checkpoint operations are attempted outside a git repository."""

    pass


class CheckpointNotFoundError(CheckpointError):
    """Raised when attempting to rollback to a non-existent checkpoint."""

    pass


class GitCheckpointManager:
    """Manages git-based checkpoints for safe rollback.

    This class provides a high-level interface for creating and managing
    checkpoints using git stash as the underlying storage mechanism.

    Thread-safety: This class is NOT thread-safe. Use external locking if
    accessing from multiple threads.

    Note: Renamed from CheckpointManager to GitCheckpointManager to be
    semantically distinct from:
    - GraphCheckpointManager (victor.framework.graph): Graph state checkpoints
    - ConversationCheckpointManager (victor.storage.checkpoints): Conversation state
    """

    PREFIX = "victor_checkpoint_"  # Prefix for checkpoint stash messages

    def __init__(self, repo_path: str = "."):
        """Initialize the checkpoint manager.

        Args:
            repo_path: Path to the git repository (default: current directory)

        Raises:
            NotAGitRepositoryError: If repo_path is not a git repository
        """
        self.repo_path = Path(repo_path).resolve()

        # Verify this is a git repository
        if not self._is_git_repo():
            raise NotAGitRepositoryError(
                f"{self.repo_path} is not a git repository. "
                "Checkpoints require git for state management."
            )

        logger.debug(f"GitCheckpointManager initialized for {self.repo_path}")

    def _is_git_repo(self) -> bool:
        """Check if the current directory is a git repository.

        Returns:
            True if inside a git repository, False otherwise
        """
        try:
            result = self._run_git_command(
                ["rev-parse", "--git-dir"], check=False, capture_output=True
            )
            return result.returncode == 0
        except Exception:
            return False

    def _run_git_command(
        self,
        args: List[str],
        check: bool = True,
        capture_output: bool = True,
    ) -> subprocess.CompletedProcess:
        """Run a git command in the repository.

        Args:
            args: Git command arguments (without 'git' prefix)
            check: Raise CalledProcessError on non-zero exit code
            capture_output: Capture stdout/stderr

        Returns:
            CompletedProcess instance

        Raises:
            subprocess.CalledProcessError: If command fails and check=True
        """
        cmd = ["git", "-C", str(self.repo_path)] + args
        logger.debug(f"Running git command: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=capture_output,
            text=True,
            check=check,
        )

        return result

    def create(self, description: str = "") -> GitCheckpoint:
        """Create a checkpoint before making changes.

        This method creates a git stash with all current changes (staged and unstaged)
        and then immediately reapplies them, preserving the working state while
        creating a recovery point.

        Args:
            description: Human-readable description of the checkpoint

        Returns:
            GitCheckpoint object with metadata

        Raises:
            CheckpointError: If checkpoint creation fails

        Example:
            >>> manager = GitCheckpointManager()
            >>> cp = manager.create("Before refactoring user auth")
            >>> print(cp.id)
            'victor_checkpoint_abc123...'
        """
        # Generate unique checkpoint ID
        checkpoint_id = f"{self.PREFIX}{uuid.uuid4().hex[:12]}"
        timestamp = datetime.now()

        # Build stash message with metadata
        stash_message = f"{checkpoint_id}"
        if description:
            stash_message += f": {description}"

        try:
            # Check if there are any changes to stash
            status = self._run_git_command(["status", "--porcelain"])
            has_changes = bool(status.stdout.strip())

            if not has_changes:
                logger.info("No changes to checkpoint (working tree clean)")
                return GitCheckpoint(
                    id=checkpoint_id,
                    timestamp=timestamp,
                    description="No changes",  # Always "No changes" when tree is clean
                    stash_ref=None,
                    stash_message=stash_message,
                )

            # Create stash with all changes (--include-untracked for safety)
            # Note: We use --keep-index to preserve staged changes
            self._run_git_command(
                [
                    "stash",
                    "push",
                    "--include-untracked",
                    "--message",
                    stash_message,
                ]
            )

            # Get the stash ref (stash@{0} is the most recent)
            stash_ref = "stash@{0}"

            # Reapply the stash to restore working state
            # Use --index to restore both staged and unstaged changes
            try:
                self._run_git_command(["stash", "apply", "--index", stash_ref])
            except subprocess.CalledProcessError:
                # If apply with --index fails, try without (may lose staging info)
                logger.warning("Failed to restore staging info, applying changes only")
                self._run_git_command(["stash", "apply", stash_ref])

            logger.info(f"Created checkpoint {checkpoint_id}: {description or '(no description)'}")

            return GitCheckpoint(
                id=checkpoint_id,
                timestamp=timestamp,
                description=description or "(no description)",
                stash_ref=stash_ref,
                stash_message=stash_message,
            )

        except subprocess.CalledProcessError as e:
            raise CheckpointError(f"Failed to create checkpoint: {e.stderr}") from e

    def rollback(self, checkpoint_id: str, drop_stash: bool = False) -> bool:
        """Restore working tree to a checkpoint state.

        This method finds the checkpoint stash and applies it to the working tree,
        effectively rolling back to the state when the checkpoint was created.

        Args:
            checkpoint_id: ID of the checkpoint to rollback to
            drop_stash: If True, remove the stash after applying (default: False)

        Returns:
            True if rollback succeeded

        Raises:
            CheckpointNotFoundError: If checkpoint doesn't exist
            CheckpointError: If rollback fails

        Example:
            >>> manager.rollback("victor_checkpoint_abc123")
            True
        """
        try:
            # Find the stash entry for this checkpoint
            stash_list = self._run_git_command(["stash", "list"])
            stash_ref = None

            for line in stash_list.stdout.split("\n"):
                if checkpoint_id in line:
                    # Extract stash ref (e.g., "stash@{2}")
                    match = re.match(r"(stash@\{\d+\})", line)
                    if match:
                        stash_ref = match.group(1)
                        break

            if not stash_ref:
                raise CheckpointNotFoundError(
                    f"GitCheckpoint {checkpoint_id} not found in stash list"
                )

            # Reset working tree to clean state
            self._run_git_command(["reset", "--hard"])

            # Apply the checkpoint stash
            try:
                self._run_git_command(["stash", "apply", "--index", stash_ref])
            except subprocess.CalledProcessError:
                # Fallback: apply without --index
                logger.warning("Failed to restore staging info, applying changes only")
                self._run_git_command(["stash", "apply", stash_ref])

            # Optionally drop the stash
            if drop_stash:
                self._run_git_command(["stash", "drop", stash_ref])
                logger.info(f"Dropped stash {stash_ref} after rollback")

            logger.info(f"Rolled back to checkpoint {checkpoint_id}")
            return True

        except CheckpointNotFoundError:
            raise
        except subprocess.CalledProcessError as e:
            raise CheckpointError(f"Failed to rollback to checkpoint: {e.stderr}") from e

    def list_checkpoints(self) -> List[GitCheckpoint]:
        """List all available checkpoints.

        Returns:
            List of GitCheckpoint objects, sorted by timestamp (most recent first)

        Example:
            >>> checkpoints = manager.list_checkpoints()
            >>> for cp in checkpoints:
            ...     print(f"{cp.timestamp}: {cp.description}")
        """
        try:
            result = self._run_git_command(["stash", "list"])
            checkpoints = []

            for line in result.stdout.split("\n"):
                if not line.strip():
                    continue

                # Parse stash list output
                # Format: stash@{N}: On branch: message
                match = re.match(r"(stash@\{\d+\}):\s+(?:On [^:]+|WIP on [^:]+):\s*(.+)", line)
                if not match:
                    continue

                stash_ref = match.group(1)
                stash_message = match.group(2)

                # Check if this is a victor checkpoint
                if not stash_message.startswith(self.PREFIX):
                    continue

                # Extract checkpoint ID and description
                parts = stash_message.split(":", 1)
                checkpoint_id = parts[0].strip()
                description = parts[1].strip() if len(parts) > 1 else ""

                # Get stash timestamp
                timestamp_str = self._run_git_command(
                    ["stash", "list", "--format=%gD", stash_ref]
                ).stdout.strip()

                try:
                    # Parse timestamp (may vary by git version)
                    timestamp = datetime.fromisoformat(timestamp_str.replace("@", ""))
                except (ValueError, AttributeError):
                    # Fallback to current time if parsing fails
                    timestamp = datetime.now()

                checkpoints.append(
                    GitCheckpoint(
                        id=checkpoint_id,
                        timestamp=timestamp,
                        description=description,
                        stash_ref=stash_ref,
                        stash_message=stash_message,
                    )
                )

            # Sort by timestamp (most recent first)
            checkpoints.sort(key=lambda cp: cp.timestamp, reverse=True)

            return checkpoints

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to list checkpoints: {e.stderr}")
            return []

    def cleanup_old(self, keep_count: int = 10) -> int:
        """Remove old checkpoint stashes to prevent accumulation.

        Keeps the N most recent checkpoints and removes the rest.

        Args:
            keep_count: Number of recent checkpoints to keep (default: 10)

        Returns:
            Number of checkpoints removed

        Example:
            >>> removed = manager.cleanup_old(keep_count=5)
            >>> print(f"Removed {removed} old checkpoints")
        """
        checkpoints = self.list_checkpoints()

        if len(checkpoints) <= keep_count:
            logger.info(f"No cleanup needed ({len(checkpoints)} <= {keep_count})")
            return 0

        # Remove old checkpoints (keep_count+1 onwards)
        to_remove = checkpoints[keep_count:]
        removed_count = 0

        for checkpoint in to_remove:
            if checkpoint.stash_ref:
                try:
                    self._run_git_command(["stash", "drop", checkpoint.stash_ref])
                    removed_count += 1
                    logger.debug(f"Removed old checkpoint {checkpoint.id}")
                except subprocess.CalledProcessError as e:
                    logger.warning(f"Failed to remove checkpoint {checkpoint.id}: {e}")

        logger.info(f"Cleaned up {removed_count} old checkpoints")
        return removed_count

    def get_checkpoint(self, checkpoint_id: str) -> Optional[GitCheckpoint]:
        """Get a specific checkpoint by ID.

        Args:
            checkpoint_id: ID of the checkpoint to retrieve

        Returns:
            GitCheckpoint object if found, None otherwise
        """
        checkpoints = self.list_checkpoints()
        for checkpoint in checkpoints:
            if checkpoint.id == checkpoint_id:
                return checkpoint
        return None
