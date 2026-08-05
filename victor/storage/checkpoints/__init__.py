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

"""Time-travel debugging via conversation state checkpoints.

This module provides LangGraph-style checkpoint functionality for conversation
state persistence, enabling replay, forking, branching, and debugging of agent sessions.

Key Features:
- Save/restore conversation state at any point
- Fork sessions from checkpoints to explore alternatives
- Diff checkpoints to understand state changes
- Auto-checkpoint every N tool calls
- SQLite backend for embedded persistence
- Tree-structured checkpoint navigation
- Named branches with merge capabilities
- Replay from any checkpoint

Usage - Basic Checkpointing:
    from victor.storage.checkpoints import ConversationCheckpointManager, SQLiteCheckpointBackend

    backend = SQLiteCheckpointBackend(storage_path)
    manager = ConversationCheckpointManager(backend)

    # Save checkpoint
    checkpoint_id = await manager.save_checkpoint(session_id, state, "before refactor")

    # List checkpoints
    checkpoints = await manager.list_checkpoints(session_id)

    # Restore state
    state = await manager.load_checkpoint(checkpoint_id)

    # Fork session
    new_state = await manager.fork_from_checkpoint(checkpoint_id, new_session_id)

Usage - Multi-Branch Workflows:
    from victor.storage.checkpoints import SQLiteCheckpointBackend
    from victor.storage.checkpoints.tree import BranchManager, CheckpointTree, MergeStrategy

    backend = SQLiteCheckpointBackend(storage_path)
    branch_mgr = BranchManager(backend)

    # Create experiment branch
    await branch_mgr.create_branch("experiment", session_id)
    await branch_mgr.checkout("experiment")

    # ... make changes, create checkpoints ...

    # Merge back to main
    result = await branch_mgr.merge("experiment", "main", session_id)

    # Visualize tree
    tree = await branch_mgr.get_tree(session_id)
    print(tree.to_ascii())
"""

from typing import TYPE_CHECKING

from victor.storage.checkpoints.protocol import (
    CheckpointMetadata,
    CheckpointDiff,
    CheckpointData,
    CheckpointManagerProtocol,
    CheckpointError,
    CheckpointNotFoundError,
    CheckpointStorageError,
    DiffType,
    FieldDiff,
)
from victor.storage.checkpoints.manager import ConversationCheckpointManager

# SQLiteCheckpointBackend requires aiosqlite - lazy import to avoid import errors
if TYPE_CHECKING:
    from victor.storage.checkpoints.backends.sqlite_backend import (
        SQLiteCheckpointBackend,
    )

from victor.storage.checkpoints.tree import (
    BranchStatus,
    MergeStrategy,
    ConflictResolution,
    BranchMetadata,
    CheckpointNode,
    MergeResult,
    ReplayStep,
    CheckpointTree,
    BranchManager,
    BranchStorageProtocol,
)

__all__ = [
    # Protocol types
    "CheckpointMetadata",
    "CheckpointDiff",
    "CheckpointData",
    "CheckpointManagerProtocol",
    "CheckpointError",
    "CheckpointNotFoundError",
    "CheckpointStorageError",
    "DiffType",
    "FieldDiff",
    # Core manager
    "ConversationCheckpointManager",
    "SQLiteCheckpointBackend",
    # Tree/Branch types
    "BranchStatus",
    "MergeStrategy",
    "ConflictResolution",
    "BranchMetadata",
    "CheckpointNode",
    "MergeResult",
    "ReplayStep",
    "CheckpointTree",
    "BranchManager",
    "BranchStorageProtocol",
]


def __getattr__(name: str):
    """Lazy import for backends with optional dependencies."""
    if name == "SQLiteCheckpointBackend":
        from victor.storage.checkpoints.backends.sqlite_backend import (
            SQLiteCheckpointBackend,
        )

        return SQLiteCheckpointBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
