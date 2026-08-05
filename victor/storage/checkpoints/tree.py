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

"""Tree-structured checkpoint management for multi-branch workflows.

This module extends the checkpoint system with LangGraph-style branching
capabilities, enabling:
- Named branches (e.g., "main", "experiment-1")
- Tree navigation (ancestors, descendants, siblings)
- Branch merging and conflict resolution
- Replay from any checkpoint through a sequence
- Branch comparison and visualization

Design Philosophy:
- Builds on existing ConversationCheckpointManager and backends
- Checkpoints form an immutable DAG (directed acyclic graph)
- Branches are lightweight pointers to head checkpoints
- Tree structure enables time-travel and parallel exploration

Usage:
    from victor.storage.checkpoints import SQLiteCheckpointBackend
    from victor.storage.checkpoints.tree import BranchManager, CheckpointTree

    backend = SQLiteCheckpointBackend()
    branch_mgr = BranchManager(backend)

    # Create a branch
    await branch_mgr.create_branch("experiment", session_id)

    # Switch to branch and make changes
    await branch_mgr.checkout("experiment")
    # ... make changes, create checkpoints ...

    # Merge back to main
    await branch_mgr.merge("experiment", "main")

    # Get tree visualization
    tree = await branch_mgr.get_tree(session_id)
    print(tree.to_ascii())
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    runtime_checkable,
)

from victor.storage.checkpoints.protocol import (
    CheckpointData,
    CheckpointDiff,
    CheckpointManagerProtocol,
    CheckpointMetadata,
    CheckpointNotFoundError,
    CheckpointError,
    DiffType,
    FieldDiff,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# =============================================================================
# Branch Data Structures
# =============================================================================


class BranchStatus(str, Enum):
    """Status of a checkpoint branch."""

    ACTIVE = "active"
    MERGED = "merged"
    ARCHIVED = "archived"
    DELETED = "deleted"


class MergeStrategy(str, Enum):
    """Strategy for merging branches."""

    FAST_FORWARD = "fast_forward"  # Move pointer if linear history
    THREE_WAY = "three_way"  # Create merge checkpoint
    REBASE = "rebase"  # Replay changes on top of target
    SQUASH = "squash"  # Combine all changes into single checkpoint


class ConflictResolution(str, Enum):
    """How to resolve conflicts during merge."""

    OURS = "ours"  # Keep current branch changes
    THEIRS = "theirs"  # Keep incoming branch changes
    MANUAL = "manual"  # Require manual resolution
    UNION = "union"  # Combine both (for compatible changes)


@dataclass
class BranchMetadata:
    """Metadata for a named checkpoint branch.

    Branches are lightweight pointers to checkpoint heads that enable
    parallel exploration of conversation states.

    Attributes:
        branch_id: Unique identifier for the branch
        name: Human-readable branch name
        session_id: Session this branch belongs to
        head_checkpoint_id: ID of the current head checkpoint
        base_checkpoint_id: ID of the checkpoint this branch started from
        created_at: When the branch was created
        updated_at: Last update timestamp
        status: Current branch status
        description: Optional branch description
        tags: Optional tags for categorization
        merge_parent_id: If merged, the checkpoint it was merged into
    """

    branch_id: str
    name: str
    session_id: str
    head_checkpoint_id: str
    base_checkpoint_id: str
    created_at: datetime
    updated_at: datetime
    status: BranchStatus = BranchStatus.ACTIVE
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    merge_parent_id: Optional[str] = None

    @classmethod
    def create(
        cls,
        name: str,
        session_id: str,
        head_checkpoint_id: str,
        base_checkpoint_id: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> "BranchMetadata":
        """Create a new branch.

        Args:
            name: Branch name (must be unique within session)
            session_id: Session identifier
            head_checkpoint_id: Current head checkpoint
            base_checkpoint_id: Where branch started (defaults to head)
            description: Optional description
            tags: Optional tags

        Returns:
            New BranchMetadata instance
        """
        now = datetime.now(timezone.utc)
        return cls(
            branch_id=f"branch_{uuid.uuid4().hex[:12]}",
            name=name,
            session_id=session_id,
            head_checkpoint_id=head_checkpoint_id,
            base_checkpoint_id=base_checkpoint_id or head_checkpoint_id,
            created_at=now,
            updated_at=now,
            description=description,
            tags=tags or [],
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "branch_id": self.branch_id,
            "name": self.name,
            "session_id": self.session_id,
            "head_checkpoint_id": self.head_checkpoint_id,
            "base_checkpoint_id": self.base_checkpoint_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "status": self.status.value,
            "description": self.description,
            "tags": self.tags,
            "merge_parent_id": self.merge_parent_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BranchMetadata":
        """Deserialize from dictionary."""
        return cls(
            branch_id=data["branch_id"],
            name=data["name"],
            session_id=data["session_id"],
            head_checkpoint_id=data["head_checkpoint_id"],
            base_checkpoint_id=data["base_checkpoint_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            status=BranchStatus(data.get("status", "active")),
            description=data.get("description"),
            tags=data.get("tags", []),
            merge_parent_id=data.get("merge_parent_id"),
        )


@dataclass
class CheckpointNode:
    """A node in the checkpoint tree.

    Represents a single checkpoint with its relationships to other
    checkpoints in the tree structure.

    Attributes:
        checkpoint_id: Unique checkpoint identifier
        parent_id: Parent checkpoint ID (None for root)
        children_ids: List of child checkpoint IDs
        branch_name: Name of branch this checkpoint belongs to
        metadata: Full checkpoint metadata
        depth: Depth in tree (root = 0)
    """

    checkpoint_id: str
    parent_id: Optional[str]
    children_ids: List[str] = field(default_factory=list)
    branch_name: Optional[str] = None
    metadata: Optional[CheckpointMetadata] = None
    depth: int = 0

    def is_root(self) -> bool:
        """Check if this is the root checkpoint."""
        return self.parent_id is None

    def is_leaf(self) -> bool:
        """Check if this is a leaf checkpoint (no children)."""
        return len(self.children_ids) == 0

    def is_branch_point(self) -> bool:
        """Check if this is a branch point (multiple children)."""
        return len(self.children_ids) > 1


@dataclass
class MergeResult:
    """Result of a branch merge operation.

    Attributes:
        success: Whether merge succeeded
        merge_checkpoint_id: ID of merge checkpoint (if created)
        strategy_used: Merge strategy that was applied
        conflicts: List of conflicts (if any)
        changes_merged: Summary of changes merged
    """

    success: bool
    merge_checkpoint_id: Optional[str] = None
    strategy_used: Optional[MergeStrategy] = None
    conflicts: List[Dict[str, Any]] = field(default_factory=list)
    changes_merged: Dict[str, int] = field(default_factory=dict)

    def has_conflicts(self) -> bool:
        """Check if merge has unresolved conflicts."""
        return len(self.conflicts) > 0


@dataclass
class ReplayStep:
    """A step in checkpoint replay.

    Represents one step when replaying from a checkpoint.

    Attributes:
        checkpoint_id: Checkpoint at this step
        action: Action to replay (tool call, message, etc.)
        state_before: State before action
        state_after: State after action
        step_index: Index in replay sequence
    """

    checkpoint_id: str
    action: Dict[str, Any]
    state_before: Optional[Dict[str, Any]] = None
    state_after: Optional[Dict[str, Any]] = None
    step_index: int = 0


# =============================================================================
# Checkpoint Tree
# =============================================================================


class CheckpointTree:
    """Tree structure for checkpoint navigation and visualization.

    Provides efficient tree operations on checkpoint history:
    - Build tree from checkpoint list
    - Navigate ancestors/descendants
    - Find common ancestors for merging
    - Generate visualizations
    """

    def __init__(self):
        """Initialize empty tree."""
        self._nodes: Dict[str, CheckpointNode] = {}
        self._root_id: Optional[str] = None
        self._branches: Dict[str, BranchMetadata] = {}

    @classmethod
    async def build_from_session(
        cls,
        backend: CheckpointManagerProtocol,
        session_id: str,
        branches: Optional[Dict[str, BranchMetadata]] = None,
    ) -> "CheckpointTree":
        """Build tree from all checkpoints in a session.

        Args:
            backend: Checkpoint storage backend
            session_id: Session to build tree for
            branches: Optional branch metadata to include

        Returns:
            Populated CheckpointTree
        """
        tree = cls()

        # Get all checkpoints for session
        checkpoints = await backend.list_checkpoints(session_id, limit=1000)

        if not checkpoints:
            return tree

        # Build nodes
        for cp in checkpoints:
            node = CheckpointNode(
                checkpoint_id=cp.checkpoint_id,
                parent_id=cp.parent_id,
                metadata=cp,
            )
            tree._nodes[cp.checkpoint_id] = node

        # Establish parent-child relationships
        for node in tree._nodes.values():
            if node.parent_id and node.parent_id in tree._nodes:
                parent = tree._nodes[node.parent_id]
                if node.checkpoint_id not in parent.children_ids:
                    parent.children_ids.append(node.checkpoint_id)
                node.depth = parent.depth + 1
            elif node.parent_id is None:
                tree._root_id = node.checkpoint_id

        # Add branch info
        if branches:
            tree._branches = branches
            for branch in branches.values():
                if branch.head_checkpoint_id in tree._nodes:
                    tree._nodes[branch.head_checkpoint_id].branch_name = branch.name

        return tree

    def get_node(self, checkpoint_id: str) -> Optional[CheckpointNode]:
        """Get node by checkpoint ID."""
        return self._nodes.get(checkpoint_id)

    def get_root(self) -> Optional[CheckpointNode]:
        """Get root node."""
        if self._root_id:
            return self._nodes.get(self._root_id)
        return None

    def get_ancestors(
        self,
        checkpoint_id: str,
        max_depth: Optional[int] = None,
    ) -> List[CheckpointNode]:
        """Get all ancestors of a checkpoint (parent to root).

        Args:
            checkpoint_id: Starting checkpoint
            max_depth: Maximum number of ancestors to return

        Returns:
            List of ancestor nodes, ordered parent to root
        """
        ancestors = []
        current = self._nodes.get(checkpoint_id)

        depth = 0
        while current and current.parent_id:
            if max_depth and depth >= max_depth:
                break
            parent = self._nodes.get(current.parent_id)
            if parent:
                ancestors.append(parent)
                current = parent
                depth += 1
            else:
                break

        return ancestors

    def get_descendants(
        self,
        checkpoint_id: str,
        max_depth: Optional[int] = None,
    ) -> List[CheckpointNode]:
        """Get all descendants of a checkpoint (children to leaves).

        Args:
            checkpoint_id: Starting checkpoint
            max_depth: Maximum depth to traverse

        Returns:
            List of descendant nodes in breadth-first order
        """
        descendants = []
        node = self._nodes.get(checkpoint_id)

        if not node:
            return descendants

        # BFS traversal
        queue = [(child_id, 1) for child_id in node.children_ids]

        while queue:
            child_id, depth = queue.pop(0)
            if max_depth and depth > max_depth:
                continue

            child = self._nodes.get(child_id)
            if child:
                descendants.append(child)
                queue.extend((grandchild_id, depth + 1) for grandchild_id in child.children_ids)

        return descendants

    def get_path(
        self,
        from_checkpoint: str,
        to_checkpoint: str,
    ) -> List[CheckpointNode]:
        """Get path between two checkpoints.

        Args:
            from_checkpoint: Starting checkpoint
            to_checkpoint: Target checkpoint

        Returns:
            List of nodes forming the path, or empty if no path exists
        """
        # Get ancestors of both
        from_ancestors = set(n.checkpoint_id for n in self.get_ancestors(from_checkpoint))
        from_ancestors.add(from_checkpoint)

        to_ancestors = set(n.checkpoint_id for n in self.get_ancestors(to_checkpoint))
        to_ancestors.add(to_checkpoint)

        # Find common ancestor
        common = from_ancestors & to_ancestors
        if not common:
            return []

        # Find the deepest common ancestor
        common_ancestor = max(
            common,
            key=lambda cid: self._nodes[cid].depth if cid in self._nodes else 0,
        )

        # Build path: from_checkpoint -> common_ancestor -> to_checkpoint
        path = []

        # Path from from_checkpoint to common ancestor
        current = from_checkpoint
        while current != common_ancestor:
            node = self._nodes.get(current)
            if node:
                path.append(node)
                current = node.parent_id
            else:
                break

        # Add common ancestor
        if common_ancestor in self._nodes:
            path.append(self._nodes[common_ancestor])

        # Path from common ancestor to to_checkpoint (reversed)
        to_path = []
        current = to_checkpoint
        while current != common_ancestor:
            node = self._nodes.get(current)
            if node:
                to_path.append(node)
                current = node.parent_id
            else:
                break
        to_path.reverse()
        path.extend(to_path)

        return path

    def find_common_ancestor(
        self,
        checkpoint_a: str,
        checkpoint_b: str,
    ) -> Optional[CheckpointNode]:
        """Find the common ancestor of two checkpoints.

        Used for three-way merge to find the base checkpoint.

        Args:
            checkpoint_a: First checkpoint
            checkpoint_b: Second checkpoint

        Returns:
            Common ancestor node, or None if none exists
        """
        ancestors_a = set(n.checkpoint_id for n in self.get_ancestors(checkpoint_a))
        ancestors_a.add(checkpoint_a)

        ancestors_b = set(n.checkpoint_id for n in self.get_ancestors(checkpoint_b))
        ancestors_b.add(checkpoint_b)

        common = ancestors_a & ancestors_b
        if not common:
            return None

        # Return deepest common ancestor
        deepest_id = max(
            common,
            key=lambda cid: self._nodes[cid].depth if cid in self._nodes else 0,
        )
        return self._nodes.get(deepest_id)

    def get_branch_points(self) -> List[CheckpointNode]:
        """Get all checkpoints that are branch points (have multiple children)."""
        return [n for n in self._nodes.values() if n.is_branch_point()]

    def get_leaves(self) -> List[CheckpointNode]:
        """Get all leaf checkpoints (no children)."""
        return [n for n in self._nodes.values() if n.is_leaf()]

    def get_branch_heads(self) -> Dict[str, CheckpointNode]:
        """Get head checkpoints for all branches."""
        heads = {}
        for branch_name, branch in self._branches.items():
            if branch.head_checkpoint_id in self._nodes:
                heads[branch_name] = self._nodes[branch.head_checkpoint_id]
        return heads

    def to_dict(self) -> Dict[str, Any]:
        """Serialize tree to dictionary."""
        return {
            "root_id": self._root_id,
            "nodes": {
                cid: {
                    "checkpoint_id": n.checkpoint_id,
                    "parent_id": n.parent_id,
                    "children_ids": n.children_ids,
                    "branch_name": n.branch_name,
                    "depth": n.depth,
                }
                for cid, n in self._nodes.items()
            },
            "branches": {name: branch.to_dict() for name, branch in self._branches.items()},
        }

    def to_ascii(self, max_depth: Optional[int] = None) -> str:
        """Generate ASCII tree visualization.

        Args:
            max_depth: Maximum depth to display

        Returns:
            ASCII art tree representation
        """
        if not self._root_id:
            return "Empty tree"

        lines = ["Checkpoint Tree", "=" * 50]
        self._ascii_node(self._root_id, "", True, lines, max_depth, 0)
        return "\n".join(lines)

    def _ascii_node(
        self,
        checkpoint_id: str,
        prefix: str,
        is_last: bool,
        lines: List[str],
        max_depth: Optional[int],
        current_depth: int,
    ) -> None:
        """Recursively build ASCII tree."""
        if max_depth and current_depth > max_depth:
            return

        node = self._nodes.get(checkpoint_id)
        if not node:
            return

        # Build node line
        connector = "└── " if is_last else "├── "
        branch_info = f" [{node.branch_name}]" if node.branch_name else ""
        desc = ""
        if node.metadata and node.metadata.description:
            desc = f" - {node.metadata.description[:30]}"

        lines.append(f"{prefix}{connector}{checkpoint_id[:12]}...{branch_info}{desc}")

        # Recurse to children
        child_prefix = prefix + ("    " if is_last else "│   ")
        for i, child_id in enumerate(node.children_ids):
            is_last_child = i == len(node.children_ids) - 1
            self._ascii_node(
                child_id,
                child_prefix,
                is_last_child,
                lines,
                max_depth,
                current_depth + 1,
            )


# =============================================================================
# Branch Manager
# =============================================================================


class BranchManager:
    """High-level API for branch operations.

    Provides git-like branching semantics for checkpoint trees:
    - create_branch: Create a new branch
    - checkout: Switch to a branch
    - merge: Merge branches together
    - delete_branch: Remove a branch
    - list_branches: List all branches
    - replay: Replay from checkpoint

    Thread Safety: NOT thread-safe. Use external locking for concurrent access.
    """

    def __init__(
        self,
        backend: CheckpointManagerProtocol,
        default_branch_name: str = "main",
    ):
        """Initialize branch manager.

        Args:
            backend: Checkpoint storage backend
            default_branch_name: Name for default branch
        """
        self.backend = backend
        self.default_branch_name = default_branch_name

        # In-memory branch storage (should be persisted in production)
        self._branches: Dict[str, Dict[str, BranchMetadata]] = {}  # session_id -> {name: branch}
        self._current_branch: Dict[str, str] = {}  # session_id -> branch_name

    async def create_branch(
        self,
        name: str,
        session_id: str,
        from_checkpoint: Optional[str] = None,
        description: Optional[str] = None,
    ) -> BranchMetadata:
        """Create a new branch.

        Args:
            name: Branch name (must be unique within session)
            session_id: Session identifier
            from_checkpoint: Checkpoint to branch from (default: current head)
            description: Optional branch description

        Returns:
            Created BranchMetadata

        Raises:
            CheckpointError: If branch name already exists
        """
        # Initialize session branches if needed
        if session_id not in self._branches:
            self._branches[session_id] = {}

        # Check for duplicate name
        if name in self._branches[session_id]:
            raise CheckpointError(f"Branch '{name}' already exists in session {session_id}")

        # Get base checkpoint
        if from_checkpoint is None:
            # Use head of current branch
            current = self._current_branch.get(session_id)
            if current and current in self._branches[session_id]:
                from_checkpoint = self._branches[session_id][current].head_checkpoint_id
            else:
                # Get most recent checkpoint
                checkpoints = await self.backend.list_checkpoints(session_id, limit=1)
                if checkpoints:
                    from_checkpoint = checkpoints[0].checkpoint_id
                else:
                    raise CheckpointError(f"No checkpoints found for session {session_id}")

        # Create branch
        branch = BranchMetadata.create(
            name=name,
            session_id=session_id,
            head_checkpoint_id=from_checkpoint,
            base_checkpoint_id=from_checkpoint,
            description=description,
        )

        self._branches[session_id][name] = branch

        logger.info(f"Created branch '{name}' from checkpoint {from_checkpoint[:12]}...")

        return branch

    async def checkout(
        self,
        branch_name: str,
        session_id: str,
    ) -> BranchMetadata:
        """Switch to a branch.

        Args:
            branch_name: Name of branch to switch to
            session_id: Session identifier

        Returns:
            BranchMetadata of checked out branch

        Raises:
            CheckpointError: If branch doesn't exist
        """
        if session_id not in self._branches:
            raise CheckpointError(f"No branches found for session {session_id}")

        if branch_name not in self._branches[session_id]:
            raise CheckpointError(f"Branch '{branch_name}' not found in session {session_id}")

        self._current_branch[session_id] = branch_name
        branch = self._branches[session_id][branch_name]

        logger.info(
            f"Checked out branch '{branch_name}' (head: {branch.head_checkpoint_id[:12]}...)"
        )

        return branch

    async def get_current_branch(self, session_id: str) -> Optional[BranchMetadata]:
        """Get the current branch for a session.

        Args:
            session_id: Session identifier

        Returns:
            Current branch or None if no branch is checked out
        """
        branch_name = self._current_branch.get(session_id)
        if branch_name and session_id in self._branches:
            return self._branches[session_id].get(branch_name)
        return None

    async def update_branch_head(
        self,
        branch_name: str,
        session_id: str,
        new_head_checkpoint_id: str,
    ) -> BranchMetadata:
        """Update a branch's head checkpoint.

        Called after creating a new checkpoint on a branch.

        Args:
            branch_name: Branch to update
            session_id: Session identifier
            new_head_checkpoint_id: New head checkpoint ID

        Returns:
            Updated BranchMetadata
        """
        if session_id not in self._branches:
            raise CheckpointError(f"No branches found for session {session_id}")

        if branch_name not in self._branches[session_id]:
            raise CheckpointError(f"Branch '{branch_name}' not found")

        branch = self._branches[session_id][branch_name]
        branch.head_checkpoint_id = new_head_checkpoint_id
        branch.updated_at = datetime.now(timezone.utc)

        return branch

    async def merge(
        self,
        source_branch: str,
        target_branch: str,
        session_id: str,
        strategy: MergeStrategy = MergeStrategy.THREE_WAY,
        state_merger: Optional[Callable[[Dict, Dict, Dict], Dict]] = None,
    ) -> MergeResult:
        """Merge one branch into another.

        Args:
            source_branch: Branch to merge from
            target_branch: Branch to merge into
            session_id: Session identifier
            strategy: Merge strategy to use
            state_merger: Optional custom function to merge states

        Returns:
            MergeResult with outcome details
        """
        if session_id not in self._branches:
            raise CheckpointError(f"No branches found for session {session_id}")

        source = self._branches[session_id].get(source_branch)
        target = self._branches[session_id].get(target_branch)

        if not source:
            raise CheckpointError(f"Source branch '{source_branch}' not found")
        if not target:
            raise CheckpointError(f"Target branch '{target_branch}' not found")

        # Build tree to find common ancestor
        tree = await CheckpointTree.build_from_session(
            self.backend, session_id, self._branches[session_id]
        )

        common_ancestor = tree.find_common_ancestor(
            source.head_checkpoint_id,
            target.head_checkpoint_id,
        )

        # Fast-forward if linear history
        if strategy == MergeStrategy.FAST_FORWARD:
            if common_ancestor and common_ancestor.checkpoint_id == target.head_checkpoint_id:
                # Target is ancestor of source - fast-forward
                target.head_checkpoint_id = source.head_checkpoint_id
                target.updated_at = datetime.now(timezone.utc)

                return MergeResult(
                    success=True,
                    merge_checkpoint_id=source.head_checkpoint_id,
                    strategy_used=MergeStrategy.FAST_FORWARD,
                    changes_merged={
                        "checkpoints": len(
                            tree.get_path(target.head_checkpoint_id, source.head_checkpoint_id)
                        )
                    },
                )
            else:
                return MergeResult(
                    success=False,
                    conflicts=[
                        {
                            "type": "not_linear",
                            "message": "Cannot fast-forward, branches have diverged",
                        }
                    ],
                )

        # Three-way merge
        if strategy == MergeStrategy.THREE_WAY:
            if not common_ancestor:
                return MergeResult(
                    success=False,
                    conflicts=[
                        {
                            "type": "no_common_ancestor",
                            "message": "Branches have no common ancestor",
                        }
                    ],
                )

            # Load all three states
            base_data = await self.backend.load_checkpoint(common_ancestor.checkpoint_id)
            source_data = await self.backend.load_checkpoint(source.head_checkpoint_id)
            target_data = await self.backend.load_checkpoint(target.head_checkpoint_id)

            # Merge states
            if state_merger:
                merged_state = state_merger(
                    base_data.state_data,
                    source_data.state_data,
                    target_data.state_data,
                )
            else:
                merged_state = self._default_state_merge(
                    base_data.state_data,
                    source_data.state_data,
                    target_data.state_data,
                )

            # Create merge checkpoint
            metadata = CheckpointMetadata.create(
                session_id=session_id,
                stage=merged_state.get("stage", "INITIAL"),
                tool_count=len(merged_state.get("tool_history", [])),
                message_count=merged_state.get("message_count", 0),
                parent_id=target.head_checkpoint_id,  # Point to target head
                description=f"Merge '{source_branch}' into '{target_branch}'",
                tags=["merge", f"from:{source_branch}", f"into:{target_branch}"],
            )

            merge_checkpoint_id = await self.backend.save_checkpoint(
                session_id=session_id,
                state_data=merged_state,
                metadata=metadata,
            )

            # Update target branch head
            target.head_checkpoint_id = merge_checkpoint_id
            target.updated_at = datetime.now(timezone.utc)

            # Mark source as merged
            source.status = BranchStatus.MERGED
            source.merge_parent_id = merge_checkpoint_id

            logger.info(
                f"Merged '{source_branch}' into '{target_branch}' -> {merge_checkpoint_id[:12]}..."
            )

            return MergeResult(
                success=True,
                merge_checkpoint_id=merge_checkpoint_id,
                strategy_used=MergeStrategy.THREE_WAY,
                changes_merged={
                    "messages": merged_state.get("message_count", 0),
                    "tools": len(merged_state.get("tool_history", [])),
                },
            )

        # Squash merge
        if strategy == MergeStrategy.SQUASH:
            source_data = await self.backend.load_checkpoint(source.head_checkpoint_id)

            # Create squash checkpoint on target
            metadata = CheckpointMetadata.create(
                session_id=session_id,
                stage=source_data.state_data.get("stage", "INITIAL"),
                tool_count=len(source_data.state_data.get("tool_history", [])),
                message_count=source_data.state_data.get("message_count", 0),
                parent_id=target.head_checkpoint_id,
                description=f"Squash merge '{source_branch}' into '{target_branch}'",
                tags=["squash", f"from:{source_branch}"],
            )

            merge_checkpoint_id = await self.backend.save_checkpoint(
                session_id=session_id,
                state_data=source_data.state_data,
                metadata=metadata,
            )

            target.head_checkpoint_id = merge_checkpoint_id
            source.status = BranchStatus.MERGED
            source.merge_parent_id = merge_checkpoint_id

            return MergeResult(
                success=True,
                merge_checkpoint_id=merge_checkpoint_id,
                strategy_used=MergeStrategy.SQUASH,
            )

        return MergeResult(
            success=False,
            conflicts=[
                {
                    "type": "unsupported",
                    "message": f"Strategy {strategy} not implemented",
                }
            ],
        )

    def _default_state_merge(
        self,
        base: Dict[str, Any],
        source: Dict[str, Any],
        target: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Default three-way state merge.

        Simple merge strategy:
        - For lists: union of items
        - For dicts: recursive merge
        - For scalars: prefer source if changed from base

        Args:
            base: Common ancestor state
            source: Source branch state
            target: Target branch state

        Returns:
            Merged state dictionary
        """
        merged = target.copy()

        for key, source_value in source.items():
            base_value = base.get(key)
            target_value = target.get(key)

            if source_value == base_value:
                # No change in source, keep target
                continue
            elif target_value == base_value:
                # Changed only in source, use source
                merged[key] = source_value
            elif isinstance(source_value, list) and isinstance(target_value, list):
                # Union of lists
                merged[key] = list(set(source_value) | set(target_value))
            elif isinstance(source_value, dict) and isinstance(target_value, dict):
                # Recursive merge
                merged[key] = self._default_state_merge(
                    base.get(key, {}),
                    source_value,
                    target_value,
                )
            else:
                # Conflict - prefer source (can be customized)
                merged[key] = source_value

        return merged

    async def delete_branch(
        self,
        branch_name: str,
        session_id: str,
        force: bool = False,
    ) -> bool:
        """Delete a branch.

        Args:
            branch_name: Branch to delete
            session_id: Session identifier
            force: Force delete even if not merged

        Returns:
            True if deleted, False if not found

        Raises:
            CheckpointError: If branch is not merged and force=False
        """
        if session_id not in self._branches:
            return False

        if branch_name not in self._branches[session_id]:
            return False

        branch = self._branches[session_id][branch_name]

        if branch.status != BranchStatus.MERGED and not force:
            raise CheckpointError(
                f"Branch '{branch_name}' has not been merged. Use force=True to delete anyway."
            )

        del self._branches[session_id][branch_name]

        # Clear current if deleted
        if self._current_branch.get(session_id) == branch_name:
            del self._current_branch[session_id]

        logger.info(f"Deleted branch '{branch_name}'")

        return True

    async def list_branches(
        self,
        session_id: str,
        include_archived: bool = False,
    ) -> List[BranchMetadata]:
        """List all branches for a session.

        Args:
            session_id: Session identifier
            include_archived: Include archived/merged branches

        Returns:
            List of branch metadata
        """
        if session_id not in self._branches:
            return []

        branches = list(self._branches[session_id].values())

        if not include_archived:
            branches = [b for b in branches if b.status == BranchStatus.ACTIVE]

        return sorted(branches, key=lambda b: b.created_at, reverse=True)

    async def get_tree(self, session_id: str) -> CheckpointTree:
        """Get the checkpoint tree for a session.

        Args:
            session_id: Session identifier

        Returns:
            Populated CheckpointTree
        """
        branches = self._branches.get(session_id, {})
        return await CheckpointTree.build_from_session(self.backend, session_id, branches)

    async def replay_from(
        self,
        checkpoint_id: str,
        session_id: str,
        action_replay: Optional[Callable[[Dict, Dict], Dict]] = None,
    ) -> List[ReplayStep]:
        """Replay conversation from a checkpoint.

        Useful for debugging or reproducing behavior from a specific point.

        Args:
            checkpoint_id: Starting checkpoint
            session_id: Session identifier
            action_replay: Optional function to replay actions

        Returns:
            List of replay steps with state changes
        """
        # Get tree and find path from checkpoint to head
        tree = await self.get_tree(session_id)
        current_branch = await self.get_current_branch(session_id)

        if not current_branch:
            raise CheckpointError(f"No current branch for session {session_id}")

        # Get path from checkpoint to head
        path = tree.get_path(checkpoint_id, current_branch.head_checkpoint_id)

        if not path:
            raise CheckpointError(
                f"No path from {checkpoint_id} to head {current_branch.head_checkpoint_id}"
            )

        steps = []
        prev_state = None

        for i, node in enumerate(path):
            # Load checkpoint data
            data = await self.backend.load_checkpoint(node.checkpoint_id)

            # Extract action from state diff
            action = {}
            if prev_state:
                # Compute what changed
                tools_before = set(prev_state.get("tool_history", []))
                tools_after = set(data.state_data.get("tool_history", []))
                new_tools = list(tools_after - tools_before)
                if new_tools:
                    action["tools_called"] = new_tools

                msg_before = prev_state.get("message_count", 0)
                msg_after = data.state_data.get("message_count", 0)
                if msg_after > msg_before:
                    action["messages_added"] = msg_after - msg_before

            step = ReplayStep(
                checkpoint_id=node.checkpoint_id,
                action=action,
                state_before=prev_state,
                state_after=data.state_data,
                step_index=i,
            )
            steps.append(step)

            # Apply action replay if provided
            if action_replay and action:
                action_replay(action, data.state_data)

            prev_state = data.state_data

        return steps

    async def ensure_default_branch(self, session_id: str) -> BranchMetadata:
        """Ensure a default branch exists for the session.

        Creates the default branch if it doesn't exist.

        Args:
            session_id: Session identifier

        Returns:
            Default branch metadata
        """
        if session_id in self._branches and self.default_branch_name in self._branches[session_id]:
            return self._branches[session_id][self.default_branch_name]

        # Check for existing checkpoints
        checkpoints = await self.backend.list_checkpoints(session_id, limit=1)
        if not checkpoints:
            raise CheckpointError(
                f"Cannot create default branch: no checkpoints in session {session_id}"
            )

        return await self.create_branch(
            name=self.default_branch_name,
            session_id=session_id,
            from_checkpoint=checkpoints[0].checkpoint_id,
            description="Default branch",
        )


# =============================================================================
# Protocol for Branch Storage
# =============================================================================


@runtime_checkable
class BranchStorageProtocol(Protocol):
    """Protocol for branch metadata persistence.

    Production implementations should persist branches to storage
    (SQLite, PostgreSQL, etc.) rather than in-memory.
    """

    async def save_branch(self, branch: BranchMetadata) -> str:
        """Save a branch and return its ID."""
        ...

    async def load_branch(self, branch_id: str) -> BranchMetadata:
        """Load a branch by ID."""
        ...

    async def load_branch_by_name(self, session_id: str, name: str) -> Optional[BranchMetadata]:
        """Load a branch by session and name."""
        ...

    async def list_branches(
        self, session_id: str, include_archived: bool = False
    ) -> List[BranchMetadata]:
        """List branches for a session."""
        ...

    async def delete_branch(self, branch_id: str) -> bool:
        """Delete a branch."""
        ...


__all__ = [
    # Enums
    "BranchStatus",
    "MergeStrategy",
    "ConflictResolution",
    # Data classes
    "BranchMetadata",
    "CheckpointNode",
    "MergeResult",
    "ReplayStep",
    # Core classes
    "CheckpointTree",
    "BranchManager",
    # Protocols
    "BranchStorageProtocol",
]
