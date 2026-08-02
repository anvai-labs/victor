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

"""Tests for checkpoint tree and branch management.

Tests cover:
- BranchMetadata creation and serialization
- CheckpointTree building and navigation
- BranchManager operations (create, checkout, merge, delete)
- Tree traversal (ancestors, descendants, common ancestor)
- Merge strategies (fast-forward, three-way, squash)
- Replay functionality
"""

import pytest
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

from victor.storage.checkpoints.protocol import (
    CheckpointData,
    CheckpointMetadata,
    CheckpointNotFoundError,
)
from victor.storage.checkpoints.tree import (
    BranchMetadata,
    BranchStatus,
    BranchManager,
    CheckpointNode,
    CheckpointTree,
    MergeResult,
    MergeStrategy,
    ReplayStep,
)

# =============================================================================
# Mock Backend
# =============================================================================


class MockCheckpointBackend:
    """Mock checkpoint backend for testing."""

    def __init__(self):
        self._checkpoints: Dict[str, CheckpointData] = {}
        self._session_checkpoints: Dict[str, List[str]] = {}

    async def save_checkpoint(
        self,
        session_id: str,
        state_data: Dict[str, Any],
        metadata: CheckpointMetadata,
    ) -> str:
        checkpoint_id = metadata.checkpoint_id

        data = CheckpointData(
            metadata=metadata,
            state_data=state_data,
        )
        self._checkpoints[checkpoint_id] = data

        if session_id not in self._session_checkpoints:
            self._session_checkpoints[session_id] = []
        self._session_checkpoints[session_id].append(checkpoint_id)

        return checkpoint_id

    async def load_checkpoint(self, checkpoint_id: str) -> CheckpointData:
        if checkpoint_id not in self._checkpoints:
            raise CheckpointNotFoundError(f"Checkpoint {checkpoint_id} not found")
        return self._checkpoints[checkpoint_id]

    async def list_checkpoints(
        self,
        session_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> List[CheckpointMetadata]:
        if session_id not in self._session_checkpoints:
            return []

        checkpoint_ids = self._session_checkpoints[session_id][offset : offset + limit]
        return [self._checkpoints[cid].metadata for cid in checkpoint_ids]

    async def delete_checkpoint(self, checkpoint_id: str) -> bool:
        if checkpoint_id in self._checkpoints:
            del self._checkpoints[checkpoint_id]
            return True
        return False

    async def get_checkpoint_metadata(self, checkpoint_id: str) -> CheckpointMetadata:
        if checkpoint_id not in self._checkpoints:
            raise CheckpointNotFoundError(f"Checkpoint {checkpoint_id} not found")
        return self._checkpoints[checkpoint_id].metadata

    async def cleanup_old_checkpoints(
        self,
        session_id: str,
        keep_count: int = 10,
    ) -> int:
        return 0


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_backend():
    """Create mock backend."""
    return MockCheckpointBackend()


@pytest.fixture
async def populated_backend(mock_backend):
    """Create backend with sample checkpoints."""
    session_id = "test_session"

    # Create linear checkpoint history
    # root -> cp1 -> cp2 -> cp3
    #               \-> cp4 (branch)
    checkpoints = [
        CheckpointMetadata.create(
            session_id=session_id,
            stage="INITIAL",
            tool_count=0,
            message_count=0,
            description="Root checkpoint",
        ),
    ]
    # Save root
    await mock_backend.save_checkpoint(
        session_id,
        {"stage": "INITIAL", "tool_history": [], "message_count": 0},
        checkpoints[0],
    )

    # cp1
    cp1 = CheckpointMetadata.create(
        session_id=session_id,
        stage="EXPLORING",
        tool_count=2,
        message_count=2,
        parent_id=checkpoints[0].checkpoint_id,
        description="After exploration",
    )
    await mock_backend.save_checkpoint(
        session_id,
        {"stage": "EXPLORING", "tool_history": ["read", "search"], "message_count": 2},
        cp1,
    )
    checkpoints.append(cp1)

    # cp2
    cp2 = CheckpointMetadata.create(
        session_id=session_id,
        stage="ANALYZING",
        tool_count=4,
        message_count=4,
        parent_id=cp1.checkpoint_id,
        description="After analysis",
    )
    await mock_backend.save_checkpoint(
        session_id,
        {
            "stage": "ANALYZING",
            "tool_history": ["read", "search", "analyze", "grep"],
            "message_count": 4,
        },
        cp2,
    )
    checkpoints.append(cp2)

    # cp3 (continuation on main)
    cp3 = CheckpointMetadata.create(
        session_id=session_id,
        stage="EXECUTING",
        tool_count=6,
        message_count=6,
        parent_id=cp2.checkpoint_id,
        description="Main branch execution",
    )
    await mock_backend.save_checkpoint(
        session_id,
        {
            "stage": "EXECUTING",
            "tool_history": ["read", "search", "analyze", "grep", "write", "edit"],
            "message_count": 6,
        },
        cp3,
    )
    checkpoints.append(cp3)

    # cp4 (branch from cp2)
    cp4 = CheckpointMetadata.create(
        session_id=session_id,
        stage="EXECUTING",
        tool_count=5,
        message_count=5,
        parent_id=cp2.checkpoint_id,
        description="Experiment branch",
    )
    await mock_backend.save_checkpoint(
        session_id,
        {
            "stage": "EXECUTING",
            "tool_history": ["read", "search", "analyze", "grep", "refactor"],
            "message_count": 5,
        },
        cp4,
    )
    checkpoints.append(cp4)

    return mock_backend, session_id, checkpoints


# =============================================================================
# Test BranchMetadata
# =============================================================================


class TestBranchMetadata:
    """Tests for BranchMetadata dataclass."""

    def test_create_branch(self):
        """Test creating a branch."""
        branch = BranchMetadata.create(
            name="experiment",
            session_id="session_123",
            head_checkpoint_id="ckpt_abc123",
            description="Testing new approach",
        )

        assert branch.name == "experiment"
        assert branch.session_id == "session_123"
        assert branch.head_checkpoint_id == "ckpt_abc123"
        assert branch.base_checkpoint_id == "ckpt_abc123"
        assert branch.status == BranchStatus.ACTIVE
        assert branch.description == "Testing new approach"
        assert branch.branch_id.startswith("branch_")

    def test_create_branch_with_base(self):
        """Test creating branch with different base."""
        branch = BranchMetadata.create(
            name="feature",
            session_id="session_123",
            head_checkpoint_id="ckpt_head",
            base_checkpoint_id="ckpt_base",
        )

        assert branch.head_checkpoint_id == "ckpt_head"
        assert branch.base_checkpoint_id == "ckpt_base"

    def test_branch_serialization(self):
        """Test branch to_dict and from_dict."""
        branch = BranchMetadata.create(
            name="test",
            session_id="session_123",
            head_checkpoint_id="ckpt_123",
            tags=["experiment", "v2"],
        )

        data = branch.to_dict()
        restored = BranchMetadata.from_dict(data)

        assert restored.name == branch.name
        assert restored.session_id == branch.session_id
        assert restored.head_checkpoint_id == branch.head_checkpoint_id
        assert restored.tags == branch.tags
        assert restored.status == branch.status


# =============================================================================
# Test CheckpointNode
# =============================================================================


class TestCheckpointNode:
    """Tests for CheckpointNode."""

    def test_node_is_root(self):
        """Test root node detection."""
        root = CheckpointNode(checkpoint_id="root", parent_id=None)
        child = CheckpointNode(checkpoint_id="child", parent_id="root")

        assert root.is_root()
        assert not child.is_root()

    def test_node_is_leaf(self):
        """Test leaf node detection."""
        leaf = CheckpointNode(checkpoint_id="leaf", parent_id="parent")
        branch = CheckpointNode(
            checkpoint_id="branch",
            parent_id="parent",
            children_ids=["child1", "child2"],
        )

        assert leaf.is_leaf()
        assert not branch.is_leaf()

    def test_node_is_branch_point(self):
        """Test branch point detection."""
        single_child = CheckpointNode(
            checkpoint_id="single",
            parent_id=None,
            children_ids=["child1"],
        )
        branch_point = CheckpointNode(
            checkpoint_id="branch",
            parent_id=None,
            children_ids=["child1", "child2"],
        )

        assert not single_child.is_branch_point()
        assert branch_point.is_branch_point()


# =============================================================================
# Test CheckpointTree
# =============================================================================


class TestCheckpointTree:
    """Tests for CheckpointTree."""

    @pytest.mark.asyncio
    async def test_build_tree_from_session(self, populated_backend):
        """Test building tree from session checkpoints."""
        backend, session_id, checkpoints = populated_backend

        tree = await CheckpointTree.build_from_session(backend, session_id)

        # Should have all checkpoints as nodes
        assert len(tree._nodes) == 5

        # Root should be first checkpoint
        root = tree.get_root()
        assert root is not None
        assert root.checkpoint_id == checkpoints[0].checkpoint_id

    @pytest.mark.asyncio
    async def test_get_ancestors(self, populated_backend):
        """Test getting ancestors of a checkpoint."""
        backend, session_id, checkpoints = populated_backend
        tree = await CheckpointTree.build_from_session(backend, session_id)

        # Get ancestors of cp3 (should be cp2, cp1, root)
        ancestors = tree.get_ancestors(checkpoints[3].checkpoint_id)

        assert len(ancestors) == 3
        assert ancestors[0].checkpoint_id == checkpoints[2].checkpoint_id  # cp2
        assert ancestors[1].checkpoint_id == checkpoints[1].checkpoint_id  # cp1
        assert ancestors[2].checkpoint_id == checkpoints[0].checkpoint_id  # root

    @pytest.mark.asyncio
    async def test_get_ancestors_with_max_depth(self, populated_backend):
        """Test getting ancestors with depth limit."""
        backend, session_id, checkpoints = populated_backend
        tree = await CheckpointTree.build_from_session(backend, session_id)

        ancestors = tree.get_ancestors(checkpoints[3].checkpoint_id, max_depth=2)

        assert len(ancestors) == 2

    @pytest.mark.asyncio
    async def test_get_descendants(self, populated_backend):
        """Test getting descendants of a checkpoint."""
        backend, session_id, checkpoints = populated_backend
        tree = await CheckpointTree.build_from_session(backend, session_id)

        # Get descendants of cp1 (should be cp2, and then cp3 and cp4)
        descendants = tree.get_descendants(checkpoints[1].checkpoint_id)

        assert len(descendants) == 3
        # cp2 should be first (direct child)
        assert descendants[0].checkpoint_id == checkpoints[2].checkpoint_id

    @pytest.mark.asyncio
    async def test_find_common_ancestor(self, populated_backend):
        """Test finding common ancestor."""
        backend, session_id, checkpoints = populated_backend
        tree = await CheckpointTree.build_from_session(backend, session_id)

        # Common ancestor of cp3 and cp4 should be cp2
        common = tree.find_common_ancestor(
            checkpoints[3].checkpoint_id,
            checkpoints[4].checkpoint_id,
        )

        assert common is not None
        assert common.checkpoint_id == checkpoints[2].checkpoint_id

    @pytest.mark.asyncio
    async def test_get_branch_points(self, populated_backend):
        """Test getting branch points."""
        backend, session_id, checkpoints = populated_backend
        tree = await CheckpointTree.build_from_session(backend, session_id)

        branch_points = tree.get_branch_points()

        # cp2 is the only branch point (has two children: cp3 and cp4)
        assert len(branch_points) == 1
        assert branch_points[0].checkpoint_id == checkpoints[2].checkpoint_id

    @pytest.mark.asyncio
    async def test_get_leaves(self, populated_backend):
        """Test getting leaf checkpoints."""
        backend, session_id, checkpoints = populated_backend
        tree = await CheckpointTree.build_from_session(backend, session_id)

        leaves = tree.get_leaves()

        # cp3 and cp4 are leaves
        assert len(leaves) == 2
        leaf_ids = {l.checkpoint_id for l in leaves}
        assert checkpoints[3].checkpoint_id in leaf_ids
        assert checkpoints[4].checkpoint_id in leaf_ids

    @pytest.mark.asyncio
    async def test_to_ascii(self, populated_backend):
        """Test ASCII tree visualization."""
        backend, session_id, checkpoints = populated_backend
        tree = await CheckpointTree.build_from_session(backend, session_id)

        ascii_tree = tree.to_ascii()

        assert "Checkpoint Tree" in ascii_tree
        assert "├──" in ascii_tree or "└──" in ascii_tree

    def test_empty_tree(self):
        """Test empty tree behavior."""
        tree = CheckpointTree()

        assert tree.get_root() is None
        assert tree.get_ancestors("nonexistent") == []
        assert tree.get_descendants("nonexistent") == []
        assert tree.to_ascii() == "Empty tree"


# =============================================================================
# Test BranchManager
# =============================================================================


class TestBranchManager:
    """Tests for BranchManager."""

    @pytest.fixture
    def branch_manager(self, mock_backend):
        """Create branch manager."""
        return BranchManager(mock_backend)

    @pytest.mark.asyncio
    async def test_create_branch(self, branch_manager, populated_backend):
        """Test creating a branch."""
        backend, session_id, checkpoints = populated_backend
        branch_manager.backend = backend

        branch = await branch_manager.create_branch(
            name="feature",
            session_id=session_id,
            from_checkpoint=checkpoints[2].checkpoint_id,
            description="New feature branch",
        )

        assert branch.name == "feature"
        assert branch.head_checkpoint_id == checkpoints[2].checkpoint_id
        assert branch.status == BranchStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_create_branch_duplicate_name(self, branch_manager, populated_backend):
        """Test creating branch with duplicate name fails."""
        backend, session_id, checkpoints = populated_backend
        branch_manager.backend = backend

        await branch_manager.create_branch("feature", session_id, checkpoints[0].checkpoint_id)

        from victor.storage.checkpoints.tree import CheckpointError

        with pytest.raises(CheckpointError, match="already exists"):
            await branch_manager.create_branch("feature", session_id, checkpoints[1].checkpoint_id)

    @pytest.mark.asyncio
    async def test_checkout_branch(self, branch_manager, populated_backend):
        """Test checking out a branch."""
        backend, session_id, checkpoints = populated_backend
        branch_manager.backend = backend

        await branch_manager.create_branch("feature", session_id, checkpoints[0].checkpoint_id)
        branch = await branch_manager.checkout("feature", session_id)

        assert branch.name == "feature"
        current = await branch_manager.get_current_branch(session_id)
        assert current.name == "feature"

    @pytest.mark.asyncio
    async def test_update_branch_head(self, branch_manager, populated_backend):
        """Test updating branch head."""
        backend, session_id, checkpoints = populated_backend
        branch_manager.backend = backend

        await branch_manager.create_branch("feature", session_id, checkpoints[0].checkpoint_id)

        updated = await branch_manager.update_branch_head(
            "feature",
            session_id,
            checkpoints[2].checkpoint_id,
        )

        assert updated.head_checkpoint_id == checkpoints[2].checkpoint_id

    @pytest.mark.asyncio
    async def test_list_branches(self, branch_manager, populated_backend):
        """Test listing branches."""
        backend, session_id, checkpoints = populated_backend
        branch_manager.backend = backend

        await branch_manager.create_branch("main", session_id, checkpoints[0].checkpoint_id)
        await branch_manager.create_branch("feature", session_id, checkpoints[1].checkpoint_id)
        await branch_manager.create_branch("experiment", session_id, checkpoints[2].checkpoint_id)

        branches = await branch_manager.list_branches(session_id)

        assert len(branches) == 3
        names = {b.name for b in branches}
        assert names == {"main", "feature", "experiment"}

    @pytest.mark.asyncio
    async def test_delete_branch(self, branch_manager, populated_backend):
        """Test deleting a branch."""
        backend, session_id, checkpoints = populated_backend
        branch_manager.backend = backend

        await branch_manager.create_branch("feature", session_id, checkpoints[0].checkpoint_id)

        # Can't delete unmerged branch without force
        from victor.storage.checkpoints.tree import CheckpointError

        with pytest.raises(CheckpointError, match="not been merged"):
            await branch_manager.delete_branch("feature", session_id)

        # Force delete works
        result = await branch_manager.delete_branch("feature", session_id, force=True)
        assert result is True

        branches = await branch_manager.list_branches(session_id)
        assert len(branches) == 0


# =============================================================================
# Test Merge Operations
# =============================================================================


class TestMergeOperations:
    """Tests for merge operations."""

    @pytest.fixture
    def branch_manager(self, mock_backend):
        """Create branch manager."""
        return BranchManager(mock_backend)

    @pytest.mark.asyncio
    async def test_fast_forward_merge(self, branch_manager, populated_backend):
        """Test fast-forward merge when target is ancestor."""
        backend, session_id, checkpoints = populated_backend
        branch_manager.backend = backend

        # Create main at cp1
        await branch_manager.create_branch("main", session_id, checkpoints[1].checkpoint_id)
        # Create feature at cp3 (ahead of main)
        await branch_manager.create_branch("feature", session_id, checkpoints[3].checkpoint_id)

        result = await branch_manager.merge(
            "feature",
            "main",
            session_id,
            strategy=MergeStrategy.FAST_FORWARD,
        )

        assert result.success
        assert result.strategy_used == MergeStrategy.FAST_FORWARD

        # Main should now point to cp3
        main = branch_manager._branches[session_id]["main"]
        assert main.head_checkpoint_id == checkpoints[3].checkpoint_id

    @pytest.mark.asyncio
    async def test_fast_forward_fails_on_diverged(self, branch_manager, populated_backend):
        """Test fast-forward fails when branches have diverged."""
        backend, session_id, checkpoints = populated_backend
        branch_manager.backend = backend

        # Create main at cp3
        await branch_manager.create_branch("main", session_id, checkpoints[3].checkpoint_id)
        # Create feature at cp4 (diverged from main)
        await branch_manager.create_branch("feature", session_id, checkpoints[4].checkpoint_id)

        result = await branch_manager.merge(
            "feature",
            "main",
            session_id,
            strategy=MergeStrategy.FAST_FORWARD,
        )

        assert not result.success
        assert result.has_conflicts()
        assert any("not_linear" in c.get("type", "") for c in result.conflicts)

    @pytest.mark.asyncio
    async def test_three_way_merge(self, branch_manager, populated_backend):
        """Test three-way merge."""
        backend, session_id, checkpoints = populated_backend
        branch_manager.backend = backend

        # Create main at cp3
        await branch_manager.create_branch("main", session_id, checkpoints[3].checkpoint_id)
        # Create feature at cp4 (diverged from main at cp2)
        await branch_manager.create_branch("feature", session_id, checkpoints[4].checkpoint_id)

        result = await branch_manager.merge(
            "feature",
            "main",
            session_id,
            strategy=MergeStrategy.THREE_WAY,
        )

        assert result.success
        assert result.strategy_used == MergeStrategy.THREE_WAY
        assert result.merge_checkpoint_id is not None

        # Feature should be marked as merged
        feature = branch_manager._branches[session_id]["feature"]
        assert feature.status == BranchStatus.MERGED

    @pytest.mark.asyncio
    async def test_squash_merge(self, branch_manager, populated_backend):
        """Test squash merge."""
        backend, session_id, checkpoints = populated_backend
        branch_manager.backend = backend

        await branch_manager.create_branch("main", session_id, checkpoints[1].checkpoint_id)
        await branch_manager.create_branch("feature", session_id, checkpoints[3].checkpoint_id)

        result = await branch_manager.merge(
            "feature",
            "main",
            session_id,
            strategy=MergeStrategy.SQUASH,
        )

        assert result.success
        assert result.strategy_used == MergeStrategy.SQUASH
        assert result.merge_checkpoint_id is not None


# =============================================================================
# Test Replay
# =============================================================================


class TestReplay:
    """Tests for replay functionality."""

    @pytest.fixture
    def branch_manager(self, mock_backend):
        """Create branch manager."""
        return BranchManager(mock_backend)

    @pytest.mark.asyncio
    async def test_replay_from_checkpoint(self, branch_manager, populated_backend):
        """Test replaying from a checkpoint."""
        backend, session_id, checkpoints = populated_backend
        branch_manager.backend = backend

        # Create and checkout main at cp3
        await branch_manager.create_branch("main", session_id, checkpoints[3].checkpoint_id)
        await branch_manager.checkout("main", session_id)

        # Replay from cp1 to current head
        steps = await branch_manager.replay_from(
            checkpoints[1].checkpoint_id,
            session_id,
        )

        assert len(steps) >= 2
        assert steps[0].checkpoint_id == checkpoints[1].checkpoint_id


# =============================================================================
# Test Tree Visualization
# =============================================================================


class TestTreeVisualization:
    """Tests for tree visualization."""

    @pytest.mark.asyncio
    async def test_get_tree(self, populated_backend):
        """Test getting tree from branch manager."""
        backend, session_id, checkpoints = populated_backend
        branch_manager = BranchManager(backend)

        await branch_manager.create_branch("main", session_id, checkpoints[3].checkpoint_id)
        await branch_manager.create_branch("feature", session_id, checkpoints[4].checkpoint_id)

        tree = await branch_manager.get_tree(session_id)

        assert len(tree._nodes) == 5
        heads = tree.get_branch_heads()
        assert "main" in heads
        assert "feature" in heads

    @pytest.mark.asyncio
    async def test_tree_to_dict(self, populated_backend):
        """Test tree serialization."""
        backend, session_id, checkpoints = populated_backend

        tree = await CheckpointTree.build_from_session(backend, session_id)
        data = tree.to_dict()

        assert "root_id" in data
        assert "nodes" in data
        assert len(data["nodes"]) == 5
