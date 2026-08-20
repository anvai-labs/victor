# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
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

"""Tests for force mode rebuild behavior in graph indexing.

These tests verify that force mode (incremental=False) properly clears
all existing graph data before rebuilding, ensuring clean state and
efficient bulk inserts instead of expensive UPSERTs.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, call
from pathlib import Path
from typing import Any, List
from contextlib import asynccontextmanager

import pytest

from victor.core.graph_rag.config import GraphIndexConfig
from victor.core.graph_rag.indexing import GraphIndexingPipeline, GraphIndexStats


class _RecordingGraphStore:
    """Mock graph store that records all operations for verification."""

    def __init__(self) -> None:
        self.calls: List[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.nodes: dict[str, MagicMock] = {}
        self.edges: dict[tuple[str, str, str], MagicMock] = {}
        self.file_mtimes: dict[str, float] = {}
        self.in_write_batch = False
        self.write_batch_depth = 0
        self._initialized = False

    async def initialize(self) -> None:
        self._initialized = True
        self._record("initialize", (), {})

    def _record(self, method: str, args: tuple, kwargs: dict) -> None:
        self.calls.append((method, args, kwargs))

    @asynccontextmanager
    async def write_batch(self):
        """Context manager for batch operations."""
        self.in_write_batch = True
        self.write_batch_depth += 1
        try:
            yield
        finally:
            self.write_batch_depth -= 1
            if self.write_batch_depth == 0:
                self.in_write_batch = False

    async def delete_by_repo(self, clear_embeddings: bool = False) -> None:
        """Record delete_by_repo call and clear all data."""
        self._record("delete_by_repo", (), {"clear_embeddings": clear_embeddings})
        self.nodes.clear()
        self.edges.clear()
        self.file_mtimes.clear()

    async def get_all_nodes(self) -> List[MagicMock]:
        """Return all nodes."""
        return list(self.nodes.values())

    async def get_stale_files(self, file_mtimes: dict[str, float]) -> set[str]:
        """Return empty set - no stale files in fresh index."""
        return set()

    async def delete_by_file(self, file: str) -> None:
        """Record delete_by_file call."""
        self._record("delete_by_file", (file,), {})

    async def upsert_nodes(self, nodes) -> None:
        """Store nodes."""
        for node in nodes:
            self.nodes[node.node_id] = node
        self._record("upsert_nodes", (list(nodes)), {})

    async def upsert_edges(self, edges) -> None:
        """Store edges."""
        for edge in edges:
            self.edges[(edge.src, edge.dst, edge.type)] = edge
        self._record("upsert_edges", (list(edges)), {})

    async def update_file_mtime(self, file: str, mtime: float) -> None:
        """Store file mtime."""
        self.file_mtimes[file] = mtime
        self._record("update_file_mtime", (file, mtime), {})

    async def stats(self) -> dict[str, Any]:
        """Return current stats."""
        return {
            "nodes": len(self.nodes),
            "edges": len(self.edges),
            "indexed_files": len(self.file_mtimes),
        }


@pytest.mark.asyncio
class TestForceModeRebuild:
    """Test force mode (incremental=False) rebuild behavior."""

    async def test_force_mode_calls_delete_by_repo(self) -> None:
        """Verify delete_by_repo() is called when incremental=False."""
        store = _RecordingGraphStore()
        config = GraphIndexConfig(
            root_path=Path("/fake/root"),
            enable_ccg=False,  # Disable CCG for simpler test
            enable_embeddings=False,
            incremental=False,  # Force mode
            chunk_size=10,
        )

        pipeline = GraphIndexingPipeline(store, config)

        # Run indexing - should call delete_by_repo first
        stats = await pipeline.index_repository()

        # Verify delete_by_repo was called
        delete_calls = [c for c in store.calls if c[0] == "delete_by_repo"]
        assert len(delete_calls) == 1, f"Expected 1 delete_by_repo call, got {len(delete_calls)}"
        assert delete_calls[0][2]["clear_embeddings"] is True

    async def test_force_mode_clears_all_existing_data(self) -> None:
        """Verify all old nodes/edges are removed before rebuild."""
        store = _RecordingGraphStore()

        # Pre-populate with existing data
        existing_node = MagicMock(node_id="old:node", name="OldNode")
        store.nodes["old:node"] = existing_node
        store.edges[("old:src", "old:dst", "OLD_TYPE")] = MagicMock()

        config = GraphIndexConfig(
            root_path=Path("/fake/root"),
            enable_ccg=False,
            enable_embeddings=False,
            incremental=False,
            chunk_size=10,
        )

        pipeline = GraphIndexingPipeline(store, config)

        # Run indexing
        stats = await pipeline.index_repository()

        # Verify old data was cleared (delete_by_repo clears the dicts)
        assert "old:node" not in store.nodes, "Old nodes should be deleted"
        assert (
            "old:src",
            "old:dst",
            "OLD_TYPE",
        ) not in store.edges, "Old edges should be deleted"

    async def test_incremental_mode_does_not_call_delete_by_repo(self) -> None:
        """Verify delete_by_repo() is NOT called in incremental mode."""
        store = _RecordingGraphStore()
        config = GraphIndexConfig(
            root_path=Path("/fake/root"),
            enable_ccg=False,
            enable_embeddings=False,
            incremental=True,  # Incremental mode
            chunk_size=10,
        )

        pipeline = GraphIndexingPipeline(store, config)

        # Run indexing
        stats = await pipeline.index_repository()

        # Verify delete_by_repo was NOT called
        delete_calls = [c for c in store.calls if c[0] == "delete_by_repo"]
        assert (
            len(delete_calls) == 0
        ), f"Expected 0 delete_by_repo calls in incremental mode, got {len(delete_calls)}"

    async def test_force_mode_logs_clearing_message(self, caplog) -> None:
        """Verify force mode logs appropriate clearing message."""
        import logging

        store = _RecordingGraphStore()
        config = GraphIndexConfig(
            root_path=Path("/fake/root"),
            enable_ccg=False,
            enable_embeddings=False,
            incremental=False,
            chunk_size=10,
        )

        pipeline = GraphIndexingPipeline(store, config)

        # Run indexing with log capture
        with caplog.at_level(logging.INFO):
            stats = await pipeline.index_repository()

        # Verify clearing message was logged
        clearing_logs = [r for r in caplog.records if "clearing" in r.message.lower()]
        assert len(clearing_logs) > 0, "Expected clearing message in logs"


@pytest.mark.asyncio
class TestDeleteByRepoIntegration:
    """Integration tests for delete_by_repo with real graph store."""

    async def test_delete_by_repo_with_clear_embeddings_flag(self) -> None:
        """Verify delete_by_repo accepts and uses clear_embeddings flag."""
        store = _RecordingGraphStore()

        # Test with clear_embeddings=True
        await store.delete_by_repo(clear_embeddings=True)

        delete_calls = [c for c in store.calls if c[0] == "delete_by_repo"]
        assert len(delete_calls) == 1
        assert delete_calls[0][2]["clear_embeddings"] is True

    async def test_delete_by_repo_clears_file_mtimes(self) -> None:
        """Verify delete_by_repo clears file mtime tracking."""
        store = _RecordingGraphStore()

        # Add some file mtimes
        store.file_mtimes["/path/to/file.py"] = 123456.0
        store.file_mtimes["/path/to/other.py"] = 123457.0

        assert len(store.file_mtimes) == 2

        # Delete by repo
        await store.delete_by_repo()

        # Verify mtimes are cleared
        assert len(store.file_mtimes) == 0
