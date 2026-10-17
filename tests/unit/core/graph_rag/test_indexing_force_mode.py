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

Force mode must prepare usable inputs before clearing existing graph data.
Write/publication failures after preflight are a separate acceptance boundary.
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

    async def test_force_mode_calls_delete_by_repo(self, tmp_path: Path) -> None:
        """Verify delete_by_repo() is called when incremental=False."""
        store = _RecordingGraphStore()
        config = GraphIndexConfig(
            root_path=tmp_path,
            enable_ccg=False,  # Disable CCG for simpler test
            enable_embeddings=False,
            incremental=False,  # Force mode
            chunk_size=10,
        )

        pipeline = GraphIndexingPipeline(store, config)

        # An explicitly empty, valid root is qualified before cleanup.
        stats = await pipeline.index_repository()
        assert stats.files_processed == 0
        assert pipeline._snapshot_qualification.status == "complete"
        assert pipeline._snapshot_qualification.selected_count == 0

        # Verify delete_by_repo was called
        delete_calls = [c for c in store.calls if c[0] == "delete_by_repo"]
        assert len(delete_calls) == 1, f"Expected 1 delete_by_repo call, got {len(delete_calls)}"
        assert delete_calls[0][2]["clear_embeddings"] is True

    async def test_force_mode_clears_all_existing_data(self, tmp_path: Path) -> None:
        """Verify all old nodes/edges are removed before rebuild."""
        store = _RecordingGraphStore()

        # Pre-populate with existing data
        existing_node = MagicMock(node_id="old:node", name="OldNode")
        store.nodes["old:node"] = existing_node
        store.edges[("old:src", "old:dst", "OLD_TYPE")] = MagicMock()

        config = GraphIndexConfig(
            root_path=tmp_path,
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

    async def test_incremental_mode_does_not_call_delete_by_repo(self, tmp_path: Path) -> None:
        """Verify delete_by_repo() is NOT called in incremental mode."""
        store = _RecordingGraphStore()
        config = GraphIndexConfig(
            root_path=tmp_path,
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


class TestPostRebuildMaintenance:
    """Co-design review item 26a: a force rebuild must run the project DB
    manager's maintain() (WAL checkpoint + VACUUM) automatically, instead of
    requiring a manual `victor db maintain`."""

    def _make_pipeline(self, incremental: bool, root: Path) -> GraphIndexingPipeline:
        return GraphIndexingPipeline(
            _RecordingGraphStore(),
            GraphIndexConfig(
                root_path=root,
                enable_ccg=False,
                enable_embeddings=False,
                incremental=incremental,
                chunk_size=10,
            ),
        )

    async def test_force_mode_runs_post_rebuild_maintenance(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """incremental=False + graph changes -> maintain() called exactly once."""
        (tmp_path / "mod.py").write_text("x = 1\n", encoding="utf-8")
        maintain = MagicMock(return_value={"path": "/fake/project.db"})
        fake_db = MagicMock()
        fake_db.maintain = maintain
        monkeypatch.setattr("victor.core.database.get_project_database", lambda _root: fake_db)

        pipeline = self._make_pipeline(incremental=False, root=tmp_path)
        stats = await pipeline.index_repository()

        assert stats.files_processed > 0, "precondition: the rebuild did something"
        maintain.assert_called_once()

    async def test_incremental_mode_skips_post_rebuild_maintenance(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Incremental refreshes must not pay the VACUUM cost."""
        (tmp_path / "mod.py").write_text("x = 1\n", encoding="utf-8")
        maintain = MagicMock()
        fake_db = MagicMock()
        fake_db.maintain = maintain
        monkeypatch.setattr("victor.core.database.get_project_database", lambda _root: fake_db)

        pipeline = self._make_pipeline(incremental=True, root=tmp_path)
        await pipeline.index_repository()

        maintain.assert_not_called()

    async def test_maintenance_failure_never_fails_the_rebuild(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """maintain() blowing up must be swallowed — an otherwise-successful
        rebuild stays successful."""
        (tmp_path / "mod.py").write_text("x = 1\n", encoding="utf-8")
        maintain = MagicMock(side_effect=RuntimeError("disk gone"))
        fake_db = MagicMock()
        fake_db.maintain = maintain
        monkeypatch.setattr("victor.core.database.get_project_database", lambda _root: fake_db)

        pipeline = self._make_pipeline(incremental=False, root=tmp_path)
        stats = await pipeline.index_repository()  # must not raise

        maintain.assert_called_once()
        assert stats.files_processed > 0


@pytest.mark.parametrize(
    "failure",
    [
        "invalid_root",
        "root_is_file",
        "discovery",
        "stat",
        "traversal",
        "parser",
        "missing",
        "syntax",
    ],
)
async def test_force_preflight_failure_preserves_graph_and_sessions(tmp_path, monkeypatch, failure):
    from victor.core.database import reset_project_database
    from victor.core.schema import Schema
    from victor.core.graph_rag import codegraph_repository
    from victor.storage.graph.protocol import GraphNode
    from victor.storage.graph.sqlite_store import SqliteGraphStore
    import sqlite3

    source = tmp_path / "selected.py"
    source.write_text("def fresh(): return 1\n")
    root = tmp_path
    if failure == "invalid_root":
        root = tmp_path / "absent"
    elif failure == "root_is_file":
        root = source
    elif failure == "syntax":
        source.write_text("def broken(\n")
    store = SqliteGraphStore(tmp_path)
    try:
        await store.upsert_nodes([GraphNode(node_id="old", type="file", name="old", file="old.py")])
        await store.update_file_mtime("old.py", 123.0)
        with sqlite3.connect(store.db_path) as conn:
            conn.execute(Schema.CONV_SESSION)
            conn.execute(
                "INSERT INTO sessions (session_id, created_at, last_activity) VALUES ('chat','now','now')"
            )
        pipeline = GraphIndexingPipeline(
            store,
            GraphIndexConfig(
                root_path=root,
                incremental=False,
                enable_ccg=False,
                enable_embeddings=False,
                enable_module_metrics=False,
                enable_subgraph_cache=False,
            ),
        )
        cleanup = AsyncMock()
        monkeypatch.setattr(store, "_clear_all_embeddings", cleanup)
        monkeypatch.setattr(store, "_delete_embeddings_for_file", AsyncMock())
        if failure == "discovery":
            monkeypatch.setattr(
                pipeline, "_discover_files", AsyncMock(side_effect=OSError("discovery failed"))
            )
        elif failure == "stat":
            original_stat = Path.stat
            selected_stats = 0

            def fail_selected_stat(path, *args, **kwargs):
                nonlocal selected_stats
                if path == source:
                    selected_stats += 1
                if path == source and selected_stats >= 2:
                    raise PermissionError("selected source stat denied")
                return original_stat(path, *args, **kwargs)

            monkeypatch.setattr(Path, "stat", fail_selected_stat)
        elif failure == "traversal":
            import os

            nested = tmp_path / "nested"
            nested.mkdir()
            (nested / "unseen.py").write_text("x = 1\n")
            original_scandir = os.scandir

            def fail_selected_directory(path):
                if Path(path) == nested:
                    raise PermissionError(13, "directory unreadable", str(nested))
                return original_scandir(path)

            monkeypatch.setattr(os, "scandir", fail_selected_directory)
        elif failure == "parser":
            monkeypatch.setattr(
                codegraph_repository,
                "prepare_repository_snapshot",
                MagicMock(side_effect=RuntimeError("parser failed")),
            )
        elif failure == "missing":
            original = codegraph_repository.prepare_repository_snapshot

            def missing(root, files):
                source.unlink()
                return original(root, files)

            monkeypatch.setattr(codegraph_repository, "prepare_repository_snapshot", missing)
        with pytest.raises((ValueError, OSError, RuntimeError)):
            await pipeline.index_repository()
        cleanup.assert_not_called()
        with sqlite3.connect(store.db_path) as conn:
            assert conn.execute("SELECT node_id FROM graph_node").fetchall() == [("old",)]
            assert conn.execute("SELECT file, mtime FROM graph_file_mtime").fetchall() == [
                ("old.py", 123.0)
            ]
            assert conn.execute("SELECT session_id FROM sessions").fetchall() == [("chat",)]
    finally:
        await store.close()
        reset_project_database(tmp_path)


async def test_cancelled_preparation_settles_before_next_root_owner(tmp_path, monkeypatch):
    from threading import Event
    from victor.core.graph_rag import codegraph_repository

    first_root, second_root = tmp_path / "first", tmp_path / "second"
    for root in (first_root, second_root):
        root.mkdir()
        (root / "selected.py").write_text("def fresh(): return 1\n")
    store = _RecordingGraphStore()
    store.nodes["old"] = MagicMock()
    pipeline = GraphIndexingPipeline(
        store,
        GraphIndexConfig(
            root_path=first_root,
            incremental=False,
            enable_ccg=False,
            enable_embeddings=False,
            enable_module_metrics=False,
            enable_subgraph_cache=False,
        ),
    )
    entered = asyncio.Event()
    release = Event()
    roots = []
    original = codegraph_repository.prepare_repository_snapshot
    loop = asyncio.get_running_loop()

    def prepare(root, files):
        roots.append(root)
        if root == first_root:
            loop.call_soon_threadsafe(entered.set)
            assert release.wait(5)
        return original(root, files)

    monkeypatch.setattr(codegraph_repository, "prepare_repository_snapshot", prepare)
    monkeypatch.setattr(pipeline, "_get_analysis_provider", lambda: None)
    monkeypatch.setattr(pipeline, "_maintain_project_db_after_rebuild", AsyncMock())
    first = asyncio.create_task(pipeline.index_repository())
    second = None
    try:
        await asyncio.wait_for(entered.wait(), 2)
        first.cancel()
        second = asyncio.create_task(pipeline.index_repository(root_path=second_root))
        await asyncio.sleep(0.02)
        first.cancel()
        await asyncio.sleep(0.02)
        assert not first.done() and not second.done()
        assert roots == [first_root]
        assert pipeline._active_root == first_root
        assert "old" in store.nodes
        assert not any(name == "delete_by_repo" for name, _, _ in store.calls)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(asyncio.shield(first), 2)
        result = await asyncio.wait_for(second, 2)
        assert result.files_processed == 1 and result.error_count == 0
        assert roots == [first_root, second_root]
        assert pipeline._active_root is None
        assert pipeline.config.root_path == first_root
        assert set(pipeline._codegraph_repository.files) == {"selected.py"}
        assert sum(name == "delete_by_repo" for name, _, _ in store.calls) == 1
    finally:
        release.set()
        tasks = [first] + ([second] if second else [])
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if pipeline._executor is not None:
            pipeline._executor.shutdown(wait=True)


async def test_override_root_qualifies_exact_file_inventory_and_storage_keys(tmp_path, monkeypatch):
    from victor.core.database import reset_project_database
    from victor.storage.graph.sqlite_store import SqliteGraphStore
    import sqlite3

    configured = tmp_path / "configured"
    selected = tmp_path / "selected"
    configured.mkdir()
    selected.mkdir()
    paths = ["a.py", ".hidden.py", "build/b.py", "pkg/direct.py", "pkg/nested/deep.py"]
    for name in paths:
        path = selected / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("def f(): return 1\n")
    store = SqliteGraphStore(configured)
    pipeline = GraphIndexingPipeline(
        store,
        GraphIndexConfig(
            root_path=configured,
            incremental=False,
            enable_ccg=False,
            enable_embeddings=False,
            enable_module_metrics=False,
            enable_subgraph_cache=False,
            exclude_patterns=["/pkg/*"],
        ),
    )
    monkeypatch.setattr(pipeline, "_get_analysis_provider", lambda: None)
    monkeypatch.setattr(pipeline, "_maintain_project_db_after_rebuild", AsyncMock())
    monkeypatch.setattr(store, "_clear_all_embeddings", AsyncMock())
    expected = set(paths) - {"pkg/direct.py"}
    try:
        stats = await pipeline.index_repository(root_path=selected)
        assert stats.error_count == 0 and stats.files_processed == len(expected)
        assert pipeline._snapshot_qualification.status == "complete"
        assert set(pipeline._codegraph_repository.files) == expected
        assert {n.file for n in await store.get_all_nodes()} == expected
        with sqlite3.connect(store.db_path) as conn:
            assert {r[0] for r in conn.execute("SELECT file FROM graph_file_mtime")} == expected
        assert pipeline.config.root_path == configured
        assert pipeline._canonical_file_str(configured / "default.py") == "default.py"
    finally:
        await store.close()
        if pipeline._executor is not None:
            pipeline._executor.shutdown(wait=True)
        reset_project_database(configured)


async def test_force_discovery_does_not_enter_explicitly_excluded_subtree(tmp_path, monkeypatch):
    import os

    ignored = tmp_path / "ignored"
    ignored.mkdir()
    (ignored / "bad.py").write_text("bad")
    keep = tmp_path / "keep.py"
    keep.write_text("x = 1\n")
    scan = os.scandir

    def deny_ignored(path):
        if Path(path) == ignored:
            raise PermissionError(13, "excluded directory must not be read", str(ignored))
        return scan(path)

    monkeypatch.setattr(os, "scandir", deny_ignored)
    pipeline = GraphIndexingPipeline(
        _RecordingGraphStore(),
        GraphIndexConfig(root_path=tmp_path, incremental=False, exclude_patterns=["/ignored/**"]),
    )
    assert await pipeline._discover_files(tmp_path) == [keep]
