# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
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

"""Content-hash staleness verification (victor-codegraph manifest contract).

Pins the two-layer incremental plan: mtime fast path first, then content-hash
verification of mtime-changed files — touch/branch-flip churn without content
change must not trigger a reparse.
"""

import os
import tempfile
from pathlib import Path

import pytest

from victor.core.graph_rag import GraphIndexConfig, GraphIndexingPipeline
from victor.storage.graph.memory_store import MemoryGraphStore
from victor.storage.graph.sqlite_store import SqliteGraphStore


@pytest.fixture
def repo():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "a.py").write_text("def a():\n    return 1\n")
        (root / "b.py").write_text("def b():\n    return 2\n")
        yield root


def _pipeline(store, root) -> GraphIndexingPipeline:
    config = GraphIndexConfig(
        root_path=root,
        enable_ccg=False,
        enable_embeddings=False,
        enable_subgraph_cache=False,
        incremental=True,
    )
    return GraphIndexingPipeline(store, config)


async def _seed_indexed_state(store, pipeline, root, files):
    """Record each file as indexed at its current mtime + content hash."""
    for f in files:
        key = pipeline._graph_file_key(f, root)
        digest = GraphIndexingPipeline._compute_file_hash(f)
        await store.update_file_mtime(key, f.stat().st_mtime, content_hash=digest)


@pytest.mark.asyncio
async def test_touched_but_unchanged_file_is_not_reindexed(repo):
    store = SqliteGraphStore(project_path=repo)
    pipeline = _pipeline(store, repo)
    files = [repo / "a.py", repo / "b.py"]
    await _seed_indexed_state(store, pipeline, repo, files)

    # Bump a.py's mtime WITHOUT changing content (touch / branch flip).
    a = repo / "a.py"
    os.utime(a, (a.stat().st_atime + 100, a.stat().st_mtime + 100))

    stats = await pipeline._prepare_incremental_work(files, repo)

    assert pipeline._files_to_process == set()
    assert stats.files_unchanged == 2

    # The stored mtime was refreshed, so the NEXT run's fast path skips it
    # without even reading the file.
    stale = await store.get_stale_files(
        {pipeline._graph_file_key(f, repo): f.stat().st_mtime for f in files}
    )
    assert stale == []


@pytest.mark.asyncio
async def test_changed_content_is_reindexed_and_hash_buffered(repo):
    store = SqliteGraphStore(project_path=repo)
    pipeline = _pipeline(store, repo)
    files = [repo / "a.py", repo / "b.py"]
    await _seed_indexed_state(store, pipeline, repo, files)

    a = repo / "a.py"
    a.write_text("def a():\n    return 42\n")
    os.utime(a, (a.stat().st_atime + 100, a.stat().st_mtime + 100))

    stats = await pipeline._prepare_incremental_work(files, repo)

    key = pipeline._graph_file_key(a, repo)
    assert pipeline._files_to_process == {key}
    assert stats.files_unchanged == 1
    # The freshly computed hash is buffered for persistence after processing.
    assert pipeline._pending_file_hashes[key] == GraphIndexingPipeline._compute_file_hash(a)


@pytest.mark.asyncio
async def test_no_stored_hash_falls_back_to_mtime_staleness(repo):
    """Files indexed before the hash column existed reindex on mtime change."""
    store = SqliteGraphStore(project_path=repo)
    pipeline = _pipeline(store, repo)
    a = repo / "a.py"
    key = pipeline._graph_file_key(a, repo)
    # Legacy state: mtime recorded, no hash.
    await store.update_file_mtime(key, a.stat().st_mtime)

    os.utime(a, (a.stat().st_atime + 100, a.stat().st_mtime + 100))
    await pipeline._prepare_incremental_work([a], repo)

    assert key in pipeline._files_to_process


@pytest.mark.asyncio
async def test_sqlite_hash_roundtrip_and_none_clears(repo):
    store = SqliteGraphStore(project_path=repo)
    await store.update_file_mtime("a.py", 1.0, content_hash="deadbeef")
    assert await store.get_file_hashes(["a.py"]) == {"a.py": "deadbeef"}

    # None must CLEAR the hash — an unknown hash can't pass verification later.
    await store.update_file_mtime("a.py", 2.0)
    assert await store.get_file_hashes(["a.py"]) == {}


@pytest.mark.asyncio
async def test_memory_store_hash_parity():
    store = MemoryGraphStore()
    await store.update_file_mtime("a.py", 1.0, content_hash="deadbeef")
    assert await store.get_file_hashes(["a.py"]) == {"a.py": "deadbeef"}
    await store.update_file_mtime("a.py", 2.0)
    assert await store.get_file_hashes(["a.py"]) == {}


def test_compute_file_hash_matches_codegraph_contract(repo):
    pytest.importorskip("victor_codegraph")
    from victor_codegraph import parse_path

    a = repo / "a.py"
    parsed = parse_path(a)
    assert parsed is not None
    assert GraphIndexingPipeline._compute_file_hash(a) == parsed.content_hash


def test_compute_file_hash_missing_file(repo):
    assert GraphIndexingPipeline._compute_file_hash(repo / "missing.py") is None


@pytest.mark.asyncio
async def test_end_to_end_touch_does_not_reparse(repo):
    """Full index -> touch -> reindex: zero files processed."""
    store = SqliteGraphStore(project_path=repo)
    pipeline = _pipeline(store, repo)
    stats1 = await pipeline.index_repository()
    assert stats1.files_processed == 2

    a = repo / "a.py"
    os.utime(a, (a.stat().st_atime + 100, a.stat().st_mtime + 100))

    pipeline2 = _pipeline(store, repo)
    stats2 = await pipeline2.index_repository()
    assert stats2.files_processed == 0
    assert stats2.files_unchanged == 2

    # And a real content change still reindexes exactly that file.
    a.write_text("def a():\n    return 99\n")
    os.utime(a, (a.stat().st_atime + 200, a.stat().st_mtime + 200))
    pipeline3 = _pipeline(store, repo)
    stats3 = await pipeline3.index_repository()
    assert stats3.files_processed == 1
