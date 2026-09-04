# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import sqlite3

from victor.storage.graph.protocol import GraphEdge, GraphNode
from victor.storage.graph.sqlite_store import SqliteGraphStore


@pytest.mark.asyncio
async def test_sqlite_graph_store_upsert_and_query(tmp_path):
    db_path = tmp_path / "graph.db"
    store = SqliteGraphStore(db_path)

    nodes = [
        GraphNode(node_id="file:main.py", type="file", name="main.py", file="main.py"),
        GraphNode(
            node_id="symbol:main.py:foo",
            type="function",
            name="foo",
            file="main.py",
            line=1,
            metadata={"signature": "foo()"},
        ),
    ]
    edges = [GraphEdge(src="file:main.py", dst="symbol:main.py:foo", type="CONTAINS")]

    await store.upsert_nodes(nodes)
    await store.upsert_edges(edges)

    stats = await store.stats()
    assert stats["nodes"] == 2
    assert stats["edges"] == 1

    found = await store.find_nodes(name="foo")
    assert len(found) == 1
    assert found[0].node_id == "symbol:main.py:foo"

    all_nodes = await store.get_all_nodes()
    assert len(all_nodes) == 2

    neighbors = await store.get_neighbors("file:main.py")
    assert len(neighbors) == 1
    assert neighbors[0].dst == "symbol:main.py:foo"


@pytest.mark.asyncio
async def test_sqlite_graph_store_persists_project_local_files_as_relative(tmp_path):
    store = SqliteGraphStore(tmp_path)
    source_file = tmp_path / "src" / "main.py"
    absolute_source = str(source_file)

    await store.upsert_nodes(
        [
            GraphNode(
                node_id="symbol:src/main.py:foo",
                type="function",
                name="foo",
                file=absolute_source,
                line=1,
            ),
            # The edge's target must exist: endpoint integrity is enforced at
            # the store layer (#903), so a dangling edge is dropped and this
            # test — whose subject is path relativization, not dangling-edge
            # acceptance — would silently have no edge row to assert on.
            GraphNode(
                node_id="symbol:src/main.py:bar",
                type="function",
                name="bar",
                file=absolute_source,
                line=10,
            ),
        ]
    )
    await store.upsert_edges(
        [
            GraphEdge(
                src="symbol:src/main.py:foo",
                dst="symbol:src/main.py:bar",
                type="CALLS",
                metadata={"file": absolute_source},
            )
        ]
    )
    await store.update_file_mtime(absolute_source, 123.0)

    with sqlite3.connect(store.db_path) as db_conn:
        node_file = db_conn.execute("SELECT file FROM graph_node").fetchone()[0]
        edge_file = db_conn.execute("SELECT file FROM graph_edge").fetchone()[0]
        mtime_file = db_conn.execute("SELECT file FROM graph_file_mtime").fetchone()[0]

    assert node_file == "src/main.py"
    assert edge_file == "src/main.py"
    assert mtime_file == "src/main.py"
    # Two symbols now live in this file (the call's caller and callee), so
    # lookup returns both — by absolute and relative path alike, which is what
    # this test is actually about.
    assert len(await store.get_nodes_by_file(absolute_source)) == 2
    assert len(await store.get_nodes_by_file("src/main.py")) == 2


@pytest.mark.asyncio
async def test_sqlite_graph_store_relative_lookup_finds_legacy_absolute_rows(tmp_path):
    store = SqliteGraphStore(tmp_path)
    source_file = tmp_path / "src" / "legacy.py"
    absolute_source = str(source_file)

    with sqlite3.connect(store.db_path) as db_conn:
        db_conn.execute(
            """
            INSERT INTO graph_node (node_id, type, name, file, line, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "symbol:absolute:legacy",
                "function",
                "legacy",
                absolute_source,
                1,
                "{}",
            ),
        )
        db_conn.execute(
            """
            INSERT INTO graph_file_mtime (file, mtime, indexed_at)
            VALUES (?, ?, ?)
            """,
            (absolute_source, 123.0, 123.0),
        )
        db_conn.commit()

    nodes = await store.get_nodes_by_file("src/legacy.py")
    stale = await store.get_stale_files({"src/legacy.py": 100.0})

    assert [node.node_id for node in nodes] == ["symbol:absolute:legacy"]
    assert stale == []

    await store.delete_by_file("src/legacy.py")

    assert await store.get_nodes_by_file("src/legacy.py") == []
    assert await store.get_indexed_files() == []


def test_sqlite_graph_store_records_project_root_metadata(tmp_path):
    store = SqliteGraphStore(tmp_path)

    with sqlite3.connect(store.db_path) as db_conn:
        rows = dict(
            db_conn.execute(
                "SELECT key, value FROM _project_metadata WHERE key IN (?, ?)",
                ("project_root", "graph_file_path_identity"),
            ).fetchall()
        )

    assert rows["project_root"] == str(tmp_path.resolve())
    assert rows["graph_file_path_identity"] == "repo_relative"


@pytest.mark.asyncio
async def test_sqlite_graph_store_traverses_both_directions_and_depth(tmp_path):
    db_path = tmp_path / "graph.db"
    store = SqliteGraphStore(db_path)

    nodes = [
        GraphNode(node_id="a", type="function", name="a", file="main.py"),
        GraphNode(node_id="b", type="function", name="b", file="main.py"),
        GraphNode(node_id="c", type="function", name="c", file="main.py"),
    ]
    edges = [
        GraphEdge(src="a", dst="b", type="CALLS"),
        GraphEdge(src="b", dst="c", type="CALLS"),
    ]

    await store.upsert_nodes(nodes)
    await store.upsert_edges(edges)

    both = await store.get_neighbors("b")
    incoming = await store.get_neighbors("c", direction="in", max_depth=2)

    assert {(edge.src, edge.dst) for edge in both} == {("a", "b"), ("b", "c")}
    assert {(edge.src, edge.dst) for edge in incoming} == {("a", "b"), ("b", "c")}


@pytest.mark.asyncio
async def test_sqlite_graph_store_write_batch_rolls_back_on_error(tmp_path):
    store = SqliteGraphStore(tmp_path)

    with pytest.raises(RuntimeError, match="boom"):
        async with store.write_batch():
            await store.upsert_nodes(
                [
                    GraphNode(
                        node_id="symbol:main.py:foo",
                        type="function",
                        name="foo",
                        file="main.py",
                        line=1,
                    )
                ]
            )
            await store.update_file_mtime("main.py", 123.0)
            raise RuntimeError("boom")

    stats = await store.stats()
    assert stats["nodes"] == 0
    assert stats["edges"] == 0
    assert stats["indexed_files"] == 0


@pytest.mark.asyncio
async def test_get_stale_files_issues_single_query_regardless_of_file_count(tmp_path):
    """Co-design review item 20a: one full-table SELECT, not one per file."""
    store = SqliteGraphStore(tmp_path)
    files = [f"pkg/mod_{i}.py" for i in range(50)]
    for f in files:
        await store.update_file_mtime(f, 100.0)

    conn = store._connect()
    queries = []
    conn.set_trace_callback(queries.append)
    try:
        stale = await store.get_stale_files(dict.fromkeys(files, 100.0))
    finally:
        conn.set_trace_callback(None)

    select_queries = [q for q in queries if q.strip().upper().startswith("SELECT")]
    assert len(select_queries) == 1
    assert stale == []


@pytest.mark.asyncio
async def test_get_file_hashes_issues_single_query_regardless_of_file_count(tmp_path):
    store = SqliteGraphStore(tmp_path)
    files = [f"pkg/mod_{i}.py" for i in range(50)]
    for f in files:
        await store.update_file_mtime(f, 100.0, content_hash=f"hash-{f}")

    conn = store._connect()
    queries = []
    conn.set_trace_callback(queries.append)
    try:
        hashes = await store.get_file_hashes(files)
    finally:
        conn.set_trace_callback(None)

    select_queries = [q for q in queries if q.strip().upper().startswith("SELECT")]
    assert len(select_queries) == 1
    assert hashes == {f: f"hash-{f}" for f in files}


@pytest.mark.asyncio
async def test_get_stale_files_batched_matches_per_file_semantics(tmp_path):
    """Batched rewrite must still detect staleness and freshness per file."""
    store = SqliteGraphStore(tmp_path)
    await store.update_file_mtime("fresh.py", 200.0)
    await store.update_file_mtime("old.py", 100.0)
    # "unknown.py" never indexed.

    stale = await store.get_stale_files({"fresh.py": 200.0, "old.py": 150.0, "unknown.py": 1.0})

    assert set(stale) == {"old.py", "unknown.py"}


@pytest.mark.asyncio
async def test_get_file_hashes_batched_omits_files_without_hash(tmp_path):
    store = SqliteGraphStore(tmp_path)
    await store.update_file_mtime("hashed.py", 1.0, content_hash="abc123")
    await store.update_file_mtime("unhashed.py", 1.0)

    hashes = await store.get_file_hashes(["hashed.py", "unhashed.py", "missing.py"])

    assert hashes == {"hashed.py": "abc123"}


@pytest.mark.asyncio
async def test_update_nodes_metadata_batches_write_and_skips_unknown_ids(tmp_path):
    store = SqliteGraphStore(tmp_path)
    await store.upsert_nodes(
        [
            GraphNode(node_id="n1", type="function", name="a", file="a.py", line=1),
            GraphNode(node_id="n2", type="function", name="b", file="b.py", line=1),
        ]
    )

    conn = store._connect()
    queries = []
    conn.set_trace_callback(queries.append)
    try:
        await store.update_nodes_metadata(
            [
                ("n1", {"embedding_ref": "emb:n1", "has_embedding": True}),
                ("n2", {"has_embedding": True}),
                ("does-not-exist", {"has_embedding": True}),
            ]
        )
    finally:
        conn.set_trace_callback(None)

    select_queries = [q for q in queries if q.strip().upper().startswith("SELECT")]
    assert len(select_queries) == 1

    nodes = await store.get_nodes_by_file("a.py")
    assert nodes[0].metadata.get("has_embedding") is True
    assert nodes[0].embedding_ref == "emb:n1"

    nodes_b = await store.get_nodes_by_file("b.py")
    assert nodes_b[0].metadata.get("has_embedding") is True


@pytest.mark.asyncio
async def test_update_nodes_metadata_empty_pairs_is_noop(tmp_path):
    store = SqliteGraphStore(tmp_path)
    await store.update_nodes_metadata([])
