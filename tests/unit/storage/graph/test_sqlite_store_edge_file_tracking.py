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

"""Edge file-tracking migration and storage tests for SqliteGraphStore."""

import sqlite3

import pytest

from victor.storage.graph.protocol import GraphEdge, GraphNode
from victor.storage.graph.sqlite_store import SqliteGraphStore


def _create_legacy_graph_schema(connection: sqlite3.Connection) -> None:
    connection.execute("""
        CREATE TABLE graph_node (
            node_id TEXT PRIMARY KEY,
            type TEXT NOT NULL,
            name TEXT NOT NULL,
            file TEXT NOT NULL,
            line INTEGER,
            end_line INTEGER,
            lang TEXT,
            signature TEXT,
            docstring TEXT,
            parent_id TEXT,
            embedding_ref TEXT,
            metadata TEXT
        )
    """)
    connection.execute("""
        CREATE TABLE graph_edge (
            src TEXT NOT NULL,
            dst TEXT NOT NULL,
            type TEXT NOT NULL,
            weight REAL,
            metadata TEXT,
            PRIMARY KEY (src, dst, type)
        )
    """)
    connection.execute("""
        CREATE TABLE graph_file_mtime (
            file TEXT PRIMARY KEY,
            mtime REAL NOT NULL,
            indexed_at REAL NOT NULL
        )
    """)


@pytest.mark.asyncio
async def test_migrate_graph_edge_file_column_and_backfill(tmp_path):
    db_path = tmp_path / "legacy_graph.db"
    legacy_conn = sqlite3.connect(db_path)
    _create_legacy_graph_schema(legacy_conn)
    legacy_conn.executemany(
        "INSERT INTO graph_node (node_id, type, name, file, metadata) VALUES (?, ?, ?, ?, '{}')",
        [
            ("node_a", "function", "a", "src.py"),
            ("node_b", "function", "b", "dst.py"),
            ("node_c", "function", "c", "dst2.py"),
        ],
    )
    legacy_conn.executemany(
        "INSERT INTO graph_edge (src, dst, type, weight, metadata) VALUES (?, ?, ?, ?, '{}')",
        [
            ("node_a", "node_b", "CALLS", 1.0),
            ("node_x", "node_b", "CALLS", 1.0),
            ("node_y", "node_z", "CALLS", 1.0),
        ],
    )
    legacy_conn.commit()
    legacy_columns = {row[1] for row in legacy_conn.execute("PRAGMA table_info(graph_edge)")}
    assert "file" not in legacy_columns
    legacy_conn.close()

    # Initialize the store to trigger migration path.
    SqliteGraphStore(db_path)

    with sqlite3.connect(db_path) as db_conn:
        columns = {row[1] for row in db_conn.execute("PRAGMA table_info(graph_edge)")}
        assert "file" in columns
        edges = {
            (row[0], row[1]): row[2]
            for row in db_conn.execute("SELECT src, dst, file FROM graph_edge ORDER BY src, dst")
        }

    assert edges[("node_a", "node_b")] == "src.py"
    assert edges[("node_x", "node_b")] == "dst.py"
    assert edges[("node_y", "node_z")] is None


@pytest.mark.asyncio
async def test_upsert_edges_stores_edge_file_hint_and_node_fallback(tmp_path):
    store = SqliteGraphStore(tmp_path / "graph.db")
    await store.upsert_nodes(
        [
            GraphNode(
                node_id="n_hint",
                type="function",
                name="hint",
                file="hint.py",
            ),
            GraphNode(
                node_id="n_node",
                type="function",
                name="node",
                file="node.py",
            ),
        ]
    )
    await store.upsert_edges(
        [
            GraphEdge(
                src="n_hint",
                dst="n_node",
                type="CALLS",
                metadata={"file": "explicit.py"},
            ),
            GraphEdge(
                src="n_node",
                dst="n_hint",
                type="CALLS",
            ),
        ]
    )

    with sqlite3.connect(store.db_path) as db_conn:
        files = {
            row[0]: row[2]
            for row in db_conn.execute("SELECT src, dst, file FROM graph_edge ORDER BY src")
        }

    assert files["n_hint"] == "explicit.py"
    assert files["n_node"] == "node.py"


@pytest.mark.asyncio
async def test_delete_by_file_removes_orphaned_edges_with_edge_file_value(tmp_path):
    db_path = tmp_path / "delete_by_file.db"
    store = SqliteGraphStore(db_path)
    await store.upsert_nodes(
        [
            GraphNode(
                node_id="file_a_node",
                type="function",
                name="a_node",
                file="a.py",
            ),
            GraphNode(
                node_id="file_b_node",
                type="function",
                name="b_node",
                file="b.py",
            ),
        ]
    )

    with sqlite3.connect(db_path) as db_conn:
        db_conn.execute(
            """
            INSERT INTO graph_edge (src, dst, type, weight, file, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("ghost", "phantom", "CALLS", 1.0, "a.py", "{}"),
        )
        db_conn.commit()

    await store.delete_by_file("a.py")

    assert len(await store.get_all_nodes()) == 1
    assert len(await store.get_all_edges()) == 0
