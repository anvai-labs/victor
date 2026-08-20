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

"""Endpoint-integrity enforcement in SqliteGraphStore.upsert_edges.

The SQLite backend used to accept edges whose endpoints were never persisted
(181 dangling edges on the repo-scale corpus), while the ProximaDB backend
rejected them — the same write produced backend-dependent graphs, and resolver
identity bugs reached disk invisibly on the default backend. Both backends now
enforce the same contract at the store layer: a dangling edge is skipped and
logged; the valid rest of the batch still lands.
"""

import logging

import pytest

from victor.storage.graph.protocol import GraphEdge, GraphNode
from victor.storage.graph.sqlite_store import SqliteGraphStore


def _node(node_id: str) -> GraphNode:
    return GraphNode(
        node_id=node_id,
        type="function",
        name=node_id,
        file=f"{node_id}.py",
    )


@pytest.mark.asyncio
async def test_dangling_edges_are_skipped_and_valid_rest_lands(tmp_path, caplog):
    store = SqliteGraphStore(tmp_path / "graph.db")
    await store.upsert_nodes([_node("a"), _node("b"), _node("c")])

    with caplog.at_level(logging.WARNING):
        await store.upsert_edges(
            [
                GraphEdge(src="a", dst="b", type="CALLS"),
                # Dangling source: skipped, not persisted, not fatal.
                GraphEdge(src="ghost", dst="b", type="CALLS"),
                GraphEdge(src="b", dst="c", type="CALLS"),
                # Dangling target: skipped as well.
                GraphEdge(src="a", dst="ghost", type="INHERITS"),
            ]
        )

    edges = await store.get_all_edges()
    pairs = sorted((e.src, e.dst) for e in edges)
    assert pairs == [("a", "b"), ("b", "c")], "only endpoint-valid edges persist"
    assert any(
        "dangling" in rec.message for rec in caplog.records
    ), "dropped edges are reported, never silent"

    await store.close()


@pytest.mark.asyncio
async def test_fully_valid_batches_are_unaffected(tmp_path):
    store = SqliteGraphStore(tmp_path / "graph.db")
    await store.upsert_nodes([_node("a"), _node("b")])

    await store.upsert_edges([GraphEdge(src="a", dst="b", type="CALLS")])
    # Upsert semantics survive: re-writing the same edge updates, not duplicates.
    await store.upsert_edges([GraphEdge(src="a", dst="b", type="CALLS", weight=0.5)])

    edges = await store.get_all_edges()
    assert len(edges) == 1
    assert edges[0].weight == 0.5

    await store.close()
