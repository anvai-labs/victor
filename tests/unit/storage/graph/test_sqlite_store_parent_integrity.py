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

"""Parent-link integrity in SqliteGraphStore.upsert_nodes.

``graph_node(parent_id)`` carries a plain (NOT deferrable) foreign key that
SQLite checks statement-by-statement, but node batches are written with one
executemany in graph-emission order — and emitters do not promise
parents-first order. Real-world Rust files (e.g. proximaDB ``src/audit/mod.rs``)
emit impl members before their impl block, so the child row preceded its
parent in the same batch and the whole file failed with ``FOREIGN KEY
constraint failed``; the pipeline's per-file retry cannot fix an ordering
problem that lives inside the single file batch.

The store now makes every batch safe before the executemany: parent links
that are in neither the batch nor the table are cleared and logged (the node
counterpart of the dangling-edge guard), and rows are ordered
parents-before-children, with a minimal one-link break (logged) for the
unorderable parent-cycle case that only colliding node ids can produce.
"""

import logging

from victor.storage.graph.protocol import GraphNode
from victor.storage.graph.sqlite_store import SqliteGraphStore


def _node(node_id: str, parent_id: str | None = None) -> GraphNode:
    return GraphNode(
        node_id=node_id,
        type="method" if parent_id else "impl",
        name=node_id,
        file="src/audit/mod.rs",
        parent_id=parent_id,
    )


async def test_child_emitted_before_parent_lands(tmp_path):
    store = SqliteGraphStore(tmp_path / "graph.db")

    # Emission order reversed: member, impl block, module root.
    await store.upsert_nodes([_node("member", "impl_blk"), _node("impl_blk", "mod"), _node("mod")])

    member = await store.get_node_by_id("member")
    impl_blk = await store.get_node_by_id("impl_blk")
    assert member is not None and member.parent_id == "impl_blk"
    assert impl_blk is not None and impl_blk.parent_id == "mod"
    assert (await store.get_node_by_id("mod")) is not None

    await store.close()


async def test_child_before_parent_lands_inside_write_batch(tmp_path):
    """The indexing pipeline writes whole files inside one write_batch."""
    store = SqliteGraphStore(tmp_path / "graph.db")

    async with store.write_batch():
        await store.upsert_nodes([_node("member", "impl_blk"), _node("impl_blk")])

    member = await store.get_node_by_id("member")
    assert member is not None and member.parent_id == "impl_blk"

    await store.close()


async def test_dangling_parent_link_is_cleared_and_logged(tmp_path, caplog):
    store = SqliteGraphStore(tmp_path / "graph.db")

    with caplog.at_level(logging.WARNING):
        # "ghost" is neither in the batch nor persisted: the link can never
        # satisfy the foreign key, but the symbol itself must still land.
        await store.upsert_nodes([_node("orphan", "ghost"), _node("ok")])

    orphan = await store.get_node_by_id("orphan")
    assert orphan is not None and orphan.parent_id is None
    assert (await store.get_node_by_id("ok")) is not None
    assert any("parent link" in rec.message for rec in caplog.records), "loss is reported"

    await store.close()


async def test_parent_cycle_is_broken_minimally_and_logged(tmp_path, caplog):
    store = SqliteGraphStore(tmp_path / "graph.db")

    with caplog.at_level(logging.WARNING):
        # a <-> b has no valid order; desc only hangs off the cycle.
        await store.upsert_nodes(
            [_node("a", "b"), _node("b", "a"), _node("desc", "a"), _node("root")]
        )

    a = await store.get_node_by_id("a")
    b = await store.get_node_by_id("b")
    desc = await store.get_node_by_id("desc")
    assert a is not None and b is not None and desc is not None
    # Exactly one link of the cycle is cleared (which one depends on set
    # iteration order); every other link — including the descendant's — survives.
    assert (a.parent_id is None) != (b.parent_id is None)
    kept = b.parent_id if a.parent_id is None else a.parent_id
    assert kept in {"a", "b"}
    assert desc.parent_id == "a"
    assert any("cycle" in rec.message for rec in caplog.records), "break is reported"

    await store.close()


async def test_self_parent_is_cleared_and_row_lands(tmp_path):
    store = SqliteGraphStore(tmp_path / "graph.db")

    await store.upsert_nodes([_node("selfp", "selfp")])

    selfp = await store.get_node_by_id("selfp")
    assert selfp is not None and selfp.parent_id is None

    await store.close()


async def test_duplicate_node_id_keeps_last_row_and_links_survive(tmp_path):
    store = SqliteGraphStore(tmp_path / "graph.db")

    # Same id twice: executemany's sequential upserts would leave the last
    # row's values; the parents-first path must preserve that outcome.
    await store.upsert_nodes(
        [
            _node("dup", "impl_blk"),
            _node("dup"),
            _node("member", "impl_blk"),
            _node("impl_blk"),
        ]
    )

    dup = await store.get_node_by_id("dup")
    member = await store.get_node_by_id("member")
    assert dup is not None and dup.parent_id is None
    assert member is not None and member.parent_id == "impl_blk"

    await store.close()
