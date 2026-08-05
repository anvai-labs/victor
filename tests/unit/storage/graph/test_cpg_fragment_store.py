# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for the durable Tier-B CPG fragment store."""

from victor.storage.graph.cpg_fragments import CpgFragmentStore
from victor.storage.graph.edge_types import EdgeType
from victor.storage.graph.proxima_store import ProximaGraphStore
from victor.storage.graph.protocol import GraphEdge, GraphNode


def _statement(node_id: str, *, file: str, line: int, scope: str) -> GraphNode:
    return GraphNode(
        node_id=node_id,
        type="statement",
        name=f"assignment:{line}",
        file=file,
        line=line,
        end_line=line,
        lang="python",
        ast_kind="assignment",
        scope_id=scope,
        statement_type="assignment",
        metadata={"variable": "value", "line": line},
    )


async def test_fragments_persist_and_support_indexed_drilldown(tmp_path):
    path = tmp_path / "cpg-fragments.sqlite3"
    first = CpgFragmentStore(path)
    await first.initialize()
    await first.upsert_nodes(
        [
            _statement("stmt:1", file="src/a.py", line=3, scope="fn:a"),
            _statement("stmt:2", file="src/a.py", line=4, scope="fn:a"),
            _statement("stmt:3", file="src/b.py", line=8, scope="fn:b"),
        ]
    )
    await first.upsert_edges(
        [
            GraphEdge("stmt:1", "stmt:2", "CFG_SUCCESSOR", metadata={"branch": "next"}),
            GraphEdge("stmt:1", "stmt:3", "DDG_DEF_USE", metadata={"variable": "value"}),
        ]
    )
    await first.close()

    reopened = CpgFragmentStore(path)
    await reopened.initialize()
    try:
        node = await reopened.get_node_by_id("stmt:1")
        assert node is not None
        assert node.metadata == {"variable": "value", "line": 3}
        assert [n.node_id for n in await reopened.get_nodes_by_scope("fn:a")] == [
            "stmt:1",
            "stmt:2",
        ]
        assert [
            n.node_id
            for n in await reopened.get_nodes_by_statement_type("assignment", file="src/b.py")
        ] == ["stmt:3"]
        assert [
            n.node_id for n in await reopened.find_nodes(name="assignment:4", file="src/a.py")
        ] == ["stmt:2"]

        cfg = await reopened.get_neighbors("stmt:1", edge_types={"CFG_SUCCESSOR"}, direction="out")
        assert [(edge.src, edge.dst, edge.type) for edge in cfg] == [
            ("stmt:1", "stmt:2", "CFG_SUCCESSOR")
        ]
        assert cfg[0].metadata == {"branch": "next"}

        stats = await reopened.stats()
        assert stats["nodes"] == 3
        assert stats["edges"] == 2
        assert stats["files"] == 2

        batches = [
            batch async for batch in reopened.iter_edges(batch_size=1, edge_types={"CFG_SUCCESSOR"})
        ]
        assert [[edge.type for edge in batch] for batch in batches] == [["CFG_SUCCESSOR"]]
    finally:
        await reopened.close()


async def test_delete_by_file_removes_its_nodes_and_incident_fragments(tmp_path):
    store = CpgFragmentStore(tmp_path / "cpg-fragments.sqlite3")
    await store.initialize()
    await store.upsert_nodes(
        [
            _statement("stmt:a", file="src/a.py", line=1, scope="fn:a"),
            _statement("stmt:b", file="src/b.py", line=1, scope="fn:b"),
        ]
    )
    await store.upsert_edges(
        [
            GraphEdge("stmt:a", "stmt:b", "DDG_DEF_USE"),
            GraphEdge("stmt:b", "stmt:b", "CFG_LOOP_BACK"),
        ]
    )

    await store.delete_by_file("src/a.py")

    assert await store.get_node_by_id("stmt:a") is None
    assert await store.get_node_by_id("stmt:b") is not None
    assert await store.get_neighbors("stmt:b", direction="both") == [
        GraphEdge("stmt:b", "stmt:b", "CFG_LOOP_BACK")
    ]

    await store.clear()
    assert await store.stats() == {"nodes": 0, "edges": 0, "files": 0}
    await store.close()


async def test_cold_only_writes_do_not_bootstrap_the_hot_graph(tmp_path):
    fragments = CpgFragmentStore(tmp_path / "cpg-fragments.sqlite3")
    store = ProximaGraphStore(
        project_path=tmp_path,
        repo="cold-only",
        cpg_store=fragments,
    )

    await store.upsert_nodes([_statement("stmt:1", file="src/a.py", line=1, scope="fn:a")])
    await store.upsert_edges([GraphEdge("stmt:1", "stmt:1", EdgeType.CFG_LOOP_BACK)])

    assert store._graph is None
    assert [
        edge.type
        for edge in await store.get_neighbors(
            "stmt:1", edge_types={EdgeType.CFG_LOOP_BACK}, direction="out"
        )
    ] == [EdgeType.CFG_LOOP_BACK]
    assert store._graph is None
    await store.close()
