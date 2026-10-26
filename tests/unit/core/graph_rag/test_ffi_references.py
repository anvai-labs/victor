"""Cross-language FFI REFERENCES edges: python wrapper → rust function."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from victor.storage.graph.protocol import GraphNode


async def _seed_ffi_graph(tmp_path: Path) -> None:
    from victor.storage.graph.sqlite_store import SqliteGraphStore

    nodes = [
        GraphNode(
            node_id="rust_fn",
            type="function",
            name="batch_extract_file_paths",
            file="rust/crates/python-bindings/src/extractor.rs",
            line=333,
            end_line=334,
            lang="rust",
        ),
        GraphNode(
            node_id="wrapper_class",
            type="class",
            name="Extractor",
            file="victor/native/rust/extractor.py",
            line=1,
            end_line=100,
            lang="python",
        ),
        GraphNode(
            node_id="wrapper_fn",
            type="function",
            name="extract",
            file="victor/native/rust/extractor.py",
            line=10,
            end_line=20,
            lang="python",
            parent_id="wrapper_class",
        ),
        GraphNode(
            node_id="ffi_stmt",
            type="statement",
            name="call:1",
            file="victor/native/rust/extractor.py",
            line=12,
            end_line=12,
            lang="python",
            statement_type="call",
            scope_id="wrapper_fn",
            metadata={"uses": ["batch_extract_file_paths", "local_var"]},
        ),
    ]
    store = SqliteGraphStore(tmp_path)
    await store.initialize()
    await store.upsert_nodes(nodes)
    await store.close()


async def _run_pass(tmp_path: Path):
    from victor.core.graph_rag.indexing import GraphIndexingPipeline
    from victor.storage.graph.sqlite_store import SqliteGraphStore

    store = SqliteGraphStore(tmp_path)
    await store.initialize()
    pipeline = GraphIndexingPipeline.__new__(GraphIndexingPipeline)
    pipeline.graph_store = store
    count = await pipeline._link_cross_language_ffi()
    conn = store._connect()
    try:
        edges = conn.execute(
            "SELECT src, dst, type, metadata FROM graph_edge WHERE type = 'REFERENCES'"
        ).fetchall()
    finally:
        conn.close()
    await store.close()
    return count, edges


@pytest.mark.asyncio
async def test_ffi_references_edges_created(tmp_path):
    """The FFI pass creates REFERENCES edges from the wrapper function to
    the rust function, deduplicated per function-pair."""
    await _seed_ffi_graph(tmp_path)
    count, edges = await _run_pass(tmp_path)
    assert count >= 1, "at least one FFI REFERENCES edge must be created"
    assert len(edges) >= 1
    ffi = [e for e in edges if "FFI" in (e[3] or "")]
    assert len(ffi) >= 1
    assert ffi[0][0] == "wrapper_fn"
    assert ffi[0][1] == "rust_fn"


@pytest.mark.asyncio
async def test_ffi_references_metadata_shape(tmp_path):
    """The FFI REFERENCES metadata carries the rust function name."""
    await _seed_ffi_graph(tmp_path)
    _, edges = await _run_pass(tmp_path)
    assert edges, "must have REFERENCES edges"
    md = json.loads(edges[0][3])
    assert md["kind"] == "FFI"
    assert md["use"] == "batch_extract_file_paths"
