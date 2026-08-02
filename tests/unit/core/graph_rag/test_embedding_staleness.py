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

"""Staleness-driven, batched graph-node embedding generation.

Pins the _generate_embeddings rewrite: content_version skip-on-match, batched
persistence through index_embedded_documents, and incremental candidate scoping.
"""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from victor.core.graph_rag import GraphIndexConfig, GraphIndexingPipeline
from victor.storage.graph.protocol import GraphNode
from victor.storage.graph.sqlite_store import SqliteGraphStore


@pytest.fixture
def store_and_root():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        yield SqliteGraphStore(project_path=root), root


def _node(node_id: str, file: str, name: str) -> GraphNode:
    return GraphNode(
        node_id=node_id,
        type="function",
        name=name,
        file=file,
        line=1,
        signature=f"{name}()",
        docstring=f"doc for {name}",
    )


def _pipeline(store, root, incremental=False) -> GraphIndexingPipeline:
    config = GraphIndexConfig(
        root_path=root,
        enable_ccg=False,
        enable_embeddings=True,
        enable_subgraph_cache=False,
        incremental=incremental,
    )
    return GraphIndexingPipeline(store, config)


def _patch_embedding_stack(monkeypatch, provider):
    """Route the embedding phase through mocks: service present, batch vectors."""
    from victor.processing import graph_embeddings as ge
    from victor.storage.embeddings import service as svc

    monkeypatch.setattr(svc, "get_embedding_service", lambda: MagicMock())

    async def _fake_embed_batch(self, nodes, graph_store):
        return {n.node_id: [0.1, 0.2, 0.3] for n in nodes}

    monkeypatch.setattr(ge.GraphAwareEmbedder, "embed_batch", _fake_embed_batch)
    monkeypatch.setattr(GraphIndexingPipeline, "_get_vector_provider", lambda self: provider)


@pytest.mark.asyncio
async def test_second_run_skips_unchanged_nodes(store_and_root, monkeypatch):
    store, root = store_and_root
    await store.upsert_nodes([_node("n1", "a.py", "alpha"), _node("n2", "b.py", "beta")])

    provider = MagicMock()
    provider.index_embedded_documents = AsyncMock()
    _patch_embedding_stack(monkeypatch, provider)

    pipeline = _pipeline(store, root)
    stats1 = await pipeline._generate_embeddings()
    assert stats1.embeddings_generated == 2
    provider.index_embedded_documents.assert_awaited()

    # Nodes are marked with content_version + has_embedding.
    nodes = await store.get_all_nodes()
    for n in nodes:
        assert n.metadata.get("has_embedding") is True
        assert n.metadata.get("content_version")
        assert n.embedding_ref == f"emb:{n.node_id}"

    # Second run: everything unchanged -> nothing re-embedded.
    provider.index_embedded_documents.reset_mock()
    stats2 = await _pipeline(store, root)._generate_embeddings()
    assert stats2.embeddings_generated == 0
    provider.index_embedded_documents.assert_not_awaited()


@pytest.mark.asyncio
async def test_changed_signature_forces_reembed(store_and_root, monkeypatch):
    store, root = store_and_root
    await store.upsert_nodes([_node("n1", "a.py", "alpha")])

    provider = MagicMock()
    provider.index_embedded_documents = AsyncMock()
    _patch_embedding_stack(monkeypatch, provider)

    await _pipeline(store, root)._generate_embeddings()

    # Change the embedded text (signature) -> fingerprint differs -> re-embed.
    changed = _node("n1", "a.py", "alpha")
    changed.signature = "alpha(x, y)"
    # Preserve embedding markers as a real reindex would (metadata survives via
    # upsert only if carried over; simulate by patching metadata directly).
    existing = (await store.get_all_nodes())[0]
    changed.metadata = dict(existing.metadata)
    await store.upsert_nodes([changed])

    stats = await _pipeline(store, root)._generate_embeddings()
    assert stats.embeddings_generated == 1


@pytest.mark.asyncio
async def test_incremental_run_scopes_to_processed_files(store_and_root, monkeypatch):
    store, root = store_and_root
    await store.upsert_nodes([_node("n1", "a.py", "alpha"), _node("n2", "b.py", "beta")])

    provider = MagicMock()
    provider.index_embedded_documents = AsyncMock()
    _patch_embedding_stack(monkeypatch, provider)

    pipeline = _pipeline(store, root, incremental=True)
    pipeline._files_to_process = {"a.py"}

    stats = await pipeline._generate_embeddings()
    assert stats.embeddings_generated == 1
    (docs,) = provider.index_embedded_documents.await_args.args
    assert [d["id"] for d in docs] == ["n1"]
    assert docs[0]["metadata"]["file_path"] == "a.py"
    assert docs[0]["metadata"]["content_version"]


@pytest.mark.asyncio
async def test_vector_persistence_failure_leaves_node_unmarked(store_and_root, monkeypatch):
    store, root = store_and_root
    await store.upsert_nodes([_node("n1", "a.py", "alpha")])

    provider = MagicMock()
    provider.index_embedded_documents = AsyncMock(side_effect=RuntimeError("disk full"))
    _patch_embedding_stack(monkeypatch, provider)

    stats = await _pipeline(store, root)._generate_embeddings()
    assert stats.embeddings_generated == 0
    assert stats.error_count >= 1
    node = (await store.get_all_nodes())[0]
    # Not marked embedded -> next run retries instead of silently skipping.
    assert not node.metadata.get("has_embedding")


@pytest.mark.asyncio
async def test_update_node_metadata_merges_and_sets_embedding_ref(store_and_root):
    store, _root = store_and_root
    await store.upsert_nodes([_node("n1", "a.py", "alpha")])

    await store.update_node_metadata("n1", {"has_embedding": True, "embedding_ref": "emb:n1"})
    await store.update_node_metadata("n1", {"content_version": "abc123"})

    node = (await store.get_all_nodes())[0]
    assert node.metadata["has_embedding"] is True
    assert node.metadata["content_version"] == "abc123"
    assert node.embedding_ref == "emb:n1"

    # Unknown node id: silent no-op.
    await store.update_node_metadata("missing", {"x": 1})
