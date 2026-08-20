# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for promoted storage protocols — TDD."""

from __future__ import annotations


from victor_contracts.verticals.protocols.storage import (
    EmbeddingConfigData,
    EmbeddingSearchResultData,
    EmbeddingServiceProtocol,
    GraphEdgeData,
    GraphNodeData,
    GraphStoreProtocol,
    VectorStoreProtocol,
)


class TestGraphNodeData:
    def test_fields(self):
        n = GraphNodeData(node_id="n1", type="function", name="foo", file="a.py")
        assert n.node_id == "n1"
        assert n.type == "function"
        assert n.name == "foo"
        assert n.file == "a.py"
        assert n.line is None
        assert n.end_line is None
        assert n.lang is None

    def test_with_all_fields(self):
        n = GraphNodeData(
            node_id="n2",
            type="class",
            name="Bar",
            file="b.py",
            line=10,
            end_line=50,
            lang="python",
            signature="class Bar:",
            docstring="A class",
            parent_id="mod1",
        )
        assert n.line == 10
        assert n.parent_id == "mod1"


class TestGraphEdgeData:
    def test_fields(self):
        e = GraphEdgeData(src="n1", dst="n2", type="CALLS")
        assert e.src == "n1"
        assert e.dst == "n2"
        assert e.type == "CALLS"
        assert e.weight is None

    def test_with_weight(self):
        e = GraphEdgeData(src="a", dst="b", type="REFERENCES", weight=0.8)
        assert e.weight == 0.8


class TestEmbeddingSearchResultData:
    def test_fields(self):
        r = EmbeddingSearchResultData(file_path="a.py", content="def foo():", score=0.95)
        assert r.file_path == "a.py"
        assert r.score == 0.95
        assert r.symbol_name is None
        assert r.line_number is None


class TestEmbeddingConfigData:
    def test_fields(self):
        c = EmbeddingConfigData(vector_store="lancedb")
        assert c.vector_store == "lancedb"
        assert c.distance_metric == "cosine"


class TestGraphStoreProtocol:
    def test_structural_check(self):
        class FakeGraphStore:
            async def upsert_nodes(self, nodes):
                pass

            async def upsert_edges(self, edges):
                pass

            async def get_neighbors(self, node_id, edge_type=None):
                return []

            async def find_nodes(self, **kwargs):
                return []

        assert isinstance(FakeGraphStore(), GraphStoreProtocol)


class TestVectorStoreProtocol:
    def test_structural_check(self):
        class FakeVectorStore:
            async def index_document(self, doc_id, content, embedding, metadata=None):
                pass

            async def search_similar(self, query_embedding, limit=10):
                return []

            async def delete_document(self, doc_id):
                pass

        assert isinstance(FakeVectorStore(), VectorStoreProtocol)


class TestEmbeddingServiceProtocol:
    def test_structural_check(self):
        class FakeEmbedding:
            async def embed_text(self, text):
                return [0.1, 0.2]

            async def embed_batch(self, texts):
                return [[0.1], [0.2]]

        assert isinstance(FakeEmbedding(), EmbeddingServiceProtocol)
