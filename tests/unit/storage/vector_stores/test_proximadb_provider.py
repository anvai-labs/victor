# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the ProximaDB embedding provider (no server required)."""

from __future__ import annotations

from victor.storage.proxima_runtime import ProximaEmbeddingMode
from victor.storage.vector_stores.proximadb_provider import (
    ProximaDBProvider,
    create_proximadb_provider,
)


def test_factory_defaults_to_memory_mode():
    provider = create_proximadb_provider()
    assert isinstance(provider, ProximaDBProvider)
    assert provider._embedding_mode is ProximaEmbeddingMode.MEMORY
    assert provider._collection_name == "code_embeddings"
    assert provider._dimension == 384
    assert provider._initialized is False


def test_factory_cold_mode():
    provider = create_proximadb_provider(embedding_mode="cold")
    assert provider._embedding_mode is ProximaEmbeddingMode.COLD


def test_search_result_conversion_with_props_and_score():
    provider = create_proximadb_provider()
    item = {
        "id": "graph/repo/node/abc",
        "score": 0.91,
        "props": {
            "text": "def login(): ...",
            "file": "src/auth.py",
            "name": "login",
            "line": 12,
        },
    }
    result = provider._to_search_result(item)
    assert result.content == "def login(): ..."
    assert result.file_path == "src/auth.py"
    assert result.symbol_name == "login"
    assert result.line_number == 12
    assert result.score == 0.91
    # The text payload is not duplicated into metadata.
    assert "text" not in result.metadata


def test_search_result_conversion_distance_fallback():
    provider = create_proximadb_provider()
    item = {"id": "x", "distance": 1.0, "metadata": {"file_path": "a.py"}}
    result = provider._to_search_result(item)
    assert result.file_path == "a.py"
    # distance 1.0 -> similarity 1/(1+1) = 0.5
    assert abs(result.score - 0.5) < 1e-9
