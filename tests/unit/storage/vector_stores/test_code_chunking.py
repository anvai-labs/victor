from __future__ import annotations

from typing import Any, Optional
from unittest.mock import patch

import pytest

from victor.storage.vector_stores.base import EmbeddingConfig
from victor.storage.vector_stores.code_chunking import (
    VictorCodegraphCodeChunker,
    create_code_chunker,
    normalize_code_chunking_strategy,
)
from victor.storage.vector_stores.proximadb_multi import ProximaDBMultiModelProvider


class StubEmbeddingModel:
    async def initialize(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def embed_text(self, text: str) -> list[float]:
        del text
        return [0.1, 0.2, 0.3, 0.4]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

    def get_dimension(self) -> int:
        return 4


class FakeClient:
    def create_collection(self, name: str, config: Any = None, **kwargs: Any) -> dict[str, Any]:
        del config, kwargs
        return {"name": name}

    def create_graph(
        self,
        graph_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        schema: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        del name, description, schema
        return {"graph_id": graph_id}


def test_canonical_chunker_owns_symbols_and_hard_budget() -> None:
    pytest.importorskip("victor_codegraph")
    content = (
        "def canonical_name(value):\n"
        "    total = value + 1\n"
        "    total += 2\n"
        "    total += 3\n"
        "    return total\n"
    )
    chunker = VictorCodegraphCodeChunker(chunk_size=8, chunk_overlap=1)

    chunks = chunker.chunk("src/example.py", content, language="python")

    assert chunks
    assert {chunk.symbol_name for chunk in chunks} == {"canonical_name"}
    assert all(len(chunk.content.split()) <= 8 for chunk in chunks)
    assert all(chunk.chunk_type == "function" for chunk in chunks)
    assert all(chunk.symbol_id for chunk in chunks)
    assert all(
        chunk.chunk_id and chunk.chunk_id.startswith(chunk.symbol_id or "") for chunk in chunks
    )


@pytest.mark.parametrize(
    "legacy_name",
    ["default", "body_aware", "symbol_span", "tree_sitter_structural", "ast_structural"],
)
def test_legacy_names_converge_on_one_implementation(legacy_name: str) -> None:
    assert normalize_code_chunking_strategy(legacy_name) == "victor_codegraph"
    assert isinstance(
        create_code_chunker(legacy_name, chunk_size=10, chunk_overlap=0),
        VictorCodegraphCodeChunker,
    )


def test_create_code_chunker_rejects_unknown_strategy() -> None:
    with pytest.raises(ValueError, match="Unknown code chunking strategy"):
        create_code_chunker("does_not_exist", chunk_size=10, chunk_overlap=0)


def test_duplicate_chunk_engines_are_not_retained() -> None:
    from victor.storage.vector_stores import code_chunking

    assert not hasattr(code_chunking, "SymbolSpanCodeChunker")
    assert not hasattr(code_chunking, "TreeSitterStructuralCodeChunker")
    assert not hasattr(code_chunking, "CodeChunkingContext")


def test_dependency_failure_uses_single_bounded_plain_fallback(monkeypatch) -> None:
    codegraph = pytest.importorskip("victor_codegraph")
    monkeypatch.setattr(
        codegraph, "chunk", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError())
    )
    content = "one two three four five six seven eight nine ten"

    chunks = VictorCodegraphCodeChunker(chunk_size=4, chunk_overlap=1).chunk(
        "src/example.py", content, language="python"
    )

    assert chunks
    assert all(len(chunk.content.split()) <= 4 for chunk in chunks)
    assert all(chunk.symbol_name is None for chunk in chunks)


def test_provider_normalizes_persisted_legacy_strategy() -> None:
    config = EmbeddingConfig(
        vector_store="proximadb_multi",
        embedding_model_type="sentence-transformers",
        embedding_model_name="all-MiniLM-L12-v2",
        distance_metric="cosine",
        extra_config={
            "workspace": "victor_test_repo",
            "dimension": 4,
            "chunk_size": 8,
            "chunk_overlap": 0,
            "code_chunking_strategy": "tree_sitter_structural",
        },
    )

    with patch(
        "victor.storage.vector_stores.proximadb_multi.create_embedding_model",
        return_value=StubEmbeddingModel(),
    ):
        provider = ProximaDBMultiModelProvider(config, client=FakeClient())

    assert provider._code_chunking_strategy == "victor_codegraph"
    assert isinstance(provider._code_chunker, VictorCodegraphCodeChunker)
