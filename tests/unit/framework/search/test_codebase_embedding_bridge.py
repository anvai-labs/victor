# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
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

"""Tests for structural codebase embedding bridge helpers."""

from __future__ import annotations

from typing import Any, Optional

import pytest

from victor.framework.search.codebase_embedding_bridge import (
    STRUCTURAL_CODEBASE_VECTOR_STORE,
    _resolve_language,
    build_codebase_index_manifest,
    enable_structural_codebase_embeddings,
    get_structural_codebase_embedding_provider_class,
    has_compatible_codebase_index_manifest,
    write_codebase_index_manifest,
)
from victor.storage.vector_stores.base import (
    BaseEmbeddingProvider,
    EmbeddingConfig,
    EmbeddingSearchResult,
)
from victor.storage.vector_stores.registry import EmbeddingRegistry


class _FakeInnerProvider(BaseEmbeddingProvider):
    """Minimal in-memory provider for bridge tests."""

    instances: list["_FakeInnerProvider"] = []

    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)
        self.indexed_documents: list[dict[str, Any]] = []
        self.deleted_doc_ids: list[str] = []
        self.deleted_files: list[str] = []
        _FakeInnerProvider.instances.append(self)

    async def initialize(self) -> None:
        self._initialized = True

    async def embed_text(self, text: str) -> list[float]:
        return [float(len(text))]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text))] for text in texts]

    async def index_document(self, doc_id: str, content: str, metadata: dict[str, Any]) -> None:
        self.indexed_documents.append(
            {"id": doc_id, "content": content, "metadata": dict(metadata)}
        )

    async def index_documents(self, documents: list[dict[str, Any]]) -> None:
        self.indexed_documents.extend(
            {
                "id": document["id"],
                "content": document["content"],
                "metadata": dict(document.get("metadata", {})),
            }
            for document in documents
        )

    async def search_similar(
        self,
        query: str,
        limit: int = 10,
        filter_metadata: Optional[dict[str, Any]] = None,
    ) -> list[EmbeddingSearchResult]:
        del query
        results: list[EmbeddingSearchResult] = []
        for index, document in enumerate(self.indexed_documents):
            metadata = dict(document["metadata"])
            if filter_metadata and any(
                metadata.get(key) != value for key, value in filter_metadata.items()
            ):
                continue
            results.append(
                EmbeddingSearchResult(
                    file_path=metadata.get("file_path", ""),
                    symbol_name=metadata.get("symbol_name"),
                    content=document["content"],
                    score=max(1.0 - (index * 0.05), 0.1),
                    line_number=metadata.get("line_number"),
                    metadata=metadata,
                )
            )
        return results[:limit]

    async def delete_document(self, doc_id: str) -> None:
        self.deleted_doc_ids.append(doc_id)
        self.indexed_documents = [doc for doc in self.indexed_documents if doc["id"] != doc_id]

    async def delete_by_file(self, file_path: str) -> int:
        before = len(self.indexed_documents)
        self.deleted_files.append(file_path)
        self.indexed_documents = [
            doc for doc in self.indexed_documents if doc["metadata"].get("file_path") != file_path
        ]
        return before - len(self.indexed_documents)

    async def clear_index(self) -> None:
        self.indexed_documents.clear()

    async def get_stats(self) -> dict[str, Any]:
        return {
            "provider": "fake",
            "total_documents": len(self.indexed_documents),
        }

    async def close(self) -> None:
        self._initialized = False


def test_codebase_index_manifest_round_trip(tmp_path) -> None:
    manifest = build_codebase_index_manifest(
        {
            "vector_store": STRUCTURAL_CODEBASE_VECTOR_STORE,
            "embedding_model_type": "sentence-transformers",
            "embedding_model_name": "BAAI/bge-small-en-v1.5",
            "extra_config": {
                "upstream_vector_store": "lancedb",
                "dimension": 384,
                "batch_size": 32,
                "code_chunking_strategy": "tree_sitter_structural",
                "chunk_size": 500,
                "chunk_overlap": 50,
                "language_overrides": {"component": "tsx", ".astro": "html"},
            },
        }
    )

    assert manifest["language_overrides"] == {".component": "tsx", ".astro": "html"}
    assert manifest["chunking_strategy"] == "victor_codegraph"

    write_codebase_index_manifest(tmp_path, manifest)
    assert has_compatible_codebase_index_manifest(tmp_path, manifest) is True

    changed_manifest = dict(manifest)
    changed_manifest["embedding_model"] = "other-model"
    assert has_compatible_codebase_index_manifest(tmp_path, changed_manifest) is False

    changed_language_manifest = dict(manifest)
    changed_language_manifest["language_overrides"] = {".component": "svelte"}
    assert has_compatible_codebase_index_manifest(tmp_path, changed_language_manifest) is False


def test_enable_structural_codebase_embeddings_rewrites_vector_store(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "victor.framework.search.codebase_embedding_bridge.register_structural_codebase_embedding_provider",
        lambda: True,
    )

    config = enable_structural_codebase_embeddings(
        {
            "vector_store": "lancedb",
            "embedding_model_type": "sentence-transformers",
            "embedding_model_name": "BAAI/bge-small-en-v1.5",
            "extra_config": {
                "structural_indexing_enabled": True,
                "code_chunking_strategy": "tree_sitter_structural",
            },
        }
    )

    assert config["vector_store"] == STRUCTURAL_CODEBASE_VECTOR_STORE
    assert config["extra_config"]["upstream_vector_store"] == "lancedb"


def test_resolve_language_honors_explicit_suffix_overrides(tmp_path) -> None:
    full_path = tmp_path / "sample.component"
    language = _resolve_language(
        full_path,
        documents=[],
        language_overrides={".component": "tsx"},
    )
    assert language == "tsx"


@pytest.mark.asyncio
async def test_structural_bridge_indexes_grounded_file_chunks(tmp_path, monkeypatch) -> None:
    pytest.importorskip("victor_coding.codebase.embeddings.base")
    provider_class = get_structural_codebase_embedding_provider_class()
    assert provider_class is not None

    fake_store_name = "test_bridge_store"
    monkeypatch.setitem(EmbeddingRegistry._providers, fake_store_name, _FakeInnerProvider)
    monkeypatch.setattr(EmbeddingRegistry, "_provider_cache", {}, raising=False)
    _FakeInnerProvider.instances.clear()

    source_dir = tmp_path / "src"
    source_dir.mkdir()
    (source_dir / "parser.py").write_text(
        "def parse_json(data):\n"
        "    if not data:\n"
        "        return {}\n"
        "    return data\n\n"
        "def parse_json_or_none(data):\n"
        "    return data or None\n",
        encoding="utf-8",
    )

    from victor_coding.codebase.embeddings.base import (
        EmbeddingConfig as CodingEmbeddingConfig,
    )

    provider = provider_class(
        CodingEmbeddingConfig(
            vector_store=STRUCTURAL_CODEBASE_VECTOR_STORE,
            persist_directory=str(tmp_path / ".victor" / "embeddings"),
            embedding_model_type="sentence-transformers",
            embedding_model_name="test-model",
            extra_config={
                "upstream_vector_store": fake_store_name,
                "workspace_root": str(tmp_path),
                "code_chunking_strategy": "symbol_span",
                "chunk_size": 20,
                "chunk_overlap": 0,
            },
        )
    )

    await provider.clear_index()
    await provider.index_documents(
        [
            {
                "id": "symbol:src/parser.py:parse_json",
                "content": "function parse_json",
                "metadata": {
                    "file_path": "src/parser.py",
                    "symbol_name": "parse_json",
                    "symbol_type": "function",
                    "line_number": 1,
                    "end_line": 4,
                },
            },
            {
                "id": "symbol:src/parser.py:parse_json_or_none",
                "content": "function parse_json_or_none",
                "metadata": {
                    "file_path": "src/parser.py",
                    "symbol_name": "parse_json_or_none",
                    "symbol_type": "function",
                    "line_number": 6,
                    "end_line": 7,
                },
            },
        ]
    )

    results = await provider.search_similar("parse json", limit=10)
    fake_provider = _FakeInnerProvider.instances[0]

    assert fake_provider.deleted_files == []
    assert any("def parse_json(data):" in doc["content"] for doc in fake_provider.indexed_documents)
    assert all(
        "function parse_json" not in doc["content"] for doc in fake_provider.indexed_documents
    )
    assert any(
        doc["metadata"].get("qualified_name") == "parse_json"
        for doc in fake_provider.indexed_documents
    )
    assert all(doc["metadata"].get("symbol_id") for doc in fake_provider.indexed_documents)
    assert all(
        doc["id"] == doc["metadata"].get("chunk_id") for doc in fake_provider.indexed_documents
    )
    assert any(result.metadata["chunking_strategy"] == "victor_codegraph" for result in results)
    assert any(result.metadata["file_path"] == "src/parser.py" for result in results)


@pytest.mark.asyncio
async def test_structural_bridge_replaces_by_file_for_incremental_updates(
    tmp_path, monkeypatch
) -> None:
    pytest.importorskip("victor_coding.codebase.embeddings.base")
    provider_class = get_structural_codebase_embedding_provider_class()
    assert provider_class is not None

    fake_store_name = "test_bridge_incremental"
    monkeypatch.setitem(EmbeddingRegistry._providers, fake_store_name, _FakeInnerProvider)
    monkeypatch.setattr(EmbeddingRegistry, "_provider_cache", {}, raising=False)
    _FakeInnerProvider.instances.clear()

    source_dir = tmp_path / "src"
    source_dir.mkdir()
    (source_dir / "parser.py").write_text(
        "def parse_json(data):\n" "    return data\n",
        encoding="utf-8",
    )

    from victor_coding.codebase.embeddings.base import (
        EmbeddingConfig as CodingEmbeddingConfig,
    )

    provider = provider_class(
        CodingEmbeddingConfig(
            vector_store=STRUCTURAL_CODEBASE_VECTOR_STORE,
            persist_directory=str(tmp_path / ".victor" / "embeddings"),
            embedding_model_type="sentence-transformers",
            embedding_model_name="test-model",
            extra_config={
                "upstream_vector_store": fake_store_name,
                "workspace_root": str(tmp_path),
                "code_chunking_strategy": "symbol_span",
            },
        )
    )

    await provider.index_document(
        doc_id="symbol:src/parser.py:parse_json",
        content="function parse_json",
        metadata={
            "file_path": "src/parser.py",
            "symbol_name": "parse_json",
            "symbol_type": "function",
            "line_number": 1,
            "end_line": 2,
        },
    )
    await provider.get_stats()

    fake_provider = _FakeInnerProvider.instances[0]
    assert fake_provider.deleted_files == ["src/parser.py"]

    await provider.delete_document("symbol:src/parser.py:parse_json")
    assert fake_provider.deleted_files == ["src/parser.py", "src/parser.py"]
