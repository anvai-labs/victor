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

"""Unit tests for LanceDB embedding provider."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from victor.storage.vector_stores.base import EmbeddingConfig, EmbeddingSearchResult
from victor.storage.vector_stores.lancedb_provider import LanceDBProvider


@pytest.fixture
def lancedb_config():
    """Create LanceDB config for testing."""
    return EmbeddingConfig(
        vector_store="lancedb",
        persist_directory="/tmp/test_lancedb",
        distance_metric="cosine",
        embedding_model_type="ollama",
        embedding_model_name="gte-Qwen2-7B-instruct",
        embedding_api_key="http://localhost:11434",
        extra_config={
            "table_name": "test_table",
            "dimension": 3584,
            "batch_size": 8,
        },
    )


@pytest.fixture
def mock_lancedb():
    """Mock LanceDB client and table."""
    backend = MagicMock(spec=["connect"])
    with (
        patch.dict("sys.modules", {"lancedb": backend}),
        patch("victor.storage.vector_stores.lancedb_provider.LANCEDB_AVAILABLE", True),
    ):
        mock_connect = backend.connect
        mock_db = MagicMock()
        mock_table = MagicMock()
        mock_table.count_rows.return_value = 0
        mock_table.schema.metadata = {}
        mock_table.schema.names = []

        # Mock list_tables() API (new)
        mock_list_response = MagicMock()
        mock_list_response.tables = []
        mock_db.list_tables.return_value = mock_list_response

        # Keep old table_names for backward compatibility in tests
        mock_db.table_names.return_value = []

        mock_db.open_table.return_value = mock_table
        mock_db.create_table.return_value = mock_table
        mock_connect.return_value = mock_db
        yield mock_connect, mock_db, mock_table


class TestLanceDBProvider:
    """Tests for LanceDBProvider."""

    def test_initialization(self, lancedb_config):
        """Test provider initialization."""
        with patch("victor.storage.vector_stores.lancedb_provider.LANCEDB_AVAILABLE", True):
            provider = LanceDBProvider(lancedb_config)

            assert provider.config == lancedb_config
            assert provider.db is None
            assert provider.table is None
            assert provider.embedding_model is None
            assert not provider._initialized

    def test_initialization_lancedb_not_available(self, lancedb_config):
        """Test initialization when LanceDB not available."""
        with patch("victor.storage.vector_stores.lancedb_provider.LANCEDB_AVAILABLE", False):
            with pytest.raises(ImportError, match="LanceDB not available"):
                LanceDBProvider(lancedb_config)

    async def test_backend_import_failure_precedes_model_initialization(self, lancedb_config):
        """A broken installed backend must fail before opening model resources."""
        with (
            patch("victor.storage.vector_stores.lancedb_provider.LANCEDB_AVAILABLE", True),
            patch.dict("sys.modules", {"lancedb": None}),
            patch("victor.storage.vector_stores.lancedb_provider.create_embedding_model") as create,
        ):
            provider = LanceDBProvider(lancedb_config)
            with pytest.raises(ModuleNotFoundError, match="lancedb"):
                await provider.initialize()
            create.assert_not_called()
            assert provider.embedding_model is None
            assert not provider._initialized

    @pytest.mark.asyncio
    async def test_initialize_persistent(self, lancedb_config, mock_lancedb):
        """Test initialization with persistent storage."""
        mock_connect, mock_db, mock_table = mock_lancedb

        with patch(
            "victor.storage.vector_stores.lancedb_provider.create_embedding_model"
        ) as mock_create:
            mock_embedding_model = AsyncMock()
            mock_embedding_model.initialize = AsyncMock()
            mock_embedding_model.get_dimension = MagicMock(return_value=3584)
            mock_create.return_value = mock_embedding_model

            provider = LanceDBProvider(lancedb_config)
            await provider.initialize()

            assert provider._initialized
            assert provider.db == mock_db
            assert provider.embedding_model == mock_embedding_model
            mock_embedding_model.initialize.assert_called_once()
            mock_connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_without_persist_directory(self, mock_lancedb):
        """Test initialization without persist directory."""
        config = EmbeddingConfig(
            vector_store="lancedb",
            persist_directory=None,  # No directory specified
            embedding_model_type="ollama",
            embedding_model_name="gte-Qwen2-7B-instruct",
            embedding_api_key="http://localhost:11434",
            extra_config={"dimension": 3584},
        )

        mock_connect, mock_db, mock_table = mock_lancedb

        with patch(
            "victor.storage.vector_stores.lancedb_provider.create_embedding_model"
        ) as mock_create:
            mock_embedding_model = AsyncMock()
            mock_embedding_model.initialize = AsyncMock()
            mock_embedding_model.get_dimension = MagicMock(return_value=3584)
            mock_create.return_value = mock_embedding_model

            provider = LanceDBProvider(config)
            await provider.initialize()

            assert provider._initialized
            # Should use default temp directory
            mock_connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_embed_text(self, lancedb_config, mock_lancedb):
        """Test single text embedding."""
        mock_connect, mock_db, mock_table = mock_lancedb

        with patch(
            "victor.storage.vector_stores.lancedb_provider.create_embedding_model"
        ) as mock_create:
            mock_embedding_model = AsyncMock()
            mock_embedding_model.initialize = AsyncMock()
            mock_embedding_model.get_dimension = MagicMock(return_value=3584)
            mock_embedding_model.embed_text.return_value = [0.1] * 3584
            mock_create.return_value = mock_embedding_model

            provider = LanceDBProvider(lancedb_config)
            await provider.initialize()

            result = await provider.embed_text("test text")

            assert len(result) == 3584
            mock_embedding_model.embed_text.assert_called_once_with("test text")

    @pytest.mark.asyncio
    async def test_embed_batch(self, lancedb_config, mock_lancedb):
        """Test batch embedding."""
        mock_connect, mock_db, mock_table = mock_lancedb

        with patch(
            "victor.storage.vector_stores.lancedb_provider.create_embedding_model"
        ) as mock_create:
            mock_embedding_model = AsyncMock()
            mock_embedding_model.initialize = AsyncMock()
            mock_embedding_model.get_dimension = MagicMock(return_value=3584)
            mock_embedding_model.embed_batch.return_value = [[0.1] * 3584, [0.2] * 3584]
            mock_create.return_value = mock_embedding_model

            provider = LanceDBProvider(lancedb_config)
            await provider.initialize()

            texts = ["text1", "text2"]
            results = await provider.embed_batch(texts)

            assert len(results) == 2
            assert all(len(emb) == 3584 for emb in results)
            mock_embedding_model.embed_batch.assert_called_once_with(texts)

    @pytest.mark.asyncio
    async def test_index_document_create_table(self, lancedb_config, mock_lancedb):
        """Test indexing single document when table doesn't exist."""
        mock_connect, mock_db, mock_table = mock_lancedb

        with patch(
            "victor.storage.vector_stores.lancedb_provider.create_embedding_model"
        ) as mock_create:
            mock_embedding_model = AsyncMock()
            mock_embedding_model.initialize = AsyncMock()
            mock_embedding_model.get_dimension = MagicMock(return_value=3584)
            mock_embedding_model.embed_text.return_value = [0.1] * 3584
            mock_create.return_value = mock_embedding_model

            provider = LanceDBProvider(lancedb_config)
            await provider.initialize()

            await provider.index_document(
                doc_id="test_id",
                content="test content",
                metadata={"file_path": "test.py"},
            )

            # Should create table since it didn't exist
            mock_db.create_table.assert_called_once()
            call_args = mock_db.create_table.call_args
            assert call_args[0][0] == "test_table"  # table name
            assert len(call_args.kwargs["data"]) == 1  # one document
            doc = call_args.kwargs["data"][0]
            assert doc["id"] == "test_id"
            assert doc["content"] == "test content"
            assert len(doc["vector"]) == 3584

    @pytest.mark.asyncio
    async def test_index_document_add_to_existing(self, lancedb_config, mock_lancedb):
        """Test indexing document when table exists."""
        mock_connect, mock_db, mock_table = mock_lancedb
        mock_db.list_tables.return_value.tables = ["test_table"]
        mock_db.table_names.return_value = ["test_table"]  # Also set for get_table_names()

        with patch(
            "victor.storage.vector_stores.lancedb_provider.create_embedding_model"
        ) as mock_create:
            mock_embedding_model = AsyncMock()
            mock_embedding_model.initialize = AsyncMock()
            mock_embedding_model.get_dimension = MagicMock(return_value=3584)
            mock_embedding_model.embed_text.return_value = [0.1] * 3584
            mock_create.return_value = mock_embedding_model

            provider = LanceDBProvider(lancedb_config)
            await provider.initialize()

            # Table should be opened
            assert provider.table == mock_table

            await provider.index_document(
                doc_id="test_id",
                content="test content",
                metadata={"file_path": "test.py"},
            )

            # Should add to existing table
            mock_table.add.assert_called_once()
            call_args = mock_table.add.call_args[0][0]
            assert len(call_args) == 1
            assert call_args[0]["id"] == "test_id"

    @pytest.mark.asyncio
    async def test_index_documents(self, lancedb_config, mock_lancedb):
        """Test batch indexing documents."""
        mock_connect, mock_db, mock_table = mock_lancedb

        with patch(
            "victor.storage.vector_stores.lancedb_provider.create_embedding_model"
        ) as mock_create:
            mock_embedding_model = AsyncMock()
            mock_embedding_model.initialize = AsyncMock()
            mock_embedding_model.get_dimension = MagicMock(return_value=3584)
            mock_embedding_model.embed_batch.return_value = [[0.1] * 3584, [0.2] * 3584]
            mock_create.return_value = mock_embedding_model

            provider = LanceDBProvider(lancedb_config)
            await provider.initialize()

            documents = [
                {
                    "id": "doc1",
                    "content": "content1",
                    "metadata": {"file_path": "test1.py"},
                },
                {
                    "id": "doc2",
                    "content": "content2",
                    "metadata": {"file_path": "test2.py"},
                },
            ]

            await provider.index_documents(documents)

            mock_db.create_table.assert_called_once()
            call_args = mock_db.create_table.call_args.kwargs["data"]
            assert len(call_args) == 2
            assert all("vector" in doc for doc in call_args)
            assert all(len(doc["vector"]) == 3584 for doc in call_args)

    @pytest.mark.asyncio
    async def test_search_similar(self, lancedb_config, mock_lancedb):
        """Test semantic similarity search."""
        mock_connect, mock_db, mock_table = mock_lancedb

        # Fix: Override list_tables to return ["test_table"]
        # We need to modify the mock's return_value directly
        mock_list_response = mock_db.list_tables.return_value
        mock_list_response.tables = ["test_table"]
        mock_db.table_names.return_value = ["test_table"]  # Also set for get_table_names()

        # Mock search results
        mock_search = MagicMock()
        mock_search_result = MagicMock()

        # Create mock result objects
        result1 = {
            "id": "doc1",
            "content": "content1",
            "file_path": "test1.py",
            "symbol_name": "func1",
            "line_number": 12,
            "vector": [0.1, 0.2, 0.3],
            "symbol_type": "function",
            "_distance": 0.1,
        }
        result2 = {
            "id": "doc2",
            "content": "content2",
            "file_path": "test2.py",
            "symbol_name": "func2",
            "_distance": 0.2,
        }

        mock_search_result.to_list.return_value = [result1, result2]
        mock_search_result.where.return_value = mock_search_result
        mock_search.limit.return_value = mock_search_result
        mock_table.search.return_value = mock_search

        with patch(
            "victor.storage.vector_stores.lancedb_provider.create_embedding_model"
        ) as mock_create:
            mock_embedding_model = AsyncMock()
            mock_embedding_model.initialize = AsyncMock()
            mock_embedding_model.get_dimension = MagicMock(return_value=3584)
            mock_embedding_model.embed_text.return_value = [0.1] * 3584
            mock_create.return_value = mock_embedding_model

            provider = LanceDBProvider(lancedb_config)
            await provider.initialize()

            results = await provider.search_similar("query text", limit=2)

            assert len(results) == 2
            assert isinstance(results[0], EmbeddingSearchResult)
            assert results[0].file_path == "test1.py"
            assert results[0].content == "content1"
            assert results[0].score > 0
            assert "vector" not in results[0].metadata
            assert "content" not in results[0].metadata
            assert "file_path" not in results[0].metadata
            assert results[0].metadata["symbol_type"] == "function"

    @pytest.mark.asyncio
    async def test_search_similar_no_table(self, lancedb_config, mock_lancedb):
        """Test search when table doesn't exist."""
        mock_connect, mock_db, mock_table = mock_lancedb

        with patch(
            "victor.storage.vector_stores.lancedb_provider.create_embedding_model"
        ) as mock_create:
            mock_embedding_model = AsyncMock()
            mock_embedding_model.initialize = AsyncMock()
            mock_embedding_model.get_dimension = MagicMock(return_value=3584)
            mock_create.return_value = mock_embedding_model

            provider = LanceDBProvider(lancedb_config)
            await provider.initialize()

            results = await provider.search_similar("query text", limit=2)

            assert results == []

    @pytest.mark.asyncio
    async def test_delete_document(self, lancedb_config, mock_lancedb):
        """Test deleting document."""
        mock_connect, mock_db, mock_table = mock_lancedb
        mock_db.list_tables.return_value.tables = ["test_table"]
        mock_db.table_names.return_value = ["test_table"]  # Also set for get_table_names()

        with patch(
            "victor.storage.vector_stores.lancedb_provider.create_embedding_model"
        ) as mock_create:
            mock_embedding_model = AsyncMock()
            mock_embedding_model.initialize = AsyncMock()
            mock_embedding_model.get_dimension = MagicMock(return_value=3584)
            mock_create.return_value = mock_embedding_model

            provider = LanceDBProvider(lancedb_config)
            await provider.initialize()

            await provider.delete_document("doc_id")

            mock_table.delete.assert_called_once_with("id = 'doc_id'")

    @pytest.mark.asyncio
    async def test_delete_document_no_table(self, lancedb_config, mock_lancedb):
        """Test delete when table doesn't exist."""
        mock_connect, mock_db, mock_table = mock_lancedb

        with patch(
            "victor.storage.vector_stores.lancedb_provider.create_embedding_model"
        ) as mock_create:
            mock_embedding_model = AsyncMock()
            mock_embedding_model.initialize = AsyncMock()
            mock_embedding_model.get_dimension = MagicMock(return_value=3584)
            mock_create.return_value = mock_embedding_model

            provider = LanceDBProvider(lancedb_config)
            await provider.initialize()

            # Should not raise error
            await provider.delete_document("doc_id")

    @pytest.mark.asyncio
    async def test_clear_index(self, lancedb_config, mock_lancedb):
        """Test clearing entire index."""
        mock_connect, mock_db, mock_table = mock_lancedb
        mock_db.list_tables.return_value.tables = ["test_table"]
        mock_db.table_names.return_value = ["test_table"]  # Also set for get_table_names()

        with patch(
            "victor.storage.vector_stores.lancedb_provider.create_embedding_model"
        ) as mock_create:
            mock_embedding_model = AsyncMock()
            mock_embedding_model.initialize = AsyncMock()
            mock_embedding_model.get_dimension = MagicMock(return_value=3584)
            mock_create.return_value = mock_embedding_model

            provider = LanceDBProvider(lancedb_config)
            await provider.initialize()

            await provider.clear_index()

            # Verify table was dropped
            mock_db.drop_table.assert_called_once_with("test_table")
            assert provider.table is None

    @pytest.mark.asyncio
    async def test_get_stats(self, lancedb_config, mock_lancedb):
        """Test getting index statistics."""
        mock_connect, mock_db, mock_table = mock_lancedb
        mock_db.list_tables.return_value.tables = ["test_table"]
        mock_db.table_names.return_value = ["test_table"]  # Also set for get_table_names()
        mock_table.count_rows.return_value = 42

        with patch(
            "victor.storage.vector_stores.lancedb_provider.create_embedding_model"
        ) as mock_create:
            mock_embedding_model = AsyncMock()
            mock_embedding_model.initialize = AsyncMock()
            mock_embedding_model.get_dimension = MagicMock(return_value=3584)
            mock_create.return_value = mock_embedding_model

            provider = LanceDBProvider(lancedb_config)
            await provider.initialize()

            stats = await provider.get_stats()

            assert stats["provider"] == "lancedb"
            assert stats["total_documents"] == 42
            assert stats["embedding_model_type"] == "ollama"
            assert stats["embedding_model_name"] == "gte-Qwen2-7B-instruct"
            assert stats["dimension"] == 3584
            assert stats["distance_metric"] == "cosine"

    @pytest.mark.asyncio
    async def test_get_stats_no_table(self, lancedb_config, mock_lancedb):
        """Test stats when table doesn't exist."""
        mock_connect, mock_db, mock_table = mock_lancedb

        with patch(
            "victor.storage.vector_stores.lancedb_provider.create_embedding_model"
        ) as mock_create:
            mock_embedding_model = AsyncMock()
            mock_embedding_model.initialize = AsyncMock()
            mock_embedding_model.get_dimension = MagicMock(return_value=3584)
            mock_create.return_value = mock_embedding_model

            provider = LanceDBProvider(lancedb_config)
            await provider.initialize()

            stats = await provider.get_stats()

            assert stats["total_documents"] == 0

    @pytest.mark.asyncio
    async def test_close(self, lancedb_config, mock_lancedb):
        """Test cleanup."""
        mock_connect, mock_db, mock_table = mock_lancedb

        with patch(
            "victor.storage.vector_stores.lancedb_provider.create_embedding_model"
        ) as mock_create:
            mock_embedding_model = AsyncMock()
            mock_embedding_model.initialize = AsyncMock()
            mock_embedding_model.get_dimension = MagicMock(return_value=3584)
            mock_embedding_model.close = AsyncMock()
            mock_create.return_value = mock_embedding_model

            provider = LanceDBProvider(lancedb_config)
            await provider.initialize()

            await provider.close()

            mock_embedding_model.close.assert_called_once()
            assert provider.embedding_model is None
            assert provider.db is None
            assert provider.table is None
            assert not provider._initialized


@pytest.fixture
def semantic_store(tmp_path):
    """Real storage, deterministic encoder; no model download or network."""
    from dataclasses import asdict, replace
    from victor_codegraph import BGE_SMALL_PROFILE

    lancedb = pytest.importorskip("lancedb")
    profile = replace(BGE_SMALL_PROFILE, dimension=2)
    provider = LanceDBProvider(EmbeddingConfig(vector_store="lancedb"))
    provider.db = lancedb.connect(str(tmp_path))
    provider._initialized = True
    provider.embedding_model = MagicMock()
    provider.embedding_model.profile = profile
    provider.embedding_model.embed_text = AsyncMock(return_value=[1.0, 0.5])
    document = {
        "id": "canonical-symbol",
        "content": "calculate_tax",
        "vector": [1.0, 0.5],
        "metadata": {
            "embedding_kind": "semantic",
            "embedding_profile_id": profile.identity,
            "embedding_profile": asdict(profile),
        },
    }
    return provider, document, profile


async def test_semantic_table_binding_survives_reopen_and_upsert(semantic_store):
    provider, document, profile = semantic_store
    await provider.index_embedded_documents([document])
    provider.table = provider.db.open_table("embeddings")
    await provider.index_embedded_documents([{**document, "content": "updated_tax"}])
    assert provider.table.count_rows() == 1
    result = await provider.search_similar("tax")
    assert result[0].content == "updated_tax"
    assert result[0].metadata["embedding_profile_id"] == profile.identity
    assert b"victor_codegraph.embedding_profile" in provider.table.schema.metadata


@pytest.mark.parametrize("model_state", ["different_revision", "unqualified"])
async def test_semantic_search_rejects_incompatible_query_before_inference(
    semantic_store, model_state
):
    from dataclasses import replace

    provider, document, profile = semantic_store
    await provider.index_embedded_documents([document])
    provider.embedding_model.profile = (
        replace(profile, revision="1" * 40) if model_state == "different_revision" else None
    )
    with pytest.raises(ValueError, match="query.*profile"):
        await provider.search_similar("tax")
    provider.embedding_model.embed_text.assert_not_awaited()


@pytest.mark.parametrize("entrypoint", ["single", "batch", "precomputed"])
async def test_profiled_collection_rejects_unprofiled_writes(semantic_store, entrypoint):
    provider, document, _ = semantic_store
    await provider.index_embedded_documents([document])
    with pytest.raises(ValueError, match="profile"):
        if entrypoint == "single":
            await provider.index_document("other", "other")
        elif entrypoint == "batch":
            await provider.index_documents([{"id": "other", "content": "other"}])
        else:
            await provider.index_embedded_documents(
                [{"id": "other", "content": "other", "vector": [0.5, 1.0]}]
            )
    assert provider.table.count_rows() == 1


async def test_orphan_row_profile_mismatch_blocks_search_and_update(semantic_store):
    provider, document, _ = semantic_store
    await provider.index_embedded_documents([document])
    # A vector row absent from SQLite is still part of this collection's contract.
    orphan = provider.table.to_arrow().to_pylist()[0]
    orphan.update(id="orphan", embedding_profile_id="0" * 64)
    provider.table.add([orphan])
    with pytest.raises(ValueError, match="mixed.*profile"):
        await provider.search_similar("tax")
    with pytest.raises(ValueError, match="mixed.*profile"):
        await provider.index_embedded_documents([document])
    assert provider.table.count_rows() == 2
    provider.embedding_model.embed_text.assert_not_awaited()


async def test_legacy_collection_requires_fresh_destination(semantic_store):
    provider, document, _ = semantic_store
    provider.table = provider.db.create_table(
        "embeddings", data=[{"id": "legacy", "content": "legacy", "vector": [1.0, 0.5]}]
    )
    with pytest.raises(ValueError, match="fresh.*collection"):
        await provider.index_embedded_documents([document])
    assert provider.table.to_arrow().to_pylist()[0]["id"] == "legacy"


@pytest.mark.parametrize(
    "bad_vector",
    [[0.0, 0.0], [float("nan"), 1.0], [1.0], [1e300, 1.0], [1e-300, 1e-300]],
)
async def test_semantic_query_rejects_failed_inference(semantic_store, bad_vector):
    provider, document, _ = semantic_store
    await provider.index_embedded_documents([document])
    provider.embedding_model.embed_text.return_value = bad_vector
    with pytest.raises(ValueError):
        await provider.search_similar("tax")


async def test_semantic_write_validates_entire_batch_before_mutation(semantic_store):
    provider, document, _ = semantic_store
    await provider.index_embedded_documents([document])
    with pytest.raises(ValueError, match="zero vector"):
        await provider.index_embedded_documents(
            [{**document, "content": "changed"}, {**document, "id": "bad", "vector": [0, 0]}]
        )
    assert provider.table.to_arrow().to_pylist()[0]["content"] == "calculate_tax"


@pytest.mark.parametrize("values", [[1e300, 1.0], [1e-300, 1e-300]])
async def test_semantic_float32_conversion_cannot_publish_invalid_vectors(semantic_store, values):
    provider, document, _ = semantic_store
    with pytest.raises(ValueError):
        await provider.index_embedded_documents([{**document, "vector": values}])
    assert provider.table is None


async def test_empty_profiled_table_retains_binding(semantic_store):
    from dataclasses import asdict, replace

    provider, document, profile = semantic_store
    await provider.index_embedded_documents([document])
    provider.table.delete("true")
    changed = replace(profile, revision="1" * 40)
    document["metadata"].update(
        embedding_profile=asdict(changed), embedding_profile_id=changed.identity
    )
    with pytest.raises(ValueError, match="fresh collection"):
        await provider.index_embedded_documents([document])


async def test_failed_semantic_upsert_preserves_previous_rows(semantic_store):
    provider, document, _ = semantic_store
    await provider.index_embedded_documents([document])
    with patch.object(
        provider.table, "merge_insert", side_effect=RuntimeError("write unavailable")
    ):
        with pytest.raises(RuntimeError, match="write unavailable"):
            await provider.index_embedded_documents([{**document, "content": "changed"}])
    assert provider.table.to_arrow().to_pylist()[0]["content"] == "calculate_tax"
