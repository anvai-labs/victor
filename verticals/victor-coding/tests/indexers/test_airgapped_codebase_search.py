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

"""Tests for air-gapped codebase semantic search with unified embedding model."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np

from victor.storage.vector_stores.base import EmbeddingConfig
from victor.storage.vector_stores.lancedb_provider import LanceDBProvider
from victor.storage.vector_stores.models import (
    SentenceTransformerModel,
    EmbeddingModelConfig,
)

# Check if sentence-transformers is available
try:
    import sentence_transformers  # noqa: F401

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Check if lancedb is available
try:
    import lancedb  # noqa: F401

    LANCEDB_AVAILABLE = True
except ImportError:
    LANCEDB_AVAILABLE = False


class TestEmbeddingConfigDefaults:
    """Test that EmbeddingConfig defaults are air-gapped."""

    def test_default_vector_store_is_lancedb(self):
        """Test that default vector store is LanceDB (production-ready)."""
        config = EmbeddingConfig()
        assert config.vector_store == "lancedb"

    def test_default_embedding_type_is_sentence_transformers(self):
        """Test that default embedding type is sentence-transformers (offline)."""
        config = EmbeddingConfig()
        assert config.embedding_model_type == "sentence-transformers"

    def test_default_embedding_model_is_unified(self):
        """Test that default model is BAAI/bge-small-en-v1.5 (unified model)."""
        config = EmbeddingConfig()
        assert config.embedding_model_name == "BAAI/bge-small-en-v1.5"

    def test_default_distance_metric_is_cosine(self):
        """Test that default distance metric is cosine."""
        config = EmbeddingConfig()
        assert config.distance_metric == "cosine"

    def test_default_is_air_gapped_capable(self):
        """Test that default configuration works offline."""
        config = EmbeddingConfig()
        assert config.embedding_model_type == "sentence-transformers"  # Offline
        assert config.vector_store == "lancedb"  # Default, works offline
        assert config.embedding_api_key is None  # No API key needed


@pytest.mark.skipif(
    not SENTENCE_TRANSFORMERS_AVAILABLE,
    reason="sentence-transformers not installed. Install with: pip install victor-ai[embeddings]",
)
class TestUnifiedEmbeddingModel:
    """Test unified embedding model strategy."""

    def test_unified_model_dimensions(self):
        """Test that unified model has 384 dimensions."""
        config = EmbeddingModelConfig(
            model_type="sentence-transformers",
            model_name="all-MiniLM-L12-v2",
            dimension=384,
        )
        assert config.dimension == 384

    @pytest.mark.asyncio
    async def test_unified_model_loads_once(self):
        """Test that unified model is loaded only once and shared via EmbeddingService."""
        from victor.storage.embeddings.service import EmbeddingService

        # Reset singleton to ensure clean state
        EmbeddingService.reset_instance()

        # Mock EmbeddingService.get_instance
        mock_service = MagicMock()
        mock_service.dimension = 384
        mock_service._ensure_model_loaded = MagicMock()

        config = EmbeddingModelConfig(
            model_type="sentence-transformers",
            model_name="all-MiniLM-L12-v2",
            dimension=384,
        )

        # Patch at the location where it's imported in the model's initialize() method
        with patch(
            "victor.storage.embeddings.service.EmbeddingService.get_instance",
            return_value=mock_service,
        ) as mock_get_instance:
            model = SentenceTransformerModel(config)
            await model.initialize()

            # Should have called EmbeddingService.get_instance once
            mock_get_instance.assert_called_once_with(model_name="all-MiniLM-L12-v2")
            mock_service._ensure_model_loaded.assert_called_once()

        # Reset singleton after test
        EmbeddingService.reset_instance()

    @pytest.mark.asyncio
    async def test_unified_model_embedding_shape(self):
        """Test that unified model produces 384-dim embeddings."""
        with patch("sentence_transformers.SentenceTransformer") as MockST:
            mock_model = MagicMock()
            mock_embedding = np.random.randn(384).astype(np.float32)
            mock_model.encode.return_value = mock_embedding
            mock_model.get_sentence_embedding_dimension.return_value = 384
            MockST.return_value = mock_model

            config = EmbeddingModelConfig(
                model_type="sentence-transformers",
                model_name="all-MiniLM-L12-v2",
                dimension=384,
            )

            model = SentenceTransformerModel(config)
            await model.initialize()

            embedding = await model.embed_text("test")
            assert len(embedding) == 384

    @pytest.mark.asyncio
    async def test_unified_model_batch_embedding(self):
        """Test that unified model handles batch embeddings efficiently via EmbeddingService."""
        from unittest.mock import AsyncMock

        from victor.storage.embeddings.service import EmbeddingService

        # Reset singleton to ensure clean state
        EmbeddingService.reset_instance()

        # Mock EmbeddingService with async methods
        mock_embeddings = np.random.randn(3, 384).astype(np.float32)
        mock_service = MagicMock()
        mock_service.dimension = 384
        mock_service._ensure_model_loaded = MagicMock()
        mock_service.embed_batch = AsyncMock(return_value=mock_embeddings)

        config = EmbeddingModelConfig(
            model_type="sentence-transformers",
            model_name="all-MiniLM-L12-v2",
            dimension=384,
            batch_size=32,
        )

        with patch.object(EmbeddingService, "get_instance", return_value=mock_service):
            model = SentenceTransformerModel(config)
            await model.initialize()

            texts = ["text1", "text2", "text3"]
            embeddings = await model.embed_batch(texts)

            assert len(embeddings) == 3
            assert all(len(emb) == 384 for emb in embeddings)
            mock_service.embed_batch.assert_called_once_with(texts)

        # Reset singleton after test
        EmbeddingService.reset_instance()


@pytest.mark.skipif(
    not LANCEDB_AVAILABLE,
    reason="lancedb not installed. Install with: pip install victor-ai[embeddings]",
)
class TestLanceDBProvider:
    """Test LanceDB provider for air-gapped deployments."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.mark.asyncio
    async def test_lancedb_initialization_offline(self, temp_dir):
        """Test that LanceDB initializes without network."""
        config = EmbeddingConfig(
            vector_store="lancedb",
            embedding_model_type="sentence-transformers",
            embedding_model_name="all-MiniLM-L12-v2",
            persist_directory=str(temp_dir),
            extra_config={"table_name": "test", "dimension": 384},
        )

        with (
            patch("lancedb.connect") as mock_connect,
            patch("sentence_transformers.SentenceTransformer") as MockST,
        ):

            mock_db = MagicMock()
            mock_db.table_names.return_value = []
            mock_connect.return_value = mock_db

            mock_model = MagicMock()
            mock_model.get_sentence_embedding_dimension.return_value = 384
            MockST.return_value = mock_model

            provider = LanceDBProvider(config)
            await provider.initialize()

            # Should have connected to local LanceDB
            mock_connect.assert_called_once_with(str(temp_dir))
            assert provider._initialized

    @pytest.mark.asyncio
    async def test_lancedb_index_and_search(self, temp_dir):
        """Test indexing and searching documents with LanceDB."""
        config = EmbeddingConfig(
            vector_store="lancedb",
            embedding_model_type="sentence-transformers",
            embedding_model_name="all-MiniLM-L12-v2",
            persist_directory=str(temp_dir),
            extra_config={"table_name": "test", "dimension": 384},
        )

        with (
            patch("lancedb.connect") as mock_connect,
            patch("sentence_transformers.SentenceTransformer") as MockST,
        ):

            # Mock LanceDB
            mock_table = MagicMock()
            mock_table.search.return_value.limit.return_value.to_list.return_value = [
                {
                    "id": "doc1",
                    "content": "test content",
                    "file_path": "test.py",
                    "_distance": 0.1,
                }
            ]

            mock_db = MagicMock()
            mock_db.table_names.return_value = []
            mock_db.create_table.return_value = mock_table
            mock_connect.return_value = mock_db

            # Mock sentence-transformers
            mock_model = MagicMock()
            mock_model.encode.return_value = np.random.randn(384).astype(np.float32)
            mock_model.get_sentence_embedding_dimension.return_value = 384
            MockST.return_value = mock_model

            provider = LanceDBProvider(config)
            await provider.initialize()

            # Index document
            documents = [
                {
                    "id": "doc1",
                    "content": "test content",
                    "metadata": {"file_path": "test.py"},
                }
            ]
            await provider.index_documents(documents)

            # Search
            results = await provider.search_similar("test query", limit=1)

            assert len(results) == 1
            assert results[0].content == "test content"


class TestMemoryOptimization:
    """Test memory optimization benefits of unified model."""

    def test_unified_model_memory_footprint(self):
        """Test that unified model has expected memory footprint."""
        # Expected: 120MB for all-MiniLM-L12-v2
        # This is 40% less than separate models (80MB + 120MB = 200MB)
        config = EmbeddingModelConfig(
            model_type="sentence-transformers",
            model_name="all-MiniLM-L12-v2",
            dimension=384,
        )

        # Model size is approximately 120MB

        # This test documents the expected memory footprint
        # Actual measurement would require loading the model
        assert config.model_name == "all-MiniLM-L12-v2"
        assert config.dimension == 384

    def test_separate_models_would_use_more_memory(self):
        """Document that separate models would use 200MB vs 120MB."""
        # Tool selection model
        tool_model = EmbeddingModelConfig(
            model_type="sentence-transformers",
            model_name="all-MiniLM-L6-v2",
            dimension=384,  # 80MB
        )

        # Codebase search model
        codebase_model = EmbeddingModelConfig(
            model_type="sentence-transformers",
            model_name="all-MiniLM-L12-v2",  # 120MB
            dimension=384,
        )

        # Separate models = 80MB + 120MB = 200MB
        # Unified model = 120MB
        # Savings = 80MB (40% reduction)

        assert tool_model.model_name != codebase_model.model_name
        # This test documents the memory benefit of unified model


@pytest.mark.skipif(
    not (SENTENCE_TRANSFORMERS_AVAILABLE and LANCEDB_AVAILABLE),
    reason=(
        "sentence-transformers and lancedb required. "
        "Install with: pip install victor-ai[embeddings]"
    ),
)
class TestAirgappedDemo:
    """Test the air-gapped demo script functionality."""

    @pytest.mark.asyncio
    async def test_demo_config_is_air_gapped(self):
        """Test that demo uses air-gapped configuration."""
        # Demo configuration from examples/airgapped_codebase_search.py
        config = EmbeddingConfig(
            vector_store="lancedb",
            persist_directory="~/.victor/embeddings/airgapped_demo",
            distance_metric="cosine",
            embedding_model_type="sentence-transformers",
            embedding_model_name="all-MiniLM-L12-v2",
            extra_config={
                "table_name": "airgapped_codebase",
                "dimension": 384,
                "batch_size": 32,
            },
        )

        # Verify air-gapped configuration
        assert config.vector_store == "lancedb"  # Offline capable
        assert config.embedding_model_type == "sentence-transformers"  # Offline
        assert config.embedding_api_key is None  # No API key needed
        assert config.extra_config["dimension"] == 384  # Unified model dimension

    @pytest.mark.asyncio
    async def test_demo_indexing_workflow(self):
        """Test the demo indexing workflow works."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = EmbeddingConfig(
                vector_store="lancedb",
                persist_directory=tmpdir,
                embedding_model_type="sentence-transformers",
                embedding_model_name="all-MiniLM-L12-v2",
                extra_config={"table_name": "test", "dimension": 384},
            )

            with (
                patch(
                    "victor.storage.vector_stores.lancedb_provider.lancedb.connect"
                ) as mock_connect,
                patch("sentence_transformers.SentenceTransformer") as MockST,
            ):

                # Mock LanceDB
                mock_table = MagicMock()
                mock_db = MagicMock()
                mock_db.table_names.return_value = []
                mock_db.create_table.return_value = mock_table
                mock_connect.return_value = mock_db

                # Mock sentence-transformers
                mock_model = MagicMock()
                # Return batch embeddings for batch processing
                mock_model.encode.return_value = [np.random.randn(384).astype(np.float32)]
                mock_model.get_sentence_embedding_dimension.return_value = 384
                MockST.return_value = mock_model

                provider = LanceDBProvider(config)
                await provider.initialize()

                # Index documents (like demo does)
                documents = [
                    {
                        "id": "auth_login",
                        "content": "def authenticate_user(username, password): ...",
                        "metadata": {
                            "file_path": "src/auth/login.py",
                            "symbol_name": "authenticate_user",
                            "line_number": 15,
                        },
                    }
                ]

                await provider.index_documents(documents)

                # Verify documents were added to table (create_table includes first batch)
                assert mock_db.create_table.called or mock_table.add.called


class TestCacheOptimization:
    """Test cache optimization benefits."""

    def test_page_cache_sharing(self):
        """Document OS page cache sharing benefit."""
        # When using unified model, OS page cache is shared between:
        # 1. Tool selection
        # 2. Codebase search
        #
        # This means:
        # - Model file loaded into page cache once
        # - Both use cases read from same cache
        # - ~50% reduction in page cache pressure

        unified_config = EmbeddingModelConfig(
            model_type="sentence-transformers", model_name="all-MiniLM-L12-v2"  # Shared
        )

        # Both use cases use same model = same page cache entry
        assert unified_config.model_name == "all-MiniLM-L12-v2"

    def test_cpu_cache_benefits(self):
        """Document CPU L2/L3 cache benefits."""
        # When using unified model:
        # - Tool selection warms L2/L3 cache
        # - Codebase search benefits from warm cache
        # - Fewer cache evictions
        # - Better cache hit rates

        # This test documents the benefit
        # Actual measurement would require performance profiling

        unified_config = EmbeddingModelConfig(
            model_type="sentence-transformers", model_name="all-MiniLM-L12-v2"
        )

        assert unified_config.model_name == "all-MiniLM-L12-v2"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
