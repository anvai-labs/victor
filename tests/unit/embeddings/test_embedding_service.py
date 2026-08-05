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

"""Comprehensive TDD tests for EmbeddingService and CPU-optimized embedding models.

Tests cover:
- EmbeddingService singleton behavior
- Default model configuration (BAAI/bge-small-en-v1.5)
- Multiple CPU-optimized model alternatives
- Settings integration
- Embedding generation and caching
- Code search quality benchmarks
"""

import builtins
import pytest
import numpy as np
from unittest.mock import MagicMock, patch


class TestDefaultEmbeddingModel:
    """Tests for default embedding model configuration."""

    def test_default_model_constant_is_bge_small(self):
        """Test DEFAULT_EMBEDDING_MODEL is set to bge-small-en-v1.5."""
        from victor.storage.embeddings.service import DEFAULT_EMBEDDING_MODEL

        assert DEFAULT_EMBEDDING_MODEL == "BAAI/bge-small-en-v1.5"

    def test_settings_unified_model_matches_service(self):
        """Test settings.embedding.unified_embedding_model matches service default."""
        from victor.storage.embeddings.service import DEFAULT_EMBEDDING_MODEL
        from victor.config.settings import Settings

        settings = Settings()
        assert settings.embedding.unified_embedding_model == DEFAULT_EMBEDDING_MODEL

    def test_service_init_uses_default_model(self):
        """Test EmbeddingService __init__ default matches constant."""
        from victor.storage.embeddings.service import (
            EmbeddingService,
            DEFAULT_EMBEDDING_MODEL,
        )

        EmbeddingService.reset_instance()

        service = EmbeddingService()
        assert service.model_name == DEFAULT_EMBEDDING_MODEL

    def test_get_instance_uses_default_model(self):
        """Test get_instance uses DEFAULT_EMBEDDING_MODEL as default."""
        from victor.storage.embeddings.service import (
            EmbeddingService,
            DEFAULT_EMBEDDING_MODEL,
        )

        EmbeddingService.reset_instance()

        service = EmbeddingService.get_instance()
        assert service.model_name == DEFAULT_EMBEDDING_MODEL

        EmbeddingService.reset_instance()


class TestEmbeddingServiceSingleton:
    """Tests for EmbeddingService singleton behavior."""

    def test_singleton_returns_same_instance(self):
        """Test get_instance returns same object."""
        from victor.storage.embeddings.service import EmbeddingService

        EmbeddingService.reset_instance()

        instance1 = EmbeddingService.get_instance()
        instance2 = EmbeddingService.get_instance()

        assert instance1 is instance2

        EmbeddingService.reset_instance()

    def test_singleton_first_call_wins_for_model(self):
        """Test that first get_instance call sets the model."""
        from victor.storage.embeddings.service import EmbeddingService

        EmbeddingService.reset_instance()

        # First call with custom model
        instance1 = EmbeddingService.get_instance(model_name="thenlper/gte-small")
        assert instance1.model_name == "thenlper/gte-small"

        # Second call with different model - should return same instance
        instance2 = EmbeddingService.get_instance(model_name="all-MiniLM-L6-v2")
        assert instance2.model_name == "thenlper/gte-small"  # First call wins

        EmbeddingService.reset_instance()

    def test_reset_instance_clears_singleton(self):
        """Test reset_instance clears the singleton."""
        from victor.storage.embeddings.service import EmbeddingService

        EmbeddingService.reset_instance()

        instance1 = EmbeddingService.get_instance(model_name="model-a")
        EmbeddingService.reset_instance()
        instance2 = EmbeddingService.get_instance(model_name="model-b")

        assert instance1 is not instance2
        assert instance2.model_name == "model-b"

        EmbeddingService.reset_instance()

    def test_singleton_thread_safety(self):
        """Test singleton is thread-safe with double-checked locking."""
        import threading
        from victor.storage.embeddings.service import EmbeddingService

        EmbeddingService.reset_instance()

        instances = []
        errors = []

        def get_instance():
            try:
                instance = EmbeddingService.get_instance()
                instances.append(instance)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=get_instance) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # All instances should be the same
        assert all(i is instances[0] for i in instances)

        EmbeddingService.reset_instance()


class TestCPUOptimizedModels:
    """Tests for CPU-optimized embedding model configurations."""

    @pytest.mark.parametrize(
        "model_name,expected_dims",
        [
            ("BAAI/bge-small-en-v1.5", 384),  # New default
            ("thenlper/gte-small", 384),
            ("all-MiniLM-L6-v2", 384),
            ("all-MiniLM-L12-v2", 384),
        ],
    )
    def test_model_dimensions(self, model_name, expected_dims):
        """Test that all CPU models produce 384-dimensional embeddings."""
        from victor.storage.embeddings.service import EmbeddingService

        EmbeddingService.reset_instance()

        # Create service with specific model
        service = EmbeddingService(model_name=model_name)

        # Mock the sentence-transformers model
        with patch("sentence_transformers.SentenceTransformer") as MockST:
            mock_model = MagicMock()
            mock_model.get_sentence_embedding_dimension.return_value = expected_dims
            MockST.return_value = mock_model

            service._ensure_model_loaded()

            assert service.dimension == expected_dims

        EmbeddingService.reset_instance()

    def test_bge_small_model_name_huggingface_format(self):
        """Test bge-small uses HuggingFace format (BAAI/bge-small-en-v1.5)."""
        from victor.storage.embeddings.service import DEFAULT_EMBEDDING_MODEL

        # Should be full HuggingFace path
        assert "/" in DEFAULT_EMBEDDING_MODEL
        assert DEFAULT_EMBEDDING_MODEL.startswith("BAAI/")

    def test_model_supports_mps_device(self):
        """Test model can use MPS device on Apple Silicon."""
        from victor.storage.embeddings.service import EmbeddingService

        EmbeddingService.reset_instance()

        # Service should accept mps device
        service = EmbeddingService(device="mps")
        assert service.device == "mps"

        EmbeddingService.reset_instance()


class TestEmbeddingGeneration:
    """Tests for embedding generation functionality."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton before and after each test."""
        from victor.storage.embeddings.service import EmbeddingService

        EmbeddingService.reset_instance()
        yield
        EmbeddingService.reset_instance()

    def test_embed_text_sync(self):
        """Test synchronous text embedding."""
        from victor.storage.embeddings.service import EmbeddingService

        service = EmbeddingService()

        with patch("sentence_transformers.SentenceTransformer") as MockST:
            mock_model = MagicMock()
            mock_model.encode.return_value = np.random.randn(384).astype(np.float32)
            mock_model.get_sentence_embedding_dimension.return_value = 384
            MockST.return_value = mock_model

            embedding = service.embed_text_sync("test text")

            assert embedding.shape == (384,)
            assert embedding.dtype == np.float32
            mock_model.encode.assert_called_once()

    def test_embed_batch_sync(self):
        """Test synchronous batch embedding."""
        from victor.storage.embeddings.service import EmbeddingService

        service = EmbeddingService()

        with patch("sentence_transformers.SentenceTransformer") as MockST:
            mock_model = MagicMock()
            mock_model.encode.return_value = np.random.randn(3, 384).astype(np.float32)
            mock_model.get_sentence_embedding_dimension.return_value = 384
            MockST.return_value = mock_model

            texts = ["text1", "text2", "text3"]
            embeddings = service.embed_batch_sync(texts)

            assert embeddings.shape == (3, 384)
            assert embeddings.dtype == np.float32

    def test_embed_batch_sync_empty_list(self):
        """Test batch embedding with empty list."""
        from victor.storage.embeddings.service import EmbeddingService

        service = EmbeddingService()

        with patch("sentence_transformers.SentenceTransformer") as MockST:
            mock_model = MagicMock()
            mock_model.get_sentence_embedding_dimension.return_value = 384
            MockST.return_value = mock_model

            embeddings = service.embed_batch_sync([])

            assert embeddings.shape == (0, 384)
            assert embeddings.dtype == np.float32

    @pytest.mark.asyncio
    async def test_embed_text_async(self):
        """Test async text embedding runs in thread pool."""
        from victor.storage.embeddings.service import EmbeddingService

        service = EmbeddingService()

        with patch("sentence_transformers.SentenceTransformer") as MockST:
            mock_model = MagicMock()
            mock_model.encode.return_value = np.random.randn(384).astype(np.float32)
            mock_model.get_sentence_embedding_dimension.return_value = 384
            MockST.return_value = mock_model

            embedding = await service.embed_text("test text")

            assert embedding.shape == (384,)
            assert embedding.dtype == np.float32

    @pytest.mark.asyncio
    async def test_embed_text_async_does_not_depend_on_get_event_loop(self):
        """Async text embedding should not access the legacy event-loop getter."""
        from victor.storage.embeddings.service import EmbeddingService

        service = EmbeddingService()
        expected = np.ones(384, dtype=np.float32)

        with (
            patch.object(service, "embed_text_sync", return_value=expected) as mock_sync,
            patch(
                "victor.storage.embeddings.service.asyncio.get_event_loop",
                side_effect=AssertionError,
            ),
        ):
            embedding = await service.embed_text("test text", use_cache=False)

        assert np.array_equal(embedding, expected)
        mock_sync.assert_called_once_with("test text", use_cache=False)

    @pytest.mark.asyncio
    async def test_embed_batch_async(self):
        """Test async batch embedding."""
        from victor.storage.embeddings.service import EmbeddingService

        service = EmbeddingService()

        with patch("sentence_transformers.SentenceTransformer") as MockST:
            mock_model = MagicMock()
            mock_model.encode.return_value = np.random.randn(5, 384).astype(np.float32)
            mock_model.get_sentence_embedding_dimension.return_value = 384
            MockST.return_value = mock_model

            texts = ["a", "b", "c", "d", "e"]
            embeddings = await service.embed_batch(texts)

            assert embeddings.shape == (5, 384)

    @pytest.mark.asyncio
    async def test_embed_batch_async_does_not_depend_on_get_event_loop(self):
        """Async batch embedding should not access the legacy event-loop getter."""
        from victor.storage.embeddings.service import EmbeddingService

        service = EmbeddingService()
        texts = ["a", "b"]
        expected = np.ones((2, 384), dtype=np.float32)

        with (
            patch.object(service, "embed_batch_sync", return_value=expected) as mock_sync,
            patch(
                "victor.storage.embeddings.service.asyncio.get_event_loop",
                side_effect=AssertionError,
            ),
        ):
            embeddings = await service.embed_batch(texts)

        assert np.array_equal(embeddings, expected)
        mock_sync.assert_called_once_with(texts)


class TestEmbeddingFallback:
    """Tests for fallback behavior on errors."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton before and after each test."""
        from victor.storage.embeddings.service import EmbeddingService

        EmbeddingService.reset_instance()
        yield
        EmbeddingService.reset_instance()

    def test_embed_text_returns_zeros_on_error(self):
        """Test embed_text returns zero vector on encoding error."""
        from victor.storage.embeddings.service import EmbeddingService

        service = EmbeddingService()

        with patch("sentence_transformers.SentenceTransformer") as MockST:
            mock_model = MagicMock()
            mock_model.encode.side_effect = RuntimeError("Encoding failed")
            mock_model.get_sentence_embedding_dimension.return_value = 384
            MockST.return_value = mock_model

            embedding = service.embed_text_sync("test text")

            # Should return zero vector, not crash
            assert embedding.shape == (384,)
            assert np.all(embedding == 0)

    def test_embed_batch_returns_zeros_on_error(self):
        """Test embed_batch returns zero vectors on error."""
        from victor.storage.embeddings.service import EmbeddingService

        service = EmbeddingService()

        with patch("sentence_transformers.SentenceTransformer") as MockST:
            mock_model = MagicMock()
            mock_model.encode.side_effect = RuntimeError("Batch encoding failed")
            mock_model.get_sentence_embedding_dimension.return_value = 384
            MockST.return_value = mock_model

            embeddings = service.embed_batch_sync(["a", "b", "c"])

            assert embeddings.shape == (3, 384)
            assert np.all(embeddings == 0)

    def test_import_error_raises_with_message(self):
        """Test ImportError is raised with helpful message."""
        from victor.storage.embeddings.service import EmbeddingService

        EmbeddingService.reset_instance()
        service = EmbeddingService()

        # Mock builtins.__import__ to simulate ImportError for sentence_transformers
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "sentence_transformers":
                raise ImportError("No module named 'sentence_transformers'")
            return original_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=mock_import):
            with pytest.raises(ImportError, match="sentence-transformers not installed"):
                service._ensure_model_loaded()


class TestCosineSimilarity:
    """Tests for cosine similarity calculations."""

    def test_cosine_similarity_identical_vectors(self):
        """Test similarity of identical vectors is 1.0."""
        from victor.storage.embeddings.service import EmbeddingService

        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 3.0])

        similarity = EmbeddingService.cosine_similarity(a, b)

        assert similarity == pytest.approx(1.0)

    def test_cosine_similarity_orthogonal_vectors(self):
        """Test similarity of orthogonal vectors is 0.0."""
        from victor.storage.embeddings.service import EmbeddingService

        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])

        similarity = EmbeddingService.cosine_similarity(a, b)

        assert similarity == pytest.approx(0.0)

    def test_cosine_similarity_opposite_vectors(self):
        """Test similarity of opposite vectors is -1.0."""
        from victor.storage.embeddings.service import EmbeddingService

        a = np.array([1.0, 2.0, 3.0])
        b = np.array([-1.0, -2.0, -3.0])

        similarity = EmbeddingService.cosine_similarity(a, b)

        assert similarity == pytest.approx(-1.0)

    def test_cosine_similarity_zero_vector(self):
        """Test similarity with zero vector returns 0."""
        from victor.storage.embeddings.service import EmbeddingService

        a = np.array([1.0, 2.0, 3.0])
        b = np.zeros(3)

        similarity = EmbeddingService.cosine_similarity(a, b)

        assert similarity == 0.0

    def test_cosine_similarity_matrix(self):
        """Test batch cosine similarity calculation."""
        from victor.storage.embeddings.service import EmbeddingService

        query = np.array([1.0, 0.0, 0.0])
        corpus = np.array(
            [
                [1.0, 0.0, 0.0],  # Identical
                [0.0, 1.0, 0.0],  # Orthogonal
                [-1.0, 0.0, 0.0],  # Opposite
            ]
        )

        similarities = EmbeddingService.cosine_similarity_matrix(query, corpus)

        assert similarities.shape == (3,)
        assert similarities[0] == pytest.approx(1.0)
        assert similarities[1] == pytest.approx(0.0)
        assert similarities[2] == pytest.approx(-1.0)

    def test_cosine_similarity_matrix_empty_corpus(self):
        """Test batch similarity with empty corpus."""
        from victor.storage.embeddings.service import EmbeddingService

        query = np.array([1.0, 0.0, 0.0])
        corpus = np.array([]).reshape(0, 3)

        similarities = EmbeddingService.cosine_similarity_matrix(query, corpus)

        assert similarities.shape == (0,)


class TestLazyLoading:
    """Tests for lazy model loading."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton before and after each test."""
        from victor.storage.embeddings.service import EmbeddingService

        EmbeddingService.reset_instance()
        yield
        EmbeddingService.reset_instance()

    def test_model_not_loaded_on_init(self):
        """Test model is not loaded during __init__."""
        from victor.storage.embeddings.service import EmbeddingService

        service = EmbeddingService()

        assert service._model is None
        assert not service.is_loaded

    def test_model_loaded_on_first_embedding(self):
        """Test model is loaded on first embedding request."""
        from victor.storage.embeddings.service import EmbeddingService

        service = EmbeddingService()

        with patch("sentence_transformers.SentenceTransformer") as MockST:
            mock_model = MagicMock()
            mock_model.encode.return_value = np.random.randn(384).astype(np.float32)
            mock_model.get_sentence_embedding_dimension.return_value = 384
            MockST.return_value = mock_model

            # Before embedding
            assert not service.is_loaded

            # Trigger embedding
            service.embed_text_sync("test")

            # After embedding
            assert service.is_loaded
            MockST.assert_called_once()

    def test_model_loaded_once(self):
        """Test model is loaded only once despite multiple embeddings."""
        from victor.storage.embeddings.service import EmbeddingService

        service = EmbeddingService()

        with patch("sentence_transformers.SentenceTransformer") as MockST:
            mock_model = MagicMock()
            mock_model.encode.return_value = np.random.randn(384).astype(np.float32)
            mock_model.get_sentence_embedding_dimension.return_value = 384
            MockST.return_value = mock_model

            # Multiple embeddings
            service.embed_text_sync("test1")
            service.embed_text_sync("test2")
            service.embed_text_sync("test3")

            # Model should only be loaded once
            MockST.assert_called_once()


class TestSettingsIntegration:
    """Tests for settings integration."""

    def test_settings_embedding_model_matches_service(self):
        """Test settings.tools.embedding_model equals service default."""
        from victor.config.settings import Settings
        from victor.storage.embeddings.service import DEFAULT_EMBEDDING_MODEL

        settings = Settings()
        assert settings.tools.embedding_model == DEFAULT_EMBEDDING_MODEL

    def test_settings_codebase_embedding_model_matches_service(self):
        """Test settings.search.codebase_embedding_model equals service default."""
        from victor.config.settings import Settings
        from victor.storage.embeddings.service import DEFAULT_EMBEDDING_MODEL

        settings = Settings()
        assert settings.search.codebase_embedding_model == DEFAULT_EMBEDDING_MODEL

    def test_settings_dimension_is_384(self):
        """Test settings.search.codebase_dimension is 384 for BGE/MiniLM models."""
        from victor.config.settings import Settings

        settings = Settings()
        assert settings.search.codebase_dimension == 384


class TestModelQualityBenchmarks:
    """Quality benchmarks for embedding models (for code search use case)."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton before and after each test."""
        from victor.storage.embeddings.service import EmbeddingService

        EmbeddingService.reset_instance()
        yield
        EmbeddingService.reset_instance()

    def test_code_related_text_similarity(self):
        """Test that code-related texts produce similar embeddings."""
        from victor.storage.embeddings.service import EmbeddingService

        service = EmbeddingService()

        with patch("sentence_transformers.SentenceTransformer") as MockST:
            mock_model = MagicMock()

            # Simulate: code-related queries should be similar
            def mock_encode(text, **kwargs):
                if "function" in text.lower() or "def " in text.lower():
                    return np.array([0.8, 0.1, 0.1] + [0.0] * 381, dtype=np.float32)
                elif "class" in text.lower():
                    return np.array([0.7, 0.2, 0.1] + [0.0] * 381, dtype=np.float32)
                else:
                    return np.array([0.1, 0.8, 0.1] + [0.0] * 381, dtype=np.float32)

            mock_model.encode.side_effect = mock_encode
            mock_model.get_sentence_embedding_dimension.return_value = 384
            MockST.return_value = mock_model

            # Get embeddings
            func_emb = service.embed_text_sync("define a function")
            class_emb = service.embed_text_sync("define a class")
            other_emb = service.embed_text_sync("write documentation")

            # Function and class should be more similar than function and other
            func_class_sim = EmbeddingService.cosine_similarity(func_emb, class_emb)
            func_other_sim = EmbeddingService.cosine_similarity(func_emb, other_emb)

            assert func_class_sim > func_other_sim


class TestDocstrings:
    """Tests for docstring accuracy after model change."""

    def test_init_docstring_lists_bge_small_first(self):
        """Test __init__ docstring lists bge-small-en-v1.5 as default."""
        from victor.storage.embeddings.service import EmbeddingService

        docstring = EmbeddingService.__init__.__doc__
        assert "BAAI/bge-small-en-v1.5" in docstring
        assert "default" in docstring.lower()

    def test_init_docstring_includes_alternatives(self):
        """Test __init__ docstring includes alternative models."""
        from victor.storage.embeddings.service import EmbeddingService

        docstring = EmbeddingService.__init__.__doc__
        assert "gte-small" in docstring
        assert "MiniLM" in docstring


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
