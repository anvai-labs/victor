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

"""Tests for the shared embedding infrastructure.

Tests cover:
- EmbeddingService singleton behavior
- StaticEmbeddingCollection caching and search
- IntentClassifier continuation/completion detection
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from victor.storage.embeddings.collections import (
    CollectionItem,
    StaticEmbeddingCollection,
)
from victor.storage.embeddings.intent_classifier import (
    COMPLETION_PHRASES,
    CONTINUATION_PHRASES,
    IntentClassifier,
    IntentResult,
    IntentType,
)
from victor.storage.embeddings.service import EmbeddingService
from victor.core.completion_markers import TASK_DONE_MARKER

# ============================================================================
# EmbeddingService Tests
# ============================================================================


class TestEmbeddingServiceSingleton:
    """Tests for EmbeddingService singleton behavior."""

    def setup_method(self):
        """Reset singleton before each test."""
        EmbeddingService.reset_instance()

    def teardown_method(self):
        """Reset singleton after each test."""
        EmbeddingService.reset_instance()

    def test_get_instance_returns_singleton(self):
        """Test that get_instance returns the same instance."""
        instance1 = EmbeddingService.get_instance()
        instance2 = EmbeddingService.get_instance()
        assert instance1 is instance2

    def test_reset_instance_clears_singleton(self):
        """Test that reset_instance clears the singleton."""
        instance1 = EmbeddingService.get_instance()
        EmbeddingService.reset_instance()
        instance2 = EmbeddingService.get_instance()
        assert instance1 is not instance2

    def test_default_model_name(self):
        """Test default model name is set correctly."""
        service = EmbeddingService.get_instance()
        assert service.model_name == "BAAI/bge-small-en-v1.5"

    def test_custom_model_name(self):
        """Test custom model name is preserved."""
        service = EmbeddingService.get_instance(model_name="all-mpnet-base-v2")
        assert service.model_name == "all-mpnet-base-v2"

    def test_is_loaded_initially_false(self):
        """Test that model is not loaded initially."""
        service = EmbeddingService.get_instance()
        assert service.is_loaded is False


class TestEmbeddingServiceEmbeddings:
    """Tests for EmbeddingService embedding generation."""

    def setup_method(self):
        """Reset singleton before each test."""
        EmbeddingService.reset_instance()

    def teardown_method(self):
        """Reset singleton after each test."""
        EmbeddingService.reset_instance()

    @pytest.fixture
    def mock_model(self):
        """Create a mock sentence transformer model."""
        mock = MagicMock()
        mock.encode.return_value = np.random.randn(384).astype(np.float32)
        mock.get_sentence_embedding_dimension.return_value = 384
        mock.device = "cpu"
        return mock

    def test_embed_text_sync_returns_numpy_array(self, mock_model):
        """Test that embed_text_sync returns numpy array."""
        with patch("sentence_transformers.SentenceTransformer", return_value=mock_model):
            service = EmbeddingService.get_instance()
            result = service.embed_text_sync("Hello world")

            assert isinstance(result, np.ndarray)
            assert result.dtype == np.float32
            assert result.shape == (384,)

    def test_embed_batch_sync_returns_2d_array(self, mock_model):
        """Test that embed_batch_sync returns 2D array."""
        mock_model.encode.return_value = np.random.randn(3, 384).astype(np.float32)

        with patch("sentence_transformers.SentenceTransformer", return_value=mock_model):
            service = EmbeddingService.get_instance()
            result = service.embed_batch_sync(["Hello", "World", "Test"])

            assert isinstance(result, np.ndarray)
            assert result.dtype == np.float32
            assert result.shape == (3, 384)

    def test_embed_batch_sync_empty_list_returns_empty_array(self, mock_model):
        """Test that empty list returns empty array."""
        with patch("sentence_transformers.SentenceTransformer", return_value=mock_model):
            service = EmbeddingService.get_instance()
            result = service.embed_batch_sync([])

            assert isinstance(result, np.ndarray)
            assert result.shape[0] == 0

    @pytest.mark.asyncio
    async def test_embed_text_async(self, mock_model):
        """Test async embed_text method."""
        with patch("sentence_transformers.SentenceTransformer", return_value=mock_model):
            service = EmbeddingService.get_instance()
            result = await service.embed_text("Hello world")

            assert isinstance(result, np.ndarray)
            assert result.dtype == np.float32

    def test_dimension_property(self, mock_model):
        """Test dimension property returns correct value."""
        with patch("sentence_transformers.SentenceTransformer", return_value=mock_model):
            service = EmbeddingService.get_instance()
            assert service.dimension == 384


class TestEmbeddingServiceCosineSimilarity:
    """Tests for cosine similarity methods."""

    def test_cosine_similarity_identical_vectors(self):
        """Test cosine similarity of identical vectors is 1."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 3.0])
        result = EmbeddingService.cosine_similarity(a, b)
        assert abs(result - 1.0) < 1e-6

    def test_cosine_similarity_orthogonal_vectors(self):
        """Test cosine similarity of orthogonal vectors is 0."""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        result = EmbeddingService.cosine_similarity(a, b)
        assert abs(result) < 1e-6

    def test_cosine_similarity_opposite_vectors(self):
        """Test cosine similarity of opposite vectors is -1."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([-1.0, -2.0, -3.0])
        result = EmbeddingService.cosine_similarity(a, b)
        assert abs(result + 1.0) < 1e-6

    def test_cosine_similarity_zero_vector_returns_zero(self):
        """Test cosine similarity with zero vector returns 0."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([0.0, 0.0, 0.0])
        result = EmbeddingService.cosine_similarity(a, b)
        assert result == 0.0

    def test_cosine_similarity_matrix(self):
        """Test cosine similarity matrix calculation."""
        query = np.array([1.0, 0.0, 0.0])
        corpus = np.array(
            [
                [1.0, 0.0, 0.0],  # Same as query
                [0.0, 1.0, 0.0],  # Orthogonal
                [0.5, 0.5, 0.0],  # Partially similar
            ]
        )
        results = EmbeddingService.cosine_similarity_matrix(query, corpus)

        assert len(results) == 3
        assert abs(results[0] - 1.0) < 1e-6  # Identical
        assert abs(results[1]) < 1e-6  # Orthogonal


# ============================================================================
# StaticEmbeddingCollection Tests
# ============================================================================


class TestStaticEmbeddingCollection:
    """Tests for StaticEmbeddingCollection."""

    def setup_method(self):
        """Reset singleton and create temp dir."""
        EmbeddingService.reset_instance()
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup."""
        EmbeddingService.reset_instance()

    @pytest.fixture
    def mock_embedding_service(self):
        """Create a mock embedding service."""
        mock = MagicMock(spec=EmbeddingService)
        mock.model_name = "test-model"
        mock.dimension = 384

        # Make embed_batch_sync return deterministic embeddings based on text
        def mock_embed_batch(texts):
            embeddings = []
            for _i, text in enumerate(texts):
                # Create deterministic embedding based on text hash
                np.random.seed(hash(text) % 2**31)
                embeddings.append(np.random.randn(384).astype(np.float32))
            return np.array(embeddings)

        mock.embed_batch_sync.side_effect = mock_embed_batch
        mock.embed_text_sync.side_effect = lambda t: mock_embed_batch([t])[0]

        return mock

    def test_collection_initialization(self, mock_embedding_service):
        """Test collection initializes correctly."""
        collection = StaticEmbeddingCollection(
            name="test",
            cache_dir=Path(self.temp_dir),
            embedding_service=mock_embedding_service,
        )

        items = [
            CollectionItem(id="1", text="Hello world"),
            CollectionItem(id="2", text="Goodbye world"),
        ]
        collection.initialize_sync(items)

        assert collection.is_initialized
        assert collection.size == 2

    def test_collection_search_returns_results(self, mock_embedding_service):
        """Test collection search returns results."""
        collection = StaticEmbeddingCollection(
            name="test",
            cache_dir=Path(self.temp_dir),
            embedding_service=mock_embedding_service,
        )

        items = [
            CollectionItem(id="1", text="Hello world"),
            CollectionItem(id="2", text="Goodbye world"),
            CollectionItem(id="3", text="Testing search"),
        ]
        collection.initialize_sync(items)

        results = collection.search_sync("Hello there", top_k=2)

        assert len(results) <= 2
        assert all(isinstance(item, CollectionItem) for item, _ in results)
        assert all(isinstance(score, float) for _, score in results)

    def test_collection_cache_saves_and_loads(self, mock_embedding_service):
        """Test collection saves to and loads from cache."""
        items = [
            CollectionItem(id="1", text="Hello world"),
            CollectionItem(id="2", text="Goodbye world"),
        ]

        # First collection - should compute embeddings
        collection1 = StaticEmbeddingCollection(
            name="cache_test",
            cache_dir=Path(self.temp_dir),
            embedding_service=mock_embedding_service,
        )
        collection1.initialize_sync(items)

        # Reset mock call count
        mock_embedding_service.embed_batch_sync.reset_mock()

        # Second collection - should load from cache
        collection2 = StaticEmbeddingCollection(
            name="cache_test",
            cache_dir=Path(self.temp_dir),
            embedding_service=mock_embedding_service,
        )
        collection2.initialize_sync(items)

        # Should not have called embed_batch_sync again
        mock_embedding_service.embed_batch_sync.assert_not_called()

    def test_collection_cache_invalidation_on_item_change(self, mock_embedding_service):
        """Test cache is invalidated when items change."""
        # First collection
        collection1 = StaticEmbeddingCollection(
            name="invalidation_test",
            cache_dir=Path(self.temp_dir),
            embedding_service=mock_embedding_service,
        )
        items1 = [CollectionItem(id="1", text="Hello world")]
        collection1.initialize_sync(items1)

        # Reset mock
        mock_embedding_service.embed_batch_sync.reset_mock()

        # Second collection with different items
        collection2 = StaticEmbeddingCollection(
            name="invalidation_test",
            cache_dir=Path(self.temp_dir),
            embedding_service=mock_embedding_service,
        )
        items2 = [CollectionItem(id="1", text="Changed text")]
        collection2.initialize_sync(items2)

        # Should have recomputed embeddings
        mock_embedding_service.embed_batch_sync.assert_called_once()

    def test_get_item_returns_correct_item(self, mock_embedding_service):
        """Test get_item returns correct item by ID."""
        collection = StaticEmbeddingCollection(
            name="test",
            cache_dir=Path(self.temp_dir),
            embedding_service=mock_embedding_service,
        )

        items = [
            CollectionItem(id="foo", text="First item", metadata={"key": "value"}),
            CollectionItem(id="bar", text="Second item"),
        ]
        collection.initialize_sync(items)

        item = collection.get_item("foo")
        assert item is not None
        assert item.id == "foo"
        assert item.text == "First item"
        assert item.metadata == {"key": "value"}

    def test_get_item_returns_none_for_missing(self, mock_embedding_service):
        """Test get_item returns None for missing ID."""
        collection = StaticEmbeddingCollection(
            name="test",
            cache_dir=Path(self.temp_dir),
            embedding_service=mock_embedding_service,
        )
        collection.initialize_sync([])

        item = collection.get_item("nonexistent")
        assert item is None


# ============================================================================
# IntentClassifier Tests
# ============================================================================


class TestIntentClassifier:
    """Tests for IntentClassifier."""

    def setup_method(self):
        """Reset singleton and create temp dir."""
        EmbeddingService.reset_instance()
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup."""
        EmbeddingService.reset_instance()

    @pytest.fixture
    def mock_embedding_service(self):
        """Create a mock embedding service with semantic understanding."""
        mock = MagicMock(spec=EmbeddingService)
        mock.model_name = "test-model"
        mock.dimension = 384

        # Create embeddings that cluster semantically similar texts
        def mock_embed_batch(texts):
            embeddings = []
            for text in texts:
                text_lower = text.lower()

                # Create base embedding based on semantic category
                np.random.seed(42)  # Consistent base
                base = np.random.randn(384).astype(np.float32)

                # Add semantic signal based on content
                if any(kw in text_lower for kw in ["let me", "i'll", "next", "going to"]):
                    # Continuation-like
                    signal = np.array([1.0] * 192 + [0.0] * 192, dtype=np.float32)
                elif any(kw in text_lower for kw in ["summary", "here are", "conclusion", "##"]):
                    # Completion-like
                    signal = np.array([0.0] * 192 + [1.0] * 192, dtype=np.float32)
                else:
                    # Neutral
                    signal = np.zeros(384, dtype=np.float32)

                embedding = base * 0.3 + signal * 0.7
                embedding = embedding / (np.linalg.norm(embedding) + 1e-9)
                embeddings.append(embedding)

            return np.array(embeddings, dtype=np.float32)

        mock.embed_batch_sync.side_effect = mock_embed_batch
        mock.embed_text_sync.side_effect = lambda t: mock_embed_batch([t])[0]

        return mock

    def test_classifier_initialization(self, mock_embedding_service):
        """Test classifier initializes correctly."""
        classifier = IntentClassifier(
            cache_dir=Path(self.temp_dir),
            embedding_service=mock_embedding_service,
        )
        classifier.initialize_sync()

        assert classifier.is_initialized

    def test_classify_continuation_intent(self, mock_embedding_service):
        """Test classifier detects continuation intent."""
        classifier = IntentClassifier(
            cache_dir=Path(self.temp_dir),
            embedding_service=mock_embedding_service,
        )
        classifier.initialize_sync()

        result = classifier.classify_intent_sync("Let me examine the code next")

        assert result.intent == IntentType.CONTINUATION
        assert result.confidence > 0.5

    def test_classify_completion_intent(self, mock_embedding_service):
        """Test classifier detects completion intent."""
        classifier = IntentClassifier(
            cache_dir=Path(self.temp_dir),
            embedding_service=mock_embedding_service,
        )
        classifier.initialize_sync()

        result = classifier.classify_intent_sync("## 1. Summary of findings\nHere are the results")

        assert result.intent == IntentType.COMPLETION
        assert result.confidence > 0.5

    def test_completion_marker_heuristic_uses_rare_victor_marker(self, mock_embedding_service):
        """Explicit rare completion markers should force completion intent."""
        classifier = IntentClassifier(
            cache_dir=Path(self.temp_dir),
            embedding_service=mock_embedding_service,
        )
        classifier.initialize_sync()

        result = classifier.classify_intent_sync(f"{TASK_DONE_MARKER} Fixed parser")

        assert result.intent == IntentType.COMPLETION

    def test_legacy_done_marker_no_longer_forces_completion(self, mock_embedding_service):
        """Generic DONE markers should not trigger the completion heuristic."""
        classifier = IntentClassifier(
            cache_dir=Path(self.temp_dir),
            embedding_service=mock_embedding_service,
            continuation_threshold=0.8,
            completion_threshold=0.8,
        )
        classifier.initialize_sync()

        result = classifier.classify_intent_sync("DONE: Fixed parser")

        assert result.intent != IntentType.COMPLETION

    def test_classify_neutral_intent(self, mock_embedding_service):
        """Test classifier returns neutral for ambiguous text."""
        classifier = IntentClassifier(
            cache_dir=Path(self.temp_dir),
            embedding_service=mock_embedding_service,
            continuation_threshold=0.8,  # High threshold
            completion_threshold=0.8,
        )
        classifier.initialize_sync()

        result = classifier.classify_intent_sync("Just some random text here")

        assert result.intent == IntentType.NEUTRAL

    def test_intends_to_continue_convenience_method(self, mock_embedding_service):
        """Test intends_to_continue convenience method."""
        classifier = IntentClassifier(
            cache_dir=Path(self.temp_dir),
            embedding_service=mock_embedding_service,
        )
        classifier.initialize_sync()

        assert classifier.intends_to_continue("Let me check the implementation")
        assert not classifier.intends_to_continue("Here is the summary of findings")

    def test_is_complete_response_convenience_method(self, mock_embedding_service):
        """Test is_complete_response convenience method."""
        classifier = IntentClassifier(
            cache_dir=Path(self.temp_dir),
            embedding_service=mock_embedding_service,
        )
        classifier.initialize_sync()

        assert classifier.is_complete_response("## Summary\nHere are the key findings")
        assert not classifier.is_complete_response("I'll examine the code next")

    def test_result_contains_top_matches(self, mock_embedding_service):
        """Test result includes top matches for debugging."""
        classifier = IntentClassifier(
            cache_dir=Path(self.temp_dir),
            embedding_service=mock_embedding_service,
        )
        classifier.initialize_sync()

        result = classifier.classify_intent_sync("Let me look at the file")

        assert len(result.top_matches) > 0
        assert all(isinstance(m, tuple) for m in result.top_matches)
        assert all(len(m) == 2 for m in result.top_matches)

    @pytest.mark.asyncio
    async def test_async_classify_intent(self, mock_embedding_service):
        """Test async classify_intent method."""

        # Mock async methods
        async def mock_embed_text_async(text):
            return mock_embedding_service.embed_text_sync(text)

        async def mock_embed_batch_async(texts):
            return mock_embedding_service.embed_batch_sync(texts)

        mock_embedding_service.embed_text = mock_embed_text_async
        mock_embedding_service.embed_batch = mock_embed_batch_async

        classifier = IntentClassifier(
            cache_dir=Path(self.temp_dir),
            embedding_service=mock_embedding_service,
        )
        await classifier.initialize()

        result = await classifier.classify_intent("Let me examine this")

        assert isinstance(result, IntentResult)


class TestIntentPhrases:
    """Tests for the canonical intent phrases."""

    def test_continuation_phrases_not_empty(self):
        """Test continuation phrases list is populated."""
        assert len(CONTINUATION_PHRASES) > 10

    def test_completion_phrases_not_empty(self):
        """Test completion phrases list is populated."""
        assert len(COMPLETION_PHRASES) > 10

    def test_continuation_phrases_are_strings(self):
        """Test all continuation phrases are strings."""
        assert all(isinstance(p, str) for p in CONTINUATION_PHRASES)

    def test_completion_phrases_are_strings(self):
        """Test all completion phrases are strings."""
        assert all(isinstance(p, str) for p in COMPLETION_PHRASES)

    def test_no_duplicate_continuation_phrases(self):
        """Test no duplicate continuation phrases."""
        assert len(CONTINUATION_PHRASES) == len(set(CONTINUATION_PHRASES))

    def test_no_duplicate_completion_phrases(self):
        """Test no duplicate completion phrases."""
        assert len(COMPLETION_PHRASES) == len(set(COMPLETION_PHRASES))


# ============================================================================
# Integration Tests (with real model)
# ============================================================================


@pytest.mark.integration
@pytest.mark.slow
class TestEmbeddingIntegration:
    """Integration tests that use the real embedding model.

    These tests are slower but verify actual embedding quality.
    Run with: pytest -m integration tests/unit/test_embeddings.py
    """

    def setup_method(self):
        """Reset singleton."""
        EmbeddingService.reset_instance()
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup."""
        EmbeddingService.reset_instance()

    def test_real_embedding_similarity(self):
        """Test that similar texts have high cosine similarity."""
        service = EmbeddingService.get_instance()

        text1 = "The cat sat on the mat"
        text2 = "A cat is sitting on a mat"
        text3 = "Python is a programming language"

        emb1 = service.embed_text_sync(text1)
        emb2 = service.embed_text_sync(text2)
        emb3 = service.embed_text_sync(text3)

        # Similar texts should have higher similarity
        sim_12 = EmbeddingService.cosine_similarity(emb1, emb2)
        sim_13 = EmbeddingService.cosine_similarity(emb1, emb3)

        assert sim_12 > sim_13, "Similar texts should have higher similarity"

    def test_real_intent_classification(self):
        """Test intent classification with real model."""
        classifier = IntentClassifier(cache_dir=Path(self.temp_dir))
        classifier.initialize_sync()

        # Test continuation
        result1 = classifier.classify_intent_sync(
            "Let me start by examining the orchestrator.py file"
        )
        assert result1.intent == IntentType.CONTINUATION

        # Test completion
        result2 = classifier.classify_intent_sync(
            "## Summary\n\nHere are the key findings from my analysis:\n\n1. First point"
        )
        assert result2.intent == IntentType.COMPLETION
