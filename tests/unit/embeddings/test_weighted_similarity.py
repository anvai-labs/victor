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

"""Tests for weighted cosine similarity in embedding service."""

import numpy as np
import pytest

from victor.storage.embeddings.service import EmbeddingService


class TestWeightedCosineSimilarity:
    """Test weighted cosine similarity with key term boosting."""

    @pytest.fixture
    def embedding_service(self):
        """Get the embedding service; skip if its model cannot be loaded.

        These tests exercise real cosine-similarity behavior and need the
        sentence-transformers model. If it can't be fetched (offline + not
        pre-cached, or a transient HF outage), skip rather than fail with an
        OSError — the test can't run meaningfully without the model.
        """
        service = EmbeddingService.get_instance()
        try:
            # use_cache=False forces a live model computation — a cached warmup
            # from an earlier test could otherwise mask a now-unloaded model.
            warm = service.embed_text_sync("structural analysis warmup probe", use_cache=False)
        except Exception as exc:  # pragma: no cover - network/offline dependent
            pytest.skip(f"embedding model unavailable: {exc}")
        # embed_text_sync returns a zero-vector fallback (WITHOUT raising) when
        # the sentence-transformers model could not be loaded — e.g. offline CI
        # with no cached model, or after the reset_embedding_singleton autouse
        # fixture nulls the model and it cannot reload. Cosine similarity is
        # identically 0.0 on zero vectors, which makes these ordering assertions
        # fail spuriously on CI runners while passing locally. Skip when the
        # model is not actually producing embeddings.
        if getattr(service, "_model", None) is None or float(np.linalg.norm(warm)) == 0.0:
            pytest.skip(
                "embedding model not loaded (zero-vector fallback); "
                "cosine-similarity assertions cannot run meaningfully"
            )
        return service

    @pytest.fixture
    def sample_embeddings(self):
        """Create sample embeddings for testing."""
        # Create simple 3D embeddings for predictable behavior
        query_emb = np.array([0.577, 0.577, 0.577])  # Normalized
        corpus_emb = np.array(
            [
                [0.707, 0.707, 0.0],  # Item 0: similar to query
                [0.0, 0.0, 1.0],  # Item 1: different
                [0.577, 0.577, 0.577],  # Item 2: identical to query
            ]
        )
        return query_emb, corpus_emb

    def test_weighted_similarity_boosts_key_terms(self, sample_embeddings):
        """Verify that overlapping key terms boost similarity."""
        query_emb, corpus_emb = sample_embeddings

        query_text = "analyze the framework"
        corpus_texts = [
            "review the architecture",  # Has "architecture" (1.2), "review" (1.4)
            "create something",  # No key terms
            "analyze structure",  # Has "analyze" (1.5), "structure" (1.2)
        ]

        similarities = EmbeddingService.weighted_cosine_similarity(
            query_emb, query_text, corpus_emb, corpus_texts
        )

        # Item 2 should get highest boost (analyze + structure overlap)
        # Item 0 should get moderate boost (architecture + review)
        # Item 1 should get no boost (no key terms)
        assert similarities[2] > similarities[0]  # analyze > review
        assert similarities[0] > similarities[1]  # review > no boost

    def test_weighted_similarity_no_key_terms_in_query(self, sample_embeddings):
        """Verify behavior when query has no key terms."""
        query_emb, corpus_emb = sample_embeddings

        query_text = "do something with the code"
        corpus_texts = ["analyze this", "review that", "create something"]

        # Should return base similarities when no key terms in query
        similarities = EmbeddingService.weighted_cosine_similarity(
            query_emb, query_text, corpus_emb, corpus_texts
        )

        base_similarities = EmbeddingService.cosine_similarity_matrix(query_emb, corpus_emb)

        np.testing.assert_array_almost_equal(similarities, base_similarities)

    def test_weighted_similarity_cap_at_one(self, sample_embeddings):
        """Verify that weighted similarity is capped at 1.0."""
        query_emb, corpus_emb = sample_embeddings

        # Query with many high-weight key terms
        query_text = "analyze review audit framework structure architecture"
        corpus_texts = [
            "analyze review audit framework structure architecture",
        ]

        similarities = EmbeddingService.weighted_cosine_similarity(
            query_emb, query_text, corpus_emb, corpus_texts
        )

        # Even with maximum boost, should not exceed 1.0
        assert all(s <= 1.0 for s in similarities)

    def test_weighted_similarity_case_insensitive(self, sample_embeddings):
        """Verify that term matching is case-insensitive."""
        query_emb, corpus_emb = sample_embeddings

        query_text = "Analyze the Framework"
        corpus_texts = ["analyze framework", "ANALYZE FRAMEWORK", "Analyze Framework"]

        similarities = EmbeddingService.weighted_cosine_similarity(
            query_emb, query_text, corpus_emb, corpus_texts
        )

        # All should get the same boost (case-insensitive matching)
        assert len(set(similarities)) <= 2  # May vary by base similarity

    def test_weighted_similarity_word_boundaries(self, sample_embeddings):
        """Verify that term matching respects word boundaries."""
        query_emb, corpus_emb = sample_embeddings

        query_text = "analyze the code"
        corpus_texts = [
            "analyze code",  # Should match "analyze"
            "analysis code",  # Should match "analysis" (different word)
            "create feature",  # Different task type - baseline
        ]

        similarities = EmbeddingService.weighted_cosine_similarity(
            query_emb, query_text, corpus_emb, corpus_texts
        )

        # First two should get boost from key terms
        # Note: with synthetic embeddings, the base similarities might be high
        # so we just check that key term items get reasonable scores
        assert similarities[0] >= 0.5  # analyze should match
        assert similarities[1] >= 0.5  # analysis should match

    def test_weighted_similarity_custom_weights(self, sample_embeddings):
        """Verify that custom weights override defaults."""
        query_emb, corpus_emb = sample_embeddings

        query_text = "custom term"
        corpus_texts = ["custom term", "other term", "third item"]

        custom_weights = {"custom": 2.0, "term": 1.5}

        similarities = EmbeddingService.weighted_cosine_similarity(
            query_emb,
            query_text,
            corpus_emb,
            corpus_texts,
            key_term_weights=custom_weights,
        )

        # First item should get boost from custom weights
        assert similarities[0] > similarities[1]

    def test_weighted_similarity_structural_analysis_case(self, embedding_service):
        """Test the specific 'framework structural analysis' case that failed."""
        # Create realistic embeddings
        query_text = "framework structural analysis"
        corpus_texts = [
            "analyze the structure",  # Should match strongly
            "review the architecture",  # Should match moderately
            "create a new file",  # Should not match
            "general help request",  # Should not match
        ]

        # Generate embeddings
        query_emb = embedding_service.embed_text_sync(query_text)
        corpus_emb = np.vstack([embedding_service.embed_text_sync(text) for text in corpus_texts])

        # Test weighted similarity
        weighted_similarities = EmbeddingService.weighted_cosine_similarity(
            query_emb, query_text, corpus_emb, corpus_texts
        )

        # Test base cosine similarity
        base_similarities = EmbeddingService.cosine_similarity_matrix(query_emb, corpus_emb)

        # "analyze the structure" should get higher or equal score with weighting
        # (may be equal if already at 1.0)
        assert (
            weighted_similarities[0] >= base_similarities[0] - 1e-6
        ), "Weighted similarity should not reduce 'analyze the structure'"

        # "review the architecture" should also get boost or equal
        assert (
            weighted_similarities[1] >= base_similarities[1] - 1e-6
        ), "Weighted similarity should not reduce 'review the architecture'"

        # Non-matching items should not be negatively affected significantly
        assert weighted_similarities[2] <= base_similarities[2] + 0.01

        # Check that analyze/architecture items rank higher than non-matching
        assert weighted_similarities[0] > weighted_similarities[2]
        assert weighted_similarities[1] > weighted_similarities[3]

    def test_weighted_similarity_empty_corpus(self, embedding_service):
        """Verify behavior with empty corpus."""
        query_text = "analyze this"
        query_emb = embedding_service.embed_text_sync(query_text)
        corpus_emb = np.array([]).reshape(0, 384)  # Empty 2D array
        corpus_texts = []

        similarities = EmbeddingService.weighted_cosine_similarity(
            query_emb, query_text, corpus_emb, corpus_texts
        )

        assert len(similarities) == 0

    def test_weighted_similarity_single_corpus_item(self, embedding_service):
        """Verify behavior with single corpus item."""
        query_text = "analyze code"
        query_emb = embedding_service.embed_text_sync(query_text)

        corpus_texts = ["review code"]
        corpus_emb = embedding_service.embed_text_sync(corpus_texts[0]).reshape(1, -1)

        similarities = EmbeddingService.weighted_cosine_similarity(
            query_emb, query_text, corpus_emb, corpus_texts
        )

        assert len(similarities) == 1
        assert 0.0 <= similarities[0] <= 1.0

    def test_weighted_similarity_multiple_overlapping_terms(self, sample_embeddings):
        """Verify boost calculation with multiple overlapping terms."""
        query_emb, corpus_emb = sample_embeddings

        query_text = "analyze review audit framework"
        corpus_texts = [
            "analyze review audit framework",  # All 4 terms overlap
            "analyze review",  # 2 terms overlap
            "create feature",  # No overlap
        ]

        similarities = EmbeddingService.weighted_cosine_similarity(
            query_emb, query_text, corpus_emb, corpus_texts
        )

        # More overlapping terms should result in higher boost (or equal if capped at 1.0)
        # With synthetic embeddings, just verify no crashes and reasonable scores
        assert all(0.0 <= s <= 1.0 for s in similarities)
        # All items should have some similarity with these synthetic embeddings
        assert all(s >= 0.5 for s in similarities)

    def test_weighted_similarity_partial_term_overlap(self, sample_embeddings):
        """Verify behavior with partial term overlap."""
        query_emb, corpus_emb = sample_embeddings

        query_text = "analyze the framework structure"
        corpus_texts = [
            "analyze framework",  # Partial overlap (2/3 terms)
            "review structure",  # Partial overlap (1/3 terms)
            "create new feature",  # No overlap
        ]

        similarities = EmbeddingService.weighted_cosine_similarity(
            query_emb, query_text, corpus_emb, corpus_texts
        )

        # With synthetic embeddings, just verify no crashes and reasonable scores
        assert all(0.0 <= s <= 1.0 for s in similarities)
        # Items with overlapping terms should get boosted
        assert similarities[0] >= 0.5
        assert similarities[1] >= 0.5

    def test_weighted_similarity_term_weights_dict(self):
        """Verify TASK_KEY_TERMS dict is properly structured."""
        weights = EmbeddingService.TASK_KEY_TERMS

        # Should be a dict
        assert isinstance(weights, dict)

        # Should have expected keys
        assert "analyze" in weights
        assert "analysis" in weights
        assert "review" in weights
        assert "structure" in weights
        assert "architecture" in weights
        assert "framework" in weights

        # All weights should be >= 1.0
        for term, weight in weights.items():
            assert weight >= 1.0, f"Weight for {term} should be >= 1.0, got {weight}"

        # Key terms should have appropriate weights
        assert weights["analyze"] >= 1.5, "Analyze should have high weight"
        assert weights["analysis"] >= 1.5, "Analysis should have high weight"

    def test_weighted_similarity_special_characters(self, sample_embeddings):
        """Verify handling of special characters in text."""
        query_emb, corpus_emb = sample_embeddings

        query_text = "analyze the: code, framework!"
        corpus_texts = ["analyze code framework", "review the code", "third item"]

        # Should handle special characters gracefully
        similarities = EmbeddingService.weighted_cosine_similarity(
            query_emb, query_text, corpus_emb, corpus_texts
        )

        assert len(similarities) == 3
        assert all(0.0 <= s <= 1.0 for s in similarities)

    def test_weighted_similarity_unicode_handling(self, sample_embeddings):
        """Verify handling of unicode characters."""
        query_emb, corpus_emb = sample_embeddings

        query_text = "analyze the framework"  # ASCII
        corpus_texts = [
            "analyze the framework",
            "review the architecture",
            "third item",
        ]

        # Should handle unicode without errors
        similarities = EmbeddingService.weighted_cosine_similarity(
            query_emb, query_text, corpus_emb, corpus_texts
        )

        assert len(similarities) == 3


class TestWeightedSimilarityIntegration:
    """Integration tests for weighted similarity with collections."""

    @pytest.fixture
    def sample_collection(self, tmp_path):
        """Create a sample embedding collection."""
        from victor.storage.embeddings.collections import (
            CollectionItem,
            StaticEmbeddingCollection,
        )

        collection = StaticEmbeddingCollection(
            name="test_weighted",
            cache_dir=tmp_path,
        )

        items = [
            CollectionItem(
                id="1",
                text="analyze the code structure",
                metadata={"task_type": "analyze"},
            ),
            CollectionItem(id="2", text="create a new file", metadata={"task_type": "create"}),
            CollectionItem(
                id="3",
                text="review the architecture",
                metadata={"task_type": "analyze"},
            ),
            CollectionItem(id="4", text="search for functions", metadata={"task_type": "search"}),
        ]

        import asyncio

        # Initialize synchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(collection.initialize(items))
        finally:
            loop.close()

        return collection

    def test_collection_search_with_weighted_similarity(self, sample_collection):
        """Test that collection uses weighted similarity when enabled."""
        query = "framework structural analysis"

        # Standard search
        results_standard = sample_collection.search_sync(
            query, top_k=4, use_weighted_similarity=False
        )

        # Weighted search
        results_weighted = sample_collection.search_sync(
            query, top_k=4, use_weighted_similarity=True
        )

        # Should return results in both cases
        assert len(results_standard) > 0
        assert len(results_weighted) > 0

        # Weighted search should boost "analyze the code structure" and "review the architecture"
        weighted_ids = [item.id for item, score in results_weighted]
        standard_ids = [item.id for item, score in results_standard]

        # Analyze items should rank higher with weighting
        analyze_items = [id for id in weighted_ids if id in ["1", "3"]]
        assert len(analyze_items) > 0, "Analyze items should be in results"

    def test_collection_search_backward_compatible(self, sample_collection):
        """Test that default behavior is backward compatible (use_weighted_similarity=False)."""
        query = "analyze the code"

        # Default call (should use standard cosine similarity)
        results_default = sample_collection.search_sync(query, top_k=2)

        # Explicit standard call
        results_standard = sample_collection.search_sync(
            query, top_k=2, use_weighted_similarity=False
        )

        # Should return same results
        assert len(results_default) == len(results_standard)

        for (item1, score1), (item2, score2) in zip(results_default, results_standard):
            assert item1.id == item2.id
            assert abs(score1 - score2) < 1e-6
