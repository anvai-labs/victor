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

"""Unit tests for SharedEncoder.

Tests the shared encoder for multi-task learning embeddings.
"""

import pytest

from victor.framework.rl.shared_encoder import (
    ContextEmbedding,
    SharedEncoder,
)


@pytest.fixture
def encoder() -> SharedEncoder:
    """Fixture for SharedEncoder."""
    return SharedEncoder()


class TestContextEmbedding:
    """Tests for ContextEmbedding."""

    def test_dimension_property(self) -> None:
        """Test dimension property."""
        embedding = ContextEmbedding(
            task_type="analysis",
            provider="anthropic",
            model="claude-3",
            vertical="coding",
            vector=[0.1, 0.2, 0.3, 0.4],
        )
        assert embedding.dimension == 4

    def test_similarity_identical(self) -> None:
        """Test similarity with identical vectors."""
        vec = [0.5, 0.5, 0.5, 0.5]
        e1 = ContextEmbedding("a", "p", "m", "v", vec)
        e2 = ContextEmbedding("b", "p", "m", "v", vec)

        similarity = e1.similarity(e2)
        assert similarity == pytest.approx(1.0, abs=0.01)

    def test_similarity_orthogonal(self) -> None:
        """Test similarity with orthogonal vectors."""
        e1 = ContextEmbedding("a", "p", "m", "v", [1.0, 0.0, 0.0, 0.0])
        e2 = ContextEmbedding("b", "p", "m", "v", [0.0, 1.0, 0.0, 0.0])

        similarity = e1.similarity(e2)
        assert similarity == pytest.approx(0.0, abs=0.01)

    def test_similarity_different_dimensions(self) -> None:
        """Test similarity with different dimensions returns 0."""
        e1 = ContextEmbedding("a", "p", "m", "v", [1.0, 0.0])
        e2 = ContextEmbedding("b", "p", "m", "v", [1.0, 0.0, 0.0])

        assert e1.similarity(e2) == 0.0


class TestSharedEncoder:
    """Tests for SharedEncoder."""

    def test_encode_returns_correct_dimension(self, encoder: SharedEncoder) -> None:
        """Test that encoding returns correct dimension."""
        embedding = encoder.encode(
            task_type="analysis",
            provider="anthropic",
            model="claude-3-opus",
            vertical="coding",
        )

        assert embedding.dimension == SharedEncoder.TOTAL_DIM
        assert embedding.task_type == "analysis"
        assert embedding.provider == "anthropic"

    def test_encode_caches_results(self, encoder: SharedEncoder) -> None:
        """Test that encoding caches results."""
        e1 = encoder.encode("analysis", "anthropic", "claude-3", "coding")
        e2 = encoder.encode("analysis", "anthropic", "claude-3", "coding")

        assert e1 is e2  # Same object from cache

    def test_different_contexts_different_embeddings(self, encoder: SharedEncoder) -> None:
        """Test that different contexts get different embeddings."""
        e1 = encoder.encode("analysis", "anthropic", "claude-3", "coding")
        e2 = encoder.encode("action", "openai", "gpt-4", "devops")

        assert e1.vector != e2.vector

    def test_similar_tasks_similar_embeddings(self, encoder: SharedEncoder) -> None:
        """Test that similar tasks have similar embeddings."""
        e_analysis = encoder.encode("analysis", "anthropic", "claude-3", "coding")
        e_explain = encoder.encode("explain", "anthropic", "claude-3", "coding")
        e_action = encoder.encode("action", "anthropic", "claude-3", "coding")

        # Analysis and explain should be more similar than analysis and action
        sim_analysis_explain = e_analysis.similarity(e_explain)
        sim_analysis_action = e_analysis.similarity(e_action)

        assert sim_analysis_explain > 0.5  # Should be similar

    def test_known_providers_have_embeddings(self, encoder: SharedEncoder) -> None:
        """Test that known providers get proper embeddings."""
        for provider in ["anthropic", "openai", "google", "deepseek", "ollama"]:
            embedding = encoder.encode("analysis", provider, "model", "coding")
            assert embedding.dimension == SharedEncoder.TOTAL_DIM
            assert all(isinstance(v, float) for v in embedding.vector)

    def test_unknown_provider_gets_embedding(self, encoder: SharedEncoder) -> None:
        """Test that unknown providers still get embeddings."""
        embedding = encoder.encode("analysis", "unknown_provider", "model", "coding")

        assert embedding.dimension == SharedEncoder.TOTAL_DIM
        assert embedding.provider == "unknown_provider"

    def test_verticals_have_distinct_embeddings(self, encoder: SharedEncoder) -> None:
        """Test that different verticals have distinct embeddings."""
        e_coding = encoder.encode("analysis", "anthropic", "claude-3", "coding")
        e_devops = encoder.encode("analysis", "anthropic", "claude-3", "devops")
        e_data = encoder.encode("analysis", "anthropic", "claude-3", "data_science")

        # Should be somewhat different
        assert e_coding.similarity(e_devops) < 0.99
        assert e_coding.similarity(e_data) < 0.99

    def test_find_similar_returns_results(self, encoder: SharedEncoder) -> None:
        """Test finding similar contexts."""
        # Create several embeddings
        encoder.encode("analysis", "anthropic", "claude-3", "coding")
        encoder.encode("search", "anthropic", "claude-3", "coding")
        encoder.encode("action", "openai", "gpt-4", "devops")

        query = encoder.encode("analysis", "openai", "gpt-4", "coding")
        similar = encoder.find_similar(query, top_k=2, min_similarity=0.0)

        assert len(similar) <= 2
        for emb, sim in similar:
            assert isinstance(sim, float)

    def test_transfer_weight_same_vertical(self, encoder: SharedEncoder) -> None:
        """Test transfer weight for same vertical."""
        e1 = encoder.encode("analysis", "anthropic", "claude-3", "coding")
        e2 = encoder.encode("search", "anthropic", "claude-3", "coding")

        weight = encoder.get_transfer_weight(e1, e2)
        assert weight >= 0.0

    def test_transfer_weight_different_vertical(self, encoder: SharedEncoder) -> None:
        """Test transfer weight for different verticals."""
        e1 = encoder.encode("analysis", "anthropic", "claude-3", "coding")
        e2 = encoder.encode("analysis", "anthropic", "claude-3", "devops")

        weight = encoder.get_transfer_weight(e1, e2)
        assert weight >= 0.0

    def test_export_metrics(self, encoder: SharedEncoder) -> None:
        """Test metrics export."""
        encoder.encode("analysis", "anthropic", "claude-3", "coding")
        encoder.encode("search", "openai", "gpt-4", "devops")

        metrics = encoder.export_metrics()

        assert metrics["cache_size"] == 2
        assert metrics["embedding_dimension"] == SharedEncoder.TOTAL_DIM
