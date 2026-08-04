"""Tests for Phase 5 config groups extracted from Settings class.

Tests cover:
- EmbeddingSettings
- ToolSelectionSettings
"""

import pytest

from victor.config.groups.embedding_config import EmbeddingSettings
from victor.config.groups.tool_selection_config import ToolSelectionSettings


class TestEmbeddingSettings:
    """Tests for EmbeddingSettings."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        settings = EmbeddingSettings()

        assert settings.unified_embedding_model == "BAAI/bge-small-en-v1.5"
        assert settings.use_semantic_tool_selection is True
        assert settings.preload_embeddings is False
        assert settings.embedding_provider == "sentence-transformers"
        assert settings.embedding_model == "BAAI/bge-small-en-v1.5"
        assert settings.codebase_vector_store == "lancedb"
        assert settings.codebase_embedding_provider == "sentence-transformers"
        assert settings.codebase_dimension == 384
        assert settings.codebase_batch_size == 32
        assert settings.codebase_structural_indexing_enabled is False
        assert settings.codebase_chunking_strategy == "victor_codegraph"
        assert settings.codebase_chunk_size == 500
        assert settings.codebase_chunk_overlap == 50
        assert settings.codebase_embedding_extra_config == {}
        assert settings.codebase_graph_store == "sqlite"
        assert settings.codebase_graph_writer_mode == "off"
        assert settings.semantic_similarity_threshold == 0.25
        assert settings.semantic_query_expansion_enabled is True
        assert settings.semantic_max_query_expansions == 5
        assert settings.enable_hybrid_search is False
        assert settings.hybrid_search_semantic_weight == 0.6
        assert settings.hybrid_search_keyword_weight == 0.4
        assert settings.enable_semantic_threshold_rl_learning is False
        assert settings.semantic_threshold_overrides == {}

    def test_custom_values(self):
        """Test custom values."""
        settings = EmbeddingSettings(
            unified_embedding_model="custom-model",
            use_semantic_tool_selection=False,
            preload_embeddings=True,
            embedding_provider="ollama",
            codebase_vector_store="lancedb",
            semantic_similarity_threshold=0.5,
            semantic_max_query_expansions=10,
            enable_hybrid_search=True,
        )

        assert settings.unified_embedding_model == "custom-model"
        assert settings.use_semantic_tool_selection is False
        assert settings.preload_embeddings is True
        assert settings.embedding_provider == "ollama"
        assert settings.codebase_vector_store == "lancedb"
        assert settings.semantic_similarity_threshold == 0.5
        assert settings.semantic_max_query_expansions == 10
        assert settings.enable_hybrid_search is True

    def test_extra_config_passthrough(self):
        """Test provider-specific extra config is preserved."""
        settings = EmbeddingSettings(codebase_embedding_extra_config={"table_name": "custom"})
        assert settings.codebase_embedding_extra_config == {"table_name": "custom"}

    def test_chunk_settings_validation(self):
        """Test structural chunk configuration validation."""
        settings = EmbeddingSettings(codebase_chunk_size=256, codebase_chunk_overlap=12)
        assert settings.codebase_chunk_size == 256
        assert settings.codebase_chunk_overlap == 12

        with pytest.raises(ValueError, match="codebase_chunk_size must be >= 1"):
            EmbeddingSettings(codebase_chunk_size=0)

        with pytest.raises(ValueError, match="codebase_chunk_overlap must be >= 0"):
            EmbeddingSettings(codebase_chunk_overlap=-1)

    def test_dimension_validation(self):
        """Test embedding dimension validation."""
        # Valid dimension
        settings = EmbeddingSettings(codebase_dimension=128)
        assert settings.codebase_dimension == 128

        # Invalid dimension (zero)
        with pytest.raises(ValueError, match="codebase_dimension must be >= 1"):
            EmbeddingSettings(codebase_dimension=0)

    def test_batch_size_validation(self):
        """Test batch size validation."""
        # Valid batch size
        settings = EmbeddingSettings(codebase_batch_size=16)
        assert settings.codebase_batch_size == 16

        # Invalid batch size (zero)
        with pytest.raises(ValueError, match="codebase_batch_size must be >= 1"):
            EmbeddingSettings(codebase_batch_size=0)

    def test_similarity_threshold_validation(self):
        """Test similarity threshold validation."""
        # Valid thresholds
        settings = EmbeddingSettings(semantic_similarity_threshold=0.0)
        assert settings.semantic_similarity_threshold == 0.0

        settings = EmbeddingSettings(semantic_similarity_threshold=1.0)
        assert settings.semantic_similarity_threshold == 1.0

        # Invalid threshold (out of range)
        with pytest.raises(
            ValueError,
            match="semantic_similarity_threshold must be between 0.0 and 1.0",
        ):
            EmbeddingSettings(semantic_similarity_threshold=1.1)

    def test_max_query_expansions_validation(self):
        """Test max query expansions validation."""
        # Valid max expansions
        settings = EmbeddingSettings(semantic_max_query_expansions=1)
        assert settings.semantic_max_query_expansions == 1

        # Invalid max expansions (zero)
        with pytest.raises(ValueError, match="semantic_max_query_expansions must be >= 1"):
            EmbeddingSettings(semantic_max_query_expansions=0)

    def test_hybrid_search_weights_validation(self):
        """Test hybrid search weights validation."""
        # Valid weights (sum to 1.0)
        settings = EmbeddingSettings(
            enable_hybrid_search=True,
            hybrid_search_semantic_weight=0.7,
            hybrid_search_keyword_weight=0.3,
        )
        assert settings.hybrid_search_semantic_weight == 0.7
        assert settings.hybrid_search_keyword_weight == 0.3

        # Invalid weights (don't sum to 1.0)
        with pytest.raises(ValueError, match="Hybrid search weights must sum to 1.0"):
            EmbeddingSettings(
                enable_hybrid_search=True,
                hybrid_search_semantic_weight=0.7,
                hybrid_search_keyword_weight=0.5,
            )


class TestToolSelectionSettings:
    """Tests for ToolSelectionSettings."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        settings = ToolSelectionSettings()

        assert settings.use_semantic_tool_selection is True
        assert settings.preload_embeddings is False
        assert settings.enable_tool_deduplication is True
        assert settings.tool_deduplication_window_size == 20
        assert settings.fallback_max_tools == 8

    def test_custom_values(self):
        """Test custom values."""
        settings = ToolSelectionSettings(
            use_semantic_tool_selection=False,
            preload_embeddings=True,
            enable_tool_deduplication=False,
            tool_deduplication_window_size=50,
            fallback_max_tools=15,
        )

        assert settings.use_semantic_tool_selection is False
        assert settings.preload_embeddings is True
        assert settings.enable_tool_deduplication is False
        assert settings.tool_deduplication_window_size == 50
        assert settings.fallback_max_tools == 15

    def test_window_size_validation(self):
        """Test deduplication window size validation."""
        # Valid window size
        settings = ToolSelectionSettings(tool_deduplication_window_size=1)
        assert settings.tool_deduplication_window_size == 1

        settings = ToolSelectionSettings(tool_deduplication_window_size=100)
        assert settings.tool_deduplication_window_size == 100

        # Invalid window size (zero)
        with pytest.raises(ValueError, match="tool_deduplication_window_size must be >= 1"):
            ToolSelectionSettings(tool_deduplication_window_size=0)

        # Invalid window size (too large)
        with pytest.raises(ValueError, match="tool_deduplication_window_size must be <= 100"):
            ToolSelectionSettings(tool_deduplication_window_size=101)

    def test_fallback_max_tools_validation(self):
        """Test fallback max tools validation."""
        # Valid max tools
        settings = ToolSelectionSettings(fallback_max_tools=1)
        assert settings.fallback_max_tools == 1

        # Invalid max tools (zero)
        with pytest.raises(ValueError, match="fallback_max_tools must be >= 1"):
            ToolSelectionSettings(fallback_max_tools=0)


class TestPhase5ConfigGroupsIntegration:
    """Integration tests for Phase 5 config groups with Settings."""

    def test_embedding_settings_in_main_settings(self):
        """Test that EmbeddingSettings is accessible from Settings."""
        from victor.config.settings import Settings

        settings = Settings()

        # Nested access is the canonical runtime contract.
        assert settings.embedding is not None
        assert isinstance(settings.embedding, EmbeddingSettings)
        assert settings.embedding.unified_embedding_model == "BAAI/bge-small-en-v1.5"
        assert settings.search.codebase_embedding_provider == "sentence-transformers"

    def test_tool_selection_settings_in_main_settings(self):
        """Test that ToolSelectionSettings is accessible from Settings."""
        from victor.config.settings import Settings

        settings = Settings()

        # Nested access only
        assert settings.tool_selection is not None
        assert isinstance(settings.tool_selection, ToolSelectionSettings)

    def test_nested_group_independence(self):
        """Test that nested groups are independent from each other."""
        from victor.config.settings import Settings

        settings = Settings()

        # Modifying one group shouldn't affect others
        embedding_id = id(settings.embedding)
        tool_selection_id = id(settings.tool_selection)

        assert embedding_id != tool_selection_id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
