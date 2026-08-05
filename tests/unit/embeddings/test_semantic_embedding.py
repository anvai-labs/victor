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

"""Tests for semantic embedding providers (sentence-transformers, Ollama, etc.)."""

import sys
from types import SimpleNamespace

import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch

from victor.tools.semantic_selector import SemanticToolSelector
from victor.tools.base import BaseTool
from victor.tools.registry import ToolRegistry


class MockTool(BaseTool):
    """Mock tool for testing."""

    @property
    def name(self) -> str:
        return "mock_tool"

    @property
    def description(self) -> str:
        return "A mock tool for testing semantic selection"

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs):
        return {"success": True}


@pytest.fixture
def mock_tool_registry():
    """Create mock tool registry with a few tools."""
    registry = ToolRegistry()
    registry.register(MockTool())
    return registry


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Create temporary cache directory."""
    cache_dir = tmp_path / "embeddings"
    cache_dir.mkdir()
    return cache_dir


class TestSentenceTransformersEmbedding:
    """Test sentence-transformers embedding provider."""

    @pytest.mark.asyncio
    async def test_default_provider_is_sentence_transformers(self):
        """Test that default provider is sentence-transformers."""
        from victor.storage.embeddings.service import DEFAULT_EMBEDDING_MODEL

        selector = SemanticToolSelector()
        assert selector.embedding_provider == "sentence-transformers"
        # Default model comes from settings.search.unified_embedding_model
        # which defaults to BAAI/bge-small-en-v1.5 (or all-MiniLM-L6-v2 as fallback)
        assert selector.embedding_model in [DEFAULT_EMBEDDING_MODEL, "all-MiniLM-L6-v2"]

    @pytest.mark.asyncio
    async def test_sentence_transformer_lazy_loading(self, temp_cache_dir):
        """Test that sentence-transformers model is loaded lazily via EmbeddingService."""
        from victor.storage.embeddings.service import EmbeddingService

        # Reset the singleton to ensure clean state
        EmbeddingService.reset_instance()

        selector = SemanticToolSelector(cache_dir=temp_cache_dir)

        # Model should not be loaded initially (EmbeddingService is lazy)
        assert EmbeddingService._instance is None or not EmbeddingService._instance.is_loaded

        # Mock the EmbeddingService's embed_text method
        mock_embedding = np.random.randn(384).astype(np.float32)
        with patch.object(EmbeddingService, "get_instance") as mock_get_instance:
            mock_service = MagicMock()
            mock_service.embed_text = AsyncMock(return_value=mock_embedding)
            mock_get_instance.return_value = mock_service

            # Get embedding should use EmbeddingService
            embedding = await selector._get_sentence_transformer_embedding("test text")

            # EmbeddingService should have been called with some model name
            mock_get_instance.assert_called_once()
            # Model name comes from settings, which may be the new default or legacy
            call_args = mock_get_instance.call_args
            model_name = call_args.kwargs.get(
                "model_name", call_args.args[0] if call_args.args else None
            )
            assert model_name is not None
            mock_service.embed_text.assert_called_once_with("test text")
            assert embedding.shape == (384,)
            assert embedding.dtype == np.float32

        # Reset singleton after test
        EmbeddingService.reset_instance()

    @pytest.mark.asyncio
    async def test_sentence_transformer_embedding_dimensions(self, temp_cache_dir):
        """Test that embeddings have correct dimensions."""
        selector = SemanticToolSelector(cache_dir=temp_cache_dir)

        with patch("sentence_transformers.SentenceTransformer") as MockST:
            mock_model = MagicMock()
            mock_model.encode.return_value = np.random.randn(384).astype(np.float32)
            MockST.return_value = mock_model

            embedding = await selector._get_embedding("test text")

            # all-MiniLM-L6-v2 produces 384-dimensional embeddings
            assert embedding.shape == (384,)
            assert embedding.dtype == np.float32

    @pytest.mark.asyncio
    async def test_sentence_transformer_fallback_on_import_error(self, temp_cache_dir):
        """Test fallback to random embedding when sentence-transformers not installed."""
        selector = SemanticToolSelector(cache_dir=temp_cache_dir)

        # Mock ImportError when loading sentence-transformers
        with patch("sentence_transformers.SentenceTransformer", side_effect=ImportError):
            # Should fall back to random embedding (better than crashing)
            embedding = await selector._get_sentence_transformer_embedding("test text")
            # Verify it's a valid embedding (384-dim)
            assert embedding.shape == (384,)
            assert embedding.dtype == np.float32


class FakeEmbeddingIndex:
    def __init__(self, vectors, labels):
        self.vectors = vectors
        self.labels = labels
        self.last_query = None

    def query(self, query, k, threshold):
        self.last_query = {"query": query, "k": k, "threshold": threshold}
        return [("mock_tool", 0.91)]


class TestEmbeddingIndexAcceleration:
    @pytest.mark.asyncio
    async def test_initialize_tool_embeddings_builds_native_index(
        self, temp_cache_dir, mock_tool_registry
    ):
        selector = SemanticToolSelector(cache_dir=temp_cache_dir)
        fake_embedding = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        fake_index_class = MagicMock(side_effect=FakeEmbeddingIndex)

        with (
            patch.object(selector, "_get_embedding", AsyncMock(return_value=fake_embedding)),
            patch.dict(
                sys.modules,
                {"victor_native": SimpleNamespace(EmbeddingIndex=fake_index_class)},
            ),
        ):
            await selector.initialize_tool_embeddings(mock_tool_registry)

        assert selector._embedding_index is not None
        fake_index_class.assert_called_once()

    @pytest.mark.asyncio
    async def test_select_relevant_tools_prefers_native_embedding_index(
        self, temp_cache_dir, mock_tool_registry
    ):
        selector = SemanticToolSelector(cache_dir=temp_cache_dir, cache_embeddings=False)
        selector._tool_embedding_cache = {"mock_tool": np.array([1.0, 0.0, 0.0], dtype=np.float32)}
        selector._embedding_index = FakeEmbeddingIndex(
            [selector._tool_embedding_cache["mock_tool"].tolist()],
            ["mock_tool"],
        )

        with (
            patch.object(
                selector,
                "_get_embedding",
                AsyncMock(return_value=np.array([1.0, 0.0, 0.0], dtype=np.float32)),
            ),
            patch.object(selector, "_batch_cosine_similarity", side_effect=AssertionError),
            patch.object(selector, "_get_relevant_categories", return_value=["mock_tool"]),
            patch.object(selector, "_get_mandatory_tools", return_value=[]),
        ):
            selected = await selector.select_relevant_tools(
                "find the relevant tool",
                mock_tool_registry,
                max_tools=1,
                similarity_threshold=0.1,
            )

        assert [tool.name for tool in selected] == ["mock_tool"]

    @pytest.mark.asyncio
    async def test_async_execution_in_thread_pool(self, temp_cache_dir):
        """Test that embedding runs async via EmbeddingService (non-blocking)."""
        from victor.storage.embeddings.service import EmbeddingService

        selector = SemanticToolSelector(cache_dir=temp_cache_dir)

        # Mock the EmbeddingService's async embed_text method
        mock_embedding = np.random.randn(384).astype(np.float32)
        with patch.object(EmbeddingService, "get_instance") as mock_get_instance:
            mock_service = MagicMock()
            mock_service.embed_text = AsyncMock(return_value=mock_embedding)
            mock_get_instance.return_value = mock_service

            # Should run without blocking
            embedding = await selector._get_sentence_transformer_embedding("test text")

            # Verify async embed_text was called on the service
            mock_service.embed_text.assert_called_once_with("test text")
            assert np.array_equal(embedding, mock_embedding)


class TestOllamaAPIEmbedding:
    """Test Ollama/vLLM/LMStudio API embedding provider."""

    @pytest.mark.asyncio
    async def test_ollama_provider_initialization(self):
        """Test Ollama provider initializes HTTP client."""
        selector = SemanticToolSelector(
            embedding_provider="ollama", embedding_model="nomic-embed-text"
        )

        assert selector.embedding_provider == "ollama"
        assert selector._client is not None

    @pytest.mark.asyncio
    async def test_api_embedding_request(self, temp_cache_dir):
        """Test API embedding makes correct HTTP request."""
        selector = SemanticToolSelector(
            embedding_provider="ollama",
            embedding_model="nomic-embed-text",
            cache_dir=temp_cache_dir,
        )

        # Mock httpx response
        mock_response = MagicMock()
        mock_response.json.return_value = {"embedding": [0.1] * 768}  # nomic-embed-text is 768-dim

        with patch.object(selector._client, "post", return_value=mock_response) as mock_post:
            mock_post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aexit__ = AsyncMock()

            await selector._get_api_embedding("test text")

            # Verify correct API endpoint and payload
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert "/api/embeddings" in str(call_args)

    @pytest.mark.asyncio
    async def test_vllm_provider(self):
        """Test vLLM provider uses same API interface."""
        selector = SemanticToolSelector(
            embedding_provider="vllm",
            embedding_model="BAAI/bge-large-en-v1.5",
            ollama_base_url="http://localhost:8000",
        )

        assert selector.embedding_provider == "vllm"
        assert selector._client is not None


class TestEmbeddingCaching:
    """Test embedding caching functionality."""

    @pytest.mark.asyncio
    async def test_cache_file_naming(self, temp_cache_dir):
        """Test cache file uses correct naming convention with project hash."""
        selector = SemanticToolSelector(
            embedding_model="all-MiniLM-L6-v2", cache_dir=temp_cache_dir
        )

        # Cache filename now includes project hash for isolation (TD-010)
        # Format: tool_embeddings_{model}_{hash}.pkl
        assert selector.cache_file.name.startswith("tool_embeddings_all-MiniLM-L6-v2")
        assert selector.cache_file.name.endswith(".pkl")

    @pytest.mark.asyncio
    async def test_cache_file_naming_with_special_chars(self, temp_cache_dir):
        """Test cache file naming handles special characters."""
        selector = SemanticToolSelector(
            embedding_model="qwen3-embedding:8b", cache_dir=temp_cache_dir
        )

        # Colons and slashes should be replaced, includes project hash
        assert selector.cache_file.name.startswith("tool_embeddings_qwen3-embedding_8b")
        assert selector.cache_file.name.endswith(".pkl")


class TestProviderFallback:
    """Test fallback logic between providers."""

    @pytest.mark.asyncio
    async def test_unsupported_provider_raises_error(self):
        """Test that unsupported provider raises NotImplementedError."""
        selector = SemanticToolSelector(embedding_provider="openai")  # Not yet implemented

        with pytest.raises(NotImplementedError, match="openai not yet supported"):
            await selector._get_embedding("test text")

    @pytest.mark.asyncio
    async def test_random_fallback_on_sentence_transformer_error(self, temp_cache_dir):
        """Test fallback to random embedding on error."""
        selector = SemanticToolSelector(cache_dir=temp_cache_dir)

        with patch("sentence_transformers.SentenceTransformer") as MockST:
            mock_model = MagicMock()
            mock_model.encode.side_effect = Exception("Model error")
            mock_model.get_sentence_embedding_dimension.return_value = 384
            MockST.return_value = mock_model

            embedding = await selector._get_sentence_transformer_embedding("test text")

            # Should return random embedding as fallback
            assert embedding.shape == (384,)
            assert embedding.dtype == np.float32


class TestToolSelectionWithEmbeddings:
    """Test end-to-end tool selection with embeddings."""

    @pytest.mark.asyncio
    async def test_tool_selection_with_sentence_transformers(
        self, mock_tool_registry, temp_cache_dir
    ):
        """Test complete tool selection flow with sentence-transformers.

        Note: select_relevant_tools filters by categories first, so mock_tool
        won't be selected unless we also mock the category filtering.
        This test verifies the embedding initialization and basic selection flow.
        """
        selector = SemanticToolSelector(cache_dir=temp_cache_dir)

        # Mock the _get_embedding method to return predictable embeddings
        async def mock_get_embedding(text: str):
            if "mock" in text.lower():
                return np.ones(384, dtype=np.float32)
            else:
                return np.zeros(384, dtype=np.float32)

        with patch.object(selector, "_get_embedding", side_effect=mock_get_embedding):
            # Initialize embeddings
            await selector.initialize_tool_embeddings(mock_tool_registry)

            # Verify embeddings were cached
            assert "mock_tool" in selector._tool_embedding_cache

            # Mock the category filtering to include our mock_tool
            with patch.object(selector, "_get_relevant_categories", return_value=["mock_tool"]):
                # Select tools
                selected_tools = await selector.select_relevant_tools(
                    "use the mock tool",
                    mock_tool_registry,
                    max_tools=5,
                    similarity_threshold=0.3,
                )

                # Should select mock_tool since we mocked the category filter
                tool_names = [t.name for t in selected_tools]
                assert len(tool_names) > 0
                assert "mock_tool" in tool_names


class TestCacheOperations:
    """Tests for cache loading and saving operations."""

    @pytest.mark.asyncio
    async def test_load_from_cache_file_not_exists(self, temp_cache_dir):
        """Test _load_from_cache returns False when file doesn't exist (covers line 270-271)."""
        selector = SemanticToolSelector(cache_dir=temp_cache_dir)
        # Cache file doesn't exist
        if selector.cache_file.exists():
            selector.cache_file.unlink()

        result = selector._load_from_cache("test_hash")
        assert result is False

    @pytest.mark.asyncio
    async def test_load_from_cache_hash_mismatch(self, temp_cache_dir):
        """Test _load_from_cache returns False on hash mismatch (covers lines 278-280)."""
        import pickle

        selector = SemanticToolSelector(cache_dir=temp_cache_dir)

        # Create a cache file with different hash
        cache_data = {
            "tools_hash": "different_hash",
            "embedding_model": selector.embedding_model,
            "embeddings": {},
        }
        with open(selector.cache_file, "wb") as f:
            pickle.dump(cache_data, f)

        result = selector._load_from_cache("expected_hash")
        assert result is False

    @pytest.mark.asyncio
    async def test_load_from_cache_model_mismatch(self, temp_cache_dir):
        """Test _load_from_cache returns False on model mismatch (covers lines 283-285)."""
        import pickle

        selector = SemanticToolSelector(cache_dir=temp_cache_dir)

        # Create a cache file with different model
        cache_data = {
            "tools_hash": "test_hash",
            "embedding_model": "different_model",
            "embeddings": {},
        }
        with open(selector.cache_file, "wb") as f:
            pickle.dump(cache_data, f)

        result = selector._load_from_cache("test_hash")
        assert result is False

    @pytest.mark.asyncio
    async def test_load_from_cache_success(self, temp_cache_dir):
        """Test _load_from_cache loads embeddings successfully (covers lines 287-291)."""
        import pickle

        selector = SemanticToolSelector(cache_dir=temp_cache_dir)

        # Create a valid cache file with all required fields including cache_version
        test_embeddings = {"tool1": np.array([0.1, 0.2, 0.3], dtype=np.float32)}
        cache_data = {
            "cache_version": selector.CACHE_VERSION,  # Required for cache validation
            "tools_hash": "test_hash",
            "embedding_model": selector.embedding_model,
            "embeddings": test_embeddings,
        }
        with open(selector.cache_file, "wb") as f:
            pickle.dump(cache_data, f)

        result = selector._load_from_cache("test_hash")
        assert result is True
        assert "tool1" in selector._tool_embedding_cache

    @pytest.mark.asyncio
    async def test_load_from_cache_corrupt_file(self, temp_cache_dir):
        """Test _load_from_cache handles corrupt cache (covers lines 293-295)."""
        selector = SemanticToolSelector(cache_dir=temp_cache_dir)

        # Write corrupt data
        with open(selector.cache_file, "wb") as f:
            f.write(b"not valid pickle data")

        result = selector._load_from_cache("test_hash")
        assert result is False

    @pytest.mark.asyncio
    async def test_save_to_cache_success(self, temp_cache_dir):
        """Test _save_to_cache saves embeddings (covers lines 303-314)."""
        selector = SemanticToolSelector(cache_dir=temp_cache_dir)
        selector._tool_embedding_cache = {"test_tool": np.array([0.1, 0.2], dtype=np.float32)}

        selector._save_to_cache("test_hash")

        assert selector.cache_file.exists()

    @pytest.mark.asyncio
    async def test_save_to_cache_failure(self, temp_cache_dir):
        """Test _save_to_cache handles write error (covers lines 316-317)."""
        selector = SemanticToolSelector(cache_dir=temp_cache_dir)

        # Make cache file read-only directory to cause write error
        with patch("builtins.open", side_effect=PermissionError("No write access")):
            # Should not raise, just log warning
            selector._save_to_cache("test_hash")


class TestFallbackAndMandatoryTools:
    """Tests for fallback and mandatory tool selection."""

    @pytest.mark.asyncio
    async def test_get_fallback_tools(self, mock_tool_registry, temp_cache_dir):
        """Test _get_fallback_tools returns common tools (covers lines 393-401)."""
        selector = SemanticToolSelector(cache_dir=temp_cache_dir)

        # Add some common tools to registry
        class ReadFileTool(BaseTool):
            @property
            def name(self):
                return "read_file"

            @property
            def description(self):
                return "Read file"

            @property
            def parameters(self):
                return {}

            async def execute(self, **kwargs):
                return {}

        mock_tool_registry.register(ReadFileTool())

        result = selector._get_fallback_tools(mock_tool_registry, max_tools=2)
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_get_mandatory_tools_with_commit(self, temp_cache_dir):
        """Test _get_mandatory_tools returns git tools for commit (covers lines 412-418)."""
        selector = SemanticToolSelector(cache_dir=temp_cache_dir)

        with patch(
            "victor.tools.semantic_selector.get_tools_matching_mandatory_keywords",
            return_value={"git"},
        ):
            result = selector._get_mandatory_tools("please commit my changes")
        # Accept any git-related tools (git, commit_msg, shell)
        assert "git" in result or "commit_msg" in result or "shell" in result

    @pytest.mark.asyncio
    async def test_get_mandatory_tools_with_test(self, temp_cache_dir):
        """Test _get_mandatory_tools returns test tools."""
        selector = SemanticToolSelector(cache_dir=temp_cache_dir)

        with patch(
            "victor.tools.semantic_selector.get_tools_matching_mandatory_keywords",
            return_value={"shell"},
        ):
            result = selector._get_mandatory_tools("run the tests")
        assert "shell" in result or "test" in result

    @pytest.mark.asyncio
    async def test_get_mandatory_tools_no_keywords(self, temp_cache_dir):
        """Test _get_mandatory_tools returns empty for generic query."""
        selector = SemanticToolSelector(cache_dir=temp_cache_dir)

        result = selector._get_mandatory_tools("hello world")
        # May return empty or minimal list
        assert isinstance(result, list)


class TestToolsHash:
    """Tests for tool hash calculation."""

    @pytest.mark.asyncio
    async def test_calculate_tools_hash(self, mock_tool_registry, temp_cache_dir):
        """Test _calculate_tools_hash returns consistent hash (covers lines 253-259)."""
        selector = SemanticToolSelector(cache_dir=temp_cache_dir)

        hash1 = selector._calculate_tools_hash(mock_tool_registry)
        hash2 = selector._calculate_tools_hash(mock_tool_registry)

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex length


class TestCostAwareSelection:
    """Tests for cost-aware tool selection."""

    @pytest.mark.asyncio
    async def test_cost_aware_selection_enabled(self, temp_cache_dir):
        """Test cost-aware selection is enabled by default."""
        selector = SemanticToolSelector(cache_dir=temp_cache_dir)
        assert selector.cost_aware_selection is True

    @pytest.mark.asyncio
    async def test_cost_penalty_factor_default(self, temp_cache_dir):
        """Test cost penalty factor default value."""
        selector = SemanticToolSelector(cache_dir=temp_cache_dir)
        assert selector.cost_penalty_factor == 0.05

    @pytest.mark.asyncio
    async def test_cost_aware_selection_disabled(self, temp_cache_dir):
        """Test cost-aware selection can be disabled."""
        selector = SemanticToolSelector(cache_dir=temp_cache_dir, cost_aware_selection=False)
        assert selector.cost_aware_selection is False


class TestConceptualQueryRouting:
    """Tests for conceptual query detection and semantic search routing (GAP-003 fix)."""

    @pytest.mark.asyncio
    async def test_is_conceptual_query_detects_inheritance_patterns(self, temp_cache_dir):
        """Test _is_conceptual_query detects inheritance-related queries."""
        selector = SemanticToolSelector(cache_dir=temp_cache_dir)

        # Should detect inheritance queries
        assert selector._is_conceptual_query("What classes inherit from BaseTool?")
        assert selector._is_conceptual_query("Find all subclasses that inherit from Provider")
        assert selector._is_conceptual_query("Show me classes that extend BaseClass")

    @pytest.mark.asyncio
    async def test_is_conceptual_query_detects_pattern_queries(self, temp_cache_dir):
        """Test _is_conceptual_query detects pattern-related queries."""
        selector = SemanticToolSelector(cache_dir=temp_cache_dir)

        # Should detect pattern queries (newly added patterns)
        assert selector._is_conceptual_query("Find error handling patterns")
        assert selector._is_conceptual_query("Show me exception handling code")
        assert selector._is_conceptual_query("Find logging implementations")
        assert selector._is_conceptual_query("Find caching patterns")

    @pytest.mark.asyncio
    async def test_is_conceptual_query_detects_related_queries(self, temp_cache_dir):
        """Test _is_conceptual_query detects 'similar to' and 'related to' queries."""
        selector = SemanticToolSelector(cache_dir=temp_cache_dir)

        assert selector._is_conceptual_query("Find code similar to the provider")
        assert selector._is_conceptual_query("Show related to authentication")

    @pytest.mark.asyncio
    async def test_is_conceptual_query_rejects_non_conceptual(self, temp_cache_dir):
        """Test _is_conceptual_query rejects non-conceptual queries."""
        selector = SemanticToolSelector(cache_dir=temp_cache_dir)

        # Literal queries - should NOT be conceptual
        assert not selector._is_conceptual_query("Read file victor/tools/base.py")
        assert not selector._is_conceptual_query("List files in directory")
        assert not selector._is_conceptual_query("Run git status")

    @pytest.mark.asyncio
    async def test_get_fallback_tools_conceptual_query(self, temp_cache_dir):
        """Test _get_fallback_tools returns semantic search for conceptual queries."""
        selector = SemanticToolSelector(cache_dir=temp_cache_dir)

        # Create a mock registry with semantic_code_search and read_file
        registry = ToolRegistry()

        class MockSearch(BaseTool):
            @property
            def name(self) -> str:
                return "search"

            @property
            def description(self) -> str:
                return "Semantic code search"

            @property
            def parameters(self) -> dict:
                return {"type": "object", "properties": {}}

            async def execute(self, **kwargs):
                return {"success": True}

        class MockRead(BaseTool):
            @property
            def name(self) -> str:
                return "read"

            @property
            def description(self) -> str:
                return "Read file"

            @property
            def parameters(self) -> dict:
                return {"type": "object", "properties": {}}

            async def execute(self, **kwargs):
                return {"success": True}

        class MockLs(BaseTool):
            @property
            def name(self) -> str:
                return "ls"

            @property
            def description(self) -> str:
                return "List directory"

            @property
            def parameters(self) -> dict:
                return {"type": "object", "properties": {}}

            async def execute(self, **kwargs):
                return {"success": True}

        registry.register(MockSearch())
        registry.register(MockRead())
        registry.register(MockLs())

        # For conceptual query, should return search + read (NOT ls)
        fallback = selector._get_fallback_tools(
            registry, max_tools=5, query="What classes inherit from BaseTool?"
        )

        assert "search" in fallback
        assert "read" in fallback
        # ls should NOT be included in conceptual fallback
        assert "ls" not in fallback

    @pytest.mark.asyncio
    async def test_get_fallback_tools_non_conceptual_query(self, temp_cache_dir):
        """Test _get_fallback_tools returns common tools for non-conceptual queries."""
        selector = SemanticToolSelector(cache_dir=temp_cache_dir)

        # Create a mock registry
        registry = ToolRegistry()

        class MockLs(BaseTool):
            @property
            def name(self) -> str:
                return "ls"

            @property
            def description(self) -> str:
                return "List directory"

            @property
            def parameters(self) -> dict:
                return {"type": "object", "properties": {}}

            async def execute(self, **kwargs):
                return {"success": True}

        registry.register(MockLs())

        # For non-conceptual query, should use common fallback tools
        fallback = selector._get_fallback_tools(
            registry, max_tools=5, query="List files in directory"
        )

        # ls should be in non-conceptual fallback
        assert "ls" in fallback

    @pytest.mark.asyncio
    async def test_conceptual_fallback_tools_constant_exists(self, temp_cache_dir):
        """Test CONCEPTUAL_FALLBACK_TOOLS constant is defined with correct tools."""
        selector = SemanticToolSelector(cache_dir=temp_cache_dir)

        # CONCEPTUAL_FALLBACK_TOOLS should exist and contain the right tools
        assert hasattr(selector, "CONCEPTUAL_FALLBACK_TOOLS")
        assert "search" in selector.CONCEPTUAL_FALLBACK_TOOLS
        assert "read" in selector.CONCEPTUAL_FALLBACK_TOOLS
        # Should NOT include exploratory tools that distract from semantic search
        assert "ls" not in selector.CONCEPTUAL_FALLBACK_TOOLS

    @pytest.mark.asyncio
    async def test_conceptual_query_patterns_extended(self, temp_cache_dir):
        """Test CONCEPTUAL_QUERY_PATTERNS includes new patterns for GAP-003."""
        selector = SemanticToolSelector(cache_dir=temp_cache_dir)

        # Check that new patterns are present
        patterns = selector.CONCEPTUAL_QUERY_PATTERNS
        assert "error handling" in patterns
        assert "exception" in patterns
        assert "logging" in patterns
        assert "caching" in patterns
        assert "similar to" in patterns
        assert "related to" in patterns


class TestSemanticMatchEventMetadata:
    """Tests for RL event metadata emitted by _emit_semantic_match_event."""

    def test_event_includes_embedding_model_and_results_count(self):
        """Verify _emit_semantic_match_event sends correct metadata keys.

        The SemanticThresholdLearner reads 'results_count', 'embedding_model',
        and 'threshold_used' from event metadata. Missing keys caused 100%
        zero_rate and embedding_model=unknown in RL stats.
        """
        selector = SemanticToolSelector(
            embedding_model="test-model-v1",
            embedding_provider="sentence-transformers",
            cache_embeddings=False,
        )

        # Create mock tools for the event
        mock_tool = MagicMock()
        mock_tool.name = "read"
        selected_tools = [(mock_tool, 0.85), (MagicMock(name="edit"), 0.72)]
        selected_tools[1][0].name = "edit"

        captured_events = []

        with patch("victor.tools.semantic_selector._get_rl_hooks") as mock_hooks_fn:
            mock_hooks = MagicMock()
            mock_hooks.emit = lambda evt: captured_events.append(evt)
            mock_hooks_fn.return_value = mock_hooks

            selector._emit_semantic_match_event(
                selected_tools=selected_tools,
                threshold=0.15,
                task_type="bug_fix",
                classification_aware=True,
                excluded_count=2,
            )

        assert len(captured_events) == 1
        event = captured_events[0]
        meta = event.metadata

        # These keys were missing and caused RL stats corruption
        assert meta["results_count"] == 2
        assert meta["embedding_model"] == "test-model-v1"
        assert meta["embedding_provider"] == "sentence-transformers"
        assert meta["threshold_used"] == 0.15

        # Existing keys still present
        assert meta["tools_selected"] == 2
        assert meta["classification_aware"] is True
        assert meta["excluded_count"] == 2

    def test_event_with_zero_results_sets_results_count_zero(self):
        """Verify results_count=0 when no tools selected."""
        selector = SemanticToolSelector(
            embedding_model="test-model",
            cache_embeddings=False,
        )

        captured_events = []

        with patch("victor.tools.semantic_selector._get_rl_hooks") as mock_hooks_fn:
            mock_hooks = MagicMock()
            mock_hooks.emit = lambda evt: captured_events.append(evt)
            mock_hooks_fn.return_value = mock_hooks

            selector._emit_semantic_match_event(
                selected_tools=[],
                threshold=0.15,
                task_type="default",
                classification_aware=False,
                excluded_count=0,
            )

        assert len(captured_events) == 1
        meta = captured_events[0].metadata
        assert meta["results_count"] == 0
        assert meta["embedding_model"] == "test-model"
        assert captured_events[0].success is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
