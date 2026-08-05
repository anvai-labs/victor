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

"""Unit tests for embedding models."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from victor.storage.vector_stores.models import (
    EmbeddingModelConfig,
    OllamaEmbeddingModel,
    SentenceTransformerModel,
    OpenAIEmbeddingModel,
    CohereEmbeddingModel,
    create_embedding_model,
)


@pytest.fixture
def ollama_config():
    """Create Ollama embedding config."""
    return EmbeddingModelConfig(
        model_type="ollama",
        model_name="qwen3-embedding:8b",
        dimension=4096,
        api_key="http://localhost:11434",
        batch_size=8,
    )


@pytest.fixture
def sentence_transformer_config():
    """Create sentence-transformers config."""
    return EmbeddingModelConfig(
        model_type="sentence-transformers",
        model_name="all-MiniLM-L6-v2",
        dimension=384,
        batch_size=32,
    )


@pytest.fixture
def openai_config():
    """Create OpenAI config."""
    return EmbeddingModelConfig(
        model_type="openai",
        model_name="text-embedding-3-small",
        dimension=1536,
        api_key="test-api-key",
        batch_size=100,
    )


class TestOllamaEmbeddingModel:
    """Tests for OllamaEmbeddingModel."""

    @pytest.mark.asyncio
    async def test_initialization(self, ollama_config):
        """Test model initialization."""
        model = OllamaEmbeddingModel(ollama_config)

        assert model.config == ollama_config
        assert model.base_url == "http://localhost:11434"
        assert model.client is None
        assert not model._initialized

    @pytest.mark.asyncio
    async def test_initialize_success(self, ollama_config):
        """Test successful initialization."""
        model = OllamaEmbeddingModel(ollama_config)

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.json.return_value = {"embedding": [0.1] * 4096}
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client

            await model.initialize()

            assert model._initialized
            assert model.client is not None
            mock_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_model_not_found(self, ollama_config):
        """Test initialization with model not found."""
        model = OllamaEmbeddingModel(ollama_config)

        with patch("httpx.AsyncClient") as mock_client_class:
            import httpx

            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 404
            http_error = httpx.HTTPStatusError(
                "404 Not Found", request=MagicMock(), response=mock_response
            )
            mock_response.raise_for_status.side_effect = http_error
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client

            with pytest.raises(RuntimeError, match="not found"):
                await model.initialize()

    @pytest.mark.asyncio
    async def test_initialize_connection_error(self, ollama_config):
        """Test initialization with connection error."""
        model = OllamaEmbeddingModel(ollama_config)

        with patch("httpx.AsyncClient") as mock_client_class:
            import httpx

            mock_client = AsyncMock()
            mock_client.post.side_effect = httpx.ConnectError("Connection refused")
            mock_client_class.return_value = mock_client

            with pytest.raises(RuntimeError, match="Cannot connect"):
                await model.initialize()

    @pytest.mark.asyncio
    async def test_embed_text(self, ollama_config):
        """Test single text embedding."""
        model = OllamaEmbeddingModel(ollama_config)
        test_embedding = [0.1, 0.2, 0.3] * 1365 + [0.1]  # 4096 values

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = AsyncMock()
            mock_response.raise_for_status = lambda: None  # Synchronous
            mock_response.json = lambda: {"embedding": test_embedding}  # Synchronous
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client

            # Mark as initialized to skip initialization
            model._initialized = True
            model.client = mock_client

            result = await model.embed_text("test text")

            assert result == test_embedding
            assert len(result) == 4096

    @pytest.mark.asyncio
    async def test_embed_batch(self, ollama_config):
        """Test batch embedding."""
        model = OllamaEmbeddingModel(ollama_config)
        test_embedding = [0.1, 0.2, 0.3] * 1365 + [0.1]

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = AsyncMock()
            mock_response.raise_for_status = lambda: None  # Synchronous
            mock_response.json = lambda: {"embedding": test_embedding}  # Synchronous
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client

            # Mark as initialized to skip initialization
            model._initialized = True
            model.client = mock_client

            texts = ["text1", "text2", "text3"]
            results = await model.embed_batch(texts)

            assert len(results) == 3
            assert all(len(emb) == 4096 for emb in results)
            assert mock_client.post.call_count == 3  # One call per text

    def test_get_dimension_qwen3(self, ollama_config):
        """Test dimension getter for Qwen3."""
        model = OllamaEmbeddingModel(ollama_config)
        assert model.get_dimension() == 4096

    def test_get_dimension_other_models(self):
        """Test dimension getter for other models."""
        config = EmbeddingModelConfig(
            model_type="ollama",
            model_name="nomic-embed-text",
            dimension=768,
        )
        model = OllamaEmbeddingModel(config)
        assert model.get_dimension() == 768

        config2 = EmbeddingModelConfig(
            model_type="ollama",
            model_name="snowflake-arctic-embed2",
            dimension=1024,
        )
        model2 = OllamaEmbeddingModel(config2)
        assert model2.get_dimension() == 1024

    @pytest.mark.asyncio
    async def test_close(self, ollama_config):
        """Test cleanup."""
        model = OllamaEmbeddingModel(ollama_config)

        with patch("httpx.AsyncClient"):
            mock_client = AsyncMock()
            mock_client.aclose = AsyncMock()
            model.client = mock_client
            model._initialized = True

            await model.close()

            mock_client.aclose.assert_called_once()
            assert model.client is None
            assert not model._initialized


class TestSentenceTransformerModel:
    """Tests for SentenceTransformerModel.

    Note: SentenceTransformerModel now uses the shared EmbeddingService singleton
    for memory efficiency (shares model with IntentClassifier and SemanticToolSelector).
    """

    @pytest.mark.asyncio
    async def test_initialization(self, sentence_transformer_config):
        """Test model initialization."""
        model = SentenceTransformerModel(sentence_transformer_config)

        assert model.config == sentence_transformer_config
        assert model._embedding_service is None
        assert not model._initialized

    @pytest.mark.asyncio
    async def test_initialize_success(self, sentence_transformer_config):
        """Test successful initialization via EmbeddingService."""
        from victor.storage.embeddings.service import EmbeddingService

        # Reset singleton to ensure clean state
        EmbeddingService.reset_instance()

        model = SentenceTransformerModel(sentence_transformer_config)

        # Mock the EmbeddingService.get_instance at the source module
        mock_service = MagicMock()
        mock_service.dimension = 384
        mock_service._ensure_model_loaded = MagicMock()

        with patch.object(EmbeddingService, "get_instance", return_value=mock_service):
            await model.initialize()

            assert model._initialized
            assert model._embedding_service == mock_service
            mock_service._ensure_model_loaded.assert_called_once()

        # Reset singleton after test
        EmbeddingService.reset_instance()

    @pytest.mark.asyncio
    async def test_initialize_import_error(self, sentence_transformer_config):
        """Test initialization with missing EmbeddingService.

        Note: Import error now happens inside EmbeddingService when it tries
        to load sentence-transformers. We test that the model handles this.
        """
        from victor.storage.embeddings.service import EmbeddingService

        model = SentenceTransformerModel(sentence_transformer_config)

        # Mock EmbeddingService.get_instance to raise ImportError
        with patch.object(
            EmbeddingService,
            "get_instance",
            side_effect=ImportError("sentence-transformers not installed"),
        ):
            with pytest.raises(ImportError, match="sentence-transformers not installed"):
                await model.initialize()

    def test_get_dimension_with_service(self, sentence_transformer_config):
        """Test dimension getter with loaded service."""
        model = SentenceTransformerModel(sentence_transformer_config)
        mock_service = MagicMock()
        mock_service.dimension = 384
        model._embedding_service = mock_service

        assert model.get_dimension() == 384

    def test_get_dimension_without_service(self, sentence_transformer_config):
        """Test dimension getter without loaded service."""
        model = SentenceTransformerModel(sentence_transformer_config)
        assert model.get_dimension() == 384


class TestOpenAIEmbeddingModel:
    """Tests for OpenAIEmbeddingModel."""

    @pytest.mark.asyncio
    async def test_initialization(self, openai_config):
        """Test model initialization."""
        model = OpenAIEmbeddingModel(openai_config)

        assert model.config == openai_config
        assert model.client is None
        assert not model._initialized

    @pytest.mark.asyncio
    async def test_initialize_success(self, openai_config):
        """Test successful initialization."""
        import importlib.util

        if importlib.util.find_spec("openai") is None:
            pytest.skip("openai not installed")

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = AsyncMock()
            mock_openai.return_value = mock_client

            model = OpenAIEmbeddingModel(openai_config)
            await model.initialize()

            assert model._initialized
            assert model.client == mock_client
            mock_openai.assert_called_once_with(api_key="test-api-key")

    @pytest.mark.asyncio
    async def test_initialize_missing_api_key(self):
        """Test initialization without API key."""
        config = EmbeddingModelConfig(
            model_type="openai",
            model_name="text-embedding-3-small",
            dimension=1536,
        )
        model = OpenAIEmbeddingModel(config)

        with pytest.raises(ValueError, match="API key required"):
            await model.initialize()

    @pytest.mark.asyncio
    async def test_embed_text(self, openai_config):
        """Test single text embedding."""
        import importlib.util

        if importlib.util.find_spec("openai") is None:
            pytest.skip("openai not installed")

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_client = AsyncMock()

            # Create mock embedding response
            mock_embedding_item = MagicMock()
            mock_embedding_item.embedding = [0.1] * 1536
            mock_response = AsyncMock()
            mock_response.data = [mock_embedding_item]

            mock_client.embeddings.create.return_value = mock_response
            mock_openai.return_value = mock_client

            model = OpenAIEmbeddingModel(openai_config)
            await model.initialize()
            result = await model.embed_text("test text")

            assert len(result) == 1536
            assert all(x == 0.1 for x in result)

    def test_get_dimension_known_models(self):
        """Test dimension getter for known OpenAI models."""
        configs = [
            ("text-embedding-3-small", 1536),
            ("text-embedding-3-large", 3072),
            ("text-embedding-ada-002", 1536),
        ]

        for model_name, expected_dim in configs:
            config = EmbeddingModelConfig(
                model_type="openai",
                model_name=model_name,
                api_key="test",
            )
            model = OpenAIEmbeddingModel(config)
            assert model.get_dimension() == expected_dim


class TestCohereEmbeddingModel:
    """Tests for CohereEmbeddingModel."""

    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test model initialization."""
        config = EmbeddingModelConfig(
            model_type="cohere",
            model_name="embed-english-v3.0",
            api_key="test-key",
        )
        model = CohereEmbeddingModel(config)

        assert model.config == config
        assert model.client is None
        assert not model._initialized

    def test_get_dimension_known_models(self):
        """Test dimension getter for known Cohere models."""
        configs = [
            ("embed-english-v3.0", 1024),
            ("embed-multilingual-v3.0", 1024),
            ("embed-english-light-v3.0", 384),
            ("embed-multilingual-light-v3.0", 384),
        ]

        for model_name, expected_dim in configs:
            config = EmbeddingModelConfig(
                model_type="cohere",
                model_name=model_name,
                api_key="test",
            )
            model = CohereEmbeddingModel(config)
            assert model.get_dimension() == expected_dim


class TestEmbeddingModelFactory:
    """Tests for create_embedding_model factory function."""

    def test_create_ollama_model(self, ollama_config):
        """Test creating Ollama model."""
        model = create_embedding_model(ollama_config)
        assert isinstance(model, OllamaEmbeddingModel)
        assert model.config == ollama_config

    def test_create_sentence_transformer_model(self, sentence_transformer_config):
        """Test creating sentence-transformers model."""
        model = create_embedding_model(sentence_transformer_config)
        assert isinstance(model, SentenceTransformerModel)
        assert model.config == sentence_transformer_config

    def test_create_openai_model(self, openai_config):
        """Test creating OpenAI model."""
        model = create_embedding_model(openai_config)
        assert isinstance(model, OpenAIEmbeddingModel)
        assert model.config == openai_config

    def test_create_unknown_model_type(self):
        """Test creating model with unknown type."""
        config = EmbeddingModelConfig(
            model_type="unknown",
            model_name="test",
        )

        with pytest.raises(ValueError, match="Unknown embedding model type"):
            create_embedding_model(config)
