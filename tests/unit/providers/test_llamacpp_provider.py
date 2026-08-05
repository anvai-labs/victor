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

"""Tests for LlamaCppProvider."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from victor.providers.llamacpp_provider import (
    LlamaCppProvider,
    _model_supports_tools,
    TOOL_CAPABLE_PATTERNS,
    DEFAULT_LLAMACPP_URLS,
)
from victor.providers.base import (
    Message,
    ToolDefinition,
    ProviderConnectionError,
)

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def llamacpp_provider():
    """Create a LlamaCppProvider instance for testing."""
    return LlamaCppProvider(
        base_url="http://localhost:8080/v1",
        timeout=120,
    )


@pytest.fixture
def sample_messages():
    """Sample messages for testing."""
    return [
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="Hello, world!"),
    ]


@pytest.fixture
def sample_tool():
    """Sample tool definition."""
    return ToolDefinition(
        name="get_weather",
        description="Get the weather for a location",
        parameters={
            "type": "object",
            "properties": {"location": {"type": "string", "description": "The city name"}},
            "required": ["location"],
        },
    )


# =============================================================================
# UTILITY FUNCTION TESTS
# =============================================================================


class TestModelSupportsTools:
    """Tests for _model_supports_tools function."""

    def test_instruct_models_support_tools(self):
        """Test instruction-tuned models support tools."""
        assert _model_supports_tools("qwen2.5-coder-7b-instruct.Q4_K_M.gguf") is True
        assert _model_supports_tools("codellama-7b-instruct") is True

    def test_chat_models_support_tools(self):
        """Test chat models support tools."""
        assert _model_supports_tools("llama-3-8b-chat.gguf") is True

    def test_coder_models_support_tools(self):
        """Test coder models support tools."""
        assert _model_supports_tools("deepseek-coder-6.7b") is True
        # starcoder2-7b actually matches "coder" pattern in the name
        assert _model_supports_tools("starcoder2-7b") is True

    def test_qwen_models_support_tools(self):
        """Test Qwen models support tools."""
        assert _model_supports_tools("qwen-7b") is True
        assert _model_supports_tools("Qwen2.5-14B") is True

    def test_llama3_supports_tools(self):
        """Test Llama 3 models support tools."""
        assert _model_supports_tools("llama-3-8b") is True
        assert _model_supports_tools("Meta-Llama-3-8B") is True

    def test_mistral_supports_tools(self):
        """Test Mistral models support tools."""
        assert _model_supports_tools("mistral-7b") is True

    def test_base_model_no_tools(self):
        """Test base models without patterns don't support tools."""
        assert _model_supports_tools("gpt2.gguf") is False
        assert _model_supports_tools("phi-2") is False


# =============================================================================
# CONSTANTS TESTS
# =============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_tool_capable_patterns(self):
        """Test TOOL_CAPABLE_PATTERNS contains expected patterns."""
        assert "instruct" in TOOL_CAPABLE_PATTERNS
        assert "coder" in TOOL_CAPABLE_PATTERNS
        assert "qwen" in TOOL_CAPABLE_PATTERNS
        assert "llama-3" in TOOL_CAPABLE_PATTERNS

    def test_default_urls(self):
        """Test DEFAULT_LLAMACPP_URLS are configured."""
        assert len(DEFAULT_LLAMACPP_URLS) >= 1
        assert "localhost:8080" in DEFAULT_LLAMACPP_URLS[0]


# =============================================================================
# PROVIDER INITIALIZATION TESTS
# =============================================================================


class TestLlamaCppProviderInit:
    """Tests for LlamaCppProvider initialization."""

    def test_init_default(self):
        """Test default initialization."""
        provider = LlamaCppProvider()
        assert provider.name == "llamacpp"
        assert "localhost:8080" in provider.base_url

    def test_init_with_custom_url(self):
        """Test initialization with custom URL."""
        provider = LlamaCppProvider(base_url="http://custom-server:9000/v1")
        assert "custom-server:9000" in provider.base_url

    def test_init_strips_v1_suffix(self):
        """Test that /v1 suffix is stripped from base_url."""
        provider = LlamaCppProvider(base_url="http://localhost:8080/v1")
        assert not provider.base_url.endswith("/v1")

    def test_init_with_timeout(self):
        """Test initialization with custom timeout."""
        provider = LlamaCppProvider(timeout=600)
        assert provider.timeout == 600

    def test_default_timeout(self):
        """Test default timeout for CPU inference."""
        provider = LlamaCppProvider()
        assert provider.timeout == 300  # 5 minutes for CPU

    def test_provider_name(self):
        """Test provider name property."""
        provider = LlamaCppProvider()
        assert provider.name == "llamacpp"

    def test_supports_tools(self):
        """Test tools support indication."""
        provider = LlamaCppProvider()
        assert provider.supports_tools() is True

    def test_supports_streaming(self):
        """Test streaming support indication."""
        provider = LlamaCppProvider()
        assert provider.supports_streaming() is True

    def test_loaded_model_initially_none(self):
        """Test loaded model is None initially."""
        provider = LlamaCppProvider()
        assert provider._loaded_model is None


# =============================================================================
# FACTORY METHOD TESTS
# =============================================================================


class TestLlamaCppProviderFactory:
    """Tests for LlamaCppProvider factory method."""

    @pytest.mark.asyncio
    async def test_create_connection_error(self):
        """Test factory method raises error when server not reachable."""
        with pytest.raises(ProviderConnectionError) as exc_info:
            await LlamaCppProvider.create(base_url="http://nonexistent:99999/v1")
        assert "Cannot connect to llama.cpp server" in str(exc_info.value.message)

    @pytest.mark.asyncio
    async def test_create_provides_suggestion(self):
        """Test factory method provides helpful suggestion on failure."""
        with pytest.raises(ProviderConnectionError) as exc_info:
            await LlamaCppProvider.create(base_url="http://nonexistent:99999/v1")
        assert "llama-server" in str(exc_info.value.details.get("suggestion", ""))


# =============================================================================
# CLIENT TESTS
# =============================================================================


class TestLlamaCppProviderClient:
    """Tests for LlamaCppProvider HTTP client."""

    def test_client_initialized(self, llamacpp_provider):
        """Test HTTP client is initialized."""
        assert llamacpp_provider.client is not None

    def test_client_timeout_configured(self, llamacpp_provider):
        """Test client timeout is configured."""
        assert llamacpp_provider.timeout == 120


# =============================================================================
# MODEL CAPABILITIES TESTS
# =============================================================================


class TestLlamaCppModelCapabilities:
    """Tests for model capability detection."""

    def test_qwen_instruct_has_tools(self):
        """Test Qwen instruct model has tool support."""
        model = "qwen2.5-coder-7b-instruct.Q4_K_M.gguf"
        assert _model_supports_tools(model) is True

    def test_codellama_instruct_has_tools(self):
        """Test CodeLlama instruct has tool support."""
        model = "codellama-7b-instruct.Q4_K_M.gguf"
        assert _model_supports_tools(model) is True

    def test_deepseek_coder_has_tools(self):
        """Test DeepSeek Coder has tool support."""
        model = "deepseek-coder-6.7b-instruct.Q4_K_M.gguf"
        assert _model_supports_tools(model) is True


# =============================================================================
# API KEY TESTS
# =============================================================================


class TestLlamaCppApiKey:
    """Tests for API key handling."""

    def test_default_api_key(self):
        """Test default API key is 'not-needed'."""
        provider = LlamaCppProvider()
        assert provider.api_key == "not-needed"

    def test_custom_api_key_allowed(self):
        """Test custom API key can be set (for auth-enabled servers)."""
        provider = LlamaCppProvider(api_key="custom-key")
        assert provider.api_key == "custom-key"
