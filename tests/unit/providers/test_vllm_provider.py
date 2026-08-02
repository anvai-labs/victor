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

"""Tests for VLLMProvider."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from victor.providers.vllm_provider import (
    VLLMProvider,
    _model_supports_tools,
    _model_uses_thinking_tags,
    TOOL_CAPABLE_MODELS,
    THINKING_TAG_MODELS,
    DEFAULT_VLLM_URLS,
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
def vllm_provider():
    """Create a VLLMProvider instance for testing."""
    return VLLMProvider(
        base_url="http://localhost:8000/v1",
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

    def test_qwen_coder_supports_tools(self):
        """Test Qwen coder models support tools."""
        assert _model_supports_tools("Qwen/Qwen2.5-Coder-7B-Instruct") is True
        assert _model_supports_tools("qwen2.5-coder-14b") is True

    def test_llama_models_support_tools(self):
        """Test Llama 3.1+ models support tools."""
        assert _model_supports_tools("meta-llama/llama-3.1-8b-instruct") is True
        assert _model_supports_tools("llama-3.3-70b") is True

    def test_deepseek_coder_supports_tools(self):
        """Test DeepSeek Coder supports tools."""
        assert _model_supports_tools("deepseek-ai/DeepSeek-Coder-V2-Lite") is True

    def test_codestral_supports_tools(self):
        """Test Codestral supports tools."""
        assert _model_supports_tools("mistralai/Codestral-22B") is True

    def test_hermes_supports_tools(self):
        """Test Hermes models support tools."""
        assert _model_supports_tools("NousResearch/Hermes-2-Pro") is True

    def test_instruct_suffix_supports_tools(self):
        """Test -instruct suffix indicates tool support."""
        assert _model_supports_tools("some-model-instruct") is True

    def test_tools_suffix_supports_tools(self):
        """Test -tools suffix indicates tool support."""
        assert _model_supports_tools("some-model-tools") is True

    def test_base_model_no_tools(self):
        """Test base models without instruct don't support tools."""
        assert _model_supports_tools("gpt2") is False
        assert _model_supports_tools("phi-2") is False


class TestModelUsesThinkingTags:
    """Tests for _model_uses_thinking_tags function."""

    def test_qwen3_uses_thinking(self):
        """Test Qwen3 uses thinking tags."""
        assert _model_uses_thinking_tags("Qwen/Qwen3-8B") is True
        assert _model_uses_thinking_tags("qwen3-coder") is True

    def test_deepseek_r1_uses_thinking(self):
        """Test DeepSeek-R1 uses thinking tags."""
        assert _model_uses_thinking_tags("deepseek-ai/DeepSeek-R1") is True
        assert _model_uses_thinking_tags("deepseek-r1-lite") is True

    def test_deepseek_reasoner_uses_thinking(self):
        """Test DeepSeek Reasoner uses thinking tags."""
        assert _model_uses_thinking_tags("deepseek-reasoner") is True

    def test_regular_models_no_thinking(self):
        """Test regular models don't use thinking tags."""
        assert _model_uses_thinking_tags("llama-3.1-8b") is False
        assert _model_uses_thinking_tags("gpt-4") is False


# =============================================================================
# CONSTANTS TESTS
# =============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_tool_capable_models_list(self):
        """Test TOOL_CAPABLE_MODELS contains expected patterns."""
        assert "qwen2.5-coder" in TOOL_CAPABLE_MODELS
        assert "llama-3.1" in TOOL_CAPABLE_MODELS
        assert "-instruct" in TOOL_CAPABLE_MODELS

    def test_thinking_tag_models_list(self):
        """Test THINKING_TAG_MODELS contains expected patterns."""
        assert "qwen3" in THINKING_TAG_MODELS
        assert "deepseek-r1" in THINKING_TAG_MODELS

    def test_default_urls(self):
        """Test DEFAULT_VLLM_URLS are configured."""
        assert len(DEFAULT_VLLM_URLS) >= 1
        assert "localhost:8000" in DEFAULT_VLLM_URLS[0]


# =============================================================================
# PROVIDER INITIALIZATION TESTS
# =============================================================================


class TestVLLMProviderInit:
    """Tests for VLLMProvider initialization."""

    def test_init_default(self):
        """Test default initialization."""
        provider = VLLMProvider()
        assert provider.name == "vllm"
        assert "localhost:8000" in provider.base_url

    def test_init_with_custom_url(self):
        """Test initialization with custom URL."""
        provider = VLLMProvider(base_url="http://custom-server:9000/v1")
        assert "custom-server:9000" in provider.base_url

    def test_init_strips_v1_suffix(self):
        """Test that /v1 suffix is stripped from base_url."""
        provider = VLLMProvider(base_url="http://localhost:8000/v1")
        assert not provider.base_url.endswith("/v1")

    def test_init_with_timeout(self):
        """Test initialization with custom timeout."""
        provider = VLLMProvider(timeout=600)
        assert provider.timeout == 600

    def test_default_timeout(self):
        """Test default timeout for large models."""
        provider = VLLMProvider()
        assert provider.timeout == 300  # 5 minutes for large models

    def test_provider_name(self):
        """Test provider name property."""
        provider = VLLMProvider()
        assert provider.name == "vllm"

    def test_supports_tools(self):
        """Test tools support indication."""
        provider = VLLMProvider()
        assert provider.supports_tools() is True

    def test_supports_streaming(self):
        """Test streaming support indication."""
        provider = VLLMProvider()
        assert provider.supports_streaming() is True


# =============================================================================
# FACTORY METHOD TESTS
# =============================================================================


class TestVLLMProviderFactory:
    """Tests for VLLMProvider factory method."""

    @pytest.mark.asyncio
    async def test_create_connection_error(self):
        """Test factory method raises error when server not reachable."""
        with pytest.raises(ProviderConnectionError) as exc_info:
            await VLLMProvider.create(base_url="http://nonexistent:99999/v1")
        assert "Cannot connect to vLLM server" in str(exc_info.value.message)

    @pytest.mark.asyncio
    async def test_create_provides_suggestion(self):
        """Test factory method provides helpful suggestion on failure."""
        with pytest.raises(ProviderConnectionError) as exc_info:
            await VLLMProvider.create(base_url="http://nonexistent:99999/v1")
        assert "Start vLLM server" in str(exc_info.value.details.get("suggestion", ""))


# =============================================================================
# CLIENT TESTS
# =============================================================================


class TestVLLMProviderClient:
    """Tests for VLLMProvider HTTP client."""

    def test_client_initialized(self, vllm_provider):
        """Test HTTP client is initialized."""
        assert vllm_provider.client is not None

    def test_client_timeout_configured(self, vllm_provider):
        """Test client timeout is configured."""
        # The provider was created with timeout=120
        assert vllm_provider.timeout == 120


# =============================================================================
# AVAILABLE MODEL TESTS
# =============================================================================


class TestVLLMAvailableModel:
    """Tests for available model tracking."""

    def test_available_model_initially_none(self, vllm_provider):
        """Test available model is None initially."""
        assert vllm_provider._available_model is None

    @pytest.mark.asyncio
    async def test_available_model_set_on_connect(self):
        """Test available model is set when connecting to server."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": [{"id": "Qwen/Qwen2.5-Coder-7B-Instruct"}]}

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response
            provider = await VLLMProvider.create(base_url="http://localhost:8000")
            assert provider._available_model == "Qwen/Qwen2.5-Coder-7B-Instruct"


# =============================================================================
# MODEL CAPABILITIES TESTS
# =============================================================================


class TestVLLMModelCapabilities:
    """Tests for model capability detection."""

    def test_qwen_instruct_has_tools(self, vllm_provider):
        """Test Qwen instruct model has tool support."""
        model = "Qwen/Qwen2.5-Coder-7B-Instruct"
        assert _model_supports_tools(model) is True

    def test_codellama_has_tools(self, vllm_provider):
        """Test CodeLlama instruct has tool support."""
        model = "codellama/CodeLlama-34b-Instruct-hf"
        assert _model_supports_tools(model) is True

    def test_base_model_no_tools(self, vllm_provider):
        """Test base model (no instruct) lacks tool support."""
        model = "meta-llama/Llama-2-7b-hf"
        assert _model_supports_tools(model) is False
