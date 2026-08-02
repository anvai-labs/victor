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

"""Tests for provider adapters.

These tests verify that provider-specific behaviors are correctly handled
by the adapter layer, ensuring proper normalization across different LLM providers.
"""

import pytest
from victor.agent.tool_calling.base import ToolCall
from victor.protocols.provider_adapter import (
    BaseProviderAdapter,
    DeepSeekAdapter,
    GrokAdapter,
    OpenAIAdapter,
    AnthropicAdapter,
    OllamaAdapter,
    GoogleAdapter,
    LMStudioAdapter,
    VLLMAdapter,
    GroqAdapter,
    MistralAdapter,
    MoonshotAdapter,
    TogetherAdapter,
    FireworksAdapter,
    AzureOpenAIAdapter,
    BedrockAdapter,
    VertexAIAdapter,
    CerebrasAdapter,
    HuggingFaceAdapter,
    ReplicateAdapter,
    OpenRouterAdapter,
    ProviderCapabilities,
    ToolCallFormat,
    get_provider_adapter,
    register_provider_adapter,
)


class TestProviderCapabilities:
    """Tests for ProviderCapabilities dataclass."""

    def test_default_values(self):
        """Test that default capabilities are sensible."""
        caps = ProviderCapabilities()

        assert caps.quality_threshold == 0.80
        assert caps.supports_thinking_tags is False
        assert caps.continuation_markers == []
        assert caps.max_continuation_attempts == 5
        assert caps.tool_call_format == ToolCallFormat.OPENAI

    def test_custom_values(self):
        """Test that custom capabilities can be set."""
        caps = ProviderCapabilities(
            quality_threshold=0.70,
            supports_thinking_tags=True,
            continuation_markers=["..."],
            max_continuation_attempts=3,
        )

        assert caps.quality_threshold == 0.70
        assert caps.supports_thinking_tags is True
        assert caps.continuation_markers == ["..."]
        assert caps.max_continuation_attempts == 3


class TestBaseProviderAdapter:
    """Tests for BaseProviderAdapter default behavior."""

    @pytest.fixture
    def adapter(self):
        return BaseProviderAdapter()

    def test_name(self, adapter):
        """Test adapter name property."""
        assert adapter.name == "base"

    def test_capabilities(self, adapter):
        """Test default capabilities."""
        caps = adapter.capabilities
        assert isinstance(caps, ProviderCapabilities)
        assert caps.quality_threshold == 0.80

    def test_continuation_detection_empty_response(self, adapter):
        """Test continuation detection for empty responses."""
        assert adapter.detect_continuation_needed("") is True
        assert adapter.detect_continuation_needed("   ") is True

    def test_continuation_detection_complete_response(self, adapter):
        """Test continuation detection for complete responses."""
        assert adapter.detect_continuation_needed("Complete response.") is False

    def test_thinking_content_extraction_no_tags(self, adapter):
        """Test thinking extraction without thinking tags."""
        thinking, content = adapter.extract_thinking_content("Regular response")
        assert thinking == ""
        assert content == "Regular response"

    def test_normalize_tool_calls_dict_format(self, adapter):
        """Test tool call normalization from dict format."""
        raw_calls = [
            {
                "id": "call_123",
                "function": {
                    "name": "read_file",
                    "arguments": {"path": "/test.py"},
                },
            }
        ]
        normalized = adapter.normalize_tool_calls(raw_calls)

        assert len(normalized) == 1
        assert normalized[0].id == "call_123"
        assert normalized[0].name == "read_file"
        assert normalized[0].arguments == {"path": "/test.py"}

    def test_should_retry_server_error(self, adapter):
        """Test retry logic for 503 errors."""
        error = Exception("503 Service Unavailable")
        is_retryable, backoff = adapter.should_retry(error)

        assert is_retryable is True
        assert backoff == 5.0

    def test_should_retry_not_retryable(self, adapter):
        """Test retry logic for non-retryable errors."""
        error = Exception("Invalid API key")
        is_retryable, backoff = adapter.should_retry(error)

        assert is_retryable is False
        assert backoff == 0.0


class TestDeepSeekAdapter:
    """Tests for DeepSeek-specific behavior."""

    @pytest.fixture
    def adapter(self):
        return DeepSeekAdapter()

    def test_name(self, adapter):
        """Test adapter name."""
        assert adapter.name == "deepseek"

    def test_capabilities(self, adapter):
        """Test DeepSeek-specific capabilities."""
        caps = adapter.capabilities

        assert caps.quality_threshold == 0.60  # Lower threshold
        assert caps.supports_thinking_tags is True
        assert caps.thinking_tag_format == "<think>...</think>"
        assert "💭 Thinking..." in caps.continuation_markers
        assert caps.max_continuation_attempts == 3  # Fewer attempts

    def test_detect_continuation_thinking_marker(self, adapter):
        """Test continuation detection for thinking marker."""
        assert adapter.detect_continuation_needed("💭 Thinking...") is True
        assert adapter.detect_continuation_needed("Complete response.") is False

    def test_detect_continuation_unclosed_think_tag(self, adapter):
        """Test continuation detection for unclosed think tags."""
        response = "<think>analyzing the code"
        assert adapter.detect_continuation_needed(response) is True

    def test_detect_continuation_closed_think_tag(self, adapter):
        """Test continuation detection for properly closed think tags."""
        response = "<think>analyzing</think>The function does..."
        assert adapter.detect_continuation_needed(response) is False

    def test_extract_thinking_content(self, adapter):
        """Test thinking content extraction from DeepSeek responses."""
        response = "<think>analyzing code structure</think>The function implements..."

        thinking, content = adapter.extract_thinking_content(response)

        assert thinking == "analyzing code structure"
        assert content == "The function implements..."

    def test_extract_thinking_content_multiple_tags(self, adapter):
        """Test thinking extraction with multiple think blocks."""
        response = "<think>first thought</think>Some text<think>second thought</think>More text"

        thinking, content = adapter.extract_thinking_content(response)

        assert "first thought" in thinking
        assert "second thought" in thinking
        assert "Some text" in content
        assert "More text" in content

    def test_extract_thinking_content_no_tags(self, adapter):
        """Test thinking extraction without tags."""
        response = "Regular response without thinking tags"

        thinking, content = adapter.extract_thinking_content(response)

        assert thinking == ""
        assert content == response


class TestGrokAdapter:
    """Tests for xAI Grok-specific behavior."""

    @pytest.fixture
    def adapter(self):
        return GrokAdapter()

    def test_name(self, adapter):
        """Test adapter name."""
        assert adapter.name == "xai"

    def test_capabilities(self, adapter):
        """Test Grok-specific capabilities."""
        caps = adapter.capabilities

        assert caps.quality_threshold == 0.80
        assert caps.supports_thinking_tags is False
        assert caps.continuation_markers == []  # Grok handles internally
        assert caps.output_deduplication is True  # Key feature

    def test_detect_continuation_rarely_needed(self, adapter):
        """Test that Grok rarely needs continuation."""
        assert adapter.detect_continuation_needed("Any response") is False
        assert adapter.detect_continuation_needed("") is True  # Only for empty


class TestOpenAIAdapter:
    """Tests for OpenAI provider adapter."""

    @pytest.fixture
    def adapter(self):
        return OpenAIAdapter()

    def test_name(self, adapter):
        """Test adapter name."""
        assert adapter.name == "openai"

    def test_capabilities(self, adapter):
        """Test OpenAI capabilities."""
        caps = adapter.capabilities

        assert caps.quality_threshold == 0.80
        assert caps.tool_call_format == ToolCallFormat.OPENAI
        assert caps.supports_parallel_tools is True


class TestAnthropicAdapter:
    """Tests for Anthropic Claude adapter."""

    @pytest.fixture
    def adapter(self):
        return AnthropicAdapter()

    def test_name(self, adapter):
        """Test adapter name."""
        assert adapter.name == "anthropic"

    def test_capabilities(self, adapter):
        """Test Anthropic capabilities."""
        caps = adapter.capabilities

        assert caps.quality_threshold == 0.85  # Higher for Claude
        assert caps.tool_call_format == ToolCallFormat.ANTHROPIC

    def test_normalize_tool_calls_anthropic_format(self, adapter):
        """Test tool call normalization from Anthropic content block format."""
        raw_calls = [
            {
                "type": "tool_use",
                "id": "toolu_123",
                "name": "read_file",
                "input": {"path": "/test.py"},
            }
        ]
        normalized = adapter.normalize_tool_calls(raw_calls)

        assert len(normalized) == 1
        assert normalized[0].id == "toolu_123"
        assert normalized[0].name == "read_file"
        assert normalized[0].arguments == {"path": "/test.py"}


class TestOllamaAdapter:
    """Tests for Ollama local model adapter."""

    @pytest.fixture
    def adapter(self):
        return OllamaAdapter()

    def test_name(self, adapter):
        """Test adapter name."""
        assert adapter.name == "ollama"

    def test_capabilities(self, adapter):
        """Test Ollama capabilities."""
        caps = adapter.capabilities

        assert caps.quality_threshold == 0.55  # Lower for local models
        assert caps.tool_call_format == ToolCallFormat.FALLBACK
        assert caps.supports_parallel_tools is False


class TestGetProviderAdapter:
    """Tests for adapter registry and factory."""

    def test_get_deepseek_adapter(self):
        """Test getting DeepSeek adapter."""
        adapter = get_provider_adapter("deepseek")
        assert isinstance(adapter, DeepSeekAdapter)

    def test_get_grok_adapter(self):
        """Test getting Grok adapter."""
        adapter = get_provider_adapter("xai")
        assert isinstance(adapter, GrokAdapter)

    def test_get_grok_by_alias(self):
        """Test getting Grok adapter by 'grok' name."""
        adapter = get_provider_adapter("grok")
        assert isinstance(adapter, GrokAdapter)

    def test_get_unknown_adapter_returns_base(self):
        """Test that unknown providers get base adapter."""
        adapter = get_provider_adapter("unknown_provider")
        assert isinstance(adapter, BaseProviderAdapter)
        assert adapter.name == "base"

    def test_case_insensitive_lookup(self):
        """Test that adapter lookup is case-insensitive."""
        adapter = get_provider_adapter("DeepSeek")
        assert isinstance(adapter, DeepSeekAdapter)

        adapter = get_provider_adapter("OPENAI")
        assert isinstance(adapter, OpenAIAdapter)


class TestRegisterProviderAdapter:
    """Tests for custom adapter registration."""

    def test_register_custom_adapter(self):
        """Test registering a custom provider adapter."""

        class CustomAdapter(BaseProviderAdapter):
            @property
            def name(self) -> str:
                return "custom"

            @property
            def capabilities(self) -> ProviderCapabilities:
                return ProviderCapabilities(quality_threshold=0.60)

        register_provider_adapter("custom_provider", CustomAdapter)
        adapter = get_provider_adapter("custom_provider")

        assert adapter.name == "custom"
        assert adapter.capabilities.quality_threshold == 0.60


class TestToolCall:
    """Tests for ToolCall dataclass."""

    def test_tool_call_creation(self):
        """Test creating a ToolCall instance."""
        call = ToolCall(
            id="call_123",
            name="read_file",
            arguments={"path": "/test.py"},
        )

        assert call.id == "call_123"
        assert call.name == "read_file"
        assert call.arguments == {"path": "/test.py"}
        assert call.raw is None

    def test_tool_call_with_raw(self):
        """Test creating a ToolCall with raw data."""
        raw_data = {"original": "data"}
        call = ToolCall(
            id="call_123",
            name="read_file",
            arguments={},
            raw=raw_data,
        )

        assert call.raw == raw_data


class TestGoogleAdapter:
    """Tests for Google/Gemini provider adapter."""

    @pytest.fixture
    def adapter(self):
        return GoogleAdapter()

    def test_name(self, adapter):
        """Test adapter name."""
        assert adapter.name == "google"

    def test_capabilities(self, adapter):
        """Test Google-specific capabilities."""
        caps = adapter.capabilities

        assert caps.quality_threshold == 0.80
        assert caps.tool_call_format == ToolCallFormat.NATIVE
        assert caps.supports_parallel_tools is True
        assert caps.streaming_chunk_size == 2048

    def test_normalize_tool_calls_function_call_format(self, adapter):
        """Test tool call normalization from Google FunctionCall format."""
        raw_calls = [
            {
                "function_call": {
                    "name": "read_file",
                    "args": {"path": "/test.py"},
                }
            }
        ]
        normalized = adapter.normalize_tool_calls(raw_calls)

        assert len(normalized) == 1
        assert normalized[0].name == "read_file"
        assert normalized[0].arguments == {"path": "/test.py"}

    def test_normalize_tool_calls_direct_format(self, adapter):
        """Test tool call normalization from direct name/args format."""
        raw_calls = [
            {
                "name": "list_files",
                "args": {"directory": "/src"},
            }
        ]
        normalized = adapter.normalize_tool_calls(raw_calls)

        assert len(normalized) == 1
        assert normalized[0].name == "list_files"
        assert normalized[0].arguments == {"directory": "/src"}


class TestLMStudioAdapter:
    """Tests for LMStudio local inference adapter."""

    @pytest.fixture
    def adapter(self):
        return LMStudioAdapter()

    def test_name(self, adapter):
        """Test adapter name."""
        assert adapter.name == "lmstudio"

    def test_capabilities(self, adapter):
        """Test LMStudio-specific capabilities."""
        caps = adapter.capabilities

        assert caps.quality_threshold == 0.75  # Updated for higher-quality local models
        assert caps.tool_call_format == ToolCallFormat.OPENAI
        assert caps.supports_parallel_tools is False
        assert caps.max_continuation_attempts == 3
        assert "..." in caps.continuation_markers

    def test_should_retry_connection_error(self, adapter):
        """Test retry logic for connection refused errors."""
        error = Exception("Connection refused")
        is_retryable, backoff = adapter.should_retry(error)

        assert is_retryable is True
        assert backoff == 5.0  # LMStudio uses 5.0s backoff for connection issues

    def test_should_retry_model_loading(self, adapter):
        """Test retry logic for model loading errors."""
        error = Exception("model is loading")
        is_retryable, backoff = adapter.should_retry(error)

        assert is_retryable is True
        assert backoff == 15.0  # Longer wait for model loading


class TestVLLMAdapter:
    """Tests for vLLM high-throughput inference adapter."""

    @pytest.fixture
    def adapter(self):
        return VLLMAdapter()

    def test_name(self, adapter):
        """Test adapter name."""
        assert adapter.name == "vllm"

    def test_capabilities(self, adapter):
        """Test vLLM-specific capabilities."""
        caps = adapter.capabilities

        assert caps.quality_threshold == 0.75
        assert caps.tool_call_format == ToolCallFormat.OPENAI
        assert caps.supports_parallel_tools is True
        assert caps.streaming_chunk_size == 512
        assert caps.max_continuation_attempts == 4


class TestGroqAdapter:
    """Tests for Groq LPU inference adapter."""

    @pytest.fixture
    def adapter(self):
        return GroqAdapter()

    def test_name(self, adapter):
        """Test adapter name."""
        assert adapter.name == "groqcloud"  # Registry name is groqcloud

    def test_capabilities(self, adapter):
        """Test Groq-specific capabilities."""
        caps = adapter.capabilities

        assert caps.quality_threshold == 0.78  # Actual threshold
        assert caps.tool_call_format == ToolCallFormat.OPENAI
        assert caps.supports_parallel_tools is True
        assert caps.streaming_chunk_size == 1024

    def test_should_retry_rate_limit(self, adapter):
        """Test retry logic with longer backoff for Groq rate limiting."""
        error = Exception("rate_limit_exceeded")
        is_retryable, backoff = adapter.should_retry(error)

        assert is_retryable is True
        assert backoff == 30.0  # Longer backoff for free tier


class TestMistralAdapter:
    """Tests for Mistral AI adapter."""

    @pytest.fixture
    def adapter(self):
        return MistralAdapter()

    def test_name(self, adapter):
        """Test adapter name."""
        assert adapter.name == "mistral"

    def test_capabilities(self, adapter):
        """Test Mistral-specific capabilities."""
        caps = adapter.capabilities

        assert caps.quality_threshold == 0.80
        assert caps.tool_call_format == ToolCallFormat.OPENAI
        assert caps.supports_parallel_tools is True


class TestMoonshotAdapter:
    """Tests for Moonshot/Kimi AI adapter."""

    @pytest.fixture
    def adapter(self):
        return MoonshotAdapter()

    def test_name(self, adapter):
        """Test adapter name."""
        assert adapter.name == "moonshot"

    def test_capabilities(self, adapter):
        """Test Moonshot-specific capabilities."""
        caps = adapter.capabilities

        assert caps.quality_threshold == 0.78  # Actual threshold
        assert caps.tool_call_format == ToolCallFormat.OPENAI
        assert caps.supports_parallel_tools is True
        assert caps.streaming_chunk_size == 2048  # Large context = larger chunks


class TestTogetherAdapter:
    """Tests for Together AI adapter."""

    @pytest.fixture
    def adapter(self):
        return TogetherAdapter()

    def test_name(self, adapter):
        """Test adapter name."""
        assert adapter.name == "together"

    def test_capabilities(self, adapter):
        """Test Together-specific capabilities."""
        caps = adapter.capabilities

        assert caps.quality_threshold == 0.75
        assert caps.tool_call_format == ToolCallFormat.OPENAI
        assert caps.supports_parallel_tools is True


class TestFireworksAdapter:
    """Tests for Fireworks AI adapter."""

    @pytest.fixture
    def adapter(self):
        return FireworksAdapter()

    def test_name(self, adapter):
        """Test adapter name."""
        assert adapter.name == "fireworks"

    def test_capabilities(self, adapter):
        """Test Fireworks-specific capabilities."""
        caps = adapter.capabilities

        assert caps.quality_threshold == 0.75
        assert caps.tool_call_format == ToolCallFormat.OPENAI
        assert caps.supports_parallel_tools is True


class TestAzureOpenAIAdapter:
    """Tests for Azure OpenAI adapter."""

    @pytest.fixture
    def adapter(self):
        return AzureOpenAIAdapter()

    def test_name(self, adapter):
        """Test adapter name."""
        assert adapter.name == "azure-openai"

    def test_capabilities(self, adapter):
        """Test Azure OpenAI-specific capabilities."""
        caps = adapter.capabilities

        assert caps.quality_threshold == 0.80
        assert caps.tool_call_format == ToolCallFormat.OPENAI
        assert caps.supports_parallel_tools is True


class TestBedrockAdapter:
    """Tests for AWS Bedrock adapter."""

    @pytest.fixture
    def adapter(self):
        return BedrockAdapter()

    def test_name(self, adapter):
        """Test adapter name."""
        assert adapter.name == "bedrock"

    def test_capabilities(self, adapter):
        """Test Bedrock-specific capabilities."""
        caps = adapter.capabilities

        assert caps.quality_threshold == 0.82  # Actual threshold
        assert caps.tool_call_format == ToolCallFormat.ANTHROPIC
        assert caps.supports_parallel_tools is True


class TestVertexAIAdapter:
    """Tests for Google Vertex AI adapter."""

    @pytest.fixture
    def adapter(self):
        return VertexAIAdapter()

    def test_name(self, adapter):
        """Test adapter name."""
        assert adapter.name == "vertexai"

    def test_capabilities(self, adapter):
        """Test Vertex AI-specific capabilities."""
        caps = adapter.capabilities

        assert caps.quality_threshold == 0.80
        assert caps.tool_call_format == ToolCallFormat.NATIVE
        assert caps.supports_parallel_tools is True


class TestCerebrasAdapter:
    """Tests for Cerebras adapter."""

    @pytest.fixture
    def adapter(self):
        return CerebrasAdapter()

    def test_name(self, adapter):
        """Test adapter name."""
        assert adapter.name == "cerebras"

    def test_capabilities(self, adapter):
        """Test Cerebras-specific capabilities."""
        caps = adapter.capabilities

        assert caps.quality_threshold == 0.78  # High quality models
        assert caps.tool_call_format == ToolCallFormat.OPENAI
        assert caps.supports_parallel_tools is True
        assert caps.output_deduplication is True  # Some models repeat content
        assert caps.supports_thinking_tags is True  # Qwen-3 has inline thinking


class TestHuggingFaceAdapter:
    """Tests for Hugging Face adapter."""

    @pytest.fixture
    def adapter(self):
        return HuggingFaceAdapter()

    def test_name(self, adapter):
        """Test adapter name."""
        assert adapter.name == "huggingface"

    def test_capabilities(self, adapter):
        """Test Hugging Face-specific capabilities."""
        caps = adapter.capabilities

        assert caps.quality_threshold == 0.70
        assert caps.tool_call_format == ToolCallFormat.FALLBACK
        assert caps.supports_parallel_tools is False


class TestReplicateAdapter:
    """Tests for Replicate adapter."""

    @pytest.fixture
    def adapter(self):
        return ReplicateAdapter()

    def test_name(self, adapter):
        """Test adapter name."""
        assert adapter.name == "replicate"

    def test_capabilities(self, adapter):
        """Test Replicate-specific capabilities."""
        caps = adapter.capabilities

        assert caps.quality_threshold == 0.72  # Actual threshold
        assert caps.tool_call_format == ToolCallFormat.FALLBACK
        assert caps.supports_parallel_tools is False

    def test_should_retry_cold_start(self, adapter):
        """Test retry logic for cold start delays."""
        error = Exception("cold start in progress")
        is_retryable, backoff = adapter.should_retry(error)

        assert is_retryable is True
        assert backoff == 15.0  # Actual backoff is 15.0s


class TestOpenRouterAdapter:
    """Tests for OpenRouter adapter."""

    @pytest.fixture
    def adapter(self):
        return OpenRouterAdapter()

    def test_name(self, adapter):
        """Test adapter name."""
        assert adapter.name == "openrouter"

    def test_capabilities(self, adapter):
        """Test OpenRouter-specific capabilities."""
        caps = adapter.capabilities

        assert caps.quality_threshold == 0.78
        assert caps.tool_call_format == ToolCallFormat.OPENAI
        assert caps.supports_parallel_tools is True


class TestAdapterRegistryAllProviders:
    """Tests for getting all provider adapters from registry."""

    def test_get_google_adapter(self):
        """Test getting Google adapter."""
        adapter = get_provider_adapter("google")
        assert isinstance(adapter, GoogleAdapter)

    def test_get_gemini_alias(self):
        """Test getting Google adapter via 'gemini' alias."""
        adapter = get_provider_adapter("gemini")
        assert isinstance(adapter, GoogleAdapter)

    def test_get_lmstudio_adapter(self):
        """Test getting LMStudio adapter."""
        adapter = get_provider_adapter("lmstudio")
        assert isinstance(adapter, LMStudioAdapter)

    def test_get_vllm_adapter(self):
        """Test getting vLLM adapter."""
        adapter = get_provider_adapter("vllm")
        assert isinstance(adapter, VLLMAdapter)

    def test_get_groq_adapter(self):
        """Test getting Groq adapter."""
        adapter = get_provider_adapter("groqcloud")
        assert isinstance(adapter, GroqAdapter)

    def test_get_groq_alias(self):
        """Test getting Groq adapter via 'groq' alias."""
        adapter = get_provider_adapter("groq")
        assert isinstance(adapter, GroqAdapter)

    def test_get_mistral_adapter(self):
        """Test getting Mistral adapter."""
        adapter = get_provider_adapter("mistral")
        assert isinstance(adapter, MistralAdapter)

    def test_get_moonshot_adapter(self):
        """Test getting Moonshot adapter."""
        adapter = get_provider_adapter("moonshot")
        assert isinstance(adapter, MoonshotAdapter)

    def test_get_kimi_alias(self):
        """Test getting Moonshot adapter via 'kimi' alias."""
        adapter = get_provider_adapter("kimi")
        assert isinstance(adapter, MoonshotAdapter)

    def test_get_together_adapter(self):
        """Test getting Together adapter."""
        adapter = get_provider_adapter("together")
        assert isinstance(adapter, TogetherAdapter)

    def test_get_fireworks_adapter(self):
        """Test getting Fireworks adapter."""
        adapter = get_provider_adapter("fireworks")
        assert isinstance(adapter, FireworksAdapter)

    def test_get_azure_openai_adapter(self):
        """Test getting Azure OpenAI adapter."""
        adapter = get_provider_adapter("azure-openai")
        assert isinstance(adapter, AzureOpenAIAdapter)

    def test_get_azure_alias(self):
        """Test getting Azure OpenAI adapter via 'azure' alias."""
        adapter = get_provider_adapter("azure")
        assert isinstance(adapter, AzureOpenAIAdapter)

    def test_get_bedrock_adapter(self):
        """Test getting Bedrock adapter."""
        adapter = get_provider_adapter("bedrock")
        assert isinstance(adapter, BedrockAdapter)

    def test_get_aws_alias(self):
        """Test getting Bedrock adapter via 'aws' alias."""
        adapter = get_provider_adapter("aws")
        assert isinstance(adapter, BedrockAdapter)

    def test_get_vertexai_adapter(self):
        """Test getting Vertex AI adapter."""
        adapter = get_provider_adapter("vertexai")
        assert isinstance(adapter, VertexAIAdapter)

    def test_get_vertex_alias(self):
        """Test getting Vertex AI adapter via 'vertex' alias."""
        adapter = get_provider_adapter("vertex")
        assert isinstance(adapter, VertexAIAdapter)

    def test_get_cerebras_adapter(self):
        """Test getting Cerebras adapter."""
        adapter = get_provider_adapter("cerebras")
        assert isinstance(adapter, CerebrasAdapter)

    def test_get_huggingface_adapter(self):
        """Test getting Hugging Face adapter."""
        adapter = get_provider_adapter("huggingface")
        assert isinstance(adapter, HuggingFaceAdapter)

    def test_get_hf_alias(self):
        """Test getting Hugging Face adapter via 'hf' alias."""
        adapter = get_provider_adapter("hf")
        assert isinstance(adapter, HuggingFaceAdapter)

    def test_get_replicate_adapter(self):
        """Test getting Replicate adapter."""
        adapter = get_provider_adapter("replicate")
        assert isinstance(adapter, ReplicateAdapter)

    def test_get_openrouter_adapter(self):
        """Test getting OpenRouter adapter."""
        adapter = get_provider_adapter("openrouter")
        assert isinstance(adapter, OpenRouterAdapter)
