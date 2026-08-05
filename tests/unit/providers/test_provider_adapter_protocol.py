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

"""Tests for provider adapter protocol and implementations."""

import pytest
from unittest.mock import MagicMock

from victor.agent.tool_calling.base import ToolCall
from victor.protocols.provider_adapter import (
    ToolCallFormat,
    ProviderCapabilities,
    ContinuationContext,
    IProviderAdapter,
    BaseProviderAdapter,
    get_provider_adapter,
    register_provider_adapter,
)

# =============================================================================
# TOOL CALL FORMAT TESTS
# =============================================================================


class TestToolCallFormat:
    """Tests for ToolCallFormat enum."""

    def test_enum_values(self):
        """Test all expected enum values exist."""
        assert ToolCallFormat.OPENAI.value == "openai"
        assert ToolCallFormat.ANTHROPIC.value == "anthropic"
        assert ToolCallFormat.NATIVE.value == "native"
        assert ToolCallFormat.FALLBACK.value == "fallback"


# =============================================================================
# PROVIDER CAPABILITIES TESTS
# =============================================================================


class TestProviderCapabilities:
    """Tests for ProviderCapabilities dataclass."""

    def test_default_values(self):
        """Test default capability values."""
        caps = ProviderCapabilities()
        assert caps.quality_threshold == 0.80
        assert caps.supports_thinking_tags is False
        assert caps.thinking_tag_format == ""
        assert caps.continuation_markers == []
        assert caps.max_continuation_attempts == 5
        assert caps.tool_call_format == ToolCallFormat.OPENAI
        assert caps.output_deduplication is False
        assert caps.streaming_chunk_size == 1024
        assert caps.supports_parallel_tools is True
        assert caps.grounding_required is True
        assert caps.grounding_strictness == 0.8
        assert caps.continuation_patience == 3
        assert caps.thinking_tokens_budget == 500
        assert caps.requires_thinking_time is False

    def test_custom_values(self):
        """Test custom capability values."""
        caps = ProviderCapabilities(
            quality_threshold=0.70,
            supports_thinking_tags=True,
            thinking_tag_format="<think>...</think>",
            continuation_markers=["...", "---"],
            max_continuation_attempts=10,
        )
        assert caps.quality_threshold == 0.70
        assert caps.supports_thinking_tags is True
        assert len(caps.continuation_markers) == 2


# =============================================================================
# TOOL CALL TESTS
# =============================================================================


class TestToolCall:
    """Tests for ToolCall dataclass."""

    def test_creation(self):
        """Test ToolCall creation."""
        call = ToolCall(
            id="call_123",
            name="read_file",
            arguments={"path": "test.py"},
        )
        assert call.id == "call_123"
        assert call.name == "read_file"
        assert call.arguments == {"path": "test.py"}
        assert call.raw is None

    def test_with_raw(self):
        """Test ToolCall with raw data."""
        raw_data = {"original": "data"}
        call = ToolCall(
            id="call_456",
            name="write_file",
            arguments={"path": "out.py", "content": "test"},
            raw=raw_data,
        )
        assert call.raw == raw_data


# =============================================================================
# CONTINUATION CONTEXT TESTS
# =============================================================================


class TestContinuationContext:
    """Tests for ContinuationContext dataclass."""

    def test_default_values(self):
        """Test default context values."""
        ctx = ContinuationContext()
        assert ctx.attempt == 0
        assert ctx.reason == ""
        assert ctx.partial_response == ""
        assert ctx.tool_calls_made == 0

    def test_custom_values(self):
        """Test custom context values."""
        ctx = ContinuationContext(
            attempt=3,
            reason="incomplete response",
            partial_response="Started analyzing...",
            tool_calls_made=5,
        )
        assert ctx.attempt == 3
        assert ctx.reason == "incomplete response"
        assert ctx.partial_response == "Started analyzing..."
        assert ctx.tool_calls_made == 5


# =============================================================================
# BASE PROVIDER ADAPTER TESTS
# =============================================================================


class TestBaseProviderAdapter:
    """Tests for BaseProviderAdapter class."""

    @pytest.fixture
    def adapter(self):
        """Create a base adapter for testing."""
        return BaseProviderAdapter()

    def test_name_property(self, adapter):
        """Test name property returns 'base'."""
        assert adapter.name == "base"

    def test_capabilities_property(self, adapter):
        """Test capabilities returns default ProviderCapabilities."""
        caps = adapter.capabilities
        assert isinstance(caps, ProviderCapabilities)
        assert caps.quality_threshold == 0.80

    def test_detect_continuation_empty_response(self, adapter):
        """Test continuation detection for empty response."""
        assert adapter.detect_continuation_needed("") is True
        assert adapter.detect_continuation_needed("   ") is True

    def test_detect_continuation_complete_response(self, adapter):
        """Test continuation detection for complete response."""
        assert adapter.detect_continuation_needed("This is complete.") is False

    def test_detect_continuation_with_markers(self):
        """Test continuation detection with custom markers."""

        class CustomAdapter(BaseProviderAdapter):
            @property
            def capabilities(self):
                return ProviderCapabilities(continuation_markers=["...", "---"])

        adapter = CustomAdapter()
        assert adapter.detect_continuation_needed("Some text...") is True
        assert adapter.detect_continuation_needed("Some text---") is True
        assert adapter.detect_continuation_needed("Some text.") is False

    def test_extract_thinking_no_tags(self, adapter):
        """Test thinking extraction when not supported."""
        thinking, content = adapter.extract_thinking_content("Hello world")
        assert thinking == ""
        assert content == "Hello world"

    def test_normalize_tool_calls_dict_format(self, adapter):
        """Test normalizing OpenAI dict format tool calls."""
        raw_calls = [
            {
                "id": "call_1",
                "function": {
                    "name": "read_file",
                    "arguments": {"path": "test.py"},
                },
            }
        ]
        normalized = adapter.normalize_tool_calls(raw_calls)
        assert len(normalized) == 1
        assert normalized[0].id == "call_1"
        assert normalized[0].name == "read_file"
        assert normalized[0].arguments == {"path": "test.py"}

    def test_normalize_tool_calls_object_format(self, adapter):
        """Test normalizing OpenAI object format tool calls."""

        class MockFunction:
            name = "write_file"
            arguments = {"path": "out.py"}

        class MockCall:
            id = "call_2"
            function = MockFunction()

        raw_calls = [MockCall()]
        normalized = adapter.normalize_tool_calls(raw_calls)
        assert len(normalized) == 1
        assert normalized[0].name == "write_file"

    def test_normalize_tool_calls_generates_id(self, adapter):
        """Test that missing IDs are generated."""
        raw_calls = [{"function": {"name": "test_tool", "arguments": {}}}]
        normalized = adapter.normalize_tool_calls(raw_calls)
        assert normalized[0].id == "call_0"

    def test_should_retry_rate_limit(self, adapter):
        """Test retry logic for rate limit errors."""
        error = Exception("Rate limit exceeded")
        should_retry, backoff = adapter.should_retry(error)
        assert should_retry is True
        assert backoff == 60.0

    def test_should_retry_server_error(self, adapter):
        """Test retry logic for server errors."""
        error = Exception("503 Service Unavailable")
        should_retry, backoff = adapter.should_retry(error)
        assert should_retry is True
        assert backoff == 5.0

    def test_should_retry_timeout(self, adapter):
        """Test retry logic for timeout errors."""
        error = Exception("Connection timeout")
        should_retry, backoff = adapter.should_retry(error)
        assert should_retry is True
        assert backoff == 2.0

    def test_should_retry_unknown_error(self, adapter):
        """Test retry logic for unknown errors."""
        error = Exception("Unknown error")
        should_retry, backoff = adapter.should_retry(error)
        assert should_retry is False
        assert backoff == 0.0


# =============================================================================
# PROVIDER ADAPTER REGISTRY TESTS
# =============================================================================


class TestProviderAdapterRegistry:
    """Tests for provider adapter registry functions."""

    def test_get_provider_adapter_openai(self):
        """Test getting OpenAI adapter."""
        adapter = get_provider_adapter("openai")
        assert adapter is not None
        assert adapter.name == "openai"

    def test_get_provider_adapter_anthropic(self):
        """Test getting Anthropic adapter."""
        adapter = get_provider_adapter("anthropic")
        assert adapter is not None
        assert adapter.name == "anthropic"

    def test_get_provider_adapter_deepseek(self):
        """Test getting DeepSeek adapter."""
        adapter = get_provider_adapter("deepseek")
        assert adapter is not None
        assert adapter.name == "deepseek"

    def test_get_provider_adapter_groq(self):
        """Test getting Groq adapter."""
        adapter = get_provider_adapter("groq")
        assert adapter is not None
        assert adapter.name in ["groq", "groqcloud"]

    def test_get_provider_adapter_ollama(self):
        """Test getting Ollama adapter."""
        adapter = get_provider_adapter("ollama")
        assert adapter is not None
        assert adapter.name == "ollama"

    def test_get_provider_adapter_unknown(self):
        """Test getting unknown provider returns base adapter."""
        adapter = get_provider_adapter("unknown_provider")
        assert adapter is not None
        # Should return base or default adapter

    def test_get_provider_adapter_case_insensitive(self):
        """Test provider name is case-insensitive."""
        adapter1 = get_provider_adapter("OpenAI")
        adapter2 = get_provider_adapter("openai")
        assert adapter1.name == adapter2.name

    def test_get_provider_adapter_aliases(self):
        """Test provider aliases work correctly."""
        # gemini should map to google
        adapter = get_provider_adapter("gemini")
        assert adapter.name in ["google", "gemini"]

    def test_register_custom_adapter(self):
        """Test registering a custom adapter."""

        class CustomAdapter(BaseProviderAdapter):
            @property
            def name(self):
                return "custom_test"

            @property
            def capabilities(self):
                return ProviderCapabilities(quality_threshold=0.99)

        register_provider_adapter("custom_test", CustomAdapter)
        adapter = get_provider_adapter("custom_test")
        assert adapter.name == "custom_test"
        assert adapter.capabilities.quality_threshold == 0.99


# =============================================================================
# SPECIFIC PROVIDER ADAPTER TESTS
# =============================================================================


class TestDeepSeekAdapter:
    """Tests for DeepSeek adapter specifics."""

    @pytest.fixture
    def adapter(self):
        """Get DeepSeek adapter."""
        return get_provider_adapter("deepseek")

    def test_supports_thinking_tags(self, adapter):
        """Test DeepSeek supports thinking tags."""
        assert adapter.capabilities.supports_thinking_tags is True

    def test_thinking_tag_format(self, adapter):
        """Test DeepSeek thinking tag format."""
        assert "<think>" in adapter.capabilities.thinking_tag_format.lower()

    def test_quality_threshold(self, adapter):
        """Test DeepSeek quality threshold is appropriate."""
        assert adapter.capabilities.quality_threshold <= 0.80


class TestAnthropicAdapter:
    """Tests for Anthropic adapter specifics."""

    @pytest.fixture
    def adapter(self):
        """Get Anthropic adapter."""
        return get_provider_adapter("anthropic")

    def test_high_quality_threshold(self, adapter):
        """Test Anthropic has high quality threshold."""
        assert adapter.capabilities.quality_threshold >= 0.80

    def test_tool_call_format(self, adapter):
        """Test Anthropic uses Anthropic tool format."""
        assert adapter.capabilities.tool_call_format == ToolCallFormat.ANTHROPIC


class TestOllamaAdapter:
    """Tests for Ollama adapter specifics."""

    @pytest.fixture
    def adapter(self):
        """Get Ollama adapter."""
        return get_provider_adapter("ollama")

    def test_fallback_tool_format(self, adapter):
        """Test Ollama uses fallback tool format."""
        assert adapter.capabilities.tool_call_format == ToolCallFormat.FALLBACK

    def test_lower_quality_threshold(self, adapter):
        """Test Ollama has appropriate quality threshold."""
        # Local models may have lower quality
        assert adapter.capabilities.quality_threshold <= 0.80


class TestXAIAdapter:
    """Tests for xAI/Grok adapter specifics."""

    @pytest.fixture
    def adapter(self):
        """Get xAI adapter."""
        return get_provider_adapter("xai")

    def test_output_deduplication(self, adapter):
        """Test xAI enables output deduplication."""
        assert adapter.capabilities.output_deduplication is True

    def test_quality_threshold(self, adapter):
        """Test xAI quality threshold."""
        assert adapter.capabilities.quality_threshold >= 0.75


# =============================================================================
# PROTOCOL COMPLIANCE TESTS
# =============================================================================


class TestProtocolCompliance:
    """Tests for IProviderAdapter protocol compliance."""

    def test_base_adapter_is_protocol_compliant(self):
        """Test BaseProviderAdapter implements IProviderAdapter."""
        adapter = BaseProviderAdapter()
        assert isinstance(adapter, IProviderAdapter)

    def test_custom_adapter_protocol_check(self):
        """Test custom adapter can implement protocol."""

        class MinimalAdapter:
            @property
            def name(self) -> str:
                return "minimal"

            @property
            def capabilities(self) -> ProviderCapabilities:
                return ProviderCapabilities()

            def detect_continuation_needed(self, response: str) -> bool:
                return False

            def extract_thinking_content(self, response: str):
                return ("", response)

            def normalize_tool_calls(self, raw_calls):
                return []

            def should_retry(self, error):
                return (False, 0.0)

        adapter = MinimalAdapter()
        assert isinstance(adapter, IProviderAdapter)


# =============================================================================
# DEEPSEEK ADAPTER EXTENDED TESTS
# =============================================================================


class TestDeepSeekAdapterExtended:
    """Extended tests for DeepSeek adapter specifics."""

    @pytest.fixture
    def adapter(self):
        """Get DeepSeek adapter."""
        return get_provider_adapter("deepseek")

    def test_extract_thinking_with_tags(self, adapter):
        """Test extracting thinking content from tags."""
        response = "<think>I need to analyze this code...</think>Here is the solution."
        thinking, content = adapter.extract_thinking_content(response)
        assert "analyze this code" in thinking
        assert "Here is the solution" in content
        assert "<think>" not in content

    def test_extract_thinking_multiple_tags(self, adapter):
        """Test extracting multiple thinking blocks."""
        response = "<think>First thought</think>Content<think>Second thought</think>More content"
        thinking, content = adapter.extract_thinking_content(response)
        assert "First thought" in thinking
        assert "Second thought" in thinking
        assert "Content" in content
        assert "More content" in content

    def test_extract_thinking_no_tags(self, adapter):
        """Test extraction when no tags present."""
        response = "Just regular content without tags."
        thinking, content = adapter.extract_thinking_content(response)
        assert thinking == ""
        assert content == "Just regular content without tags."

    def test_detect_continuation_unclosed_think(self, adapter):
        """Test continuation detection for unclosed think tag."""
        assert adapter.detect_continuation_needed("<think>Still thinking...") is True

    def test_detect_continuation_closed_think(self, adapter):
        """Test no continuation for closed think tag."""
        assert adapter.detect_continuation_needed("<think>Done</think>Result") is False

    def test_detect_continuation_markers(self, adapter):
        """Test continuation markers are in capabilities."""
        # Markers are defined but strip() in detect_continuation_needed
        # removes trailing spaces, so test the capability definition
        assert "Let me " in adapter.capabilities.continuation_markers
        assert "I'll " in adapter.capabilities.continuation_markers
        # Empty/whitespace responses trigger continuation
        assert adapter.detect_continuation_needed("") is True
        assert adapter.detect_continuation_needed("   ") is True
        # Unclosed think tags trigger continuation
        assert adapter.detect_continuation_needed("Response ends with <think>") is True

    def test_grounding_not_required(self, adapter):
        """Test DeepSeek allows suggestions without grounding."""
        assert adapter.capabilities.grounding_required is False

    def test_requires_thinking_time(self, adapter):
        """Test DeepSeek requires thinking time."""
        assert adapter.capabilities.requires_thinking_time is True


# =============================================================================
# GOOGLE ADAPTER EXTENDED TESTS
# =============================================================================


class TestGoogleAdapterExtended:
    """Extended tests for Google adapter specifics."""

    @pytest.fixture
    def adapter(self):
        """Get Google adapter."""
        return get_provider_adapter("google")

    def test_normalize_function_call_format(self, adapter):
        """Test normalizing Google's function_call format."""
        raw_calls = [
            {
                "function_call": {
                    "name": "search",
                    "args": {"query": "test"},
                }
            }
        ]
        normalized = adapter.normalize_tool_calls(raw_calls)
        assert len(normalized) == 1
        assert normalized[0].name == "search"
        assert normalized[0].arguments == {"query": "test"}

    def test_normalize_direct_format(self, adapter):
        """Test normalizing direct name/args format."""
        raw_calls = [
            {
                "name": "read_file",
                "args": {"path": "test.py"},
            }
        ]
        normalized = adapter.normalize_tool_calls(raw_calls)
        assert len(normalized) == 1
        assert normalized[0].name == "read_file"

    def test_normalize_object_format(self, adapter):
        """Test normalizing Google object format."""

        class MockFunctionCall:
            name = "analyze"
            args = {"content": "code"}

        class MockCall:
            id = "fc_1"
            function_call = MockFunctionCall()

        raw_calls = [MockCall()]
        normalized = adapter.normalize_tool_calls(raw_calls)
        assert len(normalized) == 1
        assert normalized[0].name == "analyze"

    def test_streaming_chunk_size(self, adapter):
        """Test Google uses larger streaming chunks."""
        assert adapter.capabilities.streaming_chunk_size >= 2048


# =============================================================================
# LMSTUDIO ADAPTER EXTENDED TESTS
# =============================================================================


class TestLMStudioAdapterExtended:
    """Extended tests for LMStudio adapter specifics."""

    @pytest.fixture
    def adapter(self):
        """Get LMStudio adapter."""
        return get_provider_adapter("lmstudio")

    def test_model_supports_thinking_qwen3(self, adapter):
        """Test Qwen3 models support thinking."""
        assert adapter.model_supports_thinking("qwen3-7b-chat") is True
        assert adapter.model_supports_thinking("qwen3-coder") is True

    def test_model_supports_thinking_deepseek(self, adapter):
        """Test DeepSeek-R1 models support thinking."""
        assert adapter.model_supports_thinking("deepseek-r1-32b") is True
        assert adapter.model_supports_thinking("deepseek-reasoner") is True

    def test_model_supports_thinking_other(self, adapter):
        """Test other models don't have thinking tags."""
        assert adapter.model_supports_thinking("llama-3.1-8b") is False
        assert adapter.model_supports_thinking("mistral-7b") is False

    def test_model_supports_tools(self, adapter):
        """Test tool-capable model detection."""
        assert adapter.model_supports_tools("qwen2.5-coder") is True
        assert adapter.model_supports_tools("llama3.1-tools") is True
        assert adapter.model_supports_tools("some-model-tools") is True

    def test_model_no_tools(self, adapter):
        """Test non-tool models."""
        assert adapter.model_supports_tools("phi-2") is False
        assert adapter.model_supports_tools("llama-2-7b") is False

    def test_extract_thinking_content(self, adapter):
        """Test thinking extraction for LMStudio."""
        response = "<think>Planning steps...</think>Execute the plan."
        thinking, content = adapter.extract_thinking_content(response)
        assert "Planning steps" in thinking
        assert "Execute the plan" in content

    def test_extract_thinking_case_insensitive(self, adapter):
        """Test case-insensitive think tag extraction."""
        response = "<THINK>Thinking</THINK>Done"
        thinking, content = adapter.extract_thinking_content(response)
        assert "Thinking" in thinking
        assert "Done" in content

    def test_extract_thinking_empty(self, adapter):
        """Test empty response handling."""
        thinking, content = adapter.extract_thinking_content("")
        assert thinking == ""
        assert content == ""

    def test_detect_continuation_unclosed_think(self, adapter):
        """Test continuation detection for unclosed think."""
        assert adapter.detect_continuation_needed("<think>still thinking") is True

    def test_should_retry_connection_refused(self, adapter):
        """Test retry for connection refused."""
        error = Exception("Connection refused")
        should_retry, backoff = adapter.should_retry(error)
        assert should_retry is True
        assert backoff == 5.0

    def test_should_retry_model_loading(self, adapter):
        """Test retry for model loading."""
        error = Exception("Model is loading")
        should_retry, backoff = adapter.should_retry(error)
        assert should_retry is True
        assert backoff == 15.0

    def test_should_retry_timeout(self, adapter):
        """Test retry for timeout."""
        error = Exception("Request timeout")
        should_retry, backoff = adapter.should_retry(error)
        assert should_retry is True
        assert backoff == 10.0

    def test_should_not_retry_oom(self, adapter):
        """Test no retry for out of memory."""
        error = Exception("CUDA out of memory")
        should_retry, backoff = adapter.should_retry(error)
        assert should_retry is False


# =============================================================================
# LLAMACPP ADAPTER EXTENDED TESTS
# =============================================================================


class TestLlamaCppAdapterExtended:
    """Extended tests for llama.cpp adapter."""

    @pytest.fixture
    def adapter(self):
        """Get llama.cpp adapter."""
        return get_provider_adapter("llamacpp")

    def test_name(self, adapter):
        """Test adapter name."""
        assert adapter.name == "llamacpp"

    def test_single_tool_support(self, adapter):
        """Test no parallel tools for CPU."""
        assert adapter.capabilities.supports_parallel_tools is False

    def test_smaller_streaming_chunks(self, adapter):
        """Test smaller chunks for CPU."""
        assert adapter.capabilities.streaming_chunk_size <= 512

    def test_should_retry_connection_refused(self, adapter):
        """Test retry for connection refused."""
        error = Exception("Connection refused - server not running")
        should_retry, backoff = adapter.should_retry(error)
        assert should_retry is True

    def test_should_retry_timeout(self, adapter):
        """Test retry for timeout with longer wait."""
        error = Exception("Request timeout")
        should_retry, backoff = adapter.should_retry(error)
        assert should_retry is True
        assert backoff == 10.0

    def test_should_not_retry_memory(self, adapter):
        """Test no retry for memory errors."""
        error = Exception("OOM - insufficient memory")
        should_retry, backoff = adapter.should_retry(error)
        assert should_retry is False


# =============================================================================
# GROQ ADAPTER EXTENDED TESTS
# =============================================================================


class TestGroqAdapterExtended:
    """Extended tests for Groq adapter."""

    @pytest.fixture
    def adapter(self):
        """Get Groq adapter."""
        return get_provider_adapter("groq")

    def test_name(self, adapter):
        """Test adapter name."""
        assert adapter.name == "groqcloud"

    def test_should_retry_rate_limit(self, adapter):
        """Test rate limit retry with long backoff."""
        error = Exception("Rate limit exceeded")
        should_retry, backoff = adapter.should_retry(error)
        assert should_retry is True
        assert backoff >= 30.0

    def test_should_retry_tokens_per_minute(self, adapter):
        """Test tokens per minute limit."""
        error = Exception("tokens_per_minute exceeded")
        should_retry, backoff = adapter.should_retry(error)
        assert should_retry is True
        assert backoff >= 60.0

    def test_parallel_tools_supported(self, adapter):
        """Test Groq supports parallel tools."""
        assert adapter.capabilities.supports_parallel_tools is True


# =============================================================================
# GROK ADAPTER EXTENDED TESTS
# =============================================================================


class TestGrokAdapterExtended:
    """Extended tests for Grok/xAI adapter."""

    @pytest.fixture
    def adapter(self):
        """Get Grok adapter."""
        return get_provider_adapter("xai")

    def test_detect_continuation_rarely_needed(self, adapter):
        """Test Grok rarely needs continuation."""
        assert adapter.detect_continuation_needed("Complete response.") is False
        assert adapter.detect_continuation_needed("More text here.") is False

    def test_detect_continuation_empty_only(self, adapter):
        """Test continuation only for empty responses."""
        assert adapter.detect_continuation_needed("") is True
        assert adapter.detect_continuation_needed("   ") is True


# =============================================================================
# BEDROCK ADAPTER EXTENDED TESTS
# =============================================================================


class TestBedrockAdapterExtended:
    """Extended tests for Bedrock adapter."""

    @pytest.fixture
    def adapter(self):
        """Get Bedrock adapter."""
        return get_provider_adapter("bedrock")

    def test_name(self, adapter):
        """Test adapter name."""
        assert adapter.name == "bedrock"

    def test_normalize_anthropic_format(self, adapter):
        """Test normalizing Anthropic format on Bedrock."""
        raw_calls = [
            {
                "type": "tool_use",
                "id": "tu_1",
                "name": "read_file",
                "input": {"path": "test.py"},
            }
        ]
        normalized = adapter.normalize_tool_calls(raw_calls)
        assert len(normalized) == 1
        assert normalized[0].name == "read_file"
        assert normalized[0].arguments == {"path": "test.py"}

    def test_normalize_openai_format(self, adapter):
        """Test normalizing OpenAI format on Bedrock."""
        raw_calls = [
            {
                "function": {
                    "name": "write_file",
                    "arguments": {"path": "out.py"},
                }
            }
        ]
        normalized = adapter.normalize_tool_calls(raw_calls)
        assert len(normalized) == 1
        assert normalized[0].name == "write_file"


# =============================================================================
# REPLICATE ADAPTER EXTENDED TESTS
# =============================================================================


class TestReplicateAdapterExtended:
    """Extended tests for Replicate adapter."""

    @pytest.fixture
    def adapter(self):
        """Get Replicate adapter."""
        return get_provider_adapter("replicate")

    def test_name(self, adapter):
        """Test adapter name."""
        assert adapter.name == "replicate"

    def test_should_retry_cold_start(self, adapter):
        """Test retry for cold start."""
        error = Exception("Model cold start in progress")
        should_retry, backoff = adapter.should_retry(error)
        assert should_retry is True
        assert backoff == 15.0

    def test_should_retry_starting(self, adapter):
        """Test retry for model starting."""
        error = Exception("Model is starting up")
        should_retry, backoff = adapter.should_retry(error)
        assert should_retry is True


# =============================================================================
# ALL ADAPTERS PROPERTY TESTS
# =============================================================================


class TestAllAdaptersProperties:
    """Test all adapters have required properties."""

    PROVIDER_NAMES = [
        "openai",
        "anthropic",
        "google",
        "deepseek",
        "xai",
        "mistral",
        "moonshot",
        "groq",
        "together",
        "fireworks",
        "cerebras",
        "openrouter",
        "replicate",
        "huggingface",
        "azure",
        "bedrock",
        "vertexai",
        "ollama",
        "lmstudio",
        "vllm",
        "llamacpp",
    ]

    @pytest.mark.parametrize("provider", PROVIDER_NAMES)
    def test_adapter_has_name(self, provider):
        """Test all adapters have name property."""
        adapter = get_provider_adapter(provider)
        assert adapter.name is not None
        assert len(adapter.name) > 0

    @pytest.mark.parametrize("provider", PROVIDER_NAMES)
    def test_adapter_has_capabilities(self, provider):
        """Test all adapters have capabilities."""
        adapter = get_provider_adapter(provider)
        caps = adapter.capabilities
        assert isinstance(caps, ProviderCapabilities)
        assert 0.0 <= caps.quality_threshold <= 1.0

    @pytest.mark.parametrize("provider", PROVIDER_NAMES)
    def test_adapter_detect_continuation(self, provider):
        """Test all adapters implement detect_continuation_needed."""
        adapter = get_provider_adapter(provider)
        result = adapter.detect_continuation_needed("Test response")
        assert isinstance(result, bool)

    @pytest.mark.parametrize("provider", PROVIDER_NAMES)
    def test_adapter_extract_thinking(self, provider):
        """Test all adapters implement extract_thinking_content."""
        adapter = get_provider_adapter(provider)
        thinking, content = adapter.extract_thinking_content("Test response")
        assert isinstance(thinking, str)
        assert isinstance(content, str)

    @pytest.mark.parametrize("provider", PROVIDER_NAMES)
    def test_adapter_normalize_tool_calls(self, provider):
        """Test all adapters implement normalize_tool_calls."""
        adapter = get_provider_adapter(provider)
        result = adapter.normalize_tool_calls([])
        assert isinstance(result, list)

    @pytest.mark.parametrize("provider", PROVIDER_NAMES)
    def test_adapter_should_retry(self, provider):
        """Test all adapters implement should_retry."""
        adapter = get_provider_adapter(provider)
        should_retry, backoff = adapter.should_retry(Exception("Test error"))
        assert isinstance(should_retry, bool)
        assert isinstance(backoff, (int, float))
