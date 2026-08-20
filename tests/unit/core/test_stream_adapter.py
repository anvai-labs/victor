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

"""Tests for UnifiedStreamAdapter and SDK-specific adapters."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass
from typing import Any, AsyncIterator, List, Optional

from victor.providers.stream_adapter import (
    StreamMetrics,
    SDKType,
    BaseSDKAdapter,
    OpenAIAdapter,
    AnthropicAdapter,
    GoogleAdapter,
    OllamaAdapter,
    LMStudioAdapter,
    VLLMAdapter,
    SDKAdapterRegistry,
    UnifiedStreamAdapter,
    wrap_provider,
    detect_provider_sdk,
)
from victor.providers.base import (
    BaseProvider,
    CompletionResponse,
    Message,
    StreamChunk,
    ToolDefinition,
)

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_provider():
    """Create a mock base provider."""
    provider = MagicMock(spec=BaseProvider)
    provider.name = "test"
    return provider


@pytest.fixture
def mock_openai_provider():
    """Create a mock OpenAI-style provider."""
    provider = MagicMock(spec=BaseProvider)
    provider.name = "openai"
    return provider


@pytest.fixture
def mock_anthropic_provider():
    """Create a mock Anthropic provider."""
    provider = MagicMock(spec=BaseProvider)
    provider.name = "anthropic"
    return provider


@pytest.fixture
def sample_messages():
    """Sample messages for testing."""
    return [Message(role="user", content="Hello, world!")]


# =============================================================================
# STREAM METRICS TESTS
# =============================================================================


class TestStreamMetrics:
    """Tests for StreamMetrics dataclass."""

    def test_default_values(self):
        """Test default metric values."""
        metrics = StreamMetrics()
        assert metrics.start_time == 0.0
        assert metrics.first_token_time == 0.0
        assert metrics.end_time == 0.0
        assert metrics.total_chunks == 0
        assert metrics.total_content_length == 0
        assert metrics.total_tool_calls == 0
        assert metrics.error_count == 0
        assert metrics.retries == 0

    def test_time_to_first_token(self):
        """Test time_to_first_token property."""
        metrics = StreamMetrics(start_time=1000.0, first_token_time=1000.5)
        assert metrics.time_to_first_token == 0.5

    def test_time_to_first_token_no_data(self):
        """Test time_to_first_token with no data."""
        metrics = StreamMetrics()
        assert metrics.time_to_first_token == 0.0

    def test_total_duration(self):
        """Test total_duration property."""
        metrics = StreamMetrics(start_time=1000.0, end_time=1005.0)
        assert metrics.total_duration == 5.0

    def test_total_duration_no_data(self):
        """Test total_duration with no data."""
        metrics = StreamMetrics()
        assert metrics.total_duration == 0.0

    def test_tokens_per_second(self):
        """Test tokens_per_second estimation."""
        metrics = StreamMetrics(
            start_time=1000.0,
            end_time=1001.0,  # 1 second duration
            total_content_length=400,  # ~100 tokens at 4 chars/token
        )
        assert metrics.tokens_per_second == 100.0

    def test_tokens_per_second_no_duration(self):
        """Test tokens_per_second with zero duration."""
        metrics = StreamMetrics()
        assert metrics.tokens_per_second == 0.0


# =============================================================================
# SDK TYPE TESTS
# =============================================================================


class TestSDKType:
    """Tests for SDKType enum."""

    def test_sdk_types_exist(self):
        """Test all expected SDK types exist."""
        assert SDKType.OPENAI.value == "openai"
        assert SDKType.ANTHROPIC.value == "anthropic"
        assert SDKType.GOOGLE.value == "google"
        assert SDKType.OLLAMA.value == "ollama"
        assert SDKType.LMSTUDIO.value == "lmstudio"
        assert SDKType.VLLM.value == "vllm"
        assert SDKType.UNKNOWN.value == "unknown"


# =============================================================================
# OPENAI ADAPTER TESTS
# =============================================================================


class TestOpenAIAdapter:
    """Tests for OpenAIAdapter."""

    def test_sdk_type(self):
        """Test SDK type property."""
        adapter = OpenAIAdapter()
        assert adapter.sdk_type == SDKType.OPENAI

    @pytest.mark.asyncio
    async def test_adapt_stream_chunk_passthrough(self):
        """Test that StreamChunk objects pass through unchanged."""
        adapter = OpenAIAdapter()
        chunk = StreamChunk(content="test", is_final=False)

        async def mock_stream():
            yield chunk

        results = []
        async for c in adapter.adapt_stream(mock_stream()):
            results.append(c)

        assert len(results) == 1
        assert results[0].content == "test"

    @pytest.mark.asyncio
    async def test_adapt_stream_openai_format(self):
        """Test adapting OpenAI format chunks."""
        adapter = OpenAIAdapter()

        @dataclass
        class MockDelta:
            content: str = "hello"
            tool_calls: Any = None

        @dataclass
        class MockChoice:
            delta: MockDelta
            finish_reason: Optional[str] = None

        @dataclass
        class MockChunk:
            choices: List[MockChoice]

        async def mock_stream():
            yield MockChunk(choices=[MockChoice(delta=MockDelta(content="hello"))])
            yield MockChunk(
                choices=[MockChoice(delta=MockDelta(content=" world"), finish_reason="stop")]
            )

        results = []
        async for c in adapter.adapt_stream(mock_stream()):
            results.append(c)

        assert len(results) == 2
        assert results[0].content == "hello"
        assert not results[0].is_final
        assert results[1].content == " world"
        assert results[1].is_final
        assert results[1].stop_reason == "stop"

    @pytest.mark.asyncio
    async def test_adapt_stream_with_tool_calls(self):
        """Test adapting OpenAI format with tool calls."""
        adapter = OpenAIAdapter()

        @dataclass
        class MockFunction:
            name: str = "read_file"
            arguments: str = '{"path": "test.py"}'

        @dataclass
        class MockToolCall:
            function: MockFunction = None

            def __post_init__(self):
                if self.function is None:
                    self.function = MockFunction()

        @dataclass
        class MockDelta:
            content: str = ""
            tool_calls: List[MockToolCall] = None

        @dataclass
        class MockChoice:
            delta: MockDelta
            finish_reason: Optional[str] = None

        @dataclass
        class MockChunk:
            choices: List[MockChoice]

        async def mock_stream():
            yield MockChunk(
                choices=[MockChoice(delta=MockDelta(content="", tool_calls=[MockToolCall()]))]
            )

        results = []
        async for c in adapter.adapt_stream(mock_stream()):
            results.append(c)

        assert len(results) == 1
        assert results[0].tool_calls is not None
        assert len(results[0].tool_calls) == 1
        assert results[0].tool_calls[0]["name"] == "read_file"

    def test_normalize_tool_calls_empty(self):
        """Test normalizing empty tool calls."""
        adapter = OpenAIAdapter()
        assert adapter._normalize_tool_calls(None) is None
        assert adapter._normalize_tool_calls([]) is None

    def test_detect_stream_method_stream(self):
        """Test detecting stream method."""
        adapter = OpenAIAdapter()
        provider = MagicMock()
        provider.stream = MagicMock()
        assert adapter.detect_stream_method(provider) == "stream"

    def test_detect_stream_method_stream_chat(self):
        """Test detecting stream_chat method."""
        adapter = OpenAIAdapter()
        provider = MagicMock(spec=["stream_chat"])
        assert adapter.detect_stream_method(provider) == "stream_chat"

    def test_detect_stream_method_error(self):
        """Test error when no stream method found."""
        adapter = OpenAIAdapter()
        provider = MagicMock(spec=[])
        with pytest.raises(AttributeError):
            adapter.detect_stream_method(provider)


# =============================================================================
# ANTHROPIC ADAPTER TESTS
# =============================================================================


class TestAnthropicAdapter:
    """Tests for AnthropicAdapter."""

    def test_sdk_type(self):
        """Test SDK type property."""
        adapter = AnthropicAdapter()
        assert adapter.sdk_type == SDKType.ANTHROPIC

    @pytest.mark.asyncio
    async def test_adapt_stream_chunk_passthrough(self):
        """Test that StreamChunk objects pass through unchanged."""
        adapter = AnthropicAdapter()
        chunk = StreamChunk(content="test", is_final=False)

        async def mock_stream():
            yield chunk

        results = []
        async for c in adapter.adapt_stream(mock_stream()):
            results.append(c)

        assert len(results) == 1
        assert results[0].content == "test"

    @pytest.mark.asyncio
    async def test_adapt_stream_content_block_delta(self):
        """Test adapting Anthropic content_block_delta events."""
        adapter = AnthropicAdapter()

        @dataclass
        class MockDelta:
            text: str = "hello"

        @dataclass
        class MockEvent:
            type: str
            delta: Optional[MockDelta] = None

        async def mock_stream():
            yield MockEvent(type="content_block_delta", delta=MockDelta(text="hello"))
            yield MockEvent(type="content_block_delta", delta=MockDelta(text=" world"))
            yield MockEvent(type="message_stop")

        results = []
        async for c in adapter.adapt_stream(mock_stream()):
            results.append(c)

        assert len(results) == 3
        assert results[0].content == "hello"
        assert results[1].content == " world"
        assert results[2].is_final
        assert results[2].stop_reason == "stop"

    @pytest.mark.asyncio
    async def test_adapt_stream_tool_use(self):
        """Test adapting Anthropic tool_use events."""
        adapter = AnthropicAdapter()

        @dataclass
        class MockEvent:
            type: str
            name: Optional[str] = None
            input: Optional[dict] = None

        async def mock_stream():
            yield MockEvent(type="tool_use", name="read_file", input={"path": "test.py"})

        results = []
        async for c in adapter.adapt_stream(mock_stream()):
            results.append(c)

        assert len(results) == 1
        assert results[0].tool_calls is not None
        assert results[0].tool_calls[0]["name"] == "read_file"


# =============================================================================
# GOOGLE ADAPTER TESTS
# =============================================================================


class TestGoogleAdapter:
    """Tests for GoogleAdapter."""

    def test_sdk_type(self):
        """Test SDK type property."""
        adapter = GoogleAdapter()
        assert adapter.sdk_type == SDKType.GOOGLE

    @pytest.mark.asyncio
    async def test_adapt_stream_chunk_passthrough(self):
        """Test that StreamChunk objects pass through unchanged."""
        adapter = GoogleAdapter()
        chunk = StreamChunk(content="test", is_final=False)

        async def mock_stream():
            yield chunk

        results = []
        async for c in adapter.adapt_stream(mock_stream()):
            results.append(c)

        assert len(results) == 1
        assert results[0].content == "test"

    @pytest.mark.asyncio
    async def test_adapt_stream_google_format(self):
        """Test adapting Google/Gemini format chunks."""
        adapter = GoogleAdapter()

        @dataclass
        class MockCandidate:
            finish_reason: Optional[str] = None
            content: Any = None

        @dataclass
        class MockChunk:
            text: str
            candidates: Optional[List[MockCandidate]] = None

        async def mock_stream():
            yield MockChunk(text="hello")
            yield MockChunk(
                text=" world",
                candidates=[MockCandidate(finish_reason="STOP")],
            )

        results = []
        async for c in adapter.adapt_stream(mock_stream()):
            results.append(c)

        assert len(results) == 2
        assert results[0].content == "hello"
        assert results[1].content == " world"
        assert results[1].is_final


# =============================================================================
# OLLAMA ADAPTER TESTS
# =============================================================================


class TestOllamaAdapter:
    """Tests for OllamaAdapter."""

    def test_sdk_type(self):
        """Test SDK type property."""
        adapter = OllamaAdapter()
        assert adapter.sdk_type == SDKType.OLLAMA

    @pytest.mark.asyncio
    async def test_adapt_stream_chunk_passthrough(self):
        """Test that StreamChunk objects pass through unchanged."""
        adapter = OllamaAdapter()
        chunk = StreamChunk(content="test", is_final=False)

        async def mock_stream():
            yield chunk

        results = []
        async for c in adapter.adapt_stream(mock_stream()):
            results.append(c)

        assert len(results) == 1
        assert results[0].content == "test"

    @pytest.mark.asyncio
    async def test_adapt_stream_dict_format(self):
        """Test adapting Ollama dict format."""
        adapter = OllamaAdapter()

        async def mock_stream():
            yield {"message": {"content": "hello"}, "done": False}
            yield {"message": {"content": " world"}, "done": True}

        results = []
        async for c in adapter.adapt_stream(mock_stream()):
            results.append(c)

        assert len(results) == 2
        assert results[0].content == "hello"
        assert not results[0].is_final
        assert results[1].content == " world"
        assert results[1].is_final

    @pytest.mark.asyncio
    async def test_adapt_stream_object_format(self):
        """Test adapting Ollama object format."""
        adapter = OllamaAdapter()

        @dataclass
        class MockMessage:
            content: str

        @dataclass
        class MockChunk:
            message: MockMessage
            done: bool

        async def mock_stream():
            yield MockChunk(message=MockMessage(content="hello"), done=False)
            yield MockChunk(message=MockMessage(content=" world"), done=True)

        results = []
        async for c in adapter.adapt_stream(mock_stream()):
            results.append(c)

        assert len(results) == 2
        assert results[0].content == "hello"
        assert results[1].is_final

    @pytest.mark.asyncio
    async def test_adapt_stream_with_tool_calls(self):
        """Test adapting Ollama format with tool calls."""
        adapter = OllamaAdapter()

        async def mock_stream():
            yield {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "read_file",
                                "arguments": {"path": "test.py"},
                            }
                        }
                    ],
                },
                "done": False,
            }

        results = []
        async for c in adapter.adapt_stream(mock_stream()):
            results.append(c)

        assert len(results) == 1
        assert results[0].tool_calls is not None
        assert results[0].tool_calls[0]["name"] == "read_file"

    def test_normalize_tool_calls_empty(self):
        """Test normalizing empty tool calls."""
        adapter = OllamaAdapter()
        assert adapter._normalize_tool_calls(None) is None
        assert adapter._normalize_tool_calls([]) is None


# =============================================================================
# LMSTUDIO ADAPTER TESTS
# =============================================================================


class TestLMStudioAdapter:
    """Tests for LMStudioAdapter."""

    def test_sdk_type(self):
        """Test SDK type property."""
        adapter = LMStudioAdapter()
        assert adapter.sdk_type == SDKType.LMSTUDIO

    @pytest.mark.asyncio
    async def test_adapt_stream_chunk_passthrough(self):
        """Test that StreamChunk objects pass through unchanged."""
        adapter = LMStudioAdapter()
        chunk = StreamChunk(content="test", is_final=False)

        async def mock_stream():
            yield chunk

        results = []
        async for c in adapter.adapt_stream(mock_stream()):
            results.append(c)

        assert len(results) == 1
        assert results[0].content == "test"


# =============================================================================
# VLLM ADAPTER TESTS
# =============================================================================


class TestVLLMAdapter:
    """Tests for VLLMAdapter."""

    def test_sdk_type(self):
        """Test SDK type property."""
        adapter = VLLMAdapter()
        assert adapter.sdk_type == SDKType.VLLM

    @pytest.mark.asyncio
    async def test_adapt_stream_chunk_passthrough(self):
        """Test that StreamChunk objects pass through unchanged."""
        adapter = VLLMAdapter()
        chunk = StreamChunk(content="test", is_final=False)

        async def mock_stream():
            yield chunk

        results = []
        async for c in adapter.adapt_stream(mock_stream()):
            results.append(c)

        assert len(results) == 1
        assert results[0].content == "test"

    @pytest.mark.asyncio
    async def test_adapt_stream_dict_format(self):
        """Test adapting vLLM dict format."""
        adapter = VLLMAdapter()

        async def mock_stream():
            yield {"text": "hello", "finished": False}
            yield {"text": " world", "finished": True}

        results = []
        async for c in adapter.adapt_stream(mock_stream()):
            results.append(c)

        assert len(results) == 2
        assert results[0].content == "hello"
        assert not results[0].is_final
        assert results[1].content == " world"
        assert results[1].is_final


# =============================================================================
# SDK ADAPTER REGISTRY TESTS
# =============================================================================


class TestSDKAdapterRegistry:
    """Tests for SDKAdapterRegistry."""

    def test_get_adapter_openai(self):
        """Test getting OpenAI adapter."""
        adapter = SDKAdapterRegistry.get_adapter(SDKType.OPENAI)
        assert isinstance(adapter, OpenAIAdapter)

    def test_get_adapter_anthropic(self):
        """Test getting Anthropic adapter."""
        adapter = SDKAdapterRegistry.get_adapter(SDKType.ANTHROPIC)
        assert isinstance(adapter, AnthropicAdapter)

    def test_get_adapter_google(self):
        """Test getting Google adapter."""
        adapter = SDKAdapterRegistry.get_adapter(SDKType.GOOGLE)
        assert isinstance(adapter, GoogleAdapter)

    def test_get_adapter_ollama(self):
        """Test getting Ollama adapter."""
        adapter = SDKAdapterRegistry.get_adapter(SDKType.OLLAMA)
        assert isinstance(adapter, OllamaAdapter)

    def test_get_adapter_unknown_defaults_to_openai(self):
        """Test that unknown SDK type defaults to OpenAI adapter."""
        adapter = SDKAdapterRegistry.get_adapter(SDKType.UNKNOWN)
        assert isinstance(adapter, OpenAIAdapter)

    def test_detect_sdk_type_anthropic(self):
        """Test detecting Anthropic SDK type."""
        provider = MagicMock()
        provider.name = "anthropic"
        assert SDKAdapterRegistry.detect_sdk_type(provider) == SDKType.ANTHROPIC

    def test_detect_sdk_type_google(self):
        """Test detecting Google SDK type."""
        provider = MagicMock()
        provider.name = "google"
        assert SDKAdapterRegistry.detect_sdk_type(provider) == SDKType.GOOGLE

    def test_detect_sdk_type_ollama(self):
        """Test detecting Ollama SDK type."""
        provider = MagicMock()
        provider.name = "ollama"
        assert SDKAdapterRegistry.detect_sdk_type(provider) == SDKType.OLLAMA

    def test_detect_sdk_type_lmstudio(self):
        """Test detecting LMStudio SDK type."""
        provider = MagicMock()
        provider.name = "lmstudio"
        assert SDKAdapterRegistry.detect_sdk_type(provider) == SDKType.LMSTUDIO

    def test_detect_sdk_type_vllm(self):
        """Test detecting vLLM SDK type."""
        provider = MagicMock()
        provider.name = "vllm"
        assert SDKAdapterRegistry.detect_sdk_type(provider) == SDKType.VLLM

    def test_detect_sdk_type_openai(self):
        """Test detecting OpenAI SDK type."""
        provider = MagicMock()
        provider.name = "openai"
        assert SDKAdapterRegistry.detect_sdk_type(provider) == SDKType.OPENAI

    def test_detect_sdk_type_by_class_name(self):
        """Test detecting SDK type by class name."""

        class AnthropicProvider:
            pass

        provider = AnthropicProvider()
        assert SDKAdapterRegistry.detect_sdk_type(provider) == SDKType.ANTHROPIC

    def test_detect_sdk_type_unknown(self):
        """Test detecting unknown SDK type."""
        provider = MagicMock()
        provider.name = "custom_provider"
        assert SDKAdapterRegistry.detect_sdk_type(provider) == SDKType.UNKNOWN

    def test_register_custom_adapter(self):
        """Test registering a custom adapter."""

        class CustomAdapter(BaseSDKAdapter):
            @property
            def sdk_type(self):
                return SDKType.UNKNOWN

            async def adapt_stream(self, raw_stream, **kwargs):
                async for chunk in raw_stream:
                    yield chunk

        custom = CustomAdapter()
        SDKAdapterRegistry.register_adapter(SDKType.UNKNOWN, custom)
        assert SDKAdapterRegistry.get_adapter(SDKType.UNKNOWN) is custom


# =============================================================================
# UNIFIED STREAM ADAPTER TESTS
# =============================================================================


class TestUnifiedStreamAdapter:
    """Tests for UnifiedStreamAdapter."""

    def test_init_default(self, mock_provider):
        """Test default initialization."""
        adapter = UnifiedStreamAdapter(mock_provider)
        assert adapter.provider is mock_provider
        assert adapter.sdk_type == SDKType.UNKNOWN
        assert adapter.last_metrics is None

    def test_init_with_sdk_type(self, mock_provider):
        """Test initialization with explicit SDK type."""
        adapter = UnifiedStreamAdapter(mock_provider, sdk_type=SDKType.ANTHROPIC)
        assert adapter.sdk_type == SDKType.ANTHROPIC

    def test_init_without_metrics(self, mock_provider):
        """Test initialization without metrics collection."""
        adapter = UnifiedStreamAdapter(mock_provider, collect_metrics=False)
        assert adapter._collect_metrics is False

    @pytest.mark.asyncio
    async def test_stream_basic(self, mock_provider, sample_messages):
        """Test basic streaming."""
        chunks = [
            StreamChunk(content="Hello", is_final=False),
            StreamChunk(content=" world!", is_final=True, stop_reason="stop"),
        ]

        async def mock_stream(*args, **kwargs):
            for chunk in chunks:
                yield chunk

        mock_provider.stream = mock_stream
        adapter = UnifiedStreamAdapter(mock_provider)

        results = []
        async for chunk in adapter.stream(sample_messages, model="test"):
            results.append(chunk)

        assert len(results) == 2
        assert results[0].content == "Hello"
        assert results[1].content == " world!"
        assert results[1].is_final

    @pytest.mark.asyncio
    async def test_stream_collects_metrics(self, mock_provider, sample_messages):
        """Test that streaming collects metrics."""
        chunks = [
            StreamChunk(content="Hello", is_final=False),
            StreamChunk(content=" world!", is_final=True, stop_reason="stop"),
        ]

        async def mock_stream(*args, **kwargs):
            for chunk in chunks:
                yield chunk

        mock_provider.stream = mock_stream
        adapter = UnifiedStreamAdapter(mock_provider, collect_metrics=True)

        async for _ in adapter.stream(sample_messages, model="test"):
            pass

        assert adapter.last_metrics is not None
        assert adapter.last_metrics.total_chunks == 2
        assert adapter.last_metrics.total_content_length == len("Hello world!")

    @pytest.mark.asyncio
    async def test_stream_counts_tool_calls(self, mock_provider, sample_messages):
        """Test that streaming counts tool calls."""
        chunks = [
            StreamChunk(
                content="",
                tool_calls=[{"name": "read_file", "arguments": {}}],
                is_final=False,
            ),
            StreamChunk(content="Done", is_final=True),
        ]

        async def mock_stream(*args, **kwargs):
            for chunk in chunks:
                yield chunk

        mock_provider.stream = mock_stream
        adapter = UnifiedStreamAdapter(mock_provider)

        async for _ in adapter.stream(sample_messages, model="test"):
            pass

        assert adapter.last_metrics.total_tool_calls == 1

    @pytest.mark.asyncio
    async def test_stream_error_handling(self, mock_provider, sample_messages):
        """Test error handling during streaming."""

        async def mock_stream(*args, **kwargs):
            yield StreamChunk(content="Hello", is_final=False)
            raise RuntimeError("Stream error")

        mock_provider.stream = mock_stream
        adapter = UnifiedStreamAdapter(mock_provider)

        with pytest.raises(RuntimeError):
            async for _ in adapter.stream(sample_messages, model="test"):
                pass

        assert adapter.last_metrics is not None
        assert adapter.last_metrics.error_count == 1

    @pytest.mark.asyncio
    async def test_stream_chat_alias(self, mock_provider, sample_messages):
        """Test stream_chat is an alias for stream."""
        chunks = [StreamChunk(content="test", is_final=True)]

        async def mock_stream(*args, **kwargs):
            for chunk in chunks:
                yield chunk

        mock_provider.stream = mock_stream
        adapter = UnifiedStreamAdapter(mock_provider)

        results = []
        async for chunk in adapter.stream_chat(sample_messages, model="test"):
            results.append(chunk)

        assert len(results) == 1
        assert results[0].content == "test"

    @pytest.mark.asyncio
    async def test_stream_completion_alias(self, mock_provider, sample_messages):
        """Test stream_completion is an alias for stream."""
        chunks = [StreamChunk(content="test", is_final=True)]

        async def mock_stream(*args, **kwargs):
            for chunk in chunks:
                yield chunk

        mock_provider.stream = mock_stream
        adapter = UnifiedStreamAdapter(mock_provider)

        results = []
        async for chunk in adapter.stream_completion(sample_messages, model="test"):
            results.append(chunk)

        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_collect_to_response(self, mock_provider, sample_messages):
        """Test collecting stream to CompletionResponse."""
        chunks = [
            StreamChunk(content="Hello", is_final=False),
            StreamChunk(content=" world!", is_final=False),
            StreamChunk(content="", is_final=True, stop_reason="stop"),
        ]

        async def mock_stream(*args, **kwargs):
            for chunk in chunks:
                yield chunk

        mock_provider.stream = mock_stream
        adapter = UnifiedStreamAdapter(mock_provider)

        response = await adapter.collect_to_response(sample_messages, model="test")

        assert isinstance(response, CompletionResponse)
        assert response.content == "Hello world!"
        assert response.stop_reason == "stop"
        assert response.model == "test"

    @pytest.mark.asyncio
    async def test_collect_to_response_with_tool_calls(self, mock_provider, sample_messages):
        """Test collecting stream with tool calls."""
        chunks = [
            StreamChunk(
                content="",
                tool_calls=[{"name": "read_file", "arguments": {"path": "test.py"}}],
                is_final=False,
            ),
            StreamChunk(content="", is_final=True, stop_reason="tool_use"),
        ]

        async def mock_stream(*args, **kwargs):
            for chunk in chunks:
                yield chunk

        mock_provider.stream = mock_stream
        adapter = UnifiedStreamAdapter(mock_provider)

        response = await adapter.collect_to_response(sample_messages, model="test")

        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0]["name"] == "read_file"


# =============================================================================
# CONVENIENCE FUNCTION TESTS
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_wrap_provider(self, mock_provider):
        """Test wrap_provider function."""
        adapter = wrap_provider(mock_provider)
        assert isinstance(adapter, UnifiedStreamAdapter)
        assert adapter.provider is mock_provider

    def test_detect_provider_sdk(self, mock_openai_provider):
        """Test detect_provider_sdk function."""
        sdk_type = detect_provider_sdk(mock_openai_provider)
        assert sdk_type == SDKType.OPENAI

    def test_detect_provider_sdk_anthropic(self, mock_anthropic_provider):
        """Test detect_provider_sdk for Anthropic."""
        sdk_type = detect_provider_sdk(mock_anthropic_provider)
        assert sdk_type == SDKType.ANTHROPIC
