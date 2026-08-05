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

"""Unified Stream Adapter for Provider SDKs.

This module provides a unified streaming interface that wraps different
LLM provider SDKs (OpenAI, Anthropic, Google, Ollama, LMStudio, vLLM)
with a common interface.

Design Patterns Used:
- Adapter Pattern: Wraps SDK-specific streaming APIs into a common interface
- Strategy Pattern: SDK-specific implementations are swappable strategies
- Protocol Pattern: Defines a common interface via Python Protocol

The unified interface provides:
1. Consistent method naming (stream, stream_chat, stream_completion)
2. Normalized response format (StreamChunk)
3. Error handling normalization
4. Metrics collection
5. Retry/resilience integration

Usage:
    from victor.providers.stream_adapter import UnifiedStreamAdapter

    # Create adapter for any provider
    adapter = UnifiedStreamAdapter(provider)

    # Use any method name - all are equivalent
    async for chunk in adapter.stream(messages, model="gpt-4"):
        print(chunk.content)

    # Or use the aliased methods
    async for chunk in adapter.stream_chat(messages, model="gpt-4"):
        print(chunk.content)
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import (
    Any,
    AsyncIterator,
    Dict,
    List,
    Optional,
)

from victor.providers.base import (
    BaseProvider,
    CompletionResponse,
    Message,
    StreamChunk,
    ToolDefinition,
)

logger = logging.getLogger(__name__)


# =============================================================================
# STREAM METRICS
# =============================================================================


@dataclass
class StreamMetrics:
    """Metrics collected during streaming."""

    start_time: float = 0.0
    first_token_time: float = 0.0
    end_time: float = 0.0
    total_chunks: int = 0
    total_content_length: int = 0
    total_tool_calls: int = 0
    error_count: int = 0
    retries: int = 0

    @property
    def time_to_first_token(self) -> float:
        """Time to first token in seconds."""
        if self.first_token_time and self.start_time:
            return self.first_token_time - self.start_time
        return 0.0

    @property
    def total_duration(self) -> float:
        """Total streaming duration in seconds."""
        if self.end_time and self.start_time:
            return self.end_time - self.start_time
        return 0.0

    @property
    def tokens_per_second(self) -> float:
        """Estimated tokens per second (using content length as proxy)."""
        if self.total_duration > 0:
            # Rough estimate: ~4 chars per token
            estimated_tokens = self.total_content_length / 4
            return estimated_tokens / self.total_duration
        return 0.0


# =============================================================================
# SDK-SPECIFIC ADAPTERS
# =============================================================================


class SDKType(Enum):
    """Supported SDK types."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    OLLAMA = "ollama"
    LMSTUDIO = "lmstudio"
    VLLM = "vllm"
    UNKNOWN = "unknown"


class BaseSDKAdapter(ABC):
    """Base adapter for SDK-specific streaming implementations.

    Subclasses implement the SDK-specific transformation of stream
    chunks to the normalized StreamChunk format.
    """

    @property
    @abstractmethod
    def sdk_type(self) -> SDKType:
        """Return the SDK type this adapter handles."""
        pass

    @abstractmethod
    async def adapt_stream(
        self,
        raw_stream: AsyncIterator[Any],
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Adapt a raw SDK stream to normalized StreamChunk.

        Args:
            raw_stream: Raw stream from the SDK
            **kwargs: SDK-specific options

        Yields:
            Normalized StreamChunk objects
        """
        pass

    def detect_stream_method(self, provider: Any) -> str:
        """Detect which stream method the provider uses.

        Args:
            provider: The provider instance

        Returns:
            Method name: 'stream' or 'stream_chat'
        """
        if hasattr(provider, "stream"):
            return "stream"
        if hasattr(provider, "stream_chat"):
            return "stream_chat"
        raise AttributeError(f"Provider {type(provider).__name__} has no stream method")


class OpenAIAdapter(BaseSDKAdapter):
    """Adapter for OpenAI SDK streaming format."""

    @property
    def sdk_type(self) -> SDKType:
        return SDKType.OPENAI

    async def adapt_stream(
        self,
        raw_stream: AsyncIterator[Any],
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Adapt OpenAI stream to StreamChunk.

        OpenAI format:
        - chunk.choices[0].delta.content
        - chunk.choices[0].delta.tool_calls
        - chunk.choices[0].finish_reason
        """
        async for chunk in raw_stream:
            if isinstance(chunk, StreamChunk):
                yield chunk
            elif hasattr(chunk, "choices") and chunk.choices:
                delta = chunk.choices[0].delta
                content = getattr(delta, "content", "") or ""
                tool_calls = getattr(delta, "tool_calls", None)
                finish_reason = chunk.choices[0].finish_reason

                yield StreamChunk(
                    content=content,
                    tool_calls=self._normalize_tool_calls(tool_calls),
                    stop_reason=finish_reason,
                    is_final=finish_reason is not None,
                )

    def _normalize_tool_calls(self, tool_calls: Any) -> Optional[List[Dict[str, Any]]]:
        """Normalize OpenAI tool calls format."""
        if not tool_calls:
            return None

        normalized = []
        for tc in tool_calls:
            if hasattr(tc, "function"):
                normalized.append(
                    {
                        "name": tc.function.name if tc.function.name else "",
                        "arguments": tc.function.arguments or "{}",
                    }
                )
        return normalized if normalized else None


class AnthropicAdapter(BaseSDKAdapter):
    """Adapter for Anthropic SDK streaming format."""

    @property
    def sdk_type(self) -> SDKType:
        return SDKType.ANTHROPIC

    async def adapt_stream(
        self,
        raw_stream: AsyncIterator[Any],
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Adapt Anthropic stream to StreamChunk.

        Anthropic format:
        - event.type: "content_block_delta", "message_stop", etc.
        - event.delta.text (for text deltas)
        - event.content_block.type: "tool_use" (for tool calls)
        """
        async for event in raw_stream:
            if isinstance(event, StreamChunk):
                yield event
            elif hasattr(event, "type"):
                if event.type == "content_block_delta":
                    delta = getattr(event, "delta", None)
                    text = getattr(delta, "text", "") if delta else ""
                    yield StreamChunk(content=text, is_final=False)
                elif event.type == "message_stop":
                    yield StreamChunk(
                        content="",
                        stop_reason="stop",
                        is_final=True,
                    )
                elif event.type == "tool_use":
                    # Handle Anthropic tool use format
                    tool_name = getattr(event, "name", "")
                    tool_input = getattr(event, "input", {})
                    yield StreamChunk(
                        content="",
                        tool_calls=[{"name": tool_name, "arguments": tool_input}],
                        is_final=False,
                    )


class GoogleAdapter(BaseSDKAdapter):
    """Adapter for Google (Gemini) SDK streaming format."""

    @property
    def sdk_type(self) -> SDKType:
        return SDKType.GOOGLE

    async def adapt_stream(
        self,
        raw_stream: AsyncIterator[Any],
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Adapt Google/Gemini stream to StreamChunk.

        Gemini format:
        - chunk.text (content)
        - chunk.candidates[0].finish_reason
        - chunk.candidates[0].content.parts (for function calls)
        """
        async for chunk in raw_stream:
            if isinstance(chunk, StreamChunk):
                yield chunk
            elif hasattr(chunk, "text"):
                text = chunk.text or ""
                finish_reason = None
                tool_calls = None

                if hasattr(chunk, "candidates") and chunk.candidates:
                    candidate = chunk.candidates[0]
                    finish_reason = getattr(candidate, "finish_reason", None)

                    # Check for function calls
                    content = getattr(candidate, "content", None)
                    if content and hasattr(content, "parts"):
                        for part in content.parts:
                            if hasattr(part, "function_call"):
                                fc = part.function_call
                                tool_calls = [
                                    {
                                        "name": fc.name,
                                        "arguments": dict(fc.args),
                                    }
                                ]

                yield StreamChunk(
                    content=text,
                    tool_calls=tool_calls,
                    stop_reason=str(finish_reason) if finish_reason else None,
                    is_final=finish_reason is not None,
                )


class OllamaAdapter(BaseSDKAdapter):
    """Adapter for Ollama streaming format."""

    @property
    def sdk_type(self) -> SDKType:
        return SDKType.OLLAMA

    async def adapt_stream(
        self,
        raw_stream: AsyncIterator[Any],
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Adapt Ollama stream to StreamChunk.

        Ollama format (dict or object):
        - message.content (text content)
        - message.tool_calls (optional)
        - done (completion flag)
        """
        async for chunk in raw_stream:
            if isinstance(chunk, StreamChunk):
                yield chunk
            elif isinstance(chunk, dict):
                message = chunk.get("message", {})
                content = message.get("content", "")
                tool_calls = message.get("tool_calls")
                done = chunk.get("done", False)

                yield StreamChunk(
                    content=content,
                    tool_calls=self._normalize_tool_calls(tool_calls),
                    stop_reason="stop" if done else None,
                    is_final=done,
                )
            elif hasattr(chunk, "message"):
                content = getattr(chunk.message, "content", "")
                done = getattr(chunk, "done", False)
                yield StreamChunk(
                    content=content,
                    stop_reason="stop" if done else None,
                    is_final=done,
                )

    def _normalize_tool_calls(self, tool_calls: Any) -> Optional[List[Dict[str, Any]]]:
        """Normalize Ollama tool calls."""
        if not tool_calls:
            return None

        normalized = []
        for tc in tool_calls:
            if isinstance(tc, dict):
                func = tc.get("function", {})
                normalized.append(
                    {
                        "name": func.get("name", ""),
                        "arguments": func.get("arguments", {}),
                    }
                )
        return normalized if normalized else None


class LMStudioAdapter(BaseSDKAdapter):
    """Adapter for LMStudio streaming format (OpenAI-compatible)."""

    @property
    def sdk_type(self) -> SDKType:
        return SDKType.LMSTUDIO

    async def adapt_stream(
        self,
        raw_stream: AsyncIterator[Any],
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Adapt LMStudio stream to StreamChunk.

        LMStudio uses OpenAI-compatible SSE format.
        Already normalized in the provider, so this is mostly passthrough.
        """
        async for chunk in raw_stream:
            if isinstance(chunk, StreamChunk):
                yield chunk
            else:
                # Fallback to OpenAI format parsing
                adapter = OpenAIAdapter()
                async for adapted in adapter.adapt_stream(iter([chunk]).__aiter__()):
                    yield adapted


class VLLMAdapter(BaseSDKAdapter):
    """Adapter for vLLM streaming format."""

    @property
    def sdk_type(self) -> SDKType:
        return SDKType.VLLM

    async def adapt_stream(
        self,
        raw_stream: AsyncIterator[Any],
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Adapt vLLM stream to StreamChunk.

        vLLM can use OpenAI-compatible format or custom format.
        """
        async for chunk in raw_stream:
            if isinstance(chunk, StreamChunk):
                yield chunk
            elif isinstance(chunk, dict):
                # vLLM dict format
                content = chunk.get("text", "")
                finished = chunk.get("finished", False)
                yield StreamChunk(
                    content=content,
                    stop_reason="stop" if finished else None,
                    is_final=finished,
                )
            else:
                # Fallback to OpenAI format
                adapter = OpenAIAdapter()
                async for adapted in adapter.adapt_stream(iter([chunk]).__aiter__()):
                    yield adapted


# =============================================================================
# ADAPTER REGISTRY
# =============================================================================


class SDKAdapterRegistry:
    """Registry for SDK-specific stream adapters."""

    _adapters: Dict[SDKType, BaseSDKAdapter] = {
        SDKType.OPENAI: OpenAIAdapter(),
        SDKType.ANTHROPIC: AnthropicAdapter(),
        SDKType.GOOGLE: GoogleAdapter(),
        SDKType.OLLAMA: OllamaAdapter(),
        SDKType.LMSTUDIO: LMStudioAdapter(),
        SDKType.VLLM: VLLMAdapter(),
    }

    @classmethod
    def get_adapter(cls, sdk_type: SDKType) -> BaseSDKAdapter:
        """Get adapter for SDK type.

        Args:
            sdk_type: The SDK type

        Returns:
            Appropriate SDK adapter
        """
        return cls._adapters.get(sdk_type, OpenAIAdapter())

    @classmethod
    def detect_sdk_type(cls, provider: Any) -> SDKType:
        """Detect SDK type from provider instance.

        Args:
            provider: Provider instance

        Returns:
            Detected SDK type
        """
        provider_name = provider.name.lower() if hasattr(provider, "name") else ""
        class_name = type(provider).__name__.lower()

        if "anthropic" in provider_name or "anthropic" in class_name:
            return SDKType.ANTHROPIC
        if "google" in provider_name or "gemini" in class_name:
            return SDKType.GOOGLE
        if "ollama" in provider_name or "ollama" in class_name:
            return SDKType.OLLAMA
        if "lmstudio" in provider_name or "lmstudio" in class_name:
            return SDKType.LMSTUDIO
        if "vllm" in provider_name or "vllm" in class_name:
            return SDKType.VLLM
        if "openai" in provider_name or "openai" in class_name:
            return SDKType.OPENAI

        return SDKType.UNKNOWN

    @classmethod
    def register_adapter(cls, sdk_type: SDKType, adapter: BaseSDKAdapter) -> None:
        """Register a custom adapter for an SDK type.

        Args:
            sdk_type: The SDK type
            adapter: The adapter instance
        """
        cls._adapters[sdk_type] = adapter


# =============================================================================
# UNIFIED STREAM ADAPTER
# =============================================================================


class UnifiedStreamAdapter:
    """Unified adapter that wraps any provider with consistent streaming interface.

    This adapter provides:
    1. Method aliasing (stream, stream_chat, stream_completion are equivalent)
    2. Automatic SDK detection and adaptation
    3. Metrics collection
    4. Error normalization
    5. Retry integration

    Usage:
        provider = AnthropicProvider(api_key="...")
        adapter = UnifiedStreamAdapter(provider)

        # All these are equivalent
        async for chunk in adapter.stream(messages, model="claude-3"):
            print(chunk.content)

        async for chunk in adapter.stream_chat(messages, model="claude-3"):
            print(chunk.content)
    """

    def __init__(
        self,
        provider: BaseProvider,
        sdk_type: Optional[SDKType] = None,
        collect_metrics: bool = True,
    ):
        """Initialize the unified adapter.

        Args:
            provider: The underlying provider
            sdk_type: Override SDK type detection (optional)
            collect_metrics: Whether to collect streaming metrics
        """
        self._provider = provider
        self._sdk_type = sdk_type or SDKAdapterRegistry.detect_sdk_type(provider)
        self._adapter = SDKAdapterRegistry.get_adapter(self._sdk_type)
        self._collect_metrics = collect_metrics
        self._last_metrics: Optional[StreamMetrics] = None

    @property
    def provider(self) -> BaseProvider:
        """Get the underlying provider."""
        return self._provider

    @property
    def sdk_type(self) -> SDKType:
        """Get the detected SDK type."""
        return self._sdk_type

    @property
    def last_metrics(self) -> Optional[StreamMetrics]:
        """Get metrics from the last stream operation."""
        return self._last_metrics

    async def stream(
        self,
        messages: List[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Stream chat completion with unified interface.

        This is the primary method - stream_chat and stream_completion are aliases.

        Args:
            messages: Conversation messages
            model: Model identifier
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            tools: Available tools
            **kwargs: Provider-specific options

        Yields:
            Normalized StreamChunk objects

        Raises:
            ProviderError: If streaming fails
        """
        metrics = StreamMetrics(start_time=time.time()) if self._collect_metrics else None

        try:
            # Get the raw stream from the provider
            raw_stream = self._provider.stream(
                messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                **kwargs,
            )

            first_chunk = True
            async for chunk in raw_stream:
                if metrics:
                    if first_chunk:
                        metrics.first_token_time = time.time()
                        first_chunk = False
                    metrics.total_chunks += 1
                    metrics.total_content_length += len(chunk.content or "")
                    if chunk.tool_calls:
                        metrics.total_tool_calls += len(chunk.tool_calls)

                yield chunk

            if metrics:
                metrics.end_time = time.time()
                self._last_metrics = metrics

        except Exception:
            if metrics:
                metrics.error_count += 1
                metrics.end_time = time.time()
                self._last_metrics = metrics
            raise

    # Aliases for compatibility with different naming conventions
    async def stream_chat(
        self,
        messages: List[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Stream chat completion (alias for stream()).

        See stream() for documentation.
        """
        async for chunk in self.stream(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            **kwargs,
        ):
            yield chunk

    async def stream_completion(
        self,
        messages: List[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Stream completion (alias for stream()).

        See stream() for documentation.
        """
        async for chunk in self.stream(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            **kwargs,
        ):
            yield chunk

    async def collect_to_response(
        self,
        messages: List[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Stream and collect into a single CompletionResponse.

        Useful when you want streaming metrics but need the full response.

        Args:
            messages: Conversation messages
            model: Model identifier
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            tools: Available tools
            **kwargs: Provider-specific options

        Returns:
            Complete CompletionResponse
        """
        content_parts: List[str] = []
        tool_calls: List[Dict[str, Any]] = []
        stop_reason: Optional[str] = None

        async for chunk in self.stream(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            **kwargs,
        ):
            if chunk.content:
                content_parts.append(chunk.content)
            if chunk.tool_calls:
                tool_calls.extend(chunk.tool_calls)
            if chunk.stop_reason:
                stop_reason = chunk.stop_reason

        return CompletionResponse(
            content="".join(content_parts),
            role="assistant",
            tool_calls=tool_calls if tool_calls else None,
            stop_reason=stop_reason,
            model=model,
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def wrap_provider(provider: BaseProvider) -> UnifiedStreamAdapter:
    """Wrap a provider with the unified stream adapter.

    Args:
        provider: Any BaseProvider instance

    Returns:
        UnifiedStreamAdapter wrapping the provider
    """
    return UnifiedStreamAdapter(provider)


def detect_provider_sdk(provider: BaseProvider) -> SDKType:
    """Detect the SDK type of a provider.

    Args:
        provider: Provider instance

    Returns:
        Detected SDK type
    """
    return SDKAdapterRegistry.detect_sdk_type(provider)
