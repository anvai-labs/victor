# Tutorial: Integrating a New LLM Provider

This tutorial walks you through creating a custom LLM provider for Victor. By the end, you will have a fully functional provider that supports chat completions, streaming, and tool calling.

## What You Will Build

A complete LLM provider integration that:
- Connects to any LLM API (cloud or local)
- Supports both synchronous and streaming chat completions
- Handles tool/function calling
- Integrates with Victor's circuit breaker for resilience
- Is properly registered and testable

## Prerequisites

- Python 3.11+
- Victor development environment set up (`pip install -e ./victor-contracts -e ".[dev]"`)
- API access to the LLM provider you want to integrate
- Basic understanding of async/await in Python

**Time estimate:** 45-60 minutes

---

## 1. Provider Architecture Overview

### The BaseProvider Class

All Victor providers inherit from `BaseProvider` located at `victor/providers/base.py`. This abstract base class defines the contract that every provider must fulfill.

```python
from victor.providers.base import BaseProvider

class BaseProvider(ABC):
    """Abstract base class for all LLM providers."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 3,
        use_circuit_breaker: bool = True,
        **kwargs: Any,
    ):
        # Initialization with built-in circuit breaker support
        ...
```

### Required Methods

Every provider must implement these abstract methods:

| Method | Description |
|--------|-------------|
| `name` (property) | Returns the provider identifier (e.g., "anthropic", "openai") |
| `chat()` | Sends a chat completion request and returns a response |
| `stream()` | Streams a chat completion response incrementally |
| `close()` | Closes any open connections or resources |

### Optional Capabilities

Providers can declare additional capabilities by overriding these methods:

| Method | Default | Description |
|--------|---------|-------------|
| `supports_tools()` | `False` | Whether the provider supports tool/function calling |
| `supports_streaming()` | `False` | Whether the provider supports streaming responses |
| `discover_capabilities()` | Config-based | Runtime capability discovery |
| `count_tokens()` | Character estimate | Token counting for the provider |

### Core Data Types

Victor uses standardized data types across all providers:

```python
from victor.providers.base import (
    Message,           # Input message format
    CompletionResponse,# Non-streaming response
    StreamChunk,       # Streaming response chunk
    ToolDefinition,    # Tool schema for function calling
)
```

---

## 2. Step-by-Step Implementation

### Step 1: Create the Provider File

Create a new file at `victor/providers/custom_provider.py`:

```python
# Copyright 2025 Your Name
#
# Licensed under the Apache License, Version 2.0

"""Custom LLM provider implementation.

This provider integrates with CustomLLM's API to provide chat completions,
streaming, and tool calling support.

Features:
- Native tool calling support
- Streaming responses
- Circuit breaker protection

References:
- https://customllm.example.com/docs
"""

import json
import logging
import os
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

from victor.providers.base import (
    BaseProvider,
    CompletionResponse,
    Message,
    ProviderError,
    ProviderAuthError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    StreamChunk,
    ToolDefinition,
)

logger = logging.getLogger(__name__)

# Default API endpoint
DEFAULT_BASE_URL = "https://api.customllm.example.com/v1"


class CustomLLMProvider(BaseProvider):
    """Provider for CustomLLM API.

    Features:
    - Native tool calling support
    - Streaming responses
    - Circuit breaker protection
    """

    # Default timeout in seconds
    DEFAULT_TIMEOUT = 60

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = DEFAULT_TIMEOUT,
        max_retries: int = 3,
        **kwargs: Any,
    ):
        """Initialize CustomLLM provider.

        Args:
            api_key: API key (or set CUSTOM_LLM_API_KEY env var)
            base_url: API endpoint URL
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            **kwargs: Additional configuration
        """
        # Resolve API key: parameter > env var > keyring
        resolved_key = api_key or os.environ.get("CUSTOM_LLM_API_KEY", "")
        if not resolved_key:
            try:
                from victor.config.api_keys import get_api_key
                resolved_key = get_api_key("custom_llm") or ""
            except ImportError:
                pass

        if not resolved_key:
            logger.warning(
                "CustomLLM API key not provided. Set CUSTOM_LLM_API_KEY "
                "environment variable or use 'victor keys --set custom_llm --keyring'"
            )

        # Call parent constructor (sets up circuit breaker)
        super().__init__(
            api_key=resolved_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            **kwargs,
        )

        # Initialize HTTP client
        self.client = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(timeout),
            headers={
                "Authorization": f"Bearer {resolved_key}",
                "Content-Type": "application/json",
            },
        )

    @property
    def name(self) -> str:
        """Provider name identifier."""
        return "custom_llm"

    def supports_tools(self) -> bool:
        """Whether this provider supports tool/function calling."""
        return True

    def supports_streaming(self) -> bool:
        """Whether this provider supports streaming responses."""
        return True
```

### Step 2: Implement the chat() Method

The `chat()` method sends a non-streaming request and returns a complete response:

```python
    async def chat(
        self,
        messages: List[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Send chat completion request.

        Args:
            messages: Conversation messages
            model: Model identifier (e.g., "custom-llm-large")
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            tools: Available tools for function calling
            **kwargs: Additional provider-specific parameters

        Returns:
            CompletionResponse with generated content

        Raises:
            ProviderError: If the request fails
        """
        try:
            # Build request payload
            payload = self._build_request_payload(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                stream=False,
                **kwargs,
            )

            # Execute with circuit breaker protection
            response = await self._execute_with_circuit_breaker(
                self.client.post, "/chat/completions", json=payload
            )
            response.raise_for_status()

            # Parse and return response
            result = response.json()
            return self._parse_response(result, model)

        except httpx.TimeoutException as e:
            raise ProviderTimeoutError(
                message=f"Request timed out after {self.timeout}s",
                provider=self.name,
            ) from e
        except httpx.HTTPStatusError as e:
            return self._handle_http_error(e)
        except Exception as e:
            raise ProviderError(
                message=f"Unexpected error: {str(e)}",
                provider=self.name,
                raw_error=e,
            ) from e

    def _build_request_payload(
        self,
        messages: List[Message],
        model: str,
        temperature: float,
        max_tokens: int,
        tools: Optional[List[ToolDefinition]],
        stream: bool,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Build request payload for the API.

        Args:
            messages: Conversation messages
            model: Model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            tools: Available tools
            stream: Whether to stream response
            **kwargs: Additional options

        Returns:
            Request payload dictionary
        """
        # Format messages for the API
        formatted_messages = []
        for msg in messages:
            formatted_msg: Dict[str, Any] = {
                "role": msg.role,
                "content": msg.content,
            }
            # Handle tool results if present
            if msg.tool_call_id:
                formatted_msg["tool_call_id"] = msg.tool_call_id
            formatted_messages.append(formatted_msg)

        payload: Dict[str, Any] = {
            "model": model,
            "messages": formatted_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }

        # Add tools in OpenAI-compatible format
        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
                for tool in tools
            ]
            payload["tool_choice"] = "auto"

        # Merge additional options
        for key, value in kwargs.items():
            if key not in {"api_key"} and value is not None:
                payload[key] = value

        return payload

    def _parse_response(
        self, result: Dict[str, Any], model: str
    ) -> CompletionResponse:
        """Parse API response into CompletionResponse.

        Args:
            result: Raw API response
            model: Model name

        Returns:
            Normalized CompletionResponse
        """
        choices = result.get("choices", [])
        if not choices:
            return CompletionResponse(
                content="",
                role="assistant",
                model=model,
                raw_response=result,
            )

        choice = choices[0]
        message = choice.get("message", {})
        content = message.get("content", "") or ""

        # Parse tool calls if present
        tool_calls = self._normalize_tool_calls(message.get("tool_calls"))

        # Parse usage statistics
        usage = None
        usage_data = result.get("usage")
        if usage_data:
            usage = {
                "prompt_tokens": usage_data.get("prompt_tokens", 0),
                "completion_tokens": usage_data.get("completion_tokens", 0),
                "total_tokens": usage_data.get("total_tokens", 0),
            }

        return CompletionResponse(
            content=content,
            role="assistant",
            tool_calls=tool_calls,
            stop_reason=choice.get("finish_reason"),
            usage=usage,
            model=model,
            raw_response=result,
        )

    def _normalize_tool_calls(
        self, tool_calls: Optional[List[Dict[str, Any]]]
    ) -> Optional[List[Dict[str, Any]]]:
        """Normalize tool calls from API format.

        Args:
            tool_calls: Raw tool calls from API

        Returns:
            Normalized tool calls list
        """
        if not tool_calls:
            return None

        normalized = []
        for call in tool_calls:
            if isinstance(call, dict) and "function" in call:
                function = call.get("function", {})
                name = function.get("name")
                arguments = function.get("arguments", "{}")

                # Parse JSON arguments if they're a string
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {}

                if name:
                    normalized.append({
                        "id": call.get("id", ""),
                        "name": name,
                        "arguments": arguments,
                    })

        return normalized if normalized else None
```

### Step 3: Implement the stream() Method

The `stream()` method provides streaming responses:

```python
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
        """Stream chat completion response.

        Args:
            messages: Conversation messages
            model: Model identifier
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            tools: Available tools for function calling
            **kwargs: Additional provider-specific parameters

        Yields:
            StreamChunk objects with incremental content

        Raises:
            ProviderError: If the request fails
        """
        try:
            payload = self._build_request_payload(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                stream=True,
                **kwargs,
            )

            # Track accumulated tool calls across chunks
            accumulated_tool_calls: List[Dict[str, Any]] = []

            async with self.client.stream(
                "POST", "/chat/completions", json=payload
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line.strip():
                        continue

                    # Handle Server-Sent Events format
                    if line.startswith("data: "):
                        data_str = line[6:]

                        # Check for stream end
                        if data_str.strip() == "[DONE]":
                            yield StreamChunk(
                                content="",
                                tool_calls=(
                                    accumulated_tool_calls
                                    if accumulated_tool_calls else None
                                ),
                                stop_reason="stop",
                                is_final=True,
                            )
                            break

                        try:
                            chunk_data = json.loads(data_str)
                            chunk = self._parse_stream_chunk(
                                chunk_data, accumulated_tool_calls
                            )
                            yield chunk
                        except json.JSONDecodeError:
                            logger.warning(
                                f"JSON decode error on line: {line[:100]}"
                            )

        except httpx.TimeoutException as e:
            raise ProviderTimeoutError(
                message=f"Stream timed out after {self.timeout}s",
                provider=self.name,
            ) from e
        except httpx.HTTPStatusError as e:
            raise self._handle_http_error(e)
        except Exception as e:
            raise ProviderError(
                message=f"Stream error: {str(e)}",
                provider=self.name,
                raw_error=e,
            ) from e

    def _parse_stream_chunk(
        self,
        chunk_data: Dict[str, Any],
        accumulated_tool_calls: List[Dict[str, Any]],
    ) -> StreamChunk:
        """Parse a streaming chunk from the API.

        Args:
            chunk_data: Raw chunk data
            accumulated_tool_calls: List to accumulate tool call deltas

        Returns:
            Normalized StreamChunk
        """
        choices = chunk_data.get("choices", [])
        if not choices:
            return StreamChunk(content="", is_final=False)

        choice = choices[0]
        delta = choice.get("delta", {})
        content = delta.get("content", "") or ""
        finish_reason = choice.get("finish_reason")

        # Handle tool call deltas (for streaming tool calls)
        tool_call_deltas = delta.get("tool_calls", [])
        for tc_delta in tool_call_deltas:
            idx = tc_delta.get("index", 0)

            # Ensure we have a slot for this tool call
            while len(accumulated_tool_calls) <= idx:
                accumulated_tool_calls.append({
                    "id": "",
                    "name": "",
                    "arguments": "",
                })

            # Accumulate tool call data
            if "id" in tc_delta:
                accumulated_tool_calls[idx]["id"] = tc_delta["id"]
            if "function" in tc_delta:
                func_delta = tc_delta["function"]
                if "name" in func_delta:
                    accumulated_tool_calls[idx]["name"] = func_delta["name"]
                if "arguments" in func_delta:
                    accumulated_tool_calls[idx]["arguments"] += (
                        func_delta["arguments"]
                    )

        # Finalize tool calls when stream ends
        final_tool_calls = None
        if finish_reason in ("tool_calls", "stop") and accumulated_tool_calls:
            final_tool_calls = []
            for tc in accumulated_tool_calls:
                if tc.get("name"):
                    args = tc.get("arguments", "{}")
                    try:
                        parsed_args = (
                            json.loads(args) if isinstance(args, str) else args
                        )
                    except json.JSONDecodeError:
                        parsed_args = {}
                    final_tool_calls.append({
                        "id": tc.get("id", ""),
                        "name": tc["name"],
                        "arguments": parsed_args,
                    })

        return StreamChunk(
            content=content,
            tool_calls=final_tool_calls,
            stop_reason=finish_reason,
            is_final=finish_reason is not None,
        )
```

### Step 4: Add Error Handling and close()

```python
    def _handle_http_error(self, error: httpx.HTTPStatusError) -> ProviderError:
        """Handle HTTP errors and convert to appropriate ProviderError.

        Args:
            error: The HTTP error

        Raises:
            ProviderAuthError: For authentication failures
            ProviderRateLimitError: For rate limiting
            ProviderError: For other errors
        """
        status_code = error.response.status_code
        error_body = ""
        try:
            error_body = error.response.text[:500]
        except Exception:
            pass

        if status_code == 401 or status_code == 403:
            raise ProviderAuthError(
                message=f"Authentication failed: {error_body}",
                provider=self.name,
                raw_error=error,
            )
        elif status_code == 429:
            raise ProviderRateLimitError(
                message=f"Rate limit exceeded: {error_body}",
                provider=self.name,
                status_code=429,
                raw_error=error,
            )
        else:
            raise ProviderError(
                message=f"HTTP error {status_code}: {error_body}",
                provider=self.name,
                status_code=status_code,
                raw_error=error,
            )

    async def close(self) -> None:
        """Close HTTP client and release resources."""
        await self.client.aclose()
```

---

## 3. Tool Calling Adapter (Optional)

If your provider does not support native tool calling, or you need custom parsing logic, create a tool calling adapter.

### When to Create an Adapter

- The provider returns tool calls in a non-standard format
- You need to parse tool calls from text content (fallback parsing)
- The provider has model-specific tool calling quirks

### Creating a Tool Calling Adapter

Create `victor/agent/tool_calling/custom_adapter.py`:

```python
"""Tool calling adapter for CustomLLM provider."""

from typing import Any, Dict, List, Optional

from victor.agent.tool_calling.base import (
    BaseToolCallingAdapter,
    FallbackParsingMixin,
    ToolCallingCapabilities,
    ToolCallFormat,
    ToolCallParseResult,
)
from victor.providers.base import ToolDefinition


class CustomLLMAdapter(FallbackParsingMixin, BaseToolCallingAdapter):
    """Tool calling adapter for CustomLLM.

    Handles tool call parsing and conversion for CustomLLM's format.
    """

    @property
    def provider_name(self) -> str:
        return "custom_llm"

    def get_capabilities(self) -> ToolCallingCapabilities:
        """Return tool calling capabilities for this provider/model."""
        return ToolCallingCapabilities(
            native_tool_calls=True,
            streaming_tool_calls=True,
            parallel_tool_calls=True,
            tool_choice_param=True,
            tool_call_format=ToolCallFormat.OPENAI,
            argument_format="json",
            recommended_max_tools=30,
            recommended_tool_budget=15,
        )

    def convert_tools(
        self, tools: List[ToolDefinition]
    ) -> List[Dict[str, Any]]:
        """Convert tools to provider format.

        Args:
            tools: List of standard ToolDefinition objects

        Returns:
            Provider-formatted tool definitions
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in tools
        ]

    def parse_tool_calls(
        self,
        content: str,
        raw_tool_calls: Optional[List[Dict[str, Any]]] = None,
    ) -> ToolCallParseResult:
        """Parse tool calls from response.

        Args:
            content: Response content text
            raw_tool_calls: Native tool_calls from provider

        Returns:
            ToolCallParseResult with parsed tool calls
        """
        # Try native tool calls first
        if raw_tool_calls:
            result = self.parse_native_tool_calls(
                raw_tool_calls,
                validate_name_fn=self.is_valid_tool_name,
            )
            if result.tool_calls:
                return result

        # Fall back to content parsing if native parsing failed
        return self.parse_from_content(
            content,
            validate_name_fn=self.is_valid_tool_name,
        )
```

---

## 4. Model Capabilities Configuration

Add your provider and models to `victor/config/model_capabilities.yaml`:

```yaml
# Add to provider_defaults section
provider_defaults:
  custom_llm:
    native_tool_calls: true
    streaming_tool_calls: true
    parallel_tool_calls: true
    tool_choice_param: true
    requires_strict_prompting: false
    recommended_max_tools: 30
    recommended_tool_budget: 15

# Add to models section
models:
  "custom-llm-large*":
    training:
      tool_calling: true
      code_generation: true

    providers:
      custom_llm:
        native_tool_calls: true
        streaming_tool_calls: true
        parallel_tool_calls: true

    settings:
      recommended_max_tools: 40
      recommended_tool_budget: 20
      argument_format: json

  "custom-llm-small*":
    training:
      tool_calling: true
      code_generation: false

    providers:
      custom_llm:
        native_tool_calls: true
        parallel_tool_calls: false  # Smaller model limitation

    settings:
      recommended_max_tools: 20
      recommended_tool_budget: 10
```

---

## 5. Provider Registration

### Method 1: Internal Registration

Add your provider to `victor/providers/registry.py`:

```python
def _register_default_providers() -> None:
    """Register all default providers."""
    # ... existing providers ...

    # Add your provider
    from victor.providers.custom_provider import CustomLLMProvider
    ProviderRegistry.register("custom_llm", CustomLLMProvider)
    ProviderRegistry.register("customllm", CustomLLMProvider)  # Alias
```

### Method 2: Plugin Registration (External)

For external packages, use entry points in `pyproject.toml`:

```toml
[project.entry-points."victor.providers"]
custom_llm = "victor_custom:CustomLLMProvider"
```

---

## 6. Testing Your Provider

### Unit Tests

Create `tests/unit/providers/test_custom_provider.py`:

```python
"""Unit tests for CustomLLM provider."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from victor.providers.custom_provider import CustomLLMProvider
from victor.providers.base import (
    Message,
    ToolDefinition,
    ProviderError,
    ProviderAuthError,
    ProviderRateLimitError,
)


@pytest.fixture
def custom_provider():
    """Create CustomLLMProvider instance for testing."""
    return CustomLLMProvider(
        api_key="test-api-key",
        base_url="https://api.customllm.example.com/v1",
        timeout=30,
    )


class TestCustomLLMProvider:
    """Tests for CustomLLMProvider."""

    def test_initialization(self, custom_provider):
        """Test provider initializes correctly."""
        assert custom_provider.name == "custom_llm"
        assert custom_provider.supports_tools() is True
        assert custom_provider.supports_streaming() is True

    @pytest.mark.asyncio
    async def test_chat_success(self, custom_provider):
        """Test successful chat completion."""
        # Mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": "Hello! How can I help?",
                    "role": "assistant",
                },
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 8,
                "total_tokens": 18,
            },
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(
            custom_provider.client, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value = mock_response

            messages = [Message(role="user", content="Hello")]
            response = await custom_provider.chat(
                messages=messages,
                model="custom-llm-large",
            )

            assert response.content == "Hello! How can I help?"
            assert response.role == "assistant"
            assert response.usage["prompt_tokens"] == 10

    @pytest.mark.asyncio
    async def test_chat_with_tools(self, custom_provider):
        """Test chat with tool calling."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": "",
                    "role": "assistant",
                    "tool_calls": [{
                        "id": "call_123",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "London"}',
                        },
                    }],
                },
                "finish_reason": "tool_calls",
            }],
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(
            custom_provider.client, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value = mock_response

            tools = [
                ToolDefinition(
                    name="get_weather",
                    description="Get weather for a location",
                    parameters={
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"},
                        },
                    },
                )
            ]

            messages = [Message(role="user", content="Weather in London?")]
            response = await custom_provider.chat(
                messages=messages,
                model="custom-llm-large",
                tools=tools,
            )

            assert response.tool_calls is not None
            assert len(response.tool_calls) == 1
            assert response.tool_calls[0]["name"] == "get_weather"
            assert response.tool_calls[0]["arguments"] == {"location": "London"}

    @pytest.mark.asyncio
    async def test_chat_auth_error(self, custom_provider):
        """Test authentication error handling."""
        import httpx

        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Invalid API key"

        error = httpx.HTTPStatusError(
            "Auth failed",
            request=MagicMock(),
            response=mock_response,
        )

        with patch.object(
            custom_provider.client, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value.raise_for_status.side_effect = error

            messages = [Message(role="user", content="Hello")]

            with pytest.raises(ProviderAuthError):
                await custom_provider.chat(
                    messages=messages,
                    model="custom-llm-large",
                )

    @pytest.mark.asyncio
    async def test_chat_rate_limit_error(self, custom_provider):
        """Test rate limit error handling."""
        import httpx

        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.text = "Rate limit exceeded"

        error = httpx.HTTPStatusError(
            "Rate limited",
            request=MagicMock(),
            response=mock_response,
        )

        with patch.object(
            custom_provider.client, "post", new_callable=AsyncMock
        ) as mock_post:
            mock_post.return_value.raise_for_status.side_effect = error

            messages = [Message(role="user", content="Hello")]

            with pytest.raises(ProviderRateLimitError):
                await custom_provider.chat(
                    messages=messages,
                    model="custom-llm-large",
                )

    @pytest.mark.asyncio
    async def test_close(self, custom_provider):
        """Test closing the provider."""
        with patch.object(
            custom_provider.client, "aclose", new_callable=AsyncMock
        ) as mock_close:
            await custom_provider.close()
            mock_close.assert_called_once()


class TestCustomLLMProviderStreaming:
    """Tests for streaming functionality."""

    @pytest.mark.asyncio
    async def test_stream_success(self, custom_provider):
        """Test successful streaming."""
        # Mock streaming response
        async def mock_aiter_lines():
            lines = [
                'data: {"choices":[{"delta":{"content":"Hello"}}]}',
                'data: {"choices":[{"delta":{"content":" world"}}]}',
                'data: {"choices":[{"finish_reason":"stop"}]}',
                'data: [DONE]',
            ]
            for line in lines:
                yield line

        mock_response = MagicMock()
        mock_response.aiter_lines = mock_aiter_lines
        mock_response.raise_for_status = MagicMock()

        mock_context = MagicMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_context.__aexit__ = AsyncMock()

        with patch.object(
            custom_provider.client, "stream"
        ) as mock_stream:
            mock_stream.return_value = mock_context

            messages = [Message(role="user", content="Hello")]
            chunks = []

            async for chunk in custom_provider.stream(
                messages=messages,
                model="custom-llm-large",
            ):
                chunks.append(chunk)

            # Verify chunks received
            assert len(chunks) >= 2
            content = "".join(c.content for c in chunks)
            assert "Hello" in content
```

### Running Tests

```bash
# Run provider tests
pytest tests/unit/providers/test_custom_provider.py -v

# Run with coverage
pytest tests/unit/providers/test_custom_provider.py --cov=victor/providers/custom_provider

# Run all provider tests
pytest tests/unit/providers/ -v
```

### Integration Testing

For integration testing with a real API:

```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_api_chat():
    """Integration test with real API (requires API key)."""
    import os

    api_key = os.environ.get("CUSTOM_LLM_API_KEY")
    if not api_key:
        pytest.skip("CUSTOM_LLM_API_KEY not set")

    provider = CustomLLMProvider(api_key=api_key)

    try:
        messages = [Message(role="user", content="Say 'Hello, Victor!'")]
        response = await provider.chat(
            messages=messages,
            model="custom-llm-large",
            max_tokens=50,
        )

        assert response.content
        assert len(response.content) > 0
    finally:
        await provider.close()
```

---

## 7. Error Handling Best Practices

### Error Types

Victor provides specific error types for different failure modes:

| Error Type | When to Use |
|------------|-------------|
| `ProviderError` | Base class for all provider errors |
| `ProviderAuthError` | Authentication/authorization failures (401, 403) |
| `ProviderRateLimitError` | Rate limiting (429) |
| `ProviderTimeoutError` | Request timeouts |
| `ProviderConnectionError` | Network connectivity issues |
| `ProviderInvalidResponseError` | Malformed API responses |

### Circuit Breaker

The base class includes circuit breaker support. Use it for API calls:

```python
# Protected API call
response = await self._execute_with_circuit_breaker(
    self.client.post, "/chat/completions", json=payload
)

# Check circuit state
if self.is_circuit_open():
    logger.warning("Circuit breaker is open, failing fast")

# Get circuit stats
stats = self.get_circuit_breaker_stats()
```

### Retry Strategy

For transient failures, implement exponential backoff:

```python
import asyncio
from typing import TypeVar

T = TypeVar("T")

async def retry_with_backoff(
    func,
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> T:
    """Retry a function with exponential backoff."""
    last_exception = None

    for attempt in range(max_retries):
        try:
            return await func()
        except ProviderRateLimitError as e:
            last_exception = e
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                await asyncio.sleep(delay)

    raise last_exception
```

---

## 8. Complete Provider Example

Here is the complete provider implementation file:

```python
# victor/providers/custom_provider.py
"""Complete CustomLLM provider implementation."""

import json
import logging
import os
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

from victor.providers.base import (
    BaseProvider,
    CompletionResponse,
    Message,
    ProviderError,
    ProviderAuthError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    StreamChunk,
    ToolDefinition,
)

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.customllm.example.com/v1"


class CustomLLMProvider(BaseProvider):
    """Provider for CustomLLM API."""

    DEFAULT_TIMEOUT = 60

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = DEFAULT_TIMEOUT,
        max_retries: int = 3,
        **kwargs: Any,
    ):
        resolved_key = api_key or os.environ.get("CUSTOM_LLM_API_KEY", "")
        if not resolved_key:
            try:
                from victor.config.api_keys import get_api_key
                resolved_key = get_api_key("custom_llm") or ""
            except ImportError:
                pass

        super().__init__(
            api_key=resolved_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            **kwargs,
        )

        self.client = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(timeout),
            headers={
                "Authorization": f"Bearer {resolved_key}",
                "Content-Type": "application/json",
            },
        )

    @property
    def name(self) -> str:
        return "custom_llm"

    def supports_tools(self) -> bool:
        return True

    def supports_streaming(self) -> bool:
        return True

    async def chat(
        self,
        messages: List[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        try:
            payload = self._build_request_payload(
                messages, model, temperature, max_tokens, tools, False, **kwargs
            )

            response = await self._execute_with_circuit_breaker(
                self.client.post, "/chat/completions", json=payload
            )
            response.raise_for_status()

            return self._parse_response(response.json(), model)

        except httpx.TimeoutException as e:
            raise ProviderTimeoutError(
                message=f"Request timed out after {self.timeout}s",
                provider=self.name,
            ) from e
        except httpx.HTTPStatusError as e:
            raise self._handle_http_error(e)
        except Exception as e:
            raise ProviderError(
                message=f"Unexpected error: {str(e)}",
                provider=self.name,
                raw_error=e,
            ) from e

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
        try:
            payload = self._build_request_payload(
                messages, model, temperature, max_tokens, tools, True, **kwargs
            )

            accumulated_tool_calls: List[Dict[str, Any]] = []

            async with self.client.stream(
                "POST", "/chat/completions", json=payload
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line.strip():
                        continue

                    if line.startswith("data: "):
                        data_str = line[6:]

                        if data_str.strip() == "[DONE]":
                            yield StreamChunk(
                                content="",
                                tool_calls=(
                                    accumulated_tool_calls
                                    if accumulated_tool_calls else None
                                ),
                                stop_reason="stop",
                                is_final=True,
                            )
                            break

                        try:
                            chunk_data = json.loads(data_str)
                            yield self._parse_stream_chunk(
                                chunk_data, accumulated_tool_calls
                            )
                        except json.JSONDecodeError:
                            logger.warning(f"JSON error: {line[:100]}")

        except httpx.TimeoutException as e:
            raise ProviderTimeoutError(
                message=f"Stream timed out",
                provider=self.name,
            ) from e
        except httpx.HTTPStatusError as e:
            raise self._handle_http_error(e)
        except Exception as e:
            raise ProviderError(
                message=f"Stream error: {str(e)}",
                provider=self.name,
                raw_error=e,
            ) from e

    def _build_request_payload(
        self,
        messages: List[Message],
        model: str,
        temperature: float,
        max_tokens: int,
        tools: Optional[List[ToolDefinition]],
        stream: bool,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        formatted_messages = []
        for msg in messages:
            formatted_msg: Dict[str, Any] = {
                "role": msg.role,
                "content": msg.content,
            }
            if msg.tool_call_id:
                formatted_msg["tool_call_id"] = msg.tool_call_id
            formatted_messages.append(formatted_msg)

        payload: Dict[str, Any] = {
            "model": model,
            "messages": formatted_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }

        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
                for tool in tools
            ]
            payload["tool_choice"] = "auto"

        for key, value in kwargs.items():
            if key not in {"api_key"} and value is not None:
                payload[key] = value

        return payload

    def _parse_response(
        self, result: Dict[str, Any], model: str
    ) -> CompletionResponse:
        choices = result.get("choices", [])
        if not choices:
            return CompletionResponse(
                content="", role="assistant", model=model, raw_response=result
            )

        choice = choices[0]
        message = choice.get("message", {})

        usage = None
        usage_data = result.get("usage")
        if usage_data:
            usage = {
                "prompt_tokens": usage_data.get("prompt_tokens", 0),
                "completion_tokens": usage_data.get("completion_tokens", 0),
                "total_tokens": usage_data.get("total_tokens", 0),
            }

        return CompletionResponse(
            content=message.get("content", "") or "",
            role="assistant",
            tool_calls=self._normalize_tool_calls(message.get("tool_calls")),
            stop_reason=choice.get("finish_reason"),
            usage=usage,
            model=model,
            raw_response=result,
        )

    def _normalize_tool_calls(
        self, tool_calls: Optional[List[Dict[str, Any]]]
    ) -> Optional[List[Dict[str, Any]]]:
        if not tool_calls:
            return None

        normalized = []
        for call in tool_calls:
            if "function" in call:
                function = call.get("function", {})
                name = function.get("name")
                arguments = function.get("arguments", "{}")

                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {}

                if name:
                    normalized.append({
                        "id": call.get("id", ""),
                        "name": name,
                        "arguments": arguments,
                    })

        return normalized if normalized else None

    def _parse_stream_chunk(
        self,
        chunk_data: Dict[str, Any],
        accumulated_tool_calls: List[Dict[str, Any]],
    ) -> StreamChunk:
        choices = chunk_data.get("choices", [])
        if not choices:
            return StreamChunk(content="", is_final=False)

        choice = choices[0]
        delta = choice.get("delta", {})
        finish_reason = choice.get("finish_reason")

        for tc_delta in delta.get("tool_calls", []):
            idx = tc_delta.get("index", 0)
            while len(accumulated_tool_calls) <= idx:
                accumulated_tool_calls.append({"id": "", "name": "", "arguments": ""})

            if "id" in tc_delta:
                accumulated_tool_calls[idx]["id"] = tc_delta["id"]
            if "function" in tc_delta:
                func = tc_delta["function"]
                if "name" in func:
                    accumulated_tool_calls[idx]["name"] = func["name"]
                if "arguments" in func:
                    accumulated_tool_calls[idx]["arguments"] += func["arguments"]

        final_tool_calls = None
        if finish_reason and accumulated_tool_calls:
            final_tool_calls = []
            for tc in accumulated_tool_calls:
                if tc.get("name"):
                    args = tc.get("arguments", "{}")
                    try:
                        parsed = json.loads(args) if isinstance(args, str) else args
                    except json.JSONDecodeError:
                        parsed = {}
                    final_tool_calls.append({
                        "id": tc.get("id", ""),
                        "name": tc["name"],
                        "arguments": parsed,
                    })

        return StreamChunk(
            content=delta.get("content", "") or "",
            tool_calls=final_tool_calls,
            stop_reason=finish_reason,
            is_final=finish_reason is not None,
        )

    def _handle_http_error(self, error: httpx.HTTPStatusError) -> ProviderError:
        status = error.response.status_code
        body = error.response.text[:500] if error.response.text else ""

        if status in (401, 403):
            raise ProviderAuthError(
                message=f"Authentication failed: {body}",
                provider=self.name,
                raw_error=error,
            )
        elif status == 429:
            raise ProviderRateLimitError(
                message=f"Rate limit exceeded: {body}",
                provider=self.name,
                status_code=429,
                raw_error=error,
            )
        else:
            raise ProviderError(
                message=f"HTTP error {status}: {body}",
                provider=self.name,
                status_code=status,
                raw_error=error,
            )

    async def close(self) -> None:
        await self.client.aclose()
```

---

## Summary

You have learned how to:

1. Create a provider class inheriting from `BaseProvider`
2. Implement required methods: `name`, `chat()`, `stream()`, `close()`
3. Handle tool calling with proper normalization
4. Configure model capabilities in YAML
5. Register your provider
6. Write comprehensive tests

For more advanced topics, see:
- [Tool Calling Adapters](../reference/tools/tool-calling.md)
- [Circuit Breaker Configuration](../guides/RESILIENCE.md)
- [Provider Registry](../user-guide/providers.md)

---

## Quick Reference

### Provider Checklist

- [ ] Create `victor/providers/{name}_provider.py`
- [ ] Inherit from `BaseProvider`
- [ ] Implement `name` property
- [ ] Implement `chat()` method
- [ ] Implement `stream()` method
- [ ] Implement `close()` method
- [ ] Override `supports_tools()` if applicable
- [ ] Override `supports_streaming()` if applicable
- [ ] Add to `model_capabilities.yaml`
- [ ] Register in `registry.py`
- [ ] Create unit tests
- [ ] Create integration tests (optional)

### Import Locations

```python
# Provider base classes and types
from victor.providers.base import (
    BaseProvider,
    Message,
    CompletionResponse,
    StreamChunk,
    ToolDefinition,
    ProviderError,
    ProviderAuthError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)

# Provider registry
from victor.providers.registry import ProviderRegistry

# Tool calling adapter base
from victor.agent.tool_calling.base import (
    BaseToolCallingAdapter,
    FallbackParsingMixin,
    ToolCallingCapabilities,
    ToolCallFormat,
)
```
