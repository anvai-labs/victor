# Providers API Reference

This document provides comprehensive API documentation for Victor's LLM provider system, which supports 25 different providers with a unified interface.

## Table of Contents

1. [BaseProvider Interface](#baseprovider-interface)
2. [Provider Protocols](#provider-protocols)
3. [Key Classes](#key-classes)
4. [Provider Implementation Pattern](#provider-implementation-pattern)
5. [Configuration](#configuration)
6. [Error Handling](#error-handling)

---

## BaseProvider Interface

All LLM providers in Victor inherit from `BaseProvider`, which defines the core contract for interacting with language models.

**Import:**
```python
from victor.providers.base import BaseProvider
```

### Constructor

```python
class BaseProvider(ABC):
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 3,
        use_circuit_breaker: bool = True,
        circuit_breaker_failure_threshold: int = 5,
        circuit_breaker_recovery_timeout: float = 30.0,
        **kwargs: Any,
    ):
        """Initialize provider.

        Args:
            api_key: API key for authentication (if required)
            base_url: Base URL for API endpoints
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            use_circuit_breaker: Whether to enable circuit breaker protection
            circuit_breaker_failure_threshold: Failures before opening circuit
            circuit_breaker_recovery_timeout: Seconds before testing recovery
            **kwargs: Additional provider-specific options
        """
```

### Abstract Properties

#### `name`
```python
@property
@abstractmethod
def name(self) -> str:
    """Provider name (e.g., 'anthropic', 'openai', 'ollama')."""
```

### Abstract Methods

#### `chat()`
```python
@abstractmethod
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
    """Send a chat completion request.

    Args:
        messages: List of conversation messages
        model: Model identifier (e.g., 'claude-sonnet-4-20250514')
        temperature: Sampling temperature (0-2)
        max_tokens: Maximum tokens to generate
        tools: Available tools for the model to use
        **kwargs: Additional provider-specific parameters

    Returns:
        CompletionResponse with generated content

    Raises:
        ProviderError: If the request fails
    """
```

#### `stream()`
```python
@abstractmethod
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
    """Stream a chat completion response.

    Args:
        messages: List of conversation messages
        model: Model identifier
        temperature: Sampling temperature (0-2)
        max_tokens: Maximum tokens to generate
        tools: Available tools for the model to use
        **kwargs: Additional provider-specific parameters

    Yields:
        StreamChunk objects with incremental content

    Raises:
        ProviderError: If the request fails
    """
```

#### `close()`
```python
@abstractmethod
async def close(self) -> None:
    """Close any open connections or resources."""
```

### Optional Methods

#### `supports_tools()`
```python
def supports_tools(self) -> bool:
    """Check if provider supports tool/function calling.

    Default implementation returns False. Providers that support
    tool calling should override this method to return True.

    Returns:
        True if provider supports tools, False otherwise (default)
    """
```

#### `supports_streaming()`
```python
def supports_streaming(self) -> bool:
    """Check if provider supports streaming responses.

    Default implementation returns False. Providers that support
    streaming should override this method to return True.

    Returns:
        True if provider supports streaming, False otherwise (default)
    """
```

#### `stream_chat()`
```python
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
    """Stream a chat completion response (alias for stream()).

    This is an alias for the stream() method, provided for compatibility
    with different naming conventions across SDKs.
    """
```

#### `discover_capabilities()`
```python
async def discover_capabilities(self, model: str) -> ProviderRuntimeCapabilities:
    """Discover capabilities for the given model.

    Default implementation falls back to configured limits and
    provider-declared support flags. Providers should override
    with real HTTP-based discovery when available.

    Returns:
        ProviderRuntimeCapabilities with context window, tool support, etc.
    """
```

#### `count_tokens()`
```python
async def count_tokens(self, text: str) -> int:
    """Estimate token count for given text.

    Args:
        text: Text to count tokens for

    Returns:
        Estimated token count (default: ~4 characters per token)
    """
```

### Circuit Breaker Methods

#### `is_circuit_open()`
```python
def is_circuit_open(self) -> bool:
    """Check if the circuit breaker is open (failing fast).

    Returns:
        True if circuit is open and requests will be rejected
    """
```

#### `get_circuit_breaker_stats()`
```python
def get_circuit_breaker_stats(self) -> Optional[Dict[str, Any]]:
    """Get circuit breaker statistics for monitoring.

    Returns:
        Dictionary with stats or None if circuit breaker disabled
    """
```

#### `reset_circuit_breaker()`
```python
def reset_circuit_breaker(self) -> None:
    """Manually reset the circuit breaker to closed state."""
```

---

## Provider Capabilities

Provider capabilities are a **single surface**: the `supports_*()` methods on
`BaseProvider`. Each follows the Interface Segregation Principle — it defaults
to `False`, and a provider opts in by overriding only what it supports. Because
the methods live on the base class, every provider answers every query
(Liskov), so callers use a plain method call — no `isinstance`, protocol, or
helper-function layer is needed.

```python
if provider.supports_streaming():
    async for chunk in provider.stream(messages, model=model):
        print(chunk.content)

if provider.supports_tools():
    response = await provider.chat(messages, model=model, tools=my_tools)

if provider.supports_vision():
    response = await provider.chat([image_message], model=model)
```

The capability methods on `BaseProvider` (all default `False`):

| Method | Capability |
| --- | --- |
| `supports_tools()` | Tool / function calling |
| `supports_streaming()` | Incremental streaming responses |
| `supports_vision()` | Multimodal (image) input |
| `supports_prompt_caching()` | API-level cached-token billing discount |
| `supports_kv_prefix_caching()` | KV prefix reuse (latency, not billing) |
| `supports_reasoning_effort(model)` | Per-model `reasoning_effort` parameter |

To add a capability to a provider, override the method:

```python
class MyProvider(BaseProvider):
    def supports_streaming(self) -> bool:
        return True

    def supports_tools(self) -> bool:
        return True
```

`supports_reasoning_effort(model)` takes the model id because support can vary
per model — expressiveness a boolean protocol conformance check cannot capture,
which is why the capability surface is method-based rather than Protocol-based.

---

## Key Classes

### Message

Standard message format across all providers.

```python
from victor.providers.base import Message

class Message(BaseModel):
    """Standard message format across all providers."""

    role: str
    """Message role: 'system', 'user', or 'assistant'"""

    content: str
    """Message content"""

    name: Optional[str] = None
    """Optional name for the message sender"""

    tool_calls: Optional[List[Dict[str, Any]]] = None
    """Tool calls requested by the assistant"""

    tool_call_id: Optional[str] = None
    """ID of the tool call being responded to"""

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary representation."""
```

### ToolDefinition

Standard tool definition format.

```python
from victor.providers.base import ToolDefinition

class ToolDefinition(BaseModel):
    """Standard tool definition format."""

    name: str
    """Tool name"""

    description: str
    """What the tool does"""

    parameters: Dict[str, Any]
    """JSON Schema for tool parameters"""
```

### CompletionResponse

Standard completion response format.

```python
from victor.providers.base import CompletionResponse

class CompletionResponse(BaseModel):
    """Standard completion response format."""

    content: str
    """Generated content"""

    role: str = "assistant"
    """Response role"""

    tool_calls: Optional[List[Dict[str, Any]]] = None
    """Tool calls requested by the model. Each tool call has:
    - id: Unique identifier for the tool call
    - name: Name of the tool to call
    - arguments: Dict or JSON string of arguments
    """

    stop_reason: Optional[str] = None
    """Why generation stopped (e.g., 'stop', 'length', 'tool_use')"""

    usage: Optional[Dict[str, int]] = None
    """Token usage stats:
    - prompt_tokens: Input tokens
    - completion_tokens: Output tokens
    - total_tokens: Total tokens
    """

    model: Optional[str] = None
    """Model used for generation"""

    raw_response: Optional[Dict[str, Any]] = None
    """Raw provider response for debugging"""

    metadata: Optional[Dict[str, Any]] = None
    """Additional metadata (e.g., reasoning_content for thinking models)"""
```

### StreamChunk

Streaming response chunk.

```python
from victor.providers.base import StreamChunk

class StreamChunk(BaseModel):
    """Streaming response chunk."""

    content: str = ""
    """Incremental content"""

    tool_calls: Optional[List[Dict[str, Any]]] = None
    """Tool calls (typically on final chunk)"""

    stop_reason: Optional[str] = None
    """Why generation stopped"""

    is_final: bool = False
    """Is this the final chunk"""

    metadata: Optional[Dict[str, Any]] = None
    """Additional metadata (e.g., reasoning_content)"""

    usage: Optional[Dict[str, int]] = None
    """Token usage stats (typically on final chunk). Keys:
    - prompt_tokens
    - completion_tokens
    - total_tokens
    - cache_creation_input_tokens (Anthropic)
    - cache_read_input_tokens (Anthropic)
    """
```

### ProviderRuntimeCapabilities

Runtime capabilities discovered for a provider/model pair.

```python
from victor.providers.runtime_capabilities import ProviderRuntimeCapabilities

@dataclass
class ProviderRuntimeCapabilities:
    """Runtime capabilities and limits for a provider/model pair."""

    provider: str
    """Provider name"""

    model: str
    """Model identifier"""

    context_window: int
    """Maximum context window in tokens"""

    supports_tools: bool
    """Whether the model supports tool calling"""

    supports_streaming: bool
    """Whether the model supports streaming"""

    source: str = "config"
    """How capabilities were determined: 'discovered' or 'config'"""

    raw: Optional[Dict[str, Any]] = None
    """Raw discovery response"""
```

### ProviderRegistry

Registry for managing and discovering LLM providers.

```python
from victor.providers.registry import ProviderRegistry

class ProviderRegistry:
    """Registry for LLM provider management.

    This is a static class facade that maintains backward compatibility
    while delegating to a BaseRegistry-based implementation.
    """

    @classmethod
    def register(cls, name: str, provider_class: Type[BaseProvider]) -> None:
        """Register a provider.

        Args:
            name: Provider name (e.g., "ollama", "anthropic")
            provider_class: Provider class
        """

    @classmethod
    def get(cls, name: str) -> Type[BaseProvider]:
        """Get a provider class by name.

        Args:
            name: Provider name

        Returns:
            Provider class

        Raises:
            ProviderNotFoundError: If provider not found
        """

    @classmethod
    def get_optional(cls, name: str) -> Optional[Type[BaseProvider]]:
        """Get a provider class by name, returning None if not found."""

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> BaseProvider:
        """Create a provider instance.

        Args:
            name: Provider name
            **kwargs: Provider initialization arguments

        Returns:
            Provider instance

        Raises:
            ProviderNotFoundError: If provider not found
        """

    @classmethod
    def list_providers(cls) -> List[str]:
        """List all registered provider names.

        Returns:
            List of provider names
        """

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a provider is registered."""

    @classmethod
    def unregister(cls, name: str) -> bool:
        """Unregister a provider."""

    @classmethod
    def clear(cls) -> None:
        """Clear all registered providers."""
```

### ProviderManager

Unified management of providers and models with health monitoring and fallback handling.

```python
from victor.agent.provider_manager import ProviderManager, ProviderManagerConfig

@dataclass
class ProviderManagerConfig:
    """Configuration for ProviderManager."""

    enable_health_checks: bool = True
    """Enable provider health monitoring"""

    health_check_interval: float = 60.0
    """Interval between health checks (seconds)"""

    auto_fallback: bool = True
    """Automatically fallback to healthy provider on failure"""

    fallback_providers: List[str] = field(default_factory=list)
    """Ordered list of fallback providers"""

    max_fallback_attempts: int = 3
    """Maximum fallback attempts before giving up"""


class ProviderManager:
    """Unified management of providers and models.

    Features:
    - Hot-swap providers without losing context
    - Automatic health monitoring
    - Fallback chain support
    - Switch history tracking
    - Tool capability detection
    """

    def __init__(
        self,
        settings: Any,
        initial_provider: Optional[BaseProvider] = None,
        initial_model: Optional[str] = None,
        provider_name: Optional[str] = None,
        config: Optional[ProviderManagerConfig] = None,
    ):
        """Initialize the provider manager."""

    @property
    def provider(self) -> Optional[BaseProvider]:
        """Get current provider instance."""

    @property
    def provider_name(self) -> str:
        """Get current provider name."""

    @property
    def model(self) -> str:
        """Get current model name."""

    async def switch_provider(
        self,
        provider_name: str,
        model: Optional[str] = None,
        reason: SwitchReason = SwitchReason.USER_REQUEST,
        **provider_kwargs: Any,
    ) -> bool:
        """Switch to a different provider.

        Args:
            provider_name: Name of the provider
            model: Optional model name
            reason: Reason for the switch
            **provider_kwargs: Additional provider arguments

        Returns:
            True if switch was successful
        """

    async def switch_model(
        self,
        model: str,
        reason: SwitchReason = SwitchReason.USER_REQUEST,
    ) -> bool:
        """Switch to a different model on the current provider."""

    def get_info(self) -> Dict[str, Any]:
        """Get information about current provider and model.

        Returns:
            Dictionary with provider/model info and capabilities
        """

    async def get_healthy_providers(self) -> List[str]:
        """Get list of healthy providers sorted by latency."""

    async def close(self) -> None:
        """Close provider and cleanup."""
```

### ManagedProviderFactory

Factory for creating providers with shared infrastructure (resilience, rate limiting, metrics).

```python
from victor.providers.factory import ManagedProviderFactory, ManagedProvider

class ManagedProviderFactory:
    """Factory for creating providers with shared infrastructure.

    Automatically integrates:
    - Circuit breaker and retry logic
    - Rate limiting with priority queues
    - Streaming metrics collection
    - Fallback provider chains
    """

    @classmethod
    async def create(
        cls,
        provider_name: str,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 120.0,
        enable_resilience: bool = True,
        enable_rate_limiting: bool = True,
        enable_metrics: bool = True,
        fallback_configs: Optional[List[ProviderConfig]] = None,
        **kwargs: Any,
    ) -> ManagedProvider:
        """Create an enhanced provider with shared infrastructure.

        Args:
            provider_name: Provider name (anthropic, openai, ollama, etc.)
            model: Model identifier
            api_key: API key for cloud providers
            base_url: Base URL override
            timeout: Request timeout
            enable_resilience: Enable circuit breaker and retry
            enable_rate_limiting: Enable rate limiting
            enable_metrics: Enable streaming metrics
            fallback_configs: Fallback provider configurations
            **kwargs: Additional provider-specific arguments

        Returns:
            ManagedProvider with all integrations
        """
```

---

## Provider Implementation Pattern

To implement a custom provider, follow this pattern:

```python
from typing import Any, AsyncIterator, Dict, List, Optional

from victor.providers.base import (
    BaseProvider,
    CompletionResponse,
    Message,
    ProviderError,
    StreamChunk,
    ToolDefinition,
)


class CustomProvider(BaseProvider):
    """Custom LLM provider implementation."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 3,
        **kwargs: Any,
    ):
        """Initialize custom provider.

        Args:
            api_key: API key for authentication
            base_url: Optional base URL for API
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            **kwargs: Additional configuration
        """
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            **kwargs,
        )
        # Initialize your client here
        self.client = YourAPIClient(api_key=api_key, ...)

    @property
    def name(self) -> str:
        """Provider name."""
        return "custom"

    def supports_tools(self) -> bool:
        """Return True if your provider supports tool calling."""
        return True

    def supports_streaming(self) -> bool:
        """Return True if your provider supports streaming."""
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
        """Send chat completion request.

        Args:
            messages: Conversation messages
            model: Model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            tools: Available tools
            **kwargs: Additional parameters

        Returns:
            CompletionResponse with generated content
        """
        try:
            # Convert messages to your provider's format
            formatted_messages = self._format_messages(messages)

            # Make API call with circuit breaker protection
            response = await self._execute_with_circuit_breaker(
                self.client.chat,
                messages=formatted_messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=self._convert_tools(tools) if tools else None,
                **kwargs,
            )

            # Parse and return normalized response
            return self._parse_response(response, model)

        except Exception as e:
            raise self._handle_error(e)

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

        Yields:
            StreamChunk objects with incremental content
        """
        try:
            formatted_messages = self._format_messages(messages)

            async for chunk in self.client.stream(
                messages=formatted_messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            ):
                yield StreamChunk(
                    content=chunk.text or "",
                    is_final=chunk.is_done,
                    tool_calls=chunk.tool_calls if chunk.is_done else None,
                    stop_reason="stop" if chunk.is_done else None,
                )

        except Exception as e:
            raise self._handle_error(e)

    async def close(self) -> None:
        """Close HTTP client."""
        await self.client.close()

    def _format_messages(self, messages: List[Message]) -> List[Dict]:
        """Convert standard messages to provider format."""
        return [{"role": m.role, "content": m.content} for m in messages]

    def _convert_tools(self, tools: List[ToolDefinition]) -> List[Dict]:
        """Convert standard tools to provider format."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            }
            for t in tools
        ]

    def _parse_response(self, response: Any, model: str) -> CompletionResponse:
        """Parse API response to standard format."""
        return CompletionResponse(
            content=response.text,
            role="assistant",
            tool_calls=response.tool_calls,
            stop_reason=response.stop_reason,
            usage={
                "prompt_tokens": response.usage.input,
                "completion_tokens": response.usage.output,
                "total_tokens": response.usage.total,
            },
            model=model,
        )

    def _handle_error(self, error: Exception) -> ProviderError:
        """Convert API errors to standard ProviderError."""
        raise ProviderError(
            message=f"Custom provider error: {error}",
            provider=self.name,
            raw_error=error,
        )
```

### Registering Your Provider

```python
from victor.providers.registry import ProviderRegistry

# Register your provider
ProviderRegistry.register("custom", CustomProvider)

# Create an instance
provider = ProviderRegistry.create(
    "custom",
    api_key="your-api-key",
    base_url="https://api.example.com",
)
```

---

## Configuration

### Model Capabilities

Model capabilities are configured in `victor/config/model_capabilities.yaml`. This file defines:

1. **Global Defaults** - Default capability assumptions for unknown models
2. **Provider Defaults** - Infrastructure-level capabilities for each provider
3. **Model Configurations** - Specific settings for model patterns

#### Schema Structure

```yaml
schema_version: "0.2.0"

# Global defaults for unknown models
defaults:
  training:
    tool_calling: false
    thinking_mode: false
    code_generation: false
    vision: false
    reasoning: false

  provider_support:
    native_tool_calls: false
    streaming_tool_calls: false
    parallel_tool_calls: false
    tool_choice_param: false

  settings:
    recommended_max_tools: 15
    recommended_tool_budget: 10
    requires_strict_prompting: true
    argument_format: json
    exploration_multiplier: 1.0
    continuation_patience: 10
    timeout_multiplier: 1.0
    tool_reliability: medium

# Provider-level defaults
provider_defaults:
  anthropic:
    native_tool_calls: true
    streaming_tool_calls: true
    parallel_tool_calls: true
    tool_choice_param: true
    requires_strict_prompting: false
    recommended_max_tools: 50
    recommended_tool_budget: 20

  openai:
    native_tool_calls: true
    streaming_tool_calls: true
    parallel_tool_calls: true
    # ...

  ollama:
    native_tool_calls: false  # Must be enabled per-model
    timeout_multiplier: 2.0   # Local inference needs longer timeouts
    # ...

# Model-specific configurations
models:
  "claude-*":
    training:
      tool_calling: true
      code_generation: true

    providers:
      anthropic:
        native_tool_calls: true
        streaming_tool_calls: true
        parallel_tool_calls: true

    settings:
      recommended_max_tools: 50
      recommended_tool_budget: 25
      argument_format: json
```

#### Resolution Order

Capabilities are resolved in this order (later overrides earlier):

1. `defaults`
2. `provider_defaults.<provider>`
3. `models.<pattern>.training` (inherent capabilities)
4. `models.<pattern>.providers.<provider>` (provider-specific overrides)
5. `models.<pattern>.settings` (tuning)

### Supported Providers

Victor supports 25 LLM providers out of the box:

| Provider | Name | Type | Tool Calling |
|----------|------|------|--------------|
| Anthropic | `anthropic` | Cloud | Native |
| OpenAI | `openai` | Cloud | Native |
| Google AI | `google` | Cloud | Native |
| xAI (Grok) | `xai`, `grok` | Cloud | Native |
| Azure OpenAI | `azure`, `azure-openai` | Enterprise | Native |
| AWS Bedrock | `bedrock`, `aws` | Enterprise | Native |
| Vertex AI | `vertex`, `vertexai` | Enterprise | Native |
| DeepSeek | `deepseek` | Cloud | Native |
| Mistral | `mistral` | Cloud | Native |
| Groq | `groqcloud` | Cloud | Native |
| Together AI | `together` | Cloud | Native |
| Fireworks AI | `fireworks` | Cloud | Native |
| OpenRouter | `openrouter` | Gateway | Native |
| Cerebras | `cerebras` | Cloud | Native |
| Moonshot (Kimi) | `moonshot`, `kimi` | Cloud | Native |
| ZhipuAI | `zai`, `zhipuai`, `zhipu` | Cloud | Native |
| HuggingFace | `huggingface`, `hf` | Cloud | Varies |
| Replicate | `replicate` | Cloud | Limited |
| Ollama | `ollama` | Local | Per-model |
| LMStudio | `lmstudio` | Local | Native |
| vLLM | `vllm` | Local | Native |
| llama.cpp | `llamacpp`, `llama-cpp`, `llama.cpp` | Local | Native |

---

## Error Handling

Victor provides a hierarchy of provider errors for granular error handling.

**Import:**
```python
from victor.providers.base import (
    ProviderError,
    ProviderNotFoundError,
    ProviderAuthError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderConnectionError,
    ProviderInvalidResponseError,
)
```

### Error Classes

```python
class ProviderError(Exception):
    """Base exception for all provider errors."""

    def __init__(
        self,
        message: str,
        provider: str,
        status_code: Optional[int] = None,
        raw_error: Optional[Exception] = None,
    ):
        self.message = message
        self.provider = provider
        self.status_code = status_code
        self.raw_error = raw_error


class ProviderNotFoundError(ProviderError):
    """Raised when a requested provider is not registered."""


class ProviderAuthError(ProviderError):
    """Raised when authentication fails (invalid API key, etc.)."""


class ProviderRateLimitError(ProviderError):
    """Raised when rate limit is exceeded (HTTP 429)."""


class ProviderTimeoutError(ProviderError):
    """Raised when a request times out."""


class ProviderConnectionError(ProviderError):
    """Raised when connection to provider fails."""


class ProviderInvalidResponseError(ProviderError):
    """Raised when provider returns invalid/unparseable response."""
```

### Circuit Breaker

The circuit breaker pattern prevents cascading failures when providers are unavailable.

```python
from victor.providers.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerRegistry,
)
from victor.core.circuit_breaker import CircuitBreakerError, CircuitState

# Get circuit breaker stats
stats = provider.get_circuit_breaker_stats()
# {
#     "name": "provider_AnthropicProvider",
#     "state": "closed",
#     "total_calls": 100,
#     "total_failures": 2,
#     "total_rejected": 0,
#     "failure_count": 0,
#     "success_count": 0,
# }

# Reset circuit breaker manually
provider.reset_circuit_breaker()
```

### Error Handling Example

```python
from victor.providers import (
    ProviderRegistry,
    ProviderError,
    ProviderAuthError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)

async def call_llm(messages, model):
    provider = ProviderRegistry.create("anthropic", api_key="...")

    try:
        response = await provider.chat(messages, model=model)
        return response.content

    except ProviderAuthError as e:
        logger.error(f"Authentication failed: {e.message}")
        raise

    except ProviderRateLimitError as e:
        logger.warning(f"Rate limited, retry after delay")
        await asyncio.sleep(60)
        return await call_llm(messages, model)  # Retry

    except ProviderTimeoutError as e:
        logger.warning(f"Request timed out: {e.message}")
        raise

    except ProviderError as e:
        logger.error(f"Provider error: {e.message}")
        raise

    finally:
        await provider.close()
```

---

## See Also

- [Tool System API Reference](./tools.md)
- [Workflow Engine API Reference](./workflows.md)
- [Configuration Guide](../getting-started/configuration.md)
