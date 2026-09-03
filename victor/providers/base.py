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

"""Base provider interface for LLM providers."""

import contextlib
import logging
import math
from dataclasses import dataclass
import ssl
import time
import threading
from abc import ABC, abstractmethod
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
)

logger = logging.getLogger(__name__)

from pydantic import BaseModel, Field

from victor.core.circuit_breaker import CircuitBreakerError
from victor.providers.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerRegistry,
)
from victor.providers.runtime_capabilities import ProviderRuntimeCapabilities

# -----------------------------------------------------------------------------
# Provider capabilities are a single surface: the ``supports_*()`` methods on
# BaseProvider. They default to False (Interface Segregation — providers only
# override what they support) and are answerable on every provider (Liskov), so
# callers query capabilities with a plain ``provider.supports_x()`` call. This
# is strictly more expressive than boolean Protocol conformance — e.g.
# ``supports_reasoning_effort(model)`` is per-model — so no parallel Protocol /
# ``is_*_provider`` helper layer is kept.
# -----------------------------------------------------------------------------


class Message(BaseModel):
    """Standard message format across all providers."""

    role: str = Field(
        ...,
        description=(
            "Message role. OpenAI Chat Completions supports developer, system, user, "
            "assistant, tool, and legacy function; other provider families may differ."
        ),
    )
    content: str = Field(..., description="Message content")
    name: Optional[str] = Field(default=None, description="Optional name for the message sender")
    tool_calls: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Tool calls requested by the assistant"
    )
    tool_call_id: Optional[str] = Field(
        default=None, description="ID of the tool call being responded to"
    )
    images: Optional[List[str]] = Field(
        default=None,
        description="Image data URIs (data:image/png;base64,...) for multimodal user messages",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Internal metadata (e.g. MessageSource). Never serialised to providers.",
        exclude=True,
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary representation."""
        return self.model_dump()


class ToolDefinition(BaseModel):
    """Standard tool definition format.

    The optional schema_level tracks which verbosity tier was used to
    generate the description and parameters. It is excluded from
    serialization (never sent to providers) — used only for internal
    cache-aware ordering.
    """

    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="What the tool does")
    parameters: Dict[str, Any] = Field(..., description="JSON Schema for tool parameters")
    schema_level: Optional[str] = Field(default=None, exclude=True)


class CompletionResponse(BaseModel):
    """Standard completion response format."""

    content: str = Field(..., description="Generated content")
    role: str = Field(default="assistant", description="Response role")
    tool_calls: Optional[List[Dict[str, Any]]] = Field(None, description="Tool calls requested")
    stop_reason: Optional[str] = Field(None, description="Why generation stopped")
    usage: Optional[Dict[str, int]] = Field(None, description="Token usage stats")
    model: Optional[str] = Field(None, description="Model used")
    raw_response: Optional[Dict[str, Any]] = Field(None, description="Raw provider response")
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Additional metadata (e.g., reasoning_content)"
    )


class StreamChunk(BaseModel):
    """Streaming response chunk."""

    content: str = Field(default="", description="Incremental content")
    tool_calls: Optional[List[Dict[str, Any]]] = None
    stop_reason: Optional[str] = None
    is_final: bool = Field(default=False, description="Is this the final chunk")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional metadata (e.g., reasoning_content)"
    )
    usage: Optional[Dict[str, int]] = Field(
        default=None,
        description="Token usage stats (typically on final chunk). Keys: prompt_tokens, completion_tokens, total_tokens, cache_creation_input_tokens, cache_read_input_tokens",
    )


# Provider error classes - re-exported from victor/core/errors for backward compatibility
# All error classes are defined in victor/core/errors.py as the single source of truth
from victor.core.errors import (
    ModelNotFoundError,
    ProviderError,
    ProviderNotFoundError,
    ProviderTimeoutError,
    ProviderRateLimitError,
    ProviderAuthError,
    ProviderConnectionError,
    ProviderInvalidResponseError,
)


@dataclass(frozen=True)
class CacheCostModel:
    """Characterized caching economics for a provider capability (FEP-0011).

    Replaces branching on a single boolean with a value object the prompt
    assembler can optimize against (pruning aggressiveness, stable/dynamic
    prefix split, prefix freeze size). See FEP-0011 for the full design.

    Phase 1 (this): additive only. ``BaseProvider.cache_cost_model()`` defaults
    to ``CacheCostModel(supported=self.supports_prompt_caching())`` so existing
    providers behave identically; providers may override to advertise real
    numbers. The assembler still reads the booleans today; consumption is a
    later phase.

    Attributes:
        supported: Whether this caching capability is available at all.
        read_discount: Fraction saved on a cached read (0.0–1.0).
        write_overhead: Cost multiplier for a cache write (>=1.0; 1.0 = none).
        ttl_seconds: Lifetime of a cache entry (0.0 = no TTL / indefinite).
        min_prefix_tokens: Minimum stable prefix to be eligible for caching.
        max_cache_tokens: Maximum cacheable tokens (0 = no published limit).
        prefix_granularity: Smallest cacheable unit ("token"|"message"|"system_block").
    """

    supported: bool = False
    read_discount: float = 0.0
    write_overhead: float = 1.0
    ttl_seconds: float = 0.0
    min_prefix_tokens: int = 0
    max_cache_tokens: int = 0
    prefix_granularity: str = "token"


class BaseProvider(ABC):
    """Abstract base class for all LLM providers."""

    _DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS = 15.0
    _HARD_RATE_LIMIT_COOLDOWN_SECONDS = 300.0
    _rate_limit_suppression_lock = threading.Lock()
    _rate_limit_suppression_by_key: Dict[
        tuple[str, Optional[str], Optional[str]],
        tuple[float, Optional[str], Any],
    ] = {}

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
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.extra_config = kwargs

        # Retry strategy (lazily initialized on first rate-limit retry)
        self._retry_strategy: Any = None
        self._rate_limit_suppressed_until_monotonic: Optional[float] = None
        self._rate_limit_suppression_error: Optional[ProviderRateLimitError] = None

        # Circuit breaker for resilience
        self._use_circuit_breaker = use_circuit_breaker
        self._circuit_breaker: Optional[CircuitBreaker] = None
        if use_circuit_breaker:
            # Stable key: state aggregates across instances of the same
            # provider class + endpoint. Never embed id(self) — it fragments
            # breaker state per instance and leaks one registry entry per
            # object ever created.
            base_url = getattr(self, "base_url", None) or "default"
            breaker_name = f"provider_{self.__class__.__name__}_{base_url}"
            self._circuit_breaker = CircuitBreakerRegistry.get_or_create(
                name=breaker_name,
                failure_threshold=circuit_breaker_failure_threshold,
                recovery_timeout=circuit_breaker_recovery_timeout,
                excluded_exceptions=(ProviderAuthError,),
            )

    @property
    def circuit_breaker(self) -> Optional[CircuitBreaker]:
        """Get the circuit breaker for this provider."""
        return self._circuit_breaker

    def supports_tools(self) -> bool:
        """Check if provider supports tool/function calling.

        Default implementation returns False. Providers that support tool calling
        should override this method to return True. This follows the Interface
        Segregation Principle - providers don't need to implement tool calling
        if they don't support it.

        Returns:
            True if provider supports tools, False otherwise (default)
        """
        return False

    def supports_streaming(self) -> bool:
        """Check if provider supports streaming responses.

        Default implementation returns False. Providers that support streaming
        should override this method to return True. This follows the Interface
        Segregation Principle - providers don't need to implement streaming
        if they don't support it.

        Returns:
            True if provider supports streaming, False otherwise (default)
        """
        return False

    def supports_vision(self) -> bool:
        """Check if provider supports multimodal (image) input.

        Default implementation returns False. Providers that accept messages
        with ``images`` populated should override this to return True. This
        follows the Interface Segregation Principle - providers don't need to
        implement vision if they don't support it - and completes the base
        capability contract so ``provider.supports_vision()`` is answerable on
        every provider (Liskov), matching supports_tools/supports_streaming.

        Returns:
            True if provider supports image input, False otherwise (default)
        """
        return False

    def supports_prompt_caching(self) -> bool:
        """Check if provider supports API-level prompt prefix caching.

        API-level prompt caching means the provider offers a **billing discount**
        (50-90%) on cached input tokens. For these providers, sending the full
        tool set (48 tools) every call is optimal because cached tokens are
        nearly free after the first call.

        This is distinct from KV prefix caching (see supports_kv_prefix_caching),
        which is a latency optimization at the inference engine level.

        Providers WITHOUT API-level caching (Ollama, LMStudio, llama.cpp, MLX,
        vLLM) should use per-turn semantic tool selection (5-12 tools) to
        minimize token overhead.

        Default implementation returns False.

        Returns:
            True if provider has API-level cached token discounts
        """
        return False

    def supports_kv_prefix_caching(self) -> bool:
        """Check if provider supports KV prefix caching for latency savings.

        KV prefix caching means the inference engine reuses computed key-value
        state when consecutive requests share the same prompt prefix. This
        reduces time-to-first-token (TTFT) but does NOT reduce cost — every
        token still incurs compute on the first call.

        When True, the framework should keep the system prompt + tool definitions
        stable across turns within a session so the KV cache can be reused.

        Both local engines (Ollama, vLLM, llama.cpp) and cloud APIs (Anthropic,
        OpenAI) support this. Default is False.

        Returns:
            True if provider benefits from stable prompt prefixes
        """
        return False

    def cache_cost_model(self) -> CacheCostModel:
        """Characterized economics for API-level prompt caching (FEP-0011).

        Default derives from :meth:`supports_prompt_caching` so this is purely
        additive and existing providers are unaffected. Override to advertise
        real numbers (discount %, TTL, prefix granularity, …) so the prompt
        assembler can optimize against them instead of a single boolean.

        Returns:
            A frozen :class:`CacheCostModel` describing this provider's
            API-level prompt-caching economics.
        """
        return CacheCostModel(supported=self.supports_prompt_caching())

    def kv_cache_cost_model(self) -> CacheCostModel:
        """Characterized economics for KV prefix caching (FEP-0011).

        Default derives from :meth:`supports_kv_prefix_caching`. Override to
        advertise real numbers (KV cache is latency-only; ``read_discount`` is
        typically 0.0 since there is no billing discount).

        Returns:
            A frozen :class:`CacheCostModel` describing this provider's
            KV-prefix-caching economics.
        """
        return CacheCostModel(supported=self.supports_kv_prefix_caching())

    def supports_reasoning_effort(self, model: Optional[str] = None) -> bool:
        """Check if a model accepts the ``reasoning_effort`` request parameter.

        ``reasoning_effort`` (``"low"`` / ``"medium"`` / ``"high"``) controls how
        much a reasoning model deliberates before answering. It is only valid for
        reasoning models on providers that expose it (e.g. OpenAI o-series /
        GPT-5); sending it to a model that doesn't support it is an API error.

        The framework consults this before forwarding a per-member
        ``reasoning_effort`` so the parameter is never sent to a model that would
        reject it. Default is False — providers/models opt in by overriding.

        Args:
            model: Model identifier to check (provider may key support on it).

        Returns:
            True if ``reasoning_effort`` may be forwarded for ``model``.
        """
        return False

    DEFAULT_CONTEXT_WINDOW: int = 8_192
    """Conservative default context window when model is unknown.

    Triggers semantic_select_capped strategy in the tool broadcaster, which
    is safe for any model. Override per-provider with a model lookup table.
    The single ``context_window`` definition below returns this as its
    unknown-model fallback.
    """

    def get_tool_output_format(self) -> Any:
        """Get preferred tool output format for this provider.

        This method enables provider-specific customization of tool output
        formatting, following the Strategy pattern. Default implementation
        returns plain JSON format (token-efficient for cloud providers).

        Providers can override to specify XML, TOON, or custom formats:
        - Cloud providers (OpenAI, xAI, etc.): Use default plain JSON
        - Local providers (Ollama, vLLM, llama.cpp): Override to XML format
        - Experimental: Override to TOON for structured data

        Returns:
            ToolOutputFormat specification (from victor.agent.format_strategies)

        Example:
            from victor.agent.format_strategies import ToolOutputFormat, XML_FORMAT

            class OllamaProvider(BaseProvider):
                def get_tool_output_format(self):
                    # Local models trained on XML format
                    return XML_FORMAT
        """
        from victor.agent.format_strategies import ToolOutputFormat

        return ToolOutputFormat(style="plain")

    def get_circuit_breaker_stats(self) -> Optional[Dict[str, Any]]:
        """Get circuit breaker statistics for monitoring."""
        if self._circuit_breaker:
            return self._circuit_breaker.get_stats()
        return None

    def _iter_exception_chain(self, error: Exception) -> List[Exception]:
        """Yield the wrapped exception chain without looping forever."""
        chain: List[Exception] = []
        seen: set[int] = set()
        current: Optional[BaseException] = error

        while isinstance(current, Exception) and id(current) not in seen:
            chain.append(current)
            seen.add(id(current))
            current = (
                getattr(current, "__cause__", None)
                or getattr(current, "__context__", None)
                or getattr(current, "cause", None)
            )

        return chain

    def _is_connection_error_like(self, error: Exception) -> bool:
        """Check whether an exception looks like a transient transport failure."""
        connection_exception_names = {
            "APIConnectionError",
            "ConnectError",
            "ConnectTimeout",
            "ReadError",
            "ReadTimeout",
            "RemoteProtocolError",
            "TransportError",
            "WriteError",
        }
        connection_tokens = (
            "bad record mac",
            "broken pipe",
            "connection aborted",
            "connection error",
            "connection refused",
            "connection reset",
            "remote protocol error",
            "server disconnected",
            "ssl",
            "tls",
        )

        for candidate in self._iter_exception_chain(error):
            if isinstance(candidate, (ConnectionError, ssl.SSLError)):
                return True

            if any(
                parent.__name__ in connection_exception_names for parent in type(candidate).__mro__
            ):
                return True

            candidate_str = str(candidate).lower()
            if any(token in candidate_str for token in connection_tokens):
                return True

        return False

    def _is_timeout_error_like(self, error: Exception) -> bool:
        """Check whether an exception looks like a timeout failure."""
        timeout_exception_names = {
            "APITimeoutError",
            "ConnectTimeout",
            "PoolTimeout",
            "ReadTimeout",
            "TimeoutException",
            "TimeoutError",
            "WriteTimeout",
        }
        timeout_tokens = ("timeout", "timed out")

        for candidate in self._iter_exception_chain(error):
            if isinstance(candidate, TimeoutError):
                return True

            if any(
                parent.__name__ in timeout_exception_names for parent in type(candidate).__mro__
            ):
                return True

            candidate_str = str(candidate).lower()
            if any(token in candidate_str for token in timeout_tokens):
                return True

        return False

    def _extract_error_status_code(self, error: Exception) -> Optional[int]:
        """Extract an HTTP-style status code from an exception when available."""
        status = (
            getattr(error, "status_code", None)
            or getattr(error, "code", None)
            or getattr(getattr(error, "response", None), "status_code", None)
        )
        return status if isinstance(status, int) else None

    @staticmethod
    def _sanitize_provider_error_message(error: Exception) -> str:
        """Return a bounded, user-facing provider error message."""
        response = getattr(error, "response", None)
        status_code = getattr(response, "status_code", None)
        response_text = getattr(response, "text", None)
        raw_message = (
            response_text if isinstance(response_text, str) and response_text else str(error)
        )
        message = raw_message.strip() or type(error).__name__
        lowered = message.lower()

        if "<html" in lowered or "<body" in lowered or "<!doctype html" in lowered:
            status = f" (HTTP {status_code})" if isinstance(status_code, int) else ""
            return (
                "Provider returned an HTML authentication page"
                f"{status}. Credentials may be expired, missing, or not accepted by the endpoint."
            )

        max_len = 800
        if len(message) > max_len:
            return f"{message[:max_len].rstrip()}..."
        return message

    def _looks_like_hard_rate_limit(self, error: Exception) -> bool:
        """Check whether a rate-limit error is a quota/billing exhaustion failure."""
        hard_limit_tokens = (
            "billing",
            "credit balance",
            "credits exhausted",
            "current quota",
            "hard limit",
            "insufficient balance",
            "insufficient credits",
            "insufficient quota",
            "payment required",
            "quota exceeded",
            "quota exhausted",
            "resource exhausted",
        )

        for candidate in self._iter_exception_chain(error):
            if self._extract_error_status_code(candidate) != 429 and not isinstance(
                candidate, ProviderRateLimitError
            ):
                continue

            text_parts = [str(candidate)]
            response = getattr(candidate, "response", None)
            response_text = getattr(response, "text", None)
            if isinstance(response_text, str) and response_text:
                text_parts.append(response_text)

            if any(token in " ".join(text_parts).lower() for token in hard_limit_tokens):
                return True

        return False

    def _rate_limit_cooldown_seconds(self, error: ProviderRateLimitError) -> float:
        """Determine how long to suppress follow-on calls after a rate-limit failure."""
        retry_after = getattr(error, "retry_after", None)
        if isinstance(retry_after, (int, float)) and retry_after > 0:
            return float(retry_after)
        if self._looks_like_hard_rate_limit(error):
            return self._HARD_RATE_LIMIT_COOLDOWN_SECONDS
        return self._DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS

    def _rate_limit_suppression_key(
        self, model: Optional[str] = None
    ) -> tuple[str, Optional[str], Optional[str]]:
        """Return the cross-instance cooldown key for this provider call."""
        return (self.name, model, self.base_url)

    def _record_rate_limit_suppression(
        self,
        error: ProviderRateLimitError,
        *,
        model: Optional[str] = None,
    ) -> None:
        """Suppress immediate follow-on calls after an exhausted rate-limit path."""
        cooldown_seconds = self._rate_limit_cooldown_seconds(error)
        suppressed_until = time.monotonic() + cooldown_seconds
        self._rate_limit_suppressed_until_monotonic = suppressed_until
        self._rate_limit_suppression_error = error
        key = self._rate_limit_suppression_key(model)
        with self._rate_limit_suppression_lock:
            self._rate_limit_suppression_by_key[key] = (
                suppressed_until,
                getattr(error, "message", str(error)),
                getattr(error, "raw_error", error),
            )
        logger.warning(
            "Provider %s entering rate-limit cooldown for %.1fs",
            self.name,
            cooldown_seconds,
        )

    def _raise_if_rate_limit_suppressed(self, *, model: Optional[str] = None) -> None:
        """Fail fast when this provider/model is in post-rate-limit cooldown."""
        key = self._rate_limit_suppression_key(model)
        with self._rate_limit_suppression_lock:
            shared_suppression = self._rate_limit_suppression_by_key.get(key)
            if shared_suppression is not None:
                suppressed_until, message, raw_error = shared_suppression
                remaining_seconds = suppressed_until - time.monotonic()
                if remaining_seconds <= 0:
                    self._rate_limit_suppression_by_key.pop(key, None)
                else:
                    raise ProviderRateLimitError(
                        message=(
                            "Provider temporarily suppressed after recent rate limit exhaustion"
                            if not message
                            else f"Provider temporarily suppressed after recent rate limit exhaustion: {message}"
                        ),
                        provider=self.name,
                        retry_after=int(math.ceil(remaining_seconds)),
                        status_code=429,
                        raw_error=raw_error,
                    )

        if self._rate_limit_suppressed_until_monotonic is None:
            return

        remaining_seconds = self._rate_limit_suppressed_until_monotonic - time.monotonic()
        if remaining_seconds <= 0:
            self._rate_limit_suppressed_until_monotonic = None
            self._rate_limit_suppression_error = None
            return

        prior_error = self._rate_limit_suppression_error
        raise ProviderRateLimitError(
            message=(
                "Provider temporarily suppressed after recent rate limit exhaustion"
                if prior_error is None
                else f"Provider temporarily suppressed after recent rate limit exhaustion: {prior_error.message}"
            ),
            provider=self.name,
            retry_after=int(math.ceil(remaining_seconds)),
            status_code=429,
            raw_error=getattr(prior_error, "raw_error", prior_error),
        )

    def classify_error(self, error: Exception) -> ProviderError:
        """Classify a raw exception into the appropriate ProviderError subtype.

        Providers can override this for provider-specific error handling.
        The base implementation uses a three-tier strategy:
        1. Pass through existing ProviderError subtypes unchanged
        2. Check HTTP status codes (if available on the exception)
        3. String-based pattern matching as final fallback

        Args:
            error: The raw exception from the provider API call.

        Returns:
            A ProviderError (or subtype) wrapping the original exception.
        """
        # Tier 0: Already classified
        if isinstance(error, ProviderError):
            return error

        if isinstance(error, CircuitBreakerError):
            return ProviderRateLimitError(
                message=f"Provider temporarily unavailable: {error}",
                provider=self.name,
                retry_after=error.retry_after,
                status_code=429,
                raw_error=error,
            )

        wrapped_error = (
            getattr(error, "last_error", None)
            or getattr(error, "raw_error", None)
            or getattr(error, "cause", None)
            or getattr(error, "__cause__", None)
        )
        if isinstance(wrapped_error, Exception) and wrapped_error is not error:
            return self.classify_error(wrapped_error)

        error_str = str(error).lower()

        # Tier 1: Check HTTP status code attributes
        # Also check .response.status_code for httpx.HTTPStatusError compatibility
        status = self._extract_error_status_code(error)
        if isinstance(status, int):
            if status == 401 or status == 403:
                safe_message = self._sanitize_provider_error_message(error)
                return ProviderAuthError(
                    message=f"Authentication failed: {safe_message}",
                    provider=self.name,
                    status_code=status,
                    raw_error=error,
                )
            if status == 429:
                return ProviderRateLimitError(
                    message=f"Rate limit exceeded: {error}",
                    provider=self.name,
                    status_code=429,
                    raw_error=error,
                )
            # A 404 whose body mentions a missing model is a user-actionable
            # configuration problem, not a transient/internal failure. Guard on
            # the message so unrelated 404s (e.g. a wrong endpoint) stay generic.
            if status == 404 and "model" in error_str and "not found" in error_str:
                return ModelNotFoundError(
                    provider=self.name,
                    status_code=404,
                    raw_error=error,
                )

        if self._is_timeout_error_like(error):
            timeout_seconds = getattr(self, "timeout", None)
            error_detail = str(error) or repr(error) or type(error).__name__
            logger.error(
                "Provider timeout error: provider=%s error_type=%s error_msg=%s",
                self.name,
                type(error).__name__,
                error_detail,
            )
            return ProviderTimeoutError(
                message=f"Request timed out: {error_detail}",
                provider=self.name,
                timeout=(timeout_seconds if isinstance(timeout_seconds, (int, float)) else None),
                raw_error=error,
            )

        if self._is_connection_error_like(error):
            # Log detailed connection error information for debugging
            error_type = type(error).__name__
            error_msg = str(error)
            logger.error(
                "Provider connection error: provider=%s error_type=%s error_msg=%s",
                self.name,
                error_type,
                error_msg,
            )
            return ProviderConnectionError(
                message=f"Connection failed: {error}",
                provider=self.name,
                raw_error=error,
            )

        # Tier 2: String-based classification (deprecated — providers should use
        # proper exception types or HTTP status codes instead)
        if any(
            t in error_str
            for t in (
                "auth",
                "unauthorized",
                "invalid key",
                "invalid api",
                "api_key",
                "401",
            )
        ):
            safe_message = self._sanitize_provider_error_message(error)
            logger.warning(
                "String-based error classification triggered for auth error "
                "from %s. Provider should use proper exception types.",
                self.name,
            )
            return ProviderAuthError(
                message=f"Authentication failed: {safe_message}",
                provider=self.name,
                raw_error=error,
            )
        if any(t in error_str for t in ("rate limit", "429", "too many requests")):
            logger.warning(
                "String-based error classification triggered for rate limit "
                "from %s. Provider should use proper exception types.",
                self.name,
            )
            return ProviderRateLimitError(
                message=f"Rate limit exceeded: {error}",
                provider=self.name,
                status_code=429,
                raw_error=error,
            )
        if any(t in error_str for t in ("timeout", "timed out")):
            logger.warning(
                "String-based error classification triggered for timeout "
                "from %s. Provider should use proper exception types.",
                self.name,
            )
            return ProviderTimeoutError(
                message=f"Request timed out: {error}",
                provider=self.name,
                raw_error=error,
            )

        # Default: generic ProviderError
        return ProviderError(
            message=f"{self.name} API error: {error}",
            provider=self.name,
            status_code=status if isinstance(status, int) else None,
            raw_error=error,
        )

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""
        pass

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
            model: Model identifier
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            tools: Available tools for the model to use
            **kwargs: Additional provider-specific parameters

        Returns:
            CompletionResponse with generated content

        Raises:
            ProviderError: If the request fails
        """
        pass

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
        # Abstract async generator - yield needed for mypy to recognize as generator
        if False:
            yield StreamChunk()
        raise NotImplementedError

    async def discover_capabilities(self, model: str) -> ProviderRuntimeCapabilities:
        """Discover capabilities for the given model.

        Default implementation falls back to configured limits and
        provider-declared support flags. Providers should override
        with real HTTP-based discovery when available.
        """
        from victor.config.config_loaders import get_provider_limits

        limits = get_provider_limits(self.name, model)
        return ProviderRuntimeCapabilities(
            provider=self.name,
            model=model,
            context_window=limits.context_window,
            supports_tools=self.supports_tools(),
            supports_streaming=self.supports_streaming(),
            source="config",
        )

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
        with different naming conventions across SDKs (OpenAI uses stream,
        Anthropic uses stream_chat, etc.).

        Args:
            messages: List of conversation messages
            model: Model identifier
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            tools: Available tools for the model to use
            **kwargs: Additional provider-specific parameters

        Yields:
            StreamChunk objects with incremental content
        """
        # aclosing finalizes the underlying stream() generator on-task on an early break,
        # avoiding GC-driven off-task cleanup of the httpx byte stream.
        inner = self.stream(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            **kwargs,
        )
        async with contextlib.aclosing(inner) as inner_stream:
            async for chunk in inner_stream:
                yield chunk

    async def count_tokens(self, text: str) -> int:
        """Estimate token count for given text.

        Uses the fast native token counter when available and falls back
        to word-based estimation.

        Args:
            text: Text to count tokens for

        Returns:
            Estimated token count
        """
        from victor.processing.native.tokenizer import count_tokens_fast

        return count_tokens_fast(text)

    def context_window(self, model: str) -> int:
        """Get context window size for a given model.

        Provides context window limits for common models to enable
        context-budgeted tool selection strategies. Returns safe default
        for unknown models.

        Args:
            model: Model identifier (e.g., "claude-sonnet-4-20250514", "qwen2.5-coder:7b")

        Returns:
            Context window in tokens. Returns safe default (8192) for unknown models.

        Examples:
            >>> provider = AnthropicProvider(api_key="...")
            >>> cw = provider.context_window("claude-sonnet-4-20250514")
            >>> assert cw == 200000
        """
        # Config-driven override: consult provider_context_limits.yaml first.
        # This makes model registration a YAML edit (not a code change): add a
        # pattern under `models:` or a provider block under `providers:` in
        # victor/config/provider_context_limits.yaml and it takes effect here.
        # We fall back to the hardcoded table below only if the YAML has no
        # explicit entry for this provider/model (legacy behaviour).
        try:
            from victor.config.config_loaders import (
                get_provider_limits,
                _load_yaml_cached,
                PROVIDER_LIMITS_FILE,
            )
            import fnmatch as _fnmatch

            data = _load_yaml_cached(PROVIDER_LIMITS_FILE, "provider_limits")
            providers = (data or {}).get("providers", {})
            models = (data or {}).get("models", {})

            # Only treat the YAML as authoritative when there is an EXPLICIT
            # match (provider block or model fnmatch pattern). The loader's
            # ProviderLimits default of 128000 must NOT silently override a
            # model that is correctly registered in the hardcoded table below.
            has_entry = self.name in providers or any(_fnmatch.fnmatch(model, p) for p in models)
            if has_entry:
                limits = get_provider_limits(self.name, model)
                if limits.context_window:
                    return limits.context_window
        except Exception:
            # Never let a config-load failure break the hot path; fall through
            # to the hardcoded table below.
            pass

        # Known models lookup table
        # Source: Model documentation as of 2025-04
        CONTEXT_WINDOWS = {
            # Anthropic
            "claude-sonnet-4-20250514": 200000,
            "claude-3.5-sonnet-20240620": 200000,
            "claude-3-5-sonnet-20241022": 200000,
            "claude-3-opus-20240229": 200000,
            "claude-3-haiku-20240307": 200000,
            # OpenAI
            "gpt-4o": 128000,
            "gpt-4o-mini": 128000,
            "gpt-4-turbo": 128000,
            "gpt-4": 8192,
            "gpt-3.5-turbo": 16385,
            # Google Gemini
            "gemini-2.0-flash-exp": 1000000,
            "gemini-1.5-pro": 280000,
            "gemini-1.5-flash": 280000,
            "gemini-pro": 280000,
            # DeepSeek
            "deepseek-coder": 128000,
            "deepseek-chat": 128000,
            "deepseek-coder-v2": 128000,
            # Meta Llama (via various providers)
            "llama-3.1-405b-instruct": 128000,
            "llama-3.1-70b-instruct": 128000,
            "llama-3.1-8b-instruct": 128000,
            "llama-3-1-405b-instruct": 128000,
            "llama-3-1-70b-instruct": 128000,
            # Qwen (via Ollama, vLLM, etc.)
            "qwen2.5-coder:7b": 32768,
            "qwen2.5-coder:14b": 32768,
            "qwen2.5-72b-instruct": 128000,
            "qwen2.5-7b-instruct": 128000,
            "qwen2.5-14b": 128000,
            # Edge models (small models for fast micro-decisions)
            "qwen3.5:2b": 8192,
            "qwen2.5:0.5b": 8192,
            "qwen2.5:1.5b": 8192,
            "qwen2.5:3b": 8192,
            "phi-2:2.7b": 8192,
            "phi-2.7b": 8192,
            "phi:2.7b": 8192,
            "gemma:2b": 8192,
            "gemma2:2b": 8192,
            "tinyllama:1.1b": 8192,
            "tinyllama:1.1b-chat": 8192,
            # CodeLlama
            "codellama:13b": 16384,
            "codellama:34b": 16384,
            "codellama:7b": 16384,
            # Mistral
            "mistral:7b": 32768,
            "mixtral:8x7b": 32768,
            "mixtral:8x22b": 65536,
            # Common Ollama models
            "phi:2.7b-chat": 128000,
            "gemma:7b": 8192,
        }

        # Try direct lookup. (Pattern/wildcard matching lives in the YAML
        # provider-limits path above; the hardcoded table has no "*" keys.)
        if model in CONTEXT_WINDOWS:
            return CONTEXT_WINDOWS[model]

        # Safe default for unknown models
        # Use DEFAULT_CONTEXT_WINDOW (8K) as conservative default that fits
        # even smallest models
        logger.warning(
            f"Unknown model {model}, using default context window of "
            f"{self.DEFAULT_CONTEXT_WINDOW} tokens. "
            f"Consider adding context_window mapping for this model."
        )
        return self.DEFAULT_CONTEXT_WINDOW

    @abstractmethod
    async def close(self) -> None:
        """Close any open connections or resources."""
        pass

    async def __aenter__(self) -> "BaseProvider":
        """Enable async context manager usage."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Close provider on context exit."""
        await self.close()

    def is_circuit_open(self) -> bool:
        """Check if the circuit breaker is open (failing fast).

        Returns:
            True if circuit is open and requests will be rejected
        """
        if self._circuit_breaker:
            return self._circuit_breaker.is_open
        return False

    async def _execute_with_circuit_breaker(
        self,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute a function with circuit breaker and retry-on-rate-limit.

        When ``max_retries > 0``, rate-limit errors (429) are automatically
        retried with exponential backoff and jitter, respecting ``Retry-After``
        headers. Set ``max_retries=0`` to raise immediately (old behavior).

        Args:
            func: Async function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Result from func

        Raises:
            CircuitBreakerError: If circuit is open
            ProviderRateLimitError: If rate-limit retries are exhausted
            Exception: If func raises a non-retryable error
        """
        call_model = kwargs.get("model")
        model_name = call_model if isinstance(call_model, str) else None
        self._raise_if_rate_limit_suppressed(model=model_name)

        async def _call() -> Any:
            if self._circuit_breaker:
                return await self._circuit_breaker.execute(func, *args, **kwargs)
            return await func(*args, **kwargs)

        try:
            if self.max_retries <= 0:
                return await _call()

            # Use ProviderRetryStrategy for automatic retry on 429/transient errors
            if self._retry_strategy is None:
                from victor.providers.resilience import (
                    ProviderRetryConfig,
                    ProviderRetryStrategy,
                )

                self._retry_strategy = ProviderRetryStrategy(
                    ProviderRetryConfig(
                        max_retries=self.max_retries,
                        # BaseProvider wraps the call in a circuit breaker before retry.
                        # Open circuits should fail fast instead of sleeping through
                        # the recovery timeout and re-entering as half-open.
                        retryable_exceptions=(ConnectionError, TimeoutError),
                    )
                )

            return await self._retry_strategy.execute(_call)
        except CircuitBreakerError:
            raise
        except Exception as error:
            classified = self.classify_error(error)
            if isinstance(classified, ProviderRateLimitError):
                self._record_rate_limit_suppression(classified, model=model_name)
                raise classified from error
            raise

    def reset_circuit_breaker(self) -> None:
        """Manually reset the circuit breaker to closed state."""
        if self._circuit_breaker:
            self._circuit_breaker.reset()
