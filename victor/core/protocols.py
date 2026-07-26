# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
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

"""Minimal protocol interface for breaking circular imports.

This module provides a small OrchestratorProtocol (7 methods) specifically
to decouple evaluation/adapter modules from the main orchestrator without
creating circular import chains. It is intentionally minimal.

For the full Agent public-API contract (6 composite protocols), use
``victor.framework.protocols``. Domain-specific protocols live in
``victor.protocols.*``. The three layers serve distinct purposes and
are not duplicates.

Design Pattern: Dependency Inversion Principle (SOLID)
    - High-level modules should not depend on low-level modules
    - Both should depend on abstractions (Protocols)

Circular Import Chains Addressed:
1. orchestrator ↔ evaluation ↔ agent_adapter
2. task_analyzer ↔ embeddings.task_classifier
3. runtime_intelligence_pipeline ↔ orchestrator
4. tool_calling/registry ↔ adapters

Usage Example:
    # Instead of importing concrete class:
    # from victor.agent.orchestrator import AgentOrchestrator  # Circular!

    # Import the protocol:
    from victor.core.protocols import OrchestratorProtocol

    class EvaluationHarness:
        def __init__(self, orchestrator: OrchestratorProtocol):
            self.orchestrator = orchestrator
"""

from __future__ import annotations

from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    TYPE_CHECKING,
    TypeVar,
    runtime_checkable,
)

if TYPE_CHECKING:
    from victor.providers.base import CompletionResponse, Message, StreamChunk


# Type variables for generic protocols
T = TypeVar("T")
TResult = TypeVar("TResult", covariant=True)


@runtime_checkable
class OrchestratorProtocol(Protocol):
    """Protocol for agent orchestrator functionality.

    This allows evaluation, agent_adapter, and other modules to depend on
    the orchestrator interface without importing the concrete implementation.

    Breaks chain: orchestrator → code_correction_middleware → evaluation → agent_adapter → orchestrator

    Note: Method names match AgentOrchestrator's actual interface for structural typing.
    """

    @property
    def model(self) -> str:
        """Current model identifier."""
        ...

    @property
    def provider_name(self) -> str:
        """Current provider name."""
        ...

    @property
    def tool_budget(self) -> int:
        """Maximum allowed tool calls for session."""
        ...

    @property
    def tool_calls_used(self) -> int:
        """Number of tool calls used in current session."""
        ...

    @property
    def messages(self) -> List[Any]:
        """Current conversation message history."""
        ...

    async def chat(self, user_message: str) -> Any:
        """Process a user message and return completion response.

        Args:
            user_message: The user's input message

        Returns:
            CompletionResponse with the model's reply
        """
        ...

    async def stream_chat(self, user_message: str) -> AsyncIterator[Any]:
        """Process a user message and yield streaming response chunks.

        Args:
            user_message: The user's input message

        Yields:
            StreamChunk objects with response content
        """
        ...

    def reset_conversation(self) -> None:
        """Reset the conversation history."""
        ...


@runtime_checkable
class TaskClassificationResultProtocol(Protocol):
    """Protocol for task classification result objects."""

    @property
    def task_type(self) -> Any:
        """The classified task type."""
        ...

    @property
    def confidence(self) -> float:
        """Classification confidence score 0-1."""
        ...

    @property
    def has_file_context(self) -> bool:
        """Whether the prompt mentions specific files."""
        ...

    @property
    def top_matches(self) -> List[Any]:
        """Top matching phrases with scores."""
        ...


@runtime_checkable
class TaskClassifierProtocol(Protocol):
    """Protocol for task type classification.

    Breaks chain: task_analyzer → embeddings.task_classifier → agent modules

    Note: Method signatures match TaskTypeClassifier's actual interface.
    """

    def classify(self, prompt: str) -> TaskClassificationResultProtocol:
        """Classify a prompt into a task type.

        Args:
            prompt: User prompt to classify

        Returns:
            TaskClassificationResultProtocol with task_type, confidence, etc.
        """
        ...

    def classify_sync(self, prompt: str) -> TaskClassificationResultProtocol:
        """Synchronous classification for use in sync contexts.

        Args:
            prompt: User prompt to classify

        Returns:
            TaskClassificationResultProtocol with task_type, confidence, etc.
        """
        ...


@runtime_checkable
class IntentClassificationResultProtocol(Protocol):
    """Protocol for intent classification result objects."""

    @property
    def intent_type(self) -> Any:
        """The classified intent type."""
        ...

    @property
    def confidence(self) -> float:
        """Classification confidence score 0-1."""
        ...

    @property
    def needs_continuation(self) -> bool:
        """Whether the prompt suggests continuation."""
        ...


@runtime_checkable
class IntentClassifierProtocol(Protocol):
    """Protocol for intent/continuation classification.

    Used to determine if user wants to continue or start a new task.

    Note: Method signatures match IntentClassifier's actual interface.
    """

    def classify(self, prompt: str) -> IntentClassificationResultProtocol:
        """Classify the intent of a prompt.

        Args:
            prompt: User prompt to classify

        Returns:
            IntentClassificationResultProtocol with intent_type, confidence, needs_continuation
        """
        ...

    def is_continuation(self, query: str) -> bool:
        """Check if query is a continuation of previous conversation.

        Args:
            query: User query

        Returns:
            True if this continues the previous topic
        """
        ...


@runtime_checkable
class EmbeddingServiceProtocol(Protocol):
    """Protocol for embedding generation service.

    Allows components to use embeddings without importing heavy dependencies.

    Raises:
        RuntimeError: If the service has been shut down.
        ImportError: If the underlying model library is not installed.
        OSError: If model files cannot be loaded from disk.
    """

    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        ...

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        ...

    def embed_text_sync(self, text: str) -> List[float]:
        """Synchronous embedding generation.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        ...


@runtime_checkable
class ToolCallingAdapterProtocol(Protocol):
    """Protocol for provider-specific tool calling adapters.

    Breaks chain: tool_calling/registry → adapters → base → providers
    """

    @property
    def provider_name(self) -> str:
        """Name of the provider this adapter handles."""
        ...

    def convert_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert tools to provider-specific format.

        Args:
            tools: Tool definitions in standard format

        Returns:
            Tools in provider-specific format
        """
        ...

    def parse_tool_calls(self, response: Any) -> List[Dict[str, Any]]:
        """Parse tool calls from provider response.

        Args:
            response: Provider's raw response

        Returns:
            List of parsed tool calls with name and arguments
        """
        ...

    def normalize_arguments(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Normalize tool arguments to expected format.

        Args:
            tool_name: Name of the tool
            arguments: Raw arguments from provider

        Returns:
            Normalized arguments with defaults filled in
        """
        ...


@runtime_checkable
class RuntimeIntelligencePipelineProtocol(Protocol):
    """Protocol for runtime-intelligence pipeline integration.

    Breaks chain:
        runtime_intelligence_pipeline → orchestrator → runtime_intelligence_pipeline
    """

    async def prepare_request(
        self,
        task: str,
        task_type: str,
        current_mode: str,
    ) -> Dict[str, Any]:
        """Prepare a request with intelligent optimizations.

        Args:
            task: The task description
            task_type: Classified task type
            current_mode: Current agent mode

        Returns:
            Request context with optimized parameters
        """
        ...

    async def validate_response(
        self,
        response: str,
        query: str,
        tool_calls: int,
        success: bool,
        task_type: str,
    ) -> Dict[str, Any]:
        """Validate a response for quality and grounding.

        Args:
            response: The model's response
            query: Original user query
            tool_calls: Number of tool calls made
            success: Whether the response was successful
            task_type: The task type

        Returns:
            Validation result with scores
        """
        ...

    def should_continue(self) -> tuple[bool, str]:
        """Check if processing should continue.

        Returns:
            Tuple of (should_continue, reason)
        """
        ...


IntelligentPipelineProtocol = RuntimeIntelligencePipelineProtocol


@runtime_checkable
class ProviderProtocol(Protocol):
    """Protocol for LLM providers.

    Minimal interface that all providers must implement.

    Raises:
        ProviderAuthError: If authentication credentials are invalid or expired.
        ProviderRateLimitError: If request rate or token limits are exceeded.
        ProviderTimeoutError: If the request exceeds the configured timeout.
        ProviderError: For all other provider-related failures.
    """

    @property
    def name(self) -> str:
        """Provider name."""
        ...

    def supports_tools(self) -> bool:
        """Check if provider supports tool calling."""
        ...

    async def chat(
        self,
        messages: List[Any],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Any:
        """Send a chat completion request.

        Args:
            messages: Conversation messages
            tools: Optional tool definitions
            **kwargs: Provider-specific options

        Returns:
            Completion response

        Raises:
            ProviderAuthError: If authentication fails (401/403).
            ProviderRateLimitError: If rate limited (429).
            ProviderTimeoutError: If request times out.
            ProviderError: For other API errors.
        """
        ...

    async def stream_chat(
        self,
        messages: List[Any],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """Stream a chat completion request.

        Args:
            messages: Conversation messages
            tools: Optional tool definitions
            **kwargs: Provider-specific options

        Yields:
            Stream chunks

        Raises:
            ProviderAuthError: If authentication fails (401/403).
            ProviderRateLimitError: If rate limited (429).
            ProviderTimeoutError: If request times out.
            ProviderError: For other API errors.
        """
        ...


@runtime_checkable
class CacheProtocol(Protocol):
    """Protocol for cache implementations.

    Allows components to use caching without importing specific implementations.

    All methods are safe to call concurrently. Implementations must be
    thread-safe. Methods never raise on missing keys (return None/False).

    Raises:
        OSError: If a disk-backed cache encounters I/O errors.
        MemoryError: If an in-memory cache exceeds its memory limit.
    """

    def get(self, key: str) -> Optional[Any]:
        """Get a value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        ...

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set a value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional time-to-live in seconds
        """
        ...

    def delete(self, key: str) -> bool:
        """Delete a value from cache.

        Args:
            key: Cache key

        Returns:
            True if key was deleted
        """
        ...

    def clear(self) -> None:
        """Clear all cached values."""
        ...


@runtime_checkable
class MetricsCollectorProtocol(Protocol):
    """Structural protocol for metrics collectors (FEP-0014).

    Captures the canonical "record a metric / read a snapshot" surface so callers can
    depend on *a* metrics collector abstractly instead of importing one of the five
    concrete ``MetricsCollector`` implementations (``observability``,
    ``integrations/api/event_bridge``, ``agent``, ``experiments/ab_testing``,
    ``framework/observability``).

    Derivation note (Phase 1 finding): reading all five implementations showed their
    *current* public surfaces are entirely disjoint — they share no public method
    (only ``__init__``). The FEP's expected common surface (``record_metric`` /
    ``get_snapshot``) is the ratified canonical target, not the present-day
    intersection: only ``framework/observability`` exposes ``get_snapshot`` today
    (its record method is named ``record``), and ``record_metric`` currently lives on
    other metrics services (``core.container.MetricsServiceProtocol`` implementers,
    ``NullMetricsService``, the RL coordinator). Phase 2 therefore *adapts/renames*
    the five collectors onto this protocol; none is modified in Phase 1.

    Because this is ``runtime_checkable``, ``isinstance`` verifies method *presence*
    by name only (not signatures).
    """

    def record_metric(self, name: str, value: float, **tags: str) -> None:
        """Record a single metric observation.

        Args:
            name: Metric name.
            value: Numeric value to record.
            **tags: Optional string-valued dimensional tags.
        """
        ...

    def get_snapshot(self) -> Mapping[str, Any]:
        """Return a point-in-time snapshot of collected metrics.

        Returns:
            A read-only mapping of the collector's current metric state.
        """
        ...


# Factory type for lazy service creation
ServiceFactory = Callable[[], T]


__all__ = [
    # Core Protocols
    "OrchestratorProtocol",
    "TaskClassifierProtocol",
    "IntentClassifierProtocol",
    "EmbeddingServiceProtocol",
    "ToolCallingAdapterProtocol",
    "RuntimeIntelligencePipelineProtocol",
    "IntelligentPipelineProtocol",
    "ProviderProtocol",
    "CacheProtocol",
    "MetricsCollectorProtocol",
    # Result Protocols (for classifier return types)
    "TaskClassificationResultProtocol",
    "IntentClassificationResultProtocol",
    # Types
    "ServiceFactory",
]
