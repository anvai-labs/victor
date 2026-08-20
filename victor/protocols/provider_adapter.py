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

"""Provider adapter protocol for normalizing provider-specific behaviors.

This module defines the interface for adapting provider-specific responses,
continuation detection, and quality thresholds. Implements Interface Segregation
and Dependency Inversion principles for clean provider abstraction.

Design Patterns:
- Strategy Pattern: Different providers have different adaptation strategies
- Protocol (Interface Segregation): Small, focused interfaces
- Dependency Inversion: Core code depends on abstractions, not concrete providers

Usage:
    adapter = get_provider_adapter("deepseek")

    # Check provider capabilities
    if adapter.capabilities.supports_thinking_tags:
        thinking, content = adapter.extract_thinking_content(response)

    # Detect continuation needs
    if adapter.detect_continuation_needed(response):
        # Request continuation
        pass

    # Get quality threshold for this provider
    threshold = adapter.capabilities.quality_threshold
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Protocol,
    Tuple,
    List,
    Optional,
    Any,
    runtime_checkable,
    TYPE_CHECKING,
)
from enum import Enum

if TYPE_CHECKING:
    from victor.agent.tool_calling.base import ToolCall


class ToolCallFormat(Enum):
    """Tool call format variants across providers."""

    OPENAI = "openai"  # Standard OpenAI format
    ANTHROPIC = "anthropic"  # Anthropic's content blocks
    NATIVE = "native"  # Provider's native format
    FALLBACK = "fallback"  # Text-based parsing fallback


@dataclass
class ProviderCapabilities:
    """Provider-specific capabilities and thresholds.

    Attributes:
        quality_threshold: Minimum quality score for acceptable responses (0.0-1.0)
        supports_thinking_tags: Whether provider uses <think>...</think> tags
        thinking_tag_format: Format of thinking tags (e.g., "<think>...</think>")
        continuation_markers: Strings indicating incomplete responses
        max_continuation_attempts: Maximum retries for continuations
        tool_call_format: Format for tool call parsing
        output_deduplication: Whether to deduplicate repeated content blocks
        streaming_chunk_size: Preferred chunk size for streaming
        supports_parallel_tools: Whether provider supports parallel tool calls
        grounding_required: Whether responses must be grounded in verifiable code
        grounding_strictness: Strictness of grounding verification (0.0-1.0)
        continuation_patience: Prompts without tools before flagging stuck loop
        thinking_tokens_budget: Token budget for reasoning before requiring action
        requires_thinking_time: Whether provider needs patience for reasoning
    """

    quality_threshold: float = 0.80
    supports_thinking_tags: bool = False
    thinking_tag_format: str = ""
    continuation_markers: List[str] = field(default_factory=list)
    max_continuation_attempts: int = 5
    tool_call_format: ToolCallFormat = ToolCallFormat.OPENAI
    output_deduplication: bool = False
    streaming_chunk_size: int = 1024
    supports_parallel_tools: bool = True
    grounding_required: bool = True
    grounding_strictness: float = 0.8
    continuation_patience: int = 3
    thinking_tokens_budget: int = 500
    requires_thinking_time: bool = False


# The canonical ToolCall is imported inside the normalize_tool_calls bodies:
# a module-scope import of victor.agent closes the cycle victor.protocols ->
# victor.agent -> runtime_intelligence_pipeline -> victor.protocols.provider_adapter
# and breaks any direct `import victor.protocols` entry.


@dataclass
class ContinuationContext:
    """Context for managing response continuations.

    Attributes:
        attempt: Current continuation attempt number
        reason: Reason continuation was triggered
        partial_response: Accumulated response so far
        tool_calls_made: Number of tool calls in this continuation chain
    """

    attempt: int = 0
    reason: str = ""
    partial_response: str = ""
    tool_calls_made: int = 0


@runtime_checkable
class IProviderAdapter(Protocol):
    """Interface for provider-specific behavior adaptation.

    Implementations should handle:
    1. Response normalization (thinking tags, continuations)
    2. Tool call format conversion
    3. Quality threshold adjustments
    4. Error handling and retry logic

    Example implementation:
        class DeepSeekAdapter:
            @property
            def name(self) -> str:
                return "deepseek"

            @property
            def capabilities(self) -> ProviderCapabilities:
                return ProviderCapabilities(
                    quality_threshold=0.70,
                    supports_thinking_tags=True,
                    thinking_tag_format="<think>...</think>",
                    continuation_markers=["💭 Thinking...", "<think>"],
                )
    """

    @property
    def name(self) -> str:
        """Return the provider name."""
        ...

    @property
    def capabilities(self) -> ProviderCapabilities:
        """Return provider capabilities configuration."""
        ...

    def detect_continuation_needed(self, response: str) -> bool:
        """Detect if response indicates continuation is needed.

        Args:
            response: The LLM response text

        Returns:
            True if the response appears incomplete and needs continuation
        """
        ...

    def extract_thinking_content(self, response: str) -> Tuple[str, str]:
        """Extract thinking tags and content separately.

        Args:
            response: The LLM response text

        Returns:
            Tuple of (thinking_content, main_content)
            If no thinking tags, returns ("", response)
        """
        ...

    def normalize_tool_calls(self, raw_calls: List[Any]) -> List[ToolCall]:
        """Normalize tool calls to standard format.

        Args:
            raw_calls: Provider-specific tool call data

        Returns:
            List of normalized ToolCall objects
        """
        ...

    def should_retry(self, error: Exception) -> Tuple[bool, float]:
        """Determine if error is retryable and backoff time.

        Args:
            error: The exception that occurred

        Returns:
            Tuple of (is_retryable, backoff_seconds)
        """
        ...


class BaseProviderAdapter:
    """Base implementation with sensible defaults.

    Subclasses should override methods for provider-specific behavior.
    """

    @property
    def name(self) -> str:
        """Return the provider name."""
        return "base"

    @property
    def capabilities(self) -> ProviderCapabilities:
        """Return default provider capabilities."""
        return ProviderCapabilities()

    def detect_continuation_needed(self, response: str) -> bool:
        """Default continuation detection.

        Checks for common incomplete response patterns.
        """
        if not response or not response.strip():
            return True

        # Check for configured continuation markers
        for marker in self.capabilities.continuation_markers:
            if response.strip().endswith(marker):
                return True

        return False

    def extract_thinking_content(self, response: str) -> Tuple[str, str]:
        """Default thinking content extraction.

        Returns the full response as main content if no thinking tags.
        """
        if not self.capabilities.supports_thinking_tags:
            return ("", response)

        # Default implementation - subclasses should override
        return ("", response)

    def normalize_tool_calls(self, raw_calls: List[Any]) -> List[ToolCall]:
        """Default tool call normalization for OpenAI format.

        Args:
            raw_calls: Raw tool calls from provider

        Returns:
            Normalized ToolCall list
        """
        from victor.agent.tool_calling.base import ToolCall as _ToolCall

        normalized = []
        for i, call in enumerate(raw_calls):
            if isinstance(call, dict):
                # OpenAI format
                func = call.get("function", {})
                normalized.append(
                    _ToolCall(
                        id=call.get("id", f"call_{i}"),
                        name=func.get("name", ""),
                        arguments=func.get("arguments", {}),
                        raw=call,
                    )
                )
            elif hasattr(call, "function"):
                # OpenAI object format
                normalized.append(
                    _ToolCall(
                        id=getattr(call, "id", f"call_{i}"),
                        name=call.function.name,
                        arguments=call.function.arguments,
                        raw=call,
                    )
                )
        return normalized

    def should_retry(self, error: Exception) -> Tuple[bool, float]:
        """Default retry logic with exponential backoff.

        Returns (True, backoff_seconds) for retryable errors.
        """
        error_str = str(error).lower()

        # Rate limiting
        if "rate" in error_str and "limit" in error_str:
            return (True, 60.0)

        # Temporary server errors
        if "503" in error_str or "502" in error_str:
            return (True, 5.0)

        # Timeout errors
        if "timeout" in error_str:
            return (True, 2.0)

        # Default: not retryable
        return (False, 0.0)


class DeepSeekAdapter(BaseProviderAdapter):
    """Adapter for DeepSeek provider.

    Handles:
    - <think>...</think> tag extraction
    - Provider-specific continuation markers
    - Lower quality threshold for exploratory responses
    """

    @property
    def name(self) -> str:
        return "deepseek"

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            quality_threshold=0.60,  # Lower threshold for analytical responses
            supports_thinking_tags=True,
            thinking_tag_format="<think>...</think>",
            continuation_markers=[
                "💭 Thinking...",
                "<think>",
                "Let me ",
                "I'll ",
            ],
            max_continuation_attempts=3,  # Fewer attempts to prevent loops
            tool_call_format=ToolCallFormat.OPENAI,
            output_deduplication=False,
            grounding_required=False,  # Allow suggestions and architectural analysis
            grounding_strictness=0.6,  # More lenient
            continuation_patience=5,  # More patient - thinking-heavy model
            thinking_tokens_budget=1000,  # Large budget for reasoning
            requires_thinking_time=True,  # Needs time to think
        )

    def detect_continuation_needed(self, response: str) -> bool:
        """DeepSeek-specific continuation detection."""
        if not response or not response.strip():
            return True

        response_stripped = response.strip()

        # Check for thinking tag that wasn't closed
        if "<think>" in response and "</think>" not in response:
            return True

        # Check standard markers
        for marker in self.capabilities.continuation_markers:
            if response_stripped.endswith(marker):
                return True

        return False

    def extract_thinking_content(self, response: str) -> Tuple[str, str]:
        """Extract <think>...</think> tags from DeepSeek responses."""
        import re

        think_pattern = r"<think>(.*?)</think>"
        matches = re.findall(think_pattern, response, re.DOTALL)

        thinking = "\n".join(matches) if matches else ""
        content = re.sub(think_pattern, "", response, flags=re.DOTALL).strip()

        return (thinking, content)


class GrokAdapter(BaseProviderAdapter):
    """Adapter for xAI Grok provider.

    Handles:
    - Output deduplication for repeated content blocks
    - Higher quality threshold
    - Standard tool call format
    """

    @property
    def name(self) -> str:
        return "xai"

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            quality_threshold=0.80,
            supports_thinking_tags=False,
            continuation_markers=[],  # Grok handles continuations internally
            max_continuation_attempts=5,
            tool_call_format=ToolCallFormat.OPENAI,
            output_deduplication=True,  # Grok sometimes repeats content
            grounding_required=True,  # Verify citations
            grounding_strictness=0.82,  # High strictness
            continuation_patience=3,  # Standard patience
            thinking_tokens_budget=400,  # Standard budget
            requires_thinking_time=False,  # Fast responses
        )

    def detect_continuation_needed(self, response: str) -> bool:
        """Grok rarely needs explicit continuation."""
        # Only continue if truly empty
        return not response or not response.strip()


class OpenAIAdapter(BaseProviderAdapter):
    """Adapter for OpenAI providers (GPT-4, etc)."""

    @property
    def name(self) -> str:
        return "openai"

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            quality_threshold=0.80,
            supports_thinking_tags=False,
            max_continuation_attempts=5,
            tool_call_format=ToolCallFormat.OPENAI,
            supports_parallel_tools=True,
            grounding_required=True,  # Verify citations
            grounding_strictness=0.85,  # High strictness
            continuation_patience=3,  # Standard patience
            thinking_tokens_budget=400,  # Standard budget
            requires_thinking_time=False,  # Fast responses
        )


class AnthropicAdapter(BaseProviderAdapter):
    """Adapter for Anthropic Claude providers."""

    @property
    def name(self) -> str:
        return "anthropic"

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            quality_threshold=0.85,  # Higher threshold for Claude
            supports_thinking_tags=False,
            max_continuation_attempts=5,
            tool_call_format=ToolCallFormat.ANTHROPIC,
            supports_parallel_tools=True,
            grounding_required=True,  # Strict grounding verification
            grounding_strictness=0.9,  # Very strict
            continuation_patience=3,  # Standard patience
            thinking_tokens_budget=300,  # Action-focused
            requires_thinking_time=False,  # Responds quickly
        )

    def normalize_tool_calls(self, raw_calls: List[Any]) -> List[ToolCall]:
        """Normalize Anthropic's content block format."""
        from victor.agent.tool_calling.base import ToolCall as _ToolCall

        normalized = []
        for i, call in enumerate(raw_calls):
            if isinstance(call, dict) and call.get("type") == "tool_use":
                normalized.append(
                    _ToolCall(
                        id=call.get("id", f"call_{i}"),
                        name=call.get("name", ""),
                        arguments=call.get("input", {}),
                        raw=call,
                    )
                )
        return normalized


class OllamaAdapter(BaseProviderAdapter):
    """Adapter for Ollama local models."""

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            quality_threshold=0.55,  # Lower for local models
            supports_thinking_tags=False,
            continuation_markers=["..."],
            max_continuation_attempts=3,
            tool_call_format=ToolCallFormat.FALLBACK,  # Text-based parsing
            supports_parallel_tools=False,  # Most Ollama models single-tool
            grounding_required=False,  # Local models vary widely
            grounding_strictness=0.6,  # Lenient
            continuation_patience=6,  # More patient - local models slower
            thinking_tokens_budget=800,  # Generous budget
            requires_thinking_time=True,  # Needs time to process
        )


class GoogleAdapter(BaseProviderAdapter):
    """Adapter for Google Gemini models.

    Handles:
    - 1M+ context window
    - Multimodal capabilities
    - Function calling with FunctionDeclaration format
    """

    @property
    def name(self) -> str:
        return "google"

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            quality_threshold=0.80,
            supports_thinking_tags=False,
            max_continuation_attempts=5,
            tool_call_format=ToolCallFormat.NATIVE,  # Google's native format
            supports_parallel_tools=True,
            streaming_chunk_size=2048,  # Larger chunks for Gemini
            grounding_required=True,  # Verify citations
            grounding_strictness=0.75,  # Moderate strictness
            continuation_patience=4,  # Moderate patience
            thinking_tokens_budget=500,  # Standard budget
            requires_thinking_time=False,  # Responds reasonably fast
        )

    def normalize_tool_calls(self, raw_calls: List[Any]) -> List[ToolCall]:
        """Normalize Google's FunctionCall format."""
        from victor.agent.tool_calling.base import ToolCall as _ToolCall

        normalized = []
        for i, call in enumerate(raw_calls):
            if isinstance(call, dict):
                # Google's function_call format
                if "function_call" in call:
                    fc = call["function_call"]
                    normalized.append(
                        _ToolCall(
                            id=call.get("id", f"call_{i}"),
                            name=fc.get("name", ""),
                            arguments=fc.get("args", {}),
                            raw=call,
                        )
                    )
                # Direct name/args format
                elif "name" in call:
                    normalized.append(
                        _ToolCall(
                            id=call.get("id", f"call_{i}"),
                            name=call.get("name", ""),
                            arguments=call.get("args", call.get("arguments", {})),
                            raw=call,
                        )
                    )
            elif hasattr(call, "function_call"):
                # Google object format
                fc = call.function_call
                normalized.append(
                    _ToolCall(
                        id=getattr(call, "id", f"call_{i}"),
                        name=getattr(fc, "name", ""),
                        arguments=dict(getattr(fc, "args", {})),
                        raw=call,
                    )
                )
        return normalized


class LMStudioAdapter(BaseProviderAdapter):
    """Adapter for LMStudio local inference server.

    Handles:
    - OpenAI-compatible API
    - Local model management
    - VRAM-aware model recommendations
    - Thinking tag extraction for Qwen3/DeepSeek-R1 models
    """

    # Models that output thinking tags
    THINKING_TAG_MODELS = [
        "qwen3",
        "deepseek-r1",
        "deepseek-reasoner",
    ]

    # Models with native tool calling support
    TOOL_CAPABLE_MODELS = [
        "-tools",  # Suffix pattern
        "qwen2.5-coder",
        "qwen3-coder",
        "llama3.1-tools",
        "llama3.3-tools",
        "deepseek-coder-tools",
        "deepseek-r1-tools",
    ]

    @property
    def name(self) -> str:
        return "lmstudio"

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            quality_threshold=0.75,  # Raised - many LMStudio models are high quality
            supports_thinking_tags=True,  # Enable for Qwen3/DeepSeek-R1 models
            thinking_tag_format="<think>...</think>",
            continuation_markers=["...", "Let me ", "I'll "],
            max_continuation_attempts=3,
            tool_call_format=ToolCallFormat.OPENAI,  # OpenAI-compatible
            supports_parallel_tools=False,  # Model-dependent
            grounding_required=False,  # Varies by loaded model
            grounding_strictness=0.65,  # Lenient
            continuation_patience=5,  # More patient - local is slower
            thinking_tokens_budget=1000,  # Generous budget for thinking models
            requires_thinking_time=True,  # Needs time
        )

    def model_supports_thinking(self, model: str) -> bool:
        """Check if a model outputs thinking tags.

        Args:
            model: Model name/identifier

        Returns:
            True if model uses <think>...</think> tags
        """
        model_lower = model.lower()
        return any(pattern in model_lower for pattern in self.THINKING_TAG_MODELS)

    def model_supports_tools(self, model: str) -> bool:
        """Check if a model supports native tool calling.

        Args:
            model: Model name/identifier

        Returns:
            True if model supports tool calling
        """
        model_lower = model.lower()
        return any(pattern in model_lower for pattern in self.TOOL_CAPABLE_MODELS)

    def extract_thinking_content(self, response: str) -> Tuple[str, str]:
        """Extract <think>...</think> tags from LMStudio model responses.

        Handles Qwen3 and DeepSeek-R1 models that output thinking tags.

        Args:
            response: The LLM response text

        Returns:
            Tuple of (thinking_content, main_content)
        """
        import re

        if not response:
            return ("", "")

        # Match <think>...</think> tags (case insensitive, multiline)
        think_pattern = r"<think>(.*?)</think>"
        matches = re.findall(think_pattern, response, re.DOTALL | re.IGNORECASE)

        thinking = "\n".join(matches) if matches else ""
        content = re.sub(think_pattern, "", response, flags=re.DOTALL | re.IGNORECASE).strip()

        return (thinking, content)

    def detect_continuation_needed(self, response: str) -> bool:
        """LMStudio-specific continuation detection."""
        if not response or not response.strip():
            return True

        response_stripped = response.strip()

        # Check for unclosed thinking tag
        if "<think>" in response.lower() and "</think>" not in response.lower():
            return True

        # Check standard markers
        for marker in self.capabilities.continuation_markers:
            if response_stripped.endswith(marker):
                return True

        return False

    def should_retry(self, error: Exception) -> Tuple[bool, float]:
        """LMStudio-specific retry logic for local server issues."""
        error_str = str(error).lower()

        # Connection refused - server not started
        if "connection refused" in error_str:
            return (True, 5.0)

        # Model loading / cold start
        if "loading" in error_str or "initializing" in error_str:
            return (True, 15.0)  # Longer wait for model loading

        # Timeout during model loading (first request)
        if "timeout" in error_str:
            return (True, 10.0)  # Retry with backoff

        # VRAM exhaustion
        if "out of memory" in error_str or "cuda" in error_str:
            return (False, 0.0)  # Don't retry OOM

        # Server overloaded
        if "503" in error_str or "service unavailable" in error_str:
            return (True, 5.0)

        # Fall back to base retry logic
        return super().should_retry(error)


class VLLMAdapter(BaseProviderAdapter):
    """Adapter for vLLM high-throughput inference server.

    Handles:
    - High-throughput batch inference
    - OpenAI-compatible API
    - PagedAttention for efficient memory
    """

    @property
    def name(self) -> str:
        return "vllm"

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            quality_threshold=0.75,
            supports_thinking_tags=False,
            continuation_markers=["..."],
            max_continuation_attempts=4,
            tool_call_format=ToolCallFormat.OPENAI,  # OpenAI-compatible
            output_deduplication=True,  # Some models repeat content
            supports_parallel_tools=True,  # vLLM handles parallel well
            streaming_chunk_size=512,  # Smaller chunks for faster streaming
            grounding_required=False,  # Depends on deployed model
            grounding_strictness=0.70,  # Moderate
            continuation_patience=4,  # Moderate patience
            thinking_tokens_budget=600,  # Moderate budget
            requires_thinking_time=False,  # Fast inference
        )


class LlamaCppAdapter(BaseProviderAdapter):
    """Adapter for llama.cpp server (CPU-friendly inference).

    Handles:
    - GGUF quantized models
    - CPU-optimized inference
    - Low memory footprint
    - Cross-platform support (macOS, Linux, Windows)
    """

    @property
    def name(self) -> str:
        return "llamacpp"

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            quality_threshold=0.70,  # Varies by quantization
            supports_thinking_tags=False,
            continuation_markers=["...", "Let me ", "I'll "],
            max_continuation_attempts=3,
            tool_call_format=ToolCallFormat.OPENAI,  # OpenAI-compatible
            supports_parallel_tools=False,  # Single request at a time
            streaming_chunk_size=256,  # Smaller chunks for CPU
            grounding_required=False,  # Model-dependent
            grounding_strictness=0.65,  # Lenient
            continuation_patience=5,  # More patient - CPU is slower
            thinking_tokens_budget=800,  # Generous for reasoning
            requires_thinking_time=True,  # CPU inference is slow
        )

    def should_retry(self, error: Exception) -> Tuple[bool, float]:
        """llama.cpp-specific retry logic for CPU inference issues."""
        error_str = str(error).lower()

        # Connection refused - server not started
        if "connection refused" in error_str:
            return (True, 5.0)

        # Timeout - CPU inference can be slow
        if "timeout" in error_str:
            return (True, 10.0)  # Retry with longer wait

        # Memory issues
        if "memory" in error_str or "oom" in error_str:
            return (False, 0.0)  # Don't retry memory issues

        return super().should_retry(error)


class GroqAdapter(BaseProviderAdapter):
    """Adapter for Groq LPU inference.

    Handles:
    - Ultra-fast LPU inference
    - Free tier limitations
    - Tool calling support
    """

    @property
    def name(self) -> str:
        return "groqcloud"

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            quality_threshold=0.78,
            supports_thinking_tags=False,
            max_continuation_attempts=5,
            tool_call_format=ToolCallFormat.OPENAI,
            supports_parallel_tools=True,
            streaming_chunk_size=1024,
            grounding_required=True,  # Verify citations
            grounding_strictness=0.80,  # High strictness
            continuation_patience=3,  # Standard patience
            thinking_tokens_budget=400,  # Standard budget
            requires_thinking_time=False,  # Ultra-fast inference
        )

    def should_retry(self, error: Exception) -> Tuple[bool, float]:
        """Groq-specific retry for rate limiting (free tier)."""
        error_str = str(error).lower()

        # Groq has aggressive rate limiting on free tier
        if "rate" in error_str:
            return (True, 30.0)  # Longer backoff for free tier

        # Tokens per minute limit
        if "tokens_per_minute" in error_str:
            return (True, 60.0)

        return super().should_retry(error)


class MistralAdapter(BaseProviderAdapter):
    """Adapter for Mistral AI models.

    Handles:
    - Mistral Large, Medium, Small
    - Codestral for code generation
    - Function calling
    """

    @property
    def name(self) -> str:
        return "mistral"

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            quality_threshold=0.80,
            supports_thinking_tags=False,
            max_continuation_attempts=5,
            tool_call_format=ToolCallFormat.OPENAI,
            supports_parallel_tools=True,
            grounding_required=True,  # Verify citations
            grounding_strictness=0.82,  # High strictness
            continuation_patience=3,  # Standard patience
            thinking_tokens_budget=400,  # Standard budget
            requires_thinking_time=False,  # Fast responses
        )


class MoonshotAdapter(BaseProviderAdapter):
    """Adapter for Moonshot/Kimi models.

    Handles:
    - Kimi K2 with 256K context
    - Extended reasoning capabilities
    """

    @property
    def name(self) -> str:
        return "moonshot"

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            quality_threshold=0.78,
            supports_thinking_tags=False,
            max_continuation_attempts=5,
            tool_call_format=ToolCallFormat.OPENAI,
            supports_parallel_tools=True,
            streaming_chunk_size=2048,  # Large context = larger chunks
            grounding_required=True,  # Verify citations
            grounding_strictness=0.78,  # Moderate strictness
            continuation_patience=4,  # More patient for extended reasoning
            thinking_tokens_budget=600,  # Larger budget for reasoning
            requires_thinking_time=False,  # Reasonable speed
        )


class TogetherAdapter(BaseProviderAdapter):
    """Adapter for Together AI inference platform.

    Handles:
    - Multiple open-source models
    - Serverless and dedicated deployments
    - OpenAI-compatible API
    """

    @property
    def name(self) -> str:
        return "together"

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            quality_threshold=0.75,
            supports_thinking_tags=False,
            max_continuation_attempts=4,
            tool_call_format=ToolCallFormat.OPENAI,
            supports_parallel_tools=True,
            grounding_required=True,  # Verify citations
            grounding_strictness=0.75,  # Moderate strictness
            continuation_patience=3,  # Standard patience
            thinking_tokens_budget=450,  # Standard budget
            requires_thinking_time=False,  # Fast inference
        )


class FireworksAdapter(BaseProviderAdapter):
    """Adapter for Fireworks AI inference.

    Handles:
    - Fast serverless inference
    - Multiple model support
    - Function calling
    """

    @property
    def name(self) -> str:
        return "fireworks"

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            quality_threshold=0.75,
            supports_thinking_tags=False,
            max_continuation_attempts=4,
            tool_call_format=ToolCallFormat.OPENAI,
            supports_parallel_tools=True,
            grounding_required=True,  # Verify citations
            grounding_strictness=0.75,  # Moderate strictness
            continuation_patience=3,  # Standard patience
            thinking_tokens_budget=450,  # Standard budget
            requires_thinking_time=False,  # Fast serverless inference
        )


class AzureOpenAIAdapter(BaseProviderAdapter):
    """Adapter for Azure OpenAI Service.

    Handles:
    - Enterprise Azure deployment
    - Same capabilities as OpenAI
    - Regional deployments
    """

    @property
    def name(self) -> str:
        return "azure-openai"

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            quality_threshold=0.80,
            supports_thinking_tags=False,
            max_continuation_attempts=5,
            tool_call_format=ToolCallFormat.OPENAI,
            supports_parallel_tools=True,
            grounding_required=True,  # Enterprise - verify citations
            grounding_strictness=0.85,  # High strictness
            continuation_patience=3,  # Standard patience
            thinking_tokens_budget=400,  # Standard budget
            requires_thinking_time=False,  # Fast responses
        )


class BedrockAdapter(BaseProviderAdapter):
    """Adapter for AWS Bedrock.

    Handles:
    - Claude, Llama, and other models on AWS
    - IAM authentication
    - Regional deployments
    """

    @property
    def name(self) -> str:
        return "bedrock"

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            quality_threshold=0.82,
            supports_thinking_tags=False,
            max_continuation_attempts=5,
            tool_call_format=ToolCallFormat.ANTHROPIC,  # Claude models
            supports_parallel_tools=True,
            grounding_required=True,  # Enterprise - verify citations
            grounding_strictness=0.85,  # High strictness (Claude on Bedrock)
            continuation_patience=3,  # Standard patience
            thinking_tokens_budget=400,  # Standard budget
            requires_thinking_time=False,  # Fast responses
        )

    def normalize_tool_calls(self, raw_calls: List[Any]) -> List[ToolCall]:
        """Normalize Bedrock's tool calls (varies by underlying model)."""
        from victor.agent.tool_calling.base import ToolCall as _ToolCall

        normalized = []
        for i, call in enumerate(raw_calls):
            if isinstance(call, dict):
                # Anthropic Claude format on Bedrock
                if call.get("type") == "tool_use":
                    normalized.append(
                        _ToolCall(
                            id=call.get("id", f"call_{i}"),
                            name=call.get("name", ""),
                            arguments=call.get("input", {}),
                            raw=call,
                        )
                    )
                # OpenAI-like format (Llama, etc.)
                elif "function" in call:
                    func = call.get("function", {})
                    normalized.append(
                        _ToolCall(
                            id=call.get("id", f"call_{i}"),
                            name=func.get("name", ""),
                            arguments=func.get("arguments", {}),
                            raw=call,
                        )
                    )
        return normalized


class VertexAIAdapter(BaseProviderAdapter):
    """Adapter for Google Cloud Vertex AI.

    Handles:
    - Enterprise Gemini deployment
    - Model Garden access
    - GCP authentication
    """

    @property
    def name(self) -> str:
        return "vertexai"

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            quality_threshold=0.80,
            supports_thinking_tags=False,
            max_continuation_attempts=5,
            tool_call_format=ToolCallFormat.NATIVE,
            supports_parallel_tools=True,
            streaming_chunk_size=2048,
            grounding_required=True,  # Enterprise - verify citations
            grounding_strictness=0.80,  # High strictness (Gemini enterprise)
            continuation_patience=4,  # Moderate patience
            thinking_tokens_budget=500,  # Standard budget
            requires_thinking_time=False,  # Reasonably fast
        )

    def normalize_tool_calls(self, raw_calls: List[Any]) -> List[ToolCall]:
        """Use Google adapter's normalization for Vertex."""
        google_adapter = GoogleAdapter()
        return google_adapter.normalize_tool_calls(raw_calls)


class CerebrasAdapter(BaseProviderAdapter):
    """Adapter for Cerebras wafer-scale inference.

    Handles:
    - Ultra-fast inference (1000+ tokens/sec)
    - Large context support (128K)
    - Qwen-3 thinking content filtering
    - Output deduplication for verbose models
    """

    @property
    def name(self) -> str:
        return "cerebras"

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            quality_threshold=0.78,  # High quality models
            supports_thinking_tags=True,  # Qwen-3 has inline thinking
            max_continuation_attempts=4,
            tool_call_format=ToolCallFormat.OPENAI,
            output_deduplication=True,  # Some models repeat content
            supports_parallel_tools=True,
            grounding_required=True,  # Verify citations
            grounding_strictness=0.75,  # Moderate strictness
            continuation_patience=3,  # Standard patience
            thinking_tokens_budget=600,  # Higher for Qwen-3 reasoning
            requires_thinking_time=False,  # Ultra-fast inference
        )


class HuggingFaceAdapter(BaseProviderAdapter):
    """Adapter for HuggingFace Inference API.

    Handles:
    - Hosted model inference
    - Custom model endpoints
    - Text Generation Inference (TGI)
    """

    @property
    def name(self) -> str:
        return "huggingface"

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            quality_threshold=0.70,  # Variable model quality
            supports_thinking_tags=False,
            continuation_markers=["..."],
            max_continuation_attempts=3,
            tool_call_format=ToolCallFormat.FALLBACK,  # Limited tool support
            supports_parallel_tools=False,
            grounding_required=False,  # Variable model quality
            grounding_strictness=0.65,  # Lenient
            continuation_patience=5,  # More patient
            thinking_tokens_budget=700,  # Generous budget
            requires_thinking_time=True,  # Variable speed
        )


class ReplicateAdapter(BaseProviderAdapter):
    """Adapter for Replicate model hosting.

    Handles:
    - Serverless model deployment
    - Cold start handling
    - Multiple model versions
    """

    @property
    def name(self) -> str:
        return "replicate"

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            quality_threshold=0.72,
            supports_thinking_tags=False,
            max_continuation_attempts=3,
            tool_call_format=ToolCallFormat.FALLBACK,
            supports_parallel_tools=False,
            grounding_required=False,  # Variable model quality
            grounding_strictness=0.65,  # Lenient
            continuation_patience=5,  # More patient (cold starts)
            thinking_tokens_budget=700,  # Generous budget
            requires_thinking_time=True,  # Cold start delays
        )

    def should_retry(self, error: Exception) -> Tuple[bool, float]:
        """Replicate-specific retry for cold starts."""
        error_str = str(error).lower()

        # Cold start - model being loaded
        if "cold start" in error_str or "starting" in error_str:
            return (True, 15.0)

        return super().should_retry(error)


class OpenRouterAdapter(BaseProviderAdapter):
    """Adapter for OpenRouter multi-model gateway.

    Handles:
    - Access to 100+ models
    - Automatic fallback
    - Credit-based billing
    """

    @property
    def name(self) -> str:
        return "openrouter"

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            quality_threshold=0.78,
            supports_thinking_tags=False,
            max_continuation_attempts=5,
            tool_call_format=ToolCallFormat.OPENAI,
            supports_parallel_tools=True,
            grounding_required=True,  # Verify citations
            grounding_strictness=0.78,  # Moderate strictness (varies by model)
            continuation_patience=4,  # Moderate patience
            thinking_tokens_budget=500,  # Standard budget
            requires_thinking_time=False,  # Varies by model
        )


# Registry of provider adapters
_ADAPTER_REGISTRY: dict[str, type[BaseProviderAdapter]] = {
    # Cloud providers
    "anthropic": AnthropicAdapter,
    "openai": OpenAIAdapter,
    "google": GoogleAdapter,
    "gemini": GoogleAdapter,  # Alias
    "deepseek": DeepSeekAdapter,
    "xai": GrokAdapter,
    "x-ai": GrokAdapter,  # Alias
    "grok": GrokAdapter,  # Alias
    "mistral": MistralAdapter,
    "moonshot": MoonshotAdapter,
    "kimi": MoonshotAdapter,  # Alias
    # Cloud inference platforms
    "groqcloud": GroqAdapter,
    "groq": GroqAdapter,  # Alias
    "together": TogetherAdapter,
    "fireworks": FireworksAdapter,
    "cerebras": CerebrasAdapter,
    "openrouter": OpenRouterAdapter,
    "replicate": ReplicateAdapter,
    "huggingface": HuggingFaceAdapter,
    "hf": HuggingFaceAdapter,  # Alias
    # Enterprise cloud providers
    "azure-openai": AzureOpenAIAdapter,
    "azure": AzureOpenAIAdapter,  # Alias
    "bedrock": BedrockAdapter,
    "aws": BedrockAdapter,  # Alias
    "vertexai": VertexAIAdapter,
    "vertex": VertexAIAdapter,  # Alias
    # Local inference
    "ollama": OllamaAdapter,
    "lmstudio": LMStudioAdapter,
    "vllm": VLLMAdapter,
    "llamacpp": LlamaCppAdapter,
    "llama-cpp": LlamaCppAdapter,  # Alias
    "llama.cpp": LlamaCppAdapter,  # Alias
}


def get_provider_adapter(provider_name: str) -> BaseProviderAdapter:
    """Get the appropriate adapter for a provider.

    Args:
        provider_name: Name of the provider (e.g., "deepseek", "xai", "openai")

    Returns:
        Configured provider adapter instance
    """
    adapter_class = _ADAPTER_REGISTRY.get(provider_name.lower(), BaseProviderAdapter)
    return adapter_class()


def register_provider_adapter(name: str, adapter_class: type[BaseProviderAdapter]) -> None:
    """Register a custom provider adapter.

    Args:
        name: Provider name to register
        adapter_class: Adapter class to use for this provider
    """
    _ADAPTER_REGISTRY[name.lower()] = adapter_class
