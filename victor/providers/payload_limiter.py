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

"""Payload size limiter for provider API requests.

Design Pattern: Strategy + Template Method
- PayloadLimiter protocol defines the interface
- ProviderPayloadLimiter provides configurable limits per provider
- Integrated with provider base for pre-flight payload checks

This module prevents HTTP 413 (Payload Too Large) errors by:
1. Estimating request payload size before sending
2. Truncating context when approaching limits
3. Logging warnings when payload is large
4. Providing fallback strategies (summarize, truncate oldest)

Provider-Specific Limits:
- Groq: ~4MB (LPU hardware constraint)
- Anthropic: ~25MB (generous limits)
- OpenAI: ~10MB (varies by endpoint)
- Local providers (Ollama/LMStudio): No hard limit (memory-bound)
"""

from __future__ import annotations

from victor.core.json_utils import json_dumps
import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol, Tuple

if TYPE_CHECKING:
    from victor.providers.base import Message, ToolDefinition

logger = logging.getLogger(__name__)


# Default limits (in bytes) - conservative estimates
DEFAULT_LIMITS = {
    # Cloud providers with strict limits
    "groq": 4 * 1024 * 1024,  # ~4MB - Groq LPU constraint
    "groqcloud": 4 * 1024 * 1024,  # Same as groq
    # Cloud providers with generous limits
    "anthropic": 25 * 1024 * 1024,  # ~25MB
    "openai": 10 * 1024 * 1024,  # ~10MB
    "google": 20 * 1024 * 1024,  # ~20MB
    "xai": 10 * 1024 * 1024,  # ~10MB (assumed similar to OpenAI)
    "deepseek": 10 * 1024 * 1024,  # ~10MB
    "moonshot": 15 * 1024 * 1024,  # ~15MB
    "mistral": 10 * 1024 * 1024,  # ~10MB
    "together": 10 * 1024 * 1024,  # ~10MB
    "openrouter": 10 * 1024 * 1024,  # ~10MB
    "fireworks": 10 * 1024 * 1024,  # ~10MB
    "cerebras": 5 * 1024 * 1024,  # ~5MB (LPU hardware)
    # Enterprise cloud
    "vertex": 20 * 1024 * 1024,  # ~20MB
    "azure": 10 * 1024 * 1024,  # ~10MB
    "bedrock": 25 * 1024 * 1024,  # ~25MB
    "huggingface": 5 * 1024 * 1024,  # ~5MB (free tier)
    "replicate": 10 * 1024 * 1024,  # ~10MB
    # Local providers - effectively unlimited (memory-bound)
    "ollama": 100 * 1024 * 1024,  # ~100MB (practical limit)
    "lmstudio": 100 * 1024 * 1024,  # ~100MB
    "vllm": 100 * 1024 * 1024,  # ~100MB
}

# Warning threshold as percentage of limit
WARNING_THRESHOLD = 0.80  # Warn at 80% of limit


class TruncationStrategy(Enum):
    """Strategy for truncating payload when limit is exceeded."""

    # Remove oldest messages first (keep recent context)
    TRUNCATE_OLDEST = "truncate_oldest"
    # Remove oldest tool results first (keep conversation flow)
    TRUNCATE_TOOL_RESULTS = "truncate_tool_results"
    # Summarize tool results instead of full content
    SUMMARIZE_TOOL_RESULTS = "summarize_tool_results"
    # Replace repeated long payloads with compact dictionary references
    DICTIONARY_COMPRESS = "dictionary_compress"
    # Reduce tool count by removing least relevant
    REDUCE_TOOLS = "reduce_tools"
    # Fail with informative error
    FAIL = "fail"


@dataclass
class PayloadEstimate:
    """Result of payload size estimation."""

    total_bytes: int
    messages_bytes: int
    tools_bytes: int
    overhead_bytes: int
    # Breakdown by message type
    system_bytes: int = 0
    user_bytes: int = 0
    assistant_bytes: int = 0
    tool_result_bytes: int = 0
    # Status
    exceeds_limit: bool = False
    limit_bytes: int = 0
    utilization_pct: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "total_bytes": self.total_bytes,
            "messages_bytes": self.messages_bytes,
            "tools_bytes": self.tools_bytes,
            "overhead_bytes": self.overhead_bytes,
            "system_bytes": self.system_bytes,
            "user_bytes": self.user_bytes,
            "assistant_bytes": self.assistant_bytes,
            "tool_result_bytes": self.tool_result_bytes,
            "exceeds_limit": self.exceeds_limit,
            "limit_bytes": self.limit_bytes,
            "utilization_pct": round(self.utilization_pct * 100, 1),
        }


@dataclass
class TruncationResult:
    """Result of payload truncation."""

    messages: List["Message"]
    tools: Optional[List["ToolDefinition"]]
    truncated: bool
    strategy_used: Optional[TruncationStrategy]
    messages_removed: int = 0
    tools_removed: int = 0
    bytes_saved: int = 0
    warning: Optional[str] = None
    compression_applied: bool = False
    compression_entries: int = 0


class PayloadLimiter(Protocol):
    """Protocol for payload size limiting."""

    def estimate_size(
        self,
        messages: List["Message"],
        tools: Optional[List["ToolDefinition"]] = None,
        **kwargs: Any,
    ) -> PayloadEstimate:
        """Estimate the payload size in bytes."""
        ...

    def check_limit(
        self,
        messages: List["Message"],
        tools: Optional[List["ToolDefinition"]] = None,
        **kwargs: Any,
    ) -> Tuple[bool, Optional[str]]:
        """Check if payload is within limits. Returns (ok, warning_message)."""
        ...

    def truncate_if_needed(
        self,
        messages: List["Message"],
        tools: Optional[List["ToolDefinition"]] = None,
        strategy: TruncationStrategy = TruncationStrategy.TRUNCATE_OLDEST,
        **kwargs: Any,
    ) -> TruncationResult:
        """Truncate payload if it exceeds limits."""
        ...


@dataclass
class ProviderPayloadLimiter:
    """Configurable payload limiter for providers.

    Usage:
        limiter = ProviderPayloadLimiter(provider_name="groq")
        estimate = limiter.estimate_size(messages, tools)
        if estimate.exceeds_limit:
            result = limiter.truncate_if_needed(messages, tools)
            messages = result.messages
    """

    provider_name: str
    max_payload_bytes: Optional[int] = None  # None = use default for provider
    warning_threshold: float = WARNING_THRESHOLD
    default_strategy: TruncationStrategy = TruncationStrategy.TRUNCATE_OLDEST
    # Additional options
    include_json_overhead: bool = True  # Account for JSON encoding overhead
    safety_margin: float = 0.95  # Only use 95% of limit for safety
    enable_dictionary_compression: Optional[bool] = None
    dictionary_min_match_chars: int = 160
    dictionary_preview_chars: int = 120
    dictionary_min_repeated_lines: int = 4

    def __post_init__(self) -> None:
        """Initialize limit from defaults if not provided."""
        if self.max_payload_bytes is None:
            self.max_payload_bytes = DEFAULT_LIMITS.get(
                self.provider_name.lower(),
                10 * 1024 * 1024,  # Default 10MB if unknown
            )
        if self.enable_dictionary_compression is None:
            try:
                from victor.core.feature_flags import (
                    FeatureFlag,
                    get_feature_flag_manager,
                )

                self.enable_dictionary_compression = get_feature_flag_manager().is_enabled(
                    FeatureFlag.USE_PROMPT_DICTIONARY_COMPRESSION
                )
            except Exception:
                self.enable_dictionary_compression = False

    @property
    def effective_limit(self) -> int:
        """Get effective limit with safety margin applied."""
        return int((self.max_payload_bytes or 0) * self.safety_margin)

    def estimate_size(
        self,
        messages: List["Message"],
        tools: Optional[List["ToolDefinition"]] = None,
        **kwargs: Any,
    ) -> PayloadEstimate:
        """Estimate the payload size in bytes.

        Uses JSON encoding for accurate estimation since most APIs accept JSON.

        Args:
            messages: Conversation messages
            tools: Tool definitions
            **kwargs: Additional payload fields

        Returns:
            PayloadEstimate with detailed breakdown
        """
        # Convert messages to dict representation for size calculation
        messages_data = []
        system_bytes = 0
        user_bytes = 0
        assistant_bytes = 0
        tool_result_bytes = 0

        for msg in messages:
            msg_dict = self._message_to_dict(msg)
            msg_str = json_dumps(msg_dict, ensure_ascii=False)
            msg_bytes = len(msg_str.encode("utf-8"))
            messages_data.append(msg_dict)

            # Track by role
            role = getattr(msg, "role", "user")
            if role == "system":
                system_bytes += msg_bytes
            elif role == "user":
                user_bytes += msg_bytes
            elif role == "assistant":
                assistant_bytes += msg_bytes
            elif role == "tool":
                tool_result_bytes += msg_bytes

        messages_bytes = len(json_dumps(messages_data, ensure_ascii=False).encode("utf-8"))

        # Estimate tools size
        tools_bytes = 0
        if tools:
            tools_data = [self._tool_to_dict(t) for t in tools]
            tools_bytes = len(json_dumps(tools_data, ensure_ascii=False).encode("utf-8"))

        # Estimate overhead (model name, params, etc.)
        overhead_bytes = 200  # Base overhead for typical params
        for key, value in kwargs.items():
            if value is not None:
                overhead_bytes += len(json_dumps({key: value}, ensure_ascii=False).encode("utf-8"))

        total_bytes = messages_bytes + tools_bytes + overhead_bytes

        # Calculate utilization
        limit = self.effective_limit
        utilization_pct = total_bytes / limit if limit > 0 else 0.0
        exceeds_limit = total_bytes > limit

        return PayloadEstimate(
            total_bytes=total_bytes,
            messages_bytes=messages_bytes,
            tools_bytes=tools_bytes,
            overhead_bytes=overhead_bytes,
            system_bytes=system_bytes,
            user_bytes=user_bytes,
            assistant_bytes=assistant_bytes,
            tool_result_bytes=tool_result_bytes,
            exceeds_limit=exceeds_limit,
            limit_bytes=limit,
            utilization_pct=utilization_pct,
        )

    def check_limit(
        self,
        messages: List["Message"],
        tools: Optional[List["ToolDefinition"]] = None,
        **kwargs: Any,
    ) -> Tuple[bool, Optional[str]]:
        """Check if payload is within limits.

        Args:
            messages: Conversation messages
            tools: Tool definitions
            **kwargs: Additional payload fields

        Returns:
            Tuple of (is_ok, warning_message)
            - is_ok: True if within limits
            - warning_message: Optional warning if approaching limit
        """
        estimate = self.estimate_size(messages, tools, **kwargs)

        if estimate.exceeds_limit:
            return (
                False,
                f"Payload size ({estimate.total_bytes:,} bytes) exceeds limit "
                f"({estimate.limit_bytes:,} bytes) for provider '{self.provider_name}'. "
                f"Tool results: {estimate.tool_result_bytes:,} bytes, "
                f"Messages: {estimate.messages_bytes:,} bytes.",
            )

        if estimate.utilization_pct >= self.warning_threshold:
            return (
                True,
                f"Payload at {estimate.utilization_pct * 100:.0f}% of limit for "
                f"'{self.provider_name}'. Consider truncating conversation history.",
            )

        return (True, None)

    def truncate_if_needed(
        self,
        messages: List["Message"],
        tools: Optional[List["ToolDefinition"]] = None,
        strategy: Optional[TruncationStrategy] = None,
        **kwargs: Any,
    ) -> TruncationResult:
        """Truncate payload if it exceeds limits.

        Args:
            messages: Conversation messages
            tools: Tool definitions
            strategy: Truncation strategy (default: TRUNCATE_OLDEST)
            **kwargs: Additional payload fields

        Returns:
            TruncationResult with modified messages/tools
        """
        strategy = strategy or self.default_strategy
        estimate = self.estimate_size(messages, tools, **kwargs)
        original_total_bytes = estimate.total_bytes

        if not estimate.exceeds_limit:
            return TruncationResult(
                messages=messages,
                tools=tools,
                truncated=False,
                strategy_used=None,
            )

        if strategy == TruncationStrategy.FAIL:
            return TruncationResult(
                messages=messages,
                tools=tools,
                truncated=False,
                strategy_used=strategy,
                warning=f"Payload exceeds limit: {estimate.total_bytes:,} > {estimate.limit_bytes:,} bytes",
            )

        compression_applied = False
        compression_entries = 0
        if self.enable_dictionary_compression:
            compressed_messages, compression_entries = self._apply_dictionary_compression(messages)
            if compression_entries > 0:
                compressed_estimate = self.estimate_size(compressed_messages, tools, **kwargs)
                if compressed_estimate.total_bytes < estimate.total_bytes:
                    messages = compressed_messages
                    estimate = compressed_estimate
                    compression_applied = True

                if compression_applied and not estimate.exceeds_limit:
                    return TruncationResult(
                        messages=messages,
                        tools=tools,
                        truncated=True,
                        strategy_used=TruncationStrategy.DICTIONARY_COMPRESS,
                        bytes_saved=original_total_bytes - estimate.total_bytes,
                        compression_applied=True,
                        compression_entries=compression_entries,
                    )

        # Apply truncation strategy
        if strategy == TruncationStrategy.TRUNCATE_OLDEST:
            result = self._truncate_oldest(messages, tools, estimate, **kwargs)
        elif strategy == TruncationStrategy.TRUNCATE_TOOL_RESULTS:
            result = self._truncate_tool_results(messages, tools, estimate, **kwargs)
        elif strategy == TruncationStrategy.SUMMARIZE_TOOL_RESULTS:
            result = self._summarize_tool_results(messages, tools, estimate, **kwargs)
        elif strategy == TruncationStrategy.REDUCE_TOOLS:
            result = self._reduce_tools(messages, tools, estimate, **kwargs)
        else:
            result = self._truncate_oldest(messages, tools, estimate, **kwargs)

        result.compression_applied = compression_applied
        result.compression_entries = compression_entries if compression_applied else 0
        final_estimate = self.estimate_size(result.messages, result.tools, **kwargs)
        result.bytes_saved = max(0, original_total_bytes - final_estimate.total_bytes)
        return result

    def _truncate_oldest(
        self,
        messages: List["Message"],
        tools: Optional[List["ToolDefinition"]],
        estimate: PayloadEstimate,
        **kwargs: Any,
    ) -> TruncationResult:
        """Remove oldest messages first, preserving system prompt and recent context."""
        if len(messages) <= 2:
            return TruncationResult(
                messages=messages,
                tools=tools,
                truncated=False,
                strategy_used=TruncationStrategy.TRUNCATE_OLDEST,
                warning="Cannot truncate: too few messages",
            )

        # Preserve system message (first) and recent messages
        truncated_messages = list(messages)
        messages_removed = 0
        original_bytes = estimate.total_bytes

        # Find where to start truncating (after system message if present)
        start_idx = 1 if messages and getattr(messages[0], "role", "") == "system" else 0

        while True:
            current_estimate = self.estimate_size(truncated_messages, tools, **kwargs)
            if not current_estimate.exceeds_limit:
                break

            # Remove the oldest non-system message
            if len(truncated_messages) <= start_idx + 2:
                # Keep at least last 2 messages for context
                break

            truncated_messages.pop(start_idx)
            messages_removed += 1

        bytes_saved = (
            original_bytes - self.estimate_size(truncated_messages, tools, **kwargs).total_bytes
        )

        logger.info(
            f"Truncated {messages_removed} oldest messages, saved {bytes_saved:,} bytes "
            f"for provider '{self.provider_name}'"
        )

        return TruncationResult(
            messages=truncated_messages,
            tools=tools,
            truncated=messages_removed > 0,
            strategy_used=TruncationStrategy.TRUNCATE_OLDEST,
            messages_removed=messages_removed,
            bytes_saved=bytes_saved,
        )

    def _truncate_tool_results(
        self,
        messages: List["Message"],
        tools: Optional[List["ToolDefinition"]],
        estimate: PayloadEstimate,
        **kwargs: Any,
    ) -> TruncationResult:
        """Truncate tool results to reduce payload size."""
        # Import here to avoid circular imports
        from victor.providers.base import Message as BaseMessage

        truncated_messages = []
        messages_modified = 0
        original_bytes = estimate.total_bytes

        # Maximum tool result size (truncate larger results)
        max_tool_result_size = 2000  # characters

        for msg in messages:
            if getattr(msg, "role", "") == "tool":
                content = getattr(msg, "content", "")
                if len(content) > max_tool_result_size:
                    # Truncate and add indicator
                    truncated_content = content[:max_tool_result_size] + "\n...[truncated]"
                    # Create new message with truncated content
                    new_msg = BaseMessage(
                        role="tool",
                        content=truncated_content,
                        tool_call_id=getattr(msg, "tool_call_id", None),
                    )
                    truncated_messages.append(new_msg)
                    messages_modified += 1
                else:
                    truncated_messages.append(msg)
            else:
                truncated_messages.append(msg)

        bytes_saved = (
            original_bytes - self.estimate_size(truncated_messages, tools, **kwargs).total_bytes
        )

        return TruncationResult(
            messages=truncated_messages,
            tools=tools,
            truncated=messages_modified > 0,
            strategy_used=TruncationStrategy.TRUNCATE_TOOL_RESULTS,
            messages_removed=messages_modified,
            bytes_saved=bytes_saved,
        )

    def _summarize_tool_results(
        self,
        messages: List["Message"],
        tools: Optional[List["ToolDefinition"]],
        estimate: PayloadEstimate,
        **kwargs: Any,
    ) -> TruncationResult:
        """Summarize tool results (placeholder - would need LLM call)."""
        # For now, fall back to truncation
        # A full implementation would call a small/fast LLM to summarize
        logger.warning("summarize_tool_results not fully implemented, falling back to truncate")
        return self._truncate_tool_results(messages, tools, estimate, **kwargs)

    def _reduce_tools(
        self,
        messages: List["Message"],
        tools: Optional[List["ToolDefinition"]],
        estimate: PayloadEstimate,
        **kwargs: Any,
    ) -> TruncationResult:
        """Reduce number of tools in payload."""
        if not tools or len(tools) <= 5:
            # Can't reduce further, fall back to truncate oldest
            return self._truncate_oldest(messages, tools, estimate, **kwargs)

        # Keep only core tools (remove half)
        original_tool_count = len(tools)
        reduced_tools = tools[: len(tools) // 2]
        tools_removed = original_tool_count - len(reduced_tools)
        original_bytes = estimate.total_bytes

        # Check if reduction helped
        new_estimate = self.estimate_size(messages, reduced_tools, **kwargs)
        if new_estimate.exceeds_limit:
            # Still exceeds, combine with message truncation
            result = self._truncate_oldest(messages, reduced_tools, new_estimate, **kwargs)
            result.tools_removed = tools_removed
            return result

        bytes_saved = original_bytes - new_estimate.total_bytes

        return TruncationResult(
            messages=messages,
            tools=reduced_tools,
            truncated=True,
            strategy_used=TruncationStrategy.REDUCE_TOOLS,
            tools_removed=tools_removed,
            bytes_saved=bytes_saved,
        )

    def _apply_dictionary_compression(
        self,
        messages: List["Message"],
    ) -> Tuple[List["Message"], int]:
        """Replace repeated long tool payloads with compact references."""
        from victor.providers.base import Message as BaseMessage

        compressed_messages: List["Message"] = []
        seen_tool_payloads: Dict[str, str] = {}
        compression_entries = 0

        for msg in messages:
            role = getattr(msg, "role", "")
            content = getattr(msg, "content", "")
            new_content, local_changes = self._compress_repeated_lines(content)

            if role == "tool" and len(new_content) >= self.dictionary_min_match_chars:
                ref_id = seen_tool_payloads.get(new_content)
                if ref_id is None:
                    ref_id = f"tool_result_{len(seen_tool_payloads) + 1}"
                    seen_tool_payloads[new_content] = ref_id
                else:
                    preview = (
                        new_content[: self.dictionary_preview_chars].replace("\n", " ").strip()
                    )
                    new_content = (
                        f"[dictionary-ref:{ref_id}] Repeated tool result omitted; "
                        f"identical content already appeared earlier. Preview: {preview}"
                    )
                    local_changes += 1

            if local_changes > 0:
                compression_entries += 1
                compressed_messages.append(
                    BaseMessage(
                        role=role or "user",
                        content=new_content,
                        name=getattr(msg, "name", None),
                        tool_calls=getattr(msg, "tool_calls", None),
                        tool_call_id=getattr(msg, "tool_call_id", None),
                        images=getattr(msg, "images", None),
                    )
                )
            else:
                compressed_messages.append(msg)

        return compressed_messages, compression_entries

    def _compress_repeated_lines(self, content: str) -> Tuple[str, int]:
        """Collapse long runs of identical lines inside a single payload."""
        if not content:
            return content, 0

        lines = content.splitlines()
        if len(lines) < self.dictionary_min_repeated_lines:
            return content, 0

        rewritten: List[str] = []
        changes = 0
        index = 0

        while index < len(lines):
            line = lines[index]
            next_index = index + 1
            while next_index < len(lines) and lines[next_index] == line:
                next_index += 1

            run_length = next_index - index
            if run_length >= self.dictionary_min_repeated_lines and line.strip():
                rewritten.append(line)
                rewritten.append(f"[repeated x{run_length - 1} additional identical lines omitted]")
                changes += 1
            else:
                rewritten.extend(lines[index:next_index])

            index = next_index

        if changes == 0:
            return content, 0
        return "\n".join(rewritten), changes

    def _message_to_dict(self, msg: "Message") -> Dict[str, Any]:
        """Convert message to dict for size estimation."""
        result: Dict[str, Any] = {
            "role": getattr(msg, "role", "user"),
            "content": getattr(msg, "content", ""),
        }
        if hasattr(msg, "tool_call_id") and msg.tool_call_id:
            result["tool_call_id"] = msg.tool_call_id
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            result["tool_calls"] = msg.tool_calls
        if hasattr(msg, "name") and msg.name:
            result["name"] = msg.name
        return result

    def _tool_to_dict(self, tool: "ToolDefinition") -> Dict[str, Any]:
        """Convert tool definition to dict for size estimation."""
        return {
            "type": "function",
            "function": {
                "name": getattr(tool, "name", ""),
                "description": getattr(tool, "description", ""),
                "parameters": getattr(tool, "parameters", {}),
            },
        }


def get_payload_limiter(
    provider_name: str,
    max_payload_bytes: Optional[int] = None,
    **kwargs: Any,
) -> ProviderPayloadLimiter:
    """Factory function to create a payload limiter for a provider.

    Args:
        provider_name: Name of the provider (e.g., "groq", "anthropic")
        max_payload_bytes: Optional override for max payload size
        **kwargs: Additional configuration options

    Returns:
        Configured ProviderPayloadLimiter instance
    """
    return ProviderPayloadLimiter(
        provider_name=provider_name,
        max_payload_bytes=max_payload_bytes,
        **kwargs,
    )
