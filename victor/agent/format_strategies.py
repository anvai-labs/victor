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

"""Tool output formatting strategies using Strategy pattern.

This module provides pluggable formatting strategies for tool outputs,
enabling provider-specific optimization (XML for local models, JSON for cloud).

Design Patterns:
- Strategy Pattern: Different formatting behaviors (XML, Plain, TOON)
- Protocol-Based: Type-safe strategy contracts
- Immutable: Format specs are frozen dataclasses

Usage:
    strategy = FormatStrategyFactory.create(ToolOutputFormat(style="xml"))
    formatted = strategy.format(tool_name="read", args={"path": "file.py"}, output="content")
"""

from __future__ import annotations

from victor.core.json_utils import json_dumps
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# =============================================================================
# Immutable Configuration (Value Object Pattern)
# =============================================================================


@dataclass(frozen=True)
class ToolOutputFormat:
    """Immutable specification for tool output format.

    This is a value object - frozen dataclass ensures immutability.
    Providers return this from get_tool_output_format() to specify
    their preferred formatting strategy.

    Attributes:
        style: Format style - "plain" (JSON), "xml" (tags+delimiters), "toon" (compact)
        use_delimiters: Whether to add visual delimiters around content
        delimiter_char: Character for delimiters (default "=")
        delimiter_width: Width of delimiter lines (default 50)
        include_tags: Whether to wrap in XML-like tags
        tag_name: Tag name when include_tags=True (default "TOOL_OUTPUT")
        min_json_tokens: Minimum token count before using plain format (default 10)
        prefer_compact: Use TOON for structured data (default False)

    Examples:
        # Plain JSON (token-efficient for cloud providers)
        ToolOutputFormat(style="plain")

        # XML with delimiters (for local providers trained on this format)
        ToolOutputFormat(style="xml", use_delimiters=True, delimiter_char="=")

        # TOON compact format (experimental, for structured data)
        ToolOutputFormat(style="toon", prefer_compact=True)
    """

    style: Literal["plain", "xml", "toon"] = "plain"
    use_delimiters: bool = False
    delimiter_char: str = "="
    delimiter_width: int = 50
    include_tags: bool = False
    tag_name: str = "TOOL_OUTPUT"
    min_json_tokens: int = 10
    prefer_compact: bool = False

    def __post_init__(self) -> None:
        """Validate format specification."""
        # Validate built-in styles
        builtin_styles = ("plain", "xml", "toon")
        if self.style in builtin_styles:
            if self.use_delimiters and not self.delimiter_char:
                raise ValueError("delimiter_char required when use_delimiters=True")

            if self.delimiter_width < 10:
                raise ValueError("delimiter_width must be at least 10")

            if self.min_json_tokens < 0:
                raise ValueError("min_json_tokens must be non-negative")
        # Custom styles (registered via Factory) are allowed without validation
        # They will be validated by the factory when created

    def with_style(self, style: Literal["plain", "xml", "toon"]) -> "ToolOutputFormat":
        """Create new format with different style (immutable update)."""
        return ToolOutputFormat(**{**self.__dict__, "style": style})

    def with_delimiters(self, use: bool = True) -> "ToolOutputFormat":
        """Create new format with delimiters enabled/disabled."""
        return ToolOutputFormat(**{**self.__dict__, "use_delimiters": use})


# =============================================================================
# Strategy Protocol (Interface Segregation Principle)
# =============================================================================


@runtime_checkable
class ToolFormattingStrategy(Protocol):
    """Protocol for tool output formatting strategies.

    Each strategy implements the format() method to produce
    provider-specific formatted output. This follows the
    Strategy pattern - pluggable algorithms for formatting.

    Type checking:
        if isinstance(strategy, ToolFormattingStrategy):
            formatted = strategy.format(tool_name="read", args={}, output="content")
    """

    def format(
        self,
        tool_name: str,
        args: Dict[str, Any],
        output: Any,
        format_hint: Optional[str] = None,
    ) -> str:
        """Format tool output according to strategy.

        Args:
            tool_name: Name of the tool that was executed
            args: Arguments passed to the tool
            output: Raw output from tool execution
            format_hint: Optional hint about serialization format used

        Returns:
            Formatted string ready for LLM context injection
        """
        ...

    def estimate_tokens(self, content: str) -> int:
        """Estimate token count for formatted content.

        Args:
            content: Formatted content string

        Returns:
            Estimated token count (rough approximation: 1 token ≈ 4 chars)
        """
        ...


# =============================================================================
# Concrete Strategy Implementations
# =============================================================================


class PlainFormatStrategy:
    """Plain JSON formatting strategy (token-efficient, OpenAI-compatible).

    This strategy produces minimal overhead output suitable for cloud
    providers that charge per token. No XML tags, no delimiters,
    just the raw output as JSON or string.

    Best for: OpenAI, xAI, DeepSeek, Google, Anthropic, etc.
    Token overhead: ~5 tokens per tool call
    """

    def format(
        self,
        tool_name: str,
        args: Dict[str, Any],
        output: Any,
        format_hint: Optional[str] = None,
    ) -> str:
        """Format as plain JSON or string (minimal overhead)."""
        # For simple strings, return as-is
        if isinstance(output, str) and not format_hint:
            return output

        # For structured data, wrap in minimal JSON
        try:
            if isinstance(output, (dict, list)):
                return json_dumps(output, ensure_ascii=False, default=str)
            return str(output)
        except Exception as e:
            logger.debug(f"Plain formatting failed for {tool_name}: {e}")
            return str(output)

    def estimate_tokens(self, content: str) -> int:
        """Estimate tokens (rough approximation: 1 token ≈ 4 chars)."""
        return len(content) // 4


class XmlFormatStrategy:
    """XML formatting strategy with tags and delimiters.

    This strategy produces output wrapped in XML-like tags with
    visual delimiters. This format was historically used by Victor
    and local models (Ollama, vLLM, llama.cpp) are trained on it.

    Best for: Ollama, vLLM, llama.cpp (models trained on this format)
    Token overhead: ~53 tokens per tool call

    Design Rationale:
    - Anti-hallucination: Clear boundaries prevent model confusion
    - Training data: Local models expect this format
    - Debugging: Easy to identify tool results in conversation
    """

    def __init__(self, format_spec: ToolOutputFormat):
        """Initialize XML strategy with format specification.

        Args:
            format_spec: Immutable format specification
        """
        self._spec = format_spec

    def format(
        self,
        tool_name: str,
        args: Dict[str, Any],
        output: Any,
        format_hint: Optional[str] = None,
    ) -> str:
        """Format with XML tags and optional delimiters."""
        # Convert output to string
        output_str = self._serialize_output(output, format_hint)

        # Build tag attributes
        tag_attrs = self._build_tag_attributes(tool_name, args)

        # Construct opening tag
        opening_tag = (
            f"<{self._spec.tag_name} {tag_attrs}>" if tag_attrs else f"<{self._spec.tag_name}>"
        )

        # Add delimiters if specified
        if self._spec.use_delimiters:
            delimiter = self._spec.delimiter_char * self._spec.delimiter_width
            return f"""{opening_tag}
{delimiter}
{output_str}
{delimiter}
</{self._spec.tag_name}>"""

        # No delimiters, just tags
        return f"""{opening_tag}
{output_str}
</{self._spec.tag_name}>"""

    def _serialize_output(self, output: Any, format_hint: Optional[str]) -> str:
        """Serialize output to string."""
        if isinstance(output, str):
            return output

        try:
            if isinstance(output, (dict, list)):
                return json_dumps(output, ensure_ascii=False, indent=2, default=str)
            return str(output)
        except Exception as e:
            logger.debug(f"XML serialization failed: {e}")
            return str(output)

    def _build_tag_attributes(self, tool_name: str, args: Dict[str, Any]) -> str:
        """Build XML tag attributes from tool name and args."""
        attrs = [f'tool="{tool_name}"']

        # Add relevant args (skip verbose ones)
        for key, value in args.items():
            if key not in ("operation", "subcommand"):
                # Sanitize value for XML attribute
                value_str = str(value).replace('"', "&quot;")
                if len(value_str) < 100:  # Skip very long values
                    attrs.append(f'{key}="{value_str}"')

        return " ".join(attrs)

    def estimate_tokens(self, content: str) -> int:
        """Estimate tokens including XML overhead."""
        base_tokens = len(content) // 4
        # Add overhead for tags and delimiters
        overhead = 0
        if self._spec.include_tags:
            overhead += len(f"<{self._spec.tag_name}>") + len(f"</{self._spec.tag_name}>")
        if self._spec.use_delimiters:
            overhead += self._spec.delimiter_width * 2
        return base_tokens + (overhead // 4)


class ToonFormatStrategy:
    """TOON (Token-Optimized Object Notation) formatting strategy.

    EXPERIMENTAL: Compact format achieving 30-60% token reduction
    over JSON for structured data. Uses delimiter-based serialization
    optimized for LLM tokenization.

    Best for: Structured tabular data (lists of dicts, search results)
    Token overhead: ~20 tokens per tool call
    Token savings: 30-60% on structured data vs JSON

    Example:
        Input: [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]
        JSON: 58 tokens
        TOON: 22 tokens (62% savings)

    Note: This is experimental and may not be suitable for all providers.
    """

    def __init__(self, format_spec: ToolOutputFormat):
        """Initialize TOON strategy with format specification.

        Args:
            format_spec: Immutable format specification
        """
        self._spec = format_spec

    def format(
        self,
        tool_name: str,
        args: Dict[str, Any],
        output: Any,
        format_hint: Optional[str] = None,
    ) -> str:
        """Format using TOON compact notation."""
        # For non-structured data, fall back to plain
        if not isinstance(output, (list, dict)):
            return str(output)

        try:
            return self._toon_serialize(output)
        except Exception as e:
            logger.debug(f"TOON serialization failed, falling back to plain: {e}")
            return str(output)

    def _toon_serialize(self, data: Any, indent: int = 0) -> str:
        """Serialize data to TOON format.

        TOON rules:
        - Lists: Use | delimiter between items
        - Dicts: Use : delimiter, one key-value per line
        - Strings: Quoted if contain special chars
        - Numbers: Unquoted
        - Indentation: 2 spaces per nesting level
        """
        if isinstance(data, str):
            # Quote if contains special characters
            if any(c in data for c in ["|", ":", "\n", '"']):
                return f'"{data}"'
            return data

        if isinstance(data, (int, float, bool)) or data is None:
            return str(data)

        if isinstance(data, list):
            if not data:
                return "[]"

            # Simple list: single line with | delimiter
            if all(not isinstance(item, (list, dict)) for item in data):
                items = [self._toon_serialize(item, indent) for item in data]
                return " | ".join(items)

            # Complex list: multi-line
            lines = []
            for item in data:
                serialized = self._toon_serialize(item, indent + 1)
                lines.append("  " * indent + serialized)
            return "\n".join(lines)

        if isinstance(data, dict):
            if not data:
                return "{}"

            lines = []
            for key, value in data.items():
                key_str = str(key)
                value_str = self._toon_serialize(value, indent + 1)
                lines.append("  " * indent + f"{key_str}: {value_str}")
            return "\n".join(lines)

        return str(data)

    def estimate_tokens(self, content: str) -> int:
        """Estimate tokens (TOON is typically 30-60% more efficient)."""
        # Rough estimate: TOON is ~40% more efficient than JSON
        base_tokens = len(content) // 4
        return int(base_tokens * 0.6)  # Conservative estimate


# =============================================================================
# Factory Pattern (Creation Logic Centralized)
# =============================================================================


class FormatStrategyFactory:
    """Factory for creating formatting strategies.

    This encapsulates the creation logic, following the Factory pattern.
    Strategies are created based on ToolOutputFormat specification.

    Usage:
        factory = FormatStrategyFactory()
        strategy = factory.create(ToolOutputFormat(style="xml"))
        formatted = strategy.format(tool_name="read", args={}, output="content")
    """

    _strategies: Dict[str, type] = {
        "plain": PlainFormatStrategy,
        "xml": XmlFormatStrategy,
        "toon": ToonFormatStrategy,
    }

    @classmethod
    def create(cls, format_spec: ToolOutputFormat) -> ToolFormattingStrategy:
        """Create formatting strategy based on format specification.

        Args:
            format_spec: Immutable format specification

        Returns:
            Configured formatting strategy instance

        Raises:
            ValueError: If format style is not supported
        """
        strategy_class = cls._strategies.get(format_spec.style)

        if strategy_class is None:
            raise ValueError(
                f"Unsupported format style: {format_spec.style}. "
                f"Supported: {list(cls._strategies.keys())}"
            )

        # XML and TOON strategies need format_spec, Plain doesn't
        if format_spec.style in ("xml", "toon"):
            return strategy_class(format_spec)

        return strategy_class()

    @classmethod
    def register_strategy(cls, style: str, strategy_class: type) -> None:
        """Register a custom formatting strategy (extensibility point).

        Args:
            style: Style name for the strategy
            strategy_class: Strategy class (must implement ToolFormattingStrategy)

        Example:
            class CustomFormatStrategy:
                def format(self, tool_name, args, output, format_hint=None):
                    return f"[CUSTOM] {tool_name}: {output}"

            FormatStrategyFactory.register_strategy("custom", CustomFormatStrategy)
            strategy = FormatStrategyFactory.create(ToolOutputFormat(style="custom"))
        """
        if not hasattr(strategy_class, "format"):
            raise TypeError(f"Strategy class must implement 'format' method: {strategy_class}")

        cls._strategies[style] = strategy_class
        logger.info(f"Registered custom format strategy: {style}")


# =============================================================================
# Default Format Specifications (Convenience Constants)
# =============================================================================

#: Plain JSON format (default for cloud providers)
PLAIN_FORMAT = ToolOutputFormat(style="plain")

#: XML format with delimiters (default for local providers)
XML_FORMAT = ToolOutputFormat(
    style="xml",
    use_delimiters=True,
    delimiter_char="=",
    delimiter_width=50,
    include_tags=True,
    tag_name="TOOL_OUTPUT",
)

#: TOON compact format (experimental)
TOON_FORMAT = ToolOutputFormat(
    style="toon",
    prefer_compact=True,
)
