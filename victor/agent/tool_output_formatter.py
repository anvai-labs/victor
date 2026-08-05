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

"""Tool output formatting for LLM context injection.

This module provides formatting of tool execution results into structured
text that LLMs can reliably parse. Key responsibilities:

1. **Anti-hallucination markers**: Uses clear TOOL_OUTPUT tags with ═══ delimiters
   to make tool results unmistakably authoritative.

2. **Token-optimized serialization**: Converts structured data (lists, dicts)
   to compact formats (TOON, CSV) achieving 30-60% token savings.

3. **Smart truncation**: For large outputs, uses intelligent truncation that
   preserves semantic structure rather than arbitrary cutoff.

4. **File structure extraction**: For very large files, shows structural summary
   (classes, functions) instead of raw content.

Usage:
    from victor.agent.tool_output_formatter import (
        ToolOutputFormatter,
        ToolOutputFormatterConfig,
        create_tool_output_formatter,
    )

    # Create formatter with config
    config = ToolOutputFormatterConfig(
        max_output_chars=15000,
        file_structure_threshold=50000,
        min_savings_threshold=0.15,
    )
    formatter = ToolOutputFormatter(config)

    # Format tool output
    formatted = formatter.format_tool_output(
        tool_name="read",
        args={"path": "/path/to/file.py"},
        output=file_content,
        context=FormattingContext(
            provider_name="anthropic",
            model="claude-3-sonnet",
            remaining_tokens=50000,
            max_tokens=100000,
        ),
    )

Extracted from AgentOrchestrator (December 2025) for cleaner separation of concerns.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol, Tuple
from victor.tools.tool_names import get_canonical_name

if TYPE_CHECKING:
    from victor.agent.presentation import PresentationProtocol

logger = logging.getLogger(__name__)


@dataclass
class FormattingContext:
    """Context information for formatting decisions.

    Provides information about the current state to enable
    context-aware formatting decisions (e.g., more aggressive
    compression when context is nearly full).

    Attributes:
        provider_name: Name of the provider (e.g., "openai", "ollama")
        model: Model name/identifier
        provider: Provider instance (for format preferences)
        remaining_tokens: Remaining tokens in context window
        max_tokens: Maximum context window size
        response_token_reserve: Tokens to reserve for response
    """

    provider_name: Optional[str] = None
    model: Optional[str] = None
    provider: Optional[Any] = None
    remaining_tokens: int = 50000
    max_tokens: int = 100000
    response_token_reserve: int = 4096

    @property
    def token_pressure(self) -> float:
        """Return 0.0-1.0 indicating how full the context is."""
        if self.max_tokens <= 0:
            return 0.0
        used = self.max_tokens - self.remaining_tokens
        return min(1.0, max(0.0, used / self.max_tokens))


@dataclass
class ToolOutputFormatterConfig:
    """Configuration for tool output formatting.

    Attributes:
        max_output_chars: Maximum characters before truncation (default 8000)
        file_structure_threshold: Threshold for showing structure instead of content (default 30000)
        min_savings_threshold: Minimum token savings to use optimized format (default 0.15 = 15%)
        max_classes_shown: Maximum classes to show in structure summary (default 15)
        max_functions_shown: Maximum functions to show in structure summary (default 20)
        sample_lines_start: Number of lines to show from file start (default 20)
        sample_lines_end: Number of lines to show from file end (default 15)
    """

    max_output_chars: int = 8000  # Reduced from 15000 (~47% reduction)
    file_structure_threshold: int = 30000  # Reduced from 50000 - show structure earlier
    min_savings_threshold: float = 0.15
    max_classes_shown: int = 15  # Reduced from 20
    max_functions_shown: int = 20  # Reduced from 30
    sample_lines_start: int = 20  # Reduced from 30
    sample_lines_end: int = 15  # Reduced from 20
    default_format_style: str = "plain"  # "plain" (token-efficient) or "xml" (anti-hallucination)


class TruncatorProtocol(Protocol):
    """Protocol for context-aware truncation."""

    def truncate_tool_result(self, content: str) -> Any:
        """Truncate content smartly, preserving semantic structure."""
        ...


@dataclass
class TruncationResult:
    """Result of truncation operation."""

    content: str
    truncated: bool = False
    truncated_chars: int = 0


class ToolOutputFormatter:
    """Formats tool outputs for LLM context injection.

    Handles:
    - Structured output serialization (lists, dicts -> compact formats)
    - Anti-hallucination markers (clear TOOL_OUTPUT boundaries)
    - Smart truncation for large outputs
    - File structure extraction for very large files
    """

    def __init__(
        self,
        config: Optional[ToolOutputFormatterConfig] = None,
        truncator: Optional[TruncatorProtocol] = None,
        presentation: Optional["PresentationProtocol"] = None,
    ):
        """Initialize formatter.

        Args:
            config: Configuration options (uses defaults if None)
            truncator: Optional smart truncator for context-aware truncation
            presentation: Optional presentation adapter for icons (creates default if None)
        """
        self.config = config or ToolOutputFormatterConfig()
        self._truncator = truncator
        # Lazy init for backward compatibility
        if presentation is None:
            from victor.agent.presentation import create_presentation_adapter

            self._presentation = create_presentation_adapter()
        else:
            self._presentation = presentation

    def format_tool_output(
        self,
        tool_name: str,
        args: Dict[str, Any],
        output: Any,
        context: Optional[FormattingContext] = None,
    ) -> str:
        """Format tool output with provider-specific strategy.

        Uses Strategy pattern to select appropriate formatting based on
        provider preferences:
        - Cloud providers: Plain JSON (token-efficient)
        - Local providers: XML with delimiters (cognition-optimized)
        - Experimental: TOON compact format

        For structured data (lists, dicts), uses token-optimized serialization
        to achieve 30-60% savings on tabular data.

        Args:
            tool_name: Name of the tool that was executed
            args: Arguments passed to the tool
            output: Raw output from the tool
            context: Optional formatting context for adaptive decisions

        Returns:
            Formatted string ready for LLM context injection
        """
        # CRITICAL GUARD: Check if output is a coroutine (indicates tool not awaited)
        import inspect as _inspect

        if _inspect.iscoroutine(output):
            logger.error(
                f"[FORMATTER] ❌ COROUTINE OBJECT RECEIVED: tool={tool_name}, "
                f"output={output}, type={type(output)}"
            )
            logger.error(
                "[FORMATTER] This indicates the tool function was called directly "
                "instead of being awaited through the executor."
            )
            raise RuntimeError(
                f"Tool '{tool_name}' returned coroutine object. "
                f"This indicates the tool was called directly (e.g., ls(path='.')) "
                f"instead of through the executor (await tool.execute(path='.')). "
                f"Raw output: {repr(output)}"
            )

        context = context or FormattingContext()

        # Use adaptive serialization for structured outputs
        output_str, format_hint = self._serialize_structured_output(
            tool_name, output, args, context
        )
        original_len = len(output_str)
        truncated = False

        # IMPORTANT: For file reads, check size BEFORE truncation
        # Very large files should use file structure mode, not head+tail truncation
        # This prevents losing critical content in the middle of large files
        if get_canonical_name(tool_name) == "read":
            if original_len > self.config.file_structure_threshold:
                # Skip truncation - use file structure mode instead
                return self._format_large_file_structure(args, output, output_str, original_len)

        # Use smart truncation if truncator available
        if self._truncator:
            try:
                truncation_result = self._truncator.truncate_tool_result(output_str)
                if hasattr(truncation_result, "truncated") and truncation_result.truncated:
                    truncated = True
                    output_str = truncation_result.content
                    logger.debug(
                        f"Smart truncated tool output for {tool_name}: "
                        f"{original_len:,} -> {len(output_str):,} chars "
                        f"(removed {truncation_result.truncated_chars:,} chars)"
                    )
            except Exception as e:
                logger.debug(f"Smart truncation failed, using simple: {e}")
                # Fall through to simple truncation

        # Fallback to simple truncation if needed
        if not truncated and original_len > self.config.max_output_chars:
            truncated = True
            output_str = output_str[: self.config.max_output_chars]

        # Use strategy pattern for ALL tools (provider-specific optimization)
        # This ensures cloud providers get Plain JSON (token-efficient)
        # and local providers get XML format (cognition-optimized)
        return self._format_with_strategy(
            tool_name, args, output_str, truncated, format_hint, context
        )

    def _format_with_strategy(
        self,
        tool_name: str,
        args: Dict[str, Any],
        output_str: str,
        truncated: bool,
        format_hint: Optional[str],
        context: FormattingContext,
    ) -> str:
        """Format output using provider-specific strategy.

        This implements the Strategy pattern - selecting the appropriate
        formatting strategy based on provider preferences.

        Args:
            tool_name: Name of the tool
            args: Tool arguments
            output_str: Serialized output string
            truncated: Whether output was truncated
            format_hint: Serialization format hint
            context: Formatting context with provider info

        Returns:
            Formatted output string
        """
        from victor.agent.format_strategies import FormatStrategyFactory

        # Get format specification from provider
        format_spec = self._get_format_spec(context)

        # Create strategy using factory
        try:
            strategy = FormatStrategyFactory.create(format_spec)

            # Build output object
            output_obj = output_str
            if truncated:
                output_obj = f"{output_str}\n[OUTPUT TRUNCATED]"

            # Format using strategy
            formatted = strategy.format(tool_name, args, output_obj, format_hint)

            # Estimate tokens
            estimated_tokens = strategy.estimate_tokens(formatted)

            logger.debug(
                f"Formatted {tool_name} output using {format_spec.style} strategy "
                f"({len(formatted)} chars, ~{estimated_tokens} tokens)"
            )

            # Record performance metric (non-blocking)
            self._record_format_metric(
                provider_name=context.provider_name or "unknown",
                format_style=format_spec.style,
                tool_name=tool_name,
                input_chars=len(output_str),
                output_chars=len(formatted),
                estimated_tokens=estimated_tokens,
            )

            return formatted

        except Exception as e:
            logger.warning(
                f"Strategy formatting failed for {tool_name}, falling back to generic: {e}"
            )
            return self._format_generic(tool_name, output_str, truncated, format_hint)

    def _record_format_metric(
        self,
        provider_name: str,
        format_style: str,
        tool_name: str,
        input_chars: int,
        output_chars: int,
        estimated_tokens: int,
    ) -> None:
        """Record formatting performance metric (non-blocking).

        Args:
            provider_name: Provider identifier
            format_style: Format style used
            tool_name: Tool name
            input_chars: Input character count
            output_chars: Output character count
            estimated_tokens: Estimated token count
        """
        try:
            from victor.agent.format_monitoring import FormatPerformanceMonitor

            monitor = FormatPerformanceMonitor.get_instance()
            monitor.record_format_metric(
                provider_name=provider_name,
                format_style=format_style,
                tool_name=tool_name,
                input_chars=input_chars,
                output_chars=output_chars,
                estimated_tokens=estimated_tokens,
            )
        except Exception as e:
            # Never fail formatting due to monitoring issues
            logger.debug(f"Failed to record format metric: {e}")

    def _get_format_spec(self, context: FormattingContext) -> Any:
        """Get format specification from provider or use default.

        Args:
            context: Formatting context

        Returns:
            ToolOutputFormat specification
        """
        from victor.agent.format_strategies import ToolOutputFormat

        # Check if provider has format preference
        if context.provider and hasattr(context.provider, "get_tool_output_format"):
            try:
                return context.provider.get_tool_output_format()
            except Exception as e:
                logger.debug(f"Failed to get provider format spec: {e}")

        # Default: use configured style (plain = token-efficient, xml = anti-hallucination)
        return ToolOutputFormat(style=self.config.default_format_style)

    def _format_large_file_structure(
        self,
        args: Dict[str, Any],
        output: Any,
        output_str: str,
        original_len: int,
    ) -> str:
        """Format very large files with structure summary and pagination guidance.

        Called BEFORE truncation to provide useful structure overview instead of
        head+tail truncation which loses critical middle content.
        """
        file_path = args.get("path", "unknown")
        file_content = str(output) if output is not None else output_str
        lines = file_content.split("\n")
        num_lines = len(lines)

        # Extract file structure (classes, functions)
        structure_summary = self.extract_file_structure(file_content, file_path)

        # Calculate pagination suggestions
        chunk_size = 300  # Recommended lines per chunk
        num_chunks = (num_lines + chunk_size - 1) // chunk_size

        return f"""<TOOL_OUTPUT tool="read" path="{file_path}">
═══ LARGE FILE: {original_len:,} chars / {num_lines:,} lines ═══

{structure_summary}

═══ HOW TO READ THIS FILE ═══

This file has {num_lines:,} lines. To read specific sections:

1. **Search for specific content** (FASTEST):
   read(path='{file_path}', search='function_name')
   read(path='{file_path}', search='class ClassName')

2. **Read by line range** (for context around found content):
   read(path='{file_path}', offset=0, limit={chunk_size})      # Lines 1-{chunk_size}
   read(path='{file_path}', offset={chunk_size}, limit={chunk_size})    # Lines {chunk_size+1}-{chunk_size*2}
   ... up to offset={num_lines - chunk_size}

3. **Total chunks needed**: {num_chunks} chunks of {chunk_size} lines each

═══ END OF FILE STRUCTURE ═══
</TOOL_OUTPUT>

ACTION REQUIRED: This file is too large to display fully.
- Use 'search' parameter to find specific functions/classes
- Use 'offset' and 'limit' parameters to read specific line ranges
- DO NOT re-read the full file without parameters - you will get this same structure view"""

    def _format_generic(
        self,
        tool_name: str,
        output_str: str,
        truncated: bool,
        format_hint: Optional[str],
    ) -> str:
        """Format generic tool output."""
        truncation_note = " [OUTPUT TRUNCATED]" if truncated else ""
        format_note = f"\n{format_hint}" if format_hint else ""
        return f"""<TOOL_OUTPUT tool="{tool_name}">{format_note}
{output_str}{truncation_note}
</TOOL_OUTPUT>"""

    def _serialize_structured_output(
        self,
        tool_name: str,
        output: Any,
        tool_args: Optional[Dict[str, Any]],
        context: FormattingContext,
    ) -> Tuple[str, Optional[str]]:
        """Serialize structured tool output for token efficiency.

        Uses adaptive serialization to convert lists and dicts to compact formats
        like TOON or CSV, achieving 30-60% token savings for tabular data.

        Args:
            tool_name: Name of the tool (for context)
            output: Raw output from the tool
            tool_args: Optional tool arguments (for extracting operation)
            context: Formatting context with provider/model info

        Returns:
            Tuple of (serialized_string, format_hint_or_none)
        """
        # Only serialize structured data (lists, dicts)
        if not isinstance(output, (list, dict)):
            return str(output) if output is not None else "", None

        # Skip serialization if output is too small (overhead not worth it)
        if isinstance(output, list) and len(output) < 3:
            return str(output), None
        if isinstance(output, dict) and len(output) < 3:
            return str(output), None

        try:
            # Lazy import to avoid circular dependencies
            from victor.processing.serialization import (
                get_adaptive_serializer,
                SerializationContext,
            )

            # Extract tool operation from arguments
            tool_operation = None
            if tool_args:
                tool_operation = tool_args.get("operation") or tool_args.get("subcommand")

            # Create serialization context
            ser_context = SerializationContext(
                provider=context.provider_name,
                model=context.model,
                tool_name=tool_name,
                tool_operation=tool_operation,
                remaining_context_tokens=context.remaining_tokens,
                max_context_tokens=context.max_tokens,
                response_token_reserve=context.response_token_reserve,
            )

            # Serialize using adaptive serializer
            serializer = get_adaptive_serializer()
            result = serializer.serialize(output, ser_context)

            # Only use if we got meaningful savings
            if result.estimated_savings_percent >= self.config.min_savings_threshold:
                logger.debug(
                    f"Serialized {tool_name} output: {result.format.value}, "
                    f"saved {result.estimated_savings_percent*100:.1f}% tokens"
                )
                return result.content, result.format_hint

            # Fall back to str() for low savings
            return str(output), None

        except Exception as e:
            logger.debug(f"Serialization failed for {tool_name}, using str(): {e}")
            return str(output), None

    def extract_file_structure(self, content: str, file_path: str) -> str:
        """Extract a structural summary for very large files.

        For Python files, extracts class and function definitions.
        For JS/TS files, extracts exports and functions.
        For other files, shows line count and sample lines.

        Args:
            content: Full file content
            file_path: Path to the file

        Returns:
            Structural summary string
        """
        lines = content.split("\n")
        num_lines = len(lines)
        ext = Path(file_path).suffix.lower()

        summary_parts = [f"FILE STRUCTURE: {file_path}"]
        summary_parts.append(f"Total lines: {num_lines}")

        if ext in (".py", ".pyi"):
            self._extract_python_structure(lines, summary_parts)
        elif ext in (".js", ".ts", ".jsx", ".tsx"):
            self._extract_js_structure(lines, summary_parts)

        # Add first and last few lines as sample
        summary_parts.append(f"\n--- FIRST {self.config.sample_lines_start} LINES ---")
        summary_parts.extend(lines[: self.config.sample_lines_start])
        summary_parts.append(f"\n--- LAST {self.config.sample_lines_end} LINES ---")
        summary_parts.extend(lines[-self.config.sample_lines_end :])

        return "\n".join(summary_parts)

    def _extract_python_structure(self, lines: List[str], summary_parts: List[str]) -> None:
        """Extract Python class and function definitions."""
        classes = []
        functions = []

        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("class ") and ":" in stripped:
                class_name = stripped[6:].split("(")[0].split(":")[0].strip()
                classes.append(f"  Line {i+1}: class {class_name}")
            elif stripped.startswith("def ") and "(" in stripped:
                func_name = stripped[4:].split("(")[0].strip()
                # Only top-level functions (no leading whitespace)
                if not line.startswith(" ") and not line.startswith("\t"):
                    functions.append(f"  Line {i+1}: def {func_name}()")

        if classes:
            summary_parts.append(f"\nClasses ({len(classes)}):")
            summary_parts.extend(classes[: self.config.max_classes_shown])
            if len(classes) > self.config.max_classes_shown:
                summary_parts.append(
                    f"  ... and {len(classes) - self.config.max_classes_shown} more"
                )

        if functions:
            summary_parts.append(f"\nFunctions ({len(functions)}):")
            summary_parts.extend(functions[: self.config.max_functions_shown])
            if len(functions) > self.config.max_functions_shown:
                summary_parts.append(
                    f"  ... and {len(functions) - self.config.max_functions_shown} more"
                )

    def _extract_js_structure(self, lines: List[str], summary_parts: List[str]) -> None:
        """Extract JavaScript/TypeScript exports and functions."""
        exports = []
        functions = []

        for i, line in enumerate(lines):
            stripped = line.strip()
            if "function " in stripped or "const " in stripped or "export " in stripped:
                if len(stripped) < 100:  # Skip very long lines
                    if stripped.startswith("export "):
                        exports.append(f"  Line {i+1}: {stripped[:60]}...")
                    elif "function " in stripped:
                        functions.append(f"  Line {i+1}: {stripped[:60]}...")

        if exports:
            summary_parts.append(f"\nExports ({len(exports)}):")
            summary_parts.extend(exports[: self.config.max_classes_shown])
        if functions:
            summary_parts.append(f"\nFunctions ({len(functions)}):")
            summary_parts.extend(functions[: self.config.max_classes_shown])

    def get_status_message(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        """Generate a user-friendly status message for a tool execution.

        Provides context-aware status messages showing relevant details
        (command, path, query, etc.) for different tool types.

        Args:
            tool_name: Name of the tool being executed
            tool_args: Arguments passed to the tool

        Returns:
            Status message string with icon prefix
        """
        running_icon = self._presentation.icon("running")
        canonical_tool_name = get_canonical_name(tool_name)
        display_name = tool_name or canonical_tool_name

        if canonical_tool_name == "shell":
            cmd = tool_args.get("cmd") or tool_args.get("command")
            if not cmd:
                return f"{running_icon} Running {display_name}..."
            cmd_display = cmd[:80] + "..." if len(cmd) > 80 else cmd
            return f"{running_icon} Running {display_name}: `{cmd_display}`"

        if canonical_tool_name == "ls":
            path = tool_args.get("path", ".")
            return f"{running_icon} Listing directory: {path}"

        if canonical_tool_name == "read":
            path = tool_args.get("path", "file")
            return f"{running_icon} Reading file: {path}"

        if canonical_tool_name == "edit":
            files = tool_args.get("ops") or tool_args.get("files") or tool_args.get("edits") or []
            if files and isinstance(files, list):
                paths = [f.get("path", "?") for f in files[:3] if isinstance(f, dict)]
                path_display = ", ".join(paths)
                if len(files) > 3:
                    path_display += f" (+{len(files) - 3} more)"
                return f"{running_icon} Editing: {path_display}"
            return f"{running_icon} Running {display_name}..."

        if canonical_tool_name == "write":
            path = tool_args.get("path", "file")
            return f"{running_icon} Writing file: {path}"

        if tool_name == "code_search":
            query = tool_args.get("query", "")
            query_display = query[:50] + "..." if len(query) > 50 else query
            return f"{running_icon} Searching: {query_display}"

        return f"{running_icon} Running {display_name}..."


def create_tool_output_formatter(
    config: Optional[ToolOutputFormatterConfig] = None,
    truncator: Optional[TruncatorProtocol] = None,
) -> ToolOutputFormatter:
    """Factory function to create a ToolOutputFormatter.

    Args:
        config: Optional configuration (uses defaults if None)
        truncator: Optional smart truncator for context-aware truncation

    Returns:
        Configured ToolOutputFormatter instance
    """
    return ToolOutputFormatter(config=config, truncator=truncator)


# Convenience function for one-off formatting
def format_tool_output(
    tool_name: str,
    args: Dict[str, Any],
    output: Any,
    context: Optional[FormattingContext] = None,
    config: Optional[ToolOutputFormatterConfig] = None,
) -> str:
    """Format tool output with default settings.

    Convenience function for simple use cases. For repeated formatting,
    prefer creating a ToolOutputFormatter instance.

    Args:
        tool_name: Name of the tool
        args: Tool arguments
        output: Tool output to format
        context: Optional formatting context
        config: Optional configuration

    Returns:
        Formatted output string
    """
    formatter = ToolOutputFormatter(config=config)
    return formatter.format_tool_output(tool_name, args, output, context)
