from __future__ import annotations

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

"""Response sanitization for Victor.

Handles cleaning and validation of model responses, including:
- Removing malformed XML/HTML tags
- Filtering training data leakage patterns
- Validating tool names
- Detecting garbage content from local models
- AST-based code extraction and cleanup (for LLM-generated code)
- Markdown code block extraction
- Python syntax validation
- **Real-time streaming content filtering for thinking tokens**
"""

import ast
import logging
import re
import textwrap
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional, Tuple

# Import native dispatch functions for ~3x speedup when available
from victor.processing.native.tool_extraction import (
    sanitize_response_fast,
    is_garbage_content_fast,
    strip_markup_fast,
)

logger = logging.getLogger(__name__)

# Try to import native extensions for performance
_NATIVE_AVAILABLE = False
_native = None

try:
    import victor_native as _native

    _NATIVE_AVAILABLE = True
    logger.debug(f"Native streaming filter loaded (v{_native.__version__})")
except ImportError:
    logger.debug("Native extensions not available, using Python streaming filter")


class ThinkingState(Enum):
    """State for tracking thinking blocks during streaming."""

    NORMAL = "normal"  # Outside thinking block
    IN_THINKING = "in_thinking"  # Inside a thinking block
    SUPPRESSED = "suppressed"  # Thinking content being suppressed


@dataclass
class StreamingChunkResult:
    """Result of processing a streaming chunk.

    Attributes:
        content: The processed content to display
        is_thinking: Whether this content is from thinking/reasoning
        state_changed: Whether we transitioned into/out of thinking state
        entering_thinking: True if we just entered thinking state
        exiting_thinking: True if we just exited thinking state
    """

    content: str
    is_thinking: bool = False
    state_changed: bool = False
    entering_thinking: bool = False
    exiting_thinking: bool = False


class StreamingContentFilter:
    """Real-time content filter for streaming responses.

    Processes thinking tokens and tracks thinking state during streaming.
    Designed to be called chunk-by-chunk as content arrives.

    Key features:
    - Detects DeepSeek thinking markers (<｜begin▁of▁thinking｜>, <｜end▁of▁thinking｜>)
    - Detects Qwen3 thinking blocks (<think>...</think>)
    - Tracks thinking state across chunks
    - Buffers partial tokens at chunk boundaries
    - Limits thinking content length to prevent runaway generation
    - Returns styled results for rendering thinking content dimmed/italic

    Usage:
        filter = StreamingContentFilter()

        async for chunk in stream:
            result = filter.process_chunk(chunk.content)
            if result.entering_thinking:
                show_thinking_header()
            if result.content:
                if result.is_thinking:
                    display_dimmed(result.content)
                else:
                    display_normal(result.content)
            if result.exiting_thinking:
                show_response_separator()

            if filter.should_abort():
                break  # Thinking exceeded max length
    """

    # Max chars of thinking content before aborting (prevents runaway)
    MAX_THINKING_CONTENT: int = 50000

    # Thinking token patterns — consolidated into single alternation regexes
    # so the regex engine matches all variants in one pass per chunk.
    THINKING_START_PATTERN: re.Pattern = re.compile(
        r"<｜begin▁of▁thinking｜>|<\|begin_of_thinking\|>|<think>"
    )

    THINKING_END_PATTERN: re.Pattern = re.compile(
        r"<｜end▁of▁thinking｜>|<\|end_of_thinking\|>|</think>"
    )

    # Combined pattern for stripping inline tokens (all start + end markers)
    INLINE_TOKEN_PATTERN: re.Pattern = re.compile(
        r"<｜end▁of▁thinking｜>|<｜begin▁of▁thinking｜>"
        r"|<\|end_of_thinking\|>|<\|begin_of_thinking\|>"
        r"|</think>|<think>"
    )

    # Keep list aliases for backward compatibility with any external consumers
    THINKING_START_PATTERNS: List[re.Pattern] = [THINKING_START_PATTERN]
    THINKING_END_PATTERNS: List[re.Pattern] = [THINKING_END_PATTERN]
    INLINE_TOKEN_PATTERNS: List[re.Pattern] = [INLINE_TOKEN_PATTERN]

    def __init__(self, suppress_thinking: bool = False):
        """Initialize the streaming filter.

        Args:
            suppress_thinking: If True, completely suppress thinking content.
                              If False (default), return thinking content marked as such.
        """
        self._state = ThinkingState.NORMAL
        self._buffer = ""  # Buffer for partial tokens
        self._thinking_content_length = 0
        self._total_thinking_length = 0
        self._should_abort = False
        self._abort_reason: Optional[str] = None
        self._suppress_thinking = suppress_thinking

    def reset(self) -> None:
        """Reset filter state for a new response."""
        self._state = ThinkingState.NORMAL
        self._buffer = ""
        self._thinking_content_length = 0
        self._total_thinking_length = 0
        self._should_abort = False
        self._abort_reason = None

    def process_chunk(self, chunk: str) -> StreamingChunkResult:
        """Process a content chunk, detecting thinking state transitions.

        Args:
            chunk: Raw content chunk from streaming response

        Returns:
            StreamingChunkResult with content and state information
        """
        if not chunk:
            return StreamingChunkResult(content="")

        # Combine with buffer from previous chunk
        text = self._buffer + chunk
        self._buffer = ""

        # Keep potential partial token at end of chunk
        # (handles tokens split across chunk boundaries)
        partial_start = self._find_partial_token_start(text)
        if partial_start >= 0:
            self._buffer = text[partial_start:]
            text = text[:partial_start]

        # Process based on current state
        if self._state == ThinkingState.NORMAL:
            return self._process_normal_state(text)
        else:  # IN_THINKING
            return self._process_thinking_state(text)

    def _process_normal_state(self, text: str) -> StreamingChunkResult:
        """Process text when in normal (non-thinking) state."""
        # Check for thinking start markers
        for pattern in self.THINKING_START_PATTERNS:
            match = pattern.search(text)
            if match:
                # Output text before the marker
                before = text[: match.start()]
                after = text[match.end() :]

                # Enter thinking state
                self._state = ThinkingState.IN_THINKING
                self._thinking_content_length = 0
                logger.debug("StreamingContentFilter: Entered thinking state")

                # Process remaining text in thinking state
                thinking_result = self._process_thinking_state(after)

                # Combine: normal content before, then thinking content
                # Return the before content (normal) and flag state change
                result_content = self._strip_inline_tokens(before)
                if thinking_result.content and not self._suppress_thinking:
                    # We have both normal and thinking content
                    return StreamingChunkResult(
                        content=result_content,
                        is_thinking=False,
                        state_changed=True,
                        entering_thinking=True,
                        exiting_thinking=thinking_result.exiting_thinking,
                    )
                return StreamingChunkResult(
                    content=result_content,
                    is_thinking=False,
                    state_changed=True,
                    entering_thinking=True,
                )

        # No thinking markers, strip any orphaned tokens and return
        return StreamingChunkResult(
            content=self._strip_inline_tokens(text),
            is_thinking=False,
        )

    def _process_thinking_state(self, text: str) -> StreamingChunkResult:
        """Process text when inside a thinking block."""
        # Track thinking content length
        self._thinking_content_length += len(text)
        self._total_thinking_length += len(text)

        # Check for max thinking content
        if self._thinking_content_length > self.MAX_THINKING_CONTENT:
            self._should_abort = True
            self._abort_reason = (
                f"Thinking content exceeded {self.MAX_THINKING_CONTENT} chars. "
                "Model may be stuck in a reasoning loop."
            )
            logger.warning(f"StreamingContentFilter: {self._abort_reason}")
            return StreamingChunkResult(content="", is_thinking=True)

        # Check for thinking end markers
        for pattern in self.THINKING_END_PATTERNS:
            match = pattern.search(text)
            if match:
                # Get thinking content before end marker
                thinking_content = text[: match.start()]
                after = text[match.end() :]
                self._state = ThinkingState.NORMAL
                logger.debug(
                    f"StreamingContentFilter: Exited thinking state "
                    f"(thinking content: {self._thinking_content_length} chars)"
                )

                # Process remaining text in normal state
                normal_result = self._process_normal_state(after)

                # Return thinking content with state change flag
                if self._suppress_thinking:
                    return StreamingChunkResult(
                        content=normal_result.content,
                        is_thinking=False,
                        state_changed=True,
                        exiting_thinking=True,
                    )
                return StreamingChunkResult(
                    content=self._strip_inline_tokens(thinking_content),
                    is_thinking=True,
                    state_changed=True,
                    exiting_thinking=True,
                )

        # Still in thinking state, return thinking content (or empty if suppressed)
        if self._suppress_thinking:
            return StreamingChunkResult(content="", is_thinking=True)
        return StreamingChunkResult(
            content=self._strip_inline_tokens(text),
            is_thinking=True,
        )

    def _strip_inline_tokens(self, text: str) -> str:
        """Strip any inline thinking tokens from text."""
        for pattern in self.INLINE_TOKEN_PATTERNS:
            text = pattern.sub("", text)
        return text

    def _find_partial_token_start(self, text: str) -> int:
        """Find where a partial token might start at end of text.

        Looks for incomplete tokens like '<｜begin', '<think', etc.

        Returns:
            Index where partial token starts, or -1 if none
        """
        if not text:
            return -1

        # Check last N chars for start of known tokens
        # Max token length is about 25 chars
        check_len = min(len(text), 25)
        suffix = text[-check_len:]

        # Token prefixes to look for
        prefixes = ["<｜", "<|", "<think", "</think", "<think>"]

        for prefix in prefixes:
            for i in range(1, len(prefix)):
                partial = prefix[:i]
                if suffix.endswith(partial):
                    return len(text) - i

        return -1

    def should_abort(self) -> bool:
        """Check if streaming should be aborted due to excessive thinking."""
        return self._should_abort

    @property
    def abort_reason(self) -> Optional[str]:
        """Reason for abort if should_abort() is True."""
        return self._abort_reason

    @property
    def state(self) -> ThinkingState:
        """Current thinking state."""
        return self._state

    @property
    def is_thinking(self) -> bool:
        """Whether we are currently in thinking state."""
        return self._state == ThinkingState.IN_THINKING

    @property
    def total_thinking_length(self) -> int:
        """Total length of thinking content processed."""
        return self._total_thinking_length

    def flush(self) -> StreamingChunkResult:
        """Flush any remaining buffered content.

        Call this when streaming ends to get any buffered content.

        Returns:
            StreamingChunkResult with any remaining content
        """
        result = StreamingChunkResult(
            content=self._strip_inline_tokens(self._buffer),
            is_thinking=self._state == ThinkingState.IN_THINKING,
        )
        self._buffer = ""
        return result


class _NativeFilterWrapper:
    """Python wrapper around the native Rust StreamingFilter.

    The Rust extension exposes ``process_chunk`` / ``should_abort`` /
    ``get_state`` but lacks ``is_thinking`` and ``abort_reason`` — used by
    ``handler.py``.  This thin wrapper fills those gaps so callers see a
    uniform interface regardless of which backend is active. ``flush``
    delegates to the native implementation when the wheel provides it
    (releases after U8-F7); older wheels keep the previous behavior.
    """

    def __init__(self, inner: Any, max_thinking_content: int) -> None:
        self._inner = inner
        self._max_thinking_content = max_thinking_content
        self._should_abort = False
        self._abort_reason: Optional[str] = None

    def flush(self) -> StreamingChunkResult:
        """Flush any remaining buffered content.

        The native filter buffers a potential partial thinking-tag at the end
        of each chunk; at end-of-stream those bytes belong to the output and
        must be emitted (previously dropped — U8-F7).
        """
        native_flush = getattr(self._inner, "flush", None)
        if native_flush is not None:
            return native_flush()
        return StreamingChunkResult(content="", is_thinking=self.is_thinking)

    def process_chunk(self, text: str) -> StreamingChunkResult:
        result = self._inner.process_chunk(text)
        if (
            not self._should_abort
            and self._inner.get_thinking_length() > self._max_thinking_content
        ):
            self._should_abort = True
            self._abort_reason = (
                f"Thinking content exceeded {self._max_thinking_content} chars. "
                "Model may be stuck in a reasoning loop."
            )

        if self._should_abort and (
            result.is_thinking or result.entering_thinking or result.exiting_thinking
        ):
            return StreamingChunkResult(
                content="",
                is_thinking=result.is_thinking,
                state_changed=result.state_changed,
                entering_thinking=result.entering_thinking,
                exiting_thinking=result.exiting_thinking,
            )
        return result

    def should_abort(self) -> bool:
        return self._should_abort or self._inner.should_abort()

    @property
    def abort_reason(self) -> Optional[str]:
        return self._abort_reason or self._inner.get_abort_reason()

    @property
    def is_thinking(self) -> bool:
        return self._inner.get_state() != "normal"

    def flush(self) -> StreamingChunkResult:
        # Native filter has no buffering — flush is a no-op.
        return StreamingChunkResult(content="", is_thinking=self.is_thinking)


def create_streaming_filter(
    suppress_thinking: bool = False, max_thinking_content: Optional[int] = None
) -> "StreamingContentFilter":
    """Factory function to create the best available streaming filter.

    Uses native Rust implementation when available for 2-3x better performance.
    Falls back to Python implementation otherwise.

    Args:
        suppress_thinking: If True, suppress thinking content entirely
        max_thinking_content: Max chars before aborting. When omitted, use the
            current ``StreamingContentFilter.MAX_THINKING_CONTENT`` value so test
            patches and runtime overrides apply consistently to both backends.

    Returns:
        StreamingContentFilter (native or Python implementation)
    """
    effective_max_thinking_content = (
        StreamingContentFilter.MAX_THINKING_CONTENT
        if max_thinking_content is None
        else max_thinking_content
    )
    if _NATIVE_AVAILABLE:
        return _NativeFilterWrapper(
            _native.StreamingFilter(suppress_thinking, effective_max_thinking_content),
            effective_max_thinking_content,
        )
    return StreamingContentFilter(suppress_thinking)


def strip_thinking_tokens_fast(content: str) -> str:
    """Strip thinking tokens using native implementation when available.

    This is 2-3x faster than regex-based stripping.

    Args:
        content: Text potentially containing thinking tokens

    Returns:
        Content with thinking tokens removed
    """
    if _NATIVE_AVAILABLE:
        return _native.strip_thinking_tokens(content)

    # Fallback to simple string replacement
    patterns = [
        "<｜begin▁of▁thinking｜>",
        "<｜end▁of▁thinking｜>",
        "<|begin_of_thinking|>",
        "<|end_of_thinking|>",
        "<think>",
        "</think>",
    ]
    result = content
    for pattern in patterns:
        result = result.replace(pattern, "")
    return result


@dataclass
class CodeSanitizationResult:
    """Result of code sanitization."""

    code: str  # The sanitized code
    is_valid: bool  # Whether the code is syntactically valid
    errors: List[str] = field(default_factory=list)  # List of errors/issues found
    fixes_applied: List[str] = field(default_factory=list)  # List of fixes applied
    function_names: List[str] = field(default_factory=list)  # Names of functions found


class ResponseSanitizer:
    """Sanitizes and validates model responses.

    Handles common issues from local models:
    - Repeated </function> or </parameter> tags
    - XML-like formatting artifacts
    - JSON-like tool call attempts in plain text
    - Instruction/example leakage from model training
    """

    # Patterns indicating training data leakage
    LEAKAGE_PATTERNS: List[str] = [
        r"Do not invent any new or additional parameters.*",
        r"The parameter value should be passed as a string.*",
        r"If you want to call multiple functions.*",
        r"Do NOT surround the function call.*",
        r"All parameters are required unless.*",
        r"The agent is not allowed to directly access.*",
        r"Begin by calling ls.*",
        r"shell\(cmd=.*\)",
        r"No files read yet\. Avoid file-specific claims.*",
        r'ls\(path="[^"]*"\)',
        r'read\(path="[^"]*"\)',
    ]

    # Patterns indicating garbage/malformed output
    GARBAGE_PATTERNS: List[str] = [
        r"FUNCTION_CALL\s*\{",  # Raw function call syntax
        r"</function>\s*</function>",  # Repeated closing tags
        r"<parameter[^>]*>",  # Raw parameter tags
        r'^\s*\{\s*"name":\s*"[^"]+",\s*"arguments":',  # Raw JSON tool calls
        r"^\s*<IMPORTANT>",  # Instruction leakage
        r"^\s*<important>",  # Instruction leakage (lowercase)
        r"</important>",  # Instruction leakage closing tag
        r"^\s*Do NOT",  # Instruction leakage
        r"^\s*NEVER\s+",  # Instruction leakage
        r"^\s*ALWAYS include",  # Instruction leakage
        r"^\s*- Do NOT",  # List-style instruction leakage
        r"^\s*- Use lowercase",  # Formatting instruction leakage
        r"^\s*- The parameters",  # Parameter instruction leakage
        r"the XML tag",  # Format instruction leakage
        r"backticks.*markdown",  # Format instruction leakage
        r"JSON parsing",  # Technical instruction leakage
        r"file system structure",  # Security instruction leakage
        r"\[TOOL_REQUEST\]",  # LMStudio default format leakage
        r"\[END_TOOL_REQUEST\]",  # LMStudio default format leakage
    ]

    # Thinking token patterns to strip (DeepSeek, Qwen, and other reasoning models)
    # These tokens leak through in content when running locally via Ollama
    THINKING_TOKEN_PATTERNS: List[str] = [
        r"<｜end▁of▁thinking｜>",  # DeepSeek end-of-thinking marker
        r"<｜begin▁of▁thinking｜>",  # DeepSeek begin-of-thinking marker
        r"<\|end_of_thinking\|>",  # ASCII variant
        r"<\|begin_of_thinking\|>",  # ASCII variant
        r"<think>.*?</think>",  # Qwen3 thinking blocks
        r"</think>",  # Orphaned Qwen3 thinking close tag
        r"<think>",  # Orphaned Qwen3 thinking open tag
    ]

    # Standalone shell-like command lines emitted as plain text instead of tool calls.
    # These commonly appear when smaller/local models try to "suggest" commands
    # rather than using the tool-calling interface.
    PLAIN_COMMAND_WORDS = frozenset(
        {"ls", "rg", "grep", "find", "cat", "sed", "head", "tail", "pwd", "read"}
    )
    PATHLIKE_HINT_WORDS = frozenset(
        {
            "rust",
            "victor",
            "tests",
            "docs",
            "scripts",
            "src",
            "site",
            "ui",
            "web",
            ".victor",
        }
    )

    # Patterns for invalid/hallucinated tool names
    INVALID_TOOL_PATTERNS: List[str] = [
        r"^example_",
        r"^func_",
        r"^function_",
        r"^tool_name",
        r"^my_",
        r"^test_tool",
        r"^sample_",
        r"/$",  # Ends with slash
        r"^<",  # Starts with XML tag
        r">$",  # Ends with XML tag
        r"\s",  # Contains whitespace
        r"^\d",  # Starts with number
    ]

    # ================================================================
    # Code Sanitization Patterns (for LLM-generated code)
    # ================================================================

    # Markdown code block extraction
    MARKDOWN_CODE_BLOCK = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL)

    # LLM artifacts to remove from code
    CODE_ARTIFACTS = [
        re.compile(
            r"^Here(?:'s| is) (?:the |a |my )?(?:complete |corrected |fixed )?"
            r"(?:implementation|solution|code|function).*?:",
            re.IGNORECASE | re.MULTILINE,
        ),
        re.compile(
            r"^(?:The )?(?:complete |corrected |fixed )?"
            r"(?:implementation|solution|code|function).*?:",
            re.IGNORECASE | re.MULTILINE,
        ),
        re.compile(r"^Note:.*$", re.MULTILINE),
        re.compile(r"^Explanation:.*$", re.MULTILINE),
        re.compile(r"^Output:.*$", re.MULTILINE),
        re.compile(r"^Example:.*$", re.MULTILINE),
        re.compile(r"^Test:.*$", re.MULTILINE),
        re.compile(r"^Usage:.*$", re.MULTILINE),
    ]

    # Patterns for detecting incomplete code
    INCOMPLETE_CODE_PATTERNS = [
        re.compile(r"\.\.\.$"),  # Trailing ellipsis
        re.compile(
            r"#\s*(?:TODO|FIXME|XXX|continue|rest of|implementation|more|etc).*$",
            re.IGNORECASE,
        ),
        re.compile(r"pass\s*#.*(?:implement|add|fill|complete).*$", re.IGNORECASE),
    ]

    def strip_markup(self, text: str) -> str:
        """Remove simple XML/HTML-like tags to salvage plain text.

        Uses native Rust implementation when available for ~3x speedup.

        Args:
            text: Text potentially containing markup

        Returns:
            Plain text with markup removed
        """
        if not text:
            return text
        return strip_markup_fast(text)

    def _looks_like_path_fragment(self, token: str) -> bool:
        normalized = token.strip().strip("`'\"")
        if not normalized:
            return False
        lowered = normalized.lower()
        if lowered in self.PATHLIKE_HINT_WORDS:
            return True
        if normalized.startswith((".", "/", "~")):
            return True
        if "/" in normalized or "\\" in normalized:
            return True
        return bool(re.search(r"\.[A-Za-z0-9]{1,8}$", normalized))

    def _is_malformed_parameter_line(self, line: str) -> bool:
        normalized = line.strip()
        if not normalized:
            return False
        return bool(
            re.match(r"^parameter\s*=", normalized, re.IGNORECASE)
            or re.match(r"^<parameter\b", normalized, re.IGNORECASE)
        )

    def _is_plaintext_tool_command_line(self, line: str) -> bool:
        normalized = " ".join(line.strip().split())
        if not normalized or len(normalized) > 80:
            return False
        if any(ch in normalized for ch in "(){}[]=:,"):
            return False

        tokens = normalized.split()
        if len(tokens) > 3:
            return False

        command = None
        arguments: List[str] = []
        if tokens[0].lower() in self.PLAIN_COMMAND_WORDS:
            command = tokens[0].lower()
            arguments = tokens[1:]
        elif tokens[-1].lower() in self.PLAIN_COMMAND_WORDS:
            command = tokens[-1].lower()
            arguments = tokens[:-1]
        else:
            return False

        if command == "pwd" and not arguments:
            return True
        if not arguments:
            return False
        return all(self._looks_like_path_fragment(token) for token in arguments)

    def extract_plaintext_tool_command_lines(self, text: str | None) -> List[str]:
        if not text:
            return []
        return [
            line.strip() for line in text.splitlines() if self._is_plaintext_tool_command_line(line)
        ]

    def has_tool_format_confusion(self, text: str | None) -> bool:
        if not text:
            return False
        malformed_lines = self.extract_plaintext_tool_command_lines(text)
        if not malformed_lines:
            return False
        sanitized = self.sanitize(text)
        removed_ratio = 1.0 - (len(sanitized) / max(len(text), 1))
        return len(malformed_lines) >= 2 or removed_ratio >= 0.35

    def sanitize(self, text: str) -> str:
        """Sanitize model response by removing malformed patterns.

        Uses native Rust implementation for common patterns (~3x speedup),
        then applies additional ResponseSanitizer-specific filtering.

        Handles common issues from local models:
        - Repeated </function> or </parameter> tags
        - XML-like formatting artifacts
        - JSON-like tool call attempts in plain text
        - Instruction/example leakage from model training

        Args:
            text: Raw response text from the model

        Returns:
            Cleaned text suitable for display
        """
        if not text:
            return text

        original_len = len(text)

        # Fast path: use native dispatch for common sanitization patterns
        # This handles: repeated closing tags, orphaned XML-like tags,
        # thinking tokens, training data leakage, JSON tool calls, etc.
        text = sanitize_response_fast(text)

        # Additional line-by-line filtering for plaintext tool command lines
        # (beyond what the native dispatch handles)
        lines = text.split("\n")
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            # Skip standalone shell-like command lines; these are malformed
            # plain-text tool intents, not useful user-facing content.
            if self._is_plaintext_tool_command_line(stripped):
                continue
            # Skip standalone malformed parameter artifact lines that some
            # models emit instead of structured tool-call arguments.
            if self._is_malformed_parameter_line(stripped):
                continue
            cleaned_lines.append(line)
        text = "\n".join(cleaned_lines)

        # Clean up excessive whitespace
        text = re.sub(r"\n{4,}", "\n\n\n", text)
        text = text.strip()

        # Log if significant content was removed (indicates model confusion)
        if len(text) < original_len * 0.5 and original_len > 100:
            logger.warning(
                f"Sanitization removed {original_len - len(text)} chars "
                f"({100 - len(text) * 100 // original_len}% of response) - "
                "model may be confused about tool calling format"
            )

        return text

    def is_garbage_content(self, content: str) -> bool:
        """Detect if content is garbage/malformed output from local models.

        Uses native Rust implementation for common patterns (~3x speedup),
        then checks additional ResponseSanitizer-specific patterns.

        Args:
            content: Content chunk to check

        Returns:
            True if content appears to be garbage
        """
        if not content:
            return False

        # Fast path: use native dispatch for common garbage patterns
        if is_garbage_content_fast(content):
            return True

        # Additional patterns specific to ResponseSanitizer beyond native dispatch
        additional_patterns = [
            r"^\s*<important>",  # Instruction leakage (lowercase)
            r"</important>",  # Instruction leakage closing tag
            r"^\s*ALWAYS include",  # Instruction leakage
            r"^\s*- Do NOT",  # List-style instruction leakage
            r"^\s*- Use lowercase",  # Formatting instruction leakage
            r"^\s*- The parameters",  # Parameter instruction leakage
            r"the XML tag",  # Format instruction leakage
            r"backticks.*markdown",  # Format instruction leakage
            r"JSON parsing",  # Technical instruction leakage
            r"file system structure",  # Security instruction leakage
            r"\[END_TOOL_REQUEST\]",  # LMStudio default format leakage
        ]

        for pattern in additional_patterns:
            if re.search(pattern, content, re.IGNORECASE | re.MULTILINE):
                return True

        return False

    def is_valid_tool_name(self, name: str) -> bool:
        """Check if a tool name is valid and not a hallucination.

        Args:
            name: Tool name to validate

        Returns:
            True if the tool name appears valid
        """
        if not name or not isinstance(name, str):
            return False

        # Reject common hallucinated/example tool names
        for pattern in self.INVALID_TOOL_PATTERNS:
            if re.search(pattern, name, re.IGNORECASE):
                logger.debug(f"Rejecting invalid tool name: {name}")
                return False

        # Must be alphanumeric with underscores only
        if not re.match(r"^[a-zA-Z][a-zA-Z0-9_]*$", name):
            logger.debug(f"Rejecting malformed tool name: {name}")
            return False

        return True

    # ================================================================
    # Code Sanitization Methods (for LLM-generated code)
    # ================================================================

    def sanitize_code(self, code: str) -> CodeSanitizationResult:
        """Sanitize LLM-generated Python code.

        Performs AST-based cleanup:
        1. Extract code from markdown blocks
        2. Remove LLM artifacts (preamble text)
        3. Fix indentation issues
        4. Remove incomplete trailing code
        5. Validate Python syntax
        6. Extract function names

        Args:
            code: Raw code string from LLM

        Returns:
            CodeSanitizationResult with cleaned code and metadata
        """
        errors: List[str] = []
        fixes: List[str] = []

        # Step 1: Extract code from markdown blocks
        original_code = code
        code = self._extract_code_from_markdown(code)
        if code != original_code:
            fixes.append("extracted_from_markdown")

        # Step 2: Remove LLM artifacts (preamble text)
        code, artifact_fixes = self._remove_code_artifacts(code)
        fixes.extend(artifact_fixes)

        # Step 3: Fix common indentation issues
        code, indent_fixes = self._fix_code_indentation(code)
        fixes.extend(indent_fixes)

        # Step 4: Remove trailing incomplete code
        code, incomplete_fixes = self._remove_incomplete_code(code)
        fixes.extend(incomplete_fixes)

        # Step 5: Ensure valid Python (try to extract function if needed)
        code, extract_fixes = self._ensure_valid_python(code)
        fixes.extend(extract_fixes)

        # Step 6: Validate final result
        is_valid, validation_errors = self._validate_python_syntax(code)
        errors.extend(validation_errors)

        # Extract function names
        function_names = self._extract_function_names(code)

        return CodeSanitizationResult(
            code=code,
            is_valid=is_valid,
            errors=errors,
            fixes_applied=fixes,
            function_names=function_names,
        )

    def _extract_code_from_markdown(self, code: str) -> str:
        """Extract code from markdown code blocks."""
        matches = self.MARKDOWN_CODE_BLOCK.findall(code)
        if matches:
            # Use the longest code block
            return max(matches, key=len).strip()

        # Remove any remaining backticks
        code = code.replace("```python", "").replace("```", "")
        return code.strip()

    def _remove_code_artifacts(self, code: str) -> Tuple[str, List[str]]:
        """Remove common LLM preamble/artifacts from code."""
        fixes = []
        original = code

        for pattern in self.CODE_ARTIFACTS:
            if pattern.search(code):
                code = pattern.sub("", code)

        if code != original:
            fixes.append("removed_artifacts")

        # Remove leading non-code lines before first def/class/import
        lines = code.split("\n")
        code_start_idx = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith(("def ", "class ", "import ", "from ", "@", "#")):
                code_start_idx = i
                break
            elif stripped and not stripped.startswith(
                ("def ", "class ", "import ", "from ", "@", "#", '"""', "'''")
            ):
                continue

        if code_start_idx > 0:
            code = "\n".join(lines[code_start_idx:])
            fixes.append("removed_preamble")

        return code.strip(), fixes

    def _fix_code_indentation(self, code: str) -> Tuple[str, List[str]]:
        """Fix common indentation issues in code."""
        fixes = []
        lines = code.split("\n")

        # Detect and fix mixed tabs/spaces
        has_tabs = any("\t" in line for line in lines)
        has_spaces = any(line.startswith(" ") for line in lines)

        if has_tabs and has_spaces:
            code = code.replace("\t", "    ")
            fixes.append("converted_tabs_to_spaces")
            lines = code.split("\n")

        # Detect if code is over-indented
        non_empty_lines = [line for line in lines if line.strip()]
        if non_empty_lines:
            min_indent = min(len(line) - len(line.lstrip()) for line in non_empty_lines)
            if min_indent > 0:
                code = textwrap.dedent(code)
                fixes.append("fixed_over_indentation")

        return code.strip(), fixes

    def _remove_incomplete_code(self, code: str) -> Tuple[str, List[str]]:
        """Remove trailing incomplete code."""
        fixes = []
        lines = code.split("\n")

        while lines:
            last_line = lines[-1].strip()
            is_incomplete = False

            for pattern in self.INCOMPLETE_CODE_PATTERNS:
                if pattern.search(last_line):
                    is_incomplete = True
                    break

            if is_incomplete:
                lines.pop()
                fixes.append("removed_incomplete_line")
            else:
                break

        return "\n".join(lines), fixes

    def _ensure_valid_python(self, code: str) -> Tuple[str, List[str]]:
        """Ensure code is valid Python, trying to extract function if needed."""
        fixes = []

        # First, try to parse as-is
        try:
            ast.parse(code)
            return code, fixes
        except SyntaxError:
            pass

        # Try to extract just the function definition
        extracted = self._extract_function_def(code)
        if extracted and extracted != code:
            try:
                ast.parse(extracted)
                fixes.append("extracted_function_only")
                return extracted, fixes
            except SyntaxError:
                pass

        # Try adding common missing imports
        code_with_imports = self._add_common_imports(code)
        try:
            ast.parse(code_with_imports)
            fixes.append("added_missing_imports")
            return code_with_imports, fixes
        except SyntaxError:
            pass

        # Return original if we can't fix it
        return code, fixes

    def _extract_function_def(self, code: str) -> Optional[str]:
        """Extract just the function definition from code."""
        lines = code.split("\n")
        func_lines = []
        in_function = False
        base_indent = 0

        for line in lines:
            stripped = line.lstrip()

            if stripped.startswith("def "):
                in_function = True
                base_indent = len(line) - len(stripped)
                func_lines.append(line[base_indent:])
            elif in_function:
                if line.strip() == "":
                    func_lines.append("")
                elif len(line) - len(line.lstrip()) > base_indent:
                    func_lines.append(line[base_indent:])
                elif stripped.startswith(("def ", "class ", "@")):
                    break
                elif len(line) - len(line.lstrip()) <= base_indent and line.strip():
                    break
                else:
                    func_lines.append(line[base_indent:] if len(line) > base_indent else "")

        if func_lines:
            return "\n".join(func_lines)
        return None

    def _add_common_imports(self, code: str) -> str:
        """Add commonly needed imports if they're used but not imported."""
        imports_to_add = []

        module_checks = [
            ("re", r"\bre\."),
            ("math", r"\bmath\."),
            ("itertools", r"\bitertools\."),
            ("collections", r"\bcollections\."),
            ("functools", r"\bfunctools\."),
            ("typing", r"\b(?:List|Dict|Set|Tuple|Optional|Union|Any)\b"),
        ]

        for module, pattern in module_checks:
            if re.search(pattern, code) and f"import {module}" not in code:
                if module == "typing":
                    imports_to_add.append(
                        "from typing import List, Dict, Set, Tuple, Optional, Union, Any"
                    )
                else:
                    imports_to_add.append(f"import {module}")

        if imports_to_add:
            return "\n".join(imports_to_add) + "\n\n" + code
        return code

    def _validate_python_syntax(self, code: str) -> Tuple[bool, List[str]]:
        """Validate Python syntax."""
        errors = []
        try:
            ast.parse(code)
            return True, errors
        except SyntaxError as e:
            errors.append(f"SyntaxError at line {e.lineno}: {e.msg}")
            return False, errors

    def _extract_function_names(self, code: str) -> List[str]:
        """Extract function names from code."""
        names = []
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    names.append(node.name)
        except SyntaxError:
            # Fallback to regex
            for match in re.finditer(r"def\s+(\w+)\s*\(", code):
                names.append(match.group(1))
        return names

    def is_valid_python(self, code: str) -> bool:
        """Check if code is valid Python.

        Args:
            code: Code to check

        Returns:
            True if code is syntactically valid
        """
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    def extract_function(self, code: str, function_name: Optional[str] = None) -> Optional[str]:
        """Extract a function from code.

        Args:
            code: Code containing the function
            function_name: Name of function to extract (if None, extracts first)

        Returns:
            The function code or None if not found
        """
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    if function_name is None or node.name == function_name:
                        lines = code.split("\n")
                        start = node.lineno - 1
                        end = node.end_lineno if hasattr(node, "end_lineno") else len(lines)
                        return "\n".join(lines[start:end])
        except SyntaxError:
            pass
        return None


# Module-level singleton for convenience
_sanitizer = ResponseSanitizer()


def sanitize_response(text: str) -> str:
    """Sanitize model response (convenience function).

    Args:
        text: Raw response text

    Returns:
        Cleaned text
    """
    return _sanitizer.sanitize(text)


def is_garbage_content(content: str) -> bool:
    """Detect garbage content (convenience function).

    Args:
        content: Content to check

    Returns:
        True if garbage detected
    """
    return _sanitizer.is_garbage_content(content)


def is_valid_tool_name(name: str) -> bool:
    """Validate tool name (convenience function).

    Args:
        name: Tool name to validate

    Returns:
        True if valid
    """
    return _sanitizer.is_valid_tool_name(name)


def strip_markup(text: str) -> str:
    """Strip markup from text (convenience function).

    Args:
        text: Text with markup

    Returns:
        Plain text
    """
    return _sanitizer.strip_markup(text)


# ================================================================
# Code Sanitization Convenience Functions
# ================================================================


def sanitize_code(code: str) -> CodeSanitizationResult:
    """Sanitize LLM-generated Python code (convenience function).

    Args:
        code: Raw code from LLM

    Returns:
        CodeSanitizationResult with cleaned code and metadata
    """
    return _sanitizer.sanitize_code(code)


def extract_code(code: str) -> str:
    """Extract and sanitize code, returning just the code string.

    Args:
        code: Raw code from LLM

    Returns:
        Sanitized code string
    """
    result = _sanitizer.sanitize_code(code)
    return result.code


def is_valid_python(code: str) -> bool:
    """Check if code is valid Python (convenience function).

    Args:
        code: Code to check

    Returns:
        True if code is syntactically valid
    """
    return _sanitizer.is_valid_python(code)


def extract_function(code: str, function_name: Optional[str] = None) -> Optional[str]:
    """Extract a function from code (convenience function).

    Args:
        code: Code containing the function
        function_name: Name of function to extract (if None, extracts first)

    Returns:
        The function code or None if not found
    """
    return _sanitizer.extract_function(code, function_name)
