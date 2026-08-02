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

"""Tool call extraction and response sanitization with native acceleration."""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from victor.processing.native._base import _NATIVE_AVAILABLE, _native
from victor.processing.native.observability import dispatch_with_observability
from victor.processing.native.streaming import strip_thinking_tokens

# =============================================================================
# TOOL CALL EXTRACTION
# =============================================================================

# Pre-compiled patterns for Python fallback
_FILE_PATH_PATTERNS = [
    re.compile(
        r"(?:to|file|path|in|create|write|save|modify|update|edit)\s+"
        r"[`'\"]?([a-zA-Z0-9_./-]+\.[a-zA-Z]{1,10})[`'\"]?",
        re.IGNORECASE,
    ),
    re.compile(
        r"^[`'\"]?([a-zA-Z0-9_./-]+\.[a-zA-Z]{1,10})[`'\"]?\s+(?:with|should|will)",
        re.MULTILINE,
    ),
    re.compile(
        r"(?:the|this)\s+file\s+[`'\"]?([a-zA-Z0-9_./-]+\.[a-zA-Z]{1,10})[`'\"]?",
        re.IGNORECASE,
    ),
    re.compile(r"`([a-zA-Z0-9_./-]+\.[a-zA-Z]{1,10})`"),
]

_CODE_BLOCK_PATTERN = re.compile(
    r"```(?:python|py|javascript|js|typescript|ts|bash|sh|json|yaml|yml|toml|"
    r"html|css|markdown|md|sql|go|rust|java|c|cpp|ruby|php)?\s*\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)

_INDENTED_CODE_PATTERN = re.compile(
    r"(?:^|\n)((?:[ ]{4,}|\t).+(?:\n(?:[ ]{4,}|\t).+)*)",
    re.MULTILINE,
)

_SHELL_COMMAND_PATTERNS = [
    re.compile(r"```(?:bash|sh|shell|zsh)?\s*\n(.+?)```", re.DOTALL | re.IGNORECASE),
    re.compile(r"(?:run|execute|command):\s*[`'\"](.+?)[`'\"]", re.IGNORECASE),
    re.compile(r"(?:^|\n)\$\s+(.+?)(?:\n|$)"),
]


def extract_file_path(text: str) -> Optional[str]:
    """Extract a file path from text.

    Uses native Rust implementation when available for ~3x speedup.

    Args:
        text: Text to search for file paths

    Returns:
        Extracted file path or None
    """
    if _NATIVE_AVAILABLE and hasattr(_native, "extract_file_path"):
        return _native.extract_file_path(text)

    # Pure Python fallback
    for pattern in _FILE_PATH_PATTERNS:
        match = pattern.search(text)
        if match:
            path = match.group(1)
            if "/" in path or "." in path:
                return path
    return None


def extract_code_blocks(text: str) -> List[str]:
    """Extract code blocks from text (fenced and indented).

    Uses native Rust implementation when available for ~3x speedup.

    Args:
        text: Text containing code blocks

    Returns:
        List of extracted code block contents
    """
    if _NATIVE_AVAILABLE and hasattr(_native, "extract_code_blocks"):
        return _native.extract_code_blocks(text)

    # Pure Python fallback
    blocks = []

    # Fenced code blocks
    for match in _CODE_BLOCK_PATTERN.finditer(text):
        blocks.append(match.group(1).strip())

    # Indented code blocks (if no fenced blocks found)
    if not blocks:
        for match in _INDENTED_CODE_PATTERN.finditer(text):
            block = match.group(1)
            lines = block.split("\n")
            non_empty = [line for line in lines if line.strip()]
            if non_empty:
                min_indent = min(len(line) - len(line.lstrip()) for line in non_empty)
                dedented = "\n".join(
                    line[min_indent:] if len(line) > min_indent else line for line in lines
                )
                blocks.append(dedented.strip())

    return blocks


def extract_shell_commands(text: str) -> List[str]:
    """Extract shell commands from text.

    Uses native Rust implementation when available for ~3x speedup.

    Args:
        text: Text containing shell commands

    Returns:
        List of extracted shell commands
    """
    if _NATIVE_AVAILABLE and hasattr(_native, "extract_shell_commands"):
        return _native.extract_shell_commands(text)

    # Pure Python fallback
    commands = []
    for pattern in _SHELL_COMMAND_PATTERNS:
        for match in pattern.finditer(text):
            cmd = match.group(1).strip()
            if cmd:
                commands.append(cmd)
    return commands


@dataclass
class ExtractedToolCallResult:
    """Result of tool call extraction."""

    tool_name: str
    arguments: Dict[str, Any]
    confidence: float
    source_text: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool": self.tool_name,
            "args": self.arguments,
            "confidence": self.confidence,
        }


def extract_tool_call(
    text: str,
    tool_name: str,
    current_file: Optional[str] = None,
) -> Optional[ExtractedToolCallResult]:
    """Extract a tool call from text for a specific tool.

    Uses native Rust implementation when available for ~3x speedup.

    Args:
        text: Text to extract from
        tool_name: Tool name to extract for
        current_file: Optional current file context

    Returns:
        ExtractedToolCallResult or None if extraction fails
    """
    if _NATIVE_AVAILABLE and hasattr(_native, "extract_tool_call"):
        result = _native.extract_tool_call(text, tool_name, current_file)
        if result:
            return ExtractedToolCallResult(
                tool_name=result["tool"],
                arguments=result["args"],
                confidence=result["confidence"],
                source_text=result.get("source", text[:200]),
            )
        return None

    # Pure Python fallback
    tool_lower = tool_name.lower()

    if tool_lower in ("write", "write_file"):
        file_path = extract_file_path(text) or current_file
        if not file_path:
            return None
        blocks = extract_code_blocks(text)
        if not blocks:
            return None
        content = blocks[0]
        confidence = 0.85
        if file_path.endswith(".py") and ("def " in content or "class " in content):
            confidence = 0.95
        return ExtractedToolCallResult(
            tool_name="write",
            arguments={"path": file_path, "content": content},
            confidence=confidence,
            source_text=text[:200],
        )

    elif tool_lower in ("read", "read_file"):
        file_path = extract_file_path(text)
        if not file_path:
            return None
        return ExtractedToolCallResult(
            tool_name="read",
            arguments={"path": file_path},
            confidence=0.9,
            source_text=text[:100],
        )

    elif tool_lower in ("shell", "bash", "execute", "run"):
        commands = extract_shell_commands(text)
        if not commands:
            return None
        return ExtractedToolCallResult(
            tool_name="shell",
            arguments={"command": commands[0]},
            confidence=0.75,
            source_text=text[:150],
        )

    elif tool_lower in ("ls", "list"):
        # Try backtick-wrapped paths first
        backtick_match = re.search(r"`([a-zA-Z0-9_./-]+)`", text)
        path = backtick_match.group(1) if backtick_match else "."
        return ExtractedToolCallResult(
            tool_name="ls",
            arguments={"path": path},
            confidence=0.8,
            source_text=text[:100],
        )

    return None


def batch_extract_file_paths(texts: List[str]) -> List[Optional[str]]:
    """Extract file paths from multiple texts.

    Uses native Rust implementation when available for ~5x speedup.

    Args:
        texts: List of texts to search

    Returns:
        List of extracted paths (None for texts without paths)
    """
    if _NATIVE_AVAILABLE and hasattr(_native, "batch_extract_file_paths"):
        return _native.batch_extract_file_paths(texts)

    return [extract_file_path(text) for text in texts]


# =============================================================================
# RESPONSE SANITIZATION
# =============================================================================

# Pre-compiled patterns for Python fallback
_LEAKAGE_PATTERNS = [
    re.compile(r"Do not invent any new or additional parameters.*", re.IGNORECASE),
    re.compile(r"The parameter value should be passed as a string.*", re.IGNORECASE),
    re.compile(r"If you want to call multiple functions.*", re.IGNORECASE),
    re.compile(r"Do NOT surround the function call.*", re.IGNORECASE),
    re.compile(r"All parameters are required unless.*", re.IGNORECASE),
    re.compile(r"The agent is not allowed to directly access.*", re.IGNORECASE),
    re.compile(r"Begin by calling list_directory.*", re.IGNORECASE),
]

_GARBAGE_PATTERNS = [
    re.compile(r"FUNCTION_CALL\s*\{"),
    re.compile(r"</function>\s*</function>"),
    re.compile(r"<parameter[^>]*>"),
    re.compile(r'^\s*\{\s*"name":\s*"[^"]+",\s*"arguments":', re.MULTILINE),
    re.compile(r"^\s*<IMPORTANT>", re.MULTILINE),
    re.compile(r"^\s*Do NOT", re.MULTILINE),
    re.compile(r"^\s*NEVER\s+", re.MULTILINE),
    re.compile(r"\[TOOL_REQUEST\]"),
]

_CLEANUP_PATTERNS = [
    (re.compile(r"(</\w+>\s*){3,}"), ""),  # Repeated closing tags
    (re.compile(r"</?function[^>]*>"), ""),  # Function tags
    (re.compile(r"</?parameter[^>]*>"), ""),  # Parameter tags
    (re.compile(r"</?tool[^>]*>"), ""),  # Tool tags
    (re.compile(r"</?IMPORTANT[^>]*>"), ""),  # Important tags
    (
        re.compile(r'\{"name":\s*"[^"]+",\s*"arguments":\s*\{[^}]*\}\}'),
        "",
    ),  # JSON tool calls
    (re.compile(r"\n{4,}"), "\n\n\n"),  # Excessive newlines
]


@dispatch_with_observability("sanitize_response")
def sanitize_response_fast(text: str) -> str:
    """Sanitize model response by removing malformed patterns.

    Uses native Rust implementation when available for ~3x speedup.

    Args:
        text: Raw response text from the model

    Returns:
        Cleaned text suitable for display
    """
    if _NATIVE_AVAILABLE and hasattr(_native, "sanitize_response"):
        return _native.sanitize_response(text)

    if not text:
        return text

    # Apply cleanup patterns
    for pattern, replacement in _CLEANUP_PATTERNS:
        text = pattern.sub(replacement, text)

    # Strip thinking tokens
    text = strip_thinking_tokens(text)

    # Remove leakage patterns
    for pattern in _LEAKAGE_PATTERNS:
        text = pattern.sub("", text)

    # Remove lines that are just tool call syntax
    lines = text.split("\n")
    cleaned_lines = [
        line
        for line in lines
        if not (line.strip().startswith('{"name":') or line.strip().startswith("</"))
    ]
    text = "\n".join(cleaned_lines)

    return text.strip()


@dispatch_with_observability("is_garbage_content")
def is_garbage_content_fast(content: str) -> bool:
    """Detect if content is garbage/malformed output.

    Uses native Rust implementation when available for ~3x speedup.

    Args:
        content: Content to check

    Returns:
        True if content appears to be garbage
    """
    if _NATIVE_AVAILABLE and hasattr(_native, "is_garbage_content"):
        return _native.is_garbage_content(content)

    if not content:
        return False

    for pattern in _GARBAGE_PATTERNS:
        if pattern.search(content):
            return True
    return False


def detect_leakage_patterns(text: str) -> List[Tuple[int, int, str]]:
    """Detect training data leakage patterns in text.

    Uses native Rust implementation when available for ~3x speedup.

    Args:
        text: Text to check for leakage

    Returns:
        List of (start, end, pattern_name) tuples for matches
    """
    if _NATIVE_AVAILABLE and hasattr(_native, "detect_leakage_patterns"):
        return _native.detect_leakage_patterns(text)

    # Pure Python fallback
    matches = []
    pattern_names = [
        "no_new_params",
        "string_params",
        "multiple_funcs",
        "no_surround",
        "required_params",
        "no_direct_access",
        "begin_list_dir",
    ]

    for pattern, name in zip(_LEAKAGE_PATTERNS, pattern_names):
        for m in pattern.finditer(text):
            matches.append((m.start(), m.end(), name))

    matches.sort(key=lambda x: x[0])
    return matches


@dispatch_with_observability("strip_markup")
def strip_markup_fast(text: str) -> str:
    """Remove XML/HTML-like tags to salvage plain text.

    Uses native Rust implementation when available for ~3x speedup.

    Args:
        text: Text potentially containing markup

    Returns:
        Plain text with markup removed
    """
    if _NATIVE_AVAILABLE and hasattr(_native, "strip_markup"):
        return _native.strip_markup(text)

    if not text:
        return text

    cleaned = re.sub(r"<[^>]+>", " ", text)
    return " ".join(cleaned.split())


def validate_tool_name(name: str) -> Tuple[bool, Optional[str]]:
    """Validate a tool name is not a hallucination.

    Uses native Rust implementation when available for ~3x speedup.

    Args:
        name: Tool name to validate

    Returns:
        Tuple of (is_valid, rejection_reason)
    """
    if _NATIVE_AVAILABLE and hasattr(_native, "validate_tool_name"):
        return _native.validate_tool_name(name)

    if not name or not isinstance(name, str):
        return False, "empty_or_invalid_type"

    invalid_prefixes = [
        "example_",
        "func_",
        "function_",
        "tool_name",
        "my_",
        "test_tool",
        "sample_",
    ]
    for prefix in invalid_prefixes:
        if name.startswith(prefix):
            return False, f"invalid_prefix:{prefix}"

    if name.endswith("/") or name.endswith(">"):
        return False, "invalid_suffix"
    if name.startswith("<"):
        return False, "starts_with_tag"
    if " " in name or "\t" in name:
        return False, "contains_whitespace"
    if name[0].isdigit():
        return False, "starts_with_number"
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9_]*$", name):
        return False, "invalid_characters"

    return True, None
