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

"""Output utilities for token-efficient tool responses.

This module provides shared utilities for filtering, compressing, and formatting
tool outputs to minimize token usage while preserving essential information.

Design Principles:
1. Backwards Compatibility: All new parameters have sensible defaults
2. Consistent API: Similar filtering patterns across all tools
3. Composability: Filters can be combined (grep + context lines)
4. Extensibility: Easy to add new output modes

Usage:
    from victor.tools.output_utils import (
        grep_lines,
        truncate_output,
        truncate_by_lines,
        TruncationInfo,
        format_with_line_numbers,
        OutputMode,
    )
"""

import fnmatch
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


class OutputMode(str, Enum):
    """Output mode for tool responses.

    FULL: Return complete output (default, backwards compatible)
    SUMMARY: Return counts/stats only
    COMPACT: Return compressed format (e.g., paths only)
    MATCHES: Return only matching lines with context
    """

    FULL = "full"
    SUMMARY = "summary"
    COMPACT = "compact"
    MATCHES = "matches"


@dataclass
class LineMatch:
    """Represents a matched line with context."""

    line_number: int
    content: str
    context_before: List[str] = field(default_factory=list)
    context_after: List[str] = field(default_factory=list)
    match_positions: List[Tuple[int, int]] = field(default_factory=list)


@dataclass
class GrepResult:
    """Result of grep operation on content."""

    matches: List[LineMatch]
    total_lines: int
    pattern: str
    file_path: Optional[str] = None

    @property
    def match_count(self) -> int:
        return len(self.matches)

    def to_string(self, show_line_numbers: bool = True, max_matches: int = 50) -> str:
        """Format matches as string output."""
        if not self.matches:
            return f"No matches found for pattern: {self.pattern}"

        lines = []
        header = f"[{self.match_count} matches"
        if self.file_path:
            header += f" in {self.file_path}"
        header += f" (total lines: {self.total_lines})]"
        lines.append(header)

        for i, match in enumerate(self.matches[:max_matches]):
            if match.context_before:
                for ctx_line in match.context_before:
                    if show_line_numbers:
                        lines.append(f"  {ctx_line}")
                    else:
                        lines.append(f"  {ctx_line}")

            prefix = f"{match.line_number:>5}: " if show_line_numbers else ""
            lines.append(f"{prefix}{match.content}")

            if match.context_after:
                for ctx_line in match.context_after:
                    if show_line_numbers:
                        lines.append(f"  {ctx_line}")
                    else:
                        lines.append(f"  {ctx_line}")

            if i < len(self.matches) - 1 and (match.context_before or match.context_after):
                lines.append("---")

        if len(self.matches) > max_matches:
            lines.append(f"... and {len(self.matches) - max_matches} more matches")

        return "\n".join(lines)


def grep_lines(
    content: str,
    pattern: str,
    context_before: int = 0,
    context_after: int = 0,
    case_sensitive: bool = True,
    is_regex: bool = False,
    max_matches: int = 100,
    file_path: Optional[str] = None,
) -> GrepResult:
    """Search for pattern in content and return matching lines with context.

    Args:
        content: The text content to search
        pattern: Search pattern (string or regex)
        context_before: Number of lines to include before each match
        context_after: Number of lines to include after each match
        case_sensitive: Whether search is case-sensitive
        is_regex: Whether pattern is a regex
        max_matches: Maximum number of matches to return
        file_path: Optional file path for metadata

    Returns:
        GrepResult with matching lines and context
    """
    lines = content.split("\n")
    total_lines = len(lines)
    matches: List[LineMatch] = []

    # Compile pattern
    if is_regex:
        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            regex = re.compile(pattern, flags)
        except re.error:
            # Invalid regex - treat as literal string
            regex = re.compile(re.escape(pattern), flags)
    else:
        flags = 0 if case_sensitive else re.IGNORECASE
        regex = re.compile(re.escape(pattern), flags)

    for i, line in enumerate(lines):
        if len(matches) >= max_matches:
            break

        match_iter = list(regex.finditer(line))
        if match_iter:
            # Get context lines
            ctx_before = []
            if context_before > 0:
                start = max(0, i - context_before)
                ctx_before = [f"{j+1}: {lines[j]}" for j in range(start, i)]

            ctx_after = []
            if context_after > 0:
                end = min(len(lines), i + context_after + 1)
                ctx_after = [f"{j+1}: {lines[j]}" for j in range(i + 1, end)]

            # Extract match positions
            positions = [(m.start(), m.end()) for m in match_iter]

            matches.append(
                LineMatch(
                    line_number=i + 1,
                    content=line,
                    context_before=ctx_before,
                    context_after=ctx_after,
                    match_positions=positions,
                )
            )

    return GrepResult(
        matches=matches,
        total_lines=total_lines,
        pattern=pattern,
        file_path=file_path,
    )


def filter_paths(
    paths: List[str],
    include_pattern: Optional[str] = None,
    exclude_pattern: Optional[str] = None,
    extensions: Optional[List[str]] = None,
) -> List[str]:
    """Filter paths by glob patterns and extensions.

    Args:
        paths: List of file paths to filter
        include_pattern: Glob pattern to include (e.g., "*.py", "src/**/*.ts")
        exclude_pattern: Glob pattern to exclude
        extensions: List of extensions to include (e.g., [".py", ".ts"])

    Returns:
        Filtered list of paths
    """
    result = paths

    if include_pattern:
        result = [p for p in result if fnmatch.fnmatch(p, include_pattern)]

    if exclude_pattern:
        result = [p for p in result if not fnmatch.fnmatch(p, exclude_pattern)]

    if extensions:
        ext_set = {ext if ext.startswith(".") else f".{ext}" for ext in extensions}
        result = [p for p in result if Path(p).suffix in ext_set]

    return result


def truncate_output(
    content: str,
    max_tokens: int = 4000,
    chars_per_token: float = 4.0,
    truncation_message: str = "\n[... output truncated ...]",
) -> str:
    """Truncate output to approximate token limit (legacy function).

    Note: Consider using truncate_by_lines() for cleaner line-based truncation.

    Args:
        content: Content to truncate
        max_tokens: Maximum tokens to allow
        chars_per_token: Average characters per token (conservative estimate)
        truncation_message: Message to append when truncating

    Returns:
        Truncated content with message if needed
    """
    max_chars = int(max_tokens * chars_per_token)

    if len(content) <= max_chars:
        return content

    # Truncate at a line boundary if possible
    truncated = content[:max_chars]
    last_newline = truncated.rfind("\n")
    if last_newline > max_chars * 0.8:  # Only use line boundary if not too far back
        truncated = truncated[:last_newline]

    return truncated + truncation_message


def truncate_by_lines(
    content: str,
    max_lines: int = 750,
    max_bytes: Optional[int] = None,
    start_line: int = 0,
) -> Tuple[str, "TruncationInfo"]:
    """Truncate content by line count and/or byte size.

    This function always truncates at complete line boundaries, never mid-line.
    It stops when either the line limit OR byte limit is reached (whichever first).

    Design rationale:
    - Line-based limits are more intuitive for LLMs and code
    - Character/byte limits are secondary safeguards
    - Always ending on line boundaries allows clean continuation requests

    Args:
        content: The text content to truncate
        max_lines: Maximum number of lines to include (default 512)
        max_bytes: Maximum bytes to include (None=use tool settings default)
        start_line: Starting line number (0-indexed, for offset support)

    Returns:
        Tuple of (truncated_content, TruncationInfo)

    Example:
        content = "line1\\nline2\\nline3\\n..."
        truncated, info = truncate_by_lines(content, max_lines=100)
        if info.was_truncated:
            print(f"Truncated at line {info.end_line}. Use offset={info.end_line} for more.")
    """
    # Default byte limit if not specified
    # Priority: use tool settings in unlimited mode, otherwise use 1MB safety default
    if max_bytes is None:
        if max_lines is None:
            # True unlimited mode - use tool settings (higher limit for accuracy-first)
            from victor.config.tool_settings import get_tool_settings

            tool_settings = get_tool_settings()
            # Convert MB to bytes
            max_bytes = int(tool_settings.tool_output_byte_limit_mb * 1024 * 1024)
        else:
            # Line limit is set, but no byte limit - use 1MB safety default
            max_bytes = 1024 * 1024  # 1MB safety default when line-limited

    lines = content.split("\n")
    total_lines = len(lines)

    # Apply offset if specified
    if start_line > 0:
        lines = lines[start_line:]

    result_lines: List[str] = []
    current_bytes = 0
    lines_included = 0

    for line in lines:
        line_bytes = len(line.encode("utf-8")) + 1  # +1 for newline

        # Check limits BEFORE adding the line
        if lines_included >= max_lines:
            break
        if current_bytes + line_bytes > max_bytes and result_lines:
            # Only enforce byte limit if we have at least one line
            break

        result_lines.append(line)
        current_bytes += line_bytes
        lines_included += 1

    # Build truncation info
    end_line = start_line + lines_included
    was_truncated = end_line < total_lines

    # Determine truncation reason
    truncation_reason: Optional[str] = None
    if was_truncated:
        if lines_included >= max_lines:
            truncation_reason = "line_limit"
        else:
            # Must be byte limit (we stopped before reaching line limit)
            truncation_reason = "byte_limit"

    info = TruncationInfo(
        was_truncated=was_truncated,
        total_lines=total_lines,
        lines_returned=lines_included,
        start_line=start_line,
        end_line=end_line,
        bytes_returned=current_bytes,
        truncation_reason=truncation_reason,
    )

    truncated_content = "\n".join(result_lines)

    # Add continuation hint if truncated
    if was_truncated:
        remaining = total_lines - end_line
        truncated_content += (
            f"\n\n[... {remaining} more lines. Use offset={end_line} to continue ...]"
        )

    return truncated_content, info


@dataclass
class TruncationInfo:
    """Information about how content was truncated."""

    was_truncated: bool
    """Whether the content was truncated."""

    total_lines: int
    """Total lines in the original content."""

    lines_returned: int
    """Number of lines included in the output."""

    start_line: int
    """Starting line number (0-indexed)."""

    end_line: int
    """Ending line number (exclusive, can be used as next offset)."""

    bytes_returned: int
    """Approximate bytes in the returned content."""

    truncation_reason: Optional[str] = None
    """Why truncation occurred: 'line_limit', 'byte_limit', or None."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for tool responses."""
        return {
            "was_truncated": self.was_truncated,
            "total_lines": self.total_lines,
            "lines_returned": self.lines_returned,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "bytes_returned": self.bytes_returned,
            "truncation_reason": self.truncation_reason,
        }


def format_with_line_numbers(
    content: str,
    start_line: int = 1,
) -> str:
    """Format content with line numbers.

    Args:
        content: Text content to number
        start_line: Starting line number (default 1)

    Returns:
        Content with line numbers prefixed
    """
    lines = content.split("\n")
    width = len(str(start_line + len(lines)))

    numbered = []
    for i, line in enumerate(lines):
        line_num = start_line + i
        numbered.append(f"{line_num:>{width}}\t{line}")

    return "\n".join(numbered)


def format_diff(
    original: str,
    modified: str,
    file_path: str = "file",
    context_lines: int = 3,
) -> str:
    """Generate unified diff between original and modified content.

    Args:
        original: Original content
        modified: Modified content
        file_path: File path for diff header
        context_lines: Number of context lines in diff

    Returns:
        Unified diff string
    """
    import difflib

    original_lines = original.splitlines(keepends=True)
    modified_lines = modified.splitlines(keepends=True)

    diff = difflib.unified_diff(
        original_lines,
        modified_lines,
        fromfile=f"a/{file_path}",
        tofile=f"b/{file_path}",
        n=context_lines,
    )

    return "".join(diff)


def summarize_changes(
    original: str,
    modified: str,
) -> Dict[str, Any]:
    """Generate summary statistics of changes between two contents.

    Args:
        original: Original content
        modified: Modified content

    Returns:
        Dictionary with change statistics
    """
    import difflib

    original_lines = original.splitlines()
    modified_lines = modified.splitlines()

    matcher = difflib.SequenceMatcher(None, original_lines, modified_lines)

    added = 0
    removed = 0
    changed = 0

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "insert":
            added += j2 - j1
        elif tag == "delete":
            removed += i2 - i1
        elif tag == "replace":
            changed += max(i2 - i1, j2 - j1)

    return {
        "lines_added": added,
        "lines_removed": removed,
        "lines_changed": changed,
        "original_lines": len(original_lines),
        "modified_lines": len(modified_lines),
        "similarity_ratio": matcher.ratio(),
    }


def compact_file_list(
    files: List[Dict[str, Any]],
    group_by_extension: bool = False,
    max_items: int = 100,
) -> Union[List[str], Dict[str, List[str]]]:
    """Convert detailed file list to compact format.

    Args:
        files: List of file dictionaries with 'name'/'path' and 'type' keys
        group_by_extension: Whether to group files by extension
        max_items: Maximum items to return

    Returns:
        List of file paths, or dict grouped by extension
    """
    paths = []
    for f in files[:max_items]:
        path = f.get("path") or f.get("name", "")
        if f.get("type") == "directory":
            path += "/"
        paths.append(path)

    if not group_by_extension:
        return paths

    # Group by extension
    grouped: Dict[str, List[str]] = {}
    for path in paths:
        if path.endswith("/"):
            ext = "[dirs]"
        else:
            ext = Path(path).suffix or "[no ext]"
        grouped.setdefault(ext, []).append(path)

    return grouped
