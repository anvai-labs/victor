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

"""Tool call extraction from model text for hallucinated tool call recovery.

When models mention tools in text but don't execute proper tool calls,
this module attempts to parse the text and extract the intended tool call
with arguments.

Supported tool patterns:
- write/edit: Extract file path and content from code blocks
- read: Extract file path from text
- shell/bash: Extract command from code blocks
- grep/search: Extract pattern and path from text

Example:
    extractor = ToolCallExtractor()
    result = extractor.extract_from_text(
        "I'll write to hello.py:\n```python\nprint('hello')\n```",
        mentioned_tools=["write"]
    )
    # Returns: {"tool": "write", "args": {"path": "hello.py", "content": "print('hello')"}}
"""

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from victor.tools.core_tool_aliases import canonicalize_core_tool_name

logger = logging.getLogger(__name__)


@dataclass
class ExtractedToolCall:
    """Represents a tool call extracted from model text."""

    tool_name: str
    arguments: Dict[str, Any]
    confidence: float  # 0.0-1.0, how confident we are in the extraction
    source_text: str  # The text segment that was parsed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool": self.tool_name,
            "args": self.arguments,
            "confidence": self.confidence,
        }


class ToolCallExtractor:
    """Extracts intended tool calls from model text.

    This handles the "hallucinated tool call" case where models describe
    what they would do but don't actually execute the tool call.
    """

    # Patterns for detecting file paths
    FILE_PATH_PATTERNS = [
        # Explicit path mentions
        r"(?:to|file|path|in|create|write|save|modify|update|edit|read|open|view|check)\s+(?:the\s+)?[`'\"]?([a-zA-Z0-9_./-]+\.[a-zA-Z]{1,10})[`'\"]?",
        # Path followed by "file"
        r"[`'\"]?([a-zA-Z0-9_./-]+\.[a-zA-Z]{1,10})[`'\"]?\s+file\b",
        # Path at start of sentence with action
        r"^[`'\"]?([a-zA-Z0-9_./-]+\.[a-zA-Z]{1,10})[`'\"]?\s+(?:with|should|will)",
        # After "the file" or "this file"
        r"(?:the|this)\s+file\s+[`'\"]?([a-zA-Z0-9_./-]+\.[a-zA-Z]{1,10})[`'\"]?",
    ]

    # Patterns for code blocks
    CODE_BLOCK_PATTERN = re.compile(
        r"```(?:python|py|javascript|js|typescript|ts|bash|sh|json|yaml|yml|toml|"
        r"html|css|markdown|md|sql|go|rust|java|c|cpp|ruby|php)?\s*\n(.*?)```",
        re.DOTALL | re.IGNORECASE,
    )

    # Alternative: indented code blocks (4+ spaces)
    INDENTED_CODE_PATTERN = re.compile(
        r"(?:^|\n)((?:[ ]{4,}|\t).+(?:\n(?:[ ]{4,}|\t).+)*)",
        re.MULTILINE,
    )

    # Shell command patterns
    SHELL_COMMAND_PATTERNS = [
        r"```(?:bash|sh|shell|zsh)?\s*\n(.+?)```",
        r"(?:run|execute|command):\s*[`'\"](.+?)[`'\"]",
        r"(?:^|\n)\$\s+(.+?)(?:\n|$)",
    ]

    # Grep/search patterns
    SEARCH_PATTERNS = [
        r"(?:search|grep|find|look)\s+(?:for\s+)?[`'\"](.+?)[`'\"]",
        r"pattern\s+[`'\"](.+?)[`'\"]",
    ]

    def __init__(self, tool_definitions: Optional[Dict[str, Any]] = None):
        """Initialize the extractor.

        Args:
            tool_definitions: Optional dict of tool name -> tool definition
                             for validation and parameter extraction
        """
        self._tool_definitions = tool_definitions or {}

    def extract_from_text(
        self,
        text: str,
        mentioned_tools: List[str],
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[ExtractedToolCall]:
        """Extract a tool call from model text.

        Args:
            text: The model's response text
            mentioned_tools: List of tool names detected as mentioned
            context: Optional context (e.g., current file being discussed)

        Returns:
            ExtractedToolCall if successful, None otherwise
        """
        if not mentioned_tools:
            return None

        # P0 FIX: Prioritize tools and improve extraction with lower confidence threshold
        # Priority order: ls > read > write/edit > others
        tool_priority = {
            "ls": 1,
            "list": 1,
            "read": 2,
            "write": 3,
            "edit": 3,
            "graph": 4,
            "code_search": 5,
            "grep": 6,
            "shell": 7,
        }

        # Sort mentioned tools by priority
        prioritized_tools = sorted(
            mentioned_tools,
            key=lambda t: tool_priority.get(canonicalize_core_tool_name(t).lower(), 99),
        )

        # Try extraction for each mentioned tool with lower threshold
        for tool_name in prioritized_tools:
            result = self._extract_for_tool(text, canonicalize_core_tool_name(tool_name), context)
            # P0 FIX: Lower confidence threshold from 0.5 to 0.3 for recovery attempts
            if result and result.confidence >= 0.3:
                logger.info(
                    f"[ToolCallExtractor] Extracted {tool_name} call with "
                    f"confidence {result.confidence:.2f}: {list(result.arguments.keys())}"
                )
                return result

        logger.debug(
            f"[ToolCallExtractor] Could not extract tool call from text. "
            f"Mentioned tools: {mentioned_tools}"
        )
        return None

    def _extract_for_tool(
        self,
        text: str,
        tool_name: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[ExtractedToolCall]:
        """Extract arguments for a specific tool.

        Args:
            text: Model response text
            tool_name: The tool to extract for
            context: Optional context

        Returns:
            ExtractedToolCall or None
        """
        tool_lower = canonicalize_core_tool_name(tool_name).lower()

        # Route to appropriate extractor
        if tool_lower == "write":
            return self._extract_write_call(text, context)
        elif tool_lower == "edit":
            return self._extract_edit_call(text, context)
        elif tool_lower == "read":
            return self._extract_read_call(text, context)
        elif tool_lower == "shell":
            return self._extract_shell_call(text, context)
        elif tool_lower in ("grep", "search", "find"):
            return self._extract_search_call(text, context)
        elif tool_lower in ("ls", "list"):
            return self._extract_ls_call(text, context)
        elif tool_lower in ("graph", "graph_search", "query_graph"):
            return self._extract_graph_call(text, context)

        # Generic extraction attempt
        return self._extract_generic_call(text, tool_name, context)

    def _extract_write_call(
        self, text: str, context: Optional[Dict[str, Any]] = None
    ) -> Optional[ExtractedToolCall]:
        """Extract write/create file call."""
        # Find file path
        file_path = self._extract_file_path(text)
        if not file_path:
            # Try context
            if context and "current_file" in context:
                file_path = context["current_file"]

        if not file_path:
            return None

        # Find code content
        content = self._extract_code_content(text)
        if not content:
            return None

        # Calculate confidence
        confidence = 0.6
        if file_path and content:
            confidence = 0.85
            # Boost if path matches code type
            if file_path.endswith(".py") and ("def " in content or "class " in content):
                confidence = 0.95
            elif file_path.endswith(".js") and ("function" in content or "const " in content):
                confidence = 0.95

        return ExtractedToolCall(
            tool_name="write",
            arguments={"path": file_path, "content": content},
            confidence=confidence,
            source_text=text[:200],
        )

    def _extract_edit_call(
        self, text: str, context: Optional[Dict[str, Any]] = None
    ) -> Optional[ExtractedToolCall]:
        """Extract edit file call.

        Looks for old_string/new_string patterns or diff-like content.
        """
        file_path = self._extract_file_path(text)
        if not file_path and context and "current_file" in context:
            file_path = context["current_file"]

        if not file_path:
            return None

        # Try to find old/new content patterns
        old_content, new_content = self._extract_edit_diff(text)

        if old_content and new_content:
            return ExtractedToolCall(
                tool_name="edit",
                arguments={
                    "path": file_path,
                    "old_str": old_content,
                    "new_str": new_content,
                },
                confidence=0.8,
                source_text=text[:200],
            )

        # Fall back to full content replacement
        content = self._extract_code_content(text)
        if content:
            return ExtractedToolCall(
                tool_name="write",  # Use write for full replacement
                arguments={"path": file_path, "content": content},
                confidence=0.7,
                source_text=text[:200],
            )

        return None

    def _extract_read_call(
        self, text: str, context: Optional[Dict[str, Any]] = None
    ) -> Optional[ExtractedToolCall]:
        """Extract read file call."""
        file_path = self._extract_file_path(text)
        if not file_path:
            return None

        return ExtractedToolCall(
            tool_name="read",
            arguments={"path": file_path},
            confidence=0.9,
            source_text=text[:100],
        )

    def _extract_shell_call(
        self, text: str, context: Optional[Dict[str, Any]] = None
    ) -> Optional[ExtractedToolCall]:
        """Extract shell/bash command call."""
        command = None

        # Try code block first
        for pattern in self.SHELL_COMMAND_PATTERNS:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                command = match.group(1).strip()
                break

        if not command:
            # Try to find command after common phrases
            cmd_patterns = [
                r"(?:run|execute|command|terminal):\s*(.+?)(?:\n|$)",
                r"```\s*\n(.+?)\n```",
            ]
            for pattern in cmd_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    command = match.group(1).strip()
                    break

        if not command:
            return None

        # Safety check - don't execute dangerous commands
        dangerous_patterns = [
            r"\brm\s+-rf\s+/",
            r"\bsudo\s+rm\b",
            r"\bdd\s+if=",
            r"\bmkfs\b",
            r"\b:(){ :|:& };:\b",
        ]
        for pattern in dangerous_patterns:
            if re.search(pattern, command, re.IGNORECASE):
                logger.warning(f"[ToolCallExtractor] Blocked dangerous command: {command[:50]}")
                return None

        return ExtractedToolCall(
            tool_name="shell",
            arguments={"cmd": command},
            confidence=0.75,
            source_text=text[:150],
        )

    def _extract_search_call(
        self, text: str, context: Optional[Dict[str, Any]] = None
    ) -> Optional[ExtractedToolCall]:
        """Extract grep/search call."""
        pattern = None
        path = "."

        for search_pattern in self.SEARCH_PATTERNS:
            match = re.search(search_pattern, text, re.IGNORECASE)
            if match:
                pattern = match.group(1)
                break

        if not pattern:
            return None

        # Try to find path
        file_path = self._extract_file_path(text)
        if file_path:
            path = file_path

        return ExtractedToolCall(
            tool_name="grep",
            arguments={"query": pattern, "path": path},
            confidence=0.7,
            source_text=text[:100],
        )

    def _extract_ls_call(
        self, text: str, context: Optional[Dict[str, Any]] = None
    ) -> Optional[ExtractedToolCall]:
        """Extract ls/list directory call.

        P0 FIX: Improved patterns for common hallucinated mentions like:
        - "let me explore victor/agent"
        - "I'll check the directory structure"
        - "show me what's in src/"
        """
        # Find directory path - try backtick-wrapped first
        backtick_match = re.search(r"`([a-zA-Z0-9_./-]+)`", text)
        if backtick_match:
            path = backtick_match.group(1)
            # Validate it looks like a directory (has / or is a common dir name)
            if "/" in path or path in ("src", "lib", "tests", "examples", "."):
                return ExtractedToolCall(
                    tool_name="ls",
                    arguments={"path": path},
                    confidence=0.9,  # Increased confidence for backtick-wrapped paths
                    source_text=text[:100],
                )

        # Try directory patterns - improved for hallucinated mentions
        dir_patterns = [
            # "show me what's in src/" pattern
            r"(?:show|tell)\s+(?:me\s+)?what'?s?\s+in\s+(?:the\s+)?[`'\"]?([a-zA-Z0-9_./-]+/?)[`'\"]?\s*(?:directory|folder)?",
            # "explore victor/agent" pattern (common with hallucinated calls)
            r"(?:explore|list|check|inspect|examine)\s+(?:the\s+)?[`'\"]?([a-zA-Z0-9_./-]+/[a-zA-Z0-9_./-]*)[`'\"]?",
            # "list the directory PATH"
            r"(?:list|ls)\s+(?:the\s+)?(?:directory\s+)?[`'\"]?([a-zA-Z0-9_./-]+/[a-zA-Z0-9_./-]*)[`'\"]?",
            # "what's in X"
            r"what'?s?\s+in\s+(?:the\s+)?(?:directory\s+)?[`'\"]?([a-zA-Z0-9_./-]+/?)[`'\"]?",
            r"(?:in|of)\s+[`'\"]?([a-zA-Z0-9_./-]+)[`'\"]?\s+(?:directory|folder)",
            r"(?:directory|folder)\s+[`'\"]?([a-zA-Z0-9_./-]+)[`'\"]?",
        ]

        path = "."
        confidence = 0.6  # Default confidence
        for pattern in dir_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                candidate = match.group(1)
                # Filter out common words and short matches
                skip_words = (
                    "the",
                    "a",
                    "an",
                    "this",
                    "that",
                    "to",
                    "and",
                    "or",
                    "me",
                    "in",
                    "of",
                    "for",
                    "what",
                    "whats",
                    "directory",
                    "folder",
                )
                if candidate.lower() not in skip_words and len(candidate) > 2:
                    path = candidate
                    # Boost confidence for multi-segment paths (likely real directories)
                    if "/" in candidate and len(candidate.split("/")) >= 2:
                        confidence = 0.85
                    break

        return ExtractedToolCall(
            tool_name="ls",
            arguments={"path": path},
            confidence=confidence,
            source_text=text[:100],
        )

    def _extract_graph_call(
        self, text: str, context: Optional[Dict[str, Any]] = None
    ) -> Optional[ExtractedToolCall]:
        """Extract graph query call."""
        # Try to find graph query patterns
        patterns = [
            # graph: "query" or graph("query")
            r'(?:graph|query_graph|graph_search)[\s:]*["\'\(]([^"\'\)]+)["\'\)]',
            # "search the graph for X"
            r'(?:search|find|show|query)\s+the\s+graph\s+(?:for|about|with)?\s+["\']?([^"\']+)["\']?',
            # "in the graph: X"
            r'in\s+the\s+graph[:\s]+["\']?([^"\']+)["\']?',
        ]

        query = None
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                query = match.group(1).strip()
                break

        # If no specific pattern matched, extract context around "graph" mention
        if not query and "graph" in text.lower():
            # Extract context around "graph" mention
            idx = text.lower().find("graph")
            start = max(0, idx - 50)
            end = min(len(text), idx + 100)
            query = text[start:end].strip()

            # Clean up common phrases
            query = re.sub(
                r"^(?:let me|I will|I\'ll|I\'d|can|could|please|can you|could you)\s+(?:check|see|query|search|look at|examine|explore|find|show)\s+(the\s+)?graph\s+(for|about)?\s*",
                "",
                query,
                flags=re.IGNORECASE,
            )
            query = query.strip()

        if query:
            # Determine graph mode based on keywords
            mode = "find"  # default
            if any(
                word in query.lower()
                for word in [
                    "neighbor",
                    "connected",
                    "related",
                    "calls",
                    "callee",
                    "caller",
                ]
            ):
                mode = "neighbors"
            elif any(word in query.lower() for word in ["path", "between", "connection"]):
                mode = "path"
            elif any(word in query.lower() for word in ["important", "rank", "central", "hub"]):
                mode = "pagerank"
            elif any(word in query.lower() for word in ["cluster", "group", "couple"]):
                mode = "clusters"

            return ExtractedToolCall(
                tool_name="graph",
                arguments={
                    "mode": mode,
                    "query": query[:200],  # Limit query length
                },
                confidence=0.8,  # Higher confidence for specific pattern match
                source_text=text[:200],
            )

        # Fallback: generic graph call without query
        return None

    def _extract_generic_call(
        self,
        text: str,
        tool_name: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[ExtractedToolCall]:
        """Generic extraction for unknown tools."""
        # Try to find any arguments in parentheses after tool name
        pattern = rf"{re.escape(tool_name)}\s*\(([^)]+)\)"
        match = re.search(pattern, text, re.IGNORECASE)

        if match:
            args_str = match.group(1)
            # Try to parse as key=value pairs
            args = self._parse_args_string(args_str)
            if args:
                return ExtractedToolCall(
                    tool_name=tool_name,
                    arguments=args,
                    confidence=0.5,
                    source_text=text[:100],
                )

        return None

    def _extract_file_path(self, text: str) -> Optional[str]:
        """Extract a file path from text."""
        for pattern in self.FILE_PATH_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                path = match.group(1)
                # Validate it looks like a path
                if "/" in path or "." in path:
                    return path

        # Also try backtick-wrapped paths
        backtick_match = re.search(r"`([a-zA-Z0-9_./-]+\.[a-zA-Z]{1,10})`", text)
        if backtick_match:
            return backtick_match.group(1)

        return None

    def _extract_code_content(self, text: str) -> Optional[str]:
        """Extract code content from text."""
        # Try fenced code blocks first
        match = self.CODE_BLOCK_PATTERN.search(text)
        if match:
            return match.group(1).strip()

        # Try indented code blocks
        match = self.INDENTED_CODE_PATTERN.search(text)
        if match:
            # Remove common leading whitespace
            lines = match.group(1).split("\n")
            min_indent = min(len(line) - len(line.lstrip()) for line in lines if line.strip())
            dedented = "\n".join(
                line[min_indent:] if len(line) > min_indent else line for line in lines
            )
            return dedented.strip()

        return None

    def _extract_edit_diff(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract old/new content for edit operations.

        Looks for patterns like:
        - "Replace X with Y"
        - "Change X to Y"
        - Old: ... New: ...
        """
        # Pattern: "replace X with Y" or "change X to Y"
        replace_patterns = [
            r"(?:replace|change)\s+[`'\"](.+?)[`'\"]\s+(?:with|to)\s+[`'\"](.+?)[`'\"]",
            r"(?:from|old):\s*[`'\"](.+?)[`'\"]\s*(?:to|new):\s*[`'\"](.+?)[`'\"]",
        ]

        for pattern in replace_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1), match.group(2)

        # Try to find two consecutive code blocks
        blocks = self.CODE_BLOCK_PATTERN.findall(text)
        if len(blocks) >= 2:
            # Check if text suggests old/new relationship
            if re.search(r"(?:before|old|current|existing)", text[:200], re.IGNORECASE):
                return blocks[0].strip(), blocks[1].strip()

        return None, None

    def _parse_args_string(self, args_str: str) -> Dict[str, Any]:
        """Parse a string of arguments like 'path="foo", content="bar"'."""
        args = {}
        # Match key=value or key="value" patterns
        pattern = r'(\w+)\s*=\s*(?:"([^"]*?)"|\'([^\']*?)\'|(\S+))'
        for match in re.finditer(pattern, args_str):
            key = match.group(1)
            value = match.group(2) or match.group(3) or match.group(4)
            args[key] = value
        return args


# Module-level singleton
_extractor: Optional[ToolCallExtractor] = None


def get_tool_call_extractor() -> ToolCallExtractor:
    """Get the global ToolCallExtractor instance."""
    global _extractor
    if _extractor is None:
        _extractor = ToolCallExtractor()
    return _extractor


def extract_tool_call_from_text(
    text: str,
    mentioned_tools: List[str],
    context: Optional[Dict[str, Any]] = None,
) -> Optional[ExtractedToolCall]:
    """Convenience function to extract tool call from text.

    Args:
        text: Model response text
        mentioned_tools: List of mentioned tool names
        context: Optional context dict

    Returns:
        ExtractedToolCall if successful, None otherwise
    """
    return get_tool_call_extractor().extract_from_text(text, mentioned_tools, context)
