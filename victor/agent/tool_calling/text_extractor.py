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

"""Text-based tool call extraction for models that output function calls as text.

Many open-weight models hosted on OpenAI-compatible providers (Cerebras, Groq,
Together, etc.) sometimes output tool calls as Python-like function calls in text
instead of proper structured JSON tool_calls. For example:

    read(path='victor/agent/orchestrator.py')
    shell(cmd="ls -la")
    edit(path="/path/to/file", old_str="foo", new_str="bar")

This module provides extraction logic to parse these text-based calls into
structured tool call format that Victor can execute.

Design principles (SOLID):
- Single Responsibility: Only handles text-to-tool-call extraction
- Open/Closed: New patterns can be added via EXTRACTION_PATTERNS
- Dependency Inversion: Returns standard ToolCall objects

Usage:
    from victor.agent.tool_calling.text_extractor import PythonCallExtractor

    extractor = PythonCallExtractor()
    result = extractor.extract_from_text(
        "I'll read the file: read(path='foo.py')",
        valid_tool_names={"read", "write", "shell"}
    )
    if result.tool_calls:
        for tc in result.tool_calls:
            print(f"Tool: {tc.name}, Args: {tc.arguments}")
"""

import ast
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from victor.tools.core_tool_aliases import canonicalize_core_tool_name

logger = logging.getLogger(__name__)

_TOOL_INTENT_PREFIX_RE = re.compile(
    r"""
    (?:
        \b(?:let\ me|i(?:'| wi)?ll|use|call|invoke|run|execute|try|check|inspect)\b
        .*[:\-]\s*
    )$
    """,
    re.IGNORECASE | re.VERBOSE,
)


@dataclass
class ExtractedToolCall:
    """A tool call extracted from text.

    Attributes:
        name: Tool/function name
        arguments: Parsed arguments as dict
        raw_text: Original text that was parsed
        start_pos: Start position in source text
        end_pos: End position in source text
        confidence: Confidence score (0.0-1.0)
    """

    name: str
    arguments: Dict[str, Any]
    raw_text: str = ""
    start_pos: int = 0
    end_pos: int = 0
    confidence: float = 0.8


@dataclass
class ExtractionResult:
    """Result of text-based tool call extraction.

    Attributes:
        tool_calls: List of extracted tool calls
        remaining_content: Content after removing tool calls
        parse_method: Method used for extraction
        confidence: Overall confidence score
        warnings: Any warnings during extraction
    """

    tool_calls: List[ExtractedToolCall] = field(default_factory=list)
    remaining_content: str = ""
    parse_method: str = "python_call"
    confidence: float = 0.0
    warnings: List[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """Whether extraction found any tool calls."""
        return len(self.tool_calls) > 0


# Regex patterns for Python-like function calls
# Pattern matches: function_name(arg1='value', arg2="value", arg3=123, ...)
PYTHON_CALL_PATTERN = re.compile(
    r"""
    (?P<name>[a-zA-Z_][a-zA-Z0-9_]*)  # Function name
    \s*\(\s*                           # Opening parenthesis
    (?P<args>[^)]*?)                   # Arguments (non-greedy)
    \s*\)                              # Closing parenthesis
    """,
    re.VERBOSE,
)

# Alternative pattern for multi-line calls with complex arguments
MULTILINE_CALL_PATTERN = re.compile(
    r"""
    (?P<name>[a-zA-Z_][a-zA-Z0-9_]*)  # Function name
    \s*\(\s*                           # Opening parenthesis
    (?P<args>(?:[^()]*|\([^()]*\))*)   # Arguments (handles nested parens)
    \s*\)                              # Closing parenthesis
    """,
    re.VERBOSE | re.DOTALL,
)

EXECUTION_CELL_PATTERN = re.compile(
    r"""
    ```(?P<lang>bash|sh|shell|zsh|python|py)\s*\n
    (?P<code>.*?)
    \n```
    """,
    re.IGNORECASE | re.VERBOSE | re.DOTALL,
)

# Known tool name patterns (common Victor tools)
KNOWN_TOOL_PATTERNS = frozenset(
    [
        "read",
        "read_file",
        "write",
        "write_file",
        "edit",
        "edit_file",
        "shell",
        "bash",
        "run",
        "grep",
        "find",
        "ls",
        "search",
        "replace",
        "extract",
        "analyze",
        "graph",
        "metrics",
        "scan",
        "review",
        "refs",
        "symbol",
        "overview",
        "deps",
        "workflow",
        "sandbox",
        "jira",
        "cicd",
        "docs_coverage",
        "organize_imports",
        "rename",
        "inline",
    ]
)


class PythonCallExtractor:
    """Extracts Python-like function calls from text content.

    This extractor handles cases where models output tool calls as text
    instead of structured JSON, common with open-weight models on various
    hosting providers.

    Supported formats:
    - read_file(path='foo.py')
    - shell(command="ls -la", timeout=30)
    - edit(file_path='/path', old_string="x", new_string="y")

    The extractor is designed to be:
    - Non-greedy: Only extracts calls that match known tool names
    - Safe: Uses AST parsing for argument extraction where possible
    - Configurable: Accepts custom valid tool name sets
    """

    def __init__(
        self,
        known_tools: Optional[Set[str]] = None,
        strict_mode: bool = False,
    ):
        """Initialize the extractor.

        Args:
            known_tools: Set of known/valid tool names. If None, uses defaults.
            strict_mode: If True, only extract calls for known tools.
        """
        self._known_tools = {
            canonicalize_core_tool_name(tool_name)
            for tool_name in (known_tools or KNOWN_TOOL_PATTERNS)
        }
        self._strict_mode = strict_mode

    def extract_from_text(
        self,
        content: str,
        valid_tool_names: Optional[Set[str]] = None,
    ) -> ExtractionResult:
        """Extract Python-like function calls from text content.

        Args:
            content: Text content to parse
            valid_tool_names: Optional set of valid tool names to filter by.
                If provided, only extracts calls matching these names.

        Returns:
            ExtractionResult with extracted tool calls and metadata
        """
        if not content or not content.strip():
            return ExtractionResult(remaining_content=content)

        # Explicit valid_tool_names disables generic name heuristics.
        filter_names = (
            {canonicalize_core_tool_name(name) for name in valid_tool_names}
            if valid_tool_names
            else set(self._known_tools)
        )

        tool_calls: List[ExtractedToolCall] = []
        warnings: List[str] = []
        positions_to_remove: List[Tuple[int, int]] = []

        if "shell" in filter_names:
            for match in EXECUTION_CELL_PATTERN.finditer(content):
                if not self._has_execution_cell_intent_context(content, match.start()):
                    continue
                lang = match.group("lang").lower()
                code = match.group("code").strip("\n")
                if not code.strip():
                    continue

                if lang in {"python", "py"}:
                    cmd = f"python - <<'PY'\n{code}\nPY"
                else:
                    cmd = code

                tool_calls.append(
                    ExtractedToolCall(
                        name="shell",
                        arguments={"cmd": cmd},
                        raw_text=match.group(0),
                        start_pos=match.start(),
                        end_pos=match.end(),
                        confidence=0.85,
                    )
                )
                positions_to_remove.append((match.start(), match.end()))

        # Try simple pattern first
        for match in PYTHON_CALL_PATTERN.finditer(content):
            if any(start <= match.start() < end for start, end in positions_to_remove):
                continue
            name = match.group("name")
            canonical_name = canonicalize_core_tool_name(name)
            args_str = match.group("args")
            intent_context = self._has_tool_intent_context(content, match.start(), match.end())

            if self._is_definition_context(content, match.start()):
                continue

            # Filter by known/valid tool names
            if self._strict_mode and canonical_name not in filter_names:
                continue

            # Skip if name doesn't look like a tool
            if not self._is_likely_tool_name(
                canonical_name,
                filter_names,
                exact_match_required=valid_tool_names is not None,
            ):
                continue

            # Parse arguments
            parsed_args, parse_warning = self._parse_arguments(
                args_str,
                log_warning=intent_context,
            )
            if parse_warning:
                if intent_context:
                    warnings.append(f"{name}: {parse_warning}")
                # Still try to use partial results
                if not parsed_args:
                    continue

            parsed_args = self._canonicalize_arguments(canonical_name, parsed_args)

            # Ignore incidental zero-arg references embedded in prose such as
            # implementation notes like "_helper()" or headings like "tool()".
            if not parsed_args and not intent_context:
                continue

            # Calculate confidence based on various factors
            confidence = self._calculate_confidence(canonical_name, parsed_args, filter_names)

            tool_calls.append(
                ExtractedToolCall(
                    name=canonical_name,
                    arguments=parsed_args,
                    raw_text=match.group(0),
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=confidence,
                )
            )
            positions_to_remove.append((match.start(), match.end()))

        # Calculate remaining content (remove extracted calls)
        remaining = self._remove_positions(content, positions_to_remove)

        # Calculate overall confidence
        overall_confidence = 0.0
        if tool_calls:
            overall_confidence = sum(tc.confidence for tc in tool_calls) / len(tool_calls)

        return ExtractionResult(
            tool_calls=tool_calls,
            remaining_content=remaining.strip(),
            parse_method="python_call",
            confidence=overall_confidence,
            warnings=warnings,
        )

    def _has_execution_cell_intent_context(self, content: str, start_pos: int) -> bool:
        """Return True when a fenced shell/python cell is meant to be executed."""
        prefix = content[:start_pos].rsplit("\n", 2)
        context = "\n".join(prefix[-2:]).lower()
        if not context.strip():
            return False
        intent_words = (
            "use shell",
            "with shell",
            "run this",
            "execute this",
            "run the following",
            "execute the following",
            "tool call",
        )
        return any(word in context for word in intent_words)

    def _is_definition_context(self, content: str, start_pos: int) -> bool:
        """Return True when a match occurs inside a Python definition header."""
        line_start = content.rfind("\n", 0, start_pos) + 1
        prefix = content[line_start:start_pos].strip()
        return prefix.endswith("def") or prefix.endswith("async def") or prefix.endswith("class")

    def _has_tool_intent_context(self, content: str, start_pos: int, end_pos: int) -> bool:
        """Return True when text around a match looks like an actual tool request."""
        line_start = content.rfind("\n", 0, start_pos) + 1
        line_end = content.find("\n", end_pos)
        if line_end == -1:
            line_end = len(content)

        before = content[line_start:start_pos]
        after = content[end_pos:line_end]
        stripped_before = before.strip()
        stripped_after = after.strip()

        # Ignore inline code spans and markdown examples.
        if before.count("`") % 2 == 1:
            return False

        # Bare tool call on its own line, optionally under a bullet.
        if not stripped_before:
            return not stripped_after
        if re.fullmatch(r"(?:[-*+]|\d+\.)", stripped_before):
            return not stripped_after

        # Imperative lead-ins like "Let me use:" or "I'll run:".
        if _TOOL_INTENT_PREFIX_RE.search(stripped_before):
            return True

        return False

    def _is_likely_tool_name(
        self,
        name: str,
        valid_names: Set[str],
        exact_match_required: bool = False,
    ) -> bool:
        """Check if a name is likely a tool name.

        Args:
            name: Function name to check
            valid_names: Set of valid tool names
            exact_match_required: When True, only exact valid_names matches count.

        Returns:
            True if name looks like a tool name
        """
        canonical_name = canonicalize_core_tool_name(name)

        # Direct match
        if canonical_name in valid_names:
            return True

        if exact_match_required:
            return False

        # Check against known patterns
        if canonical_name in self._known_tools:
            return True

        # Heuristics for tool-like names
        name_lower = name.lower()

        # Common tool name suffixes/prefixes
        tool_indicators = [
            "_file",
            "_dir",
            "_directory",
            "_code",
            "_text",
            "read_",
            "write_",
            "edit_",
            "get_",
            "set_",
            "list_",
            "find_",
            "search_",
            "run_",
            "execute_",
        ]

        for indicator in tool_indicators:
            if indicator in name_lower:
                return True

        # Avoid common Python builtins and methods
        builtins_to_skip = {
            "print",
            "len",
            "str",
            "int",
            "float",
            "list",
            "dict",
            "set",
            "tuple",
            "bool",
            "type",
            "range",
            "enumerate",
            "zip",
            "map",
            "filter",
            "open",
            "input",
            "format",
            "sorted",
            "reversed",
            "min",
            "max",
            "sum",
            "abs",
            "round",
            "isinstance",
            "hasattr",
            "getattr",
            "setattr",
        }

        if name_lower in builtins_to_skip:
            return False

        return False

    def _canonicalize_arguments(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize extracted arguments onto the compact canonical core-tool surface."""
        if not args:
            return args

        normalized = dict(args)

        if tool_name == "shell":
            if "command" in normalized and "cmd" not in normalized:
                normalized["cmd"] = normalized.pop("command")
            if "arg_0" in normalized and "cmd" not in normalized and len(normalized) == 1:
                normalized["cmd"] = normalized.pop("arg_0")
        elif tool_name in {"read", "write", "ls"}:
            for alias in ("file_path", "filename", "file", "directory", "dir"):
                if alias in normalized and "path" not in normalized:
                    normalized["path"] = normalized.pop(alias)
            if "arg_0" in normalized and "path" not in normalized and len(normalized) == 1:
                normalized["path"] = normalized.pop("arg_0")
        elif tool_name == "edit":
            if "file_path" in normalized and "path" not in normalized:
                normalized["path"] = normalized.pop("file_path")
            if "old_string" in normalized and "old_str" not in normalized:
                normalized["old_str"] = normalized.pop("old_string")
            if "new_string" in normalized and "new_str" not in normalized:
                normalized["new_str"] = normalized.pop("new_string")
            if "operations" in normalized and "ops" not in normalized:
                normalized["ops"] = normalized.pop("operations")
        elif tool_name in {"grep", "search", "code_search"}:
            if "pattern" in normalized and "query" not in normalized:
                normalized["query"] = normalized.pop("pattern")
            if "arg_0" in normalized and "query" not in normalized and len(normalized) == 1:
                normalized["query"] = normalized.pop("arg_0")

        return normalized

    def _parse_arguments(
        self,
        args_str: str,
        *,
        log_warning: bool = True,
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        """Parse Python-style keyword arguments from a string.

        Handles formats like:
        - path='foo.py'
        - command="ls -la", timeout=30
        - file_path='/path/to/file', content="multi\\nline"

        Args:
            args_str: Argument string without parentheses

        Returns:
            Tuple of (parsed_dict, warning_message or None)
        """
        if not args_str or not args_str.strip():
            return {}, None

        args_str = args_str.strip()

        # Method 1: Try AST parsing (safest)
        try:
            # Wrap in a function call for AST parsing
            fake_call = f"_f({args_str})"
            tree = ast.parse(fake_call, mode="eval")

            if isinstance(tree.body, ast.Call):
                args_dict = {}
                for keyword in tree.body.keywords:
                    if keyword.arg:
                        value = self._ast_to_value(keyword.value)
                        args_dict[keyword.arg] = value

                # Also handle positional args (rare but possible)
                for i, arg in enumerate(tree.body.args):
                    value = self._ast_to_value(arg)
                    args_dict[f"arg_{i}"] = value

                return args_dict, None
        except (SyntaxError, ValueError):
            pass  # Fall through to regex parsing

        # Method 2: Regex-based parsing (fallback)
        return self._parse_arguments_regex(args_str, log_warning=log_warning)

    def _ast_to_value(self, node: ast.AST) -> Any:
        """Convert AST node to Python value.

        Args:
            node: AST node

        Returns:
            Python value (str, int, float, bool, None, list, dict)
        """
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.List):
            return [self._ast_to_value(elt) for elt in node.elts]
        elif isinstance(node, ast.Dict):
            return {
                self._ast_to_value(k) if k else None: self._ast_to_value(v)
                for k, v in zip(node.keys, node.values)
            }
        elif isinstance(node, ast.Name):
            # Handle True, False, None as names
            if node.id == "True":
                return True
            elif node.id == "False":
                return False
            elif node.id == "None":
                return None
            raise ValueError(f"Non-literal identifier: {node.id}")
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            # Handle negative numbers
            return -self._ast_to_value(node.operand)
        else:
            raise ValueError(f"Unsupported AST node: {type(node).__name__}")

    def _parse_arguments_regex(
        self,
        args_str: str,
        *,
        log_warning: bool = True,
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        """Parse arguments using regex (fallback method).

        Args:
            args_str: Argument string

        Returns:
            Tuple of (parsed_dict, warning_message or None)
        """
        args_dict: Dict[str, Any] = {}
        warning = None

        # Pattern for key=value pairs
        # Handles: key='value', key="value", key=123, key=True
        arg_pattern = re.compile(
            r"""
            (\w+)\s*=\s*        # Key name and equals
            (?:
                '([^']*)'       # Single-quoted string
                |"([^"]*)"      # Double-quoted string
                |(\d+\.?\d*)    # Number
                |(True|False|None)  # Boolean/None
            )
            """,
            re.VERBOSE,
        )

        for match in arg_pattern.finditer(args_str):
            key = match.group(1)
            # Check which capture group matched
            if match.group(2) is not None:
                value = match.group(2)
            elif match.group(3) is not None:
                value = match.group(3)
            elif match.group(4) is not None:
                num_str = match.group(4)
                value = float(num_str) if "." in num_str else int(num_str)
            elif match.group(5) is not None:
                bool_str = match.group(5)
                value = {"True": True, "False": False, "None": None}.get(bool_str)
            else:
                continue

            args_dict[key] = value

        if not args_dict and args_str.strip():
            warning = f"Could not parse arguments: {args_str[:50]}..."
            warning += "\n\nExpected format: concrete literal keyword arguments"
            warning += "\nExample: read(path='/path/to/file.py', retries=3)"
            warning += (
                "\nDo NOT use: function signatures, type hints, placeholder variables, "
                "or unresolved identifiers"
            )

            if log_warning:
                logger.warning(warning)

                # Log for debugging
                logger.debug(f"Failed parse - original args_str: {args_str}")
                logger.debug("Attempted parse methods: regex, ast")

        return args_dict, warning

    def _calculate_confidence(
        self,
        name: str,
        args: Dict[str, Any],
        valid_names: Set[str],
    ) -> float:
        """Calculate confidence score for an extracted tool call.

        Args:
            name: Tool name
            args: Parsed arguments
            valid_names: Set of valid tool names

        Returns:
            Confidence score between 0.0 and 1.0
        """
        confidence = 0.5  # Base confidence

        # Boost for exact match in valid names
        if name in valid_names:
            confidence += 0.3

        # Boost for known tool patterns
        if name in self._known_tools:
            confidence += 0.15

        # Boost if has arguments (less likely to be false positive)
        if args:
            confidence += 0.1

        # Boost for common argument names
        common_args = {"path", "cmd", "content", "query", "old_str", "new_str", "ops"}
        if any(arg in common_args for arg in args.keys()):
            confidence += 0.1

        return min(confidence, 1.0)

    def _remove_positions(
        self,
        content: str,
        positions: List[Tuple[int, int]],
    ) -> str:
        """Remove specified positions from content.

        Args:
            content: Original content
            positions: List of (start, end) positions to remove

        Returns:
            Content with positions removed
        """
        if not positions:
            return content

        # Sort positions in reverse order to avoid index shifting
        sorted_positions = sorted(positions, key=lambda x: x[0], reverse=True)

        result = content
        for start, end in sorted_positions:
            result = result[:start] + result[end:]

        return result


# Singleton instance for convenience
_default_extractor: Optional[PythonCallExtractor] = None


def get_extractor() -> PythonCallExtractor:
    """Get the default PythonCallExtractor instance."""
    global _default_extractor
    if _default_extractor is None:
        _default_extractor = PythonCallExtractor()
    return _default_extractor


def extract_tool_calls_from_text(
    content: str,
    valid_tool_names: Optional[Set[str]] = None,
) -> ExtractionResult:
    """Convenience function to extract tool calls from text.

    Args:
        content: Text content to parse
        valid_tool_names: Optional set of valid tool names

    Returns:
        ExtractionResult with extracted tool calls
    """
    return get_extractor().extract_from_text(content, valid_tool_names)
