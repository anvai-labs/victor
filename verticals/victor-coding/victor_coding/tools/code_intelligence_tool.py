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


from typing import Any, Dict, List, Optional
from pathlib import Path

Query = None
QueryCursor = None

try:
    from tree_sitter import Query as _Query, QueryCursor as _QueryCursor

    Query = _Query
    QueryCursor = _QueryCursor
except ImportError:
    pass


def _get_tree_sitter_parser(language: str):
    """Get a tree-sitter parser via the capability registry."""
    from victor_contracts.capability_runtime import (
        TreeSitterParserProtocol,
        get_capability_provider,
    )

    provider = get_capability_provider(TreeSitterParserProtocol)
    if provider is None:
        return None
    return provider.get_parser(language)


def _get_analysis_provider():
    """Return the enhanced TreeSitterAnalysisProtocol provider or None.

    Only returns the provider when the registry marks it as ENHANCED — the
    null stub returns False from supports_language() so checking is_enhanced
    short-circuits the per-call lookup.
    """
    try:
        from victor_contracts.capability_runtime import (
            TreeSitterAnalysisProtocol,
            get_capability_provider,
            is_capability_enhanced,
        )

        if not is_capability_enhanced(TreeSitterAnalysisProtocol):
            return None
        return get_capability_provider(TreeSitterAnalysisProtocol)
    except Exception:
        return None


# Lightweight suffix → language map for symbol() / refs() language detection.
# Falls back to "python" because both tools have historically been Python-only.
_SUFFIX_TO_LANGUAGE = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
}


def _detect_language_from_path(file_path: str) -> str:
    return _SUFFIX_TO_LANGUAGE.get(Path(file_path).suffix.lower(), "python")


from victor.tools.base import AccessMode, DangerLevel, Priority, ExecutionCategory
from victor.tools.decorators import tool

# Tree-sitter queries to find function and class definitions in Python
PYTHON_QUERIES = {
    "function": """
    (function_definition
      name: (identifier) @function.name)
    """,
    "class": """
    (class_definition
      name: (identifier) @class.name)
    """,
    "identifier": """
    (identifier) @name
    """,
}


@tool(
    category="code_intelligence",
    priority=Priority.HIGH,  # Important for code navigation
    access_mode=AccessMode.READONLY,  # Only reads files for analysis
    danger_level=DangerLevel.SAFE,  # No side effects
    keywords=[
        "symbol",
        "find",
        "definition",
        "function",
        "class",
        "locate",
        "ast",
        # Natural language patterns for symbol lookup (semantic matching)
        "method",
        "where is",
        "show me",
        "look up",
        "go to definition",
        "find the",
        "locate the",
        "get definition",
        "code for",
        "implementation",
    ],
    stages=[
        "initial",
        "planning",
        "reading",
        "analysis",
    ],  # Available early when symbol name known
    mandatory_keywords=[
        "find",
        "find symbol",
        "find definition",
    ],  # From MANDATORY_TOOL_KEYWORDS "find" -> ["symbol", "refs"]
)
async def symbol(file_path: str, symbol_name: str) -> Optional[Dict[str, Any]]:
    """[AST-AWARE] Get FULL CODE of a function/class definition in a specific file.

    Returns the complete code block (not just a reference). Use when you KNOW:
    1. The exact file path
    2. The exact symbol name

    DIFFERS FROM:
    - search(): Finds files/snippets when you DON'T know where code is
    - refs(): Finds all USAGES of a symbol across project (not the definition)
    - graph(): Shows RELATIONSHIPS between symbols (not code content)

    Args:
        file_path: Path to Python file (.py). MUST be known.
        symbol_name: Exact name of function or class. MUST be known.

    Returns:
        Dict with symbol_name, type, file_path, start_line, end_line, code.
        None if not found. {"error": ...} on file/parse errors.

    Example:
        symbol(file_path="victor/agent/orchestrator.py", symbol_name="chat")
        # Returns full code of the chat() method with line numbers
    """
    language = _detect_language_from_path(file_path)
    try:
        with open(file_path, "rb") as f:
            content = f.read()
    except FileNotFoundError:
        return {"error": f"File not found: {file_path}"}
    except Exception as e:
        return {"error": f"An unexpected error occurred: {e}"}

    # Preferred path: ask the TreeSitterAnalysisProtocol provider for symbols
    # and look the name up in its dict output. This works for any language
    # the provider supports — not just Python.
    provider = _get_analysis_provider()
    if provider is not None and provider.supports_language(language):
        try:
            symbols = provider.extract_symbols(content, language, file_path=file_path)
        except Exception:
            symbols = None
        if symbols:
            match = next((s for s in symbols if s.get("name") == symbol_name), None)
            if match is not None:
                start_line = int(match.get("line_start") or 1)
                end_line = int(match.get("line_end") or start_line)
                code_block = _read_line_range(content, start_line, end_line)
                return {
                    "symbol_name": symbol_name,
                    "type": match.get("symbol_type") or "symbol",
                    "file_path": file_path,
                    "start_line": start_line,
                    "end_line": end_line,
                    "code": code_block,
                }
            # Provider returned symbols but the requested name was not among
            # them — treat as "not found" without falling through to the
            # legacy parser (the provider is authoritative).
            return None

    # Fallback: legacy tree-sitter parser (Python queries hardcoded).
    try:
        parser = _get_tree_sitter_parser("python")
        if parser is None or Query is None:
            return {
                "error": "Code intelligence requires the victor-coding package. "
                "Install with: pip install victor-coding"
            }

        tree = parser.parse(content)
        root_node = tree.root_node

        for symbol_type in ["function", "class"]:
            query = Query(parser.language, PYTHON_QUERIES[symbol_type])
            cursor = QueryCursor(query)
            captures_dict = cursor.captures(root_node)

            # captures_dict is {"function.name": [node1, node2, ...]} or {"class.name": [...]}
            for _capture_name, nodes in captures_dict.items():
                for node in nodes:
                    if node.text.decode("utf8") == symbol_name:
                        # We found the name identifier, now get the parent definition node
                        definition_node = node.parent
                        start_line = definition_node.start_point[0] + 1
                        end_line = definition_node.end_point[0] + 1
                        code_block = definition_node.text.decode("utf8")

                        return {
                            "symbol_name": symbol_name,
                            "type": symbol_type,
                            "file_path": file_path,
                            "start_line": start_line,
                            "end_line": end_line,
                            "code": code_block,
                        }

        return None  # Symbol not found

    except Exception as e:
        return {"error": f"An unexpected error occurred: {e}"}


def _read_line_range(content: bytes, start_line: int, end_line: int) -> str:
    """Slice ``content`` to the 1-based inclusive line range."""
    text = content.decode("utf8", errors="ignore")
    lines = text.splitlines()
    start = max(0, start_line - 1)
    end = min(len(lines), end_line)
    return "\n".join(lines[start:end])


@tool(
    category="code_intelligence",
    priority=Priority.HIGH,  # Important for code navigation
    access_mode=AccessMode.READONLY,  # Only reads files for analysis
    danger_level=DangerLevel.SAFE,  # No side effects
    keywords=[
        "refs",
        "references",
        "usage",
        "occurrences",
        "ast",
        # Natural language patterns for usage lookup (semantic matching)
        "used",
        "called",
        "invoked",
        "callers",
        "who calls",
        "all usages",
        "find usages",
        "list references",
        "where used",
    ],
    stages=[
        "initial",
        "planning",
        "reading",
        "analysis",
    ],  # Available early for impact analysis
    execution_category=ExecutionCategory.READ_ONLY,
    mandatory_keywords=[
        "find",
        "find references",
        "find usages",
    ],  # From MANDATORY_TOOL_KEYWORDS "find" -> ["symbol", "refs"]
)
async def refs(symbol_name: str, search_path: str = ".") -> List[Dict[str, Any]]:
    """[AST-AWARE] Find all USAGES of a symbol across the project.

    Project-wide scan using AST parsing. More accurate than grep (exact identifier
    matches only, no substrings). Use when you KNOW the symbol name but want to
    find WHERE it's used.

    DIFFERS FROM:
    - symbol(): Gets the DEFINITION code (not usages). Use when you know file + name.
    - search(): Finds files by CONCEPT/TEXT. Use when you DON'T know the exact name.
    - graph(): Shows RELATIONSHIPS/dependencies. Use for impact analysis, not locations.

    Args:
        symbol_name: Exact symbol name (function, class, variable) to find usages of.
        search_path: Directory to search recursively. Default: current directory.

    Returns:
        List of dicts: [{file_path, line, column, preview}, ...]
        Empty list if no references found.

    Example:
        refs(symbol_name="BaseProvider")
        # Returns all files/lines where BaseProvider is referenced

    When NOT to use:
        - Non-Python files (use grep instead)
        - Finding text patterns (use search() with mode="literal")
        - Getting the definition code (use symbol() instead)
    """
    parser = _get_tree_sitter_parser("python")
    if parser is None or Query is None:
        return [
            {
                "error": "Code intelligence requires the victor-coding package. "
                "Install with: pip install victor-coding"
            }
        ]
    references = []
    query = Query(parser.language, PYTHON_QUERIES["identifier"])

    root_path = Path(search_path)
    if not root_path.is_dir():
        return [{"error": f"Invalid search path: {search_path} is not a directory."}]

    for file_path in root_path.rglob("*.py"):
        try:
            with open(file_path, "rb") as f:
                content_bytes = f.read()

            tree = parser.parse(content_bytes)
            cursor = QueryCursor(query)
            captures_dict = cursor.captures(tree.root_node)

            # For reading lines for the preview
            content_lines = content_bytes.decode("utf8", errors="ignore").splitlines()

            # captures_dict is {"name": [node1, node2, ...]}
            for _capture_name, nodes in captures_dict.items():
                for node in nodes:
                    if node.text.decode("utf8") == symbol_name:
                        line_number = node.start_point[0] + 1
                        col_number = node.start_point[1] + 1
                        references.append(
                            {
                                "file_path": str(file_path),
                                "line": line_number,
                                "column": col_number,
                                "preview": content_lines[line_number - 1].strip(),
                            }
                        )
        except Exception:
            # Ignore files that can't be parsed
            continue

    return references


# Note: For project-wide symbol renaming, use the consolidated `rename` tool
# from refactor_tool.py with scope="project". Example:
#   rename(old_name="foo", new_name="bar", scope="project")
