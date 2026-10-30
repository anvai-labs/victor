"""Language detection + the tree-sitter grammar name map.

The Python path uses the stdlib ``ast`` (no grammar needed); everything else routes to
the generic tree-sitter extractor when the grammar pack is installed.
"""

from __future__ import annotations

import os

# Extension -> canonical language name.
EXTENSION_TO_LANGUAGE: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".hxx": "cpp",
    ".cs": "csharp",
    ".rb": "ruby",
    ".rbi": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".scala": "scala",
    ".sh": "bash",
    ".bash": "bash",
    ".lua": "lua",
    ".sql": "sql",
}

# Canonical language name -> tree-sitter-language-pack grammar name. (Most are 1:1;
# this indirection lets us split TS/TSX which share one canonical extraction path.)
TREE_SITTER_GRAMMAR: dict[str, str] = {
    "javascript": "javascript",
    "typescript": "typescript",
    "tsx": "tsx",
    "rust": "rust",
    "go": "go",
    "java": "java",
    "c": "c",
    "cpp": "cpp",
    "csharp": "csharp",
    "ruby": "ruby",
    "php": "php",
    "swift": "swift",
    "kotlin": "kotlin",
    "scala": "scala",
    "bash": "bash",
    "lua": "lua",
    "sql": "sql",
}


def detect_language(file_path: str) -> str | None:
    """Best-effort language from file extension."""

    _, ext = os.path.splitext(file_path)
    return EXTENSION_TO_LANGUAGE.get(ext.lower())
