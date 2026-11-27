"""Language detection + the tree-sitter grammar name map.

The Python path uses the stdlib ``ast`` (no grammar needed); everything else routes to
the generic tree-sitter extractor when the grammar pack is installed.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable

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


# ── ``.h`` headers: C or C++ ─────────────────────────────────────────────────
#
# ``.h`` is shared by C and C++. C++ is *not* a strict superset of C: a C header
# that names a parameter ``new``/``this``/``class`` or uses ``restrict`` or
# ``_Generic`` is an error under the C++ grammar, and a C++ header (``class``,
# ``namespace``, ``template``) is an error under the C grammar. Resolution is
# therefore content-first, then project context, then a parse-time retry with
# the other grammar (see ``parser.parse``).

AMBIGUOUS_HEADER_EXTENSIONS = frozenset({".h"})
C_FAMILY = frozenset({"c", "cpp"})

# One combined scan, strings FIRST: stripping comments before strings let a
# '/*' inside a string literal (const char *s = "/*";) eat everything up to a
# later '*/' via the DOTALL match, blanking the file's language signals; the
# naive swap has the mirror bug ('"' inside a comment). The alternation makes
# whichever construct starts first win.
_TOKEN = re.compile(
    r'"(?:\\.|[^"\\\n])*"'  # string literal (wins if it starts first)
    r"|//[^\n]*"  # line comment
    r"|/\*.*?\*/",  # block comment (DOTALL below)
    re.DOTALL,
)


def _strip_comments_and_normalize_strings(content: str) -> str:
    """Replace string literals with "" and comments with a space, in one scan."""

    def repl(match: "re.Match[str]") -> str:
        return '""' if match.group(0).startswith('"') else " "

    return _TOKEN.sub(repl, content)


_CPP_SIGNALS = tuple(
    re.compile(pattern, re.MULTILINE)
    for pattern in (
        r"^\s*namespace\b",
        r"^\s*template\s*<",
        r"\bclass\s+[A-Za-z_][\w\s]*(?::[^;{]*)?\{",  # class [EXPORT_MACRO] Name [: base] {
        r"^\s*(?:public|private|protected)\s*:",
        r"\b[A-Za-z_]\w*\s*::\s*~?[A-Za-z_]",
        r"^\s*using\s+(?:namespace\b|[A-Za-z_]\w*\s*=)",
        # C++-only keywords. ``constexpr`` and ``nullptr`` are left out: C23
        # made them C keywords as well, so they cannot tell the languages apart.
        r"\b(?:noexcept|decltype|static_cast|dynamic_cast|reinterpret_cast|"
        r"const_cast|typename|virtual)\b",
        r"\boperator\s*(?:[^\w\s(]|\(\s*\))",
        r"^\s*#\s*include\s*<[A-Za-z_]+>",  # extensionless standard library header
    )
)
_C_SIGNALS = tuple(
    re.compile(pattern, re.MULTILINE)
    for pattern in (
        r"\brestrict\b",
        r"\b_Generic\s*\(",
        # A C++ keyword used as an identifier: `int new;`, `char *this,`, `x->class`.
        r"(?:\b[A-Za-z_]\w*\s+\**|\*\s*|->\s*|\.\s*)(?:new|delete|this|class|template|"
        r"typename|private|public|protected|operator|namespace|virtual|friend|"
        r"explicit|mutable)\s*[,;)\[=]",
    )
)
_CPLUSPLUS_GUARD = re.compile(r"^\s*#\s*if(?:def|\s+defined)\s*\(?\s*__cplusplus\b", re.MULTILINE)


def classify_header(content: str) -> str | None:
    """Return ``"cpp"``/``"c"`` from a header's own text, or None when ambiguous.

    Each signal family counts once. Keywords that C23 shares with C++
    (``constexpr``, ``nullptr``) are evidence for neither side. Keywords that C23 shares with C++
    (``constexpr``, ``nullptr``) are evidence for neither side. An ``#ifdef __cplusplus`` guard marks a C
    header written to be consumable from C++; it counts toward C, so the header
    is C unless it also carries more distinct C++ constructs.
    """

    text = _strip_comments_and_normalize_strings(content)
    cpp = sum(1 for pattern in _CPP_SIGNALS if pattern.search(text))
    c = sum(1 for pattern in _C_SIGNALS if pattern.search(text))
    if _CPLUSPLUS_GUARD.search(text):
        c += 1
    if cpp > c:
        return "cpp"
    if c > cpp:
        return "c"
    return None


def resolve_language(
    file_path: str,
    content: str | None = None,
    *,
    header_default: str = "cpp",
) -> str | None:
    """Language for ``file_path``, reading ``content`` for an ambiguous ``.h``.

    ``header_default`` decides a header whose text carries no signal either way.
    It defaults to C++ because the C++ grammar accepts the shared C subset such
    a header contains; ``parse_repo`` passes ``"c"`` for a project with C
    sources and no C++ sources so a pure C library keeps its C label.
    """

    language = detect_language(file_path)
    if os.path.splitext(file_path)[1].lower() not in AMBIGUOUS_HEADER_EXTENSIONS:
        return language
    if header_default not in C_FAMILY:
        raise ValueError(f"header_default must be one of {sorted(C_FAMILY)}")
    if content is None:
        return header_default
    return classify_header(content) or header_default


def matches_languages(file_path: str, allowed: set[str] | frozenset[str] | None) -> bool:
    """Whether a path belongs to an inventory filtered to ``allowed`` languages.

    An ambiguous ``.h`` matches a filter naming either C or C++, because its
    language is only decided from its text at parse time.
    """

    language = detect_language(file_path)
    if language is None:
        return False
    if allowed is None or language in allowed:
        return True
    return os.path.splitext(file_path)[1].lower() in AMBIGUOUS_HEADER_EXTENSIONS and bool(
        allowed & C_FAMILY
    )


def project_header_default(file_paths: Iterable[str | os.PathLike[str]]) -> str:
    """C for an inventory with C sources and no C++ sources; otherwise C++."""

    has_c = False
    for path in file_paths:
        ext = os.path.splitext(os.fspath(path))[1].lower()
        if ext in AMBIGUOUS_HEADER_EXTENSIONS:
            continue
        language = EXTENSION_TO_LANGUAGE.get(ext)
        if language == "cpp":
            return "cpp"
        has_c = has_c or language == "c"
    return "c" if has_c else "cpp"
