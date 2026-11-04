"""Offline grammar loading shared by indexing and structural analysis.

Parsers belong to the caller (scan or indexing worker). Grammar packages are
existing optional dependencies; analysis never installs or downloads a grammar.
"""

from __future__ import annotations

from importlib import import_module
import re
from typing import Any

try:
    from victor_codegraph.treesitter_tables import IDENTIFIER_NODES
except ImportError:
    # The base framework also supports Python/source analysis without the
    # optional codegraph package. Keep its syntax walker importable there.
    IDENTIFIER_NODES = frozenset(
        {
            "identifier",
            "field_identifier",
            "property_identifier",
            "type_identifier",
            "shorthand_property_identifier",
            "constant",
            "simple_identifier",
            "word",
        }
    )

LANGUAGE_MODULES = {
    "python": ("tree_sitter_python", "language"),
    "javascript": ("tree_sitter_javascript", "language"),
    "typescript": ("tree_sitter_typescript", "language_typescript"),
    "tsx": ("tree_sitter_typescript", "language_tsx"),
    "go": ("tree_sitter_go", "language"),
    "rust": ("tree_sitter_rust", "language"),
    "java": ("tree_sitter_java", "language"),
    "c": ("tree_sitter_c", "language"),
    "cpp": ("tree_sitter_cpp", "language"),
    "ruby": ("tree_sitter_ruby", "language"),
    "kotlin": ("tree_sitter_kotlin", "language"),
    "scala": ("tree_sitter_scala", "language"),
    "csharp": ("tree_sitter_c_sharp", "language"),
    "php": ("tree_sitter_php", "language_php"),
    "swift": ("tree_sitter_swift", "language"),
    "lua": ("tree_sitter_lua", "language"),
    "bash": ("tree_sitter_bash", "language"),
}


def _source_frontends() -> tuple[dict[str, str], dict[str, str]]:
    """Share indexing's suffix inventory; compiler support is optional enrichment."""
    from victor.core.graph_rag.precise_providers import all_providers, extensions_for_provider

    try:
        from victor_codegraph.languages import EXTENSION_TO_LANGUAGE
    except ImportError:
        # Preserve the pre-existing base-install inventory. Full source routing
        # uses the optional graph package's canonical language registry.
        aliases = {
            ".js": "javascript",
            ".jsx": "javascript",
            ".mjs": "javascript",
            ".cjs": "javascript",
            ".tsx": "tsx",
            ".c": "c",
            ".h": "c",
        }
        languages = {}
        for provider in all_providers():
            default = (
                "cpp"
                if provider.languages == frozenset({"c", "cpp"})
                else sorted(provider.languages)[0]
            )
            languages.update(
                {
                    suffix: aliases.get(suffix, default)
                    for suffix in extensions_for_provider(provider.name)
                }
            )
    else:
        languages = {
            suffix: language
            for suffix, language in EXTENSION_TO_LANGUAGE.items()
            if language in LANGUAGE_MODULES
        }
    providers = {}
    for provider in all_providers():
        for extension in extensions_for_provider(provider.name):
            if extension in languages:
                providers[extension] = provider.name
    return languages, providers


SOURCE_LANGUAGES, SOURCE_PROVIDERS = _source_frontends()


_JNI_TOKENS = re.compile(
    rb"//[^\n]*|/\*[\s\S]*?\*/|"
    rb'(?:u8|u|U|L)?R"(?P<delimiter>[^ ()\\\t\r\n]{0,16})\([\s\S]*?\)(?P=delimiter)"|'
    rb'"(?:\\[\s\S]|[^"\\])*"|\'(?:\\[\s\S]|[^\'\\])*\'|'
    rb'(?P<export>^[ \t]*(?:extern[ \t]+"C"[ \t]+)?JNIEXPORT[ \t\r\n]+'
    rb"[\w* \t\r\n]+[ \t\r\n]+JNICALL[ \t\r\n]+(?P<jni_name>Java_[\w]+)[ \t\r\n]*\()",
    re.MULTILINE,
)


def jni_exports(data: bytes) -> set[tuple[str, int]]:
    """Conventional JNI export names/byte offsets outside comments and literals."""
    return {
        (m["jni_name"].decode("utf-8"), m.start("jni_name"))
        for m in _JNI_TOKENS.finditer(data)
        if m["export"] is not None and not re.search(rb"\bstatic\b", m["export"])
    }


def parser_input(data: bytes, language: str) -> bytes:
    """Hide known JNI declaration macros, retaining every source byte offset.

    Official C grammars cannot expand JNIEXPORT/JNICALL. Only the conventional
    export declaration prefix is blanked; bodies, strings and comments survive.
    """
    if language not in {"c", "cpp"}:
        return data
    return _JNI_TOKENS.sub(
        lambda match: (
            re.sub(rb"\b(?:JNIEXPORT|JNICALL)\b", lambda word: b" " * len(word[0]), match[0])
            if match["export"] is not None
            else match[0]
        ),
        data,
    )


def create_parser(language: str) -> Any | None:
    """Return a fresh official parser, or None when its wheel is unavailable."""
    if language not in LANGUAGE_MODULES:
        return None
    try:
        from tree_sitter import Language, Parser

        module, entry = LANGUAGE_MODULES[language]
        grammar = getattr(import_module(module), entry)()
        return Parser(grammar if isinstance(grammar, Language) else Language(grammar))
    except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
        return None
