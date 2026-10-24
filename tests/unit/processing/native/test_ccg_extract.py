"""Native CCG statement extraction — differential parity + fallback.

The native walk must be record-for-record identical to the Python walk
(``CodeContextGraphBuilder._extract_statement_nodes``) on real fixtures;
the Python implementation STAYS as oracle and fallback. Runs where the
native extension is built (CI wheels); skips cleanly on pure-Python
installs — the fallback path is pinned separately below.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from victor.core.indexing.ccg_builder import CodeContextGraphBuilder
from victor.processing.native.ccg_extract import (
    NATIVE_STATEMENT_LANGUAGES,
    extract_statements,
    native_statements_available,
)

REPO_ROOT = Path(__file__).resolve().parents[4]

PY_FIXTURES = [
    "victor/core/indexing/ccg_builder.py",
    "victor/core/graph_rag/precise_providers.py",
]
TS_FIXTURES = ["vscode-victor/src/victorClient.ts"]


def _parse_with(path: Path, language: str):
    import importlib

    import tree_sitter

    module = {
        "python": "tree_sitter_python",
        "javascript": "tree_sitter_javascript",
        "typescript": "tree_sitter_typescript",
    }[language]
    lang_obj = (
        importlib.import_module(module).language_typescript()
        if language == "typescript"
        else importlib.import_module(module).language()
    )
    ts_language = (
        tree_sitter.Language(lang_obj)
        if not isinstance(lang_obj, tree_sitter.Language)
        else lang_obj
    )
    return tree_sitter.Parser(ts_language).parse(path.read_bytes())


def _python_walk_records(path: Path, language: str):
    builder = CodeContextGraphBuilder(language=language)
    tree = _parse_with(path, language)
    nodes = builder._extract_statement_nodes(
        tree.root_node, path, path.read_text().split("\n"), source_code=None
    )
    return [
        (
            n.node_id,
            n.ast_kind,
            getattr(n, "statement_type", None) or n.metadata.get("statement_type"),
            n.line,
            n.end_line,
            n.metadata.get("content_preview", ""),
            n.visibility,
            n.scope_id or "",
        )
        for n in nodes
    ]


def _native_records(path: Path, language: str):
    records = extract_statements(language, str(path), path.read_bytes())
    assert records is not None, "native path unexpectedly unavailable"
    return [
        (
            r.node_id,
            r.node_type,
            r.statement_type,
            r.start_line,
            r.end_line,
            r.content_preview,
            r.visibility,
            r.scope_id,
        )
        for r in records
    ]


@pytest.mark.parametrize("fixture", PY_FIXTURES + TS_FIXTURES)
def test_native_walk_matches_python_walk_record_for_record(fixture):
    """Byte parity: ids, classification, spans, previews, visibility, and
    the scope cascade must match exactly — the engine's core contract."""
    language = "typescript" if fixture.endswith(".ts") else "python"
    path = REPO_ROOT / fixture
    if not path.exists():
        pytest.skip(f"fixture not present: {fixture}")
    if not native_statements_available(language):
        pytest.skip("native extension not built")
    assert _python_walk_records(path, language) == _native_records(path, language)


def test_integrated_builder_uses_native_and_matches_fallback():
    """Through the builder itself: native path (source_code provided) must
    equal the pure-Python walk (source_code withheld)."""
    path = REPO_ROOT / "victor/core/graph_rag/precise_providers.py"
    if not path.exists() or not native_statements_available("python"):
        pytest.skip("fixture or native extension unavailable")
    builder = CodeContextGraphBuilder(language="python")
    tree = _parse_with(path, "python")
    native_nodes = builder._extract_statement_nodes(
        tree.root_node, path, path.read_text().split("\n"), source_code=path.read_text()
    )
    fallback_nodes = builder._extract_statement_nodes(
        tree.root_node, path, path.read_text().split("\n"), source_code=None
    )
    key = lambda n: (  # noqa: E731
        n.node_id,
        n.name,
        n.line,
        n.end_line,
        n.statement_type,
        n.visibility,
        n.scope_id,
        n.metadata,
    )
    assert [key(n) for n in native_nodes] == [key(n) for n in fallback_nodes]


def test_unsupported_language_falls_back():
    assert "go" not in NATIVE_STATEMENT_LANGUAGES
    assert native_statements_available("go") is False
    assert extract_statements("go", "/x/main.go", b"package main") is None


def test_fallback_when_native_absent(monkeypatch):
    """Without the extension the wrapper returns None — the Python walk
    runs unchanged (pure-Python installs keep working)."""
    import victor.processing.native.ccg_extract as ccg_extract

    monkeypatch.setattr(ccg_extract, "_NATIVE_AVAILABLE", False)
    monkeypatch.setattr(ccg_extract, "native_statements_available", lambda language: False)
    assert ccg_extract.extract_statements("python", "/x/a.py", b"x = 1") is None
