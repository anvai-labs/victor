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


class TestCompleteCcgPhase2:
    """Engine phase 2: the complete CCG (CFG + CDG + DDG) natively."""

    def _native_result(self, path: Path, language: str):
        if not native_statements_available(language):
            pytest.skip("native extension not built")
        import victor_native

        return victor_native.build_ccg_batch([(language, str(path), path.read_bytes())])[0]

    def _python_result(self, path: Path, language: str):
        import asyncio

        builder = CodeContextGraphBuilder(language=language)
        tree = _parse_with(path, language)
        source = path.read_text()
        nodes = builder._extract_statement_nodes(
            tree.root_node, path, source.split("\n"), source_code=None
        )
        cfg = builder._build_cfg_edges(nodes, tree.root_node, path)
        cdg = asyncio.run(builder._build_cdg(nodes, cfg))
        ddg = asyncio.run(builder._build_ddg(tree.root_node, nodes, path, source))
        return nodes, cfg, cdg, ddg

    @pytest.mark.parametrize("fixture", ["victor/core/graph_rag/precise_providers.py"])
    def test_complete_ccg_matches_python_builders(self, fixture):
        """CFG in order; CDG as multiset (Python emits in hash-randomized
        set order — consumers upsert by key, order-insensitive by
        contract); DDG in order with variables."""
        from collections import Counter

        path = REPO_ROOT / fixture
        if not path.exists():
            pytest.skip(f"fixture not present: {fixture}")
        result = self._native_result(path, "python")
        p_nodes, p_cfg, p_cdg, p_ddg = self._python_result(path, "python")

        assert [r.node_id for r in result.statements] == [n.node_id for n in p_nodes]
        assert [(e.src, e.dst, e.edge_type) for e in result.cfg_edges] == [
            (e.src, e.dst, str(e.type.value)) for e in p_cfg
        ]
        assert Counter((e.src, e.dst, e.edge_type) for e in result.cdg_edges) == Counter(
            (e.src, e.dst, str(e.type.value)) for e in p_cdg
        )
        assert [(e.src, e.dst, e.variable) for e in result.ddg_edges] == [
            (e.src, e.dst, e.metadata.get("variable")) for e in p_ddg
        ]

    def test_integrated_native_path_equals_builtin_including_metadata(self):
        """Through _build_native_ccg: GraphNode/GraphEdge construction with
        the exact metadata shapes of the Python builders."""
        import asyncio
        from pathlib import Path as P

        path = REPO_ROOT / "victor/core/graph_rag/precise_providers.py"
        if not path.exists() or not native_statements_available("python"):
            pytest.skip("fixture or native extension unavailable")
        builder = CodeContextGraphBuilder(language="python")
        source = path.read_text()
        native = builder._build_native_ccg(P(path), "python", source)
        assert native is not None
        n_nodes, n_edges = native

        ast_root = builder._parse_source(source, P(path), "python")
        cfg_nodes, cfg_edges = asyncio.run(builder._build_cfg(ast_root, P(path), source))
        cdg = asyncio.run(builder._build_cdg(cfg_nodes, cfg_edges))
        ddg = asyncio.run(builder._build_ddg(ast_root, cfg_nodes, P(path), source))

        def t(e):
            return str(getattr(e.type, "value", e.type))

        def ekey(e):
            return (e.src, e.dst, t(e), e.weight, tuple(sorted((e.metadata or {}).items())))

        assert len(n_nodes) == len(cfg_nodes)
        assert [(n.node_id, n.name, n.line, n.statement_type) for n in n_nodes] == [
            (n.node_id, n.name, n.line, n.statement_type) for n in cfg_nodes
        ]
        # CFG + DDG in order; CDG as multiset (hash-randomized Python order).
        nat_cfg = [ekey(e) for e in n_edges if t(e).startswith("CFG")]
        nat_ddg = [ekey(e) for e in n_edges if t(e).startswith("DDG")]
        assert nat_cfg == [ekey(e) for e in cfg_edges]
        assert nat_ddg == [ekey(e) for e in ddg]
        from collections import Counter

        assert Counter(ekey(e) for e in n_edges if t(e).startswith("CDG")) == Counter(
            ekey(e) for e in cdg
        )

    def test_unsupported_language_ccg_falls_back(self):
        from victor.processing.native.ccg_extract import build_ccg, native_ccg_available

        assert native_ccg_available("go") is False
        assert build_ccg("go", "/x/main.go", b"package main") is None
