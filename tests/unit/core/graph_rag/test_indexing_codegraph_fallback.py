"""ADR-015 Phase 1: the graph-RAG symbol-extraction fallback delegates to
victor-codegraph when available (real AST) instead of the def/class regex.

Guarded with importorskip so the suite stays green where the optional package is not
installed; CI installs it (foundational), so this runs there.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("victor_codegraph")

from victor.core.graph_rag.indexing import GraphIndexingPipeline  # noqa: E402

SRC = "def a():\n    return b()\n\n\nclass C:\n    def meth(self):\n        return 2\n"


def _fallback(src: str, path: str = "m.py", language: str = "python"):
    # The method uses no instance state; call it unbound to avoid constructing the
    # heavy pipeline.
    fn = GraphIndexingPipeline.__dict__["_extract_symbols_fallback"]
    return fn(object(), src, Path(path), language)


def test_fallback_uses_victor_codegraph_ast_not_regex():
    nodes = _fallback(SRC)
    names = {n.name for n in nodes}
    # The old regex only captured top-level def/class (a, C). Real AST also yields the
    # method, proving delegation is active.
    assert {"a", "C", "meth"} <= names
    assert any(n.type == "method" for n in nodes)


def test_fallback_preserves_codegraph_v2_node_identity():
    from victor_codegraph import parse

    nodes = _fallback(SRC)
    a = next(n for n in nodes if n.name == "a")
    canonical = parse(SRC, file_path="m.py")
    expected = next(symbol.id for symbol in canonical.symbols if symbol.simple_name == "a")
    assert a.node_id == expected
