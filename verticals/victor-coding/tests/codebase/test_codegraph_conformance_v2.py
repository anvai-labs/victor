from __future__ import annotations

import pytest

victor_codegraph = pytest.importorskip("victor_codegraph")

from victor_coding.codebase.tree_sitter_analysis import TreeSitterAnalysisProvider


def test_analysis_provider_preserves_canonical_symbol_and_relation_ids():
    provider = TreeSitterAnalysisProvider()
    source = b"def caller():\n    return target()\n\ndef target():\n    return 1\n"
    canonical = victor_codegraph.parse(source.decode(), file_path="m.py")

    symbols = provider.extract_symbols(source, "python", file_path="m.py")
    assert {s["symbol_id"] for s in symbols} == {s.id for s in canonical.symbols}

    edges = provider.extract_edges(source, "python", file_path="m.py")
    canonical_call = next(r for r in canonical.relations if r.relation_type.name == "CALLS")
    delegated_call = next(e for e in edges if e["edge_type"] == "CALLS")
    assert delegated_call["source_id"] == canonical_call.from_symbol_id
    assert delegated_call["target_id"] == canonical_call.to_symbol_id
