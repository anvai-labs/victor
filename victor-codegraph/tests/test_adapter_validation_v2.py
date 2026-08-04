from __future__ import annotations

import pytest

from victor_codegraph import parse, to_proxima_records


def test_batch_embedder_cardinality_mismatch_fails():
    parsed = parse("def a(): pass\ndef b(): pass\n", file_path="m.py")
    with pytest.raises(ValueError, match="one vector per symbol"):
        to_proxima_records(parsed, "repo", batch_embedder=lambda texts: [[0.0] * 3], dim=3)


def test_embedding_dimension_mismatch_fails():
    parsed = parse("def a(): pass\n", file_path="m.py")
    with pytest.raises(ValueError, match="dimension"):
        to_proxima_records(parsed, "repo", embedder=lambda text: [0.0, 1.0], dim=3)


def test_non_finite_embedding_fails_before_storage():
    parsed = parse("def a(): pass\n", file_path="m.py")
    with pytest.raises(ValueError, match="finite numeric"):
        to_proxima_records(parsed, "repo", embedder=lambda text: [0.0, float("nan")], dim=2)


def test_repeated_calls_are_one_edge_with_all_call_sites():
    parsed = parse("def a():\n    b()\n    b()\n\ndef b(): pass\n", file_path="m.py")
    records = to_proxima_records(parsed, "repo")
    calls = [
        r for r in records if "graph_edge" in r["labels"] and r["edge"]["edge_type"] == "CALLS"
    ]
    assert len(calls) == 1
    assert [site["line"] for site in calls[0]["props"]["call_sites"]] == [2, 3]


def test_legacy_oid_mode_keeps_edge_endpoints_attached():
    parsed = parse("def a(): return b()\n\ndef b(): return 1\n", file_path="m.py")
    records = to_proxima_records(parsed, "repo", stable_oid=False)
    nodes = {r["oid"] for r in records if "graph_node" in r["labels"]}
    edge = next(r for r in records if "graph_edge" in r["labels"])
    assert edge["edge"]["from_oid"] in nodes
    assert edge["edge"]["to_oid"] in nodes
