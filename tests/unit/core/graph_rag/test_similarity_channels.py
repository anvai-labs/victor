"""Independent retrieval and explicit failures in multi-channel analysis."""

import hashlib
import pytest

pytest.importorskip("victor_codegraph.embeddings")

from victor.core.graph_rag.body_clones import BodyOptions
from victor.core.graph_rag.similarity_analysis import analyze_source


def corpus(root):
    for name in ("a.py", "b.py"):
        (root / name).write_text("def compute(x):\n value=x+1\n return value*3+4\n")


def token_report(root):
    return {
        "coverage_complete": True,
        "evidence": "token_sequence",
        "fragments": [],
        "source_manifest": {
            p.name: {"sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in root.glob("*.py")
        },
    }


def test_body_failure_preserves_other_channels(tmp_path, monkeypatch):
    from victor.core.graph_rag import body_clones, token_clones

    corpus(tmp_path)

    def unavailable(*args, **kwargs):
        raise OSError("local weights unavailable")

    monkeypatch.setattr(body_clones, "analyze_bodies", unavailable)
    monkeypatch.setattr(token_clones, "scan_tokens", lambda root, **kwargs: token_report(root))
    result = analyze_source(tmp_path, mode="hybrid", min_nodes=5)
    assert not result["coverage_complete"]
    assert result["channels"]["structural"]["families"]
    assert result["channels"]["tokens"]["coverage_complete"]
    assert "local weights" in result["channels"]["body"]["error"]


def test_snapshot_conflict_prevents_a_clean_union(tmp_path, monkeypatch):
    from victor.core.graph_rag import body_clones, token_clones

    corpus(tmp_path)
    monkeypatch.setattr(
        body_clones, "analyze_bodies", lambda *a, **k: {"coverage_complete": True, "families": []}
    )

    def changed(root, **kwargs):
        payload = token_report(root)
        payload["source_manifest"]["a.py"]["sha256"] = "f" * 64
        return payload

    monkeypatch.setattr(token_clones, "scan_tokens", changed)
    result = analyze_source(tmp_path, mode="hybrid", min_nodes=5)
    assert not result["coverage_complete"]
    assert result["snapshot_conflicts"] == ["a.py"]


def test_body_retrieval_is_not_prefiltered_by_syntax_matches(tmp_path, monkeypatch):
    from victor.core.graph_rag import body_clones, token_clones

    (tmp_path / "a.py").write_text("def first(x):\n return x+1\n")
    (tmp_path / "b.py").write_text(
        "def second(xs):\n result=[]\n for value in xs:\n  result.append(value*2)\n return result\n"
    )
    seen = []

    def body(report, **kwargs):
        seen.extend(report.units)
        assert not report.families
        return {"coverage_complete": True, "families": [{"relation": "body_similarity_candidate"}]}

    monkeypatch.setattr(body_clones, "analyze_bodies", body)
    monkeypatch.setattr(token_clones, "scan_tokens", lambda root, **kwargs: token_report(root))
    result = analyze_source(tmp_path, mode="hybrid", min_nodes=5, near_threshold=1)
    assert len(seen) == 2
    assert all(unit.source_code for unit in seen)
    assert result["channels"]["body"]["families"]


def test_body_language_scope_filters_edges_before_grouping():
    import numpy as np
    from victor.core.graph_rag.semantic_clones import SymbolRecord, find_clone_families

    records = [SymbolRecord(str(i), f"{i}.code", "f", "function", 1) for i in range(3)]
    families = find_clone_families(
        records,
        np.ones((3, 3)),
        threshold=0.9,
        language_ids=["python", "python", "rust"],
        language_scope="cross",
    )
    assert len(families) == 1
    assert families[0].matching_pairs == 2
    assert BodyOptions().identity == BodyOptions(comparison_scope="cross").identity


def test_syntax_pair_budget_does_not_block_other_channels(tmp_path, monkeypatch):
    from victor.core.graph_rag import body_clones, token_clones, structural_clones
    from victor.core.graph_rag.semantic_clones import TooManyPairsError

    corpus(tmp_path)

    def exhausted(*a, **k):
        raise TooManyPairsError("controlled syntax budget")

    monkeypatch.setattr(structural_clones, "_subtree_pairs", exhausted)
    monkeypatch.setattr(
        body_clones,
        "analyze_bodies",
        lambda report, **k: {
            "coverage_complete": True,
            "families": [],
            "functions": len(report.units),
        },
    )
    monkeypatch.setattr(token_clones, "scan_tokens", lambda root, **k: token_report(root))
    result = analyze_source(tmp_path, mode="hybrid", min_nodes=5, near_metric="subtree")
    assert not result["coverage_complete"]
    assert result["channels"]["structural"]["stats"]["matching_budget_exceeded"] == 1
    assert result["channels"]["body"]["coverage_complete"]
    assert result["channels"]["body"]["functions"] == 2
    assert result["channels"]["tokens"]["coverage_complete"]
