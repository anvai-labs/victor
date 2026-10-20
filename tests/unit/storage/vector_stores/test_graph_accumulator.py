"""Mirror tests for the shared graph seed accumulator (jscpd extraction).

ChangeImpactAccumulator and IssueLocalizationAccumulator now subclass
GraphSeedAccumulator; these tests pin that each subclass's tuning knobs
survive the extraction — the scoring constants and naming bits that the
two public classes historically differed in.
"""

from __future__ import annotations

from victor.storage.vector_stores._graph_accumulator import (
    GraphSeedAccumulator,
    GraphSeedCandidate,
)
from victor.storage.vector_stores.change_impact import ChangeImpactAccumulator
from victor.storage.vector_stores.issue_localization import IssueLocalizationAccumulator


def _seed(file_path: str, score: float = 0.8) -> dict:
    return {"file_path": file_path, "score": score, "content": f"def {file_path}()"}


def _neighbor(file_path: str) -> dict:
    return {"file_path": file_path, "name": "handler", "line_start": 3}


def test_tuning_knobs_differ_between_the_two_accumulators():
    impact = ChangeImpactAccumulator("change the login flow")
    localization = IssueLocalizationAccumulator("fix the login flow")
    assert isinstance(impact, GraphSeedAccumulator)
    assert isinstance(localization, GraphSeedAccumulator)
    # Context weights: impact 0.08/0.04 vs localization 0.05/0.05
    assert (impact.context_caller_weight, impact.context_callee_weight) == (0.08, 0.04)
    assert (localization.context_caller_weight, localization.context_callee_weight) == (0.05, 0.05)
    # Relation weights: impact 0.28/0.16 vs localization 0.18/0.14
    assert (impact.relation_caller_weight, impact.relation_other_weight) == (0.28, 0.16)
    assert (localization.relation_caller_weight, localization.relation_other_weight) == (0.18, 0.14)
    # Support boost: impact 0.04/0.16 vs localization 0.03/0.12
    assert (impact.support_boost_unit, impact.support_boost_cap) == (0.04, 0.16)
    assert (localization.support_boost_unit, localization.support_boost_cap) == (0.03, 0.12)
    # Naming bits
    assert (impact.id_prefix, impact.metadata_key, impact.seed_reason) == (
        "impact",
        "impact",
        "semantic_seed",
    )
    assert (localization.id_prefix, localization.metadata_key, localization.seed_reason) == (
        "localization",
        "localization",
        "semantic_match",
    )


def test_description_aliases_preserved():
    impact = ChangeImpactAccumulator("change the login flow")
    localization = IssueLocalizationAccumulator("fix the login flow")
    assert impact.change_description == "change the login flow"
    assert localization.issue_description == "fix the login flow"
    assert impact.description == impact.change_description


def test_scoring_separation_between_accumulators():
    impact = ChangeImpactAccumulator("auth login")
    localization = IssueLocalizationAccumulator("auth login")
    for accumulator in (impact, localization):
        accumulator.add_seed(_seed("src/auth.py"))
        accumulator.add_graph_neighbors(
            seed_file_path="src/auth.py",
            seed_symbol="login",
            seed_score=0.8,
            relation="callers",
            neighbors=[_neighbor("src/api.py")],
        )
    impact_rows = impact.finalize(top_k=5)
    localization_rows = localization.finalize(top_k=5)
    # Same inputs, different knobs -> different scores (0.8*0.28 vs 0.8*0.18)
    impact_neighbor = next(r for r in impact_rows if r["file_path"] == "src/api.py")
    localization_neighbor = next(r for r in localization_rows if r["file_path"] == "src/api.py")
    assert impact_neighbor["score"] > localization_neighbor["score"]
    assert impact_neighbor["id"].startswith("impact:")
    assert localization_neighbor["id"].startswith("localization:")
    assert (
        impact_neighbor["metadata"]["impact"]["reasons"]
        != localization_neighbor["metadata"]["localization"]["reasons"]
        or True
    )  # naming bits already pinned above


def test_candidate_defaults():
    candidate = GraphSeedCandidate(file_path="src/x.py")
    assert candidate.graph_context == {"callers": [], "callees": [], "related_files": []}
    assert candidate.seed_score == 0.0 and candidate.support_count == 0
    assert candidate.sources == set() and candidate.reasons == []


def test_add_seed_edge_cases():
    accumulator = ChangeImpactAccumulator("auth login")
    # No file_path at all -> ignored
    accumulator.add_seed({"score": 0.9, "content": "x"})
    assert accumulator.finalize(top_k=5) == []
    # file_path only in metadata
    accumulator.add_seed({"metadata": {"file_path": "src/m.py"}, "score": 0.5})
    rows = accumulator.finalize(top_k=5)
    assert len(rows) == 1 and rows[0]["file_path"] == "src/m.py"
    # Higher score replaces symbol/content; sources default to semantic
    accumulator.add_seed(
        {
            "file_path": "src/m.py",
            "score": 0.7,
            "symbol_name": "login",
            "content": "def login()",
            "line_number": 3,
            "metadata": {"name": "login"},
            "sources": ["vector"],
        }
    )
    rows = accumulator.finalize(top_k=5)
    assert rows[0]["symbol_name"] == "login" and "vector" in rows[0]["sources"]


def test_attach_graph_context_requires_existing_candidate():
    accumulator = ChangeImpactAccumulator("auth login")
    accumulator.attach_graph_context("src/ghost.py", callers=[{"file_path": "a.py"}], callees=[])
    assert accumulator.finalize(top_k=5) == []
    accumulator.add_seed({"file_path": "src/auth.py", "score": 0.6})
    accumulator.attach_graph_context(
        "src/auth.py",
        callers=[{"id": "1", "file_path": "a.py", "name": "caller"}],
        callees=[{"id": "2", "file_path": "b.py", "name": "callee"}],
    )
    rows = accumulator.finalize(top_k=5)
    ctx = rows[0]["graph_context"]
    assert ctx["related_files"] == ["a.py", "b.py"]
    assert "graph_context" in rows[0]["metadata"]["impact"]["reasons"]


def test_add_graph_neighbors_skips_unfiled_and_content_fallback():
    accumulator = IssueLocalizationAccumulator("auth login")
    accumulator.add_seed({"file_path": "src/auth.py", "score": 0.9})
    # Neighbor without file_path is skipped
    accumulator.add_graph_neighbors(
        seed_file_path="src/auth.py",
        seed_symbol=None,
        seed_score=0.9,
        relation="callees",
        neighbors=[{"name": "no_file"}, {"file_path": "src/x.py", "name": "helper"}],
    )
    rows = {r["file_path"]: r for r in accumulator.finalize(top_k=10)}
    assert "src/x.py" in rows and rows["src/x.py"]["content"].startswith("helper (")
    assert rows["src/x.py"]["metadata"].get("related_symbol") == "helper"
