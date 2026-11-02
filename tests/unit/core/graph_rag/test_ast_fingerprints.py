"""Subtree matching: exact join recall, alpha-renaming and bounded failure."""

import itertools

import numpy as np
import pytest

from victor.core.graph_rag.ast_fingerprints import NormalizedNode as N
from victor.core.graph_rag.ast_fingerprints import jaccard_pairs, subtree_fingerprints
from victor.core.graph_rag.semantic_clones import TooManyPairsError


def test_alpha_renaming_preserves_repetition_patterns_in_every_subtree():
    a = N("add", (N("v0"), N("multiply", (N("v1"), N("v0")))))
    renamed = N("add", (N("v8"), N("multiply", (N("v3"), N("v8")))))
    different = N("add", (N("v0"), N("multiply", (N("v1"), N("v2")))))
    assert subtree_fingerprints(a, "rust") == subtree_fingerprints(renamed, "rust")
    assert subtree_fingerprints(a, "rust") != subtree_fingerprints(different, "rust")
    wrapped = N("block", (N("v7"), a))
    assert set(subtree_fingerprints(a, "rust")) <= set(subtree_fingerprints(wrapped, "rust"))


@pytest.mark.parametrize("threshold", [0.3, 0.5, 0.8, 1.0])
def test_prefix_join_matches_exhaustive_jaccard(threshold):
    rng = np.random.default_rng(42)
    sets = [
        {str(v) for v in rng.choice(30, size=int(rng.integers(1, 15)), replace=False)}
        for _ in range(80)
    ]
    expected = [
        (i, j, len(a & b) / len(a | b))
        for (i, a), (j, b) in itertools.combinations(enumerate(sets), 2)
        if len(a & b) / len(a | b) >= threshold
    ]
    assert jaccard_pairs(sets, threshold) == expected


def test_candidate_budget_and_empty_sets():
    assert jaccard_pairs([[], []], 0.8) == []
    with pytest.raises(TooManyPairsError):
        jaccard_pairs([{"same"}] * 20, 0.8, max_candidates=3)


@pytest.mark.parametrize("threshold", [0, -1, float("nan"), 2])
def test_invalid_threshold_is_rejected(threshold):
    with pytest.raises(ValueError):
        jaccard_pairs([], threshold)


@pytest.mark.parametrize("same_file", [False, True])
def test_group_compression_preserves_exhaustive_components(same_file):
    from victor.core.graph_rag.structural_clones import FunctionUnit, _subtree_pairs, near_families

    rng = np.random.default_rng(21)
    units = []
    for group in range(3):
        for i in range(12):
            prints = ("shared", *(f"{group}-{k}" for k in range(4)))
            units.append(
                FunctionUnit(
                    f"{rng.integers(0, 4)}.rs",
                    f"fn{i}",
                    1,
                    20,
                    (),
                    str(group),
                    str(group),
                    subtree_hashes=prints,
                    language="rust",
                )
            )
    expected = [
        (i, j, 0.1111111111111111)
        for i, left in enumerate(units)
        for j, right in enumerate(units[i + 1 :], i + 1)
        if left.exact_hash != right.exact_hash and (same_file or left.file_path != right.file_path)
    ]
    actual = _subtree_pairs(units, 0.1, same_file)

    def members(pairs):
        return {
            frozenset((u.file_path, u.name) for u in f.members) for f in near_families(units, pairs)
        }

    assert members(actual) == members(expected)
    assert len(actual) < len(expected)


def test_two_file_bipartite_groups_remain_two_components():
    from victor.core.graph_rag.structural_clones import FunctionUnit, _subtree_pairs, near_families

    units = [
        FunctionUnit(
            file,
            f"{group}-{file}",
            1,
            20,
            (),
            group,
            group,
            subtree_hashes=("common", group),
            language="rust",
        )
        for group in ("a", "b")
        for file in ("x.rs", "y.rs")
    ]
    families = near_families(units, _subtree_pairs(units, 0.3, False))
    assert len(families) == 2
    assert {frozenset(u.name for u in f.members) for f in families} == {
        frozenset(("a-x.rs", "b-y.rs")),
        frozenset(("a-y.rs", "b-x.rs")),
    }
