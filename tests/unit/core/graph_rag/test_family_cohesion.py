"""Family explanations describe actual vector geometry, including weak links."""

import numpy as np
import pytest

from victor.core.graph_rag.family_cohesion import explain_vector_family


def test_transitive_family_exposes_a_direct_below_threshold_witness():
    theta = np.array([0, 0.2, 0.45])
    matrix = np.column_stack((np.cos(theta), np.sin(theta)))
    result = explain_vector_family(["a", "b", "c"], matrix, [0, 1, 2])
    assert result.representative_id == "b"
    assert result.witness_ids == ("a", "c")
    assert result.witness_similarity == pytest.approx(np.cos(0.45))
    assert result.witness_similarity < 0.95
    assert result.representative_min_similarity > 0.95
    assert result.member_pairs == 3
    assert result.mean_pair_similarity == pytest.approx(
        (np.cos(0.2) + np.cos(0.25) + np.cos(0.45)) / 3
    )


@pytest.mark.parametrize("seed", range(20))
def test_linear_summary_matches_independent_all_pairs_oracle(seed):
    matrix = np.random.default_rng(seed).normal(size=(11, 7))
    matrix /= np.linalg.norm(matrix, axis=1)[:, None]
    ids = [f"member-{i:02}" for i in range(len(matrix))]
    result = explain_vector_family(ids, matrix, list(range(len(matrix))), max_block_bytes=224)
    pair_scores = matrix @ matrix.T
    averages = (pair_scores.sum(axis=1) - np.diag(pair_scores)) / (len(matrix) - 1)
    representative = int(np.argmax(averages))
    assert result.representative_id == ids[representative]
    assert result.representative_mean_similarity == pytest.approx(averages[representative])
    assert result.mean_pair_similarity == pytest.approx(
        pair_scores[np.triu_indices(len(matrix), 1)].mean()
    )
    left, right = [ids.index(value) for value in result.witness_ids]
    assert left != right
    assert result.witness_similarity == pytest.approx(pair_scores[left, right])
    assert result.witness_similarity <= result.representative_min_similarity + 1e-12


def test_ties_and_input_permutation_preserve_the_same_explanation():
    matrix = np.ones((4, 3))
    ids = ["d", "a", "c", "b"]
    first = explain_vector_family(ids, matrix, [0, 1, 2, 3])
    second = explain_vector_family(ids[::-1], matrix[::-1], [0, 1, 2, 3])
    assert first == second
    assert first.representative_id == "a"
    assert first.witness_ids == ("a", "b")


def test_balanced_family_ties_do_not_depend_on_block_budget():
    random = np.random.default_rng(20261102)
    random.normal(scale=0.01, size=100)
    theta = 0.3 + random.normal(scale=0.01, size=1000)
    positive = np.column_stack((np.cos(theta), np.sin(theta)))
    matrix = np.vstack((positive, -positive))
    ids = [f"{i:06}" for i in range(len(matrix))]
    for budget in (16, 256, 8192, 8 * 1024 * 1024):
        result = explain_vector_family(
            ids, matrix, list(range(len(matrix))), max_block_bytes=budget
        )
        assert result.representative_id == "000000"
        assert result.representative_mean_similarity == pytest.approx(-1 / 1999, abs=1e-12)


def test_large_family_never_allocates_a_full_pair_matrix(monkeypatch):
    matrix = np.ones((1000, 16), dtype=np.float32)
    original = np.array
    sizes = []

    def array(*args, **kwargs):
        result = original(*args, **kwargs)
        sizes.append(result.nbytes)
        return result

    monkeypatch.setattr(np, "array", array)
    result = explain_vector_family(
        [str(i) for i in range(len(matrix))], matrix, list(range(len(matrix))), max_block_bytes=4096
    )
    assert result.member_pairs == 1000 * 999 // 2
    assert sizes and max(sizes) <= 4096


@pytest.mark.parametrize(
    "ids,matrix,rows,budget",
    [
        (["a"], [[1, 0]], [0], 1024),
        (["a", "a"], [[1, 0], [0, 1]], [0, 1], 1024),
        (["a", "b"], [[1, 0], [0, 1]], [0, 0], 1024),
        (["a", "b"], [[1, 0], [0, 1]], [0, 2], 1024),
        (["a", "b"], [[1, 0], [0, 1]], [0, 1], 8),
        (["a", "b"], [[1, 0], [0, 0]], [0, 1], 1024),
        (["a", "b"], [[1, 0], [np.nan, 1]], [0, 1], 1024),
    ],
)
def test_invalid_or_ambiguous_members_are_rejected(ids, matrix, rows, budget):
    with pytest.raises(ValueError):
        explain_vector_family(ids, np.asarray(matrix), rows, max_block_bytes=budget)
