"""Bounded vector geometry for transitive candidate families, not equivalence.

The sum of unit vectors gives each member's mean cosine to all other members
without enumerating pairs. Two farthest-member sweeps provide an actual pair
witness, not a certified global minimum. Inputs remain immutable during analysis.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


@dataclass(frozen=True)
class VectorFamilyExplanation:
    representative_id: str
    representative_mean_similarity: float
    representative_min_similarity: float
    mean_pair_similarity: float
    member_pairs: int
    witness_ids: tuple[str, str]
    witness_similarity: float
    scope: str = "all_component_member_pairs"
    representative_method: str = "maximum_mean_cosine_member"
    witness_method: str = "two_farthest_member_sweeps"
    pairwise_minimum_verified: bool = False


def _blocks(
    matrix: Any, rows: Sequence[int], batch_rows: int
) -> Iterator[tuple[int, NDArray[np.float64]]]:
    import numpy as np

    for start in range(0, len(rows), batch_rows):
        block = np.array(
            matrix[list(rows[start : start + batch_rows])], dtype=np.float64, copy=True
        )
        norms = np.linalg.norm(block, axis=1)
        if not np.isfinite(block).all() or not np.isfinite(norms).all() or (norms == 0).any():
            raise ValueError("Family vectors must have finite nonzero norms")
        block /= norms[:, None]
        yield start, block


def explain_vector_family(
    member_ids: Sequence[str],
    matrix: Any,
    rows: Sequence[int],
    *,
    max_block_bytes: int = 8 * 1024 * 1024,
) -> VectorFamilyExplanation:
    """Describe all members, independently of retrieval file/language filters.

    Four O(n*d) passes use a bounded float64 block plus O(n+d) metadata. The block
    budget excludes the existing input matrix and indexing temporaries. Mean
    cosines use normalized float64 values; retrieval retains its own precision.
    Canonical ID ordering and a 1e-12 tie tolerance make representative selection
    stable. A direct witness below a threshold disproves all-pairs cohesion; a
    witness above it does not prove every other pair exceeds that threshold.
    """
    import numpy as np

    raw = np.asarray(matrix)
    n = len(member_ids)
    if (
        not 2 <= n <= 200_000
        or len(rows) != n
        or any(not isinstance(value, str) or not value for value in member_ids)
        or len(set(member_ids)) != n
        or raw.ndim != 2
        or not 1 <= raw.shape[1] <= 65_536
        or raw.dtype.kind not in "fiu"
        or any(type(row) is not int or not 0 <= row < len(raw) for row in rows)
        or len(set(rows)) != n
        or type(max_block_bytes) is not int
        or max_block_bytes < raw.shape[1] * 8
    ):
        raise ValueError("Invalid family members, vector shape or block budget")
    order = sorted(range(n), key=lambda i: member_ids[i])
    ids = [member_ids[i] for i in order]
    selected = [rows[i] for i in order]
    batch_rows = max_block_bytes // (raw.shape[1] * 8)
    total = np.zeros(raw.shape[1], dtype=np.float64)
    diagonal = 0.0
    for _, block in _blocks(raw, selected, batch_rows):
        total += block.sum(axis=0)
        diagonal += float(np.einsum("ij,ij->", block, block))

    best, best_score = 0, float("-inf")
    for start, block in _blocks(raw, selected, batch_rows):
        scores = (block @ total - np.einsum("ij,ij->i", block, block)) / (n - 1)
        for offset, score in enumerate(scores):
            if score > best_score + 1e-12:
                best, best_score = start + offset, float(score)

    def minimum_from(index: int) -> tuple[int, float]:
        vector = np.array(raw[selected[index]], dtype=np.float64, copy=True)
        vector /= np.linalg.norm(vector)
        worst, score = -1, float("inf")
        for start, block in _blocks(raw, selected, batch_rows):
            values = block @ vector
            if start <= index < start + len(block):
                values[index - start] = np.inf
            for offset, value in enumerate(values):
                if value < score - 1e-12:
                    worst, score = start + offset, float(value)
        return worst, float(np.clip(score, -1, 1))

    first, representative_min = minimum_from(best)
    second, witness = minimum_from(first)
    left, right = sorted((ids[first], ids[second]))
    return VectorFamilyExplanation(
        representative_id=ids[best],
        representative_mean_similarity=float(np.clip(best_score, -1, 1)),
        representative_min_similarity=representative_min,
        mean_pair_similarity=float(np.clip((total @ total - diagonal) / (n * (n - 1)), -1, 1)),
        member_pairs=n * (n - 1) // 2,
        witness_ids=(left, right),
        witness_similarity=witness,
    )
