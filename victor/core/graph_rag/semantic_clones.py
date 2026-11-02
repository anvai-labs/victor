"""Semantic near-duplicate discovery over the indexed symbol embeddings.

Reads the LanceDB embedding store produced by ``victor graph index
--embeddings``, computes cross-file nearest neighbours per symbol, and
groups summary-similarity candidates. This does not compare bodies or prove
functional equivalence; identical summaries can describe different behavior. Read-only analysis; surfaced as
``victor graph similar``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_THRESHOLD = 0.90


@dataclass(frozen=True)
class SimilarPair:
    left_idx: int
    right_idx: int
    similarity: float


@dataclass
class SymbolRecord:
    id: str
    file_path: str
    symbol_name: str
    kind: str
    line_number: int
    input_hash: Optional[str] = None


@dataclass
class CloneFamily:
    similarity: float
    members: List[SymbolRecord] = field(default_factory=list)
    min_edge_similarity: Optional[float] = None
    matching_pairs: int = 0

    @property
    def files(self) -> List[str]:
        return sorted({m.file_path for m in self.members})


TEST_PATH_MARKERS = ("/tests/", "/test_", "/__tests__/", "/goldens/")


def _in_scope(file_path: str, scope: Optional[str], include_tests: bool) -> bool:
    if scope and not file_path.startswith(scope):
        return False
    if not include_tests and any(
        m in "/" + file_path.replace("\\", "/") for m in TEST_PATH_MARKERS
    ):
        return False
    return True


def _esc(value: str) -> str:
    return value.replace("'", "''")


def _scope_predicate(scope: Optional[str], include_tests: bool) -> str:
    """SQL WHERE for the store query: degenerate-name + scope + test filters.

    Pushed into LanceDB — a whole-repo store holds ~1M rows and
    materializing them client-side to filter costs minutes and gigabytes
    the pushdown avoids.
    """
    conds = ["symbol_name NOT LIKE '%:%'"]  # degenerate statement names
    if scope:
        conds.append(f"starts_with(file_path, '{_esc(scope)}')")
    if not include_tests:
        for marker in TEST_PATH_MARKERS:
            pattern = marker.replace("_", "\\_")
            conds.append(f"file_path NOT LIKE '%{_esc(pattern)}%' ESCAPE '\\'")
            conds.append(f"NOT starts_with(file_path, '{_esc(marker.lstrip('/'))}')")
    return " AND ".join(conds)


def load_embedding_symbols(
    project_root,
    scope: Optional[str] = None,
    include_tests: bool = False,
) -> Tuple[List[SymbolRecord], Any]:
    """Load (symbol records, vector matrix) from the project's LanceDB store.

    ``scope`` filters by file-path prefix; test trees are excluded unless
    ``include_tests`` — a whole-repo store embeds every symbol, and the
    duplicate scan wants source trees. Raises FileNotFoundError when the
    store is absent or empty — the caller turns that into the reindex hint.
    """
    import json
    import os
    import re

    import lancedb
    import numpy as np
    from victor_codegraph.embeddings import EmbeddingProfile

    store_dir = os.path.join(str(project_root), ".victor", "embeddings")
    if not os.path.isdir(store_dir):
        raise FileNotFoundError(store_dir)
    try:
        table = lancedb.connect(store_dir).open_table("embeddings")
    except FileNotFoundError:
        raise FileNotFoundError(store_dir) from None
    schema = table.schema
    required = {
        "id",
        "file_path",
        "symbol_name",
        "line_number",
        "vector",
        "content",
        "embedding_kind",
        "embedding_profile_id",
    }
    raw_profile = (schema.metadata or {}).get(b"victor_codegraph.embedding_profile")
    if not required.issubset(schema.names) or raw_profile is None:
        raise ValueError(
            "Embedding evidence/profile is missing; rebuild the index before semantic analysis"
        )
    try:
        profile = EmbeddingProfile(**json.loads(raw_profile))
    except (TypeError, UnicodeError, ValueError) as exc:
        raise ValueError("Invalid embedding model profile; rebuild the index") from exc
    if getattr(schema.field("vector").type, "list_size", None) != profile.dimension:
        raise ValueError("Embedding vector dimension conflicts with its model profile")
    columns = sorted(required | ({"embedding_input_hash"} & set(schema.names)))
    # Arrow keeps vector buffers contiguous. Converting vectors through Python
    # lists previously allocated hundreds of scalar Python objects per symbol.
    data = (
        table.search()
        .where(_scope_predicate(scope, include_tests))
        .select(columns)
        .limit(MAX_SYMBOLS + 1)
        .to_arrow()
    )
    if len(data) > MAX_SYMBOLS:
        raise ValueError(f"Semantic scan exceeds {MAX_SYMBOLS:,} symbols; narrow --scope")
    if not len(data):
        raise FileNotFoundError(store_dir)
    matrix = (
        data.column("vector")
        .combine_chunks()
        .values.to_numpy(zero_copy_only=False)
        .reshape(len(data), profile.dimension)
    )
    metadata = data.drop(["vector"]).to_pylist()
    records, indices = [], []
    for i, row in enumerate(metadata):
        if row["embedding_kind"] != "semantic" or row["embedding_profile_id"] != profile.identity:
            raise ValueError("Mixed or non-semantic embedding profiles; rebuild in a fresh index")
        if not _in_scope(row["file_path"], scope, include_tests) or ":" in row["symbol_name"]:
            continue
        if not include_tests and re.search(
            r"#\[\s*(?:cfg\s*\(\s*test\b|(?:[\w]+::)*test\b)", row["content"] or ""
        ):
            continue
        records.append(
            SymbolRecord(
                row["id"],
                row["file_path"],
                row["symbol_name"],
                row["embedding_kind"],
                row["line_number"] or 0,
                row.get("embedding_input_hash"),
            )
        )
        indices.append(i)
    if not records:
        raise FileNotFoundError(store_dir)
    matrix = np.asarray(matrix[indices], dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if not np.isfinite(matrix).all() or (norms == 0).any() or not np.isfinite(norms).all():
        raise ValueError("Embedding vectors must be finite and nonzero")
    return records, matrix / norms


MAX_SYMBOLS = 200_000
BLOCK_ROWS = 2048
# At 0.90 this repo yields ~280K pairs; at 0.5, 3.2B (79% of all pairs).
MAX_PAIRS = 5_000_000


class TooManyPairsError(ValueError):
    """The threshold matched more pairs than ``max_pairs`` allows."""


def make_union_find(n: int):
    """Tiny path-halving union-find over ``range(n)``.

    Shared with ``victor.native.python.graph_algo`` — one implementation,
    two callers (the native copy chose the opposite union root; component
    results are identical either way, so the shared form is safe).
    """
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    return find, union


def find_clone_families(
    records: List[SymbolRecord],
    mat: Any,
    threshold: float = DEFAULT_THRESHOLD,
    top_k: Optional[int] = None,
    include_same_file: bool = False,
    progress: Optional[Any] = None,
    max_pairs: int = MAX_PAIRS,
    max_similarity_bytes: int = 32 * 1024 * 1024,
    language_ids: Optional[List[str]] = None,
    language_scope: str = "all",
) -> List[CloneFamily]:
    """Connect threshold-matching summary vectors into candidate components.

    A family's score is its BEST matching edge, not its minimum pairwise
    similarity. Transitive members can be below threshold. Two-dimensional
    tiling bounds each cosine score matrix; default threshold joins union
    matching edges immediately instead of retaining all pairs. ``top_k`` is an
    optional lossy neighbor cap; its unique edge set is bounded by max_pairs.
    """
    import math
    import numpy as np

    n = len(records)
    if (
        language_scope not in {"all", "same", "cross"}
        or (language_ids is not None and len(language_ids) != n)
        or (language_scope != "all" and language_ids is None)
    ):
        raise ValueError("Invalid language comparison scope")
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be finite and between zero and one")
    if max_pairs < 0 or max_similarity_bytes < 4 or (top_k is not None and top_k < 1):
        raise ValueError("pair, workspace and neighbor budgets must be valid")
    matrix = np.asarray(mat, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != n or matrix.shape[1] == 0:
        raise ValueError("vector matrix shape does not match symbol records")
    if not np.isfinite(matrix).all():
        raise ValueError("vectors must be finite")
    norms = np.linalg.norm(matrix, axis=1)
    if (norms == 0).any() or not np.isfinite(norms).all():
        raise ValueError("vectors must have finite nonzero norms")
    matrix = matrix / norms[:, None]
    if n < 2:
        return []

    file_ids: Dict[str, int] = {}
    fid = np.fromiter(
        (file_ids.setdefault(r.file_path, len(file_ids)) for r in records), dtype=np.int64, count=n
    )
    language_codes: Dict[str, int] = {}
    lid = (
        np.array(
            [language_codes.setdefault(value, len(language_codes)) for value in language_ids],
            dtype=np.int64,
        )
        if language_ids is not None
        else None
    )
    find, union = make_union_find(n)
    # root -> (best edge, weakest matching edge, number of unique edges)
    components: Dict[int, Tuple[float, float, int]] = {}
    participating: set[int] = set()
    emitted: set[Tuple[int, int]] = set()
    total_pairs = 0

    def add_edge(i: int, j: int, similarity: float) -> None:
        nonlocal total_pairs
        if top_k is not None:
            edge = (min(i, j), max(i, j))
            if edge in emitted:
                return
            emitted.add(edge)
        total_pairs += 1
        if total_pairs > max_pairs:
            raise TooManyPairsError(
                f"threshold {threshold} matched over {max_pairs:,} pairs; raise the threshold or cap neighbours"
            )
        ra, rb = find(i), find(j)
        values = [components.pop(ra, (similarity, similarity, 0))]
        if ra != rb:
            values.append(components.pop(rb, (similarity, similarity, 0)))
        union(i, j)
        components[find(i)] = (
            max(similarity, *(v[0] for v in values)),
            min(similarity, *(v[1] for v in values)),
            1 + sum(v[2] for v in values),
        )
        participating.update((i, j))

    k = min(top_k, n - 1, max_pairs + 1) if top_k is not None else 0
    row_count = min(BLOCK_ROWS, max(1, max_similarity_bytes // max(4, k * 12)))
    column_count = max(1, max_similarity_bytes // (4 * row_count))
    for b0 in range(0, n, row_count):
        b1 = min(b0 + row_count, n)
        row_ids = np.arange(b0, b1)
        best_scores = np.full((b1 - b0, k), -np.inf, dtype=np.float32)
        best_ids = np.full((b1 - b0, k), -1, dtype=np.int64)
        for c0 in range(0 if top_k is not None else b0, n, column_count):
            c1 = min(c0 + column_count, n)
            col_ids = np.arange(c0, c1)
            sims = np.matmul(matrix[b0:b1], matrix[c0:c1].T)
            np.clip(sims, -1.0, 1.0, out=sims)
            allowed = (
                (row_ids[:, None] != col_ids[None, :])
                if top_k is not None
                else (row_ids[:, None] < col_ids[None, :])
            )
            if not include_same_file:
                allowed &= fid[b0:b1, None] != fid[None, c0:c1]
            if language_scope != "all" and lid is not None:
                same_language = lid[b0:b1, None] == lid[None, c0:c1]
                allowed &= same_language if language_scope == "same" else ~same_language
            if top_k is None:
                hit = allowed & (sims >= threshold)
                if total_pairs + int(np.count_nonzero(hit)) > max_pairs:
                    raise TooManyPairsError(
                        f"threshold {threshold} matched over {max_pairs:,} pairs; raise the threshold or cap neighbours"
                    )
                rows, cols = np.nonzero(hit)
                for r, c in zip(rows, cols):
                    add_edge(int(r + b0), int(c + c0), float(sims[r, c]))
            else:
                sims[~allowed] = -np.inf
                candidates = np.concatenate((best_scores, sims), axis=1)
                identities = np.concatenate(
                    (best_ids, np.broadcast_to(col_ids, sims.shape)), axis=1
                )
                chosen = np.argpartition(-candidates, k - 1, axis=1)[:, :k]
                best_scores = np.take_along_axis(candidates, chosen, axis=1)
                best_ids = np.take_along_axis(identities, chosen, axis=1)
        if top_k is not None:
            rows, cols = np.nonzero(best_scores >= threshold)
            for r, c in zip(rows, cols):
                add_edge(int(r + b0), int(best_ids[r, c]), float(best_scores[r, c]))
        if progress is not None:
            progress(b1, n)

    groups: Dict[int, List[int]] = {}
    for i in sorted(participating):
        groups.setdefault(find(i), []).append(i)
    families = [
        CloneFamily(
            similarity=components[root][0],
            min_edge_similarity=components[root][1],
            matching_pairs=components[root][2],
            members=sorted(
                (records[i] for i in members), key=lambda r: (r.file_path, r.line_number, r.id)
            ),
        )
        for root, members in groups.items()
    ]
    families.sort(key=lambda f: (-f.similarity, f.files, f.members[0].id))
    return families
