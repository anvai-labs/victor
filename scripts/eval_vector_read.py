#!/usr/bin/env python3
"""Vector read cost: LanceDB (SQLite arm) vs ProximaDB, same corpus.

This is the last unmeasured quadrant of the backend comparison, and the one
place a single-store design could still justify itself: ProximaDB co-locates
vectors with graph nodes, so a vector hit IS a node, while the SQLite arm must
search LanceDB and then join back into SQLite to turn ids into nodes.

So the measured operation is the product-level one — **query vector -> top-k
graph nodes** — not "ANN search" in isolation, which would hide the join the
SQLite arm cannot avoid. LanceDB's raw search time is reported separately so
the join cost is visible rather than merely included.

Two measurement hazards this deliberately avoids:

* `LanceDBProvider.search_similar` takes a STRING and embeds it internally,
  while ProximaDB's `semantic_search` takes a VECTOR. Timing those against each
  other would charge one arm for embedding generation and not the other. Both
  arms here are handed the same pre-computed vectors.
* Speed means nothing without correctness — an ANN index can always be fast by
  returning the wrong neighbours. Recall is scored against exact brute-force
  cosine over the same vectors, so a latency win bought with accuracy shows up.
"""

from __future__ import annotations

import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

WORKDIR = Path(sys.argv[1])
TOP_K = 10
N_QUERIES = 50


def brute_force_topk(matrix, query, k):
    """Exact cosine top-k — the ground truth both backends are scored against."""
    import numpy as np

    sims = matrix @ query
    idx = np.argpartition(-sims, min(k, len(sims) - 1))[:k]
    return list(idx[np.argsort(-sims[idx])])


async def main() -> None:
    import numpy as np

    import lancedb

    lance_dir = WORKDIR / "sqlite" / ".victor" / "embeddings"
    db = lancedb.connect(str(lance_dir))
    names = db.table_names()
    print(f"lance tables: {names}")
    table = db.open_table(names[0])
    rows = table.to_arrow().to_pylist()
    print(f"rows: {len(rows)}  fields: {sorted(rows[0].keys())[:8]}")

    vec_field = next((f for f in ("vector", "embedding") if f in rows[0]), None)
    id_field = next((f for f in ("id", "node_id", "oid", "symbol_id") if f in rows[0]), None)
    print(f"vector field={vec_field}  id field={id_field}")
    if not vec_field or not id_field:
        print("cannot identify schema; aborting rather than guessing")
        return

    ids = [r[id_field] for r in rows]
    mat = np.array([r[vec_field] for r in rows], dtype="float32")
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    mat = mat / norms
    print(f"matrix: {mat.shape}")

    rng = np.random.default_rng(1234)  # fixed seed: same queries for both arms
    picks = rng.choice(len(rows), size=min(N_QUERIES, len(rows)), replace=False)
    queries = [mat[i] for i in picks]
    truth = [set(ids[j] for j in brute_force_topk(mat, q, TOP_K)) for q in queries]

    report: dict = {"queries": len(queries), "top_k": TOP_K, "corpus_vectors": len(rows)}

    # --- Arm A: LanceDB search, then join back to SQLite for the nodes -------
    from victor.storage.graph.registry import create_graph_store

    sq = create_graph_store("sqlite", project_path=WORKDIR / "sqlite")
    await sq.initialize()

    raw_ms, full_ms, hits = [], [], []
    for q, want in zip(queries, truth):
        t0 = time.perf_counter()
        res = table.search(q).limit(TOP_K).to_list()
        raw = (time.perf_counter() - t0) * 1000
        got = [r.get(id_field) for r in res]
        # The join the SQLite arm cannot avoid: ids -> graph nodes.
        for nid in got:
            await sq.get_node_by_id(str(nid))
        full = (time.perf_counter() - t0) * 1000
        raw_ms.append(raw)
        full_ms.append(full)
        hits.append(len(set(got) & want) / len(want))
    report["lancedb"] = {
        "ann_p50_ms": round(statistics.median(raw_ms), 3),
        "ann_p95_ms": round(sorted(raw_ms)[int(len(raw_ms) * 0.95) - 1], 3),
        "with_join_p50_ms": round(statistics.median(full_ms), 3),
        "with_join_p95_ms": round(sorted(full_ms)[int(len(full_ms) * 0.95) - 1], 3),
        "recall_at_k": round(statistics.mean(hits), 3),
    }
    await sq.close()

    # --- Arm B: ProximaDB semantic_search (vector hit IS the node) ----------
    px = create_graph_store("proxima", project_path=WORKDIR / "proxima")
    await px.initialize()
    px_ms, px_hits, empties = [], [], 0
    for q, want in zip(queries, truth):
        t0 = time.perf_counter()
        nodes = await px.semantic_search(list(map(float, q)), top_k=TOP_K)
        px_ms.append((time.perf_counter() - t0) * 1000)
        got = {n.node_id for n in nodes}
        if not got:
            empties += 1
        px_hits.append(len(got & want) / len(want))
    report["proximadb"] = {
        "p50_ms": round(statistics.median(px_ms), 3),
        "p95_ms": round(sorted(px_ms)[int(len(px_ms) * 0.95) - 1], 3),
        "recall_at_k": round(statistics.mean(px_hits), 3),
        "empty_results": empties,
    }
    await px.close()

    print(json.dumps(report, indent=2))
    (WORKDIR / "vector_read.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
