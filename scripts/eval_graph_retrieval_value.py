#!/usr/bin/env python3
# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0

"""Does graph traversal change retrieval quality?

Retrieval quality is a ranked, non-deterministic surface, so it needs an eval
with a rubric rather than an assertion. Agent-level benchmarks cannot answer
this question: if the model never calls the graph tool, pass-rate is
insensitive to the graph by construction, and "no difference" gets misread as
"the graph does not help". This measures the surface where traversal
*determines* the answer.

Task class: "who calls X" / "what breaks if I change X". Ground truth is
derived mechanically from CALLS edges — no LLM, no judgement: the relevant set
for target X is every node with a CALLS edge into X. Targets are filtered to
**unambiguous** names (exactly one function node carries the name) with callers
in >= 4 distinct files, so a name-matching baseline cannot win by accident and
a probe cannot land on a same-named node that has different edges.

Two arms over the same repo:
  * seed-only  — what a keyword/symbol retriever returns for the query. No
                 edges consulted. This is the floor the graph must beat.
  * seed+graph — MultiHopRetriever, the real product hot path: seed, expand
                 across edges, rank, truncate.

Reported: recall@k and precision@k per arm, plus the **attainable ceiling** —
a target with more true callers than k caps recall at k/|callers|, so raw
recall understates a retriever that returned everything it had room for.

Usage:
    python scripts/eval_graph_retrieval_value.py [REPO] [--direction in|out|both]
                                                 [--k 20] [--json OUT]

`--direction` is the knob this harness was written to measure. Outward-only
traversal scores 0.000 here by construction: a caller is never reachable by
following edges away from the target.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import statistics
import sys
import time
from pathlib import Path

MIN_CALLER_FILES = 4
MAX_TARGETS = 8


def golden_from_sqlite(db: Path, limit: int, min_caller_files: int) -> list[dict]:
    """Targets and their true caller sets, straight from CALLS edges."""
    conn = sqlite3.connect(db)
    rows = conn.execute(
        """
        SELECT n.node_id, n.name, n.file, COUNT(DISTINCT sn.file) cf
        FROM graph_edge e
        JOIN graph_node n ON n.node_id = e.dst
        JOIN graph_node sn ON sn.node_id = e.src
        WHERE e.type='CALLS' AND n.type='function' AND length(n.name) > 8
        GROUP BY e.dst
        HAVING cf >= ?
          AND (SELECT COUNT(*) FROM graph_node x
               WHERE x.name = n.name AND x.type='function') = 1
        ORDER BY cf DESC LIMIT ?
        """,
        (min_caller_files, limit),
    ).fetchall()

    golden = []
    for node_id, name, file, caller_files in rows:
        callers = {
            row[0]
            for row in conn.execute(
                "SELECT DISTINCT src FROM graph_edge WHERE dst=? AND type='CALLS'",
                (node_id,),
            )
        }
        golden.append(
            {
                "target": name,
                "node_id": node_id,
                "file": file,
                "caller_files": caller_files,
                "relevant": callers,
            }
        )
    conn.close()
    return golden


def score(retrieved: list[str], relevant: set[str], k: int) -> tuple[float, float]:
    top = retrieved[:k]
    if not relevant:
        return 0.0, 0.0
    hits = len(set(top) & relevant)
    return hits / len(relevant), (hits / len(top) if top else 0.0)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", nargs="?", default=".", type=Path)
    parser.add_argument("--direction", default="in", choices=("in", "out", "both"))
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--targets", type=int, default=MAX_TARGETS)
    parser.add_argument("--repeats", type=int, default=3, help="timed samples per query")
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument(
        "--backend",
        default="auto",
        help="graph backend to retrieve through; 'auto' honours the repo marker",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=None,
        help="indexed store to query (defaults to REPO). Point this at one arm of "
        "a benchmark workdir to A/B backends over an identically-indexed corpus.",
    )
    parser.add_argument(
        "--golden-db",
        type=Path,
        default=None,
        help="SQLite project.db supplying ground truth (defaults to REPO/.victor/"
        "project.db). In a backend A/B this stays FIXED across arms: both arms are "
        "scored against one edge set, so a difference is retrieval, not indexing.",
    )
    args = parser.parse_args()

    repo = args.repo.resolve()
    sys.path.insert(0, str(repo))

    from victor.core.graph_rag.config import RetrievalConfig
    from victor.core.graph_rag.retrieval import MultiHopRetriever
    from victor.storage.graph.registry import create_graph_store, resolve_graph_backend

    project = (args.project or repo).resolve()
    golden_db = (args.golden_db or repo / ".victor" / "project.db").resolve()
    backend = (
        args.backend if args.backend != "auto" else resolve_graph_backend(repo, default="sqlite")
    )

    golden = golden_from_sqlite(golden_db, args.targets, MIN_CALLER_FILES)
    print(f"repo={repo}  backend={backend}  direction={args.direction}")
    print(f"project={project}")
    print(f"golden={golden_db}")
    print(f"golden targets: {len(golden)} (unambiguous, >= {MIN_CALLER_FILES} caller files)\n")

    store = create_graph_store(args.backend, project_path=project)
    await store.initialize()

    def config_for(repeat: int) -> RetrievalConfig:
        """A config whose cache key is unique to this (direction, repeat).

        `mode` participates in the query-cache key, and the cache has an L2 layer
        persisted in `project.db` that a process-local reset does not clear. So
        rather than try to clear it, every timed call is given a key nothing has
        used: each measurement is genuinely cold. Without this a repeat measures
        a dictionary lookup and reports it as retrieval latency — a 0 ms result
        that once cost a wrong diagnosis in this codebase.
        """
        return RetrievalConfig(
            seed_count=5,
            max_hops=2,
            top_k=args.k,
            mode=f"eval-{args.direction}-r{repeat}",
            direction=args.direction,
        )

    retriever = MultiHopRetriever(store, config_for(0))

    # Warm the process and the store before anything is timed: the first query
    # of a run pays import, connection and page-cache costs that belong to
    # neither backend's steady state, and comparing a cold arm against a warm
    # one is not a comparison.
    if golden:
        await retriever.retrieve(f"who calls {golden[0]['target']}", config_for(-1))

    rows = []
    for target in golden:
        relevant = target["relevant"]

        started = time.perf_counter()
        seeds = await store.search_symbols(target["target"], limit=args.k)
        seed_ms = (time.perf_counter() - started) * 1000
        seed_ids = [s.node_id for s in (seeds or []) if getattr(s, "node_id", None)]
        recall_seed, prec_seed = score(seed_ids, relevant, args.k)

        # Repeat the graph arm: one sample per target cannot separate a backend
        # difference from scheduler noise. Recall is expected to be identical
        # across repeats; if it is not, that instability is itself the finding,
        # so it is reported rather than averaged away.
        graph_samples: list[float] = []
        recalls: set[float] = set()
        graph_ids: list[str] = []
        for repeat in range(args.repeats):
            started = time.perf_counter()
            try:
                result = await retriever.retrieve(
                    f"who calls {target['target']}", config_for(repeat)
                )
                graph_ids = [node.node_id for node in result.nodes]
            except Exception as exc:  # keep the sweep running; report the gap
                graph_ids = []
                print(f"  [{target['target']}] retrieve failed: {type(exc).__name__}: {exc}")
            graph_samples.append((time.perf_counter() - started) * 1000)
            recalls.add(score(graph_ids, relevant, args.k)[0])

        graph_ms = statistics.median(graph_samples)
        if len(recalls) > 1:
            print(f"  [{target['target']}] UNSTABLE recall across repeats: {sorted(recalls)}")
        recall_graph, prec_graph = score(graph_ids, relevant, args.k)

        # A target with more callers than k cannot be fully recovered at k.
        ceiling = min(args.k, len(relevant)) / len(relevant)

        rows.append(
            {
                "target": target["target"],
                "relevant": len(relevant),
                "ceiling": ceiling,
                "recall_seed": recall_seed,
                "recall_graph": recall_graph,
                "prec_seed": prec_seed,
                "prec_graph": prec_graph,
                "seed_ms": seed_ms,
                "graph_ms": graph_ms,
            }
        )
        print(
            f"  {target['target']:32s} truth={len(relevant):3d}  "
            f"recall seed={recall_seed:.2f} graph={recall_graph:.2f} "
            f"(ceiling {ceiling:.2f})   ({seed_ms:.0f}ms / {graph_ms:.0f}ms)"
        )

    def avg(key: str) -> float:
        return statistics.mean(row[key] for row in rows) if rows else 0.0

    ceiling = avg("ceiling")
    achieved = avg("recall_graph")
    print(f"\n=== {backend}, direction={args.direction} — {len(rows)} queries, k={args.k}")
    print(f"  recall@{args.k}:    seed-only {avg('recall_seed'):.3f}   seed+graph {achieved:.3f}")
    print(f"    attainable ceiling {ceiling:.3f} -> {achieved / ceiling * 100:.0f}% of it")
    print(
        f"  precision@{args.k}: seed-only {avg('prec_seed'):.3f}   seed+graph {avg('prec_graph'):.3f}"
    )
    print(
        f"  latency p50:     seed-only {statistics.median(r['seed_ms'] for r in rows):.1f}ms   "
        f"seed+graph {statistics.median(r['graph_ms'] for r in rows):.1f}ms"
    )

    if args.json:
        args.json.write_text(
            json.dumps(
                {"backend": backend, "direction": args.direction, "k": args.k, "rows": rows},
                indent=2,
            )
        )
        print(f"\nwrote {args.json}")

    close = getattr(store, "close", None)
    if close:
        await close()


if __name__ == "__main__":
    asyncio.run(main())
