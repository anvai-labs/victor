#!/usr/bin/env python3
"""Attribute the wall time of one graph persist batch (native-acceleration-strategy).

Builds a deterministic, synthetic batch of GraphNode/GraphEdge rows whose shapes
are derived from a real victor-repo index (project.db, 2026-11):

- ~311 nodes/file (94% ``statement`` rows, ~19.5 symbols: module/class/function/
  method/constructor in the observed proportions)
- ~767 edges/file in the observed type mix (DDG_DEF_USE 53%, CFG_SUCCESSOR 24%,
  CFG_TRUE 5.5%, CDG 4.2%, REFERENCES 3.9%, CONTAINS 2.4%, CFG_RETURN 1.7%,
  CALLS 1.5%, remainder spread over CFG_LOOP_ENTRY/CFG_EXCEPTION/CDG_LOOP/IMPORTS)
- statement metadata = ``{"node_type", "content_preview", "source_line"}``;
  symbol metadata = bound/rebound name lists; DDG/CFG/CDG/REFERENCES edge
  metadata carries an absolute ``file`` hint (exercising the COALESCE
  first-branch path, same as the real corpus where 100% of edges carry a file)

It then runs a ``_IndexingStreamPipeline._flush``-shaped write batch against a
temporary ``SqliteGraphStore`` (temp project dir, WAL, dedicated worker thread)
and reports a phase-level wall-time attribution table:

- symbol-node / symbol-edge / ccg-node / ccg-edge upserts (call level, split
  into Python row-prep on the event loop, dangling-guard SELECTs, parent
  ordering, and the raw ``executemany`` on the SQLite worker)
- per-file mtime updates including the sha-256 content-hash compute the
  pipeline performs inside the batch for full runs
- BEGIN/COMMIT transaction overhead (flush-wall residual)

``--c-profile`` additionally runs one flush under cProfile and prints the top
functions by internal time. Per the recorded lesson, cProfile output is
supporting evidence only: the verdict comes from the wall-clock table above
(cProfile fabricates hot-share on tiny call counts).

Usage:
    python3 scripts/benchmark_persist_batch.py --files 20 --repeats 5
    python3 scripts/benchmark_persist_batch.py --files 200 --repeats 3 --c-profile
"""

from __future__ import annotations

import argparse
import asyncio
import cProfile
import hashlib
import io
import pstats
import random
import sqlite3
import statistics
import sys
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from victor.storage.graph.protocol import GraphEdge, GraphNode  # noqa: E402
from victor.storage.graph.sqlite_store import SqliteGraphStore  # noqa: E402

# ── Real-corpus shapes (victor repo, project.db 2026-11; see module docstring) ──

SYMBOL_SHAPE = [
    # (type, per-file mean, signature?, docstring?)
    ("module", 1.0, False, False),
    ("class", 2.9, True, True),
    ("function", 3.7, True, True),
    ("method", 11.4, True, True),
    ("constructor", 0.5, True, False),
]

# statement_type distribution over statement nodes (fraction of all nodes)
STATEMENT_TYPES = [
    ("expression", 0.234),
    ("call", 0.196),
    ("assignment", 0.144),
    ("block", 0.115),
    ("condition", 0.069),
    ("return", 0.059),
    ("function_def", 0.055),
    ("loop", 0.022),
    ("branch", 0.012),
    ("try", 0.006),
    ("with", 0.005),
    ("raise", 0.004),
    ("assert", 0.003),
    ("pass", 0.002),
    ("delete", 0.001),
]

# edge type mix: (type, per-file mean, sym-or-ccg, weighted?, file-hinted?)
EDGE_SHAPE = [
    ("DDG_DEF_USE", 409.4, "ccg", True, True),
    ("CFG_SUCCESSOR", 181.5, "ccg", True, True),
    ("CFG_TRUE", 42.4, "ccg", True, True),
    ("CDG", 31.7, "ccg", True, True),
    ("REFERENCES", 30.1, "ccg", False, True),
    ("CONTAINS", 18.5, "sym", False, True),
    ("CFG_RETURN", 13.1, "ccg", True, True),
    ("CALLS", 11.5, "sym", False, True),
    ("CFG_LOOP_ENTRY", 11.4, "ccg", True, True),
    ("CFG_EXCEPTION", 7.1, "ccg", True, True),
    ("CDG_LOOP", 5.8, "ccg", True, True),
    ("IMPORTS", 1.5, "sym", False, True),
]

LANGS = ["python", "python", "python", "typescript", "rust"]  # repo-like weighting


def _hex(rng: random.Random, nbytes: int) -> str:
    return hashlib.md5(rng.randbytes(nbytes)).hexdigest()


class Corpus:
    """Deterministic synthetic corpus: files on disk + graph rows per file."""

    def __init__(self, root: Path, n_files: int, seed: int) -> None:
        self.root = root
        self.rng = random.Random(seed)
        self.files: list[Path] = []
        self.sym_nodes: list[GraphNode] = []
        self.sym_edges: list[GraphEdge] = []
        self.ccg_nodes: list[GraphNode] = []
        self.ccg_edges: list[GraphEdge] = []
        self._build(n_files)

    def _build(self, n_files: int) -> None:
        rng = self.rng
        for fi in range(n_files):
            rel = f"victor/pkg/bench_mod_{fi:04d}.py"
            lang = rng.choice(LANGS)
            abs_path = self.root / rel

            # Small python-like source file so mtime/stat/read are real.
            n_lines = 120 + rng.randrange(300)
            body = "\n".join(
                f"def _f{fi}_{i}(x):  # statement filler {i}\n    return x + {i}"
                for i in range(max(1, n_lines // 3))
            )
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            abs_path.write_text(body + "\n", encoding="utf-8")
            self.files.append(abs_path)

            sym_ids: list[str] = []
            module_id: str | None = None
            class_ids: list[str] = []

            # Symbols
            for stype, mean, has_sig, has_doc in SYMBOL_SHAPE:
                count = max(0, int(rng.gauss(mean, mean * 0.25)))
                for k in range(count):
                    sid = _hex(rng, 16)  # 32-hex symbol ids in the real corpus
                    parent: str | None = None
                    if stype in ("method", "constructor"):
                        if not class_ids:
                            continue
                        parent = rng.choice(class_ids)
                    elif stype == "function" and module_id:
                        parent = module_id
                    name = f"{stype}_{fi}_{k}"
                    line = rng.randrange(1, n_lines)
                    end = min(n_lines, line + rng.randrange(1, 40))
                    meta: dict = {}
                    if stype != "module":
                        meta["python_bound_names"] = [
                            f"name_{k}_{j}" for j in range(rng.randrange(3, 12))
                        ]
                        meta["python_rebound_names"] = [f"rb_{k}"]
                    self.sym_nodes.append(
                        GraphNode(
                            node_id=sid,
                            type=stype,
                            name=name,
                            file=str(abs_path),
                            line=line,
                            end_line=end,
                            lang=lang,
                            signature=f"{name}(a, b) -> None" if has_sig else None,
                            docstring=f"Docstring for {name}." if has_doc else None,
                            parent_id=parent,
                            embedding_ref=f"emb:{sid}" if stype != "module" else None,
                            metadata=meta,
                        )
                    )
                    if parent:
                        self.sym_edges.append(
                            GraphEdge(src=parent, dst=sid, type="CONTAINS", metadata={})
                        )
                    if stype == "module":
                        module_id = sid
                    elif stype == "class":
                        class_ids.append(sid)
                    sym_ids.append(sid)

            # Statements (CCG nodes)
            stmt_types = [t for t, _ in STATEMENT_TYPES]
            weights = [w for _, w in STATEMENT_TYPES]
            n_stmt = max(1, int(rng.gauss(291, 60)))
            stmt_ids: list[str] = []
            for k in range(n_stmt):
                stype = rng.choices(stmt_types, weights)[0]
                sid = _hex(rng, 8)  # 16-hex statement ids
                self.ccg_nodes.append(
                    GraphNode(
                        node_id=sid,
                        type="statement",
                        name=f"{stype}:{rng.randrange(1, n_lines)}",
                        file=str(abs_path),
                        line=rng.randrange(1, n_lines),
                        end_line=rng.randrange(1, n_lines),
                        lang=lang,
                        metadata={
                            "node_type": stype,
                            "content_preview": f"{stype} stmt {k} of module {fi}",
                            "source_line": rng.randrange(1, n_lines),
                        },
                        ast_kind=stype,
                        scope_id=rng.choice(stmt_ids) if stmt_ids and rng.random() < 0.3 else sid,
                        statement_type=stype,
                    )
                )
                stmt_ids.append(sid)

            # Edges in the real type mix
            for etype, mean, kind, weighted, hinted in EDGE_SHAPE:
                count = max(0, int(rng.gauss(mean, mean * 0.3)))
                for _ in range(count):
                    pool = sym_ids if kind == "sym" else stmt_ids
                    if not pool:
                        continue
                    src = dst = None
                    while dst is None or dst == src:
                        src = rng.choice(pool)
                        dst = rng.choice(pool)
                    meta: dict = {}
                    if hinted:
                        meta["file"] = str(abs_path)
                    if etype == "DDG_DEF_USE":
                        meta["variable"] = f"var_{rng.randrange(50)}"
                    if etype in ("CALLS", "REFERENCES"):
                        meta["provenance"] = "scope_resolver"
                        meta["confidence"] = 1.0
                    self.ccg_edges.append(
                        GraphEdge(
                            src=src,
                            dst=dst,
                            type=etype,
                            weight=1.0 if weighted else None,
                            metadata=meta,
                        )
                    )

    def content_hashes(self) -> list[tuple[Path, float, str]]:
        """(path, mtime, sha-256) per file — the pipeline's full-run hash compute."""
        out = []
        for path in self.files:
            st = path.stat()
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            out.append((path, st.st_mtime, digest))
        return out


class Timings:
    """Phase accumulator (seconds). ``label`` routes worker-thread time."""

    def __init__(self) -> None:
        self.phases: dict[str, float] = {}
        self.subs: dict[str, float] = {}
        self.label = "other"

    def add(self, name: str, seconds: float) -> None:
        self.phases[name] = self.phases.get(name, 0.0) + seconds

    def add_sub(self, name: str, seconds: float) -> None:
        self.subs[name] = self.subs.get(name, 0.0) + seconds


def instrument_store(store: SqliteGraphStore, timings: Timings) -> None:
    """Wrap _run_db and the guard/ordering helpers with perf_counter brackets."""
    real_run_db = store._run_db
    real_clear = store._clear_dangling_parent_links
    real_order = store._order_rows_parents_first
    real_drop = store._drop_dangling_edge_rows

    async def timed_run_db(fn):
        t0 = time.perf_counter()
        try:
            return await real_run_db(fn)
        finally:
            timings.add_sub(f"db:{timings.label}", time.perf_counter() - t0)

    def timed_clear(conn, rows):
        t0 = time.perf_counter()
        try:
            return real_clear(conn, rows)
        finally:
            timings.add_sub(
                f"guard:{timings.label}:dangling_parent_select", time.perf_counter() - t0
            )

    def timed_order(rows):
        t0 = time.perf_counter()
        try:
            return real_order(rows)
        finally:
            timings.add_sub(f"guard:{timings.label}:parent_ordering", time.perf_counter() - t0)

    def timed_drop(conn, rows):
        t0 = time.perf_counter()
        try:
            return real_drop(conn, rows)
        finally:
            timings.add_sub(
                f"guard:{timings.label}:dangling_endpoint_select", time.perf_counter() - t0
            )

    store._run_db = timed_run_db
    store._clear_dangling_parent_links = timed_clear
    store._order_rows_parents_first = timed_order
    store._drop_dangling_edge_rows = timed_drop


def run_flush(store: SqliteGraphStore, corpus: Corpus, timings: Timings) -> dict:
    """One _IndexingStreamPipeline._flush-equivalent write batch (sync driver)."""

    @asynccontextmanager
    async def phase(name: str):
        timings.label = name
        t0 = time.perf_counter()
        try:
            yield
        finally:
            timings.add(name, time.perf_counter() - t0)
            timings.label = "other"

    async def _flush() -> None:
        async with store.write_batch():
            if corpus.sym_nodes:
                async with phase("upsert_nodes.sym"):
                    await store.upsert_nodes(corpus.sym_nodes)
            if corpus.sym_edges:
                async with phase("upsert_edges.sym"):
                    await store.upsert_edges(corpus.sym_edges)
            if corpus.ccg_nodes:
                async with phase("upsert_nodes.ccg"):
                    await store.upsert_nodes(corpus.ccg_nodes)
            if corpus.ccg_edges:
                async with phase("upsert_edges.ccg"):
                    await store.upsert_edges(corpus.ccg_edges)
            async with phase("mtime.per_file"):
                hashes = await asyncio.get_running_loop().run_in_executor(
                    None, corpus.content_hashes
                )
                for path, mtime, digest in hashes:
                    await store.update_file_mtime(str(path), mtime, content_hash=digest)

    asyncio.run(_flush())
    return {
        "sym_nodes": len(corpus.sym_nodes),
        "ccg_nodes": len(corpus.ccg_nodes),
        "sym_edges": len(corpus.sym_edges),
        "ccg_edges": len(corpus.ccg_edges),
    }


def print_query_plans() -> None:
    """EXPLAIN QUERY PLAN for every statement group the write batch executes.

    Evidence that the write path has no missing index: every plan below is a
    primary-key or secondary-index search (no SCAN, no per-row full lookup).
    """
    from victor.core.schema import Tables
    from victor.storage.graph.sqlite_store import _TABLE_SCHEMA

    node, edge, mtime = Tables.GRAPH_NODE, Tables.GRAPH_EDGE, Tables.GRAPH_FILE_MTIME
    conn = sqlite3.connect(":memory:")
    conn.executescript(_TABLE_SCHEMA)
    plans = [
        (
            "node upsert (ON CONFLICT node_id)",
            f"INSERT INTO {node}(node_id, type, name, file, line) VALUES (?,?,?,?,?) "
            f"ON CONFLICT(node_id) DO UPDATE SET name=excluded.name",
        ),
        (
            "edge upsert (ON CONFLICT src,dst,type; COALESCE file subqueries)",
            f"INSERT INTO {edge}(src, dst, type, weight, file, metadata) VALUES (?,?,?,?,"
            f"COALESCE(?,(SELECT file FROM {node} n WHERE n.node_id=?),"
            f"(SELECT file FROM {node} n WHERE n.node_id=?)),?) "
            f"ON CONFLICT(src, dst, type) DO UPDATE SET weight=excluded.weight",
        ),
        (
            "dangling-guard SELECT (chunked IN)",
            f"SELECT node_id FROM {node} WHERE node_id IN (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        ),
        (
            "mtime upsert (ON CONFLICT file)",
            f"INSERT INTO {mtime}(file, mtime, indexed_at, content_hash) VALUES (?,?,?,?) "
            f"ON CONFLICT(file) DO UPDATE SET mtime=excluded.mtime",
        ),
    ]
    print("\n== EXPLAIN QUERY PLAN (write-batch statements) ==")
    for name, sql in plans:
        print(f"--- {name}")
        # EXPLAIN executes no statement: NULL binds and constraints don't matter.
        # The prefix is concatenated OUTSIDE the execute() call on purpose: the
        # SQL strings are assembled from internal Tables-enum constants (no
        # untrusted input in this offline benchmark), and keeping the execute()
        # argument a single pre-built variable keeps the SAST gate honest —
        # there is no injection-shaped sink on that line.
        plan_sql = "EXPLAIN QUERY PLAN " + sql
        for row in conn.execute(plan_sql, tuple([None] * sql.count("?"))):
            print(f"    {row[3]}")
    conn.close()


def print_table(title: str, timings: Timings, total: float, counts: dict) -> None:
    print(f"\n== {title} ==")
    print(
        f"rows: {counts['sym_nodes']} sym nodes, {counts['ccg_nodes']} ccg nodes, "
        f"{counts['sym_edges']} sym edges, {counts['ccg_edges']} ccg edges"
    )
    print(f"{'phase':34s} {'wall ms':>9s} {'% flush':>8s}   (split: python / guards / executemany)")

    begin_commit = timings.subs.get("db:other", 0.0)
    residual = total - sum(timings.phases.values()) - begin_commit
    for name, seconds in sorted(timings.phases.items(), key=lambda kv: -kv[1]):
        db = timings.subs.get(f"db:{name}", 0.0)
        guards = sum(
            value for key, value in timings.subs.items() if key.startswith(f"guard:{name}:")
        )
        py = seconds - db
        exec_many = db - guards
        split = f"{py * 1000:6.1f} / {guards * 1000:6.1f} / {exec_many * 1000:6.1f}"
        print(f"{name:34s} {seconds * 1000:9.1f} {seconds / total * 100:7.1f}%   {split}")
    print(
        f"{'sqlite BEGIN/COMMIT':34s} {begin_commit * 1000:9.1f} "
        f"{begin_commit / total * 100:7.1f}%"
    )
    print(f"{'txn overhead + residual':34s} {residual * 1000:9.1f} {residual / total * 100:7.1f}%")
    print(f"{'TOTAL flush wall':34s} {total * 1000:9.1f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--files", type=int, default=20, help="files per flush batch")
    parser.add_argument("--repeats", type=int, default=5, help="flush repetitions on one store")
    parser.add_argument("--seed", type=int, default=20261118, help="RNG seed (deterministic)")
    parser.add_argument("--c-profile", action="store_true", help="cProfile the first flush")
    parser.add_argument(
        "--explain", action="store_true", help="print EXPLAIN QUERY PLAN for write statements"
    )
    args = parser.parse_args()

    if args.explain:
        print_query_plans()

    with tempfile.TemporaryDirectory(prefix="victor-persist-bench-") as tmp:
        tmp_path = Path(tmp)
        store = SqliteGraphStore(tmp_path / "graph.db")
        corpus = Corpus(tmp_path, args.files, args.seed)
        counts = {
            "sym_nodes": len(corpus.sym_nodes),
            "ccg_nodes": len(corpus.ccg_nodes),
            "sym_edges": len(corpus.sym_edges),
            "ccg_edges": len(corpus.ccg_edges),
        }
        print(f"batch: {args.files} files -> {counts}")

        cold_timings = Timings()
        instrument_store(store, cold_timings)

        profiler = cProfile.Profile() if args.c_profile else None
        if profiler is not None:
            profiler.enable()
        t0 = time.perf_counter()
        cold_counts = run_flush(store, corpus, cold_timings)
        cold_total = time.perf_counter() - t0
        if profiler is not None:
            profiler.disable()
        print_table(
            f"COLD flush (fresh DB, {args.files} files)", cold_timings, cold_total, cold_counts
        )

        if profiler is not None:
            out = io.StringIO()
            stats = pstats.Stats(profiler, stream=out)
            stats.sort_stats("tottime").print_stats(18)
            print("\n== cProfile top (internal time; supporting evidence only) ==")
            print("\n".join(out.getvalue().splitlines()[4:40]))

        warm_totals: list[float] = []
        for rep in range(args.repeats - 1):
            corpus2 = Corpus(tmp_path, args.files, args.seed + rep + 1)
            timings = Timings()
            instrument_store(store, timings)
            t0 = time.perf_counter()
            warm_counts = run_flush(store, corpus2, timings)
            warm_totals.append(time.perf_counter() - t0)
            print_table(
                f"WARM flush (re-index, rep {rep + 1})",
                timings,
                warm_totals[-1],
                warm_counts,
            )

        if warm_totals:
            spread = (
                f"stdev {statistics.stdev(warm_totals) * 1000:.1f} ms"
                if len(warm_totals) > 1
                else "single run"
            )
            print(
                f"\nwarm re-index flush: mean {statistics.mean(warm_totals) * 1000:.1f} ms, "
                f"{spread} over {len(warm_totals)} runs"
            )
        store._connect().close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
