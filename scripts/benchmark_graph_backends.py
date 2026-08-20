#!/usr/bin/env python3
# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Compare the SQLite and ProximaDB code-graph backends on one corpus.

This is the TD-11 step-6 measurement from
``docs/architecture/proximadb-codegraph-backend.md``: index the *same* files
through the *same* pipeline into each backend, then compare on-disk footprint
and traversal latency. The footprint number is the one that decides step 7 (the
per-repo default flip) — the design projects ~120 MB f32 / ~35 MB SQ8 for Tier-A
against a measured 2.4 GB SQLite + LanceDB pair.

**Both backends must index the same corpus path.** Symbol and statement node ids
are derived from the file path, so indexing two copies of a corpus at different
paths yields two entirely disjoint node-id sets — a diff then reports 100%
difference regardless of backend, and edge counts drift by tens purely from the
path. This script indexes one ``--corpus`` for every backend precisely so the
comparison is like-for-like; do not "helpfully" give each backend its own copy.
Measured at a shared path, SQLite and ProximaDB produce identical node sets and
identical statement-level edges.

Deliberately *not* covered: Arrow Flight bulk-load throughput. Victor has no
Arrow Flight code path at all — ingest goes through REST ``insert_records`` +
``batch_create_nodes`` — so that line of step 6 is a feature to build, not a
measurement to take. What this script reports for ingest is today's real path.

Usage::

    python scripts/benchmark_graph_backends.py --corpus victor/storage --max-files 150
    python scripts/benchmark_graph_backends.py --corpus . --backends sqlite --json out.json

The ProximaDB backend needs a ``proximadb-server`` binary on PATH (or in the
SDK's build tree); it is reported as skipped rather than failing the run.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sqlite3
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from victor.core.graph_rag.config import GraphIndexConfig  # noqa: E402
from victor.core.graph_rag.indexing import GraphIndexingPipeline  # noqa: E402
from victor.storage.graph.registry import create_graph_store  # noqa: E402


def _dir_bytes(path: Path) -> int:
    """Total bytes of every regular file under ``path`` (0 if absent)."""
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _checkpoint_sqlite_wals(path: Path) -> None:
    """Fold any `-wal` sidecars back into their database before measuring.

    A footprint number that includes a WAL the engine is about to delete is not
    a footprint; it is a snapshot of mid-flight state. Opening each database and
    running `wal_checkpoint(TRUNCATE)` makes the measurement independent of when
    the last connection happens to close. Best-effort: a database still held
    open elsewhere simply stays as it is, which is no worse than not trying.
    """
    for db in path.rglob("*.db"):
        try:
            conn = sqlite3.connect(db)
            try:
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            finally:
                conn.close()
        except sqlite3.Error:
            continue


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024.0 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024.0
    return f"{n:.1f} GB"


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return ordered[k]


async def _time_traversals(
    store: Any, seeds: List[str], *, hops: int, repeats: int
) -> Dict[str, float]:
    """Time k-hop traversal from each seed; return latency percentiles in ms."""
    latencies: List[float] = []
    for _ in range(repeats):
        for seed in seeds:
            start = time.perf_counter()
            try:
                await store.multi_hop_traverse_parallel([seed], max_hops=hops)
            except Exception:  # a backend that cannot traverse is not a timing signal
                continue
            latencies.append((time.perf_counter() - start) * 1000.0)
    if not latencies:
        return {}
    return {
        "samples": len(latencies),
        "p50_ms": round(statistics.median(latencies), 2),
        "p95_ms": round(_percentile(latencies, 95), 2),
        "mean_ms": round(statistics.fmean(latencies), 2),
    }


async def _bench_backend(
    backend: str,
    corpus: Path,
    workdir: Path,
    *,
    embeddings: bool,
    hops: int,
    seed_count: int,
    repeats: int,
) -> Dict[str, Any]:
    """Index ``corpus`` into one backend and measure it."""
    project = workdir / backend
    if project.exists():
        shutil.rmtree(project)
    project.mkdir(parents=True)

    result: Dict[str, Any] = {"backend": backend}

    try:
        store = create_graph_store(backend, project_path=project)
    except Exception as exc:
        return {**result, "skipped": f"could not create store: {exc}"}

    config = GraphIndexConfig(
        root_path=corpus,
        enable_ccg=False,  # Tier-B fragments are measured separately; keep Tier-A comparable
        enable_embeddings=embeddings,
        enable_subgraph_cache=False,
        incremental=False,
    )
    pipeline = GraphIndexingPipeline(store, config)

    try:
        started = time.perf_counter()
        stats = await pipeline.index_repository(root_path=corpus)
        result["index_seconds"] = round(time.perf_counter() - started, 2)
    except Exception as exc:
        try:
            await store.close()
        except Exception:
            pass
        return {**result, "skipped": f"indexing failed: {type(exc).__name__}: {exc}"}

    try:
        result["files_indexed"] = getattr(stats, "files_processed", None)

        # Count by reading the data back, not from stats(): backends disagree on
        # the stats() key names and some return none at all, and a footprint
        # number is meaningless unless we can prove the rows are actually there.
        nodes = await store.get_all_nodes()
        edges = await store.get_all_edges()
        result["nodes"] = len(nodes)
        result["edges"] = len(edges)

        store_stats = await store.stats()
        reported = store_stats.get("node_count", store_stats.get("nodes"))
        if reported is not None and reported != len(nodes):
            result["stats_disagreement"] = f"stats()={reported} readback={len(nodes)}"

        seeds = [n.node_id for n in nodes[:seed_count]]
        result["traversal"] = await _time_traversals(store, seeds, hops=hops, repeats=repeats)
    except Exception as exc:
        result["measure_error"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            await store.close()
        except Exception:
            pass

    # Footprint is measured after close() AND after checkpointing any SQLite
    # write-ahead logs left behind. `store.close()` is not enough: Victor holds
    # a process-global connection, so `project.db-wal` survives until the
    # interpreter exits, and it is counted as footprint even though SQLite
    # deletes it moments later. The error is large and one-sided — a repo-scale
    # SQLite arm measured 196.8 MB here and 125 MB at rest, while ProximaDB
    # measured identically both times — so it silently penalised whichever
    # backend happens to be SQLite.
    _checkpoint_sqlite_wals(project)
    result["footprint_bytes"] = _dir_bytes(project)
    result["footprint"] = _fmt_bytes(result["footprint_bytes"])
    if result.get("nodes"):
        result["bytes_per_node"] = round(result["footprint_bytes"] / result["nodes"], 1)
    return result


def _git_sha(repo: Path) -> str:
    """Short SHA of ``repo``, or a marker when it cannot be read."""
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip() or "unknown"
    except Exception:  # pragma: no cover - environment-dependent
        return "unknown"


def _git_dirty(repo: Path) -> bool:
    """True when ``repo`` has uncommitted changes (results would not reproduce)."""
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return bool(out.stdout.strip())
    except Exception:  # pragma: no cover - environment-dependent
        return False


def _describe_environment() -> Dict[str, Any]:
    """Identify exactly what is about to be measured.

    Three separate measurement sessions were invalidated by the environment
    rather than the code: a two-week-stale server binary made an
    already-fixed defect look live; an SDK missing the engine fix made a
    footprint benchmark characterise the wrong storage engine; and a shared
    venv repointed at an in-progress worktree broke embedded startup outright.
    In each case the numbers looked plausible and were about something other
    than what was claimed.

    A benchmark that cannot say what it measured produces folklore, so record
    it up front and print it beside the results.
    """
    env: Dict[str, Any] = {}

    try:
        import proximadb_sdk

        sdk_path = Path(proximadb_sdk.__file__).resolve()
        env["sdk_path"] = str(sdk_path)
        # Walk up to the checkout root (…/clients/python/src/proximadb_sdk/…).
        root = sdk_path
        for _ in range(6):
            root = root.parent
            if (root / ".git").exists():
                break
        env["sdk_repo"] = str(root)
        env["sdk_sha"] = _git_sha(root)
        env["sdk_dirty"] = _git_dirty(root)
    except Exception as exc:  # pragma: no cover - optional dependency
        env["sdk_path"] = f"unavailable ({type(exc).__name__})"
        env["sdk_dirty"] = False

    server = shutil.which("proximadb-server") or str(Path(sys.prefix) / "bin" / "proximadb-server")
    server_path = Path(server)
    if server_path.exists():
        resolved = server_path.resolve()
        stat = resolved.stat()
        env["server_path"] = str(resolved)
        env["server_bytes"] = stat.st_size
        env["server_mtime"] = time.strftime("%Y-%m-%d %H:%M", time.localtime(stat.st_mtime))
    else:
        env["server_path"] = "not found"

    victor_root = Path(__file__).resolve().parent.parent
    env["victor_sha"] = _git_sha(victor_root)
    env["victor_dirty"] = _git_dirty(victor_root)
    return env


def _print_environment(env: Dict[str, Any]) -> None:
    print("environment under measurement:")
    print(f"  victor       {env.get('victor_sha')}{'  (DIRTY)' if env.get('victor_dirty') else ''}")
    print(f"  sdk          {env.get('sdk_path')}")
    if "sdk_sha" in env:
        print(f"               {env['sdk_sha']}{'  (DIRTY)' if env.get('sdk_dirty') else ''}")
    print(f"  server       {env.get('server_path')}")
    if "server_bytes" in env:
        print(f"               {env['server_bytes']:,} bytes, built {env['server_mtime']}")


def _render(results: List[Dict[str, Any]], *, embeddings: bool) -> str:
    lines: List[str] = []
    lines.append("")
    lines.append(f"Graph backend comparison (embeddings={'on' if embeddings else 'off'})")
    lines.append("=" * 78)
    header = f"{'backend':10} {'nodes':>8} {'edges':>8} {'index s':>9} {'footprint':>12} {'B/node':>9} {'k-hop p50':>10}"
    lines.append(header)
    lines.append("-" * 78)
    for r in results:
        if r.get("skipped"):
            lines.append(f"{r['backend']:10} SKIPPED — {r['skipped']}")
            continue
        trav = r.get("traversal") or {}
        nodes = r.get("nodes")
        edges = r.get("edges")
        lines.append(
            f"{r['backend']:10} {('?' if nodes is None else nodes):>8} "
            f"{('?' if edges is None else edges):>8} "
            f"{r.get('index_seconds', 0):>9} {r.get('footprint', '?'):>12} "
            f"{r.get('bytes_per_node', '?'):>9} {trav.get('p50_ms', float('nan')):>9}ms"
        )
        if r.get("stats_disagreement"):
            lines.append(f"{'':10} WARNING stats disagreement: {r['stats_disagreement']}")
    lines.append("-" * 78)

    usable = [r for r in results if not r.get("skipped") and r.get("footprint_bytes")]
    # A footprint comparison is only meaningful between backends holding the
    # same data. Refuse to print a ratio across differing node counts.
    counts = {r.get("nodes") for r in usable}
    if len(usable) == 2 and len(counts) > 1:
        detail = ", ".join("{}={}".format(r["backend"], r.get("nodes")) for r in usable)
        lines.append(
            "footprint ratio SUPPRESSED — backends hold different node counts "
            f"({detail}); not a like-for-like comparison."
        )
        usable = []
    if len(usable) == 2:
        a, b = usable
        if b["footprint_bytes"]:
            ratio = a["footprint_bytes"] / b["footprint_bytes"]
            lines.append(
                f"footprint ratio {a['backend']}:{b['backend']} = {ratio:.2f}x "
                f"({_fmt_bytes(a['footprint_bytes'])} vs {_fmt_bytes(b['footprint_bytes'])})"
            )
    if not embeddings:
        lines.append(
            "NOTE: embeddings are OFF — this is the Tier-A graph footprint only. "
            "The 2.4 GB baseline in the design doc includes LanceDB vectors; "
            "rerun with --embeddings for the full comparison."
        )
    lines.append("")
    return "\n".join(lines)


async def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("victor/storage"))
    parser.add_argument("--backends", default="sqlite,proxima")
    parser.add_argument("--workdir", type=Path, default=None)
    parser.add_argument("--embeddings", action="store_true", help="generate vectors too")
    parser.add_argument("--hops", type=int, default=2)
    parser.add_argument("--seeds", type=int, default=25)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="measure anyway when the SDK resolves to a checkout with uncommitted changes",
    )
    args = parser.parse_args()

    env = _describe_environment()
    _print_environment(env)
    if env.get("sdk_dirty") and not args.allow_dirty:
        print(
            "\nREFUSING TO MEASURE: the proximadb SDK resolves to a checkout with "
            "uncommitted changes, so these numbers could not be reproduced or "
            "attributed to a commit.\n"
            "Point the venv at a clean checkout, or pass --allow-dirty and treat "
            "the results as indicative only.",
            file=sys.stderr,
        )
        return 3
    print()

    corpus = args.corpus.resolve()
    if not corpus.exists():
        print(f"corpus not found: {corpus}", file=sys.stderr)
        return 2

    workdir = args.workdir or Path(__import__("tempfile").mkdtemp(prefix="graph-backend-bench-"))
    workdir.mkdir(parents=True, exist_ok=True)
    print(f"corpus:  {corpus}")
    print(f"workdir: {workdir}")

    results: List[Dict[str, Any]] = []
    for backend in [b.strip() for b in args.backends.split(",") if b.strip()]:
        print(f"\n--- {backend} ---")
        res = await _bench_backend(
            backend,
            corpus,
            workdir,
            embeddings=args.embeddings,
            hops=args.hops,
            seed_count=args.seeds,
            repeats=args.repeats,
        )
        results.append(res)
        print(json.dumps(res, indent=2, default=str))

    report = _render(results, embeddings=args.embeddings)
    print(report)

    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "corpus": str(corpus),
                    "embeddings": args.embeddings,
                    "environment": env,
                    "results": results,
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
