"""Read-only duplicate-discovery profile against a representative project DB."""

from __future__ import annotations

import argparse
from contextlib import closing
import cProfile
import io
import json
from pathlib import Path
import pstats
import resource
import sqlite3
import statistics
import time
import tracemalloc

from victor.core.graph_rag.dedup import SymbolDeduplicator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")
    wall, cpu = [], []
    profiler = cProfile.Profile()
    with closing(sqlite3.connect(args.database.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        db.execute("BEGIN")
        node_count = db.execute("SELECT count(*) FROM graph_node").fetchone()[0]
        edge_count = db.execute("SELECT count(*) FROM graph_edge").fetchone()[0]
        tracemalloc.start()
        for _ in range(args.runs):
            start, start_cpu = time.perf_counter(), time.process_time()
            groups = SymbolDeduplicator(db).find_duplicates()
            wall.append(time.perf_counter() - start)
            cpu.append(time.process_time() - start_cpu)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        profiler.runcall(SymbolDeduplicator(db).find_duplicates)
    output = io.StringIO()
    pstats.Stats(profiler, stream=output).sort_stats("cumulative").print_stats(8)
    print(
        json.dumps(
            {
                "nodes": node_count,
                "edges": edge_count,
                "groups": len(groups),
                "removable_nodes": sum(len(g.node_ids) - 1 for g in groups),
                "runs": args.runs,
                "wall_median_ms": round(statistics.median(wall) * 1000, 3),
                "cpu_median_ms": round(statistics.median(cpu) * 1000, 3),
                "wall_max_ms": round(max(wall) * 1000, 3),
                "python_peak_bytes": peak,
                "process_max_rss_platform_units": resource.getrusage(
                    resource.RUSAGE_SELF
                ).ru_maxrss,
            },
            indent=2,
        )
    )
    print(output.getvalue())


if __name__ == "__main__":
    main()
