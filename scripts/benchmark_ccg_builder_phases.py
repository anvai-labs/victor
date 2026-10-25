#!/usr/bin/env python
"""Phase-level wall-clock profile of the plugin (enhanced) CCG builder.

Reproduces the boundary decision evidence recorded in
``docs/architecture/native-acceleration-strategy.md``: the statement walk
dominates (~70%) and the identifier/call extractors are a rounding error
(~4 ms/file). Phase-level timing wraps six call sites per file, not the
hot leaf functions -- per-call wrappers (cProfile included) fabricate a
large extractor share through their own call overhead; trust this script,
not instrumented profiles, for the split.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _phase_timed_builder(module: object) -> tuple[type, dict[str, float]]:
    acc = {"parse": 0.0, "hints": 0.0, "walk": 0.0, "edges": 0.0, "dedupe": 0.0, "total": 0.0}

    class Timed(module.PluginBackedCCGBuilder):  # type: ignore[name-defined]
        """build_ccg_for_file with phase counters (6 perf_counter pairs/file)."""

        async def build_ccg_for_file(self, file_path, language=None):
            t00 = time.perf_counter()
            from pathlib import Path as P

            path = P(file_path)
            plugin = self._get_plugin("python")
            parser = self._get_parser("python")
            t0 = time.perf_counter()
            source_bytes = path.read_bytes()
            tree = parser.parse(source_bytes)
            acc["parse"] += time.perf_counter() - t0
            source_lines = source_bytes.decode("utf-8", errors="ignore").splitlines()
            t0 = time.perf_counter()
            symbol_hints = self._extract_symbol_hints(tree, parser, plugin)
            acc["hints"] += time.perf_counter() - t0
            graph_nodes, records = [], []
            t0 = time.perf_counter()
            self._walk_statements(
                tree.root_node,
                file_path=path,
                source_bytes=source_bytes,
                source_lines=source_lines,
                language="python",
                parser_language="python",
                plugin=plugin,
                symbol_hints=symbol_hints,
                graph_nodes=graph_nodes,
                records=records,
            )
            acc["walk"] += time.perf_counter() - t0
            _, GraphEdge, EdgeType = module._get_graph_types()
            t0 = time.perf_counter()
            edges = self._build_cfg_edges(records, GraphEdge, EdgeType)
            edges += self._build_cdg_edges(records, GraphEdge, EdgeType)
            edges += self._build_ddg_edges(records, GraphEdge, EdgeType, str(path))
            acc["edges"] += time.perf_counter() - t0
            t0 = time.perf_counter()
            edges = self._dedupe_edges(edges)
            acc["dedupe"] += time.perf_counter() - t0
            acc["total"] += time.perf_counter() - t00
            return graph_nodes, edges

    return Timed, acc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "target",
        nargs="?",
        default="victor/core/graph_rag/precise_providers.py",
        help="file to profile (default: the ADR boundary corpus)",
    )
    parser.add_argument("--runs", type=int, default=3, help="timed runs")
    args = parser.parse_args(argv)

    target = Path(args.target)
    if not target.is_absolute():
        target = REPO / target
    if not target.is_file():
        print(f"no such file: {target}", file=sys.stderr)
        return 2

    sys.path.insert(0, str(REPO / "verticals" / "victor-coding"))
    from victor_coding.codebase import ccg_builder as module

    Timed, acc = _phase_timed_builder(module)
    builder = Timed()

    samples = []
    for run in range(args.runs):
        for key in acc:
            acc[key] = 0.0
        nodes, edges = None, None
        if run == 0:
            nodes, edges = asyncio.run(builder.build_ccg_for_file(target))  # warm-up
        for _ in range(3):
            nodes, edges = asyncio.run(builder.build_ccg_for_file(target))
        per_file = {k: v / 3 * 1000 for k, v in acc.items()}
        samples.append(per_file)

    median = {k: statistics.median(s[k] for s in samples) for k in samples[0]}
    total = median.pop("total")
    print(f"{target.name}: {len(nodes or [])} nodes / {len(edges or [])} edges")
    print(f"total {total:.1f} ms/file")
    for phase, ms in sorted(median.items(), key=lambda kv: -kv[1]):
        print(f"  {phase}: {ms:.1f} ms ({ms / total * 100:.0f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
