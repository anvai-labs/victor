"""Observe offline body-cache parity on a bounded, immutable source snapshot.

Use a fresh process for each cold/warm observation and the same Python environment
for each source revision. Library import during model loading is inside the timer;
source collection and integrity checks are outside it. JSON goes to stdout only.
This measures requested-pair scoring, not global retrieval or model accuracy.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import itertools
import json
from pathlib import Path
import platform
import sqlite3
import sys
import time
from typing import Any

from victor.core.graph_rag.body_clones import BodyOptions, LocalBodyEncoder, analyze_bodies
from victor.core.graph_rag.structural_clones import scan


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _implementation() -> dict[str, str]:
    result = {
        str(Path(__file__).resolve()): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    }
    for name in (
        "victor.core.graph_rag.body_clones",
        "victor.core.graph_rag.body_provenance",
        "victor.core.graph_rag.structural_clones",
        "victor_codegraph.embeddings",
    ):
        try:
            module = importlib.import_module(name)
        except ModuleNotFoundError as exc:
            if name.endswith(".body_provenance") and exc.name == name:
                continue  # The pre-provenance reference has no such module.
            raise
        if module.__file__ is None:
            raise RuntimeError("Producer has no inspectable source file")
        path = Path(module.__file__).resolve()
        result[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def benchmark(root: Path, cache: Path, *, state: str) -> dict[str, Any]:
    """Refuse accidental cache replacement; verify selected inputs after analysis."""
    root, cache = root.resolve(), cache.absolute()
    if state not in {"cold", "warm"}:
        raise ValueError("Cache state must be cold or warm")
    if cache.is_relative_to(root):
        raise ValueError("Keep the derived cache outside the source snapshot")
    if (state == "cold" and cache.exists()) or (state == "warm" and not cache.exists()):
        raise ValueError("Cache existence does not match the requested cold/warm state")
    implementation = _implementation()
    report = scan(root, min_nodes=40, include_tests=False, retain_source=True)
    if not 2 <= len(report.units) <= 300:
        raise ValueError("Select between 2 and 300 functions for bounded exact pair scoring")
    options = BodyOptions()
    encoder = LocalBodyEncoder(options)
    pairs = list(itertools.combinations(range(len(report.units)), 2))
    wall, cpu = time.perf_counter(), time.process_time()
    result = analyze_bodies(
        report, options=options, encoder=encoder, cache_path=cache, requested_pairs=pairs
    )
    wall, cpu = time.perf_counter() - wall, time.process_time() - cpu
    if implementation != _implementation():
        raise RuntimeError("Producer implementation changed during measurement")
    for relative, expected in report.source_manifest.items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError("Source snapshot changed during measurement")
    profile = result["profile"]
    cache_identity = profile.get("producer", {}).get("identity", options.identity)
    with sqlite3.connect(f"{cache.as_uri()}?mode=ro", uri=True) as db:
        rows = [
            [digest, hashlib.sha256(vector).hexdigest(), spans]
            for digest, vector, spans in db.execute(
                "SELECT input_hash,vector,spans FROM body_vectors_v1 WHERE profile=? ORDER BY input_hash",
                (cache_identity,),
            )
        ]
    rss = None
    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
            1 if sys.platform == "darwin" else 1024
        )
    except ImportError:
        pass
    return {
        "schema": "victor-body-cache-observation-v1",
        "scope": "body_analysis_requested_pairs_excluding_source_collection",
        "state": state,
        "wall_seconds": wall,
        "cpu_seconds": cpu,
        "process_peak_rss_bytes": rss,
        "rss_scope": "process_lifetime_peak",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "implementation": implementation,
        "source_manifest": report.source_manifest,
        "source_coverage": result["source_coverage"],
        "functions": len(report.units),
        "profile": profile,
        "stats": result["stats"],
        "coverage_complete": result["coverage_complete"],
        "errors": result["errors"],
        "cached_vectors": rows,
        "vector_manifest_sha256": _digest(rows),
        "pair_count": len(pairs),
        "pair_scores_sha256": _digest(result["pair_scores"]),
        "pair_abstentions": sum(row["score"] is None for row in result["pair_scores"]),
        "model_downloads": False,
        "accuracy_benchmark": False,
        "global_retrieval_benchmark": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--state", choices=("cold", "warm"), required=True)
    args = parser.parse_args()
    print(json.dumps(benchmark(args.path, args.cache, state=args.state), indent=2))
