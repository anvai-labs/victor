"""Observe offline body-cache parity on a bounded, immutable source snapshot.

Use a fresh process for each cold/warm observation and the same Python environment
for each source revision. Library import during model loading is inside the timer;
source collection and integrity checks are outside it. JSON goes to stdout only.
Default observations measure requested-pair scoring. --details separately times
global exact retrieval and retains diagnostics; neither mode measures accuracy.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import itertools
import json
import os
from pathlib import Path
import platform
import sqlite3
import sys
import time
from typing import Any
from urllib.parse import quote

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
        "victor.core.graph_rag.semantic_clones",
        "victor.core.graph_rag.family_cohesion",
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


def benchmark(
    root: Path, cache: Path, *, state: str, details: bool = False, owned_cache: bool = False
) -> dict[str, Any]:
    """Refuse accidental cache replacement; verify selected inputs after analysis."""
    root = root.resolve()
    if owned_cache:
        if (
            os.name != "posix"
            or cache.is_absolute()
            or len(cache.parts) != 1
            or cache.name in {".", ".."}
        ):
            raise ValueError("Owned cache requires one relative POSIX filename")
        cache_location = cache.absolute()  # Containment check only, never used for opening.
    else:
        cache = cache.absolute()
        cache_location = cache
    if state not in {"cold", "warm"}:
        raise ValueError("Cache state must be cold or warm")
    if cache_location.is_relative_to(root):
        raise ValueError("Keep the derived cache outside the source snapshot")
    if (state == "cold" and cache.exists()) or (state == "warm" and not cache.exists()):
        raise ValueError("Cache existence does not match the requested cold/warm state")
    implementation = _implementation()
    collection_wall, collection_cpu = time.perf_counter(), time.process_time()
    report = scan(root, min_nodes=40, include_tests=False, retain_source=True)
    collection_wall, collection_cpu = (
        time.perf_counter() - collection_wall,
        time.process_time() - collection_cpu,
    )
    if not 2 <= len(report.units) <= 300:
        raise ValueError("Select between 2 and 300 functions for bounded exact pair scoring")
    unit_manifest = [
        {
            "file": unit.file_path,
            "symbol": unit.name,
            "line": unit.line,
            "region_index": unit.region_index,
            "source_start": unit.source_start,
            "input_hash": hashlib.sha256(unit.source_code.encode()).hexdigest(),
        }
        for unit in report.units
    ]
    options = BodyOptions()
    encoder = LocalBodyEncoder(options)
    pairs = list(itertools.combinations(range(len(report.units)), 2))
    wall, cpu = time.perf_counter(), time.process_time()
    result = analyze_bodies(
        report, options=options, encoder=encoder, cache_path=cache, requested_pairs=pairs
    )
    wall, cpu = time.perf_counter() - wall, time.process_time() - cpu
    global_result = None
    global_wall = global_cpu = 0.0
    if details:
        # A fresh encoder proves the global path also reuses cached vectors
        # without relying on the requested-score encoder's already loaded model.
        global_wall, global_cpu = time.perf_counter(), time.process_time()
        global_result = analyze_bodies(
            report,
            options=options,
            encoder=LocalBodyEncoder(options),
            cache_path=cache,
            threshold=0.9,
            include_same_file=False,
            top_k=None,
        )
        global_wall, global_cpu = (
            time.perf_counter() - global_wall,
            time.process_time() - global_cpu,
        )
    if implementation != _implementation():
        raise RuntimeError("Producer implementation changed during measurement")
    for relative, expected in report.source_manifest.items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError("Source snapshot changed during measurement")
    profile = result["profile"]
    cache_identity = profile.get("producer", {}).get("identity", options.identity)
    cache_uri = cache.as_uri() if cache.is_absolute() else "file:" + quote(cache.as_posix())
    with sqlite3.connect(f"{cache_uri}?mode=ro&immutable=1", uri=True) as db:
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
        "cache_io_mode": "descriptor_cwd_relative_leaf" if owned_cache else "caller_path",
        "wall_seconds": wall,
        "cpu_seconds": cpu,
        "collection_wall_seconds": collection_wall,
        "collection_cpu_seconds": collection_cpu,
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
        "global_retrieval_benchmark": details,
        **(
            {
                "unit_input_hashes": [
                    hashlib.sha256(unit.source_code.encode()).hexdigest() for unit in report.units
                ],
                "unit_manifest": unit_manifest,
                "unit_manifest_sha256": _digest(unit_manifest),
                "requested_inputs": result["inputs"],
                "pair_scores": result["pair_scores"],
                "global_result": global_result,
                "global_options": {
                    "threshold": 0.9,
                    "include_same_file": False,
                    "top_k": None,
                    "comparison_scope": options.comparison_scope,
                },
                "global_wall_seconds": global_wall,
                "global_cpu_seconds": global_cpu,
            }
            if details
            else {}
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--state", choices=("cold", "warm"), required=True)
    parser.add_argument(
        "--details",
        action="store_true",
        help="Retain diagnostic pairs and separately time global retrieval",
    )
    parser.add_argument(
        "--owned-cache",
        action="store_true",
        help="Use a relative cache leaf under the caller owned cwd",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            benchmark(
                args.path,
                args.cache,
                state=args.state,
                details=args.details,
                owned_cache=args.owned_cache,
            ),
            indent=2,
        )
    )
