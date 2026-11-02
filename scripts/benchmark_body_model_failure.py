"""Probe repeated invalid-model construction with the pinned offline model.

Print JSON only. Model-library import and source collection are outside the timer.
This failure-path probe is not a throughput, model-quality or whole-CLI benchmark.
Run the same script with each candidate's source on PYTHONPATH for comparison.
"""

from __future__ import annotations

import hashlib
import argparse
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any

from victor.core.graph_rag import body_clones
from victor.core.graph_rag.body_clones import BodyOptions, LocalBodyEncoder, analyze_bodies
from victor.core.graph_rag.structural_clones import StructuralReport, extract_units, scan


def benchmark(factory: Any, report: StructuralReport | None = None) -> dict[str, Any]:
    if report is None:
        source = "def one(x):\n return x+1\n\ndef two(x):\n return x-1\n"
        units = extract_units("profile_probe.py", source, None, min_nodes=1, retain_source=True)
        report = StructuralReport([], [], {"source_files": 1, "selected_regions": 1}, units=units)
    manifest = sorted((u.file_path, u.name, u.line, u.source_hash) for u in report.units)
    options = BodyOptions(dimension=385)  # Deliberately differs from pinned BGE's 384.
    construction_seconds = []

    def measured_factory(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        try:
            return factory(*args, **kwargs)
        finally:
            construction_seconds.append(time.perf_counter() - start)

    start_cpu, start_wall = time.process_time(), time.perf_counter()
    result = analyze_bodies(
        report, options=options, encoder=LocalBodyEncoder(options, model_factory=measured_factory)
    )
    wall, cpu = time.perf_counter() - start_wall, time.process_time() - start_cpu
    rss = None
    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
            1 if sys.platform == "darwin" else 1024
        )
    except ImportError:
        pass
    return {
        "schema": "victor-body-model-failure-probe-v1",
        "scope": "invalid_profile_body_analysis_excluding_library_import_and_source_collection",
        "profile": options.identity,
        "functions": len(report.units),
        "model_constructions": len(construction_seconds),
        "construction_seconds": construction_seconds,
        "wall_seconds": wall,
        "cpu_seconds": cpu,
        "process_peak_rss_bytes": rss,
        "rss_scope": "process_lifetime_peak",
        "coverage_complete": result["coverage_complete"],
        "errors": result["errors"],
        "python": platform.python_version(),
        "platform": platform.platform(),
        "implementation": {
            "body_clones.py": hashlib.sha256(Path(body_clones.__file__).read_bytes()).hexdigest()
        },
        "source_manifest": manifest,
        "workload_sha256": hashlib.sha256(
            json.dumps(manifest, separators=(",", ":")).encode()
        ).hexdigest(),
        "model_downloads": False,
        "accuracy_benchmark": False,
    }


if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path", type=Path, help="Optional representative snapshot (min_nodes=40, excludes tests)"
    )
    args = parser.parse_args()
    report = (
        scan(args.path, min_nodes=40, include_tests=False, retain_source=True)
        if args.path
        else None
    )
    print(json.dumps(benchmark(SentenceTransformer, report), indent=2))
