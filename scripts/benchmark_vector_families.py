"""Measure vector-family discovery/explanation cost, never detector accuracy.

Run unchanged against each source revision in separate processes. Inputs and
library imports are outside the timer. Optional .npy fixtures must be derived,
reviewed local vectors; source locations are synthetic in every mode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys
import time
from typing import Any, TypedDict

from victor.core.graph_rag import semantic_clones
from victor.core.graph_rag.body_provenance import callable_code_fingerprint


class FamilyObservation(TypedDict):
    wall_seconds: float
    cpu_seconds: float
    answer_sha256: str
    families: int
    matching_edges: int
    cohesion_components: int


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _implementation() -> dict[str, str]:
    paths = [Path(__file__), Path(semantic_clones.__file__)]
    try:
        from victor.core.graph_rag import family_cohesion
    except ImportError:
        pass  # The pre-explanation reference has no such module.
    else:
        paths.append(Path(family_cohesion.__file__))
    return {
        **{str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        "actual_callable": callable_code_fingerprint(semantic_clones.find_clone_families),
    }


def benchmark(
    count: int = 5000, dimension: int = 384, runs: int = 5, fixture: Path | None = None
) -> dict[str, Any]:
    import numpy as np

    if (
        type(count) is not int
        or not 2 <= count <= 20_000
        or type(dimension) is not int
        or not 1 <= dimension <= 4096
        or type(runs) is not int
        or not 1 <= runs <= 10
    ):
        raise ValueError("Invalid vector benchmark dimensions or run count")
    fixture_hash = None
    if fixture is not None:
        if fixture.stat().st_size > 32 * 1024 * 1024:
            raise ValueError("Vector fixture exceeds 32 MiB")
        fixture_hash = hashlib.sha256(fixture.read_bytes()).hexdigest()
        base = np.load(fixture, allow_pickle=False, mmap_mode="r")
        if (
            base.ndim != 2
            or not 2 <= len(base) <= 5000
            or not 1 <= base.shape[1] <= 4096
            or base.dtype.kind not in "fiu"
        ):
            raise ValueError("Invalid vector fixture shape")
        dimension = base.shape[1]
    if count * dimension > 10_000_000:
        raise ValueError("Vector fixture/selection exceeds its element budget")
    if fixture is None:
        random = np.random.default_rng(20261102)
        centers = random.normal(size=((count + 9) // 10, dimension)).astype(np.float32)
        matrix = centers[np.arange(count) // 10] + random.normal(
            scale=0.001, size=(count, dimension)
        ).astype(np.float32)
    else:
        matrix = np.asarray(base[np.arange(count) % len(base)], dtype=np.float32)
    records = [
        semantic_clones.SymbolRecord(str(i), f"file_{i % 97}.py", f"f{i}", "function", i + 1)
        for i in range(count)
    ]
    implementation = _implementation()
    observations: list[FamilyObservation] = []
    for _ in range(runs):
        wall, cpu = time.perf_counter(), time.process_time()
        families = semantic_clones.find_clone_families(records, matrix, threshold=0.9)
        wall, cpu = time.perf_counter() - wall, time.process_time() - cpu
        answer = [
            [
                sorted(member.id for member in family.members),
                family.similarity,
                family.min_edge_similarity,
                family.matching_pairs,
            ]
            for family in families
        ]
        observations.append(
            {
                "wall_seconds": wall,
                "cpu_seconds": cpu,
                "answer_sha256": _digest(answer),
                "families": len(families),
                "matching_edges": sum(f.matching_pairs for f in families),
                "cohesion_components": sum(
                    getattr(f, "cohesion", None) is not None for f in families
                ),
            }
        )
    if implementation != _implementation() or (
        fixture is not None and hashlib.sha256(fixture.read_bytes()).hexdigest() != fixture_hash
    ):
        raise RuntimeError("Benchmark implementation or fixture changed during execution")
    if len({item["answer_sha256"] for item in observations}) != 1:
        raise RuntimeError("Family discovery changed across repeated identical inputs")
    rss = None
    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
            1 if sys.platform == "darwin" else 1024
        )
    except ImportError:
        pass
    return {
        "schema": "victor-vector-family-benchmark-v1",
        "accuracy_benchmark": False,
        "scope": "vector_family_discovery_and_explanations_excluding_imports_and_input_creation",
        "locations": "synthetic",
        "vectors": "replicated_local_fixture" if fixture else "seeded_cluster_control",
        "count": count,
        "dimension": dimension,
        "threshold": 0.9,
        "seed": 20261102,
        "fixture_sha256": fixture_hash,
        "matrix_sha256": hashlib.sha256(matrix.tobytes()).hexdigest(),
        "implementation": implementation,
        "observations": observations,
        "median_wall_seconds": statistics.median(item["wall_seconds"] for item in observations),
        "median_cpu_seconds": statistics.median(item["cpu_seconds"] for item in observations),
        "process_peak_rss_bytes": rss,
        "rss_scope": "process_lifetime_peak",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=5000)
    parser.add_argument("--dimension", type=int, default=384)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--fixture", type=Path)
    args = parser.parse_args()
    print(json.dumps(benchmark(args.count, args.dimension, args.runs, args.fixture), indent=2))
