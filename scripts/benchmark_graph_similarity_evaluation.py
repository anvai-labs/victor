"""Measure the labelled-pair protocol at its budget; never detector accuracy/retrieval."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys
import time
from typing import Any

from victor.core.graph_rag.similarity_evaluation import Corpus, Observation, calibrate, evaluate


def benchmark(pair_count: int = 50_000, runs: int = 5) -> dict[str, Any]:
    if type(pair_count) is not int or not 8 <= pair_count <= 50_000 or pair_count % 4:
        raise ValueError("Pair count must be divisible by four, between 8 and 50,000")
    if type(runs) is not int or not 1 <= runs <= 20:
        raise ValueError("Runs must be between 1 and 20")
    projects, units, pairs, scores = [], [], [], []
    for split in ("calibration", "test"):
        projects.append(
            {
                "id": split,
                "repository": f"protocol-scale:{split}",
                "revision": "v1",
                "split": split,
                "origin": "controlled_fixture",
            }
        )
        referenced: set[str] = set()
        for i in range(pair_count // 2):
            uid = f"{split}-{i}"
            positive = i % 2 == 0
            left, right = f"{split}-{i % 2500}", f"{split}-{(i % 2500 + i // 2500 + 1) % 2500}"
            referenced.update((left, right))
            pairs.append(
                {
                    "id": uid,
                    "left": left,
                    "right": right,
                    "relation": "behavior",
                    "positive": positive,
                    "family": split,
                    "rationale": "Protocol load only",
                    "domain": "No source or detector evaluated",
                    "reviewed_by": "load-generator",
                }
            )
            scores.append(Observation(uid, (0.7 if positive else 0.1) + 0.2 * i / 25000))
        for uid in sorted(referenced):
            units.append(
                {
                    "id": uid,
                    "project": split,
                    "file": f"{uid}.py",
                    "language": "python",
                    "symbol": "f",
                    "line": 1,
                    "end_line": 2,
                    "source_hash": hashlib.sha256(uid.encode()).hexdigest(),
                }
            )
    raw = {
        "schema": "victor-similarity-corpus-v1",
        "projects": projects,
        "units": units,
        "pairs": pairs,
    }
    wall, cpu = [], []
    for _ in range(runs):
        start_cpu, start_wall = time.process_time(), time.perf_counter()
        corpus = Corpus.from_dict(raw)
        policy = calibrate(corpus, scores, channel="body", profile="a" * 64)
        result = evaluate(corpus, scores, policy, profile="a" * 64)
        if not result["decision_complete"]:
            raise RuntimeError("Generated protocol workload failed its own invariants")
        wall.append(time.perf_counter() - start_wall)
        cpu.append(time.process_time() - start_cpu)
    peak_rss = None
    try:
        import resource

        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
            1 if sys.platform == "darwin" else 1024
        )
    except ImportError:
        pass
    return {
        "schema": "victor-similarity-protocol-benchmark-v1",
        "detector_retrieval_benchmark": False,
        "accuracy_benchmark": False,
        "pairs": len(pairs),
        "units": len(units),
        "runs": runs,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "wall_seconds": wall,
        "cpu_seconds": cpu,
        "median_wall_seconds": statistics.median(wall),
        "max_wall_seconds": max(wall),
        "median_cpu_seconds": statistics.median(cpu),
        "process_max_rss_bytes": peak_rss,
        "rss_scope": "process_lifetime_peak",
        "implementation": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__),
                Path(__file__).resolve().parents[1]
                / "victor/core/graph_rag/similarity_evaluation.py",
            )
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=50_000)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = benchmark(args.pairs, args.runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
