"""Observe scoped SCIP analysis in a fresh process, without inference or source execution.

Run this same observer with each implementation on PYTHONPATH. Imports, input hashing
and post-run integrity checks are outside wall/CPU timing; RSS is process lifetime peak.
File-system caches are not flushed. This is latency/parity evidence, not clone accuracy.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import stat
import sys
import time
from typing import Any

from victor.core.graph_rag.source_reader import project_source_reader, read_project_source
from victor.core.graph_rag.structural_clones import scan


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    """Hash a caller-selected regular benchmark input in bounded chunks."""
    if os.name != "posix":
        raise OSError("The scoped measurement observer requires POSIX descriptor acquisition")
    with project_source_reader(path.parent) as root:
        descriptor = os.open(
            path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=root.descriptor
        )
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise ValueError("Benchmark input must be a regular file")

            def identity(info: os.stat_result) -> tuple[int, ...]:
                return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns

            result = hashlib.sha256()
            remaining = before.st_size
            while remaining:
                chunk = os.read(descriptor, min(remaining, 64 * 1024))
                if not chunk:
                    raise RuntimeError("SCIP benchmark input changed")
                result.update(chunk)
                remaining -= len(chunk)
            if os.read(descriptor, 1):
                raise RuntimeError("SCIP benchmark input changed")
            after = os.fstat(descriptor)
            current = os.stat(path.name, dir_fd=root.descriptor, follow_symlinks=False)
            root.verify_root_path()
            if identity(before) != identity(after) or identity(before) != identity(current):
                raise RuntimeError("SCIP benchmark input changed")
            return {
                "bytes": before.st_size,
                "mtime_ns": before.st_mtime_ns,
                "sha256": result.hexdigest(),
            }
        finally:
            os.close(descriptor)


def implementation() -> dict[str, str]:
    result = {}
    for name in ("structural_clones", "precise_index", "scip_documents", "source_reader"):
        module_name = f"victor.core.graph_rag.{name}"
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if name == "scip_documents" and exc.name == module_name:
                continue  # The full-protobuf reference predates this owner.
            raise
        if module.__file__ is None:
            raise RuntimeError("Benchmark producer has no inspectable source")
        result[name] = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    return result


def benchmark(root: Path, index: Path, *, scope: str) -> dict[str, Any]:
    root, index = root.absolute(), index.absolute()
    before, producers = artifact(index), implementation()
    wall, cpu = time.perf_counter(), time.process_time()
    report = scan(root, scope=scope, scip_path=index, min_nodes=5, near_metric="subtree")
    wall, cpu = time.perf_counter() - wall, time.process_time() - cpu
    if before != artifact(index) or producers != implementation():
        raise RuntimeError("SCIP benchmark input or producer changed")
    for name, expected in report.source_manifest.items():
        data, _ = read_project_source(root, name, 8 * 1024**2)
        if hashlib.sha256(data).hexdigest() != expected:
            raise RuntimeError("Source snapshot changed during measurement")
    units = [
        (
            u.file_path,
            u.line,
            u.name,
            u.tokens,
            u.exact_hash,
            u.shape_hash,
            u.subtree_hashes,
            u.resolved_symbols,
        )
        for u in report.units
    ]
    families = [asdict(family) for family in [*report.families, *report.suppressed]]
    rss = None
    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
            1 if sys.platform == "darwin" else 1024
        )
    except ImportError:
        pass
    return {
        "schema": "victor-scip-scope-observation-v1",
        "scope": scope,
        "wall_seconds": wall,
        "cpu_seconds": cpu,
        "process_peak_rss_bytes": rss,
        "rss_scope": "process_lifetime_peak_including_integrity_checks",
        "filesystem_cache": "not_flushed",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "implementation": producers,
        "observer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "index": before,
        "source_manifest": report.source_manifest,
        "stats": report.stats,
        "coverage_complete": report.coverage_complete,
        "units": len(units),
        "unit_digest": digest(units),
        "families": len(families),
        "family_digest": digest(families),
        "accuracy_benchmark": False,
        "source_freshness_proof": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--scope", required=True)
    args = parser.parse_args()
    print(json.dumps(benchmark(args.path, args.index, scope=args.scope), indent=2))
