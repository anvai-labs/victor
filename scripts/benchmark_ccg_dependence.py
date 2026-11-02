"""Measure returned CCG construction on an explicit, trusted immutable snapshot.

Source reads, parsing and returned graph construction are timed. Imports, builder
creation, manifest hashing and JSON output are excluded. Counts describe the
returned graph; they do not qualify source control flow or semantic similarity.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any

from victor.core.indexing import ccg_builder
from victor.core.indexing.ccg_builder import CodeContextGraphBuilder
from victor.processing.native import ccg_extract

SAMPLES = (
    ("verticals/victor-coding/tests/codebase/goldens/samples/sample.py", "python"),
    ("verticals/victor-coding/tests/codebase/goldens/samples/sample.ts", "typescript"),
    ("verticals/victor-coding/tests/codebase/goldens/samples/sample.rs", "rust"),
    ("verticals/victor-coding/tests/codebase/goldens/samples/Sample.java", "java"),
    ("verticals/victor-coding/tests/codebase/goldens/samples/sample.go", "go"),
    ("victor/core/graph_rag/precise_providers.py", "python"),
    ("victor/core/graph_rag/source_reader.py", "python"),
)


class PythonBuilder(CodeContextGraphBuilder):
    def _build_native_ccg(self, *args: Any, **kwargs: Any) -> None:
        return None

    def _extract_statement_nodes_native(self, *args: Any, **kwargs: Any) -> None:
        return None


async def benchmark(
    root: Path,
    *,
    backend: str = "native",
    runs: int = 3,
    scope: str | None = None,
    max_files: int = 5000,
    max_bytes: int = 65536,
) -> dict[str, Any]:
    if backend not in {"native", "python"} or type(runs) is not int or not 1 <= runs <= 20:
        raise ValueError("Invalid CCG benchmark backend/run budget")
    if not 1 <= max_files <= 5000 or not 1 <= max_bytes <= 2_000_000:
        raise ValueError("Invalid source selection budget")
    root = root.resolve(strict=True)
    if scope is None:
        selected = list(SAMPLES)
    else:
        value = Path(scope)
        if (
            value.is_absolute()
            or ".." in value.parts
            or not (root / value).resolve().is_relative_to(root)
        ):
            raise ValueError("Scope must remain within the snapshot")
        selected = [
            (p.relative_to(root).as_posix(), "python") for p in sorted((root / value).rglob("*.py"))
        ]
    files: list[tuple[str, str]] = []
    excluded = 0
    for path, language in selected:
        source = root / path
        if source.is_symlink() or source.stat().st_size > max_bytes:
            excluded += 1
            continue
        if len(files) < max_files:
            files.append((path, language))
    manifest = {path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path, _ in files}
    implementation_paths = []
    for module in (ccg_builder, ccg_extract):
        file = module.__file__
        if file is None:
            raise RuntimeError("Missing benchmark implementation provenance")
        implementation_paths.append(Path(file))
    helper = implementation_paths[0].resolve().parents[1] / "graph_rag/control_dependence.py"
    if helper.is_file():
        implementation_paths.append(helper)
    implementation_paths.append(Path(__file__))
    implementation = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in implementation_paths
    }
    samples: list[dict[str, Any]] = []
    for run in range(runs):
        # "native" selects core extraction without provider substitution; a
        # missing/incompatible wheel still uses its typed Python fallback.
        builder = (PythonBuilder if backend == "python" else CodeContextGraphBuilder)(
            builder_resolution="native"
        )
        rows: list[dict[str, Any]] = []
        for path, language in files:
            chosen = []
            original = builder._build_native_ccg

            def observe(*args: Any, **kwargs: Any) -> Any:
                result = original(*args, **kwargs)
                chosen.append(result is not None)
                return result

            builder._build_native_ccg = observe  # type: ignore[method-assign]
            cpu, wall = time.process_time(), time.perf_counter()
            nodes, edges = await builder.build_ccg_for_file(root / path, language)
            elapsed_wall, elapsed_cpu = time.perf_counter() - wall, time.process_time() - cpu
            builder._build_native_ccg = original  # type: ignore[method-assign]
            rows.append(
                {
                    "path": path,
                    "language": language,
                    "backend": "native" if chosen == [True] else "python",
                    "wall_seconds": elapsed_wall,
                    "cpu_seconds": elapsed_cpu,
                    "nodes": len(nodes),
                    "unique_node_ids": len({n.node_id for n in nodes}),
                    "edges": len(edges),
                    "cdg_edges": sum(
                        str(getattr(e.type, "value", e.type)).startswith("CDG") for e in edges
                    ),
                    "cdg_complete": (
                        all(n.metadata.get("cdg_complete") for n in nodes)
                        if nodes and all("cdg_complete" in n.metadata for n in nodes)
                        else None
                    ),
                }
            )
        samples.append(
            {
                "run": run,
                "wall_seconds": sum(row["wall_seconds"] for row in rows),
                "cpu_seconds": sum(row["cpu_seconds"] for row in rows),
                "files": rows,
            }
        )
    if any(
        hashlib.sha256((root / path).read_bytes()).hexdigest() != value
        for path, value in manifest.items()
    ):
        raise RuntimeError("Source snapshot changed during measurement")
    if any(
        hashlib.sha256(path.read_bytes()).hexdigest() != implementation[path.name]
        for path in implementation_paths
    ):
        raise RuntimeError("Implementation changed during measurement")
    peak_rss = None
    try:
        import resource

        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
            1 if sys.platform == "darwin" else 1024
        )
    except ImportError:
        pass
    try:
        native_version = importlib.metadata.version("victor-native")
    except importlib.metadata.PackageNotFoundError:
        native_version = None
    return {
        "schema": "victor-ccg-dependence-benchmark-v1",
        "scope": "returned_core_ccg_including_source_reads_parsing_and_graph_construction",
        "program_semantics_verified": False,
        "native_version": native_version,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "source_manifest": manifest,
        "selection": {
            "scope": scope,
            "selected_files": len(files),
            "oversize_or_linked_files": excluded,
            "truncated": len(selected) - excluded > max_files,
            "max_bytes": max_bytes,
        },
        "implementation": implementation,
        "process_peak_rss_bytes": peak_rss,
        "rss_scope": "process_lifetime_peak",
        "runs": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--backend", choices=("native", "python"), default="native")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument(
        "--scope", help="Optional Python source subtree instead of seven-file controls"
    )
    parser.add_argument("--max-files", type=int, default=5000)
    parser.add_argument("--max-bytes", type=int, default=65536)
    args = parser.parse_args()
    result = asyncio.run(
        benchmark(
            args.root,
            backend=args.backend,
            runs=args.runs,
            scope=args.scope,
            max_files=args.max_files,
            max_bytes=args.max_bytes,
        )
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
