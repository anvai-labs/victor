"""Offline source-channel observation; no semantic accuracy claims or downloads."""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import resource
import sys
import time

from victor.core.graph_rag.body_clones import BodyOptions
from victor.core.graph_rag.similarity_analysis import analyze_source
from victor.core.graph_rag.token_clones import TokenOptions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--mode", choices=["body", "tokens", "hybrid"], default="hybrid")
    parser.add_argument("--jscpd", default="jscpd")
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-nodes", type=int, default=5)
    parser.add_argument("--min-tokens", type=int, default=20)
    parser.add_argument("--token-fingerprints", action="store_true")
    parser.add_argument("--token-min-lines", type=int, default=0)
    parser.add_argument("--scope")
    parser.add_argument("--same-file", action="store_true")
    parser.add_argument("--exclude-docs", action="store_true")
    args = parser.parse_args()
    implementation = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in Path("victor/core/graph_rag").glob("*.py")
        if p.name.endswith("clones.py")
        or p.name
        in {
            "source_regions.py",
            "source_reader.py",
            "numeric_literals.py",
            "clone_bindings.py",
            "similarity_analysis.py",
            "ast_fingerprints.py",
            "language_parsers.py",
            "source_inventory.py",
            "bounded_process.py",
            "token_fingerprints.py",
            "token_coverage.py",
            "copy_baseline.py",
            "copy_artifacts.py",
            "body_provenance.py",
            "family_cohesion.py",
        }
    }
    cpu, wall = time.process_time(), time.perf_counter()
    child = resource.getrusage(resource.RUSAGE_CHILDREN)
    result = analyze_source(
        args.path.resolve(),
        mode=args.mode,
        min_nodes=args.min_nodes,
        near_metric="subtree",
        include_docs=not args.exclude_docs,
        scope=args.scope,
        include_same_file=args.same_file,
        limit=200,
        body_options=BodyOptions(),
        token_options=TokenOptions(
            binary=args.jscpd,
            min_tokens=args.min_tokens,
            min_lines=args.token_min_lines,
            **({"fingerprints": True} if args.token_fingerprints else {}),
        ),
        cache_path=args.cache,
    )
    child_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    versions: dict[str, str | None] = {}
    for name in (
        "torch",
        "sentence-transformers",
        "transformers",
        "tokenizers",
        "numpy",
        "psutil",
        "tree-sitter",
        "markdown-it-py",
        "victor-codegraph",
    ):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    evidence = {
        "schema": "victor-source-channel-observation-v1",
        "accuracy_benchmark": False,
        "invocation": {
            "mode": args.mode,
            "min_nodes": args.min_nodes,
            "min_tokens": args.min_tokens,
            "token_fingerprints": args.token_fingerprints,
            "token_min_lines": args.token_min_lines,
            "scope": args.scope,
            "include_same_file": args.same_file,
            "cache_enabled": args.cache is not None,
            "include_docs": not args.exclude_docs,
            "near_metric": "subtree",
            "threshold": 0.9,
        },
        "python": platform.python_version(),
        "dependency_versions": versions,
        "wall_seconds": time.perf_counter() - wall,
        "cpu_seconds": time.process_time() - cpu,
        "child_cpu_seconds": child_after.ru_utime
        + child_after.ru_stime
        - child.ru_utime
        - child.ru_stime,
        "process_max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if sys.platform == "darwin" else 1024),
        "implementation_hashes": implementation,
        "implementation_changed_during_run": any(
            hashlib.sha256((Path("victor/core/graph_rag") / name).read_bytes()).hexdigest()
            != digest
            for name, digest in implementation.items()
        ),
        "result": result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: value
                for key, value in evidence.items()
                if key not in {"result", "implementation_hashes"}
            },
            indent=2,
        )
    )
    print("coverage_complete:", result["coverage_complete"])


if __name__ == "__main__":
    main()
