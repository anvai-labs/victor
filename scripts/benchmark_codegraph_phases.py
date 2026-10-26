#!/usr/bin/env python
"""Wall-clock phase profile of victor-codegraph symbol extraction.

The baseline harness for the symbol-extraction optimization
(docs/architecture/native-acceleration-strategy.md): attributes parse
cost per PATH (python stdlib-ast vs tree-sitter) and per LANGUAGE on the
REAL repo mix, with emit_references on/off, and separates the repository
binding pass from the per-file walk. Phase-level probes only — per-call
wrappers (cProfile included) fabricate hot-shares through their own
overhead; trust this script's numbers, not instrumented profiles.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Representative per-language samples from the real tree (mixed sizes).
SAMPLES = {
    "python": [
        "victor/core/graph_rag/precise_providers.py",
        "victor/agent/orchestrator.py",
        "victor/core/graph_rag/indexing.py",
        "tests/unit/core/graph_rag/test_ccg_builder_resolution.py",
    ],
    "rust": ["rust/crates/python-bindings/src/ccg_extract.rs"],
    "typescript": ["vscode-victor/src/victorClient.ts"],
    "javascript": ["vscode-victor/src/extension.js"],
    "yaml": [".github/workflows/ci-fast.yml"],
    "markdown": ["README.md"],
}

EXT_TO_LANG = {".py": "python", ".rs": "rust", ".ts": "typescript", ".tsx": "typescript"}


def language_of(path: Path) -> str:
    import victor_codegraph.languages as languages

    return languages.detect_language(str(path)) or EXT_TO_LANG.get(path.suffix, "?")


def time_per_file(runs: int) -> dict:
    import victor_codegraph

    results: dict[str, list[float]] = {}
    for label, rel_paths in SAMPLES.items():
        sources = []
        for rel in rel_paths:
            path = REPO / rel
            if path.is_file():
                sources.append((rel, path.read_text(encoding="utf-8", errors="ignore")))
        for emit_references in (False, True):
            key = f"{label}{'+refs' if emit_references else ''}"
            samples = []
            for _ in range(runs):
                t0 = time.perf_counter()
                for rel, source in sources:
                    victor_codegraph.parse(
                        source, language=None, file_path=rel, emit_references=emit_references
                    )
                samples.append((time.perf_counter() - t0) / max(1, len(sources)) * 1000)
            results[key] = samples
    return results


def repo_mix() -> Counter:
    """File counts per language for the real tree — the weighting for
    extrapolating per-file numbers to a full index."""
    counts: Counter = Counter()
    import victor_codegraph.languages as languages

    skip_parts = {".git", "node_modules", "__pycache__", "target", ".venv", "dist"}
    for path in REPO.rglob("*"):
        if not path.is_file() or skip_parts & set(path.parts):
            continue
        if len(counts) and path.stat().st_size > 2_000_000:
            continue
        lang = languages.detect_language(str(path))
        if lang:
            counts[lang] += 1
    return counts


def time_repo_pass(runs: int) -> dict:
    """parse_repo split: per-file walk (summed parse) vs repository binding."""
    import victor_codegraph

    root = str(REPO)
    selected = [
        str(REPO / rel) for rels in SAMPLES.values() for rel in rels if (REPO / rel).is_file()
    ]
    walk_ms, total_ms = [], []
    for _ in range(runs):
        t0 = time.perf_counter()
        for rel in selected:
            victor_codegraph.parse(
                (REPO / rel).read_text(encoding="utf-8", errors="ignore"),
                file_path=str(REPO / rel),
                emit_references=True,
            )
        walk_ms.append((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter()
        victor_codegraph.parse_repo(root, selected_paths=selected, emit_references=True)
        total_ms.append((time.perf_counter() - t0) * 1000)
    return {"walk_ms": walk_ms, "repo_total_ms": total_ms}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--skip-mix", action="store_true", help="skip the file-count walk")
    args = parser.parse_args(argv)
    # Evict the __editable__ finder FIRST: it intercepts victor_codegraph
    # before sys.path, so a worktree checkout never wins otherwise (the
    # sitecustomize also drops anvaiops paths that shadow scripts/).
    sys.path.insert(0, "/tmp/tier3-shim")
    import sitecustomize  # noqa: F401

    sys.path.insert(0, str(REPO / "victor-codegraph"))

    import statistics

    per_file = time_per_file(args.runs)
    print("=== per-file parse (ms/file, median of runs) ===")
    for key in sorted(per_file):
        print(f"  {key:<18} {statistics.median(per_file[key]):8.1f}")

    repo = time_repo_pass(args.runs)
    walk = statistics.median(repo["walk_ms"])
    total = statistics.median(repo["repo_total_ms"])
    print("=== repository pass (sample subset) ===")
    print(f"  per-file walk sum: {walk:8.1f} ms")
    print(f"  parse_repo total : {total:8.1f} ms")
    print(f"  binding overhead : {max(0.0, total - walk):8.1f} ms")

    if not args.skip_mix:
        counts = repo_mix()
        total_files = sum(counts.values())
        print(f"=== repo mix ({total_files} source files) ===")
        for lang, n in counts.most_common(10):
            print(f"  {lang:<14} {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
