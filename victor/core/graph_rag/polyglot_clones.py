"""Cross-language operation candidates and independently evidenced FFI links."""

from __future__ import annotations

from collections import defaultdict
from contextlib import closing
from itertools import chain
import json
from pathlib import Path
import sqlite3
from typing import Any, Sequence

from victor.core.graph_rag.ast_fingerprints import jaccard_pairs
from victor.core.graph_rag.portable_clones import MIN_PORTABLE_NODES
from victor.core.graph_rag.semantic_clones import MAX_PAIRS, TooManyPairsError
from victor.core.graph_rag.structural_clones import FunctionUnit, StructuralFamily, near_families

MAX_FFI_LINKS = 100_000


def cross_language_families(
    units: Sequence[FunctionUnit], threshold: float
) -> list[StructuralFamily]:
    """Join portable shapes across languages; compress repeated copies first."""
    grouped: dict[tuple[str, tuple[str, ...]], list[int]] = defaultdict(list)
    for i, unit in enumerate(units):
        if unit.portable_size >= MIN_PORTABLE_NODES and not unit.portable_error:
            grouped[(unit.language, unit.portable_hashes)].append(i)
    groups = list(grouped.values())
    matches = jaccard_pairs(
        [units[group[0]].portable_hashes for group in groups],
        threshold,
        eligible=lambda i, j: units[groups[i][0]].language != units[groups[j][0]].language,
    )
    pairs = []
    for a, b, score in matches:
        left, right = groups[a], groups[b]
        # Different languages imply distinct source files. This spanning tree
        # preserves the complete bipartite component without expanding A*B.
        for i, j in chain(((left[0], j) for j in right), ((i, right[0]) for i in left[1:])):
            pairs.append((min(i, j), max(i, j), score))
            if len(pairs) > MAX_PAIRS:
                raise TooManyPairsError("Cross-language connecting edge budget exceeded")
    families = near_families(units, pairs)
    for family in families:
        family.tier = "cross_language"
    return families


def ffi_relationships(
    root: Path,
    units: Sequence[FunctionUnit],
    source_hashes: dict[str, str],
    binding_facts: list[Any],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Read existing graph evidence and join explicit JNI declarations/exports.

    Indexed FFI edges are name-based hints in the current indexer, not verified
    compiler or runtime bindings. Both files must match the index's content
    hashes. Thin indexed wrappers need not exceed the clone-size threshold.
    """
    from victor.core.graph_rag.ffi_bindings import jni_relationships

    links, stats = jni_relationships(binding_facts)
    stats["ffi_index_available"] = 0
    database = root / ".victor" / "project.db"
    if not database.is_file():
        return links, stats
    by_name: dict[tuple[str, str], list[FunctionUnit]] = defaultdict(list)
    for unit in units:
        by_name[(unit.file_path, unit.name)].append(unit)

    def path_key(value: str) -> str:
        path = Path(value)
        if path.is_absolute():
            try:
                return path.relative_to(root).as_posix()
            except ValueError:
                return ""
        return path.as_posix()

    def endpoint(file: str, name: str, line: int, end_line: int) -> dict[str, Any] | None:
        from victor.core.graph_rag.language_parsers import SOURCE_LANGUAGES

        if not isinstance(line, int) or not isinstance(end_line, int) or not 0 < line <= end_line:
            return None
        matches = [unit for unit in by_name.get((file, name), []) if line <= unit.line <= end_line]
        language = SOURCE_LANGUAGES.get(Path(file).suffix.lower())
        if len(matches) > 1 or language is None:
            return None
        return {
            "file": file,
            "symbol": name,
            "line": matches[0].line if matches else line,
            "language": language,
            "body_compared": False,
        }

    try:
        with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)) as db:
            db.execute("PRAGMA query_only=ON")
            # File hashes and edges must belong to one index generation while
            # a watcher can commit updates concurrently in WAL mode.
            db.execute("BEGIN")
            hashes = {
                path_key(file): digest
                for file, digest in db.execute("SELECT file,content_hash FROM graph_file_mtime")
                if isinstance(file, str) and isinstance(digest, str)
            }
            rows = db.execute(
                """
                SELECT s.file,s.name,s.line,s.end_line,t.file,t.name,t.line,t.end_line,e.metadata
                FROM graph_edge e JOIN graph_node s ON s.node_id=e.src
                  JOIN graph_node t ON t.node_id=e.dst
                WHERE e.type='REFERENCES' AND e.metadata LIKE '%FFI%'
                  AND s.type IN ('function','method','constructor')
                  AND t.type IN ('function','method','constructor')
                LIMIT ?
            """,
                (MAX_FFI_LINKS + 1,),
            )
            stats["ffi_index_available"] = 1
            for i, row in enumerate(rows):
                if i == MAX_FFI_LINKS:
                    stats["ffi_truncated_links"] = 1
                    break
                (
                    source_file,
                    source_name,
                    source_line,
                    source_end,
                    target_file,
                    target_name,
                    target_line,
                    target_end,
                    metadata,
                ) = row
                try:
                    evidence = json.loads(metadata)
                except (TypeError, ValueError):
                    continue
                if not isinstance(evidence, dict) or evidence.get("kind") != "FFI":
                    continue
                if not all(
                    isinstance(value, str)
                    for value in (source_file, source_name, target_file, target_name)
                ):
                    continue
                source_file, target_file = path_key(source_file), path_key(target_file)
                if any(
                    not source_hashes.get(file) or source_hashes[file] != hashes.get(file)
                    for file in (source_file, target_file)
                ):
                    stats["ffi_stale_links"] = stats.get("ffi_stale_links", 0) + 1
                    continue
                source = endpoint(
                    source_file, source_name, source_line or 0, source_end or source_line or 0
                )
                target = endpoint(
                    target_file, target_name, target_line or 0, target_end or target_line or 0
                )
                if source is None or target is None or source["language"] == target["language"]:
                    continue
                links.append(
                    {
                        "relationship": "calls_across_ffi",
                        "evidence": "indexed_ffi_reference",
                        "confidence": "index_heuristic",
                        "equivalence_verified": False,
                        "runtime_verified": False,
                        "abi_verified": False,
                        "source": source,
                        "target": target,
                    }
                )
    except (sqlite3.Error, OSError):
        stats["ffi_index_unreadable"] = 1
    return links, stats
