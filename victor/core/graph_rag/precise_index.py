"""ADR-032 tier 3: opt-in precise-index harvesting from offline SCIP dumps.

Compiler-accurate REFERENCES edges from a SCIP index (``scip-python`` —
Sourcegraph's Pyright fork — first), harvested OFFLINE: one subprocess dump,
protobuf read, identity join, edge upsert. Never a live language server in
the query path.

Contract (mirrors the tier-2.5 capability-probe posture):

- Opt-in (``enable_precise_index``, default OFF). Absence of the toolchain
  OR the protobuf reader degrades with one explicit INFO log — indexing
  continues on the tier-2.5 graph, never fails.
- Identity join is line-drift-safe: each SCIP definition is re-keyed through
  victor-codegraph's ``stable_symbol_key`` v2 (language ‖ type ‖ file ‖ fqn
  ‖ normalized signature), the same identity the graph nodes carry.
  Occurrences that cannot be joined are counted and dropped — never guessed.
- Provenance is ``compiler`` with confidence 1.0. The harvest pass runs LAST
  in the indexing pipeline: graph_edge upserts are last-writer-wins on
  ``(src, dst, type)``, so a compiler payload replaces any scope/heuristic
  metadata on a shared CALLS/INHERITS/IMPLEMENTS key, and REFERENCES has no
  heuristic writer by construction.
- Reference→definition pairs where BOTH endpoints are joinable become
  REFERENCES edges; definition-site type relationships (base classes) become
  INHERITS edges in a follow-on — v1 emits REFERENCES only.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)

# SCIP symbol roles (scip.proto SymbolRole): bit flags; 1 = definition.
SCIP_SYMBOL_ROLE_DEFINITION = 1

_DUMP_TIMEOUT_SECONDS = 300


@dataclass(frozen=True)
class PreciseHarvest:
    """Result of one SCIP harvest pass."""

    edges: list[tuple[str, str, str, dict[str, Any]]] = field(default_factory=list)
    resolved: int = 0
    unjoined: int = 0
    skipped_reason: Optional[str] = None


def detect_precise_tool(tool: str = "auto") -> Optional[str]:
    """Locate the SCIP indexer: an explicit path, a binary on PATH, or a
    ``python -m scip_python`` module invocation. None when unavailable."""

    if tool not in ("auto", "scip-python") and Path(tool).exists():
        return tool
    binary = shutil.which("scip-python")
    if binary is not None:
        return binary
    try:
        from scip_python import scip  # noqa: F401

        return f"{_python_executable()} -m scip_python.index"
    except Exception:
        return None


def _python_executable() -> str:
    import sys

    return sys.executable or "python"


def _dump_command(tool: str) -> list[str]:
    if tool.endswith(" -m scip_python.index"):
        return [*tool.split(" "), "index"]
    return [tool, "index"]


def harvest_scip_index(
    root: Path,
    output_path: Path,
    tool: str = "auto",
    *,
    project_name: str = "victor-harvest",
) -> Optional[Path]:
    """Run one offline SCIP dump; return the index path or None on any
    failure (structured, logged, non-fatal)."""

    resolved_tool = detect_precise_tool(tool)
    if resolved_tool is None:
        logger.info("precise-index harvesting unavailable: no scip-python toolchain")
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        *_dump_command(resolved_tool),
        "--project-name",
        project_name,
        "--output",
        str(output_path),
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            cwd=str(root),
            timeout=_DUMP_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("precise-index dump failed (%s): %s", type(exc).__name__, exc)
        return None
    if result.returncode != 0 or not output_path.exists():
        logger.warning(
            "precise-index dump exited %d: %s",
            result.returncode,
            (result.stderr or "").strip()[:400],
        )
        return None
    return output_path


def _read_index(path: Path) -> Optional[Any]:
    """Parse the SCIP protobuf via the generated ``scip`` module (lazy import
    — absence degrades like toolchain absence)."""

    try:
        from scip_python import scip
    except Exception:
        try:
            import scip  # type: ignore[no-redef]
        except Exception:
            logger.info("precise-index harvesting unavailable: no SCIP protobuf reader")
            return None
    index = scip.Index()
    try:
        index.ParseFromString(path.read_bytes())
    except Exception as exc:
        logger.warning("precise-index parse failed: %s", exc)
        return None
    return index


def python_symbol_to_identity(symbol: str, relative_path: str) -> Optional[dict[str, str]]:
    """Map a SCIP python symbol string to victor-codegraph identity inputs.

    SCIP python symbols look like ``python src/pkg/mod.py`pkg.Class.method().``
    — the backtick separates the language+file marker from the FQN, with
    trailing disambiguators (``()`` parameters, ``.`` accessor). Returns the
    pieces ``stable_symbol_key`` needs: (fqn, symbol_type_name) or None when
    the symbol is not a definition-shaped python symbol we can name-map.
    """

    if not symbol.startswith("python "):
        return None
    _, _, tail = symbol.partition(" ")
    file_marker, sep, fqn = tail.partition("`")
    if not sep or not fqn:
        return None
    marker_file = file_marker.strip()
    if marker_file and marker_file != relative_path:
        return None
    # Strip SCIP disambiguator suffixes: ``()`` (empty params) / ``(a, b)`` /
    # property accessors — the victor FQN is the dotted path itself.
    if fqn.endswith("()."):
        fqn = fqn[:-3]
    elif fqn.endswith("()"):
        fqn = fqn[:-2]
    if not fqn or "." not in fqn and not fqn.isidentifier():
        return None
    return {"fqn": fqn, "symbol_type_name": "FUNCTION"}


def harvest_precise_edges(
    index_path: Path,
    root: Path,
    key_for_identity: Any,
    known_symbol_keys: set[str],
    *,
    fqn_overrides: Optional[dict[str, dict[str, str]]] = None,
) -> PreciseHarvest:
    """Join SCIP occurrences against the materialized graph nodes.

    ``key_for_identity(identity: dict) -> str`` recomputes
    victor-codegraph's ``stable_symbol_key`` (v2) from the identity inputs;
    ``known_symbol_keys`` is the set of node ids already persisted.
    ``fqn_overrides`` lets the caller supply exact (file, type, signature)
    triples keyed by FQN when the symbol table carries them — v1 maps
    definitions by FQN alone and counts misses.
    """

    index = _read_index(index_path)
    if index is None:
        return PreciseHarvest(skipped_reason="no-reader")
    return join_occurrences(index, key_for_identity, known_symbol_keys, fqn_overrides)


def join_occurrences(
    index: Any,
    key_for_identity: Any,
    known_symbol_keys: set[str],
    fqn_overrides: Optional[dict[str, dict[str, str]]] = None,
) -> PreciseHarvest:
    """The join core, over an already-parsed index (testable without a reader)."""

    # Pass 1: definition sites → identity keys.
    definition_keys: dict[str, str] = {}
    unjoined = 0
    for document in index.documents:
        relative_path = getattr(document, "relative_path", "")
        for occurrence in document.occurrences:
            if not (occurrence.symbol_roles & SCIP_SYMBOL_ROLE_DEFINITION):
                continue
            identity = python_symbol_to_identity(occurrence.symbol, relative_path)
            if identity is None:
                unjoined += 1
                continue
            override = (fqn_overrides or {}).get(identity["fqn"])
            if override:
                identity = {**identity, **override}
            key = key_for_identity(identity)
            if key in known_symbol_keys:
                definition_keys[occurrence.symbol] = key
            else:
                unjoined += 1

    # Pass 2: references → edges (both endpoints must be joinable).
    edges: list[tuple[str, str, str, dict[str, Any]]] = []
    seen: set[tuple[str, str]] = set()
    resolved = 0
    for document in index.documents:
        relative_path = getattr(document, "relative_path", "")
        for occurrence in document.occurrences:
            if occurrence.symbol_roles & SCIP_SYMBOL_ROLE_DEFINITION:
                continue
            target_key = definition_keys.get(occurrence.symbol)
            if target_key is None:
                unjoined += 1
                continue
            # Owner attribution: the enclosing definition for this reference,
            # by position — v1 uses the document's LAST definition at or
            # before the occurrence range start.
            owner = _enclosing_definition_key(document, occurrence, definition_keys)
            if owner is None or owner == target_key or (owner, target_key) in seen:
                continue
            seen.add((owner, target_key))
            resolved += 1
            edges.append(
                (
                    owner,
                    target_key,
                    "REFERENCES",
                    {"provenance": "compiler", "confidence": 1.0, "use": "load"},
                )
            )
    return PreciseHarvest(edges=edges, resolved=resolved, unjoined=unjoined)


def _enclosing_definition_key(
    document: Any, occurrence: Any, definition_keys: dict[str, str]
) -> Optional[str]:
    """Best-effort owner attribution: the definition whose range encloses
    (or most recently precedes) the occurrence."""

    from victor.core.graph_rag.codegraph_repository import ProjectedRelationEdge  # noqa: F401

    best: Optional[tuple[int, str]] = None
    position = occurrence.range.start.character if occurrence.range.start else 0
    line = occurrence.range.start.line if occurrence.range.start else 0
    offset = line * 10_000 + position
    for occ in document.occurrences:
        if not (occ.symbol_roles & SCIP_SYMBOL_ROLE_DEFINITION):
            continue
        key = definition_keys.get(occ.symbol)
        if key is None:
            continue
        start = occ.range.start
        occ_offset = (start.line if start else 0) * 10_000 + (start.character if start else 0)
        if occ_offset <= offset and (best is None or occ_offset > best[0]):
            best = (occ_offset, key)
    return best[1] if best else None
