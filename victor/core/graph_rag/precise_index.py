"""ADR-032 tier 3: opt-in precise-index harvesting from offline SCIP dumps.

Compiler-accurate REFERENCES edges from a SCIP index (``scip-python`` —
Sourcegraph's npm-distributed Pyright fork), harvested OFFLINE: one subprocess
dump, protobuf read, identity join, edge upsert. Never a live language server
in the query path.

Contract (mirrors the tier-2.5 capability-probe posture):

- Opt-in (``enable_precise_index``, default OFF). Absence of the toolchain
  OR a working reader degrades with one explicit INFO log — indexing
  continues on the tier-2.5 graph, never fails.
- Identity join is metadata-based, not hash-based: victor's stable symbol
  keys embed the codegraph fqn shape (file-dotted ``.py`` prefix + ``::``
  scope chain) plus the extracted signature, none of which the SCIP symbol
  string carries — so definitions join against the persisted nodes by
  (file, name, kind/scope agreement; scope chains re-derived from parent_id
  links when GraphNode carries no explicit field), accepting ONLY a unique
  match. Occurrences that cannot be joined are counted and dropped — never
  guessed.
- Provenance is ``compiler`` with confidence 1.0. The harvest pass runs LAST
  in the indexing pipeline: graph_edge upserts are last-writer-wins on
  ``(src, dst, type)``, so a compiler payload replaces any scope/heuristic
  metadata on a shared CALLS/INHERITS/IMPLEMENTS key. REFERENCES has no
  heuristic writer by construction.
- Reference→definition pairs where BOTH endpoints are joinable become
  REFERENCES edges; owner attribution is positional (the greatest definition
  offset at or before the reference).
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# SCIP symbol roles (scip.proto SymbolRole): bit flags; 1 = definition.
SCIP_SYMBOL_ROLE_DEFINITION = 1

# A pyright-based dump of a large monorepo takes minutes, not seconds; the
# old 300s ceiling timed real runs out and the skip was mislabeled
# "toolchain-unavailable". Configurable via GraphIndexConfig.
_DUMP_TIMEOUT_SECONDS = 1200

# scip.proto SymbolKind enum values (vendored _scip/scip.proto) mapped to the
# GraphNode.type spellings produced by project_file_symbols (enum name lower()).
_KIND_FUNCTION = 3
_KIND_METHOD = 4
_KIND_CLASS = 5

_KIND_TO_TYPE = {
    _KIND_FUNCTION: "function",
    _KIND_METHOD: "method",
    _KIND_CLASS: "class",
}

# Enclosing-symbol types whose names constitute a codegraph scope chain;
# walking parent_id through them reconstructs what python_parser carries as
# Symbol.scope_chain but GraphNode does not persist as a field.
_CONTAINER_TYPES = frozenset({"class", "function", "method"})


def _norm_rel(path: str) -> str:
    """Normalize a relative path for document/file joining (POSIX, no ./)."""

    return path.replace("\\", "/").removeprefix("./")


@dataclass(frozen=True)
class PreciseHarvest:
    """Result of one SCIP harvest pass."""

    edges: list[tuple[str, str, str, dict[str, Any]]] = field(default_factory=list)
    resolved: int = 0
    unjoined: int = 0
    skipped_reason: Optional[str] = None


def detect_precise_tool(tool: str = "auto") -> Optional[str]:
    """Locate the SCIP indexer: an explicit path or the npm binary on PATH.

    The official toolchain is npm-only (``npm install -g
    @sourcegraph/scip-python``); no pip package provides it (PyPI's ``scip``
    is an unrelated project), so there is no module form to probe.
    """

    if tool not in ("auto", "scip-python") and Path(tool).exists():
        return tool
    return shutil.which("scip-python")


def harvest_scip_index(
    root: Path,
    output_path: Path,
    tool: str = "auto",
    *,
    project_name: str = "victor-harvest",
    timeout_seconds: int = _DUMP_TIMEOUT_SECONDS,
) -> Optional[Path]:
    """Run one offline SCIP dump; return the index path or None on failure.

    ``TimeoutExpired`` is distinguished in the log: a timed-out pyright dump
    is a size problem, not a missing toolchain.
    """

    resolved_tool = detect_precise_tool(tool)
    if resolved_tool is None:
        logger.info("precise-index harvesting unavailable: no scip-python toolchain")
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [resolved_tool, "index", "--project-name", project_name, "--output", str(output_path)]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            cwd=str(root),
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        logger.warning(
            "precise-index dump timed out after %ds (raise precise_index_timeout_seconds "
            "for large corpora)",
            timeout_seconds,
        )
        return None
    except OSError as exc:
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
    """Parse the SCIP protobuf via the vendored generated bindings.

    An ImportError means the protobuf runtime is absent — the minimal
    install does not pull it — which is reader ABSENCE, not a failure:
    degrade to no-reader per the tier-3 contract.
    """

    try:
        from victor.core.graph_rag._scip import scip_pb2
    except ImportError:
        logger.info("precise-index reader unavailable: protobuf runtime not installed")
        return None

    index = scip_pb2.Index()
    try:
        index.ParseFromString(path.read_bytes())
    except Exception as exc:
        logger.warning("precise-index parse failed: %s", exc)
        return None
    return index


def python_symbol_to_parts(symbol: str) -> Optional[dict[str, str]]:
    """Split a SCIP python symbol string into file marker + dotted path.

    SCIP python symbols look like ``python src/pkg/mod.py`pkg.mod.func().``
    — the backtick separates the language+file marker from the symbol path,
    with trailing disambiguators (``()`` parameters, ``.`` accessor). Returns
    ``{"file": ..., "path": ...}`` or None when the symbol is not a
    definition-shaped python symbol we can map.
    """

    if not symbol.startswith("python "):
        return None
    _, _, tail = symbol.partition(" ")
    file_marker, sep, path = tail.partition("`")
    if not sep or not path:
        return None
    # Strip SCIP disambiguator suffixes: ``()`` (empty params) / ``(a, b)``
    # parameter lists / the trailing ``.`` accessor.
    path = path.split("(", 1)[0].rstrip(".")
    if not path:
        return None
    return {"file": file_marker.strip(), "path": path}


def _node_simple_name(path: str) -> str:
    """The codegraph simple_name for a scip symbol path (last component)."""

    return path.rsplit(".", 1)[-1]


def _scope_chain(node: Any, nodes_by_id: dict[str, Any]) -> list[str]:
    """The node's scope chain: explicit field when present, else parent links.

    GraphNode persists ``parent_id`` (CONTAINS) instead of the parser's
    ``scope_chain``; walking container-typed parents reconstructs it. An
    unknown parent id (file/module node or pruned parent) ends the walk —
    partial chains still compare truthfully below.
    """

    explicit = getattr(node, "scope_chain", None)
    if explicit is not None:
        return list(explicit)
    chain: list[str] = []
    seen: set[str] = set()
    current: Any = node
    while True:
        parent_id = getattr(current, "parent_id", None)
        if not parent_id or parent_id in seen:
            break
        parent = nodes_by_id.get(parent_id)
        if parent is None or str(getattr(parent, "type", "")).lower() not in _CONTAINER_TYPES:
            break
        seen.add(parent_id)
        chain.insert(0, parent.name)
        current = parent
    return chain


def _scope_matches(scope: list[str], name: str, path: str) -> bool:
    """Does ``[*scope, name]`` equal the scip path's trailing components?

    A node's codegraph fqn is ``file_dotted_prefix + "::" + scope chain +
    name``; the scip path is the python-visible dotted module path. Matching
    only the trailing components accepts any scip module-naming convention
    (src-layout alignment included) as long as the symbol itself lines up.
    """

    expected = [*scope, name]
    components = path.split(".")
    return len(components) >= len(expected) and components[-len(expected) :] == expected


def _kind_by_symbol(document: Any) -> dict[str, int]:
    """Symbol string → scip SymbolKind from the document's symbol table."""

    table: dict[str, int] = {}
    for info in getattr(document, "symbols", None) or []:
        symbol = getattr(info, "symbol", None)
        if symbol:
            table[symbol] = getattr(info, "kind", 0)
    return table


def harvest_precise_edges(
    index_path: Path,
    nodes_by_file: dict[str, list[Any]],
    *,
    timeout_seconds: int = _DUMP_TIMEOUT_SECONDS,
) -> PreciseHarvest:
    """Join SCIP occurrences against the materialized graph nodes.

    ``nodes_by_file`` maps each document's relative path to its persisted
    GraphNode objects (node_id/name/type/file, plus parent_id or an explicit
    scope_chain). Definitions join by (file, name) with kind/scope agreement,
    accepting only a unique match; signatures live in the persisted keys and
    are never re-derived from the scip data.
    """

    del timeout_seconds  # dump-side knob; kept for call-site compatibility
    index = _read_index(index_path)
    if index is None:
        return PreciseHarvest(skipped_reason="no-reader")
    return join_occurrences(index, nodes_by_file)


def join_occurrences(index: Any, nodes_by_file: dict[str, list[Any]]) -> PreciseHarvest:
    """The join core, over an already-parsed index (testable without a reader)."""

    nodes_by_id = {n.node_id: n for nodes in nodes_by_file.values() for n in nodes}

    def chain_for(node: Any) -> list[str]:
        return _scope_chain(node, nodes_by_id)

    # Pass 1: definition-role occurrences → node ids, unique-match only.
    definition_keys: dict[str, str] = {}
    unjoined = 0
    for document in index.documents:
        relative_path = _norm_rel(getattr(document, "relative_path", ""))
        nodes = nodes_by_file.get(relative_path) or _suffix_match(nodes_by_file, relative_path)
        kinds = _kind_by_symbol(document)
        for occurrence in document.occurrences:
            if not (occurrence.symbol_roles & SCIP_SYMBOL_ROLE_DEFINITION):
                continue
            parts = python_symbol_to_parts(occurrence.symbol)
            node_id = (
                _unique_join(parts, nodes, kinds.get(occurrence.symbol), chain_for)
                if parts is not None
                else None
            )
            if node_id is None:
                unjoined += 1
                continue
            definition_keys[occurrence.symbol] = node_id

    # Pass 2: references → edges (both endpoints must be joinable).
    edges: list[tuple[str, str, str, dict[str, Any]]] = []
    seen: set[tuple[str, str]] = set()
    resolved = 0
    for document in index.documents:
        for occurrence in document.occurrences:
            if occurrence.symbol_roles & SCIP_SYMBOL_ROLE_DEFINITION:
                continue
            target_key = definition_keys.get(occurrence.symbol)
            if target_key is None:
                unjoined += 1
                continue
            owner = _enclosing_definition_key(document, occurrence, definition_keys)
            if owner is None:
                # Reference before any joinable definition in its document:
                # unjoined evidence, never silently dropped.
                unjoined += 1
                continue
            if owner == target_key or (owner, target_key) in seen:
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


def _unique_join(
    parts: dict[str, str],
    nodes: list[Any],
    kind: Optional[int],
    chain_for: Any,
) -> Optional[str]:
    """Unique (file, name, scope, kind) match; ambiguity stays unjoined."""

    file_marker = parts["file"]
    if file_marker:
        # The symbol string NAMES its defining file: a same-named symbol in
        # another file is a different symbol and must not compete. No node
        # matching the marker at all → unjoined, never retried wider.
        candidates = [n for n in nodes if n.file.endswith(file_marker)]
        if not candidates:
            return None
    else:
        candidates = nodes
    if kind is not None and kind in _KIND_TO_TYPE:
        # scip's kind is ground truth: only nodes of that type may join. Kinds
        # we don't map (parameters, locals, ...) keep the untyped population —
        # our ignorance must not silently discard evidence.
        expected = _KIND_TO_TYPE[kind]
        typed = [n for n in candidates if str(getattr(n, "type", "")).lower() == expected]
        if not typed:
            return None
        candidates = typed
    simple = _node_simple_name(parts["path"])
    named = [n for n in candidates if n.name == simple]
    if not named:
        return None
    if len(named) == 1:
        return named[0].node_id
    # Same name, multiple scopes (e.g. methods of same-named classes):
    # disambiguate by the scip path's trailing scope components.
    matched = [n for n in named if _scope_matches(chain_for(n), n.name, parts["path"])]
    return matched[0].node_id if len(matched) == 1 else None


def _suffix_match(nodes_by_file: dict[str, list[Any]], relative_path: str) -> list[Any]:
    """Fallback document→nodes lookup for path-spelling drift (leading dirs).

    Exact normalized match first; only when that misses, a document whose
    path ENDS WITH the indexed key (or vice versa) may bind, mirroring the
    endswith semantics the file marker already uses.
    """

    for key, nodes in nodes_by_file.items():
        if key and (relative_path.endswith(key) or key.endswith(relative_path)):
            return nodes
    return []


def _occurrence_offset(occurrence: Any) -> int:
    """Start offset from whichever range encoding the producer emitted.

    scip.proto carries three encodings: the deprecated packed-int32 ``range``
    ([startLine, startChar, (endLine), (endChar)] — what the npm scip-python
    emits), and the typed flat alternatives ``single_line_range`` {line,
    start_character} / ``multi_line_range`` {start_line, start_character}.
    All normalize to line * 10_000 + character.
    """

    packed = list(getattr(occurrence, "range", ()) or ())
    if packed:
        line = packed[0] if len(packed) > 0 else 0
        character = packed[1] if len(packed) > 1 else 0
        return line * 10_000 + character
    single = _typed_range(occurrence, "single_line_range")
    if single is not None:
        return single.line * 10_000 + single.start_character
    multi = _typed_range(occurrence, "multi_line_range")
    if multi is not None:
        return multi.start_line * 10_000 + multi.start_character
    return 0


def _typed_range(occurrence: Any, field: str) -> Any:
    """A set typed-range message, or None — presence-checked on real protos.

    Proto3 message fields return a default INSTANCE when unset (not None), so
    plain getattr cannot detect presence on real messages; HasField can.
    Test fixtures carry the attribute only when set, where plain getattr is
    the presence check.
    """

    has_field = getattr(occurrence, "HasField", None)
    if has_field is not None:
        try:
            return getattr(occurrence, field) if has_field(field) else None
        except ValueError:  # field not presence-trackable on this schema
            return None
    return getattr(occurrence, field, None)


def _enclosing_definition_key(
    document: Any, occurrence: Any, definition_keys: dict[str, str]
) -> Optional[str]:
    """Best-effort owner attribution: the greatest definition offset at or
    before the occurrence offset."""

    offset = _occurrence_offset(occurrence)
    best: Optional[tuple[int, str]] = None
    for occ in document.occurrences:
        if not (occ.symbol_roles & SCIP_SYMBOL_ROLE_DEFINITION):
            continue
        key = definition_keys.get(occ.symbol)
        if key is None:
            continue
        occ_offset = _occurrence_offset(occ)
        if occ_offset <= offset and (best is None or occ_offset > best[0]):
            best = (occ_offset, key)
    return best[1] if best else None
