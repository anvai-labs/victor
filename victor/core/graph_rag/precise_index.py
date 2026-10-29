"""ADR-032 tier 3: opt-in precise-index harvesting from offline SCIP dumps.

Compiler-accurate REFERENCES edges from SCIP indexes, harvested OFFLINE:
one subprocess dump per language provider, protobuf read, metadata join,
edge upsert. Never a live language server in the query path.

Providers (contract: ADR-032 rev 1.5; ``precise_providers.py``): scip-python
(npm/Pyright) for python, scip-typescript (npm/tsserver) for TypeScript and
JavaScript, rust-analyzer's native ``scip`` CLI for Rust. Runtimes are a
provider detail; the join below is producer-agnostic and consumes the
providers' normalized symbol parts.

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
  offset at or before the reference). References to symbols defined nowhere
  in the index (stdlib, site-packages) are out-of-corpus and uncounted.
- scip-python 0.6.x emits UnspecifiedKind on every symbol information, so
  the kind agreement filter is dormant with the real tool (unmapped kind →
  no filtering, never discarding evidence) and the join rides on
  name + scope agreement; kinds still harden the join for producers that
  populate them.
"""

from __future__ import annotations

import logging
import shutil
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

# scip-python is a Node process whose pyright pass holds the whole program in
# memory: Node's default ~4 GB heap aborts (V8 OOM, core dump) around ~2,200
# files of a 6,707-file monorepo. The harvest injects a larger old-space cap
# when the environment does not set NODE_OPTIONS itself (measured need: the
# victor monorepo dumps cleanly under 16 GB).
_DUMP_NODE_OPTIONS = "--max-old-space-size=16384"

# scip.proto SymbolInformation.Kind values from the VENDORED schema, mapped
# to the GraphNode.type spellings produced by project_file_symbols (enum
# name lower()). Pinned by test against the vendored enum — these values are
# live for producers that populate kinds (scip-typescript, rust-analyzer;
# scip-python 0.6.x emits UnspecifiedKind and never filters).
_KIND_TO_TYPE = {
    7: "class",  # SymbolInformation.Kind.Class
    26: "method",  # ...Method
    17: "function",  # ...Function
    9: "constructor",  # ...Constructor
    49: "struct",  # ...Struct
    21: "interface",  # ...Interface
    11: "enum",  # ...Enum
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
    """Result of one SCIP harvest pass (possibly merged across languages).

    ``per_language`` reports each provider's outcome under the uniform
    degradation taxonomy (``ok:<n> edges`` | ``toolchain-absent`` |
    ``dump-failed`` | ``no-reader``) — polyglot repos get per-language
    honesty, never an aggregate that hides a skipped provider.

    Accounting decomposition (A0-measured on the monorepo corpus): the
    historical ``unjoined`` total conflated three different signals —
    definitions that failed the join predicate, references whose target
    never joined, and OWNERLESS references (no enclosing joined definition
    in their document — import/module-level evidence, an expected shape,
    not a join failure). Documents whose file has no codegraph node bucket
    are out-of-corpus-by-codegraph and counted separately, never as
    failures: their definitions also never enter the definition-symbol pool,
    so references to them elsewhere are out-of-corpus too rather than
    "defined but unjoinable" noise.
    """

    edges: list[tuple[str, str, str, dict[str, Any]]] = field(default_factory=list)
    resolved: int = 0
    unjoined: int = 0
    definitions_unjoined: int = 0
    references_unjoined: int = 0
    references_ownerless: int = 0
    out_of_corpus_documents: int = 0
    out_of_corpus_occurrences: int = 0
    skipped_reason: Optional[str] = None
    per_language: dict[str, str] = field(default_factory=dict)

    @property
    def accounting(self) -> str:
        """One-line decomposition for logs and per-language outcomes."""
        return (
            f"{self.definitions_unjoined} defs, {self.references_unjoined} target-refs, "
            f"{self.references_ownerless} ownerless-refs unjoined; "
            f"{self.out_of_corpus_documents} docs out-of-corpus"
        )


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
    """Run one offline scip-python dump; return the index path or None.

    Compat seam over ``ScipPythonProvider.dump`` (contract rev 1.5): the
    transport, env policy, and failure taxonomy live with the provider —
    one implementation, not a parallel copy here.
    """

    from victor.core.graph_rag.precise_providers import ScipPythonProvider

    del project_name  # provider-owned; the historical call sites used the default
    provider = ScipPythonProvider()
    resolved = provider.detect(tool)
    if resolved is None:
        logger.info("precise-index harvesting unavailable: no scip-python toolchain")
        return None
    return provider.dump(resolved, root, output_path, timeout_seconds=timeout_seconds)


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
    """scip-python symbol -> normalized parts (delegates to the provider).

    Kept as this module's public seam: the join and the tests import it
    here; the grammar itself lives with its producer in
    ``precise_providers.ScipPythonProvider``.
    """

    from victor.core.graph_rag.precise_providers import ScipPythonProvider

    return ScipPythonProvider().symbol_parts(symbol)


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
    symbol_parts: Any = None,
) -> PreciseHarvest:
    """Join SCIP occurrences against the materialized graph nodes.

    ``nodes_by_file`` maps each document's relative path to its persisted
    GraphNode objects (node_id/name/type/file, plus parent_id or an explicit
    scope_chain). Definitions join by (file, name) with kind/scope agreement,
    accepting only a unique match; signatures live in the persisted keys and
    are never re-derived from the scip data. ``symbol_parts`` is the dump's
    producer grammar (default scip-python).
    """

    del timeout_seconds  # dump-side knob; kept for call-site compatibility
    index = _read_index(index_path)
    if index is None:
        return PreciseHarvest(skipped_reason="no-reader")
    return join_occurrences(index, nodes_by_file, symbol_parts=symbol_parts)


def join_occurrences(
    index: Any,
    nodes_by_file: dict[str, list[Any]],
    *,
    symbol_parts: Any = None,
) -> PreciseHarvest:
    """The join core, over an already-parsed index (testable without a reader).

    ``symbol_parts`` is the PRODUCER grammar (contract rev 1.5) — a
    ``symbol -> parts | None`` callable owned by the provider whose dump
    this index came from. Default: the scip-python grammar (the join's
    historical behavior).
    """

    if symbol_parts is None:
        symbol_parts = python_symbol_to_parts
    nodes_by_id = {n.node_id: n for nodes in nodes_by_file.values() for n in nodes}

    def chain_for(node: Any) -> list[str]:
        return _scope_chain(node, nodes_by_id)

    # Document → node bucket, resolved ONCE and shared by both passes. A
    # document whose file has no bucket is out-of-corpus-by-codegraph: no
    # nodes were materialized for that file, so nothing in it can join —
    # its occurrences are counted separately and NEVER as join failures,
    # and its definitions stay out of the pool so references to its symbols
    # elsewhere are out-of-corpus rather than defined-but-unjoinable noise.
    doc_buckets: list[tuple[Any, list[Any]]] = []
    out_of_corpus_documents = 0
    out_of_corpus_occurrences = 0
    for document in index.documents:
        relative_path = _norm_rel(getattr(document, "relative_path", ""))
        nodes = nodes_by_file.get(relative_path)
        if nodes is None:
            nodes = _suffix_match(nodes_by_file, relative_path)
        doc_buckets.append((document, nodes))
        if not nodes:
            out_of_corpus_documents += 1
            out_of_corpus_occurrences += len(document.occurrences)

    # Pass 1: definition-role occurrences → node ids, unique-match only.
    definition_keys: dict[str, str] = {}
    definition_symbols: set[str] = set()
    definitions_unjoined = 0
    for document, nodes in doc_buckets:
        if not nodes:
            continue
        kinds = _kind_by_symbol(document)
        for occurrence in document.occurrences:
            if not (occurrence.symbol_roles & SCIP_SYMBOL_ROLE_DEFINITION):
                continue
            parts = symbol_parts(occurrence.symbol)
            if parts is None:
                # Non-joinable descriptor (module init, parameter, local):
                # absence by construction, not a join failure — uncounted.
                continue
            definition_symbols.add(occurrence.symbol)
            node_id = _unique_join(parts, nodes, kinds.get(occurrence.symbol), chain_for)
            if node_id is None:
                definitions_unjoined += 1
                continue
            definition_keys[occurrence.symbol] = node_id

    # Pass 2: references → edges (both endpoints must be joinable).
    # A reference to a symbol with NO definition anywhere in the index is
    # out-of-corpus evidence (builtins, stdlib, site-packages — or a symbol
    # whose document is out-of-corpus-by-codegraph): not joinable by design,
    # so it is neither resolved nor unjoined.
    edges: list[tuple[str, str, str, dict[str, Any]]] = []
    seen: set[tuple[str, str]] = set()
    resolved = 0
    references_unjoined = 0
    references_ownerless = 0
    for document, nodes in doc_buckets:
        if not nodes:
            continue
        for occurrence in document.occurrences:
            if occurrence.symbol_roles & SCIP_SYMBOL_ROLE_DEFINITION:
                continue
            if occurrence.symbol not in definition_symbols:
                continue
            target_key = definition_keys.get(occurrence.symbol)
            if target_key is None:
                references_unjoined += 1
                continue
            owner = _enclosing_definition_key(document, occurrence, definition_keys)
            if owner is None:
                # Ownerless reference: no enclosing JOINED definition in its
                # document — import/module-level evidence, an expected
                # shape. Counted separately so the failure metric stays
                # meaningful, never silently dropped.
                references_ownerless += 1
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
    return PreciseHarvest(
        edges=edges,
        resolved=resolved,
        unjoined=definitions_unjoined + references_unjoined + references_ownerless,
        definitions_unjoined=definitions_unjoined,
        references_unjoined=references_unjoined,
        references_ownerless=references_ownerless,
        out_of_corpus_documents=out_of_corpus_documents,
        out_of_corpus_occurrences=out_of_corpus_occurrences,
    )


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

    Exact normalized match first; only when that misses, a key that is a
    PATH-SEGMENT-aligned suffix of the document path (or vice versa) may
    bind — ``./src/m.py`` ↔ ``src/m.py`` — while ``tests/lib.py`` never
    binds ``lib.py``. Segment alignment matters: the symbols carry no file
    marker, so this fallback is the only file binding on a drift miss and
    a raw endswith would cross files.
    """

    for key, nodes in nodes_by_file.items():
        if key and (
            relative_path.endswith("/" + key)
            or key.endswith("/" + relative_path)
            or key == relative_path
        ):
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
