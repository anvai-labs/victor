"""Projection to the ProximaDB substrate-keystone ``ProximaRecord`` shape.

Per ProximaDB ``CODE_GRAPH_CORRELATED_SUBSTRATE_2026_06_22.adoc`` a code symbol is *one*
record addressable as a relational row, a graph node, and a vector at once. This adapter
emits the **shape** as plain dicts — it does not import proximadb, embed, or write. The
consumer (Victor embedded, AnvaiOps service) supplies the embedder and the DB write.
"""

from __future__ import annotations

import hashlib
import os
from typing import Any, Callable

from .model import CodeRelation, CodeSymbol, ParsedCode, stable_symbol_oid

Embedder = Callable[[str], list[float]]
BatchEmbedder = Callable[[list[str]], list[list[float]]]

# ADR-044 mixed-read gate. **P2 cutover (2026-06-28): default ON** — the record `oid` is
# the line-independent canonical form, gated behind the parity ratchet
# (tests/test_symbol_oid_parity.py: no collisions / completeness / line-shift stability).
# BOTH ids are always emitted in props so readers dual-read and legacy collections still
# resolve. Opt out per-call (`stable_oid=False`) or per-process (`VICTOR_CODEGRAPH_STABLE_OID=0`).
_STABLE_OID_ENV = "VICTOR_CODEGRAPH_STABLE_OID"


def _stable_oid_enabled(override: bool | None = None) -> bool:
    if override is not None:
        return override
    val = os.getenv(_STABLE_OID_ENV)
    if val is None or val.strip() == "":
        return True  # ADR-044 P2: canonical is the default (parity-ratchet-gated)
    return val.strip().lower() in ("1", "true", "yes", "on")


def _legacy_symbol_oid(repo_graph_id: str, symbol: CodeSymbol) -> str:
    """Line-coupled alias (today's id) — retained for the mixed-read bake."""
    return f"graph/{repo_graph_id}/node/{symbol.id}"


def _canonical_symbol_oid(repo_graph_id: str, symbol: CodeSymbol) -> str:
    """Line-independent canonical oid (ADR-044) — the correlation join key."""
    key = stable_symbol_oid(
        repo_graph_id, symbol.language, symbol.fully_qualified_name, symbol.signature
    )
    return f"graph/{repo_graph_id}/node/{key}"


def _content_version(symbol: CodeSymbol) -> str:
    """Body fingerprint for staleness/dedup — NOT identity (a body edit bumps this,
    not the oid)."""
    return hashlib.blake2b(symbol.source_code.encode("utf-8"), digest_size=8).hexdigest()


def symbol_to_record(
    symbol: CodeSymbol,
    repo_graph_id: str,
    branch_id: str = "main",
    embedder: Embedder | None = None,
    model_id: str = "bge-small-en-v1.5",
    dim: int = 384,
    *,
    stable_oid: bool | None = None,
) -> dict[str, Any]:
    """Project one symbol to a node record (row + graph node + optional vector).

    Always emits BOTH the canonical line-independent oid and the legacy line-coupled
    one (ADR-044 dual-emit); the primary record ``oid`` is the canonical one only when
    the stable-oid gate is on (``stable_oid`` arg or ``VICTOR_CODEGRAPH_STABLE_OID``),
    else legacy — so existing collections are byte-identical until cutover. Consumers
    read ``name`` / ``line`` / ``fully_qualified_name`` from props (never by parsing the
    oid), so the oid can be opaque.
    """

    legacy = _legacy_symbol_oid(repo_graph_id, symbol)
    canonical = _canonical_symbol_oid(repo_graph_id, symbol)
    oid = canonical if _stable_oid_enabled(stable_oid) else legacy
    record: dict[str, Any] = {
        "oid": oid,
        "labels": ["graph_node", "code_symbol"],
        "branch_id": branch_id,
        "props": {
            "name": symbol.simple_name,
            "fully_qualified_name": symbol.fully_qualified_name,
            "file": symbol.location.file_path,
            "line": symbol.location.start_line,
            "end_line": symbol.location.end_line,
            "lang": symbol.language,
            "ast_kind": symbol.symbol_type.name,
            "signature": symbol.signature,
            "visibility": "private" if "private" in symbol.modifiers else "public",
            "module_path": "::".join(symbol.scope_chain),
            "snippet": symbol.source_code,
            "documentation": symbol.documentation,
            # ADR-044 dual-emit: both ids always present for mixed-read resolution,
            # plus the body fingerprint (staleness/dedup, not identity).
            "stable_oid": canonical,
            "legacy_oid": legacy,
            "content_version": _content_version(symbol),
        },
        "embeddings": [],
    }
    if embedder is not None:
        record["embeddings"].append(
            {
                "model_id": model_id,
                "modality": "code",
                "dim": dim,
                "values": embedder(symbol.source_code),
            }
        )
    return record


def relation_to_record(
    relation: CodeRelation,
    repo_graph_id: str,
    branch_id: str = "main",
    *,
    id_map: dict[str, str] | None = None,
    stable_oid: bool | None = None,
) -> dict[str, Any]:
    """Project one relation to an edge record.

    Edge IDENTITY is ``(from_oid, to_oid, edge_type)`` — the call-site line is a prop,
    not identity, so the edge is line-independent once its endpoints are. ``id_map``
    (built by :func:`to_proxima_records`) maps a symbol's legacy id to its canonical
    oid; when the gate is on, endpoints resolve through it so edges and nodes agree.
    """

    def _endpoint(symbol_id: str) -> str:
        if id_map is not None and _stable_oid_enabled(stable_oid):
            mapped = id_map.get(symbol_id)
            if mapped is not None:
                return mapped
        return f"graph/{repo_graph_id}/node/{symbol_id}"

    return {
        "labels": ["graph_edge"],
        "branch_id": branch_id,
        "edge": {
            "from_oid": _endpoint(relation.from_symbol_id),
            "to_oid": _endpoint(relation.to_symbol_id),
            "edge_type": relation.relation_type.name,
        },
        "props": {
            "confidence": relation.confidence,
            "context": relation.context,
            # call-site line (0 when unknown) — a prop, NOT part of edge identity.
            "line": (relation.call_site.start_line if relation.call_site is not None else 0),
            # legacy endpoints for mixed-read resolution.
            "legacy_from_oid": f"graph/{repo_graph_id}/node/{relation.from_symbol_id}",
            "legacy_to_oid": f"graph/{repo_graph_id}/node/{relation.to_symbol_id}",
        },
    }


def to_proxima_records(
    parsed: ParsedCode,
    repo_graph_id: str,
    branch_id: str = "main",
    embedder: Embedder | None = None,
    *,
    batch_embedder: BatchEmbedder | None = None,
    model_id: str = "bge-small-en-v1.5",
    dim: int = 384,
    stable_oid: bool | None = None,
) -> list[dict[str, Any]]:
    """Project an entire parsed file to node + edge records (shapes only).

    Builds the legacy-id → canonical-oid map once so edges resolve to the same key the
    nodes use under the gate (ADR-044). Pass ``stable_oid=True`` (or set
    ``VICTOR_CODEGRAPH_STABLE_OID``) to emit canonical oids as primary.

    ``batch_embedder`` embeds every symbol's source in ONE call (preferred: real
    embedding services batch far more efficiently than a per-symbol ``embedder``
    callback). When both are supplied, ``batch_embedder`` wins.

    Staleness contract: consumers MUST compare ``props.content_version`` against
    their stored value before re-embedding a record — the oid is line-independent
    identity and does NOT change on body edits, so it cannot signal staleness.
    """

    id_map = {s.id: _canonical_symbol_oid(repo_graph_id, s) for s in parsed.symbols}
    records = [
        symbol_to_record(
            s,
            repo_graph_id,
            branch_id,
            embedder if batch_embedder is None else None,
            model_id=model_id,
            dim=dim,
            stable_oid=stable_oid,
        )
        for s in parsed.symbols
    ]
    if batch_embedder is not None and parsed.symbols:
        vectors = batch_embedder([s.source_code for s in parsed.symbols])
        for record, values in zip(records, vectors):
            record["embeddings"].append(
                {
                    "model_id": model_id,
                    "modality": "code",
                    "dim": dim,
                    "values": list(values),
                }
            )
    records.extend(
        relation_to_record(r, repo_graph_id, branch_id, id_map=id_map, stable_oid=stable_oid)
        for r in parsed.relations
    )
    return records
