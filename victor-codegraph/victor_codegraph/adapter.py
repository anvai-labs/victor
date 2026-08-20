"""Projection to the ProximaDB substrate-keystone ``ProximaRecord`` shape.

Per ProximaDB ``CODE_GRAPH_CORRELATED_SUBSTRATE_2026_06_22.adoc`` a code symbol is *one*
record addressable as a relational row, a graph node, and a vector at once. This adapter
emits the **shape** as plain dicts — it does not import proximadb, embed, or write. The
consumer (Victor embedded, AnvaiOps service) supplies the embedder and the DB write.
"""

from __future__ import annotations

import hashlib
import math
import os
from numbers import Real
from typing import Any, Callable, Protocol

from .model import CodeRelation, CodeSymbol, stable_symbol_oid

Embedder = Callable[[str], list[float]]
BatchEmbedder = Callable[[list[str]], list[list[float]]]


class RecordSource(Protocol):
    """Structural input accepted by the adapter (`ParsedCode` or repository index)."""

    symbols: list[CodeSymbol]
    relations: list[CodeRelation]


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
    return f"graph/{repo_graph_id}/node/{symbol.legacy_id or symbol.id}"


def _canonical_symbol_oid(repo_graph_id: str, symbol: CodeSymbol) -> str:
    """Line-independent canonical oid (ADR-044) — the correlation join key."""
    key = (
        symbol.id
        if symbol.identity_version == "v2"
        else stable_symbol_oid(
            repo_graph_id, symbol.language, symbol.fully_qualified_name, symbol.signature
        )
    )
    return f"graph/{repo_graph_id}/node/{key}"


def _validate_vector(values: list[float], dim: int) -> list[float]:
    vector = list(values)
    if len(vector) != dim:
        raise ValueError(f"embedding dimension mismatch: expected {dim}, received {len(vector)}")
    if any(not isinstance(value, Real) or not math.isfinite(float(value)) for value in vector):
        raise ValueError("embedding values must be finite numeric values")
    return vector


def _external_oid(repo_graph_id: str, relation: CodeRelation) -> str:
    ref = relation.target_ref
    raw = (
        "\x1f".join(
            (
                ref.name,
                ref.qualifier or "",
                str(ref.arity) if ref.arity is not None else "",
                ref.text or "",
            )
        )
        if ref is not None
        else relation.to_symbol_id
    )
    digest = hashlib.blake2b(raw.encode("utf-8"), digest_size=16).hexdigest()
    return f"graph/{repo_graph_id}/external/{digest}"


def _content_version(symbol: CodeSymbol) -> str:
    """Body fingerprint for staleness/dedup — NOT identity (a body edit bumps this,
    not the oid)."""
    explicit = symbol.metadata.get("content_version")
    if explicit:
        return str(explicit)
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
    if embedder is not None and not symbol.metadata.get("embedding_excluded"):
        record["embeddings"].append(
            {
                "model_id": model_id,
                "modality": "code",
                "dim": dim,
                "values": _validate_vector(embedder(symbol.source_code), dim),
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
        if relation.target_ref is not None and symbol_id == relation.to_symbol_id:
            return _external_oid(repo_graph_id, relation)
        if id_map is not None:
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
            "call_sites": (
                [
                    {
                        "file": relation.call_site.file_path,
                        "line": relation.call_site.start_line,
                        "column": relation.call_site.start_column,
                    }
                ]
                if relation.call_site is not None
                else []
            ),
            # legacy endpoints for mixed-read resolution.
            "legacy_from_oid": f"graph/{repo_graph_id}/node/{relation.from_symbol_id}",
            "legacy_to_oid": f"graph/{repo_graph_id}/node/{relation.to_symbol_id}",
        },
    }


def to_proxima_records(
    parsed: RecordSource,
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

    id_map: dict[str, str] = {}
    use_stable = _stable_oid_enabled(stable_oid)
    for symbol in parsed.symbols:
        primary = (
            _canonical_symbol_oid(repo_graph_id, symbol)
            if use_stable
            else _legacy_symbol_oid(repo_graph_id, symbol)
        )
        for key in (symbol.id, symbol.legacy_id):
            if key:
                id_map[key] = primary
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
        embeddable = [
            (index, symbol)
            for index, symbol in enumerate(parsed.symbols)
            if not symbol.metadata.get("embedding_excluded")
        ]
        vectors = (
            batch_embedder([symbol.source_code for _, symbol in embeddable]) if embeddable else []
        )
        if len(vectors) != len(embeddable):
            raise ValueError(
                "batch_embedder must return one vector per symbol "
                f"(expected {len(embeddable)}, received {len(vectors)})"
            )
        for (index, _symbol), values in zip(embeddable, vectors):
            record = records[index]
            record["embeddings"].append(
                {
                    "model_id": model_id,
                    "modality": "code",
                    "dim": dim,
                    "values": _validate_vector(values, dim),
                }
            )
    by_oid = [r["oid"] for r in records]
    if len(by_oid) != len(set(by_oid)):
        raise ValueError("canonical symbol identity collision in adapter input")

    external_records: dict[str, dict[str, Any]] = {}
    for relation in parsed.relations:
        if relation.target_ref is None:
            continue
        oid = _external_oid(repo_graph_id, relation)
        ref = relation.target_ref
        external_records.setdefault(
            oid,
            {
                "oid": oid,
                "labels": ["graph_node", "external_symbol"],
                "branch_id": branch_id,
                "props": {
                    "name": ref.name,
                    "qualifier": ref.qualifier,
                    "arity": ref.arity,
                    "reference_text": ref.text,
                    "resolved": False,
                },
                "embeddings": [],
            },
        )
    records.extend(external_records.values())
    edge_records: dict[tuple[str, str, str], dict[str, Any]] = {}
    for relation in parsed.relations:
        record = relation_to_record(
            relation, repo_graph_id, branch_id, id_map=id_map, stable_oid=stable_oid
        )
        edge = record["edge"]
        key = (edge["from_oid"], edge["to_oid"], edge["edge_type"])
        existing = edge_records.get(key)
        if existing is None:
            edge_records[key] = record
            continue
        sites = existing["props"]["call_sites"]
        for site in record["props"]["call_sites"]:
            if site not in sites:
                sites.append(site)
        sites.sort(key=lambda site: (site["file"], site["line"], site["column"]))
        if sites:
            existing["props"]["line"] = sites[0]["line"]
        existing["props"]["confidence"] = max(
            existing["props"]["confidence"], record["props"]["confidence"]
        )
    records.extend(edge_records.values())
    return records
