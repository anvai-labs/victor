"""Seed strategies for multi-hop retrieval, with an honest report of what ran.

Multi-hop retrieval starts from a ranked set of seed nodes and expands them
through the graph. Where those seeds come from decides what the retrieval
actually measures:

- ``structural``: the store's symbol search only (FTS5 on the sqlite store).
- ``hybrid``: the same keyword seeds fused with vector-similarity seeds from
  the node embeddings persisted at index time, via reciprocal rank fusion.

A hybrid retrieval whose vector leg is missing or fails still returns keyword
seeds, so callers must be able to tell the two apart. Every strategy returns a
:class:`SeedReport` saying which legs contributed and why one did not.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

#: Vector leg outcomes recorded in :attr:`SeedReport.semantic_leg`.
SEMANTIC_USED = "used"
SEMANTIC_EMPTY = "empty"
SEMANTIC_UNAVAILABLE = "unavailable"
SEMANTIC_NOT_REQUESTED = "not-requested"


@dataclass(frozen=True)
class SeedReport:
    """Which seed sources contributed to a retrieval, and why one did not.

    Attributes:
        mode: Seed strategy that ran (``structural`` or ``hybrid``).
        keyword_seeds: Seeds returned by the store's symbol search.
        semantic_seeds: Seeds returned by vector similarity.
        fused_seeds: Seeds handed to graph expansion.
        semantic_leg: ``used``, ``empty`` (no vectors matched or none were
            indexed), ``unavailable`` (the vector leg raised, e.g. a query
            encoder whose profile differs from the indexed one), or
            ``not-requested`` (structural mode).
        note: Human-readable reason when a leg did not contribute.
        keyword_leg: ``used``, ``empty`` (no keyword match) or
            ``unavailable`` (the store's symbol search raised).
    """

    mode: str
    keyword_seeds: int
    semantic_seeds: int
    fused_seeds: int
    semantic_leg: str
    note: Optional[str] = None
    keyword_leg: str = "used"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


VectorProviderFactory = Callable[[Any], Any]


def vector_provider_for(graph_store: Any) -> Any:
    """Vector store provider matching the graph store's embedding config.

    Returns ``None`` when no provider can be built. The provider validates the
    query encoder against the profile recorded with the indexed vectors, so a
    mismatched model raises at search time rather than ranking silently.
    """
    try:
        from victor.storage.vector_stores.registry import EmbeddingRegistry

        build = getattr(graph_store, "_build_embedding_config", None)
        if build is not None:
            config = build()
        else:
            from victor.storage.vector_stores.base import EmbeddingConfig

            config = EmbeddingConfig()
        return EmbeddingRegistry.create(config)
    except Exception as e:
        logger.debug(f"Vector provider unavailable for hybrid search: {e}")
        return None


async def structural_seed_scores(
    graph_store: Any,
    query: str,
    seed_count: int,
) -> Dict[str, float]:
    """Keyword (FTS5) symbol seeds with rank-decayed scores; no vector leg."""
    scores, _error = await _keyword_seed_scores(graph_store, query, seed_count)
    return scores


async def _keyword_nodes(
    graph_store: Any,
    query: str,
    limit: int,
) -> Tuple[List[Any], Optional[str]]:
    """The store's keyword symbol matches, plus the error if the search raised.

    A store's default search requires every query term to match, which suits
    symbol lookups but almost never matches a natural-language question
    (measured: no keyword seed for any of three research questions on an
    Apache Iceberg index). When that finds nothing for a multi-word query and
    the store supports it, the search is repeated matching any meaningful term.
    """
    try:
        nodes = list(await graph_store.search_symbols(query, limit=limit))
        if not nodes and len(query.split()) > 1 and _supports_any_term_match(graph_store):
            nodes = list(await graph_store.search_symbols(query, limit=limit, match="any"))
        return nodes, None
    except Exception as e:
        logger.warning(f"Keyword seed search failed: {e}")
        return [], f"keyword leg unavailable: {e}"


def _supports_any_term_match(graph_store: Any) -> bool:
    import inspect

    try:
        parameters = inspect.signature(graph_store.search_symbols).parameters
    except (TypeError, ValueError):
        return False
    return "match" in parameters


async def _keyword_seed_scores(
    graph_store: Any,
    query: str,
    limit: int,
) -> Tuple[Dict[str, float], Optional[str]]:
    """Rank-decayed keyword seeds, plus the error if the search raised."""
    nodes, error = await _keyword_nodes(graph_store, query, limit)
    return {n.node_id: 1.0 / (1 + rank) for rank, n in enumerate(nodes)}, error


def _keyword_leg(scores: Dict[str, float], error: Optional[str]) -> str:
    if error is not None:
        return SEMANTIC_UNAVAILABLE
    return SEMANTIC_USED if scores else SEMANTIC_EMPTY


def _join_notes(*notes: Optional[str]) -> Optional[str]:
    joined = "; ".join(n for n in notes if n)
    return joined or None


async def structural_seeds(
    graph_store: Any,
    query: str,
    seed_count: int,
) -> Tuple[Dict[str, float], SeedReport]:
    """Structural seeds plus their report."""
    scores, error = await _keyword_seed_scores(graph_store, query, seed_count)
    return scores, SeedReport(
        mode="structural",
        keyword_seeds=len(scores),
        semantic_seeds=0,
        fused_seeds=len(scores),
        semantic_leg=SEMANTIC_NOT_REQUESTED,
        note=error,
        keyword_leg=_keyword_leg(scores, error),
    )


async def hybrid_seeds(
    graph_store: Any,
    query: str,
    seed_count: int,
    *,
    vector_provider_factory: Optional[VectorProviderFactory] = None,
) -> Tuple[Dict[str, float], SeedReport]:
    """Fuse keyword seeds with vector-similarity seeds via RRF.

    Both legs are keyed by node_id (the hybrid engine's ``file_path`` field is
    used as an opaque identity key; it only needs to agree across legs). Fused
    scores are normalized to [0,1] so they feed the retriever's hop-distance
    decay proportionately. When the vector leg contributes nothing, the keyword
    seeds are returned and the report says why.

    Args:
        graph_store: Store to seed from.
        query: Natural-language query.
        seed_count: Number of seeds to return.
        vector_provider_factory: Builds the vector provider for the sqlite
            path (default :func:`vector_provider_for`).
    """
    factory = vector_provider_factory or vector_provider_for
    limit = max(seed_count * 2, seed_count)

    keyword_nodes, keyword_error = await _keyword_nodes(graph_store, query, limit)
    keyword_results: List[Dict[str, Any]] = [
        {"file_path": n.node_id, "content": n.name, "score": 1.0 / (1 + rank)}
        for rank, n in enumerate(keyword_nodes)
    ]
    keyword_leg = _keyword_leg({r["file_path"]: 1.0 for r in keyword_results}, keyword_error)

    semantic_results: List[Dict[str, Any]] = []
    note: Optional[str] = None
    failed = False
    try:
        if hasattr(graph_store, "semantic_search"):
            # Proxima-style store: co-indexed symbol vectors, query by vector.
            from victor.storage.embeddings.service import get_embedding_service

            service = get_embedding_service()
            if service is None:
                failed = True
                note = "semantic leg unavailable: no embedding service"
            else:
                query_vector = await service.embed_text(query)
                nodes = await graph_store.semantic_search(list(query_vector), top_k=limit)
                semantic_results = [
                    {"file_path": n.node_id, "content": n.name, "score": 1.0 / (1 + rank)}
                    for rank, n in enumerate(nodes)
                ]
        else:
            # Default sqlite path: node embeddings persisted to the vector
            # store by the indexing pipeline, keyed by node_id metadata.
            missing = _read_only_without_embeddings(graph_store)
            provider = None if missing else factory(graph_store)
            if missing:
                # Opening a vector store creates its directory; a read-only
                # index must not gain one.
                note = f"semantic leg empty: read-only index has no embeddings at {missing}"
            elif provider is None:
                failed = True
                note = "semantic leg unavailable: no vector store provider"
            else:
                hits = await provider.search_similar(query, limit=limit)
                semantic_results = [
                    {
                        "file_path": str(hit.metadata.get("node_id")),
                        "content": hit.content,
                        "score": hit.score,
                    }
                    for hit in hits
                    if hit.metadata.get("node_id")
                ]
    except Exception as e:
        failed = True
        note = f"semantic leg unavailable: {e}"
        logger.debug(f"Hybrid semantic leg failed: {e}")

    if not semantic_results:
        note = note or (
            "semantic leg empty (no node embeddings indexed — "
            "run `victor graph index --embeddings`); degraded to structural seeds"
        )
        scores = {r["file_path"]: r["score"] for r in keyword_results[:seed_count]}
        return scores, SeedReport(
            mode="hybrid",
            keyword_seeds=len(keyword_results),
            semantic_seeds=0,
            fused_seeds=len(scores),
            semantic_leg=SEMANTIC_UNAVAILABLE if failed else SEMANTIC_EMPTY,
            note=_join_notes(keyword_error, note),
            keyword_leg=keyword_leg,
        )

    from victor.core.search.hybrid import create_hybrid_search_engine

    engine = create_hybrid_search_engine()
    fused = engine.combine_results(semantic_results, keyword_results, max_results=seed_count)
    max_score = max((r.combined_score for r in fused), default=1.0) or 1.0
    scores = {r.file_path: r.combined_score / max_score for r in fused}
    return scores, SeedReport(
        mode="hybrid",
        keyword_seeds=len(keyword_results),
        semantic_seeds=len(semantic_results),
        fused_seeds=len(scores),
        semantic_leg=SEMANTIC_USED,
        note=keyword_error,
        keyword_leg=keyword_leg,
    )


def _read_only_without_embeddings(graph_store: Any) -> Optional[str]:
    """Embeddings path of a read-only store when that directory is absent."""
    if not getattr(graph_store, "read_only", False):
        return None
    build = getattr(graph_store, "_build_embedding_config", None)
    if build is None:
        return None
    try:
        persist = build().persist_directory
    except Exception:
        return None
    if persist and not Path(persist).expanduser().is_dir():
        return str(persist)
    return None
