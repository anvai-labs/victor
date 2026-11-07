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
        note: Human-readable reason when the vector leg did not contribute.
    """

    mode: str
    keyword_seeds: int
    semantic_seeds: int
    fused_seeds: int
    semantic_leg: str
    note: Optional[str] = None

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
    try:
        nodes = await graph_store.search_symbols(query, limit=seed_count)
    except Exception as e:
        logger.warning(f"Structural seed search failed: {e}")
        return {}
    return {n.node_id: 1.0 / (1 + rank) for rank, n in enumerate(nodes)}


async def structural_seeds(
    graph_store: Any,
    query: str,
    seed_count: int,
) -> Tuple[Dict[str, float], SeedReport]:
    """Structural seeds plus their report."""
    scores = await structural_seed_scores(graph_store, query, seed_count)
    return scores, SeedReport(
        mode="structural",
        keyword_seeds=len(scores),
        semantic_seeds=0,
        fused_seeds=len(scores),
        semantic_leg=SEMANTIC_NOT_REQUESTED,
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

    keyword_results: List[Dict[str, Any]] = []
    try:
        nodes = await graph_store.search_symbols(query, limit=limit)
        keyword_results = [
            {"file_path": n.node_id, "content": n.name, "score": 1.0 / (1 + rank)}
            for rank, n in enumerate(nodes)
        ]
    except Exception as e:
        logger.warning(f"Keyword seed search failed: {e}")

    semantic_results: List[Dict[str, Any]] = []
    note: Optional[str] = None
    failed = False
    try:
        if hasattr(graph_store, "semantic_search"):
            # Proxima-style store: co-indexed symbol vectors, query by vector.
            from victor.storage.embeddings.service import get_embedding_service

            service = get_embedding_service()
            if service is not None:
                query_vector = await service.embed_text(query)
                nodes = await graph_store.semantic_search(list(query_vector), top_k=limit)
                semantic_results = [
                    {"file_path": n.node_id, "content": n.name, "score": 1.0 / (1 + rank)}
                    for rank, n in enumerate(nodes)
                ]
        else:
            # Default sqlite path: node embeddings persisted to the vector
            # store by the indexing pipeline, keyed by node_id metadata.
            provider = factory(graph_store)
            if provider is None:
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
            note=note,
        )

    from victor.framework.search.hybrid import create_hybrid_search_engine

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
    )
