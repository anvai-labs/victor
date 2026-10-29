"""Hybrid Retrieval Gateway — unified facade over FTS5, vector ANN, and hybrid backends (Item 7).

Registered as a SINGLETON in bootstrap_new_services so callers get one instance
that is DI-resolved at construction time. Stores no per-request state.

Example::

    gateway = get_container().get(RetrievalGateway)
    results = await gateway.search(RetrievalRequest(query="auth", session_id="s1"))
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Public types
# =============================================================================


class RetrievalMode(str, Enum):
    FTS = "fts"
    VECTOR = "vector"
    HYBRID = "hybrid"


@dataclass
class RetrievalRequest:
    """Parameters for a single retrieval call."""

    query: str
    session_id: str
    limit: int = 10
    mode: RetrievalMode = RetrievalMode.HYBRID
    min_similarity: float = 0.3
    similarity_weight: float = 0.7


@dataclass
class RetrievedItem:
    """One result from the gateway; score is normalised to [0, 1]."""

    message_id: str
    score: float
    source: str
    snippet: Optional[str] = None


# =============================================================================
# Gateway
# =============================================================================


class RetrievalGateway:
    """Singleton facade that routes retrieval requests to the appropriate backend.

    Mode routing:
      HYBRID → SqliteLanceDBStore.search(SearchMode.HYBRID)   (preferred)
      VECTOR → ConversationEmbeddingStore.search_similar()
      FTS    → ConversationStore.search_messages_fts()

    Fallback: if unified_store is None, HYBRID falls back to VECTOR.
    Stores no per-request mutable state — safe to use as a singleton.
    """

    def __init__(
        self,
        *,
        fts_store: Optional[Any] = None,
        vector_store: Optional[Any] = None,
        unified_store: Optional[Any] = None,
        default_mode: RetrievalMode = RetrievalMode.HYBRID,
    ) -> None:
        self._fts_store = fts_store
        self._vector_store = vector_store
        self._unified_store = unified_store
        self._default_mode = default_mode

    async def search(self, request: RetrievalRequest) -> List[RetrievedItem]:
        """Route the request to the appropriate backend and return normalised results."""
        mode = request.mode
        if mode == RetrievalMode.HYBRID and self._unified_store is None:
            logger.debug("RetrievalGateway: unified_store unavailable, falling back to VECTOR")
            mode = RetrievalMode.VECTOR

        try:
            if mode == RetrievalMode.HYBRID:
                return await self._search_hybrid(request)
            if mode == RetrievalMode.VECTOR:
                return await self._search_vector(request)
            return await self._search_fts(request)
        except Exception as exc:
            logger.warning("RetrievalGateway search failed (mode=%s): %s", mode, exc)
            return []

    # ------------------------------------------------------------------
    # Backend-specific private methods
    # ------------------------------------------------------------------

    async def _search_hybrid(self, req: RetrievalRequest) -> List[RetrievedItem]:
        from victor.storage.unified.protocol import SearchMode, SearchParams

        params = SearchParams(
            query=req.query,
            limit=req.limit,
            mode=SearchMode.HYBRID,
            semantic_weight=req.similarity_weight,
        )
        raw = await self._unified_store.search(params)
        return [
            RetrievedItem(
                message_id=r.symbol.unified_id,
                score=self._clamp(r.score),
                source="hybrid",
            )
            for r in raw
        ]

    async def _search_vector(self, req: RetrievalRequest) -> List[RetrievedItem]:
        raw = await self._vector_store.search_similar(
            query=req.query,
            session_id=req.session_id,
            limit=req.limit,
            min_similarity=req.min_similarity,
        )
        return [
            RetrievedItem(
                message_id=r.message_id,
                score=self._clamp(r.similarity),
                source="vector",
            )
            for r in raw
        ]

    async def _search_fts(self, req: RetrievalRequest) -> List[RetrievedItem]:
        raw = self._fts_store.search_messages_fts(
            session_id=req.session_id,
            query=req.query,
            limit=req.limit,
        )

        # FTS5 BM25 scores are negative; normalise by dividing by the best (most negative) score.
        def _safe_rank(m: Any) -> float:
            v = getattr(m, "_fts_rank", None)
            try:
                return float(v) if v is not None else 0.5
            except (TypeError, ValueError):
                return 0.5

        ranks = [_safe_rank(m) for m in raw]
        max_abs = max((abs(r) for r in ranks), default=1.0) or 1.0
        return [
            RetrievedItem(
                message_id=m.id,
                score=self._clamp(abs(ranks[i]) / max_abs),
                source="fts",
                snippet=getattr(m, "content", None),
            )
            for i, m in enumerate(raw)
        ]

    @staticmethod
    def _clamp(score: float) -> float:
        return max(0.0, min(1.0, float(score)))
