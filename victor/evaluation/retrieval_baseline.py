"""Retrieval baseline harness for FTS5 / vector ANN / hybrid backends (Item 4).

Measures precision@k, recall@k, F1@k, context token proxy, and latency
(p50/p95) for any backend conforming to RetrievalBackendProtocol.

Example::

    store = ConversationStore(db_path=":memory:")
    adapter = FTSBackendAdapter(store)
    golden = [RetrievalGoldenQuery(query="auth", relevant_message_ids=["m1"])]
    metrics = await run_retrieval_baseline(adapter, golden, k=5, session_id="s1")
    print(metrics.f1_at_k)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

# =============================================================================
# Golden dataset
# =============================================================================


@dataclass
class RetrievalGoldenQuery:
    """One labelled query with its ground-truth relevant message IDs."""

    query: str
    relevant_message_ids: list[str]


# =============================================================================
# Metrics output
# =============================================================================


@dataclass
class RetrievalBaselineMetrics:
    """Aggregate retrieval quality and latency metrics for one backend run."""

    backend: str
    k: int
    precision_at_k: float
    recall_at_k: float
    f1_at_k: float
    avg_context_tokens: float
    latency_p50_ms: float
    latency_p95_ms: float
    n_queries: int
    errors: int = 0

    def summary(self) -> str:
        return (
            f"[{self.backend}] k={self.k} n={self.n_queries} "
            f"P@k={self.precision_at_k:.3f} R@k={self.recall_at_k:.3f} "
            f"F1={self.f1_at_k:.3f} p50={self.latency_p50_ms:.1f}ms"
        )


# =============================================================================
# Backend protocol
# =============================================================================


@runtime_checkable
class RetrievalBackendProtocol(Protocol):
    """Minimal async search interface that adapters must satisfy."""

    async def search(self, query: str, *, limit: int, session_id: str) -> list[str]:
        """Return up to *limit* message IDs ranked by relevance."""
        ...


# =============================================================================
# Adapters (read-only wrappers — backend method signatures unchanged)
# =============================================================================


class FTSBackendAdapter:
    """Wraps ConversationStore.search_messages_fts()."""

    name = "fts5"

    def __init__(self, store: object) -> None:
        self._store = store

    async def search(self, query: str, *, limit: int, session_id: str) -> list[str]:
        results = self._store.search_messages_fts(  # type: ignore[attr-defined]
            session_id=session_id, query=query, limit=limit
        )
        return [m.id for m in results]


class VectorBackendAdapter:
    """Wraps ConversationEmbeddingStore.search_similar()."""

    name = "vector_ann"

    def __init__(self, store: object) -> None:
        self._store = store

    async def search(self, query: str, *, limit: int, session_id: str) -> list[str]:
        results = await self._store.search_similar(  # type: ignore[attr-defined]
            query=query, session_id=session_id, limit=limit
        )
        return [r.message_id for r in results]


class HybridBackendAdapter:
    """Wraps SqliteLanceDBStore.search(SearchMode.HYBRID)."""

    name = "hybrid"

    def __init__(self, store: object) -> None:
        self._store = store

    async def search(self, query: str, *, limit: int, session_id: str) -> list[str]:
        from victor.storage.unified.protocol import SearchMode, SearchParams

        params = SearchParams(query=query, limit=limit, mode=SearchMode.HYBRID)
        results = await self._store.search(params)  # type: ignore[attr-defined]
        return [r.symbol.unified_id for r in results]


# =============================================================================
# Measurement loop
# =============================================================================


def _precision_recall_f1(
    retrieved: list[str], relevant: set[str], k: int
) -> tuple[float, float, float]:
    top_k = retrieved[:k]
    hits = sum(1 for rid in top_k if rid in relevant)
    precision = hits / k if k else 0.0
    recall = hits / len(relevant) if relevant else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def _token_proxy(message_ids: list[str]) -> float:
    """Rough proxy: assume 150 tokens per retrieved message."""
    return len(message_ids) * 150.0


async def run_retrieval_baseline(
    backend: RetrievalBackendProtocol,
    golden_queries: list[RetrievalGoldenQuery],
    *,
    k: int = 5,
    session_id: str,
) -> RetrievalBaselineMetrics:
    """Run the full retrieval evaluation loop and return aggregate metrics.

    Args:
        backend: Any object implementing RetrievalBackendProtocol.
        golden_queries: Labelled queries with ground-truth message IDs.
        k: Cut-off rank for precision/recall/F1 computation.
        session_id: Session context forwarded to the backend.

    Returns:
        RetrievalBaselineMetrics with per-query aggregates.
    """
    backend_name = getattr(backend, "name", type(backend).__name__)

    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    token_proxies: list[float] = []
    latencies_ms: list[float] = []
    errors = 0

    for gq in golden_queries:
        relevant = set(gq.relevant_message_ids)
        t0 = time.perf_counter()
        try:
            retrieved = await backend.search(gq.query, limit=k, session_id=session_id)
        except Exception:
            errors += 1
            retrieved = []
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        p, r, f = _precision_recall_f1(retrieved, relevant, k)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)
        token_proxies.append(_token_proxy(retrieved))
        latencies_ms.append(elapsed_ms)

    n = len(golden_queries)
    if not n:
        return RetrievalBaselineMetrics(
            backend=backend_name,
            k=k,
            precision_at_k=0.0,
            recall_at_k=0.0,
            f1_at_k=0.0,
            avg_context_tokens=0.0,
            latency_p50_ms=0.0,
            latency_p95_ms=0.0,
            n_queries=0,
            errors=errors,
        )

    sorted_lat = sorted(latencies_ms)
    p50_idx = int(0.5 * n)
    p95_idx = min(int(0.95 * n), n - 1)

    return RetrievalBaselineMetrics(
        backend=backend_name,
        k=k,
        precision_at_k=sum(precisions) / n,
        recall_at_k=sum(recalls) / n,
        f1_at_k=sum(f1s) / n,
        avg_context_tokens=sum(token_proxies) / n,
        latency_p50_ms=sorted_lat[p50_idx],
        latency_p95_ms=sorted_lat[p95_idx],
        n_queries=n,
        errors=errors,
    )
