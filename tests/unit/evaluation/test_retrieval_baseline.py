"""Unit tests for retrieval baseline harness (Item 4)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from victor.evaluation.retrieval_baseline import (
    RetrievalGoldenQuery,
    run_retrieval_baseline,
)


def _backend(*, returns: list[list[str]]) -> MagicMock:
    """Build a mock backend that returns successive result lists per call."""
    backend = MagicMock()
    backend.name = "mock"
    call_iter = iter(returns)

    async def _search(query, *, limit, session_id):
        try:
            return next(call_iter)
        except StopIteration:
            return []

    backend.search = _search
    return backend


# ---------------------------------------------------------------------------
# Precision / recall / F1 corner cases
# ---------------------------------------------------------------------------


class TestMetricComputation:
    async def test_perfect_backend_precision_1_recall_1(self):
        backend = _backend(returns=[["m1", "m2"]])
        golden = [RetrievalGoldenQuery(query="x", relevant_message_ids=["m1", "m2"])]
        m = await run_retrieval_baseline(backend, golden, k=2, session_id="s")
        assert m.precision_at_k == pytest.approx(1.0)
        assert m.recall_at_k == pytest.approx(1.0)
        assert m.f1_at_k == pytest.approx(1.0)

    async def test_partial_overlap_precision_recall_f1(self):
        backend = _backend(returns=[["m1", "m_noise"]])
        golden = [RetrievalGoldenQuery(query="x", relevant_message_ids=["m1", "m2"])]
        m = await run_retrieval_baseline(backend, golden, k=2, session_id="s")
        # 1 hit in top-2 → P=0.5, R=0.5, F1=0.5
        assert m.precision_at_k == pytest.approx(0.5)
        assert m.recall_at_k == pytest.approx(0.5)
        assert m.f1_at_k == pytest.approx(0.5)

    async def test_empty_results_precision_0(self):
        backend = _backend(returns=[[]])
        golden = [RetrievalGoldenQuery(query="x", relevant_message_ids=["m1"])]
        m = await run_retrieval_baseline(backend, golden, k=5, session_id="s")
        assert m.precision_at_k == pytest.approx(0.0)
        assert m.recall_at_k == pytest.approx(0.0)
        assert m.f1_at_k == pytest.approx(0.0)

    async def test_metrics_aggregate_over_multiple_queries(self):
        backend = _backend(returns=[["m1"], ["m_noise"]])
        golden = [
            RetrievalGoldenQuery(query="q1", relevant_message_ids=["m1"]),
            RetrievalGoldenQuery(query="q2", relevant_message_ids=["m2"]),
        ]
        m = await run_retrieval_baseline(backend, golden, k=1, session_id="s")
        assert m.n_queries == 2
        # First query: P=R=F1=1, second: P=R=F1=0 → averages = 0.5
        assert m.precision_at_k == pytest.approx(0.5)

    async def test_no_queries_returns_zero_metrics(self):
        backend = _backend(returns=[])
        m = await run_retrieval_baseline(backend, [], k=5, session_id="s")
        assert m.n_queries == 0
        assert m.precision_at_k == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestRetrievalBaselineErrorHandling:
    async def test_backend_exception_counted_as_error(self):
        backend = MagicMock()
        backend.name = "buggy"

        async def _raise(*a, **kw):
            raise RuntimeError("backend down")

        backend.search = _raise
        golden = [RetrievalGoldenQuery(query="x", relevant_message_ids=["m1"])]
        m = await run_retrieval_baseline(backend, golden, k=5, session_id="s")
        assert m.errors == 1
        assert m.precision_at_k == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Latency fields
# ---------------------------------------------------------------------------


class TestLatencyFields:
    async def test_latency_fields_are_non_negative(self):
        backend = _backend(returns=[["m1"]])
        golden = [RetrievalGoldenQuery(query="x", relevant_message_ids=["m1"])]
        m = await run_retrieval_baseline(backend, golden, k=1, session_id="s")
        assert m.latency_p50_ms >= 0.0
        assert m.latency_p95_ms >= 0.0
