"""Unit tests for RetrievalGateway (Item 7)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock


from victor.storage.retrieval.gateway import (
    RetrievalGateway,
    RetrievalMode,
    RetrievalRequest,
)


def _req(**kw) -> RetrievalRequest:
    defaults = {"query": "auth", "session_id": "s1", "limit": 5}
    defaults.update(kw)
    return RetrievalRequest(**defaults)


# ---------------------------------------------------------------------------
# Mode routing
# ---------------------------------------------------------------------------


class TestModeRouting:
    async def test_hybrid_mode_routes_to_unified_store(self):
        unified = AsyncMock()
        result_sym = MagicMock()
        result_sym.symbol.unified_id = "sym1"
        result_sym.score = 0.9
        unified.search = AsyncMock(return_value=[result_sym])

        gw = RetrievalGateway(unified_store=unified)
        items = await gw.search(_req(mode=RetrievalMode.HYBRID))

        unified.search.assert_awaited_once()
        assert items[0].message_id == "sym1"
        assert items[0].source == "hybrid"

    async def test_vector_mode_routes_to_embedding_store(self):
        vector = AsyncMock()
        r = MagicMock()
        r.message_id = "msg1"
        r.similarity = 0.75
        vector.search_similar = AsyncMock(return_value=[r])

        gw = RetrievalGateway(vector_store=vector)
        items = await gw.search(_req(mode=RetrievalMode.VECTOR))

        vector.search_similar.assert_awaited_once()
        assert items[0].message_id == "msg1"
        assert items[0].source == "vector"

    async def test_fts_mode_routes_to_conversation_store(self):
        fts = MagicMock()
        msg = MagicMock()
        msg.id = "fts_msg"
        fts.search_messages_fts = MagicMock(return_value=[msg])

        gw = RetrievalGateway(fts_store=fts)
        items = await gw.search(_req(mode=RetrievalMode.FTS))

        fts.search_messages_fts.assert_called_once()
        assert items[0].message_id == "fts_msg"
        assert items[0].source == "fts"


# ---------------------------------------------------------------------------
# Fallback behaviour
# ---------------------------------------------------------------------------


class TestFallback:
    async def test_hybrid_falls_back_to_vector_when_unified_store_none(self):
        vector = AsyncMock()
        r = MagicMock()
        r.message_id = "v1"
        r.similarity = 0.6
        vector.search_similar = AsyncMock(return_value=[r])

        gw = RetrievalGateway(vector_store=vector, unified_store=None)
        items = await gw.search(_req(mode=RetrievalMode.HYBRID))

        vector.search_similar.assert_awaited_once()
        assert items[0].source == "vector"

    async def test_backend_exception_returns_empty_list(self):
        vector = AsyncMock()
        vector.search_similar = AsyncMock(side_effect=RuntimeError("backend down"))

        gw = RetrievalGateway(vector_store=vector)
        items = await gw.search(_req(mode=RetrievalMode.VECTOR))

        assert items == []


# ---------------------------------------------------------------------------
# Score normalisation
# ---------------------------------------------------------------------------


class TestScoreNormalisation:
    async def test_scores_clamped_to_0_1(self):
        unified = AsyncMock()
        result = MagicMock()
        result.symbol.unified_id = "x"
        result.score = 9999.0  # way above 1.0
        unified.search = AsyncMock(return_value=[result])

        gw = RetrievalGateway(unified_store=unified)
        items = await gw.search(_req(mode=RetrievalMode.HYBRID))

        assert items[0].score <= 1.0

    async def test_negative_scores_clamped_to_0(self):
        unified = AsyncMock()
        result = MagicMock()
        result.symbol.unified_id = "y"
        result.score = -5.0
        unified.search = AsyncMock(return_value=[result])

        gw = RetrievalGateway(unified_store=unified)
        items = await gw.search(_req(mode=RetrievalMode.HYBRID))

        assert items[0].score >= 0.0


# ---------------------------------------------------------------------------
# Empty results
# ---------------------------------------------------------------------------


class TestEmptyResults:
    async def test_empty_backend_returns_empty_list(self):
        unified = AsyncMock()
        unified.search = AsyncMock(return_value=[])

        gw = RetrievalGateway(unified_store=unified)
        items = await gw.search(_req())

        assert items == []
