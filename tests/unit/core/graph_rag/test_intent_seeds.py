# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""FEP-0049 phase 4: intent seed leg + tool mode tests."""

from types import SimpleNamespace

import pytest

from victor.core.graph_rag.seeding import (
    hybrid_seeds,
    intent_provider_for,
    intent_seed_scores,
)


class _Hit:
    def __init__(self, score, node_id, chunk_id):
        self.score = score
        self.id = chunk_id
        self.content = "body text"
        self.metadata = {"symbol_id": node_id, "chunk_id": chunk_id}


class _SearchStore:
    """IntentIndexStore-shaped fake over canned hits."""

    def __init__(self, hits):
        self._hits = hits

    async def search(self, query, limit=20):
        return [
            {
                "score": h.score,
                "chunk_id": h.id,
                "content": "body text",
                "metadata": {"symbol_id": h.metadata["symbol_id"], "chunk_id": h.id},
            }
            for h in self._hits[:limit]
        ]

    async def close(self):
        pass


class TestIntentSeedScores:
    async def test_max_aggregated_per_symbol(self):
        hits = [
            _Hit(0.9, "node_b", "b#body#0"),
            _Hit(0.8, "node_a", "a#body#0"),
            _Hit(0.7, "node_b", "b#body#1"),
        ]
        scores, note = await intent_seed_scores(
            None, "rate limiter", 10, intent_store_factory=lambda _: _SearchStore(hits)
        )
        assert note is None
        assert set(scores) == {"node_a", "node_b"}
        # rank normalization: node_b's best rank is 0 -> 1.0
        assert scores["node_b"] == 1.0
        assert scores["node_a"] == pytest.approx(1.0 / 2)

    async def test_no_provider_degrades_with_note(self):
        scores, note = await intent_seed_scores(None, "q", 10, intent_store_factory=lambda _: None)
        assert scores == {}
        assert "no intent index provider" in note

    async def test_empty_index_degrades_with_hint(self):
        scores, note = await intent_seed_scores(
            None, "q", 10, intent_store_factory=lambda _: _SearchStore([])
        )
        assert scores == {}
        assert "graph index --intent" in note

    async def test_hits_without_symbol_id_skipped(self):
        hits = [_Hit(0.9, "", "x#body#0")]
        scores, note = await intent_seed_scores(
            None, "q", 10, intent_store_factory=lambda _: _SearchStore(hits)
        )
        assert scores == {}
        assert "no symbol ids" in note


class TestHybridIntentLeg:
    async def test_intent_leg_fused_when_factory_injected(self):
        hits = [_Hit(0.9, "node_intent", "i#body#0")]
        keyword_nodes = [SimpleNamespace(node_id="node_kw", name="kw")]
        keyword_error = None
        keyword_match = "all"

        async def fake_keyword_nodes(store, query, limit, allow_any=False):
            return keyword_nodes, keyword_error, keyword_match

        import victor.core.graph_rag.seeding as seeding

        original = seeding._keyword_nodes
        seeding._keyword_nodes = fake_keyword_nodes
        try:
            # No semantic store methods and no provider factory -> semantic
            # leg empty; intent leg injected -> fusion still runs.
            scores, report = await hybrid_seeds(
                SimpleNamespace(),
                "q",
                5,
                vector_provider_factory=lambda _: None,
                intent_store_factory=lambda _: _SearchStore(hits),
            )
        finally:
            seeding._keyword_nodes = original
        assert "node_intent" in scores
        assert "node_kw" in scores
        assert report.fused_seeds == len(scores)

    async def test_flag_off_byte_identical(self):
        """intent_store_factory=None (flag off) -> the intent leg is fully
        absent: same behavior as before FEP-0049."""
        hits = [_Hit(0.9, "node_intent", "i#body#0")]
        keyword_nodes = [SimpleNamespace(node_id="node_kw", name="kw")]

        async def fake_keyword_nodes(store, query, limit, allow_any=False):
            return keyword_nodes, None, "all"

        import victor.core.graph_rag.seeding as seeding

        original = seeding._keyword_nodes
        seeding._keyword_nodes = fake_keyword_nodes
        try:
            scores, report = await hybrid_seeds(
                SimpleNamespace(),
                "q",
                5,
                vector_provider_factory=lambda _: None,
                intent_store_factory=None,
            )
        finally:
            seeding._keyword_nodes = original
        assert "node_intent" not in scores
        assert report.semantic_seeds == 0


class TestIntentProviderFor:
    def test_store_without_embedding_config_returns_none(self):
        assert intent_provider_for(SimpleNamespace()) is None
