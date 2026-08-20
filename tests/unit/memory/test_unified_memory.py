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

"""Tests for unified memory coordinator and adapters.

Tests cover:
- MemoryResult and MemoryQuery dataclasses
- Ranking strategies (Relevance, Recency, Hybrid)
- UnifiedMemoryCoordinator federated search
- Memory adapters for existing systems
- Protocol conformance
"""

import asyncio
import pytest
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

from victor.storage.memory.unified import (
    MemoryType,
    MemoryResult,
    MemoryQuery,
    MemoryEvolutionTrace,
    MemoryTransferHint,
    ProactiveMemoryHint,
    MemoryProviderProtocol,
    RankingStrategyProtocol,
    RelevanceRankingStrategy,
    RecencyRankingStrategy,
    HybridRankingStrategy,
    UnifiedMemoryCoordinator,
    create_memory_coordinator,
    get_memory_coordinator,
    reset_memory_coordinator,
)

# =============================================================================
# Test Fixtures
# =============================================================================


class MockMemoryProvider:
    """Mock memory provider for testing."""

    def __init__(self, memory_type: MemoryType, results: List[MemoryResult] = None):
        self._memory_type = memory_type
        self._results = results or []
        self._stored: Dict[str, Any] = {}
        self._available = True

    @property
    def memory_type(self) -> MemoryType:
        return self._memory_type

    async def search(self, query: MemoryQuery) -> List[MemoryResult]:
        # Filter results by query
        filtered = []
        for r in self._results:
            content_str = str(r.content).lower()
            if query.query.lower() in content_str:
                filtered.append(r)
        return filtered[: query.limit]

    async def store(
        self,
        key: str,
        value: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._stored[key] = {"value": value, "metadata": metadata}

    async def get(self, key: str) -> Optional[MemoryResult]:
        if key in self._stored:
            return MemoryResult(
                source=self._memory_type,
                content=self._stored[key]["value"],
                relevance=1.0,
                id=key,
            )
        return None

    def is_available(self) -> bool:
        return self._available


@pytest.fixture
def reset_coordinator():
    """Reset coordinator before and after each test."""
    reset_memory_coordinator()
    yield
    reset_memory_coordinator()


@pytest.fixture
def sample_results() -> List[MemoryResult]:
    """Create sample memory results for testing."""
    now = time.time()
    return [
        MemoryResult(
            source=MemoryType.ENTITY,
            content={"name": "UserAuth", "type": "class"},
            relevance=0.9,
            id="ent_1",
            timestamp=now - 3600,  # 1 hour ago
        ),
        MemoryResult(
            source=MemoryType.ENTITY,
            content={"name": "AuthMiddleware", "type": "class"},
            relevance=0.7,
            id="ent_2",
            timestamp=now - 7200,  # 2 hours ago
        ),
        MemoryResult(
            source=MemoryType.CONVERSATION,
            content={"role": "user", "content": "Fix authentication bug"},
            relevance=0.8,
            id="msg_1",
            timestamp=now - 1800,  # 30 min ago
        ),
        MemoryResult(
            source=MemoryType.CONVERSATION,
            content={"role": "assistant", "content": "I found the auth issue"},
            relevance=0.6,
            id="msg_2",
            timestamp=now - 900,  # 15 min ago
        ),
    ]


# =============================================================================
# Test MemoryResult
# =============================================================================


class TestMemoryResult:
    """Tests for MemoryResult dataclass."""

    def test_create_memory_result(self):
        """Test creating a memory result."""
        result = MemoryResult(
            source=MemoryType.ENTITY,
            content={"name": "Test"},
            relevance=0.85,
        )
        assert result.source == MemoryType.ENTITY
        assert result.content == {"name": "Test"}
        assert result.relevance == 0.85
        assert result.id  # Auto-generated

    def test_memory_result_with_explicit_id(self):
        """Test memory result with explicit ID."""
        result = MemoryResult(
            source=MemoryType.CONVERSATION,
            content="test message",
            relevance=0.5,
            id="msg_123",
        )
        assert result.id == "msg_123"

    def test_memory_result_with_metadata(self):
        """Test memory result with metadata."""
        result = MemoryResult(
            source=MemoryType.GRAPH,
            content={"node": "A", "edge": "B"},
            relevance=0.7,
            metadata={"source_file": "test.py"},
            timestamp=time.time(),
        )
        assert result.metadata["source_file"] == "test.py"
        assert result.timestamp is not None

    def test_memory_result_repr(self):
        """Test memory result string representation."""
        result = MemoryResult(
            source=MemoryType.ENTITY,
            content="test",
            relevance=0.95,
            id="test_id_123456",
        )
        repr_str = repr(result)
        assert "ENTITY" in repr_str
        assert "0.95" in repr_str


# =============================================================================
# Test MemoryQuery
# =============================================================================


class TestMemoryQuery:
    """Tests for MemoryQuery dataclass."""

    def test_create_simple_query(self):
        """Test creating a simple query."""
        query = MemoryQuery(query="authentication")
        assert query.query == "authentication"
        assert query.limit == 20  # default
        assert query.memory_types is None
        assert query.min_relevance == 0.0

    def test_create_filtered_query(self):
        """Test creating a query with filters."""
        query = MemoryQuery(
            query="login",
            limit=10,
            memory_types=[MemoryType.ENTITY, MemoryType.CONVERSATION],
            filters={"entity_types": ["class", "function"]},
            min_relevance=0.5,
            session_id="session_123",
        )
        assert query.limit == 10
        assert len(query.memory_types) == 2
        assert query.filters["entity_types"] == ["class", "function"]
        assert query.min_relevance == 0.5
        assert query.session_id == "session_123"


# =============================================================================
# Test Memory Evolution
# =============================================================================


class TestMemoryEvolution:
    """Tests for shadow memory evolution and transfer hints."""

    @pytest.mark.asyncio
    async def test_records_successful_trace_when_enabled(self):
        coordinator = create_memory_coordinator(enable_memory_evolution=True)
        results = [
            MemoryResult(
                source=MemoryType.CONVERSATION,
                content="Auth findings",
                relevance=0.85,
                id="msg_auth_1",
            )
        ]

        stored = await coordinator.record_outcome(
            query="authentication failure",
            results=results,
            success=True,
            session_id="session-a",
            metadata={"gaps": ["benchmarks"]},
        )

        assert stored is True
        assert len(coordinator._evolution_traces) == 1
        trace = coordinator._evolution_traces[0]
        assert isinstance(trace, MemoryEvolutionTrace)
        assert trace.query == "authentication failure"
        assert trace.result_ids == ["msg_auth_1"]
        assert trace.source_types == ["CONVERSATION"]
        assert trace.metadata["gaps"] == ["benchmarks"]

    @pytest.mark.asyncio
    async def test_suggest_transfer_ignores_stale_trace(self):
        coordinator = create_memory_coordinator(
            enable_memory_evolution=True,
            trace_ttl_hours=0.01,
        )
        results = [
            MemoryResult(
                source=MemoryType.ENTITY,
                content="AuthService",
                relevance=0.8,
                id="ent_auth_1",
            )
        ]

        await coordinator.record_outcome(
            query="authentication failure",
            results=results,
            success=True,
            session_id="session-a",
        )
        coordinator._evolution_traces[0].timestamp -= 3600

        hints = coordinator.suggest_transfer("authentication retries", session_id="session-b")

        assert hints == []

    @pytest.mark.asyncio
    async def test_suggest_transfer_blocks_cross_project_without_policy(self):
        coordinator = create_memory_coordinator(enable_memory_evolution=True)
        results = [
            MemoryResult(
                source=MemoryType.CONVERSATION,
                content="Shared auth remediation notes",
                relevance=0.83,
                id="msg_auth_notes",
            )
        ]

        await coordinator.record_outcome(
            query="authentication failure remediation",
            results=results,
            success=True,
            session_id="session-a",
            metadata={
                "project_path": "/repo-alpha",
                "vertical_name": "coding",
            },
        )

        hints = coordinator.suggest_transfer(
            "authentication remediation checklist",
            session_id="session-b",
            project_path="/repo-beta",
            vertical_name="coding",
        )

        assert hints == []

    @pytest.mark.asyncio
    async def test_suggest_transfer_allows_shared_transfer_group_across_verticals(self):
        coordinator = create_memory_coordinator(enable_memory_evolution=True)
        results = [
            MemoryResult(
                source=MemoryType.CONVERSATION,
                content="Shared auth remediation notes",
                relevance=0.83,
                id="msg_auth_notes",
            )
        ]

        await coordinator.record_outcome(
            query="authentication failure remediation",
            results=results,
            success=True,
            session_id="session-a",
            metadata={
                "project_path": "/repo-alpha",
                "vertical_name": "coding",
                "transfer_group": "victor-core",
            },
        )

        hints = coordinator.suggest_transfer(
            "authentication remediation checklist",
            session_id="session-b",
            project_path="/repo-beta",
            vertical_name="research",
            transfer_group="victor-core",
            allow_cross_project=True,
        )

        assert len(hints) == 1
        assert hints[0].cross_session is True
        assert hints[0].metadata["transfer_scope"] == "cross_project_transfer_group"
        assert hints[0].metadata["source_project_path"] == "/repo-alpha"
        assert hints[0].metadata["source_vertical_name"] == "coding"

    @pytest.mark.asyncio
    async def test_search_all_boosts_results_with_transfer_hints(self):
        coordinator = create_memory_coordinator(
            enable_memory_evolution=True,
            transfer_boost=0.12,
        )
        preferred_result = MemoryResult(
            source=MemoryType.CONVERSATION,
            content="authentication login workflow",
            relevance=0.72,
            id="msg_preferred",
            timestamp=time.time(),
        )
        competitor_result = MemoryResult(
            source=MemoryType.ENTITY,
            content="authentication login service",
            relevance=0.75,
            id="ent_competitor",
            timestamp=preferred_result.timestamp,
        )

        conversation_provider = MockMemoryProvider(
            MemoryType.CONVERSATION,
            [preferred_result],
        )
        entity_provider = MockMemoryProvider(
            MemoryType.ENTITY,
            [competitor_result],
        )
        coordinator.register_provider(conversation_provider)
        coordinator.register_provider(entity_provider)

        await coordinator.record_outcome(
            query="authentication bug",
            results=[preferred_result],
            success=True,
            session_id="session-a",
        )

        ranked = await coordinator.search_all(
            "authentication login",
            session_id="session-b",
        )

        assert ranked[0].id == "msg_preferred"
        assert ranked[0].metadata["memory_transfer_boost"] > 0
        assert coordinator.last_transfer_hints
        assert isinstance(coordinator.last_transfer_hints[0], MemoryTransferHint)
        assert coordinator.last_transfer_hints[0].cross_session is True

    @pytest.mark.asyncio
    async def test_search_all_skips_cross_project_transfer_without_policy(self):
        coordinator = create_memory_coordinator(
            enable_memory_evolution=True,
            transfer_boost=0.12,
        )
        preferred_result = MemoryResult(
            source=MemoryType.CONVERSATION,
            content="authentication login workflow",
            relevance=0.72,
            id="msg_preferred",
            timestamp=time.time(),
        )
        competitor_result = MemoryResult(
            source=MemoryType.ENTITY,
            content="authentication login service",
            relevance=0.75,
            id="ent_competitor",
            timestamp=preferred_result.timestamp,
        )

        conversation_provider = MockMemoryProvider(
            MemoryType.CONVERSATION,
            [preferred_result],
        )
        entity_provider = MockMemoryProvider(
            MemoryType.ENTITY,
            [competitor_result],
        )
        coordinator.register_provider(conversation_provider)
        coordinator.register_provider(entity_provider)

        await coordinator.record_outcome(
            query="authentication bug",
            results=[preferred_result],
            success=True,
            session_id="session-a",
            metadata={
                "project_path": "/repo-alpha",
                "vertical_name": "coding",
            },
        )

        ranked = await coordinator.search_all(
            "authentication login",
            session_id="session-b",
            filters={
                "project_path": "/repo-beta",
                "vertical_name": "coding",
            },
        )

        assert ranked[0].id == "ent_competitor"
        assert coordinator.last_transfer_hints == []

    @pytest.mark.asyncio
    async def test_suggest_proactive_hints_uses_trace_metadata(self):
        coordinator = create_memory_coordinator(enable_memory_evolution=True)
        results = [
            MemoryResult(
                source=MemoryType.CONVERSATION,
                content="Shared auth remediation notes",
                relevance=0.83,
                id="msg_auth_notes",
            )
        ]

        await coordinator.record_outcome(
            query="authentication failure remediation",
            results=results,
            success=True,
            session_id="session-a",
            metadata={
                "project_path": "/repo-alpha",
                "gaps": ["benchmarks"],
                "next_steps": ["rerun auth middleware diagnostics"],
            },
        )

        hints = coordinator.suggest_proactive_hints(
            "authentication remediation checklist",
            session_id="session-b",
            project_path="/repo-alpha",
        )

        assert len(hints) == 1
        assert isinstance(hints[0], ProactiveMemoryHint)
        assert "benchmarks" in hints[0].hint
        assert "rerun auth middleware diagnostics" in hints[0].hint
        assert hints[0].matched_query == "authentication failure remediation"

    @pytest.mark.asyncio
    async def test_search_all_tracks_last_proactive_hints(self):
        coordinator = create_memory_coordinator(
            enable_memory_evolution=True,
            transfer_boost=0.12,
        )
        preferred_result = MemoryResult(
            source=MemoryType.CONVERSATION,
            content="authentication login workflow",
            relevance=0.72,
            id="msg_preferred",
            timestamp=time.time(),
        )
        conversation_provider = MockMemoryProvider(
            MemoryType.CONVERSATION,
            [preferred_result],
        )
        coordinator.register_provider(conversation_provider)

        await coordinator.record_outcome(
            query="authentication bug",
            results=[preferred_result],
            success=True,
            session_id="session-a",
            metadata={
                "next_steps": ["inspect auth retry counters"],
            },
        )

        await coordinator.search_all(
            "authentication login",
            session_id="session-b",
        )

        assert coordinator.last_proactive_hints
        assert isinstance(coordinator.last_proactive_hints[0], ProactiveMemoryHint)
        assert "inspect auth retry counters" in coordinator.last_proactive_hints[0].hint


# =============================================================================
# Test Ranking Strategies
# =============================================================================


class TestRelevanceRankingStrategy:
    """Tests for RelevanceRankingStrategy."""

    def test_rank_by_relevance(self, sample_results):
        """Test ranking purely by relevance."""
        strategy = RelevanceRankingStrategy()
        query = MemoryQuery(query="auth")

        ranked = strategy.rank(sample_results, query, limit=10)

        assert len(ranked) == 4
        assert ranked[0].relevance == 0.9  # Highest relevance first
        assert ranked[1].relevance == 0.8
        assert ranked[2].relevance == 0.7
        assert ranked[3].relevance == 0.6

    def test_rank_with_limit(self, sample_results):
        """Test ranking with limit."""
        strategy = RelevanceRankingStrategy()
        query = MemoryQuery(query="auth")

        ranked = strategy.rank(sample_results, query, limit=2)

        assert len(ranked) == 2
        assert ranked[0].relevance == 0.9

    def test_rank_with_min_relevance(self, sample_results):
        """Test ranking with minimum relevance filter."""
        strategy = RelevanceRankingStrategy()
        query = MemoryQuery(query="auth", min_relevance=0.75)

        ranked = strategy.rank(sample_results, query, limit=10)

        assert len(ranked) == 2
        assert all(r.relevance >= 0.75 for r in ranked)

    def test_rank_deduplicates(self):
        """Test that ranking deduplicates by ID."""
        strategy = RelevanceRankingStrategy()
        results = [
            MemoryResult(source=MemoryType.ENTITY, content="a", relevance=0.9, id="same_id"),
            MemoryResult(source=MemoryType.ENTITY, content="b", relevance=0.8, id="same_id"),
            MemoryResult(source=MemoryType.ENTITY, content="c", relevance=0.7, id="different_id"),
        ]
        query = MemoryQuery(query="test")

        ranked = strategy.rank(results, query, limit=10)

        assert len(ranked) == 2
        assert ranked[0].id == "same_id"
        assert ranked[1].id == "different_id"


class TestRecencyRankingStrategy:
    """Tests for RecencyRankingStrategy."""

    def test_rank_by_recency(self, sample_results):
        """Test ranking by recency."""
        strategy = RecencyRankingStrategy()
        query = MemoryQuery(query="auth")

        ranked = strategy.rank(sample_results, query, limit=10)

        assert len(ranked) == 4
        # Most recent first
        assert ranked[0].id == "msg_2"  # 15 min ago
        assert ranked[1].id == "msg_1"  # 30 min ago

    def test_rank_handles_none_timestamp(self):
        """Test handling results with no timestamp."""
        strategy = RecencyRankingStrategy()
        results = [
            MemoryResult(
                source=MemoryType.ENTITY,
                content="a",
                relevance=0.9,
                timestamp=time.time(),
            ),
            MemoryResult(source=MemoryType.ENTITY, content="b", relevance=0.8, timestamp=None),
        ]
        query = MemoryQuery(query="test")

        ranked = strategy.rank(results, query, limit=10)

        assert len(ranked) == 2
        assert ranked[0].timestamp is not None  # Timestamped first
        assert ranked[1].timestamp is None


class TestHybridRankingStrategy:
    """Tests for HybridRankingStrategy."""

    def test_rank_hybrid_default_weights(self, sample_results):
        """Test hybrid ranking with default weights."""
        strategy = HybridRankingStrategy()  # 0.7 relevance, 0.3 recency
        query = MemoryQuery(query="auth")

        ranked = strategy.rank(sample_results, query, limit=10)

        assert len(ranked) == 4
        # Should balance relevance and recency

    def test_rank_hybrid_custom_weights(self, sample_results):
        """Test hybrid ranking with custom weights."""
        strategy = HybridRankingStrategy(
            relevance_weight=0.5,
            recency_weight=0.5,
            recency_decay_hours=12.0,
        )
        query = MemoryQuery(query="auth")

        ranked = strategy.rank(sample_results, query, limit=10)

        assert len(ranked) == 4

    def test_recency_score_decay(self):
        """Test recency score exponential decay."""
        strategy = HybridRankingStrategy(recency_decay_hours=24.0)

        # Current time = score ~1.0
        assert strategy._recency_score(time.time()) > 0.9

        # 24 hours ago = score ~0.5 (allow wider tolerance for decay formula)
        score_24h = strategy._recency_score(time.time() - 86400)
        assert 0.4 < score_24h < 0.7, f"Expected ~0.5, got {score_24h}"

        # None timestamp = score 0.5
        assert strategy._recency_score(None) == 0.5


# =============================================================================
# Test UnifiedMemoryCoordinator
# =============================================================================


class TestUnifiedMemoryCoordinator:
    """Tests for UnifiedMemoryCoordinator."""

    @pytest.fixture
    def coordinator(self):
        """Create a fresh coordinator."""
        return create_memory_coordinator()

    @pytest.fixture
    def entity_provider(self):
        """Create mock entity provider."""
        results = [
            MemoryResult(
                source=MemoryType.ENTITY,
                content={"name": "UserAuth", "type": "class"},
                relevance=0.9,
                id="ent_1",
            ),
            MemoryResult(
                source=MemoryType.ENTITY,
                content={"name": "AuthService", "type": "class"},
                relevance=0.8,
                id="ent_2",
            ),
        ]
        return MockMemoryProvider(MemoryType.ENTITY, results)

    @pytest.fixture
    def conversation_provider(self):
        """Create mock conversation provider."""
        results = [
            MemoryResult(
                source=MemoryType.CONVERSATION,
                content={"role": "user", "content": "Fix auth bug"},
                relevance=0.7,
                id="msg_1",
            ),
        ]
        return MockMemoryProvider(MemoryType.CONVERSATION, results)

    def test_register_provider(self, coordinator, entity_provider):
        """Test registering a provider."""
        coordinator.register_provider(entity_provider)

        assert MemoryType.ENTITY in coordinator.get_registered_types()
        assert coordinator.get_provider(MemoryType.ENTITY) is entity_provider

    def test_unregister_provider(self, coordinator, entity_provider):
        """Test unregistering a provider."""
        coordinator.register_provider(entity_provider)
        result = coordinator.unregister_provider(MemoryType.ENTITY)

        assert result is True
        assert MemoryType.ENTITY not in coordinator.get_registered_types()

    def test_unregister_nonexistent_provider(self, coordinator):
        """Test unregistering a provider that doesn't exist."""
        result = coordinator.unregister_provider(MemoryType.ENTITY)
        assert result is False

    @pytest.mark.asyncio
    async def test_search_all_single_provider(self, coordinator, entity_provider):
        """Test searching with a single provider."""
        coordinator.register_provider(entity_provider)

        results = await coordinator.search_all("Auth", limit=10)

        assert len(results) == 2
        assert all(r.source == MemoryType.ENTITY for r in results)

    @pytest.mark.asyncio
    async def test_search_all_multiple_providers(
        self, coordinator, entity_provider, conversation_provider
    ):
        """Test searching across multiple providers."""
        coordinator.register_provider(entity_provider)
        coordinator.register_provider(conversation_provider)

        results = await coordinator.search_all("auth", limit=10)

        assert len(results) == 3
        sources = {r.source for r in results}
        assert MemoryType.ENTITY in sources
        assert MemoryType.CONVERSATION in sources

    @pytest.mark.asyncio
    async def test_search_all_with_type_filter(
        self, coordinator, entity_provider, conversation_provider
    ):
        """Test searching with memory type filter."""
        coordinator.register_provider(entity_provider)
        coordinator.register_provider(conversation_provider)

        results = await coordinator.search_all(
            "auth",
            limit=10,
            memory_types=[MemoryType.ENTITY],
        )

        assert all(r.source == MemoryType.ENTITY for r in results)

    @pytest.mark.asyncio
    async def test_search_all_no_providers(self, coordinator):
        """Test searching with no registered providers."""
        results = await coordinator.search_all("test")
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_search_all_unavailable_provider(self, coordinator, entity_provider):
        """Test searching with unavailable provider."""
        entity_provider._available = False
        coordinator.register_provider(entity_provider)

        results = await coordinator.search_all("Auth")

        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_search_type(self, coordinator, entity_provider, conversation_provider):
        """Test searching a specific type."""
        coordinator.register_provider(entity_provider)
        coordinator.register_provider(conversation_provider)

        results = await coordinator.search_type(MemoryType.ENTITY, "Auth")

        assert all(r.source == MemoryType.ENTITY for r in results)

    @pytest.mark.asyncio
    async def test_get(self, coordinator, entity_provider):
        """Test getting a specific item."""
        coordinator.register_provider(entity_provider)
        await entity_provider.store("test_key", {"data": "value"})

        result = await coordinator.get(MemoryType.ENTITY, "test_key")

        assert result is not None
        assert result.content == {"data": "value"}

    @pytest.mark.asyncio
    async def test_store(self, coordinator, entity_provider):
        """Test storing a value."""
        coordinator.register_provider(entity_provider)

        success = await coordinator.store(
            MemoryType.ENTITY,
            "new_key",
            {"name": "Test"},
            metadata={"created": True},
        )

        assert success is True
        assert "new_key" in entity_provider._stored

    @pytest.mark.asyncio
    async def test_store_no_provider(self, coordinator):
        """Test storing without provider."""
        success = await coordinator.store(MemoryType.ENTITY, "key", "value")
        assert success is False

    def test_get_stats(self, coordinator, entity_provider):
        """Test getting coordinator stats."""
        coordinator.register_provider(entity_provider)

        stats = coordinator.get_stats()

        assert stats["query_count"] == 0
        assert stats["error_count"] == 0
        assert "ENTITY" in stats["providers"]

    def test_set_ranking_strategy(self, coordinator):
        """Test setting ranking strategy."""
        new_strategy = RecencyRankingStrategy()
        coordinator.set_ranking_strategy(new_strategy)

        assert coordinator._ranking_strategy is new_strategy

    @pytest.mark.asyncio
    async def test_error_handling(self, coordinator):
        """Test error handling in search."""

        class FailingProvider:
            @property
            def memory_type(self):
                return MemoryType.ENTITY

            async def search(self, query):
                raise RuntimeError("Search failed")

            def is_available(self):
                return True

        coordinator.register_provider(FailingProvider())

        # Should not raise, but log warning
        results = await coordinator.search_all("test")
        assert len(results) == 0


# =============================================================================
# Test Singleton Behavior
# =============================================================================


class TestSingletonCoordinator:
    """Tests for singleton coordinator behavior."""

    def test_get_memory_coordinator_returns_same_instance(self, reset_coordinator):
        """Test that get_memory_coordinator returns singleton."""
        c1 = get_memory_coordinator()
        c2 = get_memory_coordinator()

        assert c1 is c2

    def test_reset_memory_coordinator(self, reset_coordinator):
        """Test resetting the singleton."""
        c1 = get_memory_coordinator()
        reset_memory_coordinator()
        c2 = get_memory_coordinator()

        assert c1 is not c2


# =============================================================================
# Test Protocol Conformance
# =============================================================================


class TestProtocolConformance:
    """Tests for protocol conformance."""

    def test_mock_provider_conforms_to_protocol(self):
        """Test that MockMemoryProvider conforms to MemoryProviderProtocol."""
        provider = MockMemoryProvider(MemoryType.ENTITY)

        assert isinstance(provider, MemoryProviderProtocol)
        assert hasattr(provider, "memory_type")
        assert hasattr(provider, "search")
        assert hasattr(provider, "store")
        assert hasattr(provider, "get")
        assert hasattr(provider, "is_available")

    def test_ranking_strategies_conform_to_protocol(self):
        """Test that ranking strategies conform to RankingStrategyProtocol."""
        strategies = [
            RelevanceRankingStrategy(),
            RecencyRankingStrategy(),
            HybridRankingStrategy(),
        ]

        for strategy in strategies:
            assert isinstance(strategy, RankingStrategyProtocol)
            assert hasattr(strategy, "rank")


# =============================================================================
# Integration Tests
# =============================================================================


class TestMemoryIntegration:
    """Integration tests for memory coordinator."""

    @pytest.mark.asyncio
    async def test_federated_search_ranking(self, reset_coordinator):
        """Test federated search with proper ranking."""
        coordinator = get_memory_coordinator()

        # Create providers with different results
        entity_results = [
            MemoryResult(
                source=MemoryType.ENTITY,
                content="High relevance auth",
                relevance=0.95,
                id="e1",
                timestamp=time.time() - 7200,
            ),
        ]
        conv_results = [
            MemoryResult(
                source=MemoryType.CONVERSATION,
                content="Recent auth msg",
                relevance=0.7,
                id="c1",
                timestamp=time.time() - 300,
            ),
        ]

        coordinator.register_provider(MockMemoryProvider(MemoryType.ENTITY, entity_results))
        coordinator.register_provider(MockMemoryProvider(MemoryType.CONVERSATION, conv_results))

        # Hybrid ranking should balance relevance and recency
        results = await coordinator.search_all("auth")

        assert len(results) == 2
        # Both should be present, order depends on hybrid scoring

    @pytest.mark.asyncio
    async def test_parallel_search_performance(self, reset_coordinator):
        """Test that searches run in parallel."""
        coordinator = get_memory_coordinator()

        # Create slow providers
        class SlowProvider:
            def __init__(self, memory_type, delay):
                self._memory_type = memory_type
                self._delay = delay

            @property
            def memory_type(self):
                return self._memory_type

            async def search(self, query):
                await asyncio.sleep(self._delay)
                return [
                    MemoryResult(
                        source=self._memory_type,
                        content=f"Result from {self._memory_type.name}",
                        relevance=0.8,
                        id=f"id_{self._memory_type.name}",
                    )
                ]

            def is_available(self):
                return True

        coordinator.register_provider(SlowProvider(MemoryType.ENTITY, 0.1))
        coordinator.register_provider(SlowProvider(MemoryType.CONVERSATION, 0.1))
        coordinator.register_provider(SlowProvider(MemoryType.GRAPH, 0.1))

        start = time.time()
        results = await coordinator.search_all("test")
        elapsed = time.time() - start

        assert len(results) == 3
        # If parallel, should take ~0.1s, not 0.3s
        assert elapsed < 0.25, f"Search took {elapsed:.2f}s, expected parallel execution"
