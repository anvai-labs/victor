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

"""Unit tests for prompt enrichment service.

Tests the core PromptEnrichmentService and related components:
- EnrichmentContext creation and usage
- ContextEnrichment data class
- EnrichedPrompt result handling
- Token budget enforcement
- Caching behavior
- Outcome recording for RL
"""

import asyncio
import pytest
from typing import List

from victor.framework.enrichment import (
    PromptEnrichmentService,
    EnrichmentContext,
    ContextEnrichment,
    EnrichedPrompt,
    EnrichmentOutcome,
    EnrichmentType,
    EnrichmentPriority,
    EnrichmentCache,
)

# =============================================================================
# Fixtures
# =============================================================================


class MockEnrichmentStrategy:
    """Mock enrichment strategy for testing."""

    def __init__(
        self,
        enrichments: List[ContextEnrichment] = None,
        delay_ms: float = 0,
        should_fail: bool = False,
    ):
        self._enrichments = enrichments or []
        self._delay_ms = delay_ms
        self._should_fail = should_fail

    async def get_enrichments(
        self,
        prompt: str,
        context: EnrichmentContext,
    ) -> List[ContextEnrichment]:
        if self._delay_ms > 0:
            await asyncio.sleep(self._delay_ms / 1000.0)
        if self._should_fail:
            raise RuntimeError("Mock strategy failure")
        return self._enrichments

    def get_priority(self) -> int:
        return 50

    def get_token_allocation(self) -> float:
        return 0.4


@pytest.fixture
def enrichment_service():
    """Create a basic enrichment service."""
    return PromptEnrichmentService(
        max_tokens=2000,
        timeout_ms=500,
        cache_enabled=True,
    )


@pytest.fixture
def basic_context():
    """Create a basic enrichment context."""
    return EnrichmentContext(
        session_id="test_session",
        task_type="edit",
        file_mentions=["src/main.py"],
        symbol_mentions=["calculate_total"],
    )


@pytest.fixture
def sample_enrichments():
    """Create sample enrichments for testing."""
    return [
        ContextEnrichment(
            type=EnrichmentType.KNOWLEDGE_GRAPH,
            content="Function `calculate_total` found in src/utils.py:42",
            priority=EnrichmentPriority.HIGH,
            token_estimate=50,
        ),
        ContextEnrichment(
            type=EnrichmentType.CODE_SNIPPET,
            content="def calculate_total(items): return sum(items)",
            priority=EnrichmentPriority.NORMAL,
            token_estimate=30,
        ),
    ]


# =============================================================================
# EnrichmentContext Tests
# =============================================================================


class TestEnrichmentContext:
    """Tests for EnrichmentContext data class."""

    def test_default_values(self):
        """Default values are properly initialized."""
        context = EnrichmentContext()

        assert context.session_id is None
        assert context.task_type is None
        assert context.file_mentions == []
        assert context.symbol_mentions == []
        assert context.tool_history == []
        assert context.metadata == {}

    def test_with_values(self):
        """Values are properly set."""
        context = EnrichmentContext(
            session_id="session_123",
            task_type="create",
            file_mentions=["file1.py", "file2.py"],
            symbol_mentions=["MyClass"],
            metadata={"key": "value"},
        )

        assert context.session_id == "session_123"
        assert context.task_type == "create"
        assert len(context.file_mentions) == 2
        assert "MyClass" in context.symbol_mentions
        assert context.metadata["key"] == "value"


# =============================================================================
# ContextEnrichment Tests
# =============================================================================


class TestContextEnrichment:
    """Tests for ContextEnrichment data class."""

    def test_token_estimate_auto_calculated(self):
        """Token estimate is auto-calculated from content."""
        enrichment = ContextEnrichment(
            type=EnrichmentType.KNOWLEDGE_GRAPH,
            content="A" * 400,  # 400 chars
        )

        # ~4 chars per token = 100 tokens
        assert enrichment.token_estimate == 100

    def test_explicit_token_estimate(self):
        """Explicit token estimate overrides auto-calculation."""
        enrichment = ContextEnrichment(
            type=EnrichmentType.KNOWLEDGE_GRAPH,
            content="A" * 400,
            token_estimate=50,  # Explicit value
        )

        assert enrichment.token_estimate == 50

    def test_priority_default(self):
        """Default priority is NORMAL."""
        enrichment = ContextEnrichment(
            type=EnrichmentType.CODE_SNIPPET,
            content="test",
        )

        assert enrichment.priority == EnrichmentPriority.NORMAL


# =============================================================================
# EnrichedPrompt Tests
# =============================================================================


class TestEnrichedPrompt:
    """Tests for EnrichedPrompt data class."""

    def test_properties(self, sample_enrichments):
        """Properties are computed correctly."""
        result = EnrichedPrompt(
            original_prompt="Original",
            enriched_prompt="Enriched",
            enrichments=sample_enrichments,
            total_tokens_added=80,
        )

        assert result.enrichment_count == 2
        assert EnrichmentType.KNOWLEDGE_GRAPH.value in result.enrichment_types
        assert EnrichmentType.CODE_SNIPPET.value in result.enrichment_types

    def test_from_cache_flag(self):
        """from_cache flag is properly set."""
        result = EnrichedPrompt(
            original_prompt="Original",
            enriched_prompt="Enriched",
            from_cache=True,
        )

        assert result.from_cache is True


# =============================================================================
# EnrichmentCache Tests
# =============================================================================


class TestEnrichmentCache:
    """Tests for EnrichmentCache."""

    def test_cache_miss(self):
        """Cache returns None for missing entries."""
        cache = EnrichmentCache(ttl_seconds=300)
        context = EnrichmentContext(task_type="test")

        result = cache.get("prompt", "coding", context)

        assert result is None

    def test_cache_hit(self):
        """Cache returns result for existing entries."""
        cache = EnrichmentCache(ttl_seconds=300)
        context = EnrichmentContext(task_type="test")

        original = EnrichedPrompt(
            original_prompt="prompt",
            enriched_prompt="enriched",
        )

        cache.set("prompt", "coding", context, original)
        result = cache.get("prompt", "coding", context)

        assert result is not None
        assert result.from_cache is True
        assert result.original_prompt == "prompt"

    def test_cache_eviction_on_ttl(self):
        """Cache evicts entries after TTL expires."""
        cache = EnrichmentCache(ttl_seconds=0)  # Immediate expiry
        context = EnrichmentContext(task_type="test")

        original = EnrichedPrompt(
            original_prompt="prompt",
            enriched_prompt="enriched",
        )

        cache.set("prompt", "coding", context, original)

        # Entry should be expired immediately
        import time

        time.sleep(0.01)  # Small delay to ensure TTL check

        result = cache.get("prompt", "coding", context)
        assert result is None

    def test_cache_max_entries(self):
        """Cache evicts oldest entries when at capacity."""
        cache = EnrichmentCache(ttl_seconds=300, max_entries=2)

        for i in range(3):
            context = EnrichmentContext(task_type=f"test_{i}")
            result = EnrichedPrompt(
                original_prompt=f"prompt_{i}",
                enriched_prompt=f"enriched_{i}",
            )
            cache.set(f"prompt_{i}", "coding", context, result)

        # Should have evicted oldest entry
        assert len(cache) <= 2


# =============================================================================
# PromptEnrichmentService Tests
# =============================================================================


class TestPromptEnrichmentService:
    """Tests for PromptEnrichmentService."""

    @pytest.mark.asyncio
    async def test_enrich_without_strategy(
        self,
        enrichment_service,
        basic_context,
    ):
        """Returns original prompt when no strategy registered."""
        result = await enrichment_service.enrich(
            prompt="Test prompt",
            vertical="unknown",
            context=basic_context,
        )

        assert result.original_prompt == "Test prompt"
        assert result.enriched_prompt == "Test prompt"
        assert result.enrichment_count == 0

    @pytest.mark.asyncio
    async def test_enrich_with_strategy(
        self,
        enrichment_service,
        basic_context,
        sample_enrichments,
    ):
        """Applies enrichments from registered strategy."""
        strategy = MockEnrichmentStrategy(enrichments=sample_enrichments)
        enrichment_service.register_strategy("coding", strategy)

        result = await enrichment_service.enrich(
            prompt="Test prompt",
            vertical="coding",
            context=basic_context,
        )

        assert result.enrichment_count == 2
        assert result.total_tokens_added == 80
        assert "calculate_total" in result.enriched_prompt

    @pytest.mark.asyncio
    async def test_token_budget_enforcement(
        self,
        basic_context,
    ):
        """Respects token budget when applying enrichments."""
        service = PromptEnrichmentService(max_tokens=50)

        # Create enrichments that exceed budget
        enrichments = [
            ContextEnrichment(
                type=EnrichmentType.KNOWLEDGE_GRAPH,
                content="A" * 200,  # 50 tokens
                priority=EnrichmentPriority.HIGH,
                token_estimate=50,
            ),
            ContextEnrichment(
                type=EnrichmentType.CODE_SNIPPET,
                content="B" * 200,  # 50 tokens - should be excluded
                priority=EnrichmentPriority.NORMAL,
                token_estimate=50,
            ),
        ]

        strategy = MockEnrichmentStrategy(enrichments=enrichments)
        service.register_strategy("coding", strategy)

        result = await service.enrich(
            prompt="Test prompt",
            vertical="coding",
            context=basic_context,
        )

        # Should only include first enrichment (50 tokens <= 50 budget)
        assert result.enrichment_count == 1
        assert result.total_tokens_added == 50

    @pytest.mark.asyncio
    async def test_priority_ordering(
        self,
        enrichment_service,
        basic_context,
    ):
        """Higher priority enrichments are applied first."""
        enrichments = [
            ContextEnrichment(
                type=EnrichmentType.CODE_SNIPPET,
                content="Low priority content",
                priority=EnrichmentPriority.LOW,
                token_estimate=100,
            ),
            ContextEnrichment(
                type=EnrichmentType.KNOWLEDGE_GRAPH,
                content="Critical content",
                priority=EnrichmentPriority.CRITICAL,
                token_estimate=100,
            ),
        ]

        strategy = MockEnrichmentStrategy(enrichments=enrichments)
        enrichment_service.register_strategy("coding", strategy)

        result = await enrichment_service.enrich(
            prompt="Test",
            vertical="coding",
            context=basic_context,
        )

        # Critical priority should come before low
        assert result.enrichments[0].priority == EnrichmentPriority.CRITICAL

    @pytest.mark.asyncio
    async def test_timeout_handling(
        self,
        basic_context,
    ):
        """Handles strategy timeout gracefully."""
        service = PromptEnrichmentService(timeout_ms=10)  # Very short timeout

        # Strategy that takes too long
        strategy = MockEnrichmentStrategy(
            enrichments=[
                ContextEnrichment(
                    type=EnrichmentType.KNOWLEDGE_GRAPH,
                    content="Slow content",
                )
            ],
            delay_ms=100,  # Longer than timeout
        )

        service.register_strategy("coding", strategy)

        result = await service.enrich(
            prompt="Test prompt",
            vertical="coding",
            context=basic_context,
        )

        # Should return original prompt on timeout
        assert result.enriched_prompt == "Test prompt"
        assert result.enrichment_count == 0

    @pytest.mark.asyncio
    async def test_error_handling(
        self,
        enrichment_service,
        basic_context,
    ):
        """Handles strategy errors gracefully."""
        strategy = MockEnrichmentStrategy(should_fail=True)
        enrichment_service.register_strategy("coding", strategy)

        result = await enrichment_service.enrich(
            prompt="Test prompt",
            vertical="coding",
            context=basic_context,
        )

        # Should return original prompt on error
        assert result.enriched_prompt == "Test prompt"
        assert result.enrichment_count == 0

    @pytest.mark.asyncio
    async def test_caching(
        self,
        sample_enrichments,
    ):
        """Results are cached for repeated requests."""
        # Create service with cache enabled
        service = PromptEnrichmentService(
            max_tokens=2000,
            timeout_ms=500,
            cache_enabled=True,
        )

        # Track strategy call count
        call_count = 0

        class CountingStrategy:
            async def get_enrichments(self, prompt, context):
                nonlocal call_count
                call_count += 1
                return sample_enrichments

            def get_priority(self):
                return 50

            def get_token_allocation(self):
                return 0.4

        strategy = CountingStrategy()
        service.register_strategy("coding", strategy)

        # Use exact same context for both calls
        context = EnrichmentContext(
            session_id="test_session",
            task_type="edit",
            file_mentions=["src/main.py"],
            symbol_mentions=["calculate_total"],
        )

        # First request - should call strategy
        result1 = await service.enrich(
            prompt="Test prompt",
            vertical="coding",
            context=context,
        )

        # Second request with same prompt/context - should use cache
        result2 = await service.enrich(
            prompt="Test prompt",
            vertical="coding",
            context=context,
        )

        # Strategy should only be called once (first request)
        assert call_count == 1, f"Strategy called {call_count} times, expected 1"

        # First result should not be from cache
        assert result1.from_cache is False

        # Second result should be from cache
        assert result2.from_cache is True

        # Both results should have the same enrichments
        assert result1.enrichment_count == result2.enrichment_count

    def test_register_unregister_strategy(self, enrichment_service):
        """Strategy registration and unregistration works."""
        strategy = MockEnrichmentStrategy()

        enrichment_service.register_strategy("test", strategy)
        assert "test" in enrichment_service.registered_verticals

        enrichment_service.unregister_strategy("test")
        assert "test" not in enrichment_service.registered_verticals

    def test_outcome_recording(self, enrichment_service):
        """Outcome callbacks are called correctly."""
        recorded_outcomes = []

        def callback(outcome: EnrichmentOutcome):
            recorded_outcomes.append(outcome)

        enrichment_service.on_outcome(callback)

        outcome = EnrichmentOutcome(
            enrichment_type="knowledge_graph",
            enrichment_count=2,
            task_success=True,
            quality_improvement=0.15,
        )

        enrichment_service.record_outcome(outcome)

        assert len(recorded_outcomes) == 1
        assert recorded_outcomes[0].task_success is True

    def test_property_accessors(self):
        """Property accessors work correctly."""
        service = PromptEnrichmentService(max_tokens=1000, timeout_ms=250)

        assert service.max_tokens == 1000
        assert service.timeout_ms == 250

        service.max_tokens = 500
        service.timeout_ms = 100

        assert service.max_tokens == 500
        assert service.timeout_ms == 100


# =============================================================================
# EnrichmentOutcome Tests
# =============================================================================


class TestEnrichmentOutcome:
    """Tests for EnrichmentOutcome data class."""

    def test_creation(self):
        """Outcome can be created with all fields."""
        outcome = EnrichmentOutcome(
            enrichment_type="knowledge_graph",
            enrichment_count=3,
            task_success=True,
            quality_improvement=0.2,
            task_type="edit",
            vertical="coding",
        )

        assert outcome.enrichment_type == "knowledge_graph"
        assert outcome.enrichment_count == 3
        assert outcome.task_success is True
        assert outcome.quality_improvement == 0.2
        assert outcome.task_type == "edit"
        assert outcome.vertical == "coding"

    def test_optional_fields(self):
        """Optional fields default to None."""
        outcome = EnrichmentOutcome(
            enrichment_type="web_search",
            enrichment_count=1,
            task_success=False,
            quality_improvement=-0.1,
        )

        assert outcome.task_type is None
        assert outcome.vertical is None
