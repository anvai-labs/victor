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

"""Unit tests for Query Enhancement module.

Tests cover:
- EnhancedQuery dataclass
- EntityExpandStrategy
- RewriteStrategy
- DecompositionStrategy
- QueryEnhancementPipeline
- Caching behavior
- Registry functionality
- Domain configurations
"""

from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from victor.core.query_enhancement import (
    # Types
    EnhancementTechnique,
    EnhancementContext,
    EnhancedQuery,
    QueryEnhancementConfig,
    EnhancementMetrics,
    EnhancementResult,
    DomainConfig,
    # Domain configs
    FINANCIAL_DOMAIN,
    CODE_DOMAIN,
    RESEARCH_DOMAIN,
    GENERAL_DOMAIN,
    get_domain_config,
    # Pipeline and Registry
    QueryEnhancementPipeline,
    QueryEnhancementRegistry,
    get_default_registry,
    # Strategies
    EntityExpandStrategy,
    RewriteStrategy,
    DecompositionStrategy,
)

# =============================================================================
# EnhancedQuery Dataclass Tests
# =============================================================================


class TestEnhancedQuery:
    """Tests for EnhancedQuery dataclass."""

    def test_create_basic_query(self):
        """Test creating a basic enhanced query."""
        query = EnhancedQuery(
            original="test query",
            enhanced="enhanced test query",
            technique=EnhancementTechnique.REWRITE,
        )
        assert query.original == "test query"
        assert query.enhanced == "enhanced test query"
        assert query.technique == EnhancementTechnique.REWRITE
        assert query.variants == []
        assert query.sub_queries == []
        assert query.confidence == 1.0

    def test_create_query_with_variants(self):
        """Test creating query with variants."""
        query = EnhancedQuery(
            original="test",
            enhanced="enhanced test",
            technique=EnhancementTechnique.MULTI_QUERY,
            variants=["variant 1", "variant 2"],
        )
        assert len(query.variants) == 2
        assert "variant 1" in query.variants

    def test_create_query_with_sub_queries(self):
        """Test creating query with sub-queries."""
        query = EnhancedQuery(
            original="complex query",
            enhanced="complex query",
            technique=EnhancementTechnique.DECOMPOSITION,
            sub_queries=["sub query 1", "sub query 2", "sub query 3"],
        )
        assert len(query.sub_queries) == 3

    def test_get_all_queries(self):
        """Test getting all queries including enhanced and variants."""
        query = EnhancedQuery(
            original="test",
            enhanced="enhanced test",
            technique=EnhancementTechnique.MULTI_QUERY,
            variants=["variant 1", "variant 2"],
            sub_queries=["sub query 1"],
        )
        all_queries = query.get_all_queries()
        assert "enhanced test" in all_queries
        assert "variant 1" in all_queries
        assert "variant 2" in all_queries
        assert "sub query 1" in all_queries
        # Should not include original
        assert "test" not in all_queries

    def test_get_all_queries_removes_duplicates(self):
        """Test that get_all_queries removes duplicates."""
        query = EnhancedQuery(
            original="test",
            enhanced="same query",
            technique=EnhancementTechnique.MULTI_QUERY,
            variants=["same query", "different query"],
        )
        all_queries = query.get_all_queries()
        # Should deduplicate
        assert all_queries.count("same query") == 1

    def test_query_repr(self):
        """Test query string representation."""
        query = EnhancedQuery(
            original="test",
            enhanced="enhanced",
            technique=EnhancementTechnique.REWRITE,
        )
        repr_str = repr(query)
        assert "test" in repr_str
        assert "enhanced" in repr_str
        assert "rewrite" in repr_str


# =============================================================================
# EnhancementContext Tests
# =============================================================================


class TestEnhancementContext:
    """Tests for EnhancementContext dataclass."""

    def test_create_default_context(self):
        """Test creating context with defaults."""
        context = EnhancementContext()
        assert context.domain == "general"
        assert context.entity_metadata == []
        assert context.max_variants == 3

    def test_create_financial_context(self):
        """Test creating financial domain context."""
        context = EnhancementContext(
            domain="financial",
            entity_metadata=[
                {"name": "Apple Inc", "ticker": "AAPL"},
                {"name": "Microsoft Corporation", "ticker": "MSFT"},
            ],
        )
        assert context.domain == "financial"
        assert len(context.entity_metadata) == 2

    def test_get_entity_names(self):
        """Test extracting entity names from metadata."""
        context = EnhancementContext(
            entity_metadata=[
                {"name": "Apple Inc", "ticker": "AAPL"},
                {"name": "Tesla Inc"},  # No ticker
            ]
        )
        names = context.get_entity_names()
        assert "Apple Inc" in names
        assert "AAPL" in names
        assert "Tesla Inc" in names

    def test_get_entity_names_empty(self):
        """Test get_entity_names with empty metadata."""
        context = EnhancementContext()
        names = context.get_entity_names()
        assert names == []


# =============================================================================
# EntityExpandStrategy Tests
# =============================================================================


class TestEntityExpandStrategy:
    """Tests for EntityExpandStrategy (no LLM required)."""

    @pytest.fixture
    def strategy(self):
        """Create EntityExpandStrategy instance."""
        return EntityExpandStrategy()

    def test_strategy_properties(self, strategy):
        """Test strategy properties."""
        assert strategy.name == "entity_expand"
        assert strategy.technique == EnhancementTechnique.ENTITY_EXPAND
        assert strategy.requires_llm is False

    @pytest.mark.asyncio
    async def test_enhance_with_entities(self, strategy):
        """Test enhancement with entity metadata."""
        context = EnhancementContext(
            domain="financial",
            entity_metadata=[
                {"name": "Apple Inc", "ticker": "AAPL"},
            ],
        )
        result = await strategy.enhance("What is revenue?", context)

        assert result.original == "What is revenue?"
        assert "Apple Inc" in result.enhanced or "AAPL" in result.enhanced
        assert result.technique == EnhancementTechnique.ENTITY_EXPAND
        assert result.confidence >= 0.5

    @pytest.mark.asyncio
    async def test_enhance_without_entities(self, strategy):
        """Test enhancement without entity metadata."""
        context = EnhancementContext(domain="general")
        result = await strategy.enhance("test query", context)

        # Without entities, should return original
        assert result.enhanced == "test query"
        assert result.metadata.get("no_expansion") is True

    @pytest.mark.asyncio
    async def test_enhance_avoids_duplicate_terms(self, strategy):
        """Test that existing terms aren't duplicated."""
        context = EnhancementContext(
            entity_metadata=[
                {"name": "Apple", "ticker": "AAPL"},
            ],
        )
        # Query already contains "Apple"
        result = await strategy.enhance("What is Apple revenue?", context)

        # Should not duplicate "Apple"
        enhanced_lower = result.enhanced.lower()
        assert enhanced_lower.count("apple") == 1

    @pytest.mark.asyncio
    async def test_enhance_multiple_entities(self, strategy):
        """Test enhancement with multiple entities."""
        context = EnhancementContext(
            entity_metadata=[
                {"name": "Apple Inc", "ticker": "AAPL"},
                {"name": "Microsoft Corp", "ticker": "MSFT"},
            ],
        )
        result = await strategy.enhance("Compare companies", context)

        # Should include both company terms
        assert "Apple" in result.enhanced or "AAPL" in result.enhanced
        assert "Microsoft" in result.enhanced or "MSFT" in result.enhanced

    @pytest.mark.asyncio
    async def test_max_expansion_terms_limit(self, strategy):
        """Test that expansion terms are limited."""
        # Create many entities
        entities = [{"name": f"Company{i}", "ticker": f"TK{i}"} for i in range(10)]
        context = EnhancementContext(entity_metadata=entities)

        result = await strategy.enhance("test", context)

        # Should have limited expansion (max 6 terms)
        expansion_terms = result.metadata.get("expansion_terms", [])
        assert len(expansion_terms) <= 6


# =============================================================================
# RewriteStrategy Tests
# =============================================================================


class TestRewriteStrategy:
    """Tests for RewriteStrategy (LLM-based)."""

    @pytest.fixture
    def strategy(self):
        """Create RewriteStrategy instance."""
        return RewriteStrategy()

    def test_strategy_properties(self, strategy):
        """Test strategy properties."""
        assert strategy.name == "rewrite"
        assert strategy.technique == EnhancementTechnique.REWRITE
        assert strategy.requires_llm is True

    def test_has_domain_templates(self, strategy):
        """Test that domain templates are registered."""
        assert "financial" in strategy._prompt_templates
        assert "code" in strategy._prompt_templates
        assert "general" in strategy._prompt_templates

    def test_get_prompt_template(self, strategy):
        """Test getting domain-specific template."""
        financial_template = strategy.get_prompt_template("financial")
        assert "SEC" in financial_template or "financial" in financial_template.lower()

        code_template = strategy.get_prompt_template("code")
        assert "code" in code_template.lower()

    def test_get_prompt_template_fallback(self, strategy):
        """Test template fallback for unknown domain."""
        template = strategy.get_prompt_template("unknown_domain")
        # Should fall back to general
        assert template == strategy._prompt_templates.get("general")

    def test_clean_response(self, strategy):
        """Test cleaning LLM response."""
        # Test removing prefix
        cleaned = strategy._clean_response("Rewritten query: enhanced query")
        assert cleaned == "enhanced query"

        # Test removing quotes
        cleaned = strategy._clean_response('"enhanced query"')
        assert cleaned == "enhanced query"

    def test_is_valid_rewrite(self, strategy):
        """Test rewrite validation."""
        # Valid rewrite
        assert strategy._is_valid_rewrite("short", "this is a longer rewrite")

        # Too short
        assert not strategy._is_valid_rewrite("original query", "x")

        # Too long
        long_rewrite = "x " * 1000
        assert not strategy._is_valid_rewrite("short", long_rewrite)

        # Identical (no change)
        assert not strategy._is_valid_rewrite("same query", "same query")

    @pytest.mark.asyncio
    async def test_enhance_fallback_without_llm(self, strategy):
        """Test fallback behavior when LLM unavailable."""
        # Mock _call_llm to return None
        strategy._call_llm = AsyncMock(return_value=None)

        context = EnhancementContext(domain="general")
        result = await strategy.enhance("test query", context)

        # Should fall back to original
        assert result.enhanced == "test query"
        assert result.metadata.get("fallback") is True

    @pytest.mark.asyncio
    async def test_enhance_with_llm_response(self, strategy):
        """Test enhancement with mocked LLM response."""
        # The enhanced response should be longer than original/3 and shorter than original*5
        strategy._call_llm = AsyncMock(return_value="enhanced search query terms expanded")

        context = EnhancementContext(domain="general")
        result = await strategy.enhance("test query input", context)

        assert result.enhanced == "enhanced search query terms expanded"
        assert result.confidence >= 0.8


# =============================================================================
# DecompositionStrategy Tests
# =============================================================================


class TestDecompositionStrategy:
    """Tests for DecompositionStrategy."""

    @pytest.fixture
    def strategy(self):
        """Create DecompositionStrategy instance."""
        return DecompositionStrategy()

    def test_strategy_properties(self, strategy):
        """Test strategy properties."""
        assert strategy.name == "decomposition"
        assert strategy.technique == EnhancementTechnique.DECOMPOSITION
        assert strategy.requires_llm is True

    def test_parse_sub_queries_json(self, strategy):
        """Test parsing JSON array response."""
        response = '["query 1", "query 2", "query 3"]'
        sub_queries = strategy._parse_sub_queries(response)
        assert sub_queries == ["query 1", "query 2", "query 3"]

    def test_parse_sub_queries_with_markdown(self, strategy):
        """Test parsing JSON in markdown code block."""
        response = 'Here are the queries:\n["query 1", "query 2"]'
        sub_queries = strategy._parse_sub_queries(response)
        assert "query 1" in sub_queries

    def test_parse_sub_queries_numbered_list(self, strategy):
        """Test parsing numbered list fallback."""
        response = "1. First query\n2. Second query\n3. Third query"
        sub_queries = strategy._parse_sub_queries(response)
        assert len(sub_queries) >= 2

    def test_heuristic_decompose_comparison(self, strategy):
        """Test heuristic decomposition for comparison queries."""
        context = EnhancementContext(
            entity_metadata=[
                {"name": "Apple Inc", "ticker": "AAPL"},
                {"name": "Microsoft Corp", "ticker": "MSFT"},
            ],
        )
        query = "Compare Apple and Microsoft revenue growth and profit margins"
        sub_queries = strategy._heuristic_decompose(query, context)

        # Should create sub-queries for each entity
        assert len(sub_queries) > 0

    def test_heuristic_decompose_short_query(self, strategy):
        """Test that short queries aren't decomposed."""
        context = EnhancementContext()
        query = "short query"
        sub_queries = strategy._heuristic_decompose(query, context)

        # Should not decompose short queries
        assert len(sub_queries) == 0

    @pytest.mark.asyncio
    async def test_enhance_with_llm(self, strategy):
        """Test decomposition with mocked LLM response."""
        strategy._call_llm = AsyncMock(
            return_value='["revenue query", "profit query", "growth query"]'
        )

        context = EnhancementContext(domain="financial")
        result = await strategy.enhance(
            "Compare Apple and Microsoft revenue, profit, and growth", context
        )

        assert len(result.sub_queries) == 3
        assert "revenue query" in result.sub_queries


# =============================================================================
# QueryEnhancementPipeline Tests
# =============================================================================


class TestQueryEnhancementPipeline:
    """Tests for QueryEnhancementPipeline."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return QueryEnhancementConfig(
            techniques=[EnhancementTechnique.ENTITY_EXPAND],
            enable_llm=False,  # Disable LLM for testing
        )

    @pytest.fixture
    def pipeline(self, config):
        """Create pipeline with test config."""
        return QueryEnhancementPipeline(config=config)

    @pytest.mark.asyncio
    async def test_enhance_basic(self, pipeline):
        """Test basic enhancement."""
        context = EnhancementContext(entity_metadata=[{"name": "Test Company", "ticker": "TEST"}])
        result = await pipeline.enhance("test query", context)

        assert result.original == "test query"
        assert isinstance(result, EnhancedQuery)

    @pytest.mark.asyncio
    async def test_enhance_caching(self, pipeline):
        """Test that results are cached."""
        context = EnhancementContext(entity_metadata=[{"name": "Test", "ticker": "TST"}])

        # First call
        result1 = await pipeline.enhance("test query", context)

        # Second call should hit cache
        result2 = await pipeline.enhance("test query", context)

        # Results should be equal
        assert result1.enhanced == result2.enhanced

    @pytest.mark.asyncio
    async def test_enhance_different_contexts_not_cached(self, pipeline):
        """Test that different contexts produce different cache keys."""
        context1 = EnhancementContext(entity_metadata=[{"name": "Apple", "ticker": "AAPL"}])
        context2 = EnhancementContext(entity_metadata=[{"name": "Google", "ticker": "GOOGL"}])

        result1 = await pipeline.enhance("compare revenue", context1)
        result2 = await pipeline.enhance("compare revenue", context2)

        # Different contexts should produce different results
        # (though query is same, entities differ)
        assert result1 != result2 or result1.metadata != result2.metadata

    @pytest.mark.asyncio
    async def test_fallback_to_expansion(self):
        """Test fallback to entity expansion when LLM unavailable."""
        config = QueryEnhancementConfig(
            techniques=[EnhancementTechnique.REWRITE],
            enable_llm=True,
            fallback_to_expansion=True,
        )
        pipeline = QueryEnhancementPipeline(config=config)

        # Mock LLM unavailability
        pipeline._llm_available = False

        context = EnhancementContext(entity_metadata=[{"name": "Test", "ticker": "TST"}])
        result = await pipeline.enhance("test query", context)

        # Should fall back to entity expansion
        assert result is not None

    def test_pipeline_repr(self):
        """Test pipeline string representation."""
        config = QueryEnhancementConfig(
            techniques=[
                EnhancementTechnique.REWRITE,
                EnhancementTechnique.ENTITY_EXPAND,
            ]
        )
        pipeline = QueryEnhancementPipeline(config=config)
        repr_str = repr(pipeline)

        assert "rewrite" in repr_str
        assert "entity_expand" in repr_str


# =============================================================================
# QueryEnhancementRegistry Tests
# =============================================================================


class TestQueryEnhancementRegistry:
    """Tests for QueryEnhancementRegistry."""

    @pytest.fixture
    def registry(self):
        """Create fresh registry."""
        return QueryEnhancementRegistry()

    def test_register_strategy(self, registry):
        """Test registering a strategy."""
        registry.register(EnhancementTechnique.ENTITY_EXPAND, EntityExpandStrategy)
        assert EnhancementTechnique.ENTITY_EXPAND in registry.list_techniques()

    def test_get_strategy(self, registry):
        """Test getting a registered strategy."""
        registry.register(EnhancementTechnique.ENTITY_EXPAND, EntityExpandStrategy)
        strategy = registry.get(EnhancementTechnique.ENTITY_EXPAND)

        assert strategy is not None
        assert isinstance(strategy, EntityExpandStrategy)

    def test_get_unregistered_strategy(self, registry):
        """Test getting an unregistered strategy."""
        strategy = registry.get(EnhancementTechnique.HYDE)
        assert strategy is None

    def test_list_techniques(self, registry):
        """Test listing registered techniques."""
        registry.register(EnhancementTechnique.REWRITE, RewriteStrategy)
        registry.register(EnhancementTechnique.ENTITY_EXPAND, EntityExpandStrategy)

        techniques = registry.list_techniques()
        assert EnhancementTechnique.REWRITE in techniques
        assert EnhancementTechnique.ENTITY_EXPAND in techniques

    def test_clear_registry(self, registry):
        """Test clearing the registry."""
        registry.register(EnhancementTechnique.REWRITE, RewriteStrategy)
        registry.clear()

        assert len(registry.list_techniques()) == 0

    def test_default_registry(self):
        """Test default registry has strategies."""
        default = get_default_registry()

        # Should have default strategies
        assert EnhancementTechnique.REWRITE in default.list_techniques()
        assert EnhancementTechnique.ENTITY_EXPAND in default.list_techniques()
        assert EnhancementTechnique.DECOMPOSITION in default.list_techniques()


# =============================================================================
# EnhancementMetrics Tests
# =============================================================================


class TestEnhancementMetrics:
    """Tests for EnhancementMetrics dataclass."""

    def test_create_metrics(self):
        """Test creating enhancement metrics."""
        metrics = EnhancementMetrics(
            technique=EnhancementTechnique.REWRITE,
            latency_ms=150.0,
            llm_calls=1,
            confidence=0.9,
        )
        assert metrics.technique == EnhancementTechnique.REWRITE
        assert metrics.latency_ms == 150.0
        assert metrics.llm_calls == 1

    def test_to_dict(self):
        """Test converting metrics to dictionary."""
        metrics = EnhancementMetrics(
            technique=EnhancementTechnique.ENTITY_EXPAND,
            latency_ms=50.0,
            cache_hit=True,
        )
        data = metrics.to_dict()

        assert data["technique"] == "entity_expand"
        assert data["latency_ms"] == 50.0
        assert data["cache_hit"] is True


# =============================================================================
# EnhancementResult Tests
# =============================================================================


class TestEnhancementResult:
    """Tests for EnhancementResult dataclass."""

    def test_from_query(self):
        """Test creating result from enhanced query."""
        query = EnhancedQuery(
            original="test",
            enhanced="enhanced test",
            technique=EnhancementTechnique.REWRITE,
            confidence=0.9,
            variants=["v1", "v2"],
            sub_queries=["sq1"],
        )
        result = EnhancementResult.from_query(query, latency_ms=100.0, llm_calls=1, cache_hit=False)

        assert result.success is True
        assert result.query == query
        assert result.metrics.latency_ms == 100.0
        assert result.metrics.variant_count == 2
        assert result.metrics.sub_query_count == 1

    def test_from_error(self):
        """Test creating error result."""
        result = EnhancementResult.from_error(
            original_query="test",
            error="LLM unavailable",
        )

        assert result.success is False
        assert result.error == "LLM unavailable"
        assert result.query.enhanced == "test"  # Falls back to original
        assert result.metrics.confidence == 0.0


# =============================================================================
# DomainConfig Tests
# =============================================================================


class TestDomainConfig:
    """Tests for DomainConfig and domain configurations."""

    def test_financial_domain(self):
        """Test financial domain configuration."""
        assert FINANCIAL_DOMAIN.name == "financial"
        assert EnhancementTechnique.DECOMPOSITION in FINANCIAL_DOMAIN.default_techniques
        assert "eps" in FINANCIAL_DOMAIN.term_expansions
        assert "earnings" in FINANCIAL_DOMAIN.term_expansions["eps"]

    def test_code_domain(self):
        """Test code domain configuration."""
        assert CODE_DOMAIN.name == "code"
        assert "fn" in CODE_DOMAIN.term_expansions
        assert "function" in CODE_DOMAIN.term_expansions["fn"]

    def test_research_domain(self):
        """Test research domain configuration."""
        assert RESEARCH_DOMAIN.name == "research"
        assert "ml" in RESEARCH_DOMAIN.term_expansions

    def test_general_domain(self):
        """Test general domain configuration."""
        assert GENERAL_DOMAIN.name == "general"
        # General domain has minimal config
        assert len(GENERAL_DOMAIN.term_expansions) == 0

    def test_get_domain_config(self):
        """Test getting domain config by name."""
        config = get_domain_config("financial")
        assert config == FINANCIAL_DOMAIN

        config = get_domain_config("code")
        assert config == CODE_DOMAIN

    def test_get_domain_config_unknown(self):
        """Test getting config for unknown domain."""
        config = get_domain_config("unknown_domain")
        assert config == GENERAL_DOMAIN

    def test_domain_config_default_techniques(self):
        """Test that DomainConfig sets default techniques."""
        config = DomainConfig(name="test")
        # Should have default techniques after __post_init__
        assert len(config.default_techniques) > 0


# =============================================================================
# Cache Behavior Tests
# =============================================================================


class TestCacheBehavior:
    """Tests for caching behavior."""

    @pytest.mark.asyncio
    async def test_cache_ttl_expiration(self):
        """Test that cache entries expire after TTL."""
        config = QueryEnhancementConfig(
            techniques=[EnhancementTechnique.ENTITY_EXPAND],
            enable_llm=False,
            cache_ttl_seconds=1,  # Very short TTL for testing
        )
        pipeline = QueryEnhancementPipeline(config=config)

        context = EnhancementContext(entity_metadata=[{"name": "Test", "ticker": "TST"}])

        # First call
        result1 = await pipeline.enhance("test", context)

        # Wait for TTL to expire
        await asyncio.sleep(1.5)

        # Second call should not use cache
        result2 = await pipeline.enhance("test", context)

        # Both should succeed
        assert result1.enhanced is not None
        assert result2.enhanced is not None

    @pytest.mark.asyncio
    async def test_cache_different_queries(self):
        """Test that different queries aren't confused in cache."""
        config = QueryEnhancementConfig(
            techniques=[EnhancementTechnique.ENTITY_EXPAND],
            enable_llm=False,
        )
        pipeline = QueryEnhancementPipeline(config=config)

        context = EnhancementContext(entity_metadata=[{"name": "Test", "ticker": "TST"}])

        result1 = await pipeline.enhance("query one", context)
        result2 = await pipeline.enhance("query two", context)

        # Should be different
        assert result1.original != result2.original


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for the query enhancement system."""

    @pytest.mark.asyncio
    async def test_full_enhancement_flow(self):
        """Test complete enhancement flow."""
        config = QueryEnhancementConfig(
            techniques=[
                EnhancementTechnique.ENTITY_EXPAND,
            ],
            enable_llm=False,
        )
        pipeline = QueryEnhancementPipeline(config=config)

        context = EnhancementContext(
            domain="financial",
            entity_metadata=[
                {"name": "Apple Inc", "ticker": "AAPL", "sector": "Technology"},
                {"name": "Microsoft Corp", "ticker": "MSFT", "sector": "Technology"},
            ],
        )

        result = await pipeline.enhance(
            "Compare revenue growth",
            context,
        )

        # Should have enhanced query
        assert result.original == "Compare revenue growth"
        assert len(result.enhanced) > len(result.original)
        assert result.technique == EnhancementTechnique.ENTITY_EXPAND

    @pytest.mark.asyncio
    async def test_import_from_module(self):
        """Test that all imports work correctly."""
        # This tests the module structure
        from victor.core.query_enhancement import (
            QueryEnhancementPipeline,
            EnhancementContext,
            EnhancementTechnique,
            EnhancedQuery,
            QueryEnhancementConfig,
            EnhancementMetrics,
            EnhancementResult,
            EntityExpandStrategy,
            RewriteStrategy,
            DecompositionStrategy,
            get_default_registry,
            get_domain_config,
        )

        # All imports should work
        assert QueryEnhancementPipeline is not None
        assert EnhancementContext is not None
        assert EnhancementTechnique is not None
        assert EnhancedQuery is not None

    def test_strategy_protocol_compliance(self):
        """Test that strategies comply with protocol."""
        from victor.core.query_enhancement import IQueryEnhancementStrategy

        # EntityExpandStrategy should comply
        strategy = EntityExpandStrategy()
        assert hasattr(strategy, "name")
        assert hasattr(strategy, "technique")
        assert hasattr(strategy, "requires_llm")
        assert hasattr(strategy, "enhance")

        # Check interface methods exist
        assert callable(getattr(strategy, "enhance", None))
