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

"""Unit tests for Research vertical handlers."""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock

import pytest


class MockToolResult:
    """Mock tool result for testing."""

    def __init__(self, success: bool = True, output: Any = None, error: str = None):
        self.success = success
        self.output = output
        self.error = error


class MockComputeNode:
    """Mock compute node for testing."""

    def __init__(
        self,
        node_id: str = "test_node",
        input_mapping: Dict[str, Any] = None,
        output_key: str = None,
    ):
        self.id = node_id
        self.input_mapping = input_mapping or {}
        self.output_key = output_key


class MockWorkflowContext:
    """Mock workflow context for testing."""

    def __init__(self, data: Dict[str, Any] = None):
        self._data = data or {}

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value


class TestWebScraperHandler:
    """Tests for WebScraperHandler."""

    @pytest.fixture
    def handler(self):
        from victor_research.handlers import WebScraperHandler

        return WebScraperHandler()

    @pytest.fixture
    def mock_registry(self):
        registry = MagicMock()
        registry.execute = AsyncMock(
            return_value=MockToolResult(
                success=True,
                output={"title": "Test Page", "content": "Test content"},
            )
        )
        return registry

    @pytest.mark.asyncio
    async def test_successful_scrape(self, handler, mock_registry):
        """Test successful web scraping."""
        node = MockComputeNode(
            input_mapping={
                "url": "https://example.com",
                "selectors": {"title": "h1"},
            },
            output_key="scraped_data",
        )
        context = MockWorkflowContext()

        result = await handler(node, context, mock_registry)

        assert result.status.value == "completed"
        assert result.output["success"] is True
        assert context.get("scraped_data") is not None

    @pytest.mark.asyncio
    async def test_scrape_with_context_url(self, handler, mock_registry):
        """Test URL resolution from context."""
        node = MockComputeNode(
            input_mapping={
                "url": "$ctx.target_url",
                "selectors": {},
            },
        )
        context = MockWorkflowContext({"target_url": "https://example.com"})

        result = await handler(node, context, mock_registry)

        assert result.status.value == "completed"
        mock_registry.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_scrape_failure(self, handler):
        """Test handling of scrape failure."""
        mock_registry = MagicMock()
        mock_registry.execute = AsyncMock(
            return_value=MockToolResult(success=False, error="Connection timeout")
        )

        node = MockComputeNode(input_mapping={"url": "https://example.com"})
        context = MockWorkflowContext()

        result = await handler(node, context, mock_registry)

        assert result.status.value == "failed"
        assert result.output["error"] == "Connection timeout"


class TestCitationFormatterHandler:
    """Tests for CitationFormatterHandler."""

    @pytest.fixture
    def handler(self):
        from victor_research.handlers import CitationFormatterHandler

        return CitationFormatterHandler()

    @pytest.fixture
    def mock_registry(self):
        return MagicMock()

    @pytest.mark.asyncio
    async def test_format_apa_citation(self, handler, mock_registry):
        """Test APA citation formatting."""
        node = MockComputeNode(
            input_mapping={
                "references": "refs",
                "style": "apa",
            },
            output_key="citations",
        )
        context = MockWorkflowContext(
            {
                "refs": [
                    {
                        "authors": ["Smith, J.", "Doe, A."],
                        "year": "2024",
                        "title": "Test Paper",
                        "source": "Journal of Testing",
                    }
                ]
            }
        )

        result = await handler(node, context, mock_registry)

        assert result.status.value == "completed"
        citations = context.get("citations")
        assert citations["style"] == "apa"
        assert citations["count"] == 1
        assert "Smith, J." in citations["citations"][0]

    @pytest.mark.asyncio
    async def test_format_mla_citation(self, handler, mock_registry):
        """Test MLA citation formatting."""
        node = MockComputeNode(
            input_mapping={
                "references": "refs",
                "style": "mla",
            },
        )
        context = MockWorkflowContext(
            {
                "refs": {
                    "authors": ["Smith, J."],
                    "year": "2024",
                    "title": "Test Paper",
                    "source": "Journal",
                }
            }
        )

        result = await handler(node, context, mock_registry)

        assert result.status.value == "completed"

    @pytest.mark.asyncio
    async def test_format_chicago_citation(self, handler, mock_registry):
        """Test Chicago citation formatting."""
        node = MockComputeNode(
            input_mapping={"references": "refs", "style": "chicago"},
        )
        context = MockWorkflowContext(
            {
                "refs": [
                    {
                        "authors": ["Smith, J."],
                        "year": "2024",
                        "title": "Test Paper",
                        "source": "Publisher",
                    }
                ]
            }
        )

        result = await handler(node, context, mock_registry)

        assert result.status.value == "completed"

    @pytest.mark.asyncio
    async def test_empty_references(self, handler, mock_registry):
        """Test handling of empty references."""
        node = MockComputeNode(input_mapping={"style": "apa"})
        context = MockWorkflowContext()

        result = await handler(node, context, mock_registry)

        assert result.status.value == "completed"
        assert result.output["count"] == 0


class TestHandlerRegistration:
    """Tests for handler registration."""

    def test_handlers_dict_exists(self):
        """Test HANDLERS dict is defined."""
        from victor_research.handlers import HANDLERS

        assert "web_scraper" in HANDLERS
        assert "citation_formatter" in HANDLERS

    def test_register_handlers(self):
        """Test handler registration function."""
        from victor_research.handlers import register_handlers

        # Should not raise
        register_handlers()


class TestEscapeHatches:
    """Tests for Research escape hatch conditions."""

    def test_source_coverage_sufficient(self):
        """Test sufficient source coverage."""
        from victor_research.escape_hatches import source_coverage_check

        ctx = {
            "sources": [1, 2, 3, 4, 5, 6],
            "coverage_score": 0.85,
            "min_sources": 5,
        }
        assert source_coverage_check(ctx) == "sufficient"

    def test_source_coverage_marginal(self):
        """Test marginal source coverage."""
        from victor_research.escape_hatches import source_coverage_check

        ctx = {
            "sources": [1, 2, 3],
            "coverage_score": 0.65,
            "min_sources": 5,
        }
        assert source_coverage_check(ctx) == "marginal"

    def test_source_coverage_needs_more(self):
        """Test needs more sources."""
        from victor_research.escape_hatches import source_coverage_check

        ctx = {
            "sources": [1, 2],
            "coverage_score": 0.4,
            "min_sources": 5,
        }
        assert source_coverage_check(ctx) == "needs_more"

    def test_should_search_more_proceed_on_coverage(self):
        """Test proceed when coverage is good."""
        from victor_research.escape_hatches import should_search_more

        ctx = {
            "coverage_score": 0.9,
            "search_iterations": 1,
        }
        assert should_search_more(ctx) == "proceed"

    def test_should_search_more_proceed_on_max_iterations(self):
        """Test proceed after max iterations."""
        from victor_research.escape_hatches import should_search_more

        ctx = {
            "coverage_score": 0.5,
            "search_iterations": 3,
            "max_iterations": 3,
        }
        assert should_search_more(ctx) == "proceed"

    def test_should_search_more_search_with_gaps(self):
        """Test search more when significant gaps exist."""
        from victor_research.escape_hatches import should_search_more

        ctx = {
            "coverage_score": 0.4,
            "search_iterations": 1,
            "gaps": ["gap1", "gap2", "gap3"],
        }
        assert should_search_more(ctx) == "search_more"

    def test_source_credibility_high(self):
        """Test high credibility sources."""
        from victor_research.escape_hatches import source_credibility_check

        ctx = {
            "validated_sources": [
                {"url": "a", "credibility": 0.9},
                {"url": "b", "credibility": 0.85},
            ],
        }
        assert source_credibility_check(ctx) == "high_credibility"

    def test_source_credibility_acceptable(self):
        """Test acceptable credibility sources."""
        from victor_research.escape_hatches import source_credibility_check

        ctx = {
            "validated_sources": [
                {"url": "a", "credibility": 0.6},
                {"url": "b", "credibility": 0.55},
            ],
        }
        assert source_credibility_check(ctx) == "acceptable"

    def test_source_credibility_low(self):
        """Test low credibility sources."""
        from victor_research.escape_hatches import source_credibility_check

        ctx = {
            "validated_sources": [
                {"url": "a", "credibility": 0.3},
                {"url": "b", "credibility": 0.2},
            ],
        }
        assert source_credibility_check(ctx) == "low_credibility"

    def test_source_credibility_empty(self):
        """Test empty sources returns low credibility."""
        from victor_research.escape_hatches import source_credibility_check

        ctx = {"validated_sources": []}
        assert source_credibility_check(ctx) == "low_credibility"

    def test_fact_verdict_true(self):
        """Test fact verdict true."""
        from victor_research.escape_hatches import fact_verdict

        ctx = {
            "supporting_evidence": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "refuting_evidence": [1],
            "confidence": 0.8,
        }
        assert fact_verdict(ctx) == "true"

    def test_fact_verdict_false(self):
        """Test fact verdict false."""
        from victor_research.escape_hatches import fact_verdict

        ctx = {
            "supporting_evidence": [],
            "refuting_evidence": [1, 2, 3, 4, 5],
            "confidence": 0.8,
        }
        assert fact_verdict(ctx) == "false"

    def test_fact_verdict_mixed(self):
        """Test fact verdict mixed."""
        from victor_research.escape_hatches import fact_verdict

        ctx = {
            "supporting_evidence": [1, 2],
            "refuting_evidence": [1, 2],
            "confidence": 0.8,
        }
        assert fact_verdict(ctx) == "mixed"

    def test_fact_verdict_unverifiable(self):
        """Test fact verdict unverifiable with no evidence."""
        from victor_research.escape_hatches import fact_verdict

        ctx = {
            "supporting_evidence": [],
            "refuting_evidence": [],
        }
        assert fact_verdict(ctx) == "unverifiable"

    def test_fact_verdict_low_confidence(self):
        """Test fact verdict unverifiable with low confidence."""
        from victor_research.escape_hatches import fact_verdict

        ctx = {
            "supporting_evidence": [1, 2, 3],
            "refuting_evidence": [],
            "confidence": 0.2,
        }
        assert fact_verdict(ctx) == "unverifiable"

    def test_literature_relevance_highly_relevant(self):
        """Test highly relevant paper."""
        from victor_research.escape_hatches import literature_relevance

        ctx = {
            "paper": {"relevance_score": 0.85, "citation_count": 100},
        }
        assert literature_relevance(ctx) == "highly_relevant"

    def test_literature_relevance_relevant(self):
        """Test relevant paper."""
        from victor_research.escape_hatches import literature_relevance

        ctx = {
            "paper": {"relevance_score": 0.65, "citation_count": 10},
        }
        assert literature_relevance(ctx) == "relevant"

    def test_literature_relevance_irrelevant(self):
        """Test irrelevant paper."""
        from victor_research.escape_hatches import literature_relevance

        ctx = {
            "paper": {"relevance_score": 0.2, "citation_count": 5},
        }
        assert literature_relevance(ctx) == "irrelevant"

    def test_competitive_threat_high_direct(self):
        """Test high threat from direct competitor."""
        from victor_research.escape_hatches import competitive_threat_level

        ctx = {
            "competitor": {"is_direct_competitor": True, "market_share": 0.25},
            "market_overlap": 0.8,
            "growth_rate": 0.1,
        }
        assert competitive_threat_level(ctx) == "high"

    def test_competitive_threat_emerging(self):
        """Test emerging threat from high growth."""
        from victor_research.escape_hatches import competitive_threat_level

        ctx = {
            "competitor": {"is_direct_competitor": False, "market_share": 0.05},
            "market_overlap": 0.3,
            "growth_rate": 0.6,
        }
        assert competitive_threat_level(ctx) == "emerging"

    def test_competitive_threat_low(self):
        """Test low threat level."""
        from victor_research.escape_hatches import competitive_threat_level

        ctx = {
            "competitor": {"is_direct_competitor": False, "market_share": 0.03},
            "market_overlap": 0.2,
            "growth_rate": 0.05,
        }
        assert competitive_threat_level(ctx) == "low"

    def test_merge_search_results(self):
        """Test merge search results transform."""
        from victor_research.escape_hatches import merge_search_results

        ctx = {
            "web_results": [
                {"url": "https://a.com", "title": "A"},
                {"url": "https://b.com", "title": "B"},
            ],
            "academic_results": [
                {"url": "https://c.com", "title": "C"},
            ],
            "code_results": [
                {"url": "https://github.com/d", "title": "D"},
            ],
        }
        result = merge_search_results(ctx)
        assert result["source_count"] == 4
        assert result["by_type"]["web"] == 2
        assert result["by_type"]["academic"] == 1
        assert result["by_type"]["code"] == 1

    def test_merge_search_results_deduplicates(self):
        """Test merge search results deduplicates by URL."""
        from victor_research.escape_hatches import merge_search_results

        ctx = {
            "web_results": [
                {"url": "https://a.com", "title": "A"},
            ],
            "academic_results": [
                {"url": "https://a.com", "title": "A duplicate"},
            ],
        }
        result = merge_search_results(ctx)
        assert result["source_count"] == 1

    def test_format_bibliography(self):
        """Test format bibliography transform."""
        from victor_research.escape_hatches import format_bibliography

        ctx = {
            "validated_sources": [
                {
                    "title": "Paper A",
                    "authors": ["Smith"],
                    "year": "2024",
                    "url": "https://a.com",
                },
                {
                    "title": "Paper B",
                    "authors": ["Jones"],
                    "year": "2023",
                    "url": "https://b.com",
                },
            ],
            "citation_style": "apa",
        }
        result = format_bibliography(ctx)
        assert result["count"] == 2
        assert result["style"] == "apa"
        assert len(result["entries"]) == 2
