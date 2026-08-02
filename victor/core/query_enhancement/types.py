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

"""Query Enhancement Types.

This module provides the core data types for query enhancement:
- EnhancementTechnique: Enum of available techniques
- EnhancementContext: Context for enhancement operations
- EnhancedQuery: Result of enhancement operations
- QueryEnhancementConfig: Pipeline configuration

These types are re-exported from the protocol module for convenience
and to provide a stable API surface for the core module.

Example:
    from victor.core.query_enhancement.types import (
        EnhancementTechnique,
        EnhancementContext,
        EnhancedQuery,
        QueryEnhancementConfig,
    )

    # Create context
    context = EnhancementContext(
        domain="financial",
        entity_metadata=[{"name": "Apple Inc", "ticker": "AAPL"}],
        max_variants=3,
    )

    # Create config
    config = QueryEnhancementConfig(
        techniques=[EnhancementTechnique.REWRITE, EnhancementTechnique.ENTITY_EXPAND],
        enable_llm=True,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Re-export from protocols for stable API
from victor.protocols.query_enhancement import (
    EnhancementTechnique,
    EnhancementContext,
    EnhancedQuery,
    QueryEnhancementConfig,
    IQueryEnhancementStrategy,
    IQueryEnhancementPipeline,
)


@dataclass
class EnhancementMetrics:
    """Metrics collected during query enhancement.

    Tracks performance and quality metrics for RL learning.

    Attributes:
        technique: Enhancement technique used
        latency_ms: Time taken for enhancement
        llm_calls: Number of LLM calls made
        cache_hit: Whether result was cached
        confidence: Confidence score of enhancement
        token_count: Approximate token count of enhanced query
        sub_query_count: Number of sub-queries generated
        variant_count: Number of variants generated
    """

    technique: EnhancementTechnique
    latency_ms: float = 0.0
    llm_calls: int = 0
    cache_hit: bool = False
    confidence: float = 1.0
    token_count: int = 0
    sub_query_count: int = 0
    variant_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/analytics."""
        return {
            "technique": self.technique.value,
            "latency_ms": self.latency_ms,
            "llm_calls": self.llm_calls,
            "cache_hit": self.cache_hit,
            "confidence": self.confidence,
            "token_count": self.token_count,
            "sub_query_count": self.sub_query_count,
            "variant_count": self.variant_count,
        }


@dataclass
class EnhancementResult:
    """Complete result of query enhancement including metrics.

    Combines the enhanced query with performance metrics
    for downstream analysis and RL feedback.

    Attributes:
        query: The enhanced query result
        metrics: Performance and quality metrics
        success: Whether enhancement succeeded
        error: Error message if enhancement failed
    """

    query: EnhancedQuery
    metrics: EnhancementMetrics
    success: bool = True
    error: Optional[str] = None

    @classmethod
    def from_query(
        cls,
        query: EnhancedQuery,
        latency_ms: float = 0.0,
        llm_calls: int = 0,
        cache_hit: bool = False,
    ) -> "EnhancementResult":
        """Create result from an enhanced query.

        Args:
            query: Enhanced query result
            latency_ms: Time taken for enhancement
            llm_calls: Number of LLM calls
            cache_hit: Whether result was cached

        Returns:
            EnhancementResult with populated metrics
        """
        metrics = EnhancementMetrics(
            technique=query.technique,
            latency_ms=latency_ms,
            llm_calls=llm_calls,
            cache_hit=cache_hit,
            confidence=query.confidence,
            token_count=len(query.enhanced.split()),
            sub_query_count=len(query.sub_queries),
            variant_count=len(query.variants),
        )
        return cls(query=query, metrics=metrics, success=True)

    @classmethod
    def from_error(
        cls,
        original_query: str,
        error: str,
        technique: EnhancementTechnique = EnhancementTechnique.ENTITY_EXPAND,
    ) -> "EnhancementResult":
        """Create error result.

        Args:
            original_query: Original query that failed
            error: Error message
            technique: Technique that was attempted

        Returns:
            EnhancementResult with error details
        """
        fallback_query = EnhancedQuery(
            original=original_query,
            enhanced=original_query,
            technique=technique,
            confidence=0.0,
            metadata={"error": error},
        )
        metrics = EnhancementMetrics(
            technique=technique,
            confidence=0.0,
        )
        return cls(query=fallback_query, metrics=metrics, success=False, error=error)


@dataclass
class DomainConfig:
    """Domain-specific configuration for query enhancement.

    Allows per-domain customization of enhancement behavior.

    Attributes:
        name: Domain identifier
        default_techniques: Techniques to use for this domain
        entity_patterns: Regex patterns for entity detection
        term_expansions: Dictionary of abbreviation expansions
        prompt_hints: Additional hints for LLM prompts
    """

    name: str
    default_techniques: List[EnhancementTechnique] = field(default_factory=list)
    entity_patterns: List[str] = field(default_factory=list)
    term_expansions: Dict[str, str] = field(default_factory=dict)
    prompt_hints: str = ""

    def __post_init__(self):
        """Set default techniques if not provided."""
        if not self.default_techniques:
            self.default_techniques = [
                EnhancementTechnique.REWRITE,
                EnhancementTechnique.ENTITY_EXPAND,
            ]


# Pre-defined domain configurations
FINANCIAL_DOMAIN = DomainConfig(
    name="financial",
    default_techniques=[
        EnhancementTechnique.REWRITE,
        EnhancementTechnique.ENTITY_EXPAND,
        EnhancementTechnique.DECOMPOSITION,
    ],
    entity_patterns=[
        r"\b[A-Z]{2,5}\b",  # Stock tickers
        r"\b\d{4}-\d{2}-\d{2}\b",  # Dates
    ],
    term_expansions={
        "rev": "revenue",
        "eps": "earnings per share",
        "yoy": "year over year",
        "qoq": "quarter over quarter",
        "cfo": "cash flow from operations",
        "fcf": "free cash flow",
        "roce": "return on capital employed",
        "roe": "return on equity",
        "pe": "price to earnings ratio",
    },
    prompt_hints="Focus on SEC filing terminology and financial metrics.",
)

CODE_DOMAIN = DomainConfig(
    name="code",
    default_techniques=[
        EnhancementTechnique.REWRITE,
        EnhancementTechnique.ENTITY_EXPAND,
    ],
    entity_patterns=[
        r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b",  # CamelCase
        r"\b[a-z]+(?:_[a-z]+)+\b",  # snake_case
    ],
    term_expansions={
        "fn": "function",
        "impl": "implementation",
        "cfg": "config configuration",
        "db": "database",
        "msg": "message",
        "req": "request",
        "res": "response",
        "err": "error",
        "ctx": "context",
        "init": "initialize initialization",
    },
    prompt_hints="Include common code patterns and naming conventions.",
)

RESEARCH_DOMAIN = DomainConfig(
    name="research",
    default_techniques=[
        EnhancementTechnique.REWRITE,
        EnhancementTechnique.DECOMPOSITION,
    ],
    entity_patterns=[],
    term_expansions={
        "ml": "machine learning",
        "dl": "deep learning",
        "nlp": "natural language processing",
        "cv": "computer vision",
        "rl": "reinforcement learning",
        "sota": "state of the art",
    },
    prompt_hints="Use academic terminology and include related concepts.",
)

GENERAL_DOMAIN = DomainConfig(
    name="general",
    default_techniques=[
        EnhancementTechnique.REWRITE,
        EnhancementTechnique.ENTITY_EXPAND,
    ],
    entity_patterns=[],
    term_expansions={},
    prompt_hints="",
)

# Domain registry
DOMAIN_CONFIGS: Dict[str, DomainConfig] = {
    "financial": FINANCIAL_DOMAIN,
    "code": CODE_DOMAIN,
    "research": RESEARCH_DOMAIN,
    "general": GENERAL_DOMAIN,
}


def get_domain_config(domain: str) -> DomainConfig:
    """Get configuration for a domain.

    Args:
        domain: Domain identifier

    Returns:
        DomainConfig for the domain, or GENERAL_DOMAIN if not found
    """
    return DOMAIN_CONFIGS.get(domain, GENERAL_DOMAIN)


__all__ = [
    # Core types from protocols
    "EnhancementTechnique",
    "EnhancementContext",
    "EnhancedQuery",
    "QueryEnhancementConfig",
    "IQueryEnhancementStrategy",
    "IQueryEnhancementPipeline",
    # Additional types
    "EnhancementMetrics",
    "EnhancementResult",
    "DomainConfig",
    # Domain configurations
    "FINANCIAL_DOMAIN",
    "CODE_DOMAIN",
    "RESEARCH_DOMAIN",
    "GENERAL_DOMAIN",
    "DOMAIN_CONFIGS",
    "get_domain_config",
]
