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

"""Core Query Enhancement Module.

Provides LLM-based query enhancement as a reusable core capability.
Promoted from RAG-specific to core reusable system.

Usage:
    from victor.core.query_enhancement import (
        QueryEnhancementPipeline,
        QueryEnhancementConfig,
        EnhancementContext,
        EnhancementTechnique,
    )

    # Create pipeline with config
    config = QueryEnhancementConfig(
        techniques=[EnhancementTechnique.REWRITE, EnhancementTechnique.ENTITY_EXPAND],
        enable_llm=True,
    )
    pipeline = QueryEnhancementPipeline(config)

    # Enhance a query
    result = await pipeline.enhance(
        query="What is AAPL revenue?",
        context=EnhancementContext(domain="financial"),
    )
    print(result.enhanced)  # "What is Apple Inc's total revenue and net sales?"

Strategies:
    - REWRITE: Normalize and expand queries using LLM
    - DECOMPOSITION: Break complex queries into sub-queries
    - ENTITY_EXPAND: Fast entity-based expansion (no LLM)

Types:
    - EnhancementTechnique: Enum of available techniques
    - EnhancementContext: Context for enhancement operations
    - EnhancedQuery: Result of enhancement operations
    - QueryEnhancementConfig: Pipeline configuration
    - EnhancementMetrics: Performance metrics for RL learning
    - EnhancementResult: Complete result with metrics
    - DomainConfig: Domain-specific configuration
"""

# Re-export from protocols for convenience
from victor.protocols.query_enhancement import (
    IQueryEnhancementStrategy,
    IQueryEnhancementPipeline,
    EnhancementTechnique,
    EnhancementContext,
    EnhancedQuery,
    QueryEnhancementConfig,
)

# Additional types from types module
from victor.core.query_enhancement.types import (
    EnhancementMetrics,
    EnhancementResult,
    DomainConfig,
    FINANCIAL_DOMAIN,
    CODE_DOMAIN,
    RESEARCH_DOMAIN,
    GENERAL_DOMAIN,
    DOMAIN_CONFIGS,
    get_domain_config,
)

from victor.core.query_enhancement.pipeline import QueryEnhancementPipeline
from victor.core.query_enhancement.registry import (
    QueryEnhancementRegistry,
    get_default_registry,
)

# Import strategies for explicit access
from victor.core.query_enhancement.strategies import (
    BaseQueryEnhancementStrategy,
    RewriteStrategy,
    DecompositionStrategy,
    EntityExpandStrategy,
)

__all__ = [
    # Protocols and types
    "IQueryEnhancementStrategy",
    "IQueryEnhancementPipeline",
    "EnhancementTechnique",
    "EnhancementContext",
    "EnhancedQuery",
    "QueryEnhancementConfig",
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
    # Pipeline
    "QueryEnhancementPipeline",
    # Registry
    "QueryEnhancementRegistry",
    "get_default_registry",
    # Strategies
    "BaseQueryEnhancementStrategy",
    "RewriteStrategy",
    "DecompositionStrategy",
    "EntityExpandStrategy",
]
