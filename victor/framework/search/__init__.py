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

"""Framework-level search utilities.

This module provides generic search algorithms and utilities that can be
used across all verticals. These are domain-agnostic search implementations.

Provides:
- HybridSearchEngine: RRF-based fusion of semantic and keyword search
- HybridSearchResult: Result dataclass for hybrid search
- QueryExpander: Query expansion with synonyms for better recall
- QueryExpansionConfig: Configuration for query expansion

Usage:
    from victor.framework.search import HybridSearchEngine, create_hybrid_search_engine

    engine = create_hybrid_search_engine(semantic_weight=0.6, keyword_weight=0.4)
    results = engine.combine_results(semantic_results, keyword_results)

    from victor.framework.search import QueryExpander, QueryExpansionConfig

    config = QueryExpansionConfig(expansions={"error": ["exception", "failure"]})
    expander = QueryExpander(config)
    result = expander.expand("fix error handling")
"""

from victor.framework.search.hybrid import (
    HybridSearchEngine,
    HybridSearchResult,
    create_hybrid_search_engine,
)
from victor.framework.search.code_search_quality import (
    CodeSearchQualityConfig,
    enrich_code_search_results,
    infer_code_search_chunking_strategy,
    rerank_code_search_results,
)
from victor.framework.search.codebase_embedding_bridge import (
    CODEBASE_INDEX_MANIFEST_NAME,
    DEFAULT_CODEBASE_CHUNKING_STRATEGY,
    DEFAULT_CODEBASE_CHUNK_OVERLAP,
    DEFAULT_CODEBASE_CHUNK_SIZE,
    STRUCTURAL_CODEBASE_VECTOR_STORE,
    build_codebase_index_manifest,
    enable_structural_codebase_embeddings,
    has_compatible_codebase_index_manifest,
    has_persisted_codebase_index_data,
    write_codebase_index_manifest,
)
from victor.framework.search.query_expansion import (
    QueryExpander,
    QueryExpansionConfig,
    ExpandedQuery,
    QueryExpanderProtocol,
    create_query_expander,
)
from victor.framework.search.skeleton import extract_skeleton

__all__ = [
    # Hybrid search
    "HybridSearchEngine",
    "HybridSearchResult",
    "create_hybrid_search_engine",
    "CodeSearchQualityConfig",
    "enrich_code_search_results",
    "infer_code_search_chunking_strategy",
    "rerank_code_search_results",
    "CODEBASE_INDEX_MANIFEST_NAME",
    "DEFAULT_CODEBASE_CHUNKING_STRATEGY",
    "DEFAULT_CODEBASE_CHUNK_OVERLAP",
    "DEFAULT_CODEBASE_CHUNK_SIZE",
    "STRUCTURAL_CODEBASE_VECTOR_STORE",
    "build_codebase_index_manifest",
    "enable_structural_codebase_embeddings",
    "has_compatible_codebase_index_manifest",
    "has_persisted_codebase_index_data",
    "write_codebase_index_manifest",
    # Query expansion
    "QueryExpander",
    "QueryExpansionConfig",
    "ExpandedQuery",
    "QueryExpanderProtocol",
    "create_query_expander",
    # Skeleton extraction
    "extract_skeleton",
]
