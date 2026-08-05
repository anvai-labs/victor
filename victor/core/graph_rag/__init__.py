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

"""Graph RAG Pipeline - Three-stage framework for graph-enhanced retrieval.

This module implements a Graph RAG (Retrieval-Augmented Generation) pipeline
for code intelligence:

1. **G-Indexing**: Build graph with embeddings for code symbols
2. **G-Retrieval**: Multi-hop traversal for context retrieval
3. **G-Generation**: Construct prompts with graph context

Based on research from GraphRAG and GraphCodeAgent.

Usage:
    from victor.core.graph_rag import GraphIndexingPipeline, MultiHopRetriever

    # Index a repository
    indexing_pipeline = GraphIndexingPipeline(graph_store, vector_store)
    stats = await indexing_pipeline.index_repository(Path("/path/to/repo"))

    # Retrieve relevant context
    retriever = MultiHopRetriever(graph_store, vector_store)
    subgraphs = await retriever.retrieve("How to handle authentication?",
                                         config=RetrievalConfig(max_hops=2))

    # Access query cache (PH4-005)
    from victor.core.graph_rag import get_graph_query_cache
    cache = get_graph_query_cache()
    stats = cache.get_stats()

    # Translate natural language to graph queries (PH3-001 to PH3-006)
    from victor.core.graph_rag import translate_query, list_templates
    result = await translate_query("Find functions that call parse_json", graph_store)
"""

from victor.core.graph_rag.config import (
    GraphIndexConfig,
    RetrievalConfig,
    PromptConfig,
    SubgraphConfig,
)
from victor.core.graph_rag.indexing import GraphIndexingPipeline, GraphIndexStats
from victor.core.graph_rag.retrieval import MultiHopRetriever
from victor.core.graph_rag.generation import GraphAwarePromptBuilder
from victor.core.graph_rag.query_cache import (
    GraphQueryCache,
    GraphQueryCacheConfig,
    get_graph_query_cache,
    configure_graph_query_cache,
    reset_graph_query_cache,
)

# Requirement Graph (PH5-001 to PH5-004)
try:
    from victor.core.graph_rag.requirement_graph import (
        RequirementType,
        RequirementPriority,
        RequirementStatus,
        RequirementMapping,
        RequirementSource,
        RequirementGraphBuilder,
        RequirementSimilarity,
        RequirementSimilarityCalculator,
    )

    _requirement_graph_available = True
except ImportError:
    _requirement_graph_available = False

# Query Translation (PH3-001 to PH3-006)
try:
    from victor.core.graph_rag.query_translation import (
        QueryType,
        MatchStrategy,
        QueryTemplate,
        TemplateRegistry,
        get_template_registry,
        QueryTranslator,
        TemplateBasedTranslator,
        LLMBasedTranslator,
        TranslationResult,
        translate_query,
        register_template,
        list_templates,
    )

    _query_translation_available = True
except ImportError:
    _query_translation_available = False

__all__ = [
    # Configuration
    "GraphIndexConfig",
    "RetrievalConfig",
    "PromptConfig",
    "SubgraphConfig",
    # Indexing
    "GraphIndexingPipeline",
    "GraphIndexStats",
    # Retrieval
    "MultiHopRetriever",
    # Generation
    "GraphAwarePromptBuilder",
    # Query Cache (PH4-005)
    "GraphQueryCache",
    "GraphQueryCacheConfig",
    "get_graph_query_cache",
    "configure_graph_query_cache",
    "reset_graph_query_cache",
    # Requirement Graph (PH5-001 to PH5-004)
    "RequirementType",
    "RequirementPriority",
    "RequirementStatus",
    "RequirementMapping",
    "RequirementSource",
    "RequirementGraphBuilder",
    "RequirementSimilarity",
    "RequirementSimilarityCalculator",
    # Query Translation (PH3-001 to PH3-006)
    "QueryType",
    "MatchStrategy",
    "QueryTemplate",
    "TemplateRegistry",
    "get_template_registry",
    "QueryTranslator",
    "TemplateBasedTranslator",
    "LLMBasedTranslator",
    "TranslationResult",
    "translate_query",
    "register_template",
    "list_templates",
]
