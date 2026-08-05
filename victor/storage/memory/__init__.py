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

"""Memory systems for Victor.

Provides multi-tier memory architecture with unified coordinator:

1. Entity Memory (4-tier):
   - Short-term: Current session entities
   - Working memory: LRU cache of active entities
   - Long-term: Persistent SQLite + vector storage
   - Entity graph: Relationship tracking and traversal

2. Unified Memory Coordinator:
   - Federated search across all memory systems
   - Pluggable ranking strategies
   - Automatic deduplication
   - Protocol-based adapter pattern

Example - Entity Memory:
    from victor.storage.memory import EntityMemory, EntityType, CompositeExtractor

    # Create memory with extraction
    memory = EntityMemory()
    extractor = CompositeExtractor.create_default()

    # Extract and store entities from text
    result = await extractor.extract(text)
    for entity in result.entities:
        await memory.store(entity)

    # Query entities
    entities = await memory.search("auth", entity_types=[EntityType.CLASS])

    # Get related entities via graph
    related = await memory.get_related("ent_abc123")

Example - Unified Memory Coordinator:
    from victor.storage.memory import (
        get_memory_coordinator,
        EntityMemoryAdapter,
        ConversationMemoryAdapter,
        MemoryType,
    )

    # Register adapters with coordinator
    coordinator = get_memory_coordinator()
    coordinator.register_provider(EntityMemoryAdapter(entity_memory))
    coordinator.register_provider(ConversationMemoryAdapter(conv_store))

    # Federated search across all memory
    results = await coordinator.search_all("authentication")

    # Filter by memory type
    results = await coordinator.search_all(
        query="login",
        memory_types=[MemoryType.ENTITY, MemoryType.CONVERSATION],
    )
"""

from victor.storage.memory.entity_types import (
    EntityType,
    Entity,
    EntityRelation,
    RelationType,
    CODE_ENTITY_TYPES,
    PROJECT_ENTITY_TYPES,
    CONCEPT_ENTITY_TYPES,
    PEOPLE_ENTITY_TYPES,
)
from victor.storage.memory.entity_memory import (
    EntityMemory,
    EntityMemoryConfig,
    LRUCache,
)
from victor.storage.memory.entity_graph import (
    EntityGraph,
    GraphPath,
    GraphStats,
)
from victor.storage.memory.extractors import (
    EntityExtractor,
    ExtractionResult,
    CodeEntityExtractor,
    TextEntityExtractor,
    CompositeExtractor,
    TreeSitterEntityExtractor,
    TreeSitterFileExtractor,
    has_tree_sitter,
    create_extractor,
)
from victor.storage.memory.integration import (
    EntityMemoryIntegration,
    create_entity_integration,
)
from victor.storage.memory.unified import (
    MemoryType,
    MemoryResult,
    MemoryQuery,
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
from victor.storage.memory.adapters import (
    EntityMemoryAdapter,
    ConversationMemoryAdapter,
    GraphMemoryAdapter,
    ToolResultsMemoryAdapter,
    create_entity_adapter,
    create_conversation_adapter,
    create_graph_adapter,
    create_tool_results_adapter,
)

__all__ = [
    # Entity types
    "EntityType",
    "Entity",
    "EntityRelation",
    "RelationType",
    # Type categories
    "CODE_ENTITY_TYPES",
    "PROJECT_ENTITY_TYPES",
    "CONCEPT_ENTITY_TYPES",
    "PEOPLE_ENTITY_TYPES",
    # Memory
    "EntityMemory",
    "EntityMemoryConfig",
    "LRUCache",
    # Graph
    "EntityGraph",
    "GraphPath",
    "GraphStats",
    # Extractors
    "EntityExtractor",
    "ExtractionResult",
    "CodeEntityExtractor",
    "TextEntityExtractor",
    "CompositeExtractor",
    "TreeSitterEntityExtractor",
    "TreeSitterFileExtractor",
    "has_tree_sitter",
    "create_extractor",
    # Integration
    "EntityMemoryIntegration",
    "create_entity_integration",
    # Unified Memory Coordinator
    "MemoryType",
    "MemoryResult",
    "MemoryQuery",
    "MemoryProviderProtocol",
    "RankingStrategyProtocol",
    "RelevanceRankingStrategy",
    "RecencyRankingStrategy",
    "HybridRankingStrategy",
    "UnifiedMemoryCoordinator",
    "create_memory_coordinator",
    "get_memory_coordinator",
    "reset_memory_coordinator",
    # Memory Adapters
    "EntityMemoryAdapter",
    "ConversationMemoryAdapter",
    "GraphMemoryAdapter",
    "ToolResultsMemoryAdapter",
    "create_entity_adapter",
    "create_conversation_adapter",
    "create_graph_adapter",
    "create_tool_results_adapter",
]
