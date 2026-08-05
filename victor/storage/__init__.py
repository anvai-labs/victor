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

"""Unified Storage Module for Victor.

This module consolidates all storage-related functionality into a single namespace:

- **cache**: Tiered caching with memory (L1) and disk (L2) layers
- **memory**: Multi-tier entity memory with unified coordinator
- **checkpoints**: Time-travel debugging via conversation state checkpoints
- **graph**: Generic graph storage with multiple backends
- **vector_stores**: Vector storage with multiple embedding providers
- **embeddings**: Shared embedding infrastructure and services
- **state**: Standalone state machine package

This restructuring provides:
1. Single import point for all storage functionality
2. Clear separation of concerns
3. Easier dependency management
4. Consistent API patterns across storage types

Example Usage:
    # Cache system
    from victor.storage.cache import get_cache_manager, CacheConfig

    # Memory system
    from victor.storage.memory import EntityMemory, get_memory_coordinator

    # Checkpoints
    from victor.storage.checkpoints import ConversationCheckpointManager, SQLiteCheckpointBackend

    # Graph storage
    from victor.storage.graph import create_graph_store, GraphNode

    # Vector stores
    from victor.storage.vector_stores import EmbeddingRegistry, EmbeddingConfig

    # Embeddings
    from victor.storage.embeddings import EmbeddingService

    # State machine
    from victor.storage.state import StateMachine, StateConfig

Canonical Import Paths:
    from victor.storage.cache import ...
    from victor.storage.memory import ...
    from victor.storage.checkpoints import ...
    from victor.storage.graph import ...
    from victor.storage.vector_stores import ...
    from victor.storage.embeddings import ...
    from victor.storage.state import ...
"""

# Submodule lazy loading for performance
# Users should import from specific submodules rather than from this __init__

__all__ = [
    "cache",
    "memory",
    "checkpoints",
    "graph",
    "vector_stores",
    "embeddings",
    "state",
]

__version__ = "0.5.1"
