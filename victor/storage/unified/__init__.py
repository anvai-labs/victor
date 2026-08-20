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

"""Unified Symbol Store - SOLID-based facade for graph + semantic search.

This module provides a unified interface for code symbol storage and search,
combining graph-based relationships with semantic vector search.

Usage:
    from victor.storage.unified import create_symbol_store, SearchParams

    # Create default store (SQLite + LanceDB)
    store = create_symbol_store(repo_root=Path("."))
    await store.initialize()

    # Hybrid search
    results = await store.search(SearchParams(
        query="authentication handling",
        mode=SearchMode.HYBRID,
        limit=20,
    ))

    # Get symbol with relationships
    symbol = await store.get_symbol("symbol:victor/auth.py:authenticate")
    callers = await store.get_callers(symbol.unified_id)

Backend Options:
    # Default: SQLite + LanceDB (local, air-gapped compatible)
    store = create_symbol_store(backend="sqlite+lancedb")

    # Planned: ProximaDB single-engine backend
    store = create_symbol_store(backend="proximadb")
"""

from victor.storage.unified.protocol import (
    # Unified ID
    UnifiedId,
    SymbolType,
    # Data structures
    UnifiedSymbol,
    UnifiedEdge,
    # Search
    SearchMode,
    SearchParams,
    UnifiedSearchResult,
    # Protocols
    VectorStoreProtocol,
    GraphStoreProtocol,
    UnifiedSymbolStoreProtocol,
    # Re-exports
    GraphNode,
    GraphEdge,
)

from victor.storage.unified.factory import create_symbol_store

__all__ = [
    # Factory
    "create_symbol_store",
    # Unified ID
    "UnifiedId",
    "SymbolType",
    # Data structures
    "UnifiedSymbol",
    "UnifiedEdge",
    # Search
    "SearchMode",
    "SearchParams",
    "UnifiedSearchResult",
    # Protocols
    "VectorStoreProtocol",
    "GraphStoreProtocol",
    "UnifiedSymbolStoreProtocol",
    # Re-exports
    "GraphNode",
    "GraphEdge",
]
