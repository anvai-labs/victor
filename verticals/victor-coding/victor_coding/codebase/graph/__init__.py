# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0

"""Graph storage module for codebase analysis.

This module re-exports from the core victor.storage.graph module for backward
compatibility. The graph storage infrastructure is now part of victor-core
and available to all verticals.

For new code, prefer importing directly from victor.storage.graph:
    from victor.storage.graph import GraphNode, GraphEdge, create_graph_store
"""

# Re-export from victor-core
from victor.storage.graph import (
    GraphNode,
    GraphEdge,
    GraphStoreProtocol,
    create_graph_store,
    SqliteGraphStore,
    MemoryGraphStore,
)

__all__ = [
    "GraphNode",
    "GraphEdge",
    "GraphStoreProtocol",
    "create_graph_store",
    "SqliteGraphStore",
    "MemoryGraphStore",
]
