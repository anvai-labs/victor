# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0

"""Victor Processing Module - Unified processing-related components.

This module consolidates processing-related functionality:

- **serialization**: Token-optimized serialization for LLM communication
- **native**: Rust-powered native extensions for performance
- **editing**: File editing and transaction support
- **merge**: Merge conflict resolution
- **completion**: Code completion providers for IDE integration
- **file_types**: File type detection and categorization
- **graph_profiler**: Performance profiling for graph operations (PH4-008)
- **graph_optimizations**: Optimization utilities for graph operations (PH4-008)

Example usage:
    # Import from submodules
    from victor.processing.serialization import AdaptiveSerializer
    from victor.processing.native import batch_cosine_similarity
    from victor.processing.editing import FileEditor
    from victor.processing.merge import MergeManager
    from victor.processing.completion import CompletionManager
    from victor.processing.file_types import detect_file_type
    from victor.processing.graph_profiler import GraphProfiler, profile_graph_operation
    from victor.processing.graph_optimizations import GraphOptimizer, optimize_batch_size

    # Or import submodules directly
    from victor.processing import serialization, native, editing
"""

# Expose submodules for easy access
from victor.processing import (
    serialization,
    native,
    editing,
    merge,
    completion,
    file_types,
)

# Import graph-related modules (PH4-008)
try:
    from victor.processing import graph_profiler, graph_optimizations

    _graph_modules_available = True
except ImportError:
    _graph_modules_available = False

__all__ = [
    "serialization",
    "native",
    "editing",
    "merge",
    "completion",
    "file_types",
    "graph_profiler",
    "graph_optimizations",
]
