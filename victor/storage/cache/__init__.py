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

"""Caching system for Victor.

Tiered caching architecture:
- L1: Memory cache (fast, short-lived)
- L2: Disk cache (persistent, longer-lived)

Features:
- Automatic tiering
- Namespace-scoped access
- TTL support
- Size limits
- Thread-safe

Phase 3: Extended Caching
- GenericResultCache: Extended caching for non-tool results
- Enhanced EmbeddingCache: Batch operations and preloading
- HttpConnectionPool: Connection pooling for HTTP tools
- PreloadManager: Warm cache initialization

Usage:
    from victor.storage.cache import get_cache_manager

    # Get namespace-scoped cache
    cache = get_cache_manager()
    tool_cache = cache.namespace("tools")
    tool_cache.set("key", value)
"""

from victor.storage.cache.config import CacheConfig
from victor.storage.cache.tiered_cache import TieredCache, EmbeddingCache

# New unified cache manager
from victor.storage.cache.manager import (
    CacheManager,
    CacheNamespace,
    CacheStats,
    get_cache_manager,
    set_cache_manager,
    reset_cache_manager,
    get_tools_cache,
    get_embeddings_cache,
    get_responses_cache,
    get_code_search_cache,
)

# Phase 3: Extended caching
from victor.storage.cache.generic_result_cache import (
    GenericResultCache,
    ResultType,
    CacheDependency,
    InvalidationStrategy,
    CacheKeys,
)

__all__ = [
    # Configuration
    "CacheConfig",
    # Low-level cache
    "TieredCache",
    "EmbeddingCache",
    # Unified cache manager
    "CacheManager",
    "CacheNamespace",
    "CacheStats",
    "get_cache_manager",
    "set_cache_manager",
    "reset_cache_manager",
    # Convenience accessors
    "get_tools_cache",
    "get_embeddings_cache",
    "get_responses_cache",
    "get_code_search_cache",
    # Phase 3: Extended caching
    "GenericResultCache",
    "ResultType",
    "CacheDependency",
    "InvalidationStrategy",
    "CacheKeys",
]
