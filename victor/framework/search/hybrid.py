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

"""Compatibility re-export: hybrid RRF fusion lives in victor.core.search.hybrid.

It depends only on core types, and core retrieval (victor.core.graph_rag)
fuses with it, so it belongs below the framework layer. Existing imports from
this module keep working.
"""

from victor.core.search.hybrid import (
    HybridSearchEngine,
    HybridSearchResult,
    create_hybrid_search_engine,
)

__all__ = [
    "HybridSearchEngine",
    "HybridSearchResult",
    "create_hybrid_search_engine",
]
