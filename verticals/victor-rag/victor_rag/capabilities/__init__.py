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

"""Dynamic capability definitions for the RAG vertical.

This module provides capability declarations that can be loaded
dynamically by the CapabilityLoader, enabling runtime extension
of the RAG vertical with custom functionality.

The module follows the CapabilityLoader's discovery patterns:
1. CAPABILITIES list for batch registration
2. @capability decorator for function-based capabilities
3. Capability classes for complex implementations

Example:
    # Register capabilities with loader
    from victor.framework import CapabilityLoader
    loader = CapabilityLoader()
    loader.load_from_module("victor_rag.capabilities")

    # Or use directly
    from victor_rag.capabilities import (
        get_rag_capabilities,
        RAGCapabilityProvider,
    )

This package is now split into multiple modules:
- config.py: Configuration defaults and helpers
- handlers.py: Configuration handler functions
- decorators.py: @capability decorated functions
- provider.py: RAGCapabilityProvider class and discovery
"""

# Re-export all public APIs for backward compatibility
from victor_rag.capabilities.config import (
    _INDEXING_DEFAULTS,
    _RETRIEVAL_DEFAULTS,
    _SYNTHESIS_DEFAULTS,
    _SAFETY_DEFAULTS,
    _QUERY_ENHANCEMENT_DEFAULTS,
    _RAG_DEFAULTS,
    _load_rag_config,
    _store_rag_config,
    _store_rag_section,
)

from victor_rag.capabilities.handlers import (
    configure_indexing,
    get_indexing_config,
    configure_retrieval,
    get_retrieval_config,
    configure_synthesis,
    get_synthesis_config,
    configure_safety,
    configure_query_enhancement,
)

from victor_rag.capabilities.decorators import (
    rag_indexing,
    rag_retrieval,
    rag_synthesis,
    rag_safety,
)

from victor_rag.capabilities.provider import (
    RAGCapabilityProvider,
    CAPABILITIES,
    get_rag_capabilities,
    create_rag_capability_loader,
    get_capability_configs,
)

__all__ = [
    # Config
    "_INDEXING_DEFAULTS",
    "_RETRIEVAL_DEFAULTS",
    "_SYNTHESIS_DEFAULTS",
    "_SAFETY_DEFAULTS",
    "_QUERY_ENHANCEMENT_DEFAULTS",
    "_RAG_DEFAULTS",
    "_load_rag_config",
    "_store_rag_config",
    "_store_rag_section",
    # Handlers
    "configure_indexing",
    "get_indexing_config",
    "configure_retrieval",
    "get_retrieval_config",
    "configure_synthesis",
    "get_synthesis_config",
    "configure_safety",
    "configure_query_enhancement",
    # Decorators
    "rag_indexing",
    "rag_retrieval",
    "rag_synthesis",
    "rag_safety",
    # Provider
    "RAGCapabilityProvider",
    "CAPABILITIES",
    "get_rag_capabilities",
    "create_rag_capability_loader",
    "get_capability_configs",
]
