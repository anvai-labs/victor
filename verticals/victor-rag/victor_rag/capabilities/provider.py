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

"""RAG capability provider and discovery infrastructure.

This module provides the RAGCapabilityProvider class, CAPABILITIES list
for discovery, and convenience functions for capability registration.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Set, TYPE_CHECKING

from victor_contracts import CapabilityType, OrchestratorCapability
from victor_contracts.capabilities import CapabilityEntry
from victor_contracts import BaseCapabilityProvider, CapabilityMetadata

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

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# =============================================================================
# Capability Provider Class
# =============================================================================


class RAGCapabilityProvider(BaseCapabilityProvider[Callable[..., None]]):
    """Provider for RAG-specific capabilities.

    This class provides a structured way to access and apply
    RAG capabilities to an orchestrator. It inherits from
    BaseCapabilityProvider for consistent capability registration
    and discovery across all verticals.

    Capabilities:
    - indexing: Document indexing and chunking settings
    - retrieval: Search and retrieval configuration
    - synthesis: Answer generation and citation settings
    - safety: Data filtering and validation
    - query_enhancement: Query expansion and decomposition

    Example:
        provider = RAGCapabilityProvider()

        # List available capabilities
        print(provider.list_capabilities())

        # Apply specific capabilities
        provider.apply_indexing(orchestrator, chunk_size=1024)
        provider.apply_retrieval(orchestrator, top_k=10)

        # Use BaseCapabilityProvider interface
        cap = provider.get_capability("indexing")
        if cap:
            cap(orchestrator, chunk_size=512)
    """

    def __init__(self):
        """Initialize the capability provider."""
        self._applied: Set[str] = set()
        # Map capability names to their handler functions
        self._capabilities: Dict[str, Callable[..., None]] = {
            "indexing": configure_indexing,
            "retrieval": configure_retrieval,
            "synthesis": configure_synthesis,
            "safety": configure_safety,
            "query_enhancement": configure_query_enhancement,
        }
        # Capability metadata for discovery
        self._metadata: Dict[str, CapabilityMetadata] = {
            "indexing": CapabilityMetadata(
                name="indexing",
                description="Document indexing and chunking configuration",
                version="1.0",
                tags=["indexing", "chunking", "embedding"],
            ),
            "retrieval": CapabilityMetadata(
                name="retrieval",
                description="Search and retrieval configuration",
                version="1.0",
                dependencies=["indexing"],
                tags=["retrieval", "search", "ranking"],
            ),
            "synthesis": CapabilityMetadata(
                name="synthesis",
                description="Answer generation and citation configuration",
                version="1.0",
                dependencies=["retrieval"],
                tags=["synthesis", "generation", "citations"],
            ),
            "safety": CapabilityMetadata(
                name="safety",
                description="Data filtering and validation settings",
                version="1.0",
                tags=["safety", "filtering", "validation"],
            ),
            "query_enhancement": CapabilityMetadata(
                name="query_enhancement",
                description="Query expansion and decomposition settings",
                version="1.0",
                tags=["query", "expansion", "decomposition"],
            ),
        }

    def get_capabilities(self) -> Dict[str, Callable[..., None]]:
        """Return all registered capabilities.

        Returns:
            Dictionary mapping capability names to handler functions.
        """
        return self._capabilities.copy()

    def get_capability_metadata(self) -> Dict[str, CapabilityMetadata]:
        """Return metadata for all registered capabilities.

        Returns:
            Dictionary mapping capability names to their metadata.
        """
        return self._metadata.copy()

    def apply_indexing(
        self,
        orchestrator: Any,
        **kwargs: Any,
    ) -> None:
        """Apply indexing capability.

        Args:
            orchestrator: Target orchestrator
            **kwargs: Indexing options
        """
        configure_indexing(orchestrator, **kwargs)
        self._applied.add("indexing")

    def apply_retrieval(
        self,
        orchestrator: Any,
        **kwargs: Any,
    ) -> None:
        """Apply retrieval capability.

        Args:
            orchestrator: Target orchestrator
            **kwargs: Retrieval options
        """
        configure_retrieval(orchestrator, **kwargs)
        self._applied.add("retrieval")

    def apply_synthesis(
        self,
        orchestrator: Any,
        **kwargs: Any,
    ) -> None:
        """Apply synthesis capability.

        Args:
            orchestrator: Target orchestrator
            **kwargs: Synthesis options
        """
        configure_synthesis(orchestrator, **kwargs)
        self._applied.add("synthesis")

    def apply_safety(
        self,
        orchestrator: Any,
        **kwargs: Any,
    ) -> None:
        """Apply safety capability.

        Args:
            orchestrator: Target orchestrator
            **kwargs: Safety options
        """
        configure_safety(orchestrator, **kwargs)
        self._applied.add("safety")

    def apply_query_enhancement(
        self,
        orchestrator: Any,
        **kwargs: Any,
    ) -> None:
        """Apply query enhancement capability.

        Args:
            orchestrator: Target orchestrator
            **kwargs: Query enhancement options
        """
        configure_query_enhancement(orchestrator, **kwargs)
        self._applied.add("query_enhancement")

    def apply_all(
        self,
        orchestrator: Any,
        **kwargs: Any,
    ) -> None:
        """Apply all RAG capabilities with defaults.

        Args:
            orchestrator: Target orchestrator
            **kwargs: Shared options
        """
        self.apply_indexing(orchestrator)
        self.apply_retrieval(orchestrator)
        self.apply_synthesis(orchestrator)
        self.apply_safety(orchestrator)
        self.apply_query_enhancement(orchestrator)

    def get_applied(self) -> Set[str]:
        """Get set of applied capability names.

        Returns:
            Set of applied capability names
        """
        return self._applied.copy()


# =============================================================================
# CAPABILITIES List for CapabilityLoader Discovery
# =============================================================================


CAPABILITIES: List[CapabilityEntry] = [
    CapabilityEntry(
        capability=OrchestratorCapability(
            name="rag_indexing",
            capability_type=CapabilityType.MODE,
            version="1.0",
            setter="configure_indexing",
            getter="get_indexing_config",
            description="Document indexing and chunking configuration",
        ),
        handler=configure_indexing,
        getter_handler=get_indexing_config,
    ),
    CapabilityEntry(
        capability=OrchestratorCapability(
            name="rag_retrieval",
            capability_type=CapabilityType.MODE,
            version="1.0",
            setter="configure_retrieval",
            getter="get_retrieval_config",
            description="Search and retrieval configuration",
        ),
        handler=configure_retrieval,
        getter_handler=get_retrieval_config,
    ),
    CapabilityEntry(
        capability=OrchestratorCapability(
            name="rag_synthesis",
            capability_type=CapabilityType.MODE,
            version="1.0",
            setter="configure_synthesis",
            getter="get_synthesis_config",
            description="Answer synthesis and citation configuration",
        ),
        handler=configure_synthesis,
        getter_handler=get_synthesis_config,
    ),
    CapabilityEntry(
        capability=OrchestratorCapability(
            name="rag_safety",
            capability_type=CapabilityType.SAFETY,
            version="1.0",
            setter="configure_safety",
            description="RAG safety and filtering configuration",
        ),
        handler=configure_safety,
    ),
    CapabilityEntry(
        capability=OrchestratorCapability(
            name="rag_query_enhancement",
            capability_type=CapabilityType.MODE,
            version="1.0",
            setter="configure_query_enhancement",
            description="Query expansion and decomposition configuration",
        ),
        handler=configure_query_enhancement,
    ),
]


# =============================================================================
# Convenience Functions
# =============================================================================


def get_rag_capabilities() -> List[CapabilityEntry]:
    """Get all RAG capability entries.

    Returns:
        List of capability entries for loader registration
    """
    return CAPABILITIES.copy()


def create_rag_capability_loader() -> Any:
    """Create a CapabilityLoader pre-configured for RAG vertical.

    Returns:
        CapabilityLoader with RAG capabilities registered
    """
    from victor_contracts.capabilities import create_runtime_capability_loader

    return create_runtime_capability_loader(
        CAPABILITIES,
        source_module="victor_rag.capabilities",
    )


# =============================================================================
# SOLID: Centralized Config Storage
# =============================================================================


def get_capability_configs() -> Dict[str, Any]:
    """Get RAG capability configurations for centralized storage.

    Returns default RAG configuration for VerticalContext storage.
    This replaces direct orchestrator.rag_config assignment.

    Returns:
        Dict with default RAG capability configurations
    """
    return {
        "rag_config": {
            "indexing": {
                "chunk_size": 512,
                "chunk_overlap": 50,
                "chunk_strategy": "recursive",
            },
            "retrieval": {
                "top_k": 5,
                "search_type": "hybrid",
                "min_similarity_score": 0.7,
            },
            "synthesis": {
                "citation_style": "inline",
                "include_sources": True,
                "max_context_length": 4000,
            },
            "safety": {
                "max_sources": 10,
                "enable_source_verification": True,
                "enable_fact_checking": False,
            },
            "query_enhancement": {
                "enable_expansion": True,
                "enable_decomposition": False,
                "max_query_variants": 3,
                "use_synonyms": True,
            },
        }
    }


# Re-export for backward compatibility
__all__ = [
    "RAGCapabilityProvider",
    "CAPABILITIES",
    "get_rag_capabilities",
    "create_rag_capability_loader",
    "get_capability_configs",
]
