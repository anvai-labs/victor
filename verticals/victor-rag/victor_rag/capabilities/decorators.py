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

"""RAG capability decorators for declaring capability functions.

This module provides @capability decorated functions for RAG capabilities
including indexing, retrieval, synthesis, and safety.
"""

from __future__ import annotations

from typing import Any, Callable

from victor_contracts import CapabilityType
from victor_contracts.capabilities import capability

from victor_rag.capabilities.handlers import (
    configure_indexing,
    configure_retrieval,
    configure_synthesis,
    configure_safety,
)


@capability(
    name="rag_indexing",
    capability_type=CapabilityType.MODE,
    version="1.0",
    description="Document indexing and chunking configuration",
)
def rag_indexing(
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    **kwargs: Any,
) -> Callable:
    """Indexing capability handler."""

    def handler(orchestrator: Any) -> None:
        configure_indexing(
            orchestrator,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            **kwargs,
        )

    return handler


@capability(
    name="rag_retrieval",
    capability_type=CapabilityType.MODE,
    version="1.0",
    description="Retrieval and search configuration",
    getter="get_retrieval_config",
)
def rag_retrieval(
    top_k: int = 5,
    search_type: str = "hybrid",
    **kwargs: Any,
) -> Callable:
    """Retrieval capability handler."""

    def handler(orchestrator: Any) -> None:
        configure_retrieval(
            orchestrator,
            top_k=top_k,
            search_type=search_type,
            **kwargs,
        )

    return handler


@capability(
    name="rag_synthesis",
    capability_type=CapabilityType.MODE,
    version="1.0",
    description="Answer synthesis and citation configuration",
)
def rag_synthesis(
    citation_style: str = "inline",
    include_sources: bool = True,
    **kwargs: Any,
) -> Callable:
    """Synthesis capability handler."""

    def handler(orchestrator: Any) -> None:
        configure_synthesis(
            orchestrator,
            citation_style=citation_style,
            include_sources=include_sources,
            **kwargs,
        )

    return handler


@capability(
    name="rag_safety",
    capability_type=CapabilityType.SAFETY,
    version="1.0",
    description="RAG safety and filtering configuration",
)
def rag_safety(
    filter_sensitive_data: bool = True,
    **kwargs: Any,
) -> Callable:
    """Safety capability handler."""

    def handler(orchestrator: Any) -> None:
        configure_safety(
            orchestrator,
            filter_sensitive_data=filter_sensitive_data,
            **kwargs,
        )

    return handler


# Re-export for backward compatibility
__all__ = [
    "rag_indexing",
    "rag_retrieval",
    "rag_synthesis",
    "rag_safety",
]
