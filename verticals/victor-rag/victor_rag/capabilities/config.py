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

"""RAG capability configuration defaults and helper functions.

This module provides the default configuration values for RAG capabilities
and helper functions for loading/storing configuration.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

from victor_contracts.capabilities import (
    load_capability_config,
    store_capability_config,
    update_capability_config_section,
)

# =============================================================================
# Configuration Defaults
# =============================================================================

_INDEXING_DEFAULTS: Dict[str, Any] = {
    "chunk_size": 1024,
    "chunk_overlap": 128,
    "embedding_model": "text-embedding-3-small",
    "embedding_dimensions": 1536,
    "store_backend": "lancedb",
}

_RETRIEVAL_DEFAULTS: Dict[str, Any] = {
    "top_k": 5,
    "similarity_threshold": 0.7,
    "search_type": "hybrid",
    "rerank_enabled": True,
    "max_context_tokens": 4000,
}

_SYNTHESIS_DEFAULTS: Dict[str, Any] = {
    "citation_style": "inline",
    "include_sources": True,
    "max_answer_tokens": 2000,
    "temperature": 0.3,
    "require_verification": True,
}

_SAFETY_DEFAULTS: Dict[str, Any] = {
    "filter_sensitive_data": True,
    "max_document_size_mb": 50,
    "allowed_file_types": ["pdf", "docx", "txt", "md", "py", "js", "ts", "html"],
    "validate_sources": True,
}

_QUERY_ENHANCEMENT_DEFAULTS: Dict[str, Any] = {
    "enable_expansion": True,
    "enable_decomposition": True,
    "max_query_variants": 3,
    "use_synonyms": True,
}

_RAG_DEFAULTS: Dict[str, Any] = {
    "indexing": deepcopy(_INDEXING_DEFAULTS),
    "retrieval": deepcopy(_RETRIEVAL_DEFAULTS),
    "synthesis": deepcopy(_SYNTHESIS_DEFAULTS),
    "safety": deepcopy(_SAFETY_DEFAULTS),
    "query_enhancement": deepcopy(_QUERY_ENHANCEMENT_DEFAULTS),
}


# =============================================================================
# Configuration Helper Functions
# =============================================================================


def _load_rag_config(orchestrator: Any) -> Dict[str, Any]:
    """Load the full RAG config from framework service or legacy orchestrator attr."""
    return load_capability_config(orchestrator, "rag_config", _RAG_DEFAULTS)


def _store_rag_config(orchestrator: Any, config: Dict[str, Any]) -> None:
    """Store full RAG config in framework service or legacy orchestrator attr."""
    store_capability_config(
        orchestrator,
        "rag_config",
        config,
        require_existing_attr=False,
    )


def _store_rag_section(orchestrator: Any, section: str, section_config: Dict[str, Any]) -> None:
    """Store/update a section of rag_config while preserving other sections."""
    update_capability_config_section(
        orchestrator,
        root_name="rag_config",
        section_name=section,
        section_config=section_config,
        root_defaults=_RAG_DEFAULTS,
        require_existing_attr=False,
    )


# Re-export for backward compatibility
__all__ = [
    "_INDEXING_DEFAULTS",
    "_RETRIEVAL_DEFAULTS",
    "_SYNTHESIS_DEFAULTS",
    "_SAFETY_DEFAULTS",
    "_QUERY_ENHANCEMENT_DEFAULTS",
    "_RAG_DEFAULTS",
    "_load_rag_config",
    "_store_rag_config",
    "_store_rag_section",
]
