"""SDK host adapters for search runtime helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from victor_contracts._lazy import make_lazy_getattr

if TYPE_CHECKING:
    from victor.framework.search.query_expansion import (
        ExpandedQuery,
        QueryExpander,
        QueryExpansionConfig,
        QueryExpanderProtocol,
        create_query_expander,
    )

__all__ = [  # noqa: F822
    "ExpandedQuery",
    "QueryExpander",
    "QueryExpansionConfig",
    "QueryExpanderProtocol",
    "create_query_expander",
    "create_hybrid_search_engine",
]

_LAZY_IMPORTS = {
    "get_project_paths": "victor.framework.search",
    "write_codebase_index_manifest": "victor.framework.search",
    "rerank_code_search_results": "victor.framework.search",
    "has_persisted_codebase_index_data": "victor.framework.search",
    "has_compatible_codebase_index_manifest": "victor.framework.search",
    "extract_skeleton": "victor.framework.search",
    "enrich_code_search_results": "victor.framework.search",
    "enable_structural_codebase_embeddings": "victor.framework.search",
    "build_codebase_index_manifest": "victor.framework.search",
    "STRUCTURAL_CODEBASE_VECTOR_STORE": "victor.framework.search",
    "DEFAULT_CODEBASE_CHUNK_SIZE": "victor.framework.search",
    "DEFAULT_CODEBASE_CHUNK_OVERLAP": "victor.framework.search",
    "DEFAULT_CODEBASE_CHUNKING_STRATEGY": "victor.framework.search",
    "CODEBASE_INDEX_MANIFEST_NAME": "victor.framework.search",
    "ExpandedQuery": "victor.framework.search.query_expansion",
    "QueryExpander": "victor.framework.search.query_expansion",
    "QueryExpansionConfig": "victor.framework.search.query_expansion",
    "QueryExpanderProtocol": "victor.framework.search.query_expansion",
    "create_query_expander": "victor.framework.search.query_expansion",
    "create_hybrid_search_engine": "victor.framework.search.hybrid",
}


__getattr__ = make_lazy_getattr(__name__, _LAZY_IMPORTS)
