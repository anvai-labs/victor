"""Runtime bridge: graph RAG multi-hop retrieval."""

# ruff: noqa: F822

from __future__ import annotations


from victor_contracts._lazy import make_lazy_getattr

__all__ = ["MultiHopRetriever", "RetrievalConfig"]

_LAZY_IMPORTS = {
    "MultiHopRetriever": "victor.core.graph_rag",
    "RetrievalConfig": "victor.core.graph_rag",
}


__getattr__ = make_lazy_getattr(__name__, _LAZY_IMPORTS)
