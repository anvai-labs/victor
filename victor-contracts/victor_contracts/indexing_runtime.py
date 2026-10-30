"""Runtime bridge: indexing infrastructure."""

# ruff: noqa: F822

from __future__ import annotations


from victor_contracts._lazy import make_lazy_getattr

__all__ = [
    "FileChangeEvent",
    "FileChangeType",
    "FileWatcherRegistry",
    "IndexLockRegistry",
    "GraphBuildCoordinator",
    "GraphManager",
]

_LAZY_IMPORTS = {
    "FileChangeEvent": "victor.core.indexing.file_watcher",
    "FileChangeType": "victor.core.indexing.file_watcher",
    "FileWatcherRegistry": "victor.core.indexing.file_watcher",
    "IndexLockRegistry": "victor.core.indexing.index_lock",
    "GraphBuildCoordinator": "victor.core.indexing.graph_manager",
    "GraphManager": "victor.core.indexing.graph_manager",
}


__getattr__ = make_lazy_getattr(__name__, _LAZY_IMPORTS)
