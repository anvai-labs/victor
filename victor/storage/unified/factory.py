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

"""Factory for creating UnifiedSymbolStore instances.

Supports multiple backends via factory pattern (OCP - Open/Closed Principle):
- sqlite+lancedb: Default, local storage (SQLite graph + LanceDB vectors)
- proximadb: Planned single-engine backend

Usage:
    # Default backend
    store = create_symbol_store(repo_root=Path("."))

"""

from pathlib import Path
from typing import Any, Dict, Optional

from victor.storage.unified.protocol import UnifiedSymbolStoreProtocol


def create_symbol_store(
    repo_root: Optional[Path] = None,
    backend: str = "sqlite+lancedb",
    **kwargs: Any,
) -> UnifiedSymbolStoreProtocol:
    """Create a UnifiedSymbolStore with the specified backend.

    Args:
        repo_root: Root directory of the repository. If None, uses cwd().
        backend: Storage backend to use:
            - "sqlite+lancedb": Default, SQLite graph + LanceDB vectors (local/air-gapped)
            - "proximadb": Planned single-engine backend (not yet available)
        **kwargs: Backend-specific configuration:
            - embedding_model: Override embedding model (default: sentence-transformers)
            - persist_directory: Override storage location
            - table_name: Override table names

    Returns:
        UnifiedSymbolStoreProtocol implementation

    Example:
        # Local development (air-gapped compatible)
        store = create_symbol_store(repo_root=Path("."))

    """
    if repo_root is None:
        repo_root = Path.cwd()

    if backend == "sqlite+lancedb":
        from victor.storage.unified.sqlite_lancedb import SqliteLanceDBStore

        return SqliteLanceDBStore(repo_root, **kwargs)

    elif backend == "proximadb":
        # Future: ProximaDB provides vector + relational in a single engine
        raise NotImplementedError(
            "proximadb backend planned but not yet available. "
            "Use 'sqlite+lancedb' (the default and recommended backend)."
        )

    else:
        raise ValueError(
            f"Unknown backend: {backend}. "
            f"Supported: sqlite+lancedb (default), proximadb (planned)"
        )


__all__ = ["create_symbol_store"]
