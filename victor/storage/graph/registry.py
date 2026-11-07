# Pluggable graph store factory.
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from victor.storage.graph.protocol import GraphStoreProtocol
from victor.storage.graph.sqlite_store import SqliteGraphStore
from victor.storage.graph.memory_store import MemoryGraphStore

logger = logging.getLogger(__name__)

try:
    from victor.storage.graph.duckdb_store import DuckDBGraphStore
except Exception:
    DuckDBGraphStore = None

# Per-repo backend override file: a project can opt into a non-default graph
# backend (e.g. ProximaDB) without touching global settings by writing the
# backend name into `<project>/.victor/graph_backend`. SQLite stays the default.
_BACKEND_MARKER = Path(".victor") / "graph_backend"


def resolve_graph_backend(
    project_path: Optional[Path] = None,
    *,
    default: str = "sqlite",
) -> str:
    """Resolve the graph backend for a repo (per-repo flag, default SQLite).

    Precedence: ``<project>/.victor/graph_backend`` file > ``default``. The
    marker file lets a single repo flip to ``proxima`` once parity holds without
    changing global settings.
    """
    base = Path(project_path).expanduser() if project_path else Path.cwd()
    marker = base / _BACKEND_MARKER
    try:
        if marker.is_file():
            value = marker.read_text(encoding="utf-8").strip().lower()
            if value:
                return value
    except OSError as exc:  # pragma: no cover - filesystem edge
        logger.debug("Could not read graph backend marker %s: %s", marker, exc)
    return default


def create_graph_store(
    name: str = "sqlite",
    project_path: Optional[Path] = None,
    *,
    backend: Optional[str] = None,
    read_only: bool = False,
) -> GraphStoreProtocol:
    """Create a graph store by name.

    Args:
        name: Backend name (sqlite, memory, duckdb, proxima). Pass ``"auto"`` to
            honor the per-repo ``.victor/graph_backend`` marker (default sqlite).
        project_path: Path to project root. If None, uses current directory.
        backend: Alias for name (used by victor-coding shim). Takes priority if set.
        read_only: Open an existing index for queries only; nothing is
            migrated or written, and database content never changes (SQLite may
            create empty ``-wal``/``-shm`` coordination files). Only the sqlite backend supports it; other
            backends raise ``ValueError`` rather than open read-write.

    Returns:
        GraphStoreProtocol implementation
    """
    selected_backend = str(backend if backend is not None else name or "sqlite").lower()
    if selected_backend == "auto":
        selected_backend = resolve_graph_backend(project_path)
    if read_only and selected_backend != "sqlite":
        raise ValueError(f"Graph store backend {selected_backend!r} has no read-only mode")
    if selected_backend == "sqlite":
        return SqliteGraphStore(project_path=project_path, read_only=read_only)
    if selected_backend == "memory":
        return MemoryGraphStore()
    if selected_backend == "duckdb":
        if DuckDBGraphStore is None:
            raise ValueError("DuckDB graph backend requested but duckdb is not installed")
        return DuckDBGraphStore(project_path)
    if selected_backend in {"proxima", "proximadb"}:
        # Imported lazily so the optional proximadb dependency never breaks the
        # default SQLite path at import time.
        from victor.storage.graph.proxima_store import ProximaGraphStore

        return ProximaGraphStore(project_path=project_path)
    raise ValueError(f"Unsupported graph store backend: {selected_backend}")
