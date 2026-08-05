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

"""Shared ProximaDB runtime helpers for the Code Context Graph (CCG) backend.

This module is the single place that knows how to:

- detect whether the ``proximadb_sdk`` package is importable (optional dep),
- start an **embedded** ProximaDB (local single-repo) instance, and
- derive repository and graph names used by the storage adapters.

Symbol identity and the correlated record ``oid`` are owned exclusively by
``victor-codegraph``. Storage adapters persist that emitted ``oid`` unchanged;
they do not derive an alternative identity from file, name, type, or position.

See ``docs/architecture/proximadb-codegraph-backend.md`` (TD-11/12/13) for the
design. The embedded path uses ProximaDB's in-RAM vector engine — the
equivalent of the Rust ``EmbeddingMode::Memory`` — so semantic seed→expand
scores neighbors inline. The multi-tenant **service** path (``EmbeddingMode::Cold``
/ SQ8) is gated on ProximaDB TD-127 (secondary indexes) + TD-130/131 (graph
bulk-load + REST v2 hybrid) and is treated as WIP until those land.

ProximaDB is an *optional* runtime dependency: every helper degrades gracefully
(returns ``False`` / raises a clear, actionable error) when the package or the
server binary is absent, so SQLite remains the default with no proximadb install.
"""

from __future__ import annotations

import asyncio
import enum
import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Optional-dependency detection
# ---------------------------------------------------------------------------
def is_proxima_available() -> bool:
    """Return True if the ``proximadb_sdk`` package can be imported.

    This does **not** guarantee the embedded server binary is present — use
    :func:`start_embedded_db` (which raises on a missing binary) for that.
    """
    import importlib.util

    return importlib.util.find_spec("proximadb_sdk") is not None


class ProximaUnavailableError(RuntimeError):
    """Raised when a ProximaDB-backed store is requested but cannot run.

    Carries an actionable message (install the package or build the embedded
    server binary). Callers that want SQLite-style graceful degradation should
    catch this and fall back rather than propagate.
    """


# ---------------------------------------------------------------------------
# Embedding mode (maps to the Rust EmbeddingMode enum)
# ---------------------------------------------------------------------------
class ProximaEmbeddingMode(str, enum.Enum):
    """How per-symbol vectors are held for semantic scoring.

    The Python SDK has no ``EmbeddingMode`` enum; this is Victor's encoding of
    the engine-side concept so callers/settings can express intent:

    - ``MEMORY`` — full-precision (fp32) vectors held in RAM. Used by the
      embedded/local single-repo case so semantic BFS scores neighbors inline.
      This is the default and the only mode that runs against the embedded
      PyO3/subprocess path today.
    - ``COLD`` — quantized (SQ8) vectors to bound RAM in the multi-tenant
      service deployment. Gated on TD-127/130/131; treated as WIP.
    """

    MEMORY = "memory"
    COLD = "cold"

    @classmethod
    def coerce(cls, value: "ProximaEmbeddingMode | str | None") -> "ProximaEmbeddingMode":
        if value is None:
            return cls.MEMORY
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).lower())
        except ValueError:
            logger.warning("Unknown ProximaDB embedding mode %r; defaulting to memory", value)
            return cls.MEMORY


# ---------------------------------------------------------------------------
# Repository naming (symbol identity belongs to victor-codegraph)
# ---------------------------------------------------------------------------
_REPO_SANITIZE = re.compile(r"[^A-Za-z0-9_]+")


def repo_id_from_path(project_path: Optional[Path | str]) -> str:
    """Derive a stable, filesystem-safe repo id from a project path.

    Uses the resolved directory name (lowercased, non-alphanumerics collapsed to
    ``_``). Falls back to ``repo`` for an empty/anonymous path.
    """
    if project_path is None:
        return "repo"
    name = Path(project_path).expanduser().resolve(strict=False).name or "repo"
    sanitized = _REPO_SANITIZE.sub("_", name).strip("_").lower()
    return sanitized or "repo"


def graph_id_for_repo(repo: str) -> str:
    """ORION graph id for a repo's Code Context Graph."""
    return f"{repo}_codegraph"


# ---------------------------------------------------------------------------
# Embedded bootstrap
# ---------------------------------------------------------------------------
def _pick_free_ports(count: int) -> list[int]:
    """Reserve ``count`` distinct loopback ports and return them.

    The embedded ProximaDB is a managed subprocess reached over loopback TCP, so
    it needs free REST/gRPC ports. The SDK defaults (15678/15679) frequently
    collide (Docker, a second repo, a leftover instance); picking ports the OS
    reports free avoids ``Address already in use`` on startup. There is a small
    TOCTOU window between release and the server's bind, acceptable for a
    local single-instance launch.
    """
    import socket

    held: list[socket.socket] = []
    try:
        for _ in range(count):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(("127.0.0.1", 0))
            held.append(sock)
        return [sock.getsockname()[1] for sock in held]
    finally:
        for sock in held:
            sock.close()


async def start_embedded_db(
    data_dir: Path | str,
    *,
    log_level: str = "warn",
    vector_engine: str = "SST",
    graph_engine: str = "ORION",
    binary_path: Optional[str] = None,
    timeout: float = 30.0,
    rest_port: Optional[int] = None,
    grpc_port: Optional[int] = None,
) -> Any:
    """Start and return an embedded ``EmbeddedProximaDB`` for a single repo.

    The SDK's "embedded" mode is a self-managed ``proximadb-server`` subprocess
    reached over loopback REST/gRPC (it is not in-process), so it needs ports.
    Unless explicit ports are given, free ones are selected to avoid colliding
    with the SDK defaults (15678/15679), Docker, or a leftover instance.

    Args:
        data_dir: Persistent data directory for this repo's collection/graph.
        log_level: ProximaDB log level.
        vector_engine: Vector storage engine (SST is write-optimized for code).
        graph_engine: Graph engine (ORION = WAL + in-memory).
        binary_path: Explicit path to the ``proximadb-server`` binary (optional).
        timeout: Seconds to wait for the server to become healthy.
        rest_port: Explicit REST port (default: an OS-selected free port).
        grpc_port: Explicit gRPC port (default: an OS-selected free port).

    Returns:
        A started ``EmbeddedProximaDB`` instance.

    Raises:
        ProximaUnavailableError: if the SDK is missing, the server binary cannot
            be located, or the server fails to become healthy. The message is
            actionable so callers can surface it or fall back to SQLite.
    """
    if not is_proxima_available():
        raise ProximaUnavailableError(
            "proximadb_sdk is not installed. Install it (pip install proximadb) "
            "or keep the default SQLite graph backend."
        )

    from proximadb_sdk.embedded import EmbeddedConfig, EmbeddedProximaDB

    data_path = Path(data_dir).expanduser()
    data_path.mkdir(parents=True, exist_ok=True)

    # SDK default is now UDS (portless), so port-picking is unnecessary.
    # Ports are only used when an explicit TCP transport is configured.
    if rest_port is None or grpc_port is None:
        # Leave the SDK defaults (ignored in UDS mode)
        pass

    config = EmbeddedConfig(
        data_dir=str(data_path),
        log_level=log_level,
        vector_engine=vector_engine,
        graph_engine=graph_engine,
        rest_port=rest_port or 15678,  # Fallback if explicitly set (ignored in UDS)
        grpc_port=grpc_port or 15679,  # Fallback if explicitly set (ignored in UDS)
    )
    db = EmbeddedProximaDB(config=config, binary_path=binary_path)
    try:
        await db.start(timeout=timeout)
    except (FileNotFoundError, TimeoutError, OSError) as exc:
        raise ProximaUnavailableError(
            "Could not start embedded ProximaDB: "
            f"{exc}. Build the proximadb-server binary "
            "(cd proximaDB && cargo build --release) or pass server_url for a "
            "running instance. SQLite remains the default backend."
        ) from exc
    return db


# ---------------------------------------------------------------------------
# Shared per-repo connection — one embedded instance for graph + vectors
# ---------------------------------------------------------------------------
_CONN_REGISTRY: "Dict[str, ProximaRepoConnection]" = {}
_CONN_LOCK = asyncio.Lock()


class ProximaRepoConnection:
    """One embedded ProximaDB instance shared by a repo's graph + vector stores.

    The correlated-substrate design (TD-11/12) keeps a symbol's ORION graph node
    and its HNSW vector in **one** instance under a single ``oid``. Both
    :class:`ProximaGraphStore` and ``ProximaDBProvider`` acquire the same
    connection (keyed by ``data_dir``), so the graph and the vectors are
    physically co-located and a vector hit maps to its graph node by identity
    (``vector record id == graph node id == oid``) rather than a cross-store join.

    The connection is **ref-counted**: the ``proximadb-server`` subprocess starts
    on first :meth:`acquire` and stops when the last holder calls :meth:`release`.
    """

    def __init__(self, key: str, data_dir: Path, binary_path: Optional[str]) -> None:
        self._key = key
        self._data_dir = data_dir
        self._binary_path = binary_path
        self._db: Any = None
        self._client: Any = None
        self._graphs: Dict[str, Any] = {}
        self._collections: Dict[str, Any] = {}
        self._refcount = 0

    @classmethod
    async def acquire(
        cls, data_dir: Path | str, *, binary_path: Optional[str] = None
    ) -> "ProximaRepoConnection":
        """Return the shared connection for ``data_dir``, starting it if needed."""
        key = str(Path(data_dir).expanduser().resolve(strict=False))
        async with _CONN_LOCK:
            conn = _CONN_REGISTRY.get(key)
            if conn is None:
                conn = cls(key, Path(data_dir), binary_path)
                _CONN_REGISTRY[key] = conn
            conn._refcount += 1
            try:
                await conn._ensure_started()
            except Exception:
                # Roll back the registration if startup failed.
                conn._refcount -= 1
                if conn._refcount <= 0:
                    _CONN_REGISTRY.pop(key, None)
                raise
            return conn

    async def _ensure_started(self) -> None:
        if self._db is not None:
            return
        from proximadb_sdk.unified_client import ProximaDBClient

        self._db = await start_embedded_db(self._data_dir, binary_path=self._binary_path)
        # Graph + vector ops are served over REST (the gRPC client has no graph
        # RPCs), so the shared client is pinned to the REST protocol.
        self._client = ProximaDBClient(url=self._db.rest_url, protocol="rest")

    @property
    def client(self) -> Any:
        return self._client

    @property
    def embedded_db(self) -> Any:
        return self._db

    async def graph(self, graph_id: str) -> Any:
        """Return (and cache) a ``ProximaDBGraph`` for ``graph_id`` (idempotent create)."""
        cached = self._graphs.get(graph_id)
        if cached is not None:
            return cached
        from proximadb_sdk.graph import ProximaDBGraph

        try:
            await asyncio.to_thread(self._client.create_graph, graph_id)
        except Exception as exc:  # pragma: no cover - depends on live server
            logger.debug("create_graph(%s) noop/failed: %s", graph_id, exc)
        graph = ProximaDBGraph(self._client, graph_id)
        self._graphs[graph_id] = graph
        return graph

    async def get_or_create_collection(
        self,
        name: str,
        *,
        dimension: int,
        distance_metric: str = "cosine",
        embedding_model: Any = None,
    ) -> Any:
        """Return (and cache) a vector collection on this shared instance."""
        cached = self._collections.get(name)
        if cached is not None:
            if embedding_model is not None and not getattr(cached, "has_embedding_model", True):
                cached.set_embedding_model(embedding_model)
            return cached
        collection = await self._db.create_collection(
            name,
            dimension=dimension,
            distance_metric=distance_metric,
            embedding_model=embedding_model,
        )
        self._collections[name] = collection
        return collection

    def forget_collection(self, name: str) -> None:
        """Drop a cached collection handle (e.g. after it was deleted server-side)."""
        self._collections.pop(name, None)

    async def release(self) -> None:
        """Drop one reference; stop the subprocess when the last holder releases."""
        async with _CONN_LOCK:
            self._refcount -= 1
            if self._refcount > 0:
                return
            _CONN_REGISTRY.pop(self._key, None)
            db = self._db
            self._db = None
            self._client = None
            self._graphs.clear()
            self._collections.clear()
        if db is not None:
            try:
                await db.stop()
            except Exception as exc:  # pragma: no cover
                logger.debug("Embedded ProximaDB stop failed: %s", exc)
