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

"""CheckpointerProtocol adapter over SQLiteCheckpointBackend (FEP-0047, FOU-D1).

Bridges the framework's graph ``CheckpointerProtocol`` (thread-keyed
``WorkflowCheckpoint`` save/load/list) to the storage layer's durable
``SQLiteCheckpointBackend`` so StateGraph runs survive process restarts.

Design notes (FEP-0047):
- Workflow threads are namespaced under ``wf::`` session keys so they never
  collide with conversation checkpoints sharing the same database file.
- ``WorkflowCheckpoint.to_dict()`` is stored as the backend's ``state_data``
  payload; FEP-0032's ``next_node`` rides inside it (metadata placement was
  chosen precisely so every backend persists it unchanged) and round-trips
  for free. ``parent_id`` is mirrored into the column for fork/time-travel
  tooling that reads it directly.
- The upward import of ``victor.framework.graph_checkpoint`` Protocol/dataclass
  is sanctioned dependency inversion (docs/architecture.md; layering ratchet).
- Snapshot-on-save / copy-on-load match ``MemoryCheckpointer``'s contract:
  callers mutate live state after save and mutate the resumed state after load.
"""

import asyncio
import copy
import json
import logging
import sqlite3
import sys
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, List, Optional
from datetime import datetime, timedelta, timezone

from pydantic import BaseModel

from victor.framework.graph_checkpoint import CheckpointerProtocol, WorkflowCheckpoint
from victor.storage.checkpoints.backends.sqlite_backend import SQLiteCheckpointBackend
from victor.storage.checkpoints.protocol import CheckpointMetadata

logger = logging.getLogger(__name__)

# Session keyspace prefix for workflow (graph) checkpoints. Conversation
# checkpoints use bare session ids; this namespace keeps both coexisting
# in one database file with independent retention.
WORKFLOW_SESSION_PREFIX = "wf::"

# Metadata tag marking adapter-written rows.
_WORKFLOW_TAG = "workflow"

# Page size when hydrating full checkpoints for list().
_LIST_PAGE_SIZE = 100


def _namespaced_session(thread_id: str) -> str:
    return f"{WORKFLOW_SESSION_PREFIX}{thread_id}"


_MODEL_STATE_MARKER = "__victor_model_state__"


def _encode_state(state: Any) -> Any:
    """Encode typed states without falling back to ``repr``.

    The storage serializer understands mappings and primitives, but a top-level
    Pydantic model otherwise reaches its string fallback. That preserves an
    inspection copy but makes durable resume operate on a string, breaking
    graph nodes that expect the typed state. Keep the type reference with the
    JSON-safe model fields so the adapter can reconstruct the model on load.
    """
    if not isinstance(state, BaseModel):
        return state
    model_type = type(state)
    return {
        _MODEL_STATE_MARKER: True,
        "type": f"{model_type.__module__}.{model_type.__qualname__}",
        "data": state.model_dump(mode="json"),
    }


def _decode_state(state: Any) -> Any:
    """Reconstruct a Pydantic checkpoint state encoded by :func:`_encode_state`."""
    if not (
        isinstance(state, dict)
        and state.get(_MODEL_STATE_MARKER) is True
        and isinstance(state.get("type"), str)
        and isinstance(state.get("data"), dict)
    ):
        return state

    module_name, separator, qual_name = state["type"].rpartition(".")
    if not module_name or not separator or not qual_name:
        raise ValueError("Stored typed workflow checkpoint has an invalid state type reference")
    try:
        # Resolve only classes already loaded by the running graph. Persisted
        # checkpoint data must never choose and import arbitrary modules.
        module = sys.modules.get(module_name)
        model_type = getattr(module, qual_name) if module is not None else None
        if not (isinstance(model_type, type) and issubclass(model_type, BaseModel)):
            raise TypeError("state type is not a loaded Pydantic model")
        return model_type.model_validate(state["data"])
    except Exception as exc:
        raise ValueError(
            "Stored typed workflow checkpoint cannot be restored to its declared state type"
        ) from exc


def _checkpoint_to_row(checkpoint: WorkflowCheckpoint) -> tuple:
    """Map a WorkflowCheckpoint to (session_id, state_data, metadata)."""
    parent_id = checkpoint.metadata.get("parent_id")
    metadata = CheckpointMetadata(
        checkpoint_id=checkpoint.checkpoint_id,
        session_id=_namespaced_session(checkpoint.thread_id),
        timestamp=datetime.fromtimestamp(checkpoint.timestamp, tz=timezone.utc),
        stage="graph_node",
        tool_count=0,
        message_count=0,
        parent_id=parent_id if isinstance(parent_id, str) else None,
        description=None,
        tags=[_WORKFLOW_TAG],
        version=1,
    )
    # The full WorkflowCheckpoint dict (state + metadata incl. next_node) is
    # the payload: serialization stays backend-owned and the resume position
    # round-trips without schema changes.
    payload = checkpoint.to_dict()
    payload["state"] = _encode_state(payload["state"])
    return metadata.session_id, payload, metadata


def _row_to_checkpoint(state_data: dict, fallback_thread: str) -> WorkflowCheckpoint:
    try:
        state = _decode_state(state_data["state"])
        return WorkflowCheckpoint.from_dict({**state_data, "state": state})
    except (KeyError, TypeError) as e:
        raise ValueError(
            f"Stored workflow checkpoint for thread {fallback_thread!r} has an "
            f"incompatible payload (state_schema/version mismatch): {e}"
        ) from e


class GraphSQLiteCheckpointer(CheckpointerProtocol):
    """Durable ``CheckpointerProtocol`` implementation over ``SQLiteCheckpointBackend``.

    Satisfies the framework's three-method protocol while delegating all
    persistence (schema, compression, checksums, fork columns, cleanup) to the
    existing conversation-checkpoint store.

    Retention (FEP-0047):
    - ``max_per_thread`` prunes the thread to its newest N checkpoints on every
      save (per-thread disk bound).
    - ``retention_days`` runs a one-time-per-instance sweep that deletes aged
      ``wf::`` rows (global disk bound — thread ids are per-run, so per-thread
      retention alone does not bound total rows). Scoped to the workflow
      keyspace; conversation checkpoints are never touched.
    """

    def __init__(
        self,
        backend: SQLiteCheckpointBackend,
        max_per_thread: Optional[int] = None,
        retention_days: Optional[int] = None,
    ):
        self._backend = backend
        self._max_per_thread = max_per_thread
        self._retention_days = retention_days
        self._ttl_swept = False

    @property
    def backend(self) -> SQLiteCheckpointBackend:
        return self._backend

    async def _sweep_aged_checkpoints(self) -> None:
        if self._ttl_swept or not self._retention_days or self._retention_days <= 0:
            return
        self._ttl_swept = True
        cutoff = datetime.now(timezone.utc) - timedelta(days=self._retention_days)
        deleted = await self._backend.purge_aged_checkpoints(WORKFLOW_SESSION_PREFIX, cutoff)
        if deleted:
            logger.info(f"Workflow checkpoint TTL sweep removed {deleted} aged rows")

    async def save(self, checkpoint: WorkflowCheckpoint) -> None:
        session_id, state_data, metadata = _checkpoint_to_row(checkpoint)
        # Snapshot at save: the caller's live state keeps mutating after this
        # point, so by-reference storage would retroactively rewrite earlier
        # checkpoints through shared nested dicts (MemoryCheckpointer lesson).
        # FOU-D5f: strip runtime handles (context["_execution_context"]) from
        # the persisted state — they carry futures/tasks that defeat both
        # orjson serialization and pickle, and are not checkpoint data.
        from victor.framework.graph_checkpoint import tolerant_deepcopy

        state_data = tolerant_deepcopy(state_data)
        await self._backend.save_checkpoint(
            session_id=session_id,
            state_data=state_data,
            metadata=metadata,
        )
        if self._max_per_thread and self._max_per_thread > 0:
            await self._backend.cleanup_old_checkpoints(session_id, keep_count=self._max_per_thread)
        await self._sweep_aged_checkpoints()

    async def load(self, thread_id: str) -> Optional[WorkflowCheckpoint]:
        session_id = _namespaced_session(thread_id)
        metas = await self._backend.list_checkpoints(session_id, limit=1)
        if not metas:
            return None
        data = await self._backend.load_checkpoint(metas[0].checkpoint_id)
        checkpoint = _row_to_checkpoint(data.state_data, thread_id)
        # Copy on load too: callers mutate the resumed state in place.
        return copy.deepcopy(checkpoint)

    async def list(self, thread_id: str) -> List[WorkflowCheckpoint]:
        session_id = _namespaced_session(thread_id)
        result: List[WorkflowCheckpoint] = []
        offset = 0
        # The backend orders by timestamp DESC; the protocol's contract (and
        # MemoryCheckpointer) is insertion order, so hydrate then reverse.
        while True:
            metas = await self._backend.list_checkpoints(
                session_id, limit=_LIST_PAGE_SIZE, offset=offset
            )
            if not metas:
                break
            for meta in metas:
                data = await self._backend.load_checkpoint(meta.checkpoint_id)
                result.append(_row_to_checkpoint(data.state_data, thread_id))
            if len(metas) < _LIST_PAGE_SIZE:
                break
            offset += _LIST_PAGE_SIZE
        result.reverse()
        return result


class ProjectDatabaseGraphCheckpointer(CheckpointerProtocol):
    """Store workflow checkpoints in a versioned project-database table.

    The default database is the canonical project ``project.db``.  An explicit
    path is a deployment override for measured write contention or portability;
    it uses the same schema and never changes checkpoint semantics.
    """

    _SCHEMA_VERSION = 1
    _EXPECTED_COLUMNS = frozenset(
        {
            "sequence",
            "checkpoint_id",
            "thread_id",
            "schema_version",
            "completed_node",
            "resume_target",
            "state_type",
            "durable_state",
            "checkpoint_metadata",
            "created_at",
        }
    )

    def __init__(
        self,
        db_path: Path,
        *,
        max_per_thread: Optional[int] = None,
        retention_days: Optional[int] = None,
    ):
        self.db_path = Path(db_path)
        self.max_per_thread = max_per_thread
        self.retention_days = retention_days
        self._connection: Optional[sqlite3.Connection] = None
        self._lock = threading.Lock()
        self._ttl_swept = False
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="victor-workflow-checkpoint"
        )
        weakref.finalize(self, self._executor.shutdown)

    def _connect(self) -> sqlite3.Connection:
        if self._connection is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self.db_path, timeout=5.0, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS workflow_checkpoint (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    checkpoint_id TEXT NOT NULL UNIQUE,
                    thread_id TEXT NOT NULL,
                    schema_version INTEGER NOT NULL,
                    completed_node TEXT,
                    resume_target TEXT,
                    state_type TEXT,
                    durable_state TEXT NOT NULL,
                    checkpoint_metadata TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_workflow_checkpoint_thread_sequence "
                "ON workflow_checkpoint(thread_id, sequence)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_workflow_checkpoint_thread_created "
                "ON workflow_checkpoint(thread_id, created_at)"
            )
            columns = {row[1] for row in conn.execute("PRAGMA table_info(workflow_checkpoint)")}
            missing = self._EXPECTED_COLUMNS - columns
            if missing:
                conn.close()
                raise ValueError(
                    "Existing workflow_checkpoint table is incompatible with schema "
                    f"version {self._SCHEMA_VERSION}; missing columns: {sorted(missing)}"
                )
            conn.commit()
            self._connection = conn
        return self._connection

    @staticmethod
    def _encode_durable_state(state: Any) -> tuple[Any, Optional[str]]:
        from victor.framework.graph_checkpoint import strip_runtime_handles

        encoded = _encode_state(strip_runtime_handles(state))
        if isinstance(encoded, dict) and encoded.get(_MODEL_STATE_MARKER) is True:
            return encoded["data"], encoded["type"]
        return encoded, None

    @staticmethod
    def _dump_json(value: Any, field: str) -> str:
        try:
            return json.dumps(value, allow_nan=False, separators=(",", ":"))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Workflow checkpoint {field} is not strict JSON-safe; live handles, "
                "streams, credentials, and raw provider responses must not be checkpointed"
            ) from exc

    def _save_sync(self, checkpoint: WorkflowCheckpoint) -> None:
        with self._lock:
            conn = self._connect()
            durable_state, state_type = self._encode_durable_state(checkpoint.state)
            metadata = dict(checkpoint.metadata)
            if checkpoint.next_node is not None:
                metadata["next_node"] = checkpoint.next_node
            if metadata.get("checkpoint_boundary") == "before":
                # An interrupt-before boundary has not completed this node.
                # Keep the pending node as explicit resume metadata only.
                metadata["interrupt_node"] = checkpoint.node_id
            try:
                conn.execute(
                    """
                    INSERT INTO workflow_checkpoint (
                        checkpoint_id, thread_id, schema_version, completed_node,
                        resume_target, state_type, durable_state, checkpoint_metadata,
                        created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        checkpoint.checkpoint_id,
                        checkpoint.thread_id,
                        self._SCHEMA_VERSION,
                        (
                            None
                            if metadata.get("checkpoint_boundary") == "before"
                            else checkpoint.node_id
                        ),
                        checkpoint.next_node,
                        state_type,
                        self._dump_json(durable_state, "state"),
                        self._dump_json(metadata, "metadata"),
                        checkpoint.timestamp,
                    ),
                )
                if self.max_per_thread is not None and self.max_per_thread > 0:
                    conn.execute(
                        """
                        DELETE FROM workflow_checkpoint
                        WHERE thread_id = ?
                          AND sequence NOT IN (
                              SELECT sequence FROM workflow_checkpoint
                              WHERE thread_id = ?
                              ORDER BY sequence DESC
                              LIMIT ?
                          )
                        """,
                        (checkpoint.thread_id, checkpoint.thread_id, self.max_per_thread),
                    )
                if (
                    not self._ttl_swept
                    and self.retention_days is not None
                    and self.retention_days > 0
                ):
                    self._ttl_swept = True
                    cutoff = datetime.now(timezone.utc).timestamp() - (self.retention_days * 86400)
                    conn.execute("DELETE FROM workflow_checkpoint WHERE created_at < ?", (cutoff,))
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def _row_to_checkpoint(self, row: sqlite3.Row) -> WorkflowCheckpoint:
        if row["schema_version"] != self._SCHEMA_VERSION:
            raise ValueError(
                "Unsupported workflow checkpoint schema version "
                f"{row['schema_version']} (expected {self._SCHEMA_VERSION})"
            )
        try:
            state_value = json.loads(row["durable_state"])
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("Stored workflow checkpoint state is not valid JSON") from exc
        state_type = row["state_type"]
        if state_type is not None:
            state_value = {
                _MODEL_STATE_MARKER: True,
                "type": state_type,
                "data": state_value,
            }
        try:
            metadata = json.loads(row["checkpoint_metadata"])
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("Stored workflow checkpoint metadata is not valid JSON") from exc
        return WorkflowCheckpoint(
            checkpoint_id=row["checkpoint_id"],
            thread_id=row["thread_id"],
            node_id=row["completed_node"] or metadata.get("interrupt_node") or "",
            state=_decode_state(state_value),
            timestamp=row["created_at"],
            metadata=metadata,
        )

    def _load_sync(self, thread_id: str) -> Optional[WorkflowCheckpoint]:
        with self._lock:
            row = (
                self._connect()
                .execute(
                    """
                SELECT * FROM workflow_checkpoint
                WHERE thread_id = ?
                ORDER BY sequence DESC
                LIMIT 1
                """,
                    (thread_id,),
                )
                .fetchone()
            )
            return None if row is None else copy.deepcopy(self._row_to_checkpoint(row))

    def _list_sync(self, thread_id: str) -> List[WorkflowCheckpoint]:
        with self._lock:
            rows = (
                self._connect()
                .execute(
                    """
                SELECT * FROM workflow_checkpoint
                WHERE thread_id = ?
                ORDER BY sequence ASC
                """,
                    (thread_id,),
                )
                .fetchall()
            )
            return [copy.deepcopy(self._row_to_checkpoint(row)) for row in rows]

    async def _run(self, function, *args: Any) -> Any:
        return await asyncio.get_running_loop().run_in_executor(self._executor, function, *args)

    async def save(self, checkpoint: WorkflowCheckpoint) -> None:
        await self._run(self._save_sync, checkpoint)

    async def load(self, thread_id: str) -> Optional[WorkflowCheckpoint]:
        return await self._run(self._load_sync, thread_id)

    async def list(self, thread_id: str) -> List[WorkflowCheckpoint]:
        return await self._run(self._list_sync, thread_id)


def get_default_graph_checkpointer(settings: Any = None) -> Optional[Any]:
    """Resolve the default durable workflow checkpointer.

    With no override, checkpoints use the canonical project ``project.db``.
    ``workflow_checkpoint_db_path`` is an explicit deployment exception for
    measured contention, cache lifetime, or portability; it never selects the
    former silent global ``checkpoints.db`` default.
    """
    if settings is None:
        from victor.config.settings import load_settings

        settings = load_settings()

    checkpoint_settings = getattr(settings, "checkpoint", None)
    if checkpoint_settings is None or not checkpoint_settings.workflow_checkpoints_enabled:
        return None

    configured_path = getattr(checkpoint_settings, "workflow_checkpoint_db_path", None)
    if configured_path:
        db_path = Path(configured_path)
    else:
        from victor.config.settings import get_project_paths

        db_path = get_project_paths().project_db

    return ProjectDatabaseGraphCheckpointer(
        db_path,
        max_per_thread=getattr(checkpoint_settings, "workflow_checkpoint_max_per_thread", None),
        retention_days=getattr(checkpoint_settings, "workflow_checkpoint_retention_days", None),
    )
