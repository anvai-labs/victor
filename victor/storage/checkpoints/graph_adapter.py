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

import copy
import logging
from datetime import datetime, timezone
from typing import List, Optional

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
    return metadata.session_id, checkpoint.to_dict(), metadata


def _row_to_checkpoint(state_data: dict, fallback_thread: str) -> WorkflowCheckpoint:
    try:
        return WorkflowCheckpoint.from_dict(state_data)
    except (KeyError, TypeError) as e:
        raise ValueError(
            f"Stored workflow checkpoint for thread {fallback_thread!r} has an "
            f"incompatible payload (state_schema/version mismatch): {e}"
        ) from e


class GraphSQLiteCheckpointer(CheckpointerProtocol):
    """Durable ``CheckpointerProtocol`` implementation over ``SQLiteCheckpointBackend``.

    Satisfies the framework's three-method protocol while delegating all
    persistence (schema, compression, checksums, fork columns, cleanup) to the
    existing conversation-checkpoint store. Opt-in at this layer (FOU-D1);
    durable-by-default injection at the composition roots is FOU-D2.
    """

    def __init__(self, backend: SQLiteCheckpointBackend):
        self._backend = backend

    @property
    def backend(self) -> SQLiteCheckpointBackend:
        return self._backend

    async def save(self, checkpoint: WorkflowCheckpoint) -> None:
        session_id, state_data, metadata = _checkpoint_to_row(checkpoint)
        # Snapshot at save: the caller's live state keeps mutating after this
        # point, so by-reference storage would retroactively rewrite earlier
        # checkpoints through shared nested dicts (MemoryCheckpointer lesson).
        await self._backend.save_checkpoint(
            session_id=session_id,
            state_data=copy.deepcopy(state_data),
            metadata=metadata,
        )

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
