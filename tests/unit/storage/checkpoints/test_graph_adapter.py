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

"""Tests for GraphSQLiteCheckpointer (FOU-D1, FEP-0047).

Pins the CheckpointerProtocol adapter over SQLiteCheckpointBackend:
save → fresh adapter → load round-trip with ``next_node`` intact,
ordered ``list()``, namespaced ``wf::`` session keys, loud failure when
the optional ``aiosqlite`` dependency is missing, and concurrent writers
on one database file.
"""

import asyncio
import time

import pytest

from victor.framework.graph_checkpoint import WorkflowCheckpoint
from victor.storage.checkpoints.backends.sqlite_backend import SQLiteCheckpointBackend
from victor.storage.checkpoints.graph_adapter import GraphSQLiteCheckpointer


def _checkpoint(thread_id: str, node_id: str, seq: int) -> WorkflowCheckpoint:
    return WorkflowCheckpoint(
        checkpoint_id=f"cp-{thread_id}-{seq}",
        thread_id=thread_id,
        node_id=node_id,
        state={"counter": seq, "messages": [f"m{seq}"]},
        timestamp=time.time() + seq,
        metadata={"next_node": f"node_{seq + 1}", "parent_id": None},
    )


@pytest.fixture
def adapter(tmp_path):
    backend = SQLiteCheckpointBackend(storage_path=tmp_path)
    return GraphSQLiteCheckpointer(backend)


async def test_save_then_load_roundtrip_across_adapter_instances(tmp_path):
    """A checkpoint saved by one adapter loads deep-equal from a fresh instance.

    The fresh instance proves persistence (new in-memory state, same db file)
    rather than a shared cache. ``next_node`` (FEP-0032 resume-at) must survive.
    """
    writer = GraphSQLiteCheckpointer(SQLiteCheckpointBackend(storage_path=tmp_path))
    original = _checkpoint("thread-1", "node_a", 1)
    await writer.save(original)

    reader = GraphSQLiteCheckpointer(SQLiteCheckpointBackend(storage_path=tmp_path))
    loaded = await reader.load("thread-1")

    assert loaded is not None
    assert loaded.checkpoint_id == original.checkpoint_id
    assert loaded.thread_id == original.thread_id
    assert loaded.node_id == original.node_id
    assert loaded.state == original.state
    assert loaded.next_node == "node_2"

    # Mutation isolation on load: callers mutate the resumed state in place
    # (same contract as MemoryCheckpointer's copy-on-load).
    loaded.state["counter"] = 999
    reloaded = await reader.load("thread-1")
    assert reloaded.state["counter"] == 1


async def test_save_snapshots_state_at_save_time(tmp_path):
    """Mutating the live state after save must not rewrite the stored checkpoint
    (same by-reference trap MemoryCheckpointer's adversarial review caught)."""
    adapter = GraphSQLiteCheckpointer(SQLiteCheckpointBackend(storage_path=tmp_path))
    cp = _checkpoint("thread-1", "node_a", 1)
    await adapter.save(cp)

    cp.state["counter"] = 42
    cp.metadata["next_node"] = "mutated"

    loaded = await adapter.load("thread-1")
    assert loaded.state["counter"] == 1
    assert loaded.next_node == "node_2"


async def test_list_returns_checkpoints_oldest_first(tmp_path):
    adapter = GraphSQLiteCheckpointer(SQLiteCheckpointBackend(storage_path=tmp_path))
    for seq in range(3):
        await adapter.save(_checkpoint("thread-1", f"node_{seq}", seq))
    await adapter.save(_checkpoint("thread-2", "other", 10))

    listed = await adapter.list("thread-1")
    assert [c.checkpoint_id for c in listed] == [
        "cp-thread-1-0",
        "cp-thread-1-1",
        "cp-thread-1-2",
    ]
    assert all(c.thread_id == "thread-1" for c in listed)
    # Full state present, not just metadata
    assert listed[2].state == {"counter": 2, "messages": ["m2"]}


async def test_load_unknown_thread_returns_none_and_list_empty(tmp_path):
    adapter = GraphSQLiteCheckpointer(SQLiteCheckpointBackend(storage_path=tmp_path))
    assert await adapter.load("no-such-thread") is None
    assert await adapter.list("no-such-thread") == []


async def test_workflow_threads_namespaced_from_conversation_checkpoints(tmp_path):
    """Thread keys must not collide with conversation checkpoints in the same db:
    rows are stored under a ``wf::`` session keyspace (FEP-0047)."""
    adapter = GraphSQLiteCheckpointer(SQLiteCheckpointBackend(storage_path=tmp_path))
    await adapter.save(_checkpoint("sess-1", "node_a", 1))

    # Raw backend view: adapter rows are namespaced
    rows = await adapter._backend.list_checkpoints("wf::sess-1")
    assert len(rows) == 1
    # And invisible under the bare session id used by conversation checkpoints
    assert await adapter._backend.list_checkpoints("sess-1") == []


async def test_missing_aiosqlite_raises_actionable_error(tmp_path, monkeypatch):
    """Missing optional dependency must fail loudly with the install hint,
    never silently degrade inside the adapter (FEP-0047: degradation is the
    factory's documented job, not the adapter's)."""
    import victor.storage.checkpoints.backends.sqlite_backend as backend_module

    monkeypatch.setattr(backend_module, "aiosqlite", None)
    with pytest.raises(RuntimeError, match="aiosqlite"):
        GraphSQLiteCheckpointer(SQLiteCheckpointBackend(storage_path=tmp_path))


async def test_concurrent_writers_on_shared_database(tmp_path):
    """Two adapter instances writing interleaved threads on one db file must
    both persist (WAL/busy-timeout: concurrent sessions, FEP-0047 risk note)."""
    a = GraphSQLiteCheckpointer(SQLiteCheckpointBackend(storage_path=tmp_path))
    b = GraphSQLiteCheckpointer(SQLiteCheckpointBackend(storage_path=tmp_path))

    checkpoints = [_checkpoint(f"thread-{i}", "node_a", seq) for i in (1, 2) for seq in (0, 1)]
    writers = [a, a, b, b]
    await asyncio.gather(*(w.save(cp) for w, cp in zip(writers, checkpoints)))

    reader = GraphSQLiteCheckpointer(SQLiteCheckpointBackend(storage_path=tmp_path))
    for i in (1, 2):
        listed = await reader.list(f"thread-{i}")
        assert len(listed) == 2
