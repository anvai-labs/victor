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
import sys
import time
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import BaseModel, Field, PrivateAttr

from victor.framework.graph_checkpoint import WorkflowCheckpoint
from victor.storage.checkpoints.backends.sqlite_backend import SQLiteCheckpointBackend
from victor.storage.checkpoints.graph_adapter import (
    _MODEL_STATE_MARKER,
    _decode_state,
    GraphSQLiteCheckpointer,
    get_default_graph_checkpointer,
)
from victor.storage.checkpoints.protocol import (
    CheckpointMetadata,
    CheckpointNotFoundError,
)


def _factory_settings(tmp_path, enabled: bool):
    """Minimal settings stand-in for the factory (duck-typed: .checkpoint)."""
    from types import SimpleNamespace

    return SimpleNamespace(
        checkpoint=SimpleNamespace(
            workflow_checkpoints_enabled=enabled,
            workflow_checkpoint_db_path=str(tmp_path / "checkpoints.db"),
            workflow_checkpoint_max_per_thread=20,
            workflow_checkpoint_retention_days=7,
        )
    )


def _checkpoint(thread_id: str, node_id: str, seq: int) -> WorkflowCheckpoint:
    return WorkflowCheckpoint(
        checkpoint_id=f"cp-{thread_id}-{seq}",
        thread_id=thread_id,
        node_id=node_id,
        state={"counter": seq, "messages": [f"m{seq}"]},
        timestamp=time.time() + seq,
        metadata={"next_node": f"node_{seq + 1}", "parent_id": None},
    )


class _TypedCheckpointState(BaseModel):
    query: str
    context: dict[str, object] = Field(default_factory=dict)
    _runtime_handle: object | None = PrivateAttr(default=None)


async def test_pydantic_state_round_trips_typed_and_durable(tmp_path):
    """A typed agentic-loop state must resume as its model, not a repr string.

    Runtime-only private handles are not persisted with the model fields.
    """
    state = _TypedCheckpointState(query="resume me", context={"step": 2})
    state._runtime_handle = object()
    original = WorkflowCheckpoint(
        checkpoint_id="typed-cp",
        thread_id="typed-thread",
        node_id="node_a",
        state=state,
        timestamp=time.time(),
        metadata={"next_node": "node_b"},
    )
    writer = GraphSQLiteCheckpointer(SQLiteCheckpointBackend(storage_path=tmp_path))
    await writer.save(original)

    loaded = await GraphSQLiteCheckpointer(SQLiteCheckpointBackend(storage_path=tmp_path)).load(
        "typed-thread"
    )

    assert loaded is not None
    assert isinstance(loaded.state, _TypedCheckpointState)
    assert loaded.state.query == "resume me"
    assert loaded.state.context == {"step": 2}
    assert loaded.next_node == "node_b"
    assert loaded.state._runtime_handle is None


def test_typed_state_decoder_refuses_to_import_unloaded_modules():
    """Checkpoint data must not select an import target."""
    missing_type = "victor._checkpoint_missing_state_module.MissingState"

    with pytest.raises(ValueError, match="cannot be restored"):
        _decode_state(
            {
                _MODEL_STATE_MARKER: True,
                "type": missing_type,
                "data": {"query": "untrusted"},
            }
        )

    assert missing_type.rsplit(".", 1)[0] not in sys.modules


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


# ---------------------------------------------------------------------------
# FOU-D2: retention, TTL sweep, and the default-checkpointer factory (FEP-0047)
# ---------------------------------------------------------------------------


async def test_per_thread_retention_keeps_most_recent(tmp_path):
    """With max_per_thread set, each save prunes the thread to the newest N."""
    adapter = GraphSQLiteCheckpointer(
        SQLiteCheckpointBackend(storage_path=tmp_path), max_per_thread=3
    )
    for seq in range(5):
        await adapter.save(_checkpoint("thread-1", f"node_{seq}", seq))

    listed = await adapter.list("thread-1")
    assert [c.state["counter"] for c in listed] == [2, 3, 4]


async def test_retention_off_by_default_keeps_all(tmp_path):
    adapter = GraphSQLiteCheckpointer(SQLiteCheckpointBackend(storage_path=tmp_path))
    for seq in range(7):
        await adapter.save(_checkpoint("thread-1", f"node_{seq}", seq))
    assert len(await adapter.list("thread-1")) == 7


async def test_ttl_sweep_purges_aged_workflow_rows_only(tmp_path):
    """The retention_days sweep removes aged wf:: rows on the next save and
    leaves conversation checkpoints (other keyspaces) untouched."""
    backend = SQLiteCheckpointBackend(storage_path=tmp_path)
    adapter = GraphSQLiteCheckpointer(backend, retention_days=1)

    # An aged workflow checkpoint (inserted directly with an old timestamp)
    old_meta = CheckpointMetadata(
        checkpoint_id="cp-old",
        session_id="wf::stale-thread",
        timestamp=datetime.now(timezone.utc) - timedelta(days=5),
        stage="graph_node",
        tool_count=0,
        message_count=0,
        tags=["workflow"],
    )
    await backend.save_checkpoint(
        session_id="wf::stale-thread", state_data={"old": True}, metadata=old_meta
    )
    # An aged conversation checkpoint that must survive the wf::-scoped sweep
    conv_meta = CheckpointMetadata(
        checkpoint_id="cp-conv",
        session_id="sess-conv",
        timestamp=datetime.now(timezone.utc) - timedelta(days=5),
        stage="chat",
        tool_count=0,
        message_count=0,
    )
    await backend.save_checkpoint(
        session_id="sess-conv", state_data={"conv": True}, metadata=conv_meta
    )

    # First save triggers the once-per-instance sweep
    await adapter.save(_checkpoint("fresh-thread", "node_a", 1))

    assert await backend.load_checkpoint("cp-conv") is not None
    with pytest.raises(CheckpointNotFoundError):
        await backend.load_checkpoint("cp-old")


def test_default_checkpointer_factory_disabled_returns_none(tmp_path, monkeypatch):
    settings = _factory_settings(tmp_path, enabled=False)
    assert get_default_graph_checkpointer(settings) is None


def test_default_checkpointer_factory_enabled_returns_sqlite_adapter(tmp_path):
    settings = _factory_settings(tmp_path, enabled=True)
    checkpointer = get_default_graph_checkpointer(settings)
    assert isinstance(checkpointer, GraphSQLiteCheckpointer)
    assert checkpointer.backend.db_path == tmp_path / "checkpoints.db"


def test_default_checkpointer_factory_settings_defaults(tmp_path):
    from victor.config.checkpoint_settings import CheckpointSettings

    cp = CheckpointSettings()
    assert cp.workflow_checkpoints_enabled is True
    assert cp.workflow_checkpoint_max_per_thread == 20
    assert cp.workflow_checkpoint_retention_days == 7
    # Flat keys route to the checkpoint group, so VICTOR_WORKFLOW_* env
    # overrides resolve through Settings construction.
    from victor.config.settings import Settings

    routed = Settings(
        **{
            "workflow_checkpoints_enabled": False,
            "workflow_checkpoint_max_per_thread": 5,
            "workflow_checkpoint_retention_days": 3,
            "workflow_checkpoint_db_path": "/tmp/cp.db",
        }
    )
    assert routed.checkpoint.workflow_checkpoints_enabled is False
    assert routed.checkpoint.workflow_checkpoint_max_per_thread == 5
    assert routed.checkpoint.workflow_checkpoint_retention_days == 3
    assert routed.checkpoint.workflow_checkpoint_db_path == "/tmp/cp.db"


def test_default_checkpointer_factory_missing_aiosqlite_degrades_loudly(
    tmp_path, monkeypatch, caplog
):
    import logging

    import victor.storage.checkpoints.backends.sqlite_backend as backend_module

    monkeypatch.setattr(backend_module, "aiosqlite", None)
    settings = _factory_settings(tmp_path, enabled=True)
    with caplog.at_level(logging.WARNING):
        # None = the framework in-memory default (compile(checkpointer=None));
        # the layering ratchet forbids importing the concrete class here.
        checkpointer = get_default_graph_checkpointer(settings)
    assert checkpointer is None
    assert any("aiosqlite" in r.message.lower() for r in caplog.records)
    # Second call stays silent (warn once, not per construction)
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        get_default_graph_checkpointer(settings)
    assert not any("aiosqlite" in r.message.lower() for r in caplog.records)
