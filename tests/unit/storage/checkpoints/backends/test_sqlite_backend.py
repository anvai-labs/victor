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

"""Direct tests for SQLiteCheckpointBackend (mirror suite for the backend module).

Pins the save/load/list contract and the concurrent-writer pragmas added in
FOU-D1 (WAL journal mode + busy timeout) that workflow-graph checkpoints
(``GraphSQLiteCheckpointer``) rely on when sharing one database file with the
conversation checkpoint manager.
"""

from datetime import datetime, timezone

import pytest

from victor.storage.checkpoints.backends.sqlite_backend import SQLiteCheckpointBackend
from victor.storage.checkpoints.protocol import CheckpointNotFoundError, CheckpointMetadata


def _metadata(checkpoint_id: str, session_id: str) -> CheckpointMetadata:
    return CheckpointMetadata(
        checkpoint_id=checkpoint_id,
        session_id=session_id,
        timestamp=datetime.now(timezone.utc),
        stage="test",
        tool_count=0,
        message_count=0,
    )


@pytest.fixture
def backend(tmp_path):
    return SQLiteCheckpointBackend(storage_path=tmp_path)


async def test_save_load_roundtrip(backend):
    checkpoint_id = await backend.save_checkpoint(
        session_id="sess-1", state_data={"k": "v"}, metadata=_metadata("cp-1", "sess-1")
    )
    assert checkpoint_id == "cp-1"

    loaded = await backend.load_checkpoint("cp-1")
    assert loaded.state_data == {"k": "v"}
    assert loaded.metadata.checkpoint_id == "cp-1"


async def test_load_missing_raises_not_found(backend):
    with pytest.raises(CheckpointNotFoundError):
        await backend.load_checkpoint("no-such-id")


async def test_list_orders_newest_first(backend):
    for seq in range(3):
        await backend.save_checkpoint(
            session_id="sess-1",
            state_data={"seq": seq},
            metadata=_metadata(f"cp-{seq}", "sess-1"),
        )
    listed = await backend.list_checkpoints("sess-1")
    assert [m.checkpoint_id for m in listed] == ["cp-2", "cp-1", "cp-0"]


async def test_wal_mode_and_busy_timeout_are_persistent_properties(backend):
    """FOU-D1: concurrent writers (workflow adapter + conversation manager on
    one db file) need WAL and a bounded lock wait. Both pragmas are persistent
    database properties, so they must be observable after initialization."""
    await backend._ensure_initialized()

    try:
        import aiosqlite
    except ImportError:  # pragma: no cover - aiosqlite is a test dependency
        pytest.fail("aiosqlite missing; backend construction should have raised")

    async with aiosqlite.connect(backend.db_path) as db:
        cursor = await db.execute("PRAGMA journal_mode")
        (mode,) = await cursor.fetchone()
        assert str(mode).lower() == "wal"

        cursor = await db.execute("PRAGMA busy_timeout")
        (timeout,) = await cursor.fetchone()
        assert int(timeout) == 5000


async def test_reinitialization_is_idempotent(backend):
    await backend._ensure_initialized()
    await backend._ensure_initialized()  # must not re-execute or raise
    # And the schema still serves writes after the second call
    checkpoint_id = await backend.save_checkpoint(
        session_id="sess-1", state_data={"k": 1}, metadata=_metadata("cp-ok", "sess-1")
    )
    assert checkpoint_id == "cp-ok"
