"""Concurrency regression tests for SqliteGraphStore's dedicated worker thread.

Co-design review item 20b: every async method used to run its synchronous
SQLite work directly on the event loop, so a large ``write_batch`` hogged the
loop for its whole duration and starved every other coroutine. The store now
routes all synchronous DB work through a dedicated single-worker
``ThreadPoolExecutor`` (single-worker, NOT ``asyncio.to_thread`` — the shared
pool would open one ``threading.local()`` connection per pool thread and
worsen WAL write-lock contention).

The starvation test below fails near-deterministically on the pre-fix code:
the batch coroutine never awaits mid-batch, so a heartbeat task scheduled
before the batch cannot run until the batch finishes. Post-fix, every
interior write awaits an executor round-trip, giving the loop its chance to
service the heartbeat.
"""

from pathlib import Path

import pytest

from victor.core.database import reset_project_database
from victor.storage.graph.protocol import GraphNode
from victor.storage.graph.sqlite_store import SqliteGraphStore


def _node(i: int) -> GraphNode:
    return GraphNode(node_id=f"file:f{i}.py", type="file", name=f"f{i}.py", file=f"f{i}.py")


@pytest.mark.asyncio
async def test_heartbeat_runs_during_large_write_batch(tmp_path: Path) -> None:
    """A write_batch must not starve other coroutines on the event loop.

    Starts a heartbeat task (a single ``await asyncio.sleep(0)``) before a
    write_batch of many interior upserts, and asserts mid-batch — after the
    batch's first interior write — that the heartbeat has run. On the
    pre-fix code the interior writes are fully synchronous, the batch
    coroutine never yields, and this assert fails.
    """
    db_path = tmp_path / "project.db"
    store = SqliteGraphStore(db_path)
    heartbeat_ran = False

    async def _heartbeat() -> None:
        nonlocal heartbeat_ran
        await asyncio.sleep(0)
        heartbeat_ran = True

    chunks = [
        [_node(i * 10 + j) for j in range(10)] for i in range(200)
    ]  # 2000 nodes via 200 interior writes

    async def _batch() -> None:
        nonlocal heartbeat_ran
        async with store.write_batch():
            first = True
            for chunk in chunks:
                await store.upsert_nodes(chunk)
                if first:
                    # The first interior write's executor round-trip yields
                    # the loop; from here on the heartbeat must have run.
                    first = False
                    continue
                assert heartbeat_ran, (
                    "event loop starved during write_batch: interior writes "
                    "are running synchronously on the loop"
                )
            assert heartbeat_ran

    import asyncio

    await asyncio.gather(_heartbeat(), _batch())

    stats = await store.stats()
    assert stats["nodes"] == 2000
    await store.close()
    reset_project_database(db_path)


@pytest.mark.asyncio
async def test_interior_batch_writes_share_one_transaction(tmp_path: Path) -> None:
    """Interior batch writes must all land on the batch's BEGIN IMMEDIATE
    connection (resolved via task identity on the loop, then handed to the
    worker explicitly) — a failure mid-batch rolls back everything.
    """
    db_path = tmp_path / "project.db"
    store = SqliteGraphStore(db_path)

    with pytest.raises(RuntimeError, match="boom"):
        async with store.write_batch():
            await store.upsert_nodes([_node(1), _node(2)])
            await store.upsert_nodes([_node(3)])
            raise RuntimeError("boom")

    stats = await store.stats()
    assert stats["nodes"] == 0, "interior writes must roll back with the batch"

    await store.close()
    reset_project_database(db_path)


@pytest.mark.asyncio
async def test_concurrent_store_cannot_commit_into_our_batch(tmp_path: Path) -> None:
    """Adversarial-review F1 regression: a second store instance (sharing the
    same project.db via the per-path manager singleton) must NOT be able to
    commit a partial batch out from under the batch owner.

    Pre-fix, both stores shared the manager's single thread-local connection;
    a co-writer's commit mid-batch also committed the batch's partial writes,
    so a failed batch left durable partial state. With per-store dedicated
    connections, the co-writer instead serializes on SQLite's write lock: its
    write proceeds only after the batch's rollback releases it, and the final
    state contains exactly the co-writer's rows.
    """
    import asyncio

    db_path = tmp_path / "project.db"
    store_a = SqliteGraphStore(db_path)
    store_b = SqliteGraphStore(db_path)

    async def _failing_batch() -> None:
        async with store_a.write_batch():
            for i in range(20):
                await store_a.upsert_nodes([_node(i)])
            raise RuntimeError("boom")

    async def _co_writer() -> None:
        # Issued while store_a's batch holds the write lock; with dedicated
        # connections this blocks on SQLite's busy handler until the rollback
        # releases the lock, then commits its own rows.
        await store_b.upsert_nodes([_node(900), _node(901)])

    results = await asyncio.gather(_failing_batch(), _co_writer(), return_exceptions=True)
    # The batch's own failure is intentional; the co-writer must not error.
    unexpected = [
        r
        for r in results
        if isinstance(r, BaseException) and not (isinstance(r, RuntimeError) and "boom" in str(r))
    ]
    assert (
        not unexpected
    ), f"no side should error under the dedicated-connection model: {unexpected!r}"

    stats = await store_a.stats()
    assert stats["nodes"] == 2, (
        "the failed batch must roll back fully and the co-writer's rows must "
        "be the only survivors — a partial batch must never be committed by "
        "another store's writer"
    )
    found = await store_a.find_nodes(name="f900.py")
    assert [n.node_id for n in found] == ["file:f900.py"]

    await store_a.close()
    await store_b.close()
    reset_project_database(db_path)


@pytest.mark.asyncio
async def test_reads_and_writes_interleave_through_worker(tmp_path: Path) -> None:
    """A read issued while a batch holds the lock still completes once the
    batch ends, and both paths produce consistent data through the same
    worker connection."""
    import asyncio

    db_path = tmp_path / "project.db"
    store = SqliteGraphStore(db_path)

    async def _batch() -> None:
        async with store.write_batch():
            for i in range(20):
                await store.upsert_nodes([_node(i)])

    async def _reader() -> list:
        # Issued while the batch is running; serialized by the store lock.
        await asyncio.sleep(0)
        return await store.find_nodes(type="file")

    _, found = await asyncio.gather(_batch(), _reader())
    assert len(found) == 20

    await store.close()
    reset_project_database(db_path)


@pytest.mark.asyncio
async def test_close_shuts_down_worker_without_hanging(tmp_path: Path) -> None:
    """close() must shut down the dedicated executor and close the dedicated
    connection. The store creates both lazily, so constructing-and-closing
    without any DB op never spawns a thread or opens one."""
    import asyncio
    import threading

    db_path = tmp_path / "project.db"
    store = SqliteGraphStore(db_path)
    # __init__ runs _ensure_schema via the dedicated connection, so it exists
    # post-construction, but the executor must still be lazy.
    assert store._db_executor is None, "executor must be lazy"

    await store.upsert_nodes([_node(1)])
    assert store._db_executor is not None
    worker_threads = [t for t in threading.enumerate() if t.name.startswith("sqlite-graph")]
    assert len(worker_threads) == 1

    await store.close()
    assert store._conn is None, "close() must release the dedicated connection"
    await asyncio.sleep(0.05)
    assert not [
        t for t in threading.enumerate() if t.name.startswith("sqlite-graph")
    ], "close() must stop the dedicated worker thread"

    # Ops after close() still work: the connection lazily reopens (matching
    # the pre-change behavior where close() never invalidated anything).
    await store.upsert_nodes([_node(2)])
    stats = await store.stats()
    assert stats["nodes"] == 2

    await store.close()
    reset_project_database(db_path)


@pytest.mark.parametrize("phase", ["begin", "mutation", "commit", "rollback"])
async def test_batch_cancellation_settles_before_next_owner(tmp_path, phase, caplog):
    """Cancellation never hands an active transaction to an unrelated writer."""
    import asyncio
    import sqlite3
    import threading

    store = SqliteGraphStore(tmp_path)
    started = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()

    def gate(operation):
        if operation == phase:
            loop.call_soon_threadsafe(started.set)
            assert release.wait(5), "worker was not released"

    class ControlledConnection(sqlite3.Connection):
        def execute(self, sql, parameters=()):
            result = super().execute(sql, parameters)
            if sql == "BEGIN IMMEDIATE":
                gate("begin")
            return result

        def executemany(self, sql, parameters):
            result = super().executemany(sql, parameters)
            gate("mutation")
            return result

        def commit(self):
            super().commit()
            gate("commit")  # Prove cancellation may follow an actual commit.

        def rollback(self):
            super().rollback()
            gate("rollback")

    store._conn.close()
    conn = sqlite3.connect(store.db_path, factory=ControlledConnection, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    store._db._configure_connection(conn)
    store._conn = conn

    async def batch():
        async with store.write_batch():
            async with store.write_batch():
                await store.upsert_nodes([_node(1)])
            if phase == "rollback":
                raise RuntimeError("request rollback")

    task = asyncio.create_task(batch())
    later = None
    try:
        await asyncio.wait_for(started.wait(), 2)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()  # Cleanup must survive another cancellation request.
        await asyncio.sleep(0)
        later = asyncio.create_task(store.upsert_nodes([_node(99)]))
        await asyncio.sleep(0)
        terminal_before_settlement = task.done()
        later_before_settlement = later.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        await later
        assert not terminal_before_settlement
        assert not later_before_settlement
        assert store._conn is not None and not store._conn.in_transaction
        if phase == "commit":
            assert "may have succeeded" in caplog.text
        # A separate connection observes durable effects, not uncommitted reads.
        with sqlite3.connect(store.db_path) as observer:
            ids = {row[0] for row in observer.execute("SELECT node_id FROM graph_node")}
        assert ids == ({"file:f1.py", "file:f99.py"} if phase == "commit" else {"file:f99.py"})
    finally:
        release.set()
        await asyncio.gather(task, *([later] if later else []), return_exceptions=True)
        await store.close()
        reset_project_database(tmp_path)


async def test_worker_failure_after_cancellation_is_observable(tmp_path):
    import asyncio
    import threading

    store = SqliteGraphStore(tmp_path)
    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    release = threading.Event()

    def fail():
        loop.call_soon_threadsafe(started.set)
        assert release.wait(5)
        raise RuntimeError("worker failed after cancellation")

    task = asyncio.create_task(store._run_db(fail))
    try:
        await asyncio.wait_for(started.wait(), 2)
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError) as error:
            await task
        assert isinstance(error.value.__cause__, RuntimeError)
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await store.close()
        reset_project_database(tmp_path)


@pytest.mark.parametrize("operation", ["initialize", "close"])
async def test_lifecycle_call_by_batch_owner_is_rejected(tmp_path, operation):
    store = SqliteGraphStore(tmp_path)
    try:
        with pytest.raises(RuntimeError, match="active write batch"):
            async with store.write_batch():
                await store.upsert_nodes([_node(1)])
                await getattr(store, operation)()
        assert await store.get_node_by_id("file:f1.py") is None
    finally:
        await store.close()
        reset_project_database(tmp_path)


@pytest.mark.parametrize("operation", ["initialize", "close"])
async def test_lifecycle_waits_for_batch_owner(tmp_path, operation):
    import asyncio

    store = SqliteGraphStore(tmp_path)
    started = asyncio.Event()
    release = asyncio.Event()

    async def batch():
        async with store.write_batch():
            await store.upsert_nodes([_node(1)])
            started.set()
            await release.wait()
            raise RuntimeError("rollback owner")

    task = asyncio.create_task(batch())
    lifecycle = None
    try:
        await asyncio.wait_for(started.wait(), 2)
        lifecycle = asyncio.create_task(getattr(store, operation)())
        await asyncio.sleep(0.02)
        premature = lifecycle.done()
        release.set()
        batch_result, lifecycle_result = await asyncio.gather(
            task, lifecycle, return_exceptions=True
        )
        assert isinstance(batch_result, RuntimeError)
        assert str(batch_result) == "rollback owner"
        assert lifecycle_result is None
        assert not premature
        if operation == "close":
            assert store._conn is None
            assert store._db_executor is None
        assert await store.get_node_by_id("file:f1.py") is None
    finally:
        release.set()
        await asyncio.gather(task, *([lifecycle] if lifecycle else []), return_exceptions=True)
        await store.close()
        reset_project_database(tmp_path)


@pytest.mark.parametrize("many", [False, True])
async def test_metadata_cancellation_rolls_back_before_later_write(tmp_path, monkeypatch, many):
    import asyncio

    store = SqliteGraphStore(tmp_path)
    await store.upsert_nodes([_node(1)])
    original = store._run_db
    cancelled = False

    async def cancel_after_apply(fn):
        nonlocal cancelled
        result = await original(fn)
        # Cancel only after metadata has actually changed inside the transaction.
        metadata = store._conn.execute(
            "SELECT metadata FROM graph_node WHERE node_id = ?", ("file:f1.py",)
        ).fetchone()[0]
        if not cancelled and '"new"' in metadata:
            cancelled = True
            raise asyncio.CancelledError()
        return result

    monkeypatch.setattr(store, "_run_db", cancel_after_apply)
    try:
        with pytest.raises(asyncio.CancelledError):
            if many:
                await store.update_nodes_metadata([("file:f1.py", {"new": True})])
            else:
                await store.update_node_metadata("file:f1.py", {"new": True})
        await store.upsert_nodes([_node(2)])
        assert (await store.get_node_by_id("file:f1.py")).metadata == {}
    finally:
        monkeypatch.setattr(store, "_run_db", original)
        await store.close()
        reset_project_database(tmp_path)


async def test_failed_rollback_discards_connection_before_reuse(tmp_path):
    import sqlite3
    import threading

    store = SqliteGraphStore(tmp_path)
    closed_on = []

    class FailedRollback(sqlite3.Connection):
        def rollback(self):
            raise sqlite3.OperationalError("rollback failed")

        def close(self):
            closed_on.append(threading.get_ident())
            super().close()

    store._conn.close()
    conn = sqlite3.connect(store.db_path, factory=FailedRollback, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    store._db._configure_connection(conn)
    store._conn = conn
    try:
        with pytest.raises(sqlite3.OperationalError, match="rollback failed"):
            async with store.write_batch():
                await store.upsert_nodes([_node(1)])
                raise RuntimeError("abort batch")
        assert store._conn is None
        assert closed_on and closed_on[0] != threading.get_ident()
        await store.upsert_nodes([_node(99)])
        with sqlite3.connect(store.db_path) as observer:
            assert list(observer.execute("SELECT node_id FROM graph_node")) == [("file:f99.py",)]
    finally:
        await store.close()
        reset_project_database(tmp_path)


async def test_delete_file_can_join_own_batch_without_reacquiring_lock(tmp_path, monkeypatch):
    import asyncio
    from unittest.mock import AsyncMock

    store = SqliteGraphStore(tmp_path)
    cleanup = AsyncMock()
    monkeypatch.setattr(store, "_delete_embeddings_for_file", cleanup)
    try:
        await store.upsert_nodes([_node(1)])
        with pytest.raises(RuntimeError, match="abort deletion"):
            async with asyncio.timeout(2):
                async with store.write_batch():
                    await store.delete_by_file("f1.py")
                    assert await store.get_nodes_by_file("f1.py") == []
                    raise RuntimeError("abort deletion")
        assert await store.get_node_by_id("file:f1.py") is not None
        # Vector cleanup remains outside the SQLite transaction (DATA-05).
        cleanup.assert_awaited_once_with("f1.py", ["file:f1.py"])
    finally:
        await store.close()
        reset_project_database(tmp_path)


@pytest.mark.parametrize("cancel_close", [False, True])
async def test_worker_close_settles_before_edge_probe(tmp_path, monkeypatch, cancel_close):
    import asyncio
    import sqlite3
    import threading

    from victor.storage.graph.protocol import GraphEdge

    store = SqliteGraphStore(tmp_path)
    started = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()

    class ControlledClose(sqlite3.Connection):
        def close(self):
            loop.call_soon_threadsafe(started.set)
            assert release.wait(5)
            super().close()

    await store.upsert_nodes([_node(1), _node(2)])
    store._conn.close()
    conn = sqlite3.connect(store.db_path, factory=ControlledClose, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    store._db._configure_connection(conn)
    store._conn = conn
    original = store._run_db
    edge_probe_dispatched = False
    edge_task = None

    async def observe_dispatch(fn):
        nonlocal edge_probe_dispatched
        if asyncio.current_task() is edge_task:
            edge_probe_dispatched = True
        return await original(fn)

    monkeypatch.setattr(store, "_run_db", observe_dispatch)
    closing = asyncio.create_task(store.close())
    try:
        await asyncio.wait_for(started.wait(), 2)
        if cancel_close:
            closing.cancel()
            await asyncio.sleep(0)
            closing.cancel()
        edge_task = asyncio.create_task(
            store.upsert_edges([GraphEdge(src="file:f1.py", dst="file:f2.py", type="calls")])
        )
        await asyncio.sleep(0)
        premature_close = closing.done()
        premature_probe = edge_probe_dispatched
        release.set()
        results = await asyncio.gather(closing, edge_task, return_exceptions=True)
        assert not premature_close
        assert not premature_probe, "schema work must wait for connection-close admission"
        if cancel_close:
            assert isinstance(results[0], asyncio.CancelledError)
        else:
            assert results[0] is None
        assert results[1] is None
        assert (await store.stats())["edges"] == 1
    finally:
        release.set()
        await asyncio.gather(closing, *([edge_task] if edge_task else []), return_exceptions=True)
        await store.close()
        reset_project_database(tmp_path)
