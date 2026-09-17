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
