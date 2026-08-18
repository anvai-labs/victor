# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
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

"""Deterministic release of embedded ProximaDB connections (victor#911).

Embedded servers are spawned setsid and outlive their parent, so a holder
that is dropped without releasing leaves a server running against the data
dir — where it keeps rewriting state, silently resurrecting data a caller
believed it had cleared, and accumulating across iterations (seven orphans
were found in one session).

`ProximaRepoConnection.release()` already stopped the subprocess on the last
release; nothing called it. These tests pin the seams that now do, and give
harnesses an assertable "am I clean?" signal instead of hope.
"""

import pytest

from victor.storage import proxima_runtime as rt


class FakeDB:
    def __init__(self):
        self.stopped = 0

    async def stop(self):
        self.stopped += 1


def _register(key: str) -> tuple[rt.ProximaRepoConnection, FakeDB]:
    """Put a fake live connection in the registry, as acquire() would."""
    conn = rt.ProximaRepoConnection.__new__(rt.ProximaRepoConnection)
    conn._key = key
    conn._db = FakeDB()
    conn._client = object()
    conn._graphs = {}
    conn._collections = {}
    conn._collection_generations = {}
    conn._refcount = 1
    rt._CONN_REGISTRY[key] = conn
    return conn, conn._db


@pytest.fixture(autouse=True)
def _clean_registry():
    rt._CONN_REGISTRY.clear()
    yield
    rt._CONN_REGISTRY.clear()


def test_live_connection_count_reports_registry_size():
    assert rt.live_connection_count() == 0
    _register("a")
    _register("b")
    assert (
        rt.live_connection_count() == 2
    ), "harnesses need an assertable signal, not a hope, that teardown worked"


@pytest.mark.asyncio
async def test_release_all_stops_every_server_and_empties_the_registry():
    _, db_a = _register("a")
    _, db_b = _register("b")

    await rt.release_all_connections()

    assert db_a.stopped == 1
    assert db_b.stopped == 1
    assert rt.live_connection_count() == 0


@pytest.mark.asyncio
async def test_release_all_is_idempotent():
    _, db = _register("a")
    await rt.release_all_connections()
    await rt.release_all_connections()
    assert db.stopped == 1, "a stopped server is not stopped twice"
    assert rt.live_connection_count() == 0


@pytest.mark.asyncio
async def test_release_all_never_raises_when_stop_fails():
    """Teardown runs on failure paths; it must not mask the original error."""

    class Exploding:
        async def stop(self):
            raise RuntimeError("boom")

    conn = rt.ProximaRepoConnection.__new__(rt.ProximaRepoConnection)
    conn._key = "x"
    conn._db = Exploding()
    conn._client = None
    conn._graphs = {}
    conn._collections = {}
    conn._collection_generations = {}
    conn._refcount = 1
    rt._CONN_REGISTRY["x"] = conn

    await rt.release_all_connections()  # must not raise
    assert rt.live_connection_count() == 0


@pytest.mark.asyncio
async def test_close_quietly_tolerates_missing_and_failing_stores():
    from victor.evaluation.swe_bench_loader import _close_quietly

    class NoClose:
        pass

    class Failing:
        async def close(self):
            raise RuntimeError("boom")

    class Ok:
        def __init__(self):
            self.closed = False

        async def close(self):
            self.closed = True

    await _close_quietly(None)
    await _close_quietly(NoClose())
    await _close_quietly(Failing())
    ok = Ok()
    await _close_quietly(ok)
    assert ok.closed, "a closable store must actually be closed"
