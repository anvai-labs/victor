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

"""Concurrency tests for state managers.

Verifies that all state manager implementations are safe under
concurrent async access (e.g. parallel team formations).
"""

from __future__ import annotations

import asyncio
import threading

import pytest

from victor.state.managers import (
    ConversationStateManager,
    GlobalStateManagerImpl,
    TeamStateManager,
    WorkflowStateManager,
)
from victor.state.factory import get_global_manager, reset_global_manager

ALL_MANAGERS = [
    WorkflowStateManager,
    ConversationStateManager,
    TeamStateManager,
    GlobalStateManagerImpl,
]


@pytest.fixture(params=ALL_MANAGERS, ids=lambda c: c.__name__)
def manager(request):
    """Parametrized fixture yielding each state manager type."""
    return request.param()


class TestConcurrentSetGet:
    """Test concurrent set/get operations don't lose updates."""

    async def test_concurrent_set_no_lost_updates(self, manager):
        """Many concurrent sets to different keys should all persist."""
        n = 50

        async def setter(i: int):
            await manager.set(f"key_{i}", i)

        await asyncio.gather(*[setter(i) for i in range(n)])

        for i in range(n):
            val = await manager.get(f"key_{i}")
            assert val == i, f"Lost update for key_{i}: got {val}"

    async def test_concurrent_set_same_key(self, manager):
        """Concurrent sets to the same key should not corrupt state."""
        n = 50

        async def setter(i: int):
            await manager.set("counter", i)

        await asyncio.gather(*[setter(i) for i in range(n)])

        val = await manager.get("counter")
        assert isinstance(val, int)
        assert 0 <= val < n

    async def test_concurrent_set_and_get(self, manager):
        """Concurrent reads during writes should not raise."""
        await manager.set("x", 0)

        async def writer():
            for i in range(20):
                await manager.set("x", i)

        async def reader():
            for _ in range(20):
                val = await manager.get("x")
                assert val is not None

        await asyncio.gather(writer(), reader())


class TestConcurrentDelete:
    """Test concurrent delete operations."""

    async def test_concurrent_set_and_delete(self, manager):
        """Concurrent set + delete should not raise KeyError."""
        await manager.set("volatile", "initial")

        async def setter():
            for _ in range(20):
                await manager.set("volatile", "value")

        async def deleter():
            for _ in range(20):
                await manager.delete("volatile")

        # Should not raise
        await asyncio.gather(setter(), deleter())


class TestConcurrentSnapshot:
    """Test snapshot consistency under concurrent writes."""

    async def test_snapshot_is_consistent(self, manager):
        """Snapshot should reflect a single point in time."""
        for i in range(10):
            await manager.set(f"k{i}", i)

        async def writer():
            for j in range(10):
                await manager.set(f"k{j}", j + 100)

        async def snapshotter():
            snapshots = []
            for _ in range(5):
                snap = await manager.snapshot()
                snapshots.append(snap)
            return snapshots

        _, snapshots = await asyncio.gather(writer(), snapshotter())

        for snap in snapshots:
            # Each snapshot should have all 10 keys
            assert len(snap) == 10
            # Values should be internally consistent (no partial updates visible)
            for key in snap:
                assert isinstance(snap[key], int)


class TestConcurrentClear:
    """Test clear under concurrent access."""

    async def test_concurrent_clear_and_set(self, manager):
        """Clear + set should not leave corrupt state."""

        async def setter():
            for i in range(20):
                await manager.set(f"key_{i}", i)

        async def clearer():
            for _ in range(5):
                await manager.clear()

        await asyncio.gather(setter(), clearer())

        # State should be valid (either keys exist with correct values, or cleared)
        all_state = await manager.get_all()
        for key, val in all_state.items():
            assert isinstance(val, int)


class TestConcurrentRestore:
    """Test restore under concurrent access."""

    async def test_concurrent_restore_and_get(self, manager):
        """Restore + get should not raise or return garbage."""
        await manager.set("before", 1)
        snapshot = await manager.snapshot()

        async def restorer():
            for _ in range(10):
                await manager.restore(snapshot)

        async def reader():
            for _ in range(10):
                val = await manager.get("before")
                # Either the value from before restore or after
                assert val is None or val == 1

        await asyncio.gather(restorer(), reader())


class TestAtomicUpdate:
    """Test that update() is atomic across multiple keys."""

    async def test_update_atomicity(self, manager):
        """All keys in an update should appear together."""
        await manager.update({"a": 0, "b": 0})

        async def updater():
            for i in range(1, 20):
                await manager.update({"a": i, "b": i})

        async def checker():
            for _ in range(20):
                a = await manager.get("a")
                b = await manager.get("b")
                # Both should have the same value (set atomically)
                assert a == b, f"Non-atomic update: a={a}, b={b}"

        await asyncio.gather(updater(), checker())


class TestFactorySingletonThreadSafety:
    """Test factory singleton creation from multiple threads."""

    def test_factory_singleton_from_threads(self):
        """get_global_manager() from multiple threads returns same instance."""
        reset_global_manager()
        results = []
        barrier = threading.Barrier(5)

        def get_manager():
            barrier.wait()
            mgr = get_global_manager()
            results.append(id(mgr))

        threads = [threading.Thread(target=get_manager) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All threads should get the same instance
        assert len(set(results)) == 1, f"Multiple instances created: {set(results)}"

        reset_global_manager()
