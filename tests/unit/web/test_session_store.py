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

"""Behavioral tests for the injectable web session store (P0-B)."""

import asyncio

import pytest

from web.server.session_store import (
    InMemorySessionStore,
    SessionLimitReached,
    SessionStore,
    WebSession,
    get_session_store,
    set_session_store,
)
import web.server.session_store as store_module


def _session(session_id: str, token: str | None = None) -> WebSession:
    return WebSession(session_id=session_id, agent=object(), session_token=token)


class TestInMemorySessionStore:
    async def test_add_get_count_contains(self):
        store = InMemorySessionStore(max_sessions=5)
        session, created = await store.add(_session("s1", token="t1"))

        assert created is True
        assert await store.count() == 1
        assert await store.contains("s1")
        assert (await store.get("s1")) is session
        assert await store.get("missing") is None

    async def test_cap_enforced_atomically(self):
        store = InMemorySessionStore(max_sessions=2)
        await store.add(_session("s1"))
        await store.add(_session("s2"))

        assert not await store.has_capacity()
        with pytest.raises(SessionLimitReached):
            await store.add(_session("s3"))
        assert await store.count() == 2

    async def test_duplicate_add_returns_existing_not_created(self):
        store = InMemorySessionStore(max_sessions=5)
        original, _ = await store.add(_session("s1"))
        duplicate, created = await store.add(_session("s1"))

        assert created is False
        assert duplicate is original
        assert await store.count() == 1

    async def test_acquire_connection_touches_and_increments(self):
        store = InMemorySessionStore(max_sessions=5)
        session, _ = await store.add(_session("s1"))
        before = session.last_activity

        acquired = await store.acquire_connection("s1")
        assert acquired is session
        assert acquired.connection_count == 1
        assert acquired.last_activity >= before
        assert await store.acquire_connection("missing") is None

    async def test_release_connection_floors_at_zero(self):
        store = InMemorySessionStore(max_sessions=5)
        await store.add(_session("s1"))

        await store.release_connection("s1")
        await store.release_connection("s1")
        assert (await store.get("s1")).connection_count == 0
        await store.release_connection("missing")  # no-op, no raise

    async def test_touch_updates_last_activity(self):
        store = InMemorySessionStore(max_sessions=5)
        session, _ = await store.add(_session("s1"))
        session.last_activity = 0.0

        await store.touch("s1")
        assert session.last_activity > 0.0
        await store.touch("missing")  # no-op, no raise

    async def test_pop_idle_removes_only_expired_and_revokes_tokens(self):
        store = InMemorySessionStore(max_sessions=5)
        stale, _ = await store.add(_session("stale", token="t-stale"))
        fresh, _ = await store.add(_session("fresh", token="t-fresh"))
        stale.last_activity = 0.0  # epoch — long expired

        expired = await store.pop_idle(idle_timeout=3600)

        assert [s.session_id for s in expired] == ["stale"]
        assert not await store.contains("stale")
        assert await store.contains("fresh")
        assert "t-stale" not in store._tokens
        assert "t-fresh" in store._tokens

    async def test_pop_all_clears_sessions_and_tokens(self):
        store = InMemorySessionStore(max_sessions=5)
        await store.add(_session("s1", token="t1"))
        await store.add(_session("s2", token="t2"))
        await store.bind_token("extra", "s1")

        popped = await store.pop_all()

        assert {s.session_id for s in popped} == {"s1", "s2"}
        assert await store.count() == 0
        assert store._tokens == {}

    async def test_bind_and_revoke_token(self):
        store = InMemorySessionStore(max_sessions=5)
        await store.bind_token("tok", "s1")
        assert store._tokens == {"tok": "s1"}
        await store.revoke_token("tok")
        await store.revoke_token("unknown")  # no-op
        assert store._tokens == {}

    async def test_concurrent_adds_respect_cap_exactly(self):
        store = InMemorySessionStore(max_sessions=10)

        async def try_add(i: int) -> bool:
            try:
                _, created = await store.add(_session(f"s{i}"))
                return created
            except SessionLimitReached:
                return False

        results = await asyncio.gather(*(try_add(i) for i in range(50)))
        assert sum(results) == 10
        assert await store.count() == 10

    async def test_protocol_conformance(self):
        assert isinstance(InMemorySessionStore(max_sessions=1), SessionStore)


class TestStoreInjection:
    def test_get_before_set_raises(self, monkeypatch):
        monkeypatch.setattr(store_module, "_session_store", None)
        with pytest.raises(RuntimeError, match="not configured"):
            get_session_store()

    def test_set_then_get_roundtrip(self, monkeypatch):
        monkeypatch.setattr(store_module, "_session_store", None)
        store = InMemorySessionStore(max_sessions=1)
        assert set_session_store(store) is store
        assert get_session_store() is store
