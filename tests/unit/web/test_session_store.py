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
from dataclasses import replace

import pytest

from web.server.session_store import (
    InMemorySessionStore,
    PrincipalSessionLimitReached,
    SessionConnectionLimitReached,
    SessionDraining,
    SessionLimitReached,
    SessionReservationExpired,
    SessionReservationInvalid,
    SessionReservation,
    SessionScope,
    SessionStore,
    SessionState,
    SessionTokenRejected,
    SessionTurnConflict,
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


class TestScopedSessionAdmission:
    async def test_reservation_scope_is_server_owned_and_admitted_before_agent(self):
        store = InMemorySessionStore(max_sessions=2)

        reservation = await store.reserve_scope("alice", "workspace-a")

        assert reservation.scope == SessionScope(
            principal_id="alice",
            workspace_id="workspace-a",
            session_id=reservation.scope.session_id,
        )
        assert reservation.state is SessionState.RESERVED
        assert reservation.lease_expires_at > 0.0
        assert await store.count() == 1
        assert await store.get_scoped(reservation.scope) is None

    async def test_reserved_session_initializes_and_activates_without_add(self):
        store = InMemorySessionStore(max_sessions=1)
        agent = object()
        reservation = await store.reserve_scope("alice", "workspace-a")

        initializing = await store.begin_initialization(reservation)
        active = await store.complete_initialization(initializing, agent)

        assert initializing.state is SessionState.INITIALIZING
        assert active.agent is agent
        assert active.scope == reservation.scope
        assert active.state is SessionState.ACTIVE
        assert await store.get_scoped(reservation.scope) is active
        assert await store.count() == 1

    async def test_initialization_failure_releases_reserved_capacity(self):
        store = InMemorySessionStore(max_sessions=1)
        reservation = await store.reserve_scope("alice", "workspace-a")
        initializing = await store.begin_initialization(reservation)

        await store.fail_initialization(initializing)

        assert await store.count() == 0
        replacement = await store.reserve_scope("bob", "workspace-a")
        assert replacement.scope.principal_id == "bob"

    async def test_expired_initialization_lease_releases_capacity(self):
        current = [100.0]
        store = InMemorySessionStore(
            max_sessions=1,
            reservation_lease_seconds=10,
            clock=lambda: current[0],
        )
        reservation = await store.reserve_scope("alice", "workspace-a")
        current[0] = 111.0

        with pytest.raises(SessionReservationExpired):
            await store.begin_initialization(reservation)

        assert await store.count() == 0
        assert await store.reserve_scope("bob", "workspace-a")

    async def test_completion_requires_the_initializing_transition(self):
        store = InMemorySessionStore(max_sessions=1)
        reservation = await store.reserve_scope("alice", "workspace-a")

        with pytest.raises(SessionReservationInvalid):
            await store.complete_initialization(reservation, agent=object())

        assert await store.count() == 1
        initializing = await store.begin_initialization(reservation)
        assert await store.complete_initialization(initializing, agent=object())

    async def test_tampered_reservation_lease_cannot_extend_admission(self):
        current = [100.0]
        store = InMemorySessionStore(
            max_sessions=1,
            reservation_lease_seconds=10,
            clock=lambda: current[0],
        )
        reservation = await store.reserve_scope("alice", "workspace-a")
        current[0] = 111.0
        extended = SessionReservation(
            reservation_id=reservation.reservation_id,
            scope=reservation.scope,
            state=SessionState.RESERVED,
            lease_expires_at=current[0] + 10,
        )

        with pytest.raises(SessionReservationInvalid):
            await store.begin_initialization(extended)
        with pytest.raises(SessionReservationExpired):
            await store.begin_initialization(reservation)

        assert await store.count() == 0

    async def test_reservation_ids_are_unique_across_pending_admissions(self, monkeypatch):
        store = InMemorySessionStore(max_sessions=3, max_sessions_per_principal=1)
        generated = iter(("collision", "collision", "replacement"))
        monkeypatch.setattr(
            store_module.uuid,
            "uuid4",
            lambda: type("GeneratedId", (), {"hex": next(generated)})(),
        )

        first = await store.reserve_scope("alice", "workspace-a")
        second = await store.reserve_scope("bob", "workspace-a")

        assert first.reservation_id == "collision"
        assert second.reservation_id == "replacement"
        assert first.reservation_id != second.reservation_id
        assert await store.count() == 2

    async def test_exact_scope_lookup_does_not_expose_other_scopes(self):
        store = InMemorySessionStore(max_sessions=2)
        active = await store.complete_initialization(
            await store.begin_initialization(await store.reserve_scope("alice", "workspace-a")),
            agent=object(),
        )
        wrong_workspace = SessionScope("alice", "workspace-b", active.scope.session_id)
        wrong_principal = SessionScope("bob", "workspace-a", active.scope.session_id)

        assert await store.get_scoped(wrong_workspace) is None
        assert await store.get_scoped(wrong_principal) is None
        assert not await store.contains_scoped(wrong_workspace)
        assert await store.acquire_scoped_connection(wrong_workspace) is None
        await store.touch_scoped(wrong_workspace)
        assert await store.get_scoped(active.scope) is active

    async def test_global_limit_counts_active_and_reserved_sessions(self):
        store = InMemorySessionStore(max_sessions=2)
        await store.complete_initialization(
            await store.begin_initialization(await store.reserve_scope("alice", "workspace-a")),
            agent=object(),
        )
        await store.reserve_scope("bob", "workspace-a")

        assert not await store.has_capacity()
        with pytest.raises(SessionLimitReached):
            await store.reserve_scope("carol", "workspace-a")
        assert await store.count() == 2

    async def test_principal_limit_counts_active_and_reserved_sessions(self):
        store = InMemorySessionStore(max_sessions=10, max_sessions_per_principal=1)

        await store.reserve_scope("alice", "workspace-a")
        with pytest.raises(PrincipalSessionLimitReached):
            await store.reserve_scope("alice", "workspace-b")

        assert await store.reserve_scope("bob", "workspace-a")
        assert await store.count() == 2

    async def test_concurrent_reservations_admit_exact_capacity(self):
        store = InMemorySessionStore(max_sessions=10, max_sessions_per_principal=1)

        async def reserve(principal: str) -> bool:
            try:
                await store.reserve_scope(principal, "workspace-a")
                return True
            except SessionLimitReached:
                return False

        results = await asyncio.gather(*(reserve(f"principal-{i}") for i in range(50)))
        assert sum(results) == 10
        assert await store.count() == 10

    async def test_concurrent_reservations_admit_exact_principal_capacity(self):
        store = InMemorySessionStore(max_sessions=10, max_sessions_per_principal=2)

        async def reserve() -> bool:
            try:
                await store.reserve_scope("alice", "workspace-a")
                return True
            except PrincipalSessionLimitReached:
                return False

        results = await asyncio.gather(*(reserve() for _ in range(20)))
        assert sum(results) == 2
        assert await store.count() == 2

    async def test_scoped_connections_are_bounded_and_releasable(self):
        store = InMemorySessionStore(max_sessions=1, max_connections_per_session=1)
        active = await store.complete_initialization(
            await store.begin_initialization(await store.reserve_scope("alice", "workspace-a")),
            agent=object(),
        )

        assert await store.acquire_scoped_connection(active.scope) is active
        with pytest.raises(SessionConnectionLimitReached):
            await store.acquire_scoped_connection(active.scope)

        await store.release_scoped_connection(active.scope)
        assert await store.acquire_scoped_connection(active.scope) is active

    async def test_same_scope_connections_share_the_active_session(self):
        store = InMemorySessionStore(max_sessions=1, max_connections_per_session=2)
        active = await store.complete_initialization(
            await store.begin_initialization(await store.reserve_scope("alice", "workspace-a")),
            agent=object(),
        )

        first = await store.acquire_scoped_connection(active.scope)
        second = await store.acquire_scoped_connection(active.scope)

        assert first is active
        assert second is active
        assert active.connection_count == 2
        await store.release_scoped_connection(active.scope)
        assert active.connection_count == 1

    async def test_session_admits_one_turn_and_disconnect_does_not_release_it(self):
        store = InMemorySessionStore(max_sessions=1)
        active = await store.complete_initialization(
            await store.begin_initialization(await store.reserve_scope("alice", "workspace-a")),
            agent=object(),
        )

        turn = await store.acquire_turn(active.scope, turn_id="turn-1")
        with pytest.raises(SessionTurnConflict):
            await store.acquire_turn(active.scope, turn_id="turn-2")

        await store.release_scoped_connection(active.scope)
        with pytest.raises(SessionTurnConflict):
            await store.acquire_turn(active.scope, turn_id="turn-3")

        await store.release_turn(active.scope, "turn-2")
        with pytest.raises(SessionTurnConflict):
            await store.acquire_turn(active.scope, turn_id="turn-4")

        await store.release_turn(active.scope, turn.turn_id)
        assert (await store.acquire_turn(active.scope, turn_id="turn-5")).turn_id == "turn-5"

    async def test_draining_session_refuses_connections_and_turns_then_pops_once(self):
        store = InMemorySessionStore(max_sessions=1)
        active = await store.complete_initialization(
            await store.begin_initialization(await store.reserve_scope("alice", "workspace-a")),
            agent=object(),
        )

        draining = await store.begin_drain(active.scope)

        assert draining.state is SessionState.DRAINING
        with pytest.raises(SessionDraining):
            await store.acquire_scoped_connection(active.scope)
        with pytest.raises(SessionDraining):
            await store.acquire_turn(active.scope, turn_id="turn-1")
        assert await store.pop_scope(active.scope) is active
        assert await store.pop_scope(active.scope) is None

    async def test_scope_removal_revokes_its_turn_and_scoped_tokens(self):
        now = 1000.0
        store = InMemorySessionStore(max_sessions=1, clock=lambda: now)
        active = await store.complete_initialization(
            await store.begin_initialization(await store.reserve_scope("alice", "workspace-a")),
            agent=object(),
        )
        turn = await store.acquire_turn(active.scope, turn_id="turn-1")
        first_token = await store.bind_scoped_token(
            active.scope,
            token="token-a",
            token_nonce="nonce-a",
            issued_at=now,
            expires_at=now + 60,
        )
        second_token = await store.bind_scoped_token(
            active.scope,
            token="token-b",
            token_nonce="nonce-b",
            issued_at=now,
            expires_at=now + 60,
        )

        await store.pop_scope(active.scope)

        assert await store.get_scoped(active.scope) is None
        assert await store.count() == 0
        for token in (first_token, second_token):
            with pytest.raises(SessionTokenRejected, match="revoked"):
                await store.resolve_scoped_token(token, active.scope)
        with pytest.raises(SessionTurnConflict):
            await store.acquire_turn(active.scope, turn.turn_id)

    async def test_idle_close_removes_exact_scope_and_revokes_its_token(self):
        now = 100.0
        store = InMemorySessionStore(max_sessions=1, clock=lambda: now)
        active = await store.complete_initialization(
            await store.begin_initialization(await store.reserve_scope("alice", "workspace-a")),
            agent=object(),
        )
        token = await store.bind_scoped_token(
            active.scope,
            token="token-a",
            token_nonce="nonce-a",
            issued_at=now,
            expires_at=now + 3600,
        )
        active.last_activity = 0.0

        expired = await store.pop_idle(idle_timeout=60)

        assert expired == [active]
        assert await store.get_scoped(active.scope) is None
        with pytest.raises(SessionTokenRejected, match="revoked"):
            await store.resolve_scoped_token(token, active.scope)

    async def test_scoped_tokens_bind_exact_scope_epoch_and_revocation(self):
        current = [100.0]
        store = InMemorySessionStore(max_sessions=1, clock=lambda: current[0])
        active = await store.complete_initialization(
            await store.begin_initialization(await store.reserve_scope("alice", "workspace-a")),
            agent=object(),
        )

        token = await store.bind_scoped_token(
            active.scope,
            token="token-a",
            token_nonce="nonce-a",
            issued_at=current[0],
            expires_at=current[0] + 60,
        )

        assert token.store_epoch == store.store_epoch
        assert await store.resolve_scoped_token(token, active.scope) is active
        with pytest.raises(SessionTokenRejected, match="revoked"):
            await store.resolve_scoped_token(
                replace(token, token_nonce="unknown-nonce"), active.scope
            )
        with pytest.raises(SessionTokenRejected, match="revoked"):
            await store.resolve_scoped_token(
                replace(token, expires_at=current[0] + 3600), active.scope
            )
        with pytest.raises(SessionTokenRejected, match="scope mismatch"):
            await store.resolve_scoped_token(
                token,
                SessionScope("alice", "workspace-b", active.scope.session_id),
            )

        current[0] += 61
        with pytest.raises(SessionTokenRejected, match="expired"):
            await store.resolve_scoped_token(token, active.scope)

        current[0] = 100.0
        await store.revoke_scoped_token("token-a")
        with pytest.raises(SessionTokenRejected, match="revoked"):
            await store.resolve_scoped_token(token, active.scope)

    async def test_restart_epoch_rejects_process_local_token(self):
        now = 100.0
        old_store = InMemorySessionStore(max_sessions=1, clock=lambda: now)
        active = await old_store.complete_initialization(
            await old_store.begin_initialization(
                await old_store.reserve_scope("alice", "workspace-a")
            ),
            agent=object(),
        )
        token = await old_store.bind_scoped_token(
            active.scope,
            token="token-a",
            token_nonce="nonce-a",
            issued_at=now,
            expires_at=now + 60,
        )
        restarted_store = InMemorySessionStore(max_sessions=1, clock=lambda: now)

        assert restarted_store.store_epoch != old_store.store_epoch
        with pytest.raises(SessionTokenRejected, match="store epoch"):
            await restarted_store.resolve_scoped_token(token, active.scope)
