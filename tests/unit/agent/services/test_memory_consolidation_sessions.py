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

"""Consolidation wiring tests (Codex finding 15, MEDIUM).

The A2b ledger started empty, staging was ONE process-wide list on the
shared consolidator, confirm committed that entire list (mixing sessions),
each episode was written separately before the list cleared (a later DB
error duplicated the already-committed prefix), and the first shared
consolidator construction froze auto-capture for later sessions.

Under test: stable proposal IDs, owner/session membership on proposals, a
retention bound, atomic/idempotent confirmation, the ACTUAL conversation
source wired at session end, and confirm / discard / retry / opt-out
across TWO sessions on one shared consolidator.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, List
from unittest.mock import Mock

import pytest

from victor.agent.services.memory_consolidation import (
    MemoryConsolidator,
    get_shared_memory_consolidator,
    reset_shared_memory,
)
from victor.agent.services.session_service import SessionService
from victor.storage.memory.preference_episodic import PreferenceEpisodicStore

_LEDGER_A = [
    {"role": "user", "content": "session A renamed the deploy script"},
    {"role": "user", "content": "session A pins python 3.12"},
]
_LEDGER_B = [
    {"role": "user", "content": "session B prefers ruff over flake8"},
]


@pytest.fixture
def dbs(tmp_path: Path):
    from victor.core.database import (
        get_database,
        get_project_database,
        reset_database,
        reset_project_database,
    )

    reset_database()
    reset_project_database(tmp_path)
    global_db = get_database(tmp_path / "victor.db")
    project_db = get_project_database(tmp_path)
    yield global_db, project_db
    reset_database()
    reset_project_database(tmp_path)


@pytest.fixture
def store(dbs) -> PreferenceEpisodicStore:
    global_db, project_db = dbs
    return PreferenceEpisodicStore(global_db=global_db, project_db=project_db)


# =============================================================================
# Stable proposal IDs + membership
# =============================================================================


def test_proposal_ids_are_stable_per_session_and_content(store: PreferenceEpisodicStore):
    consolidator = MemoryConsolidator(store)
    first = consolidator.propose(_LEDGER_A, session_id="s1")
    second = MemoryConsolidator(store).propose(_LEDGER_A, session_id="s1")

    assert [p.id for p in first] == [p.id for p in second]
    assert all(p.id.startswith("prop_") for p in first)
    # Different session, same content -> different membership, different id.
    other = MemoryConsolidator(store).propose(_LEDGER_A, session_id="s2")
    assert [p.id for p in first] != [p.id for p in other]


def test_proposals_carry_session_and_owner_membership(store: PreferenceEpisodicStore):
    consolidator = MemoryConsolidator(store)
    proposals = consolidator.propose(_LEDGER_A, session_id="s1", owner_client_id="alice")

    assert all(p.session_id == "s1" for p in proposals)
    assert all(p.metadata.get("owner_client_id") == "alice" for p in proposals)
    assert all(p.metadata.get("proposal_id") == p.id for p in proposals)


# =============================================================================
# Two-session isolation on ONE shared consolidator
# =============================================================================


def test_confirm_is_session_scoped_across_two_sessions(store: PreferenceEpisodicStore):
    """Confirmation of session A must not commit session B's proposals."""
    consolidator = MemoryConsolidator(store)
    consolidator.run(_LEDGER_A, session_id="sess-a", owner_client_id="alice")
    consolidator.run(_LEDGER_B, session_id="sess-b", owner_client_id="bob")

    stored_a = consolidator.commit_staged(session_id="sess-a")
    assert [ep.summary for ep in stored_a] == [
        "session A renamed the deploy script",
        "session A pins python 3.12",
    ]

    # Session B's proposals are untouched and still confirmable.
    assert [p.summary for p in consolidator.staged_proposals(session_id="sess-b")] == [
        "session B prefers ruff over flake8"
    ]
    stored_b = consolidator.commit_staged(session_id="sess-b")
    assert [ep.summary for ep in stored_b] == ["session B prefers ruff over flake8"]

    # Every stored episode belongs to its own session.
    for ep in store._recent_episodes():
        assert ep.session_id in {"sess-a", "sess-b"}
    assert len(store._recent_episodes()) == 3


def test_discard_is_session_scoped_across_two_sessions(store: PreferenceEpisodicStore):
    consolidator = MemoryConsolidator(store)
    consolidator.run(_LEDGER_A, session_id="sess-a")
    consolidator.run(_LEDGER_B, session_id="sess-b")

    assert consolidator.discard_staged(session_id="sess-a") == 2
    # Session B's pending proposals survive A's discard.
    assert len(consolidator.staged_proposals(session_id="sess-b")) == 1
    assert store._recent_episodes() == []


def test_opt_out_stores_nothing_across_two_sessions(store: PreferenceEpisodicStore):
    """Confirm-by-default: end-of-session runs on two sessions store nothing
    until each session's own confirm path runs."""
    consolidator = MemoryConsolidator(store)  # auto_capture defaults False
    consolidator.run(_LEDGER_A, session_id="sess-a")
    consolidator.run(_LEDGER_B, session_id="sess-b")

    assert consolidator.staged_proposals() != []
    assert store._recent_episodes() == []  # negative gate: nothing stored


# =============================================================================
# Idempotent / non-duplicating confirmation (retry after DB error)
# =============================================================================


class _FlakyStore(PreferenceEpisodicStore):
    """Store whose SECOND write explodes with a non-refusal DB error."""

    def __init__(self, real: PreferenceEpisodicStore) -> None:
        super().__init__(global_db=real._global_db, project_db=real._project_db)
        self.calls = 0
        self.fail_on = 2

    def record_episode(self, summary: str, **kwargs: Any):
        self.calls += 1
        if self.calls == self.fail_on:
            raise RuntimeError("simulated database failure")
        return super().record_episode(summary, **kwargs)


def test_retry_after_mid_commit_failure_never_duplicates_prefix(
    store: PreferenceEpisodicStore,
):
    """A DB error on the second write must leave the committed prefix
    committed exactly once; the retry completes only the remainder."""
    consolidator = MemoryConsolidator(store)
    consolidator.run(_LEDGER_A, session_id="sess-a")
    staged: List[Any] = consolidator.staged_proposals(session_id="sess-a")
    assert len(staged) == 2

    flaky = _FlakyStore(store)
    consolidator._store = flaky
    with pytest.raises(RuntimeError):
        consolidator.commit_staged(session_id="sess-a")

    # First proposal committed, second (and only it) still staged.
    assert flaky.calls == 2
    assert [p.id for p in consolidator.staged_proposals(session_id="sess-a")] == [staged[1].id]

    # Repair the store and retry.
    consolidator._store = store
    stored = consolidator.commit_staged(session_id="sess-a")

    assert [ep.summary for ep in stored] == ["session A pins python 3.12"]
    # The committed prefix is NOT duplicated: exactly 2 episodes total.
    assert len(store._recent_episodes()) == 2
    assert {ep.summary for ep in store._recent_episodes()} == {
        "session A renamed the deploy script",
        "session A pins python 3.12",
    }


def test_recommit_of_already_committed_proposals_is_a_noop(store: PreferenceEpisodicStore):
    """A stale confirm surface re-offering committed proposals stores nothing
    new (stable-ID idempotency)."""
    consolidator = MemoryConsolidator(store)
    proposals = consolidator.propose(_LEDGER_A, session_id="sess-a")
    consolidator.commit(proposals)

    stored_again = consolidator.commit(proposals)
    assert stored_again == []
    assert len(store._recent_episodes()) == 2


def test_refused_proposal_is_dropped_not_retried_forever(store: PreferenceEpisodicStore):
    from victor.storage.memory.preference_episodic import Episode, MemoryScope

    consolidator = MemoryConsolidator(store)
    oversized = Episode(
        id="prop_refused",
        scope=MemoryScope.PROJECT,
        summary="z" * 100000,
        session_id="sess-a",
        metadata={"proposed": True, "proposal_id": "prop_refused"},
    )
    consolidator._staged.setdefault("sess-a", []).append(oversized)

    stored = consolidator.commit_staged(session_id="sess-a")
    assert stored == []
    assert consolidator.staged_proposals(session_id="sess-a") == []


# =============================================================================
# Retention bound on staged state
# =============================================================================


def test_staged_retention_bound_evicts_oldest(store: PreferenceEpisodicStore):
    consolidator = MemoryConsolidator(store, max_staged=3)
    for i in range(5):
        consolidator.run(
            [{"role": "user", "content": f"distinct fact number {i}"}],
            session_id=f"sess-{i}",
        )

    total = len(consolidator.staged_proposals())
    assert total == 3  # bound holds
    # Oldest (lowest session index) evicted first.
    summaries = {p.summary for p in consolidator.staged_proposals()}
    assert summaries == {
        "distinct fact number 2",
        "distinct fact number 3",
        "distinct fact number 4",
    }


# =============================================================================
# The actual conversation source wired at session end
# =============================================================================


class _FakeConversationManager:
    """The bound memory manager: provider-format context messages."""

    def __init__(self, messages: List[dict]):
        self._messages = messages
        self.asked_include_system: bool | None = None

    def create_session(self, project_path=None):
        return "mem-1"

    def get_context_messages(self, session_id: str, include_system: bool = True):
        self.asked_include_system = include_system
        return self._messages


def _service(store: PreferenceEpisodicStore, manager=None) -> SessionService:
    service = SessionService(session_state_manager=Mock())
    service.bind_runtime_components(
        memory_consolidator=MemoryConsolidator(store),
        memory_manager=manager,
        memory_session_id="mem-1" if manager else None,
        owner_client_id="alice",
    )
    return service


def test_end_session_proposes_from_the_live_conversation_source(
    store: PreferenceEpisodicStore,
):
    """No registered ledger: the hook reads the bound memory manager's
    context messages — the production wiring, not an empty ledger."""
    manager = _FakeConversationManager(
        [
            {"role": "user", "content": "we moved CI to the merge queue"},
            {"role": "assistant", "content": "acknowledged"},
            {"role": "system", "content": "system frames are never proposed"},
        ]
    )
    service = _service(store, manager)
    asyncio.run(service.create_session("sess-live"))

    asyncio.run(service.consolidate_memory())

    assert manager.asked_include_system is False
    pending = service.pending_memory_proposals()
    assert [p.summary for p in pending] == ["we moved CI to the merge queue"]
    # Membership: session + owner stamped on every proposal.
    assert all(p.session_id == "sess-live" for p in pending)
    assert all(p.metadata.get("owner_client_id") == "alice" for p in pending)
    assert store._recent_episodes() == []  # confirm-by-default still holds


def test_end_session_without_any_source_stores_nothing(store: PreferenceEpisodicStore):
    service = _service(store, manager=None)
    asyncio.run(service.create_session("sess-empty"))
    report = asyncio.run(service.consolidate_memory())

    assert report["enabled"] is True
    assert report["proposed"] == 0
    assert store._recent_episodes() == []


def test_registered_ledger_still_wins_over_conversation_source(
    store: PreferenceEpisodicStore,
):
    manager = _FakeConversationManager([{"role": "user", "content": "from the conversation store"}])
    service = _service(store, manager)
    service.register_session_ledger([{"role": "user", "content": "from the ledger"}])
    asyncio.run(service.create_session("sess-x"))
    asyncio.run(service.consolidate_memory())

    assert [p.summary for p in service.pending_memory_proposals()] == ["from the ledger"]


def test_two_sessions_via_the_service_seam_are_isolated(store: PreferenceEpisodicStore):
    """Two sequential sessions on the shared consolidator: session B's
    confirm path does not touch session A's staged proposals."""
    consolidator = MemoryConsolidator(store)

    service_a = SessionService(session_state_manager=Mock())
    service_a.bind_runtime_components(
        memory_consolidator=consolidator,
        owner_client_id="alice",
    )
    service_a.register_session_ledger(list(_LEDGER_A))
    asyncio.run(service_a.create_session("sess-a"))
    asyncio.run(service_a.end_session())

    service_b = SessionService(session_state_manager=Mock())
    service_b.bind_runtime_components(
        memory_consolidator=consolidator,
        owner_client_id="bob",
    )
    service_b.register_session_ledger(list(_LEDGER_B))
    asyncio.run(service_b.create_session("sess-b"))
    asyncio.run(service_b.end_session())

    # Session B confirms ITS OWN proposals only.
    stored_b = service_b.commit_memory_proposals()
    assert [ep.summary for ep in stored_b] == ["session B prefers ruff over flake8"]

    # Session A's proposals are still pending, and confirmable separately.
    assert [p.summary for p in service_a.pending_memory_proposals()] == [
        "session A renamed the deploy script",
        "session A pins python 3.12",
    ]
    stored_a = consolidator.commit_staged(session_id="sess-a")
    assert len(stored_a) == 2
    assert len(store._recent_episodes()) == 3


# =============================================================================
# Shared-handle posture is not frozen by first construction
# =============================================================================


@pytest.fixture
def shared_handle():
    reset_shared_memory()
    yield
    reset_shared_memory()


def test_shared_consolidator_auto_capture_tracks_later_calls(shared_handle):
    """The first construction must not freeze auto-capture for later
    sessions: a later settings-derived call updates the shared instance."""
    first = get_shared_memory_consolidator(auto_capture=False)
    assert first.auto_capture is False

    later = get_shared_memory_consolidator(auto_capture=True)
    assert later is first  # still one process-wide handle
    assert first.auto_capture is True  # ...and the posture took effect


def test_confirm_after_a_new_session_started_still_commits_the_old_session(
    store: PreferenceEpisodicStore,
):
    """Negative-path guard: the confirm surface running after a NEW session
    began must scope to the session that actually staged the proposals."""
    consolidator = MemoryConsolidator(store)
    service = SessionService(session_state_manager=Mock())
    service.bind_runtime_components(memory_consolidator=consolidator)
    service.register_session_ledger(list(_LEDGER_A))
    asyncio.run(service.create_session("sess-a"))
    asyncio.run(service.end_session())

    # A new session starts on the SAME service before the confirm path runs.
    asyncio.run(service.create_session("sess-b"))
    stored = service.commit_memory_proposals()

    assert [ep.summary for ep in stored] == [
        "session A renamed the deploy script",
        "session A pins python 3.12",
    ]
    assert len(store._recent_episodes()) == 2
