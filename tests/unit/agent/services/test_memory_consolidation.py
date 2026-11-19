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

"""FOU-A2b unit suite: end-of-session memory consolidation (FEP-0048).

Covers the consolidator lifecycle (propose -> commit / discard), the
confirm-by-default guarantee (the negative test is the point: with the
default ``memory_auto_capture=False`` nothing is stored without the
explicit confirm path), propose-time dropping of secret-shaped and
oversized candidates, and the Session-service seam (bound collaborator +
end_session hook + confirm/discard paths, no behavior change when no
consolidator is bound).

Databases are fixture-injected onto ``tmp_path`` per the A2a conventions.
No network.
"""

from pathlib import Path
from unittest.mock import Mock

import pytest

from victor.agent.services.memory_consolidation import (
    DEFAULT_MAX_PROPOSALS,
    MemoryConsolidator,
)
from victor.agent.services.session_service import SessionService
from victor.config.memory_settings import MemorySettings
from victor.core.database import (
    get_database,
    get_project_database,
    reset_database,
    reset_project_database,
)
from victor.storage.memory.preference_episodic import (
    MAX_MEMORY_CONTENT_BYTES,
    MemoryScope,
    PreferenceEpisodicStore,
)

_GITHUB_PAT = "ghp_" + "a" * 36

_LEDGER = [
    {"role": "user", "content": "we renamed the service to sandhi-gateway"},
    {"role": "assistant", "content": "noted, the rename is in the graph"},
    {"role": "user", "content": "this repo prefers pytest over unittest"},
]


@pytest.fixture
def dbs(tmp_path: Path):
    """Fixture-injected global + project database managers on tmp_path."""
    reset_database()
    reset_project_database(tmp_path)
    global_db = get_database(tmp_path / "victor.db")
    project_db = get_project_database(tmp_path)
    yield global_db, project_db
    reset_database()
    reset_project_database(tmp_path)


@pytest.fixture
def store(dbs) -> PreferenceEpisodicStore:
    """Store wired to the tmp-path databases."""
    global_db, project_db = dbs
    return PreferenceEpisodicStore(global_db=global_db, project_db=project_db)


# =============================================================================
# Consolidator lifecycle: propose -> commit / discard
# =============================================================================


def test_propose_returns_unstored_candidates(store: PreferenceEpisodicStore):
    consolidator = MemoryConsolidator(store)
    proposals = consolidator.propose(_LEDGER)

    # Human-side entries only, ledger order.
    assert [p.summary for p in proposals] == [
        "we renamed the service to sandhi-gateway",
        "this repo prefers pytest over unittest",
    ]
    assert all(p.scope is MemoryScope.PROJECT for p in proposals)
    assert all(p.metadata.get("proposed") is True for p in proposals)
    # Proposing never writes.
    assert store.episodes_for_thread("") == []


def test_commit_stores_proposed_episodes(store: PreferenceEpisodicStore):
    consolidator = MemoryConsolidator(store)
    proposals = consolidator.propose(_LEDGER)

    stored = consolidator.commit(proposals)
    assert len(stored) == 2
    assert all(ep.id.startswith("epis_") for ep in stored)  # real store ids
    assert all(not ep.metadata.get("proposed") for ep in stored)

    recent = store._recent_episodes()
    assert {ep.summary for ep in recent} == {p.summary for p in proposals}


def test_discard_writes_nothing(store: PreferenceEpisodicStore):
    consolidator = MemoryConsolidator(store)
    result = consolidator.run(_LEDGER)
    assert result.staged

    discarded = consolidator.discard()
    assert discarded == len(result.staged)
    assert consolidator.staged_proposals() == []
    assert store._recent_episodes() == []  # nothing was ever stored


def test_propose_dedupes_and_ignores_non_user_roles(store: PreferenceEpisodicStore):
    consolidator = MemoryConsolidator(store)
    ledger = [
        {"role": "user", "content": "same fact"},
        {"role": "user", "content": "same fact"},
        {"role": "assistant", "content": "same fact"},
        {"role": "tool", "content": "same fact"},
        {"role": "user", "content": "   "},
    ]
    proposals = consolidator.propose(ledger)
    assert [p.summary for p in proposals] == ["same fact"]


def test_propose_accepts_object_shaped_entries(store: PreferenceEpisodicStore):
    class Entry:
        role = "user"
        content = "object-shaped ledger entry"

    proposals = MemoryConsolidator(store).propose([Entry()])
    assert [p.summary for p in proposals] == ["object-shaped ledger entry"]


def test_propose_truncates_long_messages_to_summary_length(store: PreferenceEpisodicStore):
    consolidator = MemoryConsolidator(store)
    ledger = [{"role": "user", "content": "x" * 5000}]
    proposals = consolidator.propose(ledger)
    assert len(proposals[0].summary) == 280


# =============================================================================
# Write-path policy applies at propose time
# =============================================================================


def test_propose_drops_secret_shaped_candidates(store: PreferenceEpisodicStore):
    consolidator = MemoryConsolidator(store)
    ledger = [
        {"role": "user", "content": f"the api token is {_GITHUB_PAT} rotate it"},
        {"role": "user", "content": "benign fact survives"},
    ]
    proposals = consolidator.propose(ledger)
    assert [p.summary for p in proposals] == ["benign fact survives"]


def test_propose_bounds_long_messages_below_the_store_cap(
    store: PreferenceEpisodicStore,
):
    """Oversized ledger entries summarize to bounded, store-safe proposals.

    Proposals are summaries (truncated before validation), so a huge
    message becomes a bounded candidate rather than a refusal; the byte-cap
    refusal itself is fail-closed at commit (next test).
    """
    consolidator = MemoryConsolidator(store)
    ledger = [
        {"role": "user", "content": "y" * (MAX_MEMORY_CONTENT_BYTES + 1)},
        {"role": "user", "content": "short fact survives"},
    ]
    proposals = consolidator.propose(ledger)
    assert [p.summary for p in proposals] == [
        "y" * 280,
        "short fact survives",
    ]
    assert all(len(p.summary.encode("utf-8")) <= MAX_MEMORY_CONTENT_BYTES for p in proposals)


def test_commit_refuses_oversized_mutated_proposal(store: PreferenceEpisodicStore):
    """A proposal oversized after propose time is refused at commit."""
    from victor.storage.memory.preference_episodic import Episode

    consolidator = MemoryConsolidator(store)
    oversized = Episode(
        id="proposed_oversized",
        scope=MemoryScope.PROJECT,
        summary="z" * (MAX_MEMORY_CONTENT_BYTES + 1),
        metadata={"proposed": True},
    )
    stored = consolidator.commit([oversized])
    assert stored == []
    assert store._recent_episodes() == []


def test_commit_is_fail_closed_per_item(store: PreferenceEpisodicStore):
    """A proposal mutated after propose time still cannot persist a secret."""
    consolidator = MemoryConsolidator(store)
    good = consolidator.propose([{"role": "user", "content": "good fact"}])[0]

    from victor.storage.memory.preference_episodic import Episode

    smuggled = Episode(
        id="proposed_smuggled",
        scope=MemoryScope.PROJECT,
        summary=f"credential {_GITHUB_PAT} in a mutated proposal",
        metadata={"proposed": True},
    )

    stored = consolidator.commit([good, smuggled])
    assert [ep.summary for ep in stored] == ["good fact"]
    assert store._recent_episodes() == [stored[0]]


# =============================================================================
# Confirm-by-default (the negative tests are the point)
# =============================================================================


def test_run_default_stages_and_stores_nothing(store: PreferenceEpisodicStore):
    consolidator = MemoryConsolidator(store)  # auto_capture defaults False
    assert consolidator.auto_capture is False

    result = consolidator.run(_LEDGER)
    assert result.proposed
    assert result.staged
    assert result.nothing_stored
    assert consolidator.staged_proposals() == list(result.proposed)
    # Negative gate: NOTHING stored without confirmation.
    assert store._recent_episodes() == []


def test_commit_staged_is_the_confirm_path(store: PreferenceEpisodicStore):
    consolidator = MemoryConsolidator(store)
    consolidator.run(_LEDGER)

    stored = consolidator.commit_staged()
    assert len(stored) == 2
    assert consolidator.staged_proposals() == []
    assert {ep.summary for ep in store._recent_episodes()} == {p.summary for p in stored}


def test_discard_staged_is_the_reject_path(store: PreferenceEpisodicStore):
    consolidator = MemoryConsolidator(store)
    consolidator.run(_LEDGER)

    assert consolidator.discard_staged() == 2
    assert store._recent_episodes() == []


def test_auto_capture_optin_commits_immediately(store: PreferenceEpisodicStore):
    consolidator = MemoryConsolidator(store, auto_capture=True)
    result = consolidator.run(_LEDGER)

    assert not result.staged
    assert len(result.stored) == 2
    assert consolidator.staged_proposals() == []
    assert len(store._recent_episodes()) == 2


def test_proposal_bound_limits_flood(store: PreferenceEpisodicStore):
    consolidator = MemoryConsolidator(store, max_proposals=2)
    ledger = [{"role": "user", "content": f"fact number {i} is distinct"} for i in range(10)]
    proposals = consolidator.propose(ledger)
    assert len(proposals) == 2
    assert DEFAULT_MAX_PROPOSALS == 10


def test_memory_settings_default_is_off():
    """The settings flag ships default-off (FEP-0048 confirm-by-default)."""
    assert MemorySettings().memory_auto_capture is False


# =============================================================================
# Session-service seam
# =============================================================================


@pytest.fixture
def session_service() -> SessionService:
    return SessionService(session_state_manager=Mock())


def test_service_without_consolidator_reports_disabled(session_service):
    import asyncio

    report = asyncio.run(session_service.consolidate_memory())
    assert report == {"enabled": False, "proposed": 0, "stored": 0, "staged": 0}
    assert session_service.pending_memory_proposals() == []
    assert session_service.commit_memory_proposals() == []
    assert session_service.discard_memory_proposals() == 0


def _service_with_consolidator(store: PreferenceEpisodicStore) -> SessionService:
    service = SessionService(session_state_manager=Mock())
    consolidator = MemoryConsolidator(store)
    service.bind_runtime_components(memory_consolidator=consolidator)
    return service


def test_end_session_confirm_by_default_stores_nothing(
    store: PreferenceEpisodicStore,
):
    import asyncio

    service = _service_with_consolidator(store)
    service.register_session_ledger(list(_LEDGER))
    asyncio.run(service.create_session("sess-1"))

    asyncio.run(service.end_session())

    # Negative gate: end-of-session hook staged, stored nothing.
    assert store._recent_episodes() == []
    pending = service.pending_memory_proposals()
    assert [p.summary for p in pending] == [
        "we renamed the service to sandhi-gateway",
        "this repo prefers pytest over unittest",
    ]

    # Human confirms -> stored.
    stored = service.commit_memory_proposals()
    assert len(stored) == 2
    assert len(store._recent_episodes()) == 2


def test_end_session_reject_path_discards(store: PreferenceEpisodicStore):
    import asyncio

    service = _service_with_consolidator(store)
    service.register_session_ledger(list(_LEDGER))
    asyncio.run(service.create_session("sess-1"))
    asyncio.run(service.end_session())

    assert service.discard_memory_proposals() == 2
    assert store._recent_episodes() == []
    assert service.pending_memory_proposals() == []


def test_end_session_with_auto_capture_stores(
    store: PreferenceEpisodicStore,
):
    import asyncio

    service = SessionService(session_state_manager=Mock())
    service.bind_runtime_components(
        memory_consolidator=MemoryConsolidator(store, auto_capture=True)
    )
    service.register_session_ledger(list(_LEDGER))
    asyncio.run(service.create_session("sess-1"))
    asyncio.run(service.end_session())

    assert len(store._recent_episodes()) == 2


def test_no_consolidator_end_session_is_unchanged():
    import asyncio

    service = SessionService(session_state_manager=Mock())
    asyncio.run(service.create_session("sess-1"))
    asyncio.run(service.end_session())  # must not raise
    assert service.get_current_session_id() is None


def test_no_ledger_registered_stores_nothing(store: PreferenceEpisodicStore):
    import asyncio

    service = _service_with_consolidator(store)
    asyncio.run(service.create_session("sess-1"))
    asyncio.run(service.end_session())
    assert store._recent_episodes() == []


# =============================================================================
# Shared handle (FOU-A2d): the confirm surfaces (TUI /memory, API routes)
# and the production Session service all drive ONE process-scoped
# consolidator, so staged proposals proposed by end_session are visible to
# the confirm path.
# =============================================================================


@pytest.fixture
def shared_handle():
    from victor.agent.services.memory_consolidation import reset_shared_memory

    reset_shared_memory()
    yield
    reset_shared_memory()


def test_shared_consolidator_is_memoized(shared_handle):
    from victor.agent.services.memory_consolidation import (
        get_shared_memory_consolidator,
    )

    first = get_shared_memory_consolidator()
    assert get_shared_memory_consolidator() is first


def test_reset_shared_memory_clears_both_handles(shared_handle):
    from victor.agent.services.memory_consolidation import (
        get_shared_memory_consolidator,
        get_shared_memory_store,
        reset_shared_memory,
    )

    get_shared_memory_store()
    get_shared_memory_consolidator()
    reset_shared_memory()

    new_store = get_shared_memory_store()
    assert get_shared_memory_consolidator()._store is new_store


def test_injected_store_and_consolidator_are_returned(shared_handle, store):
    from victor.agent.services.memory_consolidation import (
        get_shared_memory_consolidator,
        get_shared_memory_store,
        set_shared_memory_consolidator,
        set_shared_memory_store,
    )

    set_shared_memory_store(store)
    consolidator = MemoryConsolidator(store)
    set_shared_memory_consolidator(consolidator)

    assert get_shared_memory_store() is store
    assert get_shared_memory_consolidator() is consolidator


def test_shared_store_is_memoized(shared_handle):
    from victor.agent.services.memory_consolidation import get_shared_memory_store

    first = get_shared_memory_store()
    assert get_shared_memory_store() is first


def test_bootstrap_session_service_binds_the_shared_consolidator(dbs, shared_handle):
    """Production wiring: the container-created SessionService stages
    proposals onto the same consolidator the /memory surfaces confirm."""
    from victor.agent.services.memory_consolidation import (
        get_shared_memory_consolidator,
        set_shared_memory_store,
    )
    from victor.core.bootstrap_services import _create_session_service
    from victor.core.container import ServiceContainer
    from victor.storage.memory.preference_episodic import PreferenceEpisodicStore

    set_shared_memory_store(PreferenceEpisodicStore(global_db=dbs[0], project_db=dbs[1]))

    service = _create_session_service(ServiceContainer())

    assert service._memory_consolidator is get_shared_memory_consolidator()
    # Confirm-by-default: the production binding never auto-captures.
    assert service._memory_consolidator.auto_capture is False
    # And the service-level seam resolves against it.
    assert service.pending_memory_proposals() == []
