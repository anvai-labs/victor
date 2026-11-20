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

"""FOU-A2a unit suite: PreferenceEpisodicStore (FEP-0048).

Covers the increment's gates: round-trip for both content classes, scope
isolation (user rows never leak into project queries and vice versa — the
negative tests are the point), secret-scan refusal with the typed error,
oversized refusal, forget by id and by content, thread-linked episode
queries, ranking interop through the coordinator, and two-database placement
(user rows in the global db, project rows in the project db).

Databases are fixture-injected onto ``tmp_path``; no network, no real
``~/.victor`` writes.
"""

from pathlib import Path

import pytest

from victor.core.database import (
    DatabaseManager,
    ProjectDatabaseManager,
    get_database,
    get_project_database,
    reset_database,
    reset_project_database,
)
from victor.core.schema import Tables
from victor.storage.memory.adapters import PreferenceEpisodicAdapter
from victor.storage.memory.preference_episodic import (
    Episode,
    MAX_MEMORY_CONTENT_BYTES,
    MemoryScope,
    MemoryWriteRefused,
    OversizedContentRefused,
    Preference,
    PreferenceEpisodicStore,
    SecretContentRefused,
)
from victor.storage.memory.unified import (
    MemoryProviderProtocol,
    MemoryQuery,
    MemoryResult,
    MemoryType,
    UnifiedMemoryCoordinator,
)

_GITHUB_PAT = "ghp_" + "a" * 36  # canonical GITHUB_PAT_PATTERN, matches on Rust and Python paths
_PRIVATE_KEY_HEADER = "-----BEGIN RSA PRIVATE KEY-----"


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
# Round-trip
# =============================================================================


def test_preference_round_trip_user_scope(store: PreferenceEpisodicStore):
    pref = store.set_preference(
        "always use uv, never pip",
        scope=MemoryScope.USER,
        source="user-correction",
        confidence=0.9,
        metadata={"captured_from": "chat"},
    )
    assert isinstance(pref, Preference)
    assert pref.id.startswith("pref_")
    assert pref.scope is MemoryScope.USER
    assert pref.created_at > 0
    assert pref.updated_at == pref.created_at

    found = store.get_preferences(MemoryScope.USER)
    assert len(found) == 1
    assert found[0].id == pref.id
    assert found[0].content == "always use uv, never pip"
    assert found[0].source == "user-correction"
    assert found[0].confidence == pytest.approx(0.9)
    assert found[0].metadata == {"captured_from": "chat"}

    by_id = store._db_for(MemoryScope.USER).query_one(
        f"SELECT content FROM {Tables.MEMORY_PREFERENCE} WHERE id = ?", (pref.id,)
    )
    assert by_id is not None and by_id[0] == "always use uv, never pip"


def test_preference_round_trip_project_scope(store: PreferenceEpisodicStore):
    pref = store.set_preference("this project uses pytest", scope=MemoryScope.PROJECT)
    found = store.get_preferences(MemoryScope.PROJECT)
    assert [p.id for p in found] == [pref.id]
    assert found[0].content == "this project uses pytest"


def test_episode_round_trip_with_thread_linkage(store: PreferenceEpisodicStore):
    episode = store.record_episode(
        "renamed the service to sandhi-gateway",
        session_id="sess-1",
        thread_id="thread-7",
        metadata={"turn": 3},
        timestamp=1_700_000_000.0,
    )
    assert isinstance(episode, Episode)
    assert episode.id.startswith("epis_")
    assert episode.scope is MemoryScope.PROJECT

    found = store.episodes_for_thread("thread-7")
    assert len(found) == 1
    assert found[0].id == episode.id
    assert found[0].summary == "renamed the service to sandhi-gateway"
    assert found[0].session_id == "sess-1"
    assert found[0].timestamp == pytest.approx(1_700_000_000.0)
    assert found[0].metadata == {"turn": 3}


async def test_protocol_store_routes_to_typed_helpers(store: PreferenceEpisodicStore):
    await store.store("k1", "prefer tables over prose", {"scope": "user"})
    await store.store(
        "k2",
        "migrated CI to shard matrix",
        {"kind": "episode", "thread_id": "t1", "session_id": "s1"},
    )
    assert len(store.get_preferences(MemoryScope.USER)) == 1
    assert len(store.episodes_for_thread("t1")) == 1


# =============================================================================
# Scope isolation (the negative tests are the point)
# =============================================================================


def test_user_preferences_never_appear_in_project_queries(store: PreferenceEpisodicStore):
    store.set_preference("user naming convention", scope=MemoryScope.USER)
    store.set_preference("user prefers short commits", scope=MemoryScope.USER)

    assert store.get_preferences(MemoryScope.PROJECT) == []

    results = store.search(MemoryQuery(query="convention commits", filters={"scope": "project"}))
    assert results == []


def test_project_preferences_never_appear_in_user_queries(store: PreferenceEpisodicStore):
    store.set_preference("project runs tests via make", scope=MemoryScope.PROJECT)

    assert store.get_preferences(MemoryScope.USER) == []

    results = store.search(MemoryQuery(query="tests", filters={"scope": "user"}))
    assert results == []


def test_episodes_are_project_scope_only(store: PreferenceEpisodicStore):
    store.record_episode("deployed v2 to staging", thread_id="t2")

    # An unscoped search sees it; a user-scoped search must not.
    unscoped = store.search(MemoryQuery(query="staging"))
    assert any(r.source is MemoryType.EPISODE for r in unscoped)
    user_scoped = store.search(MemoryQuery(query="staging", filters={"scope": "user"}))
    assert user_scoped == []


def test_unscoped_search_keeps_scope_metadata(store: PreferenceEpisodicStore):
    store.set_preference("user likes rust", scope=MemoryScope.USER)
    store.set_preference("project lints with ruff", scope=MemoryScope.PROJECT)

    results = store.search(MemoryQuery(query="likes rust lints ruff"))
    scopes = {r.metadata["scope"] for r in results}
    assert scopes == {"user", "project"}


def test_invalid_scope_is_rejected(store: PreferenceEpisodicStore):
    with pytest.raises(ValueError):
        store.search(MemoryQuery(query="x", filters={"scope": "workspace"}))


# =============================================================================
# Secret-scan refusal
# =============================================================================


@pytest.mark.parametrize(
    "secret_content",
    [
        f"my token is {_GITHUB_PAT} please remember it",
        f"{_PRIVATE_KEY_HEADER}\nMIIEpAIBAAKCAQEA...",
    ],
    ids=["github-pat", "private-key-block"],
)
def test_secret_shaped_preference_is_refused(store: PreferenceEpisodicStore, secret_content):
    with pytest.raises(SecretContentRefused) as excinfo:
        store.set_preference(secret_content, scope=MemoryScope.USER)
    assert isinstance(excinfo.value, MemoryWriteRefused)
    assert excinfo.value.secret_types
    # Nothing was persisted.
    assert store.get_preferences(MemoryScope.USER) == []


def test_secret_shaped_episode_is_refused(store: PreferenceEpisodicStore):
    with pytest.raises(SecretContentRefused):
        store.record_episode(f"pasted credential {_GITHUB_PAT} into config")
    assert store.episodes_for_thread("") == []
    assert store._recent_episodes() == []


async def test_protocol_store_propagates_refusal(store: PreferenceEpisodicStore):
    with pytest.raises(SecretContentRefused):
        await store.store("k", f"key {_GITHUB_PAT}", {"scope": "user"})


@pytest.mark.parametrize(
    "benign_content",
    [
        "my secret project name",
        "the secret ingredient is patience",
        "password policies are documented elsewhere",
        "the password manager recommends vaults",
    ],
    ids=["secret-word", "secret-ingredient", "password-policies", "password-manager"],
)
def test_benign_prose_with_secret_words_is_stored(store: PreferenceEpisodicStore, benign_content):
    """The word "secret"/"password" in prose is not a credential.

    FOU-A2b follow-up: the shared scanner's generic keyword patterns used
    to treat bare whitespace as a key-value separator, so this benign
    corpus was refused. Key-value-shaped text must still refuse (see
    tests/unit/security/test_secrets_scanner.py).
    """
    pref = store.set_preference(benign_content, scope=MemoryScope.USER)
    found = store.get_preferences(MemoryScope.USER)
    assert [p.id for p in found] == [pref.id]


def test_validate_content_is_public_and_matches_write_policy(
    store: PreferenceEpisodicStore,
):
    """Write-path callers pre-screen with the same policy the store enforces."""
    store.validate_content("perfectly fine directive")  # no raise

    with pytest.raises(SecretContentRefused):
        store.validate_content(f"token {_GITHUB_PAT}")
    with pytest.raises(OversizedContentRefused):
        store.validate_content("x" * (MAX_MEMORY_CONTENT_BYTES + 1))


# =============================================================================
# Oversized refusal
# =============================================================================


def test_oversized_preference_is_refused(store: PreferenceEpisodicStore):
    oversized = "word " * (MAX_MEMORY_CONTENT_BYTES // 4 + 64)
    assert len(oversized.encode("utf-8")) > MAX_MEMORY_CONTENT_BYTES
    with pytest.raises(OversizedContentRefused) as excinfo:
        store.set_preference(oversized, scope=MemoryScope.USER)
    assert isinstance(excinfo.value, MemoryWriteRefused)
    assert store.get_preferences(MemoryScope.USER) == []


def test_oversized_episode_is_refused(store: PreferenceEpisodicStore):
    oversized = "x" * (MAX_MEMORY_CONTENT_BYTES + 1)
    with pytest.raises(OversizedContentRefused):
        store.record_episode(oversized)
    assert store._recent_episodes() == []


def test_custom_cap_is_honored(tmp_path: Path, dbs):
    global_db, project_db = dbs
    strict = PreferenceEpisodicStore(
        global_db=global_db, project_db=project_db, max_content_bytes=32
    )
    with pytest.raises(OversizedContentRefused):
        strict.set_preference("this is longer than thirty-two bytes by far", scope=MemoryScope.USER)


# =============================================================================
# Forget
# =============================================================================


def test_forget_by_id(store: PreferenceEpisodicStore):
    keep = store.set_preference("keep me", scope=MemoryScope.USER)
    drop = store.set_preference("drop me", scope=MemoryScope.USER)

    assert store.forget(preference_id=drop.id) == 1
    remaining = [p.id for p in store.get_preferences(MemoryScope.USER)]
    assert remaining == [keep.id]
    assert store.forget(preference_id=drop.id) == 0


def test_forget_by_id_spans_scopes(store: PreferenceEpisodicStore):
    proj = store.set_preference("project-scoped fact", scope=MemoryScope.PROJECT)
    assert store.forget(preference_id=proj.id) == 1
    assert store.get_preferences(MemoryScope.PROJECT) == []


def test_forget_by_content_match_requires_scope(store: PreferenceEpisodicStore):
    store.set_preference("pip is fine", scope=MemoryScope.USER)
    with pytest.raises(ValueError):
        store.forget(content_match="pip")


def test_forget_by_content_match_is_scope_bounded(store: PreferenceEpisodicStore):
    store.set_preference("use pip", scope=MemoryScope.USER)
    store.set_preference("use pip", scope=MemoryScope.PROJECT)

    assert store.forget(content_match="pip", scope=MemoryScope.PROJECT) == 1
    # Only the project-scope row was deleted; the user row is untouched.
    assert len(store.get_preferences(MemoryScope.USER)) == 1
    assert store.get_preferences(MemoryScope.PROJECT) == []


def test_forget_by_content_match_escapes_like_wildcards(store: PreferenceEpisodicStore):
    store.set_preference("literally 100% coverage", scope=MemoryScope.USER)
    store.set_preference("coverage at 95", scope=MemoryScope.USER)
    # "%" must be literal, not a wildcard: only the exact substring matches.
    assert store.forget(content_match="100%", scope=MemoryScope.USER) == 1
    remaining = [p.content for p in store.get_preferences(MemoryScope.USER)]
    assert remaining == ["coverage at 95"]


def test_forget_with_no_criteria_is_rejected(store: PreferenceEpisodicStore):
    with pytest.raises(ValueError):
        store.forget()


# =============================================================================
# Finding 13 (Codex review): empty/whitespace content_match must never wipe a
# scope — rejected in the STORE before any transaction opens, rows preserved.
# =============================================================================


def test_forget_empty_content_match_is_rejected(store: PreferenceEpisodicStore):
    store.set_preference("always use uv", scope=MemoryScope.USER)
    store.set_preference("uses pytest", scope=MemoryScope.USER)

    with pytest.raises(ValueError, match="content_match"):
        store.forget(content_match="", scope=MemoryScope.USER)

    # The empty contains-match would have matched every row in the scope.
    remaining = sorted(p.content for p in store.get_preferences(MemoryScope.USER))
    assert remaining == ["always use uv", "uses pytest"]


def test_forget_whitespace_content_match_is_rejected(store: PreferenceEpisodicStore):
    store.set_preference("always use uv", scope=MemoryScope.PROJECT)
    store.set_preference("uses pytest", scope=MemoryScope.PROJECT)

    with pytest.raises(ValueError, match="content_match"):
        store.forget(content_match="   \t\n ", scope=MemoryScope.PROJECT)

    remaining = sorted(p.content for p in store.get_preferences(MemoryScope.PROJECT))
    assert remaining == ["always use uv", "uses pytest"]


def test_forget_empty_match_refusal_is_typed(store: PreferenceEpisodicStore):
    from victor.storage.memory.preference_episodic import EmptyContentMatchRefused

    with pytest.raises(EmptyContentMatchRefused):
        store.forget(content_match=" ", scope=MemoryScope.USER)


def test_forget_empty_match_without_scope_is_still_rejected(store: PreferenceEpisodicStore):
    """The input defect is rejected before the scope requirement is reached."""
    with pytest.raises(ValueError, match="content_match"):
        store.forget(content_match="")


def test_forget_nonempty_match_still_works(store: PreferenceEpisodicStore):
    """Positive control: the guard rejects only empty/whitespace matches."""
    store.set_preference("use uv", scope=MemoryScope.USER)
    store.set_preference("use pip", scope=MemoryScope.USER)
    assert store.forget(content_match="uv", scope=MemoryScope.USER) == 1
    assert [p.content for p in store.get_preferences(MemoryScope.USER)] == ["use pip"]


# =============================================================================
# Finding 4 (Codex review): store-level cross-principal guard. When a
# principal context is present (store-bound or call-supplied), addressing the
# store as a different principal is refused before any database work.
# =============================================================================


def test_cross_principal_write_is_refused(dbs):
    global_db, project_db = dbs
    guarded = PreferenceEpisodicStore(
        global_db=global_db, project_db=project_db, principal_id="alice"
    )
    with pytest.raises(ValueError, match="bob"):
        guarded.set_preference("bob's directive", scope=MemoryScope.USER, on_behalf_of="bob")
    # Nothing was written under alice's principal context.
    assert [p.content for p in guarded.get_preferences(MemoryScope.USER)] == []


def test_cross_principal_read_and_forget_are_refused(dbs):
    global_db, project_db = dbs
    guarded = PreferenceEpisodicStore(
        global_db=global_db, project_db=project_db, principal_id="alice"
    )
    pref = guarded.set_preference("alice's directive", scope=MemoryScope.USER)

    with pytest.raises(ValueError, match="bob"):
        guarded.get_preferences(MemoryScope.USER, on_behalf_of="bob")
    with pytest.raises(ValueError, match="bob"):
        guarded.forget(preference_id=pref.id, on_behalf_of="bob")
    with pytest.raises(ValueError, match="bob"):
        guarded.forget(content_match="directive", scope=MemoryScope.USER, on_behalf_of="bob")
    with pytest.raises(ValueError, match="bob"):
        guarded.record_episode("bob's episode", on_behalf_of="bob")

    # The row survives every refused addressing attempt.
    assert [p.content for p in guarded.get_preferences(MemoryScope.USER)] == ["alice's directive"]


def test_same_principal_addressing_is_allowed(dbs):
    global_db, project_db = dbs
    guarded = PreferenceEpisodicStore(
        global_db=global_db, project_db=project_db, principal_id="alice"
    )
    guarded.set_preference("alice's directive", scope=MemoryScope.USER, on_behalf_of="alice")
    assert [p.content for p in guarded.get_preferences(MemoryScope.USER, on_behalf_of="alice")] == [
        "alice's directive"
    ]
    guarded.record_episode("alice's episode", on_behalf_of="alice")
    assert guarded.recent_episodes() != []


def test_store_without_principal_context_accepts_named_caller(dbs):
    """No principal context anywhere: the legacy single-operator behavior."""
    global_db, project_db = dbs
    unbound = PreferenceEpisodicStore(global_db=global_db, project_db=project_db)
    unbound.set_preference("anyone's directive", scope=MemoryScope.USER, on_behalf_of="whoever")
    assert [p.content for p in unbound.get_preferences(MemoryScope.USER)] == ["anyone's directive"]


# =============================================================================
# Thread-linked episode queries
# =============================================================================


def test_episodes_for_thread_filters_and_orders(store: PreferenceEpisodicStore):
    store.record_episode("old event", thread_id="t1", timestamp=1_000.0)
    store.record_episode("new event", thread_id="t1", timestamp=2_000.0)
    store.record_episode("other thread", thread_id="t2", timestamp=3_000.0)

    found = store.episodes_for_thread("t1")
    assert [e.summary for e in found] == ["new event", "old event"]  # newest first
    assert store.episodes_for_thread("missing") == []


def test_episode_search_with_thread_filter(store: PreferenceEpisodicStore):
    store.record_episode("thread one migrated the database", thread_id="t1")
    store.record_episode("thread two renamed the service", thread_id="t2")

    results = store.search(MemoryQuery(query="migrated database", filters={"thread_id": "t1"}))
    assert len(results) == 1
    assert results[0].content == "thread one migrated the database"
    assert results[0].metadata["thread_id"] == "t1"


# =============================================================================
# Ranking interop (MemoryProviderProtocol / coordinator)
# =============================================================================


def test_store_satisfies_protocol(store: PreferenceEpisodicStore):
    assert isinstance(store, MemoryProviderProtocol)


def test_search_returns_ranked_memory_results(store: PreferenceEpisodicStore):
    store.set_preference("the project uses pytest for testing", scope=MemoryScope.PROJECT)
    store.record_episode("pytest suite was sharded last week", thread_id="t1")

    results = store.search(MemoryQuery(query="pytest testing", limit=10))
    assert results, "matching memories must be returned"
    assert all(isinstance(r, MemoryResult) for r in results)
    assert all(r.relevance > 0.0 for r in results)
    assert results == sorted(results, key=lambda r: r.relevance, reverse=True)
    kinds = {r.metadata["kind"] for r in results}
    assert kinds == {"preference", "episode"}
    # Full-overlap preference (stability + confidence) outranks the
    # partial-overlap recency-decayed episode.
    assert results[0].metadata["kind"] == "preference"


async def test_adapter_serves_only_its_lane(store: PreferenceEpisodicStore):
    store.set_preference("prefer descriptive commit messages", scope=MemoryScope.PROJECT)
    store.record_episode("commit message linting was enabled", thread_id="t1")

    pref_adapter = PreferenceEpisodicAdapter(store, lane=MemoryType.PREFERENCE)
    epi_adapter = PreferenceEpisodicAdapter(store, lane=MemoryType.EPISODE)
    assert pref_adapter.memory_type is MemoryType.PREFERENCE
    assert epi_adapter.memory_type is MemoryType.EPISODE
    with pytest.raises(ValueError):
        PreferenceEpisodicAdapter(store, lane=MemoryType.ENTITY)

    query = MemoryQuery(query="commit messages")
    pref_results = await pref_adapter.search(query)
    epi_results = await epi_adapter.search(query)
    assert all(r.source is MemoryType.PREFERENCE for r in pref_results)
    assert all(r.source is MemoryType.EPISODE for r in epi_results)
    assert all(
        r.provenance is not None and r.provenance.store_class == "PreferenceEpisodicStore"
        for r in pref_results + epi_results
    )
    assert pref_results[0].provenance.lane is MemoryType.PREFERENCE


async def test_coordinator_federated_search_interop(store: PreferenceEpisodicStore):
    coordinator = UnifiedMemoryCoordinator(enable_memory_evolution=False)
    coordinator.register_provider(PreferenceEpisodicAdapter(store))
    coordinator.register_provider(PreferenceEpisodicAdapter(store, lane=MemoryType.EPISODE))

    store.set_preference("this repo prefers pytest over unittest", scope=MemoryScope.PROJECT)
    store.record_episode("unittest usage was removed from the repo", thread_id="t9")

    results = await coordinator.search_all("pytest unittest repo")
    assert results, "coordinator must surface memory results"
    kinds = {r.metadata.get("kind") for r in results}
    assert kinds <= {"preference", "episode"}
    assert kinds  # at least one memory lane contributed

    episode_only = await coordinator.search_all(
        "unittest usage removed", memory_types=[MemoryType.EPISODE]
    )
    assert episode_only
    assert all(r.source is MemoryType.EPISODE for r in episode_only)


async def test_adapter_get_and_is_available(store: PreferenceEpisodicStore):
    pref = store.set_preference("adapter get probe", scope=MemoryScope.PROJECT)
    episode = store.record_episode("episode get probe", thread_id="t1")

    pref_adapter = PreferenceEpisodicAdapter(store)
    epi_adapter = PreferenceEpisodicAdapter(store, lane=MemoryType.EPISODE)

    got_pref = await pref_adapter.get(pref.id)
    assert got_pref is not None and got_pref.source is MemoryType.PREFERENCE
    assert await epi_adapter.get(pref.id) is None  # lane-mismatched id is not served

    got_episode = await epi_adapter.get(episode.id)
    assert got_episode is not None and got_episode.source is MemoryType.EPISODE
    assert await pref_adapter.get("does-not-exist") is None

    assert pref_adapter.is_available() is True


# =============================================================================
# Two-database placement
# =============================================================================


def test_user_rows_live_in_global_db_project_rows_in_project_db(
    store: PreferenceEpisodicStore, dbs, tmp_path: Path
):
    global_db, project_db = dbs
    user_pref = store.set_preference("global scope fact", scope=MemoryScope.USER)
    project_pref = store.set_preference("project scope fact", scope=MemoryScope.PROJECT)
    episode = store.record_episode("project episode", thread_id="t1")

    # User-scope row: in the GLOBAL database, NOT in the project database.
    global_rows = global_db.query(f"SELECT id FROM {Tables.MEMORY_PREFERENCE}")
    assert [r["id"] for r in global_rows] == [user_pref.id]
    project_pref_rows = project_db.query(f"SELECT id FROM {Tables.MEMORY_PREFERENCE}")
    assert [r["id"] for r in project_pref_rows] == [project_pref.id]

    # Episodes: project database only — the global db never grows an
    # episode table (physical isolation is the isolation boundary).
    episode_rows = project_db.query(f"SELECT id FROM {Tables.MEMORY_EPISODE}")
    assert [r["id"] for r in episode_rows] == [episode.id]
    assert Tables.MEMORY_EPISODE not in global_db.get_tables()

    # Databases really are the tmp-path files the fixture injected.
    assert global_db.db_path == tmp_path / "victor.db"
    assert project_db.db_path == tmp_path / ".victor" / "project.db"


def test_scope_query_cannot_cross_databases(store: PreferenceEpisodicStore, dbs):
    """Defense in depth: the scope column also partitions rows within a db."""
    global_db, project_db = dbs
    store.set_preference("cross probe", scope=MemoryScope.USER)
    store.set_preference("cross probe", scope=MemoryScope.PROJECT)

    assert (
        global_db.query_one(
            f"SELECT COUNT(*) FROM {Tables.MEMORY_PREFERENCE} WHERE scope = 'project'"
        )[0]
        == 0
    )
    assert (
        project_db.query_one(
            f"SELECT COUNT(*) FROM {Tables.MEMORY_PREFERENCE} WHERE scope = 'user'"
        )[0]
        == 0
    )


# =============================================================================
# Availability / misc
# =============================================================================


def test_is_available_reflects_missing_tables(tmp_path: Path, dbs):
    import sqlite3

    global_db, project_db = dbs
    store = PreferenceEpisodicStore(global_db=global_db, project_db=project_db)
    assert store.is_available() is True

    with sqlite3.connect(tmp_path / "victor.db") as conn:
        conn.execute(f"DROP TABLE {Tables.MEMORY_PREFERENCE}")
    assert store.is_available() is False


def test_database_managers_are_injected_not_singleton(store: PreferenceEpisodicStore, dbs):
    global_db, project_db = dbs
    assert isinstance(global_db, DatabaseManager)
    assert isinstance(project_db, ProjectDatabaseManager)
    assert store._db_for(MemoryScope.USER) is global_db
    assert store._db_for(MemoryScope.PROJECT) is project_db
