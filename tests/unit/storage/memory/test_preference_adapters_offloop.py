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

"""Off-loop, lane-selected reads for the preference/episode adapters (finding 19).

Codex finding 19: both async lane adapters ran the synchronous SQLite search
plus Python ranking ON the event loop, materializing up to 500 rows per scope
per scan, and each adapter re-read BOTH lanes then filtered. Contract pinned
here:

- ``PreferenceEpisodicAdapter.search`` offloads the store scan + ranking to a
  worker thread (``asyncio.to_thread``) — the event loop thread never runs it;
- the lane is selected BEFORE reading: the PREFERENCE adapter never reads the
  episode table and vice versa (each scope's scan is bounded to its lane);
- results are unchanged by the hop (same lane-filtered results as the direct
  store search).
"""

import threading
from pathlib import Path

import pytest

from victor.core.database import (
    get_database,
    get_project_database,
    reset_database,
    reset_project_database,
)
from victor.storage.memory.adapters import PreferenceEpisodicAdapter
from victor.storage.memory.preference_episodic import MemoryScope, PreferenceEpisodicStore
from victor.storage.memory.unified import MemoryQuery, MemoryType


@pytest.fixture
def dbs(tmp_path: Path):
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


@pytest.fixture
def seeded_store(store: PreferenceEpisodicStore) -> PreferenceEpisodicStore:
    store.set_preference("always use uv for packaging", scope=MemoryScope.USER)
    store.set_preference("deploy target is staging", scope=MemoryScope.PROJECT)
    store.record_episode("renamed the service to cart", thread_id="t1")
    return store


def _query(text: str, **filters) -> MemoryQuery:
    return MemoryQuery(query=text, limit=8, filters=filters or None)


async def test_adapter_search_offloads_store_scan_off_the_loop(seeded_store, monkeypatch):
    """The synchronous SQLite scan + ranking runs on a worker thread."""
    loop_thread = threading.get_ident()
    scan_threads = []
    original = seeded_store.search

    def _spy_search(query):
        scan_threads.append(threading.get_ident())
        return original(query)

    monkeypatch.setattr(seeded_store, "search", _spy_search)

    adapter = PreferenceEpisodicAdapter(seeded_store, lane=MemoryType.PREFERENCE)
    results = await adapter.search(_query("packaging uv", scope="user"))

    assert results, "precondition: the seeded preference surfaces"
    assert scan_threads == [loop_thread] or all(t != loop_thread for t in scan_threads)
    assert scan_threads and all(t != loop_thread for t in scan_threads)


async def test_preference_lane_never_reads_episode_table(seeded_store, monkeypatch):
    """Lane selected before reading: the PREFERENCE lane skips episode reads."""
    adapter = PreferenceEpisodicAdapter(seeded_store, lane=MemoryType.PREFERENCE)

    def _explode(*args, **kwargs):
        raise AssertionError("episode table read on the preference lane")

    monkeypatch.setattr(seeded_store, "_recent_episodes", _explode)
    monkeypatch.setattr(seeded_store, "episodes_for_thread", _explode)

    results = await adapter.search(_query("packaging uv", scope="user"))
    assert all(r.metadata.get("kind") == "preference" for r in results)
    assert any("uv" in r.content for r in results)


async def test_episode_lane_never_reads_preference_tables(seeded_store, monkeypatch):
    """Lane selected before reading: the EPISODE lane skips preference reads."""
    adapter = PreferenceEpisodicAdapter(seeded_store, lane=MemoryType.EPISODE)

    def _explode(*args, **kwargs):
        raise AssertionError("preference table read on the episode lane")

    monkeypatch.setattr(seeded_store, "get_preferences", _explode)

    results = await adapter.search(_query("renamed service", scope="project"))
    assert results, "precondition: the seeded episode surfaces"
    assert all(r.metadata.get("kind") == "episode" for r in results)


async def test_offloaded_results_match_direct_store_search(seeded_store):
    """The thread hop does not change results (differential parity)."""
    preference_adapter = PreferenceEpisodicAdapter(seeded_store, lane=MemoryType.PREFERENCE)
    episode_adapter = PreferenceEpisodicAdapter(seeded_store, lane=MemoryType.EPISODE)

    lane_results = await preference_adapter.search(_query("packaging uv", scope="user"))
    direct = [
        r
        for r in seeded_store.search(_query("packaging uv", scope="user"))
        if r.metadata.get("kind") == "preference"
    ]
    assert [(r.id, round(r.relevance, 6)) for r in lane_results] == [
        (r.id, round(r.relevance, 6)) for r in direct
    ]

    episode_results = await episode_adapter.search(_query("renamed service", scope="project"))
    assert all(r.metadata.get("kind") == "episode" for r in episode_results)


async def test_adapter_get_stays_off_the_loop(seeded_store, monkeypatch):
    """Point lookups through the adapter also leave the event loop."""
    loop_thread = threading.get_ident()
    get_threads = []
    adapter = PreferenceEpisodicAdapter(seeded_store, lane=MemoryType.PREFERENCE)

    def _spy_get_by_id_sync(key):
        get_threads.append(threading.get_ident())
        return None

    monkeypatch.setattr(seeded_store, "_get_by_id_sync", _spy_get_by_id_sync)

    result = await adapter.get("unknown_id")
    assert result is None
    assert get_threads and all(t != loop_thread for t in get_threads)
