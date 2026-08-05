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

"""FEP-0029: expiry + GC for paused runs — shared across both store backends."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from victor.agent.paused_run_store import (
    InMemoryPausedRunStore,
    ProjectDbPausedRunStore,
)

_REQ = {"id": "r", "title": "t"}


@pytest.fixture(params=["memory", "project"])
def store(request: Any, tmp_path: Path) -> Any:
    if request.param == "memory":
        return InMemoryPausedRunStore()
    return ProjectDbPausedRunStore(db_path=tmp_path / ".victor" / "project.db")


def _save(store: Any, created_at: float) -> str:
    return store.save(session_id="s", agent_id="a", approval_request=_REQ, created_at=created_at)


def test_expire_pending_marks_only_stale_timestamped_runs(store: Any) -> None:
    old = _save(store, created_at=1000.0)  # ancient
    recent = _save(store, created_at=9000.0)  # fresh
    unset = _save(store, created_at=0.0)  # no timestamp → never auto-expire

    # now=10000, max_age=3600 → cutoff 6400: only `old` is past it.
    n = store.expire_pending(max_age_seconds=3600, now=10000.0)
    assert n == 1
    assert store.get(old).status == "expired"
    assert store.get(recent).status == "awaiting_approval"
    assert store.get(unset).status == "awaiting_approval"
    # Expired runs drop out of the pending list.
    assert {r.run_id for r in store.list_pending()} == {recent, unset}


def test_expire_pending_is_idempotent(store: Any) -> None:
    _save(store, created_at=1000.0)
    assert store.expire_pending(max_age_seconds=1, now=10000.0) == 1
    assert store.expire_pending(max_age_seconds=1, now=10000.0) == 0  # nothing left to expire


def test_purge_deletes_terminal_rows_but_keeps_pending(store: Any) -> None:
    pending = _save(store, created_at=1000.0)
    resumed = _save(store, created_at=1000.0)
    expired = _save(store, created_at=1000.0)
    store.mark_resumed(resumed)
    store.expire_pending(max_age_seconds=1, now=2000.0)  # marks `expired` (and `pending`… )

    # Re-add a genuinely-pending one AFTER expiry so it stays pending.
    fresh = _save(store, created_at=5000.0)

    removed = store.purge(before=3000.0)  # terminal rows created before 3000
    assert removed >= 2  # the resumed + expired terminal rows are gone
    assert store.get(resumed) is None and store.get(expired) is None
    # A pending run created before the cutoff is NOT purged (only terminal rows are).
    assert store.get(fresh) is not None


def test_purge_keeps_recent_terminal_rows(store: Any) -> None:
    r = _save(store, created_at=9000.0)
    store.mark_resumed(r)
    assert store.purge(before=3000.0) == 0  # created_at 9000 > 3000 → kept
    assert store.get(r) is not None
