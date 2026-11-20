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

"""FOU-A2d unit suite: /memory + /memory/proposals API routes (FEP-0048).

Covers list (empty + populated, scope filter), set (success + structured 4xx
refusals where the matched secret value is never echoed), delete by id and
by content (with the required-scope negative), and the consolidation
proposal lifecycle (stage -> list -> confirm -> stored; discard -> gone).
Auth posture mirrors the sibling management routes (config/rl): plain
management API, no per-route key dependency.

Databases are fixture-injected onto ``tmp_path`` per the memory-suite
conventions. No network.
"""

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from victor.agent.services.memory_consolidation import MemoryConsolidator
from victor.storage.memory.preference_episodic import (
    MemoryScope,
    PreferenceEpisodicStore,
)

_LEDGER = [
    {"role": "user", "content": "we renamed the service to sandhi-gateway"},
    {"role": "assistant", "content": "noted"},
    {"role": "user", "content": "this repo prefers pytest over unittest"},
]

_GITHUB_PAT = "ghp_" + "a" * 36


class _Server:
    """Stand-in matching the create_router(server) factory convention."""

    def claim_shared_state(self, client_id, surface: str):
        return None


@pytest.fixture
def client(tmp_path: Path):
    from victor.core.database import (
        get_database,
        get_project_database,
        reset_database,
        reset_project_database,
    )
    from victor.agent.services.memory_consolidation import (
        set_shared_memory_consolidator,
        set_shared_memory_store,
    )
    from victor.integrations.api.routes.memory_routes import create_router

    reset_database()
    reset_project_database(tmp_path)
    store = PreferenceEpisodicStore(
        global_db=get_database(tmp_path / "victor.db"),
        project_db=get_project_database(tmp_path),
    )
    consolidator = MemoryConsolidator(store)
    set_shared_memory_store(store)
    set_shared_memory_consolidator(consolidator)

    app = FastAPI()
    app.include_router(create_router(_Server()))
    yield TestClient(app), store, consolidator

    from victor.agent.services.memory_consolidation import reset_shared_memory

    reset_shared_memory()
    reset_database()
    reset_project_database(tmp_path)


class TestListMemories:
    def test_list_empty(self, client):
        http, _store, _consolidator = client
        response = http.get("/memory")

        assert response.status_code == 200
        payload = response.json()
        assert payload["preferences"] == []
        assert payload["episodes"] == []

    def test_list_populated(self, client):
        http, store, _consolidator = client
        store.set_preference("always use uv", scope=MemoryScope.USER)
        store.set_preference("uses pytest", scope=MemoryScope.PROJECT, source="tool")
        store.record_episode("renamed the service", session_id="s1")

        response = http.get("/memory")

        assert response.status_code == 200
        payload = response.json()
        assert [p["content"] for p in payload["preferences"]] == [
            "always use uv",
            "uses pytest",
        ]
        pref = payload["preferences"][1]
        assert pref["scope"] == "project"
        assert pref["source"] == "tool"
        assert pref["confidence"] == 1.0
        assert pref["id"].startswith("pref_")
        assert [e["summary"] for e in payload["episodes"]] == ["renamed the service"]

    def test_list_scope_filter(self, client):
        http, store, _consolidator = client
        store.set_preference("user fact", scope=MemoryScope.USER)
        store.set_preference("project fact", scope=MemoryScope.PROJECT)

        response = http.get("/memory", params={"scope": "user"})

        assert response.status_code == 200
        payload = response.json()
        assert [p["content"] for p in payload["preferences"]] == ["user fact"]

    def test_list_rejects_unknown_scope(self, client):
        http, _store, _consolidator = client
        response = http.get("/memory", params={"scope": "galaxy"})
        assert response.status_code == 400


class TestSetMemory:
    def test_set_success(self, client):
        http, store, _consolidator = client
        response = http.post("/memory", json={"content": "always use uv"})

        assert response.status_code == 200
        payload = response.json()
        assert payload["success"] is True
        assert payload["scope"] == "user"
        assert payload["preference_id"].startswith("pref_")
        stored = store.get_preferences(MemoryScope.USER)
        assert [p.content for p in stored] == ["always use uv"]

    def test_set_project_scope(self, client):
        http, store, _consolidator = client
        response = http.post("/memory", json={"content": "uses pytest", "scope": "project"})

        assert response.status_code == 200
        assert response.json()["scope"] == "project"
        assert [p.content for p in store.get_preferences(MemoryScope.PROJECT)] == ["uses pytest"]

    def test_set_secret_refusal_is_structured_4xx_and_never_echoes_the_secret(self, client):
        http, store, _consolidator = client
        response = http.post("/memory", json={"content": f"token: {_GITHUB_PAT}"})

        assert response.status_code == 400
        payload = response.json()
        assert payload["success"] is False
        assert payload["refusal"] == "SecretContentRefused"
        assert "error" in payload
        body = response.text
        assert _GITHUB_PAT not in body, "matched secret value must never be echoed"
        assert store.get_preferences(MemoryScope.USER) == []
        assert store.get_preferences(MemoryScope.PROJECT) == []

    def test_set_oversized_refusal(self, client):
        http, store, _consolidator = client
        response = http.post("/memory", json={"content": "x" * 5000})

        assert response.status_code == 400
        payload = response.json()
        assert payload["refusal"] == "OversizedContentRefused"
        assert store.get_preferences(MemoryScope.USER) == []

    def test_set_unknown_scope_is_400(self, client):
        http, _store, _consolidator = client
        response = http.post("/memory", json={"content": "x", "scope": "galaxy"})
        assert response.status_code == 400


class TestDeleteMemory:
    def test_delete_by_id(self, client):
        http, store, _consolidator = client
        pref = store.set_preference("always use uv", scope=MemoryScope.USER)

        response = http.delete("/memory", params={"preference_id": pref.id})

        assert response.status_code == 200
        assert response.json()["forgotten"] == 1
        assert store.get_preferences(MemoryScope.USER) == []

    def test_delete_unknown_id_reports_zero(self, client):
        http, _store, _consolidator = client
        response = http.delete("/memory", params={"preference_id": "pref_nope"})
        assert response.status_code == 200
        assert response.json()["forgotten"] == 0

    def test_delete_by_content_requires_scope(self, client):
        http, _store, _consolidator = client
        response = http.delete("/memory", params={"content_match": "uv"})
        assert response.status_code == 400

    def test_delete_by_content_with_scope(self, client):
        http, store, _consolidator = client
        store.set_preference("always use uv", scope=MemoryScope.PROJECT)
        store.set_preference("always use pip", scope=MemoryScope.PROJECT)

        response = http.delete("/memory", params={"content_match": "uv", "scope": "project"})

        assert response.status_code == 200
        assert response.json()["forgotten"] == 1
        assert [p.content for p in store.get_preferences(MemoryScope.PROJECT)] == ["always use pip"]

    def test_delete_without_target_is_400(self, client):
        http, _store, _consolidator = client
        response = http.delete("/memory")
        assert response.status_code == 400

    # Finding 13 (Codex review): an empty or whitespace-only content_match
    # must be rejected with rows preserved — pinned from the API layer.
    @pytest.mark.parametrize("empty_match", ["", "   ", "\t\n "])
    def test_delete_empty_content_match_is_rejected_and_rows_preserved(self, client, empty_match):
        http, store, _consolidator = client
        store.set_preference("always use uv", scope=MemoryScope.USER)
        store.set_preference("uses pytest", scope=MemoryScope.USER)

        response = http.delete("/memory", params={"content_match": empty_match, "scope": "user"})

        assert response.status_code == 400
        payload = response.json()
        assert payload["success"] is False
        assert "content_match" in payload["error"]
        remaining = sorted(p.content for p in store.get_preferences(MemoryScope.USER))
        assert remaining == ["always use uv", "uses pytest"]


class TestProposalLifecycle:
    def test_stage_list_confirm(self, client):
        http, store, consolidator = client
        consolidator.run(_LEDGER)

        response = http.get("/memory/proposals")
        assert response.status_code == 200
        proposals = response.json()["proposals"]
        assert [p["summary"] for p in proposals] == [
            "we renamed the service to sandhi-gateway",
            "this repo prefers pytest over unittest",
        ]

        confirm = http.post("/memory/proposals/confirm")
        assert confirm.status_code == 200
        assert confirm.json()["count"] == 2

        assert consolidator.staged_proposals() == []
        summaries = [e.summary for e in store.recent_episodes(limit=10)]
        assert "we renamed the service to sandhi-gateway" in summaries
        assert "this repo prefers pytest over unittest" in summaries

        assert http.get("/memory/proposals").json()["proposals"] == []

    def test_confirm_with_nothing_staged(self, client):
        http, _store, _consolidator = client
        response = http.post("/memory/proposals/confirm")
        assert response.status_code == 200
        assert response.json()["count"] == 0

    def test_discard_drops_proposals_without_writing(self, client):
        http, store, consolidator = client
        consolidator.run(_LEDGER)

        response = http.post("/memory/proposals/discard")
        assert response.status_code == 200
        assert response.json()["discarded"] == 2
        assert consolidator.staged_proposals() == []
        assert store.recent_episodes(limit=10) == []

    def test_proposals_list_empty(self, client):
        http, _store, _consolidator = client
        response = http.get("/memory/proposals")
        assert response.status_code == 200
        assert response.json()["proposals"] == []


# =============================================================================
# Finding 4 (Codex review): enforced single-principal posture for /memory.
# When the server posture refuses (more than one configured credential
# without an explicit single-operator declaration), every /memory surface
# returns the typed MultiPrincipalRefused payload and the store is untouched.
# =============================================================================


class _MultiPrincipalServer(_Server):
    """Stand-in whose posture refuses shared-state surfaces."""

    def __init__(self) -> None:
        self.refused_surfaces: list[str] = []

    def claim_shared_state(self, client_id, surface: str):
        self.refused_surfaces.append(surface)
        return {
            "success": False,
            "refusal": "MultiPrincipalRefused",
            "surface": surface,
            "error": (
                "memory is refused while more than one API credential is "
                "configured: this server resolves one shared memory store and "
                "cannot attribute it per credential until owner isolation "
                "lands (VAS-06 / FOU-A4a). Declare single-trusted-operator "
                "mode explicitly to keep the shared posture."
            ),
        }


@pytest.fixture
def multi_principal_client(tmp_path: Path):
    from victor.core.database import (
        get_database,
        get_project_database,
        reset_database,
        reset_project_database,
    )
    from victor.agent.services.memory_consolidation import (
        set_shared_memory_consolidator,
        set_shared_memory_store,
    )
    from victor.integrations.api.routes.memory_routes import create_router

    reset_database()
    reset_project_database(tmp_path)
    store = PreferenceEpisodicStore(
        global_db=get_database(tmp_path / "victor.db"),
        project_db=get_project_database(tmp_path),
    )
    set_shared_memory_store(store)
    set_shared_memory_consolidator(MemoryConsolidator(store))

    server = _MultiPrincipalServer()
    app = FastAPI()
    app.include_router(create_router(server))
    yield TestClient(app), store, server

    from victor.agent.services.memory_consolidation import reset_shared_memory

    reset_shared_memory()
    reset_database()
    reset_project_database(tmp_path)


class TestMultiPrincipalPosture:
    def test_list_refused(self, multi_principal_client):
        http, store, server = multi_principal_client
        response = http.get("/memory")
        assert response.status_code == 409
        payload = response.json()
        assert payload["refusal"] == "MultiPrincipalRefused"
        assert "single-trusted-operator" in payload["error"]
        assert payload["surface"] == "memory"
        assert server.refused_surfaces == ["memory"]

    def test_set_refused_and_store_untouched(self, multi_principal_client):
        http, store, server = multi_principal_client
        response = http.post("/memory", json={"content": "always use uv"})
        assert response.status_code == 409
        assert response.json()["refusal"] == "MultiPrincipalRefused"
        assert store.get_preferences(MemoryScope.USER) == []

    def test_forget_refused_and_rows_preserved(self, multi_principal_client):
        http, store, _server = multi_principal_client
        store.set_preference("always use uv", scope=MemoryScope.USER)
        response = http.delete("/memory", params={"preference_id": "pref_any"})
        assert response.status_code == 409
        assert sorted(p.content for p in store.get_preferences(MemoryScope.USER)) == [
            "always use uv"
        ]

    def test_proposal_surfaces_refused(self, multi_principal_client):
        http, _store, server = multi_principal_client
        assert http.get("/memory/proposals").status_code == 409
        assert http.post("/memory/proposals/confirm").status_code == 409
        assert http.post("/memory/proposals/discard").status_code == 409
        assert len(server.refused_surfaces) == 3
