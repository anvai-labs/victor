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

"""Single-trusted-operator posture enforcement (Codex findings 3/4, wave 1).

The HTTP server resolves ONE server-wide client and ONE shared memory store:
conversation addressing (chat), export and reset (and /memory) carry no
owner/session in the lookup, so a second authenticated credential can read or
destroy the first user's conversation and preferences. Until owner isolation
lands (VAS-06 / FOU-A4a, paused pending D4), the enforced posture is:
more than one configured credential + no explicit single-operator declaration
=> the shared-state surfaces refuse with the typed
``MultiPrincipalRefused`` payload (HTTP 409), and the first user's data is
preserved.

Acceptance pinned here (RED-first per the Codex contract):
- two-credential interleaved chat/export/reset/stream/resume: the SECOND
  credential is refused with the typed limitation error (positive);
- the first user's conversation is intact after every refused attempt
  (preservation);
- single-credential and keyless servers are behaviorally unchanged
  (negative — legit operators unaffected);
- the explicit ``single_operator_mode`` declaration restores the shared
  multi-key posture for one trusted operator.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, AsyncIterator, Dict, List, Optional

import pytest
from httpx import ASGITransport, AsyncClient

from victor.integrations.api import fastapi_server
from victor.storage.memory.preference_episodic import MemoryScope


class _FakeVictorClient:
    """Records conversation-addressing calls so preservation is observable."""

    def __init__(self) -> None:
        self.chats: List[str] = []
        self.streams: List[str] = []
        self.resets: int = 0
        self.closed: bool = False

    async def initialize(self) -> Any:
        return self

    async def chat(self, message: str) -> SimpleNamespace:
        self.chats.append(message)
        return SimpleNamespace(content=f"echo:{message}", tool_calls=[], status="ok")

    async def stream_chat(self, message: str) -> AsyncIterator[SimpleNamespace]:
        self.streams.append(message)
        yield SimpleNamespace(content=f"stream:{message}", tool_calls=None)

    async def reset_conversation(self) -> None:
        self.resets += 1

    async def get_messages(
        self, limit: Optional[int] = None, role: Optional[str] = None
    ) -> List[Dict[str, str]]:
        return [{"role": "user", "content": "alice's private context"}]

    async def resume(self, run_id: str, decision: Any) -> SimpleNamespace:
        return SimpleNamespace(content="resumed", tool_calls=[], status="ok")

    async def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_client() -> _FakeVictorClient:
    return _FakeVictorClient()


def _create_server(
    monkeypatch,
    tmp_path: Path,
    fake_client: _FakeVictorClient,
    **server_kwargs: Any,
) -> fastapi_server.VictorFastAPIServer:
    monkeypatch.setattr(
        fastapi_server,
        "load_fastapi_router_registrations",
        lambda *, workspace_root: [],
    )
    server = fastapi_server.VictorFastAPIServer(
        workspace_root=str(tmp_path),
        enable_graphql=False,
        **server_kwargs,
    )
    server._victor_client = fake_client
    return server


_ALICE_HEADERS = {"Authorization": "Bearer sk-alice"}
_BOB_HEADERS = {"Authorization": "Bearer sk-bob"}
_CHAT_BODY = {"messages": [{"role": "user", "content": "hello"}]}


def _assert_refusal_payload(payload: Dict[str, Any], surface: str) -> None:
    assert payload["success"] is False
    assert payload["refusal"] == "MultiPrincipalRefused"
    assert payload["surface"] == surface
    # The error must NAME the limitation: shared state, multi-credential
    # posture, and the path out (explicit single-operator declaration).
    error = payload["error"]
    assert "single-trusted-operator" in error
    assert "shared" in error


# =============================================================================
# Finding 3: two-credential interleaved chat / export / reset / stream
# =============================================================================


@pytest.mark.asyncio
async def test_second_credential_chat_refused_first_users_conversation_intact(
    monkeypatch, tmp_path, fake_client
):
    server = _create_server(
        monkeypatch,
        tmp_path,
        fake_client,
        api_keys={"sk-alice": "alice", "sk-bob": "bob"},
    )
    transport = ASGITransport(app=server.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        ok = await client.post("/chat", json=_CHAT_BODY, headers=_ALICE_HEADERS)
        refused = await client.post("/chat", json=_CHAT_BODY, headers=_BOB_HEADERS)
        after = await client.post("/chat", json=_CHAT_BODY, headers=_ALICE_HEADERS)

    assert ok.status_code == 200
    assert refused.status_code == 409
    _assert_refusal_payload(refused.json(), "chat")
    # Preservation: bob's refused turn never reached the shared conversation.
    assert fake_client.chats == ["hello", "hello"]
    assert after.status_code == 200


@pytest.mark.asyncio
async def test_second_credential_export_refused_and_first_users_data_not_disclosed(
    monkeypatch, tmp_path, fake_client
):
    server = _create_server(
        monkeypatch,
        tmp_path,
        fake_client,
        api_keys={"sk-alice": "alice", "sk-bob": "bob"},
    )
    transport = ASGITransport(app=server.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post("/chat", json=_CHAT_BODY, headers=_ALICE_HEADERS)
        refused = await client.get("/conversation/export", headers=_BOB_HEADERS)
        allowed = await client.get("/conversation/export", headers=_ALICE_HEADERS)

    assert refused.status_code == 409
    _assert_refusal_payload(refused.json(), "conversation/export")
    assert "alice's private context" not in refused.text
    assert allowed.status_code == 200
    assert "alice's private context" in allowed.text


@pytest.mark.asyncio
async def test_second_credential_reset_refused_and_conversation_preserved(
    monkeypatch, tmp_path, fake_client
):
    server = _create_server(
        monkeypatch,
        tmp_path,
        fake_client,
        api_keys={"sk-alice": "alice", "sk-bob": "bob"},
    )
    transport = ASGITransport(app=server.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post("/chat", json=_CHAT_BODY, headers=_ALICE_HEADERS)
        refused = await client.post("/conversation/reset", headers=_BOB_HEADERS)
        resumed = await client.post("/chat", json=_CHAT_BODY, headers=_ALICE_HEADERS)

    assert refused.status_code == 409
    _assert_refusal_payload(refused.json(), "conversation/reset")
    assert fake_client.resets == 0, "a refused reset must not destroy shared history"
    assert resumed.status_code == 200
    assert fake_client.chats == ["hello", "hello"]


@pytest.mark.asyncio
async def test_second_credential_stream_and_resume_refused(monkeypatch, tmp_path, fake_client):
    server = _create_server(
        monkeypatch,
        tmp_path,
        fake_client,
        api_keys={"sk-alice": "alice", "sk-bob": "bob"},
    )
    transport = ASGITransport(app=server.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        claim = await client.post("/chat", json=_CHAT_BODY, headers=_ALICE_HEADERS)
        assert claim.status_code == 200
        stream_refused = await client.post("/chat/stream", json=_CHAT_BODY, headers=_BOB_HEADERS)
        wire_refused = await client.post("/chat/wire-stream", json=_CHAT_BODY, headers=_BOB_HEADERS)
        resume_refused = await client.post(
            "/chat/resume",
            json={"run_id": "run-1", "approved": True},
            headers=_BOB_HEADERS,
        )

    assert stream_refused.status_code == 409
    _assert_refusal_payload(stream_refused.json(), "chat")
    assert wire_refused.status_code == 409
    _assert_refusal_payload(wire_refused.json(), "chat")
    assert resume_refused.status_code == 409
    _assert_refusal_payload(resume_refused.json(), "chat/resume")
    # Refusal happens before the shared client is touched: alice's turn is
    # the only stream/chat activity on the shared conversation.
    assert fake_client.streams == []
    assert fake_client.chats == ["hello"]


# =============================================================================
# Negative controls: single-credential and keyless operators unaffected
# =============================================================================


@pytest.mark.asyncio
async def test_single_credential_posture_unchanged(monkeypatch, tmp_path, fake_client):
    server = _create_server(monkeypatch, tmp_path, fake_client, api_keys={"sk-solo": "solo"})
    transport = ASGITransport(app=server.app)
    headers = {"Authorization": "Bearer sk-solo"}
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        chat = await client.post("/chat", json=_CHAT_BODY, headers=headers)
        export = await client.get("/conversation/export", headers=headers)
        reset = await client.post("/conversation/reset", headers=headers)

    assert chat.status_code == 200
    assert export.status_code == 200
    assert reset.status_code == 200
    assert fake_client.resets == 1
    assert fake_client.chats == ["hello"]


@pytest.mark.asyncio
async def test_keyless_unauthenticated_server_unaffected(monkeypatch, tmp_path, fake_client):
    server = _create_server(
        monkeypatch,
        tmp_path,
        fake_client,
        allow_unauthenticated=True,
    )
    transport = ASGITransport(app=server.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        chat = await client.post("/chat", json=_CHAT_BODY)
        reset = await client.post("/conversation/reset")

    assert chat.status_code == 200
    assert reset.status_code == 200


@pytest.mark.asyncio
async def test_explicit_single_operator_declaration_restores_shared_multi_key_posture(
    monkeypatch, tmp_path, fake_client
):
    """The documented operator escape: one trusted operator with key aliases."""
    server = _create_server(
        monkeypatch,
        tmp_path,
        fake_client,
        api_keys={"sk-alice": "operator", "sk-bob": "operator"},
        single_operator_mode=True,
    )
    transport = ASGITransport(app=server.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        via_bob = await client.post("/chat", json=_CHAT_BODY, headers=_BOB_HEADERS)
        reset = await client.post("/conversation/reset", headers=_BOB_HEADERS)

    assert via_bob.status_code == 200
    assert reset.status_code == 200
    assert fake_client.chats == ["hello"]
    assert fake_client.resets == 1


@pytest.mark.asyncio
async def test_single_operator_mode_env_alias(monkeypatch, tmp_path, fake_client):
    monkeypatch.setenv("VICTOR_SINGLE_OPERATOR_MODE", "1")
    server = _create_server(
        monkeypatch,
        tmp_path,
        fake_client,
        api_keys={"sk-alice": "operator", "sk-bob": "operator"},
    )
    assert server.claim_shared_state("operator", "chat") is None


def test_posture_helper_rejects_on_two_keys_without_declaration(tmp_path):
    server = fastapi_server.VictorFastAPIServer(
        workspace_root=str(tmp_path),
        enable_graphql=False,
        allow_unauthenticated=True,
        api_keys={"k1": "a", "k2": "b"},
    )
    # First principal claims; a different credential is refused.
    assert server.claim_shared_state("a", "chat") is None
    assert server.claim_shared_state("b", "chat") is not None
    assert server.claim_shared_state("a", "chat") is None
    assert server._shared_state_principal == "a"


def test_posture_helper_fails_closed_on_unresolved_principal(tmp_path):
    """Posture active + no resolved principal -> refuse (never address shared
    state unattributed)."""
    server = fastapi_server.VictorFastAPIServer(
        workspace_root=str(tmp_path),
        enable_graphql=False,
        allow_unauthenticated=True,
        api_keys={"k1": "a", "k2": "b"},
    )
    assert server.claim_shared_state(None, "chat") is not None


def test_multi_principal_refused_is_typed_and_names_the_limitation():
    from victor.integrations.api.operator_posture import MultiPrincipalRefused

    exc = MultiPrincipalRefused("chat")
    assert "single-trusted-operator" in str(exc)
    assert "shared" in str(exc)


# =============================================================================
# Finding 4: /memory under the REAL server (auth dependency resolves
# request.state.client_id; sticky claim over the shared store).
# =============================================================================


@pytest.fixture
def memory_env(tmp_path):
    from victor.agent.services.memory_consolidation import (
        reset_shared_memory,
        set_shared_memory_store,
    )
    from victor.core.database import (
        get_database,
        get_project_database,
        reset_database,
        reset_project_database,
    )
    from victor.storage.memory.preference_episodic import PreferenceEpisodicStore

    reset_database()
    reset_project_database(tmp_path)
    store = PreferenceEpisodicStore(
        global_db=get_database(tmp_path / "victor.db"),
        project_db=get_project_database(tmp_path),
    )
    set_shared_memory_store(store)
    yield store
    reset_shared_memory()
    reset_database()
    reset_project_database(tmp_path)


@pytest.mark.asyncio
async def test_memory_second_credential_refused_and_store_rows_preserved(
    monkeypatch, tmp_path, memory_env
):
    store = memory_env
    monkeypatch.setattr(
        fastapi_server,
        "load_fastapi_router_registrations",
        lambda *, workspace_root: [],
    )
    server = fastapi_server.VictorFastAPIServer(
        workspace_root=str(tmp_path),
        enable_graphql=False,
        api_keys={"sk-alice": "alice", "sk-bob": "bob"},
    )
    transport = ASGITransport(app=server.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        ok = await client.post(
            "/memory",
            json={"content": "always use uv"},
            headers=_ALICE_HEADERS,
        )
        refused_set = await client.post(
            "/memory",
            json={"content": "bob's directive"},
            headers=_BOB_HEADERS,
        )
        refused_list = await client.get("/memory", headers=_BOB_HEADERS)
        refused_forget = await client.delete(
            "/memory", params={"preference_id": "pref_any"}, headers=_BOB_HEADERS
        )

    assert ok.status_code == 200
    assert refused_set.status_code == 409
    _assert_refusal_payload(refused_set.json(), "memory")
    assert refused_list.status_code == 409
    assert refused_forget.status_code == 409
    # Preservation: bob's refused writes/reads/forgets never touched the store.
    assert [p.content for p in store.get_preferences(MemoryScope.USER)] == ["always use uv"]
