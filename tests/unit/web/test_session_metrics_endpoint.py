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

"""Tests for the ``GET /session/{id}/metrics`` observability endpoint.

Backs the failure mode measured in execution logs (repeated tool timeouts +
runaway context growth): clients poll per-session telemetry without touching
the websocket. Reads only the framework telemetry port on the stored agent.
"""

import pytest

from web.server.session_store import InMemorySessionStore, WebSession, set_session_store

pytestmark = pytest.mark.asyncio


@pytest.fixture
def store():
    """Fresh in-memory store wired into the app module for each test."""
    import web.server.main as main_mod

    original = main_mod.SESSION_STORE
    main_mod.SESSION_STORE = set_session_store(InMemorySessionStore(max_sessions=4))
    yield main_mod.SESSION_STORE
    main_mod.SESSION_STORE = original


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    import web.server.main as main_mod

    with TestClient(main_mod.app) as test_client:
        yield test_client


class _Agent:
    """Stub agent exposing the framework telemetry port."""

    async def get_session_metrics(self):
        return {"context_tokens": 591745, "turns": 3}

    def get_last_turn_cost(self):
        return {"tokens_in": 12000, "cost_usd": 0.04}


async def test_metrics_returns_framework_telemetry(store, client):
    session = WebSession(session_id="s1", agent=_Agent())
    await store.add(session)

    resp = client.get("/session/s1/metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert body["session_id"] == "s1"
    assert body["metrics"]["context_tokens"] == 591745
    assert body["last_turn"]["cost_usd"] == 0.04


async def test_metrics_unknown_session_404(store, client):
    resp = client.get("/session/does-not-exist/metrics")
    assert resp.status_code == 404


async def test_metrics_agent_without_port_degrades_gracefully(store, client):
    session = WebSession(session_id="s2", agent=object())
    await store.add(session)

    resp = client.get("/session/s2/metrics")
    assert resp.status_code == 200
    assert resp.json() == {"session_id": "s2"}


async def test_metrics_agent_error_does_not_500(store, client):
    class _Broken:
        async def get_session_metrics(self):
            raise RuntimeError("boom")

    session = WebSession(session_id="s3", agent=_Broken())
    await store.add(session)

    resp = client.get("/session/s3/metrics")
    assert resp.status_code == 200
    assert resp.json() == {"session_id": "s3"}
