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

"""FEP-0037 conformance guard: the frozen /v1/classify contract.

These tests are the enforcement mechanism for the versioned surface — a
change to the request/response shape, status codes, or attribution binding
that breaks any assertion here is a contract violation requiring a FEP update
(and, if breaking, a /v2 rather than an in-place edit).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi.testclient import TestClient

from victor.integrations.api import fastapi_server
from victor.integrations.api.routes.classify_routes import (
    _extract_json_object,
    _resolve_schema,
)

_TRIAGE_RESULT = {
    "sensitive": True,
    "category": "money",
    "reason": "payment confirmation requested",
    "confidence": 0.9,
}


class _FakeProvider:
    """Scripted provider: returns queued CompletionResponse-like objects."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[dict] = []
        self.closed = False
        self.model = "fake-model"

    async def chat(self, messages, *, model, temperature, max_tokens, **kwargs):
        self.calls.append({"messages": messages, "model": model})
        content = self._responses.pop(0) if self._responses else ""
        return SimpleNamespace(
            content=content,
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            model="fake-model",
        )

    async def close(self):
        self.closed = True


class _FakeOrchestrator:
    def __init__(self, provider: _FakeProvider):
        self.provider_manager = SimpleNamespace(current_provider=provider)
        self.model = "fake-model"


class _HangingProvider(_FakeProvider):
    async def chat(self, messages, *, model, temperature, max_tokens, **kwargs):
        import asyncio

        await asyncio.sleep(3600)
        raise AssertionError("should have been cancelled by timeout")


def _create_server(monkeypatch, tmp_path: Path, orchestrator, **server_kwargs):
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
    server._orchestrator = orchestrator
    return server


def _client(server) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=server.app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


# ── pure helpers ─────────────────────────────────────────────────────────────


def test_extract_json_object_tolerates_prose_and_fences():
    text = 'Sure! ```json {"a": 1} ``` hope that helps'
    assert _extract_json_object(text) == {"a": 1}


def test_extract_json_object_rejects_non_objects():
    assert _extract_json_object("[1, 2]") is None
    assert _extract_json_object("no json") is None
    assert _extract_json_object("") is None


def test_resolve_schema_unknown_preset_is_rejected():
    import pytest
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        _resolve_schema(SimpleNamespace(preset="nope", output_schema=None))
    assert exc.value.status_code == 422


def test_resolve_schema_preset_is_frozen_contract():
    schema = _resolve_schema(SimpleNamespace(preset="triage.v1", output_schema=None))
    assert schema["required"] == ["sensitive", "category", "reason"]


# ── frozen contract ──────────────────────────────────────────────────────────


def _make_server(monkeypatch, tmp_path, provider, api_keys=None):
    return _create_server(monkeypatch, tmp_path, _FakeOrchestrator(provider), api_keys=api_keys)


def test_conformance_route_and_contract_shape(monkeypatch, tmp_path):
    """FROZEN: method, path, and response field set. Breaking these = /v2."""
    provider = _FakeProvider([json.dumps(_TRIAGE_RESULT)])
    server = _make_server(monkeypatch, tmp_path, provider)
    openapi = server.app.openapi()
    op = openapi["paths"]["/v1/classify"]["post"]
    assert op["tags"] == ["Classify"]
    body = op["requestBody"]["content"]["application/json"]["schema"]
    props = body["$ref"].split("/")[-1]
    req_props = openapi["components"]["schemas"][props]["properties"]
    for field in ("input", "schema", "preset", "provider", "model", "max_tokens", "timeout_ms"):
        assert field in req_props, field
    resp_ref = op["responses"]["200"]["content"]["application/json"]["schema"]["$ref"]
    resp_props = openapi["components"]["schemas"][resp_ref.split("/")[-1]]["properties"]
    for field in ("result", "usage", "model", "latency_ms", "error", "parse_retries"):
        assert field in resp_props, field


@pytest.mark.asyncio
async def test_classify_success_shape_and_usage(monkeypatch, tmp_path):
    provider = _FakeProvider([json.dumps(_TRIAGE_RESULT)])
    server = _make_server(monkeypatch, tmp_path, provider)
    async with _client(server) as client:
        response = await client.post(
            "/v1/classify",
            json={
                "input": "pay this invoice",
                "preset": "triage.v1",
                "sender": "+1555",
                "source": "whatsapp",
            },
        )
    assert response.status_code == 200
    body = response.json()
    assert body["result"] == _TRIAGE_RESULT
    assert body["error"] is None
    assert body["usage"]["total_tokens"] == 15
    assert body["model"] == "fake-model"
    assert body["latency_ms"] >= 0
    assert provider.calls[0]["messages"][0]["role"] == "system"
    assert '"sensitive"' in provider.calls[0]["messages"][0]["content"]
    assert "pay this invoice" in provider.calls[0]["messages"][1]["content"]


@pytest.mark.asyncio
async def test_classify_retries_unparseable_then_422(monkeypatch, tmp_path):
    provider = _FakeProvider(["not json at all", "still not json"])
    server = _make_server(monkeypatch, tmp_path, provider)
    async with _client(server) as client:
        response = await client.post(
            "/v1/classify",
            json={"input": "x", "preset": "triage.v1", "timeout_ms": 5000},
        )
    assert response.status_code == 422
    body = response.json()
    assert body["error"] and "parseable JSON" in body["error"]
    assert body["latency_ms"] >= 0
    assert len(provider.calls) == 2  # retried exactly once


@pytest.mark.asyncio
async def test_classify_recovers_on_second_attempt(monkeypatch, tmp_path):
    provider = _FakeProvider(["oops", json.dumps(_TRIAGE_RESULT)])
    server = _make_server(monkeypatch, tmp_path, provider)
    async with _client(server) as client:
        response = await client.post(
            "/v1/classify",
            json={"input": "x", "preset": "triage.v1", "timeout_ms": 20000},
        )
    assert response.status_code == 200
    assert response.json()["result"] == _TRIAGE_RESULT


@pytest.mark.asyncio
async def test_classify_timeout_returns_504_with_latency(monkeypatch, tmp_path):
    provider = _HangingProvider([])
    server = _make_server(monkeypatch, tmp_path, provider)
    async with _client(server) as client:
        response = await client.post(
            "/v1/classify",
            json={"input": "x", "preset": "triage.v1", "timeout_ms": 300},
        )
    assert response.status_code == 504
    body = response.json()
    assert "timeout" in body["error"]
    assert body["latency_ms"] >= 250


@pytest.mark.asyncio
async def test_classify_empty_input_is_422(monkeypatch, tmp_path):
    server = _make_server(monkeypatch, tmp_path, _FakeProvider([]))
    async with _client(server) as client:
        response = await client.post("/v1/classify", json={"input": "   ", "preset": "triage.v1"})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_classify_unknown_preset_is_422(monkeypatch, tmp_path):
    server = _make_server(monkeypatch, tmp_path, _FakeProvider([]))
    async with _client(server) as client:
        response = await client.post("/v1/classify", json={"input": "x", "preset": "nope.v9"})
    assert response.status_code == 422
    assert "unknown preset" in response.json()["error"]


@pytest.mark.asyncio
async def test_classify_requires_schema_or_preset(monkeypatch, tmp_path):
    server = _make_server(monkeypatch, tmp_path, _FakeProvider([]))
    async with _client(server) as client:
        response = await client.post("/v1/classify", json={"input": "x"})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_classify_managed_provider_override_is_closed(monkeypatch, tmp_path):
    """Provider override creates a managed provider and MUST dispose it."""
    created: list = []

    class _Managed(_FakeProvider):
        async def chat(self, messages, *, model, temperature, max_tokens, **kwargs):
            return SimpleNamespace(
                content=json.dumps(_TRIAGE_RESULT),
                usage={"total_tokens": 7},
                model="managed-model",
            )

    def fake_factory(provider_name, model, api_key):
        p = _Managed([json.dumps(_TRIAGE_RESULT)])
        p.managed_model = model
        created.append(p)
        return p

    import victor.config.api_keys as api_keys_mod
    import victor.providers.factory as factory_mod

    monkeypatch.setattr(api_keys_mod, "get_api_key", lambda name: "sk-test")
    monkeypatch.setattr(factory_mod.ManagedProviderFactory, "create", staticmethod(fake_factory))

    server = _make_server(monkeypatch, tmp_path, _FakeProvider([]))
    async with _client(server) as client:
        response = await client.post(
            "/v1/classify",
            json={"input": "x", "preset": "triage.v1", "provider": "zai", "model": "glm-5.3"},
        )
    assert response.status_code == 200
    assert created and created[0].closed is True
    assert created[0].managed_model == "glm-5.3"


@pytest.mark.asyncio
async def test_classify_binds_authenticated_subject(monkeypatch, tmp_path):
    """FEP-0020 attribution join: the auth subject reaches the usage record."""
    captured: dict = {}

    import victor.core.context as context_mod

    original = context_mod.bind_attribution

    def spy(subject_id):
        captured["subject"] = subject_id
        return original(subject_id)

    monkeypatch.setattr("victor.integrations.api.routes.classify_routes.bind_attribution", spy)
    provider = _FakeProvider([json.dumps(_TRIAGE_RESULT)])
    server = _make_server(monkeypatch, tmp_path, provider, api_keys={"sk-hub": "hub"})
    async with _client(server) as client:
        response = await client.post(
            "/v1/classify",
            json={"input": "x", "preset": "triage.v1"},
            headers={"Authorization": "Bearer sk-hub"},
        )
    assert response.status_code == 200
    assert captured["subject"] == "hub"


@pytest.mark.asyncio
async def test_classify_rejects_bad_key(monkeypatch, tmp_path):
    provider = _FakeProvider([json.dumps(_TRIAGE_RESULT)])
    server = _make_server(monkeypatch, tmp_path, provider, api_keys={"sk-hub": "hub"})
    async with _client(server) as client:
        response = await client.post(
            "/v1/classify",
            json={"input": "x", "preset": "triage.v1"},
            headers={"Authorization": "Bearer wrong"},
        )
    assert response.status_code == 401
    assert provider.calls == []
