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
    assert body["error"] and "JSON object after retry" in body["error"]
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

    async def fake_factory(provider_name, model, **kwargs):
        # REAL signature: ManagedProviderFactory.create is async (P1 finding).
        p = _Managed([json.dumps(_TRIAGE_RESULT)])
        p.managed_model = model
        created.append(p)
        return p

    import victor.providers.factory as factory_mod

    server = _make_server(monkeypatch, tmp_path, _FakeProvider([]))
    monkeypatch.setattr(
        "victor.config.settings.load_settings",
        lambda **kwargs: SimpleNamespace(get_provider_settings=lambda name: {"api_key": "sk-test"}),
    )
    monkeypatch.setattr(factory_mod.ManagedProviderFactory, "create", staticmethod(fake_factory))
    server._get_orchestrator = AsyncMock(
        side_effect=AssertionError("No agent needed for explicit classify route")
    )
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


@pytest.mark.asyncio
async def test_classify_schema_violation_retries_then_422(monkeypatch, tmp_path):
    """FEP-0037: the output schema is ENFORCED — a shape-valid JSON object
    that violates required fields is a retry, then a 422."""
    provider = _FakeProvider(
        [
            json.dumps({"sensitive": True}),  # missing required 'category'/'reason'
            json.dumps({"sensitive": True}),  # still violating after the retry
        ]
    )
    server = _make_server(monkeypatch, tmp_path, provider)
    async with _client(server) as client:
        response = await client.post(
            "/v1/classify",
            json={"input": "x", "preset": "triage.v1", "timeout_ms": 20000},
        )
    assert response.status_code == 422
    assert len(provider.calls) == 2  # retried exactly once
    assert "schema-conformant" in response.json()["error"]


@pytest.mark.asyncio
async def test_classify_schema_conforming_second_attempt_passes(monkeypatch, tmp_path):
    provider = _FakeProvider(
        [
            json.dumps({"unexpected": "shape"}),
            json.dumps({"sensitive": False, "category": "none", "reason": "ok"}),
        ]
    )
    server = _make_server(monkeypatch, tmp_path, provider)
    async with _client(server) as client:
        response = await client.post(
            "/v1/classify",
            json={"input": "x", "preset": "triage.v1", "timeout_ms": 20000},
        )
    assert response.status_code == 200
    assert response.json()["result"]["sensitive"] is False


@pytest.mark.asyncio
async def test_classify_schema_and_preset_are_mutually_exclusive(monkeypatch, tmp_path):
    provider = _FakeProvider([])
    server = _make_server(monkeypatch, tmp_path, provider)
    async with _client(server) as client:
        response = await client.post(
            "/v1/classify",
            json={"input": "x", "preset": "triage.v1", "schema": {"type": "object"}},
        )
    assert response.status_code == 422
    assert "mutually exclusive" in response.json()["error"]


@pytest.mark.asyncio
async def test_classify_error_responses_carry_request_id_header(monkeypatch, tmp_path):
    provider = _FakeProvider(["garbage", "more garbage"])
    server = _make_server(monkeypatch, tmp_path, provider)
    async with _client(server) as client:
        response = await client.post(
            "/v1/classify",
            json={"input": "x", "preset": "triage.v1", "timeout_ms": 20000},
        )
    assert response.status_code == 422
    assert response.headers.get("x-victor-request-id", "").startswith("cls-")


@pytest.mark.asyncio
async def test_classify_forwards_reasoning_effort_on_both_parse_attempts(monkeypatch, tmp_path):
    class Provider(_FakeProvider):
        async def chat(self, messages, **kwargs):
            self.calls.append(kwargs)
            content = "invalid" if len(self.calls) == 1 else json.dumps(_TRIAGE_RESULT)
            return SimpleNamespace(content=content, model="gpt-6-luna", usage={})

    provider = Provider([])
    server = _make_server(monkeypatch, tmp_path, provider)
    async with _client(server) as client:
        result = await client.post(
            "/v1/classify",
            json={
                "input": "hello",
                "preset": "triage.v1",
                "model": "gpt-6-luna",
                "reasoning_effort": "medium",
            },
        )
    assert result.status_code == 200
    assert len(provider.calls) == 2
    assert all(call.get("reasoning_effort") == "medium" for call in provider.calls)


@pytest.mark.asyncio
async def test_classify_invalid_effort_rejected_without_provider_call(monkeypatch, tmp_path):
    provider = _FakeProvider([])
    server = _make_server(monkeypatch, tmp_path, provider)
    async with _client(server) as client:
        result = await client.post(
            "/v1/classify",
            json={"input": "hello", "preset": "triage.v1", "reasoning_effort": "typo"},
        )
    assert result.status_code == 422
    assert not provider.calls


@pytest.mark.asyncio
async def test_classify_override_keeps_configured_gateway(monkeypatch):
    from victor.integrations.api.routes.classify_routes import ClassifyRequest, _resolve_provider

    configured = {
        "api_key": "synthetic-key",
        "gateway": {"url": "https://gateway.example", "virtual_key": "synthetic-key"},
        "timeout": 45,
    }
    settings = SimpleNamespace(get_provider_settings=lambda name: dict(configured))
    monkeypatch.setattr("victor.config.settings.load_settings", lambda **kwargs: settings)
    created = AsyncMock(return_value=object())
    monkeypatch.setattr("victor.providers.factory.ManagedProviderFactory.create", created)
    provider, disposable = await _resolve_provider(
        None,
        ClassifyRequest(input="hello", preset="triage.v1", provider="openai", model="gpt-6-luna"),
    )
    assert provider is disposable
    assert created.call_args.kwargs["gateway"] == configured["gateway"]
    assert created.call_args.kwargs["enable_resilience"] is False
    assert created.call_args.kwargs["enable_rate_limiting"] is False


@pytest.mark.asyncio
async def test_classify_gateway_policy_denial_is_terminal_and_sanitized(monkeypatch, tmp_path):
    from victor.core.errors import ProviderPolicyError

    class Provider(_FakeProvider):
        async def chat(self, **kwargs):
            self.calls.append(kwargs)
            raise ProviderPolicyError("policy_blocked", receipt="a" * 32)

    provider = Provider([])
    server = _make_server(monkeypatch, tmp_path, provider)
    async with _client(server) as client:
        result = await client.post(
            "/v1/classify", json={"input": "synthetic", "preset": "triage.v1"}
        )
    assert result.status_code == 403
    assert len(provider.calls) == 1
    assert result.headers["x-sandhi-policy-receipt"] == "a" * 32
    assert result.json()["latency_ms"] >= 0


@pytest.mark.asyncio
async def test_classify_schema_requires_key_and_exposes_only_request_contract(
    monkeypatch, tmp_path
):
    provider = _FakeProvider([])
    server = _make_server(
        monkeypatch, tmp_path, provider, api_keys={"synthetic-key": "message-hub"}
    )
    async with _client(server) as client:
        denied = await client.get("/v1/classify/schema")
        allowed = await client.get(
            "/v1/classify/schema", headers={"Authorization": "Bearer synthetic-key"}
        )
    assert denied.status_code == 401
    assert allowed.status_code == 200
    assert "reasoning_effort" in allowed.json()["properties"]
    assert "paths" not in allowed.json()
    assert not provider.calls


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "timeout", "cancel"])
async def test_real_managed_classify_owns_dispatch_and_cleanup(monkeypatch, tmp_path, outcome):
    import asyncio
    from victor.providers.factory import ManagedProviderFactory

    started = asyncio.Event()
    cancelled = []

    class Provider(_FakeProvider):
        name = "openai"

        async def chat(self, **kwargs):
            self.calls.append(kwargs)
            started.set()
            if outcome == "success":
                return SimpleNamespace(
                    content=json.dumps(_TRIAGE_RESULT), model="gpt-6-luna", usage={}
                )
            try:
                await asyncio.sleep(3600)
            finally:
                cancelled.append(True)

    base = Provider([])
    server = _make_server(monkeypatch, tmp_path, base)
    monkeypatch.setattr(
        "victor.config.settings.load_settings",
        lambda **k: SimpleNamespace(get_provider_settings=lambda name: {"api_key": "synthetic"}),
    )
    monkeypatch.setattr("victor.providers.factory.ProviderRegistry.create", lambda *a, **k: base)
    created = []
    original = ManagedProviderFactory.create_from_config

    async def create(config):
        managed = await original(config)
        created.append(managed)
        return managed

    monkeypatch.setattr(ManagedProviderFactory, "create_from_config", create)
    try:
        async with _client(server) as client:
            task = asyncio.create_task(
                client.post(
                    "/v1/classify",
                    json={
                        "input": "hello",
                        "preset": "triage.v1",
                        "provider": "openai",
                        "model": "gpt-6-luna",
                        "timeout_ms": 250,
                    },
                )
            )
            if outcome == "cancel":
                await asyncio.wait_for(started.wait(), 3)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                result = await asyncio.wait_for(task, 3)
                assert result.status_code == (200 if outcome == "success" else 504)
        assert created and created[0]._request_manager is None
        assert base.closed
        if outcome != "success":
            assert cancelled
    finally:
        for managed in created:
            await managed.shutdown()


@pytest.mark.asyncio
async def test_parse_retry_shares_operation_deadline(monkeypatch, tmp_path):
    import asyncio

    class Provider(_FakeProvider):
        async def chat(self, **kwargs):
            self.calls.append(kwargs)
            await asyncio.sleep(0.20)
            return SimpleNamespace(content="invalid", model="fake-model", usage={})

    base = Provider([])
    server = _make_server(monkeypatch, tmp_path, base)
    async with _client(server) as client:
        result = await client.post(
            "/v1/classify", json={"input": "hello", "preset": "triage.v1", "timeout_ms": 300}
        )
    assert result.status_code == 504
    assert len(base.calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("raised", "expected_status"),
    [
        ("provider error: Blocked content keyword: secret", 422),
        ("provider error: connection reset by peer", 502),
    ],
)
async def test_classify_provider_failure_status_triage(
    monkeypatch, tmp_path, raised, expected_status
):
    """Permanent content-policy rejections return 422 (route to human
    review, never retry); transient provider failures keep 502. The
    response body stays sanitized in both cases."""

    class Provider(_FakeProvider):
        async def chat(self, **kwargs):
            self.calls.append(kwargs)
            raise RuntimeError(raised)

    provider = Provider([])
    server = _make_server(monkeypatch, tmp_path, provider)
    async with _client(server) as client:
        result = await client.post(
            "/v1/classify", json={"input": "synthetic", "preset": "triage.v1"}
        )
    assert result.status_code == expected_status
    assert result.json()["error"] == "provider call failed"
    assert raised not in result.json()["error"]


@pytest.mark.asyncio
async def test_classify_falls_back_to_managed_default_provider(monkeypatch, tmp_path):
    """A freshly started server has no bootstrapped current_provider — classify
    must build a managed provider from the configured defaults instead of 503."""
    import victor.config.settings as settings_mod
    import victor.providers.factory as factory_mod
    from victor.integrations.api import fastapi_server as fs_mod

    created: list = []

    class _Managed(_FakeProvider):
        async def chat(self, messages, *, model, temperature, max_tokens, **kwargs):
            return SimpleNamespace(
                content=json.dumps(_TRIAGE_RESULT),
                usage={"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
                model="managed-default",
            )

    async def fake_factory(provider_name, model, **kwargs):
        # REAL call shape: create(provider_name, model, **provider_settings).
        p = _Managed([])
        created.append((provider_name, model, p))
        return p

    class _NoCurrentManager:
        current_provider = None  # lazy bootstrap has not run

    class _Orch:
        provider_manager = _NoCurrentManager()
        model = None

    # Server bootstraps with REAL settings; the defaults stub is applied only
    # for the request, after bootstrap has completed.
    monkeypatch.setattr(fs_mod, "load_fastapi_router_registrations", lambda *, workspace_root: [])
    server = fs_mod.VictorFastAPIServer(workspace_root=str(tmp_path), enable_graphql=False)
    server._orchestrator = _Orch()

    class _ProviderCfg:
        default_provider = "zai"
        default_model = "glm-x"

    class _Settings:
        provider = _ProviderCfg()

        def get_provider_settings(self, name):
            return {"api_key": "sk-test"}

    def fake_load_settings(fresh=False):
        return _Settings()

    async with _client(server) as client:
        monkeypatch.setattr(settings_mod, "load_settings", fake_load_settings)
        monkeypatch.setattr(
            factory_mod.ManagedProviderFactory, "create", staticmethod(fake_factory)
        )
        response = await client.post(
            "/v1/classify",
            json={"input": "x", "preset": "triage.v1", "timeout_ms": 20000},
        )

    assert response.status_code == 200
    assert response.json()["result"] == _TRIAGE_RESULT
    provider_name, model, p = created[0]
    assert (provider_name, model) == ("zai", "glm-x")
    assert p.closed is True


@pytest.mark.asyncio
async def test_classify_endpoint_override_reaches_factory_and_instructions_reach_prompt(
    monkeypatch, tmp_path
):
    """Additive /v1 fields: `endpoint` overrides the provider base_url;
    `instructions` fold into the system prompt."""
    import victor.config.settings as settings_mod
    import victor.providers.factory as factory_mod

    server = _make_server(monkeypatch, tmp_path, _FakeProvider([]))
    captured: dict = {}

    class _RecordingManaged(_FakeProvider):
        async def chat(self, messages, *, model, temperature, max_tokens, **kwargs):
            captured["system"] = messages[0]["content"]
            return SimpleNamespace(
                content=json.dumps(_TRIAGE_RESULT),
                usage={"total_tokens": 7},
                model="managed",
            )

    async def spy_factory(cls, provider_name, model, **kwargs):
        captured["factory"] = (provider_name, model, kwargs.get("base_url"))
        return _RecordingManaged([])

    monkeypatch.setattr(factory_mod.ManagedProviderFactory, "create", classmethod(spy_factory))

    class _ProviderCfg:
        default_provider = "inferflux"
        default_model = "qwen3-coder-30b"

    class _Settings:
        provider = _ProviderCfg()

        def get_provider_settings(self, name):
            return {"api_key": "sk-test"}

    monkeypatch.setattr(settings_mod, "load_settings", lambda fresh=False: _Settings())

    instruction = "Treat the private category as confidential matters."
    async with _client(server) as client:
        response = await client.post(
            "/v1/classify",
            json={
                "input": "x",
                "preset": "triage.v1",
                "endpoint": "http://127.0.0.1:18081/v1",
                "instructions": instruction,
            },
        )

    assert response.status_code == 200
    assert captured["factory"] == (
        "inferflux",
        "qwen3-coder-30b",
        "http://127.0.0.1:18081/v1",
    )
    assert instruction in captured["system"]
