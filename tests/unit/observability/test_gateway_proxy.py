# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
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

"""FEP-0020 Phase 3 — reverse-proxy serve mode.

Requires ``victor[gateway]`` (sandhi-gateway + fastapi) and respx; skipped
otherwise.
"""

from __future__ import annotations

import pytest

pytest.importorskip("sandhi_gateway", reason="requires the victor[gateway] extra")
pytest.importorskip("fastapi", reason="requires the victor[gateway] extra")
respx = pytest.importorskip("respx", reason="requires respx (dev/test)")

import httpx  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from victor.observability.gateway_proxy import (  # noqa: E402
    GatewayConfig,
    GatewayVirtualKey,
    _usage_from_data_line,
    build_gateway_app,
)

UPSTREAM = "http://up.test/v1/chat/completions"

OPENAI_RESP = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "model": "gpt-4o",
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}
    ],
    "usage": {"prompt_tokens": 30, "completion_tokens": 10, "total_tokens": 40},
}


def _config(budget: int | None = None) -> GatewayConfig:
    return GatewayConfig(
        virtual_keys=[
            GatewayVirtualKey(
                key_id="alice-openai",
                token="vk-secret",
                subject_id="alice",
                group_id="team-a",
                provider="openai",
                upstream_base_url="http://up.test/v1",
                upstream_api_key="sk-real-shared",
                budget_tokens=budget,
            )
        ]
    )


def _client(budget: int | None = None) -> TestClient:
    return TestClient(build_gateway_app(_config(budget)))


def test_missing_or_bad_token_is_401() -> None:
    client = _client()
    assert client.post("/v1/chat/completions", json={}).status_code == 401
    assert (
        client.post(
            "/v1/chat/completions", headers={"Authorization": "Bearer wrong"}, json={}
        ).status_code
        == 401
    )


@respx.mock
def test_forwards_and_meters_per_subject() -> None:
    route = respx.post(UPSTREAM).mock(return_value=httpx.Response(200, json=OPENAI_RESP))
    client = _client()

    r = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer vk-secret"},
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200
    assert r.json()["usage"]["prompt_tokens"] == 30  # body passed through

    # The proxy forwarded with the *real* upstream key, not the client's virtual key.
    assert route.calls.last.request.headers["authorization"] == "Bearer sk-real-shared"

    usage = client.get("/gateway/usage").json()
    assert usage["event_count"] == 1
    assert usage["by_subject"]["alice"] == 40
    assert usage["by_group"]["team-a"] == 40
    ev = usage["events"][0]
    assert ev["subject_id"] == "alice"
    assert ev["group_id"] == "team-a"
    assert ev["tokens_in"] == 30 and ev["tokens_out"] == 10


@respx.mock
def test_budget_enforced_with_429() -> None:
    respx.post(UPSTREAM).mock(return_value=httpx.Response(200, json=OPENAI_RESP))
    client = _client(budget=40)  # exactly one 40-token call fits
    headers = {"Authorization": "Bearer vk-secret"}
    body = {"model": "gpt-4o", "messages": []}

    r1 = client.post("/v1/chat/completions", headers=headers, json=body)
    assert r1.status_code == 200  # records 40 → scope now at the cap

    r2 = client.post("/v1/chat/completions", headers=headers, json=body)
    assert r2.status_code == 429
    assert r2.json()["error"]["type"] == "budget_exceeded"


def test_keys_endpoint_never_leaks_secrets() -> None:
    client = _client(budget=100)
    resp = client.get("/gateway/keys")
    keys = resp.json()
    assert keys[0]["key_id"] == "alice-openai"
    assert keys[0]["subject_id"] == "alice"
    assert keys[0]["budget_tokens"] == 100
    assert keys[0]["spent_tokens"] == 0
    assert "vk-secret" not in resp.text and "sk-real-shared" not in resp.text


def test_healthz() -> None:
    client = _client()
    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert body["virtual_keys"] == 1


@respx.mock
def test_streaming_passes_through_and_meters_final_usage() -> None:
    sse = (
        b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
        b'data: {"choices":[],"usage":{"prompt_tokens":12,"completion_tokens":4}}\n\n'
        b"data: [DONE]\n\n"
    )
    respx.post(UPSTREAM).mock(return_value=httpx.Response(200, content=sse))
    client = _client()

    r = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer vk-secret"},
        json={"model": "gpt-4o", "stream": True, "messages": []},
    )
    assert r.status_code == 200
    assert b"[DONE]" in r.content  # byte-exact pass-through

    usage = client.get("/gateway/usage").json()
    assert usage["event_count"] == 1
    ev = usage["events"][0]
    assert ev["tokens_in"] == 12 and ev["tokens_out"] == 4


@respx.mock
def test_streaming_meters_usage_when_upstream_drops_mid_stream() -> None:
    """A torn upstream stream must still meter the usage it already parsed.

    Real usage arrives in an early chunk; the upstream then drops before ``[DONE]``
    (a client disconnect raises the same way at ``yield chunk``). Pre-fix,
    ``meter_tokens`` sat after the ``async with`` with no try/finally, so the
    exception skipped it and the call settled zero — the same class of metering
    hole ADR-0005 D1 closed in sandhi.
    """

    async def dropping_upstream():
        yield b'data: {"choices":[],"usage":{"prompt_tokens":12,"completion_tokens":4}}\n\n'
        raise RuntimeError("upstream disconnected mid-stream")

    respx.post(UPSTREAM).mock(return_value=httpx.Response(200, content=dropping_upstream()))
    client = _client()

    try:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            headers={"Authorization": "Bearer vk-secret"},
            json={"model": "gpt-4o", "stream": True, "messages": []},
        ) as resp:
            for _ in resp.iter_bytes():  # drain what arrives; the stream tears mid-way
                pass
    except Exception:
        # The torn upstream surfaces to the client as some transport error
        # (raw RuntimeError under TestClient; an httpx protocol error in prod).
        # We don't care which — the assertion below is the real gate: the usage
        # the tee already parsed must still have been metered.
        pass

    usage = client.get("/gateway/usage").json()
    assert usage["event_count"] == 1, usage
    ev = usage["events"][0]
    assert ev["tokens_in"] == 12 and ev["tokens_out"] == 4


def test_usage_line_parser() -> None:
    assert _usage_from_data_line("data: [DONE]") is None
    assert _usage_from_data_line(": keep-alive comment") is None
    assert _usage_from_data_line('data: {"choices":[]}') is None
    assert _usage_from_data_line("data: not-json") is None
    assert _usage_from_data_line('data: {"usage":{"prompt_tokens":7,"completion_tokens":3}}') == {
        "prompt_tokens": 7,
        "completion_tokens": 3,
    }
