# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
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

"""Sandhi run cost tree (sandhi PR #149) — ``x-sandhi-run-id`` wire header.

In gateway mode (TD-0003 P3) every provider call is routed through the Sandhi
proxy; stamping the agent run's stable identifier (victor's session id from the
execution-context correlation spine) as ``x-sandhi-run-id`` lets the proxy
build the persisted per-run cost tree (``GET /admin/usage/run/{run_id}``).

Contract:
- gateway mode + bound session -> header present, value == session id;
- direct mode -> never sends the header (no gateway to consume it);
- gateway mode without a bound session -> no header (nothing to attribute);
- existing wire headers are preserved; an explicit caller-set run id wins.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import victor.providers.sandhi_transport as st
from victor.core import context as ctx
from victor.providers.base import Message
from victor.providers.deepseek_provider import DeepSeekProvider


class FakeTypedProvider:
    def __init__(self) -> None:
        self.requests: list[dict] = []

    async def complete_json(self, request_json: str) -> str:
        self.requests.append(json.loads(request_json))
        return json.dumps(
            {
                "schema_version": "1",
                "id": "r1",
                "model": "deepseek-chat",
                "output": {"content": "hello", "tool_calls": []},
                "finish_reason": "stop",
                "usage": {
                    "tokens_in": 6,
                    "tokens_out": 5,
                    "cache_creation_tokens": 0,
                    "cache_read_tokens": 0,
                },
            }
        )


class FakeRuntime:
    def __init__(self, handle: FakeTypedProvider) -> None:
        self.handle = handle
        self.calls: list[tuple] = []

    def provider(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.handle


def install_runtime(monkeypatch) -> FakeRuntime:
    runtime = FakeRuntime(FakeTypedProvider())
    monkeypatch.setattr(st, "_sg", SimpleNamespace(ProviderRuntime=lambda: runtime))
    return runtime


def make_gateway_provider(**extra) -> DeepSeekProvider:
    return DeepSeekProvider(
        api_key="real-upstream-key",
        base_url="https://api.deepseek.com/v1",
        gateway={"url": "http://localhost:8600", "virtual_key": "vk_test_123"},
        **extra,
    )


@pytest.fixture
def bound_session():
    token = ctx.set_session_id("run-abc-123")
    try:
        yield "run-abc-123"
    finally:
        ctx.session_id.reset(token)


@pytest.mark.asyncio
async def test_gateway_mode_stamps_run_id_header(monkeypatch, bound_session):
    runtime = install_runtime(monkeypatch)
    provider = make_gateway_provider()

    await provider.chat([Message(role="user", content="hi")], model="deepseek-chat")

    _, kwargs = runtime.calls[0]
    headers = json.loads(kwargs["headers_json"])
    assert headers["x-sandhi-run-id"] == bound_session


@pytest.mark.asyncio
async def test_gateway_mode_without_session_sends_no_run_header(monkeypatch):
    runtime = install_runtime(monkeypatch)
    token = ctx.set_session_id("")
    try:
        provider = make_gateway_provider()
        await provider.chat([Message(role="user", content="hi")], model="deepseek-chat")
    finally:
        ctx.session_id.reset(token)

    _, kwargs = runtime.calls[0]
    headers = json.loads(kwargs["headers_json"]) if "headers_json" in kwargs else {}
    assert "x-sandhi-run-id" not in headers


@pytest.mark.asyncio
async def test_direct_mode_never_sends_run_header(monkeypatch, bound_session):
    """Direct-provider mode is byte-identical: no sandhi tree headers on the wire."""
    runtime = install_runtime(monkeypatch)
    provider = DeepSeekProvider(api_key="k", base_url="https://api.deepseek.com/v1")

    await provider.chat([Message(role="user", content="hi")], model="deepseek-chat")

    _, kwargs = runtime.calls[0]
    headers = json.loads(kwargs["headers_json"]) if "headers_json" in kwargs else {}
    assert "x-sandhi-run-id" not in headers


@pytest.mark.asyncio
async def test_gateway_mode_preserves_existing_wire_headers(monkeypatch, bound_session):
    runtime = install_runtime(monkeypatch)
    provider = make_gateway_provider()
    provider._wire_headers = {"originator": "victor"}

    await provider.chat([Message(role="user", content="hi")], model="deepseek-chat")

    _, kwargs = runtime.calls[0]
    headers = json.loads(kwargs["headers_json"])
    assert headers["originator"] == "victor"
    assert headers["x-sandhi-run-id"] == bound_session


@pytest.mark.asyncio
async def test_gateway_mode_explicit_run_header_wins(monkeypatch, bound_session):
    """A caller-provided x-sandhi-run-id wire header is never clobbered."""
    runtime = install_runtime(monkeypatch)
    provider = make_gateway_provider()
    provider._wire_headers = {"x-sandhi-run-id": "explicit-run"}

    await provider.chat([Message(role="user", content="hi")], model="deepseek-chat")

    _, kwargs = runtime.calls[0]
    headers = json.loads(kwargs["headers_json"])
    assert headers["x-sandhi-run-id"] == "explicit-run"


@pytest.mark.asyncio
async def test_gateway_handle_is_rebuilt_per_run_and_reused_within_one(monkeypatch):
    """The FFI handle cache keys on the run id: reused within a run, fresh per run."""
    runtime = install_runtime(monkeypatch)
    provider = make_gateway_provider()

    token = ctx.set_session_id("run-1")
    try:
        await provider.chat([Message(role="user", content="a")], model="deepseek-chat")
        await provider.chat([Message(role="user", content="b")], model="deepseek-chat")
        assert len(runtime.calls) == 1  # cached within the run

        ctx.set_session_id("run-2")
        await provider.chat([Message(role="user", content="c")], model="deepseek-chat")
    finally:
        ctx.session_id.reset(token)

    assert len(runtime.calls) == 2
    headers_first = json.loads(runtime.calls[0][1]["headers_json"])
    headers_second = json.loads(runtime.calls[1][1]["headers_json"])
    assert headers_first["x-sandhi-run-id"] == "run-1"
    assert headers_second["x-sandhi-run-id"] == "run-2"
