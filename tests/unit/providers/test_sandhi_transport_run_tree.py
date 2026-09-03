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

The per-call headers also carry an ``Idempotency-Key`` — the LOGICAL-call
identity (sandhi TD-0021 P4 sender half): its preferred source is the ``call_id``
correlation var bound by the retry owner (ResilientProvider), so every
Python-level retry AND fallback of one logical call shares one key and the
gateway's meter counts the call once; with no binding, a fresh key per
invocation (each invocation its own logical call); direct mode -> never sent
(no gateway ledger to dedup against). The Rust-internal retry reuse is pinned
sandhi-side (resilience.rs re-sends the same HeaderMap), not here.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Optional

import pytest

import victor.providers.sandhi_transport as st
from victor.core import context as ctx
from victor.providers.base import Message
from victor.providers.deepseek_provider import DeepSeekProvider


class FakeTypedProvider:
    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.wire_headers: list[Optional[str]] = []

    async def complete_json(self, request_json: str, wire_headers_json=None) -> str:
        self.requests.append(json.loads(request_json))
        self.wire_headers.append(wire_headers_json)
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

    def stream_json(self, request_json: str, wire_headers_json=None):
        self.requests.append(json.loads(request_json))
        self.wire_headers.append(wire_headers_json)

        async def events():
            for event in (
                {"event": "response_start", "id": "r2", "model": "deepseek-chat"},
                {"event": "text_delta", "delta": "he"},
                {
                    "event": "usage",
                    "usage": {
                        "tokens_in": 6,
                        "tokens_out": 5,
                        "cache_creation_tokens": 0,
                        "cache_read_tokens": 0,
                    },
                },
                {"event": "finish", "finish_reason": "stop"},
            ):
                yield json.dumps(event)

        return events()


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
async def test_session_affinity_rides_neutral_metadata_in_both_modes(monkeypatch, bound_session):
    """Conversation affinity (sandhi ADR-0008 D3) is request metadata, never a body field.

    In direct mode there is no gateway header to carry it, so the session id rides
    the typed request's neutral ``metadata`` block — the typed runtime maps it onto
    catalog-declared vendor affinity headers (e.g. InferFlux's KV/prefix-cache key).
    """
    runtime = install_runtime(monkeypatch)

    provider = DeepSeekProvider(api_key="k", base_url="https://api.deepseek.com/v1")
    await provider.chat([Message(role="user", content="hi")], model="deepseek-chat")
    request = runtime.handle.requests[0]
    assert request["metadata"]["session_id"] == bound_session
    assert "session_id" not in request  # never a top-level wire-body field

    runtime2 = install_runtime(monkeypatch)
    provider2 = make_gateway_provider()
    await provider2.chat([Message(role="user", content="hi")], model="deepseek-chat")
    request2 = runtime2.handle.requests[0]
    assert request2["metadata"]["session_id"] == bound_session
    _, kwargs = runtime2.calls[0]
    headers = json.loads(kwargs["headers_json"])
    # On the proxy path the ingress header is what the proxy reads for affinity.
    assert headers["x-sandhi-session"] == bound_session


def _is_idempotency_key(value: object) -> bool:
    """A per-logical-call key as minted by ``_wire_call_headers``: uuid4().hex."""
    return (
        isinstance(value, str) and len(value) == 32 and all(c in "0123456789abcdef" for c in value)
    )


@pytest.fixture
def bound_turn():
    import victor.core.context as ctx

    token = ctx.turn_id.set("turn-aaa-111")
    try:
        yield "turn-aaa-111"
    finally:
        ctx.turn_id.reset(token)


@pytest.mark.asyncio
async def test_gateway_mode_stamps_step_id_per_call(monkeypatch, bound_turn):
    """TD-0022 D1 consumer side: the turn id rides the PER-CALL wire_headers_json, not
    handle-static headers — so the transport handle is reused across turns while the
    step id changes every call. This is what makes sandhi's run cost tree step-aware."""
    runtime = install_runtime(monkeypatch)
    provider = make_gateway_provider()

    await provider.chat([Message(role="user", content="hi")], model="deepseek-chat")

    (headers,) = (json.loads(h) for h in runtime.handle.wire_headers)
    assert headers["x-sandhi-step-id"] == bound_turn
    assert _is_idempotency_key(headers["Idempotency-Key"])


@pytest.mark.asyncio
async def test_step_id_changes_per_turn_without_handle_rebuild(monkeypatch):
    """Two turns, two distinct step ids — and ONE runtime.provider() invocation: the
    handle cache is untouched by per-call headers (pool + circuit state survive)."""
    import victor.core.context as ctx

    runtime = install_runtime(monkeypatch)
    provider = make_gateway_provider()

    token = ctx.turn_id.set("turn-1")
    try:
        await provider.chat([Message(role="user", content="a")], model="deepseek-chat")
        ctx.turn_id.set("turn-2")
        await provider.chat([Message(role="user", content="b")], model="deepseek-chat")
    finally:
        ctx.turn_id.reset(token)

    assert len(runtime.calls) == 1, "per-call headers must NOT rebuild the handle"
    first, second = (json.loads(h) for h in runtime.handle.wire_headers)
    assert first["x-sandhi-step-id"] == "turn-1"
    assert second["x-sandhi-step-id"] == "turn-2"
    # The idempotency key is minted per LOGICAL call — the two turns are two calls.
    assert first["Idempotency-Key"] != second["Idempotency-Key"]


@pytest.mark.asyncio
async def test_stream_path_stamps_step_id_per_call(monkeypatch, bound_turn):
    """The stream seam carries the per-call step id exactly like complete — a future
    edit to one branch only must not silently drop the other (review finding)."""
    import victor.core.context as ctx

    runtime = install_runtime(monkeypatch)
    provider = make_gateway_provider()

    async def collect():
        async for _ in provider.stream([Message(role="user", content="hi")], model="deepseek-chat"):
            pass

    await collect()
    headers = json.loads(runtime.handle.wire_headers[0])
    assert headers["x-sandhi-step-id"] == bound_turn
    assert _is_idempotency_key(headers["Idempotency-Key"])

    # Second turn: header changes, handle still reused.
    token = ctx.turn_id.set("turn-bbb-222")
    try:
        await collect()
    finally:
        ctx.turn_id.reset(token)
    headers_b = json.loads(runtime.handle.wire_headers[-1])
    assert headers_b["x-sandhi-step-id"] == "turn-bbb-222"
    assert headers_b["Idempotency-Key"] != headers["Idempotency-Key"]


@pytest.mark.asyncio
async def test_direct_mode_sends_no_step_header(monkeypatch, bound_turn):
    """Direct (non-gateway) mode stays byte-identical: no per-call headers at all —
    neither the step id nor the idempotency key (no gateway ledger to dedup against)."""
    runtime = install_runtime(monkeypatch)
    provider = DeepSeekProvider(api_key="k", base_url="https://api.deepseek.com/v1")

    await provider.chat([Message(role="user", content="hi")], model="deepseek-chat")

    assert runtime.handle.wire_headers[0] is None


@pytest.mark.asyncio
async def test_gateway_mode_mints_fresh_idempotency_key_per_logical_call(monkeypatch, bound_turn):
    """With no retry owner binding a call id (a bare provider used directly), each
    invocation is its own logical call and mints a fresh key — two invocations
    under the SAME turn must not share one."""
    runtime = install_runtime(monkeypatch)
    provider = make_gateway_provider()

    await provider.chat([Message(role="user", content="a")], model="deepseek-chat")
    await provider.chat([Message(role="user", content="b")], model="deepseek-chat")

    first, second = (json.loads(h) for h in runtime.handle.wire_headers)
    assert first["x-sandhi-step-id"] == second["x-sandhi-step-id"] == bound_turn
    assert _is_idempotency_key(first["Idempotency-Key"])
    assert _is_idempotency_key(second["Idempotency-Key"])
    assert first["Idempotency-Key"] != second["Idempotency-Key"]


@pytest.mark.asyncio
async def test_bound_call_id_pins_the_idempotency_key_across_retries(monkeypatch, bound_turn):
    """TD-0021 P4 sender half: when the retry owner (ResilientProvider) has bound a
    call id, EVERY re-entry into the transport under that binding — a Python-level
    retry or fallback of one logical call — carries the SAME Idempotency-Key, so
    the gateway's meter counts the logical call once (meter counts logical;
    enforcement counts physical). This is exactly what a resilience retry does:
    re-invoke provider.chat inside one chat() boundary."""
    import victor.core.context as ctx

    runtime = install_runtime(monkeypatch)
    provider = make_gateway_provider()

    token = ctx.set_call_id("call-xyz-1")
    try:
        # Two transport invocations under one logical call (attempt + retry).
        await provider.chat([Message(role="user", content="a")], model="deepseek-chat")
        await provider.chat([Message(role="user", content="a")], model="deepseek-chat")
    finally:
        ctx.call_id.reset(token)

    first, second = (json.loads(h) for h in runtime.handle.wire_headers)
    assert first["Idempotency-Key"] == second["Idempotency-Key"] == "call-xyz-1"


@pytest.mark.asyncio
async def test_gateway_without_bound_turn_sends_no_step_header(monkeypatch):
    """No turn -> no step header, but the idempotency key still rides: dedup is
    transport-retry protection, orthogonal to turn attribution."""
    runtime = install_runtime(monkeypatch)
    provider = make_gateway_provider()

    await provider.chat([Message(role="user", content="hi")], model="deepseek-chat")

    headers = json.loads(runtime.handle.wire_headers[0])
    assert "x-sandhi-step-id" not in headers
    assert _is_idempotency_key(headers["Idempotency-Key"])


@pytest.mark.asyncio
async def test_step_id_never_rides_neutral_metadata(monkeypatch, bound_turn, bound_session):
    """The KV-cache invariant: metadata maps onto vendor affinity headers, so a per-turn
    value there would change the cache key every turn. metadata carries session_id ONLY —
    neither the step id nor the idempotency key ever enters the request body."""
    runtime = install_runtime(monkeypatch)
    provider = make_gateway_provider()

    await provider.chat([Message(role="user", content="hi")], model="deepseek-chat")

    request = runtime.handle.requests[0]
    assert set(request.get("metadata", {}).keys()) == {"session_id"}


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
async def test_per_call_idempotency_key_overrides_a_static_one(monkeypatch, bound_turn):
    """The deliberate precedence asymmetry with x-sandhi-run-id: a STATIC
    caller-set Idempotency-Key is overridden by the per-call key — a static key
    would dedup every call of the dedup window into one metered event. Pinned so
    the asymmetry is a decision, not an accident."""
    runtime = install_runtime(monkeypatch)
    provider = make_gateway_provider()
    provider._wire_headers = {"Idempotency-Key": "caller-static-key"}

    await provider.chat([Message(role="user", content="hi")], model="deepseek-chat")

    headers = json.loads(runtime.handle.wire_headers[0])
    assert headers["Idempotency-Key"] != "caller-static-key"
    assert _is_idempotency_key(headers["Idempotency-Key"])


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
