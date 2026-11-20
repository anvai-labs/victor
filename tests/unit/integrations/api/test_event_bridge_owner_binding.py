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

"""EventBridge owner binding (Codex finding 9, HIGH).

The /ws/events EventBridge socket authenticated a key but registered a
RANDOM connection id with no principal; the default subscription was
broad ("*"); delivery checked only event type + an optional caller-selected
correlation id. Owner filtering for chat broadcasts therefore did not
extend to this transport: one authenticated user received another's
protected events.

Fix under test: the transport-verified principal is BOUND to each
subscription record at registration (immutable afterwards), and every
protected event (a stored resource owner) is rechecked at DELIVERY time
against the live record's bound owner and its still-active registration.
Two principals receiving simultaneous events are exercised on BOTH WS
transports (/ws chat broadcast and /ws/events EventBridge).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional

import pytest
from fastapi.testclient import TestClient

from victor.integrations.api import fastapi_server
from victor.integrations.api.event_bridge import (
    BridgeEvent,
    BridgeEventType,
    ClientRegistry,
    EventBroadcaster,
)

# ---------------------------------------------------------------------------
# Unit level: subscription records bind owners, delivery rechecks them
# ---------------------------------------------------------------------------


class _CapturingClient:
    """A registered broadcaster client that records delivered payloads."""

    def __init__(self, broadcaster: EventBroadcaster, client_id: str, owner: Optional[str]):
        self.received: List[dict] = []
        broadcaster.add_client(client_id, self._send, owner_client_id=owner)

    def _send(self, payload: str) -> None:
        self.received.append(json.loads(payload))


def _protected(event_type: BridgeEventType, owner: Optional[str], marker: str) -> BridgeEvent:
    return BridgeEvent(
        type=event_type,
        data={"marker": marker, "owner_client_id": owner} if owner else {"marker": marker},
        owner_client_id=owner,
    )


async def _deliver_all(broadcaster: EventBroadcaster) -> None:
    """Drive the engine's delivery for every event buffered so far."""
    for event in list(broadcaster._recent_events):
        await broadcaster._send_to_clients(event)


@pytest.fixture
def broadcaster_solo():
    """A fresh broadcaster singleton for registry-level tests."""
    EventBroadcaster.reset_instance()
    broadcaster = EventBroadcaster()
    yield broadcaster
    EventBroadcaster.reset_instance()


async def test_protected_event_reaches_only_the_owning_principal(broadcaster_solo):
    """Positive: owner-scoped delivery on the EventBridge registry."""
    broadcaster = broadcaster_solo
    alice = _CapturingClient(broadcaster, "conn-alice", "alice")
    bob = _CapturingClient(broadcaster, "conn-bob", "bob")

    broadcaster.broadcast_sync(_protected(BridgeEventType.TOOL_START, "alice", "a1"))
    broadcaster.broadcast_sync(_protected(BridgeEventType.TOOL_START, "bob", "b1"))
    await _deliver_all(broadcaster)

    assert [e["data"]["marker"] for e in alice.received] == ["a1"]
    assert [e["data"]["marker"] for e in bob.received] == ["b1"]


async def test_unbound_client_never_receives_protected_events(broadcaster_solo):
    """Negative (fail closed): an ownerless subscription record — keyless or
    not yet authenticated — is never admitted to a protected event."""
    broadcaster = broadcaster_solo
    anonymous = _CapturingClient(broadcaster, "conn-anon", None)

    broadcaster.broadcast_sync(_protected(BridgeEventType.TOOL_START, "alice", "secret"))
    await _deliver_all(broadcaster)

    assert anonymous.received == []


async def test_ownerless_internal_event_reaches_every_subscriber(broadcaster_solo):
    """Internal events with no bound resource owner keep the historical
    all-subscribers behavior."""
    broadcaster = broadcaster_solo
    alice = _CapturingClient(broadcaster, "conn-alice", "alice")
    bob = _CapturingClient(broadcaster, "conn-bob", "bob")

    broadcaster.broadcast_sync(_protected(BridgeEventType.METRICS_UPDATE, None, "internal"))
    await _deliver_all(broadcaster)

    assert [e["data"]["marker"] for e in alice.received] == ["internal"]
    assert [e["data"]["marker"] for e in bob.received] == ["internal"]


async def test_owner_in_data_only_is_honored(broadcaster_solo):
    """Producers may bind the resource owner purely via
    ``data["owner_client_id"]`` (the route call-site shape)."""
    broadcaster = broadcaster_solo
    alice = _CapturingClient(broadcaster, "conn-alice", "alice")
    bob = _CapturingClient(broadcaster, "conn-bob", "bob")

    broadcaster.broadcast_sync(
        BridgeEvent(
            type=BridgeEventType.TOOL_COMPLETE,
            data={"marker": "d1", "owner_client_id": "alice"},
        )
    )
    await _deliver_all(broadcaster)

    assert [e["data"]["marker"] for e in alice.received] == ["d1"]
    assert bob.received == []


async def test_ended_lease_receives_nothing_after_removal(broadcaster_solo):
    """Delivery-time recheck against the LIVE registry: a disconnected
    client's record is gone, so nothing is admitted to it."""
    broadcaster = broadcaster_solo
    alice = _CapturingClient(broadcaster, "conn-alice", "alice")

    broadcaster.broadcast_sync(_protected(BridgeEventType.TOOL_START, "alice", "live"))
    await _deliver_all(broadcaster)
    assert len(alice.received) == 1

    broadcaster.remove_client("conn-alice")  # lease ended
    broadcaster.broadcast_sync(_protected(BridgeEventType.TOOL_START, "alice", "after"))
    await _deliver_all(broadcaster)

    assert [e["data"]["marker"] for e in alice.received] == ["live"]


def test_subscription_update_cannot_rebind_the_owner():
    """Negative: a client message (subscribe frame) must not be able to
    widen admission to another principal's protected events."""
    registry = ClientRegistry()
    registry.add("conn-1", lambda _p: None, owner_client_id="alice")
    registry.update_subscriptions("conn-1", {"*"}, correlation_id=None)
    assert registry.get("conn-1").owner_client_id == "alice"


async def test_recent_events_replay_is_owner_scoped(broadcaster_solo):
    """The replay buffer applies the same admission as live delivery."""
    broadcaster = broadcaster_solo
    broadcaster.broadcast_sync(_protected(BridgeEventType.TOOL_START, "alice", "a-only"))
    broadcaster.broadcast_sync(_protected(BridgeEventType.TOOL_START, "bob", "b-only"))
    broadcaster.broadcast_sync(_protected(BridgeEventType.TOOL_START, None, "internal"))

    alice_view = broadcaster.get_recent_events(owner_client_id="alice")
    bob_view = broadcaster.get_recent_events(owner_client_id="bob")

    # Newest first.
    assert [e.data["marker"] for e in alice_view] == ["internal", "a-only"]
    assert [e.data["marker"] for e in bob_view] == ["internal", "b-only"]


# ---------------------------------------------------------------------------
# Transport 1: /ws/events (EventBridge) with two simultaneous principals
# ---------------------------------------------------------------------------


def _create_server(monkeypatch, tmp_path: Path, api_keys: Dict[str, str]):
    monkeypatch.setattr(
        fastapi_server,
        "load_fastapi_router_registrations",
        lambda *, workspace_root: [],
    )
    return fastapi_server.VictorFastAPIServer(
        workspace_root=str(tmp_path),
        enable_graphql=False,
        api_keys=api_keys,
    )


def _recv_bounded(client: TestClient, ws, timeout: float = 5.0) -> dict:
    """Receive one WebSocket frame with a hard time bound (fail, not hang)."""
    import anyio

    async def _recv():
        with anyio.fail_after(timeout):
            return await ws._send_rx.receive()

    message = client.portal.call(_recv)
    ws._raise_on_close(message)
    return json.loads(message["text"])


def test_ws_events_two_principals_receive_only_their_own_events(monkeypatch, tmp_path):
    """Two authenticated principals connected SIMULTANEOUSLY on /ws/events:
    owner-bound subscription records admit each principal only to its own
    protected events."""
    server = _create_server(monkeypatch, tmp_path, {"sk-alice": "alice", "sk-bob": "bob"})

    with TestClient(server.app) as client:
        # The lifespan builds the EventBridge; grab the live broadcaster.
        broadcaster = server._event_bridge._broadcaster
        with client.websocket_connect("/ws/events?api_key=sk-alice") as ws_alice:
            with client.websocket_connect("/ws/events?api_key=sk-bob") as ws_bob:
                # Both subscribe (broad default) — the owner binding holds.
                for ws in (ws_alice, ws_bob):
                    ws.send_json({"type": "subscribe", "categories": ["all"]})
                    assert ws.receive_json()["type"] == "subscribed"

                owners = {conn.owner_client_id for _, conn in broadcaster._registry.items()}
                assert owners == {"alice", "bob"}, "verified owners must be bound"

                # Simultaneous protected events, one per principal.
                broadcaster.broadcast_sync(
                    BridgeEvent(
                        type=BridgeEventType.TOOL_START,
                        data={"marker": "alice-tool", "owner_client_id": "alice"},
                        owner_client_id="alice",
                    )
                )
                broadcaster.broadcast_sync(
                    BridgeEvent(
                        type=BridgeEventType.TOOL_START,
                        data={"marker": "bob-tool", "owner_client_id": "bob"},
                        owner_client_id="bob",
                    )
                )

                assert _recv_bounded(client, ws_alice)["data"]["marker"] == "alice-tool"
                assert _recv_bounded(client, ws_bob)["data"]["marker"] == "bob-tool"

                # A later protected event for bob reaches bob only; alice's
                # socket must NOT observe it (negative gate).
                broadcaster.broadcast_sync(
                    BridgeEvent(
                        type=BridgeEventType.TOOL_COMPLETE,
                        data={"marker": "bob-final", "owner_client_id": "bob"},
                        owner_client_id="bob",
                    )
                )
                assert _recv_bounded(client, ws_bob)["data"]["marker"] == "bob-final"
                with pytest.raises(Exception):
                    _recv_bounded(client, ws_alice, timeout=0.4)


# ---------------------------------------------------------------------------
# Transport 2: /ws chat broadcast with two simultaneous principals
# ---------------------------------------------------------------------------


class _FakeWS:
    """Hashable stand-in for a Starlette WebSocket (delivery maps key on it)."""

    def __init__(self, owner: Optional[str], authenticated: bool = True):
        self.state = SimpleNamespace(authenticated=authenticated, client_id=owner)
        self.sent: List[dict] = []

    async def send_json(self, message: dict) -> None:
        self.sent.append(message)


async def _register_ws(server, owner: str):
    ws = _FakeWS(owner)
    server._ws_clients.append(ws)
    server._register_ws_delivery(ws)
    return ws


async def test_ws_chat_broadcast_owner_scoped_for_two_principals(monkeypatch, tmp_path):
    """Simultaneous principals on /ws: the existing owner-scoped broadcast
    must keep admitting each principal only to its own events (guard the
    /ws transport against regression while /ws/events gains the same)."""
    server = _create_server(monkeypatch, tmp_path, {"sk-alice": "alice", "sk-bob": "bob"})
    ws_alice = await _register_ws(server, "alice")
    ws_bob = await _register_ws(server, "bob")
    try:
        await server._broadcast_ws(
            {"type": "agent_event", "event": "tool_started", "data": {"marker": "a1"}},
            owner_client_id="alice",
        )
        await server._broadcast_ws(
            {"type": "agent_event", "event": "tool_started", "data": {"marker": "b1"}},
            owner_client_id="bob",
        )
        # Let the per-client sender tasks drain.
        await asyncio.sleep(0.05)
    finally:
        for ws in (ws_alice, ws_bob):
            server._teardown_ws_delivery(ws, reason="test")

    assert [m["data"]["marker"] for m in ws_alice.sent] == ["a1"]
    assert [m["data"]["marker"] for m in ws_bob.sent] == ["b1"]


async def test_ws_chat_broadcast_denies_unauthenticated_socket(monkeypatch, tmp_path):
    """Negative: a socket that never completed the auth handshake receives
    no broadcast even for ownerless internal events."""
    server = _create_server(monkeypatch, tmp_path, {"sk-alice": "alice", "sk-bob": "bob"})
    ws_anon = _FakeWS(None, authenticated=False)
    server._ws_clients.append(ws_anon)
    # Keyed servers defer delivery registration until auth succeeds; an
    # unauthenticated socket has no delivery record by construction.

    await server._broadcast_ws({"type": "agent_event", "event": "tool_started", "data": {}})
    await asyncio.sleep(0.05)

    assert ws_anon.sent == []
