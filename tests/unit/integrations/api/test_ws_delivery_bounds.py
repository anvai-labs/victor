# Copyright 2026 Vijaykumar Singh <vijaykumar@anvaiops.com>
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

"""Bounded per-client WebSocket broadcast delivery (VAS-06/10, TD-03).

``_broadcast_ws`` awaited each recipient's ``send_json`` sequentially: a
socket whose send buffer never drained (a wedged TCP peer) blocked the
broadcast loop for EVERY recipient, and there was no bound at all on how
much a slow client could stall producers. Per-client bounded queues plus a
per-client sender task with a send deadline fix the hazard:

- a slow client never delays a healthy client's delivery;
- overflow sheds the oldest NONESSENTIAL progress events (coalescing);
- terminal/approval events are never dropped — if one cannot enqueue
  behind queued essential state, the slow client is disconnected and the
  gap is recorded instead of silently losing final state (TD-03);
- a wedged socket is torn down by the send deadline, not left blocking.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional

import pytest

from victor.integrations.api import fastapi_server


def _progress(step: int) -> dict:
    return {"type": "agent_event", "event": "task_update", "data": {"step": step}}


def _terminal(run_id: str = "wf-1") -> dict:
    return {"type": "agent_event", "event": "workflow_completed", "data": {"id": run_id}}


def _approval(approval_id: str = "apr-1") -> dict:
    return {"type": "tool_approval_resolved", "approval_id": approval_id, "approved": True}


class _FakeSocket:
    """Stand-in for a Starlette WebSocket.

    ``block_forever`` simulates the hazard: a client that never reads so
    the send never completes (a full TCP send buffer).
    """

    def __init__(
        self,
        *,
        block_forever: bool = False,
        send_delay: float = 0.0,
        authenticated: bool = True,
    ) -> None:
        self.state = SimpleNamespace(authenticated=authenticated, client_id="alice")
        self.sent: List[dict] = []
        self.closed: List[tuple] = []
        self.block_forever = block_forever
        self.send_delay = send_delay
        self._release: Optional[asyncio.Event] = None

    async def send_json(self, message: dict) -> None:
        if self.block_forever:
            self._release = asyncio.Event()
            await self._release.wait()
        if self.send_delay:
            await asyncio.sleep(self.send_delay)
        self.sent.append(message)

    async def close(self, code: int = 1000, reason: str = "") -> None:
        self.closed.append((code, reason))


def _create_server(monkeypatch, tmp_path: Path, **server_kwargs):
    monkeypatch.setattr(
        fastapi_server,
        "load_fastapi_router_registrations",
        lambda *, workspace_root: [],
    )
    server = fastapi_server.VictorFastAPIServer(
        workspace_root=str(tmp_path),
        enable_graphql=False,
        allow_unauthenticated=True,
        **server_kwargs,
    )
    return server


def _register(server, ws: _FakeSocket):
    """Register a fake socket exactly as the /ws route does."""
    server._ws_clients.append(ws)
    return server._register_ws_delivery(ws)


async def _wait_for(predicate, timeout: float = 2.0) -> bool:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.01)
    return predicate()


# ---------------------------------------------------------------------------
# The core hazard: one slow socket must not block other recipients
# ---------------------------------------------------------------------------


class TestSlowClientIsolation:
    async def test_wedged_socket_does_not_delay_healthy_client(self, monkeypatch, tmp_path):
        """RED at baseline: _broadcast_ws awaited each socket sequentially, so
        a wedged peer stalled the whole loop. The broadcast call itself must
        return bounded-time and the healthy client must receive the event."""
        server = _create_server(monkeypatch, tmp_path)
        slow = _FakeSocket(block_forever=True)
        healthy = _FakeSocket()
        _register(server, slow)
        _register(server, healthy)

        await asyncio.wait_for(
            server._broadcast_ws(_progress(1)),
            timeout=2.0,
        )

        delivered = await _wait_for(lambda: bool(healthy.sent))
        assert delivered, "healthy client starved by wedged peer socket"
        assert healthy.sent[0] == _progress(1)

    async def test_slow_client_teardown_leaves_healthy_flow_untouched(self, monkeypatch, tmp_path):
        """After the deadline tears the wedged client down, later broadcasts
        still reach the healthy recipient."""
        server = _create_server(monkeypatch, tmp_path)
        server._ws_send_deadline = 0.05
        slow = _FakeSocket(block_forever=True)
        healthy = _FakeSocket()
        _register(server, slow)
        _register(server, healthy)

        await server._broadcast_ws(_progress(1))

        torn_down = await _wait_for(lambda: slow not in server._ws_deliveries, timeout=5.0)
        assert torn_down, "wedged client not disconnected by send deadline"

        healthy.sent.clear()
        await asyncio.wait_for(server._broadcast_ws(_progress(2)), timeout=2.0)
        assert await _wait_for(lambda: bool(healthy.sent))
        assert healthy.sent[-1] == _progress(2)
        assert healthy in server._ws_deliveries, "healthy client must stay connected"


# ---------------------------------------------------------------------------
# Send deadline tears down a wedged socket
# ---------------------------------------------------------------------------


class TestSendDeadline:
    async def test_deadline_disconnects_wedged_client(self, monkeypatch, tmp_path):
        """A socket whose send never completes is disconnected by the send
        deadline instead of holding its sender task (and the broadcast path)
        hostage."""
        server = _create_server(monkeypatch, tmp_path)
        server._ws_send_deadline = 0.05
        slow = _FakeSocket(block_forever=True)
        _register(server, slow)

        await server._broadcast_ws(_progress(1))

        torn_down = await _wait_for(lambda: slow not in server._ws_deliveries, timeout=5.0)
        assert torn_down
        assert slow not in server._ws_clients

    async def test_send_failure_disconnects_only_the_failing_client(self, monkeypatch, tmp_path):
        """A socket that errors on send is removed; other recipients keep
        receiving."""
        server = _create_server(monkeypatch, tmp_path)

        class _ExplodingSocket(_FakeSocket):
            async def send_json(self, message: dict) -> None:
                raise RuntimeError("socket gone")

        bad = _ExplodingSocket()
        good = _FakeSocket()
        _register(server, bad)
        _register(server, good)

        await asyncio.wait_for(server._broadcast_ws(_progress(1)), timeout=2.0)

        gone = await _wait_for(lambda: bad not in server._ws_deliveries)
        assert gone
        assert await _wait_for(lambda: bool(good.sent))


# ---------------------------------------------------------------------------
# Overflow policy: shed oldest progress, never essential state
# ---------------------------------------------------------------------------


class TestOverflowPolicy:
    async def test_progress_overflow_sheds_oldest_progress_keeps_newest(
        self, monkeypatch, tmp_path
    ):
        server = _create_server(monkeypatch, tmp_path)
        server._ws_delivery_queue_size = 3
        slow = _FakeSocket(block_forever=True)
        delivery = _register(server, slow)
        # Sender consumed the first item and is wedged inside send_json; the
        # queue is effectively frozen from here on.
        await server._broadcast_ws(_progress(0))
        await _wait_for(lambda: slow._release is not None)

        for step in range(1, 5):
            await server._broadcast_ws(_progress(step))

        assert delivery.dropped_nonessential >= 1
        queued = delivery.queue.snapshot()
        assert all(item["event"] == "task_update" for item in queued)
        assert queued[-1] == _progress(4), "newest progress must be enqueued (drop-oldest)"
        assert delivery.queue.qsize() <= server._ws_delivery_queue_size
        assert slow in server._ws_deliveries, "progress overflow must NOT disconnect"

    async def test_terminal_event_survives_progress_overflow_in_order(self, monkeypatch, tmp_path):
        """A full queue of progress events must never cost the final terminal
        event: shedding progress makes room for it, in order."""
        server = _create_server(monkeypatch, tmp_path)
        server._ws_delivery_queue_size = 3
        slow = _FakeSocket(block_forever=True)
        delivery = _register(server, slow)
        await server._broadcast_ws(_progress(0))
        await _wait_for(lambda: slow._release is not None)
        for step in range(1, 4):
            await server._broadcast_ws(_progress(step))

        await server._broadcast_ws(_terminal())

        queued = delivery.queue.snapshot()
        assert queued[-1] == _terminal(), "terminal event must be the last enqueued"
        assert queued.count(_terminal()) == 1
        assert slow in server._ws_deliveries

    async def test_approval_event_is_essential_not_sheddable(self, monkeypatch, tmp_path):
        """Approval state changes ride the essential path: a full queue sheds
        progress before them, and they are never the dropped side of an
        overflow."""
        server = _create_server(monkeypatch, tmp_path)
        server._ws_delivery_queue_size = 2
        slow = _FakeSocket(block_forever=True)
        delivery = _register(server, slow)
        await server._broadcast_ws(_progress(0))
        await _wait_for(lambda: slow._release is not None)
        await server._broadcast_ws(_progress(1))

        await server._broadcast_ws(_approval())

        queued = delivery.queue.snapshot()
        assert queued[-1] == _approval()
        assert _approval() not in queued[:-1], "approval event duplicated/dropped"

    async def test_nonessential_newcomer_dropped_when_queue_all_essential(
        self, monkeypatch, tmp_path
    ):
        """When nothing in the queue is sheddable, a nonessential newcomer is
        the one dropped — never a queued essential event, never a disconnect."""
        server = _create_server(monkeypatch, tmp_path)
        server._ws_delivery_queue_size = 2
        slow = _FakeSocket(block_forever=True)
        delivery = _register(server, slow)
        await server._broadcast_ws(_approval("apr-a"))
        # First item consumed by the wedged sender; queue holds nothing, fill
        # with essential events directly.
        delivery.queue.put(_approval("apr-b"))
        await _wait_for(lambda: slow._release is not None)
        delivery.queue.put(_approval("apr-c"))
        assert delivery.queue.full()

        await server._broadcast_ws(_progress(9))

        queued = delivery.queue.snapshot()
        assert queued == [_approval("apr-b"), _approval("apr-c")]
        assert delivery.dropped_nonessential >= 1
        assert slow in server._ws_deliveries, "nonessential overflow must not disconnect"

    async def test_terminal_overflow_disconnects_and_records_gap(self, monkeypatch, tmp_path):
        """A terminal event that cannot fit behind an all-essential queue
        means final state would be lost: disconnect the slow client and
        record the gap instead of silently dropping (TD-03)."""
        server = _create_server(monkeypatch, tmp_path)
        server._ws_delivery_queue_size = 2
        slow = _FakeSocket(block_forever=True)
        delivery = _register(server, slow)
        # The sender consumes this item and wedges inside send_json.
        delivery.queue.put(_approval("apr-b"))
        await _wait_for(lambda: slow._release is not None)
        # Now fill the frozen queue to capacity with essential events only.
        delivery.queue.put(_approval("apr-c"))
        delivery.queue.put(_approval("apr-d"))
        assert delivery.queue.full()

        assert server._ws_delivery_gaps == 0
        await server._broadcast_ws(_terminal())

        assert server._ws_delivery_gaps == 1, "essential overflow must record the gap"
        gone = await _wait_for(lambda: slow not in server._ws_deliveries)
        assert gone, "slow client must be disconnected on terminal overflow"
        assert slow not in server._ws_clients

    async def test_healthy_clients_unaffected_by_peer_terminal_overflow(
        self, monkeypatch, tmp_path
    ):
        server = _create_server(monkeypatch, tmp_path)
        server._ws_delivery_queue_size = 1
        slow = _FakeSocket(block_forever=True)
        healthy = _FakeSocket()
        delivery = _register(server, slow)
        good_delivery = _register(server, healthy)
        await server._broadcast_ws(_approval("apr-a"))
        await _wait_for(lambda: slow._release is not None)
        delivery.queue.put(_approval("apr-b"))

        await server._broadcast_ws(_terminal())

        assert server._ws_delivery_gaps == 1
        assert await _wait_for(lambda: len(healthy.sent) >= 2)
        assert healthy in server._ws_deliveries
        assert good_delivery.dropped_nonessential == 0


# ---------------------------------------------------------------------------
# Background-agent terminal events must survive overflow (TD-03, review fix)
# ---------------------------------------------------------------------------


class TestAgentTerminalEventsPreserved:
    async def test_agent_completed_sheds_progress_not_final_state(self, monkeypatch, tmp_path):
        """agent_completed (BackgroundAgentManager terminal variant) riding a
        full queue of progress events must evict progress — never be the
        dropped side (regression: the closed terminal whitelist shed it)."""
        server = _create_server(monkeypatch, tmp_path)
        server._ws_delivery_queue_size = 2
        slow = _FakeSocket(block_forever=True)
        delivery = _register(server, slow)
        await server._broadcast_ws({"type": "agent_event", "event": "agent_started"})
        await _wait_for(lambda: slow._release is not None)
        # Fill the frozen queue to capacity with sheddable progress.
        delivery.queue.put({"type": "agent_event", "event": "agent_progress"})
        delivery.queue.put({"type": "agent_event", "event": "agent_progress"})
        assert delivery.queue.full()

        await server._broadcast_ws({"type": "agent_event", "event": "agent_completed"})

        queued = delivery.queue.snapshot()
        assert queued[-1]["event"] == "agent_completed"
        assert delivery.dropped_nonessential >= 1
        assert slow in server._ws_deliveries

    async def test_agent_error_cannot_fit_all_essential_disconnects(self, monkeypatch, tmp_path):
        server = _create_server(monkeypatch, tmp_path)
        server._ws_delivery_queue_size = 1
        slow = _FakeSocket(block_forever=True)
        delivery = _register(server, slow)
        await server._broadcast_ws({"type": "agent_event", "event": "agent_completed"})
        await _wait_for(lambda: slow._release is not None)
        delivery.queue.put({"type": "agent_event", "event": "agent_cancelled"})
        assert delivery.queue.full()

        await server._broadcast_ws({"type": "agent_event", "event": "agent_error"})

        assert server._ws_delivery_gaps == 1
        assert await _wait_for(lambda: slow not in server._ws_deliveries)


# ---------------------------------------------------------------------------
# Single-writer invariant: ALL frames flow through the per-client queue
# ---------------------------------------------------------------------------


class TestSingleWriter:
    async def test_response_frames_enqueue_instead_of_direct_send(self, monkeypatch, tmp_path):
        """The socket's sender task must be its ONLY writer (uvicorn raises
        on overlapping sends; a direct pong colliding with a queued broadcast
        would tear down a healthy client mid-chat). Right after the handler
        returns, the frame is in the queue and nothing has touched the wire."""
        server = _create_server(monkeypatch, tmp_path)
        ws = _FakeSocket()
        delivery = _register(server, ws)

        await server._handle_ws_message(ws, {"type": "ping"})

        assert ws.sent == [], "pong must not bypass the per-client queue"
        assert delivery.queue.snapshot() == [{"type": "pong"}]
        # The sender task drains it: the client still receives the frame.
        assert await _wait_for(lambda: ws.sent == [{"type": "pong"}])

    async def test_broadcast_enqueued_before_pong_is_delivered_before_pong(
        self, monkeypatch, tmp_path
    ):
        """FIFO through one queue makes the ordering that the ws auth gate's
        positive control asserts (broadcast before pong) guaranteed rather
        than scheduler-dependent."""
        server = _create_server(monkeypatch, tmp_path)
        ws = _FakeSocket()
        delivery = _register(server, ws)

        await server._broadcast_ws(_progress(1))
        await server._handle_ws_message(ws, {"type": "ping"})

        assert delivery.queue.snapshot() == [_progress(1), {"type": "pong"}]
        assert await _wait_for(lambda: ws.sent == [_progress(1), {"type": "pong"}])


# ---------------------------------------------------------------------------
# Configuration knobs
# ---------------------------------------------------------------------------


class TestConfigKnobs:
    async def test_queue_size_knob_respected(self, monkeypatch, tmp_path):
        server = _create_server(monkeypatch, tmp_path)
        server._ws_delivery_queue_size = 7
        delivery = _register(server, _FakeSocket())
        assert delivery.queue.maxsize == 7

    async def test_deadline_knob_respected(self, monkeypatch, tmp_path):
        server = _create_server(monkeypatch, tmp_path)
        server._ws_send_deadline = 0.05
        slow = _FakeSocket(block_forever=True)
        _register(server, slow)
        await server._broadcast_ws(_progress(1))
        assert await _wait_for(lambda: slow not in server._ws_deliveries, timeout=3.0)

    async def test_default_queue_size_is_bounded(self, monkeypatch, tmp_path):
        server = _create_server(monkeypatch, tmp_path)
        assert 0 < server._ws_delivery_queue_size <= 10_000
        assert 0 < server._ws_send_deadline <= 60.0

    def test_flat_env_knobs_reach_the_server(self, monkeypatch, tmp_path):
        """VICTOR_SERVER_WS_DELIVERY_QUEUE_SIZE / _SEND_DEADLINE_SECONDS must
        map like every sibling server knob (LEGACY_ENV_MAPPINGS)."""
        from victor.config.settings import reset_settings_cache

        monkeypatch.setenv("VICTOR_SERVER_WS_DELIVERY_QUEUE_SIZE", "13")
        monkeypatch.setenv("VICTOR_SERVER_WS_SEND_DEADLINE_SECONDS", "2.5")
        reset_settings_cache()
        server = _create_server(monkeypatch, tmp_path)
        try:
            assert server._ws_delivery_queue_size == 13
            assert server._ws_send_deadline == 2.5
        finally:
            reset_settings_cache()

    def test_settings_reject_nonpositive_queue_size(self):
        from victor.config.groups.server_config import ServerSettings

        with pytest.raises(ValueError):
            ServerSettings(server_ws_delivery_queue_size=0)

    def test_settings_reject_nonpositive_deadline(self):
        from victor.config.groups.server_config import ServerSettings

        with pytest.raises(ValueError):
            ServerSettings(server_ws_send_deadline_seconds=0)


# ---------------------------------------------------------------------------
# Per-classification semantics (unit-level)
# ---------------------------------------------------------------------------


class TestEssentialClassification:
    def test_approval_and_terminal_types_are_essential(self):
        assert not fastapi_server._is_nonessential_progress(_approval())
        assert not fastapi_server._is_nonessential_progress(_terminal())
        for event in (
            "workflow_failed",
            "workflow_cancelled",
            "workflow_paused",
            "team_completed",
            "team_failed",
            "team_cancelled",
            # BackgroundAgentManager terminal variants (TD-03: final state
            # must never be shed).
            "agent_completed",
            "agent_error",
            "agent_cancelled",
        ):
            message = {"type": "agent_event", "event": event, "data": {}}
            assert not fastapi_server._is_nonessential_progress(message), event

    def test_declared_progress_is_sheddable(self):
        for message in (
            {"type": "agent_event", "event": "task_update", "data": {}},
            {"type": "agent_event", "event": "workflow_started", "data": {}},
            {"type": "agent_event", "event": "team_started", "data": {}},
            {"type": "agent_event", "event": "team_created", "data": {}},
            {"type": "agent_event", "event": "agent_started", "data": {}},
            {"type": "agent_event", "event": "agent_progress", "data": {}},
            {"type": "agent_event", "event": "agent_tool_call", "data": {}},
            {"type": "agent_event", "event": "agent_tool_result", "data": {}},
        ):
            assert fastapi_server._is_nonessential_progress(message), message

    def test_unknown_agent_events_default_to_essential(self):
        """Inverted default (TD-03): agent_event is essential unless its
        event name is in the declared progress allowlist — a closed progress
        whitelist cannot silently shed future terminal variants."""
        for message in (
            {"type": "agent_event", "event": "future_event", "data": {}},
            {"type": "agent_event"},
            {"type": "agent_event", "event": None},
        ):
            assert not fastapi_server._is_nonessential_progress(message), message

    def test_unknown_broadcast_types_default_to_essential(self):
        """Only declared agent_event progress is coalescible; other broadcast
        types are conservatively essential (never silently shed)."""
        for message in (
            {"type": "tool_cancelled", "approval_id": "x"},
            {"type": "something_new"},
            {},
        ):
            assert not fastapi_server._is_nonessential_progress(message), message


# ---------------------------------------------------------------------------
# Route-level wiring: delivery state follows connect/disconnect
# ---------------------------------------------------------------------------


class TestRouteLifecycle:
    def test_delivery_registered_on_connect_removed_on_disconnect(self, monkeypatch, tmp_path):
        from fastapi.testclient import TestClient

        class _FakeOrchestrator:
            async def chat(self, message: str):
                return SimpleNamespace(content=f"echo:{message}", tool_calls=[])

            async def stream_chat(self, message: str):
                yield SimpleNamespace(content=f"reply:{message}", metadata={})

            async def graceful_shutdown(self) -> None:
                return None

        server = _create_server(monkeypatch, tmp_path)
        server._orchestrator = _FakeOrchestrator()

        with TestClient(server.app) as client:
            with client.websocket_connect("/ws") as ws:
                assert len(server._ws_deliveries) == 1
                ws.send_json({"type": "ping"})
                assert ws.receive_json()["type"] == "pong"
            removed = client.portal.call(
                lambda: asyncio.wait_for(
                    _wait_for(lambda: len(server._ws_deliveries) == 0), timeout=5.0
                )
            )
        assert removed, "delivery state leaked after disconnect"
        assert server._ws_clients == []

    def test_keyed_server_defers_delivery_until_auth(self, monkeypatch, tmp_path):
        """Unauthenticated sockets on a keyed server are excluded from
        broadcast admission (VAS-05h), so they must not hold queue+sender
        resources either: registration happens at auth success."""
        from fastapi.testclient import TestClient

        server = _create_server(monkeypatch, tmp_path, api_keys={"sk-alice": "alice"})

        with TestClient(server.app) as client:
            with client.websocket_connect("/ws") as ws:
                assert len(server._ws_deliveries) == 0, "unauth socket got delivery state"

                ws.send_json({"type": "ping"})
                assert ws.receive_json()["type"] == "pong"
                assert len(server._ws_deliveries) == 0

                ws.send_json({"type": "auth", "api_key": "sk-alice"})
                assert ws.receive_json()["type"] == "auth_success"
                assert len(server._ws_deliveries) == 1, "auth did not register delivery"
