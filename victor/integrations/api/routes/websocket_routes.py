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

"""WebSocket routes: /ws, /ws/events."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import TYPE_CHECKING

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

if TYPE_CHECKING:
    from victor.integrations.api.fastapi_server import VictorFastAPIServer

logger = logging.getLogger(__name__)

# Fallback for the bounded /ws auth handshake when the server does not
# define ``_ws_auth_deadline`` (VAS-05h).
DEFAULT_WS_AUTH_DEADLINE_SECONDS = 30.0


def create_router(server: "VictorFastAPIServer") -> APIRouter:
    """Create WebSocket routes bound to *server*."""
    router = APIRouter()

    @router.websocket("/ws")
    async def websocket_handler(websocket: WebSocket) -> None:
        """Handle WebSocket connections."""
        await websocket.accept()
        server._ws_clients.append(websocket)
        # Bounded per-client delivery (VAS-06/10, TD-03): this socket's
        # broadcasts flow through its own bounded queue + sender task; the
        # broadcast producer never awaits this socket directly. Keyed
        # servers defer registration until the auth handshake succeeds —
        # unauthenticated sockets are excluded from broadcast admission
        # anyway (VAS-05h), so they don't need queue+task resources for the
        # bounded handshake window.
        if not server.api_keys:
            server._ensure_ws_delivery(websocket)
        logger.info(f"WebSocket client connected. Total: {len(server._ws_clients)}")

        # Bounded handshake (VAS-05h): on a keyed server an unauthenticated
        # socket may not occupy the broadcast recipient set indefinitely.
        # The deadline is a TOTAL window from accept, not a per-message
        # timeout — an unauthenticated client cannot extend it by dripping
        # frames. Once the ``auth`` handshake completes the socket receives
        # without a timeout; keyless servers keep the old unconditional
        # receive.
        auth_deadline = (
            float(getattr(server, "_ws_auth_deadline", DEFAULT_WS_AUTH_DEADLINE_SECONDS))
            if server.api_keys
            else None
        )
        loop = asyncio.get_running_loop()
        connected_at = loop.time()

        try:
            while True:
                if auth_deadline is not None and not getattr(
                    getattr(websocket, "state", None), "authenticated", False
                ):
                    remaining = auth_deadline - (loop.time() - connected_at)
                    if remaining <= 0:
                        raise asyncio.TimeoutError()
                    data = await asyncio.wait_for(websocket.receive_json(), timeout=remaining)
                else:
                    data = await websocket.receive_json()
                await server._handle_ws_message(websocket, data)
        except WebSocketDisconnect:
            pass
        except asyncio.TimeoutError:
            logger.info(
                "WebSocket client failed to authenticate within " f"{auth_deadline}s; closing"
            )
            try:
                await websocket.close(code=1008, reason="Authentication deadline exceeded")
            except Exception:
                pass
        finally:
            # Drop the bounded delivery state (cancels the sender task) —
            # teardown also removes the socket from _ws_clients.
            server._teardown_ws_delivery(websocket, reason="disconnected")
            if websocket in server._ws_clients:
                server._ws_clients.remove(websocket)
            logger.info(f"WebSocket client disconnected. Total: {len(server._ws_clients)}")

    @router.websocket("/ws/events")
    async def events_websocket_handler(websocket: WebSocket) -> None:
        """Handle EventBridge WebSocket connections for real-time events.

        Owner binding (Codex finding 9): the WS auth dependency resolved
        the connecting principal onto ``websocket.state.client_id``; that
        verified owner is bound to the subscription record so protected
        events (a stored resource owner) are delivered only to the owning
        principal. Keyless servers bind ``None`` — an unbound record is
        never admitted to a protected event (fail closed).
        """
        await websocket.accept()
        server._event_clients.append(websocket)
        client_id = uuid.uuid4().hex[:12]
        owner_client_id = getattr(getattr(websocket, "state", None), "client_id", None)
        logger.info(
            f"EventBridge client {client_id} connected. " f"Total: {len(server._event_clients)}"
        )

        async def send_event(message: str) -> None:
            try:
                await websocket.send_text(message)
            except Exception:
                pass

        if server._event_bridge:
            server._event_bridge._broadcaster.add_client(
                client_id,
                send_event,
                owner_client_id=owner_client_id,
            )

        try:
            while True:
                data = await websocket.receive_json()
                msg_type = data.get("type", "")

                # Inbound payload cap (Codex finding 22): the advertised
                # ``server.server_max_message_bytes`` control is enforced on
                # this transport too — oversized frames are refused, not
                # processed.
                max_message_bytes = int(getattr(server, "_ws_max_message_bytes", 0) or 0)
                if max_message_bytes:
                    payload_bytes = len(json.dumps(data, separators=(",", ":")).encode("utf-8"))
                    if payload_bytes > max_message_bytes:
                        await websocket.send_json(
                            {
                                "type": "error",
                                "message": (
                                    f"Message exceeds server_max_message_bytes "
                                    f"({payload_bytes} > {max_message_bytes})"
                                ),
                            }
                        )
                        continue

                if msg_type == "subscribe":
                    categories = data.get("categories", ["all"])
                    normalized = server._event_bridge._broadcaster.normalize_subscriptions(
                        categories
                    )
                    correlation_id = data.get("correlation_id")
                    server._event_bridge._broadcaster.update_subscriptions(
                        client_id,
                        normalized,
                        correlation_id=(
                            correlation_id if isinstance(correlation_id, str) else None
                        ),
                    )
                    logger.debug(f"Client {client_id} subscribed to: {categories}")
                    await websocket.send_json(
                        {
                            "type": "subscribed",
                            "categories": categories,
                            "correlation_id": (
                                correlation_id if isinstance(correlation_id, str) else None
                            ),
                        }
                    )

                elif msg_type == "unsubscribe":
                    server._event_bridge._broadcaster.update_subscriptions(
                        client_id,
                        set(),
                        correlation_id=None,
                    )
                    await websocket.send_json({"type": "unsubscribed"})

                elif msg_type == "ping":
                    await websocket.send_json({"type": "pong"})

        except WebSocketDisconnect:
            pass
        finally:
            if server._event_bridge:
                server._event_bridge._broadcaster.remove_client(client_id)
            if websocket in server._event_clients:
                server._event_clients.remove(websocket)
            logger.info(
                f"EventBridge client {client_id} disconnected. "
                f"Total: {len(server._event_clients)}"
            )

    return router
