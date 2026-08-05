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

import logging
import uuid
from typing import TYPE_CHECKING

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

if TYPE_CHECKING:
    from victor.integrations.api.fastapi_server import VictorFastAPIServer

logger = logging.getLogger(__name__)


def create_router(server: "VictorFastAPIServer") -> APIRouter:
    """Create WebSocket routes bound to *server*."""
    router = APIRouter()

    @router.websocket("/ws")
    async def websocket_handler(websocket: WebSocket) -> None:
        """Handle WebSocket connections."""
        await websocket.accept()
        server._ws_clients.append(websocket)
        logger.info(f"WebSocket client connected. Total: {len(server._ws_clients)}")

        try:
            while True:
                data = await websocket.receive_json()
                await server._handle_ws_message(websocket, data)
        except WebSocketDisconnect:
            pass
        finally:
            if websocket in server._ws_clients:
                server._ws_clients.remove(websocket)
            logger.info(f"WebSocket client disconnected. Total: {len(server._ws_clients)}")

    @router.websocket("/ws/events")
    async def events_websocket_handler(websocket: WebSocket) -> None:
        """Handle EventBridge WebSocket connections for real-time events."""
        await websocket.accept()
        server._event_clients.append(websocket)
        client_id = uuid.uuid4().hex[:12]
        logger.info(
            f"EventBridge client {client_id} connected. " f"Total: {len(server._event_clients)}"
        )

        async def send_event(message: str) -> None:
            try:
                await websocket.send_text(message)
            except Exception:
                pass

        if server._event_bridge:
            server._event_bridge._broadcaster.add_client(client_id, send_event)

        try:
            while True:
                data = await websocket.receive_json()
                msg_type = data.get("type", "")

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
