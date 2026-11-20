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

"""Conversation / History routes: /conversation/reset, /conversation/export, /undo, /redo, /history."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse

from victor.integrations.api.change_tracker_ops import (
    change_history,
    redo_last_change,
    undo_last_change,
)
from victor.integrations.api.operator_posture import posture_json_response, request_principal

if TYPE_CHECKING:
    from victor.integrations.api.fastapi_server import VictorFastAPIServer

logger = logging.getLogger(__name__)


def create_router(server: "VictorFastAPIServer") -> APIRouter:
    """Create conversation / history routes bound to *server*."""
    router = APIRouter()

    def _posture_refusal(surface: str, http_request: Request) -> Optional[JSONResponse]:
        """Enforced single-trusted-operator posture (Codex finding 3): reset
        and export address the ONE server-wide conversation with no owner in
        the lookup, so under the multi-credential posture a second credential
        must never reach them (sticky first-principal claim)."""
        return posture_json_response(
            server.claim_shared_state(request_principal(http_request), surface), surface
        )

    @router.post("/conversation/reset", tags=["Conversation"])
    async def reset_conversation(http_request: Request) -> JSONResponse:
        """Reset conversation history."""
        refusal = _posture_refusal("conversation/reset", http_request)
        if refusal is not None:
            return refusal
        await server._record_rl_feedback()
        # ✅ NEW: Use VictorClient (service layer) instead of direct orchestrator access
        await server.reset_conversation()
        return JSONResponse({"success": True, "message": "Conversation reset"})

    @router.get("/conversation/export", tags=["Conversation"])
    async def export_conversation(http_request: Request, format: str = Query("json")) -> Any:
        """Export conversation history."""
        refusal = _posture_refusal("conversation/export", http_request)
        if refusal is not None:
            return refusal
        # ✅ NEW: Use VictorClient (service layer) instead of direct orchestrator access
        messages = await server.get_conversation_messages()

        if format == "markdown":
            content = server._format_messages_markdown(messages)
            return StreamingResponse(
                iter([content]),
                media_type="text/markdown",
            )
        else:
            return JSONResponse({"messages": messages})

    @router.post("/undo", tags=["History"])
    async def undo() -> JSONResponse:
        """Undo last change."""
        return JSONResponse(undo_last_change())

    @router.post("/redo", tags=["History"])
    async def redo() -> JSONResponse:
        """Redo last undone change."""
        return JSONResponse(redo_last_change())

    @router.get("/history", tags=["History"])
    async def history(limit: int = Query(10, ge=1, le=100)) -> JSONResponse:
        """Get change history."""
        return JSONResponse(change_history(limit))

    return router
