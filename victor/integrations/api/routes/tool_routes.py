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

"""Tool routes: /tools, /tools/approve, /tools/pending."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from victor.integrations.api.fastapi_server import ToolApprovalRequest


class ToolCancelRequest(BaseModel):
    """Body for POST /tools/cancel."""

    tool_call_id: str


if TYPE_CHECKING:
    from victor.integrations.api.fastapi_server import VictorFastAPIServer

logger = logging.getLogger(__name__)


def create_router(server: "VictorFastAPIServer") -> APIRouter:
    """Create tool routes bound to *server*."""
    router = APIRouter(tags=["Tools"])

    @router.get("/tools")
    async def list_tools() -> JSONResponse:
        """List available tools with metadata."""
        try:
            from victor.tools.registry import ToolRegistry

            registry = ToolRegistry()
            tools_info = []

            for tool in registry.list_tools():
                cost_tier = (
                    tool.cost_tier.value
                    if hasattr(tool, "cost_tier") and tool.cost_tier
                    else "free"
                )
                category = server._get_tool_category(tool.name)

                tools_info.append(
                    {
                        "name": tool.name,
                        "description": tool.description or "",
                        "category": category,
                        "cost_tier": cost_tier,
                        "parameters": (tool.parameters if hasattr(tool, "parameters") else {}),
                        "is_dangerous": server._is_dangerous_tool(tool.name),
                        "requires_approval": cost_tier in ("medium", "high")
                        or server._is_dangerous_tool(tool.name),
                    }
                )

            tools_info.sort(key=lambda t: (t["category"], t["name"]))

            return JSONResponse(
                {
                    "tools": tools_info,
                    "total": len(tools_info),
                    "categories": list({t["category"] for t in tools_info}),
                }
            )

        except Exception as e:
            logger.exception("List tools error")
            return JSONResponse({"tools": [], "total": 0, "error": str(e)})

    @router.post("/tools/approve")
    async def approve_tool(request: ToolApprovalRequest) -> JSONResponse:
        """Approve or reject a pending tool execution."""
        if request.approval_id in server._pending_tool_approvals:
            approval = server._pending_tool_approvals.pop(request.approval_id)
            approval["approved"] = request.approved
            approval["resolved"] = True

            await server._broadcast_ws(
                {
                    "type": "tool_approval_resolved",
                    "approval_id": request.approval_id,
                    "approved": request.approved,
                    "tool_name": approval.get("tool_name"),
                }
            )

            return JSONResponse(
                {
                    "success": True,
                    "approval_id": request.approval_id,
                    "approved": request.approved,
                }
            )
        else:
            raise HTTPException(status_code=404, detail="Approval not found")

    @router.get("/tools/pending")
    async def pending_approvals() -> JSONResponse:
        """Get list of pending tool approvals."""
        pending = [
            {
                "approval_id": aid,
                "tool_name": info.get("tool_name"),
                "arguments": info.get("arguments"),
                "danger_level": info.get("danger_level"),
                "cost_tier": info.get("cost_tier"),
                "created_at": info.get("created_at"),
            }
            for aid, info in server._pending_tool_approvals.items()
            if not info.get("resolved")
        ]

        return JSONResponse({"pending": pending, "count": len(pending)})

    @router.post("/tools/cancel")
    async def cancel_tool(request: ToolCancelRequest) -> JSONResponse:
        """Cancel a pending tool execution by its tool_call_id (or approval_id)."""
        target = request.tool_call_id
        match_id = None
        for aid, info in server._pending_tool_approvals.items():
            if aid == target or info.get("tool_call_id") == target:
                match_id = aid
                break

        if match_id is None:
            return JSONResponse({"tool_call_id": target, "cancelled": False, "reason": "not_found"})

        approval = server._pending_tool_approvals.pop(match_id)
        approval["approved"] = False
        approval["resolved"] = True
        # Unblock any waiter, mirroring how an approval resolves.
        event = approval.get("event")
        if event is not None and hasattr(event, "set"):
            event.set()

        try:
            await server._broadcast_ws(
                {
                    "type": "tool_cancelled",
                    "approval_id": match_id,
                    "tool_call_id": target,
                    "tool_name": approval.get("tool_name"),
                }
            )
        except Exception:
            logger.debug("broadcast tool_cancelled failed", exc_info=True)

        return JSONResponse({"tool_call_id": target, "cancelled": True})

    return router
