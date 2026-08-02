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

"""Agent routes: /agents/start, /agents, /agents/{id}, /agents/{id}/cancel, /agents/clear."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Optional

from fastapi import APIRouter, Body, Query
from fastapi.responses import JSONResponse
from victor.runtime.chat_runtime import resolve_chat_service, resolve_chat_runtime

if TYPE_CHECKING:
    from victor.integrations.api.fastapi_server import VictorFastAPIServer

logger = logging.getLogger(__name__)


def create_router(server: "VictorFastAPIServer") -> APIRouter:
    """Create agent routes bound to *server*."""
    router = APIRouter(tags=["Agents"])

    @router.post("/agents/start")
    async def start_agent(
        task: str = Body(..., description="Task for the agent to execute"),
        mode: str = Body("build", description="Agent mode: build, plan, explore"),
        name: Optional[str] = Body(None, description="Display name for the agent"),
    ) -> JSONResponse:
        """Start a new background agent."""
        try:
            from victor.agent.background_agent import (
                get_agent_manager,
                init_agent_manager,
            )

            manager = get_agent_manager()
            if manager is None:
                orchestrator = await server._get_orchestrator()
                chat_service = resolve_chat_service(orchestrator) or resolve_chat_runtime(
                    orchestrator
                )
                manager = init_agent_manager(
                    orchestrator=orchestrator,
                    chat_service=chat_service,
                    max_concurrent=4,
                    event_callback=lambda t, d: asyncio.create_task(
                        server._broadcast_agent_event(t, d)
                    ),
                )

            agent_id = await manager.start_agent(
                task=task,
                mode=mode,
                name=name,
            )

            return JSONResponse(
                {
                    "success": True,
                    "agent_id": agent_id,
                    "message": f"Agent started: {agent_id}",
                }
            )

        except RuntimeError as e:
            return JSONResponse({"error": str(e)}, status_code=429)
        except Exception:
            logger.exception("Failed to start agent")
            return JSONResponse({"error": "Internal server error"}, status_code=500)

    @router.get("/agents")
    async def list_agents(
        status: Optional[str] = Query(None, description="Filter by status"),
        limit: int = Query(20, ge=1, le=100),
    ) -> JSONResponse:
        """List background agents."""
        try:
            from victor.agent.background_agent import (
                get_agent_manager,
                AgentStatus,
            )

            manager = get_agent_manager()
            if manager is None:
                return JSONResponse({"agents": []})

            status_filter = None
            if status:
                try:
                    status_filter = AgentStatus(status)
                except ValueError:
                    pass

            agents = manager.list_agents(status=status_filter, limit=limit)
            return JSONResponse(
                {
                    "agents": agents,
                    "active_count": manager.active_count,
                }
            )

        except Exception:
            logger.exception("Failed to list agents")
            return JSONResponse({"error": "Internal server error"}, status_code=500)

    @router.get("/agents/{agent_id}")
    async def get_agent(agent_id: str) -> JSONResponse:
        """Get a specific agent's status."""
        try:
            from victor.agent.background_agent import get_agent_manager

            manager = get_agent_manager()
            if manager is None:
                return JSONResponse({"error": "No agents running"}, status_code=404)

            agent_data = manager.get_agent_status(agent_id)
            if agent_data is None:
                return JSONResponse({"error": "Agent not found"}, status_code=404)

            return JSONResponse(agent_data)

        except Exception:
            logger.exception("Failed to get agent")
            return JSONResponse({"error": "Internal server error"}, status_code=500)

    @router.post("/agents/{agent_id}/cancel")
    async def cancel_agent(agent_id: str) -> JSONResponse:
        """Cancel a running agent."""
        try:
            from victor.agent.background_agent import get_agent_manager

            manager = get_agent_manager()
            if manager is None:
                return JSONResponse({"error": "No agents running"}, status_code=404)

            cancelled = await manager.cancel_agent(agent_id)
            if cancelled:
                return JSONResponse(
                    {
                        "success": True,
                        "message": f"Agent {agent_id} cancelled",
                    }
                )
            else:
                return JSONResponse(
                    {"error": "Agent not found or not running"},
                    status_code=404,
                )

        except Exception:
            logger.exception("Failed to cancel agent")
            return JSONResponse({"error": "Internal server error"}, status_code=500)

    @router.post("/agents/clear")
    async def clear_agents() -> JSONResponse:
        """Clear completed/failed/cancelled agents."""
        try:
            from victor.agent.background_agent import get_agent_manager

            manager = get_agent_manager()
            if manager is None:
                return JSONResponse({"cleared": 0})

            cleared = manager.clear_completed()
            return JSONResponse(
                {
                    "success": True,
                    "cleared": cleared,
                    "message": f"Cleared {cleared} agents",
                }
            )

        except Exception:
            logger.exception("Failed to clear agents")
            return JSONResponse({"error": "Internal server error"}, status_code=500)

    return router
