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

"""Team routes: /teams, /teams/{id}, /teams/{id}/start, /teams/{id}/cancel, etc."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse

if TYPE_CHECKING:
    from victor.integrations.api.fastapi_server import VictorFastAPIServer

logger = logging.getLogger(__name__)

# Module-level in-memory team storage
_teams: Dict[str, Dict[str, Any]] = {}
_team_messages: Dict[str, List[Dict[str, Any]]] = {}


def create_router(server: "VictorFastAPIServer") -> APIRouter:
    """Create team routes bound to *server*."""
    router = APIRouter(tags=["Teams"])

    @router.post("/teams")
    async def create_team(request: Request) -> JSONResponse:
        """Create a new agent team."""
        try:
            data = await request.json()
            name = data.get("name", "Unnamed Team")
            goal = data.get("goal", "")
            formation = data.get("formation", "sequential")
            members = data.get("members", [])
            total_tool_budget = data.get("total_tool_budget", 100)

            team_id = str(uuid.uuid4())[:8]

            team_members = []
            for i, m in enumerate(members):
                member = {
                    "id": m.get("id", f"member_{i+1}"),
                    "role": m.get("role", "executor"),
                    "name": m.get("name", f"Agent {i+1}"),
                    "goal": m.get("goal", ""),
                    "status": "pending",
                    "tool_budget": total_tool_budget // len(members) if members else 0,
                    "tools_used": 0,
                    "discoveries": [],
                    "agent_category": m.get(
                        "agent_category",
                        "supervisor" if m.get("is_manager", False) else "specialist",
                    ),
                    "is_manager": m.get("is_manager", False),
                }
                team_members.append(member)

            if formation == "hierarchical" and team_members:
                team_members[0]["agent_category"] = "supervisor"
                team_members[0]["is_manager"] = True

            _teams[team_id] = {
                "id": team_id,
                "name": name,
                "goal": goal,
                "formation": formation,
                "status": "draft",
                "members": team_members,
                "total_tool_budget": total_tool_budget,
                "total_tools_used": 0,
                "start_time": None,
                "end_time": None,
                "current_step": None,
                "output": None,
                "error": None,
            }
            _team_messages[team_id] = []

            await server._broadcast_ws(
                {
                    "type": "agent_event",
                    "event": "team_created",
                    "data": _teams[team_id],
                    "timestamp": time.time(),
                }
            )

            return JSONResponse(
                {
                    "success": True,
                    "team_id": team_id,
                    "message": f"Team '{name}' created",
                }
            )

        except Exception:
            logger.exception("Failed to create team")
            return JSONResponse({"error": "Internal server error"}, status_code=500)

    @router.get("/teams")
    async def list_teams(
        status: Optional[str] = Query(None, description="Filter by status"),
    ) -> JSONResponse:
        """List all teams."""
        teams_list = []
        for team in _teams.values():
            if status is None or team.get("status") == status:
                teams_list.append(team)
        return JSONResponse({"teams": teams_list})

    @router.get("/teams/{team_id}")
    async def get_team(team_id: str) -> JSONResponse:
        """Get team details."""
        if team_id not in _teams:
            return JSONResponse({"error": "Team not found"}, status_code=404)
        return JSONResponse(_teams[team_id])

    @router.post("/teams/{team_id}/start")
    async def start_team(team_id: str) -> JSONResponse:
        """Start team execution."""
        if team_id not in _teams:
            return JSONResponse({"error": "Team not found"}, status_code=404)

        team = _teams[team_id]
        if team["status"] not in ("draft", "paused"):
            return JSONResponse(
                {"error": f"Cannot start team in status: {team['status']}"},
                status_code=400,
            )

        team["status"] = "running"
        team["start_time"] = time.time()

        for member in team["members"]:
            member["status"] = "pending"

        await server._broadcast_ws(
            {
                "type": "agent_event",
                "event": "team_started",
                "data": team,
                "timestamp": time.time(),
            }
        )

        async def execute_team():
            try:
                from victor.teams import (
                    TeamConfig,
                    TeamAgentCategory,
                    TeamMember,
                    TeamFormation,
                    create_coordinator,
                )
                from victor.agent.subagents import SubAgentRole

                orchestrator = await server._get_orchestrator()

                role_map = {
                    "researcher": SubAgentRole.RESEARCHER,
                    "planner": SubAgentRole.PLANNER,
                    "executor": SubAgentRole.EXECUTOR,
                    "reviewer": SubAgentRole.REVIEWER,
                    "tester": SubAgentRole.TESTER,
                }
                formation_map = {
                    "sequential": TeamFormation.SEQUENTIAL,
                    "parallel": TeamFormation.PARALLEL,
                    "hierarchical": TeamFormation.HIERARCHICAL,
                    "pipeline": TeamFormation.PIPELINE,
                }

                members = []
                for m in team["members"]:
                    role = role_map.get(m["role"], SubAgentRole.EXECUTOR)
                    members.append(
                        TeamMember(
                            id=m["id"],
                            role=role,
                            name=m["name"],
                            goal=m["goal"],
                            tool_budget=m["tool_budget"],
                            agent_category=TeamAgentCategory(
                                m.get(
                                    "agent_category",
                                    ("supervisor" if m.get("is_manager", False) else "specialist"),
                                )
                            ),
                            is_manager=m.get("is_manager", False),
                        )
                    )

                config = TeamConfig(
                    name=team["name"],
                    goal=team["goal"],
                    members=members,
                    formation=formation_map.get(team["formation"], TeamFormation.SEQUENTIAL),
                    total_tool_budget=team["total_tool_budget"],
                )

                coordinator = create_coordinator(
                    orchestrator=orchestrator,
                    enable_observability=False,
                    enable_rl=False,
                )
                result = await coordinator.execute_team(config)

                if team_id in _teams:
                    team["status"] = "completed" if result.success else "failed"
                    team["end_time"] = time.time()
                    team["output"] = result.final_output
                    team["total_tools_used"] = result.total_tool_calls

                    for member in team["members"]:
                        if member["id"] in result.member_results:
                            mr = result.member_results[member["id"]]
                            member["status"] = "completed" if mr.success else "failed"
                            member["tools_used"] = mr.tool_calls_used
                            member["discoveries"] = mr.discoveries

                    await server._broadcast_ws(
                        {
                            "type": "agent_event",
                            "event": ("team_completed" if result.success else "team_failed"),
                            "data": team,
                            "timestamp": time.time(),
                        }
                    )

            except Exception as e:
                logger.exception(f"Team {team_id} execution error")
                if team_id in _teams:
                    team["status"] = "failed"
                    team["error"] = str(e)
                    team["end_time"] = time.time()

                    await server._broadcast_ws(
                        {
                            "type": "agent_event",
                            "event": "team_failed",
                            "data": team,
                            "timestamp": time.time(),
                        }
                    )

        asyncio.create_task(execute_team())

        return JSONResponse(
            {
                "success": True,
                "message": f"Team {team_id} started",
            }
        )

    @router.post("/teams/{team_id}/cancel")
    async def cancel_team(team_id: str) -> JSONResponse:
        """Cancel a running team."""
        if team_id not in _teams:
            return JSONResponse({"error": "Team not found"}, status_code=404)

        team = _teams[team_id]
        if team["status"] != "running":
            return JSONResponse(
                {"error": f"Cannot cancel team in status: {team['status']}"},
                status_code=400,
            )

        team["status"] = "cancelled"
        team["end_time"] = time.time()

        await server._broadcast_ws(
            {
                "type": "agent_event",
                "event": "team_cancelled",
                "data": team,
                "timestamp": time.time(),
            }
        )

        return JSONResponse(
            {
                "success": True,
                "message": f"Team {team_id} cancelled",
            }
        )

    @router.post("/teams/clear")
    async def clear_teams() -> JSONResponse:
        """Clear completed/failed/cancelled teams."""
        cleared = 0
        to_delete = []
        for tid, team in _teams.items():
            if team["status"] in ("completed", "failed", "cancelled"):
                to_delete.append(tid)

        for tid in to_delete:
            del _teams[tid]
            if tid in _team_messages:
                del _team_messages[tid]
            cleared += 1

        return JSONResponse(
            {
                "success": True,
                "cleared": cleared,
            }
        )

    @router.get("/teams/{team_id}/messages")
    async def get_team_messages(team_id: str) -> JSONResponse:
        """Get team inter-agent messages."""
        if team_id not in _teams:
            return JSONResponse({"error": "Team not found"}, status_code=404)

        messages = _team_messages.get(team_id, [])
        return JSONResponse({"messages": messages})

    return router
