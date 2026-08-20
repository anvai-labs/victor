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

"""Plan routes: /plans, /plans/{id}, /plans/{id}/approve, /plans/{id}/execute."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import TYPE_CHECKING, Any, Dict

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

if TYPE_CHECKING:
    from victor.integrations.api.fastapi_server import VictorFastAPIServer

logger = logging.getLogger(__name__)

# Module-level in-memory plan storage (shared across requests)
_plans: Dict[str, Dict[str, Any]] = {}


def create_router(server: "VictorFastAPIServer") -> APIRouter:
    """Create plan routes bound to *server*."""
    router = APIRouter(tags=["Plans"])

    @router.get("/plans")
    async def list_plans() -> JSONResponse:
        """List all plans."""
        plans_list = []
        for plan_id, plan in _plans.items():
            plans_list.append(
                {
                    "id": plan_id,
                    "title": plan.get("title", ""),
                    "description": plan.get("description", ""),
                    "status": plan.get("status", "draft"),
                    "created_at": plan.get("created_at"),
                    "approved_at": plan.get("approved_at"),
                    "executed_at": plan.get("executed_at"),
                    "steps": plan.get("steps", []),
                }
            )
        return JSONResponse({"plans": plans_list})

    @router.post("/plans")
    async def create_plan(request: Request) -> JSONResponse:
        """Create a new plan."""
        try:
            data = await request.json()
            title = data.get("title", "Untitled Plan")
            description = data.get("description", "")
            steps = data.get("steps", [])

            plan_id = str(uuid.uuid4())[:8]

            _plans[plan_id] = {
                "id": plan_id,
                "title": title,
                "description": description,
                "status": "draft",
                "created_at": time.time(),
                "approved_at": None,
                "executed_at": None,
                "completed_at": None,
                "steps": steps,
                "current_step": 0,
                "output": "",
            }

            return JSONResponse(
                {
                    "id": plan_id,
                    "status": "draft",
                    "message": f"Plan created: {title}",
                }
            )

        except Exception:
            logger.exception("Failed to create plan")
            return JSONResponse({"error": "Internal server error"}, status_code=500)

    @router.get("/plans/{plan_id}")
    async def get_plan(plan_id: str) -> JSONResponse:
        """Get plan details."""
        if plan_id not in _plans:
            return JSONResponse({"error": "Plan not found"}, status_code=404)
        return JSONResponse(_plans[plan_id])

    @router.post("/plans/{plan_id}/approve")
    async def approve_plan(plan_id: str) -> JSONResponse:
        """Approve a plan for execution."""
        if plan_id not in _plans:
            return JSONResponse({"error": "Plan not found"}, status_code=404)

        plan = _plans[plan_id]
        if plan.get("status") != "draft":
            return JSONResponse(
                {"error": f"Plan is not in draft status (status: {plan.get('status')})"},
                status_code=400,
            )

        plan["status"] = "approved"
        plan["approved_at"] = time.time()

        return JSONResponse(
            {
                "success": True,
                "message": f"Plan {plan_id} approved",
                "status": "approved",
            }
        )

    @router.post("/plans/{plan_id}/execute")
    async def execute_plan(plan_id: str) -> JSONResponse:
        """Execute an approved plan."""
        if plan_id not in _plans:
            return JSONResponse({"error": "Plan not found"}, status_code=404)

        plan = _plans[plan_id]
        if plan.get("status") != "approved":
            return JSONResponse(
                {"error": f"Plan must be approved before execution (status: {plan.get('status')})"},
                status_code=400,
            )

        plan["status"] = "executing"
        plan["executed_at"] = time.time()

        async def execute_steps():
            try:
                steps = plan.get("steps", [])
                for i, step in enumerate(steps):
                    if plan_id not in _plans:
                        break
                    plan["current_step"] = i
                    step_desc = step.get("description", step) if isinstance(step, dict) else step
                    plan["output"] += f"\n## Step {i+1}: {step_desc}\n"
                    if isinstance(step, dict):
                        step["status"] = "completed"

                if plan_id in _plans:
                    plan["status"] = "completed"
                    plan["completed_at"] = time.time()
            except Exception as e:
                logger.exception(f"Plan {plan_id} execution error")
                if plan_id in _plans:
                    plan["status"] = "failed"
                    plan["error"] = str(e)

        asyncio.create_task(execute_steps())

        return JSONResponse(
            {
                "success": True,
                "message": f"Plan {plan_id} execution started",
                "status": "executing",
            }
        )

    @router.delete("/plans/{plan_id}")
    async def delete_plan(plan_id: str) -> JSONResponse:
        """Delete a plan."""
        if plan_id not in _plans:
            return JSONResponse({"error": "Plan not found"}, status_code=404)

        del _plans[plan_id]
        return JSONResponse({"success": True, "message": f"Plan {plan_id} deleted"})

    return router
