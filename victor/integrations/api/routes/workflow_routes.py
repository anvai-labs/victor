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

"""Workflow routes: /workflows/*."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

from fastapi import (
    APIRouter,
    HTTPException,
    Query,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import HTMLResponse, JSONResponse

from victor.integrations.api.graph_export import get_execution_state

if TYPE_CHECKING:
    from victor.integrations.api.fastapi_server import VictorFastAPIServer

logger = logging.getLogger(__name__)


def create_router(server: "VictorFastAPIServer") -> APIRouter:
    """Create workflow routes bound to *server*."""
    router = APIRouter(tags=["Workflows"])

    @router.get("/workflows/templates")
    async def list_workflow_templates() -> JSONResponse:
        """List available workflow templates."""
        try:
            from victor.workflows import get_workflow_registry

            registry = get_workflow_registry()
            templates = []

            for workflow_id, workflow_def in registry._workflows.items():
                templates.append(
                    {
                        "id": workflow_id,
                        "name": workflow_def.name,
                        "description": workflow_def.description or "",
                        "category": workflow_def.metadata.get("category", "General"),
                        "steps": [
                            {
                                "id": node.id,
                                "name": node.name or node.id,
                                "type": node.type.value,
                                "role": getattr(node, "role", None),
                                "goal": getattr(node, "goal", None),
                            }
                            for node in workflow_def.nodes.values()
                        ],
                        "tags": workflow_def.metadata.get("tags", []),
                        "estimated_duration": workflow_def.metadata.get("estimated_duration"),
                    }
                )

            return JSONResponse({"templates": templates})

        except ImportError:
            return JSONResponse({"templates": []})
        except Exception:
            logger.exception("Failed to list workflow templates")
            return JSONResponse({"error": "Internal server error"}, status_code=500)

    @router.get("/workflows/templates/{template_id}")
    async def get_workflow_template(template_id: str) -> JSONResponse:
        """Get workflow template details."""
        try:
            from victor.workflows import get_workflow_registry

            registry = get_workflow_registry()
            workflow_def = registry.get(template_id)

            if workflow_def is None:
                return JSONResponse({"error": "Template not found"}, status_code=404)

            return JSONResponse(
                {
                    "id": template_id,
                    "name": workflow_def.name,
                    "description": workflow_def.description or "",
                    "category": workflow_def.metadata.get("category", "General"),
                    "steps": [
                        {
                            "id": node.id,
                            "name": node.name or node.id,
                            "type": node.type.value,
                            "role": getattr(node, "role", None),
                            "goal": getattr(node, "goal", None),
                        }
                        for node in workflow_def.nodes.values()
                    ],
                    "tags": workflow_def.metadata.get("tags", []),
                }
            )

        except ImportError:
            return JSONResponse({"error": "Workflows module not available"}, status_code=404)
        except Exception:
            logger.exception("Failed to get workflow template")
            return JSONResponse({"error": "Internal server error"}, status_code=500)

    @router.post("/workflows/execute")
    async def execute_workflow(request: Request) -> JSONResponse:
        """Execute a workflow."""
        try:
            data = await request.json()
            template_id = data.get("template_id")
            parameters = data.get("parameters", {})

            if not template_id:
                return JSONResponse({"error": "template_id required"}, status_code=400)

            from victor.workflows import get_workflow_registry, WorkflowExecutor

            registry = get_workflow_registry()
            workflow_def = registry.get(template_id)

            if workflow_def is None:
                return JSONResponse({"error": "Template not found"}, status_code=404)

            execution_id = str(uuid.uuid4())[:8]

            server._workflow_executions[execution_id] = {
                "id": execution_id,
                "workflow_id": template_id,
                "workflow_name": workflow_def.name,
                "status": "running",
                "parameters": parameters,
                "current_step": None,
                "progress": 0,
                "start_time": time.time(),
                "end_time": None,
                "steps": [
                    {
                        "id": node.id,
                        "name": node.name or node.id,
                        "type": node.type.value,
                        "status": "pending",
                    }
                    for node in workflow_def.nodes.values()
                ],
                "output": None,
                "error": None,
            }

            await server._broadcast_ws(
                {
                    "type": "agent_event",
                    "event": "workflow_started",
                    "data": server._workflow_executions[execution_id],
                    "timestamp": time.time(),
                }
            )

            async def run_workflow():
                try:
                    orchestrator = await server._get_orchestrator()
                    executor = WorkflowExecutor(orchestrator)

                    result = await executor.execute(
                        workflow_def,
                        initial_context=parameters,
                    )

                    if execution_id in server._workflow_executions:
                        exec_state = server._workflow_executions[execution_id]
                        exec_state["status"] = "completed" if result.success else "failed"
                        exec_state["end_time"] = time.time()
                        exec_state["progress"] = 100
                        exec_state["output"] = (
                            str(result.final_output) if result.final_output else None
                        )

                        for step in exec_state["steps"]:
                            node_result = result.node_results.get(step["id"])
                            if node_result:
                                step["status"] = "completed" if node_result.success else "failed"
                                step["duration"] = (
                                    node_result.duration_ms / 1000
                                    if node_result.duration_ms
                                    else None
                                )

                        await server._broadcast_ws(
                            {
                                "type": "agent_event",
                                "event": (
                                    "workflow_completed" if result.success else "workflow_failed"
                                ),
                                "data": exec_state,
                                "timestamp": time.time(),
                            }
                        )

                except Exception as e:
                    logger.exception(f"Workflow {execution_id} error")
                    if execution_id in server._workflow_executions:
                        exec_state = server._workflow_executions[execution_id]
                        exec_state["status"] = "failed"
                        exec_state["error"] = str(e)
                        exec_state["end_time"] = time.time()

                        await server._broadcast_ws(
                            {
                                "type": "agent_event",
                                "event": "workflow_failed",
                                "data": exec_state,
                                "timestamp": time.time(),
                            }
                        )

            asyncio.create_task(run_workflow())

            return JSONResponse(
                {
                    "success": True,
                    "execution_id": execution_id,
                    "message": f"Workflow '{workflow_def.name}' started",
                }
            )

        except ImportError:
            return JSONResponse({"error": "Workflows module not available"}, status_code=500)
        except Exception:
            logger.exception("Failed to execute workflow")
            return JSONResponse({"error": "Internal server error"}, status_code=500)

    @router.get("/workflows/executions")
    async def list_workflow_executions(
        status: Optional[str] = Query(None, description="Filter by status"),
    ) -> JSONResponse:
        """List workflow executions."""
        executions = []
        for exec_state in server._workflow_executions.values():
            if status is None or exec_state.get("status") == status:
                executions.append(exec_state)
        return JSONResponse({"executions": executions})

    @router.get("/workflows/executions/{execution_id}")
    async def get_workflow_execution(execution_id: str) -> JSONResponse:
        """Get workflow execution details."""
        if execution_id not in server._workflow_executions:
            return JSONResponse({"error": "Execution not found"}, status_code=404)
        return JSONResponse(server._workflow_executions[execution_id])

    @router.post("/workflows/executions/{execution_id}/cancel")
    async def cancel_workflow_execution(execution_id: str) -> JSONResponse:
        """Cancel a workflow execution."""
        if execution_id not in server._workflow_executions:
            return JSONResponse({"error": "Execution not found"}, status_code=404)

        exec_state = server._workflow_executions[execution_id]
        if exec_state["status"] != "running":
            return JSONResponse(
                {"error": f"Cannot cancel execution in status: {exec_state['status']}"},
                status_code=400,
            )

        exec_state["status"] = "cancelled"
        exec_state["end_time"] = time.time()

        await server._broadcast_ws(
            {
                "type": "agent_event",
                "event": "workflow_cancelled",
                "data": exec_state,
                "timestamp": time.time(),
            }
        )

        return JSONResponse(
            {
                "success": True,
                "message": f"Workflow execution {execution_id} cancelled",
            }
        )

    # ---- Workflow Visualization ----

    @router.get("/workflows/{workflow_id}/graph")
    async def get_workflow_graph(workflow_id: str) -> JSONResponse:
        """Get static workflow graph structure for visualization."""
        try:
            if workflow_id in server._workflow_executions:
                exec_data = server._workflow_executions[workflow_id]
                if "graph" in exec_data:
                    return JSONResponse(exec_data["graph"])

            raise HTTPException(
                status_code=404,
                detail=f"Workflow {workflow_id} not found. Execute the workflow first.",
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to get graph for workflow {workflow_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to export graph: {str(e)}")

    @router.get("/workflows/{workflow_id}/execution")
    async def get_workflow_execution_status(workflow_id: str) -> JSONResponse:
        """Get current workflow execution status."""
        try:
            if workflow_id not in server._workflow_executions:
                raise HTTPException(
                    status_code=404,
                    detail=f"Workflow execution {workflow_id} not found",
                )

            exec_state = get_execution_state(workflow_id, server._workflow_executions)
            return JSONResponse(exec_state.to_dict())

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to get execution state for {workflow_id}: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to get execution state: {str(e)}",
            )

    @router.websocket("/workflows/{workflow_id}/stream")
    async def workflow_websocket_stream(websocket: WebSocket, workflow_id: str) -> None:
        """Real-time workflow execution event stream via WebSocket."""
        await websocket.accept()

        if not server._workflow_event_bridge:
            await websocket.close(code=1011, reason="Workflow event bridge not initialized")
            return

        try:
            await server._workflow_event_bridge.handle_websocket_connection(
                websocket, workflow_id, client_id=uuid.uuid4().hex[:12]
            )
        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected for workflow {workflow_id}")
        except Exception as e:
            logger.error(f"WebSocket error for workflow {workflow_id}: {e}")
            try:
                await websocket.close(code=1011, reason=str(e))
            except Exception:
                pass

    @router.get("/workflows/visualize/{workflow_id}", response_class=HTMLResponse)
    async def visualize_workflow(workflow_id: str) -> HTMLResponse:
        """Serve HTML page with interactive workflow visualization."""
        try:
            template_path = Path(__file__).parent.parent / "templates" / "workflow_visualizer.html"

            if not template_path.exists():
                html_content = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Victor Workflow Visualization</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 40px; }}
                        .error {{ color: red; }}
                    </style>
                </head>
                <body>
                    <h1>Workflow Visualization Not Available</h1>
                    <p class="error">Template file not found: {template_path}</p>
                    <p>The workflow visualization feature requires the template file to be installed.</p>
                </body>
                </html>
                """
                return HTMLResponse(content=html_content, status_code=200)

            with open(template_path, "r", encoding="utf-8") as f:
                template_html = f.read()

            return HTMLResponse(content=template_html)

        except Exception as e:
            logger.error(f"Failed to serve visualization for {workflow_id}: {e}")
            return HTMLResponse(
                content=f"<html><body><h1>Error</h1><p>{str(e)}</p></body></html>",
                status_code=500,
            )

    return router
