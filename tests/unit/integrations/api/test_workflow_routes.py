"""Workflow API reports the compatibility executor's actual result contract."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from victor.integrations.api.routes import workflow_routes
from victor.workflows.context import WorkflowContext, WorkflowResult
from victor.workflows.definition import TransformNode, WorkflowDefinition
from victor.workflows.registry import WorkflowRegistry
from victor_contracts.workflows import ExecutorNodeStatus, NodeResult


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["completed", "failed", "paused", "exception"])
async def test_execution_reports_result_and_pause_without_attribute_errors(monkeypatch, outcome):
    workflow = WorkflowDefinition(
        name="test",
        nodes={
            "first": TransformNode(id="first", name="first"),
            "second": TransformNode(id="second", name="second"),
        },
        start_node="first",
    )
    node_result = NodeResult(
        node_id="first",
        status=(ExecutorNodeStatus.FAILED if outcome == "failed" else ExecutorNodeStatus.COMPLETED),
        duration_seconds=0.125,
    )
    result = WorkflowResult(
        workflow_name="test",
        success=outcome != "failed",
        context=WorkflowContext(data={"answer": 0}, node_results={"first": node_result}),
        error="node failed" if outcome == "failed" else None,
    )
    result.interrupted = outcome == "paused"
    result.interrupt_node = "second" if result.interrupted else None
    executor = Mock(execute=AsyncMock(return_value=result))
    if outcome == "exception":
        executor.execute.side_effect = RuntimeError("engine failed")
    factory = Mock(return_value=executor)
    monkeypatch.setattr(
        "victor.workflows.runtime_executor_factory.create_legacy_workflow_executor", factory
    )
    monkeypatch.setattr(
        "victor.workflows.get_global_registry", lambda: Mock(get=Mock(return_value=workflow))
    )
    server = SimpleNamespace(
        _workflow_executions={}, _get_orchestrator=AsyncMock(return_value=object())
    )
    finished = asyncio.Event()

    async def broadcast(event):
        if event["event"] != "workflow_started":
            finished.set()

    server._broadcast_ws = AsyncMock(side_effect=broadcast)
    app = FastAPI()
    app.include_router(workflow_routes.create_router(server))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/workflows/execute", json={"template_id": "test", "parameters": {"input": 7}}
        )
        assert response.status_code == 200
        await asyncio.wait_for(finished.wait(), timeout=2)

    execution = server._workflow_executions[response.json()["execution_id"]]
    expected_status = "failed" if outcome == "exception" else outcome
    assert execution["status"] == expected_status
    assert server._broadcast_ws.call_args.args[0]["event"] == f"workflow_{expected_status}"
    factory.assert_called_once_with(server._get_orchestrator.return_value)
    executor.execute.assert_awaited_once_with(workflow, initial_context={"input": 7})
    if outcome == "exception":
        assert execution["error"] == "engine failed"
        return
    assert execution["output"] == "{'answer': 0}"
    assert execution["steps"][0]["duration"] == 0.125
    assert execution["error"] == result.error
    if outcome == "paused":
        assert execution["end_time"] is None
        assert execution["progress"] == 50
        assert execution["interrupt_node"] == "second"
        assert execution["steps"][1]["status"] == "pending"
    else:
        assert execution["end_time"] is not None


@pytest.mark.asyncio
async def test_template_routes_use_public_registry_and_definition_contracts(monkeypatch):
    registry = WorkflowRegistry()
    registry.register(
        WorkflowDefinition(
            name="real-template",
            nodes={"start": TransformNode(id="start", name="start")},
            start_node="start",
        )
    )
    monkeypatch.setattr("victor.workflows.get_global_registry", lambda: registry)
    app = FastAPI()
    app.include_router(workflow_routes.create_router(SimpleNamespace()))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        listing = await client.get("/workflows/templates")
        details = await client.get("/workflows/templates/real-template")
        missing = await client.get("/workflows/templates/missing")

    assert listing.status_code == details.status_code == 200
    assert listing.json()["templates"][0]["id"] == "real-template"
    assert details.json()["steps"][0]["type"] == "transform"
    assert missing.status_code == 404
