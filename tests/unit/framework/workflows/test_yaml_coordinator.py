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

from unittest.mock import AsyncMock, Mock, patch

import pytest

from victor.framework.coordinators.yaml_coordinator import YAMLWorkflowCoordinator
from victor.workflows.context import WorkflowContext, WorkflowResult
from victor.workflows.definition import TransformNode, WorkflowDefinition


def test_get_executor_uses_runtime_executor_factory() -> None:
    coordinator = YAMLWorkflowCoordinator()
    sentinel = object()

    with patch(
        "victor.framework.coordinators.yaml_coordinator.create_legacy_workflow_executor",
        return_value=sentinel,
    ) as factory:
        assert coordinator._get_executor() is sentinel
        assert coordinator._get_executor() is sentinel

    factory.assert_called_once_with()


@pytest.mark.asyncio
@pytest.mark.parametrize("success", [True, False])
@pytest.mark.parametrize("interrupted", [True, False])
async def test_definition_execution_passes_definition_and_preserves_result(success, interrupted):
    workflow = WorkflowDefinition(
        name="definition",
        nodes={"start": TransformNode(id="start", name="start")},
        start_node="start",
    )
    execution_result = WorkflowResult(
        workflow_name=workflow.name,
        success=success,
        context=WorkflowContext(data={"answer": 42}),
        error=None if success else "transform failed",
        interrupted=interrupted,
        interrupt_node="start" if interrupted else None,
    )
    executor = Mock(execute=AsyncMock(return_value=execution_result))
    coordinator = YAMLWorkflowCoordinator(executor=executor, use_unified_compiler=False)
    with patch.object(coordinator, "load_workflow", return_value=workflow):
        result = await coordinator.execute(
            "workflow.yaml", initial_state={"input": 21}, thread_id="thread", timeout=3
        )

    executor.execute.assert_awaited_once_with(
        workflow, initial_context={"input": 21}, thread_id="thread", timeout=3
    )
    assert result.success is success
    assert result.final_state == {"answer": 42}
    assert result.error == execution_result.error
    assert result.interrupted is interrupted
    assert result.interrupt_node == execution_result.interrupt_node


@pytest.mark.asyncio
async def test_definition_execution_reports_executor_exception():
    executor = Mock(execute=AsyncMock(side_effect=RuntimeError("engine failed")))
    coordinator = YAMLWorkflowCoordinator(executor=executor, use_unified_compiler=False)
    with patch.object(coordinator, "load_workflow", return_value=Mock()):
        result = await coordinator.execute("workflow.yaml")

    assert not result.success
    assert result.error == "engine failed"


def test_get_streaming_executor_uses_runtime_executor_factory() -> None:
    coordinator = YAMLWorkflowCoordinator()
    sentinel = object()

    with patch(
        "victor.framework.coordinators.yaml_coordinator.create_legacy_streaming_workflow_executor",
        return_value=sentinel,
    ) as factory:
        assert coordinator._get_streaming_executor() is sentinel
        assert coordinator._get_streaming_executor() is sentinel

    factory.assert_called_once_with()
