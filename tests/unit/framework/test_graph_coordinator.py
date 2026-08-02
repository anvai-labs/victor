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

"""Tests for GraphTurnExecutor compiler routing."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from victor.framework.coordinators.graph_coordinator import GraphTurnExecutor


class TestGraphTurnExecutorCompilerRouting:
    """Verify graph execution uses canonical compiler entrypoints."""

    @pytest.mark.asyncio
    async def test_execute_workflow_graph_uses_unified_compiler(self):
        coordinator = GraphTurnExecutor()
        graph = MagicMock(name="workflow_graph")
        cached_graph = MagicMock(name="cached_graph")
        cached_graph.invoke = AsyncMock(return_value={"result": "ok"})
        compiler = MagicMock(name="unified_compiler")
        compiler.compile_graph.return_value = cached_graph

        with patch.object(coordinator, "_get_unified_compiler", return_value=compiler) as getter:
            result = await coordinator.execute_workflow_graph(
                graph,
                initial_state={"input": "value"},
                use_node_runners=True,
            )

        getter.assert_called_once_with(use_node_runners=True)
        compiler.compile_graph.assert_called_once_with(graph)
        cached_graph.invoke.assert_called_once_with({"input": "value"})
        assert result.success is True
        assert result.final_state == {"result": "ok"}

    @pytest.mark.asyncio
    async def test_execute_definition_compiled_uses_unified_compiler(self):
        runner_registry = MagicMock(name="runner_registry")
        coordinator = GraphTurnExecutor(runner_registry=runner_registry)
        workflow = MagicMock(name="workflow_definition")
        cached_graph = MagicMock(name="cached_graph")
        cached_graph.invoke = AsyncMock(return_value={"status": "done"})
        compiler = MagicMock(name="unified_compiler")
        compiler.compile_definition.return_value = cached_graph

        with patch.object(coordinator, "_get_unified_compiler", return_value=compiler) as getter:
            result = await coordinator.execute_definition_compiled(
                workflow,
                initial_state={"input": "value"},
            )

        getter.assert_called_once_with(use_node_runners=True)
        compiler.compile_definition.assert_called_once_with(workflow)
        cached_graph.invoke.assert_called_once_with({"input": "value"})
        assert result.success is True
        assert result.final_state == {"status": "done"}

    def test_set_runner_registry_clears_cached_unified_compilers(self):
        coordinator = GraphTurnExecutor()
        with_runners = MagicMock(name="with_runners")
        without_runners = MagicMock(name="without_runners")
        coordinator._unified_compilers = {
            True: with_runners,
            False: without_runners,
        }
        runner_registry = MagicMock(name="runner_registry")

        coordinator.set_runner_registry(runner_registry)

        assert coordinator._runner_registry is runner_registry
        assert coordinator._unified_compilers == {}
