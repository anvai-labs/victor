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

"""Integration tests for StateGraph vs Legacy agentic loop parity.

This module tests that the new StateGraph-based agentic loop produces
equivalent results to the legacy while loop implementation.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from victor.core.feature_flags import FeatureFlag, is_feature_enabled
from victor.framework.agentic_loop import AgenticLoop


class TestFeatureFlagIntegration:
    """Tests for feature flag integration in AgenticLoop."""

    @pytest.mark.asyncio
    async def test_stategraph_path_disabled_by_default(self):
        """Test that StateGraph path is disabled by default."""
        assert not is_feature_enabled(FeatureFlag.USE_STATEGRAPH_AGENTIC_LOOP)

    @pytest.mark.asyncio
    async def test_agentic_loop_with_flag_disabled(self):
        """Test AgenticLoop uses legacy path when flag is disabled."""
        from victor.core.feature_flags import get_feature_flag_manager

        # Ensure flag is disabled
        manager = get_feature_flag_manager()
        manager.set(FeatureFlag.USE_STATEGRAPH_AGENTIC_LOOP, False)

        try:
            mock_orchestrator = MagicMock()
            mock_orchestrator.planning_coordinator = AsyncMock()
            mock_orchestrator.planning_coordinator.chat_with_planning = AsyncMock(
                return_value=MagicMock(content="Plan content")
            )
            mock_orchestrator.chat = AsyncMock(return_value="Response")

            loop = AgenticLoop(
                orchestrator=mock_orchestrator,
                max_iterations=2,
            )

            result = await loop.run("Test query")

            # Should return a LoopResult
            assert hasattr(result, "success")
            assert hasattr(result, "iterations")
            assert hasattr(result, "total_duration")

        finally:
            # Clear override
            manager.clear_runtime_override(FeatureFlag.USE_STATEGRAPH_AGENTIC_LOOP)

    @pytest.mark.asyncio
    async def test_agentic_loop_with_flag_enabled(self):
        """Test AgenticLoop uses StateGraph path when flag is enabled."""
        from victor.core.feature_flags import get_feature_flag_manager

        # Enable flag
        manager = get_feature_flag_manager()
        manager.set(FeatureFlag.USE_STATEGRAPH_AGENTIC_LOOP, True)

        try:
            mock_orchestrator = MagicMock()

            loop = AgenticLoop(
                orchestrator=mock_orchestrator,
                max_iterations=2,
            )

            result = await loop.run("Test query")

            # Should return a LoopResult from StateGraph executor
            assert hasattr(result, "success")
            assert hasattr(result, "total_duration")
            assert result.metadata.get("executor_type") == "stategraph"

        finally:
            # Clear override
            manager.clear_runtime_override(FeatureFlag.USE_STATEGRAPH_AGENTIC_LOOP)


class TestStateGraphExecutor:
    """Tests for StateGraph-based executor."""

    @pytest.mark.asyncio
    async def test_executor_with_mock_services(self):
        """Test executor with mocked services injected."""
        from victor.framework.agentic_graph.executor import AgenticLoopGraphExecutor

        mock_context = MagicMock()

        executor = AgenticLoopGraphExecutor(
            execution_context=mock_context,
            max_iterations=3,
        )

        # Verify executor structure
        assert executor.max_iterations == 3
        assert executor.graph is not None
        assert executor.compiled is not None

        # Get execution stats
        stats = executor.get_execution_stats()
        assert stats["max_iterations"] == 3
        assert len(stats["graph_nodes"]) >= 4

    @pytest.mark.asyncio
    async def test_executor_run_with_injected_services(self):
        """Test executor with injected services."""
        from victor.framework.agentic_graph.executor import AgenticLoopGraphExecutor

        mock_context = MagicMock()
        mock_turn_executor = AsyncMock()
        mock_turn_executor.execute_turn = AsyncMock(
            return_value=MagicMock(
                response="Done",
                tool_results=[],
                has_tool_calls=False,
                tool_calls_count=0,
                successful_tool_count=0,
                failed_tool_count=0,
                all_tools_blocked=False,
                is_qa_response=False,
                content="Done",
            )
        )

        class FakeRuntimeIntelligence:
            async def analyze_turn(self, *args, **kwargs):
                raise AssertionError("analyze_turn should be replaced in this test")

        mock_rt = FakeRuntimeIntelligence()
        mock_rt.analyze_turn = AsyncMock(
            return_value=MagicMock(
                intent=MagicMock(value="query"),
                complexity="low",
                task_analysis=MagicMock(task_type="general"),
                confidence=0.9,
            ),
        )

        executor = AgenticLoopGraphExecutor(
            execution_context=mock_context,
            max_iterations=2,
        )

        # Inject services
        executor.turn_executor = mock_turn_executor
        executor.runtime_intelligence = mock_rt

        result = await executor.run("Test query")

        assert result is not None
        assert hasattr(result, "success")
        assert hasattr(result, "iterations")


class TestResultCompatibility:
    """Tests for result compatibility between implementations."""

    @pytest.mark.asyncio
    async def test_loop_result_structure(self):
        """Test that LoopResult has expected structure."""
        from victor.framework.agentic_graph.executor import LoopResult

        result = LoopResult(
            success=True,
            response="Test response",
            iterations=2,
            termination_reason="complete",
        )

        assert result.success is True
        assert result.response == "Test response"
        assert result.iterations == 2
        assert result.termination_reason == "complete"

    @pytest.mark.asyncio
    async def test_loop_result_from_graph_result(self):
        """Test creating LoopResult from graph execution result."""
        from victor.framework.agentic_graph.executor import LoopResult
        from victor.framework.agentic_graph.state import AgenticLoopStateModel

        state = AgenticLoopStateModel(query="Test", iteration=2)
        state = state.model_copy(
            update={
                "action_result": {"response": "Complete"},
                "evaluation": {"decision": "complete"},
            }
        )

        mock_graph_result = MagicMock()

        result = LoopResult.from_graph_result(mock_graph_result, state)

        assert result.success is True
        assert result.iterations == 2
        assert result.termination_reason == "complete"
