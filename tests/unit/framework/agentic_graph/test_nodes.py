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

"""Tests for agentic loop nodes (perceive, plan, act, evaluate)."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from victor.framework.graph import CopyOnWriteState
from victor.framework.agentic_graph.state import (
    AgenticLoopStateModel,
    create_initial_state,
)
from victor.framework.agentic_graph.nodes import (
    perceive_node,
    plan_node,
    act_node,
    evaluate_node,
    decide_edge,
    _build_fast_path_plan,
    _default_evaluation,
)


class TestPerceiveNode:
    """Tests for perceive_node."""

    @pytest.mark.asyncio
    async def test_perceive_node_basic(self):
        """Test basic perception execution."""
        state = create_initial_state(query="Write a function")

        # Create mock perception result
        mock_perception_result = MagicMock(
            intent=MagicMock(value="write"),
            complexity="medium",
            task_analysis=MagicMock(task_type="code_generation"),
            confidence=0.8,
        )

        # Create mock runtime intelligence
        mock_rt = AsyncMock()
        mock_rt.analyze_turn = AsyncMock(return_value=mock_perception_result)

        result = await perceive_node(state, runtime_intelligence=mock_rt)

        assert result.stage == "perceive"
        assert result.perception is not None
        assert result.task_type == "code_generation"
        assert result.complexity == "medium"

    @pytest.mark.asyncio
    async def test_perceive_node_with_context(self):
        """Test perception with additional context."""
        state = create_initial_state(
            query="Fix the bug",
            context={"project": "victor", "file": "agent.py"},
        )

        mock_perception_result = MagicMock(
            intent=MagicMock(value="edit"),
            complexity="low",
            task_analysis=MagicMock(task_type="debugging"),
            confidence=0.9,
        )

        mock_rt = AsyncMock()
        mock_rt.analyze_turn = AsyncMock(return_value=mock_perception_result)

        result = await perceive_node(state, runtime_intelligence=mock_rt)

        assert result.perception is not None
        assert result.task_type == "debugging"

    @pytest.mark.asyncio
    async def test_perceive_node_error_handling(self):
        """Test perception error handling."""
        state = create_initial_state(query="Test")

        # No service provided - should use fallback
        result = await perceive_node(state)

        assert result.stage == "perceive"
        # Should have fallback perception
        assert result.perception is not None

    @pytest.mark.asyncio
    async def test_perceive_node_accepts_dict_input(self):
        """Perceive node should normalize raw dict input at the boundary."""
        result = await perceive_node({"query": "Test dict input"})

        assert isinstance(result, AgenticLoopStateModel)
        assert result.stage == "perceive"
        assert result.iteration == 1


class TestPlanNode:
    """Tests for plan_node."""

    @pytest.mark.asyncio
    async def test_plan_node_with_planning_enabled(self):
        """Test planning when LLM planning is enabled."""
        state = create_initial_state(query="Write tests")
        state = state.model_copy(
            update={
                "perception": {"intent": "write"},
                "task_type": "code_generation",
            }
        )

        mock_coordinator = AsyncMock()
        mock_coordinator.chat_with_planning = AsyncMock(
            return_value=MagicMock(
                content="Plan: write unit tests",
                tool_calls=["code_search", "write_file"],
            )
        )

        result = await plan_node(
            state, planning_coordinator=mock_coordinator, use_llm_planning=True
        )

        assert result.stage == "plan"
        assert result.plan is not None
        assert result.planning_events[-1]["selection_policy"] == "llm_planning"
        assert result.planning_events[-1]["used_llm_planning"] is True

    @pytest.mark.asyncio
    async def test_plan_node_lifts_plan_execution_state_from_response_metadata(self):
        """LLM planning metadata should become graph-checkpointable state."""
        state = create_initial_state(query="Write tests")
        execution_state = {
            "plan_id": "plan-1",
            "execution_mode": "team_adapter",
            "success": True,
            "step_statuses": {"1": "completed"},
        }

        mock_coordinator = AsyncMock()
        mock_coordinator.chat_with_planning = AsyncMock(
            return_value=MagicMock(
                content="Plan: write unit tests",
                tool_calls=["code_search", "write_file"],
                metadata={"plan_execution_state": execution_state},
            )
        )

        result = await plan_node(
            state,
            planning_coordinator=mock_coordinator,
            use_llm_planning=True,
        )

        assert result.plan_execution_state == execution_state
        assert result.planning_events[-1]["has_plan_execution_state"] is True

    @pytest.mark.asyncio
    async def test_plan_node_fast_path(self):
        """Test planning with fast path (no LLM)."""
        state = create_initial_state(query="Simple task")
        state = state.model_copy(
            update={
                "perception": {"intent": "read"},
                "complexity": "low",
            }
        )

        result = await plan_node(state, use_llm_planning=False)

        assert result.stage == "plan"
        assert result.plan is not None
        # Fast path should have simple plan structure
        assert result.plan.get("approach") == "fast_path"
        assert result.planning_events[-1]["selection_policy"] == "heuristic_fast_path"
        assert result.planning_events[-1]["used_llm_planning"] is False


class TestActNode:
    """Tests for act_node."""

    @pytest.mark.asyncio
    async def test_act_node_with_turn_executor(self):
        """Test action execution via TurnExecutor."""
        state = create_initial_state(query="Write code")
        state = state.model_copy(
            update={
                "plan": {"tool_calls": ["write_file"]},
                "task_type": "code_generation",
            }
        )

        mock_executor = AsyncMock()
        mock_executor.execute_turn = AsyncMock(
            return_value=MagicMock(
                response="Code written",
                tool_results=[{"tool": "write_file", "status": "success"}],
            )
        )

        result = await act_node(state, turn_executor=mock_executor)

        assert result.stage == "act"
        assert result.action_result is not None
        assert len(result.tool_results) > 0

    @pytest.mark.asyncio
    async def test_act_node_without_plan(self):
        """Test action node when no plan exists."""
        state = create_initial_state(query="Help me")

        result = await act_node(state, turn_executor=None)

        assert result.stage == "act"
        # Should have a basic action result
        assert result.action_result is not None

    @pytest.mark.asyncio
    async def test_act_node_records_degradation_event_on_error(self):
        """Action failures should be recorded in graph-side degradation metadata."""
        state = create_initial_state(query="Write code")
        mock_executor = AsyncMock()
        mock_executor.execute_turn = AsyncMock(side_effect=RuntimeError("provider failed"))

        result = await act_node(state, turn_executor=mock_executor)

        assert result.action_result["error"] == "provider failed"
        assert result.degradation_events[-1]["failure_type"] == "ACTION_ERROR"
        assert result.degradation_events[-1]["source"] == "agentic_graph"


class TestEvaluateNode:
    """Tests for evaluate_node."""

    @pytest.mark.asyncio
    async def test_evaluate_node_continue(self):
        """Test evaluation that returns continue decision."""
        state = create_initial_state(query="Write code")
        state = state.model_copy(
            update={
                "action_result": {"output": "Started"},
                "progress_scores": [0.3],
            }
        )

        evaluator = AsyncMock()
        evaluator.evaluate = AsyncMock(
            return_value=MagicMock(
                decision="continue",
                score=0.5,
                reason="Progress made",
            )
        )

        result = await evaluate_node(state, evaluator=evaluator)

        assert result.stage == "evaluate"
        assert result.evaluation is not None
        assert result.evaluation["decision"] == "continue"
        assert len(result.progress_scores) == 2

    @pytest.mark.asyncio
    async def test_evaluate_node_complete(self):
        """Test evaluation that returns complete decision."""
        state = create_initial_state(query="Simple task")
        state = state.model_copy(
            update={
                "action_result": {"output": "Done"},
                "progress_scores": [0.9],
            }
        )

        evaluator = AsyncMock()
        evaluator.evaluate = AsyncMock(
            return_value=MagicMock(
                decision="complete",
                score=0.95,
                reason="Task completed",
            )
        )

        result = await evaluate_node(state, evaluator=evaluator)

        assert result.evaluation["decision"] == "complete"

    @pytest.mark.asyncio
    async def test_evaluate_node_with_fulfillment_check(self):
        """Test evaluation with fulfillment detector."""
        state = create_initial_state(query="Write function")
        state = state.model_copy(
            update={
                "action_result": {"output": "Function written"},
            }
        )

        detector = MagicMock()
        detector.is_fulfilled = MagicMock(return_value=True)
        detector.get_fulfillment_result = MagicMock(
            return_value={"is_fulfilled": True, "confidence": 0.9}
        )

        result = await evaluate_node(
            state,
            fulfillment_detector=detector,
            enable_fulfillment_check=True,
        )

        assert result.fulfillment is not None
        assert result.fulfillment["is_fulfilled"] is True

    @pytest.mark.asyncio
    async def test_evaluate_node_records_degradation_event_on_error(self):
        """Evaluation failures should be recorded in graph-side degradation metadata."""
        state = create_initial_state(query="Write code")
        evaluator = AsyncMock()
        evaluator.evaluate = AsyncMock(side_effect=RuntimeError("judge unavailable"))

        result = await evaluate_node(state, evaluator=evaluator)

        assert result.evaluation["error"] == "judge unavailable"
        assert result.degradation_events[-1]["failure_type"] == "EVALUATION_ERROR"
        assert result.degradation_events[-1]["source"] == "agentic_graph"


class TestDecideEdge:
    """Tests for decide_edge conditional routing."""

    def test_decide_edge_continue(self):
        """Test decide edge routes to perceive for continue."""
        state = create_initial_state(query="Task")
        state = state.model_copy(
            update={
                "evaluation": {"decision": "continue"},
                "iteration": 1,
            }
        )

        next_node = decide_edge(state)
        assert next_node == "perceive"

    def test_decide_edge_complete(self):
        """Test decide edge routes to END for complete."""
        state = create_initial_state(query="Task")
        state = state.model_copy(
            update={
                "evaluation": {"decision": "complete"},
            }
        )

        next_node = decide_edge(state)
        assert next_node == "__end__"

    def test_decide_edge_fail(self):
        """Test decide edge routes to END for fail."""
        state = create_initial_state(query="Task")
        state = state.model_copy(
            update={
                "evaluation": {"decision": "fail"},
            }
        )

        next_node = decide_edge(state)
        assert next_node == "__end__"

    def test_decide_edge_retry(self):
        """Test decide edge routes to act for retry."""
        state = create_initial_state(query="Task")
        state = state.model_copy(
            update={
                "evaluation": {"decision": "retry"},
            }
        )

        next_node = decide_edge(state)
        assert next_node == "act"

    def test_decide_edge_max_iterations(self):
        """Test decide edge stops at max iterations."""
        state = create_initial_state(query="Task", max_iterations=3)
        state = state.model_copy(
            update={
                "iteration": 3,
                "evaluation": {"decision": "continue"},
            }
        )

        next_node = decide_edge(state)
        assert next_node == "__end__"

    def test_decide_edge_default_continue(self):
        """Test decide edge defaults to continue when no evaluation."""
        state = create_initial_state(query="Task")
        state = state.model_copy(update={"iteration": 1})

        next_node = decide_edge(state)
        assert next_node == "perceive"

    def test_decide_edge_accepts_copy_on_write_state(self):
        """Decide edge should normalize wrapped state before routing."""
        state = create_initial_state(query="Task")
        state = state.model_copy(update={"evaluation": {"decision": "complete"}})

        next_node = decide_edge(CopyOnWriteState(state))
        assert next_node == "__end__"


class TestFastPathPlanning:
    """Tests for fast-path planning logic."""

    def test_fast_path_plan_write_intent(self):
        """Test fast-path plan for write intent."""
        state = create_initial_state(query="Write code")
        state = state.model_copy(
            update={
                "perception": {"intent": "write"},
                "task_type": "code_generation",
            }
        )

        plan = _build_fast_path_plan(state)

        assert plan["approach"] == "fast_path"
        assert "write_file" in plan.get("tool_calls", [])

    def test_fast_path_plan_read_intent(self):
        """Test fast-path plan for read intent."""
        state = create_initial_state(query="Read code")
        state = state.model_copy(
            update={
                "perception": {"intent": "read"},
            }
        )

        plan = _build_fast_path_plan(state)

        assert "read_file" in plan.get("tool_calls", [])

    def test_fast_path_plan_debugging(self):
        """Test fast-path plan for debugging tasks."""
        state = create_initial_state(query="Fix bug")
        state = state.model_copy(
            update={
                "task_type": "debugging",
            }
        )

        plan = _build_fast_path_plan(state)

        assert "grep" in plan.get("tool_calls", [])


class TestDefaultEvaluation:
    """Tests for default evaluation logic."""

    def test_default_evaluation_error(self):
        """Test evaluation with error result."""
        state = create_initial_state(query="Test")
        state = state.model_copy(
            update={
                "action_result": {"error": "Something failed"},
            }
        )

        evaluation = _default_evaluation(state)

        assert evaluation["decision"] == "fail"
        assert evaluation["score"] == 0.0

    def test_default_evaluation_complete(self):
        """Test evaluation detects completion."""
        state = create_initial_state(query="Test")
        state = state.model_copy(
            update={
                "action_result": {"response": "Task completed successfully"},
            }
        )

        evaluation = _default_evaluation(state)

        assert evaluation["decision"] == "complete"
        assert evaluation["score"] == 0.9

    def test_default_evaluation_continue(self):
        """Test evaluation defaults to continue."""
        state = create_initial_state(query="Test")
        state = state.model_copy(
            update={
                "action_result": {"response": "Making progress"},
            }
        )

        evaluation = _default_evaluation(state)

        assert evaluation["decision"] == "continue"


class TestNodeIntegration:
    """Integration tests for node chains."""

    @pytest.mark.asyncio
    async def test_perceive_to_plan_flow(self):
        """Test flow from perceive to plan."""
        state = create_initial_state(query="Write code")

        # Create mock perception
        mock_perception_result = MagicMock(
            intent=MagicMock(value="write"),
            complexity="medium",
            task_analysis=MagicMock(task_type="code_generation"),
            confidence=0.8,
        )

        mock_rt = AsyncMock()
        mock_rt.analyze_turn = AsyncMock(return_value=mock_perception_result)

        # Execute perceive
        state = await perceive_node(state, runtime_intelligence=mock_rt)

        # Verify state is ready for planning
        assert state.stage == "perceive"
        assert state.task_type is not None

        # Execute plan (fast path)
        state = await plan_node(state, use_llm_planning=False)

        assert state.stage == "plan"
        assert state.plan is not None

    @pytest.mark.asyncio
    async def test_full_loop_single_iteration(self):
        """Test single full loop iteration."""
        state = create_initial_state(query="Help")

        # Perceive (mocked)
        mock_perception_result = MagicMock(
            intent=MagicMock(value="query"),
            complexity="low",
            task_analysis=MagicMock(task_type="general"),
            confidence=0.9,
        )

        mock_rt = AsyncMock()
        mock_rt.analyze_turn = AsyncMock(return_value=mock_perception_result)

        state = await perceive_node(state, runtime_intelligence=mock_rt)

        # Plan (fast path)
        state = await plan_node(state, use_llm_planning=False)

        # Act (mocked)
        mock_executor = AsyncMock()
        mock_executor.execute_turn = AsyncMock(
            return_value=MagicMock(
                response="Here's help",
                tool_results=[],
            )
        )
        state = await act_node(state, turn_executor=mock_executor)

        # Evaluate (mocked)
        evaluator = AsyncMock()
        evaluator.evaluate = AsyncMock(
            return_value=MagicMock(
                decision="complete",
                score=0.9,
                reason="Done",
            )
        )
        state = await evaluate_node(state, evaluator=evaluator)

        # Verify complete flow
        assert state.stage == "evaluate"
        assert state.evaluation["decision"] == "complete"

        # Decide
        next_node = decide_edge(state)
        assert next_node == "__end__"
