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
from unittest.mock import AsyncMock, MagicMock

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
    async def test_perceive_node_fallback_is_json_serializable(self):
        """Fallback labels must survive durable JSON checkpoint serialization."""
        result = await perceive_node(create_initial_state(query="Write a function"))

        assert result.perception == {
            "intent": "write",
            "complexity": "medium",
            "confidence": 0.5,
        }
        assert result.model_dump(mode="json")["perception"] == result.perception

    async def test_perceive_node_accepts_dict_input(self):
        """Perceive node should normalize raw dict input at the boundary."""
        result = await perceive_node({"query": "Test dict input"})

        assert isinstance(result, AgenticLoopStateModel)
        assert result.stage == "perceive"
        assert result.iteration == 1

    @pytest.mark.asyncio
    async def test_perceive_node_preserves_run_owned_intent_across_drift(self):
        """Later perception drift cannot rewrite the run's authorization scope."""
        state = create_initial_state(query="Explain this repository")
        state = state.model_copy(update={"original_intent": "read_only"})
        perception = MagicMock(
            intent="write_allowed",
            complexity="medium",
            task_analysis=MagicMock(task_type="research"),
            confidence=0.9,
        )
        runtime_intelligence = AsyncMock(analyze_turn=AsyncMock(return_value=perception))

        result = await perceive_node(state, runtime_intelligence=runtime_intelligence)

        assert result.perception["intent"] == "write_allowed"
        assert result.original_intent == "read_only"
        assert result.model_dump(mode="json")["original_intent"] == "read_only"

    @pytest.mark.asyncio
    async def test_perceive_node_captures_first_intent_as_durable_primitive(self):
        """The first perception establishes the run-owned intent as JSON data."""
        state = create_initial_state(query="Explain this repository")
        perception = MagicMock(
            intent="read_only",
            complexity="medium",
            task_analysis=MagicMock(task_type="research"),
            confidence=0.9,
        )
        runtime_intelligence = AsyncMock(analyze_turn=AsyncMock(return_value=perception))

        result = await perceive_node(state, runtime_intelligence=runtime_intelligence)

        assert result.original_intent == "read_only"
        assert result.model_dump(mode="json")["original_intent"] == "read_only"


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
        mock_executor._is_question_only = lambda message: False
        mock_executor.execute_turn = AsyncMock(
            return_value=MagicMock(
                response="Code written",
                tool_results=[{"tool": "write_file", "status": "success"}],
                disposition="continue",
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
        mock_executor._is_question_only = lambda message: False
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

        from types import SimpleNamespace
        from unittest.mock import AsyncMock

        # Real FulfillmentDetector contract: async check_fulfillment returning
        # a result with is_fulfilled/score/reason (FOU-D5f: the graph node
        # previously called a phantom get_fulfillment_result API).
        detector = SimpleNamespace(
            check_fulfillment=AsyncMock(
                return_value=SimpleNamespace(is_fulfilled=True, score=0.9, reason="done")
            )
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
        mock_executor._is_question_only = lambda message: False
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


class TestActNodeResponseText:
    """FOU-D5f soak finding: act_node stored the raw CompletionResponse object
    in action_result['response']; _default_evaluation called .lower() on it and
    crashed into the fallback 'continue' decision — every run spun to the
    iteration cap and never completed. The node must store the response TEXT
    (the shape LoopResult.from_graph_result and evaluation both consume)."""

    @staticmethod
    def _turn(response):
        from victor.agent.services.turn_execution_runtime import TurnResult
        from victor.providers.base import CompletionResponse

        return TurnResult(response=CompletionResponse(content=response))

    async def test_act_node_stores_response_text_not_object(self):
        from types import SimpleNamespace
        from unittest.mock import AsyncMock

        from victor.framework.agentic_graph.nodes import act_node
        from victor.framework.agentic_graph.state import AgenticLoopStateModel

        te = SimpleNamespace(execute_turn=AsyncMock(return_value=self._turn("SOAK-OK")))
        state = AgenticLoopStateModel(query="task", max_iterations=5)
        result = await act_node(state, turn_executor=te)
        assert result.action_result.content == "SOAK-OK"

    async def test_default_evaluation_completes_on_response_text(self):
        from types import SimpleNamespace

        from victor.framework.agentic_graph.nodes import _default_evaluation

        state = SimpleNamespace(action_result={"response": "The task completed successfully."})
        evaluation = _default_evaluation(state)
        assert evaluation["decision"] == "complete"

    async def test_default_evaluation_survives_object_response(self):
        """Defense in depth: even if an object leaks into the dict, evaluation
        must coerce instead of crashing into the fallback 'continue'."""
        from types import SimpleNamespace

        from victor.providers.base import CompletionResponse

        from victor.framework.agentic_graph.nodes import _default_evaluation

        state = SimpleNamespace(
            action_result={"response": CompletionResponse(content="done, finished successfully")}
        )
        evaluation = _default_evaluation(state)
        assert evaluation["decision"] == "complete"


class TestEvaluateNodeRealFulfillmentAPI:
    """FOU-D5f soak finding: the graph evaluate node called a phantom
    ``get_fulfillment_result`` API — the real ``FulfillmentDetector`` exposes
    ``async check_fulfillment``. Every production evaluation crashed into the
    fallback 'continue', so graph runs never completed."""

    async def test_evaluate_node_uses_check_fulfillment_and_records_result(self):
        from types import SimpleNamespace
        from unittest.mock import AsyncMock

        from victor.framework.agentic_graph.nodes import evaluate_node
        from victor.framework.agentic_graph.state import AgenticLoopStateModel

        detector = SimpleNamespace(
            check_fulfillment=AsyncMock(
                return_value=SimpleNamespace(is_fulfilled=True, score=0.95, reason="criteria met")
            )
        )
        state = AgenticLoopStateModel(query="task", max_iterations=5)

        result = await evaluate_node(
            state,
            evaluator=None,
            fulfillment_detector=detector,
            enable_fulfillment_check=True,
        )

        detector.check_fulfillment.assert_awaited_once()
        assert result.fulfillment is not None
        assert result.fulfillment["is_fulfilled"] is True
        assert result.fulfillment["score"] == 0.95

    async def test_fulfillment_failure_does_not_crash_evaluation(self):
        from types import SimpleNamespace
        from unittest.mock import AsyncMock

        from victor.framework.agentic_graph.nodes import evaluate_node
        from victor.framework.agentic_graph.state import AgenticLoopStateModel

        detector = SimpleNamespace(
            check_fulfillment=AsyncMock(side_effect=RuntimeError("strategy missing"))
        )
        state = AgenticLoopStateModel(query="task", max_iterations=5)

        result = await evaluate_node(
            state,
            evaluator=None,
            fulfillment_detector=detector,
            enable_fulfillment_check=True,
        )

        # Evaluation survives: a decision exists and the failure is contained.
        assert result.evaluation is not None


class TestTurnResultFlowsToEvaluator:
    """FOU-D5f root cause: EnhancedCompletionEvaluator keys its Q&A shortcut
    (and every disposition-aware branch) off ``isinstance(action_result,
    TurnResult)`` — but the graph path handed it a plain dict, so the model
    answering a simple question could never COMPLETE and every run spun to
    the iteration cap ('Insufficient progress' forever)."""

    @staticmethod
    def _turn(content):
        from victor.agent.services.turn_execution_runtime import TurnResult
        from victor.providers.base import CompletionResponse

        return TurnResult(response=CompletionResponse(content=content))

    async def test_act_completion_observer_publishes_durable_record(self):
        from victor.framework.agentic_graph.executor import _turn_record_observer
        from victor.framework.agentic_graph.nodes import act_node
        from victor.framework.agentic_graph.state import AgenticLoopStateModel
        from victor.framework.agentic_graph.turn_record import TurnRecordChannel
        from types import SimpleNamespace
        from unittest.mock import AsyncMock

        te = SimpleNamespace(execute_turn=AsyncMock(return_value=self._turn("SOAK-OK")))
        state = AgenticLoopStateModel(query="task", max_iterations=5)
        channel = TurnRecordChannel()
        completed = await act_node(state, turn_executor=te)

        await _turn_record_observer(channel)("complete", "act", completed, None, 0.1)
        await _turn_record_observer(channel)("complete", "plan", completed, None, 0.1)

        assert completed.action_result.content == "SOAK-OK"
        published = channel.latest
        assert published is not None and published.content == "SOAK-OK"
        assert completed.to_dict()["action_result"]["schema_version"] == "victor.turn.record.v1"

    async def test_evaluate_node_passes_turn_result_to_evaluator(self):
        from types import SimpleNamespace
        from unittest.mock import AsyncMock

        from victor.agent.services.turn_execution_runtime import TurnResult
        from victor.framework.agentic_graph.nodes import evaluate_node
        from victor.framework.agentic_graph.state import AgenticLoopStateModel

        from victor.framework.agentic_graph.turn_record import TurnRecord

        turn = self._turn("SOAK-OK")
        evaluator = AsyncMock()
        evaluator.evaluate = AsyncMock(
            return_value=SimpleNamespace(decision="complete", score=0.9, reason="qa")
        )
        state = AgenticLoopStateModel(query="task", max_iterations=5)
        state = state.model_copy(
            update={
                "stage": "act",
                "action_result": TurnRecord.from_turn_result(turn),
            }
        )

        await evaluate_node(state, evaluator=evaluator, enable_fulfillment_check=False)

        passed = (
            evaluator.evaluate.call_args.kwargs.get("action_result")
            or evaluator.evaluate.call_args.args[1]
        )
        assert isinstance(passed, TurnResult)

    async def test_real_evaluator_completes_qa_answer_on_graph_path(self):
        """The production failure, pinned: a tool-free answer must COMPLETE
        through the real EnhancedCompletionEvaluator on the graph path. The
        runtime marks question-task turns ``is_qa_response=True``; with the
        TurnResult flowing through to the evaluator, the Q&A shortcut fires."""
        from victor.agent.services.turn_execution_runtime import TurnDisposition
        from victor.framework.agentic_graph.nodes import evaluate_node
        from victor.framework.agentic_graph.state import AgenticLoopStateModel
        from victor.framework.enhanced_completion_evaluation import EnhancedCompletionEvaluator

        from victor.framework.agentic_graph.turn_record import TurnRecord

        turn = self._turn("SOAK-OK")
        turn.is_qa_response = True
        turn.disposition = TurnDisposition.COMPLETION_CANDIDATE
        evaluator = EnhancedCompletionEvaluator()
        state = AgenticLoopStateModel(query="Reply with exactly: SOAK-OK", max_iterations=5)
        state = state.model_copy(
            update={
                "stage": "act",
                "action_result": TurnRecord.from_turn_result(turn),
            }
        )

        result = await evaluate_node(state, evaluator=evaluator, enable_fulfillment_check=False)
        decision = result.evaluation["decision"]
        assert getattr(decision, "value", decision) == "complete"


class TestActNodePassesQaSignal:
    """FOU-D5f final seam: the graph ACT node must derive is_qa_task from the
    runtime's _is_question_only like the legacy loop does — otherwise the
    runtime never marks Q&A turns and simple questions never complete."""

    async def test_act_node_passes_is_qa_task_for_questions(self):
        from types import SimpleNamespace
        from unittest.mock import AsyncMock

        from victor.framework.agentic_graph.nodes import act_node
        from victor.framework.agentic_graph.state import AgenticLoopStateModel

        captured = {}

        async def execute_turn(**kwargs):
            captured.update(kwargs)
            from victor.providers.base import CompletionResponse

            return SimpleNamespace(response=CompletionResponse(content="SOAK-OK"), tool_results=[])

        te = SimpleNamespace(
            execute_turn=AsyncMock(side_effect=execute_turn),
            _is_question_only=lambda message: True,
        )
        state = AgenticLoopStateModel(query="What is this?", max_iterations=5)
        await act_node(state, turn_executor=te)
        assert captured.get("is_qa_task") is True

    async def test_act_node_defaults_is_qa_task_false_without_hook(self):
        from types import SimpleNamespace
        from unittest.mock import AsyncMock

        from victor.framework.agentic_graph.nodes import act_node
        from victor.framework.agentic_graph.state import AgenticLoopStateModel

        captured = {}

        async def execute_turn(**kwargs):
            captured.update(kwargs)
            from victor.providers.base import CompletionResponse

            return SimpleNamespace(response=CompletionResponse(content="done"), tool_results=[])

        te = SimpleNamespace(execute_turn=AsyncMock(side_effect=execute_turn))
        state = AgenticLoopStateModel(query="task", max_iterations=5)
        await act_node(state, turn_executor=te)
        assert captured.get("is_qa_task") is False
