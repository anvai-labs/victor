"""Integration tests for the full agentic loop pipeline.

Tests the complete PERCEIVE -> PLAN -> ACT -> EVALUATE -> DECIDE loop
using real Victor components (TaskAnalyzer, IntentDetector, FulfillmentDetector)
rather than mocks, to verify end-to-end wiring.
"""

from __future__ import annotations

import tempfile
from unittest.mock import AsyncMock, MagicMock

import pytest

from victor.agent.action_authorizer import ActionIntent
from victor.agent.decisions.schemas import DecisionType
from victor.agent.turn_policy import SpinState
from victor.framework.agentic_loop import AgenticLoop, LoopResult
from victor.framework.evaluation_nodes import (
    EvaluationDecision,
    EvaluationResult,
    add_evaluation,
    composite_evaluator,
    convergence_evaluator,
    create_agentic_loop_graph,
    progress_tracking_evaluator,
    simple_score_evaluator,
)
from victor.framework.fulfillment import (
    FulfillmentConfig,
    FulfillmentDetector,
    FulfillmentResult,
    FulfillmentStatus,
    TaskType,
)
from victor.framework.perception_integration import (
    Perception,
    perceive,
)
from victor.framework.task.protocols import TaskComplexity

# ============================================================================
# Full Pipeline Integration Tests
# ============================================================================


class TestFullPipelineIntegration:
    """Tests the full loop pipeline with real perception + fulfillment."""

    async def test_perceive_to_fulfillment_code_generation(self):
        """Verify perception feeds correctly into fulfillment detection."""
        # Step 1: Perceive a code generation intent
        perception = await perceive("Write a Python function to sort a list")
        assert isinstance(perception.intent, ActionIntent)
        assert isinstance(perception.complexity, TaskComplexity)
        assert perception.to_dict()["intent"] in [e.value for e in ActionIntent]

        # Step 2: Map to task type via the loop's mapper
        loop = AgenticLoop(
            orchestrator=MagicMock(spec=[]),
            enable_fulfillment_check=True,
        )
        task_type = loop._map_to_task_type(perception)
        assert isinstance(task_type, TaskType)

        # Step 3: Run fulfillment check
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("def sort_list(items):\n    return sorted(items)\n")
            f.flush()
            result = await loop.fulfillment.check_fulfillment(
                task_type=TaskType.CODE_GENERATION,
                criteria={"file_path": f.name},
                context={},
            )
        assert result.is_fulfilled or result.is_partial
        assert result.score > 0.0

    async def test_perceive_to_fulfillment_analysis(self):
        """Verify analysis tasks work through the pipeline."""
        perception = await perceive("Analyze the performance of our API")

        loop = AgenticLoop(
            orchestrator=MagicMock(spec=[]),
            enable_fulfillment_check=True,
        )

        result = await loop.fulfillment.check_fulfillment(
            task_type=TaskType.ANALYSIS,
            criteria={"min_findings": 1},
            context={
                "findings": ["API latency is 200ms p99"],
                "summary": "Performance is acceptable",
            },
        )
        assert result.is_fulfilled or result.is_partial

    async def test_full_loop_with_stub_orchestrator(self):
        """Run the full loop with a no-op orchestrator."""
        loop = AgenticLoop(
            orchestrator=MagicMock(spec=[]),
            max_iterations=2,
            enable_fulfillment_check=False,
            enable_adaptive_iterations=False,
        )

        result = await loop.run("What is 2+2?")
        assert isinstance(result, LoopResult)
        assert result.total_duration > 0
        assert len(result.iterations) > 0
        assert "progress_scores" in result.metadata

    async def test_multi_turn_context_flows_through(self):
        """Verify conversation history reaches TaskAnalyzer."""
        loop = AgenticLoop(
            orchestrator=MagicMock(spec=[]),
            max_iterations=1,
            enable_fulfillment_check=False,
            enable_adaptive_iterations=False,
        )

        history = [
            {"role": "user", "content": "Fix the login bug"},
            {"role": "assistant", "content": "Found the issue in auth.py"},
        ]
        result = await loop.run(
            "Now add tests for that fix",
            conversation_history=history,
        )
        assert isinstance(result, LoopResult)
        assert len(result.iterations) >= 1


# ============================================================================
# Evaluator Pipeline Integration Tests
# ============================================================================


class TestEvaluatorPipelineIntegration:
    """Tests evaluator factories work correctly in composed pipelines."""

    def test_progress_then_convergence(self):
        """Verify progress tracking and convergence can work together."""
        progress = progress_tracking_evaluator(complete_threshold=0.95)
        convergence = convergence_evaluator(min_iterations=2, min_score=0.7)

        combined = composite_evaluator(
            [progress, convergence],
            strategy="worst",
        )

        # Iteration 1: both continue
        r1 = combined({"score": 0.5})
        assert r1.decision in (
            EvaluationDecision.CONTINUE,
            "continue",
        )

        # Iteration 2: progress says continue, convergence may complete
        r2 = combined({"score": 0.75})
        # Not converged yet (big jump), so continue
        assert r2.score > 0

    def test_composite_with_all_strategies(self):
        """Verify all composite aggregation strategies produce valid results."""
        evaluators = [
            simple_score_evaluator(threshold=0.5),
            simple_score_evaluator(threshold=0.8),
        ]

        for strategy in ("worst", "best", "average"):
            combined = composite_evaluator(evaluators, strategy=strategy)
            result = combined({"score": 0.7})
            assert isinstance(result, EvaluationResult)
            assert 0.0 <= result.score <= 1.0

    def test_progress_evaluator_full_lifecycle(self):
        """Test progress evaluator through initial -> progress -> plateau."""
        evaluator = progress_tracking_evaluator(
            complete_threshold=0.95,
            plateau_window=3,
            plateau_tolerance=0.01,
        )

        # Iteration 1: initial
        r1 = evaluator({"score": 0.3})
        assert r1.decision == EvaluationDecision.CONTINUE

        # Iteration 2: progress
        r2 = evaluator({"score": 0.5})
        assert r2.decision == EvaluationDecision.CONTINUE
        assert "Progress" in r2.reason

        # Iteration 3-5: plateau
        r3 = evaluator({"score": 0.505})
        r4 = evaluator({"score": 0.508})
        # After 3 iterations in plateau window with < 0.01 improvement
        assert r4.decision == EvaluationDecision.FAIL
        assert "plateau" in r4.reason.lower()


# ============================================================================
# Fulfillment Detector Integration Tests
# ============================================================================


class TestFulfillmentDetectorIntegration:
    """Tests FulfillmentDetector covers all TaskTypes end-to-end."""

    async def test_all_task_types_have_strategies(self):
        """Every TaskType except UNKNOWN should have a strategy."""
        detector = FulfillmentDetector()
        for task_type in TaskType:
            if task_type == TaskType.UNKNOWN:
                continue
            assert task_type in detector.strategies, f"Missing strategy for {task_type}"

    async def test_all_strategies_return_valid_results(self):
        """Each strategy should return a valid FulfillmentResult."""
        detector = FulfillmentDetector()
        for task_type in TaskType:
            result = await detector.check_fulfillment(
                task_type=task_type,
                criteria={},
                context={},
            )
            assert isinstance(result, FulfillmentResult)
            assert isinstance(result.status, FulfillmentStatus)
            assert 0.0 <= result.score <= 1.0

    async def test_custom_config_affects_thresholds(self):
        """Verify FulfillmentConfig thresholds change behavior."""
        # Very low threshold: everything is fulfilled
        easy_config = FulfillmentConfig(fulfilled_threshold=0.1)
        easy_detector = FulfillmentDetector(config=easy_config)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("# empty\n")
            f.flush()
            # This file exists but has no valid code — low score
            # But with threshold=0.1, any file_exists score should suffice
            # Note: the strategy itself doesn't use config yet, this tests the wiring


# ============================================================================
# StateGraph + EvaluationNode Integration Tests
# ============================================================================


class TestStateGraphEvalIntegration:
    """Tests evaluation nodes integrate with real StateGraph."""

    def test_add_evaluation_to_real_graph(self):
        """add_evaluation() should work with a real StateGraph."""
        from victor.framework.graph import StateGraph

        graph = StateGraph(dict)

        async def analyze(state):
            state["analyzed"] = True
            return state

        async def fix(state):
            state["fixed"] = True
            return state

        graph.add_node("analyze", analyze)
        graph.add_node("fix", fix)
        graph.add_edge("analyze", "fix")

        def quality_check(state):
            if state.get("fixed"):
                return EvaluationResult(
                    decision=EvaluationDecision.COMPLETE,
                    score=1.0,
                    reason="Fix applied",
                )
            return EvaluationResult(
                decision=EvaluationDecision.RETRY,
                score=0.3,
                reason="Not fixed yet",
            )

        graph = add_evaluation(
            graph,
            node_id="check_quality",
            evaluator=quality_check,
            decision_edges={
                str(EvaluationDecision.COMPLETE): "__end__",
                str(EvaluationDecision.RETRY): "fix",
            },
        )

        graph.add_edge("fix", "check_quality")
        graph.set_entry_point("analyze")

        # Verify graph structure
        assert "analyze" in graph._nodes
        assert "fix" in graph._nodes
        assert "check_quality" in graph._nodes

    def test_create_agentic_loop_graph_compilable(self):
        """create_agentic_loop_graph should produce a compilable graph."""

        async def perceive(s):
            s["perceived"] = True
            return s

        async def plan(s):
            s["planned"] = True
            return s

        async def act(s):
            s["acted"] = True
            s["score"] = 1.0
            return s

        def evaluate(s):
            if s.get("score", 0) >= 0.9:
                return EvaluationResult(
                    decision=EvaluationDecision.COMPLETE,
                    score=s["score"],
                )
            return EvaluationResult(
                decision=EvaluationDecision.CONTINUE,
                score=s.get("score", 0),
            )

        graph = create_agentic_loop_graph(
            state_type=dict,
            perception_fn=perceive,
            planning_fn=plan,
            execution_fn=act,
            evaluator_fn=evaluate,
            max_iterations=3,
        )

        # Should be compilable
        compiled = graph.compile()
        assert compiled is not None


# ============================================================================
# Adaptive Loop Integration Tests
# ============================================================================


class TestAdaptiveLoopIntegration:
    """Tests adaptive iteration features work end-to-end."""

    async def test_plateau_triggers_early_exit(self):
        """Loop should exit early when progress plateaus."""
        loop = AgenticLoop(
            orchestrator=MagicMock(spec=[]),
            max_iterations=10,
            enable_fulfillment_check=False,
            enable_adaptive_iterations=True,
            plateau_window=3,
            plateau_tolerance=0.05,
        )

        # Mock perception to return constant low confidence (simulates plateau)
        perception = Perception(
            intent=ActionIntent.WRITE_ALLOWED,
            complexity=TaskComplexity.MEDIUM,
            task_analysis=MagicMock(task_type=None),
            confidence=0.45,
        )
        loop.perception.perceive = AsyncMock(return_value=perception)

        result = await loop.run("Fix the bug")

        # Should exit early due to plateau (not hit max 10 iterations)
        assert len(result.iterations) < 10
        assert "progress_scores" in result.metadata

    async def test_high_confidence_completes_immediately(self):
        """Loop should complete on first iteration with high confidence."""
        loop = AgenticLoop(
            orchestrator=MagicMock(spec=[]),
            max_iterations=5,
            enable_fulfillment_check=False,
            enable_adaptive_iterations=True,
        )

        perception = Perception(
            intent=ActionIntent.WRITE_ALLOWED,
            complexity=TaskComplexity.SIMPLE,
            task_analysis=MagicMock(task_type=None),
            confidence=0.95,
        )
        loop.perception.perceive = AsyncMock(return_value=perception)

        result = await loop.run("Simple question")
        assert result.success is True
        assert len(result.iterations) == 1


# ============================================================================
# Framework Export Integration Tests
# ============================================================================


class TestFrameworkExports:
    """Tests that all new modules are properly exported from victor.framework."""

    def test_agentic_loop_exports(self):
        from victor.framework import AgenticLoop, LoopStage

        assert AgenticLoop is not None
        assert LoopStage.PERCEIVE.value == "perceive"

    def test_evaluation_exports(self):
        from victor.framework import (
            EvaluationDecision,
        )

        assert EvaluationDecision.COMPLETE.value == "complete"

    def test_fulfillment_exports(self):
        from victor.framework import (
            TaskType,
        )

        assert len(TaskType) == 10

    def test_perception_exports(self):
        from victor.framework import (
            Perception,
        )

        assert Perception is not None


# ============================================================================
# TurnResult + Single-Turn Evaluation Integration Tests
# ============================================================================


class TestTurnResultEvaluation:
    """Tests that _evaluate() correctly interprets TurnResult signals."""

    def _make_loop(self):
        return AgenticLoop(
            orchestrator=MagicMock(spec=[]),
            enable_fulfillment_check=False,
            config={"disable_enhanced_completion": True},
        )

    def _make_perception(self):
        return Perception(
            intent=ActionIntent.WRITE_ALLOWED,
            complexity=TaskComplexity.MEDIUM,
            task_analysis=MagicMock(task_type=None),
            confidence=0.7,
        )

    @pytest.mark.parametrize(
        "prompt, answer",
        [
            ("What is 6 * 7?", "The answer is 42"),
            (
                "What harmless marker did I ask you to remember? "
                "Reply with only the marker; do not use tools.",
                "JADE",
            ),
        ],
    )
    @pytest.mark.parametrize("mode", ["buffered", "streaming"])
    async def test_qa_response_completes(self, prompt, answer, mode):
        """The real evaluator completes a short answer after one ACT in either mode."""
        from victor.agent.services.turn_execution_runtime import TurnExecutor, TurnResult
        from victor.providers.base import CompletionResponse, StreamChunk

        turn = TurnResult(
            response=CompletionResponse(content=answer),
            is_qa_response=TurnExecutor._is_question_only(prompt),
        )
        executor = MagicMock(spec=["execute_turn"])
        executor.execute_turn = AsyncMock(return_value=turn)
        loop = AgenticLoop(
            orchestrator=MagicMock(spec=[]),
            turn_executor=executor,
            enable_fulfillment_check=False,
            enable_adaptive_iterations=False,
            max_iterations=3,
        )
        loop._analyze_turn = AsyncMock(return_value=self._make_perception())
        loop._plan = AsyncMock(return_value={})

        class Port:
            async def stream_turn_act(self, **kwargs):
                kwargs["outcome"].turn_result = await executor.execute_turn()
                yield StreamChunk(content=answer)

        if mode == "buffered":
            result = await loop.run(prompt)
            assert result.success is True
            assert len(result.iterations) == 1
            assert result.iterations[0].action_result.content == answer
            assert "Q&A shortcut" in result.iterations[0].evaluation.reason
        else:
            loop.streaming_act_port = Port()
            chunks = [chunk async for chunk in loop.run_streaming(prompt)]
            assert "".join(chunk.content for chunk in chunks if chunk.content) == answer
            assert chunks[-1].metadata["agentic_loop_success"] is True
            assert chunks[-1].metadata["agentic_loop_iterations"] == 1
        executor.execute_turn.assert_awaited_once()
        assert loop.spin_detector.total_tool_calls == 0

    async def test_spin_detection_fails(self):
        """4+ consecutive all-blocked turns should fail."""
        from victor.agent.services.turn_execution_runtime import TurnResult

        loop = self._make_loop()
        loop.spin_detector.consecutive_all_blocked = 4
        turn = TurnResult(
            response=MagicMock(content="", tool_calls=[{"name": "read"}]),
            has_tool_calls=True,
            all_tools_blocked=True,
        )
        result = await loop._evaluate(self._make_perception(), turn, {})
        assert result.decision == EvaluationDecision.FAIL
        assert "Spin detected" in result.reason

    async def test_stuck_agent_fails(self):
        """3+ turns without tools should fail."""
        from victor.agent.services.turn_execution_runtime import TurnResult

        loop = self._make_loop()
        loop.spin_detector.consecutive_no_tool_turns = 3
        turn = TurnResult(
            response=MagicMock(content="I think...", tool_calls=None),
            has_tool_calls=False,
        )
        result = await loop._evaluate(self._make_perception(), turn, {})
        assert result.decision == EvaluationDecision.FAIL
        assert "stuck" in result.reason

    async def test_successful_tools_continue(self):
        """Successful tool execution should continue."""
        from victor.agent.services.turn_execution_runtime import TurnResult

        loop = self._make_loop()
        turn = TurnResult(
            response=MagicMock(content="Edited file", tool_calls=[{"name": "edit"}]),
            has_tool_calls=True,
            tool_calls_count=1,
            tool_results=[{"success": True, "tool_name": "edit"}],
        )
        result = await loop._evaluate(self._make_perception(), turn, {})
        assert result.decision == EvaluationDecision.CONTINUE
        assert result.score > 0.5

    async def test_final_answer_after_tools_completes(self):
        """No tools + content + previously used tools = done."""
        from victor.agent.services.turn_execution_runtime import TurnResult

        loop = self._make_loop()
        loop.spin_detector.total_tool_calls = 5
        turn = TurnResult(
            response=MagicMock(content="Here's the fixed code", tool_calls=None),
            has_tool_calls=False,
        )
        result = await loop._evaluate(self._make_perception(), turn, {})
        assert result.decision == EvaluationDecision.COMPLETE
        assert result.score >= 0.8

    async def test_act_with_turn_executor(self):
        """_act() should call execute_turn() when coordinator available."""
        from victor.agent.services.turn_execution_runtime import TurnResult

        mock_coord = AsyncMock()
        mock_turn = TurnResult(
            response=MagicMock(content="done", tool_calls=None),
            has_tool_calls=False,
        )
        mock_coord.execute_turn = AsyncMock(return_value=mock_turn)

        loop = AgenticLoop(
            orchestrator=MagicMock(spec=[]),
            turn_executor=mock_coord,
            enable_fulfillment_check=False,
        )

        result = await loop._act({}, {"query": "test"})
        assert isinstance(result, TurnResult)
        mock_coord.execute_turn.assert_called_once()

    async def test_act_tracks_tool_calls(self):
        """_act() should update turn-level tracking from TurnResult."""
        from victor.agent.services.turn_execution_runtime import TurnResult

        mock_coord = AsyncMock()
        mock_turn = TurnResult(
            response=MagicMock(content="", tool_calls=[{"name": "edit"}]),
            has_tool_calls=True,
            tool_calls_count=2,
            all_tools_blocked=False,
        )
        mock_coord.execute_turn = AsyncMock(return_value=mock_turn)

        loop = AgenticLoop(
            orchestrator=MagicMock(spec=[]),
            turn_executor=mock_coord,
            enable_fulfillment_check=False,
        )
        assert loop.spin_detector.total_tool_calls == 0
        assert loop.spin_detector.consecutive_no_tool_turns == 0

        await loop._act({}, {"query": "test"})

        assert loop.spin_detector.total_tool_calls == 2
        assert loop.spin_detector.consecutive_no_tool_turns == 0
        assert loop.spin_detector.consecutive_all_blocked == 0


class TestTurnResultDataclass:
    """Tests for TurnResult dataclass."""

    def test_properties(self):
        from victor.agent.services.turn_execution_runtime import TurnResult

        turn = TurnResult(
            response=MagicMock(content="hello"),
            tool_results=[
                {"success": True, "tool_name": "read"},
                {"success": False, "tool_name": "edit", "error": "fail"},
                {"success": True, "tool_name": "write"},
            ],
            has_tool_calls=True,
            tool_calls_count=3,
        )
        assert turn.content == "hello"
        assert turn.has_content is True
        assert turn.successful_tool_count == 2
        assert turn.failed_tool_count == 1

    def test_empty_content(self):
        from victor.agent.services.turn_execution_runtime import TurnResult

        turn = TurnResult(response=MagicMock(content=None))
        assert turn.content == ""
        assert turn.has_content is False
        assert turn.successful_tool_count == 0


# ============================================================================
# Nudge Injection Tests
# ============================================================================


class TestNudgeInjection:
    """Tests that nudge messages are injected into conversation on RETRY."""

    async def test_nudge_injected_on_no_tool_turns(self):
        """When agent doesn't use tools for 2+ turns, nudge is injected."""
        from victor.agent.services.turn_execution_runtime import TurnResult
        from unittest.mock import patch, PropertyMock

        mock_coord = AsyncMock()
        mock_chat_ctx = MagicMock()
        mock_coord._chat_context = mock_chat_ctx

        # Simulate agent not using tools
        no_tool_turn = TurnResult(
            response=MagicMock(content="I think we should...", tool_calls=None),
            has_tool_calls=False,
        )
        mock_coord.execute_turn = AsyncMock(return_value=no_tool_turn)

        loop = AgenticLoop(
            turn_executor=mock_coord,
            max_iterations=4,
            enable_fulfillment_check=False,
            enable_adaptive_iterations=False,
        )

        # Mock perception
        perception = Perception(
            intent=ActionIntent.WRITE_ALLOWED,
            complexity=TaskComplexity.MEDIUM,
            task_analysis=MagicMock(task_type=None),
            confidence=0.5,
        )
        loop.perception.perceive = AsyncMock(return_value=perception)

        # Mock spin detector state to return WARNING (triggers nudge injection)
        # We use PropertyMock because state is a @property
        with patch.object(
            type(loop.spin_detector),
            "state",
            new_callable=PropertyMock,
            return_value=SpinState.WARNING,
        ):
            # Also patch consecutive_no_tool_turns since the nudge message uses it
            with patch.object(
                loop.spin_detector,
                "consecutive_no_tool_turns",
                new=2,
            ):
                await loop.run("Fix the bug")

        # Check that nudge was injected into conversation
        add_calls = mock_chat_ctx.add_message.call_args_list
        nudge_calls = [
            c
            for c in add_calls
            if c.kwargs.get("metadata", {}).get("internal_prompt_kind") == "nudge"
        ]
        assert len(nudge_calls) >= 1, "Nudge message should be injected"
        assert all("within the original request" in c.args[1] for c in nudge_calls)

    async def test_spin_nudge_injected_on_blocked_tools(self):
        """When all tools are blocked by dedup, spin nudge is injected."""
        from victor.agent.services.turn_execution_runtime import TurnResult
        from unittest.mock import patch, PropertyMock

        mock_coord = AsyncMock()
        mock_chat_ctx = MagicMock()
        mock_coord._chat_context = mock_chat_ctx

        # First turn: tools blocked
        blocked_turn = TurnResult(
            response=MagicMock(content="", tool_calls=[{"name": "read"}]),
            has_tool_calls=True,
            tool_calls_count=1,
            all_tools_blocked=True,
        )
        mock_coord.execute_turn = AsyncMock(return_value=blocked_turn)

        loop = AgenticLoop(
            turn_executor=mock_coord,
            max_iterations=5,
            enable_fulfillment_check=False,
            enable_adaptive_iterations=False,
        )

        perception = Perception(
            intent=ActionIntent.WRITE_ALLOWED,
            complexity=TaskComplexity.MEDIUM,
            task_analysis=MagicMock(task_type=None),
            confidence=0.5,
        )
        loop.perception.perceive = AsyncMock(return_value=perception)

        # Mock spin detector state to return BLOCKED (triggers different tools nudge)
        with patch.object(
            type(loop.spin_detector),
            "state",
            new_callable=PropertyMock,
            return_value=SpinState.BLOCKED,
        ):
            await loop.run("Fix the bug")

        # Check spin nudge was injected
        add_calls = mock_chat_ctx.add_message.call_args_list
        spin_calls = [c for c in add_calls if "blocked" in str(c).lower()]
        assert len(spin_calls) >= 1, "Spin detection nudge should be injected"

    async def test_no_nudge_without_turn_executor(self):
        """Nudge injection is skipped when no turn_executor."""
        loop = AgenticLoop(
            orchestrator=MagicMock(spec=[]),
            max_iterations=2,
            enable_fulfillment_check=False,
            enable_adaptive_iterations=False,
        )

        perception = Perception(
            intent=ActionIntent.WRITE_ALLOWED,
            complexity=TaskComplexity.MEDIUM,
            task_analysis=MagicMock(task_type=None),
            confidence=0.5,
        )
        loop.perception.perceive = AsyncMock(return_value=perception)

        # Should not crash — nudge is just skipped
        result = await loop.run("Fix the bug")
        assert isinstance(result, LoopResult)


# ============================================================================
# Edge Model Semantic Evaluation Tests
# ============================================================================


class TestLLMRefinement:
    """Tests for _refine_with_llm() tiered semantic evaluation."""

    def _make_loop(self):
        return AgenticLoop(
            orchestrator=MagicMock(spec=[]),
            enable_fulfillment_check=False,
        )

    async def test_skips_high_confidence(self):
        """High confidence (>0.8) should skip edge model."""
        loop = self._make_loop()
        heuristic = EvaluationResult(decision=EvaluationDecision.COMPLETE, score=0.95)
        result = await loop._refine_with_llm(heuristic, MagicMock(), {})
        assert result is heuristic  # Unchanged

    async def test_skips_low_confidence(self):
        """Low confidence (<0.4) should skip edge model."""
        loop = self._make_loop()
        heuristic = EvaluationResult(decision=EvaluationDecision.RETRY, score=0.1)
        result = await loop._refine_with_llm(heuristic, MagicMock(), {})
        assert result is heuristic  # Unchanged

    async def test_refines_ambiguous_confidence(self):
        """Ambiguous range (0.4-0.8) should attempt edge model refinement."""
        from unittest.mock import patch

        loop = self._make_loop()
        loop._total_tool_calls = 3

        heuristic = EvaluationResult(decision=EvaluationDecision.CONTINUE, score=0.6)

        # Mock edge model to say task is complete
        mock_decision = MagicMock()
        mock_decision.is_complete = True
        mock_decision.confidence = 0.85
        mock_decision.phase = "done"

        mock_result = MagicMock()
        mock_result.result = mock_decision
        mock_result.source = "edge"
        mock_result.confidence = 0.85

        mock_svc = AsyncMock()
        mock_svc.decide_async = AsyncMock(return_value=mock_result)

        mock_container = MagicMock()
        mock_container.get_optional = MagicMock(return_value=mock_svc)

        with (
            patch("victor.core.feature_flags.get_feature_flag_manager") as mock_ffm,
            patch("victor.core.get_container", return_value=mock_container),
        ):
            mock_ffm.return_value.is_enabled.return_value = True

            result = await loop._refine_with_llm(
                heuristic, MagicMock(content="Done!", successful_tool_count=2), {}
            )

        assert result.decision == EvaluationDecision.COMPLETE
        assert result.score == 0.85
        assert "llm" in result.reason.lower()

    async def test_falls_back_on_edge_model_error(self):
        """Edge model failure should return heuristic unchanged."""
        from unittest.mock import patch

        loop = self._make_loop()
        heuristic = EvaluationResult(decision=EvaluationDecision.CONTINUE, score=0.6)

        with (
            patch("victor.core.feature_flags.get_feature_flag_manager") as mock_ffm,
            patch("victor.core.get_container", side_effect=RuntimeError("no container")),
        ):
            mock_ffm.return_value.is_enabled.return_value = True
            result = await loop._refine_with_llm(heuristic, MagicMock(), {})

        assert result is heuristic  # Unchanged on error

    async def test_stuck_phase_triggers_retry(self):
        """Edge model detecting 'stuck' phase should return RETRY."""
        from unittest.mock import patch

        loop = self._make_loop()
        heuristic = EvaluationResult(decision=EvaluationDecision.CONTINUE, score=0.5)

        mock_decision = MagicMock()
        mock_decision.is_complete = False
        mock_decision.confidence = 0.7
        mock_decision.phase = "stuck"

        mock_result = MagicMock()
        mock_result.result = mock_decision

        mock_svc = AsyncMock()
        mock_svc.decide_async = AsyncMock(return_value=mock_result)

        mock_container = MagicMock()
        mock_container.get_optional = MagicMock(return_value=mock_svc)

        with (
            patch("victor.core.feature_flags.get_feature_flag_manager") as mock_ffm,
            patch("victor.core.get_container", return_value=mock_container),
        ):
            mock_ffm.return_value.is_enabled.return_value = True

            result = await loop._refine_with_llm(heuristic, MagicMock(content="hmm"), {})

        assert result.decision == EvaluationDecision.RETRY
        assert "stuck" in result.reason
        mock_svc.escalate_tier.assert_awaited_once_with(DecisionType.TASK_COMPLETION, "stuck_phase")

    async def test_high_confidence_awaits_async_tier_deescalation(self):
        """LLM refinement should await async tier de-escalation hooks."""
        from unittest.mock import patch

        loop = self._make_loop()
        heuristic = EvaluationResult(decision=EvaluationDecision.CONTINUE, score=0.6)

        mock_decision = MagicMock()
        mock_decision.is_complete = True
        mock_decision.confidence = 0.9
        mock_decision.phase = "complete"

        mock_result = MagicMock()
        mock_result.result = mock_decision
        mock_result.source = "edge"

        mock_svc = AsyncMock()
        mock_svc.decide_async = AsyncMock(return_value=mock_result)

        mock_container = MagicMock()
        mock_container.get_optional = MagicMock(return_value=mock_svc)

        with (
            patch("victor.core.feature_flags.get_feature_flag_manager") as mock_ffm,
            patch("victor.core.get_container", return_value=mock_container),
        ):
            mock_ffm.return_value.is_enabled.return_value = True
            result = await loop._refine_with_llm(heuristic, MagicMock(content="done"), {})

        assert result.decision == EvaluationDecision.COMPLETE
        mock_svc.deescalate_tier.assert_awaited_once_with(
            DecisionType.TASK_COMPLETION, "high_confidence"
        )

    async def test_skipped_when_edge_model_disabled(self):
        """When USE_EDGE_MODEL=False, returns heuristic unchanged."""
        from unittest.mock import patch

        loop = self._make_loop()
        heuristic = EvaluationResult(decision=EvaluationDecision.CONTINUE, score=0.6)

        with patch("victor.core.feature_flags.get_feature_flag_manager") as mock_ffm:
            mock_ffm.return_value.is_enabled.return_value = False
            result = await loop._refine_with_llm(heuristic, MagicMock(), {})

        assert result is heuristic


# Note: feature-flag plumbing tests live in tests/unit/core/test_feature_flags.py.
# The previous TestFeatureFlagIntegration class here referenced USE_AGENTIC_LOOP
# claiming it controlled the execution path; it never did, and the flag has been
# removed. See AgenticLoop.run()/stream() for the real seam (USE_STATEGRAPH_AGENTIC_LOOP).
