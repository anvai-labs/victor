"""Tests for victor.framework.evaluation_nodes module.

Tests evaluation checkpoints, decision routing, and StateGraph extensions
for agentic loop evaluation.
"""

from __future__ import annotations


import pytest

from victor.framework.evaluation_nodes import (
    EvaluationCheckpoint,
    EvaluationDecision,
    EvaluationNode,
    EvaluationResult,
    add_evaluation,
    composite_evaluator,
    convergence_evaluator,
    create_agentic_loop_graph,
    error_count_evaluator,
    progress_tracking_evaluator,
    simple_score_evaluator,
)

# ============================================================================
# EvaluationDecision enum tests
# ============================================================================


class TestEvaluationDecision:
    """Tests for EvaluationDecision enum."""

    def test_values(self):
        assert EvaluationDecision.CONTINUE.value == "continue"
        assert EvaluationDecision.RETRY.value == "retry"
        assert EvaluationDecision.COMPLETE.value == "complete"
        assert EvaluationDecision.ESCALATE.value == "escalate"
        assert EvaluationDecision.FAIL.value == "fail"


# ============================================================================
# EvaluationResult tests
# ============================================================================


class TestEvaluationResult:
    """Tests for EvaluationResult dataclass."""

    def test_should_continue_with_enum(self):
        r = EvaluationResult(decision=EvaluationDecision.CONTINUE, score=0.5)
        assert r.should_continue is True
        assert r.should_retry is False
        assert r.should_complete is False
        assert r.should_fail is False

    def test_should_continue_with_string(self):
        r = EvaluationResult(decision="continue", score=0.5)
        assert r.should_continue is True

    def test_should_retry_with_enum(self):
        r = EvaluationResult(decision=EvaluationDecision.RETRY, score=0.2)
        assert r.should_retry is True
        assert r.should_continue is False

    def test_should_complete_with_enum(self):
        r = EvaluationResult(decision=EvaluationDecision.COMPLETE, score=1.0)
        assert r.should_complete is True

    def test_should_fail_with_enum(self):
        r = EvaluationResult(decision=EvaluationDecision.FAIL, score=0.0)
        assert r.should_fail is True

    def test_should_complete_with_string(self):
        r = EvaluationResult(decision="complete", score=1.0)
        assert r.should_complete is True

    def test_default_values(self):
        r = EvaluationResult(decision="continue")
        assert r.score == 0.5
        assert r.reason == ""
        assert r.metrics == {}
        assert r.metadata == {}


# ============================================================================
# EvaluationNode tests
# ============================================================================


class TestEvaluationNode:
    """Tests for EvaluationNode."""

    def test_init(self):
        def eval_fn(state):
            return EvaluationResult(decision="continue")

        node = EvaluationNode(
            id="test_node",
            evaluator=eval_fn,
            decision_edges={"continue": "next", "retry": "prev"},
        )
        assert node.id == "test_node"
        assert node.checkpoint is True
        assert node.decision_edges == {"continue": "next", "retry": "prev"}

    async def test_evaluate_sync_evaluator(self):
        def eval_fn(state):
            return EvaluationResult(
                decision="continue",
                score=state.get("score", 0.0),
            )

        node = EvaluationNode(
            id="test",
            evaluator=eval_fn,
            decision_edges={"continue": "next"},
        )
        result = await node.evaluate({"score": 0.8})
        assert result.decision == "continue"
        assert result.score == 0.8

    async def test_evaluate_async_evaluator(self):
        async def eval_fn(state):
            return EvaluationResult(
                decision=EvaluationDecision.COMPLETE,
                score=1.0,
            )

        node = EvaluationNode(
            id="test",
            evaluator=eval_fn,
            decision_edges={str(EvaluationDecision.COMPLETE): "end"},
        )
        result = await node.evaluate({})
        assert result.should_complete

    def test_get_next_node_string_key(self):
        node = EvaluationNode(
            id="test",
            evaluator=lambda s: None,
            decision_edges={"continue": "next", "retry": "prev"},
        )
        result = EvaluationResult(decision="continue")
        assert node.get_next_node(result) == "next"

    def test_get_next_node_enum_key(self):
        node = EvaluationNode(
            id="test",
            evaluator=lambda s: None,
            decision_edges={
                str(EvaluationDecision.CONTINUE): "next",
                str(EvaluationDecision.FAIL): "end",
            },
        )
        result = EvaluationResult(decision=EvaluationDecision.CONTINUE)
        assert node.get_next_node(result) == "next"

    def test_get_next_node_enum_value_fallback(self):
        node = EvaluationNode(
            id="test",
            evaluator=lambda s: None,
            decision_edges={"continue": "next", "retry": "prev"},
        )
        result = EvaluationResult(decision=EvaluationDecision.CONTINUE)
        assert node.get_next_node(result) == "next"

    def test_get_next_node_invalid_decision(self):
        node = EvaluationNode(
            id="test",
            evaluator=lambda s: None,
            decision_edges={"continue": "next"},
        )
        result = EvaluationResult(decision="nonexistent")
        with pytest.raises(ValueError, match="has no mapped edge"):
            node.get_next_node(result)

    def test_no_checkpoint(self):
        node = EvaluationNode(
            id="test",
            evaluator=lambda s: None,
            decision_edges={},
            checkpoint=False,
        )
        assert node.checkpoint is False


# ============================================================================
# EvaluationCheckpoint tests
# ============================================================================


class TestEvaluationCheckpoint:
    """Tests for EvaluationCheckpoint."""

    def test_to_dict(self):
        from datetime import datetime

        result = EvaluationResult(
            decision=EvaluationDecision.CONTINUE,
            score=0.75,
            reason="Progress looks good",
        )
        checkpoint = EvaluationCheckpoint(
            checkpoint_id="cp_1",
            node_id="eval_node",
            timestamp=datetime(2026, 1, 1, 12, 0, 0),
            state={"key": "value"},
            result=result,
            iteration=3,
        )
        d = checkpoint.to_dict()
        assert d["checkpoint_id"] == "cp_1"
        assert d["node_id"] == "eval_node"
        assert d["score"] == 0.75
        assert d["reason"] == "Progress looks good"
        assert d["iteration"] == 3
        assert d["state_keys"] == ["key"]


# ============================================================================
# Utility evaluator tests
# ============================================================================


class TestSimpleScoreEvaluator:
    """Tests for simple_score_evaluator factory."""

    def test_above_threshold(self):
        evaluator = simple_score_evaluator(threshold=0.7)
        result = evaluator({"score": 0.8})
        assert result.decision == "continue"
        assert result.score == 0.8

    def test_below_threshold(self):
        evaluator = simple_score_evaluator(threshold=0.7)
        result = evaluator({"score": 0.5})
        assert result.decision == "retry"
        assert result.score == 0.5

    def test_at_threshold(self):
        evaluator = simple_score_evaluator(threshold=0.7)
        result = evaluator({"score": 0.7})
        assert result.decision == "continue"

    def test_missing_score_key(self):
        evaluator = simple_score_evaluator(threshold=0.7)
        result = evaluator({})
        assert result.decision == "retry"
        assert result.score == 0.0

    def test_custom_score_key(self):
        evaluator = simple_score_evaluator(threshold=0.5, score_key="quality")
        result = evaluator({"quality": 0.9})
        assert result.decision == "continue"


class TestErrorCountEvaluator:
    """Tests for error_count_evaluator factory."""

    def test_no_errors(self):
        evaluator = error_count_evaluator(max_errors=0)
        result = evaluator({"errors": []})
        assert result.decision == "continue"
        assert result.score == 1.0

    def test_errors_within_limit(self):
        evaluator = error_count_evaluator(max_errors=2)
        result = evaluator({"errors": ["e1"]})
        assert result.decision == "continue"

    def test_errors_exceed_limit(self):
        evaluator = error_count_evaluator(max_errors=0)
        result = evaluator({"errors": ["e1", "e2"]})
        assert result.decision == "retry"
        assert result.score == 0.0

    def test_integer_error_count(self):
        evaluator = error_count_evaluator(max_errors=1)
        result = evaluator({"errors": 2})
        assert result.decision == "retry"

    def test_missing_errors_key(self):
        evaluator = error_count_evaluator(max_errors=0)
        result = evaluator({})
        assert result.decision == "continue"

    def test_custom_error_key(self):
        evaluator = error_count_evaluator(max_errors=0, error_key="warnings")
        result = evaluator({"warnings": ["w1"]})
        assert result.decision == "retry"


# ============================================================================
# add_evaluation() StateGraph integration tests
# ============================================================================


class TestAddEvaluation:
    """Tests for add_evaluation() StateGraph extension."""

    def test_add_evaluation_adds_node(self):
        from victor.framework.graph import StateGraph

        graph = StateGraph(dict)
        graph.add_node("start", lambda s: s)

        def evaluator(state):
            return EvaluationResult(decision="continue", score=0.5)

        graph = add_evaluation(
            graph,
            node_id="check",
            evaluator=evaluator,
            decision_edges={"continue": "start"},
        )

        assert "check" in graph._nodes

    def test_add_evaluation_returns_graph(self):
        from victor.framework.graph import StateGraph

        graph = StateGraph(dict)
        result = add_evaluation(
            graph,
            node_id="eval",
            evaluator=lambda s: EvaluationResult(decision="continue"),
            decision_edges={"continue": "__end__"},
        )
        assert result is graph


# ============================================================================
# create_agentic_loop_graph() tests
# ============================================================================


class TestCreateAgenticLoopGraph:
    """Tests for create_agentic_loop_graph factory."""

    def test_creates_graph_with_all_nodes(self):
        async def perceive(s):
            return s

        async def plan(s):
            return s

        async def act(s):
            return s

        def evaluate(s):
            return EvaluationResult(decision=EvaluationDecision.COMPLETE)

        graph = create_agentic_loop_graph(
            state_type=dict,
            perception_fn=perceive,
            planning_fn=plan,
            execution_fn=act,
            evaluator_fn=evaluate,
        )
        assert "perceive" in graph._nodes
        assert "plan" in graph._nodes
        assert "act" in graph._nodes
        assert "evaluate" in graph._nodes

    def test_graph_has_entry_point(self):
        graph = create_agentic_loop_graph(
            state_type=dict,
            perception_fn=lambda s: s,
            planning_fn=lambda s: s,
            execution_fn=lambda s: s,
            evaluator_fn=lambda s: EvaluationResult(decision="complete"),
        )
        assert graph._entry_point == "perceive"


# ============================================================================
# Progress tracking evaluator tests
# ============================================================================


class TestProgressTrackingEvaluator:
    """Tests for progress_tracking_evaluator."""

    def test_first_iteration_continues(self):
        evaluator = progress_tracking_evaluator(complete_threshold=0.9)
        result = evaluator({"score": 0.3})
        assert result.decision == EvaluationDecision.CONTINUE

    def test_completes_at_threshold(self):
        evaluator = progress_tracking_evaluator(complete_threshold=0.9)
        result = evaluator({"score": 0.95})
        assert result.decision == EvaluationDecision.COMPLETE

    def test_detects_progress(self):
        evaluator = progress_tracking_evaluator(complete_threshold=0.9)
        evaluator({"score": 0.3})
        result = evaluator({"score": 0.5})
        assert result.decision == EvaluationDecision.CONTINUE
        assert "Progress" in result.reason

    def test_detects_regression(self):
        evaluator = progress_tracking_evaluator(complete_threshold=0.9)
        evaluator({"score": 0.5})
        result = evaluator({"score": 0.4})
        assert result.decision == EvaluationDecision.RETRY

    def test_detects_plateau(self):
        evaluator = progress_tracking_evaluator(
            complete_threshold=0.9,
            plateau_window=3,
            plateau_tolerance=0.02,
        )
        evaluator({"score": 0.5})
        evaluator({"score": 0.505})
        result = evaluator({"score": 0.51})
        assert result.decision == EvaluationDecision.FAIL
        assert "plateau" in result.reason.lower()

    def test_tracks_history_in_metrics(self):
        evaluator = progress_tracking_evaluator()
        evaluator({"score": 0.2})
        result = evaluator({"score": 0.4})
        assert result.metrics["progress_history"] == [0.2, 0.4]


# ============================================================================
# Composite evaluator tests
# ============================================================================


class TestCompositeEvaluator:
    """Tests for composite_evaluator."""

    def test_worst_strategy(self):
        evaluator = composite_evaluator(
            [
                simple_score_evaluator(threshold=0.5),
                error_count_evaluator(max_errors=0),
            ],
            strategy="worst",
        )
        result = evaluator({"score": 0.8, "errors": ["e1"]})
        assert result.score == 0.0  # Error evaluator returns 0.0

    def test_best_strategy(self):
        evaluator = composite_evaluator(
            [
                simple_score_evaluator(threshold=0.9),
                error_count_evaluator(max_errors=0),
            ],
            strategy="best",
        )
        result = evaluator({"score": 0.8, "errors": []})
        assert result.score == 1.0  # Error evaluator returns 1.0

    def test_average_strategy(self):
        evaluator = composite_evaluator(
            [
                simple_score_evaluator(threshold=0.5),
                simple_score_evaluator(threshold=0.5),
            ],
            strategy="average",
        )
        result = evaluator({"score": 0.8})
        assert result.score == 0.8

    def test_empty_evaluators(self):
        evaluator = composite_evaluator([], strategy="worst")
        result = evaluator({})
        assert result.decision == EvaluationDecision.CONTINUE


# ============================================================================
# Convergence evaluator tests
# ============================================================================


class TestConvergenceEvaluator:
    """Tests for convergence_evaluator."""

    def test_continues_below_min_iterations(self):
        evaluator = convergence_evaluator(min_iterations=3)
        evaluator({"score": 0.7})
        result = evaluator({"score": 0.7})
        assert result.decision == EvaluationDecision.CONTINUE

    def test_converges_above_min_score(self):
        evaluator = convergence_evaluator(
            min_iterations=2,
            convergence_threshold=0.01,
            min_score=0.6,
        )
        evaluator({"score": 0.75})
        result = evaluator({"score": 0.755})
        assert result.decision == EvaluationDecision.COMPLETE

    def test_fails_converged_below_min_score(self):
        evaluator = convergence_evaluator(
            min_iterations=2,
            convergence_threshold=0.01,
            min_score=0.8,
        )
        evaluator({"score": 0.5})
        result = evaluator({"score": 0.505})
        assert result.decision == EvaluationDecision.FAIL

    def test_continues_when_not_converged(self):
        evaluator = convergence_evaluator(
            min_iterations=2,
            convergence_threshold=0.01,
        )
        evaluator({"score": 0.5})
        result = evaluator({"score": 0.7})
        assert result.decision == EvaluationDecision.CONTINUE
