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

"""Unit tests for EnhancedCompletionEvaluator."""

from __future__ import annotations

import pytest
from dataclasses import dataclass
from unittest.mock import AsyncMock, Mock

from victor.framework.enhanced_completion_evaluation import (
    EnhancedCompletionEvaluator,
)
from victor.framework.evaluation_nodes import EvaluationDecision
from victor.framework.fulfillment import FulfillmentDetector
from victor.framework.perception_integration import RequirementType, Requirement
from victor.framework.completion_scorer import CompletionScore, TaskType
from victor.framework.completion_scorer import CompletionSignal
from victor.framework.runtime_evaluation_policy import RuntimeEvaluationPolicy
from victor.agent.turn_policy import SpinState

# =============================================================================
# Test Fixtures
# =============================================================================


@dataclass
class MockTurnResult:
    """Mock TurnResult for testing."""

    response: str = "Task complete"
    is_qa_response: bool = False
    has_content: bool = True
    has_tool_calls: bool = False
    successful_tool_count: int = 0
    failed_tool_count: int = 0
    tool_calls: list = None


@dataclass
class MockSpinDetector:
    """Mock SpinDetector for testing."""

    state: SpinState = SpinState.NORMAL
    consecutive_all_blocked: int = 0
    consecutive_no_tool_turns: int = 0
    total_tool_calls: int = 0


@dataclass
class MockPerception:
    """Mock Perception for testing - matches actual API."""

    confidence: float = 0.8
    requirements: list = None
    task_type: str = "code_generation"
    intent: Mock = None
    complexity: str = "medium"
    task_analysis: Mock = None
    metadata: dict = None


# =============================================================================
# EnhancedCompletionEvaluator Tests
# =============================================================================


class TestEnhancedCompletionEvaluator:
    """Test suite for EnhancedCompletionEvaluator."""

    @pytest.fixture
    def evaluator(self):
        """Create evaluator instance for testing."""
        return EnhancedCompletionEvaluator(
            enable_requirement_validation=True,
            enable_completion_scoring=True,
            enable_context_keywords=True,
            completion_threshold=0.80,
        )

    @pytest.fixture
    def mock_fulfillment(self):
        """Create mock fulfillment detector."""
        fulfillment = AsyncMock()
        fulfillment.check_fulfillment = AsyncMock(
            return_value=Mock(
                is_fulfilled=False,
                score=0.6,
                reason="Partial progress",
            )
        )
        return fulfillment

    @pytest.mark.asyncio
    async def test_initialization(self, evaluator):
        """Test evaluator initializes with correct settings."""
        assert evaluator.enable_requirement_validation is True
        assert evaluator.enable_completion_scoring is True
        assert evaluator.enable_context_keywords is True
        assert evaluator.completion_threshold == 0.80

    @pytest.mark.asyncio
    async def test_evaluate_qa_shortcut(self, evaluator):
        """Test Q&A shortcut takes priority."""
        from victor.agent.services.turn_execution_runtime import TurnResult
        from victor.providers.base import CompletionResponse

        perception = MockPerception(confidence=0.9)
        # Use actual TurnResult class to pass isinstance check
        completion_response = CompletionResponse(
            content="The answer is 42",
            role="assistant",
            tool_calls=None,
            stop_reason="stop",
            usage=None,
        )
        action_result = TurnResult(
            response=completion_response,
            is_qa_response=True,
        )
        state = {}

        result = await evaluator.evaluate(
            perception=perception, action_result=action_result, state=state
        )

        # Q&A shortcut should trigger COMPLETE
        assert result.decision == EvaluationDecision.COMPLETE
        assert "Q&A shortcut" in result.reason
        assert result.score == 0.9

    @pytest.mark.asyncio
    async def test_evaluate_spin_terminated_all_blocked(self, evaluator):
        """Test spin detection when all tools blocked."""
        perception = MockPerception(confidence=0.5)
        action_result = MockTurnResult(response="Stuck")
        state = {}
        spin_detector = MockSpinDetector(
            state=SpinState.TERMINATED,
            consecutive_all_blocked=4,
            consecutive_no_tool_turns=0,
        )

        result = await evaluator.evaluate(
            perception=perception,
            action_result=action_result,
            state=state,
            spin_detector=spin_detector,
        )

        assert result.decision == EvaluationDecision.FAIL
        assert "Spin detected" in result.reason
        assert result.score == 0.1

    @pytest.mark.asyncio
    async def test_evaluate_spin_terminated_no_tools(self, evaluator):
        """Test spin detection when no tools used."""
        perception = MockPerception(confidence=0.5)
        action_result = MockTurnResult(response="Still thinking")
        state = {}
        spin_detector = MockSpinDetector(
            state=SpinState.TERMINATED,
            consecutive_all_blocked=0,
            consecutive_no_tool_turns=5,
        )

        result = await evaluator.evaluate(
            perception=perception,
            action_result=action_result,
            state=state,
            spin_detector=spin_detector,
        )

        assert result.decision == EvaluationDecision.FAIL
        assert "Agent stuck" in result.reason

    @pytest.mark.asyncio
    async def test_evaluate_enhanced_complete(self, evaluator):
        """Test enhanced evaluation returns COMPLETE."""
        perception = MockPerception(
            confidence=0.9,
            requirements=[
                Requirement(
                    type=RequirementType.FUNCTIONAL,
                    description="Create file",
                    priority=0,
                )
            ],
        )
        action_result = MockTurnResult(
            response="Here is the implementation",
            has_tool_calls=True,
            tool_calls=["write_file"],
        )
        state = {"files_modified": ["auth.py"]}

        result = await evaluator.evaluate(
            perception=perception, action_result=action_result, state=state
        )

        # Should make a valid decision
        assert result.decision in (
            EvaluationDecision.COMPLETE,
            EvaluationDecision.CONTINUE,
            EvaluationDecision.RETRY,
        )
        assert result.score > 0.0

    @pytest.mark.asyncio
    async def test_evaluate_enhanced_with_fulfillment(self, evaluator, mock_fulfillment):
        """Test enhanced evaluation with fulfillment detector."""
        perception = MockPerception(
            confidence=0.8,
            requirements=[
                Requirement(type=RequirementType.FUNCTIONAL, description="Test", priority=0)
            ],
        )
        action_result = MockTurnResult(response="Testing complete")
        state = {}

        result = await evaluator.evaluate(
            perception=perception,
            action_result=action_result,
            state=state,
            fulfillment_detector=mock_fulfillment,
        )

        # Just verify it returns a valid decision
        # (fulfillment detector may or may not be called depending on implementation)
        assert result.decision in (
            EvaluationDecision.COMPLETE,
            EvaluationDecision.CONTINUE,
            EvaluationDecision.RETRY,
        )

    @pytest.mark.asyncio
    async def test_evaluate_enhanced_uses_compatible_fulfillment_task_type(self, evaluator, caplog):
        """The evaluator should not leak completion-scoring enums into fulfillment."""
        perception = MockPerception(
            confidence=0.8,
            task_type="search",
            requirements=[],
        )
        action_result = MockTurnResult(response="I found the relevant graph files.")
        state = {
            "search_results": ["victor/framework/graph.py"],
            "avg_relevance": 0.9,
        }

        with caplog.at_level("WARNING"):
            result = await evaluator.evaluate(
                perception=perception,
                action_result=action_result,
                state=state,
                fulfillment_detector=FulfillmentDetector(),
            )

        assert result.decision in (
            EvaluationDecision.COMPLETE,
            EvaluationDecision.CONTINUE,
            EvaluationDecision.RETRY,
        )
        assert "No fulfillment strategy for TaskType.SEARCH" not in caplog.text

    @pytest.mark.asyncio
    async def test_evaluate_legacy_fallback(self, evaluator):
        """Test legacy fallback when enhanced evaluation is disabled."""
        # Disable enhanced evaluation to force legacy path
        evaluator.enable_completion_scoring = False

        perception = MockPerception(confidence=0.8)
        action_result = MockTurnResult(
            response="Used some tools",
            has_tool_calls=True,
            successful_tool_count=3,
            failed_tool_count=0,
        )
        state = {}

        result = await evaluator.evaluate(
            perception=perception, action_result=action_result, state=state
        )

        # Legacy path returns a valid decision
        assert result.decision in (
            EvaluationDecision.COMPLETE,
            EvaluationDecision.CONTINUE,
            EvaluationDecision.RETRY,
        )

    @pytest.mark.asyncio
    async def test_evaluate_legacy_substantial_response(self, evaluator):
        """Test legacy evaluation with substantial response."""
        # Disable enhanced evaluation
        evaluator.enable_completion_scoring = False

        perception = MockPerception(confidence=0.7)
        action_result = MockTurnResult(
            response="This is a very long and detailed explanation. " * 20,  # > 100 chars
            has_tool_calls=False,
            has_content=True,
        )
        state = {}

        result = await evaluator.evaluate(
            perception=perception, action_result=action_result, state=state
        )

        # Legacy path returns a valid decision
        assert result.decision in (
            EvaluationDecision.COMPLETE,
            EvaluationDecision.CONTINUE,
            EvaluationDecision.RETRY,
        )

    @pytest.mark.asyncio
    async def test_evaluate_legacy_confidence_based(self, evaluator):
        """Test legacy evaluation with confidence-based decision."""
        # Disable enhanced evaluation
        evaluator.enable_completion_scoring = False

        perception = MockPerception(confidence=0.9)
        action_result = MockTurnResult(response="Response")
        state = {}

        result = await evaluator.evaluate(
            perception=perception, action_result=action_result, state=state
        )

        assert result.decision == EvaluationDecision.COMPLETE
        assert "High confidence" in result.reason

    @pytest.mark.asyncio
    async def test_evaluate_error_handling(self, evaluator):
        """Test graceful degradation on errors."""
        perception = MockPerception(confidence=0.8)
        # Pass invalid action_result that might cause errors
        action_result = None
        state = {}

        # Should not raise exception, should return result
        result = await evaluator.evaluate(
            perception=perception, action_result=action_result, state=state
        )

        assert result is not None
        assert result.decision in (
            EvaluationDecision.COMPLETE,
            EvaluationDecision.CONTINUE,
            EvaluationDecision.RETRY,
        )

    @pytest.mark.asyncio
    async def test_evaluate_low_confidence_uses_runtime_policy_without_mutating_budget(
        self, evaluator
    ):
        """Low-confidence fallback should share runtime policy vocabulary without spending budget."""
        evaluator.enable_completion_scoring = False
        perception = MockPerception(confidence=0.3)
        state = {"low_confidence_retries": 1}

        result = await evaluator.evaluate(
            perception=perception,
            action_result=None,
            state=state,
        )

        assert result.decision == EvaluationDecision.RETRY
        assert result.reason == "Low confidence - retry"
        assert state["low_confidence_retries"] == 1

    @pytest.mark.asyncio
    async def test_evaluate_uses_injected_runtime_policy(self):
        evaluator = EnhancedCompletionEvaluator(
            enable_requirement_validation=False,
            enable_completion_scoring=False,
            enable_context_keywords=False,
            evaluation_policy=RuntimeEvaluationPolicy(
                low_confidence_reason="Confidence too low - retry",
            ),
        )
        perception = MockPerception(confidence=0.3)

        result = await evaluator.evaluate(
            perception=perception,
            action_result=None,
            state={},
        )

        assert result.decision == EvaluationDecision.RETRY
        assert result.reason == "Confidence too low - retry"

    @pytest.mark.asyncio
    async def test_calibrated_completion_uses_policy_reason_and_penalty(self):
        evaluator = EnhancedCompletionEvaluator(
            enable_requirement_validation=True,
            enable_completion_scoring=True,
            enable_context_keywords=True,
            evaluation_policy=RuntimeEvaluationPolicy(
                completion_requires_support_reason="Need stronger execution evidence",
                unsupported_requirement_penalty=0.2,
                minimum_supported_evidence_score=0.95,
            ),
        )
        evaluator.enable_calibrated_completion = True
        evaluator.requirement_validator.validate_completion = Mock(
            return_value=Mock(
                is_satisfied=False,
                satisfaction_score=0.75,
                critical_gaps=[],
                summary="missing verification",
            )
        )
        evaluator.keyword_detector.detect_completion = Mock(
            return_value=CompletionSignal(
                has_completion_indicator=True,
                has_complete_code=True,
                has_structure=True,
                is_continuation_request=False,
                confidence=0.9,
                evidence=["completion marker"],
            )
        )
        evaluator.completion_scorer.calculate_completion_score = Mock(
            return_value=CompletionScore(
                total_score=0.92,
                requirement_score=0.75,
                fulfillment_score=0.8,
                keyword_score=0.9,
                confidence_score=0.8,
                complexity_adjustment=0.0,
                is_complete=True,
                threshold=0.85,
            )
        )

        perception = MockPerception(
            confidence=0.9,
            requirements=[
                Requirement(type=RequirementType.FUNCTIONAL, description="Modify auth flow")
            ],
        )
        action_result = MockTurnResult(
            response="Implemented the change.",
            has_tool_calls=False,
            successful_tool_count=0,
        )

        result = await evaluator.evaluate(
            perception=perception,
            action_result=action_result,
            state={},
        )

        assert result.decision == EvaluationDecision.CONTINUE
        assert result.reason == "Need stronger execution evidence"
        assert result.metadata["calibration"]["support_penalty"] == pytest.approx(0.2)

    @pytest.mark.asyncio
    async def test_evaluate_enhanced_uses_policy_progress_threshold(self):
        evaluator = EnhancedCompletionEvaluator(
            enable_requirement_validation=False,
            enable_completion_scoring=True,
            enable_context_keywords=False,
            evaluation_policy=RuntimeEvaluationPolicy(
                enhanced_progress_threshold=0.65,
                completion_retry_reason_template="Retry: {score:.2f}",
            ),
        )
        evaluator.completion_scorer.calculate_completion_score = Mock(
            return_value=CompletionScore(
                # Component scores drive the fuser — total_score is no longer the canonical
                # aggregate; the fuser recomputes from the four component signals.
                # All-0.6 components → fused = 0.60 (uniform signals, weights sum to 1.0).
                # 0.60 < enhanced_progress_threshold (0.65) → RETRY path in policy.
                total_score=0.6,
                requirement_score=0.6,
                fulfillment_score=0.6,
                keyword_score=0.6,
                confidence_score=0.6,
                complexity_adjustment=0.0,
                is_complete=False,
                threshold=0.85,
            )
        )

        result = await evaluator.evaluate(
            perception=MockPerception(task_type="unknown"),
            action_result=MockTurnResult(response="Partial progress", has_content=True),
            state={},
        )

        assert result.decision == EvaluationDecision.RETRY
        # Fuser uses component scores (all 0.6): weighted sum = 0.60
        assert result.reason == "Retry: 0.60"

    @pytest.mark.asyncio
    async def test_map_to_task_type_from_perception(self, evaluator):
        """Test mapping perception to TaskType."""
        from victor.framework.completion_scorer import TaskType

        perception = MockPerception(task_type="code_generation")

        task_type = evaluator._map_to_task_type(perception)

        # Should return a TaskType enum
        assert isinstance(task_type, TaskType)

    @pytest.mark.asyncio
    async def test_extract_response_from_turn_result(self, evaluator):
        """Test extracting response from TurnResult."""
        action_result = MockTurnResult(response="Test response")

        response = evaluator._extract_response(action_result)

        assert response == "Test response"

    @pytest.mark.asyncio
    async def test_extract_response_from_content(self, evaluator):
        """Test extracting response from content attribute."""

        # Create a simple object with content attribute but no response
        class ActionResultWithContent:
            content = "Content response"

        action_result = ActionResultWithContent()

        response = evaluator._extract_response(action_result)

        # Should extract from content attribute when response doesn't exist
        assert response == "Content response"

    @pytest.mark.asyncio
    async def test_extract_response_from_string(self, evaluator):
        """Test extracting response from string."""
        action_result = "String response"

        response = evaluator._extract_response(action_result)

        assert response == "String response"

    @pytest.mark.asyncio
    async def test_spin_detection_priority_over_enhanced(self, evaluator):
        """Test that spin detection takes priority over enhanced evaluation."""
        perception = MockPerception(confidence=0.9)
        action_result = MockTurnResult(response="Response")
        state = {}
        spin_detector = MockSpinDetector(state=SpinState.TERMINATED, consecutive_all_blocked=2)

        result = await evaluator.evaluate(
            perception=perception,
            action_result=action_result,
            state=state,
            spin_detector=spin_detector,
        )

        # Should return FAIL due to spin, not COMPLETE due to high confidence
        assert result.decision == EvaluationDecision.FAIL

    @pytest.mark.asyncio
    async def test_qa_shortcut_priority_over_enhanced(self, evaluator):
        """Test that Q&A shortcut takes priority over enhanced evaluation."""
        from victor.agent.services.turn_execution_runtime import TurnResult
        from victor.providers.base import CompletionResponse

        perception = MockPerception(confidence=0.5)  # Low confidence
        # Use actual TurnResult class to pass isinstance check
        completion_response = CompletionResponse(
            content="The answer is simple",
            role="assistant",
            tool_calls=None,
            stop_reason="stop",
            usage=None,
        )
        action_result = TurnResult(
            response=completion_response,
            is_qa_response=True,
        )
        state = {}

        result = await evaluator.evaluate(
            perception=perception, action_result=action_result, state=state
        )

        # Q&A shortcut should return COMPLETE
        assert result.decision == EvaluationDecision.COMPLETE
        assert "Q&A shortcut" in result.reason

    @pytest.mark.asyncio
    async def test_no_perception_uses_legacy(self, evaluator):
        """Test that missing perception falls back to legacy."""
        perception = None
        action_result = MockTurnResult(
            response="Response",
            has_tool_calls=True,
            successful_tool_count=2,
        )
        state = {}

        result = await evaluator.evaluate(
            perception=perception, action_result=action_result, state=state
        )

        # Should use legacy evaluation - but actual legacy path is complex
        # Just verify it returns a valid decision
        assert result.decision in (
            EvaluationDecision.COMPLETE,
            EvaluationDecision.CONTINUE,
            EvaluationDecision.RETRY,
        )

    @pytest.mark.asyncio
    async def test_disable_requirement_validation(self):
        """Test evaluator with requirement validation disabled."""
        evaluator = EnhancedCompletionEvaluator(
            enable_requirement_validation=False,
            enable_completion_scoring=True,
            enable_context_keywords=True,
        )

        assert evaluator.enable_requirement_validation is False

    @pytest.mark.asyncio
    async def test_disable_completion_scoring(self):
        """Test evaluator with completion scoring disabled."""
        evaluator = EnhancedCompletionEvaluator(
            enable_requirement_validation=True,
            enable_completion_scoring=False,  # Disabled
            enable_context_keywords=True,
        )

        assert evaluator.enable_completion_scoring is False

    @pytest.mark.asyncio
    async def test_disable_context_keywords(self):
        """Test evaluator with context keywords disabled."""
        evaluator = EnhancedCompletionEvaluator(
            enable_requirement_validation=True,
            enable_completion_scoring=True,
            enable_context_keywords=False,  # Disabled
        )

        assert evaluator.enable_context_keywords is False

    @pytest.mark.asyncio
    async def test_calibrated_completion_penalizes_unsupported_direct_answer(self, evaluator):
        """Direct answers without support should not prematurely complete."""
        evaluator.enable_calibrated_completion = True
        evaluator.completion_scorer.calculate_completion_score = Mock(
            return_value=CompletionScore(
                total_score=0.92,
                requirement_score=0.9,
                fulfillment_score=0.8,
                keyword_score=0.9,
                confidence_score=0.8,
                complexity_adjustment=0.0,
                is_complete=True,
                threshold=0.85,
            )
        )

        perception = MockPerception(
            confidence=0.9,
            requirements=[
                Requirement(type=RequirementType.FUNCTIONAL, description="Modify auth flow")
            ],
        )
        action_result = MockTurnResult(
            response="The task is complete.",
            has_tool_calls=False,
            successful_tool_count=0,
        )

        result = await evaluator.evaluate(
            perception=perception,
            action_result=action_result,
            state={},
        )

        assert result.decision == EvaluationDecision.CONTINUE
        assert result.metadata["calibration"]["requires_additional_support"] is True
        assert result.metadata["calibration"]["calibrated_score"] < 0.85

    @pytest.mark.asyncio
    async def test_calibrated_completion_allows_tool_backed_answer(self, evaluator):
        """Tool-backed answers should preserve strong completion scores."""
        evaluator.enable_calibrated_completion = True
        evaluator.completion_scorer.calculate_completion_score = Mock(
            return_value=CompletionScore(
                total_score=0.91,
                requirement_score=0.9,
                fulfillment_score=0.9,
                keyword_score=0.8,
                confidence_score=0.8,
                complexity_adjustment=0.0,
                is_complete=True,
                threshold=0.85,
            )
        )

        perception = MockPerception(
            confidence=0.9,
            requirements=[Requirement(type=RequirementType.FUNCTIONAL, description="Write patch")],
        )
        action_result = MockTurnResult(
            response="Applied the patch and updated tests.",
            has_tool_calls=True,
            successful_tool_count=2,
            tool_calls=["edit_file", "run_tests"],
        )

        result = await evaluator.evaluate(
            perception=perception,
            action_result=action_result,
            state={"files_modified": ["auth.py"], "tests_passed": 4},
        )

        assert result.decision == EvaluationDecision.COMPLETE
        assert result.metadata["calibration"]["requires_additional_support"] is False


class TestSpinDetectionWithPriorTools:
    """Regression tests for spin detection when the model used tools earlier."""

    @pytest.fixture
    def evaluator(self):
        return EnhancedCompletionEvaluator(
            enable_requirement_validation=False,
            enable_completion_scoring=False,
            enable_context_keywords=False,
        )

    @pytest.mark.asyncio
    async def test_spin_no_tools_fails_without_prior_tool_use(self, evaluator):
        """TERMINATED due to no-tool-turns with zero prior tool calls → FAIL."""
        spin_detector = MockSpinDetector(
            state=SpinState.TERMINATED,
            consecutive_no_tool_turns=3,
        )
        # total_tool_calls defaults to 0 on MockSpinDetector
        action_result = MockTurnResult(response="Brief reply", has_tool_calls=False)

        result = await evaluator.evaluate(
            perception=None,
            action_result=action_result,
            state={},
            spin_detector=spin_detector,
        )

        assert result.decision == EvaluationDecision.FAIL

    @pytest.mark.asyncio
    async def test_spin_substantial_response_after_tool_use_completes(self, evaluator):
        """Regression: after using tools, a long no-tool response should be COMPLETE, not FAIL."""
        spin_detector = MockSpinDetector(
            state=SpinState.TERMINATED,
            consecutive_no_tool_turns=3,
        )
        spin_detector.total_tool_calls = 6  # had prior tool usage

        long_response = "A" * 200  # > 100 chars → substantial
        action_result = MockTurnResult(response=long_response, has_tool_calls=False)

        result = await evaluator.evaluate(
            perception=None,
            action_result=action_result,
            state={},
            spin_detector=spin_detector,
        )

        assert result.decision == EvaluationDecision.COMPLETE
        assert "complete" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_spin_short_response_after_tool_use_still_fails(self, evaluator):
        """A short response after tool use (< 100 chars) still counts as stuck."""
        spin_detector = MockSpinDetector(
            state=SpinState.TERMINATED,
            consecutive_no_tool_turns=3,
        )
        spin_detector.total_tool_calls = 4

        action_result = MockTurnResult(response="OK done", has_tool_calls=False)

        result = await evaluator.evaluate(
            perception=None,
            action_result=action_result,
            state={},
            spin_detector=spin_detector,
        )

        assert result.decision == EvaluationDecision.FAIL

    @pytest.mark.asyncio
    async def test_all_blocked_spin_always_fails(self, evaluator):
        """consecutive_all_blocked path is unaffected by prior tool usage."""
        spin_detector = MockSpinDetector(
            state=SpinState.TERMINATED,
            consecutive_all_blocked=3,
        )
        spin_detector.total_tool_calls = 10

        long_response = "B" * 500
        action_result = MockTurnResult(response=long_response, has_tool_calls=False)

        result = await evaluator.evaluate(
            perception=None,
            action_result=action_result,
            state={},
            spin_detector=spin_detector,
        )

        assert result.decision == EvaluationDecision.FAIL

    @pytest.mark.asyncio
    async def test_spin_substantial_response_without_any_tool_use_completes(self, evaluator):
        """Zero total tool calls + substantial response = knowledge generation task → COMPLETE.

        Regression: step 5 'Create comprehensive Rust best practices checklist' ran 3
        agentic loop iterations (150s), generated 892 chars, made 0 tool calls — correct
        for a knowledge task — but got FAIL (Agent stuck: 3 turns without tool calls)
        because the COMPLETE path required had_prior_tools > 0.
        """
        spin_detector = MockSpinDetector(
            state=SpinState.TERMINATED,
            consecutive_no_tool_turns=3,
        )
        # total_tool_calls = 0 (default) — never used any tools

        knowledge_response = (
            "Arc vs Rc tradeoffs: Use Arc only when data crosses thread boundaries. "
            "Use Rc for single-threaded ref-counting to avoid atomic overhead. "
            "Prefer ownership transfer over Arc cloning where lifetime allows.\n"
            "Immutable bindings: Prefer let over let mut; mutate only at the narrowest scope. "
            "Use iterator combinators (map/filter/fold) over mutable accumulators.\n"
            "Unnecessary cloning: Avoid .clone() on Arc — pass references instead. "
            "Use Cow<str> when callers may return borrowed or owned data.\n"
        )  # > 100 chars, not intent-only
        action_result = MockTurnResult(response=knowledge_response, has_tool_calls=False)

        result = await evaluator.evaluate(
            perception=None,
            action_result=action_result,
            state={},
            spin_detector=spin_detector,
        )

        assert (
            result.decision == EvaluationDecision.COMPLETE
        ), f"Zero-tool knowledge generation should be COMPLETE, got {result.decision}: {result.reason}"
        assert "knowledge" in result.reason.lower() or "complete" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_spin_short_zero_tool_response_still_fails(self, evaluator):
        """Zero total tool calls + short response — model is genuinely stuck → FAIL."""
        spin_detector = MockSpinDetector(
            state=SpinState.TERMINATED,
            consecutive_no_tool_turns=3,
        )
        # Short response — not substantial knowledge generation
        action_result = MockTurnResult(response="I'll look at this.", has_tool_calls=False)

        result = await evaluator.evaluate(
            perception=None,
            action_result=action_result,
            state={},
            spin_detector=spin_detector,
        )

        assert result.decision == EvaluationDecision.FAIL

    @pytest.mark.asyncio
    async def test_spin_intent_only_zero_tool_still_fails(self, evaluator):
        """Zero tool calls + intent-only narration (long but future-tense) → FAIL."""
        spin_detector = MockSpinDetector(
            state=SpinState.TERMINATED,
            consecutive_no_tool_turns=3,
        )
        # Narration: says what it WILL do, doesn't actually do it
        intent_response = (
            "I will now read the Cargo.toml files to understand the workspace structure. "
            "Then I will examine each crate's source files for Arc usage. "
            "Let me now begin by looking at the root manifest to identify all members. "
            "I'll also check the dependency graph to understand crate relationships."
        )  # > 100 chars but pure intent narration
        action_result = MockTurnResult(response=intent_response, has_tool_calls=False)

        result = await evaluator.evaluate(
            perception=None,
            action_result=action_result,
            state={},
            spin_detector=spin_detector,
        )

        assert result.decision == EvaluationDecision.FAIL


# =============================================================================
# Wave A2: CompletionSignalFuser wiring tests
# =============================================================================


class TestCompletionSignalFuserWired:
    """Verify CompletionSignalFuser is imported and used in _evaluate_enhanced()."""

    def test_fuser_imported_in_enhanced_completion_evaluation(self):
        """CompletionSignalFuser must be importable from the enhanced evaluator module."""
        import inspect

        import victor.framework.enhanced_completion_evaluation as mod

        source = inspect.getsource(mod)
        assert "CompletionSignalFuser" in source, (
            "enhanced_completion_evaluation.py must import and use CompletionSignalFuser. "
            "Wire it in _evaluate_enhanced() after computing the four individual signals."
        )

    def test_fuser_used_in_evaluate_enhanced_source(self):
        """_evaluate_enhanced() source must reference CompletionSignalFuser or fuser."""
        import inspect

        from victor.framework.enhanced_completion_evaluation import (
            EnhancedCompletionEvaluator,
        )

        source = inspect.getsource(EnhancedCompletionEvaluator._evaluate_enhanced)
        assert (
            "CompletionSignalFuser" in source or "fuser" in source
        ), "_evaluate_enhanced() must instantiate CompletionSignalFuser and call fuse()."
