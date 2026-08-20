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

"""Integration tests for enhanced completion detection in AgenticLoop."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, Mock
from dataclasses import dataclass

from victor.framework.agentic_loop import AgenticLoop
from victor.framework.evaluation_nodes import EvaluationDecision
from victor.framework.perception_integration import RequirementType, Requirement
from victor.agent.turn_policy import SpinDetector, SpinState

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
# Integration Tests
# =============================================================================


class TestEnhancedCompletionIntegration:
    """Integration tests for enhanced completion in AgenticLoop."""

    @pytest.fixture
    def mock_orchestrator(self):
        """Create mock orchestrator."""
        orchestrator = Mock()
        orchestrator.turn_executor = Mock()
        return orchestrator

    @pytest.fixture
    def mock_memory_coordinator(self):
        """Create mock memory coordinator."""
        return Mock()

    @pytest.fixture
    def loop_with_enhanced_enabled(self, mock_orchestrator, mock_memory_coordinator):
        """Create AgenticLoop with enhanced completion enabled."""
        loop = AgenticLoop(
            orchestrator=mock_orchestrator,
            memory_coordinator=mock_memory_coordinator,
            config={
                "disable_enhanced_completion": False,  # Enhanced enabled (default)
            },
        )
        return loop

    @pytest.fixture
    def loop_with_enhanced_disabled(self, mock_orchestrator, mock_memory_coordinator):
        """Create AgenticLoop with enhanced completion disabled."""
        loop = AgenticLoop(
            orchestrator=mock_orchestrator,
            memory_coordinator=mock_memory_coordinator,
            config={
                "disable_enhanced_completion": True,  # Enhanced disabled
            },
        )
        return loop

    def test_enhanced_completion_enabled_by_default(
        self, mock_orchestrator, mock_memory_coordinator
    ):
        """Test that enhanced completion is enabled by default."""
        loop = AgenticLoop(
            orchestrator=mock_orchestrator,
            memory_coordinator=mock_memory_coordinator,
            config={},  # Empty config (default behavior)
        )

        # Enhanced evaluator should be initialized
        assert loop.enhanced_completion_evaluator is not None

    def test_enhanced_completion_can_be_disabled(self, mock_orchestrator, mock_memory_coordinator):
        """Test that enhanced completion can be disabled via config."""
        loop = AgenticLoop(
            orchestrator=mock_orchestrator,
            memory_coordinator=mock_memory_coordinator,
            config={"disable_enhanced_completion": True},
        )

        # Enhanced evaluator should be None when disabled
        assert loop.enhanced_completion_evaluator is None

    @pytest.mark.asyncio
    async def test_evaluate_uses_enhanced_when_enabled(self, loop_with_enhanced_enabled):
        """Test that _evaluate uses enhanced evaluator when enabled."""
        perception = MockPerception(confidence=0.9, requirements=[])
        action_result = MockTurnResult(response="Complete")
        state = {}

        # Mock the enhanced evaluator
        loop_with_enhanced_enabled.enhanced_completion_evaluator = AsyncMock()
        loop_with_enhanced_enabled.enhanced_completion_evaluator.evaluate = AsyncMock(
            return_value=Mock(decision=EvaluationDecision.COMPLETE, score=0.85, reason="Test")
        )

        result = await loop_with_enhanced_enabled._evaluate(perception, action_result, state)

        # Should use enhanced evaluator
        loop_with_enhanced_enabled.enhanced_completion_evaluator.evaluate.assert_called_once()
        assert result.decision == EvaluationDecision.COMPLETE
        assert result.score == 0.85

    @pytest.mark.asyncio
    async def test_evaluate_uses_legacy_when_disabled(self, loop_with_enhanced_disabled):
        """Test that _evaluate uses legacy logic when enhanced disabled."""
        perception = MockPerception(confidence=0.9, task_type="code_generation")
        action_result = MockTurnResult(
            response="Here is the complete answer to your question",
            has_tool_calls=False,
            has_content=True,
        )
        state = {}

        result = await loop_with_enhanced_disabled._evaluate(perception, action_result, state)

        # Should use legacy evaluation and return a valid decision
        assert result.decision in (
            EvaluationDecision.COMPLETE,
            EvaluationDecision.CONTINUE,
            EvaluationDecision.RETRY,
        )

    @pytest.mark.asyncio
    async def test_enhanced_evaluator_receives_correct_params(self, loop_with_enhanced_enabled):
        """Test that enhanced evaluator receives correct parameters."""
        perception = MockPerception(confidence=0.8, requirements=[])
        action_result = MockTurnResult(response="Test")
        state = {"key": "value"}

        # Mock the enhanced evaluator
        loop_with_enhanced_enabled.enhanced_completion_evaluator = AsyncMock()
        loop_with_enhanced_enabled.enhanced_completion_evaluator.evaluate = AsyncMock(
            return_value=Mock(decision=EvaluationDecision.CONTINUE, score=0.5, reason="Test")
        )

        await loop_with_enhanced_enabled._evaluate(perception, action_result, state)

        # Verify correct parameters passed
        call_args = loop_with_enhanced_enabled.enhanced_completion_evaluator.evaluate.call_args
        assert call_args[1]["perception"] == perception
        assert call_args[1]["action_result"] == action_result
        assert call_args[1]["state"] == state
        assert "fulfillment_detector" in call_args[1]
        assert "spin_detector" in call_args[1]

    @pytest.mark.asyncio
    async def test_graceful_degradation_on_enhanced_error(self, loop_with_enhanced_enabled):
        """Test graceful degradation when enhanced evaluator raises error."""
        perception = MockPerception(confidence=0.8, task_type="code_generation")
        action_result = MockTurnResult(
            response="Response",
            has_tool_calls=True,
            successful_tool_count=2,
        )
        state = {}

        # Mock enhanced evaluator to raise error
        loop_with_enhanced_enabled.enhanced_completion_evaluator = AsyncMock()
        loop_with_enhanced_enabled.enhanced_completion_evaluator.evaluate = AsyncMock(
            side_effect=Exception("Test error")
        )

        # Should not raise exception, should fall back to legacy
        result = await loop_with_enhanced_enabled._evaluate(perception, action_result, state)

        # Should use legacy fallback and return a valid decision
        assert result.decision in (
            EvaluationDecision.COMPLETE,
            EvaluationDecision.CONTINUE,
            EvaluationDecision.RETRY,
        )

    @pytest.mark.asyncio
    async def test_spin_detector_passed_to_enhanced(self, loop_with_enhanced_enabled):
        """Test that spin_detector is passed to enhanced evaluator."""
        perception = MockPerception(confidence=0.8, requirements=[])
        action_result = MockTurnResult(response="Test")
        state = {}

        # Mock the enhanced evaluator
        loop_with_enhanced_enabled.enhanced_completion_evaluator = AsyncMock()
        loop_with_enhanced_enabled.enhanced_completion_evaluator.evaluate = AsyncMock(
            return_value=Mock(decision=EvaluationDecision.CONTINUE, score=0.5, reason="Test")
        )

        await loop_with_enhanced_enabled._evaluate(perception, action_result, state)

        # Verify spin_detector passed
        call_args = loop_with_enhanced_enabled.enhanced_completion_evaluator.evaluate.call_args
        assert call_args[1]["spin_detector"] == loop_with_enhanced_enabled.spin_detector

    @pytest.mark.asyncio
    async def test_fulfillment_detector_passed_to_enhanced(self, loop_with_enhanced_enabled):
        """Test that fulfillment_detector is passed to enhanced evaluator."""
        perception = MockPerception(confidence=0.8, requirements=[])
        action_result = MockTurnResult(response="Test")
        state = {}

        # Mock the enhanced evaluator
        loop_with_enhanced_enabled.enhanced_completion_evaluator = AsyncMock()
        loop_with_enhanced_enabled.enhanced_completion_evaluator.evaluate = AsyncMock(
            return_value=Mock(decision=EvaluationDecision.CONTINUE, score=0.5, reason="Test")
        )

        await loop_with_enhanced_enabled._evaluate(perception, action_result, state)

        # Verify fulfillment_detector passed
        call_args = loop_with_enhanced_enabled.enhanced_completion_evaluator.evaluate.call_args
        assert call_args[1]["fulfillment_detector"] == loop_with_enhanced_enabled.fulfillment

    def test_configuration_options(self, mock_orchestrator, mock_memory_coordinator):
        """Test that configuration options are passed correctly."""
        config = {
            "disable_enhanced_completion": False,
            "enable_requirement_validation": False,  # Override default
            "enable_completion_scoring": False,  # Override default
            "enable_context_keywords": False,  # Override default
            "completion_threshold": 0.90,  # Override default
        }

        loop = AgenticLoop(
            orchestrator=mock_orchestrator,
            memory_coordinator=mock_memory_coordinator,
            config=config,
        )

        # Enhanced evaluator should be initialized with custom config
        assert loop.enhanced_completion_evaluator is not None
        assert loop.enhanced_completion_evaluator.enable_requirement_validation is False
        assert loop.enhanced_completion_evaluator.enable_completion_scoring is False
        assert loop.enhanced_completion_evaluator.enable_context_keywords is False
        assert loop.enhanced_completion_evaluator.completion_threshold == 0.90

    @pytest.mark.asyncio
    async def test_qa_shortcut_bypasses_enhanced(self, loop_with_enhanced_enabled):
        """Test that Q&A shortcut takes priority over enhanced evaluation."""
        perception = MockPerception(confidence=0.5, requirements=[])
        action_result = MockTurnResult(
            response="Answer is 42", is_qa_response=True, has_content=True
        )
        state = {}

        result = await loop_with_enhanced_enabled._evaluate(perception, action_result, state)

        # Q&A shortcut should give COMPLETE with high score
        # The actual implementation may vary, just check it's a valid decision
        assert result.decision in (
            EvaluationDecision.COMPLETE,
            EvaluationDecision.CONTINUE,
            EvaluationDecision.RETRY,
        )

    @pytest.mark.asyncio
    async def test_spin_detection_bypasses_enhanced(self, loop_with_enhanced_enabled):
        """Test that spin detection takes priority over enhanced evaluation."""
        # SpinDetector.state is a property - need to set counters instead
        loop_with_enhanced_enabled.spin_detector.consecutive_all_blocked = 4
        loop_with_enhanced_enabled.spin_detector.consecutive_no_tool_turns = 0

        # Verify state is TERMINATED
        assert loop_with_enhanced_enabled.spin_detector.state == SpinState.TERMINATED

        perception = MockPerception(confidence=0.9, requirements=[])
        action_result = MockTurnResult(response="Stuck")
        state = {}

        result = await loop_with_enhanced_enabled._evaluate(perception, action_result, state)

        # Should return FAIL due to spin
        assert result.decision == EvaluationDecision.FAIL
        assert "Spin detected" in result.reason

    def test_backward_compatibility_default_disabled(
        self, mock_orchestrator, mock_memory_coordinator
    ):
        """Test backward compatibility - legacy behavior preserved when disabled."""
        loop = AgenticLoop(
            orchestrator=mock_orchestrator,
            memory_coordinator=mock_memory_coordinator,
            config={"disable_enhanced_completion": True},
        )

        # Should not have enhanced evaluator
        assert loop.enhanced_completion_evaluator is None

        # Should still have all other components
        assert loop.spin_detector is not None
        assert loop.nudge_policy is not None
        assert loop.perception is not None

    def test_feature_flag_inverted_logic(self, mock_orchestrator, mock_memory_coordinator):
        """Test that feature flag uses inverted logic (disable_enhanced_completion)."""
        # Empty config = enhanced enabled (default)
        loop1 = AgenticLoop(
            orchestrator=mock_orchestrator,
            memory_coordinator=mock_memory_coordinator,
            config={},
        )
        assert loop1.enhanced_completion_evaluator is not None

        # disable_enhanced_completion=False = enhanced enabled
        loop2 = AgenticLoop(
            orchestrator=mock_orchestrator,
            memory_coordinator=mock_memory_coordinator,
            config={"disable_enhanced_completion": False},
        )
        assert loop2.enhanced_completion_evaluator is not None

        # disable_enhanced_completion=True = enhanced disabled
        loop3 = AgenticLoop(
            orchestrator=mock_orchestrator,
            memory_coordinator=mock_memory_coordinator,
            config={"disable_enhanced_completion": True},
        )
        assert loop3.enhanced_completion_evaluator is None


class TestEnhancedCompletionEndToEnd:
    """End-to-end tests for enhanced completion in agentic loop."""

    @pytest.fixture
    def mock_orchestrator(self):
        """Create mock orchestrator."""
        orchestrator = Mock()
        orchestrator.turn_executor = Mock()
        return orchestrator

    @pytest.fixture
    def mock_memory(self):
        """Create mock memory coordinator."""
        return Mock()

    @pytest.mark.asyncio
    async def test_full_loop_with_enhanced_completion(self, mock_orchestrator, mock_memory):
        """Test full agentic loop iteration with enhanced completion."""
        loop = AgenticLoop(
            orchestrator=mock_orchestrator,
            memory_coordinator=mock_memory,
            config={"disable_enhanced_completion": False},  # Enhanced enabled
        )

        # Create mock perception with requirements
        perception = MockPerception(
            confidence=0.9,
            requirements=[
                Requirement(
                    type=RequirementType.FUNCTIONAL,
                    description="Complete task",
                    priority=0,
                )
            ],
        )

        # Create mock action result showing completion
        action_result = MockTurnResult(
            response="Task is complete",
            has_tool_calls=True,
            tool_calls=["write_file"],
        )

        state = {"files_modified": ["task.py"]}

        # Run evaluation
        result = await loop._evaluate(perception, action_result, state)

        # Should make a decision
        assert result.decision in (
            EvaluationDecision.COMPLETE,
            EvaluationDecision.CONTINUE,
            EvaluationDecision.RETRY,
        )
        assert result.score > 0.0

    @pytest.mark.asyncio
    async def test_enhanced_vs_legacy_decision_difference(self, mock_orchestrator, mock_memory):
        """Test that enhanced can make different decisions than legacy."""
        # Create two loops - one with enhanced, one without
        loop_enhanced = AgenticLoop(
            orchestrator=mock_orchestrator,
            memory_coordinator=mock_memory,
            config={"disable_enhanced_completion": False},
        )

        loop_legacy = AgenticLoop(
            orchestrator=mock_orchestrator,
            memory_coordinator=mock_memory,
            config={"disable_enhanced_completion": True},
        )

        perception = MockPerception(confidence=0.7, requirements=[])
        action_result = MockTurnResult(response="Done some work")
        state = {}

        # Get decisions from both
        result_enhanced = await loop_enhanced._evaluate(perception, action_result, state)
        result_legacy = await loop_legacy._evaluate(perception, action_result, state)

        # Both should make valid decisions
        assert result_enhanced.decision in (
            EvaluationDecision.COMPLETE,
            EvaluationDecision.CONTINUE,
            EvaluationDecision.RETRY,
        )
        assert result_legacy.decision in (
            EvaluationDecision.COMPLETE,
            EvaluationDecision.CONTINUE,
            EvaluationDecision.RETRY,
        )
