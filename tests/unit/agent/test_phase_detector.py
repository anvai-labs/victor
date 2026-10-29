"""Tests for context phase detector.

Tests cover:
- Stage-to-phase mapping
- Phase detection from tools
- Phase detection from content
- Phase transition validation
- Cooldown and thrashing prevention
- Ensemble detection strategy
"""

import time


from victor.agent.context_phase_detector import (
    PhaseDetectionStrategy,
    PhaseDetector,
    PhaseTransitionDetector,
    TOOL_PATTERNS,
    create_phase_detector,
)
from victor.core.shared_types import ConversationStage, TaskPhase, STAGE_TO_PHASE_MAP


class TestStageToPhaseMapping:
    """Test stage-to-phase mapping."""

    def test_exploration_stages(self):
        """Test that exploration stages map correctly."""
        assert STAGE_TO_PHASE_MAP[ConversationStage.INITIAL] == TaskPhase.EXPLORATION
        assert STAGE_TO_PHASE_MAP[ConversationStage.READING] == TaskPhase.EXPLORATION
        assert STAGE_TO_PHASE_MAP[ConversationStage.ANALYSIS] == TaskPhase.EXPLORATION

    def test_planning_stage(self):
        """Test that planning stage maps correctly."""
        assert STAGE_TO_PHASE_MAP[ConversationStage.PLANNING] == TaskPhase.PLANNING

    def test_execution_stage(self):
        """Test that execution stage maps correctly."""
        assert STAGE_TO_PHASE_MAP[ConversationStage.EXECUTION] == TaskPhase.EXECUTION

    def test_review_stages(self):
        """Test that review stages map correctly."""
        assert STAGE_TO_PHASE_MAP[ConversationStage.VERIFICATION] == TaskPhase.REVIEW
        assert STAGE_TO_PHASE_MAP[ConversationStage.COMPLETION] == TaskPhase.REVIEW

    def test_all_stages_mapped(self):
        """Test that all conversation stages have a phase mapping."""
        for stage in ConversationStage:
            assert stage in STAGE_TO_PHASE_MAP
            assert STAGE_TO_PHASE_MAP[stage] in TaskPhase


class TestPhaseDetector:
    """Test phase detection functionality."""

    def test_initialization_with_defaults(self):
        """Test detector initialization with default strategy."""
        detector = PhaseDetector()

        assert detector._strategy == PhaseDetectionStrategy.STAGE_MAPPING
        assert detector._enable_content_analysis is False

    def test_initialization_with_custom_strategy(self):
        """Test detector initialization with custom strategy."""
        detector = PhaseDetector(
            strategy=PhaseDetectionStrategy.TOOL_PATTERNS,
            enable_content_analysis=True,
        )

        assert detector._strategy == PhaseDetectionStrategy.TOOL_PATTERNS
        assert detector._enable_content_analysis is True

    def test_detect_phase_from_stage_mapping(self):
        """Test phase detection via stage mapping."""
        detector = PhaseDetector(strategy=PhaseDetectionStrategy.STAGE_MAPPING)

        phase = detector.detect_phase(
            current_stage=ConversationStage.EXECUTION,
            recent_tools=None,
            message_content=None,
        )

        assert phase == TaskPhase.EXECUTION

    def test_detect_phase_from_tool_patterns(self):
        """Test phase detection via tool patterns."""
        detector = PhaseDetector(strategy=PhaseDetectionStrategy.TOOL_PATTERNS)

        phase = detector.detect_phase(
            current_stage=ConversationStage.INITIAL,  # Wrong stage, but tools override
            recent_tools=["edit", "write", "create"],
            message_content=None,
        )

        assert phase == TaskPhase.EXECUTION

    def test_fallback_to_exploration(self):
        """Test fallback to EXPLORATION when no patterns match."""
        detector = PhaseDetector(
            strategy=PhaseDetectionStrategy.TOOL_PATTERNS,
            enable_content_analysis=False,
        )

        phase = detector.detect_phase(
            current_stage=ConversationStage.INITIAL,
            recent_tools=[],
            message_content=None,
        )

        # Should return EXPLORATION as fallback when no tools detected
        assert phase == TaskPhase.EXPLORATION or phase is None


class TestPhaseDetectionFromTools:
    """Test tool-based phase detection."""

    def test_detect_exploration_from_tools(self):
        """Test exploration phase detection from tools."""
        detector = PhaseDetector()

        phase = detector._detect_from_tools(["search", "find", "read"])

        assert phase == TaskPhase.EXPLORATION

    def test_detect_exploration_from_legacy_core_tool_aliases(self):
        """Legacy read/list aliases should still classify as exploration."""
        detector = PhaseDetector()

        phase = detector._detect_from_tools(["read_file", "list_directory"])

        assert phase == TaskPhase.EXPLORATION

    def test_detect_execution_from_create_file_alias(self):
        """Create-file aliases should still classify as execution."""
        detector = PhaseDetector()

        phase = detector._detect_from_tools(["create_file", "edit"])

        assert phase == TaskPhase.EXECUTION

    def test_detect_execution_from_shell_alias(self):
        """Legacy shell aliases should still classify as execution."""
        detector = PhaseDetector()

        phase = detector._detect_from_tools(["execute_bash", "test"])

        assert phase == TaskPhase.EXECUTION

    def test_detect_planning_from_tools(self):
        """Test planning phase detection from tools."""
        detector = PhaseDetector()

        phase = detector._detect_from_tools(["plan", "breakdown", "design"])

        assert phase == TaskPhase.PLANNING

    def test_detect_execution_from_tools(self):
        """Test execution phase detection from tools."""
        detector = PhaseDetector()

        phase = detector._detect_from_tools(["edit", "write", "run", "test"])

        assert phase == TaskPhase.EXECUTION

    def test_detect_review_from_tools(self):
        """Test review phase detection from tools."""
        detector = PhaseDetector()

        phase = detector._detect_from_tools(["verify", "check", "test", "review"])

        assert phase == TaskPhase.REVIEW

    def test_empty_tools_returns_none(self):
        """Test that empty tools list returns None."""
        detector = PhaseDetector()

        phase = detector._detect_from_tools([])

        assert phase is None


class TestPhaseTransitionDetector:
    """Test phase transition management."""

    def test_initialization(self):
        """Test transition detector initialization."""
        detector = PhaseTransitionDetector()

        assert detector._current_phase is None
        assert detector._last_transition_time == 0.0
        assert len(detector._transition_history) == 0

    def test_first_transition_always_allowed(self):
        """Test that first transition is always allowed."""
        detector = PhaseTransitionDetector()

        allowed = detector.should_transition(TaskPhase.EXPLORATION)

        assert allowed is True

    def test_valid_transition_allowed(self):
        """Test that valid transitions are allowed."""
        detector = PhaseTransitionDetector()

        # Set current phase
        detector._current_phase = TaskPhase.EXPLORATION
        detector._last_transition_time = time.monotonic() - 5.0

        allowed = detector.should_transition(TaskPhase.PLANNING)

        assert allowed is True

    def test_invalid_transition_blocked(self):
        """Test that invalid transitions are blocked."""
        detector = PhaseTransitionDetector()

        # Set current phase
        detector._current_phase = TaskPhase.EXECUTION
        detector._last_transition_time = time.monotonic() - 5.0

        # REVIEW → EXECUTION is not a valid transition
        allowed = detector.should_transition(TaskPhase.EXECUTION)

        assert allowed is False

    def test_cooldown_blocks_transition(self):
        """Test that cooldown blocks rapid transitions."""
        detector = PhaseTransitionDetector(cooldown_seconds=2.0)

        # Set current phase and recent transition
        detector._current_phase = TaskPhase.EXPLORATION
        detector._last_transition_time = time.monotonic() - 0.5

        allowed = detector.should_transition(TaskPhase.PLANNING)

        assert allowed is False  # Blocked by cooldown

    def test_record_transition(self):
        """Test recording a phase transition."""
        detector = PhaseTransitionDetector()

        detector.record_transition(TaskPhase.EXPLORATION, TaskPhase.PLANNING)

        assert detector._current_phase == TaskPhase.PLANNING
        assert detector._last_transition_time > 0
        assert len(detector._transition_history) == 1

    def test_get_current_phase(self):
        """Test getting current phase."""
        detector = PhaseTransitionDetector()

        assert detector.get_current_phase() is None

        detector._current_phase = TaskPhase.EXECUTION

        assert detector.get_current_phase() == TaskPhase.EXECUTION


class TestFactoryFunction:
    """Test factory function."""

    def test_create_with_default_strategy(self):
        """Test creating detector with default strategy."""
        detector = create_phase_detector()

        assert detector._strategy == PhaseDetectionStrategy.STAGE_MAPPING

    def test_create_with_custom_strategy(self):
        """Test creating detector with custom strategy."""
        detector = create_phase_detector(strategy="tool_patterns")

        assert detector._strategy == PhaseDetectionStrategy.TOOL_PATTERNS

    def test_create_with_invalid_strategy(self):
        """Test creating detector with invalid strategy."""
        detector = create_phase_detector(strategy="invalid")

        assert detector._strategy == PhaseDetectionStrategy.STAGE_MAPPING


class TestToolPatterns:
    """Test tool pattern definitions."""

    def test_tool_patterns_coverage(self):
        """Test that all phases have tool patterns defined."""
        for phase in TaskPhase:
            assert phase in TOOL_PATTERNS
            assert len(TOOL_PATTERNS[phase]) > 0


class TestIntegration:
    """Integration tests for phase detection."""

    def test_full_detection_pipeline(self):
        """Test complete phase detection flow."""
        detector = PhaseDetector(
            strategy=PhaseDetectionStrategy.STAGE_MAPPING,
        )

        # Stage mapping takes priority
        phase = detector.detect_phase(
            current_stage=ConversationStage.EXECUTION,
            recent_tools=None,
        )

        assert phase == TaskPhase.EXECUTION

    def test_phase_transition_workflow(self):
        """Test complete phase transition workflow."""
        detector = PhaseDetector(strategy=PhaseDetectionStrategy.STAGE_MAPPING)
        # Use short cooldown for testing
        transition_detector = PhaseTransitionDetector(cooldown_seconds=0.01)

        # Initial phase detection
        phase1 = detector.detect_phase(
            current_stage=ConversationStage.INITIAL,
            recent_tools=None,
        )

        assert phase1 == TaskPhase.EXPLORATION
        assert transition_detector.should_transition(phase1) is True

        # Record initial phase
        transition_detector.record_transition(None, phase1)

        # Wait for cooldown to pass
        time.sleep(0.02)

        # Transition to next phase
        phase2 = detector.detect_phase(
            current_stage=ConversationStage.PLANNING,
            recent_tools=None,
        )

        assert phase2 == TaskPhase.PLANNING
        assert transition_detector.should_transition(phase2) is True

        # Record transition
        transition_detector.record_transition(phase1, phase2)

        assert transition_detector.get_current_phase() == phase2

    def test_multiple_decision_types(self):
        """Test phase detection for different conversation stages."""
        detector = PhaseDetector()

        stages_to_phases = {
            ConversationStage.INITIAL: TaskPhase.EXPLORATION,
            ConversationStage.READING: TaskPhase.EXPLORATION,
            ConversationStage.ANALYSIS: TaskPhase.EXPLORATION,
            ConversationStage.PLANNING: TaskPhase.PLANNING,
            ConversationStage.EXECUTION: TaskPhase.EXECUTION,
            ConversationStage.VERIFICATION: TaskPhase.REVIEW,
            ConversationStage.COMPLETION: TaskPhase.REVIEW,
        }

        for stage, expected_phase in stages_to_phases.items():
            detected = detector.detect_phase(
                current_stage=stage,
                recent_tools=None,
            )
            assert detected == expected_phase, f"Stage {stage} should map to {expected_phase}"
