"""Unit tests for victor.state package.

Tests the standalone state machine implementation with protocols,
generic state machine, and integration with conversation state.
"""

from typing import Any, Dict, List, Optional

import pytest

from victor.storage.state import (
    ConversationStage,
    StateConfig,
    StateMachine,
    StateTransition,
)


class TestStateConfig:
    """Tests for StateConfig dataclass."""

    def test_valid_config(self):
        """Valid config should be created successfully."""
        config = StateConfig(
            stages=["A", "B", "C"],
            initial_stage="A",
            transitions={"A": ["B"], "B": ["C"], "C": []},
        )
        assert config.stages == ["A", "B", "C"]
        assert config.initial_stage == "A"

    def test_invalid_initial_stage(self):
        """Invalid initial stage should raise ValueError."""
        with pytest.raises(ValueError, match="Initial stage"):
            StateConfig(
                stages=["A", "B"],
                initial_stage="C",  # Not in stages
            )

    def test_invalid_transition_source(self):
        """Invalid transition source should raise ValueError."""
        with pytest.raises(ValueError, match="Transition source"):
            StateConfig(
                stages=["A", "B"],
                initial_stage="A",
                transitions={"X": ["B"]},  # X not in stages
            )

    def test_invalid_transition_target(self):
        """Invalid transition target should raise ValueError."""
        with pytest.raises(ValueError, match="Transition target"):
            StateConfig(
                stages=["A", "B"],
                initial_stage="A",
                transitions={"A": ["C"]},  # C not in stages
            )


class TestStateMachine:
    """Tests for generic StateMachine."""

    @pytest.fixture
    def basic_config(self) -> StateConfig:
        """Create a basic config for tests."""
        return StateConfig(
            stages=["START", "MIDDLE", "END"],
            initial_stage="START",
            transitions={
                "START": ["MIDDLE"],
                "MIDDLE": ["END", "START"],
                "END": [],
            },
            stage_tools={
                "START": {"read", "search"},
                "MIDDLE": {"write", "edit"},
                "END": {"commit"},
            },
        )

    def test_initial_state(self, basic_config: StateConfig):
        """Machine should start in initial state."""
        machine = StateMachine(basic_config)
        assert machine.get_stage() == "START"

    def test_valid_transition(self, basic_config: StateConfig):
        """Valid transitions should succeed."""
        machine = StateMachine(basic_config)
        success = machine.transition_to("MIDDLE")
        assert success
        assert machine.get_stage() == "MIDDLE"

    def test_invalid_transition(self, basic_config: StateConfig):
        """Invalid transitions should fail."""
        machine = StateMachine(basic_config)
        # Can't go directly from START to END
        success = machine.transition_to("END")
        assert not success
        assert machine.get_stage() == "START"

    def test_get_stage_tools(self, basic_config: StateConfig):
        """get_stage_tools should return configured tools."""
        machine = StateMachine(basic_config)
        tools = machine.get_stage_tools()
        assert tools == {"read", "search"}

        machine.transition_to("MIDDLE")
        tools = machine.get_stage_tools()
        assert tools == {"write", "edit"}

    def test_get_valid_transitions(self, basic_config: StateConfig):
        """get_valid_transitions should return valid next stages."""
        machine = StateMachine(basic_config)
        valid = machine.get_valid_transitions()
        assert valid == ["MIDDLE"]

        machine.transition_to("MIDDLE")
        valid = machine.get_valid_transitions()
        assert set(valid) == {"END", "START"}

    def test_can_transition_to(self, basic_config: StateConfig):
        """can_transition_to should check validity."""
        machine = StateMachine(basic_config)
        assert machine.can_transition_to("MIDDLE")
        assert not machine.can_transition_to("END")
        assert not machine.can_transition_to("INVALID")

    def test_same_state_transition(self, basic_config: StateConfig):
        """Transitioning to same state should return True."""
        machine = StateMachine(basic_config)
        success = machine.transition_to("START")
        assert success

    def test_reset(self, basic_config: StateConfig):
        """reset should return to initial state."""
        machine = StateMachine(basic_config)
        machine.transition_to("MIDDLE")
        machine.transition_to("END")

        machine.reset()
        assert machine.get_stage() == "START"
        assert machine.transition_count == 0

    def test_history_tracking(self, basic_config: StateConfig):
        """Machine should track transition history."""
        machine = StateMachine(basic_config)
        machine.transition_to("MIDDLE")
        machine.transition_to("END")

        assert len(machine.history) == 2
        assert machine.history[0].from_stage == "START"
        assert machine.history[0].to_stage == "MIDDLE"
        assert machine.history[1].from_stage == "MIDDLE"
        assert machine.history[1].to_stage == "END"

    def test_transition_count(self, basic_config: StateConfig):
        """transition_count should track total transitions."""
        machine = StateMachine(basic_config)
        assert machine.transition_count == 0

        machine.transition_to("MIDDLE")
        assert machine.transition_count == 1

        machine.transition_to("START")  # Back
        assert machine.transition_count == 2

    def test_observer_notification(self, basic_config: StateConfig):
        """Observers should be notified of transitions."""
        machine = StateMachine(basic_config)
        transitions: List[tuple] = []

        class TestObserver:
            def on_transition(
                self, old_stage: str, new_stage: str, context: Dict[str, Any]
            ) -> None:
                transitions.append((old_stage, new_stage))

        machine.add_observer(TestObserver())
        machine.transition_to("MIDDLE")

        assert len(transitions) == 1
        assert transitions[0] == ("START", "MIDDLE")

    def test_observer_removal(self, basic_config: StateConfig):
        """Observers should be removable."""
        machine = StateMachine(basic_config)
        transitions: List[tuple] = []

        class TestObserver:
            def on_transition(
                self, old_stage: str, new_stage: str, context: Dict[str, Any]
            ) -> None:
                transitions.append((old_stage, new_stage))

        observer = TestObserver()
        remove = machine.add_observer(observer)

        machine.transition_to("MIDDLE")
        assert len(transitions) == 1

        remove()  # Remove observer

        machine.transition_to("START")
        assert len(transitions) == 1  # No new transitions recorded

    def test_validator(self, basic_config: StateConfig):
        """Validators should be able to block transitions."""

        class BlockingValidator:
            def validate(
                self, current: str, target: str, context: Dict[str, Any]
            ) -> tuple[bool, Optional[str]]:
                if target == "MIDDLE":
                    return False, "MIDDLE is blocked"
                return True, None

        machine = StateMachine(basic_config, validators=[BlockingValidator()])

        success = machine.transition_to("MIDDLE")
        assert not success
        assert machine.get_stage() == "START"

    def test_to_dict(self, basic_config: StateConfig):
        """to_dict should serialize state."""
        machine = StateMachine(basic_config)
        machine.transition_to("MIDDLE")

        data = machine.to_dict()
        assert data["current_stage"] == "MIDDLE"
        assert data["transition_count"] == 1
        assert len(data["history"]) == 1

    def test_from_dict(self, basic_config: StateConfig):
        """from_dict should restore state."""
        data = {"current_stage": "MIDDLE", "transition_count": 5}
        machine = StateMachine.from_dict(data, basic_config)

        assert machine.get_stage() == "MIDDLE"
        assert machine.transition_count == 5


class TestStateMachineCooldown:
    """Tests for state machine cooldown behavior."""

    def test_cooldown_blocks_rapid_transitions(self):
        """Cooldown should block rapid transitions."""
        config = StateConfig(
            stages=["A", "B"],
            initial_stage="A",
            transitions={"A": ["B"], "B": ["A"]},
            cooldown_seconds=1.0,  # 1 second cooldown
        )
        machine = StateMachine(config)

        # First transition should succeed
        assert machine.transition_to("B")

        # Immediate second transition should be blocked
        assert not machine.transition_to("A")

    def test_backward_threshold(self):
        """Backward transitions should require higher confidence."""
        config = StateConfig(
            stages=["A", "B", "C"],
            initial_stage="A",
            transitions={"A": ["B", "C"], "B": ["A", "C"], "C": ["A", "B"]},
            backward_threshold=0.9,
        )
        machine = StateMachine(config)
        machine.transition_to("C")

        # Low confidence backward transition should fail
        assert not machine.transition_to("A", confidence=0.5)

        # High confidence backward transition should succeed
        assert machine.transition_to("A", confidence=0.95)


class TestStateProtocol:
    """Tests for StateProtocol compliance."""

    def test_state_machine_implements_protocol(self):
        """StateMachine should implement StateProtocol."""
        config = StateConfig(
            stages=["A", "B"],
            initial_stage="A",
            transitions={"A": ["B"], "B": []},
        )
        machine = StateMachine(config)

        # Check protocol methods exist and work
        assert isinstance(machine.get_stage(), str)
        assert isinstance(machine.transition_to("B"), bool)
        assert isinstance(machine.get_stage_tools(), set)


class TestConversationStateMachineIntegration:
    """Tests for ConversationStateMachine re-exported from state package."""

    def test_import_from_state_package(self):
        """ConversationStateMachine should be importable from state package."""
        from victor.storage.state import ConversationStateMachine

        machine = ConversationStateMachine()
        assert machine.get_stage() == ConversationStage.INITIAL

    def test_stage_keywords_available(self):
        """STAGE_KEYWORDS should be available from state package."""
        from victor.storage.state import STAGE_KEYWORDS

        assert ConversationStage.PLANNING in STAGE_KEYWORDS
        assert "plan" in STAGE_KEYWORDS[ConversationStage.PLANNING]


class TestStateTransition:
    """Tests for StateTransition dataclass."""

    def test_transition_creation(self):
        """StateTransition should be creatable with all fields."""
        transition = StateTransition(
            from_stage="A",
            to_stage="B",
            confidence=0.8,
            context={"reason": "test"},
        )
        assert transition.from_stage == "A"
        assert transition.to_stage == "B"
        assert transition.confidence == 0.8
        assert transition.context == {"reason": "test"}
        assert transition.timestamp > 0

    def test_transition_defaults(self):
        """StateTransition should have sensible defaults."""
        transition = StateTransition(from_stage="A", to_stage="B")
        assert transition.confidence == 1.0
        assert transition.context == {}
