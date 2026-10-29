"""Tests for DACS-inspired focus sessions in context assembly."""

from dataclasses import dataclass


@dataclass
class MockMessage:
    role: str
    content: str


class TestFocusPhaseDetection:
    def test_method_exists(self):
        from victor.agent.conversation.assembler import TurnBoundaryContextAssembler

        assert hasattr(TurnBoundaryContextAssembler, "_detect_focus_phase")

    def test_detect_exploration_phase(self):
        from victor.agent.conversation.assembler import TurnBoundaryContextAssembler

        messages = [
            MockMessage("assistant", "read_file result: def hello()..."),
            MockMessage("assistant", "code_search found 3 matches..."),
            MockMessage("assistant", "grep result: line 42..."),
        ]
        assert TurnBoundaryContextAssembler._detect_focus_phase(messages) == "exploration"

    def test_detect_mutation_phase(self):
        from victor.agent.conversation.assembler import TurnBoundaryContextAssembler

        messages = [
            MockMessage("assistant", "edit result: successfully replaced old_str"),
            MockMessage("assistant", "write_file created new_module.py"),
            MockMessage("assistant", "edit result: added import statement"),
        ]
        assert TurnBoundaryContextAssembler._detect_focus_phase(messages) == "mutation"

    def test_detect_execution_phase(self):
        from victor.agent.conversation.assembler import TurnBoundaryContextAssembler

        messages = [
            MockMessage("assistant", "bash output: 5 tests passed"),
            MockMessage("assistant", "git status: 2 files modified"),
            MockMessage("assistant", "test result: all green"),
        ]
        assert TurnBoundaryContextAssembler._detect_focus_phase(messages) == "execution"

    def test_mixed_when_no_tools(self):
        from victor.agent.conversation.assembler import TurnBoundaryContextAssembler

        messages = [MockMessage("user", "hello"), MockMessage("assistant", "Hi!")]
        assert TurnBoundaryContextAssembler._detect_focus_phase(messages) == "mixed"

    def test_empty_messages(self):
        from victor.agent.conversation.assembler import TurnBoundaryContextAssembler

        assert TurnBoundaryContextAssembler._detect_focus_phase([]) == "mixed"


class TestFocusScoring:
    def test_method_exists(self):
        from victor.agent.conversation.assembler import TurnBoundaryContextAssembler

        assert hasattr(TurnBoundaryContextAssembler, "_apply_focus_scoring")

    def test_mutation_phase_compresses_old_reads(self):
        from victor.agent.conversation.assembler import TurnBoundaryContextAssembler

        assembler = TurnBoundaryContextAssembler.__new__(TurnBoundaryContextAssembler)
        messages = [
            MockMessage("assistant", "read_file result: old content..."),
            MockMessage("assistant", "edit result: updated auth module"),
        ]
        adjusted = TurnBoundaryContextAssembler._apply_focus_scoring(
            assembler, messages, [1.0, 1.0], "mutation"
        )
        assert adjusted[0] < 1.0  # Read compressed
        assert adjusted[1] > 1.0  # Edit boosted

    def test_mixed_phase_no_adjustment(self):
        from victor.agent.conversation.assembler import TurnBoundaryContextAssembler

        assembler = TurnBoundaryContextAssembler.__new__(TurnBoundaryContextAssembler)
        messages = [MockMessage("assistant", "some content")]
        adjusted = TurnBoundaryContextAssembler._apply_focus_scoring(
            assembler, messages, [1.0], "mixed"
        )
        assert adjusted == [1.0]

    def test_exploration_phase_compresses_old_edits(self):
        from victor.agent.conversation.assembler import TurnBoundaryContextAssembler

        assembler = TurnBoundaryContextAssembler.__new__(TurnBoundaryContextAssembler)
        messages = [
            MockMessage("assistant", "edit result: old modification"),
            MockMessage("assistant", "read_file result: current content"),
        ]
        adjusted = TurnBoundaryContextAssembler._apply_focus_scoring(
            assembler, messages, [1.0, 1.0], "exploration"
        )
        assert adjusted[0] < 1.0
        assert adjusted[1] > 1.0

    def test_mutation_phase_boosts_edits(self):
        from victor.agent.conversation.assembler import TurnBoundaryContextAssembler

        assembler = TurnBoundaryContextAssembler.__new__(TurnBoundaryContextAssembler)
        messages = [MockMessage("assistant", "edit result: replaced function body")]
        adjusted = TurnBoundaryContextAssembler._apply_focus_scoring(
            assembler, messages, [1.0], "mutation"
        )
        assert adjusted[0] > 1.0
