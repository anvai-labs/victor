"""Tests for semantic context deduplication (DCE-inspired)."""

from dataclasses import dataclass


@dataclass
class MockMessage:
    role: str
    content: str


class TestSemanticDeduplication:
    """Test _deduplicate_semantic filter in context assembler."""

    def test_method_exists(self):
        from victor.agent.conversation.assembler import TurnBoundaryContextAssembler

        assert hasattr(TurnBoundaryContextAssembler, "_deduplicate_semantic")

    def test_identical_messages_deduplicated(self):
        """Two messages with identical content -> only one kept."""
        from victor.agent.conversation.assembler import TurnBoundaryContextAssembler

        assembler = TurnBoundaryContextAssembler.__new__(TurnBoundaryContextAssembler)

        msgs = [
            MockMessage(
                role="assistant",
                content="File content of src/auth.py: def hello(name): return f'Hello {name}'",
            ),
            MockMessage(
                role="assistant",
                content="File content of src/auth.py: def hello(name): return f'Hello {name}'",
            ),
            MockMessage(role="user", content="Now edit it"),
        ]

        result = TurnBoundaryContextAssembler._deduplicate_semantic(assembler, msgs)
        # One duplicate removed
        assert len(result) < len(msgs)

    def test_different_messages_preserved(self):
        """Messages with different content are all preserved."""
        from victor.agent.conversation.assembler import TurnBoundaryContextAssembler

        assembler = TurnBoundaryContextAssembler.__new__(TurnBoundaryContextAssembler)

        msgs = [
            MockMessage(role="assistant", content="File auth.py has authentication logic"),
            MockMessage(role="assistant", content="File db.py has database connection pooling"),
            MockMessage(role="user", content="Compare the two approaches"),
        ]

        result = TurnBoundaryContextAssembler._deduplicate_semantic(assembler, msgs)
        assert len(result) == len(msgs)

    def test_near_duplicate_tool_results(self):
        """Tool results that differ only slightly (timestamps, whitespace) are deduplicated."""
        from victor.agent.conversation.assembler import TurnBoundaryContextAssembler

        assembler = TurnBoundaryContextAssembler.__new__(TurnBoundaryContextAssembler)

        msgs = [
            MockMessage(
                role="assistant",
                content="read_file result:\ndef hello():\n    return 'world'\n",
            ),
            MockMessage(
                role="assistant",
                content="read_file result:\ndef hello():\n    return 'world'\n\n",
            ),
            MockMessage(role="user", content="What does it do?"),
        ]

        result = TurnBoundaryContextAssembler._deduplicate_semantic(assembler, msgs)
        assert len(result) < len(msgs)

    def test_short_messages_not_deduplicated(self):
        """Short messages (< 50 chars) are never deduplicated -- they're cheap."""
        from victor.agent.conversation.assembler import TurnBoundaryContextAssembler

        assembler = TurnBoundaryContextAssembler.__new__(TurnBoundaryContextAssembler)

        msgs = [
            MockMessage(role="user", content="yes"),
            MockMessage(role="user", content="yes"),
            MockMessage(role="user", content="ok"),
        ]

        result = TurnBoundaryContextAssembler._deduplicate_semantic(assembler, msgs)
        # Short messages preserved even if identical
        assert len(result) == len(msgs)

    def test_empty_list_returns_empty(self):
        from victor.agent.conversation.assembler import TurnBoundaryContextAssembler

        assembler = TurnBoundaryContextAssembler.__new__(TurnBoundaryContextAssembler)
        result = TurnBoundaryContextAssembler._deduplicate_semantic(assembler, [])
        assert result == []
