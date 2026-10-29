"""Tests for compaction event sourcing."""

from unittest.mock import MagicMock, patch


from victor.core.event_sourcing import (
    ContextCompactedEvent,
    SessionResumedEvent,
    DomainEvent,
)


class TestCompactionEvents:
    def test_compacted_event_emitted_on_compaction(self):
        """Verify ContextCompactor emits event via emit_event_sync after compaction."""
        from victor.agent.context_compactor import ContextCompactor, CompactorConfig

        mock_controller = MagicMock()
        mock_controller.get_context_metrics.side_effect = [
            MagicMock(utilization=0.95, char_count=10000, is_overflow_risk=False),
            MagicMock(utilization=0.5, char_count=5000),
        ]
        mock_controller.smart_compact_history.return_value = 5
        # Create mock messages with proper content attribute (string, not MagicMock)
        mock_messages = []
        for _ in range(10):
            msg = MagicMock()
            msg.content = "some message content"
            msg.role = "user"
            mock_messages.append(msg)
        mock_controller.messages = mock_messages

        mock_event_bus = MagicMock()

        compactor = ContextCompactor(
            controller=mock_controller,
            config=CompactorConfig(
                proactive_threshold=0.9,
                enable_proactive=True,
            ),
            event_bus=mock_event_bus,
        )

        with patch("victor.core.events.emit_helper.emit_event_sync") as mock_emit:
            compactor.check_and_compact()

            mock_emit.assert_called_once()
            call_args = mock_emit.call_args
            assert call_args[0][0] is mock_event_bus
            assert call_args[0][1] == "context.compacted"
            data = call_args[0][2]
            assert data["messages_removed"] == 5
            assert data["chars_freed"] == 5000
            assert data["trigger"] == "threshold"

    def test_no_event_without_event_bus(self):
        """No error when event_bus is None."""
        from victor.agent.context_compactor import ContextCompactor, CompactorConfig

        mock_controller = MagicMock()
        mock_controller.get_context_metrics.side_effect = [
            MagicMock(utilization=0.95, char_count=10000, is_overflow_risk=False),
            MagicMock(utilization=0.5, char_count=5000),
        ]
        mock_controller.smart_compact_history.return_value = 5
        # Create mock messages with proper content attribute (string, not MagicMock)
        mock_messages = []
        for _ in range(10):
            msg = MagicMock()
            msg.content = "some message content"
            msg.role = "user"
            mock_messages.append(msg)
        mock_controller.messages = mock_messages

        compactor = ContextCompactor(
            controller=mock_controller,
            config=CompactorConfig(
                proactive_threshold=0.9,
                enable_proactive=True,
            ),
            event_bus=None,
        )
        # Should not raise
        action = compactor.check_and_compact()
        assert action.messages_removed == 5

    def test_event_contains_correct_metrics(self):
        event = ContextCompactedEvent(
            messages_removed=10,
            chars_freed=5000,
            trigger="threshold",
            summary="Removed 10 messages, freed ~1250 tokens",
        )
        assert event.messages_removed == 10
        assert event.chars_freed == 5000
        assert event.trigger == "threshold"
        assert "1250" in event.summary
        assert isinstance(event, DomainEvent)

    def test_session_resumed_event_structure(self):
        event = SessionResumedEvent(
            session_id="test-123",
            messages_restored=25,
            ledger_entries_restored=8,
        )
        assert event.session_id == "test-123"
        assert event.messages_restored == 25
        assert event.ledger_entries_restored == 8
        assert isinstance(event, DomainEvent)
        assert event.event_id  # Should have auto-generated ID
