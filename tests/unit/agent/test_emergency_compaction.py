# Copyright 2025 Vijaykumar Singh
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for Discard-All emergency compaction (P2-2).

TDD approach: Tests written first, then implementation.

Emergency compaction is triggered when context is critically full (>95%)
and standard compaction isn't enough. It aggressively discards content
while preserving user intent.
"""

import pytest
from dataclasses import dataclass
from typing import Optional

from victor.agent.emergency_compaction import (
    EmergencyCompactionConfig,
    EmergencyCompactor,
    should_trigger_emergency_compaction,
    emergency_compact,
)


@dataclass
class MockMessage:
    """Mock message for testing."""

    role: str
    content: str
    tool_call_id: Optional[str] = None


class TestEmergencyCompactionConfig:
    """Test suite for emergency compaction configuration."""

    def test_default_config_exists(self):
        """EmergencyCompactionConfig should have sensible defaults."""
        config = EmergencyCompactionConfig()

        assert config.critical_threshold == 0.95
        assert config.min_messages_after_compact == 3
        assert config.preserve_user_messages is True
        assert config.preserve_system_prompt is True
        assert config.max_tool_result_age == 2

    def test_custom_config_overrides_defaults(self):
        """Custom configuration should override defaults."""
        config = EmergencyCompactionConfig(
            critical_threshold=0.90,
            min_messages_after_compact=5,
        )

        assert config.critical_threshold == 0.90
        assert config.min_messages_after_compact == 5

    def test_config_validates_ranges(self):
        """Config should validate parameter ranges."""
        # Invalid threshold should raise error
        with pytest.raises(ValueError):
            EmergencyCompactionConfig(critical_threshold=1.5)

        with pytest.raises(ValueError):
            EmergencyCompactionConfig(critical_threshold=-0.1)


class TestEmergencyCompactionTrigger:
    """Test suite for emergency compaction trigger conditions."""

    def test_trigger_at_critical_threshold(self):
        """Should trigger at critical threshold (95%)."""
        assert should_trigger_emergency_compaction(
            utilization=0.95,
            standard_compaction_failed=True,
        )

    def test_trigger_above_critical_threshold(self):
        """Should trigger above critical threshold."""
        assert should_trigger_emergency_compaction(
            utilization=0.98,
            standard_compaction_failed=True,
        )

    def test_no_trigger_below_critical_threshold(self):
        """Should not trigger below critical threshold."""
        assert not should_trigger_emergency_compaction(
            utilization=0.90,
            standard_compaction_failed=True,
        )

    def test_no_trigger_if_standard_compaction_succeeded(self):
        """Should not trigger if standard compaction succeeded."""
        assert not should_trigger_emergency_compaction(
            utilization=0.98,
            standard_compaction_failed=False,
        )

    def test_trigger_at_exact_threshold_with_failure(self):
        """Should trigger at exact threshold with standard failure."""
        assert should_trigger_emergency_compaction(
            utilization=0.95,
            standard_compaction_failed=True,
        )


class TestEmergencyCompactor:
    """Test suite for EmergencyCompactor class."""

    def test_preserves_system_prompt(self):
        """System prompt should always be preserved."""
        messages = [
            MockMessage("system", "You are a helpful assistant."),
            MockMessage("user", "Hello"),
            MockMessage("assistant", "Hi there!"),
            MockMessage("tool", "Result here"),
        ]

        compactor = EmergencyCompactor()
        result = compactor.compact(messages, current_turn=10)

        # System message should be preserved
        assert any(m.role == "system" for m in result)

    def test_preserves_user_messages(self):
        """User messages should be preserved by default."""
        messages = [
            MockMessage("system", "System prompt"),
            MockMessage("user", "First message"),
            MockMessage("assistant", "Response"),
            MockMessage("user", "Second message"),
            MockMessage("assistant", "Another response"),
        ]

        compactor = EmergencyCompactor(config=EmergencyCompactionConfig(inject_warning=False))
        result = compactor.compact(messages, current_turn=10)

        # Both user messages should be preserved (warning disabled)
        user_messages = [m for m in result if m.role == "user"]
        assert len(user_messages) == 2

    def test_discards_old_tool_results(self):
        """Old tool results should be discarded when there are many messages."""
        # Create a message list that exceeds the minimum
        messages = [
            MockMessage("system", "System prompt"),
            MockMessage("user", "Read file"),
            MockMessage("tool", "Old result 1"),
            MockMessage("assistant", "Let me check"),
            MockMessage("tool", "Old result 2"),
            MockMessage("user", "Read another file"),
            MockMessage("tool", "Recent result"),
        ]
        # Add more messages to exceed minimum and trigger actual compaction
        for i in range(10):
            messages.append(MockMessage("assistant", f"Response {i}"))
            messages.append(MockMessage("tool", f"Tool result {i}"))

        compactor = EmergencyCompactor(
            config=EmergencyCompactionConfig(
                min_messages_after_compact=5,  # Low minimum to force compaction
                max_tool_result_age=2,  # Only keep 2 recent tool results
                inject_warning=False,
            )
        )
        result = compactor.compact(messages, current_turn=10)

        # Verify that tool results were limited (we added many, should keep only recent)
        tool_results = [m for m in result if m.role == "tool"]
        # Should keep approximately max_tool_result_age tool results
        # (give some tolerance for the minimum message requirement)
        assert len(tool_results) < 10  # Definitely fewer than all 13

    def test_keeps_minimum_messages(self):
        """Should keep at least minimum messages after compaction."""
        messages = [MockMessage("system", "System")]
        messages.extend(MockMessage("user", f"Message {i}") for i in range(20))
        messages.insert(1, MockMessage("user", "First user message"))

        compactor = EmergencyCompactor(
            config=EmergencyCompactionConfig(min_messages_after_compact=5)
        )
        result = compactor.compact(messages, current_turn=10)

        # Should keep at least 5 messages
        assert len(result) >= 5

    def test_injects_compaction_warning(self):
        """Should inject warning message about emergency compaction."""
        messages = [
            MockMessage("system", "System prompt"),
            MockMessage("user", "Message"),
            MockMessage("assistant", "Response"),
        ]

        compactor = EmergencyCompactor()
        result = compactor.compact(messages, current_turn=10)

        # Should have a user message about compaction (either original or injected)
        user_messages = [m for m in result if m.role == "user"]
        assert len(user_messages) >= 1  # At least the original user message

        # Check if warning was added (result should have more messages than input
        # when inject_warning is True)
        assert len(result) > len(messages) or any(
            "compact" in m.content.lower() or "reset" in m.content.lower() for m in result
        )

    def test_preserves_most_recent_messages(self):
        """Most recent messages should be preserved."""
        messages = [
            MockMessage("system", "System"),
        ]
        for i in range(10):
            messages.append(MockMessage("user", f"User {i}"))
            messages.append(MockMessage("assistant", f"Assistant {i}"))

        compactor = EmergencyCompactor(
            config=EmergencyCompactionConfig(min_messages_after_compact=4)
        )
        result = compactor.compact(messages, current_turn=20)

        # Most recent messages should be preserved
        # Check that last user message is there
        assert any("User 9" in m.content for m in result if m.role == "user")

    def test_returns_compaction_summary(self):
        """Compaction should return summary of what was done."""
        messages = [
            MockMessage("system", "System"),
            MockMessage("user", "Message 1"),
            MockMessage("assistant", "Response 1"),
            MockMessage("tool", "Tool result"),
        ]

        compactor = EmergencyCompactor()
        result, summary = compactor.compact_with_summary(messages, current_turn=10)

        # Summary should contain information
        assert summary is not None
        assert len(summary) > 0
        assert any(
            keyword in summary.lower() for keyword in ["removed", "discarded", "compact", "message"]
        )


class TestEmergencyCompactFunction:
    """Test suite for emergency_compact convenience function."""

    def test_function_works_with_simple_list(self):
        """Convenience function should work with simple message list."""
        messages = [
            MockMessage("system", "System"),
            MockMessage("user", "Message"),
            MockMessage("assistant", "Response"),
        ]

        result = emergency_compact(messages, current_turn=10)

        # Should return compacted messages
        assert isinstance(result, list)
        assert len(result) > 0  # At least system prompt

    def test_function_handles_empty_list(self):
        """Should handle empty message list gracefully."""
        result = emergency_compact([], current_turn=10)

        # Should return empty list or minimal list
        assert isinstance(result, list)

    def test_function_preserves_system_prompt(self):
        """Convenience function should preserve system prompt."""
        messages = [
            MockMessage("system", "System prompt"),
            MockMessage("user", "Message"),
        ]

        result = emergency_compact(messages, current_turn=10)

        # System prompt should be preserved
        assert any(m.role == "system" for m in result)


class TestEmergencyCompactionSafety:
    """Test suite for emergency compaction safety features."""

    def test_never_discards_all_messages(self):
        """Should never discard all messages - always keep something."""
        messages = [
            MockMessage("system", "System"),
            MockMessage("user", "User"),
            MockMessage("assistant", "Assistant"),
        ]

        compactor = EmergencyCompactor(
            config=EmergencyCompactionConfig(
                min_messages_after_compact=1,
                preserve_user_messages=False,  # Even with this False
            )
        )
        result = compactor.compact(messages, current_turn=10)

        # Should always keep at least one message
        assert len(result) >= 1

    def test_logs_emergency_compaction_event(self):
        """Emergency compaction should log an event."""
        from unittest.mock import patch

        messages = [MockMessage("system", "System"), MockMessage("user", "User")]

        compactor = EmergencyCompactor()

        with patch("victor.agent.emergency_compaction.logger") as mock_logger:
            compactor.compact(messages, current_turn=10)

            # Should have logged a warning about emergency compaction
            assert mock_logger.warning.called

    def test_configures_for_aggressive_mode(self):
        """Should support aggressive mode configuration."""
        config = EmergencyCompactionConfig(
            critical_threshold=0.90,  # Lower threshold
            min_messages_after_compact=2,  # Fewer messages
            max_tool_result_age=1,  # Discard more tool results
        )

        compactor = EmergencyCompactor(config=config)

        assert compactor.config.critical_threshold == 0.90
        assert compactor.config.min_messages_after_compact == 2
