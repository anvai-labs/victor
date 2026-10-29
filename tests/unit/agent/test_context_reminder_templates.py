# Copyright 2025 Vijaykumar Singh
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for task-specific context reminder templates (P1 refinement).

TDD approach: Tests written first, then implementation.

Task-specific reminder templates provide context-aware continuation prompts
after compaction, improving task continuity (from OpenDev research).
"""

from victor.agent.context_reminder_templates import (
    get_reminder_for_task,
    get_post_compaction_reminder,
)


class TestTaskReminderTemplates:
    """Test suite for task-specific reminder templates."""

    def test_get_reminder_for_known_task_types(self):
        """Known task types should return appropriate reminders."""
        # Analysis tasks
        reminder = get_reminder_for_task("analysis")
        assert "analyzing" in reminder.lower() or "analysis" in reminder.lower()
        assert "continue" in reminder.lower()

        # Implementation tasks
        reminder = get_reminder_for_task("implementation")
        assert "implement" in reminder.lower()

        # Debugging tasks
        reminder = get_reminder_for_task("debugging")
        assert "debug" in reminder.lower() or "troubleshoot" in reminder.lower()

    def test_get_reminder_for_unknown_task_returns_default(self):
        """Unknown task types should return default reminder."""
        reminder = get_reminder_for_task("unknown_task")
        assert "continue" in reminder.lower()

    def test_get_reminder_with_context(self):
        """Reminder with context should include context in template."""
        reminder = get_reminder_for_task("analysis", context={"files": "file1.py, file2.py"})
        assert (
            "file1.py" in reminder or "{files}" not in reminder
        )  # Either filled or has placeholder

    def test_reminder_templates_are_concise(self):
        """Reminders should be concise (< 200 chars)."""
        for task_type in ["analysis", "implementation", "debugging", "coding"]:
            reminder = get_reminder_for_task(task_type)
            assert len(reminder) < 200, f"{task_type} reminder too long: {len(reminder)} chars"

    def test_all_reminders_use_user_role(self):
        """All reminders should suggest user role injection (OpenDev finding)."""
        # Reminders should be formatted for user role (higher salience)
        # This is checked by ensuring they don't sound like system commands
        system_like = ["you must", "required to", "do not"]
        for task_type in ["analysis", "implementation", "debugging"]:
            reminder = get_reminder_for_task(task_type)
            # Check that reminder is conversational, not authoritarian
            assert not any(phrase in reminder.lower() for phrase in system_like)


class TestPostCompactionReminder:
    """Test suite for post-compaction specific reminders."""

    def test_post_compaction_includes_compaction_context(self):
        """Post-compaction reminder should mention compaction."""
        reminder = get_post_compaction_reminder(
            task_type="analysis",
            compaction_summary="Removed 64 messages",
            messages_removed=64,
        )
        assert "compacted" in reminder.lower() or "removed" in reminder.lower()

    def test_post_compaction_includes_task_guidance(self):
        """Post-compaction reminder should include task-specific guidance."""
        reminder = get_post_compaction_reminder(
            task_type="analysis",
            compaction_summary="Removed 10 messages",
            messages_removed=10,
        )
        assert "analyzing" in reminder.lower() or "analysis" in reminder.lower()

    def test_post_compaction_with_deepseek_high_removal(self):
        """High message removal should get more detailed reminder."""
        # DeepSeek with 64+ messages removed
        reminder = get_post_compaction_reminder(
            task_type="coding",
            compaction_summary="Removed 64 messages",
            messages_removed=64,
        )
        assert len(reminder) > 50  # Should provide substantial guidance

    def test_post_compaction_with_low_removal(self):
        """Low message removal should get simpler reminder."""
        reminder = get_post_compaction_reminder(
            task_type="analysis",
            compaction_summary="Removed 5 messages",
            messages_removed=5,
        )
        # Should be brief (< 200 chars for simple reminder)
        assert len(reminder) < 200


class TestReminderIntegration:
    """Integration tests for reminder system with compaction."""

    def test_reminder_frequency_capping(self):
        """Reminder frequency should be capped per reminder type."""
        # This tests that we don't spam reminders
        # Frequency capping is handled by ReminderConfig.frequency
        # Here we just verify the config exists
        from victor.agent.context_reminder import ReminderConfig

        config = ReminderConfig(
            enabled=True,
            frequency=3,  # Inject every 3 tool calls
            priority=50,
        )
        assert config.frequency == 3
        assert config.enabled is True

    def test_compaction_reminder_config_exists(self):
        """COMPACTION reminder type should exist."""
        from victor.agent.context_reminder import ReminderType

        assert hasattr(ReminderType, "COMPACTION")
        assert ReminderType.COMPACTION.value == "compaction"

    def test_compaction_reminder_has_default_config(self):
        """COMPACTION reminder should have sensible default configuration."""
        from victor.agent.context_reminder import ContextReminderManager, ReminderType

        # Should have config for COMPACTION reminder
        configs = ContextReminderManager.DEFAULT_CONFIGS
        assert ReminderType.COMPACTION in configs
        config = configs[ReminderType.COMPACTION]

        assert config.enabled is True  # Should be enabled by default
        assert config.frequency >= 1  # Should have a frequency
        assert config.priority > 0  # Should have priority

    def test_reminder_priority_ordering(self):
        """Reminders should be prioritizable."""
        from victor.agent.context_reminder import ReminderConfig

        # Create configs with different priorities
        high_priority = ReminderConfig(frequency=1, priority=90)
        low_priority = ReminderConfig(frequency=1, priority=10)

        assert high_priority.priority > low_priority.priority
