# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
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

"""Tests for context reminder manager and related functionality."""

import pytest

from victor.agent.context_reminder import (
    ReminderType,
    ReminderConfig,
    ContextState,
    ContextReminderManager,
    create_reminder_manager,
    get_evidence_reminder,
)

# =============================================================================
# REMINDER TYPE TESTS
# =============================================================================


class TestReminderType:
    """Tests for ReminderType enum."""

    def test_enum_values(self):
        """Test all expected enum values exist."""
        assert ReminderType.EVIDENCE.value == "evidence"
        assert ReminderType.BUDGET.value == "budget"
        assert ReminderType.TASK_HINT.value == "task_hint"
        assert ReminderType.GROUNDING.value == "grounding"
        assert ReminderType.PROGRESS.value == "progress"
        assert ReminderType.CUSTOM.value == "custom"


# =============================================================================
# REMINDER CONFIG TESTS
# =============================================================================


class TestReminderConfig:
    """Tests for ReminderConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ReminderConfig()
        assert config.enabled is True
        assert config.frequency == 1
        assert config.priority == 50
        assert config.max_tokens == 0
        assert config.provider_overrides == {}

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ReminderConfig(
            enabled=False,
            frequency=5,
            priority=90,
            max_tokens=100,
            provider_overrides={"ollama": 2},
        )
        assert config.enabled is False
        assert config.frequency == 5
        assert config.priority == 90
        assert config.max_tokens == 100
        assert config.provider_overrides["ollama"] == 2

    def test_get_frequency_for_provider_with_override(self):
        """Test getting frequency with provider override."""
        config = ReminderConfig(
            frequency=5,
            provider_overrides={"ollama": 2, "anthropic": 6},
        )
        assert config.get_frequency_for_provider("ollama") == 2
        assert config.get_frequency_for_provider("anthropic") == 6
        assert config.get_frequency_for_provider("OLLAMA") == 2  # Case insensitive

    def test_get_frequency_for_provider_without_override(self):
        """Test getting frequency without provider override."""
        config = ReminderConfig(frequency=5)
        assert config.get_frequency_for_provider("unknown") == 5
        assert config.get_frequency_for_provider("openai") == 5


# =============================================================================
# CONTEXT STATE TESTS
# =============================================================================


class TestContextState:
    """Tests for ContextState dataclass."""

    def test_default_values(self):
        """Test default state values."""
        state = ContextState()
        assert state.observed_files == set()
        assert state.executed_tools == []
        assert state.tool_calls_made == 0
        # No default budget: the runtime supplies the enforced number via
        # update_state(). A hardcoded default here gets stated to the model as fact.
        assert state.tool_budget is None
        assert state.task_complexity == "medium"
        assert state.task_hint == ""
        assert state.last_reminder_at == 0
        assert state.reminder_history == {}

    def test_custom_values(self):
        """Test custom state values."""
        state = ContextState(
            observed_files={"file1.py", "file2.py"},
            executed_tools=["read_file", "search"],
            tool_calls_made=5,
            tool_budget=20,
            task_complexity="high",
            task_hint="Focus on performance",
        )
        assert len(state.observed_files) == 2
        assert len(state.executed_tools) == 2
        assert state.tool_calls_made == 5
        assert state.tool_budget == 20
        assert state.task_complexity == "high"
        assert state.task_hint == "Focus on performance"


# =============================================================================
# CONTEXT REMINDER MANAGER TESTS
# =============================================================================


class TestContextReminderManager:
    """Tests for ContextReminderManager."""

    @pytest.fixture
    def manager(self):
        """Create a default manager."""
        return ContextReminderManager()

    @pytest.fixture
    def google_manager(self):
        """Create a manager for Google provider."""
        return ContextReminderManager(provider="google")

    @pytest.fixture
    def ollama_manager(self):
        """Create a manager for Ollama provider."""
        return ContextReminderManager(provider="ollama")

    def test_init_default(self, manager):
        """Test default initialization."""
        assert manager.provider == "unknown"
        assert manager.state is not None
        assert manager._reminder_count == 0

    def test_init_with_provider(self, google_manager):
        """Test initialization with provider."""
        assert google_manager.provider == "google"

    def test_init_with_custom_configs(self):
        """Test initialization with custom configs."""
        custom_config = {ReminderType.EVIDENCE: ReminderConfig(frequency=10)}
        manager = ContextReminderManager(configs=custom_config)
        assert manager.configs[ReminderType.EVIDENCE].frequency == 10

    def test_reset(self, manager):
        """Test state reset."""
        manager.update_state(tool_calls=5, observed_files={"test.py"})
        manager._reminder_count = 10

        manager.reset()

        assert manager.state.tool_calls_made == 0
        assert manager.state.observed_files == set()
        assert manager._reminder_count == 0

    def test_update_state_observed_files(self, manager):
        """Test updating observed files."""
        manager.update_state(observed_files={"file1.py", "file2.py"})
        assert "file1.py" in manager.state.observed_files
        assert "file2.py" in manager.state.observed_files

    def test_update_state_executed_tool(self, manager):
        """Test updating executed tools."""
        manager.update_state(executed_tool="read_file")
        manager.update_state(executed_tool="search")

        assert manager.state.executed_tools == ["read_file", "search"]

    def test_update_state_tool_calls(self, manager):
        """Test updating tool call count."""
        manager.update_state(tool_calls=5)
        assert manager.state.tool_calls_made == 5

    def test_update_state_tool_budget(self, manager):
        """Test updating tool budget."""
        manager.update_state(tool_budget=20)
        assert manager.state.tool_budget == 20

    def test_update_state_task_complexity(self, manager):
        """Test updating task complexity."""
        manager.update_state(task_complexity="high")
        assert manager.state.task_complexity == "high"

    def test_update_state_task_hint(self, manager):
        """Test updating task hint."""
        manager.update_state(task_hint="Focus on testing")
        assert manager.state.task_hint == "Focus on testing"

    def test_add_observed_file(self, manager):
        """Test adding a single observed file."""
        manager.add_observed_file("test.py")
        manager.add_observed_file("main.py")

        assert "test.py" in manager.state.observed_files
        assert "main.py" in manager.state.observed_files


class TestShouldInjectReminder:
    """Tests for should_inject_reminder method."""

    def test_disabled_reminder(self):
        """Test disabled reminder not injected."""
        config = {ReminderType.EVIDENCE: ReminderConfig(enabled=False)}
        manager = ContextReminderManager(configs=config)

        assert manager.should_inject_reminder(ReminderType.EVIDENCE) is False

    def test_missing_config(self):
        """Test missing config returns False."""
        manager = ContextReminderManager(configs={})
        assert manager.should_inject_reminder(ReminderType.EVIDENCE) is False

    def test_task_hint_only_once(self):
        """Test task hint is only injected once."""
        manager = ContextReminderManager()
        manager.update_state(task_hint="Focus on testing")

        # First check should be True
        assert manager.should_inject_reminder(ReminderType.TASK_HINT) is True

        # Mark as injected
        manager.state.reminder_history[ReminderType.TASK_HINT] = "Focus on testing"

        # Second check should be False
        assert manager.should_inject_reminder(ReminderType.TASK_HINT) is False

    def test_task_hint_not_set(self):
        """Test task hint not injected if not set."""
        manager = ContextReminderManager()
        # Returns empty string (falsy) when no task hint
        assert not manager.should_inject_reminder(ReminderType.TASK_HINT)

    def test_budget_reminder_when_low(self):
        """Test budget reminder injected when running low."""
        manager = ContextReminderManager()
        manager.update_state(tool_budget=10, tool_calls=6)

        # 4 remaining - should inject
        assert manager.should_inject_reminder(ReminderType.BUDGET) is True

    def test_budget_reminder_when_high(self):
        """Test budget reminder not injected when plenty remaining."""
        manager = ContextReminderManager()
        manager.update_state(tool_budget=10, tool_calls=2)

        # 8 remaining - should not inject
        assert manager.should_inject_reminder(ReminderType.BUDGET) is False

    def test_budget_reminder_when_exhausted(self):
        """Test budget reminder not injected when exhausted."""
        manager = ContextReminderManager()
        manager.update_state(tool_budget=10, tool_calls=10)

        # 0 remaining - should not inject (nothing to do)
        assert manager.should_inject_reminder(ReminderType.BUDGET) is False

    def test_frequency_based_injection(self):
        """Test frequency-based reminder injection."""
        config = {ReminderType.EVIDENCE: ReminderConfig(frequency=3, enabled=True)}
        manager = ContextReminderManager(configs=config)

        # At start (0 calls since last reminder)
        assert manager.should_inject_reminder(ReminderType.EVIDENCE) is False

        # After 3 tool calls
        manager.update_state(tool_calls=3)
        assert manager.should_inject_reminder(ReminderType.EVIDENCE) is True


class TestReminderFormatters:
    """Tests for reminder formatting methods."""

    def test_format_evidence_reminder_with_files(self):
        """Test evidence reminder with files."""
        manager = ContextReminderManager()
        manager.update_state(observed_files={"main.py", "test.py"})

        reminder = manager._format_evidence_reminder()

        assert "[FILES:" in reminder
        assert "main.py" in reminder
        assert "test.py" in reminder

    def test_format_evidence_reminder_no_files(self):
        """Test evidence reminder without files."""
        manager = ContextReminderManager()

        reminder = manager._format_evidence_reminder()

        # Phrased as an actionable status (not a bare flag a small model misreads
        # as a prohibition). Assert intent rather than exact wording.
        assert "No files read yet" in reminder
        assert "read" in reminder.lower()

    def test_format_evidence_reminder_many_files(self):
        """Test evidence reminder truncates many files."""
        manager = ContextReminderManager()
        files = {f"file{i}.py" for i in range(15)}
        manager.update_state(observed_files=files)

        reminder = manager._format_evidence_reminder()

        assert "... and 5 more" in reminder

    def test_format_budget_reminder_critical(self):
        """Test budget reminder when critical."""
        manager = ContextReminderManager()
        manager.update_state(tool_budget=10, tool_calls=8)

        reminder = manager._format_budget_reminder()

        assert "2 tool calls remaining" in reminder
        assert "wrap up soon" in reminder

    def test_format_budget_reminder_low(self):
        """Test budget reminder when low but not critical."""
        manager = ContextReminderManager()
        manager.update_state(tool_budget=10, tool_calls=5)

        reminder = manager._format_budget_reminder()

        assert "5 tool calls remaining" in reminder
        assert "wrap up" not in reminder

    def test_format_budget_reminder_plenty(self):
        """Test budget reminder when plenty remaining."""
        manager = ContextReminderManager()
        manager.update_state(tool_budget=10, tool_calls=2)

        reminder = manager._format_budget_reminder()

        assert reminder == ""

    def test_format_task_hint(self):
        """Test task hint formatting."""
        manager = ContextReminderManager()
        manager.update_state(task_hint="  Focus on testing  ")

        hint = manager._format_task_hint()

        assert hint == "Focus on testing"

    def test_format_task_hint_empty(self):
        """Test task hint formatting when empty."""
        manager = ContextReminderManager()

        hint = manager._format_task_hint()

        assert hint == ""

    def test_format_progress_reminder(self):
        """Test progress reminder formatting."""
        manager = ContextReminderManager()
        manager.update_state(executed_tool="read_file")
        manager.update_state(executed_tool="search")
        manager.update_state(executed_tool="edit")

        reminder = manager._format_progress_reminder()

        assert "[Progress:" in reminder
        assert "3 tools used" in reminder
        assert "read_file" in reminder
        assert "search" in reminder
        assert "edit" in reminder

    def test_format_progress_reminder_empty(self):
        """Test progress reminder when no tools executed."""
        manager = ContextReminderManager()

        reminder = manager._format_progress_reminder()

        assert reminder == ""

    def test_format_grounding_reminder(self):
        """Test grounding reminder formatting."""
        manager = ContextReminderManager()

        reminder = manager._format_grounding_reminder()

        assert "Ground responses" in reminder
        assert "tool output only" in reminder


class TestGetReminder:
    """Tests for get_reminder method."""

    def test_get_reminder_not_needed(self):
        """Test get_reminder returns None when not needed."""
        manager = ContextReminderManager()

        reminder = manager.get_reminder(ReminderType.TASK_HINT)

        assert reminder is None

    def test_get_reminder_custom_formatter(self):
        """Test get_reminder with custom formatter."""

        def custom_formatter(state):
            return f"Custom: {len(state.observed_files)} files"

        manager = ContextReminderManager(custom_formatters={ReminderType.CUSTOM: custom_formatter})
        manager.configs[ReminderType.CUSTOM] = ReminderConfig(enabled=True)
        manager.update_state(observed_files={"a.py", "b.py"}, tool_calls=1)

        reminder = manager.get_reminder(ReminderType.CUSTOM)

        assert reminder == "Custom: 2 files"

    def test_get_reminder_updates_history(self):
        """Test get_reminder updates history."""
        manager = ContextReminderManager()
        manager.update_state(task_hint="Testing hint")

        reminder = manager.get_reminder(ReminderType.TASK_HINT)

        assert ReminderType.TASK_HINT in manager.state.reminder_history
        assert manager.state.reminder_history[ReminderType.TASK_HINT] == reminder

    def test_get_reminder_no_repeat(self):
        """Test get_reminder doesn't repeat same content."""
        manager = ContextReminderManager()
        manager.update_state(observed_files={"test.py"}, tool_calls=3)

        # First reminder
        reminder1 = manager.get_reminder(ReminderType.EVIDENCE)
        assert reminder1 is not None

        # Update call count but same files
        manager.update_state(tool_calls=6)

        # Second reminder should be None (no change)
        reminder2 = manager.get_reminder(ReminderType.EVIDENCE)
        assert reminder2 is None


class TestGetConsolidatedReminder:
    """Tests for get_consolidated_reminder method."""

    def test_consolidated_reminder_empty(self):
        """Test consolidated reminder when none needed."""
        manager = ContextReminderManager()

        reminder = manager.get_consolidated_reminder()

        assert reminder is None

    def test_consolidated_reminder_single(self):
        """Test consolidated reminder with single item."""
        manager = ContextReminderManager()
        manager.update_state(task_hint="Focus on testing")

        reminder = manager.get_consolidated_reminder()

        assert reminder is not None
        assert "Focus on testing" in reminder

    def test_consolidated_reminder_multiple(self):
        """Test consolidated reminder combines multiple."""
        manager = ContextReminderManager()
        manager.update_state(
            task_hint="Focus on testing",
            tool_budget=10,
            tool_calls=7,
            observed_files={"test.py"},
        )

        reminder = manager.get_consolidated_reminder(force=True)

        assert reminder is not None
        assert " | " in reminder  # Multiple parts separated

    def test_consolidated_reminder_force(self):
        """Test force parameter bypasses frequency check in loop."""
        manager = ContextReminderManager()
        # Set conditions that would normally trigger reminders
        manager.update_state(
            observed_files={"test.py"},
            tool_budget=10,
            tool_calls=7,  # Low budget triggers reminder
        )

        reminder = manager.get_consolidated_reminder(force=True)

        # With low budget, budget reminder should trigger
        assert reminder is not None

    def test_consolidated_reminder_updates_tracking(self):
        """Test consolidated reminder updates tracking."""
        manager = ContextReminderManager()
        manager.update_state(task_hint="Testing", tool_calls=5)

        manager.get_consolidated_reminder()

        assert manager.state.last_reminder_at == 5
        assert manager._reminder_count == 1


class TestGetMinimalReminder:
    """Tests for get_minimal_reminder method."""

    def test_minimal_reminder_empty(self):
        """Test minimal reminder when nothing critical."""
        manager = ContextReminderManager()
        manager.update_state(tool_budget=10, tool_calls=2)

        reminder = manager.get_minimal_reminder()

        assert reminder is None

    def test_minimal_reminder_low_budget(self):
        """Test minimal reminder with low budget."""
        manager = ContextReminderManager()
        manager.update_state(tool_budget=10, tool_calls=8)

        reminder = manager.get_minimal_reminder()

        assert reminder is not None
        assert "2 calls left" in reminder

    def test_minimal_reminder_files_read(self):
        """Test minimal reminder with files read."""
        manager = ContextReminderManager()
        manager.update_state(observed_files={"a.py", "b.py", "c.py"})

        reminder = manager.get_minimal_reminder()

        assert reminder is not None
        assert "3 files read" in reminder

    def test_minimal_reminder_both(self):
        """Test minimal reminder with both conditions."""
        manager = ContextReminderManager()
        manager.update_state(
            tool_budget=10,
            tool_calls=8,
            observed_files={"a.py", "b.py"},
        )

        reminder = manager.get_minimal_reminder()

        assert reminder is not None
        assert "2 calls left" in reminder
        assert "2 files read" in reminder
        assert " | " in reminder


class TestConfigureForProvider:
    """Tests for configure_for_provider method."""

    def test_configure_cloud_provider(self):
        """Test configuration for cloud providers."""
        manager = ContextReminderManager()
        manager.configure_for_provider("anthropic")

        assert manager.provider == "anthropic"
        assert manager.configs[ReminderType.EVIDENCE].frequency == 4
        assert manager.configs[ReminderType.GROUNDING].frequency == 15

    def test_configure_local_provider(self):
        """Test configuration for local providers."""
        manager = ContextReminderManager()
        manager.configure_for_provider("ollama")

        assert manager.provider == "ollama"
        assert manager.configs[ReminderType.EVIDENCE].frequency == 2
        assert manager.configs[ReminderType.GROUNDING].frequency == 5


class TestGetStats:
    """Tests for get_stats method."""

    def test_stats_initial(self):
        """Test initial stats."""
        manager = ContextReminderManager(provider="openai")

        stats = manager.get_stats()

        assert stats["total_reminders"] == 0
        assert stats["tool_calls_made"] == 0
        assert stats["files_observed"] == 0
        assert stats["tools_executed"] == 0
        assert stats["provider"] == "openai"

    def test_stats_after_activity(self):
        """Test stats after activity."""
        manager = ContextReminderManager()
        manager.update_state(
            observed_files={"a.py", "b.py"},
            tool_calls=5,
            task_hint="Testing",
        )
        manager.update_state(executed_tool="read_file")
        manager.update_state(executed_tool="search")
        manager.get_consolidated_reminder()

        stats = manager.get_stats()

        assert stats["total_reminders"] == 1
        assert stats["tool_calls_made"] == 5
        assert stats["files_observed"] == 2
        assert stats["tools_executed"] == 2


# =============================================================================
# CONVENIENCE FUNCTION TESTS
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_reminder_manager(self):
        """Test create_reminder_manager function."""
        manager = create_reminder_manager(
            provider="google",
            task_complexity="high",
            tool_budget=20,
        )

        assert manager.provider == "google"
        assert manager.state.task_complexity == "high"
        assert manager.state.tool_budget == 20

    def test_get_evidence_reminder(self):
        """Test get_evidence_reminder function."""
        files = {"main.py", "test.py"}

        reminder = get_evidence_reminder(files, provider="openai")

        assert "[FILES:" in reminder
        assert "main.py" in reminder
        assert "test.py" in reminder

    def test_get_evidence_reminder_empty(self):
        """Test get_evidence_reminder with empty files."""
        reminder = get_evidence_reminder(set())

        assert "No files read yet" in reminder


# =============================================================================
# EDGE CASES
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_provider_case_insensitive(self):
        """Test provider name is case insensitive."""
        manager = ContextReminderManager(provider="GOOGLE")
        assert manager.provider == "google"

    def test_custom_formatter_none_return(self):
        """Test custom formatter returning empty string."""

        def empty_formatter(state):
            return ""

        manager = ContextReminderManager(custom_formatters={ReminderType.CUSTOM: empty_formatter})
        manager.configs[ReminderType.CUSTOM] = ReminderConfig(enabled=True)
        manager.update_state(tool_calls=1)

        reminder = manager.get_reminder(ReminderType.CUSTOM)

        assert reminder is None

    def test_budget_exactly_at_threshold(self):
        """Test budget reminder at exact threshold."""
        manager = ContextReminderManager()
        manager.update_state(tool_budget=10, tool_calls=5)

        # 5 remaining - at threshold
        assert manager.should_inject_reminder(ReminderType.BUDGET) is True

    def test_progress_shows_recent_tools(self):
        """Test progress only shows recent 3 tools."""
        manager = ContextReminderManager()
        for i in range(5):
            manager.update_state(executed_tool=f"tool_{i}")

        reminder = manager._format_progress_reminder()

        assert "5 tools used" in reminder
        assert "tool_2" in reminder
        assert "tool_3" in reminder
        assert "tool_4" in reminder
        assert "tool_0" not in reminder  # Not in recent 3

    def test_evidence_files_sorted(self):
        """Test evidence files are sorted."""
        manager = ContextReminderManager()
        manager.update_state(observed_files={"z.py", "a.py", "m.py"})

        reminder = manager._format_evidence_reminder()

        # Check sorted order
        idx_a = reminder.index("a.py")
        idx_m = reminder.index("m.py")
        idx_z = reminder.index("z.py")
        assert idx_a < idx_m < idx_z


# =============================================================================
# BUDGET TRUTHFULNESS REGRESSION TESTS
# =============================================================================
#
# Regression cover for the sandhi-cdfbc589 refusal (2026-07-26): the reminder
# manager was constructed with a hardcoded budget of 10 that no enforcer used,
# then counted down against it in messages injected into the model's context.
# The model measured the claim as false and stopped working.


class TestBudgetIsNeverInvented:
    """The reminder must state a budget only when the runtime supplied one."""

    def test_no_budget_reminder_when_budget_unknown(self):
        """With no budget supplied, no countdown is emitted at any call count."""
        manager = ContextReminderManager()

        for calls in (0, 1, 5, 9, 50, 500):
            manager.update_state(tool_calls=calls)
            assert manager._format_budget_reminder() == ""
            assert manager.should_inject_reminder(ReminderType.BUDGET) is False

    def test_minimal_reminder_omits_budget_when_unknown(self):
        """get_minimal_reminder must not fabricate a 'calls left' figure."""
        manager = ContextReminderManager()
        manager.update_state(tool_calls=100, observed_files={"a.py"})

        minimal = manager.get_minimal_reminder()

        assert minimal is not None
        assert "calls left" not in minimal
        assert "1 files read" in minimal

    def test_consolidated_reminder_has_no_budget_claim_when_unknown(self):
        """The consolidated string the model actually sees carries no budget."""
        manager = ContextReminderManager()
        manager.update_state(tool_calls=9, executed_tools=["read", "shell"])

        reminder = manager.get_consolidated_reminder(force=True) or ""

        assert "remaining" not in reminder
        assert "wrap up" not in reminder

    def test_budget_reminder_returns_once_runtime_supplies_one(self):
        """Supplying the enforced budget re-enables the countdown."""
        manager = ContextReminderManager()
        manager.update_state(tool_calls=18, tool_budget=20)

        assert "2 tool calls remaining" in manager._format_budget_reminder()

    def test_factory_does_not_default_a_budget(self):
        """create_reminder_manager must not seed a budget of its own."""
        manager = create_reminder_manager(provider="zai", task_complexity="simple")

        assert manager.state.tool_budget is None


class TestProgressCountMatchesCallCount:
    """Progress must report the same quantity the enforcer counts."""

    def test_executed_tools_records_every_tool_in_a_parallel_batch(self):
        """A batch of N calls contributes N names, not one."""
        manager = ContextReminderManager()

        manager.update_state(executed_tools=["read"], tool_calls=1)
        manager.update_state(executed_tools=["code", "code", "code"], tool_calls=4)
        manager.update_state(executed_tools=["shell", "shell"], tool_calls=6)

        reminder = manager._format_progress_reminder()

        assert "6 tools used" in reminder
        assert manager.state.tool_calls_made == 6
        assert len(manager.state.executed_tools) == manager.state.tool_calls_made

    def test_singular_executed_tool_still_supported(self):
        """The single-name form remains valid for non-batched callers."""
        manager = ContextReminderManager()
        manager.update_state(executed_tool="read", tool_calls=1)

        assert manager.state.executed_tools == ["read"]
