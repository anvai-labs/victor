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

"""Tests for BudgetManager and related classes."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from victor.agent.protocols import (
    BudgetConfig,
    BudgetStatus,
    BudgetType,
    IBudgetManager,
)
from victor.agent.budget_manager import (
    BudgetManager,
    BudgetState,
    WRITE_TOOLS,
    is_write_tool,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def default_config():
    """Create a default BudgetConfig."""
    return BudgetConfig()


@pytest.fixture
def custom_config():
    """Create a custom BudgetConfig."""
    return BudgetConfig(
        base_tool_calls=50,
        base_iterations=100,
        base_exploration=30,
        base_action=20,
    )


@pytest.fixture
def manager(default_config):
    """Create a BudgetManager with default config."""
    return BudgetManager(config=default_config)


@pytest.fixture
def custom_manager(custom_config):
    """Create a BudgetManager with custom config."""
    return BudgetManager(config=custom_config)


# =============================================================================
# is_write_tool Tests
# =============================================================================


class TestIsWriteTool:
    """Tests for is_write_tool function."""

    def test_recognizes_write_file(self):
        """Test recognizes write_file as write tool."""
        assert is_write_tool("write_file") is True

    def test_recognizes_shell(self):
        """Test recognizes shell as write tool."""
        assert is_write_tool("shell") is True

    def test_recognizes_edit(self):
        """Test recognizes edit as write tool."""
        assert is_write_tool("edit") is True

    def test_rejects_read_file(self):
        """Test rejects read_file as non-write tool."""
        assert is_write_tool("read_file") is False

    def test_case_insensitive(self):
        """Test is case insensitive."""
        assert is_write_tool("SHELL") is True
        assert is_write_tool("Write_File") is True

    def test_write_tools_set(self):
        """Test WRITE_TOOLS contains expected tools."""
        assert "write" in WRITE_TOOLS
        assert "shell" in WRITE_TOOLS
        assert "edit" in WRITE_TOOLS
        assert "git_commit" in WRITE_TOOLS

    def test_recognizes_execute_bash_alias(self):
        """Legacy execute_bash alias should still resolve as a write tool."""
        assert is_write_tool("execute_bash") is True

    def test_recognizes_notebook_edit_as_write_tool(self):
        """Notebook edits should be treated as write operations consistently."""
        assert is_write_tool("notebook_edit") is True


# =============================================================================
# BudgetState Tests
# =============================================================================


class TestBudgetState:
    """Tests for BudgetState dataclass."""

    def test_default_values(self):
        """Test default values."""
        state = BudgetState()
        assert state.current == 0
        assert state.base_maximum == 0
        assert state.last_tool is None

    def test_with_values(self):
        """Test with explicit values."""
        state = BudgetState(current=5, base_maximum=20, last_tool="read_file")
        assert state.current == 5
        assert state.base_maximum == 20
        assert state.last_tool == "read_file"


# =============================================================================
# BudgetStatus Tests
# =============================================================================


class TestBudgetStatus:
    """Tests for BudgetStatus dataclass."""

    def test_default_values(self):
        """Test default values."""
        status = BudgetStatus(budget_type=BudgetType.TOOL_CALLS)
        assert status.current == 0
        assert status.effective_maximum == 0
        assert status.is_exhausted is False

    def test_remaining_property(self):
        """Test remaining property calculates correctly."""
        status = BudgetStatus(
            budget_type=BudgetType.TOOL_CALLS,
            current=5,
            effective_maximum=20,
        )
        assert status.remaining == 15

    def test_remaining_never_negative(self):
        """Test remaining never goes negative."""
        status = BudgetStatus(
            budget_type=BudgetType.TOOL_CALLS,
            current=25,
            effective_maximum=20,
        )
        assert status.remaining == 0


# =============================================================================
# BudgetManager Initialization Tests
# =============================================================================


class TestBudgetManagerInit:
    """Tests for BudgetManager initialization."""

    def test_initializes_all_budget_types(self, manager):
        """Test all budget types are initialized."""
        assert BudgetType.TOOL_CALLS in manager._budgets
        assert BudgetType.ITERATIONS in manager._budgets
        assert BudgetType.EXPLORATION in manager._budgets
        assert BudgetType.ACTION in manager._budgets

    def test_default_multipliers(self, manager):
        """Test default multipliers are 1.0."""
        assert manager._model_multiplier == 1.0
        assert manager._mode_multiplier == 1.0
        assert manager._productivity_multiplier == 1.0

    def test_custom_config_applied(self, custom_manager):
        """Test custom config is applied."""
        tool_status = custom_manager.get_status(BudgetType.TOOL_CALLS)
        assert tool_status.base_maximum == 50

    def test_callback_not_set(self, manager):
        """Test callback not set by default."""
        assert manager._on_exhausted is None


# =============================================================================
# BudgetManager.get_status Tests
# =============================================================================


class TestBudgetManagerGetStatus:
    """Tests for BudgetManager.get_status() method."""

    def test_returns_correct_type(self, manager):
        """Test returns BudgetStatus."""
        status = manager.get_status(BudgetType.TOOL_CALLS)
        assert isinstance(status, BudgetStatus)
        assert status.budget_type == BudgetType.TOOL_CALLS

    def test_current_starts_at_zero(self, manager):
        """Test current starts at zero."""
        status = manager.get_status(BudgetType.TOOL_CALLS)
        assert status.current == 0

    def test_is_exhausted_false_initially(self, manager):
        """Test is_exhausted is False initially."""
        status = manager.get_status(BudgetType.TOOL_CALLS)
        assert status.is_exhausted is False

    def test_reflects_multipliers(self, manager):
        """Test status reflects current multipliers."""
        manager.set_model_multiplier(1.5)
        status = manager.get_status(BudgetType.TOOL_CALLS)
        assert status.model_multiplier == 1.5


# =============================================================================
# BudgetManager.consume Tests
# =============================================================================


class TestBudgetManagerConsume:
    """Tests for BudgetManager.consume() method."""

    def test_consume_increases_current(self, manager):
        """Test consume increases current count."""
        initial = manager.get_status(BudgetType.TOOL_CALLS).current
        manager.consume(BudgetType.TOOL_CALLS)
        after = manager.get_status(BudgetType.TOOL_CALLS).current
        assert after == initial + 1

    def test_consume_custom_amount(self, manager):
        """Test consume with custom amount."""
        manager.consume(BudgetType.TOOL_CALLS, amount=5)
        status = manager.get_status(BudgetType.TOOL_CALLS)
        assert status.current == 5

    def test_returns_true_when_available(self, manager):
        """Test returns True when budget available."""
        result = manager.consume(BudgetType.TOOL_CALLS)
        assert result is True

    def test_returns_false_when_exhausted(self, custom_manager):
        """Test returns False when budget exhausted."""
        # Exhaust the budget
        custom_manager._budgets[BudgetType.TOOL_CALLS].current = 50
        result = custom_manager.consume(BudgetType.TOOL_CALLS)
        assert result is False

    def test_calls_on_exhausted_callback(self, custom_manager):
        """Test calls on_exhausted callback when budget exhausted."""
        callback = MagicMock()
        custom_manager._on_exhausted = callback

        # Consume until exhausted
        custom_manager._budgets[BudgetType.TOOL_CALLS].current = 49
        custom_manager.consume(BudgetType.TOOL_CALLS)

        callback.assert_called_once_with(BudgetType.TOOL_CALLS)


# =============================================================================
# BudgetManager.is_exhausted Tests
# =============================================================================


class TestBudgetManagerIsExhausted:
    """Tests for BudgetManager.is_exhausted() method."""

    def test_false_when_available(self, manager):
        """Test returns False when budget available."""
        assert manager.is_exhausted(BudgetType.TOOL_CALLS) is False

    def test_true_when_exhausted(self, custom_manager):
        """Test returns True when budget exhausted."""
        # Exhaust the budget
        custom_manager._budgets[BudgetType.TOOL_CALLS].current = 50
        assert custom_manager.is_exhausted(BudgetType.TOOL_CALLS) is True


# =============================================================================
# BudgetManager Multiplier Tests
# =============================================================================


class TestBudgetManagerMultipliers:
    """Tests for BudgetManager multiplier methods."""

    def test_set_model_multiplier(self, manager):
        """Test set_model_multiplier updates value."""
        manager.set_model_multiplier(1.5)
        assert manager._model_multiplier == 1.5

    def test_model_multiplier_clamped_min(self, manager):
        """Test model multiplier clamped to minimum."""
        manager.set_model_multiplier(0.1)
        assert manager._model_multiplier == 0.5

    def test_model_multiplier_clamped_max(self, manager):
        """Test model multiplier clamped to maximum."""
        manager.set_model_multiplier(10.0)
        assert manager._model_multiplier == 3.0

    def test_set_mode_multiplier(self, manager):
        """Test set_mode_multiplier updates value."""
        manager.set_mode_multiplier(2.5)
        assert manager._mode_multiplier == 2.5

    def test_mode_multiplier_clamped_min(self, manager):
        """Test mode multiplier clamped to minimum."""
        manager.set_mode_multiplier(0.1)
        assert manager._mode_multiplier == 0.5

    def test_mode_multiplier_clamped_max(self, manager):
        """Test mode multiplier clamped to maximum."""
        manager.set_mode_multiplier(10.0)
        assert manager._mode_multiplier == 5.0

    def test_set_productivity_multiplier(self, manager):
        """Test set_productivity_multiplier updates value."""
        manager.set_productivity_multiplier(0.8)
        assert manager._productivity_multiplier == 0.8

    def test_productivity_multiplier_clamped_min(self, manager):
        """Test productivity multiplier clamped to minimum."""
        manager.set_productivity_multiplier(0.1)
        assert manager._productivity_multiplier == 0.5

    def test_productivity_multiplier_clamped_max(self, manager):
        """Test productivity multiplier clamped to maximum."""
        manager.set_productivity_multiplier(10.0)
        assert manager._productivity_multiplier == 3.0


# =============================================================================
# BudgetManager Effective Max Tests
# =============================================================================


class TestBudgetManagerEffectiveMax:
    """Tests for effective maximum calculation."""

    def test_effective_max_with_multipliers(self, custom_manager):
        """Test effective max applies multipliers."""
        custom_manager.set_model_multiplier(1.5)
        custom_manager.set_mode_multiplier(2.0)
        custom_manager.set_productivity_multiplier(1.0)

        status = custom_manager.get_status(BudgetType.TOOL_CALLS)
        # 50 * 1.5 * 2.0 * 1.0 = 150
        assert status.effective_maximum == 150

    def test_effective_max_minimum_one(self, manager):
        """Test effective max is at least 1."""
        # Even with very low multipliers
        manager.set_model_multiplier(0.5)
        manager.set_mode_multiplier(0.5)
        manager.set_productivity_multiplier(0.5)

        status = manager.get_status(BudgetType.TOOL_CALLS)
        assert status.effective_maximum >= 1


# =============================================================================
# BudgetManager.reset Tests
# =============================================================================


class TestBudgetManagerReset:
    """Tests for BudgetManager.reset() method."""

    def test_reset_all_budgets(self, manager):
        """Test reset all budgets."""
        # Consume some budget
        manager.consume(BudgetType.TOOL_CALLS, amount=10)
        manager.consume(BudgetType.EXPLORATION, amount=5)

        # Reset all
        manager.reset()

        # Verify all reset
        assert manager.get_status(BudgetType.TOOL_CALLS).current == 0
        assert manager.get_status(BudgetType.EXPLORATION).current == 0

    def test_reset_specific_budget(self, manager):
        """Test reset specific budget."""
        # Consume some budgets
        manager.consume(BudgetType.TOOL_CALLS, amount=10)
        manager.consume(BudgetType.EXPLORATION, amount=5)

        # Reset only tool calls
        manager.reset(BudgetType.TOOL_CALLS)

        # Verify only tool calls reset
        assert manager.get_status(BudgetType.TOOL_CALLS).current == 0
        assert manager.get_status(BudgetType.EXPLORATION).current == 5


# =============================================================================
# BudgetManager.get_prompt_budget_info Tests
# =============================================================================


class TestBudgetManagerGetPromptBudgetInfo:
    """Tests for BudgetManager.get_prompt_budget_info() method."""

    def test_returns_dict(self, manager):
        """Test returns dictionary."""
        info = manager.get_prompt_budget_info()
        assert isinstance(info, dict)

    def test_contains_tool_budget_info(self, manager):
        """Test contains tool budget info."""
        info = manager.get_prompt_budget_info()
        assert "tool_budget" in info
        assert "tool_calls_used" in info
        assert "tool_calls_remaining" in info

    def test_contains_exploration_info(self, manager):
        """Test contains exploration info."""
        info = manager.get_prompt_budget_info()
        assert "exploration_budget" in info
        assert "exploration_used" in info
        assert "exploration_remaining" in info

    def test_contains_action_info(self, manager):
        """Test contains action info."""
        info = manager.get_prompt_budget_info()
        assert "action_budget" in info
        assert "action_used" in info

    def test_reflects_usage(self, manager):
        """Test info reflects actual usage."""
        manager.consume(BudgetType.TOOL_CALLS, amount=5)
        info = manager.get_prompt_budget_info()
        assert info["tool_calls_used"] == 5


# =============================================================================
# IBudgetManager Protocol Tests
# =============================================================================


class TestIBudgetManagerProtocol:
    """Tests for IBudgetManager protocol compliance."""

    def test_manager_implements_protocol(self, manager):
        """Test that BudgetManager implements IBudgetManager."""
        assert isinstance(manager, IBudgetManager)

    def test_protocol_has_required_methods(self):
        """Test that IBudgetManager defines required methods."""
        assert hasattr(IBudgetManager, "get_status")
        assert hasattr(IBudgetManager, "consume")
        assert hasattr(IBudgetManager, "set_model_multiplier")
        assert hasattr(IBudgetManager, "set_mode_multiplier")
        assert hasattr(IBudgetManager, "get_prompt_budget_info")


# =============================================================================
# BudgetType Tests
# =============================================================================


class TestBudgetType:
    """Tests for BudgetType enum."""

    def test_has_required_types(self):
        """Test has all required budget types."""
        assert hasattr(BudgetType, "TOOL_CALLS")
        assert hasattr(BudgetType, "ITERATIONS")
        assert hasattr(BudgetType, "EXPLORATION")
        assert hasattr(BudgetType, "ACTION")

    def test_values_are_strings(self):
        """Test values are strings."""
        assert BudgetType.TOOL_CALLS.value == "tool_calls"
        assert BudgetType.ITERATIONS.value == "iterations"


# =============================================================================
# Integration Tests
# =============================================================================


class TestBudgetManagerIntegration:
    """Integration tests for BudgetManager."""

    def test_exploration_action_separation(self, manager):
        """Test exploration and action budgets are separate."""
        manager.consume(BudgetType.EXPLORATION, amount=5)
        manager.consume(BudgetType.ACTION, amount=3)

        assert manager.get_status(BudgetType.EXPLORATION).current == 5
        assert manager.get_status(BudgetType.ACTION).current == 3

    def test_multiplier_affects_exhaustion(self, custom_manager):
        """Test multipliers affect when budget exhausts."""
        # Base is 50, with 2x mode multiplier = 100
        custom_manager.set_mode_multiplier(2.0)

        # Consume 60 (would exhaust at base 50)
        for _ in range(60):
            custom_manager.consume(BudgetType.TOOL_CALLS)

        # Should not be exhausted yet (effective max is 100)
        assert custom_manager.is_exhausted(BudgetType.TOOL_CALLS) is False

        # Consume 50 more (total 110)
        for _ in range(50):
            custom_manager.consume(BudgetType.TOOL_CALLS)

        # Now should be exhausted
        assert custom_manager.is_exhausted(BudgetType.TOOL_CALLS) is True

    def test_reset_preserves_multipliers(self, manager):
        """Test reset preserves multiplier settings."""
        manager.set_model_multiplier(1.5)
        manager.set_mode_multiplier(2.0)
        manager.consume(BudgetType.TOOL_CALLS, amount=10)

        manager.reset()

        assert manager._model_multiplier == 1.5
        assert manager._mode_multiplier == 2.0
        assert manager.get_status(BudgetType.TOOL_CALLS).current == 0


# =============================================================================
# BudgetManager.record_tool_call Tests
# =============================================================================


class TestBudgetManagerRecordToolCall:
    """Tests for BudgetManager.record_tool_call() method."""

    def test_record_read_tool(self, manager):
        """Test recording a read tool call."""
        result = manager.record_tool_call("read_file")
        assert result is True
        assert manager.get_status(BudgetType.TOOL_CALLS).current == 1
        assert manager.get_status(BudgetType.EXPLORATION).current == 1
        assert manager.get_status(BudgetType.ACTION).current == 0

    def test_record_write_tool(self, manager):
        """Test recording a write tool call."""
        result = manager.record_tool_call("write_file")
        assert result is True
        assert manager.get_status(BudgetType.TOOL_CALLS).current == 1
        assert manager.get_status(BudgetType.ACTION).current == 1
        assert manager.get_status(BudgetType.EXPLORATION).current == 0

    def test_record_shell_tool(self, manager):
        """Test recording shell tool as write operation."""
        result = manager.record_tool_call("shell")
        assert result is True
        assert manager.get_status(BudgetType.ACTION).current == 1

    def test_explicit_is_write_operation(self, manager):
        """Test explicit is_write_operation parameter."""
        result = manager.record_tool_call("custom_tool", is_write_operation=True)
        assert result is True
        assert manager.get_status(BudgetType.ACTION).current == 1

    def test_updates_last_tool(self, manager):
        """Test that record_tool_call updates last_tool."""
        manager.record_tool_call("semantic_search")
        state = manager._budgets[BudgetType.EXPLORATION]
        assert state.last_tool == "semantic_search"

    def test_returns_false_when_exhausted(self, custom_manager):
        """Test returns False when budget is exhausted."""
        # Exhaust tool calls budget
        custom_manager._budgets[BudgetType.TOOL_CALLS].current = 50
        result = custom_manager.record_tool_call("read_file")
        assert result is False


# =============================================================================
# BudgetManager.set_base_budget Tests
# =============================================================================


class TestBudgetManagerSetBaseBudget:
    """Tests for BudgetManager.set_base_budget() method."""

    def test_set_base_budget(self, manager):
        """Test setting base budget for a type."""
        manager.set_base_budget(BudgetType.EXPLORATION, 100)
        state = manager._budgets[BudgetType.EXPLORATION]
        assert state.base_maximum == 100

    def test_base_budget_minimum_one(self, manager):
        """Test base budget is at least 1."""
        manager.set_base_budget(BudgetType.ACTION, 0)
        state = manager._budgets[BudgetType.ACTION]
        assert state.base_maximum == 1

    def test_affects_effective_max(self, manager):
        """Test setting base budget affects effective max."""
        manager.set_base_budget(BudgetType.TOOL_CALLS, 200)
        status = manager.get_status(BudgetType.TOOL_CALLS)
        assert status.effective_maximum == 200


# =============================================================================
# BudgetManager.set_on_exhausted Tests
# =============================================================================


class TestBudgetManagerSetOnExhausted:
    """Tests for BudgetManager.set_on_exhausted() method."""

    def test_set_callback(self, manager):
        """Test setting exhaustion callback."""
        callback = MagicMock()
        manager.set_on_exhausted(callback)
        assert manager._on_exhausted is callback

    def test_callback_replaces_previous(self, manager):
        """Test new callback replaces previous."""
        callback1 = MagicMock()
        callback2 = MagicMock()
        manager.set_on_exhausted(callback1)
        manager.set_on_exhausted(callback2)
        assert manager._on_exhausted is callback2


# =============================================================================
# BudgetManager.update_from_mode Tests
# =============================================================================


class TestBudgetManagerUpdateFromMode:
    """Tests for BudgetManager.update_from_mode() method."""

    def test_uses_mixin_multiplier(self, manager):
        """Test update_from_mode uses ModeAwareMixin."""
        # Mock the exploration_multiplier property
        manager._mode_controller = MagicMock()
        manager._mode_controller.config.exploration_multiplier = 2.5

        manager.update_from_mode()
        assert manager._mode_multiplier == 2.5


# =============================================================================
# BudgetManager.get_diagnostics Tests
# =============================================================================


class TestBudgetManagerGetDiagnostics:
    """Tests for BudgetManager.get_diagnostics() method."""

    def test_returns_dict(self, manager):
        """Test returns dictionary."""
        diag = manager.get_diagnostics()
        assert isinstance(diag, dict)

    def test_contains_multipliers(self, manager):
        """Test contains multipliers section."""
        diag = manager.get_diagnostics()
        assert "multipliers" in diag
        assert "model" in diag["multipliers"]
        assert "mode" in diag["multipliers"]
        assert "productivity" in diag["multipliers"]
        assert "combined" in diag["multipliers"]

    def test_contains_budgets(self, manager):
        """Test contains budgets section."""
        diag = manager.get_diagnostics()
        assert "budgets" in diag
        assert "tool_calls" in diag["budgets"]
        assert "exploration" in diag["budgets"]
        assert "action" in diag["budgets"]

    def test_budget_details(self, manager):
        """Test budget details are included."""
        manager.consume(BudgetType.TOOL_CALLS, amount=5)
        diag = manager.get_diagnostics()

        tool_budget = diag["budgets"]["tool_calls"]
        assert tool_budget["current"] == 5
        assert "base_maximum" in tool_budget
        assert "effective_maximum" in tool_budget
        assert "remaining" in tool_budget
        assert "utilization" in tool_budget
        assert "is_exhausted" in tool_budget
        assert "last_tool" in tool_budget

    def test_combined_multiplier_calculation(self, manager):
        """Test combined multiplier is calculated correctly."""
        manager.set_model_multiplier(1.5)
        manager.set_mode_multiplier(2.0)
        manager.set_productivity_multiplier(1.0)

        diag = manager.get_diagnostics()
        assert diag["multipliers"]["combined"] == 3.0


# =============================================================================
# create_budget_manager Factory Tests
# =============================================================================


class TestCreateBudgetManager:
    """Tests for create_budget_manager factory function."""

    def test_creates_with_defaults(self):
        """Test creates manager with default config."""
        from victor.agent.budget_manager import create_budget_manager

        manager = create_budget_manager()
        assert isinstance(manager, BudgetManager)
        assert manager._model_multiplier == 1.0
        assert manager._mode_multiplier == 1.0

    def test_creates_with_custom_config(self):
        """Test creates manager with custom config."""
        from victor.agent.budget_manager import create_budget_manager

        config = BudgetConfig(base_tool_calls=100, base_exploration=50)
        manager = create_budget_manager(config=config)

        status = manager.get_status(BudgetType.TOOL_CALLS)
        assert status.base_maximum == 100

    def test_applies_multipliers(self):
        """Test applies initial multipliers."""
        from victor.agent.budget_manager import create_budget_manager

        manager = create_budget_manager(
            model_multiplier=1.5,
            mode_multiplier=2.0,
            productivity_multiplier=0.8,
        )

        assert manager._model_multiplier == 1.5
        assert manager._mode_multiplier == 2.0
        assert manager._productivity_multiplier == 0.8


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestBudgetManagerEdgeCases:
    """Tests for edge cases and error handling."""

    def test_unknown_budget_type_consume(self, manager):
        """Test consume with unknown budget type returns False."""
        # Create a mock budget type that's not in the manager
        # This is tricky since BudgetType is an enum
        # Instead, test by directly manipulating _budgets
        manager._budgets.clear()
        result = manager.consume(BudgetType.TOOL_CALLS)
        assert result is False

    def test_get_status_unknown_type(self, manager):
        """Test get_status with unknown budget type returns exhausted status."""
        manager._budgets.clear()
        status = manager.get_status(BudgetType.TOOL_CALLS)
        assert status.is_exhausted is True
        assert status.current == 0

    def test_calculate_effective_max_unknown_type(self, manager):
        """Test _calculate_effective_max with unknown type returns 0."""
        manager._budgets.clear()
        result = manager._calculate_effective_max(BudgetType.TOOL_CALLS)
        assert result == 0
