"""Temporary budget failures must roll back or stop the runtime explicitly."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from victor.agent.services.chat_stream_helpers import ChatStreamHelperMixin
from victor.agent.services.runtime_overrides import (
    OverrideRestorationError,
    apply_overrides,
    attribute_change,
    restore_overrides,
)
from victor.agent.services.turn_execution_runtime import TurnExecutor
from victor.agent.planning.team_execution import PlanningTeamExecutionAdapter


class Helper(ChatStreamHelperMixin):
    def __init__(self, owner):
        self._orchestrator = owner


class BudgetOwner:
    def __init__(self):
        self._value = 8
        self.reject = None

    @property
    def tool_budget(self):
        return self._value

    @tool_budget.setter
    def tool_budget(self, value):
        self._value = value  # Simulate partial mutation before setter failure.
        if value == self.reject:
            raise RuntimeError("budget setter failed")


def runtime(streaming, owner):
    if streaming:
        instance = Helper(owner)
        return (
            instance,
            instance._apply_stream_runtime_overrides,
            instance._restore_stream_runtime_overrides,
        )
    instance = object.__new__(TurnExecutor)
    instance._orchestrator = owner
    instance._chat_context = SimpleNamespace()
    instance._tool_context = SimpleNamespace()
    return (
        instance,
        instance._apply_runtime_context_overrides,
        instance._restore_runtime_context_overrides,
    )


@pytest.mark.parametrize("streaming", [False, True])
def test_budget_setter_failure_rolls_back_every_applied_owner(streaming):
    broken = BudgetOwner()
    broken.reject = 2
    owner = SimpleNamespace(tool_budget=8, _tool_pipeline=SimpleNamespace(config=broken))
    instance, apply, restore = runtime(streaming, owner)
    with pytest.raises(RuntimeError, match="setter failed"):
        apply({"tool_budget": 2})
    assert owner.tool_budget == broken.tool_budget == 8
    assert not hasattr(owner, "_runtime_tool_context_overrides")
    if not streaming:
        assert not hasattr(instance._chat_context, "_runtime_context_overrides")
    broken.reject = None
    snapshot = apply({"tool_budget": 2})
    assert owner.tool_budget == broken.tool_budget == 2
    restore(snapshot)
    assert owner.tool_budget == broken.tool_budget == 8


@pytest.mark.parametrize("streaming", [False, True])
def test_restore_failure_attempts_other_owners_and_blocks_reuse(streaming):
    broken = BudgetOwner()
    owner = SimpleNamespace(tool_budget=8, _tool_pipeline=SimpleNamespace(config=broken))
    instance, apply, restore = runtime(streaming, owner)
    snapshot = apply({"tool_budget": 2})
    broken.reject = 8
    with pytest.raises(OverrideRestorationError):
        restore(snapshot)
    assert owner.tool_budget == 8
    assert not hasattr(owner, "_runtime_tool_context_overrides")
    with pytest.raises(OverrideRestorationError):
        apply({})
    other, other_apply, _ = runtime(not streaming, owner)
    with pytest.raises(OverrideRestorationError):
        other_apply({})


def test_snapshot_read_failure_happens_before_any_mutation():
    owner = SimpleNamespace(budget=8)
    changes = [attribute_change("budget", owner, "budget", 2)]
    changes.append(("broken", MagicMock(side_effect=RuntimeError("read failed")), MagicMock(), 2))
    with pytest.raises(RuntimeError, match="read failed"):
        apply_overrides(changes)
    assert owner.budget == 8


def test_registered_mode_failure_does_not_fall_back_to_build():
    from victor.agent.mode_controller import get_mode_controller

    container = MagicMock()
    container.is_registered.return_value = True
    container.get.side_effect = RuntimeError("configured mode unavailable")
    with patch("victor.core.container.get_container", return_value=container):
        with pytest.raises(RuntimeError, match="configured mode unavailable"):
            get_mode_controller()


def test_parent_tool_failure_does_not_fall_back_to_broader_registry():
    adapter = object.__new__(PlanningTeamExecutionAdapter)
    registry = MagicMock()
    adapter.orchestrator = SimpleNamespace(
        get_enabled_tools=MagicMock(side_effect=RuntimeError()), tools=registry
    )
    with pytest.raises(RuntimeError, match="parent tool permissions"):
        adapter._available_parent_tools()
    registry.list_tools.assert_not_called()
    adapter.orchestrator.get_enabled_tools.return_value = set()
    adapter.orchestrator.get_enabled_tools.side_effect = None
    assert adapter._available_parent_tools() == set()
    registry.list_tools.assert_not_called()
