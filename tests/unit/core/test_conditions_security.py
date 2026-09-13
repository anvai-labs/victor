"""Workflow and debug condition syntax must not execute Python expressions."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from victor.core.conditions import compile_condition
from victor.ui.slash.commands.debug import DebugCommand
from victor.workflows.handlers import ConditionalBranchHandler


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os')",
        "x.__class__",
        "x[0]",
        "(lambda: 1)()",
        "[v for v in x]",
        "True or x.run()",
        "2 ** 100",
        "x := 3",
    ],
)
def test_condition_rejects_executable_or_unsupported_syntax(expression):
    with pytest.raises(ValueError):
        compile_condition(expression)


def test_conditions_compare_plain_data_and_short_circuit():
    predicate = compile_condition("count >= 2 and enabled and label in labels")
    assert predicate({"count": 2, "enabled": True, "label": "ok", "labels": ["ok"]})
    assert not predicate({"count": 1})
    assert compile_condition("True or missing")({})
    assert not compile_condition("False and missing")({})
    assert compile_condition("-count < 0")({"count": 2})
    assert not predicate({})


def test_custom_operands_never_run_operator_methods():
    class Hostile:
        def __eq__(self, other):
            pytest.fail("custom equality executed")

        def __bool__(self):
            pytest.fail("custom truthiness executed")

    assert not compile_condition("value == 1")({"value": Hostile()})
    assert not compile_condition("value")({"value": Hostile()})
    assert not compile_condition("1 in values")({"values": [Hostile()]})


def test_workflow_condition_uses_same_data_only_evaluator():
    handler = ConditionalBranchHandler()
    assert handler._evaluate_condition("${count} >= 2 and $enabled", {"count": 3, "enabled": True})
    assert not handler._evaluate_condition("value.run()", {"value": object()})


def test_debug_rejects_invalid_condition_before_setting_breakpoint():
    manager = MagicMock()
    ctx = SimpleNamespace(agent=SimpleNamespace(breakpoint_manager=manager), console=MagicMock())
    command = DebugCommand()
    command._handle_break(ctx, ["node", "--condition", "x.__class__"])
    manager.set_breakpoint.assert_not_called()
    command._handle_break(ctx, ["node", "--condition", "count > 2"])
    predicate = manager.set_breakpoint.call_args.kwargs["condition"]
    assert predicate({"count": 3})
    assert not predicate({"count": 1})


def test_expression_size_is_bounded():
    with pytest.raises(ValueError):
        compile_condition("x" * 4097)
    with pytest.raises(ValueError):
        compile_condition(" and ".join(["True"] * 300))


def test_aliases_cycles_and_total_value_budget_are_rejected():
    predicate = compile_condition("value")
    value = [1]
    for _ in range(10):
        value = [value] * 10
    assert not predicate({"value": value})
    cycle = []
    cycle.append(cycle)
    assert not predicate({"value": cycle})
    assert not predicate({"value": [list(range(100)) for _ in range(101)]})
    assert predicate({"value": [[1], [2]]})


def test_state_mapping_and_keys_cannot_run_lookup_hooks():
    class State(dict):
        def __contains__(self, key):
            pytest.fail("custom contains executed")

        def __getitem__(self, key):
            pytest.fail("custom getitem executed")

    class Key:
        def __hash__(self):
            return hash("value")

        def __eq__(self, other):
            pytest.fail("custom key equality executed")

    predicate = compile_condition("value")
    assert not predicate(State(value=1))
    assert not predicate({Key(): 1})
    handler = ConditionalBranchHandler()
    assert not handler._evaluate_condition("value", State(value=1))
    assert not handler._evaluate_condition("value", {Key(): 1})
