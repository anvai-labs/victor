"""Tests for BudgetController."""

from victor.tools.budget_controller import BudgetController


class TestBudgetController:
    """Tests for budget controller."""

    def test_initial_state(self):
        bc = BudgetController(budget=10)
        assert bc.get_remaining() == 10
        assert not bc.is_exhausted()
        assert bc.consumed == 0

    def test_consume_basic(self):
        bc = BudgetController(budget=3)
        assert bc.consume("tool_a")
        assert bc.get_remaining() == 2
        assert bc.consume("tool_b")
        assert bc.consume("tool_c")
        assert bc.is_exhausted()

    def test_consume_over_budget(self):
        bc = BudgetController(budget=1)
        assert bc.consume("tool_a")
        assert not bc.consume("tool_b")
        assert bc.is_exhausted()

    def test_custom_tool_costs(self):
        bc = BudgetController(budget=10, tool_costs={"expensive": 5, "cheap": 1})
        assert bc.consume("expensive")
        assert bc.get_remaining() == 5
        assert bc.consume("cheap")
        assert bc.get_remaining() == 4

    def test_tool_counts(self):
        bc = BudgetController(budget=10)
        bc.consume("tool_a")
        bc.consume("tool_a")
        bc.consume("tool_b")
        assert bc.tool_counts == {"tool_a": 2, "tool_b": 1}

    def test_reset(self):
        bc = BudgetController(budget=5)
        bc.consume("tool_a")
        bc.consume("tool_b")
        bc.reset()
        assert bc.get_remaining() == 5
        assert bc.tool_counts == {}
        assert not bc.is_exhausted()

    def test_budget_setter(self):
        bc = BudgetController(budget=5)
        bc.budget = 20
        assert bc.budget == 20
        assert bc.get_remaining() == 20
