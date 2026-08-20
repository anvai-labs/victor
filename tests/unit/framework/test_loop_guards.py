"""Tests for the agentic-loop spin guards (Phase 1 instrumentation)."""

import logging

from victor.framework.loop.guards import (
    LoopGuards,
    current_guards,
    note_perception,
    reset_current_guards,
    set_current_guards,
)


class TestPerceptionSpinSignal:
    def test_warns_when_perception_far_outpaces_iterations(self, caplog):
        g = LoopGuards(perception_spin_factor=4)
        g.note_iteration()  # 1 healthy iteration
        with caplog.at_level(logging.WARNING):
            for _ in range(10):  # 10 perception calls against 1 iteration
                g.note_perception()
        assert "possible perception/restart spin" in caplog.text
        assert g.perception_calls == 10 and g.iteration_count == 1

    def test_no_warning_on_healthy_one_per_iteration(self, caplog):
        g = LoopGuards(perception_spin_factor=4)
        with caplog.at_level(logging.WARNING):
            for _ in range(20):
                g.note_iteration()
                g.note_perception()
        assert "spin" not in caplog.text

    def test_warns_only_once(self, caplog):
        g = LoopGuards(perception_spin_factor=2)
        g.note_iteration()
        with caplog.at_level(logging.WARNING):
            for _ in range(20):
                g.note_perception()
        assert caplog.text.count("possible perception/restart spin") == 1


class TestTimers:
    def test_check_total_none_never_breaches(self):
        assert LoopGuards().check_total(None) is False

    def test_check_total_breaches_past_deadline(self, caplog):
        g = LoopGuards()
        g.run_started_monotonic -= 100  # pretend 100s elapsed
        with caplog.at_level(logging.WARNING):
            assert g.check_total(10) is True
        assert "wall-clock" in caplog.text

    def test_check_phase_soft_cap(self, caplog):
        g = LoopGuards()
        g.phase_started_monotonic -= 30
        with caplog.at_level(logging.WARNING):
            assert g.check_phase("perceive", 5) is True
        assert g.check_phase("perceive", None) is False


class TestContextVarSeam:
    def test_note_perception_uses_current_guards(self):
        g = LoopGuards()
        token = set_current_guards(g)
        try:
            assert current_guards() is g
            note_perception()
            note_perception()
            assert g.perception_calls == 2
        finally:
            reset_current_guards(token)
        assert current_guards() is None

    def test_note_perception_is_noop_outside_a_loop(self):
        # No guard installed → no crash, no-op.
        assert current_guards() is None
        note_perception()  # must not raise


class TestEnforce:
    def test_no_config_never_raises(self):
        g = LoopGuards()
        g.note_iteration()
        g.enforce(total_timeout=None, effective_max=10)  # off → no raise

    def test_wall_clock_deadline_raises(self):
        import pytest

        from victor.framework.loop.guards import LoopSpinError

        g = LoopGuards()
        g.run_started_monotonic -= 100  # pretend 100s elapsed
        with pytest.raises(LoopSpinError, match="wall-clock deadline"):
            g.enforce(total_timeout=10, effective_max=10)

    def test_perception_backstop_raises(self):
        import pytest

        from victor.framework.loop.guards import LoopSpinError

        g = LoopGuards()
        g.iteration_count = 3
        g.perception_calls = 3 * 8 + 1  # > effective_max * backstop_factor
        with pytest.raises(LoopSpinError, match="perception backstop"):
            g.enforce(total_timeout=None, effective_max=3, backstop_factor=8)

    def test_backstop_healthy_does_not_raise(self):
        g = LoopGuards()
        g.iteration_count = 20
        g.perception_calls = 20  # once per iteration
        g.enforce(total_timeout=None, effective_max=20, backstop_factor=8)
