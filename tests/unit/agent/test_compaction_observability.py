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

"""Unit tests for compaction observability helpers.

These helpers implement the [compaction-check]/[compaction-result] traces,
the budget probe, trigger evaluation, and stampede suppression extracted
from the context_compactor hotspot (ratchet TD-R1 -> TD-14).
"""

import logging
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

from victor.agent.compaction_observability import (
    CompactionAction,
    CompactionTrigger,
    coerce_float,
    coerce_int,
    duplicate_suppression_age,
    evaluate_trigger,
    log_compaction_result,
    probe_budget_state,
    skip_reason,
    threshold_origin_label,
    trace_compaction_check,
    trigger_reason,
)


def _metrics(**overrides):
    base = {
        "estimated_tokens": 90_000,
        "char_count": 360_000,
        "message_count": 12,
        "is_overflow_risk": False,
        "max_context_chars": 800_000,
        "max_context_tokens": 100_000,
        "context_window_source": "model-pattern",
        "last_actual_prompt_tokens": 88_000,
        "session_prompt_tokens": 500_000,
        "utilization": 0.9,
        "char_utilization": 0.45,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


class TestCoercion:
    def test_coerce_int_numeric(self):
        assert coerce_int(5) == 5
        assert coerce_int("7") == 7

    def test_coerce_int_rejects_non_numeric(self):
        assert coerce_int(None) is None
        assert coerce_int(True) is None
        assert coerce_int("abc") is None

    def test_coerce_int_never_raises_on_mocks(self):
        # MagicMock configures __int__ (returns 1); coercion must stay numeric
        # so trace formatting never breaks on metrics stand-ins.
        assert isinstance(coerce_int(MagicMock()), int)

    def test_coerce_float_numeric(self):
        assert coerce_float(0.75) == 0.75
        assert coerce_float("0.5") == 0.5

    def test_coerce_float_rejects_non_numeric(self):
        assert coerce_float(None) is None
        assert coerce_float("abc") is None

    def test_coerce_float_never_raises_on_mocks(self):
        assert isinstance(coerce_float(MagicMock()), float)


class TestProbeBudgetState:
    def test_reads_controller_budget(self):
        controller = SimpleNamespace(budget_compaction_due=lambda: (True, 500_000, 400_000))
        assert probe_budget_state(controller) == (True, 500_000, 400_000)

    def test_none_controller(self):
        assert probe_budget_state(None) == (False, 0, None)

    def test_missing_method(self):
        assert probe_budget_state(SimpleNamespace()) == (False, 0, None)

    def test_mock_return_shape_rejected(self):
        controller = SimpleNamespace(budget_compaction_due=MagicMock())
        assert probe_budget_state(controller) == (False, 0, None)


class TestThresholdOriginLabel:
    def test_adaptive_wins(self):
        assert threshold_origin_label(True, [object()], None, True) == "adaptive"

    def test_phase_label(self):
        phase = SimpleNamespace(value="execution")
        assert threshold_origin_label(False, None, phase, True) == "phase:execution"

    def test_default(self):
        assert threshold_origin_label(False, None, None, True) == "default"


class TestReasons:
    def test_skip_reason_token_space(self):
        assert "headroom 10,000 tok" in skip_reason(_metrics())

    def test_skip_reason_char_space(self):
        assert "char-space" in skip_reason(_metrics(max_context_tokens=None))

    def test_trigger_reasons(self):
        assert trigger_reason("manual", (False, 0, None), True) == "forced"
        assert trigger_reason("overflow", (False, 0, None), False) == "overflow risk"
        assert (
            trigger_reason("budget", (True, 500_000, 400_000), False)
            == "session prefill 500,000 >= limit 400,000"
        )
        assert (
            trigger_reason("threshold", (False, 0, None), False) == "utilization crossed threshold"
        )


class TestEvaluateTrigger:
    def _budget(self, due=False):
        return (due, 500_000, 400_000)

    def test_overflow_wins(self):
        result = evaluate_trigger(
            logging.getLogger("test"),
            _metrics(is_overflow_risk=True),
            force=False,
            enable_proactive=True,
            utilization=0.5,
            threshold=0.75,
            origin="default",
            budget_state=self._budget(due=True),
        )
        assert result is CompactionTrigger.OVERFLOW

    def test_budget_fires_independently_of_low_utilization(self):
        """The dead-gating regression: budget due + utilization 0.4 fires."""
        result = evaluate_trigger(
            logging.getLogger("test"),
            _metrics(),
            force=False,
            enable_proactive=True,
            utilization=0.4,
            threshold=0.75,
            origin="default",
            budget_state=self._budget(due=True),
        )
        assert result is CompactionTrigger.BUDGET

    def test_threshold(self):
        result = evaluate_trigger(
            logging.getLogger("test"),
            _metrics(),
            force=False,
            enable_proactive=True,
            utilization=0.8,
            threshold=0.75,
            origin="default",
            budget_state=self._budget(),
        )
        assert result is CompactionTrigger.THRESHOLD

    def test_none_when_below_threshold(self):
        result = evaluate_trigger(
            logging.getLogger("test"),
            _metrics(),
            force=False,
            enable_proactive=True,
            utilization=0.5,
            threshold=0.75,
            origin="default",
            budget_state=self._budget(),
        )
        assert result is CompactionTrigger.NONE

    def test_proactive_disabled_blocks_threshold_but_not_budget(self):
        result = evaluate_trigger(
            logging.getLogger("test"),
            _metrics(),
            force=False,
            enable_proactive=False,
            utilization=0.99,
            threshold=0.75,
            origin="default",
            budget_state=self._budget(),
        )
        assert result is CompactionTrigger.NONE
        result = evaluate_trigger(
            logging.getLogger("test"),
            _metrics(),
            force=False,
            enable_proactive=False,
            utilization=0.99,
            threshold=0.75,
            origin="default",
            budget_state=self._budget(due=True),
        )
        assert result is CompactionTrigger.BUDGET

    def test_force_wins(self):
        result = evaluate_trigger(
            logging.getLogger("test"),
            _metrics(),
            force=True,
            enable_proactive=False,
            utilization=0.1,
            threshold=0.75,
            origin="default",
            budget_state=self._budget(),
        )
        assert result is CompactionTrigger.MANUAL

    def test_traces_are_emitted(self, caplog):
        with caplog.at_level(logging.INFO, logger="test.trace"):
            evaluate_trigger(
                logging.getLogger("test.trace"),
                _metrics(),
                force=False,
                enable_proactive=True,
                utilization=0.5,
                threshold=0.75,
                origin="default",
                budget_state=self._budget(),
            )
        assert any(
            "[compaction-check]" in r.message and "decision=skip" in r.message
            for r in caplog.records
        )

    def test_mock_metrics_never_raise(self):
        result = evaluate_trigger(
            logging.getLogger("test"),
            MagicMock(),
            force=False,
            enable_proactive=True,
            utilization=0.9,
            threshold=0.75,
            origin="default",
            budget_state=self._budget(),
        )
        # Mock is_overflow_risk is truthy -> overflow; the point: no exception.
        assert result is CompactionTrigger.OVERFLOW


class TestDuplicateSuppression:
    def test_recent_compaction_suppresses(self):
        controller = SimpleNamespace(
            last_compaction_age_seconds=5.0, MIN_COMPACTION_INTERVAL_SECONDS=60.0
        )
        assert duplicate_suppression_age(controller) == 5.0

    def test_old_compaction_not_suppressed(self):
        controller = SimpleNamespace(
            last_compaction_age_seconds=120.0, MIN_COMPACTION_INTERVAL_SECONDS=60.0
        )
        assert duplicate_suppression_age(controller) is None

    def test_mock_controller_never_suppresses(self):
        assert duplicate_suppression_age(MagicMock()) is None

    def test_missing_attributes(self):
        assert duplicate_suppression_age(None) is None
        assert duplicate_suppression_age(SimpleNamespace()) is None


class TestTraceEmitters:
    def test_check_trace_content(self, caplog):
        with caplog.at_level(logging.INFO, logger="test.trace2"):
            trace_compaction_check(
                logging.getLogger("test.trace2"),
                _metrics(),
                threshold=0.75,
                origin="phase:execution",
                decision="trigger:overflow",
                reason="overflow risk",
                budget=(True, 500_000, 400_000),
            )
        line = next(r.message for r in caplog.records if "[compaction-check]" in r.message)
        assert "window=100,000 (model-pattern)" in line
        assert "util_token=90.0%" in line
        assert "threshold=75%(phase:execution)" in line
        assert "budget=500,000/400,000 DUE" in line
        assert "decision=trigger:overflow" in line

    def test_result_trace_content(self, caplog):
        with caplog.at_level(logging.INFO, logger="test.trace3"):
            log_compaction_result(
                logging.getLogger("test.trace3"),
                action_taken=True,
                trigger_value="overflow",
                messages_removed=57,
                truncations_applied=0,
                tokens_freed=33_992,
                tokens_before=145_200,
                utilization=1.0,
                metrics_after=_metrics(estimated_tokens=111_208, utilization=0.95),
                started_at=time.monotonic() - 0.008,
            )
        line = next(r.message for r in caplog.records if "[compaction-result]" in r.message)
        assert "trigger=overflow" in line
        assert "removed=57" in line
        assert "145,200->111,208" in line
        assert "100.0%->95.0%" in line
        assert "100,000/model-pattern" in line


class TestMovedTypes:
    def test_compaction_action_reexport_identity(self):
        """The enum must be the SAME object across import sites (identity checks)."""
        from victor.agent.context_compactor import CompactionAction as ReexportedAction
        from victor.agent.context_compactor import CompactionTrigger as ReexportedTrigger

        assert ReexportedTrigger is CompactionTrigger
        assert ReexportedAction is CompactionAction
        assert CompactionTrigger.BUDGET.value == "budget"

    def test_action_taken_property(self):
        action = CompactionAction(trigger=CompactionTrigger.BUDGET, messages_removed=3)
        assert action.action_taken is True
        assert CompactionAction(trigger=CompactionTrigger.NONE).action_taken is False
