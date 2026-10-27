"""FEP-0041 Phase 3: batch plan detection in the AgenticLoop."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from victor.core.batch_plan import BatchPlan


def _make_loop() -> Any:
    """A minimal AgenticLoop-shaped object for testing _maybe_parse_batch_plan."""
    from victor.framework.agentic_loop import AgenticLoop

    loop = AgenticLoop.__new__(AgenticLoop)
    return loop


class TestBatchPlanDetection:
    def test_dict_with_steps_detected(self):
        loop = _make_loop()
        state: dict = {}
        plan = {
            "steps": [
                {"tool_name": "read", "inputs": {"file_path": "a.py"}},
                {"tool_name": "edit", "input_refs": {"file_path": "$step0.file_path"}},
            ],
            "rationale": "read then edit",
        }
        loop._maybe_parse_batch_plan(state, plan)
        assert "_batch_plan" in state
        assert isinstance(state["_batch_plan"], BatchPlan)
        assert len(state["_batch_plan"].steps) == 2

    def test_non_batch_plan_ignored(self):
        loop = _make_loop()
        state: dict = {}
        loop._maybe_parse_batch_plan(state, {"action": "read", "file": "a.py"})
        assert "_batch_plan" not in state

    def test_none_plan_ignored(self):
        loop = _make_loop()
        state: dict = {}
        loop._maybe_parse_batch_plan(state, None)
        assert "_batch_plan" not in state

    def test_string_plan_ignored(self):
        loop = _make_loop()
        state: dict = {}
        loop._maybe_parse_batch_plan(state, "just a string plan")
        assert "_batch_plan" not in state

    def test_stale_batch_plan_cleared(self):
        """When a subsequent plan is NOT a batch plan, the stale detection is removed."""
        loop = _make_loop()
        state: dict = {"_batch_plan": "stale"}
        loop._maybe_parse_batch_plan(state, {"action": "single_step"})
        assert "_batch_plan" not in state

    def test_empty_steps_not_detected(self):
        loop = _make_loop()
        state: dict = {}
        loop._maybe_parse_batch_plan(state, {"steps": []})
        assert "_batch_plan" not in state
