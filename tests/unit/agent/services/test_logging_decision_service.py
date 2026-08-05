# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License").

"""Always-on LoggingDecisionService — guarantees decisions are captured.

Closes the FEP-0012 loop's data gap: ``log_decision`` is only reached via a
registered decision service, so with no backend (--no-edge-model + no
artifact) nothing was logged. This service is the heuristic pass-through that
logs every call.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from victor.agent.services.logging_decision_service import LoggingDecisionService
from victor.agent.services.protocols.decision_service import DecisionResult


def test_is_healthy_always_true():
    assert LoggingDecisionService().is_healthy() is True


def test_decide_sync_returns_heuristic_and_logs():
    svc = LoggingDecisionService()
    with patch("victor.agent.services.logging_decision_service.log_decision") as m:
        result = svc.decide_sync(
            "task_completion",
            {"msg": "tests pass"},
            heuristic_result="complete",
            heuristic_confidence=0.9,
        )
    assert isinstance(result, DecisionResult)
    assert result.result == "complete"  # heuristic returned unchanged
    assert result.source == "heuristic"
    assert result.confidence == 0.9
    m.assert_called_once()
    assert m.call_args.kwargs["decision_type"] == "task_completion"
    assert m.call_args.kwargs["source"] == "heuristic"


@pytest.mark.asyncio
async def test_decide_and_decide_async_log():
    svc = LoggingDecisionService()
    with patch("victor.agent.services.logging_decision_service.log_decision") as m:
        r1 = await svc.decide("task_completion", {"msg": "a"}, heuristic_result="x")
        r2 = await svc.decide_async("task_completion", {"msg": "b"}, heuristic_result="y")
    assert r1.result == "x" and r2.result == "y"
    assert m.call_count == 2


def test_logging_failure_does_not_break_decision():
    """A logging error must never propagate — decisions still return."""
    svc = LoggingDecisionService()
    with patch(
        "victor.agent.services.logging_decision_service.log_decision",
        side_effect=RuntimeError("disk full"),
    ):
        result = svc.decide_sync("task_completion", {}, heuristic_result="ok")
    assert result.result == "ok"  # decision still returned


def test_decision_type_enum_value_logged():
    """log_decision receives the enum's .value (the canonical string key)."""
    from enum import Enum

    class FakeType(Enum):
        TASK_COMPLETION = "task_completion"

    svc = LoggingDecisionService()
    with patch("victor.agent.services.logging_decision_service.log_decision") as m:
        svc.decide_sync(FakeType.TASK_COMPLETION, {})
    assert m.call_args.kwargs["decision_type"] == "task_completion"
