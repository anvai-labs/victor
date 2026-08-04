# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
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

"""Tests for the injectable edge-model judge seam in PerTurnAuditor (EVR-6)."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from victor.framework.agentic_loop import AgenticLoop
from victor.framework.evaluation_nodes import EvaluationDecision, EvaluationResult
from victor.framework.per_turn_auditor import (
    AuditSignal,
    AuditVerdict,
    PerTurnAuditor,
    PerTurnAuditorConfig,
)


class _Judge:
    def __init__(self, signal):
        self._signal = signal
        self.calls = 0

    def judge(self, action_result, state):
        self.calls += 1
        return self._signal


def _turn(content="ok", tools=None):
    return SimpleNamespace(content=content, tool_results=tools or [])


def _complete():
    return EvaluationResult(decision=EvaluationDecision.COMPLETE, score=0.9)


def _auditor(judge):
    return PerTurnAuditor(PerTurnAuditorConfig(enabled=True), judge=judge)


def test_judge_alarm_overrides_heuristic_continue():
    judge = _Judge(AuditSignal(AuditVerdict.ALARM, "off-track"))
    out = _auditor(judge).apply(_complete(), _turn(content="looks fine"))
    assert out.should_retry
    assert "off-track" in out.reason
    assert judge.calls == 1


def test_judge_none_keeps_heuristic_continue():
    out = _auditor(_Judge(None)).apply(_complete(), _turn(content="looks fine"))
    assert out.should_complete


def test_heuristic_alarm_short_circuits_the_judge():
    judge = _Judge(AuditSignal(AuditVerdict.CONTINUE))  # would say continue, but must not be asked
    out = _auditor(judge).apply(_complete(), _turn(content="", tools=[]))  # degenerate
    assert out.should_retry
    assert judge.calls == 0


def test_judge_exception_is_tolerated():
    class _Boom:
        def judge(self, action_result, state):
            raise RuntimeError("boom")

    out = PerTurnAuditor(PerTurnAuditorConfig(enabled=True), judge=_Boom()).apply(
        _complete(), _turn(content="fine")
    )
    assert out.should_complete  # judge error → keep the heuristic CONTINUE


def test_no_judge_is_deterministic_only():
    out = PerTurnAuditor(PerTurnAuditorConfig(enabled=True)).apply(
        _complete(), _turn(content="fine")
    )
    assert out.should_complete


def test_loop_threads_judge_into_auditor():
    judge = _Judge(None)
    loop = AgenticLoop(
        orchestrator=MagicMock(spec=[]),
        enable_fulfillment_check=False,
        config={"enable_per_turn_auditor": True},
        per_turn_judge=judge,
    )
    assert loop.per_turn_auditor._judge is judge
