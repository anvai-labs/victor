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

"""Tests for the edge-model per-turn judge (EVR-6)."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from victor.agent.decisions.schemas import DecisionType, TurnAuditDecision
from victor.agent.edge_turn_judge import EdgeTurnJudge
from victor.framework.per_turn_auditor import AuditVerdict


def _decision(verdict, *, source="llm", reason=""):
    # Duck-typed DecisionResult (the judge reads .source / .result via getattr).
    return SimpleNamespace(
        source=source,
        result=TurnAuditDecision(verdict=verdict, confidence=0.9, reason=reason),
    )


def _turn(content="hello", tools=None):
    return SimpleNamespace(content=content, tool_results=tools or [])


def _service(decision):
    svc = MagicMock()
    svc.decide_sync.return_value = decision
    return svc


def test_maps_alarm_verdict():
    sig = EdgeTurnJudge(_service(_decision("alarm", reason="stuck in a loop"))).judge(_turn(), None)
    assert sig is not None
    assert sig.verdict is AuditVerdict.ALARM
    assert "stuck in a loop" in sig.reason


def test_maps_continue_verdict():
    sig = EdgeTurnJudge(_service(_decision("continue"))).judge(_turn(), None)
    assert sig is not None
    assert sig.verdict is AuditVerdict.CONTINUE


def test_ignores_non_llm_source():
    # A heuristic/timeout fallback is not a real judgment → no opinion.
    judge = EdgeTurnJudge(_service(_decision("alarm", source="timeout_fallback")))
    assert judge.judge(_turn(), None) is None


def test_none_service_returns_none():
    assert EdgeTurnJudge(None).judge(_turn(), None) is None


def test_decide_sync_exception_is_tolerated():
    svc = MagicMock()
    svc.decide_sync.side_effect = RuntimeError("edge boom")
    assert EdgeTurnJudge(svc).judge(_turn(), None) is None


def test_passes_turn_signals_in_context():
    svc = _service(_decision("continue"))
    turn = _turn(content="x", tools=[{"name": "edit", "success": False}])
    EdgeTurnJudge(svc).judge(turn, None)
    _args, kwargs = svc.decide_sync.call_args
    # decision_type positional, context positional
    assert svc.decide_sync.call_args[0][0] is DecisionType.TURN_AUDIT
    context = svc.decide_sync.call_args[0][1]
    assert context["has_tool_calls"] is True
    assert context["all_tools_failed"] is True


def test_turn_audit_registered_in_prompts():
    from victor.agent.decisions.prompts import DECISION_PROMPTS

    assert DecisionType.TURN_AUDIT in DECISION_PROMPTS
    assert DECISION_PROMPTS[DecisionType.TURN_AUDIT].schema is TurnAuditDecision
