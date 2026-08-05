# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
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

"""Tests for the per-turn-auditor wiring in AgenticLoop (EVR-6, ADR-012 seam)."""

from unittest.mock import MagicMock

from victor.agent.services.turn_execution_runtime import TurnResult
from victor.framework.agentic_loop import AgenticLoop, AgenticLoopConfig
from victor.framework.evaluation_nodes import EvaluationDecision, EvaluationResult
from victor.framework.per_turn_auditor import PerTurnAuditor, PerTurnAuditorConfig
from victor.providers.base import CompletionResponse


def _turn(content="", *, tool_results=None):
    return TurnResult(
        response=CompletionResponse(content=content, metadata={}),
        tool_results=tool_results or [],
        is_qa_response=False,
    )


def _complete(score=0.9):
    return EvaluationResult(decision=EvaluationDecision.COMPLETE, score=score, reason="core done")


def _loop(*, auditor=None, core_result=None):
    """AgenticLoop.__new__ pattern (mirrors test_agentic_loop_effect_gate)."""
    loop = AgenticLoop.__new__(AgenticLoop)
    loop.effect_gate = None  # isolate the auditor seam
    loop.per_turn_auditor = auditor

    async def _core(perception, action_result, state):
        return core_result

    loop._evaluate_core = _core
    return loop


def _enabled(max_alarms=2):
    return PerTurnAuditor(PerTurnAuditorConfig(enabled=True, max_alarms=max_alarms))


# --- config --------------------------------------------------------------------------------------


def test_config_defaults_off():
    cfg = AgenticLoopConfig()
    assert cfg.enable_per_turn_auditor is False
    assert cfg.per_turn_auditor_max_alarms == 2


def test_config_from_dict_accepts_auditor_keys():
    cfg = AgenticLoopConfig.from_dict(
        {"enable_per_turn_auditor": True, "per_turn_auditor_max_alarms": 1}
    )
    assert cfg.enable_per_turn_auditor is True
    assert cfg.per_turn_auditor_max_alarms == 1
    assert "enable_per_turn_auditor" not in cfg.extra_config


# --- construction --------------------------------------------------------------------------------


def test_loop_constructs_disabled_auditor_by_default():
    loop = AgenticLoop(orchestrator=MagicMock(spec=[]), enable_fulfillment_check=False)
    assert isinstance(loop.per_turn_auditor, PerTurnAuditor)
    assert loop.per_turn_auditor.enabled is False


def test_loop_constructs_enabled_auditor_from_config():
    loop = AgenticLoop(
        orchestrator=MagicMock(spec=[]),
        enable_fulfillment_check=False,
        config={"enable_per_turn_auditor": True, "per_turn_auditor_max_alarms": 3},
    )
    assert loop.per_turn_auditor.enabled is True
    assert loop.per_turn_auditor.config.max_alarms == 3


# --- _evaluate seam ------------------------------------------------------------------------------


async def test_default_off_passthrough_returns_core_identity():
    core = _complete()
    loop = _loop(auditor=PerTurnAuditor(PerTurnAuditorConfig(enabled=False)), core_result=core)
    assert await loop._evaluate(None, _turn(""), {}) is core  # strict no-op


async def test_missing_auditor_attribute_is_tolerated():
    core = _complete()
    loop = _loop(auditor=None, core_result=core)
    assert await loop._evaluate(None, _turn(""), {}) is core


async def test_degenerate_complete_downgraded_to_retry():
    loop = _loop(auditor=_enabled(), core_result=_complete())
    result = await loop._evaluate(None, _turn(""), {})
    assert result.decision == EvaluationDecision.RETRY
    assert result.reason.startswith("per-turn auditor alarm")


async def test_complete_with_content_stands():
    loop = _loop(auditor=_enabled(), core_result=_complete())
    result = await loop._evaluate(None, _turn("here is the answer"), {})
    assert result.decision == EvaluationDecision.COMPLETE


async def test_complete_with_successful_tool_stands():
    loop = _loop(auditor=_enabled(), core_result=_complete())
    turn = _turn("", tool_results=[{"tool_name": "write", "success": True}])
    result = await loop._evaluate(None, turn, {})
    assert result.decision == EvaluationDecision.COMPLETE


async def test_non_complete_core_untouched():
    core = EvaluationResult(decision=EvaluationDecision.CONTINUE, score=0.5)
    loop = _loop(auditor=_enabled(), core_result=core)
    assert await loop._evaluate(None, _turn(""), {}) is core


async def test_exhausted_budget_annotates_and_allows():
    loop = _loop(auditor=_enabled(max_alarms=1), core_result=_complete())
    assert (await loop._evaluate(None, _turn(""), {})).decision == EvaluationDecision.RETRY
    r2 = await loop._evaluate(None, _turn(""), {})
    assert r2.decision == EvaluationDecision.COMPLETE
    assert r2.metadata["per_turn_auditor_exhausted"] is True
