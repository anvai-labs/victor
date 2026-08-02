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

"""Tests for the effect-gate wiring in AgenticLoop (EVR-4, ADR-010)."""

from unittest.mock import MagicMock

from victor.agent.services.turn_execution_runtime import TurnResult
from victor.framework.agentic_loop import AgenticLoop, AgenticLoopConfig
from victor.framework.effect_gate import EffectGate, EffectGateConfig
from victor.framework.evaluation_nodes import EvaluationDecision, EvaluationResult
from victor.providers.base import CompletionResponse


def _turn(content="", *, tool_results=None, is_qa=False, response_metadata=None):
    return TurnResult(
        response=CompletionResponse(content=content, metadata=response_metadata or {}),
        tool_results=tool_results or [],
        is_qa_response=is_qa,
    )


def _complete(score=0.9):
    return EvaluationResult(decision=EvaluationDecision.COMPLETE, score=score, reason="core done")


def _loop(*, gate=None, core_result=None):
    """AgenticLoop.__new__ pattern (mirrors test_agentic_loop_rubric_strategy)."""
    loop = AgenticLoop.__new__(AgenticLoop)
    loop.effect_gate = gate

    async def _core(perception, action_result, state):
        return core_result

    loop._evaluate_core = _core
    return loop


def _enabled_gate(max_downgrades=2):
    return EffectGate(EffectGateConfig(enabled=True, max_downgrades=max_downgrades))


# --- config --------------------------------------------------------------------------------------


def test_config_defaults_off():
    cfg = AgenticLoopConfig()
    assert cfg.enable_effect_gate is False
    assert cfg.effect_gate_max_downgrades == 2


def test_config_from_dict_accepts_gate_keys():
    cfg = AgenticLoopConfig.from_dict({"enable_effect_gate": True, "effect_gate_max_downgrades": 1})
    assert cfg.enable_effect_gate is True
    assert cfg.effect_gate_max_downgrades == 1
    assert "enable_effect_gate" not in cfg.extra_config


# --- construction --------------------------------------------------------------------------------


def test_loop_constructs_disabled_gate_by_default():
    loop = AgenticLoop(orchestrator=MagicMock(spec=[]), enable_fulfillment_check=False)
    assert isinstance(loop.effect_gate, EffectGate)
    assert loop.effect_gate.enabled is False


def test_loop_constructs_enabled_gate_from_config():
    loop = AgenticLoop(
        orchestrator=MagicMock(spec=[]),
        enable_fulfillment_check=False,
        config={"enable_effect_gate": True, "effect_gate_max_downgrades": 3},
    )
    assert loop.effect_gate.enabled is True
    assert loop.effect_gate.config.max_downgrades == 3


# --- _evaluate wrapper ---------------------------------------------------------------------------


async def test_default_off_passthrough_returns_core_result_identity():
    core = _complete()
    loop = _loop(gate=EffectGate(EffectGateConfig(enabled=False)), core_result=core)
    result = await loop._evaluate(None, _turn("done"), {"task_type": "edit"})
    assert result is core  # strict no-op: same object, no annotation
    assert core.metadata == {}


async def test_missing_gate_attribute_is_tolerated():
    core = _complete()
    loop = _loop(gate=None, core_result=core)
    assert await loop._evaluate(None, _turn("done"), {}) is core


async def test_complete_without_effect_downgraded_to_retry():
    loop = _loop(gate=_enabled_gate(), core_result=_complete())
    result = await loop._evaluate(None, _turn("I fixed the bug"), {"task_type": "bug_fix"})
    assert result.decision == EvaluationDecision.RETRY
    assert result.reason.startswith("completion-without-effect")
    assert result.metadata["completion_without_effect"] is True


async def test_complete_stands_when_turn_recorded_a_write():
    loop = _loop(gate=_enabled_gate(), core_result=_complete())
    turn = _turn(
        "done",
        tool_results=[{"tool_name": "write", "args": {"path": "a.py"}, "success": True}],
    )
    result = await loop._evaluate(None, turn, {"task_type": "edit"})
    assert result.decision == EvaluationDecision.COMPLETE


async def test_effect_from_earlier_turn_satisfies_later_summary_turn():
    gate = _enabled_gate()
    loop = _loop(gate=gate, core_result=_complete())
    effectful = _turn(
        "",
        tool_results=[{"tool_name": "edit", "args": {"path": "b.py"}, "success": True}],
    )
    # Turn 1: effectful turn evaluates (records evidence), turn 2: no-tool summary completes.
    await loop._evaluate(None, effectful, {"task_type": "refactor"})
    result = await loop._evaluate(None, _turn("summary of the change"), {"task_type": "refactor"})
    assert result.decision == EvaluationDecision.COMPLETE


async def test_qa_direct_answer_completes():
    loop = _loop(gate=_enabled_gate(), core_result=_complete())
    result = await loop._evaluate(None, _turn("The answer is 42", is_qa=True), {})
    assert result.decision == EvaluationDecision.COMPLETE


async def test_team_execution_bypassed():
    loop = _loop(gate=_enabled_gate(), core_result=_complete())
    turn = _turn("team output", response_metadata={"execution_mode": "team_execution"})
    result = await loop._evaluate(None, turn, {"task_type": "edit"})
    assert result.decision == EvaluationDecision.COMPLETE
    assert result.metadata["effect_gate"] == {"bypassed": "team_execution"}


async def test_exhausted_budget_annotates_and_allows():
    loop = _loop(gate=_enabled_gate(max_downgrades=1), core_result=_complete())
    state = {"task_type": "edit"}
    r1 = await loop._evaluate(None, _turn("done"), state)
    assert r1.decision == EvaluationDecision.RETRY
    r2 = await loop._evaluate(None, _turn("done"), state)
    assert r2.decision == EvaluationDecision.COMPLETE
    assert r2.metadata["effect_gate_exhausted"] is True


async def test_non_complete_core_results_untouched():
    core = EvaluationResult(decision=EvaluationDecision.CONTINUE, score=0.5, reason="progress")
    loop = _loop(gate=_enabled_gate(), core_result=core)
    assert await loop._evaluate(None, _turn("working"), {"task_type": "edit"}) is core


# --- reset wiring --------------------------------------------------------------------------------


def test_gate_reset_clears_session_state():
    gate = _enabled_gate()
    gate.record(
        _turn("", tool_results=[{"tool_name": "write", "args": {"path": "a.py"}, "success": True}]),
        {},
    )
    gate._downgrades = 2
    gate.reset()
    assert gate.ledger.candidate_effects() == []
    assert gate._downgrades == 0
