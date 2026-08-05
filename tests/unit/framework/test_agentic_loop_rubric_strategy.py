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

"""Tests for the rubric completion strategy wiring in AgenticLoop (EVR-3b, ADR-009)."""

from types import SimpleNamespace

from victor.agent.services.turn_execution_runtime import TurnResult
from victor.framework.agentic_loop import AgenticLoop, AgenticLoopConfig
from victor.framework.evaluation_nodes import EvaluationDecision, EvaluationResult
from victor.providers.base import CompletionResponse


def _turn(content="", *, tool_calls=None):
    return TurnResult(
        response=CompletionResponse(content=content, tool_calls=tool_calls),
        has_tool_calls=bool(tool_calls),
    )


def _scripted_rubric(*, complete: bool):
    return SimpleNamespace(
        evaluate=lambda task_family, content, context: SimpleNamespace(
            complete=complete,
            aggregate=0.9 if complete else 0.4,
            reason="scripted",
            to_dict=lambda: {"complete": complete},
        )
    )


def _loop(evaluator, *, strategy="rubric", enhanced=None):
    loop = AgenticLoop.__new__(AgenticLoop)
    loop.config = AgenticLoopConfig(completion_strategy=strategy)
    loop.rubric_completion_evaluator = evaluator
    loop.enhanced_completion_evaluator = enhanced
    loop.fulfillment = None
    loop.spin_detector = None
    return loop


# --- config --------------------------------------------------------------------------------------


def test_config_default_strategy_is_enhanced():
    assert AgenticLoopConfig().completion_strategy == "enhanced"


def test_config_from_dict_accepts_rubric_strategy():
    cfg = AgenticLoopConfig.from_dict({"completion_strategy": "rubric"})
    assert cfg.completion_strategy == "rubric"
    assert "completion_strategy" not in cfg.extra_config  # known field, not spilled to extra


# --- _build_rubric_evaluator (ADR-011 fallback contract) -----------------------------------------


def test_enhanced_strategy_builds_no_evaluator():
    assert AgenticLoop._build_rubric_evaluator("enhanced", None) is None
    assert AgenticLoop._build_rubric_evaluator("enhanced", lambda p: "x") is None


def test_rubric_without_llm_judge_reverts_to_enhanced():
    # No rubric_complete_fn -> None (defer to enhanced), NOT the uncalibrated heuristic (a=-0.092).
    for strategy in ("rubric", "hybrid"):
        assert AgenticLoop._build_rubric_evaluator(strategy, None) is None


def test_rubric_with_llm_judge_builds_evaluator():
    async def judge(_prompt):
        return "correctness: score=0.9 confidence=0.9"

    for strategy in ("rubric", "hybrid"):
        ev = AgenticLoop._build_rubric_evaluator(strategy, judge)
        assert ev is not None and hasattr(ev, "aevaluate")  # the async LLM-judge evaluator


# --- _rubric_completion_result -------------------------------------------------------------------


async def test_defers_when_no_rubric_evaluator():
    assert await _loop(None)._rubric_completion_result(None, _turn("done"), {}) is None


async def test_complete_when_rubric_passes():
    res = await _loop(_scripted_rubric(complete=True))._rubric_completion_result(
        None, _turn("a real final answer"), {"task_type": "qa"}
    )
    assert res is not None and res.decision == EvaluationDecision.COMPLETE
    assert res.reason.startswith("[rubric]")


async def test_continue_when_rubric_fails():
    res = await _loop(_scripted_rubric(complete=False))._rubric_completion_result(
        None, _turn("partial"), {"task_type": "qa"}
    )
    assert res is not None and res.decision == EvaluationDecision.CONTINUE


async def test_defers_when_tools_pending():
    res = await _loop(_scripted_rubric(complete=True))._rubric_completion_result(
        None, _turn("text", tool_calls=[{"name": "read"}]), {}
    )
    assert res is None  # tools pending -> not a completion candidate


async def test_defers_on_empty_content():
    res = await _loop(_scripted_rubric(complete=True))._rubric_completion_result(
        None, _turn(""), {}
    )
    assert res is None


async def test_uses_async_evaluator_when_present():
    class _AsyncEval:
        async def aevaluate(self, *, task_family, content, context):
            return SimpleNamespace(
                complete=True, aggregate=0.88, reason="async", to_dict=lambda: {}
            )

    res = await _loop(_AsyncEval())._rubric_completion_result(
        None, _turn("answer"), {"task_type": "qa"}
    )
    assert res is not None and res.decision == EvaluationDecision.COMPLETE and res.score == 0.88


# --- hybrid (rubric AND enhanced must agree) -----------------------------------------------------


def _enhanced(decision):
    async def _evaluate(**kwargs):
        return EvaluationResult(decision=decision, score=0.5, reason="enh")

    return SimpleNamespace(evaluate=_evaluate)


async def test_hybrid_continues_when_enhanced_disagrees():
    # rubric says COMPLETE but enhanced says CONTINUE → hybrid downgrades to CONTINUE
    loop = _loop(
        _scripted_rubric(complete=True),
        strategy="hybrid",
        enhanced=_enhanced(EvaluationDecision.CONTINUE),
    )
    res = await loop._rubric_completion_result(None, _turn("intent-only"), {"task_type": "qa"})
    assert res is not None and res.decision == EvaluationDecision.CONTINUE
    assert "hybrid" in res.reason


async def test_hybrid_completes_when_both_agree():
    loop = _loop(
        _scripted_rubric(complete=True),
        strategy="hybrid",
        enhanced=_enhanced(EvaluationDecision.COMPLETE),
    )
    res = await loop._rubric_completion_result(None, _turn("real answer"), {"task_type": "qa"})
    assert res is not None and res.decision == EvaluationDecision.COMPLETE
