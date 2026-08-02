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

"""Offline tests for AgenticLoop.run_streaming (FEP-0007 Addendum A, step 2).

These drive ``run_streaming`` with a fake streaming ACT port and stubbed phase methods, so they
assert the shared loop wiring — that streaming reuses the same PERCEIVE/PLAN/EVALUATE/DECIDE phase
sequence as run() and differs only in the ACT (yielding chunks + producing a TurnResult). The
port is unwired in production; these cover the framework loop in isolation.
"""

from types import SimpleNamespace

import pytest

from victor.agent.turn_policy import SpinDetector
from victor.framework.agentic_loop import AgenticLoop, StreamingTurnOutcome
from victor.framework.evaluation_nodes import EvaluationDecision, EvaluationResult
from victor.providers.base import StreamChunk


def _fake_port(turns):
    """A streaming ACT port: ``turns[i]`` is turn i+1's list of chunk strings.

    Each turn yields its chunks and stamps a TurnResult-shaped object onto the outcome holder.
    """

    class _Port:
        async def stream_turn_act(self, *, query, state, perception, plan, turn_index, outcome):
            chunks = turns[turn_index - 1] if turn_index - 1 < len(turns) else []
            for text in chunks:
                yield StreamChunk(content=text)
            outcome.turn_result = SimpleNamespace(content="".join(chunks), tag=f"turn{turn_index}")

    return _Port()


def _loop(port, *, evaluations, turn_executor=None):
    """A bare AgenticLoop with the run_streaming collaborators stubbed."""
    loop = AgenticLoop.__new__(AgenticLoop)
    loop.streaming_act_port = port
    loop.max_iterations = 5
    loop._progress_scores = []
    loop._last_perception = None
    loop.turn_executor = turn_executor
    # run_streaming now records per-turn spin (+ advances the temperature ratchet) like the buffered
    # _act path (ADR-013), so it needs a real SpinDetector rather than a bare stub.
    loop.spin_detector = SpinDetector()
    loop.turn_evaluation_controller = SimpleNamespace(reset=lambda: None)
    loop.criteria_builder = SimpleNamespace(reset=lambda: None)
    loop.nudge_policy = SimpleNamespace(
        evaluate=lambda spin: SimpleNamespace(should_inject=False),
        budget_warning=lambda i, m: SimpleNamespace(should_inject=False),
    )

    async def fake_analyze(query, context, history):
        return SimpleNamespace(
            to_dict=lambda: {"query": query},
            task_analysis=SimpleNamespace(task_type="qa"),
        )

    async def fake_plan(perception, state):
        return {"plan": "noop"}

    evals = iter(evaluations)

    async def fake_evaluate(perception, action_result, state):
        # Record what ACT produced so tests can assert the EVALUATE input.
        state["_seen_action_result"] = action_result
        return next(evals)

    loop._analyze_turn = fake_analyze
    loop._plan = fake_plan
    loop._evaluate = fake_evaluate
    loop._apply_backslide_guard = lambda evaluation: evaluation
    return loop


async def test_run_streaming_yields_chunks_and_completes_on_first_turn():
    port = _fake_port([["hello ", "world"]])
    loop = _loop(
        port,
        evaluations=[
            EvaluationResult(decision=EvaluationDecision.COMPLETE, score=1.0, reason="done")
        ],
    )

    chunks = [c async for c in loop.run_streaming("q")]

    assert [c.content for c in chunks] == ["hello ", "world"]
    assert loop._progress_scores == [1.0]


async def test_run_streaming_iterates_until_complete():
    port = _fake_port([["a"], ["b"]])
    loop = _loop(
        port,
        evaluations=[
            EvaluationResult(decision=EvaluationDecision.CONTINUE, score=0.4, reason="more"),
            EvaluationResult(decision=EvaluationDecision.COMPLETE, score=0.9, reason="done"),
        ],
    )

    chunks = [c async for c in loop.run_streaming("q")]

    # Both turns' chunks stream through in order; progress recorded per turn.
    assert [c.content for c in chunks] == ["a", "b"]
    assert loop._progress_scores == [0.4, 0.9]


async def test_run_streaming_feeds_act_turn_result_to_evaluate():
    port = _fake_port([["payload"]])
    captured = {}

    loop = _loop(
        port,
        evaluations=[
            EvaluationResult(decision=EvaluationDecision.COMPLETE, score=1.0, reason="done")
        ],
    )

    original_evaluate = loop._evaluate

    async def spy_evaluate(perception, action_result, state):
        captured["action_result"] = action_result
        return await original_evaluate(perception, action_result, state)

    loop._evaluate = spy_evaluate

    _ = [c async for c in loop.run_streaming("q")]

    # The TurnResult the ACT port produced is exactly what EVALUATE consumed.
    assert captured["action_result"].tag == "turn1"
    assert captured["action_result"].content == "payload"


async def test_run_streaming_stops_on_fail():
    port = _fake_port([["x"]])
    loop = _loop(
        port,
        evaluations=[EvaluationResult(decision=EvaluationDecision.FAIL, score=0.0, reason="bad")],
    )

    chunks = [c async for c in loop.run_streaming("q")]

    assert [c.content for c in chunks] == ["x"]
    assert loop._progress_scores == [0.0]


async def test_run_streaming_requires_a_port():
    loop = _loop(None, evaluations=[])

    with pytest.raises(RuntimeError, match="streaming ACT port"):
        async for _ in loop.run_streaming("q"):
            pass


def test_extract_turn_content_handles_empty_tool_turn():
    """Regression: a tool-only TurnResult has empty content; extraction must return "" (a str),
    never the CompletionResponse object (which crashed the content-repetition .strip() check).
    """
    from victor.agent.services.turn_execution_runtime import TurnResult
    from victor.providers.base import CompletionResponse

    tool_turn = TurnResult(
        response=CompletionResponse(content="", tool_calls=[{"name": "read", "arguments": {}}]),
        tool_results=[{"success": True, "name": "read"}],
        has_tool_calls=True,
        tool_calls_count=1,
    )
    assert AgenticLoop._extract_turn_content(tool_turn) == ""

    text_turn = TurnResult(response=CompletionResponse(content="done"))
    assert AgenticLoop._extract_turn_content(text_turn) == "done"

    # Raw string response and a bare CompletionResponse both coerce to a string.
    assert AgenticLoop._extract_turn_content("plain string") == "plain string"
    assert (
        AgenticLoop._extract_turn_content(SimpleNamespace(response=CompletionResponse(content="x")))
        == "x"
    )


def _nudge_loop(chat_ctx, *, should_inject_nudge, should_inject_budget=False):
    loop = AgenticLoop.__new__(AgenticLoop)
    loop.turn_executor = SimpleNamespace(_chat_context=chat_ctx)
    loop.spin_detector = object()
    loop.nudge_policy = SimpleNamespace(
        evaluate=lambda spin: SimpleNamespace(
            should_inject=should_inject_nudge,
            role="system",
            message="use a tool",
            nudge_type=SimpleNamespace(value="USE_TOOLS"),
        ),
        budget_warning=lambda i, m: SimpleNamespace(
            should_inject=should_inject_budget,
            role="system",
            message="budget low",
        ),
    )
    return loop


def test_inject_decide_nudges_adds_message_on_continue():
    added = []
    chat_ctx = SimpleNamespace(
        add_message=lambda role, message, metadata=None: added.append((role, message))
    )
    loop = _nudge_loop(chat_ctx, should_inject_nudge=True)

    loop._inject_decide_nudges(
        EvaluationResult(decision=EvaluationDecision.CONTINUE, score=0.3, reason="x"),
        1,
        5,
    )

    assert added == [("system", "use a tool")]


def test_inject_decide_nudges_noop_on_terminal_decision():
    added = []
    chat_ctx = SimpleNamespace(add_message=lambda *a, **k: added.append(a))
    loop = _nudge_loop(chat_ctx, should_inject_nudge=True)

    loop._inject_decide_nudges(
        EvaluationResult(decision=EvaluationDecision.COMPLETE, score=1.0, reason="done"),
        1,
        5,
    )

    assert added == []


def test_inject_decide_nudges_noop_without_turn_executor():
    loop = AgenticLoop.__new__(AgenticLoop)
    loop.turn_executor = None
    # Should simply return without touching nudge_policy / chat context.
    loop._inject_decide_nudges(
        EvaluationResult(decision=EvaluationDecision.CONTINUE, score=0.2, reason="x"),
        2,
        5,
    )


def _terminal_loop(total_tool_calls=0):
    loop = AgenticLoop.__new__(AgenticLoop)
    loop.spin_detector = SimpleNamespace(total_tool_calls=total_tool_calls)
    return loop


def _turn(content, *, tool_calls=None):
    from victor.agent.services.turn_execution_runtime import TurnResult
    from victor.providers.base import CompletionResponse

    return TurnResult(
        response=CompletionResponse(content=content, tool_calls=tool_calls),
        has_tool_calls=bool(tool_calls),
    )


def test_is_terminal_answer_substantial_no_tools_true():
    assert _terminal_loop()._is_terminal_answer(_turn("x" * 150)) is True


def test_is_terminal_answer_short_after_prior_tools_true():
    # Short final answer counts once tools have already run this task.
    assert _terminal_loop(total_tool_calls=2)._is_terminal_answer(_turn("Done.")) is True


def test_is_terminal_answer_short_no_prior_tools_false():
    assert _terminal_loop()._is_terminal_answer(_turn("short")) is False


def test_is_terminal_answer_tool_call_turn_false():
    assert (
        _terminal_loop()._is_terminal_answer(_turn("x" * 150, tool_calls=[{"name": "read"}]))
        is False
    )


def test_is_terminal_answer_intent_only_false():
    # Prior tools satisfy the substance gate, but intent narration is not a final answer.
    loop = _terminal_loop(total_tool_calls=2)
    assert loop._is_terminal_answer(_turn("I'll now read the file and summarize it.")) is False


def test_is_terminal_answer_continuation_request_false():
    text = "Here is the summary so far. Would you like me to continue with the next file? " * 2
    assert _terminal_loop()._is_terminal_answer(_turn(text)) is False


def test_is_terminal_answer_refusal_false():
    # A refusal must not be treated as a delivered answer (would force a
    # high-confidence COMPLETE / success=True for a turn that declined the task),
    # even after prior tools ran.
    loop = _terminal_loop(total_tool_calls=2)
    refusal = (
        "I'm sorry, but I can't read any files right now, so I don't have the "
        "information needed to give a grounded review of this codebase."
    )
    assert loop._is_terminal_answer(_turn(refusal)) is False


def test_is_terminal_answer_findings_not_misclassified_as_refusal():
    # A genuine completion that merely contains "cannot find" is still terminal.
    loop = _terminal_loop(total_tool_calls=2)
    answer = "I reviewed the modules and cannot find any correctness bugs. " * 3
    assert loop._is_terminal_answer(_turn(answer)) is True


def test_is_refusal_response_patterns():
    loop = AgenticLoop.__new__(AgenticLoop)
    assert loop._is_refusal_response("I can't read any files right now")
    assert loop._is_refusal_response("I cannot access that path")
    assert loop._is_refusal_response("...don't have the information needed...")
    assert loop._is_refusal_response("I'm unable to complete this")
    # Negatives: legitimate findings / answers
    assert not loop._is_refusal_response("I cannot find any bugs in this file.")
    assert not loop._is_refusal_response("Here are three improvement areas: ...")
    assert not loop._is_refusal_response("")


def test_is_intent_only_response_meta_deliberation_narration_true():
    # Regression: the transcript failure mode. The model narrates imminent
    # action ("Executing now", "Going now", "Calling now", "no more
    # deliberation") without ever invoking a tool. The first line does NOT
    # start with an intent prefix, so the OLD first-line-only check let this
    # through as a "substantial answer" and exited the loop before any tool
    # ran. The full-response density check must now classify it as intent-only.
    loop = AgenticLoop.__new__(AgenticLoop)
    narration = (
        "Not yet — I hadn't started merging. Executing now.\n\n"
        "Final command: gh pr checks 435. Going. (Calling.)\n\n"
        "I need to stop the meta-deliberation loop from before and just execute. "
        "Making the call. (Done.)\n\n"
        "Executing with gh pr checks. Going. (Final.)\n\n"
        "Producing the call."
    )
    assert loop._is_intent_only_response(narration) is True


def test_is_intent_only_response_real_answer_with_code_false():
    # A real answer carrying a fenced code block must NOT be flagged as
    # intent-only, even if it happens to mention an action word.
    loop = AgenticLoop.__new__(AgenticLoop)
    answer = (
        "Here are the results:\n\n"
        "```bash\n"
        "$ gh pr view 435\n"
        "```()\n\n"
        "The PR is not yet merged and is mergeable. I recommend --squash."
    )
    assert loop._is_intent_only_response(answer) is False


def test_is_intent_only_response_real_answer_with_table_false():
    loop = AgenticLoop.__new__(AgenticLoop)
    answer = (
        "Checks summary:\n\n"
        "| Name | Status | Conclusion |\n"
        "|------|--------|------------|\n"
        "| CI   | pass   | success    |\n\n"
        "All checks are green."
    )
    assert loop._is_intent_only_response(answer) is False


def test_is_terminal_answer_meta_deliberation_not_terminal():
    # End-to-end guard: even after prior tools ran (which satisfies the
    # substance gate), meta-deliberation narration is NOT a terminal answer —
    # the loop must continue rather than exit before the tool is invoked.
    loop = _terminal_loop(total_tool_calls=2)
    narration = (
        "Executing now. Going. (Calling.) I need to stop the meta-deliberation "
        "loop and just execute. Making the call. (Done.) No more deliberation."
    )
    assert loop._is_terminal_answer(_turn(narration)) is False


async def test_run_streaming_emits_prompt_reward_on_completion(monkeypatch):
    """run_streaming attributes the terminal turn's outcome to served prompt
    candidates — streaming parity with run() (FEP-0017)."""
    port = _fake_port([["hello"]])
    loop = _loop(
        port,
        evaluations=[
            EvaluationResult(decision=EvaluationDecision.COMPLETE, score=0.9, reason="done")
        ],
    )
    loop.runtime_intelligence = SimpleNamespace(
        _last_served_prompt_identities=[
            SimpleNamespace(
                provider="zai",
                prompt_candidate_hash="h1",
                section_name="GROUNDING_RULES",
                prompt_section_name="GROUNDING_RULES",
            )
        ]
    )
    captured: list = []
    monkeypatch.setattr(
        "victor.agent.services.prompt_optimization_reward.emit_prompt_candidate_outcome",
        lambda identities, *, completion_score, **kw: captured.append(completion_score),
    )

    [c async for c in loop.run_streaming("q")]

    assert captured == [0.9]  # terminal evaluation score, once


async def test_run_streaming_emits_prompt_reward_on_exhaustion(monkeypatch):
    """Loop exhausting without a terminal decision still attributes the final
    turn's evaluation (mirrors run()'s epilogue)."""
    port = _fake_port([["a"], ["b"]])
    loop = _loop(
        port,
        evaluations=[
            EvaluationResult(decision=EvaluationDecision.CONTINUE, score=0.3, reason="more"),
            EvaluationResult(decision=EvaluationDecision.CONTINUE, score=0.5, reason="more"),
        ],
    )
    loop.max_iterations = 2  # exhaust after two CONTINUE turns (no terminal return)
    loop.runtime_intelligence = SimpleNamespace(
        _last_served_prompt_identities=[
            SimpleNamespace(
                provider="zai",
                prompt_candidate_hash="h1",
                section_name="GROUNDING_RULES",
                prompt_section_name="GROUNDING_RULES",
            )
        ]
    )
    captured: list = []
    monkeypatch.setattr(
        "victor.agent.services.prompt_optimization_reward.emit_prompt_candidate_outcome",
        lambda identities, *, completion_score, **kw: captured.append(completion_score),
    )

    [c async for c in loop.run_streaming("q")]

    assert captured == [0.5]  # final evaluation score (not the mid-loop 0.3), once


async def test_run_streaming_verify_gate_fires_and_reenters():
    """FEP-0018: when a verifier is set and the agent claims COMPLETE, the loop
    verifies; on failure it injects feedback + re-enters. On the next COMPLETE,
    verify passes -> exits."""
    from victor.framework.verification import VerificationResult

    port = _fake_port([["hello"], ["fixed"]])
    loop = _loop(
        port,
        evaluations=[
            EvaluationResult(decision=EvaluationDecision.COMPLETE, score=1.0, reason="done"),
            EvaluationResult(decision=EvaluationDecision.COMPLETE, score=1.0, reason="done"),
        ],
    )

    call_count = [0]

    class _MockVerifier:
        async def verify(self, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return VerificationResult(
                    passed=9,
                    total=10,
                    raw_output="test_x failed",
                    feedback="9/10 - fix test_x",
                )
            return VerificationResult(
                passed=10,
                total=10,
                raw_output="all passed",
            )

    loop._verifier = _MockVerifier()
    loop._max_verify_retries = 2
    loop._verify_retries = 0
    loop.turn_executor = SimpleNamespace(
        _chat_context=SimpleNamespace(add_message=lambda **kw: None)
    )

    chunks = [c async for c in loop.run_streaming("q")]

    assert [c.content for c in chunks] == ["hello", "fixed"]  # both turns ran
    assert call_count[0] == 2  # verify called twice (fail -> re-enter -> pass)
