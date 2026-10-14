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

"""Run/stream behavioral-parity battery — the FEP-0007 unified-loop gate.

Addendum A (framing B) makes streaming a pure I/O mode of the one research-rooted loop:
``Agent.run(p)`` (buffered ``AgenticLoop.run``) and ``Agent.stream(p)`` (streaming, now
``AgenticLoop.run_streaming`` via ``StreamingChatExecutor.run_unified``) drive the SAME
PERCEIVE/PLAN/ACT/EVALUATE/DECIDE loop and differ only in I/O. So they must produce the SAME
executed tool sequence and the SAME answer for each QA scenario. This battery drives BOTH real
loops from one scripted script and asserts that parity across the whole battery.

History: before the step-3 cutover the two loops diverged on multi-step tasks — but that was a
latent ``run()`` bug (a tool-only turn's empty content fell through to a ``CompletionResponse``
object and crashed the content-repetition check), not PPAED eagerness. With the cutover (streaming
driving the unified loop) and that bug fixed, all scenarios reach parity.
"""

import pytest

from .parity_harness import (
    SCENARIOS,
    ScriptedProvider,
    build_streaming_orchestrator,
    capture_buffered_transcript,
    capture_transcript,
)

pytestmark = [pytest.mark.integration, pytest.mark.agents]


async def _run_both(scenario):
    """Drive the buffered and streaming loops from the same script (separate provider instances)."""
    buffered_orch = build_streaming_orchestrator(
        ScriptedProvider(list(scenario.turns)),
        scenario.tools,
        max_iterations=scenario.max_iterations,
        failing_tool_names=scenario.failing_tools,
    )
    streaming_orch = build_streaming_orchestrator(
        ScriptedProvider(list(scenario.turns)),
        scenario.tools,
        max_iterations=scenario.max_iterations,
        failing_tool_names=scenario.failing_tools,
    )
    buffered = await capture_buffered_transcript(buffered_orch, scenario.message)
    streaming = await capture_transcript(streaming_orch, scenario.message)
    return buffered, streaming


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s.id for s in SCENARIOS])
async def test_run_stream_parity(scenario):
    """Buffered and streaming loops agree on tool sequence + answer for the scenario."""
    buffered, streaming = await _run_both(scenario)

    assert not buffered.errored, f"{scenario.id} buffered: {buffered.error}"
    assert not streaming.errored, f"{scenario.id} streaming: {streaming.error}"

    # Primary parity signal: identical executed tool sequence (same actions, same order).
    assert buffered.tool_calls == streaming.tool_calls, (
        f"{scenario.id} ({scenario.label}): buffered tools {buffered.tool_calls} != "
        f"streaming tools {streaming.tool_calls}"
    )

    # Same answer present in both (streaming concatenates per-turn emits, so use containment,
    # not byte-equality, which the existing characterization battery also relies on).
    if scenario.expect_content_contains:
        needle = scenario.expect_content_contains.lower()
        assert needle in buffered.content.lower(), (
            f"{scenario.id}: expected {scenario.expect_content_contains!r} in buffered "
            f"{buffered.content!r}"
        )
        assert needle in streaming.content.lower(), (
            f"{scenario.id}: expected {scenario.expect_content_contains!r} in streaming "
            f"{streaming.content!r}"
        )


async def test_parity_battery_is_non_empty():
    """Guard: the battery actually runs scenarios (so the parity assertions above mean something)."""
    assert SCENARIOS, "expected a non-empty QA battery"


@pytest.mark.parametrize("with_tools", [False, True])
@pytest.mark.parametrize("mode", ["buffered", "streaming"])
@pytest.mark.parametrize(
    "policy",
    ["absent", "redact", "block", "empty", "malformed", "bad-content", "bad-allowed", "failure"],
)
async def test_unified_response_policy_and_terminal_verdict(mode, policy, with_tools, monkeypatch):
    """Real service/ACT/loop/event wiring: withheld output never causes a second call."""
    from unittest.mock import AsyncMock
    from victor.framework.agentic_loop import AgenticLoop
    from victor.framework.evaluation_nodes import EvaluationDecision, EvaluationResult
    from victor.framework._internal import stream_with_events
    from victor.framework.events import EventType
    from victor.framework.policies import (
        BlockPatternPolicy,
        MessagePolicyGate,
        Phase,
        PolicyEngine,
        RedactContentPolicy,
    )
    from .parity_harness import TurnScript

    original = "The result is sk-abc123. " + "This is the verified answer. " * 6
    tool_calls = [{"id": "read1", "name": "read", "arguments": {"path": "example.py"}}]
    provider = ScriptedProvider(
        [TurnScript(content=original, tool_calls=tool_calls if with_tools else None)]
    )
    policies = []
    if policy == "redact":
        policies = [RedactContentPolicy([r"sk-\w+"], placeholder="[KEY]")]
    elif policy == "block":
        policies = [BlockPatternPolicy([r"sk-\w+"], phases={Phase.RESPONSE}, reason="withheld")]
    elif policy == "empty":
        policies = [RedactContentPolicy([r".+"], placeholder="")]
    gate = MessagePolicyGate(PolicyEngine(policies)) if policies else None
    monkeypatch.setattr(
        "victor.agent.factory.coordination_builders.build_message_policy_gate", lambda *args: gate
    )
    if policy in {"malformed", "bad-content", "bad-allowed", "failure"}:
        from types import SimpleNamespace
        from victor.framework.policies import GateResult

        gate = SimpleNamespace(
            gate_request=AsyncMock(side_effect=lambda text: GateResult(allowed=True, content=text)),
            gate_response=AsyncMock(return_value=None),
        )
        if policy == "bad-content":
            gate.gate_response.return_value = GateResult(allowed=True, content=None)
        elif policy == "bad-allowed":
            gate.gate_response.return_value = GateResult(allowed="false", content=original)
        if policy == "failure":
            gate.gate_response.side_effect = RuntimeError("policy backend unavailable")
    if gate is not None and policy not in {"malformed", "bad-content", "bad-allowed", "failure"}:
        gate.gate_response = AsyncMock(wraps=gate.gate_response)
    orch = build_streaming_orchestrator(provider, ["read"] if with_tools else None)
    evaluate = AsyncMock(
        return_value=EvaluationResult(
            decision=EvaluationDecision.COMPLETE, score=1.0, reason="done"
        )
    )
    monkeypatch.setattr(AgenticLoop, "_evaluate", evaluate)

    denied = policy in {"block", "empty", "malformed", "bad-content", "bad-allowed", "failure"}
    if mode == "streaming":
        events = [event async for event in stream_with_events(orch, "Give the result.")]
        content = "".join(event.content for event in events if event.type == EventType.CONTENT)
        ends = [event for event in events if event.type == EventType.STREAM_END]
        assert len(ends) == 1
        assert ends[0].success is not denied
        assert len([event for event in events if event.type == EventType.ERROR]) == int(denied)
    else:
        from victor.core.errors import AgentError

        if denied:
            with pytest.raises(AgentError) as error:
                await orch.chat("Give the result.")
            assert error.value.recoverable is False
            content = ""
        else:
            response = await orch.chat("Give the result.")
            content = response.content
            assert response.metadata["agentic_loop_success"] is True
    assert provider.turn_index == 1
    assert orch._parity_tool_log == (["read"] if with_tools and not denied else [])
    if gate is not None:
        gate.gate_response.assert_awaited_once_with(original)
    if denied:
        assert not content
        evaluate.assert_not_awaited()
    else:
        expected = original.replace("sk-abc123", "[KEY]") if policy == "redact" else original
        assert content.strip() == expected.strip()
        evaluate.assert_awaited_once()
    if policy != "absent":
        assert "sk-abc123" not in content
        assert all("sk-abc123" not in (message.content or "") for message in orch.messages)


@pytest.mark.parametrize("stop_at", ["before-act", "evaluate-empty", "evaluate-partial"])
async def test_buffered_guard_stop_preserves_completed_act_without_recovery(stop_at, monkeypatch):
    """Real public service/loop/ACT wiring preserves work not yet in an iteration."""
    from unittest.mock import AsyncMock

    from victor.framework.agentic_loop import AgenticLoop
    from victor.framework.evaluation_nodes import EvaluationDecision, EvaluationResult
    from victor.framework.loop.guards import current_guards
    from .parity_harness import TurnScript

    content = "Current partial answer." if stop_at == "evaluate-partial" else ""
    provider = ScriptedProvider(
        [
            TurnScript(
                content=content,
                tool_calls=[{"id": "read1", "name": "read", "arguments": {"path": "example.py"}}],
            )
        ]
    )
    chat = provider.chat
    usage = {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14}

    async def counted_chat(*args, **kwargs):
        response = await chat(*args, **kwargs)
        response.usage = usage.copy()
        response.metadata = {"request_id": "current"}
        return response

    monkeypatch.setattr(provider, "chat", counted_chat)
    orch = build_streaming_orchestrator(provider, ["read"])
    original_run = AgenticLoop.run
    observed = {}

    async def guarded_run(self, *args, **kwargs):
        if stop_at == "before-act":
            self.config.total_timeout = 0
        result = await original_run(self, *args, **kwargs)
        observed["loop"] = result
        return result

    async def stop_during_evaluation(self, perception, action_result, state):
        observed["act"] = action_result
        guards = current_guards()
        assert guards is not None
        # Deterministic expiry of the actual owner's clock; no sleep or new timer.
        guards.total_timeout = 0
        return EvaluationResult(decision=EvaluationDecision.CONTINUE, reason="stale evaluation")

    monkeypatch.setattr(AgenticLoop, "run", guarded_run)
    monkeypatch.setattr(AgenticLoop, "_evaluate", stop_during_evaluation)
    from victor.agent.response_completer import ResponseCompleter

    recovery = AsyncMock(side_effect=AssertionError("Recovery after terminal guard"))
    monkeypatch.setattr(ResponseCompleter, "ensure_response", recovery)
    response = await orch.chat("Inspect example.py and report what you find.")

    recovery.assert_not_awaited()
    assert response.metadata["agentic_loop_success"] is False
    assert response.metadata["agentic_loop_iterations"] == 0
    assert "deadline" in response.metadata["agentic_loop_error"]
    assert observed["loop"].iterations == []
    assert provider.turn_index == (0 if stop_at == "before-act" else 1)
    assert orch._parity_tool_log == ([] if stop_at == "before-act" else ["read"])
    assert response.content == content
    if stop_at != "before-act":
        assert response is observed["act"].response
        assert response.usage == usage
        assert response.metadata["request_id"] == "current"
        assert observed["act"].successful_tool_count == 1
        assert any(message.role == "tool" for message in orch.messages)
