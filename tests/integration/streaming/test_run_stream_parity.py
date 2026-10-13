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


@pytest.mark.parametrize("policy", ["absent", "redact", "block", "empty"])
async def test_unified_response_policy_and_terminal_verdict(policy, monkeypatch):
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
    provider = ScriptedProvider([TurnScript(content=original)])
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
    orch = build_streaming_orchestrator(provider)
    evaluate = AsyncMock(
        return_value=EvaluationResult(
            decision=EvaluationDecision.COMPLETE, score=1.0, reason="done"
        )
    )
    monkeypatch.setattr(AgenticLoop, "_evaluate", evaluate)

    events = [event async for event in stream_with_events(orch, "Give the result.")]
    content = "".join(event.content for event in events if event.type == EventType.CONTENT)
    ends = [event for event in events if event.type == EventType.STREAM_END]
    assert len(ends) == 1
    assert provider.turn_index == 1
    if policy in {"block", "empty"}:
        assert ends[0].success is False
        assert not content
        assert len([event for event in events if event.type == EventType.ERROR]) == 1
        evaluate.assert_not_awaited()
    else:
        assert ends[0].success is True
        expected = original.replace("sk-abc123", "[KEY]") if policy == "redact" else original
        assert content.strip() == expected.strip()
        evaluate.assert_awaited_once()
    if policy != "absent":
        assert "sk-abc123" not in content
        assert all("sk-abc123" not in (message.content or "") for message in orch.messages)
