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
