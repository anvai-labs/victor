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

"""Streaming-loop characterization battery — the FEP-0007 regression gate.

These tests pin the canonical streaming loop (``AgenticLoop.run_streaming`` driven via
``StreamingChatExecutor.run_unified``) on the QA battery: each scenario must yield the expected
answer + tool sequence with no streaming traceback. There is a single streaming path — no feature
flag and no legacy fallback — so these characterization transcripts ARE the regression gate: any
refactor of the loop must keep them byte-stable.
"""

from __future__ import annotations

import pytest

from .parity_harness import (
    SCENARIOS,
    build_for_scenario,
    capture_transcript,
)

pytestmark = [pytest.mark.integration, pytest.mark.agents]

_SCENARIO_IDS = [s.id for s in SCENARIOS]


@pytest.mark.parametrize("scenario", SCENARIOS, ids=_SCENARIO_IDS)
async def test_streaming_loop_characterization(scenario):
    """Each battery scenario runs through the real streaming loop without a streaming traceback,
    producing the expected answer + tool sequence. This is the regression baseline the loop must
    preserve across refactors."""
    orch = build_for_scenario(scenario)
    transcript = await capture_transcript(orch, scenario.message)

    assert not transcript.errored, f"{scenario.id} ({scenario.label}): {transcript.error}"

    if scenario.expect_content_contains:
        assert scenario.expect_content_contains.lower() in transcript.content.lower(), (
            f"{scenario.id}: expected {scenario.expect_content_contains!r} in "
            f"{transcript.content!r}"
        )

    if scenario.expect_tools is not None:
        assert (
            transcript.tool_calls == scenario.expect_tools
        ), f"{scenario.id}: tool sequence {transcript.tool_calls} != {scenario.expect_tools}"


async def test_spin_scenario_terminates():
    """U1: an identical repeated tool call must be stopped by the loop's spin/repetition guard
    well before max_iterations — i.e. the loop converges instead of hanging."""
    scenario = next(s for s in SCENARIOS if s.id == "U1")
    orch = build_for_scenario(scenario)
    transcript = await capture_transcript(orch, scenario.message)

    assert not transcript.errored, transcript.error
    # The guard must terminate before exhausting the scripted turns / iteration ceiling.
    assert len(transcript.tool_calls) < scenario.max_iterations, (
        f"spin guard did not converge: {len(transcript.tool_calls)} tool calls "
        f"(max_iterations={scenario.max_iterations})"
    )
