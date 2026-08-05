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

"""Spin-ratchet activation (ADR-013, PR-E): the resolution seam auto-applies the orchestrator's
per-session RatchetState, advanced by the loop from the SpinDetector signal."""

from types import SimpleNamespace

from victor.agent.services.temperature_resolution import resolve_effective_temperature
from victor.framework.capabilities.task_hints import TaskTypeHintCapabilityProvider
from victor.framework.temperature import (
    RatchetState,
    SpinSignal,
    build_default_resolver,
)


def _orch(ratchet: RatchetState, *, base=0.6, model="claude"):
    return SimpleNamespace(
        temperature_resolver=build_default_resolver(hint_provider=TaskTypeHintCapabilityProvider()),
        profile_task_temperatures={},
        settings_task_temperatures={},
        temperature_ratchet_state=ratchet,
        temperature=base,
        model=model,
    )


def test_ratchet_auto_applied_from_orchestrator_state():
    ratchet = RatchetState(steps=2)  # two stalled turns accumulated
    orch = _orch(ratchet)
    # base 0.6 (no hint for this task) + 2*0.05 ratchet = 0.70
    value = resolve_effective_temperature(
        orch, task_type="conversational", model="claude", base_temperature=0.6
    )
    assert abs(value - 0.70) < 1e-9


def test_ratchet_zero_steps_is_noop():
    orch = _orch(RatchetState(steps=0))
    value = resolve_effective_temperature(
        orch, task_type="conversational", model="claude", base_temperature=0.6
    )
    assert value == 0.6


def test_ratchet_resets_on_progress_then_escalates_on_spin():
    """Mirror the loop's per-turn advance: progress resets, stalled/spinning turns escalate."""
    ratchet = RatchetState()
    # stalled turns escalate
    ratchet.record_turn(SpinSignal(spin_state="warning", made_progress=False))
    ratchet.record_turn(SpinSignal(spin_state="stuck", made_progress=False))
    assert ratchet.steps == 2
    # a tool-using (progress) turn resets
    ratchet.record_turn(SpinSignal(spin_state="normal", made_progress=True))
    assert ratchet.steps == 0
    orch = _orch(ratchet)
    assert (
        resolve_effective_temperature(
            orch, task_type="conversational", model="claude", base_temperature=0.6
        )
        == 0.6
    )


def test_ratchet_capped_at_0_9():
    orch = _orch(RatchetState(steps=100))
    value = resolve_effective_temperature(
        orch, task_type="conversational", model="claude", base_temperature=0.6
    )
    assert value == 0.9  # hard cap, well below the degeneration cliff


def _turn(*, has_tool_calls, blocked=False):
    return SimpleNamespace(
        has_tool_calls=has_tool_calls,
        all_tools_blocked=blocked,
        tool_calls_count=1 if has_tool_calls else 0,
        tool_signatures=(),
        tool_results=[{"tool_name": "read"}] if has_tool_calls else [],
    )


def test_shared_record_spin_and_ratchet_used_by_both_paths():
    """The shared _record_spin_and_ratchet (buffered _act + streaming run_streaming) updates the
    spin detector AND advances the orchestrator ratchet — the basis for buffered↔streaming symmetry.
    """
    from victor.agent.turn_policy import SpinDetector
    from victor.framework.agentic_loop import AgenticLoop

    loop = AgenticLoop.__new__(AgenticLoop)
    loop.spin_detector = SpinDetector()
    ratchet = RatchetState()
    loop.orchestrator = SimpleNamespace(temperature_ratchet_state=ratchet)

    # a stalled (no-tool) turn records spin and (once spin warns/sticks) escalates the ratchet
    for _ in range(4):
        loop._record_spin_and_ratchet(_turn(has_tool_calls=False))
    assert ratchet.steps >= 1  # escalated on stall
    # a tool-using (progress) turn resets the ratchet
    loop._record_spin_and_ratchet(_turn(has_tool_calls=True))
    assert ratchet.steps == 0
    # None turn is a safe no-op
    loop._record_spin_and_ratchet(None)
    assert ratchet.steps == 0
