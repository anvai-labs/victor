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

"""Streaming cutover (ADR-013, PR-D): shared resolution helper + buffered↔streaming parity."""

from enum import Enum
from types import SimpleNamespace

from victor.agent.services.temperature_resolution import (
    normalize_task_type,
    resolve_effective_temperature,
)
from victor.agent.services.turn_execution_runtime import TurnExecutor
from victor.framework.capabilities.task_hints import TaskTypeHintCapabilityProvider
from victor.framework.temperature import build_default_resolver


def _orch(*, base=0.7, profile_map=None, settings_map=None, with_resolver=True):
    return SimpleNamespace(
        temperature_resolver=(
            build_default_resolver(hint_provider=TaskTypeHintCapabilityProvider())
            if with_resolver
            else None
        ),
        profile_task_temperatures=profile_map or {},
        settings_task_temperatures=settings_map or {},
        temperature=base,
        model="claude",
    )


# --- shared helper ------------------------------------------------------------------------------


def test_helper_explicit_override_wins():
    orch = _orch()
    assert (
        resolve_effective_temperature(
            orch,
            task_type="debug",
            model="claude",
            base_temperature=0.7,
            explicit_override=0.42,
        )
        == 0.42
    )


def test_helper_delegates_to_resolver():
    orch = _orch()
    assert (
        resolve_effective_temperature(orch, task_type="debug", model="claude", base_temperature=0.7)
        == 0.1
    )  # debug task-hint constant


def test_helper_fallback_without_resolver():
    orch = _orch(with_resolver=False)
    assert (
        resolve_effective_temperature(
            orch, task_type="debug", model="claude", base_temperature=0.55
        )
        == 0.55
    )


def test_helper_profile_per_task_beats_constant():
    orch = _orch(profile_map={"debug": 0.25})
    assert (
        resolve_effective_temperature(orch, task_type="debug", model="claude", base_temperature=0.7)
        == 0.25
    )


def test_normalize_task_type():
    class TT(Enum):
        ANALYSIS = "analysis"

    assert normalize_task_type("debug") == "debug"
    assert normalize_task_type(TT.ANALYSIS) == "analysis"
    assert normalize_task_type(None) is None


def test_resolve_with_enum_task_type_does_not_raise():
    """Regression: perception passes a TaskType *enum*; the resolver's get_hint needs a string.

    Before normalization this raised AttributeError('TaskType' has no attribute 'lower') inside
    resolve(), which crashed the buffered turn and broke run/stream parity.
    """

    class TaskType(Enum):
        ANALYZE = "analyze"

    orch = _orch(base=0.0)
    # passing the enum must resolve to the 'analyze' task-hint constant (0.6), not raise
    value = resolve_effective_temperature(
        orch, task_type=TaskType.ANALYZE, model="claude", base_temperature=0.0
    )
    assert value == 0.6


# --- buffered ↔ streaming parity ----------------------------------------------------------------


def test_buffered_path_matches_shared_helper():
    """The buffered executor must resolve to exactly what the shared helper produces (same seam)."""
    orch = _orch(base=0.7, profile_map={"plan": 0.5})
    ex = TurnExecutor.__new__(TurnExecutor)
    ex._provider_context = SimpleNamespace(temperature=0.7, model="claude", provider=None)
    ex._resolve_orchestrator = lambda: orch

    for task_type in ("debug", "analyze", "plan", "search", "totally_unmapped"):
        buffered = ex._resolve_turn_temperature(task_type, "claude", None)
        shared = resolve_effective_temperature(
            orch, task_type=task_type, model="claude", base_temperature=0.7
        )
        assert buffered == shared, f"parity broken for {task_type}: {buffered} != {shared}"
