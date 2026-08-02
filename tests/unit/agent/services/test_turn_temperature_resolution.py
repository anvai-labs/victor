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

"""Buffered cutover (ADR-013, PR-C): TurnExecutor temperature resolution via the unified resolver."""

from types import SimpleNamespace

from victor.agent.services.turn_execution_runtime import TurnExecutor
from victor.framework.capabilities.task_hints import TaskTypeHintCapabilityProvider
from victor.framework.temperature import build_default_resolver


def _executor(resolver, *, base_temp=0.7, model="claude", profile_map=None, settings_map=None):
    ex = TurnExecutor.__new__(TurnExecutor)
    ex._provider_context = SimpleNamespace(
        temperature=base_temp, model=model, provider=SimpleNamespace(name="zai")
    )
    orch = SimpleNamespace(
        temperature_resolver=resolver,
        profile_task_temperatures=profile_map or {},
        settings_task_temperatures=settings_map or {},
    )
    ex._resolve_orchestrator = lambda: orch
    return ex


def _real_resolver():
    # Real task-hint constants (debug 0.1 … analyze 0.6); global default 0.6.
    return build_default_resolver(hint_provider=TaskTypeHintCapabilityProvider())


# --- _derive_task_type ---------------------------------------------------------------------------


def test_derive_task_type_from_str_dict_object():
    assert TurnExecutor._derive_task_type("debug") == "debug"
    assert TurnExecutor._derive_task_type({"task_type": "search"}) == "search"
    assert TurnExecutor._derive_task_type(SimpleNamespace(task_type="edit")) == "edit"
    assert TurnExecutor._derive_task_type(None) is None


# --- _resolve_turn_temperature -------------------------------------------------------------------


def test_explicit_override_wins():
    ex = _executor(_real_resolver())
    assert ex._resolve_turn_temperature("debug", "claude", explicit_override=0.42) == 0.42


def test_task_hint_constant_applied():
    ex = _executor(_real_resolver())
    # debug hint constant = 0.1
    assert ex._resolve_turn_temperature("debug", "claude", None) == 0.1


def test_unknown_task_falls_back_to_profile_base():
    ex = _executor(_real_resolver(), base_temp=0.7)
    assert ex._resolve_turn_temperature("conversational_unmapped", "claude", None) == 0.7


def test_profile_per_task_overrides_constant():
    ex = _executor(_real_resolver(), profile_map={"debug": 0.25})
    # profile-per-task (0.25) beats the task-hint constant (0.1)
    assert ex._resolve_turn_temperature("debug", "claude", None) == 0.25


def test_fallback_to_provider_temperature_when_no_resolver():
    ex = TurnExecutor.__new__(TurnExecutor)
    ex._provider_context = SimpleNamespace(temperature=0.55, model="claude", provider=None)
    ex._resolve_orchestrator = lambda: None
    assert ex._resolve_turn_temperature("debug", "claude", None) == 0.55


def test_model_bounds_clamp_applied():
    ex = _executor(_real_resolver(), base_temp=0.95, model="qwen3.5")
    # profile base 0.95 clamped to qwen max 0.9
    assert ex._resolve_turn_temperature("unmapped", "qwen3.5", None) == 0.9
