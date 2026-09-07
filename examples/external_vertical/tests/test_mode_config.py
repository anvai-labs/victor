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

"""Mode-config tests - runnable with only victor-contracts installed."""

from __future__ import annotations

from victor_contracts.verticals.mode_config import ModeConfig

from victor_security import SecurityAssistant, SecurityModeConfigProvider
from victor_security.mode_config import _SECURITY_MODES


def test_provider_resolves_quick_and_deep_modes() -> None:
    provider = SecurityModeConfigProvider()

    configs = provider.get_mode_configs()
    assert set(configs) == {"quick", "deep"}
    for name, config in configs.items():
        assert isinstance(config, ModeConfig), name
        assert config.tool_budget == _SECURITY_MODES[name].tool_budget
        assert config.max_iterations == _SECURITY_MODES[name].max_iterations

    assert configs["quick"].tool_budget < configs["deep"].tool_budget


def test_default_mode_and_budgets_resolve() -> None:
    provider = SecurityModeConfigProvider()

    assert provider.get_default_mode() == "quick"
    assert provider.get_default_mode() in provider.get_mode_configs()
    assert provider.get_default_tool_budget() == 12
    assert provider.get_default_tool_budget("unknown_task") == 12


def test_task_budgets_match_assistant_task_type_hints() -> None:
    provider = SecurityModeConfigProvider()
    hints = SecurityAssistant.get_task_type_hints()

    assert hints, "assistant should declare task type hints"
    for task_type, hint in hints.items():
        assert provider.get_tool_budget_for_task(task_type) == hint["tool_budget"], task_type


def test_allowed_stages_are_consistent_with_assistant_stages() -> None:
    stage_names = set(SecurityAssistant.get_stages())

    for name, definition in _SECURITY_MODES.items():
        assert definition.allowed_stages, name
        assert set(definition.allowed_stages) <= stage_names, name


def test_complexity_mapping_targets_known_modes() -> None:
    provider = SecurityModeConfigProvider()
    known_modes = set(provider.get_mode_configs())

    for complexity in ("trivial", "simple", "moderate", "complex", "highly_complex"):
        assert provider.get_mode_for_complexity(complexity) in known_modes
    assert provider.get_mode_for_complexity("unmapped") in known_modes
