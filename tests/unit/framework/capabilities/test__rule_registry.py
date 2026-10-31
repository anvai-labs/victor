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

"""Unit tests for the shared custom-rule registry base."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pytest

from victor.framework.capabilities._rule_registry import CustomRuleRegistry
from victor.framework.capabilities.grounding_rules import GroundingRulesCapability
from victor.framework.capabilities.safety_rules import SafetyRulesCapabilityProvider


@dataclass
class _Rule:
    rule_id: str
    weight: int = 0


class _Registry(CustomRuleRegistry[_Rule]):
    DEFAULTS = [_Rule("builtin", 1)]

    def get_rules(self) -> List[_Rule]:
        return [*self.DEFAULTS, *self._custom_rules]


def test_add_get_remove_round_trip_invalidates_cache() -> None:
    reg = _Registry()
    reg._rule_cache = {"x": []}
    reg.add_rule(_Rule("a"))
    assert reg._rule_cache is None
    assert reg.get_rule("a") == _Rule("a")
    reg._rule_cache = {"x": []}
    assert reg.remove_rule("a") is True
    assert reg._rule_cache is None
    assert reg.get_rule("a") is None


def test_remove_miss_keeps_cache_and_defaults() -> None:
    reg = _Registry()
    reg._rule_cache = {"x": []}
    assert reg.remove_rule("builtin") is False
    assert reg.remove_rule("absent") is False
    assert reg._rule_cache == {"x": []}
    assert reg.get_rule("builtin") == _Rule("builtin", 1)


def test_remove_and_get_use_first_match() -> None:
    reg = _Registry(custom_rules=[_Rule("dup", 1), _Rule("dup", 2)])
    assert reg.get_rule("dup") == _Rule("dup", 1)
    assert reg.remove_rule("dup") is True
    assert reg._custom_rules == [_Rule("dup", 2)]


def test_init_state() -> None:
    given = [_Rule("a")]
    reg = _Registry(custom_rules=given, strict_mode=True)
    assert reg._custom_rules is given and reg._strict_mode is True and reg._rule_cache is None
    assert _Registry()._custom_rules == []


def test_get_rules_is_abstract() -> None:
    with pytest.raises(NotImplementedError):
        CustomRuleRegistry[_Rule]().get_rule("a")


@pytest.mark.parametrize("cls", [GroundingRulesCapability, SafetyRulesCapabilityProvider])
def test_rule_capabilities_share_the_registry(cls: type) -> None:
    assert issubclass(cls, CustomRuleRegistry)
    assert "remove_rule" not in vars(cls) and "__init__" not in vars(cls)
