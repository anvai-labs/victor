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

"""Unit tests for the framework grounding-rules capability."""

from __future__ import annotations

from victor.framework.capabilities.grounding_rules import (
    GroundingRule,
    GroundingRulesCapability,
    GroundingRulesPresets,
    RuleCategory,
)


def _rule(rule_id: str = "custom_x", priority: int = 99) -> GroundingRule:
    return GroundingRule(
        rule_id=rule_id, category=RuleCategory.PRIVACY, text="Custom text.", priority=priority
    )


def test_defaults_are_sorted_by_priority_and_filterable() -> None:
    cap = GroundingRulesCapability()
    rules = cap.get_rules()
    assert [r.priority for r in rules] == sorted((r.priority for r in rules), reverse=True)
    git = cap.get_rules(RuleCategory.GIT_SAFETY)
    assert git and all(r.category is RuleCategory.GIT_SAFETY for r in git)
    assert cap.list_categories() == list(RuleCategory)


def test_add_get_and_remove_custom_rule() -> None:
    cap = GroundingRulesCapability()
    cap._rule_cache = {"stale": []}
    cap.add_rule(_rule())
    assert cap._rule_cache is None
    assert cap.get_rule("custom_x") is not None

    cap._rule_cache = {"stale": []}
    assert cap.remove_rule("custom_x") is True
    assert cap._rule_cache is None
    assert cap.get_rule("custom_x") is None
    assert cap.remove_rule("custom_x") is False


def test_remove_rule_only_touches_custom_rules() -> None:
    cap = GroundingRulesCapability()
    cap._rule_cache = {"kept": []}
    assert cap.remove_rule("read_before_write") is False
    assert cap.get_rule("read_before_write") is not None
    assert cap._rule_cache == {"kept": []}


def test_remove_rule_removes_first_match_only() -> None:
    cap = GroundingRulesCapability(custom_rules=[_rule("dup", 10), _rule("dup", 20)])
    assert cap.remove_rule("dup") is True
    assert [r.priority for r in cap.get_rules() if r.rule_id == "dup"] == [20]


def test_init_defaults_and_shares_the_given_list() -> None:
    cap = GroundingRulesCapability()
    assert cap._custom_rules == [] and cap._strict_mode is False and cap._rule_cache is None
    given = [_rule()]
    cap = GroundingRulesCapability(custom_rules=given, strict_mode=True)
    assert cap._custom_rules is given and cap._strict_mode is True
    cap._rule_cache = {"x": []}
    cap.clear_cache()
    assert cap._rule_cache is None


def test_vertical_rules_text_and_presets() -> None:
    cap = GroundingRulesPresets.coding()
    assert cap.get_rule("coding_lint_check") is not None
    assert all(r.applies_to("coding") for r in cap.get_vertical_rules("coding"))
    assert "coding_lint_check" not in {r.rule_id for r in cap.get_vertical_rules("devops")}
    text = cap.get_rules_text("coding")
    assert text.startswith("# Grounding Rules") and "# Additional Rules for Coding" in text
    custom = cap.get_custom_rules({"privacy": ["a", "b"], "bogus": ["c"]})
    assert list(custom) == ["privacy"] and [r.rule_id for r in custom["privacy"]] == [
        "custom_privacy_0",
        "custom_privacy_1",
    ]
