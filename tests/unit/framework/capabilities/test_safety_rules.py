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

"""Unit tests for the framework safety-rules capability."""

from __future__ import annotations

from victor.framework.capabilities.safety_rules import (
    SafetyAction,
    SafetyCategory,
    SafetyRule,
    SafetyRulesCapabilityProvider,
)


def _rule(rule_id: str = "custom_block", severity: int = 9) -> SafetyRule:
    return SafetyRule(
        rule_id=rule_id,
        category=SafetyCategory.SYSTEM,
        pattern=r"shutdown",
        description="Shutting down the host",
        action=SafetyAction.BLOCK,
        severity=severity,
        tool_names=["shell"],
    )


def test_builtin_rules_block_warn_and_confirm() -> None:
    cap = SafetyRulesCapabilityProvider()
    assert cap.is_operation_safe("git", ["push", "--force"]) is False
    assert cap.is_operation_safe("git", ["status"]) is True
    assert cap.get_warnings("git", ["reset", "--hard"])
    assert cap.get_required_confirmations("shell", ["mkfs /dev/sda"])
    severities = [r.severity for r in cap.get_rules()]
    assert severities == sorted(severities, reverse=True)
    assert cap.list_categories() == list(SafetyCategory)


def test_add_rule_invalidates_cache_and_takes_effect() -> None:
    cap = SafetyRulesCapabilityProvider()
    assert cap.is_operation_safe("shell", ["shutdown", "now"]) is True  # builds the cache
    assert cap._rule_cache is not None
    cap.add_rule(_rule())
    assert cap._rule_cache is None
    assert cap.is_operation_safe("shell", ["shutdown", "now"]) is False
    assert cap.get_rules(SafetyCategory.SYSTEM)[0].rule_id == "custom_block"


def test_remove_rule_invalidates_cache_and_only_touches_custom_rules() -> None:
    cap = SafetyRulesCapabilityProvider(custom_rules=[_rule()])
    assert cap.get_rule("custom_block") is not None
    assert cap.remove_rule("git_reset_hard") is False
    assert cap._rule_cache is not None  # untouched on a miss
    assert cap.remove_rule("custom_block") is True
    assert cap._rule_cache is None
    assert cap.get_rule("custom_block") is None
    assert cap.is_operation_safe("shell", ["shutdown", "now"]) is True
    assert cap.remove_rule("custom_block") is False


def test_remove_rule_removes_first_match_only() -> None:
    cap = SafetyRulesCapabilityProvider(custom_rules=[_rule("dup", 3), _rule("dup", 4)])
    assert cap.remove_rule("dup") is True
    assert [r.severity for r in cap.get_rules() if r.rule_id == "dup"] == [4]


def test_init_defaults_and_shares_the_given_list() -> None:
    cap = SafetyRulesCapabilityProvider()
    assert cap._custom_rules == [] and cap._strict_mode is False and cap._rule_cache is None
    given = [_rule()]
    cap = SafetyRulesCapabilityProvider(custom_rules=given, strict_mode=True)
    assert cap._custom_rules is given and cap._strict_mode is True


def test_invalid_regex_falls_back_to_literal_match() -> None:
    rule = SafetyRule(
        rule_id="lit",
        category=SafetyCategory.SHELL,
        pattern="rm [",
        description="literal",
        action=SafetyAction.WARN,
    )
    assert rule.matches("shell", ["rm", "[x"]) is True
    assert rule.matches("shell", ["ls"]) is False
    assert rule.to_dict()["action"] == "warn"
