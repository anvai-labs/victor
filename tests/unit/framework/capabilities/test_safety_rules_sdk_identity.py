"""SafetyCategory/SafetyAction in the framework safety-rules capability are the SDK enums.

Before the round-4 dedup the framework re-declared both enums, so
``victor.framework.capabilities.SafetyCategory.GIT`` was NOT equal to
``victor.framework.extensions.SafetyCategory.GIT`` (plain ``Enum`` members of
distinct classes never compare equal). Both paths now yield the SDK object.
"""

from __future__ import annotations

import importlib

import pytest

PAIRS = [
    ("victor.framework.capabilities.safety_rules", "victor_contracts.safety.runtime"),
    ("victor.framework.capabilities", "victor_contracts.safety"),
    ("victor.framework.capabilities", "victor.framework.extensions"),
]


@pytest.mark.parametrize("name", ["SafetyCategory", "SafetyAction"])
@pytest.mark.parametrize("core_mod, sdk_mod", PAIRS)
def test_core_safety_enum_is_the_sdk_object(core_mod: str, sdk_mod: str, name: str) -> None:
    core_obj = getattr(importlib.import_module(core_mod), name)
    assert core_obj is getattr(importlib.import_module(sdk_mod), name)


def test_safety_enum_members_values_and_order_preserved() -> None:
    from victor.framework.capabilities.safety_rules import SafetyAction, SafetyCategory

    assert [(m.name, m.value) for m in SafetyCategory] == [
        ("GIT", "git"),
        ("FILE", "file"),
        ("SHELL", "shell"),
        ("DOCKER", "docker"),
        ("NETWORK", "network"),
        ("SYSTEM", "system"),
    ]
    assert [(m.name, m.value) for m in SafetyAction] == [
        ("ALLOW", "allow"),
        ("WARN", "warn"),
        ("BLOCK", "block"),
        ("REQUIRE_CONFIRMATION", "require_confirmation"),
    ]
    # Plain Enums (not str-mixins): a member is not equal to its raw value.
    assert SafetyCategory.GIT != "git"
    assert SafetyAction.BLOCK != "block"


def test_provider_rules_use_sdk_enums_and_local_rule_semantics() -> None:
    from victor.framework.capabilities.safety_rules import (
        SafetyRule,
        SafetyRulesCapabilityProvider,
    )
    from victor_contracts.safety import SafetyAction as SDKSafetyAction
    from victor_contracts.safety import SafetyCategory as SDKSafetyCategory
    from victor_contracts.safety import SafetyRule as SDKSafetyRule

    provider = SafetyRulesCapabilityProvider()
    git_rules = provider.get_rules(SDKSafetyCategory.GIT)
    assert git_rules
    assert all(rule.category is SDKSafetyCategory.GIT for rule in git_rules)
    assert all(isinstance(rule.action, SDKSafetyAction) for rule in git_rules)
    assert provider.list_categories() == list(SDKSafetyCategory)

    # SafetyRule deliberately stays a framework class: its matching is
    # case-sensitive and does not canonicalize tool names (SDK rule does).
    assert SafetyRule is not SDKSafetyRule
    rule = SafetyRule(
        rule_id="r",
        category=SDKSafetyCategory.SHELL,
        pattern="rm -rf",
        description="d",
        action=SDKSafetyAction.BLOCK,
    )
    assert rule.matches("shell", ["rm", "-rf", "/"])
    assert not rule.matches("shell", ["RM", "-RF", "/"])
    assert rule.to_dict()["category"] == "shell"
    assert rule.to_dict()["action"] == "block"


def test_config_safety_level_binding_is_pinned() -> None:
    """config binds the SDK framework enum, not the SDK top-level str-Enum fork."""
    import victor.framework.config as config
    from victor_contracts.safety.framework import SafetyLevel as FrameworkLevel

    assert config.SafetyLevel is FrameworkLevel
    assert config.SafetyLevel.LOW.value == "low"
    # The SDK top-level SafetyLevel is still a separate str-Enum fork; pin that
    # this binding is NOT it, so an accidental swap cannot pass silently.
    import victor_contracts

    assert config.SafetyLevel is not victor_contracts.SafetyLevel
