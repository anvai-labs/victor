# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for the SafetyCoordinator-backed enhanced DevOps safety extension."""

from victor_contracts import SafetyAction, SafetyRule, SafetyCategory

from victor_devops.safety_enhanced import (
    DevOpsSafetyRules,
    EnhancedDevOpsSafetyExtension,
)


class TestDevOpsSafetyRules:
    def test_all_rules_aggregates_every_domain(self):
        rules = DevOpsSafetyRules.get_all_rules()
        rule_ids = {r.rule_id for r in rules}

        assert len(rules) == len(rule_ids), "duplicate rule ids"
        # one representative rule from each domain
        assert "devops_docker_system_prune" in rule_ids
        assert "devops_k8s_delete_namespace" in rule_ids
        assert "devops_terraform_destroy" in rule_ids
        assert "devops_deploy_production" in rule_ids
        assert "devops_stop_critical_service" in rule_ids

    def test_catastrophic_operations_are_blocked(self):
        blocked = {
            r.rule_id for r in DevOpsSafetyRules.get_all_rules() if r.action == SafetyAction.BLOCK
        }

        assert "devops_k8s_delete_namespace" in blocked
        assert "devops_terraform_destroy" in blocked

    def test_severities_within_scale(self):
        for rule in DevOpsSafetyRules.get_all_rules():
            assert 1 <= rule.severity <= 10, rule.rule_id

    def test_confirmation_rules_have_prompts(self):
        for rule in DevOpsSafetyRules.get_all_rules():
            if rule.action == SafetyAction.REQUIRE_CONFIRMATION:
                assert rule.confirmation_prompt, rule.rule_id


class TestEnhancedExtension:
    def test_blocks_namespace_deletion(self):
        extension = EnhancedDevOpsSafetyExtension()

        result = extension.check_operation("shell", ["kubectl", "delete", "namespace", "prod"])

        assert result.is_safe is False
        assert result.action == SafetyAction.BLOCK
        assert any(r.rule_id == "devops_k8s_delete_namespace" for r in result.matched_rules)

    def test_blocks_terraform_destroy_auto_approve(self):
        extension = EnhancedDevOpsSafetyExtension()

        result = extension.check_operation("shell", ["terraform", "destroy", "-auto-approve"])

        assert result.is_safe is False

    def test_safe_command_passes(self):
        extension = EnhancedDevOpsSafetyExtension()

        assert extension.is_operation_safe("shell", ["ls", "-la"]) is True

    def test_custom_rules_can_be_disabled(self):
        extension = EnhancedDevOpsSafetyExtension(enable_custom_rules=False)

        rule_ids = {r.rule_id for r in extension.get_coordinator().list_rules()}
        assert "devops_k8s_delete_namespace" not in rule_ids

    def test_add_and_remove_custom_rule(self):
        extension = EnhancedDevOpsSafetyExtension()
        rule = SafetyRule(
            rule_id="test_custom_rule",
            category=SafetyCategory.SHELL,
            pattern=r"forbidden_command",
            description="Test rule",
            action=SafetyAction.BLOCK,
            severity=9,
            tool_names=["shell"],
        )

        extension.add_custom_rule(rule)
        result = extension.check_operation("shell", ["forbidden_command"])
        assert result.is_safe is False

        assert extension.remove_rule("test_custom_rule") is True
        assert extension.is_operation_safe("shell", ["forbidden_command"]) is True

    def test_tool_restrictions_cover_infrastructure_tools(self):
        restrictions = EnhancedDevOpsSafetyExtension().get_tool_restrictions()

        assert "docker" in restrictions
        assert "kubectl" in restrictions
        assert any("terraform destroy" in arg for arg in restrictions["shell"])

    def test_bash_patterns_from_contracts(self):
        patterns = EnhancedDevOpsSafetyExtension().get_bash_patterns()

        assert patterns, "expected BUILD_DEPLOY_PATTERNS to be exposed"

    def test_file_patterns_empty(self):
        assert EnhancedDevOpsSafetyExtension().get_file_patterns() == []
