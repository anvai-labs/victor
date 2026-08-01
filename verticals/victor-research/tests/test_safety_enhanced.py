# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for the SafetyCoordinator-backed enhanced research safety extension."""

from victor_contracts import SafetyAction, SafetyCategory, SafetyRule

from victor_research.safety_enhanced import (
    EnhancedResearchSafetyExtension,
    ResearchSafetyRules,
)


class TestResearchSafetyRules:
    def test_all_rules_aggregates_every_domain(self):
        rules = ResearchSafetyRules.get_all_rules()
        rule_ids = {r.rule_id for r in rules}

        assert len(rules) == len(rule_ids), "duplicate rule ids"
        # one representative rule from each domain
        assert "research_bulk_scrape" in rule_ids
        assert "research_delete_experiments" in rule_ids
        assert "research_large_compute" in rule_ids
        assert "research_auto_publish" in rule_ids

    def test_auto_publish_is_most_severe(self):
        rules = {r.rule_id: r for r in ResearchSafetyRules.get_all_rules()}

        assert rules["research_auto_publish"].severity == 9
        assert rules["research_auto_publish"].action == SafetyAction.REQUIRE_CONFIRMATION

    def test_sensitive_data_collection_requires_confirmation(self):
        rules = {r.rule_id: r for r in ResearchSafetyRules.get_all_rules()}

        rule = rules["research_sensitive_data"]
        assert rule.action == SafetyAction.REQUIRE_CONFIRMATION
        assert "privacy" in rule.confirmation_prompt

    def test_severities_within_scale(self):
        for rule in ResearchSafetyRules.get_all_rules():
            assert 1 <= rule.severity <= 10, rule.rule_id

    def test_confirmation_rules_have_prompts(self):
        for rule in ResearchSafetyRules.get_all_rules():
            if rule.action == SafetyAction.REQUIRE_CONFIRMATION:
                assert rule.confirmation_prompt, rule.rule_id


class TestEnhancedExtension:
    def test_bulk_scrape_requires_confirmation(self):
        extension = EnhancedResearchSafetyExtension()

        result = extension.check_operation("shell", ["scrape", "--bulk", "example.com"])

        assert result.is_safe is False
        assert result.action == SafetyAction.REQUIRE_CONFIRMATION
        assert any(r.rule_id == "research_bulk_scrape" for r in result.matched_rules)

    def test_auto_publish_requires_confirmation(self):
        extension = EnhancedResearchSafetyExtension()

        result = extension.check_operation("shell", ["publish", "--auto"])

        assert result.is_safe is False

    def test_safe_command_passes(self):
        extension = EnhancedResearchSafetyExtension()

        assert extension.is_operation_safe("shell", ["ls", "-la"]) is True

    def test_custom_rules_can_be_disabled(self):
        extension = EnhancedResearchSafetyExtension(enable_custom_rules=False)

        rule_ids = {r.rule_id for r in extension.get_coordinator().list_rules()}
        assert "research_bulk_scrape" not in rule_ids

    def test_add_and_remove_custom_rule(self):
        extension = EnhancedResearchSafetyExtension()
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

    def test_tool_restrictions_cover_web_and_shell(self):
        restrictions = EnhancedResearchSafetyExtension().get_tool_restrictions()

        assert any("scrape" in arg for arg in restrictions["web"])
        assert any("experiments" in arg for arg in restrictions["shell"])

    def test_no_bash_or_file_patterns(self):
        """Research relies on coordinator rules, not raw pattern lists."""
        extension = EnhancedResearchSafetyExtension()

        assert extension.get_bash_patterns() == []
        assert extension.get_file_patterns() == []
