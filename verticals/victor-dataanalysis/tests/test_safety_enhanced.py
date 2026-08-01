# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for the SafetyCoordinator-backed enhanced data analysis safety extension."""

from victor.framework.extensions import SafetyAction

from victor_dataanalysis.safety_enhanced import (
    DataAnalysisSafetyRules,
    EnhancedDataAnalysisSafetyExtension,
)


class TestDataAnalysisSafetyRules:
    def test_expected_rules_defined(self):
        rules = DataAnalysisSafetyRules.get_all_rules()
        rule_ids = {r.rule_id for r in rules}

        assert len(rules) == len(rule_ids), "duplicate rule ids"
        assert rule_ids == {
            "dataanalysis_delete_data",
            "dataanalysis_overwrite_original",
            "dataanalysis_share_sensitive",
        }

    def test_overwriting_original_data_is_blocked(self):
        rules = {r.rule_id: r for r in DataAnalysisSafetyRules.get_all_rules()}

        assert rules["dataanalysis_overwrite_original"].action == SafetyAction.BLOCK
        assert rules["dataanalysis_overwrite_original"].severity == 10

    def test_confirmation_rules_have_prompts(self):
        for rule in DataAnalysisSafetyRules.get_all_rules():
            if rule.action == SafetyAction.REQUIRE_CONFIRMATION:
                assert rule.confirmation_prompt, rule.rule_id

    def test_severities_within_scale(self):
        for rule in DataAnalysisSafetyRules.get_all_rules():
            assert 1 <= rule.severity <= 10, rule.rule_id


class TestEnhancedExtension:
    def test_data_deletion_requires_confirmation(self):
        extension = EnhancedDataAnalysisSafetyExtension()

        result = extension.check_operation("shell", ["delete", "sales.csv"])

        assert result.is_safe is False
        assert any(r.rule_id == "dataanalysis_delete_data" for r in result.matched_rules)

    def test_sharing_data_requires_confirmation(self):
        extension = EnhancedDataAnalysisSafetyExtension()

        result = extension.check_operation("shell", ["upload", "customer", "data"])

        assert result.is_safe is False
        assert any(r.rule_id == "dataanalysis_share_sensitive" for r in result.matched_rules)

    def test_safe_analysis_command_passes(self):
        extension = EnhancedDataAnalysisSafetyExtension()

        assert extension.is_operation_safe("shell", ["python", "analyze.py"]) is True

    def test_rules_registered_with_coordinator(self):
        extension = EnhancedDataAnalysisSafetyExtension()

        rule_ids = {r.rule_id for r in extension.get_coordinator().list_rules()}
        assert "dataanalysis_delete_data" in rule_ids
        assert "dataanalysis_overwrite_original" in rule_ids

    def test_pattern_surfaces_are_empty(self):
        extension = EnhancedDataAnalysisSafetyExtension()

        assert extension.get_bash_patterns() == []
        assert extension.get_file_patterns() == []
        assert extension.get_tool_restrictions() == {}

    def test_safety_stats_available(self):
        extension = EnhancedDataAnalysisSafetyExtension()
        extension.check_operation("shell", ["ls"])

        stats = extension.get_safety_stats()
        assert isinstance(stats, dict)
        assert stats
