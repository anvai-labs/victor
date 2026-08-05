# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
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

"""Unit tests for SafetyCoordinator."""

import pytest

from victor_contracts.safety import (
    SafetyAction,
    SafetyCategory,
    SafetyCheckResult,
    SafetyCoordinator,
    SafetyRule,
    SafetyStats,
)


class TestSafetyRule:
    """Test suite for SafetyRule."""

    def test_rule_creation(self):
        """Test creating a safety rule."""
        rule = SafetyRule(
            rule_id="test_rule",
            category=SafetyCategory.GIT,
            pattern=r"push.*--force",
            description="Test force push rule",
            action=SafetyAction.BLOCK,
            severity=8,
        )

        assert rule.rule_id == "test_rule"
        assert rule.category == SafetyCategory.GIT
        assert rule.action == SafetyAction.BLOCK
        assert rule.severity == 8

    def test_matches_tool_name(self):
        """Test pattern matching against tool name."""
        rule = SafetyRule(
            rule_id="git_force",
            category=SafetyCategory.GIT,
            pattern=r"git",
            description="Git operations",
            action=SafetyAction.WARN,
            tool_names=["git"],
        )

        assert rule.matches("git", ["status"])
        assert not rule.matches("docker", ["ps"])

    def test_matches_args(self):
        """Test pattern matching against arguments."""
        rule = SafetyRule(
            rule_id="force_push",
            category=SafetyCategory.GIT,
            pattern=r"push.*--force",
            description="Force push",
            action=SafetyAction.BLOCK,
            tool_names=["git"],
        )

        assert rule.matches("git", ["push", "--force", "origin", "main"])
        assert not rule.matches("git", ["push", "origin", "main"])

    def test_matches_canonicalizes_legacy_tool_names(self):
        """Test tool-name aliases normalize before matching."""
        rule = SafetyRule(
            rule_id="system_write",
            category=SafetyCategory.FILE,
            pattern=r"/etc/",
            description="Write to system directory",
            action=SafetyAction.BLOCK,
            tool_names=["write_file", "edit_files"],
        )

        assert rule.tool_names == ["write", "edit"]
        assert rule.matches("write", ["/etc/hosts", "content"])
        assert rule.matches("edit_file", ["/etc/hosts", "content"])

    def test_to_dict(self):
        """Test converting rule to dictionary."""
        rule = SafetyRule(
            rule_id="test",
            category=SafetyCategory.FILE,
            pattern=r"rm.*-rf",
            description="Recursive delete",
            action=SafetyAction.REQUIRE_CONFIRMATION,
            confirmation_prompt="Really delete?",
        )

        rule_dict = rule.to_dict()

        assert rule_dict["rule_id"] == "test"
        assert rule_dict["category"] == "file"
        assert rule_dict["action"] == "require_confirmation"
        assert rule_dict["confirmation_prompt"] == "Really delete?"


class TestSafetyCoordinator:
    """Test suite for SafetyCoordinator."""

    def test_initialization_with_defaults(self):
        """Test coordinator initialization with default rules."""
        coordinator = SafetyCoordinator()

        # Should have default rules loaded
        rules = coordinator.list_rules()
        assert len(rules) > 0

    def test_initialization_without_defaults(self):
        """Test coordinator initialization without default rules."""
        coordinator = SafetyCoordinator(enable_default_rules=False)

        rules = coordinator.list_rules()
        assert len(rules) == 0

    def test_register_rule(self):
        """Test registering a safety rule."""
        coordinator = SafetyCoordinator(enable_default_rules=False)

        rule = SafetyRule(
            rule_id="test_rule",
            category=SafetyCategory.SHELL,
            pattern=r"rm.*-rf",
            description="Recursive delete",
            action=SafetyAction.BLOCK,
            severity=9,
        )

        coordinator.register_rule(rule)

        retrieved = coordinator.get_rule("test_rule")
        assert retrieved is not None
        assert retrieved.rule_id == "test_rule"

    def test_unregister_rule(self):
        """Test unregistering a safety rule."""
        coordinator = SafetyCoordinator(enable_default_rules=False)

        rule = SafetyRule(
            rule_id="test",
            category=SafetyCategory.SHELL,
            pattern=r"test",
            description="Test",
            action=SafetyAction.WARN,
        )

        coordinator.register_rule(rule)
        assert coordinator.get_rule("test") is not None

        result = coordinator.unregister_rule("test")
        assert result is True
        assert coordinator.get_rule("test") is None

    def test_unregister_nonexistent_rule(self):
        """Test unregistering a rule that doesn't exist."""
        coordinator = SafetyCoordinator()

        result = coordinator.unregister_rule("nonexistent")
        assert result is False

    def test_list_rules(self):
        """Test listing all rules."""
        coordinator = SafetyCoordinator(enable_default_rules=False)

        coordinator.register_rule(
            SafetyRule(
                rule_id="rule1",
                category=SafetyCategory.GIT,
                pattern="git",
                description="Git rule",
                action=SafetyAction.WARN,
            )
        )
        coordinator.register_rule(
            SafetyRule(
                rule_id="rule2",
                category=SafetyCategory.DOCKER,
                pattern="docker",
                description="Docker rule",
                action=SafetyAction.WARN,
            )
        )

        all_rules = coordinator.list_rules()
        assert len(all_rules) == 2

        git_rules = coordinator.list_rules(category=SafetyCategory.GIT)
        assert len(git_rules) == 1
        assert git_rules[0].rule_id == "rule1"

    def test_check_safety_no_match(self):
        """Test safety check with no matching rules."""
        coordinator = SafetyCoordinator(enable_default_rules=False)

        result = coordinator.check_safety("safe_tool", ["safe", "args"])

        assert result.is_safe is True
        assert result.action == SafetyAction.ALLOW
        assert len(result.matched_rules) == 0

    def test_check_safety_with_block(self):
        """Test safety check that results in block."""
        coordinator = SafetyCoordinator(enable_default_rules=False)

        coordinator.register_rule(
            SafetyRule(
                rule_id="block_rule",
                category=SafetyCategory.SHELL,
                pattern=r"format",
                description="Format disk",
                action=SafetyAction.BLOCK,
                severity=10,
                tool_names=["shell"],
            )
        )

        result = coordinator.check_safety("shell", ["format", "/dev/sda1"])

        assert result.is_safe is False
        assert result.action == SafetyAction.BLOCK
        assert len(result.matched_rules) == 1
        assert result.block_reason == "Format disk"

    def test_check_safety_with_warn(self):
        """Test safety check that results in warning."""
        coordinator = SafetyCoordinator(enable_default_rules=False)

        coordinator.register_rule(
            SafetyRule(
                rule_id="warn_rule",
                category=SafetyCategory.DOCKER,
                pattern=r"rm",
                description="Remove container",
                action=SafetyAction.WARN,
                severity=5,
                tool_names=["docker"],
            )
        )

        result = coordinator.check_safety("docker", ["rm", "container_id"])

        assert result.is_safe is True
        assert result.action == SafetyAction.WARN
        assert len(result.warnings) > 0

    def test_check_safety_with_confirmation(self):
        """Test safety check that requires confirmation."""
        coordinator = SafetyCoordinator(enable_default_rules=False)

        coordinator.register_rule(
            SafetyRule(
                rule_id="confirm_rule",
                category=SafetyCategory.GIT,
                pattern=r"push.*--force",
                description="Force push",
                action=SafetyAction.REQUIRE_CONFIRMATION,
                severity=7,
                confirmation_prompt="Are you sure you want to force push?",
                tool_names=["git"],
            )
        )

        result = coordinator.check_safety("git", ["push", "--force"])

        assert result.is_safe is False
        assert result.action == SafetyAction.REQUIRE_CONFIRMATION
        assert result.confirmation_prompt == "Are you sure you want to force push?"

    def test_check_safety_max_severity(self):
        """Test max severity blocking."""
        coordinator = SafetyCoordinator(
            enable_default_rules=False,
            max_severity_to_allow=5,
        )

        coordinator.register_rule(
            SafetyRule(
                rule_id="high_severity",
                category=SafetyCategory.SHELL,
                pattern=r"dangerous",
                description="Dangerous command",
                action=SafetyAction.WARN,
                severity=8,  # Above max_severity_to_allow
            )
        )

        result = coordinator.check_safety("shell", ["dangerous", "command"])

        # Should be blocked even though action is WARN
        assert result.is_safe is False
        assert result.action == SafetyAction.BLOCK

    def test_check_safety_strict_mode(self):
        """Test strict mode promotes warnings to blocks."""
        coordinator = SafetyCoordinator(
            enable_default_rules=False,
            strict_mode=True,
        )

        coordinator.register_rule(
            SafetyRule(
                rule_id="warn_rule",
                category=SafetyCategory.FILE,
                pattern=r"delete",
                description="Delete file",
                action=SafetyAction.WARN,
                severity=3,
            )
        )

        result = coordinator.check_safety("file", ["delete", "test.txt"])

        # In strict mode, WARN becomes BLOCK
        assert result.is_safe is False
        assert result.action == SafetyAction.BLOCK

    def test_is_operation_safe(self):
        """Test the is_operation_safe convenience method."""
        coordinator = SafetyCoordinator(enable_default_rules=False)

        coordinator.register_rule(
            SafetyRule(
                rule_id="block_rule",
                category=SafetyCategory.SHELL,
                pattern=r"dangerous",
                description="Dangerous",
                action=SafetyAction.BLOCK,
                tool_names=["shell"],
            )
        )

        assert coordinator.is_operation_safe("shell", ["safe", "command"])
        assert not coordinator.is_operation_safe("shell", ["dangerous"])

    def test_get_stats(self):
        """Test getting safety statistics."""
        coordinator = SafetyCoordinator(enable_default_rules=False)

        coordinator.register_rule(
            SafetyRule(
                rule_id="warn_rule",
                category=SafetyCategory.DOCKER,
                pattern=r"rm",
                description="Remove",
                action=SafetyAction.WARN,
                tool_names=["docker"],
            )
        )
        coordinator.register_rule(
            SafetyRule(
                rule_id="block_rule",
                category=SafetyCategory.SHELL,
                pattern=r"format",
                description="Format",
                action=SafetyAction.BLOCK,
                tool_names=["shell"],
            )
        )

        # Make some checks
        coordinator.check_safety("docker", ["rm", "container"])
        coordinator.check_safety("shell", ["format", "/dev/sda1"])
        coordinator.check_safety("safe_tool", ["args"])

        stats = coordinator.get_stats()

        assert stats.total_checks == 3
        assert stats.warned_operations == 1
        assert stats.blocked_operations == 1
        assert "warn_rule" in stats.rule_hits
        assert "block_rule" in stats.rule_hits

    def test_get_stats_dict(self):
        """Test getting statistics as dictionary."""
        coordinator = SafetyCoordinator()

        coordinator.check_safety("safe_tool", ["args"])

        stats_dict = coordinator.get_stats_dict()

        assert isinstance(stats_dict, dict)
        assert "total_checks" in stats_dict
        assert "blocked_operations" in stats_dict
        assert "rule_hits" in stats_dict

    def test_reset_stats(self):
        """Test resetting statistics."""
        coordinator = SafetyCoordinator()

        coordinator.check_safety("tool", ["args"])

        stats_before = coordinator.get_stats()
        assert stats_before.total_checks > 0

        coordinator.reset_stats()

        stats_after = coordinator.get_stats()
        assert stats_after.total_checks == 0

    def test_set_strict_mode(self):
        """Test setting strict mode."""
        coordinator = SafetyCoordinator(enable_default_rules=False)

        assert coordinator._strict_mode is False

        coordinator.set_strict_mode(True)
        assert coordinator._strict_mode is True

        coordinator.set_strict_mode(False)
        assert coordinator._strict_mode is False

    def test_set_max_severity_to_allow(self):
        """Test setting max severity to allow."""
        coordinator = SafetyCoordinator()

        coordinator.set_max_severity_to_allow(7)
        assert coordinator._max_severity_to_allow == 7

    def test_clear_rules(self):
        """Test clearing all rules."""
        coordinator = SafetyCoordinator()

        # Should have default rules
        assert len(coordinator.list_rules()) > 0

        coordinator.clear_rules()

        assert len(coordinator.list_rules()) == 0

    def test_get_observability_data(self):
        """Test getting observability data."""
        coordinator = SafetyCoordinator()

        obs_data = coordinator.get_observability_data()

        assert obs_data["source_type"] == "coordinator"
        assert obs_data["coordinator_type"] == "safety"
        assert "stats" in obs_data
        assert "config" in obs_data
        assert obs_data["config"]["total_rules"] > 0

    def test_default_git_force_push_rule(self):
        """Test that default git force push rule exists and works."""
        coordinator = SafetyCoordinator(enable_default_rules=True)

        result = coordinator.check_safety("git", ["push", "--force", "origin", "main"])

        # Should block force push to main
        assert result.is_safe is False
        assert result.action == SafetyAction.BLOCK

    def test_default_git_force_push_non_main(self):
        """Test force push to non-main branch requires confirmation."""
        coordinator = SafetyCoordinator(enable_default_rules=True)

        result = coordinator.check_safety("git", ["push", "--force", "origin", "feature"])

        # Should require confirmation (not block)
        assert result.is_safe is False
        assert result.action == SafetyAction.REQUIRE_CONFIRMATION

    def test_default_file_overwrite_rule_blocks_canonical_write(self):
        """Test default system-path rule matches canonical write tool names."""
        coordinator = SafetyCoordinator(enable_default_rules=True)

        result = coordinator.check_safety("write", ["/etc/hosts", "127.0.0.1 localhost"])

        assert result.is_safe is False
        assert result.action == SafetyAction.BLOCK
        assert result.block_reason == "Write to system directory"

    def test_default_file_overwrite_rule_blocks_canonical_edit(self):
        """Test default system-path rule matches canonical edit tool names."""
        coordinator = SafetyCoordinator(enable_default_rules=True)

        result = coordinator.check_safety("edit", ["/usr/bin/python", "replace shebang"])

        assert result.is_safe is False
        assert result.action == SafetyAction.BLOCK


class TestSafetyCheckResult:
    """Test suite for SafetyCheckResult."""

    def test_to_dict(self):
        """Test converting result to dictionary."""
        result = SafetyCheckResult(
            is_safe=False,
            action=SafetyAction.BLOCK,
            block_reason="Test reason",
        )

        result_dict = result.to_dict()

        assert result_dict["is_safe"] is False
        assert result_dict["action"] == "block"
        assert result_dict["block_reason"] == "Test reason"
