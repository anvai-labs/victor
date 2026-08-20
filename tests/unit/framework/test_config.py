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

"""Tests for victor.framework.config module (Phase 5: Framework-level configs).

These tests verify the new framework-level configuration classes
promoted from verticals to eliminate duplication.
"""

import pytest

from victor.agent.config import UnifiedAgentConfig as CanonicalUnifiedAgentConfig
from victor.framework.config import (
    UnifiedAgentConfig,
    SafetyConfig,
    StyleConfig,
    ToolConfig,
    SafetyLevel,
    SafetyRule,
    SafetyEnforcer,
    ExecutionConfig,
)

# =============================================================================
# Agent Config Re-export Tests
# =============================================================================


class TestAgentConfigExports:
    """Tests for framework-facing agent config exports."""

    def test_unified_agent_config_re_export(self):
        """framework.config should re-export the canonical UnifiedAgentConfig."""
        assert UnifiedAgentConfig is CanonicalUnifiedAgentConfig


# =============================================================================
# SafetyConfig Tests
# =============================================================================


class TestSafetyConfig:
    """Tests for SafetyConfig class."""

    def test_default_values(self):
        """SafetyConfig should have sensible defaults."""
        config = SafetyConfig()

        assert config.level == SafetyLevel.MEDIUM
        assert config.require_confirmation is False
        assert config.blocked_operations == []
        assert config.audit_log is True
        assert config.dry_run is False

    def test_custom_values(self):
        """SafetyConfig should accept custom values."""
        config = SafetyConfig(
            level=SafetyLevel.HIGH,
            require_confirmation=True,
            blocked_operations=["rm -rf /", "git push --force"],
            audit_log=False,
            dry_run=True,
        )

        assert config.level == SafetyLevel.HIGH
        assert config.require_confirmation is True
        assert config.blocked_operations == ["rm -rf /", "git push --force"]
        assert config.audit_log is False
        assert config.dry_run is True

    def test_from_dict(self):
        """from_dict should create SafetyConfig from dict."""
        data = {
            "level": "high",
            "require_confirmation": True,
            "blocked_operations": ["dangerous_op"],
            "audit_log": False,
            "dry_run": True,
        }

        config = SafetyConfig.from_dict(data)

        assert config.level == SafetyLevel.HIGH
        assert config.require_confirmation is True
        assert config.blocked_operations == ["dangerous_op"]
        assert config.audit_log is False
        assert config.dry_run is True

    def test_from_dict_defaults(self):
        """from_dict should use defaults for missing values."""
        config = SafetyConfig.from_dict({})

        assert config.level == SafetyLevel.MEDIUM
        assert config.require_confirmation is False
        assert config.audit_log is True
        assert config.dry_run is False

    def test_to_dict(self):
        """to_dict should serialize SafetyConfig."""
        config = SafetyConfig(
            level=SafetyLevel.HIGH,
            require_confirmation=True,
            blocked_operations=["test"],
        )

        data = config.to_dict()

        assert data["level"] == "high"
        assert data["require_confirmation"] is True
        assert data["blocked_operations"] == ["test"]
        assert data["audit_log"] is True


# =============================================================================
# StyleConfig Tests
# =============================================================================


class TestStyleConfig:
    """Tests for StyleConfig class."""

    def test_default_values(self):
        """StyleConfig should have None defaults."""
        config = StyleConfig()

        assert config.formatter is None
        assert config.formatter_options == {}
        assert config.linter is None
        assert config.linter_options == {}
        assert config.style_guide is None

    def test_custom_values(self):
        """StyleConfig should accept custom values."""
        config = StyleConfig(
            formatter="black",
            formatter_options={"line_length": 100},
            linter="ruff",
            linter_options={"select": ["E", "F"]},
            style_guide="PEP8",
        )

        assert config.formatter == "black"
        assert config.formatter_options == {"line_length": 100}
        assert config.linter == "ruff"
        assert config.linter_options == {"select": ["E", "F"]}
        assert config.style_guide == "PEP8"

    def test_from_dict(self):
        """from_dict should create StyleConfig from dict."""
        data = {
            "formatter": "prettier",
            "formatter_options": {"trailing_comma": True},
            "linter": "eslint",
            "linter_options": {"extends": "airbnb"},
            "style_guide": "AirBnB",
        }

        config = StyleConfig.from_dict(data)

        assert config.formatter == "prettier"
        assert config.formatter_options == {"trailing_comma": True}
        assert config.linter == "eslint"
        assert config.style_guide == "AirBnB"


# =============================================================================
# ToolConfig Tests
# =============================================================================


class TestToolConfig:
    """Tests for ToolConfig class."""

    def test_default_values(self):
        """ToolConfig should have sensible defaults."""
        config = ToolConfig()

        assert config.enabled_tools == []
        assert config.disabled_tools == []
        assert config.tool_settings == {}
        assert config.max_tool_budget == 100
        assert config.require_confirmation is False

    def test_custom_values(self):
        """ToolConfig should accept custom values."""
        config = ToolConfig(
            enabled_tools=["read", "write"],
            disabled_tools=["shell"],
            tool_settings={"docker": {"runtime": "python3.12"}},
            max_tool_budget=50,
            require_confirmation=True,
        )

        assert config.enabled_tools == ["read", "write"]
        assert config.disabled_tools == ["shell"]
        assert config.tool_settings == {"docker": {"runtime": "python3.12"}}
        assert config.max_tool_budget == 50
        assert config.require_confirmation is True

    def test_from_dict(self):
        """from_dict should create ToolConfig from dict."""
        data = {
            "enabled_tools": ["read"],
            "disabled_tools": ["write"],
            "tool_settings": {"git": {"default_branch": "main"}},
            "max_tool_budget": 75,
            "require_confirmation": True,
        }

        config = ToolConfig.from_dict(data)

        assert config.enabled_tools == ["read"]
        assert config.disabled_tools == ["write"]
        assert config.tool_settings == {"git": {"default_branch": "main"}}
        assert config.max_tool_budget == 75

    def test_is_tool_enabled_whitelist(self):
        """is_tool_enabled should check whitelist."""
        config = ToolConfig(enabled_tools=["read", "write"])

        assert config.is_tool_enabled("read") is True
        assert config.is_tool_enabled("write") is True
        assert config.is_tool_enabled("shell") is False

    def test_is_tool_enabled_blacklist(self):
        """is_tool_enabled should check blacklist."""
        config = ToolConfig(disabled_tools=["shell", "git"])

        assert config.is_tool_enabled("read") is True
        assert config.is_tool_enabled("shell") is False
        assert config.is_tool_enabled("git") is False

    def test_is_tool_enabled_no_restrictions(self):
        """is_tool_enabled should allow all when no restrictions."""
        config = ToolConfig()

        assert config.is_tool_enabled("read") is True
        assert config.is_tool_enabled("write") is True
        assert config.is_tool_enabled("shell") is True

    def test_get_tool_setting(self):
        """get_tool_setting should return tool-specific setting."""
        config = ToolConfig(
            tool_settings={
                "docker": {"runtime": "python3.12"},
                "git": {"default_branch": "main"},
            }
        )

        assert config.get_tool_setting("docker", "runtime") == "python3.12"
        assert config.get_tool_setting("git", "default_branch") == "main"
        assert config.get_tool_setting("unknown", "key") is None
        assert config.get_tool_setting("docker", "unknown_key", "default") == "default"


# =============================================================================
# SafetyEnforcer Tests
# =============================================================================


class TestSafetyEnforcer:
    """Tests for SafetyEnforcer class."""

    def test_initialization(self):
        """SafetyEnforcer should initialize with config."""
        config = SafetyConfig(level=SafetyLevel.HIGH)
        enforcer = SafetyEnforcer(config)

        assert enforcer.config == config
        assert enforcer.rules == []

    def test_add_rule(self):
        """add_rule should register safety rule."""
        enforcer = SafetyEnforcer(config=SafetyConfig())
        rule = SafetyRule(
            name="test_rule",
            description="Test rule",
            check_fn=lambda op: "dangerous" in op,
            level=SafetyLevel.HIGH,
        )

        enforcer.add_rule(rule)

        assert len(enforcer.rules) == 1
        assert enforcer.rules[0] == rule

    def test_add_rule_duplicate(self):
        """add_rule should raise for duplicate rule names."""
        enforcer = SafetyEnforcer(config=SafetyConfig())
        rule1 = SafetyRule(name="test", description="Test", check_fn=lambda op: True)
        rule2 = SafetyRule(name="test", description="Test 2", check_fn=lambda op: False)

        enforcer.add_rule(rule1)

        with pytest.raises(ValueError, match="already registered"):
            enforcer.add_rule(rule2)

    def test_remove_rule(self):
        """remove_rule should remove rule by name."""
        enforcer = SafetyEnforcer(config=SafetyConfig())
        rule = SafetyRule(name="test", description="Test", check_fn=lambda op: True)
        enforcer.add_rule(rule)

        result = enforcer.remove_rule("test")

        assert result is True
        assert len(enforcer.rules) == 0

    def test_remove_rule_not_found(self):
        """remove_rule should return False for unknown rule."""
        enforcer = SafetyEnforcer(config=SafetyConfig())

        result = enforcer.remove_rule("unknown")

        assert result is False

    def test_check_operation_allowed(self):
        """check_operation should allow safe operations."""
        enforcer = SafetyEnforcer(config=SafetyConfig())

        allowed, reason = enforcer.check_operation("safe operation")

        assert allowed is True
        assert reason is None

    def test_check_operation_blocked_by_config(self):
        """check_operation should block operations in blocked_operations list."""
        config = SafetyConfig(blocked_operations=["rm -rf /", "git push --force"])
        enforcer = SafetyEnforcer(config)

        allowed, reason = enforcer.check_operation("rm -rf /")

        assert allowed is False
        assert "Blocked by configuration" in reason
        assert "rm -rf /" in reason

    def test_check_operation_blocked_by_rule_high(self):
        """check_operation should block when rule level is HIGH."""
        config = SafetyConfig(level=SafetyLevel.MEDIUM)
        enforcer = SafetyEnforcer(config)
        enforcer.add_rule(
            SafetyRule(
                name="block_force_push",
                description="Block force push",
                check_fn=lambda op: "git push --force" in op,
                level=SafetyLevel.HIGH,
            )
        )

        allowed, reason = enforcer.check_operation("git push --force origin main")

        assert allowed is False
        assert "block_force_push" in reason

    def test_check_operation_blocked_by_rule_medium(self):
        """check_operation should block MEDIUM rules unless config is LOW."""
        config = SafetyConfig(level=SafetyLevel.MEDIUM)
        enforcer = SafetyEnforcer(config)
        enforcer.add_rule(
            SafetyRule(
                name="warn_push",
                description="Warn about push",
                check_fn=lambda op: "git push" in op,
                level=SafetyLevel.MEDIUM,
            )
        )

        allowed, reason = enforcer.check_operation("git push origin main")

        assert allowed is False
        assert "warn_push" in reason

    def test_check_operation_warn_only_low(self):
        """check_operation should allow LOW level rules."""
        config = SafetyConfig(level=SafetyLevel.MEDIUM)
        enforcer = SafetyEnforcer(config)
        enforcer.add_rule(
            SafetyRule(
                name="recommend_tests",
                description="Recommend tests",
                check_fn=lambda op: "git commit" in op,
                level=SafetyLevel.LOW,
            )
        )

        allowed, reason = enforcer.check_operation("git commit -m 'test'")

        assert allowed is True
        assert reason is None  # LOW level doesn't block at MEDIUM config

    def test_check_operation_dry_run(self):
        """check_operation should always allow in dry_run mode."""
        config = SafetyConfig(dry_run=True, blocked_operations=["rm -rf /"])
        enforcer = SafetyEnforcer(config)

        allowed, reason = enforcer.check_operation("rm -rf /")

        assert allowed is True
        assert "[DRY RUN]" in reason

    def test_get_rules_by_level(self):
        """get_rules_by_level should filter rules by level."""
        enforcer = SafetyEnforcer(config=SafetyConfig())
        enforcer.add_rule(
            SafetyRule(
                name="high",
                description="High",
                check_fn=lambda op: True,
                level=SafetyLevel.HIGH,
            )
        )
        enforcer.add_rule(
            SafetyRule(
                name="medium",
                description="Medium",
                check_fn=lambda op: True,
                level=SafetyLevel.MEDIUM,
            )
        )
        enforcer.add_rule(
            SafetyRule(
                name="low",
                description="Low",
                check_fn=lambda op: True,
                level=SafetyLevel.LOW,
            )
        )

        high_rules = enforcer.get_rules_by_level(SafetyLevel.HIGH)
        medium_rules = enforcer.get_rules_by_level(SafetyLevel.MEDIUM)

        assert len(high_rules) == 1
        assert high_rules[0].name == "high"
        assert len(medium_rules) == 1
        assert medium_rules[0].name == "medium"

    def test_clear_rules(self):
        """clear_rules should remove all rules."""
        enforcer = SafetyEnforcer(config=SafetyConfig())
        enforcer.add_rule(SafetyRule(name="test1", description="Test", check_fn=lambda op: True))
        enforcer.add_rule(SafetyRule(name="test2", description="Test", check_fn=lambda op: True))

        assert len(enforcer.rules) == 2

        enforcer.clear_rules()

        assert len(enforcer.rules) == 0


# =============================================================================
# Integration Tests with Vertical Safety Rules
# =============================================================================


class TestVerticalSafetyIntegration:
    """Tests for vertical-specific safety rule integration."""

    def test_coding_safety_git_rules(self):
        """Coding git safety rules should block dangerous git operations."""
        from victor.framework.safety import create_git_safety_rules

        enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))
        create_git_safety_rules(enforcer, block_force_push=True, block_main_push=True)

        # Test force push blocking
        allowed, reason = enforcer.check_operation("git push --force origin main")
        assert allowed is False
        assert "force" in reason.lower()

        # Test main push blocking
        allowed, reason = enforcer.check_operation("git push origin main")
        assert allowed is False
        assert "main" in reason.lower() or "push" in reason.lower()

    def test_coding_safety_file_rules(self):
        """Coding file safety rules should block destructive commands."""
        from victor.framework.safety import create_file_safety_rules

        enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))
        create_file_safety_rules(enforcer, block_destructive_commands=True)

        # Test destructive command blocking
        allowed, reason = enforcer.check_operation("rm -rf /")
        assert allowed is False
        assert "destructive" in reason.lower() or "rm -rf" in reason.lower()

    # NOTE: Tests for vertical-specific safety rules have been migrated to their
    # respective vertical repositories:
    # - victor-devops: tests/test_safety.py (deployment, container, infrastructure rules)
    # - victor-rag: tests/test_safety.py (deletion, ingestion rules)
    # - victor-research: tests/test_safety.py (source, content rules)
    # - victor-dataanalysis: tests/test_safety.py (PII, export rules)
    # - victor-coding: tests/safety/test_safety_integration.py (all coding rules)

    def test_benchmark_safety_repository_rules(self):
        """BenchmarkSafetyExtension should block operations on production paths."""
        from victor.benchmark.safety import BenchmarkSafetyExtension

        ext = BenchmarkSafetyExtension()

        # Test blocking operations on production paths (benchmark_production_modify rule)
        is_safe = ext.check_operation("shell", ["/production/deploy"])
        assert not is_safe

        # Test blocking git force push (benchmark_git_force_push rule, tool_names=["git"])
        is_safe = ext.check_operation("git", ["push", "--force", "origin", "main"])
        assert not is_safe

    def test_benchmark_safety_resource_rules(self):
        """BenchmarkSafetyExtension should block dangerous system commands."""
        from victor.benchmark.safety import BenchmarkSafetyExtension

        ext = BenchmarkSafetyExtension()

        # Test blocking dangerous commands (benchmark_dangerous_commands rule)
        is_safe = ext.check_operation("shell", ["rm -rf /"])
        assert not is_safe

        # Test blocking format commands
        is_safe = ext.check_operation("execute_bash", ["format C:"])
        assert not is_safe

    def test_benchmark_safety_test_rules(self):
        """BenchmarkSafetyExtension should block production modifications."""
        from victor.benchmark.safety import BenchmarkSafetyExtension

        ext = BenchmarkSafetyExtension()

        # Test blocking production path operations
        is_safe = ext.check_operation("shell", ["deploy to /prod/server"])
        assert not is_safe

        # Test blocking release branch operations
        is_safe = ext.check_operation("shell", ["push to release"])
        assert not is_safe

    def test_benchmark_safety_data_rules(self):
        """BenchmarkSafetyExtension should block database/data file deletion."""
        from victor.benchmark.safety import BenchmarkSafetyExtension

        ext = BenchmarkSafetyExtension()

        # Test blocking database file deletion (benchmark_file_deletion rule)
        is_safe = ext.check_operation("file_delete", ["benchmark.db"])
        assert not is_safe

        # Test blocking data file deletion
        is_safe = ext.check_operation("file_delete", ["important_database"])
        assert not is_safe

    def test_benchmark_safety_extension_has_rules(self):
        """BenchmarkSafetyExtension should register safety rules on creation."""
        from victor.benchmark.safety import BenchmarkSafetyExtension

        ext = BenchmarkSafetyExtension()

        # Should have rules registered
        coordinator = ext.get_coordinator()
        rules = coordinator.list_rules()
        assert len(rules) > 0

        # Should have stats tracking
        stats = ext.get_safety_stats()
        assert isinstance(stats, dict)


# =============================================================================
# SafetyConfig Validation Tests — Wave R
# =============================================================================


class TestSafetyConfigValidation:
    """SafetyConfig should validate enum values in from_dict."""

    def test_from_dict_with_invalid_level_raises(self):
        """from_dict should raise ValueError for invalid safety level."""
        with pytest.raises(ValueError, match="Invalid safety level"):
            SafetyConfig.from_dict({"level": "invalid_level"})

        with pytest.raises(ValueError, match="Invalid safety level"):
            SafetyConfig.from_dict({"level": "not_a_real_level"})

    def test_from_dict_with_valid_level_passes(self):
        """from_dict should accept valid safety levels."""
        config = SafetyConfig.from_dict({"level": "low"})
        assert config.level == SafetyLevel.LOW

        config = SafetyConfig.from_dict({"level": "medium"})
        assert config.level == SafetyLevel.MEDIUM

        config = SafetyConfig.from_dict({"level": "high"})
        assert config.level == SafetyLevel.HIGH

    def test_from_dict_with_case_insensitive_level(self):
        """from_dict should handle case-insensitive level values."""
        config = SafetyConfig.from_dict({"level": "LOW"})
        assert config.level == SafetyLevel.LOW

        config = SafetyConfig.from_dict({"level": "High"})
        assert config.level == SafetyLevel.HIGH

        config = SafetyConfig.from_dict({"level": "MeDiUm"})
        assert config.level == SafetyLevel.MEDIUM


# =============================================================================
# ExecutionConfig Validation Tests — Wave P
# =============================================================================


class TestExecutionConfigValidation:
    """ExecutionConfig numeric fields must have valid ranges."""

    def test_default_config_passes(self):
        """Default ExecutionConfig should pass all validations."""
        config = ExecutionConfig()
        assert config.max_iterations == 25
        assert config.timeout is None
        assert config.recursion_limit == 100

    def test_max_iterations_below_one_raises(self):
        """max_iterations must be >= 1."""
        with pytest.raises(ValueError, match="max_iterations must be >= 1"):
            ExecutionConfig(max_iterations=0)

        with pytest.raises(ValueError, match="max_iterations must be >= 1"):
            ExecutionConfig(max_iterations=-10)

    def test_timeout_below_point_one_raises(self):
        """timeout must be >= 0.1 if set."""
        with pytest.raises(ValueError, match="timeout must be >= 0.1"):
            ExecutionConfig(timeout=0.0)

        with pytest.raises(ValueError, match="timeout must be >= 0.1"):
            ExecutionConfig(timeout=0.01)

    def test_timeout_none_passes(self):
        """timeout=None should pass (no limit)."""
        config = ExecutionConfig(timeout=None)
        assert config.timeout is None

    def test_recursion_limit_below_one_raises(self):
        """recursion_limit must be >= 1."""
        with pytest.raises(ValueError, match="recursion_limit must be >= 1"):
            ExecutionConfig(recursion_limit=0)

        with pytest.raises(ValueError, match="recursion_limit must be >= 1"):
            ExecutionConfig(recursion_limit=-1)

    def test_boundary_values_pass(self):
        """Boundary values (1, 0.1) should pass validation."""
        config = ExecutionConfig(
            max_iterations=1,
            timeout=0.1,
            recursion_limit=1,
        )
        assert config.max_iterations == 1
        assert config.timeout == 0.1
        assert config.recursion_limit == 1

    def test_large_values_pass(self):
        """Large reasonable values should pass validation."""
        config = ExecutionConfig(
            max_iterations=1000000,
            timeout=86400.0,  # 1 day
            recursion_limit=10000,
        )
        assert config.max_iterations == 1000000
        assert config.timeout == 86400.0
        assert config.recursion_limit == 10000
