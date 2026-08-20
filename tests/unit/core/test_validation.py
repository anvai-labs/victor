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

"""Tests for configuration validation module."""

import pytest
from pathlib import Path
from tempfile import TemporaryDirectory

from victor.core.validation import (
    AgentConfigSchema,
    CacheConfigSchema,
    ConfigurationBuilder,
    ConfigValidator,
    DependencyRule,
    EnumRule,
    ModelConfigSchema,
    ObservabilityConfigSchema,
    PathRule,
    ProviderConfigSchema,
    RangeRule,
    RegexRule,
    ResilienceConfigSchema,
    ToolConfigSchema,
    ValidationIssue,
    ConfigValidationResult,
    ValidationSeverity,
    validate_agent_config,
    validate_model_config,
    validate_provider_config,
)

# =============================================================================
# ConfigValidationResult Tests
# =============================================================================


class TestConfigValidationResult:
    """Tests for ConfigValidationResult."""

    def test_empty_result_is_valid(self):
        """Empty result should be valid."""
        result = ConfigValidationResult()
        assert result.is_valid is True
        assert len(result.errors) == 0
        assert len(result.warnings) == 0

    def test_result_with_error_is_invalid(self):
        """Result with error should be invalid."""
        result = ConfigValidationResult()
        result.add_error("field", "Something went wrong")

        assert result.is_valid is False
        assert len(result.errors) == 1
        assert result.errors[0].message == "Something went wrong"

    def test_result_with_only_warnings_is_valid(self):
        """Result with only warnings should still be valid."""
        result = ConfigValidationResult()
        result.add_warning("field", "Consider changing this")

        assert result.is_valid is True
        assert len(result.warnings) == 1

    def test_merge_results(self):
        """Test merging results."""
        result1 = ConfigValidationResult()
        result1.add_error("field1", "Error 1")

        result2 = ConfigValidationResult()
        result2.add_error("field2", "Error 2")
        result2.add_warning("field3", "Warning")

        result1.merge(result2)

        assert len(result1.errors) == 2
        assert len(result1.warnings) == 1

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = ConfigValidationResult()
        result.add_error("field", "Error", code="E001")

        data = result.to_dict()

        assert data["is_valid"] is False
        assert data["error_count"] == 1
        assert data["warning_count"] == 0
        assert len(data["issues"]) == 1
        assert data["issues"][0]["code"] == "E001"


class TestValidationIssue:
    """Tests for ValidationIssue."""

    def test_str_format(self):
        """Test string formatting."""
        issue = ValidationIssue(
            path="config.api_key",
            message="API key is required",
            severity=ValidationSeverity.ERROR,
        )

        formatted = str(issue)
        assert "[ERROR]" in formatted
        assert "config.api_key" in formatted
        assert "API key is required" in formatted


# =============================================================================
# Validation Rules Tests
# =============================================================================


class TestRegexRule:
    """Tests for RegexRule."""

    def test_valid_match(self):
        """Test valid regex match."""
        rule = RegexRule(r"^sk-[a-zA-Z0-9]+$")
        result = rule.validate("sk-abc123", {})

        assert result.is_valid

    def test_invalid_match(self):
        """Test invalid regex match."""
        rule = RegexRule(r"^sk-[a-zA-Z0-9]+$")
        result = rule.validate("invalid-key", {})

        assert not result.is_valid

    def test_none_value_allowed(self):
        """Test None value is allowed."""
        rule = RegexRule(r"^sk-")
        result = rule.validate(None, {})

        assert result.is_valid


class TestRangeRule:
    """Tests for RangeRule."""

    def test_in_range(self):
        """Test value in range."""
        rule = RangeRule(min_value=0, max_value=100)
        result = rule.validate(50, {})

        assert result.is_valid

    def test_below_min(self):
        """Test value below minimum."""
        rule = RangeRule(min_value=0, max_value=100)
        result = rule.validate(-1, {})

        assert not result.is_valid

    def test_above_max(self):
        """Test value above maximum."""
        rule = RangeRule(min_value=0, max_value=100)
        result = rule.validate(101, {})

        assert not result.is_valid

    def test_exclusive_bounds(self):
        """Test exclusive bounds."""
        rule = RangeRule(min_value=0, max_value=100, exclusive=True)

        # Boundary values should fail
        assert not rule.validate(0, {}).is_valid
        assert not rule.validate(100, {}).is_valid

        # Interior values should pass
        assert rule.validate(50, {}).is_valid


class TestEnumRule:
    """Tests for EnumRule."""

    def test_allowed_value(self):
        """Test allowed value."""
        rule = EnumRule({"red", "green", "blue"})
        result = rule.validate("red", {})

        assert result.is_valid

    def test_disallowed_value(self):
        """Test disallowed value."""
        rule = EnumRule({"red", "green", "blue"})
        result = rule.validate("yellow", {})

        assert not result.is_valid

    def test_case_insensitive(self):
        """Test case insensitive matching."""
        rule = EnumRule({"RED", "GREEN", "BLUE"}, case_insensitive=True)
        result = rule.validate("red", {})

        assert result.is_valid


class TestPathRule:
    """Tests for PathRule."""

    def test_path_exists(self):
        """Test path existence check."""
        with TemporaryDirectory() as tmpdir:
            rule = PathRule(must_exist=True)
            result = rule.validate(tmpdir, {})
            assert result.is_valid

    def test_path_not_exists(self):
        """Test non-existent path."""
        rule = PathRule(must_exist=True)
        result = rule.validate("/nonexistent/path/12345", {})

        assert not result.is_valid

    def test_must_be_dir(self):
        """Test directory check."""
        with TemporaryDirectory() as tmpdir:
            rule = PathRule(must_be_dir=True)
            result = rule.validate(tmpdir, {})
            assert result.is_valid


class TestDependencyRule:
    """Tests for DependencyRule."""

    def test_dependency_met(self):
        """Test when dependency is met."""
        rule = DependencyRule(required_if={"type": "api"})
        context = {"type": "api"}

        result = rule.validate("some_value", context)
        assert result.is_valid

    def test_dependency_not_met(self):
        """Test when required but value missing."""
        rule = DependencyRule(required_if={"type": "api"})
        context = {"type": "api"}

        result = rule.validate(None, context)
        assert not result.is_valid


# =============================================================================
# Pydantic Schema Tests
# =============================================================================


class TestProviderConfigSchema:
    """Tests for ProviderConfigSchema."""

    def test_valid_config(self):
        """Test valid provider configuration."""
        config = ProviderConfigSchema(
            name="anthropic",
            api_key="sk-real-key-12345",
            timeout=300,
        )

        assert config.name == "anthropic"
        assert config.api_key == "sk-real-key-12345"

    def test_placeholder_api_key_rejected(self):
        """Test placeholder API keys are rejected."""
        with pytest.raises(Exception):
            ProviderConfigSchema(
                name="anthropic",
                api_key="your-api-key",
            )

    def test_base_url_normalized(self):
        """Test base URL trailing slash removed."""
        config = ProviderConfigSchema(
            name="openai",
            base_url="https://api.openai.com/v1/",
        )

        assert config.base_url == "https://api.openai.com/v1"

    def test_invalid_base_url(self):
        """Test invalid base URL rejected."""
        with pytest.raises(Exception):
            ProviderConfigSchema(
                name="test",
                base_url="ftp://invalid.com",
            )


class TestModelConfigSchema:
    """Tests for ModelConfigSchema."""

    def test_valid_config(self):
        """Test valid model configuration."""
        config = ModelConfigSchema(
            model="claude-3-sonnet-20240229",
            temperature=0.7,
            max_tokens=4096,
        )

        assert config.model == "claude-3-sonnet-20240229"
        assert config.temperature == 0.7

    def test_temperature_bounds(self):
        """Test temperature bounds validation."""
        with pytest.raises(Exception):
            ModelConfigSchema(model="test", temperature=3.0)

        with pytest.raises(Exception):
            ModelConfigSchema(model="test", temperature=-1.0)

    def test_model_name_validation(self):
        """Test model name validation."""
        # Valid names
        config = ModelConfigSchema(model="gpt-4")
        assert config.model == "gpt-4"

        config = ModelConfigSchema(model="claude-3-opus/latest")
        assert config.model == "claude-3-opus/latest"


class TestToolConfigSchema:
    """Tests for ToolConfigSchema."""

    def test_valid_config(self):
        """Test valid tool configuration."""
        config = ToolConfigSchema(
            enabled=True,
            timeout=60,
            cost_tier="MEDIUM",
        )

        assert config.enabled is True
        assert config.cost_tier == "MEDIUM"

    def test_cost_tier_normalized(self):
        """Test cost tier normalized to uppercase."""
        config = ToolConfigSchema(cost_tier="low")
        assert config.cost_tier == "LOW"


class TestCacheConfigSchema:
    """Tests for CacheConfigSchema."""

    def test_valid_config(self):
        """Test valid cache configuration."""
        config = CacheConfigSchema(
            enabled=True,
            ttl_seconds=3600,
            max_size_mb=100,
            strategy="lru",
        )

        assert config.ttl_seconds == 3600
        assert config.strategy == "lru"

    def test_strategy_normalized(self):
        """Test strategy normalized to lowercase."""
        config = CacheConfigSchema(strategy="LRU")
        assert config.strategy == "lru"


class TestResilienceConfigSchema:
    """Tests for ResilienceConfigSchema."""

    def test_valid_config(self):
        """Test valid resilience configuration."""
        config = ResilienceConfigSchema(
            circuit_breaker_enabled=True,
            failure_threshold=5,
            retry_enabled=True,
            max_retries=3,
        )

        assert config.circuit_breaker_enabled is True
        assert config.failure_threshold == 5


class TestObservabilityConfigSchema:
    """Tests for ObservabilityConfigSchema."""

    def test_valid_config(self):
        """Test valid observability configuration."""
        config = ObservabilityConfigSchema(
            metrics_enabled=True,
            tracing_enabled=True,
            logging_level="DEBUG",
            export_format="json",
        )

        assert config.logging_level == "DEBUG"
        assert config.export_format == "json"

    def test_logging_level_normalized(self):
        """Test logging level normalized."""
        config = ObservabilityConfigSchema(logging_level="debug")
        assert config.logging_level == "DEBUG"


# =============================================================================
# ConfigValidator Tests
# =============================================================================


class TestConfigValidator:
    """Tests for ConfigValidator."""

    def test_validate_valid_config(self):
        """Test validating valid configuration."""
        validator = ConfigValidator()
        config = {
            "name": "anthropic",
            "api_key": "sk-test-123456",
            "timeout": 300,
        }

        result = validator.validate(config, ProviderConfigSchema)

        assert result.is_valid

    def test_validate_invalid_config(self):
        """Test validating invalid configuration."""
        validator = ConfigValidator()
        config = {
            "name": "",  # Empty name
            "timeout": -1,  # Negative timeout
        }

        result = validator.validate(config, ProviderConfigSchema)

        assert not result.is_valid
        assert len(result.errors) > 0

    def test_custom_rules(self):
        """Test adding custom validation rules."""
        validator = ConfigValidator()
        validator.add_rule("api_key", RegexRule(r"^sk-"))

        config = {
            "name": "test",
            "api_key": "invalid-key",  # Doesn't start with sk-
        }

        result = validator.validate(config, ProviderConfigSchema)

        # Should have error from custom rule
        assert not result.is_valid


# =============================================================================
# ConfigurationBuilder Tests
# =============================================================================


class TestConfigurationBuilder:
    """Tests for ConfigurationBuilder."""

    def test_build_basic_config(self):
        """Test building basic configuration."""
        config = (
            ConfigurationBuilder()
            .with_provider("anthropic")
            .with_api_key("sk-test-123")
            .with_model("claude-3-sonnet")
            .build()
        )

        assert config["provider"]["name"] == "anthropic"
        assert config["provider"]["api_key"] == "sk-test-123"
        assert config["model"]["model"] == "claude-3-sonnet"

    def test_build_with_all_options(self):
        """Test building with all options."""
        config = (
            ConfigurationBuilder()
            .with_provider("openai")
            .with_api_key("sk-test")
            .with_base_url("https://api.openai.com/v1")
            .with_timeout(120)
            .with_model("gpt-4")
            .with_temperature(0.5)
            .with_max_tokens(2000)
            .with_cache({"enabled": True})
            .with_resilience({"retry_enabled": True})
            .with_observability({"metrics_enabled": True})
            .build()
        )

        assert config["provider"]["timeout"] == 120
        assert config["model"]["temperature"] == 0.5
        assert config["cache"]["enabled"] is True

    def test_build_validated(self):
        """Test building with validation."""
        builder = ConfigurationBuilder().with_provider("anthropic").with_model("claude-3-sonnet")

        # This should work if provider config is valid
        config = builder.build_validated()
        assert config.provider.name == "anthropic"
        assert config.model.model == "claude-3-sonnet"


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience validation functions."""

    def test_validate_provider_config(self):
        """Test validate_provider_config function."""
        result = validate_provider_config(
            {
                "name": "anthropic",
                "timeout": 300,
            }
        )

        assert result.is_valid

    def test_validate_model_config(self):
        """Test validate_model_config function."""
        result = validate_model_config(
            {
                "model": "gpt-4",
                "temperature": 0.7,
            }
        )

        assert result.is_valid

    def test_validate_model_config_invalid(self):
        """Test validate_model_config with invalid config."""
        result = validate_model_config(
            {
                "model": "",  # Empty model
                "temperature": 3.0,  # Out of range
            }
        )

        assert not result.is_valid


class TestAgentConfigSchema:
    """Tests for complete agent configuration."""

    def test_valid_complete_config(self):
        """Test valid complete configuration."""
        config = AgentConfigSchema(
            provider=ProviderConfigSchema(name="anthropic"),
            model=ModelConfigSchema(model="claude-3-sonnet"),
            tools=ToolConfigSchema(enabled=True),
            cache=CacheConfigSchema(enabled=True),
        )

        assert config.provider.name == "anthropic"
        assert config.model.model == "claude-3-sonnet"

    def test_minimal_config(self):
        """Test minimal valid configuration."""
        config = AgentConfigSchema(
            provider=ProviderConfigSchema(name="test"),
            model=ModelConfigSchema(model="test-model"),
        )

        assert config.tools is None
        assert config.cache is None
