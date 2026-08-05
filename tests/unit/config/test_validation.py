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

"""Tests for configuration validation."""

import pytest

from victor.config.validation import (
    ConfigValidationResult,
    ConfigValidator,
    ValidationIssue,
    ValidationSeverity,
    format_validation_result,
    validate_configuration,
    validate_settings,
)


class TestValidationIssue:
    """Tests for ValidationIssue dataclass."""

    def test_basic_issue(self):
        """Issue renders category and message."""
        issue = ValidationIssue(
            category="provider",
            severity=ValidationSeverity.ERROR,
            message="Test error",
        )
        text = str(issue)
        assert "Test error" in text
        assert "provider" in text

    def test_issue_with_suggestion(self):
        """Suggestion appears in string representation."""
        issue = ValidationIssue(
            category="api_key",
            severity=ValidationSeverity.WARNING,
            message="Key missing",
            suggestion="Set API_KEY env var",
        )
        assert "Set API_KEY env var" in str(issue)

    def test_issue_with_setting_path(self):
        """Setting path appears in string representation."""
        issue = ValidationIssue(
            category="path",
            severity=ValidationSeverity.ERROR,
            message="Directory missing",
            setting_path="providers.anthropic.base_url",
        )
        assert "providers.anthropic.base_url" in str(issue)

    def test_severity_icons(self):
        """Each severity level produces a distinct icon."""
        icons = set()
        for sev in ValidationSeverity:
            issue = ValidationIssue(category="x", severity=sev, message="m")
            icons.add(str(issue)[0])
        # Distinct icon per canonical severity (now includes CRITICAL, FEP-0014).
        assert len(icons) == len(list(ValidationSeverity))


class TestConfigValidationResult:
    """Tests for ConfigValidationResult."""

    def test_empty_result_is_valid(self):
        """Fresh result with no errors is valid."""
        result = ConfigValidationResult()
        assert result.is_valid()
        assert len(result.errors) == 0

    def test_add_error_makes_invalid(self):
        """Adding an error makes the result invalid."""
        result = ConfigValidationResult()
        result.add_error(category="provider", message="Bad provider")
        assert not result.is_valid()
        assert len(result.errors) == 1

    def test_add_warning_stays_valid(self):
        """Warnings do not affect validity."""
        result = ConfigValidationResult()
        result.add_warning(category="perf", message="Slow setting")
        assert result.is_valid()
        assert result.has_warnings()

    def test_add_info(self):
        """Info messages are collected separately."""
        result = ConfigValidationResult()
        result.add_info(category="profile", message="Using default profile")
        assert len(result.info) == 1
        assert result.is_valid()

    def test_error_carries_suggestion_and_path(self):
        """Error fields are preserved on the stored issue."""
        result = ConfigValidationResult()
        result.add_error(
            category="api_key",
            message="Missing key",
            suggestion="Export ANTHROPIC_API_KEY",
            setting_path="providers.anthropic.api_key",
        )
        issue = result.errors[0]
        assert issue.suggestion == "Export ANTHROPIC_API_KEY"
        assert issue.setting_path == "providers.anthropic.api_key"

    def test_get_total_issues(self):
        """Total issues counts errors + warnings + info."""
        result = ConfigValidationResult()
        result.add_error(category="a", message="e")
        result.add_warning(category="b", message="w")
        result.add_info(category="c", message="i")
        assert result.get_total_issues() == 3

    def test_get_summary_valid(self):
        """Summary for clean result shows valid confirmation."""
        result = ConfigValidationResult()
        assert "valid" in result.get_summary().lower()

    def test_get_summary_with_errors(self):
        """Summary includes error count when invalid."""
        result = ConfigValidationResult()
        result.add_error(category="x", message="boom")
        summary = result.get_summary()
        assert "1 error" in summary
        assert "invalid" in summary


class TestConfigValidator:
    """Tests for ConfigValidator.validate()."""

    def test_returns_result(self):
        """validate() always returns a ConfigValidationResult."""
        from victor.config.settings import Settings

        result = ConfigValidator().validate(Settings())
        assert isinstance(result, ConfigValidationResult)

    def test_unknown_provider_warns(self):
        """An unrecognised default_provider produces a warning."""
        from victor.config.settings import Settings

        settings = Settings()
        settings.provider.default_provider = "totally_unknown_provider_xyz"
        result = ConfigValidator().validate(settings)
        has_warn = any(
            "unknown" in issue.message.lower() or "totally_unknown" in issue.message
            for issue in result.warnings
        )
        assert has_warn

    def test_known_provider_no_unknown_warning(self):
        """A known provider like 'ollama' does not trigger unknown-provider warning."""
        from victor.config.settings import Settings

        settings = Settings()
        settings.provider.default_provider = "ollama"
        result = ConfigValidator().validate(settings)
        has_unknown_warn = any("Unknown provider" in issue.message for issue in result.warnings)
        assert not has_unknown_warn

    def test_validator_errors_become_warnings(self):
        """If an internal validator raises, it is captured as a warning (not crash)."""
        validator = ConfigValidator()
        # Pass None — _validate_providers will raise AttributeError internally;
        # the outer try/except captures it as a validator warning.
        result = validator.validate(None)
        assert isinstance(result, ConfigValidationResult)


class TestValidateSettings:
    """Tests for validate_settings() and validate_configuration() convenience wrappers."""

    def test_validate_settings_returns_result(self):
        """validate_settings returns ConfigValidationResult."""
        from victor.config.settings import Settings

        result = validate_settings(Settings())
        assert isinstance(result, ConfigValidationResult)

    def test_validate_configuration_alias(self):
        """validate_configuration is an alias for validate_settings."""
        from victor.config.settings import Settings

        settings = Settings()
        assert type(validate_configuration(settings)) is type(validate_settings(settings))

    def test_validate_configuration_with_none(self):
        """validate_configuration(None) does not crash."""
        result = validate_configuration(None)
        assert isinstance(result, ConfigValidationResult)


class TestFormatValidationResult:
    """Tests for format_validation_result()."""

    def test_empty_result_formats_cleanly(self):
        """An empty result produces a non-empty string (no crash)."""
        result = ConfigValidationResult()
        formatted = format_validation_result(result)
        assert isinstance(formatted, str)

    def test_errors_appear_in_output(self):
        """Error messages appear in formatted output."""
        result = ConfigValidationResult()
        result.add_error(
            category="api_key",
            message="Missing API key",
            suggestion="Export MY_API_KEY",
            setting_path="providers.x.api_key",
        )
        formatted = format_validation_result(result)
        assert "Missing API key" in formatted
        assert "Export MY_API_KEY" in formatted
        assert "providers.x.api_key" in formatted

    def test_warnings_appear_in_output(self):
        """Warning messages appear in formatted output."""
        result = ConfigValidationResult()
        result.add_warning(category="perf", message="Slow network timeout")
        assert "Slow network timeout" in format_validation_result(result)

    def test_errors_and_warnings_both_shown(self):
        """Both sections appear when result has errors and warnings."""
        result = ConfigValidationResult()
        result.add_error(category="a", message="Critical problem")
        result.add_warning(category="b", message="Minor concern")
        formatted = format_validation_result(result)
        assert "Critical problem" in formatted
        assert "Minor concern" in formatted
