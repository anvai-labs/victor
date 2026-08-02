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

"""Configuration validation framework.

Provides early validation of settings before orchestrator initialization,
separating warnings from errors and providing actionable suggestions.
"""

from dataclasses import dataclass, field
from typing import List, Optional

# FEP-0014 Phase 2a: re-export the canonical severity instead of redefining it.
from victor.core.validation import ValidationSeverity


@dataclass
class ValidationIssue:
    """A single validation issue found during configuration validation."""

    category: str  # e.g., "provider", "profile", "api_key"
    severity: ValidationSeverity
    message: str
    suggestion: Optional[str] = None
    setting_path: Optional[str] = None  # e.g., "providers.anthropic.api_key"

    def __str__(self) -> str:
        """Format validation issue for display."""
        severity_icon = {
            ValidationSeverity.CRITICAL: "🛑",
            ValidationSeverity.ERROR: "✗",
            ValidationSeverity.WARNING: "⚠",
            ValidationSeverity.INFO: "ℹ",
        }[self.severity]

        result = f"{severity_icon} [{self.category}] {self.message}"
        if self.suggestion:
            result += f"\n    💡 {self.suggestion}"
        if self.setting_path:
            result += f"\n    📍 Setting: {self.setting_path}"
        return result


@dataclass
class ConfigValidationResult:
    """Result of configuration validation.

    Contains lists of errors, warnings, and info messages found during validation.
    Provides methods to check validity and get formatted output.
    """

    errors: List[ValidationIssue] = field(default_factory=list)
    warnings: List[ValidationIssue] = field(default_factory=list)
    info: List[ValidationIssue] = field(default_factory=list)

    def add_error(
        self,
        category: str,
        message: str,
        suggestion: Optional[str] = None,
        setting_path: Optional[str] = None,
    ) -> None:
        """Add an error to the validation result."""
        self.errors.append(
            ValidationIssue(
                category=category,
                severity=ValidationSeverity.ERROR,
                message=message,
                suggestion=suggestion,
                setting_path=setting_path,
            )
        )

    def add_warning(
        self,
        category: str,
        message: str,
        suggestion: Optional[str] = None,
        setting_path: Optional[str] = None,
    ) -> None:
        """Add a warning to the validation result."""
        self.warnings.append(
            ValidationIssue(
                category=category,
                severity=ValidationSeverity.WARNING,
                message=message,
                suggestion=suggestion,
                setting_path=setting_path,
            )
        )

    def add_info(
        self,
        category: str,
        message: str,
        suggestion: Optional[str] = None,
        setting_path: Optional[str] = None,
    ) -> None:
        """Add an info message to the validation result."""
        self.info.append(
            ValidationIssue(
                category=category,
                severity=ValidationSeverity.INFO,
                message=message,
                suggestion=suggestion,
                setting_path=setting_path,
            )
        )

    def is_valid(self) -> bool:
        """Check if configuration is valid (no errors).

        Returns:
            True if there are no errors, False otherwise
        """
        return len(self.errors) == 0

    def has_warnings(self) -> bool:
        """Check if configuration has warnings.

        Returns:
            True if there are warnings, False otherwise
        """
        return len(self.warnings) > 0

    def get_total_issues(self) -> int:
        """Get total number of issues (errors + warnings + info).

        Returns:
            Total count of all issues
        """
        return len(self.errors) + len(self.warnings) + len(self.info)

    def get_summary(self) -> str:
        """Get a summary of validation results.

        Returns:
            Formatted summary string
        """
        parts = []
        if self.errors:
            parts.append(f"{len(self.errors)} error(s)")
        if self.warnings:
            parts.append(f"{len(self.warnings)} warning(s)")
        if self.info:
            parts.append(f"{len(self.info)} info")

        if not parts:
            return "✓ Configuration is valid"

        status = "invalid" if self.errors else "valid with warnings"
        return f"Configuration {status}: {', '.join(parts)}"

    def format_for_display(self) -> str:
        """Format all validation issues for display.

        Returns:
            Formatted string with all issues grouped by severity
        """
        lines = []

        if self.errors:
            lines.append("Errors:")
            for error in self.errors:
                lines.append(f"  {error}")

        if self.warnings:
            if self.errors:
                lines.append("")
            lines.append("Warnings:")
            for warning in self.warnings:
                lines.append(f"  {warning}")

        if self.info:
            if self.errors or self.warnings:
                lines.append("")
            lines.append("Info:")
            for info_msg in self.info:
                lines.append(f"  {info_msg}")

        return "\n".join(lines)


class ConfigValidator:
    """Validates configuration settings before initialization.

    Provides validation methods for different aspects of configuration:
    - Provider configuration
    - API keys
    - Profile settings
    - File paths
    - Feature flags
    """

    def __init__(self):
        """Initialize the validator."""
        self._validators = {
            "providers": self._validate_providers,
            "api_keys": self._validate_api_keys,
            "profiles": self._validate_profiles,
            "paths": self._validate_paths,
        }

    def validate(self, settings) -> ConfigValidationResult:
        """Validate all aspects of configuration.

        Args:
            settings: Settings object to validate

        Returns:
            ConfigValidationResult with all issues found
        """
        result = ConfigValidationResult()

        # Run all validators
        for name, validator in self._validators.items():
            try:
                validator(settings, result)
            except Exception as e:
                # Don't let validator errors break validation
                result.add_warning(
                    category="validator",
                    message=f"Validator '{name}' failed: {e}",
                    suggestion="This may indicate a bug in the validator",
                )

        return result

    def _validate_providers(self, settings, result: ConfigValidationResult) -> None:
        """Validate provider configuration.

        Args:
            settings: Settings object
            result: Validation result to add issues to
        """
        # Check if default_provider is known
        known_providers = [
            "ollama",
            "anthropic",
            "openai",
            "google",
            "cohere",
            "groq",
            "deepseek",
            "openrouter",
            "zhipu",
            "mistral",
            "azure",
            "aws",
            "lambdalabs",
            "replicate",
            "together",
            "xai",
            "wangrui",
        ]

        if settings.provider and settings.provider.default_provider:
            provider = settings.provider.default_provider
            if provider not in known_providers:
                result.add_warning(
                    category="provider",
                    message=f"Unknown provider '{provider}'",
                    suggestion=f"Known providers: {', '.join(known_providers[:5])}...",
                    setting_path="provider.default_provider",
                )

    def _validate_api_keys(self, settings, result: ConfigValidationResult) -> None:
        """Validate API key configuration.

        Args:
            settings: Settings object
            result: Validation result to add issues to
        """
        # Check if provider has API key when required
        if settings.provider and settings.provider.default_provider:
            provider = settings.provider.default_provider

            # Providers that require API keys
            requires_key = ["anthropic", "openai", "google", "cohere", "groq"]

            if provider in requires_key:
                # Check for API key in flat fields
                key_field = f"{provider}_api_key"
                api_key = getattr(settings, key_field, None)
                if api_key is None or (
                    hasattr(api_key, "get_secret_value") and not api_key.get_secret_value()
                ):
                    result.add_error(
                        category="api_key",
                        message=f"API key not set for provider '{provider}'",
                        suggestion=f"Set {provider.upper()}_API_KEY environment variable",
                        setting_path=key_field,
                    )

    def _validate_profiles(self, settings, result: ConfigValidationResult) -> None:
        """Validate profile configuration.

        Args:
            settings: Settings object
            result: Validation result to add issues to
        """
        # Check if profile exists
        if hasattr(settings, "profile"):
            profile = settings.profile
            # Profile validation would go here
            # For now, just add info about which profile is active
            result.add_info(
                category="profile",
                message=f"Using profile: {profile}",
            )

    def _validate_paths(self, settings, result: ConfigValidationResult) -> None:
        """Validate file system paths.

        Args:
            settings: Settings object
            result: Validation result to add issues to
        """
        import os

        # Check common paths
        paths_to_check = []

        if hasattr(settings, "codebase_persist_directory"):
            path = settings.codebase_persist_directory
            if path:
                paths_to_check.append(("codebase_persist_directory", path))

        for setting_name, path in paths_to_check:
            if path and not os.path.exists(os.path.dirname(path)):
                result.add_warning(
                    category="path",
                    message=f"Parent directory does not exist for {setting_name}",
                    suggestion=f"Create directory: mkdir -p {os.path.dirname(path)}",
                    setting_path=setting_name,
                )


def validate_settings(settings) -> ConfigValidationResult:
    """Validate settings and return result.

    This is a convenience function that creates a ConfigValidator
    and runs validation on the provided settings.

    Args:
        settings: Settings object to validate

    Returns:
        ConfigValidationResult with all issues found
    """
    validator = ConfigValidator()
    return validator.validate(settings)


# Alias for backward compatibility with existing code
validate_configuration = validate_settings


def format_validation_result(result: ConfigValidationResult) -> str:
    """Format validation result for display.

    This is a convenience function that formats the validation result
    in a user-friendly way for CLI output.

    Args:
        result: ConfigValidationResult to format

    Returns:
        Formatted string for display
    """
    return result.format_for_display()
