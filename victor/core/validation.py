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

"""Configuration validation framework with Pydantic schemas.

This module provides a comprehensive validation framework for Victor configurations,
implementing enterprise design patterns for reliable configuration management.

Design Patterns:
- Builder Pattern: ConfigurationBuilder for fluent config construction
- Validator Pattern: Composable validation rules
- Result Pattern: ConfigValidationResult for error handling
- Strategy Pattern: Pluggable validation strategies

Example:
    from victor.core.validation import (
        ConfigValidator,
        ConfigValidationResult,
        ProviderConfigSchema,
        ToolConfigSchema,
    )

    # Validate configuration
    validator = ConfigValidator()
    result = validator.validate(config_dict, ProviderConfigSchema)

    if result.is_valid:
        print("Configuration is valid")
    else:
        for error in result.errors:
            print(f"Error: {error.path} - {error.message}")

    # Use fluent builder
    config = (
        ConfigurationBuilder()
        .with_provider("anthropic")
        .with_model("claude-3-sonnet")
        .with_api_key("sk-...")
        .build()
    )
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Pattern,
    Set,
    Type,
    TypeVar,
    Union,
)

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

# =============================================================================
# Validation Result Pattern
# =============================================================================


class ValidationSeverity(str, Enum):
    """Severity levels for validation issues.

    Canonical severity enum for the whole tree (FEP-0014). The member set is the
    superset ``{INFO, WARNING, ERROR, CRITICAL}`` — ``CRITICAL`` is adopted from the
    ``framework/middleware.py`` variant so no layer loses a level. The pre-existing
    members and their string values are unchanged for backward compatibility; only
    ``CRITICAL`` is additive.

    Note:
        Because this is a ``str`` Enum, the native ``<`` / ``>=`` operators compare
        the underlying *string values* (alphabetical), which does NOT match logical
        severity order. Use :func:`severity_rank` for severity-aware comparisons.
    """

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    CRITICAL = "critical"


# Logical ordering of severities (ascending). Used instead of the native ``str``
# Enum comparison operators, which order by string value rather than by severity.
_SEVERITY_ORDER: Dict[ValidationSeverity, int] = {
    ValidationSeverity.INFO: 0,
    ValidationSeverity.WARNING: 1,
    ValidationSeverity.ERROR: 2,
    ValidationSeverity.CRITICAL: 3,
}


def severity_rank(severity: ValidationSeverity) -> int:
    """Return the logical rank of a severity (higher == more severe).

    Args:
        severity: The severity to rank.

    Returns:
        Integer rank where ``INFO=0 < WARNING=1 < ERROR=2 < CRITICAL=3``.
    """
    return _SEVERITY_ORDER[severity]


@dataclass
class ValidationIssue:
    """Single validation issue."""

    path: str
    message: str
    severity: ValidationSeverity = ValidationSeverity.ERROR
    code: Optional[str] = None
    value: Optional[Any] = None

    def __str__(self) -> str:
        """Format as string."""
        prefix = f"[{self.severity.value.upper()}]"
        return f"{prefix} {self.path}: {self.message}"


@dataclass
class ConfigValidationResult:
    """Result of configuration validation.

    Renamed from ValidationResult to be semantically distinct:
    - ToolValidationResult (victor.tools.base): Tool parameter validation
    - ConfigValidationResult (here): Configuration validation with ValidationIssue list
    - ContentValidationResult (victor.framework.middleware): Content validation with fixed_content
    - ParameterValidationResult (victor.agent.parameter_enforcer): Parameter enforcement
    - CodeValidationResult (victor.evaluation.correction.types): Code validation

    Implements the Result Pattern for handling validation outcomes.
    """

    issues: List[ValidationIssue] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Check if validation passed (no errors)."""
        return not any(i.severity == ValidationSeverity.ERROR for i in self.issues)

    @property
    def errors(self) -> List[ValidationIssue]:
        """Get all errors."""
        return [i for i in self.issues if i.severity == ValidationSeverity.ERROR]

    @property
    def warnings(self) -> List[ValidationIssue]:
        """Get all warnings."""
        return [i for i in self.issues if i.severity == ValidationSeverity.WARNING]

    def add_error(
        self,
        path: str,
        message: str,
        code: Optional[str] = None,
        value: Optional[Any] = None,
    ) -> "ConfigValidationResult":
        """Add an error."""
        self.issues.append(
            ValidationIssue(
                path=path,
                message=message,
                severity=ValidationSeverity.ERROR,
                code=code,
                value=value,
            )
        )
        return self

    def add_warning(
        self,
        path: str,
        message: str,
        code: Optional[str] = None,
    ) -> "ConfigValidationResult":
        """Add a warning."""
        self.issues.append(
            ValidationIssue(
                path=path,
                message=message,
                severity=ValidationSeverity.WARNING,
                code=code,
            )
        )
        return self

    def merge(self, other: "ConfigValidationResult") -> "ConfigValidationResult":
        """Merge another result into this one."""
        self.issues.extend(other.issues)
        return self

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "issues": [
                {
                    "path": i.path,
                    "message": i.message,
                    "severity": i.severity.value,
                    "code": i.code,
                }
                for i in self.issues
            ],
        }


# =============================================================================
# Canonical Validation Result (FEP-0014)
# =============================================================================


@dataclass(frozen=True)
class ValidationResult:
    """Canonical validation result (FEP-0014).

    The single "bool + list of problems" shape for the whole tree. It supersedes the
    five divergent ``ValidationResult`` variants (``is_valid``/``status``,
    ``errors``/``issues``/``error_message``) that previously lived in ``tools``,
    ``config``, ``workflows`` and ``framework``. Consumers are migrated onto this
    type in FEP-0014 Phase 2; Phase 1 only introduces it additively.

    This is distinct from :class:`ConfigValidationResult` (kept as-is): that type is a
    mutable builder with ``add_error``/``merge`` helpers used by the config validator,
    whereas this is a frozen value object for passing results across layer boundaries.

    Attributes:
        is_valid: Whether validation passed.
        issues: All validation issues (of any severity).
    """

    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> List[ValidationIssue]:
        """Issues at severity ``ERROR`` or higher (i.e. ``ERROR`` or ``CRITICAL``).

        Uses :func:`severity_rank` rather than the native ``str`` Enum ``>=`` so the
        comparison is by logical severity, not alphabetical string value.
        """
        threshold = severity_rank(ValidationSeverity.ERROR)
        return [i for i in self.issues if severity_rank(i.severity) >= threshold]

    @property
    def error_message(self) -> str:
        """Convenience: newline-joined messages of the error-level issues.

        Empty string when there are no error-level issues.
        """
        return "\n".join(i.message for i in self.errors)


# =============================================================================
# Validation Rules (Strategy Pattern)
# =============================================================================


class ValidationRule(ABC):
    """Abstract base for validation rules."""

    @abstractmethod
    def validate(self, value: Any, context: Dict[str, Any]) -> ConfigValidationResult:
        """Validate a value.

        Args:
            value: Value to validate.
            context: Additional context for validation.

        Returns:
            Validation result.
        """
        pass


class RegexRule(ValidationRule):
    """Validate against a regex pattern."""

    def __init__(
        self,
        pattern: Union[str, Pattern],
        message: str = "Value does not match expected pattern",
        code: str = "regex_mismatch",
    ) -> None:
        """Initialize regex rule.

        Args:
            pattern: Regex pattern.
            message: Error message.
            code: Error code.
        """
        self._pattern = re.compile(pattern) if isinstance(pattern, str) else pattern
        self._message = message
        self._code = code

    def validate(self, value: Any, context: Dict[str, Any]) -> ConfigValidationResult:
        """Validate value matches pattern."""
        result = ConfigValidationResult()

        if value is None:
            return result

        if not isinstance(value, str):
            result.add_error("", "Value must be a string for regex validation")
            return result

        if not self._pattern.match(value):
            result.add_error("", self._message, code=self._code, value=value)

        return result


class RangeRule(ValidationRule):
    """Validate numeric range."""

    def __init__(
        self,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        exclusive: bool = False,
    ) -> None:
        """Initialize range rule.

        Args:
            min_value: Minimum value (inclusive by default).
            max_value: Maximum value (inclusive by default).
            exclusive: Use exclusive bounds.
        """
        self._min = min_value
        self._max = max_value
        self._exclusive = exclusive

    def validate(self, value: Any, context: Dict[str, Any]) -> ConfigValidationResult:
        """Validate value is in range."""
        result = ConfigValidationResult()

        if value is None:
            return result

        if not isinstance(value, (int, float)):
            result.add_error("", "Value must be numeric")
            return result

        if self._min is not None:
            if self._exclusive and value <= self._min:
                result.add_error("", f"Value must be greater than {self._min}")
            elif not self._exclusive and value < self._min:
                result.add_error("", f"Value must be at least {self._min}")

        if self._max is not None:
            if self._exclusive and value >= self._max:
                result.add_error("", f"Value must be less than {self._max}")
            elif not self._exclusive and value > self._max:
                result.add_error("", f"Value must be at most {self._max}")

        return result


class EnumRule(ValidationRule):
    """Validate value is one of allowed values."""

    def __init__(
        self,
        allowed: Set[Any],
        case_insensitive: bool = False,
    ) -> None:
        """Initialize enum rule.

        Args:
            allowed: Set of allowed values.
            case_insensitive: Case insensitive string matching.
        """
        self._allowed = allowed
        self._case_insensitive = case_insensitive

    def validate(self, value: Any, context: Dict[str, Any]) -> ConfigValidationResult:
        """Validate value is allowed."""
        result = ConfigValidationResult()

        if value is None:
            return result

        check_value = value
        check_allowed = self._allowed

        if self._case_insensitive and isinstance(value, str):
            check_value = value.lower()
            check_allowed = {v.lower() if isinstance(v, str) else v for v in self._allowed}

        if check_value not in check_allowed:
            allowed_str = ", ".join(str(v) for v in self._allowed)
            result.add_error("", f"Value must be one of: {allowed_str}", value=value)

        return result


class PathRule(ValidationRule):
    """Validate path exists or is valid."""

    def __init__(
        self,
        must_exist: bool = False,
        must_be_file: bool = False,
        must_be_dir: bool = False,
        writable: bool = False,
    ) -> None:
        """Initialize path rule.

        Args:
            must_exist: Path must exist.
            must_be_file: Path must be a file.
            must_be_dir: Path must be a directory.
            writable: Path must be writable.
        """
        self._must_exist = must_exist
        self._must_be_file = must_be_file
        self._must_be_dir = must_be_dir
        self._writable = writable

    def validate(self, value: Any, context: Dict[str, Any]) -> ConfigValidationResult:
        """Validate path."""
        result = ConfigValidationResult()

        if value is None:
            return result

        path = Path(value) if isinstance(value, str) else value

        if self._must_exist and not path.exists():
            result.add_error("", f"Path does not exist: {path}")
            return result

        if path.exists():
            if self._must_be_file and not path.is_file():
                result.add_error("", f"Path is not a file: {path}")

            if self._must_be_dir and not path.is_dir():
                result.add_error("", f"Path is not a directory: {path}")

            if self._writable:
                import os

                if not os.access(path, os.W_OK):
                    result.add_error("", f"Path is not writable: {path}")

        return result


class DependencyRule(ValidationRule):
    """Validate field dependencies."""

    def __init__(
        self,
        required_if: Dict[str, Any],
        message: Optional[str] = None,
    ) -> None:
        """Initialize dependency rule.

        Args:
            required_if: Dict of field -> value that triggers requirement.
            message: Custom error message.
        """
        self._required_if = required_if
        self._message = message

    def validate(self, value: Any, context: Dict[str, Any]) -> ConfigValidationResult:
        """Validate dependencies."""
        result = ConfigValidationResult()

        # Check if conditions are met
        for field_name, expected in self._required_if.items():
            if context.get(field_name) == expected:
                if value is None:
                    msg = self._message or f"Field is required when {field_name}={expected}"
                    result.add_error("", msg)
                    break

        return result


# =============================================================================
# Pydantic Configuration Schemas
# =============================================================================


class ProviderConfigSchema(BaseModel):
    """Schema for provider configuration validation."""

    name: str = Field(..., min_length=1, description="Provider name")
    api_key: Optional[str] = Field(None, description="API key")
    base_url: Optional[str] = Field(None, description="Base URL for API")
    timeout: int = Field(300, ge=1, le=3600, description="Request timeout in seconds")
    max_retries: int = Field(3, ge=0, le=10, description="Maximum retry attempts")
    organization: Optional[str] = Field(None, description="Organization ID")

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v: Optional[str]) -> Optional[str]:
        """Validate API key format."""
        if v is None:
            return v

        # Basic security check - don't allow obvious placeholder keys
        placeholders = {"your-api-key", "xxx", "placeholder", "test", "demo"}
        if v.lower() in placeholders:
            raise ValueError("API key appears to be a placeholder")

        return v

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, v: Optional[str]) -> Optional[str]:
        """Validate base URL format."""
        if v is None:
            return v

        if not v.startswith(("http://", "https://")):
            raise ValueError("Base URL must start with http:// or https://")

        return v.rstrip("/")


class ModelConfigSchema(BaseModel):
    """Schema for model configuration."""

    model: str = Field(..., min_length=1, description="Model identifier")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int = Field(4096, ge=1, le=200000, description="Maximum tokens")
    top_p: float = Field(1.0, ge=0.0, le=1.0, description="Nucleus sampling")
    presence_penalty: float = Field(0.0, ge=-2.0, le=2.0, description="Presence penalty")
    frequency_penalty: float = Field(0.0, ge=-2.0, le=2.0, description="Frequency penalty")

    @field_validator("model")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        """Validate model name format."""
        # Allow common model naming patterns
        valid_pattern = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._:/-]*$")
        if not valid_pattern.match(v):
            raise ValueError("Model name contains invalid characters")
        return v


class ToolConfigSchema(BaseModel):
    """Schema for tool configuration."""

    enabled: bool = Field(True, description="Whether tool is enabled")
    timeout: int = Field(30, ge=1, le=300, description="Tool execution timeout")
    max_retries: int = Field(2, ge=0, le=5, description="Maximum retry attempts")
    cost_tier: str = Field(
        "FREE",
        description="Tool cost tier",
    )

    @field_validator("cost_tier")
    @classmethod
    def validate_cost_tier(cls, v: str) -> str:
        """Validate cost tier."""
        valid_tiers = {"FREE", "LOW", "MEDIUM", "HIGH"}
        if v.upper() not in valid_tiers:
            raise ValueError(f"Invalid cost tier. Must be one of: {valid_tiers}")
        return v.upper()


class CacheConfigSchema(BaseModel):
    """Schema for cache configuration."""

    enabled: bool = Field(True, description="Whether caching is enabled")
    ttl_seconds: int = Field(3600, ge=0, le=86400, description="Cache TTL in seconds")
    max_size_mb: int = Field(100, ge=1, le=10000, description="Maximum cache size in MB")
    strategy: str = Field("lru", description="Eviction strategy")

    @field_validator("strategy")
    @classmethod
    def validate_strategy(cls, v: str) -> str:
        """Validate eviction strategy."""
        valid = {"lru", "lfu", "ttl", "fifo"}
        if v.lower() not in valid:
            raise ValueError(f"Invalid strategy. Must be one of: {valid}")
        return v.lower()


class ResilienceConfigSchema(BaseModel):
    """Schema for resilience configuration."""

    circuit_breaker_enabled: bool = Field(True, description="Enable circuit breaker")
    failure_threshold: int = Field(5, ge=1, le=100, description="Failures before opening")
    recovery_timeout: float = Field(30.0, ge=1.0, le=300.0, description="Recovery timeout")
    retry_enabled: bool = Field(True, description="Enable retry")
    max_retries: int = Field(3, ge=0, le=10, description="Maximum retries")
    base_delay: float = Field(1.0, ge=0.1, le=60.0, description="Base delay between retries")
    rate_limit_enabled: bool = Field(False, description="Enable rate limiting")
    requests_per_second: float = Field(10.0, ge=0.1, le=1000.0, description="Rate limit")


class ObservabilityConfigSchema(BaseModel):
    """Schema for observability configuration."""

    metrics_enabled: bool = Field(True, description="Enable metrics collection")
    tracing_enabled: bool = Field(False, description="Enable distributed tracing")
    logging_level: str = Field("INFO", description="Logging level")
    export_format: str = Field("json", description="Export format for events")

    @field_validator("logging_level")
    @classmethod
    def validate_logging_level(cls, v: str) -> str:
        """Validate logging level."""
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid:
            raise ValueError(f"Invalid logging level. Must be one of: {valid}")
        return v.upper()

    @field_validator("export_format")
    @classmethod
    def validate_export_format(cls, v: str) -> str:
        """Validate export format."""
        valid = {"json", "jsonl", "csv", "otel"}
        if v.lower() not in valid:
            raise ValueError(f"Invalid export format. Must be one of: {valid}")
        return v.lower()


class AgentConfigSchema(BaseModel):
    """Schema for complete agent configuration."""

    provider: ProviderConfigSchema
    model: ModelConfigSchema
    tools: Optional[ToolConfigSchema] = None
    cache: Optional[CacheConfigSchema] = None
    resilience: Optional[ResilienceConfigSchema] = None
    observability: Optional[ObservabilityConfigSchema] = None

    @model_validator(mode="after")
    def validate_config(self) -> "AgentConfigSchema":
        """Cross-field validation."""
        # Example: Validate provider-model compatibility
        return self


# =============================================================================
# Config Validator (Facade)
# =============================================================================


T = TypeVar("T", bound=BaseModel)


class ConfigValidator:
    """Facade for configuration validation.

    Provides a unified interface for validating configurations against
    Pydantic schemas with custom validation rules.
    """

    def __init__(self) -> None:
        """Initialize validator."""
        self._custom_rules: Dict[str, List[ValidationRule]] = {}

    def add_rule(
        self,
        field_path: str,
        rule: ValidationRule,
    ) -> "ConfigValidator":
        """Add custom validation rule.

        Args:
            field_path: Dot-separated path to field.
            rule: Validation rule.

        Returns:
            Self for chaining.
        """
        if field_path not in self._custom_rules:
            self._custom_rules[field_path] = []
        self._custom_rules[field_path].append(rule)
        return self

    def validate(
        self,
        data: Dict[str, Any],
        schema: Type[T],
    ) -> ConfigValidationResult:
        """Validate data against schema.

        Args:
            data: Configuration data.
            schema: Pydantic model class.

        Returns:
            Validation result.
        """
        result = ConfigValidationResult()

        # First try Pydantic validation
        try:
            schema.model_validate(data)
        except ValidationError as e:
            for error in e.errors():
                path = ".".join(str(loc) for loc in error["loc"])
                result.add_error(
                    path=path,
                    message=error["msg"],
                    code=error["type"],
                    value=error.get("input"),
                )

        # Apply custom rules
        for field_path, rules in self._custom_rules.items():
            value = self._get_nested(data, field_path)
            for rule in rules:
                rule_result = rule.validate(value, data)
                # Prefix errors with field path
                for issue in rule_result.issues:
                    issue.path = f"{field_path}.{issue.path}".rstrip(".")
                result.merge(rule_result)

        return result

    def _get_nested(self, data: Dict[str, Any], path: str) -> Any:
        """Get nested value by dot-separated path."""
        parts = path.split(".")
        current = data

        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
            else:
                return None

        return current


# =============================================================================
# Configuration Builder (Builder Pattern)
# =============================================================================


class ConfigurationBuilder:
    """Fluent builder for configuration construction.

    Example:
        config = (
            ConfigurationBuilder()
            .with_provider("anthropic")
            .with_model("claude-3-sonnet")
            .with_api_key("sk-...")
            .with_temperature(0.7)
            .build()
        )
    """

    def __init__(self) -> None:
        """Initialize builder."""
        self._config: Dict[str, Any] = {
            "provider": {},
            "model": {},
        }

    def with_provider(self, name: str) -> "ConfigurationBuilder":
        """Set provider name."""
        self._config["provider"]["name"] = name
        return self

    def with_api_key(self, key: str) -> "ConfigurationBuilder":
        """Set API key."""
        self._config["provider"]["api_key"] = key
        return self

    def with_base_url(self, url: str) -> "ConfigurationBuilder":
        """Set base URL."""
        self._config["provider"]["base_url"] = url
        return self

    def with_timeout(self, seconds: int) -> "ConfigurationBuilder":
        """Set request timeout."""
        self._config["provider"]["timeout"] = seconds
        return self

    def with_model(self, model: str) -> "ConfigurationBuilder":
        """Set model name."""
        self._config["model"]["model"] = model
        return self

    def with_temperature(self, temp: float) -> "ConfigurationBuilder":
        """Set temperature."""
        self._config["model"]["temperature"] = temp
        return self

    def with_max_tokens(self, tokens: int) -> "ConfigurationBuilder":
        """Set max tokens."""
        self._config["model"]["max_tokens"] = tokens
        return self

    def with_tools(self, config: Dict[str, Any]) -> "ConfigurationBuilder":
        """Set tool configuration."""
        self._config["tools"] = config
        return self

    def with_cache(self, config: Dict[str, Any]) -> "ConfigurationBuilder":
        """Set cache configuration."""
        self._config["cache"] = config
        return self

    def with_resilience(self, config: Dict[str, Any]) -> "ConfigurationBuilder":
        """Set resilience configuration."""
        self._config["resilience"] = config
        return self

    def with_observability(self, config: Dict[str, Any]) -> "ConfigurationBuilder":
        """Set observability configuration."""
        self._config["observability"] = config
        return self

    def build(self) -> Dict[str, Any]:
        """Build the configuration dictionary."""
        return self._config.copy()

    def build_validated(self) -> AgentConfigSchema:
        """Build and validate the configuration.

        Returns:
            Validated AgentConfigSchema.

        Raises:
            ValidationError: If validation fails.
        """
        return AgentConfigSchema.model_validate(self._config)


# =============================================================================
# Convenience Functions
# =============================================================================


def validate_provider_config(config: Dict[str, Any]) -> ConfigValidationResult:
    """Validate provider configuration.

    Args:
        config: Provider configuration dict.

    Returns:
        Validation result.
    """
    validator = ConfigValidator()
    return validator.validate(config, ProviderConfigSchema)


def validate_model_config(config: Dict[str, Any]) -> ConfigValidationResult:
    """Validate model configuration.

    Args:
        config: Model configuration dict.

    Returns:
        Validation result.
    """
    validator = ConfigValidator()
    return validator.validate(config, ModelConfigSchema)


def validate_agent_config(config: Dict[str, Any]) -> ConfigValidationResult:
    """Validate complete agent configuration.

    Args:
        config: Agent configuration dict.

    Returns:
        Validation result.
    """
    validator = ConfigValidator()
    return validator.validate(config, AgentConfigSchema)


__all__ = [
    # Severity and Issues
    "ValidationSeverity",
    "severity_rank",
    "ValidationIssue",
    "ValidationResult",
    "ConfigValidationResult",
    # Rules
    "ValidationRule",
    "RegexRule",
    "RangeRule",
    "EnumRule",
    "PathRule",
    "DependencyRule",
    # Config Schemas
    "ProviderConfigSchema",
    "ModelConfigSchema",
    "ToolConfigSchema",
    "CacheConfigSchema",
    "ResilienceConfigSchema",
    "ObservabilityConfigSchema",
    "AgentConfigSchema",
    # Validator and Builder
    "ConfigValidator",
    "ConfigurationBuilder",
    # Convenience Functions
    "validate_provider_config",
    "validate_model_config",
    "validate_agent_config",
    # Re-export ValidationError from pydantic
    "ValidationError",
]
