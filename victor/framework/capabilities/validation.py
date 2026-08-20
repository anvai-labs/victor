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

"""Validation capability for framework-level validation rules.

This module provides a pluggable validation system with common validators
that can be used across verticals and tool pipelines.

Design Pattern: Capability Provider + Strategy Pattern
- Pluggable validation system
- Common validators for files, syntax, configuration
- Extensible validator registry
- Chain of responsibility for validation

Integration Points:
    - Tool pipeline: Validate tool arguments before execution
    - Orchestrator: Validate configuration on startup
    - Verticals: Register domain-specific validators

Example:
    capability = ValidationCapabilityProvider()
    capability.register_validator("file_path", FilePathValidator())

    result = capability.validate("file_path", "/path/to/file.txt")
    if not result.is_valid:
        print(result.error_message)

Phase 1: Promote Generic Capabilities to Framework
"""

from __future__ import annotations

from victor.core.json_utils import json_loads
from json import JSONDecodeError
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union

# FEP-0014 Phase 2a: re-export the canonical severity instead of redefining it.
from victor.core.validation import ValidationSeverity


@dataclass
class CapabilityValidationResult:
    """Result of a validation operation.

    Attributes:
        is_valid: Whether validation passed
        error_message: Error message if validation failed
        severity: Severity of validation failure
        details: Additional details about the validation result
        validator_id: ID of the validator that produced this result
    """

    is_valid: bool
    error_message: str = ""
    severity: ValidationSeverity = ValidationSeverity.ERROR
    details: Dict[str, Any] = field(default_factory=dict)
    validator_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "error_message": self.error_message,
            "severity": self.severity.value,
            "details": self.details,
            "validator_id": self.validator_id,
        }

    @classmethod
    def success(cls, details: Optional[Dict[str, Any]] = None) -> "CapabilityValidationResult":
        """Create a successful validation result."""
        return cls(is_valid=True, details=details or {})

    @classmethod
    def failure(
        cls,
        message: str,
        severity: ValidationSeverity = ValidationSeverity.ERROR,
        details: Optional[Dict[str, Any]] = None,
        validator_id: str = "",
    ) -> "CapabilityValidationResult":
        """Create a failed validation result."""
        return cls(
            is_valid=False,
            error_message=message,
            severity=severity,
            details=details or {},
            validator_id=validator_id,
        )


class Validator(ABC):
    """Abstract base class for validators.

    Validators implement specific validation logic for different
    types of data and operations.
    """

    @property
    @abstractmethod
    def validator_id(self) -> str:
        """Unique identifier for this validator."""
        pass

    @abstractmethod
    def validate(
        self, value: Any, context: Optional[Dict[str, Any]] = None
    ) -> CapabilityValidationResult:
        """Validate a value.

        Args:
            value: The value to validate
            context: Optional context for validation

        Returns:
            CapabilityValidationResult with validation outcome
        """
        pass


class FilePathValidator(Validator):
    """Validator for file paths.

    Validates that file paths are:
    - Well-formed (valid path syntax)
    - Within allowed directories (if specified)
    - Not attempting path traversal attacks
    """

    def __init__(
        self,
        allowed_directories: Optional[List[str]] = None,
        allow_traversal: bool = False,
        require_exists: bool = False,
    ):
        """Initialize the file path validator.

        Args:
            allowed_directories: List of allowed root directories
            allow_traversal: Whether to allow path traversal (../)
            require_exists: Whether path must exist
        """
        self._allowed_dirs = [Path(d).resolve() for d in (allowed_directories or [])]
        self._allow_traversal = allow_traversal
        self._require_exists = require_exists

    @property
    def validator_id(self) -> str:
        return "file_path"

    def validate(
        self, value: Any, context: Optional[Dict[str, Any]] = None
    ) -> CapabilityValidationResult:
        """Validate a file path.

        Args:
            value: File path to validate (string or Path)
            context: Optional validation context

        Returns:
            CapabilityValidationResult
        """
        if not isinstance(value, (str, Path)):
            return CapabilityValidationResult.failure(
                f"File path must be a string or Path object, got {type(value).__name__}",
                validator_id=self.validator_id,
            )

        try:
            path = Path(value).resolve()
        except Exception as e:
            return CapabilityValidationResult.failure(
                f"Invalid file path: {e}",
                validator_id=self.validator_id,
            )

        # Check for path traversal
        if not self._allow_traversal:
            if ".." in str(value):
                return CapabilityValidationResult.failure(
                    "Path traversal (../) is not allowed",
                    severity=ValidationSeverity.ERROR,
                    validator_id=self.validator_id,
                )

        # Check allowed directories
        if self._allowed_dirs:
            is_allowed = any(
                str(path).startswith(str(allowed_dir)) for allowed_dir in self._allowed_dirs
            )
            if not is_allowed:
                return CapabilityValidationResult.failure(
                    f"Path is not within allowed directories: {self._allowed_dirs}",
                    severity=ValidationSeverity.ERROR,
                    validator_id=self.validator_id,
                )

        # Check if path exists (if required)
        if self._require_exists and not path.exists():
            return CapabilityValidationResult.failure(
                f"Path does not exist: {path}",
                severity=ValidationSeverity.ERROR,
                validator_id=self.validator_id,
            )

        return CapabilityValidationResult.success(
            details={"resolved_path": str(path), "exists": path.exists()}
        )


class CodeSyntaxValidator(Validator):
    """Validator for code syntax.

    Validates that code strings have valid syntax for their language.
    Supports Python, JavaScript, TypeScript, JSON, YAML, etc.
    """

    def __init__(self, language: str = "python"):
        """Initialize the code syntax validator.

        Args:
            language: Programming language (python, javascript, typescript, json, yaml)
        """
        self._language = language.lower()

    @property
    def validator_id(self) -> str:
        return f"code_syntax_{self._language}"

    def validate(
        self, value: Any, context: Optional[Dict[str, Any]] = None
    ) -> CapabilityValidationResult:
        """Validate code syntax.

        Args:
            value: Code string to validate
            context: Optional validation context

        Returns:
            CapabilityValidationResult
        """
        if not isinstance(value, str):
            return CapabilityValidationResult.failure(
                f"Code must be a string, got {type(value).__name__}",
                validator_id=self.validator_id,
            )

        try:
            if self._language == "python":
                return self._validate_python(value)
            elif self._language in ("javascript", "js"):
                return self._validate_javascript(value)
            elif self._language in ("typescript", "ts"):
                return self._validate_typescript(value)
            elif self._language == "json":
                return self._validate_json(value)
            elif self._language == "yaml":
                return self._validate_yaml(value)
            else:
                return CapabilityValidationResult.failure(
                    f"Unsupported language: {self._language}",
                    severity=ValidationSeverity.WARNING,
                    validator_id=self.validator_id,
                )
        except Exception as e:
            return CapabilityValidationResult.failure(
                f"Syntax validation error: {e}",
                validator_id=self.validator_id,
            )

    def _validate_python(self, code: str) -> CapabilityValidationResult:
        """Validate Python syntax."""
        try:
            import ast

            ast.parse(code)
            return CapabilityValidationResult.success()
        except SyntaxError as e:
            return CapabilityValidationResult.failure(
                f"Python syntax error at line {e.lineno}: {e.msg}",
                details={"line": e.lineno, "offset": e.offset},
                validator_id=self.validator_id,
            )

    def _validate_javascript(self, code: str) -> CapabilityValidationResult:
        """Validate JavaScript syntax."""
        # Basic validation - look for obvious syntax issues
        # Full validation would require a JS parser
        open_braces = code.count("{")
        close_braces = code.count("}")
        open_parens = code.count("(")
        close_parens = code.count(")")

        if open_braces != close_braces:
            return CapabilityValidationResult.failure(
                f"Unbalanced braces: {open_braces} open, {close_braces} close",
                severity=ValidationSeverity.WARNING,
                validator_id=self.validator_id,
            )

        if open_parens != close_parens:
            return CapabilityValidationResult.failure(
                f"Unbalanced parentheses: {open_parens} open, {close_parens} close",
                severity=ValidationSeverity.WARNING,
                validator_id=self.validator_id,
            )

        return CapabilityValidationResult.success()

    def _validate_typescript(self, code: str) -> CapabilityValidationResult:
        """Validate TypeScript syntax (basic checks)."""
        # TypeScript syntax validation would require a TS parser
        # For now, do basic brace/parenthesis balancing
        return self._validate_javascript(code)

    def _validate_json(self, code: str) -> CapabilityValidationResult:
        """Validate JSON syntax."""
        try:
            import json

            json_loads(code)
            return CapabilityValidationResult.success()
        except JSONDecodeError as e:
            return CapabilityValidationResult.failure(
                f"JSON syntax error at line {e.lineno}, column {e.colno}: {e.msg}",
                details={"line": e.lineno, "column": e.colno},
                validator_id=self.validator_id,
            )

    def _validate_yaml(self, code: str) -> CapabilityValidationResult:
        """Validate YAML syntax."""
        try:
            import yaml

            yaml.safe_load(code)
            return CapabilityValidationResult.success()
        except yaml.YAMLError as e:
            return CapabilityValidationResult.failure(
                f"YAML syntax error: {e}",
                validator_id=self.validator_id,
            )


class ConfigurationValidator(Validator):
    """Validator for configuration dictionaries.

    Validates that configuration dictionaries:
    - Have required keys
    - Have valid value types
    - Pass custom validation rules
    """

    def __init__(
        self,
        required_keys: Optional[List[str]] = None,
        key_types: Optional[Dict[str, Type]] = None,
        custom_validators: Optional[Dict[str, Callable[[Any], bool]]] = None,
    ):
        """Initialize the configuration validator.

        Args:
            required_keys: List of required key names
            key_types: Expected types for specific keys
            custom_validators: Custom validation functions for keys
        """
        self._required_keys = set(required_keys or [])
        self._key_types = key_types or {}
        self._custom_validators = custom_validators or {}

    @property
    def validator_id(self) -> str:
        return "configuration"

    def validate(
        self, value: Any, context: Optional[Dict[str, Any]] = None
    ) -> CapabilityValidationResult:
        """Validate a configuration dictionary.

        Args:
            value: Configuration dictionary to validate
            context: Optional validation context

        Returns:
            CapabilityValidationResult
        """
        if not isinstance(value, dict):
            return CapabilityValidationResult.failure(
                f"Configuration must be a dictionary, got {type(value).__name__}",
                validator_id=self.validator_id,
            )

        # Check required keys
        missing_keys = self._required_keys - set(value.keys())
        if missing_keys:
            return CapabilityValidationResult.failure(
                f"Missing required configuration keys: {missing_keys}",
                validator_id=self.validator_id,
            )

        # Check key types
        for key, expected_type in self._key_types.items():
            if key in value and not isinstance(value[key], expected_type):
                return CapabilityValidationResult.failure(
                    f"Configuration key '{key}' must be of type {expected_type.__name__}, "
                    f"got {type(value[key]).__name__}",
                    validator_id=self.validator_id,
                )

        # Run custom validators
        for key, validator_func in self._custom_validators.items():
            if key in value:
                try:
                    if not validator_func(value[key]):
                        return CapabilityValidationResult.failure(
                            f"Configuration key '{key}' failed custom validation",
                            validator_id=self.validator_id,
                        )
                except Exception as e:
                    return CapabilityValidationResult.failure(
                        f"Error validating configuration key '{key}': {e}",
                        validator_id=self.validator_id,
                    )

        return CapabilityValidationResult.success()


class OutputFormatValidator(Validator):
    """Validator for output format requirements.

    Validates that output meets format requirements:
    - Maximum length
    - Required patterns present
    - Forbidden patterns absent
    """

    def __init__(
        self,
        max_length: Optional[int] = None,
        required_patterns: Optional[List[str]] = None,
        forbidden_patterns: Optional[List[str]] = None,
    ):
        """Initialize the output format validator.

        Args:
            max_length: Maximum allowed length
            required_patterns: Regex patterns that must be present
            forbidden_patterns: Regex patterns that must NOT be present
        """
        self._max_length = max_length
        self._required_patterns = [re.compile(p) for p in (required_patterns or [])]
        self._forbidden_patterns = [re.compile(p) for p in (forbidden_patterns or [])]

    @property
    def validator_id(self) -> str:
        return "output_format"

    def validate(
        self, value: Any, context: Optional[Dict[str, Any]] = None
    ) -> CapabilityValidationResult:
        """Validate output format.

        Args:
            value: Output string to validate
            context: Optional validation context

        Returns:
            CapabilityValidationResult
        """
        if not isinstance(value, str):
            return CapabilityValidationResult.failure(
                f"Output must be a string, got {type(value).__name__}",
                validator_id=self.validator_id,
            )

        # Check max length
        if self._max_length and len(value) > self._max_length:
            return CapabilityValidationResult.failure(
                f"Output exceeds maximum length of {self._max_length} characters "
                f"(got {len(value)} characters)",
                validator_id=self.validator_id,
            )

        # Check required patterns
        missing_patterns = []
        for pattern in self._required_patterns:
            if not pattern.search(value):
                missing_patterns.append(pattern.pattern)

        if missing_patterns:
            return CapabilityValidationResult.failure(
                f"Output missing required patterns: {missing_patterns}",
                validator_id=self.validator_id,
            )

        # Check forbidden patterns
        found_forbidden = []
        for pattern in self._forbidden_patterns:
            if pattern.search(value):
                found_forbidden.append(pattern.pattern)

        if found_forbidden:
            return CapabilityValidationResult.failure(
                f"Output contains forbidden patterns: {found_forbidden}",
                validator_id=self.validator_id,
            )

        return CapabilityValidationResult.success()


class ValidationCapabilityProvider:
    """Generic validation capability provider.

    Provides a pluggable validation system with common validators:
    - File path validation
    - Code syntax validation
    - Configuration validation
    - Output format validation

    Validators can be registered and chained for comprehensive validation.

    Attributes:
        strict_mode: Whether to stop validation on first error
    """

    def __init__(self, strict_mode: bool = False):
        """Initialize the validation capability provider.

        Args:
            strict_mode: Whether to stop on first validation error
        """
        self._strict_mode = strict_mode
        self._validators: Dict[str, Validator] = {}
        self._register_default_validators()

    def _register_default_validators(self) -> None:
        """Register default validators."""
        # File path validators
        self.register_validator(FilePathValidator())
        self.register_validator(
            FilePathValidator(require_exists=True),
            validator_id="file_path_exists",
        )

        # Code syntax validators
        for lang in ["python", "javascript", "typescript", "json", "yaml"]:
            self.register_validator(CodeSyntaxValidator(language=lang))

        # Configuration validator
        self.register_validator(ConfigurationValidator())

        # Output format validator
        self.register_validator(OutputFormatValidator())

    def register_validator(self, validator: Validator, validator_id: Optional[str] = None) -> None:
        """Register a validator.

        Args:
            validator: Validator instance to register
            validator_id: Optional custom ID (uses validator.validator_id if None)
        """
        vid = validator_id or validator.validator_id
        self._validators[vid] = validator

    def unregister_validator(self, validator_id: str) -> bool:
        """Unregister a validator.

        Args:
            validator_id: ID of validator to unregister

        Returns:
            True if validator was removed, False if not found
        """
        if validator_id in self._validators:
            del self._validators[validator_id]
            return True
        return False

    def get_validator(self, validator_id: str) -> Optional[Validator]:
        """Get a registered validator.

        Args:
            validator_id: ID of the validator

        Returns:
            Validator instance or None if not found
        """
        return self._validators.get(validator_id)

    def list_validators(self) -> List[str]:
        """List all registered validator IDs.

        Returns:
            List of validator IDs
        """
        return list(self._validators.keys())

    def validate(
        self,
        validator_id: str,
        value: Any,
        context: Optional[Dict[str, Any]] = None,
    ) -> CapabilityValidationResult:
        """Validate a value using a specific validator.

        Args:
            validator_id: ID of the validator to use
            value: Value to validate
            context: Optional validation context

        Returns:
            CapabilityValidationResult

        Raises:
            KeyError: If validator_id is not registered
        """
        validator = self.get_validator(validator_id)
        if validator is None:
            raise KeyError(f"Validator not found: {validator_id}")

        return validator.validate(value, context=context)

    def validate_all(
        self,
        validations: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[CapabilityValidationResult]:
        """Validate multiple values.

        Args:
            validations: Dictionary mapping validator_id to value
            context: Optional validation context

        Returns:
            List of ValidationResults
        """
        results = []

        for validator_id, value in validations.items():
            try:
                result = self.validate(validator_id, value, context=context)
                results.append(result)

                # Stop on error if strict mode is enabled
                if self._strict_mode and not result.is_valid:
                    if result.severity == ValidationSeverity.ERROR:
                        break
            except KeyError as e:
                results.append(
                    CapabilityValidationResult.failure(
                        f"Validator not found: {e}",
                        validator_id=validator_id,
                    )
                )

        return results

    def validate_chain(
        self,
        validator_ids: List[str],
        value: Any,
        context: Optional[Dict[str, Any]] = None,
    ) -> CapabilityValidationResult:
        """Validate a value through a chain of validators.

        Args:
            validator_ids: List of validator IDs to apply in order
            value: Value to validate
            context: Optional validation context

        Returns:
            CapabilityValidationResult from first failed validator, or success if all pass
        """
        for validator_id in validator_ids:
            result = self.validate(validator_id, value, context=context)
            if not result.is_valid:
                return result

        return CapabilityValidationResult.success()


__all__ = [
    "ValidationCapabilityProvider",
    "Validator",
    "FilePathValidator",
    "CodeSyntaxValidator",
    "ConfigurationValidator",
    "OutputFormatValidator",
    "CapabilityValidationResult",
    "ValidationSeverity",
]
