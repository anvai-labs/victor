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

"""Stage contract protocols for LSP compliance.

This module defines the StageContract protocol and related validation
to ensure Liskov Substitution Principle (LSP) compliance for vertical
stage definitions.

Design Pattern: Protocol + Strategy
- StageContract: Protocol defining valid stage contract
- StageValidator: Validates stage definitions against contract
- Enforces required stages, valid transitions, naming conventions

Integration Point:
    Update VerticalBase.get_stages() to validate against StageContract

Phase 2: Fix SOLID Violations via Coordinator Extraction
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Protocol, runtime_checkable
from enum import Enum


class ValidationError(Enum):
    """Types of validation errors for stage definitions.

    NOTE: This enum is duplicated in victor_contracts.verticals.protocols.promoted_types
    to maintain SDK independence and avoid circular imports. When updating,
    please update BOTH definitions to maintain consistency.
    """

    MISSING_REQUIRED_STAGE = "missing_required_stage"
    INVALID_TRANSITION = "invalid_transition"
    INVALID_STAGE_NAME = "invalid_stage_name"
    MISSING_NEXT_STAGES = "missing_next_stages"
    CIRCULAR_TRANSITION = "circular_transition"
    INVALID_KEYWORDS = "invalid_keywords"
    INVALID_DESCRIPTION = "invalid_description"


@dataclass
class StageValidationResult:
    """Result of stage contract validation.

    Attributes:
        is_valid: Whether the stage definition passes validation
        errors: List of validation errors
        warnings: List of validation warnings
        details: Additional validation details

    NOTE: This class is duplicated in victor_contracts.verticals.protocols.promoted_types
    to maintain SDK independence and avoid circular imports. When updating,
    please update BOTH definitions to maintain consistency.
    """

    is_valid: bool
    errors: List[tuple[ValidationError, str]]  # (error_type, message)
    warnings: List[str]
    details: Dict[str, Any]

    def add_error(self, error_type: ValidationError, message: str) -> None:
        """Add a validation error.

        Args:
            error_type: Type of validation error
            message: Error message
        """
        self.errors.append((error_type, message))
        self.is_valid = False

    def add_warning(self, message: str) -> None:
        """Add a validation warning.

        Args:
            message: Warning message
        """
        self.warnings.append(message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "errors": [(e.value, m) for e, m in self.errors],
            "warnings": self.warnings,
            "details": self.details,
        }


@runtime_checkable
class StageContract(Protocol):
    """Protocol defining the contract for stage definitions.

    This protocol ensures LSP compliance by defining what constitutes
    a valid stage definition. Verticals that override get_stages()
    must return definitions that satisfy this contract.

    Required Stages:
        - INITIAL: Starting stage for all workflows
        - COMPLETION: Final stage for all workflows

    Valid Transitions:
        - Must be acyclic (no circular references)
        - Must reference existing stages
        - COMPLETION must have no next_stages (terminal)

    Stage Names:
        - Must be uppercase with underscores
        - Must be unique within a workflow

    Stage Attributes:
        - name: Stage name (string)
        - description: Human-readable description (string, non-empty)
        - keywords: List of keywords for stage matching (list of strings)
        - next_stages: Set of valid next stage names (set of strings)
    """

    # Required stage names
    REQUIRED_STAGES: frozenset[str] = frozenset({"INITIAL", "COMPLETION"})

    # Terminal stages (no next_stages allowed)
    TERMINAL_STAGES: frozenset[str] = frozenset({"COMPLETION"})

    # Reserved stage name prefixes
    RESERVED_PREFIXES: frozenset[str] = frozenset({"_", "SYSTEM", "INTERNAL"})


class StageValidator:
    """Validator for stage contract compliance.

    Validates that stage definitions satisfy the StageContract protocol.
    """

    def __init__(
        self,
        strict_mode: bool = False,
        allow_custom_stages: bool = True,
    ):
        """Initialize the stage validator.

        Args:
            strict_mode: If True, treat warnings as errors
            allow_custom_stages: If True, allow stages beyond required ones
        """
        self._strict_mode = strict_mode
        self._allow_custom_stages = allow_custom_stages

    def validate(
        self, stages: Dict[str, Any], stage_name: str = "default"
    ) -> StageValidationResult:
        """Validate stage definitions against contract.

        Args:
            stages: Dictionary of stage definitions
            stage_name: Name of the stage set being validated (for error messages)

        Returns:
            StageValidationResult with validation outcome
        """
        result = StageValidationResult(
            is_valid=True,
            errors=[],
            warnings=[],
            details={"stage_count": len(stages), "stage_name": stage_name},
        )

        # Check for required stages
        self._validate_required_stages(stages, result)

        # Check stage names
        self._validate_stage_names(stages, result)

        # Check stage attributes
        for stage_name, stage_def in stages.items():
            self._validate_stage_attributes(stage_name, stage_def, result)

        # Check transitions
        self._validate_transitions(stages, result)

        # Treat warnings as errors in strict mode
        if self._strict_mode and result.warnings:
            for warning in result.warnings:
                result.add_error(ValidationError.INVALID_TRANSITION, f"[STRICT] {warning}")
            result.warnings.clear()

        return result

    def _validate_required_stages(
        self, stages: Dict[str, Any], result: StageValidationResult
    ) -> None:
        """Validate that all required stages are present.

        Args:
            stages: Stage definitions to validate
            result: Validation result to update
        """
        missing = StageContract.REQUIRED_STAGES - set(stages.keys())
        if missing:
            result.add_error(
                ValidationError.MISSING_REQUIRED_STAGE,
                f"Missing required stages: {sorted(missing)}",
            )

    def _validate_stage_names(self, stages: Dict[str, Any], result: StageValidationResult) -> None:
        """Validate stage name format.

        Args:
            stages: Stage definitions to validate
            result: Validation result to update
        """
        for stage_name in stages.keys():
            # Check for reserved prefixes
            if any(stage_name.startswith(prefix) for prefix in StageContract.RESERVED_PREFIXES):
                result.add_error(
                    ValidationError.INVALID_STAGE_NAME,
                    f"Stage '{stage_name}' uses reserved prefix",
                )

            # Check naming convention (uppercase with underscores allowed but not required for single-word names)
            if not stage_name.isupper():
                result.add_warning(f"Stage '{stage_name}' should use UPPERCASE naming convention")

    def _validate_stage_attributes(
        self, stage_name: str, stage_def: Any, result: StageValidationResult
    ) -> None:
        """Validate individual stage attributes.

        Args:
            stage_name: Name of the stage
            stage_def: Stage definition
            result: Validation result to update
        """
        if not isinstance(stage_def, dict):
            result.add_error(
                ValidationError.INVALID_DESCRIPTION,
                f"Stage '{stage_name}' must be a dictionary",
            )
            return

        # Check required attributes
        if "name" not in stage_def:
            result.add_error(
                ValidationError.INVALID_DESCRIPTION,
                f"Stage '{stage_name}' missing 'name' attribute",
            )

        if "description" not in stage_def:
            result.add_warning(f"Stage '{stage_name}' missing 'description' attribute")
        elif (
            not isinstance(stage_def.get("description"), str)
            or not stage_def.get("description").strip()
        ):
            result.add_error(
                ValidationError.INVALID_DESCRIPTION,
                f"Stage '{stage_name}' has invalid 'description' (must be non-empty string)",
            )

        # Validate keywords
        keywords = stage_def.get("keywords", [])
        if not isinstance(keywords, list):
            result.add_error(
                ValidationError.INVALID_KEYWORDS,
                f"Stage '{stage_name}' 'keywords' must be a list",
            )
        else:
            for kw in keywords:
                if not isinstance(kw, str):
                    result.add_warning(
                        f"Stage '{stage_name}' has non-string keyword: {type(kw).__name__}"
                    )

        # Validate next_stages
        next_stages = stage_def.get("next_stages")
        if next_stages is None:
            result.add_warning(f"Stage '{stage_name}' missing 'next_stages' attribute")
        elif not isinstance(next_stages, (set, list)):
            result.add_error(
                ValidationError.MISSING_NEXT_STAGES,
                f"Stage '{stage_name}' 'next_stages' must be a set or list",
            )

        # Terminal stages should not have next_stages
        if stage_name in StageContract.TERMINAL_STAGES and next_stages:
            result.add_error(
                ValidationError.INVALID_TRANSITION,
                f"Terminal stage '{stage_name}' should not have next_stages",
            )

    def _validate_transitions(self, stages: Dict[str, Any], result: StageValidationResult) -> None:
        """Validate stage transitions.

        Args:
            stages: Stage definitions to validate
            result: Validation result to update
        """
        # Build transition graph
        graph: Dict[str, Set[str]] = {}
        for stage_name, stage_def in stages.items():
            next_stages = stage_def.get("next_stages", set())
            if isinstance(next_stages, list):
                next_stages = set(next_stages)
            graph[stage_name] = set(next_stages) if next_stages else set()

        # Check for circular transitions
        self._check_circular_transitions(graph, result)

        # Check that all referenced stages exist
        for from_stage, to_stages in graph.items():
            for to_stage in to_stages:
                if to_stage not in stages:
                    result.add_error(
                        ValidationError.INVALID_TRANSITION,
                        f"Invalid transition from '{from_stage}' to non-existent stage '{to_stage}'",
                    )

        # Check that all non-terminal stages have at least one transition
        for stage_name in stages:
            if stage_name not in StageContract.TERMINAL_STAGES:
                if not graph.get(stage_name):
                    result.add_warning(
                        f"Non-terminal stage '{stage_name}' has no next_stages defined"
                    )

    def _check_circular_transitions(
        self, graph: Dict[str, Set[str]], result: StageValidationResult
    ) -> None:
        """Check for circular transitions using depth-first search.

        Args:
            graph: Transition graph
            result: Validation result to update
        """
        visited: Set[str] = set()
        rec_stack: Set[str] = set()

        def dfs(node: str, path: List[str]) -> None:
            """Depth-first search for cycles."""
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in graph.get(node, set()):
                if neighbor not in visited:
                    dfs(neighbor, path)
                elif neighbor in rec_stack:
                    # Found a cycle
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    result.add_error(
                        ValidationError.CIRCULAR_TRANSITION,
                        f"Circular transition detected: {' -> '.join(cycle)}",
                    )

            path.pop()
            rec_stack.remove(node)

        for node in graph:
            if node not in visited:
                dfs(node, [])


class StageContractMixin:
    """Mixin for classes that need stage contract validation.

    Provides convenient methods for validating stage definitions.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the mixin."""
        super().__init__(*args, **kwargs)
        self._stage_validator = StageValidator()

    def validate_stages(
        self, stages: Dict[str, Any], stage_name: str = "default"
    ) -> StageValidationResult:
        """Validate stage definitions.

        Args:
            stages: Stage definitions to validate
            stage_name: Name of the stage set

        Returns:
            StageValidationResult
        """
        return self._stage_validator.validate(stages, stage_name)

    def ensure_stage_contract(self, stages: Dict[str, Any], stage_name: str = "default") -> None:
        """Ensure stage definitions satisfy contract, raise if not.

        Args:
            stages: Stage definitions to validate
            stage_name: Name of the stage set

        Raises:
            ValueError: If validation fails
        """
        result = self.validate_stages(stages, stage_name)
        if not result.is_valid:
            error_msgs = [msg for _, msg in result.errors]
            raise ValueError(
                f"Stage contract validation failed for '{stage_name}':\n"
                + "\n".join(f"  - {msg}" for msg in error_msgs)
            )


def validate_stage_contract(
    stages: Dict[str, Any], stage_name: str = "default", strict: bool = False
) -> StageValidationResult:
    """Convenience function to validate stage contract.

    Args:
        stages: Stage definitions to validate
        stage_name: Name of the stage set
        strict: Whether to use strict validation

    Returns:
        StageValidationResult

    Example:
        stages = {
            "INITIAL": StageDefinition(...),
            "COMPLETION": StageDefinition(...),
        }
        result = validate_stage_contract(stages)
        if not result.is_valid:
            for error_type, msg in result.errors:
                print(f"Error: {msg}")
    """
    validator = StageValidator(strict_mode=strict)
    return validator.validate(stages, stage_name)


__all__ = [
    "StageContract",
    "StageValidator",
    "StageValidationResult",
    "ValidationError",
    "validate_stage_contract",
    "StageContractMixin",
]
