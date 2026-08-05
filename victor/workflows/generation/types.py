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

"""Type definitions for workflow validation and refinement system.

This module provides the core data structures used throughout the
workflow generation and validation pipeline.

Design Principles (SOLID):
    - SRP: Each type has a single, clear purpose
    - OCP: Types are extensible via inheritance
    - LSP: Base types can be substituted with specialized types
    - ISP: Focused interfaces for each validation layer
    - DIP: High-level modules depend on these abstractions

Key Types:
    - WorkflowValidationError: Error with location and suggestions for workflow definitions
    - WorkflowGenerationValidationResult: Aggregated result from all validation layers
    - RefinementResult: Result from automated refinement
    - RefinementHistory: Track refinement iterations

Example:
    from victor.workflows.generation.types import (
        WorkflowValidationError,
        WorkflowGenerationValidationResult,
        ErrorSeverity,
        ErrorCategory,
    )

    error = WorkflowValidationError(
        category=ErrorCategory.SCHEMA,
        severity=ErrorSeverity.ERROR,
        message="Missing required field: 'type'",
        location="nodes[0]",
        suggestion="Add 'type' field to node definition"
    )
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from datetime import datetime


class ErrorSeverity(Enum):
    """Severity levels for validation errors.

    Ordered from most severe to least severe.
    Used for error prioritization and filtering.
    """

    CRITICAL = "critical"  # Must fix - workflow cannot run
    ERROR = "error"  # Should fix - will cause runtime issues
    WARNING = "warning"  # Nice to fix - may cause issues
    INFO = "info"  # Informational - no action required

    def __lt__(self, other: "ErrorSeverity") -> bool:
        """Compare severity for sorting."""
        order = [
            ErrorSeverity.CRITICAL,
            ErrorSeverity.ERROR,
            ErrorSeverity.WARNING,
            ErrorSeverity.INFO,
        ]
        return order.index(self) < order.index(other)

    def __str__(self) -> str:
        return self.value


class ErrorCategory(Enum):
    """Categories of validation errors.

    Maps to the 4 validation layers:
    - SCHEMA: Layer 1 - JSON/YAML structure validation
    - STRUCTURE: Layer 2 - Graph topology validation
    - SEMANTIC: Layer 3 - Node semantics validation
    - SECURITY: Layer 4 - Security and safety validation
    """

    SCHEMA = "schema"
    STRUCTURE = "structure"
    SEMANTIC = "semantic"
    SECURITY = "security"


@dataclass
class WorkflowValidationError:
    """Validation error for workflow definitions with context and suggestions.

    Attributes:
        category: Type of error (schema/structure/semantic/security)
        severity: How severe the error is
        message: Human-readable error description
        location: Path to error location (JSON path format)
        suggestion: Optional fix suggestion
        value: Optional actual value that caused error
        error_code: Optional machine-readable error code
        context: Optional additional context (node data, etc.)

    Example:
        error = WorkflowValidationError(
            category=ErrorCategory.SCHEMA,
            severity=ErrorSeverity.ERROR,
            message="Missing required field: 'type'",
            location="nodes[0]",
            suggestion="Add 'type' field with one of: agent, compute, condition"
        )
    """

    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    location: str
    suggestion: Optional[str] = None
    value: Optional[Any] = None
    error_code: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON export."""
        return {
            "category": self.category.value,
            "severity": self.severity.value,
            "message": self.message,
            "location": self.location,
            "suggestion": self.suggestion,
            "value": str(self.value) if self.value is not None else None,
            "error_code": self.error_code,
            "context": self.context,
        }

    def __str__(self) -> str:
        """Format as human-readable string."""
        severity_str = f"[{self.severity.value.upper()}]"
        location_str = f"@ {self.location}" if self.location else ""
        base = f"{severity_str} {self.message} {location_str}"
        if self.suggestion:
            base += f"\n  💡 {self.suggestion}"
        return base


@dataclass
class WorkflowGenerationValidationResult:
    """Aggregated result from all workflow validation layers.

    Attributes:
        is_valid: True if no critical or error-level issues
        schema_errors: Errors from schema validation layer
        structure_errors: Errors from structure validation layer
        semantic_errors: Errors from semantic validation layer
        security_errors: Errors from security validation layer
        validation_timestamp: When validation was performed
        workflow_name: Optional name of workflow validated

    Example:
        result = WorkflowGenerationValidationResult(
            is_valid=False,
            schema_errors=[schema_error],
            structure_errors=[structure_error]
        )

        if not result.is_valid:
            print(f"Found {len(result.all_errors)} errors")
            for error in result.critical_errors:
                print(error)
    """

    is_valid: bool
    schema_errors: List[WorkflowValidationError] = field(default_factory=list)
    structure_errors: List[WorkflowValidationError] = field(default_factory=list)
    semantic_errors: List[WorkflowValidationError] = field(default_factory=list)
    security_errors: List[WorkflowValidationError] = field(default_factory=list)
    validation_timestamp: datetime = field(default_factory=datetime.now)
    workflow_name: Optional[str] = None

    @property
    def all_errors(self) -> List[WorkflowValidationError]:
        """Get all errors from all categories."""
        return (
            self.schema_errors + self.structure_errors + self.semantic_errors + self.security_errors
        )

    @property
    def critical_errors(self) -> List[WorkflowValidationError]:
        """Get only critical errors."""
        return [e for e in self.all_errors if e.severity == ErrorSeverity.CRITICAL]

    @property
    def error_count(self) -> Dict[str, int]:
        """Get count of errors by severity."""
        counts = {"critical": 0, "error": 0, "warning": 0, "info": 0}
        for error in self.all_errors:
            counts[error.severity.value] += 1
        return counts

    @property
    def category_count(self) -> Dict[str, int]:
        """Get count of errors by category."""
        return {
            "schema": len(self.schema_errors),
            "structure": len(self.structure_errors),
            "semantic": len(self.semantic_errors),
            "security": len(self.security_errors),
        }

    def summary(self) -> str:
        """Get human-readable summary."""
        if self.is_valid:
            base = "✓ Workflow validation passed"
            if self.all_errors:
                warnings = [e for e in self.all_errors if e.severity == ErrorSeverity.WARNING]
                if warnings:
                    base += f" ({len(warnings)} warnings)"
            return base

        counts = self.error_count
        total = sum(counts.values())
        return (
            f"✗ Validation failed: {total} issues "
            f"({counts['critical']} critical, {counts['error']} errors, "
            f"{counts['warning']} warnings)"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON export."""
        return {
            "is_valid": self.is_valid,
            "summary": self.summary(),
            "error_counts": self.error_count,
            "category_counts": self.category_count,
            "errors": {
                "schema": [e.to_dict() for e in self.schema_errors],
                "structure": [e.to_dict() for e in self.structure_errors],
                "semantic": [e.to_dict() for e in self.semantic_errors],
                "security": [e.to_dict() for e in self.security_errors],
            },
            "validation_timestamp": self.validation_timestamp.isoformat(),
            "workflow_name": self.workflow_name,
        }

    def group_by_node(self) -> Dict[str, List[WorkflowValidationError]]:
        """Group errors by node ID for focused reporting."""
        grouped: Dict[str, List[WorkflowValidationError]] = {}

        for error in self.all_errors:
            node_id = self._extract_node_id(error.location)
            if node_id not in grouped:
                grouped[node_id] = []
            grouped[node_id].append(error)

        return grouped

    def _extract_node_id(self, location: str) -> str:
        """Extract node ID from location path."""
        if "nodes[" in location:
            # Extract from "nodes[node_id]" or "nodes[0].field"
            import re

            match = re.search(r"nodes\[([^\]]+)\]", location)
            if match:
                return match.group(1)
        return "workflow"  # Workflow-level error


@dataclass
class RefinementResult:
    """Result from automated refinement of a workflow.

    Attributes:
        success: True if refinement was successful
        refined_schema: The refined workflow schema (dict/WorkflowDefinition)
        iterations: Number of refinement iterations performed
        fixes_applied: List of fix descriptions applied
        validation_result: Final validation result after refinement
        original_errors: Errors that were present before refinement
        remaining_errors: Errors that remain after refinement
        convergence_achieved: True if error count decreased significantly

    Example:
        result = RefinementResult(
            success=True,
            refined_schema=fixed_workflow,
            iterations=2,
            fixes_applied=["Added missing 'type' field", "Fixed edge reference"]
        )

        print(f"Applied {len(result.fixes_applied)} fixes in {result.iterations} iterations")
    """

    success: bool
    refined_schema: Any
    iterations: int
    fixes_applied: List[str] = field(default_factory=list)
    validation_result: Optional[WorkflowGenerationValidationResult] = None
    original_errors: List[WorkflowValidationError] = field(default_factory=list)
    remaining_errors: List[WorkflowValidationError] = field(default_factory=list)
    convergence_achieved: bool = False

    @property
    def errors_fixed(self) -> int:
        """Number of errors that were fixed."""
        return len(self.original_errors) - len(self.remaining_errors)

    @property
    def fix_rate(self) -> float:
        """Percentage of errors that were fixed."""
        if not self.original_errors:
            return 1.0
        return self.errors_fixed / len(self.original_errors)

    def summary(self) -> str:
        """Get human-readable summary."""
        if self.success:
            base = f"✓ Refinement successful in {self.iterations} iterations"
            if self.fixes_applied:
                base += f"\n  Fixes applied: {len(self.fixes_applied)}"
            if self.errors_fixed > 0:
                base += f"\n  Errors fixed: {self.errors_fixed}/{len(self.original_errors)}"
                base += f" ({self.fix_rate*100:.1f}%)"
            return base
        return f"✗ Refinement failed after {self.iterations} iterations"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "success": self.success,
            "iterations": self.iterations,
            "fixes_applied": self.fixes_applied,
            "errors_fixed": self.errors_fixed,
            "fix_rate": self.fix_rate,
            "convergence_achieved": self.convergence_achieved,
            "original_error_count": len(self.original_errors),
            "remaining_error_count": len(self.remaining_errors),
            "validation_result": (
                self.validation_result.to_dict() if self.validation_result else None
            ),
        }


@dataclass
class RefinementIteration:
    """Single iteration in the refinement loop.

    Attributes:
        iteration_number: Which iteration this is (0-indexed)
        workflow_schema: Workflow schema at start of iteration
        validation_result: Validation result at start of iteration
        refinement_type: Type of refinement applied (auto_fix, llm_refine, etc.)
        changes_made: List of changes applied
        duration_ms: How long this iteration took

    Example:
        iteration = RefinementIteration(
            iteration_number=0,
            workflow_schema=initial_workflow,
            validation_result=initial_validation,
            refinement_type="auto_fix"
        )
    """

    iteration_number: int
    workflow_schema: Any
    validation_result: WorkflowGenerationValidationResult
    refinement_type: str = "unknown"
    changes_made: List[str] = field(default_factory=list)
    duration_ms: Optional[float] = None

    @property
    def error_count(self) -> int:
        """Number of errors at this iteration."""
        return len(self.validation_result.all_errors)


@dataclass
class RefinementHistory:
    """Complete history of refinement iterations.

    Attributes:
        iterations: List of refinement iterations
        final_result: Final refinement result
        total_duration_ms: Total time spent refining
        converged: True if refinement converged to a valid state
        convergence_iteration: Iteration where convergence was achieved (if any)

    Example:
        history = RefinementHistory()

        for i in range(max_iterations):
            iteration = await refine_once(workflow, i)
            history.add_iteration(iteration)
            if iteration.validation_result.is_valid:
                break

        print(f"Converged after {history.convergence_iteration} iterations")
    """

    iterations: List[RefinementIteration] = field(default_factory=list)
    final_result: Optional[RefinementResult] = None
    total_duration_ms: float = 0.0
    converged: bool = False
    convergence_iteration: Optional[int] = None

    def add_iteration(self, iteration: RefinementIteration) -> None:
        """Add an iteration to the history."""
        self.iterations.append(iteration)
        if iteration.duration_ms:
            self.total_duration_ms += iteration.duration_ms

    def mark_converged(self, iteration_number: int) -> None:
        """Mark that refinement converged."""
        self.converged = True
        self.convergence_iteration = iteration_number

    def get_error_progression(self) -> List[int]:
        """Get list of error counts per iteration."""
        return [iter.error_count for iter in self.iterations]

    def has_convergence(self) -> bool:
        """Check if error counts are decreasing."""
        if len(self.iterations) < 2:
            return False

        error_counts = self.get_error_progression()
        # Check if last 2 iterations show improvement
        return error_counts[-2] > error_counts[-1]

    def summary(self) -> str:
        """Get human-readable summary."""
        if not self.iterations:
            return "No refinement iterations performed"

        base = f"Refinement history: {len(self.iterations)} iterations"
        if self.converged:
            base += f"\n✓ Converged at iteration {self.convergence_iteration}"
        else:
            base += "\n✗ Did not converge"

        if self.total_duration_ms > 0:
            base += f"\nTotal time: {self.total_duration_ms:.2f}ms"

        error_counts = self.get_error_progression()
        base += f"\nError progression: {error_counts[0]} → {error_counts[-1]}"

        return base


@dataclass
class WorkflowFix:
    """A single fix applied during refinement.

    Attributes:
        fix_type: Type of fix (schema_add, structure_remove, etc.)
        description: Human-readable description
        location: Where the fix was applied
        before_value: Value before fix (for diff)
        after_value: Value after fix (for diff)
        auto_applied: True if automatically applied (vs. manual)

    Example:
        fix = WorkflowFix(
            fix_type="schema_add",
            description="Added missing 'type' field",
            location="nodes[0]",
            after_value="agent",
            auto_applied=True
        )
    """

    fix_type: str
    description: str
    location: str
    before_value: Optional[Any] = None
    after_value: Optional[Any] = None
    auto_applied: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "fix_type": self.fix_type,
            "description": self.description,
            "location": self.location,
            "before": str(self.before_value) if self.before_value else None,
            "after": str(self.after_value) if self.after_value else None,
            "auto_applied": self.auto_applied,
        }


__all__ = [
    # Enums
    "ErrorSeverity",
    "ErrorCategory",
    # Core types
    "WorkflowValidationError",
    "WorkflowGenerationValidationResult",
    "RefinementResult",
    "RefinementIteration",
    "RefinementHistory",
    "WorkflowFix",
]
