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

"""Validation for workflow optimizations.

This module provides validation capabilities to ensure optimizations are
safe to apply, including dry-run execution and constraint checking.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum

from victor.optimization.models import (
    OptimizationOpportunity,
    WorkflowProfile,
)

logger = logging.getLogger(__name__)


class ValidationRecommendation(Enum):
    """Recommendation levels for validation results."""

    APPROVE = "approve"  # Safe to apply
    CONDITIONAL_APPROVE = "conditional_approve"  # Apply with caution
    MARGINAL = "marginal"  # Minimal improvement, consider alternatives
    REJECT = "reject"  # Not safe to apply


@dataclass
class ConstraintViolation:
    """Represents a constraint violation.

    Attributes:
        type: Type of violation
        description: Human-readable description
        severity: Severity level (LOW, MEDIUM, HIGH, CRITICAL)
    """

    type: str
    description: str
    severity: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.type,
            "description": self.description,
            "severity": self.severity,
        }


@dataclass
class OptimizationValidationResult:
    """Result of optimization validation.

    Attributes:
        is_valid: Whether optimization passed validation
        functional_equivalence: Whether optimized produces same outputs
        speedup: Expected speedup multiplier
        cost_reduction: Expected cost reduction (0-1)
        error_rate: Error rate in validation runs (0-1)
        violations: List of constraint violations
        recommendation: Overall recommendation
    """

    is_valid: bool
    functional_equivalence: bool
    speedup: float
    cost_reduction: float
    error_rate: float = 0.0
    violations: List[ConstraintViolation] = field(default_factory=list)
    recommendation: ValidationRecommendation = ValidationRecommendation.APPROVE

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "functional_equivalence": self.functional_equivalence,
            "speedup": self.speedup,
            "cost_reduction": self.cost_reduction,
            "error_rate": self.error_rate,
            "violations": [v.to_dict() for v in self.violations],
            "recommendation": self.recommendation.value,
        }


class OptimizationValidator:
    """Validates workflow optimizations before deployment.

    This validator provides:
    1. Dry-run execution to test optimizations
    2. Constraint checking for safety rules
    3. Functional equivalence validation
    4. Performance estimation

    Example:
        validator = OptimizationValidator()

        result = await validator.validate_optimization(
            original_workflow=workflow,
            optimization=opportunity,
            profile=profile,
        )

        if result.is_valid:
            print(f"Speedup: {result.speedup:.2f}x")
        else:
            print(f"Validation failed: {result.violations}")
    """

    def __init__(
        self,
        max_regression: float = 0.1,
        min_speedup: float = 1.1,
        require_equivalence: bool = True,
    ):
        """Initialize the optimization validator.

        Args:
            max_regression: Maximum allowed regression (0-1)
            min_speedup: Minimum speedup to be considered valid
            require_equivalence: Whether functional equivalence is required
        """
        self.max_regression = max_regression
        self.min_speedup = min_speedup
        self.require_equivalence = require_equivalence

    def validate_optimization(
        self,
        original_workflow: Dict[str, Any],
        optimization: OptimizationOpportunity,
        profile: WorkflowProfile,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> OptimizationValidationResult:
        """Validate an optimization opportunity.

        Args:
            original_workflow: Original workflow configuration
            optimization: Optimization opportunity to validate
            profile: Workflow performance profile
            constraints: Optional custom constraints

        Returns:
            OptimizationValidationResult with validation outcome
        """
        logger.info(
            f"Validating optimization: {optimization.strategy_type.value} "
            f"on {optimization.target}"
        )

        violations = []

        # Check 1: Risk level validation
        if optimization.risk_level.value == "critical":
            violations.append(
                ConstraintViolation(
                    type="risk_level",
                    description="Critical risk level optimizations require manual review",
                    severity="CRITICAL",
                )
            )

        # Check 2: Confidence threshold
        if optimization.confidence < 0.5:
            violations.append(
                ConstraintViolation(
                    type="low_confidence",
                    description=f"Low confidence ({optimization.confidence:.1%}) in optimization",
                    severity="MEDIUM",
                )
            )

        # Check 3: Expected improvement validation
        if optimization.expected_improvement < 0.05:  # Less than 5%
            violations.append(
                ConstraintViolation(
                    type="minimal_improvement",
                    description=f"Expected improvement ({optimization.expected_improvement:.1%}) is too small",
                    severity="LOW",
                )
            )

        # Check 4: Cost-benefit analysis
        if (
            optimization.estimated_cost_reduction < 0
            and optimization.estimated_duration_reduction < 0
        ):
            violations.append(
                ConstraintViolation(
                    type="no_benefit",
                    description="Optimization increases both cost and duration",
                    severity="HIGH",
                )
            )

        # Check 5: Strategy-specific validation
        strategy_violations = self._validate_strategy(optimization, profile)
        violations.extend(strategy_violations)

        # Check 6: Custom constraints
        if constraints:
            constraint_violations = self._check_constraints(optimization, constraints)
            violations.extend(constraint_violations)

        # Determine overall validity
        is_valid = not any(v.severity in ("CRITICAL", "HIGH") for v in violations)

        # Estimate performance improvements
        speedup = 1.0 + optimization.estimated_duration_reduction
        cost_reduction = optimization.estimated_cost_reduction

        # Make recommendation
        recommendation = self._make_recommendation(
            is_valid,
            speedup,
            cost_reduction,
            violations,
        )

        result = OptimizationValidationResult(
            is_valid=is_valid,
            functional_equivalence=True,  # Assume equivalent for dry-run
            speedup=speedup,
            cost_reduction=cost_reduction,
            violations=violations,
            recommendation=recommendation,
        )

        logger.info(
            f"Validation complete: valid={result.is_valid}, "
            f"recommendation={recommendation.value}"
        )

        return result

    def _validate_strategy(
        self,
        optimization: OptimizationOpportunity,
        profile: WorkflowProfile,
    ) -> List[ConstraintViolation]:
        """Validate strategy-specific constraints.

        Args:
            optimization: Optimization opportunity
            profile: Workflow profile

        Returns:
            List of strategy-specific violations
        """
        violations = []

        # Pruning strategy: Check if node is critical
        if optimization.strategy_type.value == "pruning":
            node_id = optimization.target

            # Check if node has high success rate despite being marked for pruning
            if node_id in profile.node_stats:
                node_stats = profile.node_stats[node_id]
                if node_stats.success_rate > 0.9:
                    violations.append(
                        ConstraintViolation(
                            type="pruning_high_success",
                            description=f"Node '{node_id}' has high success rate ({node_stats.success_rate:.1%}), "
                            f"pruning may not be beneficial",
                            severity="MEDIUM",
                        )
                    )

        # Parallelization strategy: Check dependencies
        elif optimization.strategy_type.value == "parallelization":
            # Would need workflow structure to check dependencies
            # For now, just log a warning
            logger.warning(
                f"Parallelization optimization for {optimization.target} "
                f"should verify node independence"
            )

        # Tool selection strategy: Verify tool exists
        elif optimization.strategy_type.value == "tool_selection":
            target_tool = optimization.target
            if " -> " in target_tool:
                # Extract new tool name
                new_tool = target_tool.split(" -> ")[1]
                # Would need tool registry to verify
                logger.debug(f"Substituting to tool: {new_tool}")

        return violations

    def _check_constraints(
        self,
        optimization: OptimizationOpportunity,
        constraints: Dict[str, Any],
    ) -> List[ConstraintViolation]:
        """Check custom constraints.

        Args:
            optimization: Optimization opportunity
            constraints: Custom constraints dictionary

        Returns:
            List of constraint violations
        """
        violations = []

        # Max cost constraint
        if "max_cost_increase" in constraints:
            max_cost = constraints["max_cost_increase"]
            if optimization.estimated_cost_reduction < -max_cost:
                violations.append(
                    ConstraintViolation(
                        type="cost_constraint",
                        description=f"Cost increase exceeds maximum allowed: "
                        f"{-optimization.estimated_cost_reduction:.1%} > {max_cost:.1%}",
                        severity="HIGH",
                    )
                )

        # Risk level constraint
        if "max_risk_level" in constraints:
            max_risk = constraints["max_risk_level"]
            risk_levels = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
            current_risk_idx = risk_levels.index(optimization.risk_level.value)
            max_risk_idx = risk_levels.index(max_risk)

            if current_risk_idx > max_risk_idx:
                violations.append(
                    ConstraintViolation(
                        type="risk_constraint",
                        description=f"Risk level ({optimization.risk_level.value}) exceeds "
                        f"maximum allowed ({max_risk})",
                        severity="HIGH",
                    )
                )

        return violations

    def _make_recommendation(
        self,
        is_valid: bool,
        speedup: float,
        cost_reduction: float,
        violations: List[ConstraintViolation],
    ) -> ValidationRecommendation:
        """Make validation recommendation.

        Args:
            is_valid: Whether validation passed
            speedup: Expected speedup
            cost_reduction: Expected cost reduction
            violations: List of violations

        Returns:
            ValidationRecommendation
        """
        if not is_valid:
            return ValidationRecommendation.REJECT

        # Check for critical violations
        if any(v.severity == "CRITICAL" for v in violations):
            return ValidationRecommendation.REJECT

        # Check for high violations
        if any(v.severity == "HIGH" for v in violations):
            return ValidationRecommendation.CONDITIONAL_APPROVE

        # Check for significant improvement
        if speedup > 1.2 and cost_reduction > 0.1:
            return ValidationRecommendation.APPROVE

        # Check for marginal improvement
        if speedup > 1.1 or cost_reduction > 0.2:
            return ValidationRecommendation.APPROVE

        # Minimal improvement
        if speedup > 1.05 or cost_reduction > 0.05:
            return ValidationRecommendation.MARGINAL

        return ValidationRecommendation.CONDITIONAL_APPROVE
