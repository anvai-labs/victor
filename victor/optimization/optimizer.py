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

"""Workflow optimizer facade.

This module provides the WorkflowOptimizer facade class that integrates
all optimization components into a simple, high-level API.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass

from victor.optimization.models import (
    Bottleneck,
    OptimizationOpportunity,
    WorkflowProfile,
)
from victor.optimization.profiler import WorkflowProfiler
from victor.optimization.strategies import create_strategy
from victor.optimization.generator import (
    WorkflowVariant,
    WorkflowVariantGenerator,
)
from victor.optimization.search import (
    HillClimbingOptimizer,
    SimulatedAnnealingOptimizer,
    OptimizationResult,
)
from victor.optimization.evaluator import (
    VariantEvaluator,
    EvaluationMode,
    EvaluationResult,
)
from victor.experiments.tracking import ExperimentTracker

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for workflow optimization.

    Attributes:
        search_algorithm: Search algorithm to use ("hill_climbing" or "simulated_annealing")
        max_iterations: Maximum iterations for search
        enable_validation: Whether to validate variants before returning
        evaluation_mode: Mode for variant evaluation
        auto_apply: Whether to automatically apply optimizations (DANGEROUS)
        min_confidence: Minimum confidence threshold for suggestions
        min_improvement: Minimum expected improvement threshold
    """

    search_algorithm: str = "hill_climbing"
    max_iterations: int = 50
    enable_validation: bool = True
    evaluation_mode: EvaluationMode = EvaluationMode.ESTIMATION
    auto_apply: bool = False
    min_confidence: float = 0.6
    min_improvement: float = 0.1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "search_algorithm": self.search_algorithm,
            "max_iterations": self.max_iterations,
            "enable_validation": self.enable_validation,
            "evaluation_mode": self.evaluation_mode.value,
            "auto_apply": self.auto_apply,
            "min_confidence": self.min_confidence,
            "min_improvement": self.min_improvement,
        }


class WorkflowOptimizer:
    """Facade for workflow optimization.

    This class provides a high-level API for analyzing, profiling, and
    optimizing workflows. It integrates the profiler, strategies, search
    algorithms, and evaluator into a simple interface.

    Example:
        optimizer = WorkflowOptimizer()

        # Analyze workflow
        profile = await optimizer.analyze_workflow(
            workflow_id="my_workflow",
            experiment_tracker=tracker,
        )

        # Get suggestions
        suggestions = await optimizer.suggest_optimizations(
            workflow_id="my_workflow",
            experiment_tracker=tracker,
        )

        # Optimize workflow
        result = await optimizer.optimize_workflow(
            workflow_id="my_workflow",
            workflow_config=config,
            experiment_tracker=tracker,
        )

        print(f"Best variant: {result.best_variant.variant_id}")
        print(f"Expected improvement: {result.best_variant.expected_improvement:.1%}")
    """

    def __init__(
        self,
        config: Optional[OptimizationConfig] = None,
        experiment_tracker: Optional[ExperimentTracker] = None,
    ):
        """Initialize the workflow optimizer.

        Args:
            config: Optimization configuration
            experiment_tracker: Experiment tracker instance
        """
        self.config = config or OptimizationConfig()
        self.experiment_tracker = experiment_tracker

        # Initialize components
        self.profiler = WorkflowProfiler()
        self.variant_generator = WorkflowVariantGenerator(
            validate_structure=self.config.enable_validation,
        )
        self.evaluator = VariantEvaluator(
            recommendation_threshold=self.config.min_improvement,
        )

        # Initialize search algorithm
        if self.config.search_algorithm == "hill_climbing":
            self.search_optimizer = HillClimbingOptimizer(
                variant_generator=self.variant_generator,
            )
        elif self.config.search_algorithm == "simulated_annealing":
            self.search_optimizer = SimulatedAnnealingOptimizer(
                variant_generator=self.variant_generator,
            )
        else:
            raise ValueError(f"Unsupported search algorithm: {self.config.search_algorithm}")

    async def analyze_workflow(
        self,
        workflow_id: str,
        experiment_tracker: Optional[ExperimentTracker] = None,
        min_executions: int = 3,
    ) -> Optional[WorkflowProfile]:
        """Analyze a workflow to detect bottlenecks.

        Args:
            workflow_id: Workflow to analyze
            experiment_tracker: Experiment tracker (uses instance tracker if None)
            min_executions: Minimum executions required for analysis

        Returns:
            WorkflowProfile if analysis successful, None otherwise
        """
        logger.info(f"Analyzing workflow: {workflow_id}")

        tracker = experiment_tracker or self.experiment_tracker

        if not tracker:
            raise ValueError(
                "Experiment tracker required. "
                "Provide tracker parameter or initialize WorkflowOptimizer with tracker."
            )

        profile = await self.profiler.profile_workflow(
            workflow_id=workflow_id,
            experiment_tracker=tracker,
            min_executions=min_executions,
        )

        if profile:
            logger.info(
                f"Analysis complete: {len(profile.bottlenecks)} bottlenecks, "
                f"{len(profile.opportunities)} opportunities"
            )
        else:
            logger.warning(f"Unable to generate profile for workflow: {workflow_id}")

        return profile

    async def suggest_optimizations(
        self,
        workflow_id: str,
        experiment_tracker: Optional[ExperimentTracker] = None,
        min_executions: int = 3,
        max_suggestions: int = 10,
    ) -> List[OptimizationOpportunity]:
        """Generate optimization suggestions for a workflow.

        Args:
            workflow_id: Workflow to analyze
            experiment_tracker: Experiment tracker
            min_executions: Minimum executions required
            max_suggestions: Maximum number of suggestions to return

        Returns:
            List of optimization opportunities sorted by expected improvement
        """
        logger.info(f"Generating optimization suggestions for: {workflow_id}")

        # Analyze workflow
        profile = await self.analyze_workflow(
            workflow_id=workflow_id,
            experiment_tracker=experiment_tracker,
            min_executions=min_executions,
        )

        if not profile:
            return []

        # Filter opportunities by confidence and improvement
        filtered_opportunities = [
            opp
            for opp in profile.opportunities
            if opp.confidence >= self.config.min_confidence
            and opp.expected_improvement >= self.config.min_improvement
        ]

        # Sort by expected improvement
        filtered_opportunities.sort(
            key=lambda o: o.expected_improvement,
            reverse=True,
        )

        # Limit to max suggestions
        suggestions = filtered_opportunities[:max_suggestions]

        logger.info(
            f"Generated {len(suggestions)} suggestions "
            f"(from {len(profile.opportunities)} total opportunities)"
        )

        return suggestions

    async def optimize_workflow(
        self,
        workflow_id: str,
        workflow_config: Dict[str, Any],
        experiment_tracker: Optional[ExperimentTracker] = None,
        opportunities: Optional[List[OptimizationOpportunity]] = None,
        custom_score_function: Optional[Callable[[WorkflowVariant], float]] = None,
    ) -> Optional[OptimizationResult]:
        """Optimize a workflow using search algorithms.

        Args:
            workflow_id: Workflow to optimize
            workflow_config: Current workflow configuration
            experiment_tracker: Experiment tracker
            opportunities: Specific opportunities to explore (auto-detected if None)
            custom_score_function: Optional custom scoring function

        Returns:
            OptimizationResult with best variant found
        """
        logger.info(f"Optimizing workflow: {workflow_id}")

        # Get opportunities if not provided
        if opportunities is None:
            opportunities = await self.suggest_optimizations(
                workflow_id=workflow_id,
                experiment_tracker=experiment_tracker,
            )

            if not opportunities:
                logger.warning(f"No optimization opportunities found for: {workflow_id}")
                return None

        logger.info(f"Using {len(opportunities)} opportunities for optimization")

        # Run optimization
        result = await self.search_optimizer.optimize_workflow(
            workflow_config=workflow_config,
            profile=await self.analyze_workflow(
                workflow_id=workflow_id,
                experiment_tracker=experiment_tracker,
            ),
            opportunities=opportunities,
            max_iterations=self.config.max_iterations,
            score_function=custom_score_function,
        )

        logger.info(
            f"Optimization complete: {result.iterations} iterations, "
            f"best score {result.best_score:.3f}, "
            f"converged={result.converged}"
        )

        return result

    async def validate_variant(
        self,
        variant: WorkflowVariant,
        profile: WorkflowProfile,
        test_inputs: Optional[List[Dict[str, Any]]] = None,
    ) -> EvaluationResult:
        """Validate a workflow variant.

        Args:
            variant: Variant to validate
            profile: Original workflow profile
            test_inputs: Test inputs for dry-run validation

        Returns:
            EvaluationResult with validation scores
        """
        logger.info(f"Validating variant: {variant.variant_id}")

        # Evaluate variant
        result = await self.evaluator.evaluate_variant(
            variant=variant,
            profile=profile,
            mode=self.config.evaluation_mode,
            test_inputs=test_inputs,
        )

        # Log validation result
        logger.info(
            f"Validation result: score={result.overall_score:.3f}, "
            f"recommend={result.recommendation}, "
            f"confidence={result.confidence:.1%}"
        )

        return result

    async def apply_optimization(
        self,
        workflow_id: str,
        workflow_config: Dict[str, Any],
        opportunity: OptimizationOpportunity,
        experiment_tracker: Optional[ExperimentTracker] = None,
        validate_before_applying: bool = True,
    ) -> Optional[WorkflowVariant]:
        """Apply a single optimization to a workflow.

        Args:
            workflow_id: Workflow to optimize
            workflow_config: Current workflow configuration
            opportunity: Optimization opportunity to apply
            experiment_tracker: Experiment tracker
            validate_before_applying: Whether to validate before applying

        Returns:
            WorkflowVariant if successful, None otherwise
        """
        logger.info(f"Applying optimization {opportunity.strategy_type.value} " f"to {workflow_id}")

        # Get profile for validation
        profile = await self.analyze_workflow(
            workflow_id=workflow_id,
            experiment_tracker=experiment_tracker,
        )

        if not profile:
            logger.warning(f"Cannot apply optimization without profile for: {workflow_id}")
            return None

        # Generate variant
        variant = await self.variant_generator.generate_variant(
            workflow_config=workflow_config,
            opportunity=opportunity,
            profile=profile,
        )

        if not variant:
            logger.warning(f"Failed to generate variant for opportunity: {opportunity.target}")
            return None

        # Validate if requested
        if validate_before_applying:
            validation_result = await self.validate_variant(
                variant=variant,
                profile=profile,
            )

            if not validation_result.recommendation:
                logger.warning(
                    f"Variant {variant.variant_id} not recommended: "
                    f"score={validation_result.overall_score:.3f}"
                )
                # Return variant anyway, user can decide
            else:
                logger.info(f"Variant {variant.variant_id} validated successfully")

        logger.info(f"Applied optimization, created variant: {variant.variant_id}")

        return variant

    def set_config(self, **kwargs) -> None:
        """Update optimizer configuration.

        Args:
            **kwargs: Configuration parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"Updated config: {key} = {value}")
            else:
                logger.warning(f"Unknown configuration parameter: {key}")

    def get_config(self) -> OptimizationConfig:
        """Get current optimizer configuration.

        Returns:
            Current configuration
        """
        return self.config
