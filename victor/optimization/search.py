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

"""Search algorithms for workflow optimization.

This module provides search algorithms for finding optimal workflow
configurations, including hill climbing and simulated annealing.
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass
from copy import deepcopy
import math

from victor.optimization.models import (
    OptimizationOpportunity,
    OptimizationStrategyType,
    WorkflowProfile,
)
from victor.optimization.generator import (
    WorkflowVariant,
    WorkflowVariantGenerator,
)

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Result of workflow optimization.

    Attributes:
        best_variant: Best workflow variant found
        best_score: Best score achieved
        iterations: Number of iterations performed
        converged: Whether optimization converged
        score_history: History of scores across iterations
    """

    best_variant: Optional[WorkflowVariant]
    best_score: float
    iterations: int
    converged: bool
    score_history: List[float] = None

    def __post_init__(self):
        if self.score_history is None:
            self.score_history = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "best_variant_id": (self.best_variant.variant_id if self.best_variant else None),
            "best_score": self.best_score,
            "iterations": self.iterations,
            "converged": self.converged,
            "score_history": self.score_history,
        }


class HillClimbingOptimizer:
    """Hill climbing optimization for workflows.

    This optimizer iteratively improves a workflow by:
    1. Starting with the initial workflow
    2. Generating neighbor variants (single changes)
    3. Evaluating each neighbor
    4. Moving to the best neighbor if it improves the score
    5. Repeating until convergence or max iterations

    Example:
        optimizer = HillClimbingOptimizer()

        result = await optimizer.optimize_workflow(
            workflow_config=config,
            profile=profile,
            max_iterations=50,
        )

        print(f"Best score: {result.best_score:.3f}")
        print(f"Improvement: {result.best_variant.expected_improvement:.1%}")
    """

    def __init__(
        self,
        variant_generator: Optional[WorkflowVariantGenerator] = None,
        objective_weights: Optional[Dict[str, float]] = None,
    ):
        """Initialize the hill climbing optimizer.

        Args:
            variant_generator: Variant generator instance
            objective_weights: Weights for multi-objective scoring
        """
        self.variant_generator = variant_generator or WorkflowVariantGenerator()

        # Default objective weights (can be customized)
        self.objective_weights = objective_weights or {
            "duration": 0.4,  # Weight for duration reduction
            "cost": 0.4,  # Weight for cost reduction
            "quality": 0.2,  # Weight for quality preservation
        }

    async def optimize_workflow(
        self,
        workflow_config: Dict[str, Any],
        profile: WorkflowProfile,
        opportunities: List[OptimizationOpportunity],
        max_iterations: int = 50,
        convergence_threshold: float = 0.01,
        score_function: Optional[Callable[[WorkflowVariant], float]] = None,
    ) -> OptimizationResult:
        """Optimize workflow using hill climbing.

        Args:
            workflow_config: Initial workflow configuration
            profile: Workflow performance profile
            opportunities: List of optimization opportunities to explore
            max_iterations: Maximum number of iterations
            convergence_threshold: Minimum improvement to continue
            score_function: Optional custom scoring function

        Returns:
            OptimizationResult with best variant found
        """
        logger.info(
            f"Starting hill climbing optimization with {len(opportunities)} opportunities, "
            f"max {max_iterations} iterations"
        )

        # Initialize with current workflow
        current_config = workflow_config
        current_score = self._evaluate_config(
            current_config,
            profile,
            score_function,
        )

        best_config = current_config
        best_score = current_score
        best_variant = None

        score_history = [current_score]

        # Track which opportunities have been applied
        applied_opportunities: List[int] = []

        for iteration in range(max_iterations):
            logger.info(
                f"Iteration {iteration + 1}/{max_iterations}, current score: {current_score:.3f}"
            )

            # Generate neighbors by applying opportunities
            neighbors = await self._generate_neighbors(
                current_config,
                profile,
                opportunities,
                applied_opportunities,
            )

            if not neighbors:
                logger.info("No more neighbors to explore, converging")
                break

            # Evaluate neighbors
            best_neighbor = None
            best_neighbor_score = current_score

            for neighbor_config, opp_idx in neighbors:
                score = self._evaluate_config(
                    neighbor_config,
                    profile,
                    score_function,
                )

                if score > best_neighbor_score:
                    best_neighbor_score = score
                    best_neighbor = (neighbor_config, opp_idx)

            # Check if improvement found
            improvement = best_neighbor_score - current_score

            if improvement > convergence_threshold:
                # Move to best neighbor
                current_config = best_neighbor[0]
                current_score = best_neighbor_score
                applied_opportunities.append(best_neighbor[1])

                logger.info(
                    f"Iteration {iteration + 1}: Improvement {improvement:.3f}, "
                    f"new score {current_score:.3f}"
                )

                # Update best
                if current_score > best_score:
                    best_config = current_config
                    best_score = current_score

            else:
                # No significant improvement, converged
                logger.info(
                    f"Converged at iteration {iteration + 1}, "
                    f"improvement {improvement:.3f} below threshold {convergence_threshold}"
                )
                break

            score_history.append(current_score)

        # Create best variant
        if applied_opportunities:
            # Generate variant from best config
            # (Simplified - would need to track actual changes)
            best_variant = WorkflowVariant(
                variant_id=f"{profile.workflow_id}_optimized",
                base_workflow_id=profile.workflow_id,
                changes=[],
                expected_improvement=(best_score - score_history[0]) / max(score_history[0], 1e-6),
                risk_level="medium",
                config=best_config,
            )

        converged = improvement <= convergence_threshold if iteration > 0 else True

        result = OptimizationResult(
            best_variant=best_variant,
            best_score=best_score,
            iterations=iteration + 1,
            converged=converged,
            score_history=score_history,
        )

        logger.info(
            f"Optimization complete: {result.iterations} iterations, "
            f"best score {result.best_score:.3f}, converged={result.converged}"
        )

        return result

    async def _generate_neighbors(
        self,
        config: Dict[str, Any],
        profile: WorkflowProfile,
        opportunities: List[OptimizationOpportunity],
        applied_indices: List[int],
    ) -> List[tuple[Dict[str, Any], int]]:
        """Generate neighbor configurations.

        Args:
            config: Current configuration
            profile: Workflow profile
            opportunities: Available optimization opportunities
            applied_indices: Indices of opportunities already applied

        Returns:
            List of (neighbor_config, opportunity_index) tuples
        """
        neighbors = []

        # Generate neighbors by applying unapplied opportunities
        for i, opportunity in enumerate(opportunities):
            # Skip if already applied
            if i in applied_indices:
                continue

            # Generate variant with this opportunity
            variant = await self.variant_generator.generate_variant(
                config,
                opportunity,
                profile,
            )

            if variant:
                neighbors.append((variant.config, i))

        # Limit number of neighbors to avoid explosion
        max_neighbors = 10
        if len(neighbors) > max_neighbors:
            # Prioritize by expected improvement
            neighbors = sorted(
                neighbors,
                key=lambda n: opportunities[n[1]].expected_improvement,
                reverse=True,
            )[:max_neighbors]

        return neighbors

    def _evaluate_config(
        self,
        config: Dict[str, Any],
        profile: WorkflowProfile,
        score_function: Optional[Callable[[WorkflowVariant], float]] = None,
    ) -> float:
        """Evaluate workflow configuration.

        Args:
            config: Workflow configuration
            profile: Original workflow profile
            score_function: Optional custom scoring function

        Returns:
            Score (higher is better)
        """
        if score_function:
            # Use custom scoring function
            # Create a temporary variant for scoring
            variant = WorkflowVariant(
                variant_id="temp",
                base_workflow_id=profile.workflow_id,
                changes=[],
                expected_improvement=0.0,
                risk_level="low",
                config=config,
            )
            return score_function(variant)

        # Default scoring: multi-objective based on profile
        # Higher score = better (lower duration, lower cost, higher success rate)

        # Normalize metrics
        duration_score = 1.0 / (profile.total_duration + 1.0)
        cost_score = 1.0 / (profile.total_cost + 1.0)
        quality_score = profile.success_rate

        # Weighted combination
        score = (
            self.objective_weights["duration"] * duration_score
            + self.objective_weights["cost"] * cost_score
            + self.objective_weights["quality"] * quality_score
        )

        return score


class SimulatedAnnealingOptimizer:
    """Simulated annealing optimization for workflows.

    This optimizer uses simulated annealing to escape local optima:
    1. Start with high temperature
    2. Generate random neighbor
    3. Accept if better, or accept with probability if worse
    4. Gradually cool down temperature
    5. Return best solution found

    Example:
        optimizer = SimulatedAnnealingOptimizer()

        result = await optimizer.optimize_workflow(
            workflow_config=config,
            profile=profile,
            opportunities=opportunities,
            initial_temperature=100.0,
        )
    """

    def __init__(
        self,
        variant_generator: Optional[WorkflowVariantGenerator] = None,
        initial_temperature: float = 100.0,
        cooling_rate: float = 0.95,
        min_temperature: float = 0.1,
    ):
        """Initialize the simulated annealing optimizer.

        Args:
            variant_generator: Variant generator instance
            initial_temperature: Starting temperature
            cooling_rate: Temperature cooling rate per iteration
            min_temperature: Minimum temperature (stopping condition)
        """
        self.variant_generator = variant_generator or WorkflowVariantGenerator()
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.min_temperature = min_temperature

    async def optimize_workflow(
        self,
        workflow_config: Dict[str, Any],
        profile: WorkflowProfile,
        opportunities: List[OptimizationOpportunity],
        score_function: Optional[Callable[[WorkflowVariant], float]] = None,
    ) -> OptimizationResult:
        """Optimize workflow using simulated annealing.

        Args:
            workflow_config: Initial workflow configuration
            profile: Workflow performance profile
            opportunities: List of optimization opportunities
            score_function: Optional custom scoring function

        Returns:
            OptimizationResult with best variant found
        """
        logger.info(
            f"Starting simulated annealing optimization with {len(opportunities)} opportunities"
        )

        # Scoring function
        def score_config(config: Dict[str, Any]) -> float:
            variant = WorkflowVariant(
                variant_id="temp",
                base_workflow_id=profile.workflow_id,
                changes=[],
                expected_improvement=0.0,
                risk_level="low",
                config=config,
            )
            if score_function:
                return score_function(variant)
            else:
                # Simple scoring
                return 1.0 / (profile.total_duration + 1.0)

        # Initialize
        current_config = workflow_config
        current_score = score_config(current_config)
        best_config = current_config
        best_score = current_score

        temperature = self.initial_temperature
        iteration = 0
        score_history = [current_score]

        # Annealing loop
        while temperature > self.min_temperature:
            # Generate random neighbor
            if opportunities:
                opportunity = random.choice(opportunities)
                variant = await self.variant_generator.generate_variant(
                    current_config,
                    opportunity,
                    profile,
                )

                if not variant:
                    temperature *= self.cooling_rate
                    iteration += 1
                    continue

                neighbor_config = variant.config
                neighbor_score = score_config(neighbor_config)
            else:
                # No opportunities, can't improve
                break

            # Calculate acceptance probability
            delta = neighbor_score - current_score

            if delta > 0:
                # Better solution, accept
                accept = True
            else:
                # Worse solution, accept with probability
                probability = math.exp(delta / temperature)
                accept = random.random() < probability

            if accept:
                current_config = neighbor_config
                current_score = neighbor_score

                # Track best
                if current_score > best_score:
                    best_config = current_config
                    best_score = current_score

            # Cool down
            temperature *= self.cooling_rate
            iteration += 1

            if iteration % 10 == 0:
                logger.info(
                    f"Iteration {iteration}: T={temperature:.2f}, "
                    f"score={current_score:.3f}, best={best_score:.3f}"
                )

            score_history.append(current_score)

        # Create best variant
        best_variant = WorkflowVariant(
            variant_id=f"{profile.workflow_id}_annealed",
            base_workflow_id=profile.workflow_id,
            changes=[],
            expected_improvement=(best_score - score_history[0]) / max(score_history[0], 1e-6),
            risk_level="medium",
            config=best_config,
        )

        result = OptimizationResult(
            best_variant=best_variant,
            best_score=best_score,
            iterations=iteration,
            converged=temperature < self.min_temperature,
            score_history=score_history,
        )

        logger.info(
            f"Simulated annealing complete: {result.iterations} iterations, "
            f"best score {result.best_score:.3f}"
        )

        return result
