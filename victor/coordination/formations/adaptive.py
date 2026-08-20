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

"""Adaptive formation for dynamic coordination strategy switching.

This module provides AdaptiveFormation, which monitors task execution
and dynamically switches between coordination strategies based on
performance metrics.

Formation Pattern:
    Adaptive Wrapper → (Orchestration | Hierarchy | Consensus | ...)

Adaptation Strategies:
    - PERFORMANCE: Switch based on execution time
    - ERROR_RATE: Switch based on error frequency
    - FEEDBACK: Switch based on quality feedback

Formation Cycle:
    1. OrchestrationFormation (baseline)
    2. HierarchyFormation (for large/complex tasks)
    3. ConsensusFormation (for high accuracy requirements)

SOLID Principles:
- SRP: Adaptation logic only
- OCP: Extensible via custom formation cycles
- LSP: Substitutable with other formations
- DIP: Depends on TeamContext and BaseFormationStrategy abstractions

Usage:
    from victor.coordination.formations.adaptive import AdaptiveFormation
    from victor.coordination.formations.base import TeamContext

    # Create adaptive formation
    formation = AdaptiveFormation(
        adaptation_strategy="performance",
        max_switches=3,
    )

    # Create context with agents
    context = TeamContext("team-1", "adaptive")
    context.set("agent1", agent1)
    context.set("agent2", agent2)

    # Execute with automatic formation switching
    results = await formation.execute(agents, context, task)

    # Check which formation was used
    print(f"Formation used: {results[0].metadata['current_formation']}")
    print(f"Switches made: {results[0].metadata['formation_switches']}")
"""

from __future__ import annotations

import logging
import time
from enum import Enum
from typing import Any, Dict, List, Optional

from victor.coordination.formations.base import BaseFormationStrategy, TeamContext
from victor.teams.types import AgentMessage, MemberResult

logger = logging.getLogger(__name__)


class AdaptationStrategy(str, Enum):
    """Strategy for determining when to switch formations."""

    PERFORMANCE = "performance"
    ERROR_RATE = "error_rate"
    FEEDBACK = "feedback"


class AdaptiveFormation(BaseFormationStrategy):
    """Dynamically switches formation based on performance.

    Monitors execution and switches to better formation if performance
    degrades below acceptable thresholds.

    Formation Cycle:
        1. OrchestrationFormation (baseline, fast)
        2. HierarchyFormation (divide-and-conquer for large tasks)
        3. ConsensusFormation (high accuracy through voting)

    Adaptation Strategies:
    - PERFORMANCE: Switch if execution time exceeds threshold
    - ERROR_RATE: Switch if error rate exceeds threshold
    - FEEDBACK: Switch if quality score is low

    SOLID: SRP (adaptation logic only), OCP (extensible strategies)

    Attributes:
        adaptation_strategy: How to evaluate performance
        max_switches: Maximum number of formation switches
        performance_threshold: Threshold (0-1) for switching
        formation_cycle: Ordered list of formations to try

    Example:
        >>> formation = AdaptiveFormation(
        ...     adaptation_strategy="performance",
        ...     max_switches=3,
        ... )
        >>>
        >>> context = TeamContext("team-1", "adaptive")
        >>> results = await formation.execute(agents, context, task)
        >>>
        >>> # Check adaptation
        >>> print(f"Used: {results[0].metadata['current_formation']}")
        >>> print(f"Performance: {results[0].metadata['performance_score']}")
    """

    def __init__(
        self,
        adaptation_strategy: str = "performance",
        max_switches: int = 3,
        performance_threshold: float = 0.5,
        max_duration_seconds: float = 60.0,
        formation_cycle: Optional[List[str]] = None,
    ):
        """Initialize the adaptive formation.

        Args:
            adaptation_strategy: Strategy for switching formations
                - "performance": Switch based on execution time
                - "error_rate": Switch based on error frequency
                - "feedback": Switch based on quality feedback
            max_switches: Maximum number of formation switches (default: 3)
            performance_threshold: Performance score threshold (0-1) below
                which to switch formations (default: 0.5)
            max_duration_seconds: Maximum acceptable execution time in seconds
                for performance strategy (default: 60.0)
            formation_cycle: Ordered list of formation names to try
                (default: ["orchestration", "hierarchy", "consensus"])
        """
        self.adaptation_strategy = AdaptationStrategy(adaptation_strategy)
        self.max_switches = max_switches
        self.performance_threshold = performance_threshold
        self.max_duration_seconds = max_duration_seconds

        # Formation cycle (order to try formations)
        self.formation_cycle = formation_cycle or [
            "sequential",
            "hierarchical",
            "consensus",
        ]

        # State tracking
        self._current_formation: Optional[BaseFormationStrategy] = None
        self._current_formation_name: Optional[str] = None
        self._formation_index: int = 0
        self._formation_history: List[Dict[str, Any]] = []
        self._switch_count: int = 0
        self._performance_scores: Dict[str, List[float]] = {}

    async def execute(
        self,
        agents: List[Any],
        context: TeamContext,
        task: AgentMessage,
    ) -> List[MemberResult]:
        """Execute task with adaptive formation switching.

        Args:
            agents: List of available agents
            context: Team context
            task: Task message to process

        Returns:
            List of MemberResult with metadata:
                - result: Result from execution
                - current_formation: Name of formation used
                - performance_score: Performance evaluation score
                - formation_switches: Number of switches made
                - formation_history: List of formations tried
        """
        # Initialize formation if needed
        if self._current_formation is None:
            self._select_initial_formation(task, context)

        # Track execution
        start_time = time.time()
        formation_used = self._current_formation_name

        try:
            # Execute with current formation
            logger.info(
                f"AdaptiveFormation executing with {formation_used} "
                f"(switch {self._switch_count}/{self.max_switches})"
            )

            results = await self._current_formation.execute(agents, context, task)

            # Calculate performance metrics
            duration = time.time() - start_time
            performance_score = self._evaluate_performance(results, duration)

            # Track performance score for this formation
            if formation_used not in self._performance_scores:
                self._performance_scores[formation_used] = []
            self._performance_scores[formation_used].append(performance_score)

            # Check if we should switch formations
            if self._should_switch_formation(performance_score, results):
                logger.info(
                    f"Performance score {performance_score:.2f} below threshold "
                    f"{self.performance_threshold}, switching formations"
                )
                self._switch_formation()

            # Extract result (handle different result formats)
            result = self._extract_result(results)

            # Record formation usage
            self._formation_history.append(
                {
                    "formation": formation_used,
                    "timestamp": time.time(),
                    "duration": duration,
                    "performance_score": performance_score,
                }
            )

            return [
                MemberResult(
                    member_id="adaptive_formation",
                    success=True,
                    output=str(result) if result else "",
                    metadata={
                        "result": result,
                        "current_formation": formation_used,
                        "performance_score": performance_score,
                        "formation_switches": self._switch_count,
                        "formation_history": list(self._formation_history),
                        "adaptation_strategy": self.adaptation_strategy.value,
                        "formation": "adaptive",
                    },
                )
            ]

        except Exception as e:
            logger.error(f"AdaptiveFormation failed with {formation_used}: {e}")

            # Try switching formations on error
            if self._switch_count < self.max_switches:
                logger.info("Error occurred, switching formations")
                self._switch_formation()

                # Retry with new formation
                return await self.execute(agents, context, task)

            return [
                MemberResult(
                    member_id="adaptive_formation",
                    success=False,
                    output="",
                    error=str(e),
                    metadata={
                        "current_formation": formation_used,
                        "formation_switches": self._switch_count,
                        "formation": "adaptive",
                    },
                )
            ]

    def _select_initial_formation(self, task: AgentMessage, context: TeamContext) -> None:
        """Select initial formation based on task characteristics.

        Args:
            task: Task message to analyze
            context: Team context containing optional formation hints
        """
        formation_hint = self._resolve_initial_formation_hint(context)
        if formation_hint:
            if formation_hint in self.formation_cycle:
                self._formation_index = self.formation_cycle.index(formation_hint)
                self._load_formation()
                return

            self._current_formation_name = formation_hint
            self._current_formation = self._create_formation(formation_hint)
            return

        # Use task length/complexity to select initial formation
        task_size = len(task.content)

        if task_size > 1000:
            # Large task: start with hierarchical for divide-and-conquer
            self._formation_index = 1  # hierarchical
        elif task_size > 500:
            # Medium task: start with sequential
            self._formation_index = 0  # sequential
        else:
            # Small task: start with sequential
            self._formation_index = 0  # sequential

        self._load_formation()

    @staticmethod
    def _resolve_initial_formation_hint(context: TeamContext) -> Optional[str]:
        """Resolve explicit formation override from shared context or metadata."""
        raw_hint = context.get("initial_formation_hint") or context.get("formation_hint")
        if not raw_hint:
            return None
        return str(raw_hint).strip().lower()

    def _load_formation(self) -> None:
        """Load the formation at current index in cycle."""
        formation_name = self.formation_cycle[self._formation_index]
        self._current_formation_name = formation_name
        self._current_formation = self._create_formation(formation_name)

    def _create_formation(self, formation_name: str) -> BaseFormationStrategy:
        """Create a formation instance by name.

        Args:
            formation_name: Name of formation to create

        Returns:
            Formation instance
        """
        # Lazy import to avoid circular dependencies
        if formation_name == "sequential":
            from victor.coordination.formations.sequential import (
                SequentialFormation,
            )

            return SequentialFormation()
        elif formation_name == "hierarchical":
            from victor.coordination.formations.hierarchical import (
                HierarchicalFormation,
            )

            return HierarchicalFormation()
        elif formation_name == "consensus":
            from victor.coordination.formations.consensus import ConsensusFormation

            return ConsensusFormation()
        else:
            # Fallback to sequential
            from victor.coordination.formations.sequential import (
                SequentialFormation,
            )

            logger.warning(f"Unknown formation {formation_name}, using sequential")
            return SequentialFormation()

    def _evaluate_performance(self, results: List[MemberResult], duration: float) -> float:
        """Evaluate performance of execution.

        Args:
            results: Results from formation execution
            duration: Execution duration in seconds

        Returns:
            Performance score (0-1, higher is better)
        """
        if self.adaptation_strategy == AdaptationStrategy.PERFORMANCE:
            # Score based on execution time
            # 1.0 = instant, 0.0 = max_duration_seconds or longer
            score = max(0, 1 - (duration / self.max_duration_seconds))
            return score

        elif self.adaptation_strategy == AdaptationStrategy.ERROR_RATE:
            # Score based on success rate
            if not results:
                return 0.0

            successful = sum(1 for r in results if r.success)
            score = successful / len(results)
            return score

        elif self.adaptation_strategy == AdaptationStrategy.FEEDBACK:
            # Score based on quality feedback in metadata
            if not results or not results[0].metadata:
                return 0.5  # Neutral score

            # Look for quality score in metadata
            quality_score = results[0].metadata.get("quality_score", 0.5)
            return float(quality_score)

        else:
            # Default: neutral score
            return 0.5

    def _should_switch_formation(
        self, performance_score: float, results: List[MemberResult]
    ) -> bool:
        """Determine if formation should be switched.

        Args:
            performance_score: Current performance score
            results: Results from execution

        Returns:
            True if formation should be switched
        """
        # Check switch limit
        if self._switch_count >= self.max_switches:
            return False

        # Check performance threshold
        if performance_score < self.performance_threshold:
            return True

        # Check for errors
        if self.adaptation_strategy == AdaptationStrategy.ERROR_RATE:
            if any(not r.success for r in results):
                return True

        return False

    def _switch_formation(self) -> None:
        """Switch to next formation in cycle."""
        # Record switch
        old_formation = self._current_formation_name
        self._switch_count += 1

        # Move to next formation in cycle
        self._formation_index = (self._formation_index + 1) % len(self.formation_cycle)

        # Load new formation
        self._load_formation()

        logger.info(
            f"Switched formation from {old_formation} to {self._current_formation_name} "
            f"(switch {self._switch_count}/{self.max_switches})"
        )

    def _extract_result(self, results: List[MemberResult]) -> Any:
        """Extract result from MemberResult list.

        Args:
            results: List of member results

        Returns:
            Extracted result or None
        """
        if not results:
            return None

        # Return result from first successful result
        for result in results:
            if result.success:
                return result.result

        # Fallback to first result
        return results[0].result if results else None

    def validate_context(self, context: TeamContext) -> bool:
        """Validate that context has required agents.

        Args:
            context: Team context to validate

        Returns:
            True if context has at least 2 agents
        """
        # Check if any agents are registered in context
        agent_count = 0
        for value in context.shared_state.values():
            if hasattr(value, "execute") and hasattr(value, "id"):
                agent_count += 1

        if agent_count < 2:
            logger.warning(
                f"AdaptiveFormation context has only {agent_count} agents, "
                "recommending at least 2"
            )

        return agent_count >= 1

    def supports_early_termination(self) -> bool:
        """Check if formation supports early termination.

        Adaptive formation delegates to underlying formation,
        so it depends on the current formation.

        Returns:
            True (may support early termination depending on current formation)
        """
        return True

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of performance across all formations tried.

        Returns:
            Dictionary with performance statistics
        """
        summary = {
            "formations_tried": list(self._performance_scores.keys()),
            "total_switches": self._switch_count,
            "current_formation": self._current_formation_name,
            "adaptation_strategy": self.adaptation_strategy.value,
        }

        # Add stats for each formation
        for formation, scores in self._performance_scores.items():
            if scores:
                summary[f"{formation}_avg_score"] = sum(scores) / len(scores)
                summary[f"{formation}_executions"] = len(scores)

        return summary


__all__ = ["AdaptiveFormation", "AdaptationStrategy"]
