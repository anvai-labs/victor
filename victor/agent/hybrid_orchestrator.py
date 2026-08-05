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

"""Hybrid orchestration router with complexity-based routing."""

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from victor.agent.complexity_detector import (
    ComplexityAnalysis,
    ComplexityLevel,
    QueryComplexityDetector,
)
from victor.framework.rl.orchestration.bayesian_orchestrator import (
    BayesianOrchestrationService,
)
from victor.framework.rl.consensus.bayesian_consensus import (
    BayesianConsensusBuilder,
)


@dataclass
class OrchestrationResult:
    """Result from orchestration (simple or Bayesian)."""

    decision: str
    confidence: float
    agent_contributions: Dict[str, Any]
    orchestration_type: str  # "simple" or "bayesian"
    complexity_analysis: Optional[ComplexityAnalysis] = None
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class HybridOrchestrationRouter:
    """Routes queries to simple or Bayesian orchestration based on complexity."""

    def __init__(
        self,
        complexity_detector: Optional[QueryComplexityDetector] = None,
        bayesian_service: Optional[BayesianOrchestrationService] = None,
        enable_bayesian: bool = True,
        force_bayesian: bool = False,
        track_performance: bool = True,
    ):
        """Initialize hybrid orchestration router.

        Args:
            complexity_detector: Query complexity analyzer
            bayesian_service: Bayesian orchestration service
            enable_bayesian: Whether to enable Bayesian orchestration
            force_bayesian: Force all queries through Bayesian (for testing)
            track_performance: Track performance metrics
        """
        self.complexity_detector = complexity_detector or QueryComplexityDetector()
        self.bayesian_service = bayesian_service
        self.enable_bayesian = enable_bayesian
        self.force_bayesian = force_bayesian
        self.track_performance = track_performance

        # Performance tracking
        self.performance_stats = {
            "simple_count": 0,
            "simple_total_latency_ms": 0.0,
            "bayesian_count": 0,
            "bayesian_total_latency_ms": 0.0,
            "total_queries": 0,
        }

    def route_query(
        self,
        query: str,
        agent_messages: Dict[str, str],
        agent_confidences: Optional[Dict[str, float]] = None,
        task_type: str = "general",
    ) -> OrchestrationResult:
        """Route query to appropriate orchestration strategy.

        Args:
            query: User query string
            agent_messages: Dict mapping agent_id to message
            agent_confidences: Optional dict mapping agent_id to confidence
            task_type: Type of task (e.g., "code_edit", "question_answering")

        Returns:
            OrchestrationResult with decision and metadata
        """
        start_time = time.time()

        # Analyze complexity
        complexity_analysis = self.complexity_detector.analyze(query)

        # Determine orchestration type
        if self.force_bayesian or (
            self.enable_bayesian and self.complexity_detector.should_use_bayesian(query)
        ):
            result = self._bayesian_orchestration(
                query,
                agent_messages,
                agent_confidences,
                task_type,
                complexity_analysis,
            )
        else:
            result = self._simple_orchestration(
                query, agent_messages, agent_confidences, complexity_analysis
            )

        # Add latency
        result.latency_ms = (time.time() - start_time) * 1000
        result.complexity_analysis = complexity_analysis

        # Track performance
        if self.track_performance:
            self._track_performance(result)

        return result

    def _simple_orchestration(
        self,
        query: str,
        agent_messages: Dict[str, str],
        agent_confidences: Optional[Dict[str, float]],
        complexity_analysis: ComplexityAnalysis,
    ) -> OrchestrationResult:
        """Simple majority vote orchestration.

        Fast path for simple queries.
        """
        # Count votes
        votes = {"Yes": 0, "No": 0, "Uncertain": 0}
        for agent_id, message in agent_messages.items():
            message_lower = message.lower().strip()
            if any(
                word in message_lower for word in ["yes", "yeah", "yep", "correct", "right", "true"]
            ):
                votes["Yes"] += 1
            elif any(
                word in message_lower for word in ["no", "nope", "incorrect", "wrong", "false"]
            ):
                votes["No"] += 1
            else:
                votes["Uncertain"] += 1

        # Determine decision
        if votes["Yes"] > votes["No"]:
            decision = "Yes"
        elif votes["No"] > votes["Yes"]:
            decision = "No"
        else:
            decision = "Uncertain"

        # Calculate confidence
        total_votes = sum(votes.values())
        max_votes = max(votes.values())
        confidence = max_votes / total_votes if total_votes > 0 else 0.0

        # Agent contributions
        agent_contributions = {
            agent_id: {"message": message, "vote": self._categorize_message(message)}
            for agent_id, message in agent_messages.items()
        }

        return OrchestrationResult(
            decision=decision,
            confidence=confidence,
            agent_contributions=agent_contributions,
            orchestration_type="simple",
            metadata={
                "vote_counts": votes,
                "reasoning": "Simple majority vote",
            },
        )

    def _bayesian_orchestration(
        self,
        query: str,
        agent_messages: Dict[str, str],
        agent_confidences: Optional[Dict[str, float]],
        task_type: str,
        complexity_analysis: ComplexityAnalysis,
    ) -> OrchestrationResult:
        """Bayesian orchestration with belief states and reliability weighting.

        Sophisticated path for complex queries.
        """
        if not self.bayesian_service:
            # Fallback to simple if Bayesian service not available
            return self._simple_orchestration(
                query, agent_messages, agent_confidences, complexity_analysis
            )

        try:
            # Create belief state
            from victor.agent.bayesian_task_analysis import BayesianTaskAnalysis
            from victor.agent.task_analyzer import TaskComplexity

            # Map complexity level to TaskComplexity
            task_complexity_map = {
                ComplexityLevel.SIMPLE: TaskComplexity.SIMPLE,
                ComplexityLevel.MODERATE: TaskComplexity.MODERATE,
                ComplexityLevel.COMPLEX: TaskComplexity.COMPLEX,
            }

            belief = self.bayesian_service.create_belief_state(
                task_type=task_type,
                complexity=task_complexity_map.get(
                    complexity_analysis.level, TaskComplexity.MODERATE
                ),
                tool_budget=10,
                initial_belief={"Yes": 0.5, "No": 0.5},
            )

            # Update with agent messages
            for agent_id, message in agent_messages.items():
                confidence = agent_confidences.get(agent_id, 0.8) if agent_confidences else 0.8
                self.bayesian_service.update_belief_with_message(
                    belief_id=belief.belief_id,
                    agent_id=agent_id,
                    message=message,
                    confidence=confidence,
                )

            # Get current belief state
            current_belief = self.bayesian_service.get_belief_state(belief.belief_id)

            # Compute consensus
            consensus_builder = BayesianConsensusBuilder(self.bayesian_service)
            consensus = consensus_builder.compute_consensus(
                belief_id=belief.belief_id,
                agent_messages=agent_messages,
                strategy="weighted_bayesian",
            )

            # Determine decision from consensus
            decision = consensus["recommended_outcome"]

            # Agent contributions
            agent_contributions = {}
            for agent_id, message in agent_messages.items():
                # Get reliability weight
                reliability_stats = (
                    self.bayesian_service.reliability_learner.get_agent_reliability_stats(agent_id)
                )
                reliability = (
                    reliability_stats["expected_reliability"] if reliability_stats else 0.7
                )

                agent_contributions[agent_id] = {
                    "message": message,
                    "reliability": reliability,
                    "weight": (
                        reliability_stats.get("alpha_reliability", 1.0)
                        if reliability_stats
                        else 1.0
                    ),
                }

            # Cleanup belief state
            self.bayesian_service.cleanup_belief_state(belief.belief_id)

            return OrchestrationResult(
                decision=decision,
                confidence=consensus["confidence"],
                agent_contributions=agent_contributions,
                orchestration_type="bayesian",
                metadata={
                    "agreement_level": consensus["agreement_level"],
                    "belief_entropy": current_belief.belief_entropy,
                    "reasoning": "Bayesian consensus with reliability weighting",
                },
            )

        except Exception as e:
            # Fallback to simple on error
            import logging

            logging.warning(f"Bayesian orchestration failed, falling back to simple: {e}")
            return self._simple_orchestration(
                query, agent_messages, agent_confidences, complexity_analysis
            )

    def _categorize_message(self, message: str) -> str:
        """Categorize message as Yes/No/Uncertain."""
        message_lower = message.lower().strip()
        if any(
            word in message_lower for word in ["yes", "yeah", "yep", "correct", "right", "true"]
        ):
            return "Yes"
        elif any(word in message_lower for word in ["no", "nope", "incorrect", "wrong", "false"]):
            return "No"
        else:
            return "Uncertain"

    def _track_performance(self, result: OrchestrationResult):
        """Track performance metrics."""
        self.performance_stats["total_queries"] += 1

        if result.orchestration_type == "simple":
            self.performance_stats["simple_count"] += 1
            self.performance_stats["simple_total_latency_ms"] += result.latency_ms
        else:
            self.performance_stats["bayesian_count"] += 1
            self.performance_stats["bayesian_total_latency_ms"] += result.latency_ms

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics.

        Returns:
            Dict with performance metrics
        """
        stats = self.performance_stats.copy()

        # Calculate averages
        if stats["simple_count"] > 0:
            stats["avg_simple_latency_ms"] = (
                stats["simple_total_latency_ms"] / stats["simple_count"]
            )
        else:
            stats["avg_simple_latency_ms"] = 0.0

        if stats["bayesian_count"] > 0:
            stats["avg_bayesian_latency_ms"] = (
                stats["bayesian_total_latency_ms"] / stats["bayesian_count"]
            )
        else:
            stats["avg_bayesian_latency_ms"] = 0.0

        if stats["total_queries"] > 0:
            stats["bayesian_percentage"] = (stats["bayesian_count"] / stats["total_queries"]) * 100
        else:
            stats["bayesian_percentage"] = 0.0

        return stats

    def reset_performance_stats(self):
        """Reset performance statistics."""
        self.performance_stats = {
            "simple_count": 0,
            "simple_total_latency_ms": 0.0,
            "bayesian_count": 0,
            "bayesian_total_latency_ms": 0.0,
            "total_queries": 0,
        }
