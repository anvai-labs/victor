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

"""Bayesian task analysis with belief state tracking for orchestration.

This module extends TaskAnalysis with Bayesian reasoning capabilities,
enabling principled decision-making under uncertainty in agentic workflows.

Based on: "Position: agentic AI orchestration should be Bayes-consistent"
(arXiv:2605.00742, ICML 2026)
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from victor.agent.task_analyzer import TaskAnalysis, TaskComplexity, UnifiedTaskType

logger = logging.getLogger(__name__)


@dataclass
class BayesianTaskAnalysis(TaskAnalysis):
    """Extend TaskAnalysis with Bayesian belief state for orchestration.

    This class maintains a posterior distribution over task outcomes,
    enabling Value-of-Information computations and uncertainty-aware decisions.

    Attributes:
        outcome_belief: Posterior distribution P(Y=y|D) over task outcomes
        belief_entropy: Shannon entropy H[Y|D] of the belief state
        belief_variance: Variance of the belief distribution (uncertainty metric)
        agent_likelihoods: Per-agent observation models P(z|Y)
        agent_reliability: Per-agent reliability weights α_i for evidence downweighting
        expected_voi: Expected Value of Information for next query
        query_cost_estimate: Estimated cost (in entropy units) for next query

    Example:
        >>> analysis = BayesianTaskAnalysis(
        ...     complexity=TaskComplexity.SIMPLE,
        ...     tool_budget=10,
        ...     unified_task_type=UnifiedTaskType.EDIT,
        ...     outcome_belief={"success": 0.5, "failure": 0.5},
        ... )
        >>> # Agent suggests success
        >>> likelihood = {"success": 0.8, "failure": 0.2}
        >>> analysis.compute_posterior(analysis.outcome_belief, likelihood)
        >>> # Belief should shift toward success
        >>> assert analysis.outcome_belief["success"] > 0.5
    """

    # Bayesian belief state
    outcome_belief: Dict[str, float] = field(default_factory=dict)
    belief_entropy: float = 0.0
    belief_variance: float = 0.0

    # Observation model parameters
    agent_likelihoods: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Reliability weights (α_i from paper)
    agent_reliability: Dict[str, float] = field(default_factory=dict)

    # Value of Information tracking
    expected_voi: float = 0.0
    query_cost_estimate: float = 0.0

    # Unique identifier for belief state (set by orchestration service)
    belief_id: Optional[str] = None

    def __post_init__(self):
        """Initialize belief state and compute entropy if provided."""
        # Compute entropy if belief state provided
        if self.outcome_belief:
            self._update_entropy_and_variance()

    def compute_posterior(
        self,
        prior: Dict[str, float],
        likelihood: Dict[str, float],
        agent_id: Optional[str] = None,
    ) -> None:
        """Compute Bayesian posterior: P(Y|D,z) ∝ P(Y|D) * P(z|Y).

        Args:
            prior: Current belief P(Y=y|D) before observation
            likelihood: Observation model P(z|Y=y) for each outcome
            agent_id: Optional agent ID for reliability weighting

        Updates outcome_belief in place with normalized posterior.
        """
        # Apply reliability downweighting if agent provided
        if agent_id and agent_id in self.agent_reliability:
            reliability = self.agent_reliability[agent_id]
            likelihood = {outcome: prob * reliability for outcome, prob in likelihood.items()}

        # Compute unnormalized posterior
        unnormalized = {}
        for outcome, prior_prob in prior.items():
            likelihood_prob = likelihood.get(outcome, 1.0)
            unnormalized[outcome] = prior_prob * likelihood_prob

        # Normalize
        total = sum(unnormalized.values())
        if total > 0:
            self.outcome_belief = {outcome: prob / total for outcome, prob in unnormalized.items()}
        else:
            # Fallback to uniform if normalization fails
            n_outcomes = len(prior)
            self.outcome_belief = dict.fromkeys(prior.keys(), 1.0 / n_outcomes)

        # Update entropy and variance
        self._update_entropy_and_variance()

        logger.debug(
            f"Posterior update: {self.outcome_belief}, " f"entropy={self.belief_entropy:.4f}"
        )

    def _update_entropy_and_variance(self) -> None:
        """Update entropy and variance metrics from current belief."""
        if not self.outcome_belief:
            self.belief_entropy = 0.0
            self.belief_variance = 0.0
            return

        # Compute Shannon entropy: H[Y] = -∑ p(y) log p(y)
        entropy = 0.0
        for prob in self.outcome_belief.values():
            if prob > 0:
                entropy -= prob * math.log(prob)

        self.belief_entropy = entropy

        # Compute variance (for binary outcome: Var = p * (1-p))
        if len(self.outcome_belief) == 2:
            # For binary distribution
            probs = list(self.outcome_belief.values())
            self.belief_variance = probs[0] * probs[1]
        else:
            # For multi-outcome: E[X²] - (E[X])²
            # Treat outcomes as 0, 1, 2, ...
            mean = sum(i * prob for i, (_, prob) in enumerate(sorted(self.outcome_belief.items())))
            mean_sq = sum(
                (i**2) * prob for i, (_, prob) in enumerate(sorted(self.outcome_belief.items()))
            )
            self.belief_variance = mean_sq - (mean**2)

    def compute_voi(self, agent_id: str, query_cost: float) -> float:
        """Compute Expected Value of Information: E[H[Y|D] - H[Y|D,z]] - cost.

        Args:
            agent_id: Agent being considered for query
            query_cost: Cost of query (in entropy units)

        Returns:
            Expected information gain minus query cost. Positive values
            indicate the query is worth making.

        Note:
            This is a simplified approximation. Full VoI requires computing
            E[H[Y|D,z]] = ∑_z P(z|D) * H[Y|D,z], which can be expensive.
            We approximate using current entropy as a proxy.
        """
        # Current entropy: H[Y|D]
        current_entropy = self.belief_entropy

        # Approximate expected posterior entropy
        # If agent is reliable, we expect significant entropy reduction
        # If agent is unreliable, we expect less reduction

        reliability = self.agent_reliability.get(agent_id, 0.5)
        expected_entropy_reduction = current_entropy * reliability

        # Expected information gain
        expected_gain = expected_entropy_reduction

        # Subtract cost
        voi = expected_gain - query_cost

        self.expected_voi = voi
        self.query_cost_estimate = query_cost

        logger.debug(
            f"VoI for {agent_id}: E[IG]={expected_gain:.4f}, "
            f"cost={query_cost:.4f}, VoI={voi:.4f}"
        )

        return voi

    def get_belief_dict(self) -> Dict[str, Any]:
        """Serialize belief state for storage/transmission.

        Returns:
            Dictionary containing belief state suitable for JSON serialization
        """
        return {
            "outcome_belief": self.outcome_belief,
            "belief_entropy": self.belief_entropy,
            "belief_variance": self.belief_variance,
            "agent_reliability": self.agent_reliability,
            "expected_voi": self.expected_voi,
            "query_cost_estimate": self.query_cost_estimate,
        }

    def should_query_agent(self, agent_id: str, query_cost: float) -> bool:
        """Decision rule: query if EVOI > 0.

        Args:
            agent_id: Agent being considered
            query_cost: Cost of query

        Returns:
            True if the expected information gain exceeds cost
        """
        voi = self.compute_voi(agent_id, query_cost)
        return voi > 0

    def get_uncertainty_level(self) -> str:
        """Get categorical uncertainty level from belief state.

        Returns:
            "high" if entropy > 0.5, "medium" if > 0.2, "low" otherwise
        """
        if self.belief_entropy > 0.5:
            return "high"
        elif self.belief_entropy > 0.2:
            return "medium"
        else:
            return "low"

    def get_most_likely_outcome(self) -> tuple[str, float]:
        """Get the most likely task outcome and its probability.

        Returns:
            Tuple of (outcome_name, probability)
        """
        if not self.outcome_belief:
            return ("unknown", 0.0)

        most_likely = max(self.outcome_belief.items(), key=lambda x: x[1])
        return most_likely
