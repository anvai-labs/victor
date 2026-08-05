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

"""
Credit Assignment for Agentic RL in Victor.

Based on arXiv:2604.09459 "From Reasoning to Agentic: Credit Assignment in
Reinforcement Learning for Large Language Models".

This module provides multi-granularity credit assignment mechanisms for:
- Token-level credit for reasoning chains (500-30K tokens)
- Step-level credit for multi-turn workflows (100K-1M tokens)
- Agent-level credit for multi-agent scenarios

Two regimes from the paper:
1. **Reasoning RL**: Intra-chain credit assignment for CoT generation
2. **Agentic RL**: Episode-level credit for multi-turn interaction

Usage:
    from victor.framework.rl.credit_assignment import (
        CreditAssignmentIntegration,
        CreditMethodology,
        CreditGranularity,
    )

    # Create integration
    ca = CreditAssignmentIntegration()

    # Assign credit to trajectory
    signals = ca.assign_credit(
        trajectory=action_metadata_list,
        rewards=reward_list,
        methodology=CreditMethodology.GAE,
    )

    # Get attribution for specific agent
    attribution = ca.get_agent_attribution(agent_id="agent_1")
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    TypeVar,
)

from typing_extensions import TypeAlias

logger = logging.getLogger(__name__)

# Import RL types
from victor.framework.rl.base import RLOutcome

if TYPE_CHECKING:
    try:
        from victor.framework.graph import StateGraph
    except ImportError:
        StateGraph = None  # type: ignore


# ============================================================================
# Core Data Structures
# ============================================================================


class CreditGranularity(Enum):
    """Granularity levels for credit assignment.

    Maps to the two regimes from the paper:
    - Reasoning RL: token, segment
    - Agentic RL: step, turn, agent, episode
    """

    TOKEN = "token"  # Individual token in reasoning chain
    SEGMENT = "segment"  # Group of tokens (e.g., reasoning step)
    STEP = "step"  # Single action within a turn
    TURN = "turn"  # Complete agent interaction (request-response)
    AGENT = "agent"  # Agent-level across multiple turns
    EPISODE = "episode"  # Full trajectory from start to terminal state


class CreditMethodology(Enum):
    """Credit assignment methodologies from the paper (arXiv:2604.09459).

    Implemented methods (9):
    - Value-based: Monte Carlo, TD(λ), GAE, N-step returns
    - Game-theoretic: Shapley values
    - Information-theoretic: Hindsight (HER-style)
    - Counterfactual: C3 (leave-one-out)
    - Bifurcation: CARL (entropy-based critical action detection)
    - Critic-based: LLM-as-Critic (Ollama edge model, falls back to GAE)

    Planned methods (3) — raise NotImplementedError if used directly:
    - Actor-Critic (needs neural value network), CCPO, HCAPO
    """

    # Value-based methods (implemented)
    MONTE_CARLO = "monte_carlo"  # Full return credit → SegmentLevelCreditAssigner
    TEMPORAL_DIFFERENCE = "td"  # TD(λ) bootstrapped credit → TokenLevelCreditAssigner
    GAE = "gae"  # Generalized Advantage Estimation → EpisodeLevelCreditAssigner
    N_STEP_RETURNS = "n_step"  # N-step with bifurcation → TurnLevelCreditAssigner

    # Game-theoretic methods (implemented)
    SHAPLEY = "shapley"  # Shapley value attribution → MultiAgentCreditAssigner

    # Information-theoretic (implemented)
    HINDSIGHT = "hindsight"  # HER-style goal relabeling → HindsightCreditAssigner

    # Counterfactual (implemented)
    C3 = "c3"  # Leave-one-out counterfactual → CounterfactualCreditAssigner

    # Bifurcation (implemented)
    CARL = "carl"  # Entropy-based critical action detection → CARLCreditAssigner

    # Critic-based (implemented)
    LLM_AS_CRITIC = "llm_critic"  # LLM evaluates actions → LLMCriticCreditAssigner

    # Planned — not yet implemented (will raise NotImplementedError)
    ACTOR_CRITIC = "actor_critic"  # Separate value network
    CCPO = "ccpo"  # Counterfactual Credit-weighted PPO
    HCAPO = "hcapo"  # Hierarchical Credit Assignment PPO


@dataclass(frozen=True)
class ActionMetadata:
    """Metadata associated with an action for credit attribution.

    Captures context needed to assign credit fairly:
    - Agent/team responsible
    - Position in trajectory
    - Tool/method used
    - Timestamp and duration
    """

    agent_id: str
    team_id: Optional[str] = None
    turn_index: int = 0
    step_index: int = 0
    tool_name: Optional[str] = None
    method_name: Optional[str] = None
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    duration_ms: int = 0
    parent_action_id: Optional[str] = None
    action_id: str = ""
    session_id: Optional[str] = None

    def __post_init__(self):
        # Generate action_id if not provided
        if not self.action_id:
            object.__setattr__(self, "action_id", f"action_{self.agent_id}_{self.timestamp}")


@dataclass
class CreditSignal:
    """A credit assignment for a single action or agent.

    Combines reward with attribution metadata:
    - raw_reward: Observed reward (could be sparse)
    - credit: Assigned credit (densified via CA methods)
    - confidence: How confident the CA method is
    - methodology: Which CA method produced this
    """

    action_id: str
    raw_reward: float
    credit: float
    confidence: float = 0.0
    methodology: Optional[CreditMethodology] = None
    granularity: CreditGranularity = CreditGranularity.STEP
    metadata: Optional[ActionMetadata] = None
    attribution: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "action_id": self.action_id,
            "raw_reward": self.raw_reward,
            "credit": self.credit,
            "confidence": self.confidence,
            "methodology": self.methodology.value if self.methodology else None,
            "granularity": self.granularity.value,
            "metadata": (
                {
                    "agent_id": self.metadata.agent_id,
                    "team_id": self.metadata.team_id,
                    "turn_index": self.metadata.turn_index,
                    "step_index": self.metadata.step_index,
                    "tool_name": self.metadata.tool_name,
                    "method_name": self.metadata.method_name,
                    "timestamp": self.metadata.timestamp,
                    "duration_ms": self.metadata.duration_ms,
                }
                if self.metadata
                else None
            ),
            "attribution": self.attribution,
        }


@dataclass
class TrajectorySegment:
    """A segment of a trajectory for credit assignment.

    Represents a contiguous sequence of actions for segment-level credit.
    """

    segment_id: str
    action_ids: List[str]
    start_time: float
    end_time: float
    agent_id: str
    outcome: Optional[float] = None
    parent_segment_id: Optional[str] = None

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def action_count(self) -> int:
        return len(self.action_ids)


@dataclass
class CreditAssignmentConfig:
    """Configuration for credit assignment behavior."""

    methodology: CreditMethodology = CreditMethodology.GAE
    granularity: CreditGranularity = CreditGranularity.STEP
    gamma: float = 0.99  # Discount factor
    lambda_gae: float = 0.95  # GAE parameter
    n_step: int = 5  # N-step return parameter
    credit_confidence_threshold: float = 0.5
    hindsight_ratio: float = 0.8
    shapley_sampling_count: int = 10
    enable_bifurcation_detection: bool = True
    bifurcation_threshold: float = 0.3


# ============================================================================
# Protocols
# ============================================================================


T = TypeVar("T")


class CreditAssignmentProvider(Protocol[T]):
    """Protocol for credit assignment services."""

    def assign_credit(
        self,
        trajectory: List[T],
        rewards: List[float],
        config: Optional[CreditAssignmentConfig] = None,
    ) -> List[CreditSignal]:
        """Assign credit to actions in a trajectory."""
        ...

    def get_credit(self, action_id: str) -> Optional[CreditSignal]:
        """Retrieve credit for a specific action."""
        ...

    def get_attribution(
        self,
        agent_id: str,
        granularity: CreditGranularity = CreditGranularity.AGENT,
    ) -> Dict[str, float]:
        """Get credit attribution for an agent."""
        ...

    def reset(self) -> None:
        """Reset internal state."""
        ...


# ============================================================================
# Base Classes
# ============================================================================


class BaseCreditAssigner(ABC, Generic[T]):
    """Base class for credit assignment implementations."""

    def __init__(self, config: Optional[CreditAssignmentConfig] = None):
        self.config = config or CreditAssignmentConfig()
        self._credit_store: Dict[str, CreditSignal] = {}
        self._attribution_cache: Dict[str, Dict[str, float]] = {}

    @abstractmethod
    def assign_credit(
        self,
        trajectory: List[T],
        rewards: List[float],
        config: Optional[CreditAssignmentConfig] = None,
    ) -> List[CreditSignal]:
        """Assign credit to trajectory. Must be implemented by subclasses."""
        ...

    def get_credit(self, action_id: str) -> Optional[CreditSignal]:
        """Retrieve credit for a specific action."""
        return self._credit_store.get(action_id)

    def get_attribution(
        self,
        agent_id: str,
        granularity: CreditGranularity = CreditGranularity.AGENT,
    ) -> Dict[str, float]:
        """Get credit attribution for an agent."""
        cache_key = f"{agent_id}:{granularity.value}"
        if cache_key in self._attribution_cache:
            return self._attribution_cache[cache_key]

        attribution: Dict[str, float] = defaultdict(float)
        for signal in self._credit_store.values():
            if signal.metadata and signal.metadata.agent_id == agent_id:
                for contributor, amount in signal.attribution.items():
                    attribution[contributor] += amount

        self._attribution_cache[cache_key] = dict(attribution)
        return self._attribution_cache[cache_key]

    def reset(self) -> None:
        """Reset internal state."""
        self._credit_store.clear()
        self._attribution_cache.clear()

    def _compute_returns(self, rewards: List[float], gamma: Optional[float] = None) -> List[float]:
        """Compute discounted returns (Monte Carlo)."""
        gamma = gamma or self.config.gamma
        returns = []
        running_return = 0.0
        for reward in reversed(rewards):
            running_return = reward + gamma * running_return
            returns.insert(0, running_return)
        return returns


# ============================================================================
# Reasoning RL Credit Assigners (Token/Segment Level)
# ============================================================================


class TokenLevelCreditAssigner(BaseCreditAssigner[str]):
    """Credit assignment at token level for reasoning chains.

    Uses TD learning with value function bootstrapping for long CoT chains.
    """

    def __init__(
        self,
        config: Optional[CreditAssignmentConfig] = None,
        value_function: Optional[Callable[[List[str], int], float]] = None,
    ):
        super().__init__(config)
        self._value_function = value_function or self._default_value_function

    def assign_credit(
        self,
        trajectory: List[str],
        rewards: List[float],
        config: Optional[CreditAssignmentConfig] = None,
    ) -> List[CreditSignal]:
        """Assign credit to tokens in reasoning chain using TD(λ).

        Standard forward-view TD(λ): credit for token t is the λ-return,
        which is a weighted combination of n-step returns. Equivalent to
        GAE at token granularity: credit_t = Σ_{l=0}^{T-t-1} (γλ)^l δ_{t+l}
        where δ_t = r_t + γ V(t+1) - V(t).
        """
        if len(trajectory) != len(rewards):
            raise ValueError("Trajectory and rewards must have same length")

        cfg = config or self.config
        n = len(trajectory)
        signals = []

        # Compute value estimates for each position
        values = [self._value_function(trajectory, t) for t in range(n)]
        values.append(0.0)  # Terminal value

        # Compute TD errors: δ_t = r_t + γ V(t+1) - V(t)
        td_errors = [rewards[t] + cfg.gamma * values[t + 1] - values[t] for t in range(n)]

        # Compute GAE-style credit (backward pass):
        # credit_t = Σ_{l=0}^{T-t-1} (γλ)^l δ_{t+l}
        credits = [0.0] * n
        running = 0.0
        for t in reversed(range(n)):
            running = td_errors[t] + cfg.gamma * cfg.lambda_gae * running
            credits[t] = running

        for t in range(n):
            action_id = f"token_{t}_{hash(trajectory[t]) & 0xffff}"
            signal = CreditSignal(
                action_id=action_id,
                raw_reward=rewards[t],
                credit=credits[t],
                confidence=min(abs(credits[t]) / (abs(td_errors[t]) + 1e-8), 1.0),
                methodology=CreditMethodology.TEMPORAL_DIFFERENCE,
                granularity=CreditGranularity.TOKEN,
            )
            signals.append(signal)
            self._credit_store[action_id] = signal

        return signals

    def _default_value_function(self, tokens: List[str], index: int) -> float:
        """Default value function (exponential decay from end)."""
        remaining = len(tokens) - index
        return 1.0 / (1.0 + len(tokens) - remaining)


class SegmentLevelCreditAssigner(BaseCreditAssigner[TrajectorySegment]):
    """Credit assignment at segment level using Monte Carlo returns."""

    def assign_credit(
        self,
        trajectory: List[TrajectorySegment],
        rewards: List[float],
        config: Optional[CreditAssignmentConfig] = None,
    ) -> List[CreditSignal]:
        """Assign credit to segments using Monte Carlo returns."""
        if len(trajectory) != len(rewards):
            raise ValueError("Trajectory and rewards must have same length")

        cfg = config or self.config
        signals = []
        returns = self._compute_returns(rewards, cfg.gamma)

        for i, segment in enumerate(trajectory):
            credit = returns[i]

            # Distribute credit within segment
            segment_attribution = {}
            if segment.action_ids:
                per_action_credit = credit / len(segment.action_ids)
                for action_id in segment.action_ids:
                    segment_attribution[action_id] = per_action_credit

            signal = CreditSignal(
                action_id=segment.segment_id,
                raw_reward=rewards[i],
                credit=credit,
                confidence=1.0,
                methodology=CreditMethodology.MONTE_CARLO,
                granularity=CreditGranularity.SEGMENT,
                attribution=segment_attribution,
            )
            signals.append(signal)
            self._credit_store[segment.segment_id] = signal

        return signals


# ============================================================================
# Agentic RL Credit Assigners (Step/Turn/Agent Level)
# ============================================================================


class EpisodeLevelCreditAssigner(BaseCreditAssigner[ActionMetadata]):
    """Episode-level credit assignment using GAE.

    Implements Generalized Advantage Estimation for bias-variance tradeoff.
    """

    def assign_credit(
        self,
        trajectory: List[ActionMetadata],
        rewards: List[float],
        config: Optional[CreditAssignmentConfig] = None,
    ) -> List[CreditSignal]:
        """Assign credit using GAE (Generalized Advantage Estimation)."""
        if len(trajectory) != len(rewards):
            raise ValueError("Trajectory and rewards must have same length")

        cfg = config or self.config
        signals = []

        # Estimate values
        values = self._estimate_values(rewards)

        # Compute GAE advantages
        advantages = self._compute_gae(rewards, values, cfg.gamma, cfg.lambda_gae)

        # Credit = advantage + value (baseline)
        for i, action_meta in enumerate(trajectory):
            credit = advantages[i] + values[i]

            signal = CreditSignal(
                action_id=action_meta.action_id,
                raw_reward=rewards[i],
                credit=credit,
                confidence=0.8,
                methodology=CreditMethodology.GAE,
                granularity=cfg.granularity,
                metadata=action_meta,
            )
            signals.append(signal)
            self._credit_store[action_meta.action_id] = signal

        return signals

    def _estimate_values(self, rewards: List[float]) -> List[float]:
        """Estimate state values using discounted future returns.

        Without a learned critic, the best critic-free baseline is the
        discounted return from each state: V(s_t) ≈ Σ_{k=0}^{T-t} γ^k r_{t+k}.
        This is then used by GAE to compute advantages.
        """
        gamma = self.config.gamma
        n = len(rewards)
        values = [0.0] * n
        running = 0.0
        for t in reversed(range(n)):
            running = rewards[t] + gamma * running
            values[t] = running
        return values

    def _compute_gae(
        self, rewards: List[float], values: List[float], gamma: float, lambda_gae: float
    ) -> List[float]:
        """Compute Generalized Advantage Estimation."""
        advantages = []
        last_advantage = 0.0

        for t in reversed(range(len(rewards))):
            if t + 1 < len(rewards):
                next_value = values[t + 1]
            else:
                next_value = 0.0

            delta = rewards[t] + gamma * next_value - values[t]
            advantage = delta + gamma * lambda_gae * last_advantage
            advantages.insert(0, advantage)
            last_advantage = advantage

        return advantages


class TurnLevelCreditAssigner(BaseCreditAssigner[ActionMetadata]):
    """Turn-level credit assignment for multi-turn workflows.

    Implements bifurcation point detection for critical actions.
    """

    def __init__(
        self,
        config: Optional[CreditAssignmentConfig] = None,
    ):
        super().__init__(config)
        self._critical_actions: set[str] = set()

    def assign_credit(
        self,
        trajectory: List[ActionMetadata],
        rewards: List[float],
        config: Optional[CreditAssignmentConfig] = None,
    ) -> List[CreditSignal]:
        """Assign credit at turn level with critical action detection."""
        if len(trajectory) != len(rewards):
            raise ValueError("Trajectory and rewards must have same length")

        cfg = config or self.config
        signals = []

        # Group by turn
        turn_groups: Dict[int, List[tuple[int, ActionMetadata, float]]] = defaultdict(list)
        for i, action_meta in enumerate(trajectory):
            turn_idx = action_meta.turn_index
            turn_groups[turn_idx].append((i, action_meta, rewards[i]))

        # Compute turn-level returns
        for turn_idx, group in sorted(turn_groups.items()):
            turn_rewards = [r for _, _, r in group]
            turn_return = sum(turn_rewards) * (cfg.gamma ** len(turn_rewards))

            # Detect critical actions
            has_critical = False
            for idx, action_meta, _ in group:
                if self._is_critical_action(action_meta, rewards, idx):
                    has_critical = True
                    self._critical_actions.add(action_meta.action_id)

            # Assign credit with boost for critical actions
            for idx, action_meta, reward in group:
                credit = turn_return / len(group)
                if action_meta.action_id in self._critical_actions:
                    credit *= 1.5  # Boost critical actions

                signal = CreditSignal(
                    action_id=action_meta.action_id,
                    raw_reward=reward,
                    credit=credit,
                    confidence=0.9 if has_critical else 0.7,
                    methodology=CreditMethodology.N_STEP_RETURNS,
                    granularity=CreditGranularity.TURN,
                    metadata=action_meta,
                )
                signals.append(signal)
                self._credit_store[action_meta.action_id] = signal

        return signals

    def _is_critical_action(self, action: ActionMetadata, rewards: List[float], idx: int) -> bool:
        """Detect if action is a bifurcation point."""
        if not self.config.enable_bifurcation_detection:
            return False

        if idx + 1 < len(rewards):
            future_variance = (
                abs(rewards[idx + 1]) - abs(rewards[idx])
                if idx + 2 < len(rewards)
                else abs(rewards[idx + 1])
            )
            return future_variance > self.config.bifurcation_threshold

        return False


class HindsightCreditAssigner(BaseCreditAssigner[ActionMetadata]):
    """Hindsight credit assignment for failed trajectories.

    Re-frames failures as successes with different goals.
    """

    def __init__(
        self,
        config: Optional[CreditAssignmentConfig] = None,
        goal_generator: Optional[Callable[[List[ActionMetadata]], List[Any]]] = None,
    ):
        super().__init__(config)
        self._goal_generator = goal_generator or self._default_goal_generator

    def assign_credit(
        self,
        trajectory: List[ActionMetadata],
        rewards: List[float],
        config: Optional[CreditAssignmentConfig] = None,
    ) -> List[CreditSignal]:
        """Assign credit with hindsight goal relabeling (HER-style).

        For successful trajectories, delegates to GAE.
        For failed trajectories, applies hindsight by treating each achieved
        intermediate state as a retrospective "goal". Actions that made
        progress toward any achieved state receive positive credit, extracting
        learning value from failures.
        """
        cfg = config or self.config
        total_reward = sum(rewards)

        # If trajectory succeeded overall, use standard GAE
        if total_reward >= 0:
            return EpisodeLevelCreditAssigner(cfg).assign_credit(trajectory, rewards, cfg)

        # Failed trajectory — apply hindsight goal relabeling
        return self._assign_hindsight_credit(trajectory, rewards, cfg)

    def _assign_hindsight_credit(
        self,
        trajectory: List[ActionMetadata],
        rewards: List[float],
        cfg: CreditAssignmentConfig,
    ) -> List[CreditSignal]:
        """Assign credit using hindsight goal relabeling.

        Hindsight Experiencer Replay (HER) insight: a failed trajectory
        is a *successful* trajectory for the goal it actually achieved.

        For each action, we compute:
        1. Original reward (negative, since trajectory failed)
        2. Hindsight goals: sample `hindsight_ratio` fraction of future
           achieved states as retrospective goals
        3. For each hindsight goal, the action gets positive credit if it
           contributed to reaching that achieved state (measured by reward
           improvement in the sub-trajectory leading to the goal)
        4. Final credit = blend of original reward and hindsight bonus
        """
        n = len(trajectory)
        if n == 0:
            return []

        signals = []
        # Generate hindsight goals from achieved intermediate states
        goals = self._goal_generator(trajectory)
        num_goals = max(1, int(len(goals) * cfg.hindsight_ratio))
        selected_goals = goals[:num_goals]

        for i, action_meta in enumerate(trajectory):
            raw = rewards[i] if i < len(rewards) else 0.0

            # Compute hindsight bonus: how many future "goals" does this
            # action help reach? Actions earlier in the trajectory that
            # precede achieved goals get credit for enabling them.
            hindsight_bonus = 0.0
            for goal_action in selected_goals:
                goal_idx = next(
                    (j for j, a in enumerate(trajectory) if a.action_id == goal_action.action_id),
                    -1,
                )
                if goal_idx > i:
                    # This action preceded the achieved goal —
                    # credit decays with distance (γ^distance)
                    distance = goal_idx - i
                    hindsight_bonus += cfg.gamma**distance

            # Normalize by number of goals
            if num_goals > 0:
                hindsight_bonus /= num_goals

            # Blend: original credit (weighted down) + hindsight bonus
            credit = raw * (1.0 - cfg.hindsight_ratio) + hindsight_bonus * cfg.hindsight_ratio

            signal = CreditSignal(
                action_id=action_meta.action_id,
                raw_reward=raw,
                credit=credit,
                confidence=cfg.hindsight_ratio * min(1.0, hindsight_bonus + 0.3),
                methodology=CreditMethodology.HINDSIGHT,
                granularity=cfg.granularity,
                metadata=action_meta,
            )
            signals.append(signal)
            self._credit_store[action_meta.action_id] = signal

        return signals

    def _default_goal_generator(self, trajectory: List[ActionMetadata]) -> List[ActionMetadata]:
        """Generate hindsight goals from achieved intermediate states.

        Uses the last portion of the trajectory as achieved "goals" —
        these are states the agent actually reached, even if the overall
        task failed.
        """
        # Use latter half of trajectory as achieved goals
        midpoint = len(trajectory) // 2
        return trajectory[midpoint:] if len(trajectory) > 2 else trajectory


class MultiAgentCreditAssigner(BaseCreditAssigner[ActionMetadata]):
    """Credit assignment for multi-agent scenarios using Shapley values."""

    def assign_credit(
        self,
        trajectory: List[ActionMetadata],
        rewards: List[float],
        config: Optional[CreditAssignmentConfig] = None,
    ) -> List[CreditSignal]:
        """Assign credit using Shapley values for fair attribution."""
        if len(trajectory) != len(rewards):
            raise ValueError("Trajectory and rewards must have same length")

        cfg = config or self.config

        # Group by agent
        agent_groups: Dict[str, List[tuple[int, ActionMetadata, float]]] = defaultdict(list)
        for i, action_meta in enumerate(trajectory):
            agent_groups[action_meta.agent_id].append((i, action_meta, rewards[i]))

        # Compute Shapley values
        shapley_values = self._compute_shapley_values(agent_groups, sum(rewards), cfg)

        # Assign credit with Shapley attribution
        signals = []
        for agent_id, agent_actions in agent_groups.items():
            agent_credit = shapley_values.get(agent_id, 0.0)

            for idx, action_meta, reward in agent_actions:
                per_action_credit = agent_credit / len(agent_actions)

                # Attribution to other agents
                attribution = {agent_id: per_action_credit}
                for other_agent in shapley_values:
                    if other_agent != agent_id:
                        attribution[other_agent] = (
                            shapley_values[other_agent] * 0.1 / len(agent_actions)
                        )

                signal = CreditSignal(
                    action_id=action_meta.action_id,
                    raw_reward=reward,
                    credit=per_action_credit,
                    confidence=0.85,
                    methodology=CreditMethodology.SHAPLEY,
                    granularity=CreditGranularity.AGENT,
                    metadata=action_meta,
                    attribution=attribution,
                )
                signals.append(signal)
                self._credit_store[action_meta.action_id] = signal

        return signals

    def _compute_shapley_values(
        self,
        agent_groups: Dict[str, List[tuple[int, ActionMetadata, float]]],
        total_reward: float,
        config: CreditAssignmentConfig,
    ) -> Dict[str, float]:
        """Compute Shapley values using Monte Carlo permutation sampling.

        For each random permutation of agents, computes the marginal contribution
        of each agent: V(S ∪ {i}) - V(S), where V(S) is the coalition value
        (total reward from agents in set S). Averaged over all sampled permutations.

        Satisfies the Shapley efficiency axiom: Σ φ_i = V(N) = total_reward.
        """
        import random

        agents = list(agent_groups.keys())
        shapley: Dict[str, float] = dict.fromkeys(agents, 0.0)

        # Single agent case — assign full reward
        if len(agents) == 1:
            return {agents[0]: total_reward}

        # Precompute per-agent reward sums for coalition value function
        agent_reward: Dict[str, float] = {
            aid: sum(r for _, _, r in actions) for aid, actions in agent_groups.items()
        }

        # Use deterministic seed for reproducibility when sampling count is small
        rng = random.Random(42)

        for _ in range(config.shapley_sampling_count):
            perm = agents[:]
            rng.shuffle(perm)

            # V(S) = sum of rewards from agents in coalition S
            coalition_value = 0.0
            for agent in perm:
                # Marginal contribution: V(S ∪ {agent}) - V(S)
                marginal = agent_reward[agent]
                shapley[agent] += marginal / config.shapley_sampling_count
                coalition_value += agent_reward[agent]

        # The above gives exact Shapley values when V is additive (V(S) = Σ_{i∈S} v_i).
        # For interaction effects, we apply a synergy adjustment based on
        # deviation from additivity: if total_reward != sum of individual rewards,
        # distribute the surplus/deficit proportionally.
        additive_total = sum(agent_reward.values())
        synergy = total_reward - additive_total
        if abs(synergy) > 1e-10 and abs(additive_total) > 1e-10:
            for agent in agents:
                # Distribute synergy proportional to each agent's share
                share = agent_reward[agent] / additive_total
                shapley[agent] += synergy * share

        return shapley


# ============================================================================
# Counterfactual Credit Assigner (C3)
# ============================================================================


class CounterfactualCreditAssigner(BaseCreditAssigner[ActionMetadata]):
    """Counterfactual credit assignment via leave-one-out analysis (C3).

    For each action i in the trajectory, computes:
        credit_i = V(trajectory) - V(trajectory without action i)

    where V(T) is the discounted return of trajectory T. The marginal
    impact of removing each action reveals how critical it was.

    Reference: arXiv:2604.09459 Section 4.3 (C3 — Counterfactual Credit)
    """

    def assign_credit(
        self,
        trajectory: List[ActionMetadata],
        rewards: List[float],
        config: Optional[CreditAssignmentConfig] = None,
    ) -> List[CreditSignal]:
        """Assign credit using leave-one-out counterfactual analysis."""
        if len(trajectory) != len(rewards):
            raise ValueError("Trajectory and rewards must have same length")

        cfg = config or self.config
        n = len(trajectory)
        signals = []

        # Compute full trajectory value: V(T) = Σ γ^t r_t
        full_value = sum(rewards[t] * cfg.gamma**t for t in range(n))

        for i in range(n):
            # V(T \ {i}): trajectory value without action i
            # Remaining actions shift forward; rewards of removed action = 0
            counterfactual_value = sum(
                rewards[t] * cfg.gamma ** (t if t < i else t - 1) for t in range(n) if t != i
            )

            # Credit = marginal contribution of action i
            credit = full_value - counterfactual_value

            signal = CreditSignal(
                action_id=trajectory[i].action_id,
                raw_reward=rewards[i],
                credit=credit,
                confidence=0.9,  # High confidence (exact computation, no sampling)
                methodology=CreditMethodology.C3,
                granularity=cfg.granularity,
                metadata=trajectory[i],
            )
            signals.append(signal)
            self._credit_store[trajectory[i].action_id] = signal

        return signals


# ============================================================================
# Critical Action Identifier (CARL)
# ============================================================================


class CriticalActionIdentifier:
    """Identifies bifurcation points in trajectories.

    Implements Critical Action Refinement Learning (CARL) with entropy-based
    detection. Points where the reward distribution has high local entropy
    indicate the trajectory could have gone either way — these are
    the critical decision points where the agent's choice mattered most.

    Reference: arXiv:2604.09459 Section 4.2 (CARL — entropy-based)
    """

    def __init__(
        self,
        threshold: float = 0.3,
        window_size: int = 3,
    ):
        self.threshold = threshold
        self.window_size = window_size

    def identify(
        self,
        trajectory: List[Any],
        rewards: List[float],
        outcomes: Optional[List[float]] = None,
    ) -> List[int]:
        """Identify critical action indices."""
        critical_indices = []

        # Entropy-based bifurcation detection (primary method)
        critical_indices.extend(self._find_entropy_bifurcations(rewards))

        # High variance points (secondary)
        critical_indices.extend(self._find_variance_bifurcations(rewards))

        # Outcome shift points
        if outcomes:
            critical_indices.extend(self._find_outcome_shifts(outcomes))

        # Gradient changes
        critical_indices.extend(self._find_gradient_changes(rewards))

        return sorted(set(critical_indices))

    def _find_entropy_bifurcations(self, rewards: List[float]) -> List[int]:
        """Find high-entropy points where the trajectory could branch.

        Entropy is computed over a discretized reward distribution in
        each sliding window. High entropy = unpredictable outcomes =
        critical decision point.
        """
        import math

        critical = []
        if len(rewards) < 2 * self.window_size + 1:
            return critical

        for i in range(self.window_size, len(rewards) - self.window_size):
            window = rewards[i - self.window_size : i + self.window_size + 1]

            # Discretize rewards into positive/negative/neutral bins
            pos = sum(1 for r in window if r > 0.1)
            neg = sum(1 for r in window if r < -0.1)
            neu = len(window) - pos - neg
            total = len(window)

            # Compute entropy over 3-bin distribution
            entropy = 0.0
            for count in [pos, neg, neu]:
                if count > 0:
                    p = count / total
                    entropy -= p * math.log2(p)

            # Max entropy for 3 bins = log2(3) ≈ 1.585
            # Threshold: normalized entropy > threshold
            normalized_entropy = entropy / math.log2(3) if entropy > 0 else 0.0
            if normalized_entropy > self.threshold:
                critical.append(i)

        return critical

    def _find_variance_bifurcations(self, rewards: List[float]) -> List[int]:
        """Find points with high local variance."""
        critical = []
        for i in range(self.window_size, len(rewards) - self.window_size):
            window = rewards[i - self.window_size : i + self.window_size + 1]
            variance = sum((r - sum(window) / len(window)) ** 2 for r in window) / len(window)
            if variance > self.threshold:
                critical.append(i)
        return critical

    def _find_outcome_shifts(self, outcomes: List[float]) -> List[int]:
        """Find points where outcome changes direction."""
        critical = []
        for i in range(1, len(outcomes) - 1):
            before = outcomes[i - 1]
            current = outcomes[i]
            after = outcomes[i + 1]
            if (current - before) * (after - current) < 0:
                critical.append(i)
        return critical

    def _find_gradient_changes(self, rewards: List[float]) -> List[int]:
        """Find points where reward gradient changes significantly."""
        critical = []
        for i in range(2, len(rewards) - 1):
            grad_before = rewards[i - 1] - rewards[i - 2]
            grad_after = rewards[i] - rewards[i - 1]
            if abs(grad_after - grad_before) > self.threshold:
                critical.append(i)
        return critical


class CARLCreditAssigner(BaseCreditAssigner[ActionMetadata]):
    """CARL: Critical Action Refinement Learning.

    Combines GAE credit assignment with entropy-based bifurcation detection.
    Critical actions (high-entropy bifurcation points) receive boosted credit,
    while non-critical actions receive standard GAE credit. This focuses
    learning signal on the actions that mattered most.

    Reference: arXiv:2604.09459 Section 4.2 (CARL)
    """

    def __init__(self, config: Optional[CreditAssignmentConfig] = None):
        super().__init__(config)
        self._identifier = CriticalActionIdentifier(
            threshold=config.bifurcation_threshold if config else 0.3,
            window_size=3,
        )
        self._gae = EpisodeLevelCreditAssigner(config)

    def assign_credit(
        self,
        trajectory: List[ActionMetadata],
        rewards: List[float],
        config: Optional[CreditAssignmentConfig] = None,
    ) -> List[CreditSignal]:
        """Assign credit with CARL: GAE base + entropy-boosted critical actions."""
        if len(trajectory) != len(rewards):
            raise ValueError("Trajectory and rewards must have same length")

        cfg = config or self.config

        # Step 1: Standard GAE credit assignment
        gae_signals = self._gae.assign_credit(trajectory, rewards, cfg)

        # Step 2: Identify critical actions via entropy-based detection
        critical_indices = set(self._identifier.identify(trajectory, rewards))

        # Step 3: Boost critical actions, attenuate non-critical
        # CARL insight: focus gradient updates on bifurcation points
        boost_factor = 1.5
        attenuate_factor = 0.7

        signals = []
        for i, gae_signal in enumerate(gae_signals):
            is_critical = i in critical_indices
            factor = boost_factor if is_critical else attenuate_factor
            credit = gae_signal.credit * factor

            signal = CreditSignal(
                action_id=gae_signal.action_id,
                raw_reward=gae_signal.raw_reward,
                credit=credit,
                confidence=0.95 if is_critical else gae_signal.confidence,
                methodology=CreditMethodology.CARL,
                granularity=cfg.granularity,
                metadata=gae_signal.metadata,
            )
            signals.append(signal)
            self._credit_store[signal.action_id] = signal

        return signals


# ============================================================================
# LLM-as-Critic Credit Assigner
# ============================================================================


class LLMCriticCreditAssigner(BaseCreditAssigner[ActionMetadata]):
    """LLM-as-Critic: uses a cheap/fast LLM to evaluate action quality.

    Instead of mathematical value estimation, asks an LLM to score each
    action's contribution to the trajectory outcome. Falls back to GAE
    when no LLM is available.

    Uses the same provider infrastructure as GEPA (Ollama edge model by
    default — free, local, fast). The LLM receives a structured prompt
    with the action context and returns a quality score.

    Reference: arXiv:2604.09459 Section 3.3 (LLM-as-Judge / CAPO pattern)
    """

    def __init__(
        self,
        config: Optional[CreditAssignmentConfig] = None,
        provider_name: str = "ollama",
        model: str = "qwen3.5:2b",
    ):
        super().__init__(config)
        self._provider_name = provider_name
        self._model = model
        self._provider: Optional[Any] = None
        self._gae_fallback = EpisodeLevelCreditAssigner(config)

    def assign_credit(
        self,
        trajectory: List[ActionMetadata],
        rewards: List[float],
        config: Optional[CreditAssignmentConfig] = None,
    ) -> List[CreditSignal]:
        """Assign credit using LLM evaluation of each action."""
        if len(trajectory) != len(rewards):
            raise ValueError("Trajectory and rewards must have same length")

        cfg = config or self.config

        # Try LLM-based evaluation
        llm_scores = self._evaluate_with_llm(trajectory, rewards)
        if llm_scores is None:
            # Fallback to GAE if LLM unavailable
            logger.debug("LLM-as-Critic unavailable, falling back to GAE")
            return self._gae_fallback.assign_credit(trajectory, rewards, cfg)

        # Combine LLM scores with raw rewards
        signals = []
        for i, action_meta in enumerate(trajectory):
            llm_score = llm_scores[i] if i < len(llm_scores) else 0.5
            # Credit = blend of LLM evaluation and raw reward
            credit = 0.6 * llm_score + 0.4 * rewards[i]

            signal = CreditSignal(
                action_id=action_meta.action_id,
                raw_reward=rewards[i],
                credit=credit,
                confidence=0.85,  # LLM judgment has moderate confidence
                methodology=CreditMethodology.LLM_AS_CRITIC,
                granularity=cfg.granularity,
                metadata=action_meta,
            )
            signals.append(signal)
            self._credit_store[action_meta.action_id] = signal

        return signals

    def _evaluate_with_llm(
        self,
        trajectory: List[ActionMetadata],
        rewards: List[float],
    ) -> Optional[List[float]]:
        """Ask LLM to evaluate each action's contribution.

        Returns a list of scores in [0, 1] for each action, or None if
        the LLM is unavailable.
        """
        provider = self._get_provider()
        if provider is None:
            return None

        # Build a concise evaluation prompt
        total_reward = sum(rewards)
        outcome = "successful" if total_reward > 0 else "failed"

        actions_text = []
        for i, (meta, reward) in enumerate(zip(trajectory, rewards)):
            tool = meta.tool_name or meta.method_name or "unknown"
            actions_text.append(f"  {i}: tool={tool}, reward={reward:+.2f}")

        prompt = (
            f"/no_think\n"
            f"Rate each action's contribution to this {outcome} trajectory "
            f"(total reward: {total_reward:+.2f}).\n\n"
            f"Actions:\n" + "\n".join(actions_text[:20]) + "\n\n"
            "Reply with ONLY a comma-separated list of scores (0.0 to 1.0), "
            "one per action. Example: 0.8, 0.3, 0.9, 0.1"
        )

        try:
            from victor.core.async_utils import run_sync_in_thread
            from victor.providers.base import Message

            messages = [Message(role="user", content=prompt)]
            response = run_sync_in_thread(
                provider.chat(
                    messages=messages,
                    model=self._model,
                    max_tokens=200,
                    temperature=0.3,  # Low temp for consistent scoring
                ),
                timeout=15.0,
            )

            if not response or not response.content:
                return None

            # Parse scores from response
            return self._parse_scores(response.content, len(trajectory))

        except Exception as e:
            logger.debug("LLM-as-Critic evaluation failed: %s", e)
            return None

    def _parse_scores(self, response: str, expected_count: int) -> Optional[List[float]]:
        """Parse comma-separated scores from LLM response."""
        import re

        # Strip thinking artifacts
        content = response
        if "<think>" in content:
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

        # Extract numbers
        numbers = re.findall(r"(\d+\.?\d*)", content)
        if not numbers:
            return None

        scores = []
        for n in numbers[:expected_count]:
            try:
                score = float(n)
                scores.append(min(1.0, max(0.0, score)))
            except ValueError:
                scores.append(0.5)

        # Pad with 0.5 if LLM returned fewer scores than actions
        while len(scores) < expected_count:
            scores.append(0.5)

        return scores

    def _get_provider(self) -> Optional[Any]:
        """Get or create LLM provider (same pattern as GEPA)."""
        if self._provider is not None:
            return self._provider
        if not self._provider_name:
            return None
        try:
            from importlib import import_module

            from victor.providers.sandhi_transport import resolve_transport_class

            mod = import_module(f"victor.providers.{self._provider_name}_provider")
            native_cls = getattr(mod, f"{self._provider_name.title()}Provider")
            # Sandhi owns transport: resolve the typed variant so chat()/stream()
            # run via the FFI binding rather than the (removed) raw wire path.
            self._provider = resolve_transport_class(self._provider_name, native_cls, {})()
            return self._provider
        except Exception as e:
            logger.debug(
                "Failed to create %s provider for LLM-as-Critic: %s",
                self._provider_name,
                e,
            )
            return None


# ============================================================================
# Main Integration Class
# ============================================================================


class CreditAssignmentIntegration:
    """Main integration point for credit assignment in Victor.

    Provides unified interface for all credit assignment methods.
    """

    _assigners: ClassVar[Dict[CreditMethodology, type]] = {
        CreditMethodology.MONTE_CARLO: SegmentLevelCreditAssigner,
        CreditMethodology.GAE: EpisodeLevelCreditAssigner,
        CreditMethodology.TEMPORAL_DIFFERENCE: TokenLevelCreditAssigner,
        CreditMethodology.HINDSIGHT: HindsightCreditAssigner,
        CreditMethodology.SHAPLEY: MultiAgentCreditAssigner,
        CreditMethodology.N_STEP_RETURNS: TurnLevelCreditAssigner,
        CreditMethodology.C3: CounterfactualCreditAssigner,
        CreditMethodology.CARL: CARLCreditAssigner,
        CreditMethodology.LLM_AS_CRITIC: LLMCriticCreditAssigner,
    }

    def __init__(
        self,
        default_config: Optional[CreditAssignmentConfig] = None,
        state_graph: Optional["StateGraph"] = None,
    ):
        self.default_config = default_config or CreditAssignmentConfig()
        self.state_graph = state_graph
        self._active_assigners: Dict[CreditMethodology, BaseCreditAssigner] = {}
        self._trajectory_history: List[Dict[str, Any]] = []
        self._critical_identifier = CriticalActionIdentifier()

    def assign_credit(
        self,
        trajectory: List[Any],
        rewards: List[float],
        methodology: Optional[CreditMethodology] = None,
        config: Optional[CreditAssignmentConfig] = None,
    ) -> List[CreditSignal]:
        """Main entry point for credit assignment."""
        cfg = config or self.default_config
        method = methodology or cfg.methodology

        # Get or create assigner
        assigner = self._get_assigner(method, cfg)

        # Assign credit
        signals = assigner.assign_credit(trajectory, rewards, cfg)

        # Store history
        self._trajectory_history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "methodology": method.value,
                "trajectory_length": len(trajectory),
                "total_reward": sum(rewards),
            }
        )

        return signals

    # Methodologies that have no assigner implementation yet
    _UNIMPLEMENTED: ClassVar[set] = {
        CreditMethodology.ACTOR_CRITIC,
        CreditMethodology.CCPO,
        CreditMethodology.HCAPO,
    }

    def _get_assigner(
        self, methodology: CreditMethodology, config: CreditAssignmentConfig
    ) -> BaseCreditAssigner:
        """Get or create assigner for methodology."""
        if methodology in self._UNIMPLEMENTED:
            raise NotImplementedError(
                f"Credit methodology '{methodology.value}' is planned but not yet "
                f"implemented. Available methods: "
                f"{', '.join(m.value for m in self._assigners)}"
            )
        if methodology not in self._active_assigners:
            assigner_class = self._assigners.get(methodology)
            if assigner_class is None:
                raise ValueError(f"Unknown credit methodology: {methodology}")
            self._active_assigners[methodology] = assigner_class(config)

        return self._active_assigners[methodology]

    def get_credit(self, action_id: str) -> Optional[CreditSignal]:
        """Get credit for action from any active assigner."""
        for assigner in self._active_assigners.values():
            credit = assigner.get_credit(action_id)
            if credit:
                return credit
        return None

    def get_agent_attribution(
        self, agent_id: str, granularity: CreditGranularity = CreditGranularity.AGENT
    ) -> Dict[str, float]:
        """Get attribution for agent across all assigners."""
        attribution: Dict[str, float] = defaultdict(float)
        for assigner in self._active_assigners.values():
            agent_attr = assigner.get_attribution(agent_id, granularity)
            for contributor, amount in agent_attr.items():
                attribution[contributor] += amount
        return dict(attribution)

    def identify_critical_actions(self, trajectory: List[Any], rewards: List[float]) -> List[int]:
        """Identify bifurcation points in trajectory."""
        return self._critical_identifier.identify(trajectory, rewards)

    def get_trajectory_summary(self) -> Dict[str, Any]:
        """Get summary of all processed trajectories."""
        if not self._trajectory_history:
            return {"count": 0, "total_reward": 0.0}

        return {
            "count": len(self._trajectory_history),
            "total_reward": sum(t["total_reward"] for t in self._trajectory_history),
            "avg_trajectory_length": sum(t["trajectory_length"] for t in self._trajectory_history)
            / len(self._trajectory_history),
            "methodologies_used": list(set(t["methodology"] for t in self._trajectory_history)),
        }

    def reset(self) -> None:
        """Reset all state."""
        for assigner in self._active_assigners.values():
            assigner.reset()
        self._active_assigners.clear()
        self._trajectory_history.clear()


# ============================================================================
# StateGraph Integration
# ============================================================================


class StateGraphCreditMixin:
    """Mixin to add credit assignment to StateGraph."""

    @property
    def credit_assigner(self) -> Optional[CreditAssignmentIntegration]:
        """Get or create credit assignment integration."""
        if not hasattr(self, "_credit_assigner"):
            # Check if StateGraph is available
            is_state_graph = False
            try:
                from victor.framework.graph import StateGraph as SG

                is_state_graph = isinstance(self, SG)
            except ImportError:
                pass

            self._credit_assigner = CreditAssignmentIntegration(
                state_graph=self if is_state_graph else None
            )
        return self._credit_assigner

    def assign_transition_credit(
        self,
        state_transitions: List[Dict[str, Any]],
        rewards: List[float],
        methodology: Optional[CreditMethodology] = None,
    ) -> List[CreditSignal]:
        """Assign credit for StateGraph transitions."""
        # Create metadata from transitions
        trajectory = []
        for i, trans in enumerate(state_transitions):
            meta = ActionMetadata(
                agent_id=trans.get("agent_id", "graph"),
                action_id=f"transition_{i}_{trans.get('from', '_')}_{trans.get('to', '_')}",
                turn_index=trans.get("turn_index", 0),
                step_index=i,
                method_name=trans.get("action", "__edge__"),
            )
            trajectory.append(meta)

        return self.credit_assigner.assign_credit(trajectory, rewards, methodology)


# ============================================================================
# Utilities
# ============================================================================


def compute_credit_metrics(
    signals: List[CreditSignal],
) -> Dict[str, float]:
    """Compute aggregate metrics from credit signals."""
    if not signals:
        return {
            "count": 0,
            "total_credit": 0.0,
            "avg_confidence": 0.0,
            "credit_std": 0.0,
        }

    credits = [s.credit for s in signals]
    confidences = [s.confidence for s in signals]

    import statistics

    return {
        "count": len(signals),
        "total_credit": sum(credits),
        "avg_credit": sum(credits) / len(credits),
        "avg_confidence": sum(confidences) / len(confidences),
        "credit_std": statistics.stdev(credits) if len(credits) > 1 else 0.0,
        "positive_ratio": sum(1 for c in credits if c > 0) / len(credits),
    }


def visualize_credit_assignment(
    signals: List[CreditSignal],
    output_path: Optional[str] = None,
) -> str:
    """Generate ASCII visualization of credit assignment."""
    if not signals:
        return "No credit signals to visualize"

    lines = ["Credit Assignment Visualization", "=" * 50]

    for i, signal in enumerate(signals[:20]):
        confidence_bar = "█" * int(signal.confidence * 10)
        credit_str = f"{signal.credit:+.3f}"
        lines.append(
            f"{i:3d} | {signal.action_id[:20]:20s} | {credit_str:>8s} | "
            f"{signal.methodology.value if signal.methodology else 'N/A':12s} | "
            f"{confidence_bar}"
        )

    if len(signals) > 20:
        lines.append(f"... and {len(signals) - 20} more")

    result = "\n".join(lines)

    if output_path:
        with open(output_path, "w") as f:
            f.write(result)

    return result


# ============================================================================
# Exports
# ============================================================================


__all__ = [
    # Core types
    "CreditGranularity",
    "CreditMethodology",
    "ActionMetadata",
    "CreditSignal",
    "TrajectorySegment",
    "CreditAssignmentConfig",
    # Protocols
    "CreditAssignmentProvider",
    # Base classes
    "BaseCreditAssigner",
    # Reasoning RL assigners
    "TokenLevelCreditAssigner",
    "SegmentLevelCreditAssigner",
    # Agentic RL assigners
    "EpisodeLevelCreditAssigner",
    "TurnLevelCreditAssigner",
    "HindsightCreditAssigner",
    # Multi-agent
    "MultiAgentCreditAssigner",
    # Counterfactual
    "CounterfactualCreditAssigner",
    # Bifurcation (CARL)
    "CARLCreditAssigner",
    "CriticalActionIdentifier",
    # LLM-as-Critic
    "LLMCriticCreditAssigner",
    # Main integration
    "CreditAssignmentIntegration",
    "StateGraphCreditMixin",
    # Utilities
    "compute_credit_metrics",
    "visualize_credit_assignment",
]
