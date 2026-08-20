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

"""Adaptive mode controller with reinforcement learning for optimal transitions.

Architecture (State Machine + Q-Learning patterns):
- Tracks conversation state (explore, plan, build, review)
- Learns optimal mode transitions from user feedback
- Optimizes tool budgets based on task type and success patterns
- Supports continuation context management

Key Features:
1. Q-Learning for Mode Transitions:
   - State: (current_mode, task_type, tool_usage_ratio, quality_score)
   - Action: (transition_to_mode, adjust_tool_budget)
   - Reward: User satisfaction, task completion, quality scores

2. Tool Budget Optimization:
   - Learns optimal budgets per task type
   - Adjusts based on success/failure patterns
   - Respects user-defined limits

3. Continuation Management:
   - Tracks context across continuations
   - Preserves learned state between sessions
   - Intelligent context summarization

4. Real-time Adaptation:
   - Adjusts during conversation based on patterns
   - Early termination detection
   - Loop detection and recovery

Usage:
    controller = AdaptiveModeController(profile_name="local-qwen")

    # Get recommended action
    action = controller.get_recommended_action(
        current_mode="explore",
        task_type="analysis",
        tool_calls_made=5,
        tool_budget=10,
        quality_score=0.7,
    )

    # Record outcome for learning
    controller.record_outcome(
        success=True,
        quality_score=0.85,
        user_satisfied=True,
    )
"""

import hashlib
from victor.core.json_utils import json_dumps
import logging
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from victor.config.orchestrator_constants import BUDGET_LIMITS
from victor.core.schema import Tables

logger = logging.getLogger(__name__)


class AdaptiveAgentMode(Enum):
    """Extended agent operation modes for adaptive mode control.

    This is an extension of the base AgentMode (victor.agent.mode_controller.AgentMode)
    with additional modes (REVIEW, COMPLETE) for adaptive behavior tracking.

    Renamed from AgentMode to be semantically distinct:
    - AgentMode (victor.agent.mode_controller): Base 3 modes (BUILD, PLAN, EXPLORE)
    - AdaptiveAgentMode: Extended 5 modes for adaptive control
    - RLAgentMode: Extended 5 modes for RL state machine
    """

    EXPLORE = "explore"  # Understanding codebase
    PLAN = "plan"  # Creating implementation plan
    BUILD = "build"  # Writing/modifying code
    REVIEW = "review"  # Reviewing changes
    COMPLETE = "complete"  # Task finished


# Backward compatibility alias
AgentMode = AdaptiveAgentMode


class RatioLevel(str, Enum):
    """Discretized ratio bucket for adaptive mode state."""

    LOW = "low"  # < 0.25
    MID_LOW = "mid_low"  # 0.25-0.5
    MID_HIGH = "mid_high"  # 0.5-0.75
    HIGH = "high"  # > 0.75


class QualityLevel(str, Enum):
    """Discretized quality score bucket."""

    POOR = "poor"  # < 0.4
    FAIR = "fair"  # 0.4-0.6
    GOOD = "good"  # 0.6-0.8
    EXCELLENT = "excellent"  # > 0.8


class TransitionTrigger(Enum):
    """Triggers for mode transitions."""

    USER_REQUEST = "user_request"  # User explicitly requested
    TASK_COMPLETE = "task_complete"  # Task objective achieved
    BUDGET_LOW = "budget_low"  # Tool/iteration budget low
    QUALITY_THRESHOLD = "quality_threshold"  # Quality score threshold met
    PATTERN_DETECTED = "pattern_detected"  # Learned pattern suggests transition
    LOOP_DETECTED = "loop_detected"  # Repetitive behavior detected
    ERROR_RECOVERY = "error_recovery"  # Recovering from error


@dataclass
class ModeState:
    """Current state for mode decision making."""

    mode: AgentMode
    task_type: str
    tool_calls_made: int
    tool_budget: int
    iteration_count: int
    iteration_budget: int
    quality_score: float
    grounding_score: float
    time_in_mode_seconds: float
    recent_tool_success_rate: float

    def to_state_key(self) -> str:
        """Convert to discrete state key for Q-learning."""
        # Discretize continuous values
        tool_ratio = self._discretize_ratio(self.tool_calls_made / max(self.tool_budget, 1))
        iter_ratio = self._discretize_ratio(self.iteration_count / max(self.iteration_budget, 1))
        quality_bucket = self._discretize_quality(self.quality_score)
        grounding_bucket = self._discretize_quality(self.grounding_score)

        return f"{self.mode.value}:{self.task_type}:{tool_ratio.value}:{iter_ratio.value}:{quality_bucket.value}:{grounding_bucket.value}"

    def _discretize_ratio(self, ratio: float) -> RatioLevel:
        """Discretize a ratio to buckets."""
        if ratio < 0.25:
            return RatioLevel.LOW
        elif ratio < 0.5:
            return RatioLevel.MID_LOW
        elif ratio < 0.75:
            return RatioLevel.MID_HIGH
        else:
            return RatioLevel.HIGH

    def _discretize_quality(self, score: float) -> QualityLevel:
        """Discretize quality score."""
        if score < 0.4:
            return QualityLevel.POOR
        elif score < 0.6:
            return QualityLevel.FAIR
        elif score < 0.8:
            return QualityLevel.GOOD
        else:
            return QualityLevel.EXCELLENT


@dataclass
class ModeAction:
    """Action to take in the current state."""

    target_mode: AgentMode
    adjust_tool_budget: int = 0  # +/- adjustment
    should_continue: bool = True
    reason: str = ""
    confidence: float = 0.5

    def __repr__(self) -> str:
        return (
            f"ModeAction(target={self.target_mode.value}, "
            f"budget_adj={self.adjust_tool_budget:+d}, "
            f"continue={self.should_continue}, conf={self.confidence:.2f})"
        )


@dataclass
class TransitionEvent:
    """Record of a mode transition."""

    from_mode: AgentMode
    to_mode: AgentMode
    trigger: TransitionTrigger
    state_before: ModeState
    action_taken: ModeAction
    timestamp: datetime = field(default_factory=datetime.now)

    # Outcome (filled in after transition)
    outcome_success: Optional[bool] = None
    outcome_quality: Optional[float] = None
    reward: Optional[float] = None


class QLearningStore:
    """SQLite-backed Q-learning storage for mode transitions.

    Stores Q-values for state-action pairs and learns from outcomes.
    Uses consolidated project.db via ProjectDatabaseManager.
    """

    def __init__(self, project_path: Optional[Path] = None):
        """Initialize the Q-learning store.

        Args:
            project_path: Ignored - RL data is stored in global database for cross-project learning.
        """
        from victor.core.database import DatabaseManager

        self._db = DatabaseManager()
        self.db_path = self._db.db_path

        # Q-learning parameters
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_rate = 0.1  # Epsilon for epsilon-greedy

        self._ensure_tables()

    def _ensure_tables(self) -> None:
        """Migrate legacy private RL tables into unified schema (idempotent)."""
        from victor.framework.rl.migration import RLTableMigrator

        conn = self._db.get_connection()
        RLTableMigrator(conn).run_if_needed(
            "mode_transition", RLTableMigrator.migrate_mode_transition
        )

    def get_q_value(self, state_key: str, action_key: str) -> float:
        """Get Q-value for a state-action pair."""
        conn = self._db.get_connection()
        row = conn.execute(
            f"SELECT q_value FROM {Tables.RL_Q_VALUE} "
            f"WHERE learner_id = 'mode_transition' AND state_key = ? AND action_key = ?",
            (state_key, action_key),
        ).fetchone()

        return row[0] if row else 0.0

    def get_all_actions(self, state_key: str) -> Dict[str, float]:
        """Get all Q-values for a state."""
        conn = self._db.get_connection()
        rows = conn.execute(
            f"SELECT action_key, q_value FROM {Tables.RL_Q_VALUE} "
            f"WHERE learner_id = 'mode_transition' AND state_key = ?",
            (state_key,),
        ).fetchall()

        return {row[0]: row[1] for row in rows}

    def update_q_value(
        self,
        state_key: str,
        action_key: str,
        reward: float,
        next_state_key: Optional[str] = None,
    ) -> float:
        """Update Q-value using Q-learning update rule.

        Q(s,a) = Q(s,a) + α * (r + γ * max(Q(s',a')) - Q(s,a))

        Returns:
            New Q-value
        """
        conn = self._db.get_connection()

        # Get current Q-value and visit count
        row = conn.execute(
            f"SELECT q_value, visit_count FROM {Tables.RL_Q_VALUE} "
            f"WHERE learner_id = 'mode_transition' AND state_key = ? AND action_key = ?",
            (state_key, action_key),
        ).fetchone()
        current_q = row[0] if row else 0.0
        visit_count = row[1] if row else 0

        # Get max Q-value for next state (if provided)
        max_next_q = 0.0
        if next_state_key:
            next_actions = self.get_all_actions(next_state_key)
            if next_actions:
                max_next_q = max(next_actions.values())

        # Q-learning update
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )

        now = datetime.now().isoformat()
        conn.execute(
            f"INSERT OR REPLACE INTO {Tables.RL_Q_VALUE} "
            f"(learner_id, state_key, action_key, q_value, visit_count, last_updated) "
            f"VALUES ('mode_transition', ?, ?, ?, ?, ?)",
            (state_key, action_key, new_q, visit_count + 1, now),
        )
        conn.commit()

        return new_q

    def record_transition(
        self,
        profile_name: str,
        from_mode: AgentMode,
        to_mode: AgentMode,
        trigger: TransitionTrigger,
        state_key: str,
        action_key: str,
        reward: Optional[float] = None,
    ) -> None:
        """Record a transition for analysis."""
        conn = self._db.get_connection()
        conn.execute(
            f"INSERT INTO {Tables.RL_TRANSITION} "
            f"(learner_id, from_state, to_state, action, reward, metadata, created_at) "
            f"VALUES ('mode_transition', ?, ?, ?, ?, ?, ?)",
            (
                from_mode.value,
                to_mode.value,
                action_key,
                reward,
                json_dumps(
                    {
                        "trigger": trigger.value,
                        "state_key": state_key,
                        "profile_name": profile_name,
                    }
                ),
                datetime.now().isoformat(),
            ),
        )
        conn.commit()

    def update_task_stats(
        self,
        task_type: str,
        tool_budget_used: int,
        quality_score: float,
        completed: bool,
        tool_budget_total: int = 0,
        budget_exhausted: bool = False,
    ) -> None:
        """Update task-type statistics with outcome-aware budget learning.

        The budget learning considers both success/failure and efficiency:
        - Success with few tools → gradual decrease (efficient)
        - Failure or low quality → gradual increase (needs more resources)
        - Budget exhausted with failure → stronger increase signal

        Args:
            task_type: Type of task
            tool_budget_used: Number of tool calls made
            quality_score: Quality score (0-1)
            completed: Whether task completed successfully
            tool_budget_total: Total budget that was available
            budget_exhausted: Whether budget was exhausted
        """
        conn = self._db.get_connection()

        # Load current stats from rl_task_stat (multi-row per task_type)
        rows = conn.execute(
            f"SELECT stat_key, stat_value, sample_count FROM {Tables.RL_TASK_STAT} "
            f"WHERE learner_id = 'mode_transition' AND task_type = ?",
            (task_type,),
        ).fetchall()

        if rows:
            stat_map = {r[0]: r[1] for r in rows}
            count = int(rows[0][2]) + 1  # sample_count lives on each row; take from first
            current_budget = int(stat_map.get("optimal_tool_budget", 10))
            current_quality = float(stat_map.get("avg_quality_score", 0.5))
            current_completion = float(stat_map.get("avg_completion_rate", 0.5))

            # Outcome-aware budget adjustment
            base_alpha = 0.05

            if completed and quality_score >= 0.7:
                if tool_budget_used < current_budget * 0.5:
                    target_budget = max(tool_budget_used + 3, current_budget - 2)
                    new_budget = int((1 - base_alpha) * current_budget + base_alpha * target_budget)
                else:
                    alpha = base_alpha * 0.5
                    new_budget = int((1 - alpha) * current_budget + alpha * tool_budget_used)
            else:
                if budget_exhausted:
                    increase = max(5, int(current_budget * 0.2))
                    new_budget = min(current_budget + increase, 100)
                    logger.debug(
                        f"[QLearningStore] Budget exhaustion failure for {task_type}: "
                        f"{current_budget} -> {new_budget}"
                    )
                elif quality_score < 0.5:
                    increase = max(2, int(current_budget * 0.1))
                    new_budget = min(current_budget + increase, 100)
                else:
                    new_budget = min(current_budget + 1, 100)

            min_budget = self._get_min_budget_for_task(task_type)
            new_budget = max(min_budget, new_budget)

            alpha = 0.1
            new_quality = (1 - alpha) * current_quality + alpha * quality_score
            completion_rate = (1 - alpha) * current_completion + alpha * (1.0 if completed else 0.0)
        else:
            count = 1
            new_budget = max(tool_budget_used + 5, self._get_min_budget_for_task(task_type))
            new_quality = quality_score
            completion_rate = 1.0 if completed else 0.0

        now = datetime.now().isoformat()
        for stat_key, stat_value in (
            ("optimal_tool_budget", float(new_budget)),
            ("avg_quality_score", new_quality),
            ("avg_completion_rate", completion_rate),
        ):
            conn.execute(
                f"INSERT OR REPLACE INTO {Tables.RL_TASK_STAT} "
                f"(learner_id, task_type, stat_key, stat_value, sample_count, updated_at) "
                f"VALUES ('mode_transition', ?, ?, ?, ?, ?)",
                (task_type, stat_key, stat_value, count, now),
            )
        conn.commit()

    def _get_min_budget_for_task(self, task_type: str) -> int:
        """Get minimum viable budget for a task type.

        This prevents learned budgets from dropping below usable thresholds.
        """
        # Minimum budgets by task type - these are floors, not targets
        min_budgets = {
            "code_generation": 8,
            "create_simple": 5,
            "create": 10,
            "edit": 10,
            "search": 8,
            "action": 15,
            "analysis_deep": 20,
            "analyze": 15,
            "design": 25,
            "general": 15,
        }
        return min_budgets.get(task_type, 10)

    def get_task_stats(self, task_type: str) -> Dict[str, Any]:
        """Get statistics for a task type."""
        conn = self._db.get_connection()
        rows = conn.execute(
            f"SELECT stat_key, stat_value, sample_count FROM {Tables.RL_TASK_STAT} "
            f"WHERE learner_id = 'mode_transition' AND task_type = ?",
            (task_type,),
        ).fetchall()

        result: Dict[str, Any] = {
            "task_type": task_type,
            "optimal_tool_budget": 10,
            "avg_quality_score": 0.5,
            "avg_completion_rate": 0.5,
            "sample_count": 0,
        }
        for stat_key, stat_value, sample_count in rows:
            if stat_key in result:
                result[stat_key] = stat_value
            result["sample_count"] = sample_count
        return result


class AdaptiveModeController:
    """Controller for adaptive mode management with reinforcement learning.

    Uses Q-learning to optimize:
    - Mode transitions (explore → plan → build → review)
    - Tool budget allocation
    - Continuation decisions
    """

    # Valid mode transitions
    VALID_TRANSITIONS = {
        AgentMode.EXPLORE: [AgentMode.PLAN, AgentMode.BUILD, AgentMode.COMPLETE],
        AgentMode.PLAN: [AgentMode.BUILD, AgentMode.EXPLORE, AgentMode.COMPLETE],
        AgentMode.BUILD: [AgentMode.REVIEW, AgentMode.EXPLORE, AgentMode.COMPLETE],
        AgentMode.REVIEW: [AgentMode.BUILD, AgentMode.COMPLETE],
        AgentMode.COMPLETE: [],
    }

    # Default tool budgets by task type.
    #
    # Aligned with BUDGET_LIMITS / COMPLEXITY_BUDGETS so the RL-recommended budget
    # path cannot collapse a real task to a handful of calls. These are the *defaults*
    # consulted by get_optimal_tool_budget() before any learned override; previously
    # they were tiny vestigial values (analyze=8, analysis_deep=15) that contradicted
    # the framework-wide generous-exploration policy and could prematurely stop tasks.
    DEFAULT_TOOL_BUDGETS = {
        "code_generation": BUDGET_LIMITS.simple_task,  # 20
        "create_simple": BUDGET_LIMITS.simple_task,  # 20
        "create": BUDGET_LIMITS.medium_task,  # 50
        "edit": BUDGET_LIMITS.medium_task,  # 50
        "search": BUDGET_LIMITS.medium_task,  # 50
        "action": BUDGET_LIMITS.action_task,  # 200
        "analysis_deep": BUDGET_LIMITS.analysis_task,  # 500
        "analyze": BUDGET_LIMITS.complex_task,  # 100
        "design": BUDGET_LIMITS.complex_task,  # 100
        "general": BUDGET_LIMITS.medium_task,  # 50
    }

    # Provider-aware iteration thresholds for loop detection
    # Reasoning models (DeepSeek, o1) need more iterations before being considered stuck
    # Efficient models (Claude, GPT-4, Grok) can be detected as stuck earlier
    PROVIDER_ITERATION_THRESHOLDS = {
        "deepseek": {"min_iterations_before_loop": 5, "no_tool_threshold": 3},
        "anthropic": {"min_iterations_before_loop": 3, "no_tool_threshold": 2},
        "openai": {"min_iterations_before_loop": 3, "no_tool_threshold": 2},
        "xai": {"min_iterations_before_loop": 3, "no_tool_threshold": 2},
        "google": {"min_iterations_before_loop": 4, "no_tool_threshold": 2},
        "ollama": {"min_iterations_before_loop": 4, "no_tool_threshold": 3},
        "default": {"min_iterations_before_loop": 3, "no_tool_threshold": 2},
    }

    # Provider-aware quality thresholds
    # Reasoning models may produce lower scores due to verbose explanations
    PROVIDER_QUALITY_THRESHOLDS = {
        "deepseek": {"min_quality": 0.65, "grounding_threshold": 0.60},
        "anthropic": {"min_quality": 0.75, "grounding_threshold": 0.70},
        "openai": {"min_quality": 0.75, "grounding_threshold": 0.70},
        "xai": {"min_quality": 0.75, "grounding_threshold": 0.70},
        "google": {"min_quality": 0.70, "grounding_threshold": 0.65},
        "ollama": {"min_quality": 0.65, "grounding_threshold": 0.60},
        "default": {"min_quality": 0.70, "grounding_threshold": 0.65},
    }

    def __init__(
        self,
        profile_name: str = "default",
        q_store: Optional[QLearningStore] = None,
        provider_name: Optional[str] = None,
        model_name: Optional[str] = None,
        provider_adapter: Optional[Any] = None,
        mode_transition_learner: Optional[Any] = None,
    ):
        """Initialize the adaptive mode controller.

        Args:
            profile_name: Profile name for tracking
            q_store: Optional Q-learning store (creates one if not provided)
            provider_name: Provider name for provider-aware thresholds (e.g., "deepseek", "anthropic")
            model_name: Model name for RL tracking (e.g., "deepseek-chat", "gpt-4")
            provider_adapter: Optional provider adapter for capability-based thresholds
            mode_transition_learner: Optional ModeTransitionLearner for unified RL tracking
        """
        self.profile_name = profile_name
        self._q_store = q_store or QLearningStore()
        self._provider_name = self._normalize_provider_name(provider_name)
        self._model_name = model_name or profile_name  # Fall back to profile name
        self._provider_adapter = provider_adapter
        self._mode_transition_learner = mode_transition_learner

        # Current state tracking
        self._current_state: Optional[ModeState] = None
        self._current_action: Optional[ModeAction] = None
        self._pending_transition: Optional[TransitionEvent] = None

        # Session tracking
        self._session_start = datetime.now()
        self._mode_history: List[Tuple[AgentMode, datetime]] = []
        self._total_reward = 0.0

        # Consecutive no-tool iterations tracking for loop detection
        self._no_tool_iterations = 0

        if mode_transition_learner:
            logger.info("RL: AdaptiveModeController using unified ModeTransitionLearner")

    def _normalize_provider_name(self, provider_name: Optional[str]) -> str:
        """Normalize provider name for threshold lookup.

        Args:
            provider_name: Raw provider name (may include model info)

        Returns:
            Normalized provider key for threshold lookup
        """
        if not provider_name:
            return "default"

        # Extract base provider from full names like "deepseek:deepseek-chat"
        provider = provider_name.lower().split(":")[0].strip()

        # Map common variations
        provider_map = {
            "deepseek": "deepseek",
            "anthropic": "anthropic",
            "claude": "anthropic",
            "openai": "openai",
            "gpt": "openai",
            "xai": "xai",
            "grok": "xai",
            "google": "google",
            "gemini": "google",
            "ollama": "ollama",
            "lmstudio": "ollama",  # Treat LMStudio similar to Ollama
            "vllm": "ollama",
        }

        return provider_map.get(provider, "default")

    def get_iteration_thresholds(self) -> Dict[str, int]:
        """Get provider-specific iteration thresholds for loop detection.

        Returns:
            Dict with 'min_iterations_before_loop' and 'no_tool_threshold'
        """
        return self.PROVIDER_ITERATION_THRESHOLDS.get(
            self._provider_name, self.PROVIDER_ITERATION_THRESHOLDS["default"]
        )

    def get_quality_thresholds(self) -> Dict[str, float]:
        """Get provider-specific quality thresholds.

        Returns:
            Dict with 'min_quality' and 'grounding_threshold'
        """
        # Use provider adapter's capabilities if available
        if self._provider_adapter:
            caps = self._provider_adapter.capabilities
            return {
                "min_quality": caps.quality_threshold,
                "grounding_threshold": caps.grounding_strictness,
            }

        # Fall back to hardcoded provider-specific thresholds
        return self.PROVIDER_QUALITY_THRESHOLDS.get(
            self._provider_name, self.PROVIDER_QUALITY_THRESHOLDS["default"]
        )

    def set_provider(self, provider_name: str) -> None:
        """Update the provider for threshold selection.

        Args:
            provider_name: New provider name
        """
        self._provider_name = self._normalize_provider_name(provider_name)
        logger.debug(f"[AdaptiveModeController] Provider set to: {self._provider_name}")

    def get_recommended_action(
        self,
        current_mode: str,
        task_type: str,
        tool_calls_made: int,
        tool_budget: int,
        iteration_count: int = 0,
        iteration_budget: int = 20,
        quality_score: float = 0.5,
        grounding_score: float = 0.5,
        time_in_mode_seconds: float = 0.0,
        recent_tool_success_rate: float = 1.0,
    ) -> ModeAction:
        """Get recommended action based on current state.

        Uses epsilon-greedy policy: explore with probability epsilon,
        otherwise exploit best known action.

        Args:
            current_mode: Current agent mode
            task_type: Task type being processed
            tool_calls_made: Number of tool calls made
            tool_budget: Total tool call budget
            iteration_count: Current iteration count
            iteration_budget: Total iteration budget
            quality_score: Current quality score (0-1)
            grounding_score: Grounding verification score (0-1)
            time_in_mode_seconds: Time spent in current mode
            recent_tool_success_rate: Recent tool call success rate

        Returns:
            Recommended ModeAction
        """
        # Build current state
        # Ensure current_mode is a string (could be int from ConversationStage.value)
        try:
            mode_str = str(current_mode).lower() if current_mode else "explore"
            mode_enum = AgentMode(mode_str)
        except ValueError:
            mode_enum = AgentMode.EXPLORE

        state = ModeState(
            mode=mode_enum,
            task_type=task_type,
            tool_calls_made=tool_calls_made,
            tool_budget=tool_budget,
            iteration_count=iteration_count,
            iteration_budget=iteration_budget,
            quality_score=quality_score,
            grounding_score=grounding_score,
            time_in_mode_seconds=time_in_mode_seconds,
            recent_tool_success_rate=recent_tool_success_rate,
        )

        self._current_state = state
        state_key = state.to_state_key()

        # Get possible actions
        possible_actions = self._get_possible_actions(state)

        if not possible_actions:
            # No valid actions - stay in current mode
            action = ModeAction(
                target_mode=mode_enum,
                should_continue=True,
                reason="No valid transitions available",
                confidence=1.0,
            )
            self._current_action = action
            return action

        # Epsilon-greedy selection
        if random.random() < self._q_store.exploration_rate:
            # Explore: random action
            action = random.choice(possible_actions)
            action.reason = f"Exploring (ε={self._q_store.exploration_rate:.2f})"
        else:
            # Exploit: best known action
            action = self._select_best_action(state_key, possible_actions)

        self._current_action = action

        # Create pending transition event
        if action.target_mode != mode_enum:
            self._pending_transition = TransitionEvent(
                from_mode=mode_enum,
                to_mode=action.target_mode,
                trigger=self._infer_trigger(state, action),
                state_before=state,
                action_taken=action,
            )

        logger.debug(f"[AdaptiveModeController] State: {state_key}, " f"Recommended: {action}")

        return action

    def _get_possible_actions(self, state: ModeState) -> List[ModeAction]:
        """Get all possible actions from current state."""
        actions = []
        current_mode = state.mode

        # Stay in current mode (with budget adjustments)
        for budget_adj in [-2, 0, 2, 5]:
            actions.append(
                ModeAction(
                    target_mode=current_mode,
                    adjust_tool_budget=budget_adj,
                    should_continue=True,
                    confidence=0.5,
                )
            )

        # Transitions to valid modes
        for target_mode in self.VALID_TRANSITIONS.get(current_mode, []):
            actions.append(
                ModeAction(
                    target_mode=target_mode,
                    adjust_tool_budget=0,
                    should_continue=target_mode != AgentMode.COMPLETE,
                    confidence=0.5,
                )
            )

        # Check for termination conditions
        if state.quality_score > 0.8 and state.tool_calls_made > 2:
            actions.append(
                ModeAction(
                    target_mode=AgentMode.COMPLETE,
                    should_continue=False,
                    reason="High quality achieved",
                    confidence=0.8,
                )
            )

        if state.tool_calls_made >= state.tool_budget:
            actions.append(
                ModeAction(
                    target_mode=AgentMode.COMPLETE,
                    should_continue=False,
                    reason="Tool budget exhausted",
                    confidence=0.9,
                )
            )

        return actions

    def _select_best_action(
        self,
        state_key: str,
        possible_actions: List[ModeAction],
    ) -> ModeAction:
        """Select best action using Q-values."""
        q_values = self._q_store.get_all_actions(state_key)

        best_action = possible_actions[0]
        best_q = float("-inf")

        for action in possible_actions:
            action_key = self._action_to_key(action)
            q_value = q_values.get(action_key, 0.0)

            # Add heuristic bonuses
            q_value += self._get_heuristic_bonus(action)

            if q_value > best_q:
                best_q = q_value
                best_action = action

        best_action.confidence = self._q_to_confidence(best_q)
        best_action.reason = f"Best Q-value: {best_q:.2f}"

        return best_action

    def _action_to_key(self, action: ModeAction) -> str:
        """Convert action to string key."""
        return f"{action.target_mode.value}:{action.adjust_tool_budget:+d}"

    def _get_heuristic_bonus(self, action: ModeAction) -> float:
        """Get heuristic bonus for an action."""
        bonus = 0.0

        # Bonus for completing when quality is high
        if action.target_mode == AgentMode.COMPLETE:
            if self._current_state and self._current_state.quality_score > 0.8:
                bonus += 0.3

        # Penalty for unnecessary mode switches
        if self._current_state and action.target_mode != self._current_state.mode:
            if self._current_state.tool_calls_made < 2:
                bonus -= 0.2  # Don't switch too early

        return bonus

    def _q_to_confidence(self, q_value: float) -> float:
        """Convert Q-value to confidence score (0-1)."""
        # Sigmoid function to map Q-value to [0, 1]
        return 1.0 / (1.0 + math.exp(-q_value))

    def _infer_trigger(self, state: ModeState, action: ModeAction) -> TransitionTrigger:
        """Infer the trigger for a transition."""
        if action.target_mode == AgentMode.COMPLETE:
            if state.quality_score > 0.8:
                return TransitionTrigger.QUALITY_THRESHOLD
            if state.tool_calls_made >= state.tool_budget:
                return TransitionTrigger.BUDGET_LOW

        if state.recent_tool_success_rate < 0.5:
            return TransitionTrigger.ERROR_RECOVERY

        return TransitionTrigger.PATTERN_DETECTED

    def record_outcome(
        self,
        success: bool,
        quality_score: float,
        user_satisfied: bool = True,
        completed: bool = False,
    ) -> float:
        """Record outcome and update Q-values.

        Call this after the action has been executed to learn from the outcome.

        Args:
            success: Whether the action was successful
            quality_score: Quality score of the result
            user_satisfied: Whether user indicated satisfaction
            completed: Whether the task was completed

        Returns:
            Reward value used for learning
        """
        if not self._current_state or not self._current_action:
            return 0.0

        # Calculate reward
        reward = self._calculate_reward(
            success=success,
            quality_score=quality_score,
            user_satisfied=user_satisfied,
            completed=completed,
        )

        self._total_reward += reward

        # Update Q-value
        state_key = self._current_state.to_state_key()
        action_key = self._action_to_key(self._current_action)

        self._q_store.update_q_value(state_key, action_key, reward)

        # Record transition if there was one
        if self._pending_transition:
            self._pending_transition.outcome_success = success
            self._pending_transition.outcome_quality = quality_score
            self._pending_transition.reward = reward

            self._q_store.record_transition(
                profile_name=self.profile_name,
                from_mode=self._pending_transition.from_mode,
                to_mode=self._pending_transition.to_mode,
                trigger=self._pending_transition.trigger,
                state_key=state_key,
                action_key=action_key,
                reward=reward,
            )

            self._mode_history.append(
                (
                    self._pending_transition.to_mode,
                    datetime.now(),
                )
            )

            self._pending_transition = None

        # Update task stats with outcome-aware budget learning
        # Detect if budget was exhausted (used >= budget)
        budget_exhausted = self._current_state.tool_calls_made >= self._current_state.tool_budget

        self._q_store.update_task_stats(
            task_type=self._current_state.task_type,
            tool_budget_used=self._current_state.tool_calls_made,
            quality_score=quality_score,
            completed=completed,
            tool_budget_total=self._current_state.tool_budget,
            budget_exhausted=budget_exhausted
            and not success,  # Only counts as exhaustion failure if not successful
        )

        # Record to unified ModeTransitionLearner for centralized observability
        self._record_to_rl_learner(
            success=success,
            quality_score=quality_score,
            completed=completed,
            state_key=state_key,
            action_key=action_key,
            reward=reward,
        )

        logger.debug(
            f"[AdaptiveModeController] Recorded outcome: "
            f"success={success}, quality={quality_score:.2f}, reward={reward:.2f}, "
            f"budget_exhausted={budget_exhausted}"
        )

        return reward

    def _record_to_rl_learner(
        self,
        success: bool,
        quality_score: float,
        completed: bool,
        state_key: str,
        action_key: str,
        reward: float,
    ) -> None:
        """Record outcome to unified ModeTransitionLearner for centralized RL tracking.

        Args:
            success: Whether action was successful
            quality_score: Quality score of result
            completed: Whether task was completed
            state_key: State key for Q-learning
            action_key: Action key for Q-learning
            reward: Calculated reward value
        """
        if not self._mode_transition_learner:
            return

        try:
            from victor.framework.rl.base import RLOutcome

            # Build outcome for the learner
            from_mode = (
                self._pending_transition.from_mode.value
                if self._pending_transition
                else (self._current_state.mode.value if self._current_state else "explore")
            )
            to_mode = (
                self._pending_transition.to_mode.value
                if self._pending_transition
                else (self._current_action.target_mode.value if self._current_action else "explore")
            )

            outcome = RLOutcome(
                provider=self._provider_name or "unknown",
                model=self._model_name or "unknown",
                success=success,
                quality_score=quality_score,
                task_type=(self._current_state.task_type if self._current_state else "general"),
                metadata={
                    "from_mode": from_mode,
                    "to_mode": to_mode,
                    "state_key": state_key,
                    "action_key": action_key,
                    "task_completed": completed,
                    "tool_budget_used": (
                        self._current_state.tool_calls_made if self._current_state else 0
                    ),
                    "tool_budget_total": (
                        self._current_state.tool_budget if self._current_state else 10
                    ),
                },
            )

            self._mode_transition_learner.record_outcome(outcome)
            logger.debug(f"RL: Recorded mode transition to unified learner: {from_mode}→{to_mode}")

        except Exception as e:
            logger.warning(f"RL: Failed to record to ModeTransitionLearner: {e}")

    def _calculate_reward(
        self,
        success: bool,
        quality_score: float,
        user_satisfied: bool,
        completed: bool,
    ) -> float:
        """Calculate reward for reinforcement learning.

        Reward components:
        - Success: +1.0 if successful, -0.5 if failed
        - Quality: +quality_score (0-1)
        - User satisfaction: +0.5 if satisfied
        - Completion: +0.3 if completed task
        - Efficiency: Bonus for using fewer resources
        """
        reward = 0.0

        # Success component
        reward += 1.0 if success else -0.5

        # Quality component
        reward += quality_score

        # User satisfaction
        if user_satisfied:
            reward += 0.5

        # Completion bonus
        if completed:
            reward += 0.3

        # Efficiency bonus (less tool calls = more efficient)
        if self._current_state:
            budget_ratio = self._current_state.tool_calls_made / max(
                self._current_state.tool_budget, 1
            )
            if budget_ratio < 0.5 and success:
                reward += 0.2  # Efficient completion

        return reward

    def get_optimal_tool_budget(self, task_type: str) -> int:
        """Get learned optimal tool budget for a task type.

        Consults both local QLearningStore and unified ModeTransitionLearner,
        preferring learner's recommendation if it has more samples.
        """
        # Get from local store
        local_stats = self._q_store.get_task_stats(task_type)
        local_budget = local_stats.get(
            "optimal_tool_budget", self.DEFAULT_TOOL_BUDGETS.get(task_type, 10)
        )
        local_samples = local_stats.get("sample_count", 0)

        # Try unified learner if available
        if self._mode_transition_learner:
            try:
                learner_budget = self._mode_transition_learner.get_optimal_budget(task_type)
                learner_stats = self._mode_transition_learner.get_task_stats(task_type)
                learner_samples = learner_stats.get("sample_count", 0)

                # Prefer source with more samples
                if learner_samples > local_samples:
                    logger.debug(
                        f"RL: Using ModeTransitionLearner budget ({learner_budget}) "
                        f"over local ({local_budget}) for {task_type}"
                    )
                    return learner_budget

            except Exception as e:
                logger.debug(f"RL: Could not get budget from learner: {e}")

        return local_budget

    def should_continue(
        self,
        tool_calls_made: int,
        tool_budget: int,
        quality_score: float,
        iteration_count: int,
        iteration_budget: int,
        current_tool_calls: int = 0,
    ) -> Tuple[bool, str]:
        """Determine if the agent should continue or stop.

        Uses provider-aware thresholds for better loop detection.

        Args:
            tool_calls_made: Total tool calls made in session
            tool_budget: Tool budget limit
            quality_score: Current quality score (0-1)
            iteration_count: Current iteration count
            iteration_budget: Iteration budget limit
            current_tool_calls: Tool calls made in current iteration (for loop detection)

        Returns:
            Tuple of (should_continue, reason)
        """
        # Get provider-specific thresholds
        _iter_thresholds = self.get_iteration_thresholds()  # noqa: F841
        quality_thresholds = self.get_quality_thresholds()

        # Hard limits
        if tool_calls_made >= tool_budget:
            return False, "Tool budget exhausted"

        if iteration_count >= iteration_budget:
            return False, "Iteration budget exhausted"

        # Provider-aware quality threshold
        quality_threshold = quality_thresholds.get("min_quality", 0.70)
        if quality_score > quality_threshold + 0.15:  # High quality achieved
            return (
                False,
                f"High quality achieved ({quality_score:.2f} > {quality_threshold + 0.15:.2f})",
            )

        # Provider-aware loop detection
        loop_detected, loop_reason = self.check_loop_detection(
            iteration_count=iteration_count,
            current_tool_calls=current_tool_calls,
        )
        if loop_detected:
            return False, loop_reason

        # Check for patterns suggesting completion
        if self._current_state:
            state_key = self._current_state.to_state_key()
            q_values = self._q_store.get_all_actions(state_key)

            # If completing has highest Q-value, suggest stopping
            complete_key = f"{AgentMode.COMPLETE.value}:+0"
            if complete_key in q_values:
                if all(q_values[complete_key] >= v for v in q_values.values()):
                    return False, "Learned pattern suggests completion"

        return True, "Continue processing"

    def check_loop_detection(
        self,
        iteration_count: int,
        current_tool_calls: int,
    ) -> Tuple[bool, str]:
        """Check for stuck continuation loops using provider-aware thresholds.

        Args:
            iteration_count: Current iteration count
            current_tool_calls: Tool calls made in current iteration

        Returns:
            Tuple of (is_stuck, reason)
        """
        iter_thresholds = self.get_iteration_thresholds()
        min_iterations = iter_thresholds.get("min_iterations_before_loop", 3)
        no_tool_threshold = iter_thresholds.get("no_tool_threshold", 2)

        # Track consecutive no-tool iterations
        if current_tool_calls == 0:
            self._no_tool_iterations += 1
        else:
            self._no_tool_iterations = 0

        # Check if stuck: past minimum iterations and no progress
        if iteration_count >= min_iterations and self._no_tool_iterations >= no_tool_threshold:
            logger.warning(
                f"[AdaptiveModeController] Loop detected for {self._provider_name}: "
                f"{self._no_tool_iterations} consecutive iterations with no tool calls "
                f"(threshold: {no_tool_threshold}, provider: {self._provider_name})"
            )
            return True, (
                f"Stuck continuation loop detected: {self._no_tool_iterations} iterations "
                f"with no tool calls (provider threshold: {no_tool_threshold})"
            )

        return False, ""

    def reset_loop_tracking(self) -> None:
        """Reset loop detection tracking for a new conversation."""
        self._no_tool_iterations = 0

    def get_session_stats(self) -> Dict[str, Any]:
        """Get statistics for the current session."""
        session_duration = (datetime.now() - self._session_start).total_seconds()

        return {
            "profile_name": self.profile_name,
            "session_duration_seconds": session_duration,
            "total_reward": self._total_reward,
            "mode_transitions": len(self._mode_history),
            "modes_visited": [m.value for m, _ in self._mode_history],
            "exploration_rate": self._q_store.exploration_rate,
        }

    def reset_session(self) -> None:
        """Reset session tracking."""
        self._session_start = datetime.now()
        self._mode_history = []
        self._total_reward = 0.0
        self._current_state = None
        self._current_action = None
        self._pending_transition = None

    def adjust_exploration_rate(self, new_rate: float) -> None:
        """Adjust the exploration rate for Q-learning."""
        self._q_store.exploration_rate = max(0.0, min(1.0, new_rate))

    def decay_exploration_rate(self, decay_factor: float = 0.99) -> None:
        """Decay exploration rate over time."""
        self._q_store.exploration_rate *= decay_factor


# Module-level convenience function
def get_mode_controller(profile_name: str = "default") -> AdaptiveModeController:
    """Get or create a mode controller for a profile.

    Args:
        profile_name: Profile name

    Returns:
        AdaptiveModeController instance
    """
    return AdaptiveModeController(profile_name=profile_name)
