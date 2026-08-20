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

"""Team composition learner using Q-learning.

This module provides TeamCompositionLearner, which uses Q-learning to
discover optimal team compositions for different task types.

The learner tracks:
- Best formation per task type (sequential vs parallel vs hierarchical)
- Optimal role distributions (researchers vs executors vs reviewers)
- Budget allocation patterns that maximize success and quality

Database:
    Uses the unified database at ~/.victor/victor.db via victor.core.database.
"""

from __future__ import annotations

from victor.core.json_utils import json_dumps, json_loads
import logging
import random
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from victor.agent.subagents import SubAgentRole

# Import canonical types from victor.teams.types
from victor.teams.types import TeamAgentCategory, TeamConfig, TeamFormation, TeamMember
from victor.agent.teams.metrics import (
    TaskCategory,
    TeamMetrics,
    CompositionStats,
    categorize_task,
)
from victor.core.database import get_database
from victor.core.schema import Tables

logger = logging.getLogger(__name__)


@dataclass
class TeamRecommendation:
    """Recommendation for team composition.

    Attributes:
        formation: Recommended formation
        role_distribution: Recommended role counts
        suggested_budget: Suggested total tool budget
        confidence: Confidence in recommendation 0.0-1.0
        reason: Explanation for recommendation
        sample_size: Number of past executions informing this
        is_baseline: Whether this is a default baseline
    """

    formation: TeamFormation
    role_distribution: Dict[str, int]
    suggested_budget: int
    confidence: float
    reason: str
    sample_size: int = 0
    is_baseline: bool = False

    def to_team_config(
        self,
        name: str,
        goal: str,
        member_goals: Optional[Dict[str, str]] = None,
    ) -> TeamConfig:
        """Convert recommendation to TeamConfig.

        Args:
            name: Team name
            goal: Team goal
            member_goals: Optional goals per role

        Returns:
            TeamConfig based on recommendation
        """
        members = []
        member_goals = member_goals or {}

        # Budget per member
        total_members = sum(self.role_distribution.values())
        budget_per_member = self.suggested_budget // max(total_members, 1)

        role_map = {
            "researcher": SubAgentRole.RESEARCHER,
            "planner": SubAgentRole.PLANNER,
            "executor": SubAgentRole.EXECUTOR,
            "reviewer": SubAgentRole.REVIEWER,
            "tester": SubAgentRole.TESTER,
        }

        member_idx = 0
        for role_name, count in self.role_distribution.items():
            role = role_map.get(role_name, SubAgentRole.EXECUTOR)
            role_goal = member_goals.get(role_name, f"Handle {role_name} tasks")

            for i in range(count):
                member_idx += 1
                members.append(
                    TeamMember(
                        id=f"{role_name}_{i + 1}",
                        role=role,
                        name=f"{role_name.title()} {i + 1}",
                        goal=role_goal,
                        tool_budget=budget_per_member,
                        agent_category=(
                            TeamAgentCategory.SUPERVISOR
                            if self.formation == TeamFormation.HIERARCHICAL and member_idx == 1
                            else TeamAgentCategory.SPECIALIST
                        ),
                        is_manager=(
                            self.formation == TeamFormation.HIERARCHICAL and member_idx == 1
                        ),
                    )
                )

        return TeamConfig(
            name=name,
            goal=goal,
            members=members,
            formation=self.formation,
            total_tool_budget=self.suggested_budget,
        )


# Default compositions for cold start
DEFAULT_COMPOSITIONS: Dict[TaskCategory, TeamRecommendation] = {
    TaskCategory.EXPLORATION: TeamRecommendation(
        formation=TeamFormation.PARALLEL,
        role_distribution={"researcher": 2},
        suggested_budget=30,
        confidence=0.5,
        reason="Parallel research for broad exploration",
        is_baseline=True,
    ),
    TaskCategory.IMPLEMENTATION: TeamRecommendation(
        formation=TeamFormation.SEQUENTIAL,
        role_distribution={"planner": 1, "executor": 1},
        suggested_budget=40,
        confidence=0.5,
        reason="Plan then execute for implementation",
        is_baseline=True,
    ),
    TaskCategory.REVIEW: TeamRecommendation(
        formation=TeamFormation.SEQUENTIAL,
        role_distribution={"researcher": 1, "reviewer": 1},
        suggested_budget=25,
        confidence=0.5,
        reason="Research context then review",
        is_baseline=True,
    ),
    TaskCategory.TESTING: TeamRecommendation(
        formation=TeamFormation.PIPELINE,
        role_distribution={"researcher": 1, "tester": 1},
        suggested_budget=30,
        confidence=0.5,
        reason="Research code then write tests",
        is_baseline=True,
    ),
    TaskCategory.REFACTORING: TeamRecommendation(
        formation=TeamFormation.SEQUENTIAL,
        role_distribution={"researcher": 1, "planner": 1, "executor": 1},
        suggested_budget=50,
        confidence=0.5,
        reason="Understand, plan, then refactor",
        is_baseline=True,
    ),
    TaskCategory.DOCUMENTATION: TeamRecommendation(
        formation=TeamFormation.SEQUENTIAL,
        role_distribution={"researcher": 1, "executor": 1},
        suggested_budget=25,
        confidence=0.5,
        reason="Research code then document",
        is_baseline=True,
    ),
    TaskCategory.DEBUGGING: TeamRecommendation(
        formation=TeamFormation.HIERARCHICAL,
        role_distribution={"researcher": 2, "executor": 1},
        suggested_budget=45,
        confidence=0.5,
        reason="Parallel investigation with central coordinator",
        is_baseline=True,
    ),
    TaskCategory.PLANNING: TeamRecommendation(
        formation=TeamFormation.SEQUENTIAL,
        role_distribution={"researcher": 1, "planner": 1},
        suggested_budget=25,
        confidence=0.5,
        reason="Research then plan",
        is_baseline=True,
    ),
    TaskCategory.MIXED: TeamRecommendation(
        formation=TeamFormation.HIERARCHICAL,
        role_distribution={"researcher": 1, "executor": 1, "reviewer": 1},
        suggested_budget=45,
        confidence=0.5,
        reason="Balanced team for mixed tasks",
        is_baseline=True,
    ),
}


class TeamCompositionLearner:
    """Q-learning based team composition optimizer.

    Learns optimal team compositions by tracking outcomes and using
    Q-learning to update composition values.

    The learner maintains:
    - Q-values for each (task_category, composition) pair
    - Statistics for computing confidence
    - Exploration/exploitation balance via epsilon-greedy

    Example:
        learner = TeamCompositionLearner()

        # Get recommendation for a task
        rec = learner.suggest_team(
            task_type="exploration",
            context={"complexity": "high"}
        )

        # Execute team...
        result = await coordinator.execute_team(rec.to_team_config(...))

        # Record outcome for learning
        learner.record_outcome(
            task_type="exploration",
            config=team_config,
            result=result,
        )
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
        epsilon: float = 0.1,
        min_samples_for_confidence: int = 5,
    ):
        """Initialize the learner.

        Args:
            db_path: Path to SQLite database - legacy, now uses unified database
            learning_rate: Q-learning alpha (how fast to update)
            discount_factor: Q-learning gamma (future reward weight)
            epsilon: Exploration probability for epsilon-greedy
            min_samples_for_confidence: Minimum samples for high confidence
        """
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.min_samples = min_samples_for_confidence

        # Use unified database from victor.core.database
        self._db_manager = get_database()
        self.db = self._db_manager.get_connection()
        self.db_path = self._db_manager.db_path
        self._ensure_tables()

        logger.info(f"TeamCompositionLearner initialized (unified db={self.db_path})")

    def _ensure_tables(self) -> None:
        """Create database tables if they don't exist.

        Uses Tables constants from victor.core.schema for consistent naming.
        """
        cursor = self.db.cursor()

        # Composition stats table (agent_team_config)
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {Tables.AGENT_TEAM_CONFIG} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                config_key TEXT UNIQUE NOT NULL,
                formation TEXT NOT NULL,
                role_counts TEXT NOT NULL,
                task_category TEXT,
                execution_count INTEGER DEFAULT 0,
                success_count INTEGER DEFAULT 0,
                avg_quality REAL DEFAULT 0.5,
                avg_duration REAL DEFAULT 0,
                q_value REAL DEFAULT 0.5,
                updated_at TEXT DEFAULT (datetime('now'))
            )
            """)

        # Execution history table (agent_team_run)
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {Tables.AGENT_TEAM_RUN} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                team_id TEXT NOT NULL,
                task_category TEXT,
                formation TEXT,
                role_counts TEXT,
                member_count INTEGER,
                budget_used INTEGER,
                tools_used INTEGER,
                success INTEGER,
                quality_score REAL,
                duration_seconds REAL,
                created_at TEXT DEFAULT (datetime('now'))
            )
            """)

        # Indexes
        cursor.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_team_config_category
            ON {Tables.AGENT_TEAM_CONFIG}(task_category)
            """)
        cursor.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_team_run_category
            ON {Tables.AGENT_TEAM_RUN}(task_category)
            """)

        self.db.commit()

    def suggest_team(
        self,
        task_type: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> TeamRecommendation:
        """Suggest optimal team composition for a task.

        Uses epsilon-greedy strategy: mostly exploit best known composition,
        occasionally explore alternatives.

        Args:
            task_type: Task type or description
            context: Optional context (complexity, file count, etc.)

        Returns:
            TeamRecommendation with suggested composition
        """
        # Categorize task
        category = categorize_task(task_type)
        context = context or {}

        # Epsilon-greedy: sometimes explore
        if random.random() < self.epsilon:
            return self._explore_composition(category, context)

        # Otherwise exploit: get best known composition
        return self._exploit_best(category, context)

    def _exploit_best(
        self,
        category: TaskCategory,
        context: Dict[str, Any],
    ) -> TeamRecommendation:
        """Get the best known composition for a category.

        Args:
            category: Task category
            context: Task context

        Returns:
            Best TeamRecommendation based on Q-values
        """
        cursor = self.db.cursor()

        # Get compositions for this category, ordered by Q-value
        cursor.execute(
            f"""
            SELECT config_key, formation, role_counts, q_value,
                   execution_count, avg_quality, success_count
            FROM {Tables.AGENT_TEAM_CONFIG}
            WHERE task_category = ?
            ORDER BY q_value DESC
            LIMIT 1
            """,
            (category.value,),
        )
        row = cursor.fetchone()

        if row is None:
            # No data - return baseline
            return DEFAULT_COMPOSITIONS.get(category, DEFAULT_COMPOSITIONS[TaskCategory.MIXED])

        (
            comp_key,
            formation,
            role_counts,
            q_value,
            total_exec,
            avg_quality,
            successes,
        ) = row

        role_counts = json_loads(role_counts)
        # Estimate budget based on execution count and typical team size
        avg_budget = 40  # Default budget

        # Compute confidence based on sample size
        confidence = min(1.0, total_exec / (self.min_samples * 2))

        # Adjust for context if provided
        if context.get("complexity") == "high":
            avg_budget = int(avg_budget * 1.2)
        elif context.get("complexity") == "low":
            avg_budget = int(avg_budget * 0.8)

        return TeamRecommendation(
            formation=TeamFormation(formation),
            role_distribution=role_counts,
            suggested_budget=max(20, avg_budget),
            confidence=confidence,
            reason=f"Best performer for {category.value} (Q={q_value:.2f})",
            sample_size=total_exec,
            is_baseline=False,
        )

    def _explore_composition(
        self,
        category: TaskCategory,
        context: Dict[str, Any],
    ) -> TeamRecommendation:
        """Explore an alternative composition.

        Either tries a random variation or a less-used composition.

        Args:
            category: Task category
            context: Task context

        Returns:
            Exploratory TeamRecommendation
        """
        # Get baseline
        baseline = DEFAULT_COMPOSITIONS.get(category, DEFAULT_COMPOSITIONS[TaskCategory.MIXED])

        # Randomly vary the baseline
        formations = list(TeamFormation)
        formation = random.choice(formations)

        # Vary role counts
        roles = ["researcher", "planner", "executor", "reviewer", "tester"]
        role_weights = {
            TaskCategory.EXPLORATION: {"researcher": 0.6, "executor": 0.2},
            TaskCategory.IMPLEMENTATION: {"executor": 0.5, "planner": 0.3},
            TaskCategory.REVIEW: {"reviewer": 0.5, "researcher": 0.3},
            TaskCategory.TESTING: {"tester": 0.5, "researcher": 0.3},
            TaskCategory.DEBUGGING: {"researcher": 0.4, "executor": 0.4},
        }

        weights = role_weights.get(category, {"executor": 0.4, "researcher": 0.3})

        # Generate random role distribution
        role_distribution = {}
        total_members = random.randint(2, 4)

        for _ in range(total_members):
            # Weighted random selection
            weighted_roles = []
            for role in roles:
                weight = weights.get(role, 0.1)
                weighted_roles.extend([role] * int(weight * 10))

            if weighted_roles:
                selected = random.choice(weighted_roles)
            else:
                selected = random.choice(roles)

            role_distribution[selected] = role_distribution.get(selected, 0) + 1

        # Vary budget
        budget_base = baseline.suggested_budget
        budget = budget_base + random.randint(-10, 20)
        budget = max(20, min(100, budget))

        return TeamRecommendation(
            formation=formation,
            role_distribution=role_distribution,
            suggested_budget=budget,
            confidence=0.3,
            reason=f"Exploration: trying {formation.value} with varied roles",
            sample_size=0,
            is_baseline=False,
        )

    @property
    def name(self) -> str:
        """Learner name for BaseLearner compatibility."""
        return "team_composition"

    def record_outcome(self, outcome: Any) -> None:
        """Record outcome from RLOutcome (BaseLearner interface).

        This method accepts an RLOutcome object from the RLCoordinator
        and extracts team-specific data from its metadata.

        Args:
            outcome: RLOutcome object with team metadata
        """
        from victor.framework.rl.base import RLOutcome

        if isinstance(outcome, RLOutcome):
            # Extract team info from metadata
            metadata = outcome.metadata or {}
            team_name = metadata.get("team_name", "unknown")
            formation_str = metadata.get("formation", "sequential")
            member_count = metadata.get("member_count", 1)
            tool_calls = metadata.get("tool_calls", 0)
            duration = metadata.get("duration_seconds", 0.0)

            # Create a simplified metrics record
            category = categorize_task(outcome.task_type)

            # Compute composition key from formation
            role_str = f"executor={member_count}"
            comp_key = f"{formation_str}:{role_str}"

            cursor = self.db.cursor()

            # Store in run history
            cursor.execute(
                f"""
                INSERT INTO {Tables.AGENT_TEAM_RUN}
                (team_id, task_category, formation, role_counts, member_count,
                 budget_used, tools_used, success, quality_score, duration_seconds)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    team_name,
                    category.value,
                    formation_str,
                    json_dumps({"executor": member_count}),
                    member_count,
                    tool_calls,  # Approximate budget as tool calls
                    tool_calls,
                    1 if outcome.success else 0,
                    outcome.quality_score,
                    duration,
                ),
            )

            # Update Q-values
            reward = outcome.quality_score - 0.5 if outcome.success else -0.5

            cursor.execute(
                f"""
                SELECT q_value, execution_count, success_count
                FROM {Tables.AGENT_TEAM_CONFIG}
                WHERE config_key = ? AND task_category = ?
                """,
                (comp_key, category.value),
            )
            row = cursor.fetchone()

            if row:
                q_value = row[0]  # Only q_value needed for update
            else:
                q_value = 0.5

            # Q-learning update
            new_q = q_value + self.learning_rate * (reward - q_value)
            new_q = max(0.0, min(1.0, new_q))

            cursor.execute(
                f"""
                INSERT INTO {Tables.AGENT_TEAM_CONFIG}
                (config_key, formation, role_counts, task_category,
                 execution_count, success_count, avg_quality, avg_duration, q_value)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(config_key) DO UPDATE SET
                    execution_count = execution_count + 1,
                    success_count = success_count + ?,
                    avg_quality = (avg_quality * execution_count + ?) / (execution_count + 1),
                    avg_duration = (avg_duration * execution_count + ?) / (execution_count + 1),
                    q_value = ?,
                    updated_at = datetime('now')
                """,
                (
                    comp_key,
                    formation_str,
                    json_dumps({"executor": member_count}),
                    category.value,
                    1,
                    1 if outcome.success else 0,
                    outcome.quality_score,
                    duration,
                    new_q,
                    # For UPDATE clause
                    1 if outcome.success else 0,
                    outcome.quality_score,
                    duration,
                    new_q,
                ),
            )
            self.db.commit()

            logger.debug(
                f"RL: Team {team_name} outcome: Q={q_value:.3f}->{new_q:.3f}, "
                f"success={outcome.success}"
            )
        else:
            # Fallback: assume old-style call
            logger.warning(
                "TeamCompositionLearner.record_outcome received non-RLOutcome; "
                "use record_team_outcome for direct team recording"
            )

    def record_team_outcome(
        self,
        task_type: str,
        config: TeamConfig,
        result: Any,  # TeamResult
        quality_override: Optional[float] = None,
    ) -> None:
        """Record outcome and update Q-values (direct team interface).

        Args:
            task_type: Task type or description
            config: Team configuration used
            result: TeamResult from execution
            quality_override: Optional quality score override
        """
        # Categorize and create metrics
        category = categorize_task(task_type)
        metrics = TeamMetrics.from_result(config, result, category)

        if quality_override is not None:
            metrics.quality_score = quality_override

        # Store in history
        self._store_execution(metrics)

        # Update Q-values
        self._update_q_values(metrics)

        logger.debug(
            f"Recorded team outcome: {category.value}, "
            f"success={metrics.success}, quality={metrics.quality_score:.2f}"
        )

    def _store_execution(self, metrics: TeamMetrics) -> None:
        """Store execution in history table.

        Args:
            metrics: Execution metrics
        """
        cursor = self.db.cursor()
        cursor.execute(
            f"""
            INSERT INTO {Tables.AGENT_TEAM_RUN}
            (team_id, task_category, formation, role_counts, member_count,
             budget_used, tools_used, success, quality_score, duration_seconds)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                metrics.team_id,
                metrics.task_category.value,
                metrics.formation.value,
                json_dumps(metrics.role_distribution),
                metrics.member_count,
                metrics.total_tool_budget,
                metrics.tools_used,
                1 if metrics.success else 0,
                metrics.quality_score,
                metrics.duration_seconds,
            ),
        )
        self.db.commit()

    def _update_q_values(self, metrics: TeamMetrics) -> None:
        """Update Q-values using Q-learning update rule.

        Q(s,a) = Q(s,a) + α * (reward + γ * max_Q(s') - Q(s,a))

        Args:
            metrics: Execution metrics
        """
        cursor = self.db.cursor()

        # Compute composition key
        role_str = ",".join(
            f"{role}={count}" for role, count in sorted(metrics.role_distribution.items())
        )
        comp_key = f"{metrics.formation.value}:{role_str}"

        # Compute reward
        reward = self._compute_reward(metrics)

        # Get current Q-value
        cursor.execute(
            f"""
            SELECT q_value, execution_count, success_count, avg_quality, avg_duration
            FROM {Tables.AGENT_TEAM_CONFIG}
            WHERE config_key = ? AND task_category = ?
            """,
            (comp_key, metrics.task_category.value),
        )
        row = cursor.fetchone()

        if row is None:
            # New composition - initialize
            q_value = 0.5  # Neutral starting point
            total_exec = 0
            successes = 0
            avg_quality = 0.5
            avg_duration = 0.0
        else:
            q_value, total_exec, successes, avg_quality, avg_duration = row

        # Get max Q-value for this category (for Q-learning update)
        cursor.execute(
            f"""
            SELECT MAX(q_value) FROM {Tables.AGENT_TEAM_CONFIG}
            WHERE task_category = ?
            """,
            (metrics.task_category.value,),
        )
        max_q_row = cursor.fetchone()
        max_q = max_q_row[0] if max_q_row and max_q_row[0] else 0.5

        # Q-learning update
        new_q = q_value + self.learning_rate * (reward + self.discount_factor * max_q - q_value)
        new_q = max(0.0, min(1.0, new_q))

        # Update stats using exponential moving average
        new_total_exec = total_exec + 1
        new_successes = successes + (1 if metrics.success else 0)
        # EMA with alpha=0.2 for quality and duration
        alpha = 0.2
        new_avg_quality = alpha * metrics.quality_score + (1 - alpha) * avg_quality
        new_avg_duration = alpha * metrics.duration_seconds + (1 - alpha) * avg_duration

        # Upsert
        cursor.execute(
            f"""
            INSERT INTO {Tables.AGENT_TEAM_CONFIG}
            (config_key, formation, role_counts, task_category,
             execution_count, success_count, avg_quality, avg_duration, q_value)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(config_key) DO UPDATE SET
                execution_count = excluded.execution_count,
                success_count = excluded.success_count,
                avg_quality = excluded.avg_quality,
                avg_duration = excluded.avg_duration,
                q_value = excluded.q_value,
                updated_at = datetime('now')
            """,
            (
                comp_key,
                metrics.formation.value,
                json_dumps(metrics.role_distribution),
                metrics.task_category.value,
                new_total_exec,
                new_successes,
                new_avg_quality,
                new_avg_duration,
                new_q,
            ),
        )
        self.db.commit()

        logger.debug(
            f"Updated Q-value for {comp_key}: {q_value:.3f} -> {new_q:.3f} "
            f"(reward={reward:.3f})"
        )

    def _compute_reward(self, metrics: TeamMetrics) -> float:
        """Compute reward signal from execution metrics.

        Args:
            metrics: Execution metrics

        Returns:
            Reward value between -1.0 and 1.0
        """
        if not metrics.success:
            return -0.5  # Penalty for failure

        # Base reward from quality
        reward = metrics.quality_score - 0.5  # Normalize around 0

        # Bonus for efficiency
        efficiency = metrics.compute_efficiency()
        reward += 0.2 * (efficiency - 0.5)

        # Speed bonus (normalized to typical execution)
        if metrics.duration_seconds < 30:
            reward += 0.1
        elif metrics.duration_seconds > 120:
            reward -= 0.1

        return max(-1.0, min(1.0, reward))

    def get_stats(self, task_category: Optional[TaskCategory] = None) -> Dict[str, Any]:
        """Get learner statistics.

        Args:
            task_category: Optional filter by category

        Returns:
            Statistics dictionary
        """
        cursor = self.db.cursor()

        if task_category:
            cursor.execute(
                f"""
                SELECT COUNT(*), SUM(execution_count), AVG(q_value)
                FROM {Tables.AGENT_TEAM_CONFIG}
                WHERE task_category = ?
                """,
                (task_category.value,),
            )
        else:
            cursor.execute(f"""
                SELECT COUNT(*), SUM(execution_count), AVG(q_value)
                FROM {Tables.AGENT_TEAM_CONFIG}
                """)

        row = cursor.fetchone()
        composition_count, total_executions, avg_q = row

        # Get top compositions
        if task_category:
            cursor.execute(
                f"""
                SELECT config_key, q_value, execution_count
                FROM {Tables.AGENT_TEAM_CONFIG}
                WHERE task_category = ?
                ORDER BY q_value DESC
                LIMIT 5
                """,
                (task_category.value,),
            )
        else:
            cursor.execute(f"""
                SELECT config_key, q_value, execution_count
                FROM {Tables.AGENT_TEAM_CONFIG}
                ORDER BY q_value DESC
                LIMIT 5
                """)

        top_compositions = [
            {"key": row[0], "q_value": row[1], "executions": row[2]} for row in cursor.fetchall()
        ]

        return {
            "composition_count": composition_count or 0,
            "total_executions": total_executions or 0,
            "avg_q_value": avg_q or 0.5,
            "top_compositions": top_compositions,
            "learning_rate": self.learning_rate,
            "epsilon": self.epsilon,
        }

    def reset(self) -> None:
        """Reset all learned data."""
        cursor = self.db.cursor()
        cursor.execute(f"DELETE FROM {Tables.AGENT_TEAM_CONFIG}")
        cursor.execute(f"DELETE FROM {Tables.AGENT_TEAM_RUN}")
        self.db.commit()
        logger.info("TeamCompositionLearner reset")

    def close(self) -> None:
        """Close database connection."""
        self.db.close()


# Singleton instance
_global_learner: Optional[TeamCompositionLearner] = None


def get_team_learner() -> TeamCompositionLearner:
    """Get the global team composition learner.

    Returns:
        Global TeamCompositionLearner instance
    """
    global _global_learner
    if _global_learner is None:
        _global_learner = TeamCompositionLearner()
    return _global_learner


__all__ = [
    "TeamRecommendation",
    "TeamCompositionLearner",
    "get_team_learner",
    "DEFAULT_COMPOSITIONS",
]
