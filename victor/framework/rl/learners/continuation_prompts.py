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

"""RL learner for optimal continuation prompt limits per provider:model:task.

This learner determines the optimal max_continuation_prompts value - the total
budget for continuation prompts before forcing completion.

This is DIFFERENT from continuation_patience (learned by ContinuationPatienceLearner):
- max_continuation_prompts: Total budget (e.g., 10 prompts max)
- continuation_patience: No-tool tolerance (e.g., 5 prompts without tools)

Strategy:
- If stuck rate > 30% → Decrease max prompts (model gets stuck frequently)
- If quality < 0.6 and few prompts used → Increase max prompts (needs more time)
- If quality > 0.8 and many prompts used → Decrease max prompts (wasting time)
- Converge to optimal [1, 20] range

Migrated from: victor/agent/continuation_learner.py
"""

import logging
from typing import Optional

from victor.framework.rl.base import BaseLearner, RLOutcome, RLRecommendation
from victor.core.schema import Tables
from victor.framework.rl.migration import RLTableMigrator

logger = logging.getLogger(__name__)


class ContinuationPromptLearner(BaseLearner):
    """Learn optimal max_continuation_prompts per provider:model:task.

    Uses multi-armed bandit approach with weighted moving average to find
    optimal continuation prompt limits for each provider:model:task_type combination.

    Attributes:
        name: Always "continuation_prompts"
        db: SQLite database connection
        learning_rate: Adjustment rate (0.0-1.0)
    """

    def _ensure_tables(self) -> None:
        """Migrate legacy per-learner tables to unified RL tables."""
        RLTableMigrator(self.db).run_if_needed(
            self.name, RLTableMigrator.migrate_continuation_prompts
        )

    def record_outcome(self, outcome: RLOutcome) -> None:
        """Record continuation prompts outcome.

        Expected metadata:
        - continuation_prompts_used: How many continuation prompts were sent
        - max_prompts_configured: The max limit that was configured
        - stuck_loop_detected: Was stuck continuation loop detected?
        - forced_completion: Was completion forced?
        - tool_calls_total: Total tool calls made

        Args:
            outcome: Outcome with continuation prompts data
        """
        context_key = self._get_context_key(outcome.provider, outcome.model, outcome.task_type)

        cursor = self.db.cursor()

        # Load existing stats from rl_task_stat
        cursor.execute(
            f"SELECT stat_key, stat_value FROM {Tables.RL_TASK_STAT}"
            f" WHERE learner_id = ? AND task_type = ?",
            (self.name, context_key),
        )
        row_map = {r["stat_key"]: r["stat_value"] for r in (dict(row) for row in cursor.fetchall())}

        # Load current_max_prompts from rl_param
        cursor.execute(
            f"SELECT param_value FROM {Tables.RL_PARAM}"
            f" WHERE learner_id = ? AND param_key = 'current_max_prompts' AND context = ?",
            (self.name, context_key),
        )
        param_row = cursor.fetchone()
        default_max = outcome.metadata.get("max_prompts_configured", 6)
        current_max_prompts = int(param_row["param_value"]) if param_row else default_max

        total_sessions = int(row_map.get("total_sessions", 0))
        successful_sessions = int(row_map.get("successful_sessions", 0))
        stuck_loop_count = int(row_map.get("stuck_loop_count", 0))
        forced_completion_count = int(row_map.get("forced_completion_count", 0))
        quality_sum = row_map.get("quality_sum", 0.0)
        prompts_sum = row_map.get("prompts_sum", 0.0)

        # Update counts
        total_sessions += 1
        if outcome.success:
            successful_sessions += 1
        if outcome.metadata.get("stuck_loop_detected", False):
            stuck_loop_count += 1
        if outcome.metadata.get("forced_completion", False):
            forced_completion_count += 1

        decay = 0.9
        prompts_used = outcome.metadata.get("continuation_prompts_used", 0)
        quality_sum = decay * quality_sum + outcome.quality_score
        prompts_sum = decay * prompts_sum + prompts_used
        avg_quality_score = quality_sum / (1 + (total_sessions - 1) * decay)
        avg_prompts_used = prompts_sum / (1 + (total_sessions - 1) * decay)

        current_max_prompts = outcome.metadata.get("max_prompts_configured", current_max_prompts)

        stats = {
            "context_key": context_key,
            "total_sessions": total_sessions,
            "successful_sessions": successful_sessions,
            "stuck_loop_count": stuck_loop_count,
            "avg_quality_score": avg_quality_score,
            "avg_prompts_used": avg_prompts_used,
            "current_max_prompts": current_max_prompts,
        }

        if total_sessions >= 3:
            self._update_max_prompts(stats)
            current_max_prompts = stats.get("recommended_max_prompts") or current_max_prompts

        ts = outcome.timestamp
        for stat_key, stat_value in (
            ("total_sessions", float(total_sessions)),
            ("successful_sessions", float(successful_sessions)),
            ("stuck_loop_count", float(stuck_loop_count)),
            ("forced_completion_count", float(forced_completion_count)),
            ("avg_quality_score", avg_quality_score),
            ("avg_prompts_used", avg_prompts_used),
            ("quality_sum", quality_sum),
            ("prompts_sum", prompts_sum),
        ):
            cursor.execute(
                f"""
                INSERT OR REPLACE INTO {Tables.RL_TASK_STAT}
                (learner_id, task_type, stat_key, stat_value, sample_count, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (self.name, context_key, stat_key, stat_value, total_sessions, ts),
            )

        cursor.execute(
            f"""
            INSERT OR REPLACE INTO {Tables.RL_PARAM}
            (learner_id, param_key, param_value, context, sample_count, updated_at)
            VALUES (?, 'current_max_prompts', ?, ?, ?, ?)
            """,
            (self.name, float(current_max_prompts), context_key, total_sessions, ts),
        )
        self.db.commit()

        logger.debug(
            f"RL: Recorded continuation_prompts outcome for {context_key} "
            f"(sessions={stats['total_sessions']}, "
            f"stuck={stats['stuck_loop_count']}, quality={stats['avg_quality_score']:.2f})"
        )

    def _update_max_prompts(self, stats: dict) -> None:
        """Update max_prompts recommendation based on stats.

        Strategy:
        1. If stuck rate > 30% → Decrease (model gets stuck frequently)
        2. If quality < 0.6 and few prompts → Increase (needs more time)
        3. If quality > 0.8 and many prompts → Decrease (wasting time)
        4. If low success rate → Increase (model struggling)

        Args:
            stats: Statistics dictionary
        """
        total = stats["total_sessions"]
        stuck_rate = stats["stuck_loop_count"] / total if total > 0 else 0.0
        success_rate = stats["successful_sessions"] / total if total > 0 else 0.0
        quality = stats["avg_quality_score"]
        avg_prompts = stats["avg_prompts_used"]
        current = stats["current_max_prompts"]

        adjustment = 0

        # High stuck rate → Decrease
        if stuck_rate > 0.3:
            adjustment = -2 if stuck_rate > 0.5 else -1
            logger.debug(
                f"RL: High stuck rate ({stuck_rate:.1%}) for {stats['context_key']} → "
                f"decrease by {abs(adjustment)}"
            )

        # Low quality + few prompts → Increase
        elif quality < 0.6 and avg_prompts < current * 0.7:
            adjustment = 2 if quality < 0.4 else 1
            logger.debug(
                f"RL: Low quality ({quality:.2f}) + few prompts ({avg_prompts:.1f}/{current}) "
                f"for {stats['context_key']} → increase by {adjustment}"
            )

        # High quality + many prompts → Decrease
        elif quality > 0.8 and avg_prompts > current * 0.8:
            adjustment = -1
            logger.debug(
                f"RL: High quality ({quality:.2f}) + many prompts ({avg_prompts:.1f}/{current}) "
                f"for {stats['context_key']} → decrease by 1"
            )

        # Low success rate → Increase
        elif success_rate < 0.5 and total >= 5:
            adjustment = 1
            logger.debug(
                f"RL: Low success rate ({success_rate:.1%}) for {stats['context_key']} → "
                f"increase by 1"
            )

        # Apply adjustment with bounds [1, 20]
        new_max = max(1, min(20, current + adjustment))

        if new_max != current:
            stats["recommended_max_prompts"] = new_max
            logger.info(
                f"RL: Updated continuation_prompts for {stats['context_key']}: "
                f"{current} → {new_max} "
                f"(stuck={stuck_rate:.1%}, quality={quality:.2f}, success={success_rate:.1%})"
            )
        else:
            stats["recommended_max_prompts"] = current

    def get_recommendation(
        self, provider: str, model: str, task_type: str
    ) -> Optional[RLRecommendation]:
        """Get recommended max continuation prompts.

        Args:
            provider: Provider name
            model: Model name
            task_type: Task type

        Returns:
            Recommendation with max prompts value and confidence, or None
        """
        context_key = self._get_context_key(provider, model, task_type)

        cursor = self.db.cursor()
        cursor.execute(
            f"SELECT stat_key, stat_value FROM {Tables.RL_TASK_STAT}"
            f" WHERE learner_id = ? AND task_type = ?",
            (self.name, context_key),
        )
        row_map = {r["stat_key"]: r["stat_value"] for r in (dict(row) for row in cursor.fetchall())}

        cursor.execute(
            f"SELECT param_value FROM {Tables.RL_PARAM}"
            f" WHERE learner_id = ? AND param_key = 'current_max_prompts' AND context = ?",
            (self.name, context_key),
        )
        param_row = cursor.fetchone()

        if not row_map and param_row is None:
            return None

        total_sessions = int(row_map.get("total_sessions", 0))
        if total_sessions < 3:
            return None

        stuck_loop_count = int(row_map.get("stuck_loop_count", 0))
        successful_sessions = int(row_map.get("successful_sessions", 0))
        current_max_prompts = int(param_row["param_value"]) if param_row else 6

        confidence = min(1.0, total_sessions / 20.0)
        stuck_rate = stuck_loop_count / total_sessions
        success_rate = successful_sessions / total_sessions

        if stuck_rate > 0.3 or success_rate < 0.5:
            confidence *= 0.7

        return RLRecommendation(
            value=current_max_prompts,
            confidence=confidence,
            reason=f"Learned from {total_sessions} sessions "
            f"(stuck={stuck_rate:.1%}, success={success_rate:.1%})",
            sample_size=total_sessions,
            is_baseline=False,
        )

    def _compute_reward(self, outcome: RLOutcome) -> float:
        """Compute reward signal from outcome.

        Reward based on:
        - Success: +1.0
        - Quality: weighted component
        - Stuck loop: -0.5
        - Forced completion: -0.3

        Args:
            outcome: Outcome to compute reward for

        Returns:
            Reward value (-1.0 to 1.0)
        """
        reward = 1.0 if outcome.success else -0.5

        # Quality component (weight 30%)
        reward = 0.7 * reward + 0.3 * (outcome.quality_score * 2 - 1)

        # Penalties
        if outcome.metadata.get("stuck_loop_detected", False):
            reward -= 0.5

        if outcome.metadata.get("forced_completion", False):
            reward -= 0.3

        # Tool usage bonus
        tool_calls = outcome.metadata.get("tool_calls_total", 0)
        if tool_calls > 5:
            reward += 0.1

        return max(-1.0, min(1.0, reward))
