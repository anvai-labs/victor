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

"""RL learner for optimal continuation patience per provider:model:task.

Learns the optimal `continuation_patience` value - how many prompts without
tool calls before flagging as stuck loop.

Currently, this is a static value in provider adapters:
- DeepSeek: continuation_patience=5
- Claude: continuation_patience=3

This learner refines these baselines based on actual stuck loop outcomes.

Strategy:
- Track false_positive_rate: Model flagged as stuck but eventually made progress
- Track missed_stuck_loops: Model continued without making progress
- If FP rate > 20% → Increase patience (too eager to flag)
- If missed loops > 10% → Decrease patience (too lenient)
- Converge to optimal [1, 15] range
"""

import logging
from typing import Optional

from victor.framework.rl.base import BaseLearner, RLOutcome, RLRecommendation
from victor.core.schema import Tables
from victor.framework.rl.migration import RLTableMigrator

logger = logging.getLogger(__name__)


class ContinuationPatienceLearner(BaseLearner):
    """Learn optimal continuation_patience per provider:model:task.

    This learns how patient we should be before flagging a continuation loop
    as stuck (i.e., how many prompts without tool calls are acceptable).

    Attributes:
        name: Always "continuation_patience"
        db: SQLite database connection
        learning_rate: Adjustment rate (0.0-1.0)
    """

    def _ensure_tables(self) -> None:
        """Migrate legacy per-learner tables to unified RL tables."""
        RLTableMigrator(self.db).run_if_needed(
            self.name, RLTableMigrator.migrate_continuation_patience
        )

    def record_outcome(self, outcome: RLOutcome) -> None:
        """Record continuation patience outcome.

        Expected metadata:
        - continuation_prompts: How many prompts were sent without tool calls
        - patience_threshold: What the patience was configured to
        - flagged_as_stuck: Was it flagged as stuck loop?
        - actually_stuck: Was it actually stuck (true positive)?
        - eventually_made_progress: Did model eventually recover and make progress?

        Args:
            outcome: Outcome with continuation patience data
        """
        context_key = self._get_context_key(outcome.provider, outcome.model, outcome.task_type)

        cursor = self.db.cursor()

        # Load existing stats from rl_task_stat
        cursor.execute(
            f"SELECT stat_key, stat_value FROM {Tables.RL_TASK_STAT}"
            f" WHERE learner_id = ? AND task_type = ?",
            (self.name, context_key),
        )
        row_map = {
            row_dict["stat_key"]: row_dict["stat_value"]
            for row_dict in (dict(r) for r in cursor.fetchall())
        }

        # Load current_patience from rl_param
        cursor.execute(
            f"SELECT param_value FROM {Tables.RL_PARAM}"
            f" WHERE learner_id = ? AND param_key = 'current_patience' AND context = ?",
            (self.name, context_key),
        )
        patience_row = cursor.fetchone()
        baseline = self._get_provider_baseline("continuation_patience") or 3
        current_patience = int(patience_row["param_value"]) if patience_row else baseline

        total_sessions = int(row_map.get("total_sessions", 0))
        false_positives = int(row_map.get("false_positives", 0))
        true_positives = int(row_map.get("true_positives", 0))
        missed_stuck_loops = int(row_map.get("missed_stuck_loops", 0))

        # Update counts
        total_sessions += 1
        flagged_as_stuck = outcome.metadata.get("flagged_as_stuck", False)
        actually_stuck = outcome.metadata.get("actually_stuck", False)

        if flagged_as_stuck and not actually_stuck:
            false_positives += 1
        elif flagged_as_stuck and actually_stuck:
            true_positives += 1
        elif not flagged_as_stuck and actually_stuck:
            missed_stuck_loops += 1

        stats = {
            "context_key": context_key,
            "total_sessions": total_sessions,
            "false_positives": false_positives,
            "true_positives": true_positives,
            "missed_stuck_loops": missed_stuck_loops,
            "current_patience": current_patience,
        }

        if total_sessions >= 5:
            self._update_patience(stats)
            current_patience = stats["current_patience"]

        # Write stats to rl_task_stat
        ts = outcome.timestamp
        for stat_key, stat_value in (
            ("total_sessions", float(total_sessions)),
            ("false_positives", float(false_positives)),
            ("true_positives", float(true_positives)),
            ("missed_stuck_loops", float(missed_stuck_loops)),
        ):
            cursor.execute(
                f"""
                INSERT OR REPLACE INTO {Tables.RL_TASK_STAT}
                (learner_id, task_type, stat_key, stat_value, sample_count, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (self.name, context_key, stat_key, stat_value, total_sessions, ts),
            )

        # Write current_patience to rl_param
        cursor.execute(
            f"""
            INSERT OR REPLACE INTO {Tables.RL_PARAM}
            (learner_id, param_key, param_value, context, sample_count, updated_at)
            VALUES (?, 'current_patience', ?, ?, ?, ?)
            """,
            (self.name, float(current_patience), context_key, total_sessions, ts),
        )
        self.db.commit()

        logger.debug(
            f"RL: Recorded continuation_patience outcome for {context_key} "
            f"(sessions={stats['total_sessions']}, FP={stats['false_positives']}, "
            f"TP={stats['true_positives']}, missed={stats['missed_stuck_loops']})"
        )

    def _update_patience(self, stats: dict) -> None:
        """Update patience recommendation based on stats.

        Strategy:
        - High false positive rate → Increase patience
        - High missed stuck loop rate → Decrease patience
        - Balance between catching stuck loops and avoiding false alarms

        Args:
            stats: Statistics dictionary
        """
        total = stats["total_sessions"]
        fp_rate = stats["false_positives"] / total if total > 0 else 0.0
        missed_rate = stats["missed_stuck_loops"] / total if total > 0 else 0.0

        current_patience = stats["current_patience"]
        adjustment = 0

        # High false positive rate → Increase patience
        if fp_rate > 0.2:
            adjustment = 2 if fp_rate > 0.3 else 1
            logger.debug(
                f"RL: High FP rate ({fp_rate:.1%}) for {stats['context_key']} → "
                f"increase patience by {adjustment}"
            )

        # High missed stuck loop rate → Decrease patience
        elif missed_rate > 0.1:
            adjustment = -2 if missed_rate > 0.2 else -1
            logger.debug(
                f"RL: High missed loop rate ({missed_rate:.1%}) for {stats['context_key']} → "
                f"decrease patience by {abs(adjustment)}"
            )

        # Apply adjustment with bounds [1, 15]
        new_patience = max(1, min(15, current_patience + adjustment))

        if new_patience != current_patience:
            stats["current_patience"] = new_patience
            logger.info(
                f"RL: Updated continuation_patience for {stats['context_key']}: "
                f"{current_patience} → {new_patience} "
                f"(FP={fp_rate:.1%}, missed={missed_rate:.1%})"
            )

    def get_recommendation(
        self, provider: str, model: str, task_type: str
    ) -> Optional[RLRecommendation]:
        """Get recommended continuation patience.

        Args:
            provider: Provider name
            model: Model name
            task_type: Task type

        Returns:
            Recommendation with patience value and confidence, or None
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
            f" WHERE learner_id = ? AND param_key = 'current_patience' AND context = ?",
            (self.name, context_key),
        )
        patience_row = cursor.fetchone()

        if not row_map and patience_row is None:
            baseline = self._get_provider_baseline("continuation_patience")
            if baseline is not None:
                return RLRecommendation(
                    value=baseline,
                    confidence=0.0,
                    reason="Provider baseline (no learning data yet)",
                    sample_size=0,
                    is_baseline=True,
                )
            return None

        total_sessions = int(row_map.get("total_sessions", 0))
        false_positives = int(row_map.get("false_positives", 0))
        missed_stuck_loops = int(row_map.get("missed_stuck_loops", 0))
        baseline_patience = self._get_provider_baseline("continuation_patience") or 3
        current_patience = int(patience_row["param_value"]) if patience_row else baseline_patience

        if total_sessions < 5:
            return RLRecommendation(
                value=self._get_provider_baseline("continuation_patience") or current_patience,
                confidence=0.0,
                reason=f"Insufficient data ({total_sessions} sessions)",
                sample_size=total_sessions,
                is_baseline=True,
            )

        confidence = min(1.0, total_sessions / 20.0)
        fp_rate = false_positives / total_sessions
        missed_rate = missed_stuck_loops / total_sessions
        if fp_rate > 0.2 or missed_rate > 0.1:
            confidence *= 0.7

        return RLRecommendation(
            value=current_patience,
            confidence=confidence,
            reason=f"Learned from {total_sessions} sessions "
            f"(FP={fp_rate:.1%}, missed={missed_rate:.1%})",
            sample_size=total_sessions,
            is_baseline=False,
        )

    def _compute_reward(self, outcome: RLOutcome) -> float:
        """Compute reward signal from outcome.

        Args:
            outcome: Outcome to compute reward for

        Returns:
            Reward value (-1.0 to 1.0)
        """
        # Reward for correct stuck loop detection
        flagged = outcome.metadata.get("flagged_as_stuck", False)
        actually_stuck = outcome.metadata.get("actually_stuck", False)

        if flagged and actually_stuck:
            # True positive - good!
            return 1.0
        elif not flagged and not actually_stuck:
            # True negative - good!
            return 1.0
        elif flagged and not actually_stuck:
            # False positive - bad
            return -0.5
        elif not flagged and actually_stuck:
            # Missed stuck loop - bad
            return -1.0

        return 0.0
