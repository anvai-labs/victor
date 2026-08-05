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

"""RL learner for optimal semantic similarity thresholds per embedding model:task:tool.

This learner determines the optimal similarity threshold for semantic code search
to balance precision and recall based on observed search quality metrics.

This is used by semantic search tools to dynamically adjust thresholds:
- code_search (hybrid keyword + semantic)
- semantic_code_search (pure semantic)

Strategy:
- If zero-result rate > 30% → Lower threshold (increase recall)
- If low-quality rate > 30% → Raise threshold (increase precision)
- Track per (embedding_model, task_type, tool_name) context
- Converge to optimal [0.1, 0.9] range

Migrated from: victor/codebase/semantic_threshold_learner.py
"""

import logging
import sqlite3
from typing import Optional

from victor.framework.rl.base import BaseLearner, RLOutcome, RLRecommendation
from victor.core.schema import Tables
from victor.framework.rl.migration import RLTableMigrator

logger = logging.getLogger(__name__)


class SemanticThresholdLearner(BaseLearner):
    """Learn optimal semantic similarity thresholds per embedding_model:task:tool.

    Uses multi-armed bandit approach with weighted moving average to find
    optimal similarity thresholds for each context combination.

    Attributes:
        name: Always "semantic_threshold"
        db: SQLite database connection
        learning_rate: Adjustment rate (0.0-1.0)
    """

    def _ensure_tables(self) -> None:
        """Migrate legacy per-learner tables to unified RL tables."""
        RLTableMigrator(self.db).run_if_needed(
            self.name, RLTableMigrator.migrate_semantic_threshold
        )

    def _load_context_stats(self, cursor: sqlite3.Cursor, context_key: str) -> dict:
        """Load all stat_keys for a context_key from rl_task_stat."""
        cursor.execute(
            f"SELECT stat_key, stat_value, sample_count FROM {Tables.RL_TASK_STAT}"
            f" WHERE learner_id = ? AND task_type = ?",
            (self.name, context_key),
        )
        row_map: dict = {}
        for row in cursor.fetchall():
            if hasattr(row, "keys"):
                row_dict = dict(row)
                stat_key = row_dict["stat_key"]
                stat_value = row_dict["stat_value"]
                sample_count = row_dict["sample_count"]
            else:
                stat_key, stat_value, sample_count = row
            row_map[stat_key] = (stat_value, sample_count)
        return row_map

    def _upsert_stat(
        self,
        cursor: sqlite3.Cursor,
        context_key: str,
        stat_key: str,
        stat_value: float,
        sample_count: int,
        ts: str,
    ) -> None:
        """Upsert a single stat row into rl_task_stat."""
        cursor.execute(
            f"""
            INSERT OR REPLACE INTO {Tables.RL_TASK_STAT}
            (learner_id, task_type, stat_key, stat_value, sample_count, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (self.name, context_key, stat_key, stat_value, sample_count, ts),
        )

    def record_outcome(self, outcome: RLOutcome) -> None:
        """Record semantic search outcome.

        Expected metadata:
        - embedding_model: Model used (e.g., "bge-small", "all-MiniLM-L12-v2")
        - tool_name: Tool name ("code_search", "semantic_code_search")
        - query: Search query
        - results_count: Number of results returned
        - threshold_used: Similarity threshold that was used
        - false_negatives: True if returned 0 results when shouldn't have
        - false_positives: True if returned irrelevant results

        Args:
            outcome: Outcome with semantic search data
        """
        # Check RL mode — skip writes in "none" mode
        try:
            from victor.config.settings import load_settings

            rl_mode = getattr(load_settings(), "rl_mode", "selective")
        except Exception:
            rl_mode = "selective"

        if rl_mode == "none":
            return

        # Extract metadata
        embedding_model = outcome.metadata.get("embedding_model", "unknown")
        tool_name = outcome.metadata.get("tool_name", "code_search")

        context_key = self._get_context_key(embedding_model, outcome.task_type, tool_name)

        cursor = self.db.cursor()

        # Load existing stats from rl_task_stat
        row_map = self._load_context_stats(cursor, context_key)

        def _sv(key: str, default: float) -> float:
            return row_map[key][0] if key in row_map else default

        total_searches = int(_sv("total_searches", 0))
        zero_result_count = int(_sv("zero_result_count", 0))
        low_quality_count = int(_sv("low_quality_count", 0))
        results_sum = _sv("results_sum", 0.0)
        threshold_sum = _sv("threshold_sum", 0.0)
        prev_recommended = _sv("recommended_threshold", -1.0)

        # Update counts
        total_searches += 1
        results_count = outcome.metadata.get("results_count", 0)
        if results_count == 0:
            zero_result_count += 1
        if outcome.metadata.get("false_positives", False):
            low_quality_count += 1

        # Update weighted moving average (decay=0.9)
        decay = 0.9
        threshold_used = outcome.metadata.get("threshold_used", 0.5)
        results_sum = decay * results_sum + results_count
        threshold_sum = decay * threshold_sum + threshold_used
        avg_results_count = results_sum / (1 + (total_searches - 1) * decay)
        avg_threshold = threshold_sum / (1 + (total_searches - 1) * decay)

        # Build stats dict for _update_threshold
        stats = {
            "context_key": context_key,
            "total_searches": total_searches,
            "zero_result_count": zero_result_count,
            "low_quality_count": low_quality_count,
            "avg_results_count": avg_results_count,
            "avg_threshold": avg_threshold,
            "recommended_threshold": None,
        }

        if total_searches >= 5:
            self._update_threshold(stats)

        recommended = stats.get("recommended_threshold")

        # Dirty-check: skip write only when recommendation is stable
        if rl_mode == "selective" and row_map and recommended is not None and prev_recommended >= 0:
            if abs(recommended - prev_recommended) < 0.01:
                logger.debug(
                    "RL: semantic_threshold recommendation unchanged for %s, skipping",
                    context_key,
                )
                return

        # Upsert all stats to rl_task_stat
        ts = outcome.timestamp
        for stat_key, stat_value in (
            ("total_searches", float(total_searches)),
            ("zero_result_count", float(zero_result_count)),
            ("low_quality_count", float(low_quality_count)),
            ("avg_results_count", avg_results_count),
            ("avg_threshold", avg_threshold),
            ("results_sum", results_sum),
            ("threshold_sum", threshold_sum),
        ):
            self._upsert_stat(cursor, context_key, stat_key, stat_value, total_searches, ts)

        if recommended is not None:
            self._upsert_stat(
                cursor,
                context_key,
                "recommended_threshold",
                recommended,
                total_searches,
                ts,
            )

        self.db.commit()

        logger.debug(
            f"RL: Recorded semantic_threshold outcome for {context_key} "
            f"(searches={stats['total_searches']}, "
            f"zero_rate={stats['zero_result_count']/stats['total_searches']:.1%}, "
            f"threshold={stats['avg_threshold']:.2f})"
        )

    def _update_threshold(self, stats: dict) -> None:
        """Update threshold recommendation based on stats.

        Strategy:
        1. High zero-result rate (>30%) → Lower threshold (increase recall)
        2. High low-quality rate (>30%) → Raise threshold (increase precision)
        3. Good balance → Minor adjustment toward optimal result count
        4. Bounds: [0.1, 0.9]

        Args:
            stats: Statistics dictionary
        """
        total = stats["total_searches"]
        zero_rate = stats["zero_result_count"] / total if total > 0 else 0.0
        low_quality_rate = stats["low_quality_count"] / total if total > 0 else 0.0
        avg_results = stats["avg_results_count"]
        current = stats["avg_threshold"]

        adjustment = 0.0

        # High false negative rate → Lower threshold (increase recall)
        if zero_rate > 0.3:
            adjustment = -0.1 if zero_rate > 0.5 else -0.05
            logger.debug(
                f"RL: High zero-result rate ({zero_rate:.1%}) for {stats['context_key']} → "
                f"lower threshold by {abs(adjustment):.2f}"
            )

        # High false positive rate → Raise threshold (increase precision)
        elif low_quality_rate > 0.3:
            adjustment = 0.1 if low_quality_rate > 0.5 else 0.05
            logger.debug(
                f"RL: High low-quality rate ({low_quality_rate:.1%}) for {stats['context_key']} → "
                f"raise threshold by {adjustment:.2f}"
            )

        # Good balance → Minor adjustment toward optimal result count
        elif zero_rate < 0.1 and low_quality_rate < 0.1:
            if avg_results < 3:
                # Too few results on average → lower threshold slightly
                adjustment = -0.02
                logger.debug(
                    f"RL: Low avg results ({avg_results:.1f}) for {stats['context_key']} → "
                    f"lower threshold by {abs(adjustment):.2f}"
                )
            elif avg_results > 20:
                # Too many results on average → raise threshold slightly
                adjustment = 0.02
                logger.debug(
                    f"RL: High avg results ({avg_results:.1f}) for {stats['context_key']} → "
                    f"raise threshold by {adjustment:.2f}"
                )

        # Apply adjustment with bounds [0.15, 0.9]
        recommended = current + adjustment
        recommended = max(0.15, min(0.9, recommended))

        stats["recommended_threshold"] = recommended

        # Skip log spam when threshold is stable (no meaningful update)
        if abs(recommended - current) < 0.01:
            return

        logger.debug(
            f"RL: Updated semantic_threshold for {stats['context_key']}: "
            f"{current:.2f} → {recommended:.2f} "
            f"(zero_rate={zero_rate:.1%}, low_quality_rate={low_quality_rate:.1%})"
        )

    def get_recommendation(
        self, provider: str, model: str, task_type: str
    ) -> Optional[RLRecommendation]:
        """Get recommended semantic similarity threshold.

        Note: For semantic threshold learner, provider/model params are
        actually embedding_model/tool_name, but we maintain signature
        compatibility with BaseLearner.

        Args:
            provider: Embedding model name (e.g., "all-MiniLM-L12-v2")
            model: Tool name (e.g., "code_search", "semantic_code_search")
            task_type: Task type

        Returns:
            Recommendation with threshold value and confidence, or None
        """
        embedding_model = provider
        tool_name = model

        context_key = self._get_context_key(embedding_model, task_type, tool_name)

        cursor = self.db.cursor()
        row_map = self._load_context_stats(cursor, context_key)

        if not row_map:
            return None

        def _sv(key: str, default: float) -> float:
            return row_map[key][0] if key in row_map else default

        total_searches = int(_sv("total_searches", 0))
        if total_searches < 5:
            return None

        zero_result_count = _sv("zero_result_count", 0.0)
        low_quality_count = _sv("low_quality_count", 0.0)
        avg_threshold = _sv("avg_threshold", 0.5)
        recommended_threshold = _sv("recommended_threshold", -1.0)

        confidence = min(1.0, total_searches / 30.0)
        zero_rate = zero_result_count / total_searches
        low_quality_rate = low_quality_count / total_searches

        if zero_rate > 0.3 or low_quality_rate > 0.3:
            confidence *= 0.7

        recommended = recommended_threshold if recommended_threshold >= 0 else avg_threshold

        return RLRecommendation(
            value=recommended,
            confidence=confidence,
            reason=f"Learned from {total_searches} searches "
            f"(zero_rate={zero_rate:.1%}, low_quality_rate={low_quality_rate:.1%})",
            sample_size=total_searches,
            is_baseline=False,
        )

    def _compute_reward(self, outcome: RLOutcome) -> float:
        """Compute reward signal from outcome.

        Reward based on:
        - Non-zero results: +0.5
        - Quality score: weighted component
        - False negatives: -0.5
        - False positives: -0.3
        - Result count in optimal range [3, 20]: +0.2

        Args:
            outcome: Outcome to compute reward for

        Returns:
            Reward value (-1.0 to 1.0)
        """
        reward = 0.0

        # Non-zero results bonus
        results_count = outcome.metadata.get("results_count", 0)
        if results_count > 0:
            reward += 0.5

        # Quality component (weight 40%)
        reward += 0.4 * (outcome.quality_score * 2 - 1)

        # Penalties
        if outcome.metadata.get("false_negatives", False):
            reward -= 0.5

        if outcome.metadata.get("false_positives", False):
            reward -= 0.3

        # Optimal result count bonus
        if 3 <= results_count <= 20:
            reward += 0.2

        return max(-1.0, min(1.0, reward))
