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

"""Meta-learning coordinator for cross-session pattern consolidation.

Extends the existing RLCoordinator with session-level aggregation that
persists UsageAnalytics in-memory summaries to the RL database for long-term
trend analysis. Does NOT recreate session aggregation — it bridges the existing
UsageAnalytics.get_session_summary() into the persistent RL store.

Usage:
    coord = get_meta_learning_coordinator()
    summary = coord.aggregate_session_metrics(repo_id="my-project")
    trends = coord.detect_long_term_trends(repo_id="my-project", days=30)
"""

from victor.core.json_utils import json_loads
from json import JSONDecodeError
import logging
from typing import Any, Dict, List, Optional

from victor.framework.rl.coordinator import RLCoordinator, get_rl_coordinator
from victor.core.schema import Tables

logger = logging.getLogger(__name__)


class MetaLearningCoordinator(RLCoordinator):
    """Extends RLCoordinator with cross-session meta-learning capabilities.

    Key design decisions:
    - Reuses UsageAnalytics.get_session_summary() — does not reimplement it
    - Stores aggregated summaries in the existing rl_outcome table (session_summary column)
    - Detects trends by querying historical rl_outcome rows, not a new table
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Import here to avoid circular dependency at module level
        from victor.agent.usage_analytics import UsageAnalytics

        self._analytics = UsageAnalytics.get_instance()

    def aggregate_session_metrics(self, repo_id: Optional[str] = None) -> Dict[str, Any]:
        """Aggregate in-memory session metrics and persist them to the RL database.

        Uses the existing UsageAnalytics.get_session_summary() for computation
        and writes the result to the rl_outcome table (session_summary column)
        so future calls can query historical trends.

        Args:
            repo_id: Repository identifier for isolation (optional)

        Returns:
            Session summary dict from UsageAnalytics (same shape every time)
        """
        summary = self._analytics.get_session_summary()

        if summary.get("status") == "no_sessions":
            return summary

        # Persist via the existing UsageAnalytics bridge method
        persisted = self._analytics.persist_to_rl_database(repo_id=repo_id)
        if persisted:
            logger.debug("MetaLearning: persisted session summary (repo=%s)", repo_id)

        return summary

    def detect_long_term_trends(
        self, repo_id: Optional[str] = None, days: int = 30
    ) -> Dict[str, Any]:
        """Detect patterns across long time windows using historical rl_outcome rows.

        Queries the session_summary column (written by aggregate_session_metrics)
        and computes trend direction for key metrics.

        Args:
            repo_id: Repository to scope the query (None = global)
            days: Look-back window in days

        Returns:
            Dict with trend analysis for each metric
        """
        try:
            cursor = self.db.cursor()

            where_clause = "WHERE task_type = 'session_summary'"
            params: List[Any] = []

            if repo_id:
                where_clause += " AND repo_id = ?"
                params.append(repo_id)

            where_clause += f" AND created_at >= datetime('now', '-{int(days)} days')"

            cursor.execute(
                f"SELECT session_summary FROM {Tables.RL_OUTCOME} "
                f"{where_clause} ORDER BY created_at ASC",
                params,
            )
            rows = cursor.fetchall()
        except Exception as e:
            logger.debug("MetaLearning: trend query failed: %s", e)
            return {"status": "unavailable", "error": str(e)}

        if not rows:
            return {"status": "no_historical_data", "days": days}

        summaries = []
        for row in rows:
            raw = row[0] if isinstance(row, (list, tuple)) else row["session_summary"]
            if raw:
                try:
                    summaries.append(json_loads(raw))
                except (JSONDecodeError, TypeError):
                    pass

        if len(summaries) < 2:
            return {"status": "insufficient_data", "snapshots": len(summaries)}

        return self._compute_trends(summaries)

    def _compute_trends(self, summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute trend direction for each numeric metric across snapshots."""
        trend_keys = [
            "avg_turns_per_session",
            "avg_tool_calls_per_session",
            "avg_tokens_per_session",
            "avg_session_duration_seconds",
            "total_sessions",
        ]

        trends: Dict[str, Any] = {
            "status": "ok",
            "snapshot_count": len(summaries),
            "metrics": {},
        }

        for key in trend_keys:
            values = [s.get(key) for s in summaries if s.get(key) is not None]
            if len(values) < 2:
                continue

            first_half = values[: len(values) // 2]
            second_half = values[len(values) // 2 :]
            avg_first = sum(first_half) / len(first_half)
            avg_second = sum(second_half) / len(second_half)

            if avg_first == 0:
                delta_pct = 0.0
            else:
                delta_pct = (avg_second - avg_first) / avg_first * 100

            trends["metrics"][key] = {
                "direction": ("up" if delta_pct > 2 else ("down" if delta_pct < -2 else "stable")),
                "delta_pct": round(delta_pct, 1),
                "latest": values[-1],
                "earliest": values[0],
            }

        return trends

    def get_consolidated_recommendations(
        self, repo_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Merge UsageAnalytics recommendations with RL coordinator patterns.

        Calls the existing get_optimization_recommendations() from UsageAnalytics
        and augments with historical pattern data from the RL database.

        Args:
            repo_id: Repository to scope RL queries

        Returns:
            Combined list of optimization recommendations
        """
        # Existing in-memory recommendations (don't recreate this logic)
        analytics_recs = self._analytics.get_optimization_recommendations()

        # Augment with RL-learned patterns
        rl_recs = []
        try:
            learner = self.get_learner("user_feedback")
            if learner:
                stats = learner.get_feedback_stats()  # type: ignore[attr-defined]
                if stats.get("total_feedback", 0) > 0:
                    rl_recs.append(
                        {
                            "priority": "medium",
                            "category": "user_feedback",
                            "issue": f"User avg rating: {stats['avg_rating']:.2f}",
                            "action": (
                                "Investigate low-rated sessions"
                                if stats["avg_rating"] < 0.7
                                else "Maintain current approach"
                            ),
                        }
                    )
        except Exception as e:
            logger.debug("MetaLearning: user_feedback enrichment skipped: %s", e)

        return analytics_recs + rl_recs


_meta_coordinator: Optional[MetaLearningCoordinator] = None


def get_meta_learning_coordinator() -> "RLCoordinator":
    """Get or create the MetaLearningCoordinator if the feature flag is enabled.

    Returns MetaLearningCoordinator when USE_LEARNING_FROM_EXECUTION is enabled
    (the default), or falls back to the base RLCoordinator for gradual rollout.
    Disable with: VICTOR_USE_LEARNING_FROM_EXECUTION=false
    """
    from victor.core.feature_flags import FeatureFlag, get_feature_flag_manager

    if not get_feature_flag_manager().is_enabled(FeatureFlag.USE_LEARNING_FROM_EXECUTION):
        logger.debug("MetaLearning: USE_LEARNING_FROM_EXECUTION disabled, using base coordinator")
        return get_rl_coordinator()

    global _meta_coordinator
    if _meta_coordinator is None:
        base = get_rl_coordinator()
        _meta_coordinator = MetaLearningCoordinator(db_path=str(base.db_path))
    return _meta_coordinator
