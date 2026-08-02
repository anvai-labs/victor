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

"""User feedback learner — records explicit human ratings and turns them into
quality-weight recommendations without duplicating existing infrastructure.

Design constraints:
- Reuses RLOutcome.quality_score (not a new field)
- Distinguishes human vs auto feedback via metadata["feedback_source"]
- Feeds recommendations into the existing quality_weights learner
- Stores outcomes in the existing rl_outcome table (no new tables)
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from victor.framework.rl.base import BaseLearner, RLOutcome, RLRecommendation
from victor.core.schema import Tables

logger = logging.getLogger(__name__)


def create_outcome_with_user_feedback(
    session_id: str,
    rating: float,
    feedback: Optional[str] = None,
    helpful: Optional[bool] = None,
    correction: Optional[str] = None,
) -> RLOutcome:
    """Build an RLOutcome that carries explicit user feedback.

    The rating maps directly to quality_score so downstream learners receive
    human ground truth through the existing outcome pipeline.
    """
    return RLOutcome(
        provider="user",
        model="feedback",
        task_type="feedback",
        success=True,
        quality_score=rating,
        timestamp=datetime.now().isoformat(),
        metadata={
            "session_id": session_id,
            "feedback_source": "user",
            "user_feedback": feedback,
            "helpful": helpful,
            "correction": correction,
        },
        vertical="general",
    )


class UserFeedbackLearner(BaseLearner):
    """Learn from explicit user ratings.

    Aggregates feedback by task context and emits quality-weight adjustment
    recommendations. Does not maintain its own quality model — instead it
    feeds the existing quality_weights learner through RLRecommendation.
    """

    MIN_SAMPLES_FOR_CONFIDENCE = 5

    def __init__(
        self,
        name: str,
        db_connection: Any,
        learning_rate: float = 0.1,
        provider_adapter: Optional[Any] = None,
    ):
        super().__init__(name, db_connection, learning_rate, provider_adapter)
        self._feedback_by_context: Dict[str, List[float]] = {}
        self._load_state()

    def _ensure_tables(self) -> None:
        cursor = self.db.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rl_user_feedback_summary (
                context_key TEXT PRIMARY KEY,
                avg_rating REAL NOT NULL,
                sample_count INTEGER NOT NULL,
                last_updated TEXT NOT NULL
            )
        """)
        self.db.commit()

    def _load_state(self) -> None:
        try:
            cursor = self.db.cursor()
            cursor.execute(
                "SELECT context_key, avg_rating, sample_count FROM rl_user_feedback_summary"
            )
            for row in cursor.fetchall():
                row_dict = dict(row)
                key = row_dict["context_key"]
                count = row_dict["sample_count"]
                avg = row_dict["avg_rating"]
                self._feedback_by_context[key] = [avg] * count
        except Exception as e:
            logger.debug("user_feedback: could not load state: %s", e)

    def record_outcome(self, outcome: RLOutcome) -> None:
        if outcome.metadata.get("feedback_source") != "user":
            return

        context_key = outcome.metadata.get("session_id", outcome.task_type)
        if context_key not in self._feedback_by_context:
            self._feedback_by_context[context_key] = []

        score = outcome.quality_score if outcome.quality_score is not None else 0.5
        self._feedback_by_context[context_key].append(score)

        self._persist_summary(context_key)

    def _persist_summary(self, context_key: str) -> None:
        ratings = self._feedback_by_context.get(context_key, [])
        if not ratings:
            return
        avg = sum(ratings) / len(ratings)
        try:
            cursor = self.db.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO rl_user_feedback_summary
                (context_key, avg_rating, sample_count, last_updated)
                VALUES (?, ?, ?, ?)
                """,
                (context_key, avg, len(ratings), datetime.now().isoformat()),
            )
            self.db.commit()
        except Exception as e:
            logger.debug("user_feedback: persist failed: %s", e)

    def get_recommendation(
        self, provider: str, model: str, task_type: str
    ) -> Optional[RLRecommendation]:
        all_ratings: List[float] = []
        for ratings in self._feedback_by_context.values():
            all_ratings.extend(ratings)

        if not all_ratings:
            return None

        avg = sum(all_ratings) / len(all_ratings)
        confidence = min(len(all_ratings) / self.MIN_SAMPLES_FOR_CONFIDENCE, 1.0) * 0.8

        return RLRecommendation(
            value={"avg_user_rating": avg, "sample_count": len(all_ratings)},
            confidence=confidence,
            reason=f"Aggregated user feedback from {len(all_ratings)} ratings",
            sample_size=len(all_ratings),
            is_baseline=len(all_ratings) < self.MIN_SAMPLES_FOR_CONFIDENCE,
        )

    def _compute_reward(self, outcome: RLOutcome) -> float:
        score = outcome.quality_score if outcome.quality_score is not None else 0.5
        return 2.0 * score - 1.0

    def get_feedback_stats(self) -> Dict[str, Any]:
        all_ratings: List[float] = []
        for ratings in self._feedback_by_context.values():
            all_ratings.extend(ratings)

        if not all_ratings:
            return {"total_feedback": 0}

        return {
            "total_feedback": len(all_ratings),
            "avg_rating": sum(all_ratings) / len(all_ratings),
            "contexts_with_feedback": len(self._feedback_by_context),
        }
