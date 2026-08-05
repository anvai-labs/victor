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

"""Multi-task learning coordinator for RL framework.

This module provides a meta-learner that coordinates learning across
multiple verticals (coding, devops, data_science), enabling knowledge
transfer and faster learning for new domains.

Architecture:
    ┌─────────────────────────────────────────┐
    │         MULTI-TASK COORDINATOR          │
    │  ┌──────────────────────────────────┐   │
    │  │      Shared Encoder               │   │
    │  │  (Task + Provider + Model)        │   │
    │  └──────────────────────────────────┘   │
    │                  │                       │
    │    ┌─────────────┼─────────────┐        │
    │    ▼             ▼             ▼        │
    │ ┌─────────┐ ┌─────────┐ ┌─────────┐    │
    │ │ Coding  │ │ DevOps  │ │ Data    │    │
    │ │ Head    │ │ Head    │ │ Science │    │
    │ └─────────┘ └─────────┘ └─────────┘    │
    └─────────────────────────────────────────┘

Key Features:
1. Cross-vertical transfer learning
2. Shared representations for similar tasks
3. Warm-start for new verticals using existing knowledge
4. Adaptive weighting between vertical-specific and shared knowledge

Sprint 5: Advanced RL Patterns
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from victor.framework.rl.base import BaseLearner, RLOutcome, RLRecommendation
from victor.core.constants import DEFAULT_VERTICAL
from victor.framework.rl.shared_encoder import (
    ContextEmbedding,
    get_shared_encoder,
)

logger = logging.getLogger(__name__)


@dataclass
class VerticalHead:
    """Vertical-specific learning head.

    Maintains Q-values and statistics for a specific vertical.

    Attributes:
        vertical: Vertical name (coding, devops, data_science)
        q_values: Q-values indexed by context key
        sample_counts: Sample counts per context
        success_rates: Running success rates
    """

    vertical: str
    q_values: Dict[str, float] = field(default_factory=dict)
    sample_counts: Dict[str, int] = field(default_factory=dict)
    success_rates: Dict[str, float] = field(default_factory=dict)

    def get_q_value(self, context_key: str, default: float = 0.5) -> float:
        """Get Q-value for a context."""
        return self.q_values.get(context_key, default)

    def update(
        self,
        context_key: str,
        reward: float,
        learning_rate: float,
    ) -> None:
        """Update Q-value for a context.

        Args:
            context_key: Context identifier
            reward: Observed reward
            learning_rate: Update rate
        """
        current_q = self.q_values.get(context_key, 0.5)
        self.q_values[context_key] = current_q + learning_rate * (reward - current_q)
        self.sample_counts[context_key] = self.sample_counts.get(context_key, 0) + 1

        # Update running success rate
        count = self.sample_counts[context_key]
        current_rate = self.success_rates.get(context_key, 0.5)
        self.success_rates[context_key] = (
            current_rate * (count - 1) + (1.0 if reward > 0 else 0.0)
        ) / count


class MultiTaskLearner(BaseLearner):
    """Multi-task meta-learner for cross-vertical transfer.

    Coordinates learning across multiple verticals using shared
    representations. When a new vertical or task type is encountered,
    it can leverage knowledge from similar existing contexts.

    Algorithm:
    1. Encode context using SharedEncoder
    2. Find similar contexts from other verticals
    3. Compute weighted combination of:
       - Vertical-specific Q-value (if exists)
       - Transferred Q-values from similar contexts
    4. Update both vertical-specific and shared representations

    Transfer Learning Strategy:
    - High similarity (>0.8): Transfer 60% of knowledge
    - Medium similarity (0.5-0.8): Transfer 30% of knowledge
    - Low similarity (<0.5): No transfer

    Attributes:
        name: Always "multi_task"
        db: SQLite database connection
        learning_rate: Base learning rate
        transfer_rate: How much to weight transferred knowledge
    """

    # Transfer learning parameters
    DEFAULT_TRANSFER_RATE = 0.3
    HIGH_SIMILARITY_THRESHOLD = 0.8
    MEDIUM_SIMILARITY_THRESHOLD = 0.5

    # Minimum samples before confident transfer
    MIN_SAMPLES_FOR_TRANSFER = 5

    def __init__(
        self,
        name: str,
        db_connection: Any,
        learning_rate: float = 0.1,
        provider_adapter: Optional[Any] = None,
        transfer_rate: float = DEFAULT_TRANSFER_RATE,
    ):
        """Initialize multi-task learner.

        Args:
            name: Learner name
            db_connection: SQLite database connection
            learning_rate: Base learning rate
            provider_adapter: Optional provider adapter
            transfer_rate: Transfer learning weight
        """
        super().__init__(name, db_connection, learning_rate, provider_adapter)

        self.transfer_rate = transfer_rate

        # Shared encoder for embeddings
        self._encoder = get_shared_encoder(db_connection)

        # Vertical-specific heads
        self._heads: Dict[str, VerticalHead] = {}

        # Global Q-values (shared across verticals)
        self._global_q_values: Dict[str, float] = {}
        self._global_sample_counts: Dict[str, int] = {}

        # Load state
        self._load_state()

    def _ensure_tables(self) -> None:
        """Create tables for multi-task learning."""
        cursor = self.db.cursor()

        # Vertical-specific Q-values
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS multi_task_vertical_q (
                vertical TEXT NOT NULL,
                context_key TEXT NOT NULL,
                q_value REAL NOT NULL,
                sample_count INTEGER NOT NULL DEFAULT 0,
                success_rate REAL NOT NULL DEFAULT 0.5,
                last_updated TEXT NOT NULL,
                PRIMARY KEY (vertical, context_key)
            )
            """)

        # Global Q-values (shared)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS multi_task_global_q (
                context_key TEXT PRIMARY KEY,
                q_value REAL NOT NULL,
                sample_count INTEGER NOT NULL DEFAULT 0,
                last_updated TEXT NOT NULL
            )
            """)

        # Transfer history
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS multi_task_transfers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_vertical TEXT NOT NULL,
                target_vertical TEXT NOT NULL,
                source_context TEXT NOT NULL,
                target_context TEXT NOT NULL,
                similarity REAL NOT NULL,
                transfer_weight REAL NOT NULL,
                timestamp TEXT NOT NULL
            )
            """)

        # Indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_multi_task_vertical
            ON multi_task_vertical_q(vertical)
            """)

        self.db.commit()
        logger.debug("RL: multi_task tables ensured")

    def _load_state(self) -> None:
        """Load state from database."""
        cursor = self.db.cursor()

        try:
            # Load vertical Q-values
            cursor.execute("SELECT * FROM multi_task_vertical_q")
            for row in cursor.fetchall():
                row_dict = dict(row)
                vertical = row_dict["vertical"]

                if vertical not in self._heads:
                    self._heads[vertical] = VerticalHead(vertical=vertical)

                head = self._heads[vertical]
                context_key = row_dict["context_key"]
                head.q_values[context_key] = row_dict["q_value"]
                head.sample_counts[context_key] = row_dict["sample_count"]
                head.success_rates[context_key] = row_dict["success_rate"]

            # Load global Q-values
            cursor.execute("SELECT * FROM multi_task_global_q")
            for row in cursor.fetchall():
                row_dict = dict(row)
                context_key = row_dict["context_key"]
                self._global_q_values[context_key] = row_dict["q_value"]
                self._global_sample_counts[context_key] = row_dict["sample_count"]

            if self._heads:
                logger.info(f"RL: Loaded multi-task state for {len(self._heads)} verticals")

        except Exception as e:
            logger.debug(f"RL: Could not load multi-task state: {e}")

    def record_outcome(self, outcome: RLOutcome) -> None:
        """Record outcome and update multi-task Q-values.

        Updates both vertical-specific and global Q-values,
        enabling knowledge transfer.

        Args:
            outcome: Outcome with provider, model, task, success
        """
        vertical = outcome.vertical
        context_key = self._get_context_key(outcome.provider, outcome.model, outcome.task_type)

        # Compute reward
        reward = self._compute_reward(outcome)

        # Ensure vertical head exists
        if vertical not in self._heads:
            self._heads[vertical] = VerticalHead(vertical=vertical)

        head = self._heads[vertical]

        # Update vertical-specific Q-value
        head.update(context_key, reward, self.learning_rate)

        # Update global Q-value
        current_global = self._global_q_values.get(context_key, 0.5)
        self._global_q_values[context_key] = current_global + self.learning_rate * (
            reward - current_global
        )
        self._global_sample_counts[context_key] = self._global_sample_counts.get(context_key, 0) + 1

        # Perform transfer learning update
        self._transfer_update(outcome, reward)

        # Save to database
        self._save_to_db(vertical, context_key, head)

        logger.debug(
            f"RL: multi_task recorded for {vertical}:{context_key} " f"(reward={reward:.3f})"
        )

    def _compute_reward(self, outcome: RLOutcome) -> float:
        """Compute reward signal from outcome.

        Args:
            outcome: Outcome to compute reward for

        Returns:
            Reward value (-1.0 to 1.0)
        """
        # Base reward from success
        base_reward = 1.0 if outcome.success else -0.5

        # Modulate by quality score
        quality_factor = outcome.quality_score

        # Combine
        reward = base_reward * 0.6 + quality_factor * 0.4

        # Normalize to [-1, 1]
        return max(-1.0, min(1.0, (reward - 0.5) * 2))

    def _transfer_update(self, outcome: RLOutcome, reward: float) -> None:
        """Perform transfer learning update to similar contexts.

        Args:
            outcome: Current outcome
            reward: Computed reward
        """
        # Get embedding for current context
        embedding = self._encoder.encode(
            task_type=outcome.task_type,
            provider=outcome.provider,
            model=outcome.model,
            vertical=outcome.vertical,
        )

        # Find similar contexts in other verticals
        for other_vertical, head in self._heads.items():
            if other_vertical == outcome.vertical:
                continue

            for context_key, q_value in head.q_values.items():
                # Check if enough samples for transfer
                if head.sample_counts.get(context_key, 0) < self.MIN_SAMPLES_FOR_TRANSFER:
                    continue

                # Parse context key to get embedding
                parts = context_key.split(":")
                if len(parts) != 3:
                    continue

                other_embedding = self._encoder.encode(
                    task_type=parts[2],
                    provider=parts[0],
                    model=parts[1],
                    vertical=other_vertical,
                )

                # Compute transfer weight
                transfer_weight = self._compute_transfer_weight(embedding, other_embedding)

                if transfer_weight > 0:
                    # Apply transfer update (smaller rate)
                    transfer_lr = self.learning_rate * transfer_weight * 0.5
                    current_q = head.q_values.get(context_key, 0.5)
                    head.q_values[context_key] = current_q + transfer_lr * (reward - current_q)

    def _compute_transfer_weight(
        self,
        source: ContextEmbedding,
        target: ContextEmbedding,
    ) -> float:
        """Compute transfer learning weight.

        Args:
            source: Source context embedding
            target: Target context embedding

        Returns:
            Transfer weight [0, 1]
        """
        similarity = source.similarity(target)

        if similarity >= self.HIGH_SIMILARITY_THRESHOLD:
            return self.transfer_rate * 2  # High transfer
        elif similarity >= self.MEDIUM_SIMILARITY_THRESHOLD:
            return self.transfer_rate  # Medium transfer
        else:
            return 0.0  # No transfer

    def get_recommendation(
        self,
        provider: str,
        model: str,
        task_type: str,
        vertical: str = DEFAULT_VERTICAL,
    ) -> Optional[RLRecommendation]:
        """Get recommendation combining vertical and transferred knowledge.

        Args:
            provider: Provider name
            model: Model name
            task_type: Task type
            vertical: Vertical (coding, devops, data_science)

        Returns:
            Recommendation with Q-value and confidence
        """
        context_key = self._get_context_key(provider, model, task_type)

        # Get vertical-specific Q-value
        vertical_q = None
        vertical_samples = 0

        if vertical in self._heads:
            head = self._heads[vertical]
            if context_key in head.q_values:
                vertical_q = head.q_values[context_key]
                vertical_samples = head.sample_counts.get(context_key, 0)

        # Get global Q-value
        global_q = self._global_q_values.get(context_key)
        global_samples = self._global_sample_counts.get(context_key, 0)

        # Get transferred Q-value from similar contexts
        transferred_q, transfer_samples = self._get_transferred_q(
            provider, model, task_type, vertical
        )

        # Combine Q-values
        if vertical_q is not None:
            # Have vertical-specific data
            if transferred_q is not None and vertical_samples < 10:
                # Blend with transfer for low sample count
                blend_weight = min(1.0, vertical_samples / 10)
                q_value = blend_weight * vertical_q + (1 - blend_weight) * transferred_q
                total_samples = vertical_samples + transfer_samples // 2
            else:
                q_value = vertical_q
                total_samples = vertical_samples
            is_baseline = False
        elif transferred_q is not None:
            # Use transferred knowledge
            q_value = transferred_q
            total_samples = transfer_samples
            is_baseline = True
        elif global_q is not None:
            # Fall back to global
            q_value = global_q
            total_samples = global_samples
            is_baseline = True
        else:
            # No data - return baseline
            return RLRecommendation(
                value=0.5,
                confidence=0.3,
                reason="No learned data, using baseline",
                sample_size=0,
                is_baseline=True,
            )

        # Compute confidence
        confidence = self._compute_confidence(total_samples)

        return RLRecommendation(
            value=q_value,
            confidence=confidence,
            reason=f"Multi-task Q-value from {total_samples} samples",
            sample_size=total_samples,
            is_baseline=is_baseline,
        )

    def _get_transferred_q(
        self,
        provider: str,
        model: str,
        task_type: str,
        target_vertical: str,
    ) -> Tuple[Optional[float], int]:
        """Get transferred Q-value from similar contexts.

        Args:
            provider: Provider name
            model: Model name
            task_type: Task type
            target_vertical: Target vertical

        Returns:
            (transferred_q_value, sample_count) or (None, 0)
        """
        target_embedding = self._encoder.encode(
            task_type=task_type,
            provider=provider,
            model=model,
            vertical=target_vertical,
        )

        transferred_values = []
        total_samples = 0

        for other_vertical, head in self._heads.items():
            if other_vertical == target_vertical:
                continue

            for context_key, q_value in head.q_values.items():
                samples = head.sample_counts.get(context_key, 0)
                if samples < self.MIN_SAMPLES_FOR_TRANSFER:
                    continue

                # Parse context
                parts = context_key.split(":")
                if len(parts) != 3:
                    continue

                other_embedding = self._encoder.encode(
                    task_type=parts[2],
                    provider=parts[0],
                    model=parts[1],
                    vertical=other_vertical,
                )

                transfer_weight = self._compute_transfer_weight(other_embedding, target_embedding)

                if transfer_weight > 0:
                    transferred_values.append((q_value, transfer_weight, samples))
                    total_samples += samples

        if not transferred_values:
            return None, 0

        # Weighted average
        total_weight = sum(w for _, w, _ in transferred_values)
        if total_weight == 0:
            return None, 0

        transferred_q = sum(q * w for q, w, _ in transferred_values) / total_weight
        return transferred_q, total_samples

    def _compute_confidence(self, sample_count: int) -> float:
        """Compute confidence from sample count.

        Args:
            sample_count: Number of samples

        Returns:
            Confidence [0.3, 0.95]
        """
        if sample_count == 0:
            return 0.3
        return min(0.95, 0.3 + 0.65 * (1 - math.exp(-sample_count / 20)))

    def _save_to_db(
        self,
        vertical: str,
        context_key: str,
        head: VerticalHead,
    ) -> None:
        """Save state to database.

        Args:
            vertical: Vertical name
            context_key: Context key
            head: Vertical head
        """
        cursor = self.db.cursor()
        timestamp = datetime.now().isoformat()

        # Save vertical Q-value
        cursor.execute(
            """
            INSERT OR REPLACE INTO multi_task_vertical_q
            (vertical, context_key, q_value, sample_count, success_rate, last_updated)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                vertical,
                context_key,
                head.q_values.get(context_key, 0.5),
                head.sample_counts.get(context_key, 0),
                head.success_rates.get(context_key, 0.5),
                timestamp,
            ),
        )

        # Save global Q-value
        cursor.execute(
            """
            INSERT OR REPLACE INTO multi_task_global_q
            (context_key, q_value, sample_count, last_updated)
            VALUES (?, ?, ?, ?)
            """,
            (
                context_key,
                self._global_q_values.get(context_key, 0.5),
                self._global_sample_counts.get(context_key, 0),
                timestamp,
            ),
        )

        self.db.commit()

    def get_vertical_stats(self, vertical: str) -> Dict[str, Any]:
        """Get statistics for a specific vertical.

        Args:
            vertical: Vertical name

        Returns:
            Dictionary with vertical stats
        """
        if vertical not in self._heads:
            return {
                "vertical": vertical,
                "contexts": 0,
                "total_samples": 0,
                "avg_success_rate": 0.0,
            }

        head = self._heads[vertical]
        total_samples = sum(head.sample_counts.values())
        avg_success = (
            sum(head.success_rates.values()) / len(head.success_rates)
            if head.success_rates
            else 0.0
        )

        return {
            "vertical": vertical,
            "contexts": len(head.q_values),
            "total_samples": total_samples,
            "avg_success_rate": avg_success,
            "avg_q_value": (
                sum(head.q_values.values()) / len(head.q_values) if head.q_values else 0.5
            ),
        }

    def get_transfer_stats(self) -> Dict[str, Any]:
        """Get transfer learning statistics.

        Returns:
            Dictionary with transfer stats
        """
        # Count potential transfers
        potential_transfers = 0
        for v1 in self._heads:
            for v2 in self._heads:
                if v1 != v2:
                    potential_transfers += len(self._heads[v1].q_values) * len(
                        self._heads[v2].q_values
                    )

        return {
            "verticals": list(self._heads.keys()),
            "potential_transfer_pairs": potential_transfers,
            "transfer_rate": self.transfer_rate,
            "encoder_stats": self._encoder.export_metrics(),
        }

    def export_metrics(self) -> Dict[str, Any]:
        """Export learner metrics.

        Returns:
            Dictionary with learner stats
        """
        total_samples = sum(sum(head.sample_counts.values()) for head in self._heads.values())

        vertical_stats = {v: self.get_vertical_stats(v) for v in self._heads}

        return {
            "learner": self.name,
            "verticals": len(self._heads),
            "total_samples": total_samples,
            "global_contexts": len(self._global_q_values),
            "learning_rate": self.learning_rate,
            "transfer_rate": self.transfer_rate,
            "vertical_stats": vertical_stats,
            "transfer_stats": self.get_transfer_stats(),
        }
