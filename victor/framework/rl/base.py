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

"""Base classes and data models for RL framework."""

from victor.core.json_utils import json_dumps, json_loads
import logging
import math
import os
import random
import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, ClassVar, Dict, List, Optional, TYPE_CHECKING
from victor.core.constants import DEFAULT_VERTICAL
from victor.core.sql_utils import validate_identifier

if TYPE_CHECKING:
    from victor.protocols.provider_adapter import IProviderAdapter

logger = logging.getLogger(__name__)


def saturating_confidence(samples: float, scale: float) -> float:
    """Confidence that grows with evidence and saturates at 0.95.

    The shared learner policy: 0.5 with no evidence, approaching 0.95 as
    ``samples`` grows; ``scale`` sets how many samples count as "a lot".
    """
    return min(0.95, 0.5 + 0.45 * (1 - math.exp(-samples / scale)))


def sample_confidence(samples: float, min_samples: float, scale: float) -> "tuple[float, bool]":
    """``(confidence, is_baseline)`` with a low-confidence warm-up period.

    Below ``min_samples`` the recommendation is a baseline whose confidence
    ramps from 0.3 to 0.5; afterwards :func:`saturating_confidence` applies.
    """
    if samples < min_samples:
        return 0.3 + 0.2 * (samples / min_samples), True
    return saturating_confidence(samples, scale), False


def resolve_rl_rng_seed(name: str) -> Optional[int]:
    """Resolve a deterministic RNG seed from ``VICTOR_RL_SEED`` (None = entropy).

    Shared by all RL stochastic components so they draw from an injectable,
    reproducible source instead of the global ``random`` module. The component
    ``name`` is mixed in so each gets a distinct but reproducible stream from
    the same base seed. Returns None (system entropy) when the env var is unset.
    """
    raw = os.environ.get("VICTOR_RL_SEED")
    if raw is None or raw == "":
        return None
    try:
        base = int(raw)
    except ValueError:
        base = hash(raw)
    return (base ^ (hash(name) & 0xFFFFFFFF)) & 0x7FFFFFFF


@dataclass
class RLOutcome:
    """Base class for RL outcome records.

    This is recorded after every session/operation to provide feedback
    for learning optimal parameter values.

    Attributes:
        provider: Provider name (e.g., "anthropic", "deepseek")
        model: Model name (e.g., "claude-3-opus", "deepseek-chat")
        task_type: Task type ("analysis", "action", "default")
        success: Whether the operation completed successfully
        quality_score: Quality score 0.0-1.0 from grounding/user feedback
        timestamp: ISO timestamp of outcome
        metadata: Domain-specific metadata (varies by learner)
        vertical: Which vertical this came from (coding, devops, data_science)
    """

    provider: str
    model: str
    task_type: str
    success: bool
    quality_score: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    vertical: str = DEFAULT_VERTICAL
    # Per-segment process reward {turn/action index: credit} (Vision P6). Transient/observability —
    # blended into quality_score at outcome construction, so it is not persisted to the DB.
    segment_rewards: Optional[Dict[int, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "provider": self.provider,
            "model": self.model,
            "task_type": self.task_type,
            "success": self.success,
            "quality_score": self.quality_score,
            "timestamp": self.timestamp,
            "metadata": json_dumps(self.metadata),
            "vertical": self.vertical,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RLOutcome":
        """Create from dictionary."""
        metadata = data.get("metadata", "{}")
        if isinstance(metadata, str):
            metadata = json_loads(metadata)

        return cls(
            provider=data["provider"],
            model=data["model"],
            task_type=data["task_type"],
            success=data["success"],
            quality_score=data["quality_score"],
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            metadata=metadata,
            vertical=data.get("vertical", "coding"),
        )


@dataclass
class RLRecommendation:
    """Base class for RL recommendations.

    Returned when querying a learner for optimal parameter value.

    Attributes:
        value: The recommended value (type varies by learner)
        confidence: Confidence in recommendation 0.0-1.0
        reason: Human-readable explanation of recommendation
        sample_size: Number of outcomes used to compute recommendation
        is_baseline: Whether this is a provider baseline (no learning yet)
    """

    value: Any
    confidence: float
    reason: str
    sample_size: int
    is_baseline: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def learner_name(self) -> Optional[str]:
        """Compatibility accessor for extended learners."""
        value = self.metadata.get("learner_name")
        return value if isinstance(value, str) else None

    @property
    def recommendation_type(self) -> Optional[str]:
        """Compatibility accessor for extended learners."""
        value = self.metadata.get("recommendation_type")
        return value if isinstance(value, str) else None

    @property
    def key(self) -> Optional[str]:
        """Compatibility accessor for extended learners."""
        value = self.metadata.get("key")
        return value if isinstance(value, str) else None

    def __str__(self) -> str:
        """Human-readable representation."""
        baseline_str = " (baseline)" if self.is_baseline else ""
        return (
            f"RLRecommendation(value={self.value}, "
            f"confidence={self.confidence:.2f}, "
            f"samples={self.sample_size}{baseline_str})"
        )


class BaseLearner(ABC):
    """Abstract base class for all RL learners.

    Provides common infrastructure:
    - SQLite database access (via coordinator)
    - Provider adapter integration for baselines
    - Telemetry collection
    - Statistics tracking

    Subclasses implement domain-specific learning logic.
    """

    # Learners on the unified RL tables set this to their
    # ``RLTableMigrator.migrate_*`` function (wrapped in ``staticmethod``);
    # the default ``_ensure_tables`` then runs it once per database.
    _legacy_migration: ClassVar[Optional[Callable[[sqlite3.Connection], None]]] = None

    def __init__(
        self,
        name: str,
        db_connection: Any,  # sqlite3.Connection
        learning_rate: float = 0.1,
        provider_adapter: Optional["IProviderAdapter"] = None,
    ):
        """Initialize base learner.

        Args:
            name: Learner name (e.g., "continuation_patience")
            db_connection: SQLite database connection from coordinator
            learning_rate: How aggressively to adjust values (0.0-1.0)
            provider_adapter: Optional provider adapter for baselines
        """
        self.name = validate_identifier(name)
        self.db = db_connection
        self.learning_rate = learning_rate
        self.provider_adapter = provider_adapter

        # Per-learner stochastic source. Learners must draw from ``self._rng``
        # (NOT the global ``random`` module) so RL decisions are reproducible and
        # testable without monkeypatching stdlib. Unseeded by default (system
        # entropy = prior behavior); set ``VICTOR_RL_SEED`` for a deterministic,
        # per-learner-distinct stream, or call ``seed_rng()`` in tests.
        self._rng: random.Random = random.Random(resolve_rl_rng_seed(name))

        # Ensure tables exist
        self._ensure_tables()

    def seed_rng(self, seed: int) -> None:
        """Seed this learner's RNG for deterministic draws (tests/reproducibility)."""
        self._rng.seed(seed)

    def _ensure_tables(self) -> None:
        """Ensure required database tables exist.

        The default runs ``_legacy_migration`` once per database. Learners
        that own their tables override this to create them; table names
        should be prefixed with the learner name to avoid conflicts.

        Raises:
            TypeError: the learner neither sets ``_legacy_migration`` nor
                overrides this method (it would have no storage).
        """
        migrate = type(self)._legacy_migration
        if migrate is None:
            raise TypeError(
                f"{type(self).__name__} must set _legacy_migration or override _ensure_tables"
            )
        from victor.framework.rl.migration import RLTableMigrator

        RLTableMigrator(self.db).run_if_needed(self.name, migrate)

    def _read_task_stats(self, task_type: str) -> Dict[str, Any]:
        """This learner's ``rl_task_stat`` rows for ``task_type`` as ``{stat_key: value}``."""
        from victor.core.schema import Tables

        cursor = self.db.cursor()
        cursor.execute(
            f"SELECT stat_key, stat_value FROM {Tables.RL_TASK_STAT}"
            f" WHERE learner_id = ? AND task_type = ?",
            (self.name, task_type),
        )
        return {row["stat_key"]: row["stat_value"] for row in map(dict, cursor.fetchall())}

    def _read_params(self) -> List[Dict[str, Any]]:
        """All of this learner's ``rl_param`` rows (param_key, param_value, context, sample_count)."""
        from victor.core.schema import Tables

        cursor = self.db.cursor()
        cursor.execute(
            f"SELECT param_key, param_value, context, sample_count FROM {Tables.RL_PARAM}"
            f" WHERE learner_id = ?",
            (self.name,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def _read_param(self, param_key: str, context: str) -> Optional[Any]:
        """One ``rl_param`` value of this learner, or None when unset."""
        from victor.core.schema import Tables

        cursor = self.db.cursor()
        cursor.execute(
            f"SELECT param_value FROM {Tables.RL_PARAM}"
            f" WHERE learner_id = ? AND param_key = ? AND context = ?",
            (self.name, param_key, context),
        )
        row = cursor.fetchone()
        return dict(row)["param_value"] if row else None

    @abstractmethod
    def record_outcome(self, outcome: RLOutcome) -> None:
        """Record an outcome and update learned values.

        Args:
            outcome: Outcome data with provider, model, success, quality, etc.
        """
        pass

    @abstractmethod
    def get_recommendation(
        self, provider: str, model: str, task_type: str
    ) -> Optional[RLRecommendation]:
        """Get recommendation for given context.

        Args:
            provider: Provider name
            model: Model name
            task_type: Task type

        Returns:
            Recommendation with value and confidence, or None if no data
        """
        pass

    def _compute_reward(self, outcome: RLOutcome) -> float:
        """Compute reward signal from outcome.

        Default = the canonical reward derived from the outcome's success +
        quality_score (``victor.framework.rl.reward.reward_from_outcome``). Learners
        MAY override this for domain-specific *shaping* (e.g. ``tool_selector`` weights
        success/completion/grounding/efficiency); overriding is fully supported. This
        concrete default removes the need to reimplement the trivial case and gives
        one canonical formula instead of several drifting ones.

        Args:
            outcome: Outcome to compute reward for

        Returns:
            Reward value in ``[0.0, 1.0]``.
        """
        from victor.framework.rl.reward import reward_from_outcome

        return reward_from_outcome(outcome)

    def _get_provider_baseline(self, param_name: str) -> Any:
        """Get baseline value from provider adapter if available.

        Args:
            param_name: Parameter name (e.g., "continuation_patience")

        Returns:
            Baseline value from provider capabilities, or None
        """
        if not self.provider_adapter:
            return None

        return getattr(self.provider_adapter.capabilities, param_name, None)

    def _get_context_key(self, provider: str, model: str, task_type: str) -> str:
        """Generate context key for storage.

        Args:
            provider: Provider name
            model: Model name
            task_type: Task type

        Returns:
            Key string like "anthropic:claude-3-opus:analysis"
        """
        return f"{provider}:{model}:{task_type}"

    def export_metrics(self) -> Dict[str, Any]:
        """Export learner metrics for monitoring.

        Returns:
            Dictionary with learner stats and recommendations
        """
        cursor = self.db.cursor()

        # Get table name (each learner has its own stats table)
        stats_table = f"{self.name}_stats"

        try:
            cursor.execute(f"SELECT COUNT(*) FROM {stats_table}")
            total_contexts = cursor.fetchone()[0]

            cursor.execute(f"""
                SELECT SUM(total_sessions) FROM {stats_table}
                """)
            total_sessions = cursor.fetchone()[0] or 0

            return {
                "learner": self.name,
                "total_contexts": total_contexts,
                "total_sessions": total_sessions,
            }
        except Exception as e:
            logger.warning(f"RL: Failed to export metrics for {self.name}: {e}")
            return {"learner": self.name, "error": str(e)}
