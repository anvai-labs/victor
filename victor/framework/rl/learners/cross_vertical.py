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

"""Cross-vertical transfer learning for sharing patterns across verticals.

This learner identifies common patterns (e.g., "edit tasks benefit from BUILD mode")
and applies them across verticals. It enables knowledge transfer between coding,
devops, data_analysis, and research verticals.

Strategy:
- Analyze RL outcomes across all verticals
- Identify patterns with consistent behavior (high confidence)
- Generate recommendations applicable to new/underrepresented verticals
- Reduce cold-start problem for new verticals

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                 CrossVerticalLearner                             │
    ├─────────────────────────────────────────────────────────────────┤
    │  1. Pattern Discovery: Find consistent patterns across verticals│
    │  2. Confidence Scoring: Score patterns by cross-vertical support│
    │  3. Recommendation: Apply patterns to new verticals             │
    │  4. Observation: Track pattern application success              │
    └─────────────────────────────────────────────────────────────────┘

Usage:
    from victor.framework.rl.learners.cross_vertical import CrossVerticalLearner

    learner = CrossVerticalLearner("cross_vertical", db_connection)

    # Get patterns applicable to a specific vertical
    patterns = learner.get_shared_patterns(target_vertical="devops")

    # Apply patterns in a cold-start scenario
    recommendation = learner.get_recommendation(
        task_type="edit",
        target_vertical="research",
    )
"""

from __future__ import annotations

from victor.core.json_utils import json_dumps, json_loads
from json import JSONDecodeError
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from victor.framework.rl.base import BaseLearner, RLOutcome, RLRecommendation
from victor.core.schema import Tables

logger = logging.getLogger(__name__)


@dataclass
class SharedPattern:
    """A pattern that applies across verticals.

    Attributes:
        task_type: Task type this pattern applies to
        pattern_name: Descriptive name for the pattern
        avg_quality: Average quality score across verticals
        confidence: Confidence in the pattern (0.0-1.0)
        source_verticals: Verticals that contributed to this pattern
        recommended_mode: Recommended mode for this task type
        recommendation: Textual recommendation
        sample_count: Number of samples supporting this pattern
    """

    task_type: str
    pattern_name: str
    avg_quality: float
    confidence: float
    source_verticals: List[str] = field(default_factory=list)
    recommended_mode: Optional[str] = None
    recommendation: str = ""
    sample_count: int = 0


class CrossVerticalLearner(BaseLearner):
    """Shares learning patterns across verticals.

    Identifies common patterns (e.g., "edit tasks benefit from BUILD mode")
    and applies them across verticals. This enables knowledge transfer
    between coding, devops, data_analysis, and research verticals.

    Attributes:
        name: Learner name (should be "cross_vertical")
        db: SQLite database connection
        learning_rate: Not used directly (pattern-based learning)
        min_samples_for_pattern: Minimum samples to consider a pattern valid
        min_verticals_for_transfer: Minimum verticals to consider a pattern transferable
    """

    # Minimum samples to consider a pattern valid
    MIN_SAMPLES_FOR_PATTERN = 10

    # Minimum verticals to consider a pattern transferable
    MIN_VERTICALS_FOR_TRANSFER = 2

    # Confidence threshold for high-confidence patterns
    HIGH_CONFIDENCE_THRESHOLD = 0.7

    def __init__(
        self,
        name: str,
        db_connection: Any,
        learning_rate: float = 0.1,
        provider_adapter: Optional[Any] = None,
        min_samples: int = MIN_SAMPLES_FOR_PATTERN,
        min_verticals: int = MIN_VERTICALS_FOR_TRANSFER,
    ):
        """Initialize cross-vertical learner.

        Args:
            name: Learner name
            db_connection: SQLite database connection
            learning_rate: Not directly used (pattern-based)
            provider_adapter: Optional provider adapter
            min_samples: Minimum samples for valid pattern
            min_verticals: Minimum verticals for transfer
        """
        super().__init__(
            name=name,
            db_connection=db_connection,
            learning_rate=learning_rate,
            provider_adapter=provider_adapter,
        )
        self._min_samples = min_samples
        self._min_verticals = min_verticals
        self._ensure_tables()

        logger.debug(
            f"CrossVerticalLearner initialized: "
            f"min_samples={min_samples}, min_verticals={min_verticals}"
        )

    def _ensure_tables(self) -> None:
        """Ensure required database tables exist."""
        cursor = self.db.cursor()

        # Table for storing discovered patterns
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {Tables.RL_PATTERN} (
                pattern_id TEXT PRIMARY KEY,
                task_type TEXT NOT NULL,
                pattern_name TEXT NOT NULL,
                avg_quality REAL NOT NULL,
                confidence REAL NOT NULL,
                source_verticals TEXT NOT NULL,
                recommended_mode TEXT,
                recommendation TEXT,
                sample_count INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

        # Table for tracking pattern applications
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {Tables.RL_PATTERN_USE} (
                application_id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_id TEXT NOT NULL,
                target_vertical TEXT NOT NULL,
                applied_at TEXT NOT NULL,
                success INTEGER,
                quality_score REAL,
                feedback TEXT
            )
        """)

        self.db.commit()

    def get_shared_patterns(
        self,
        target_vertical: Optional[str] = None,
        min_confidence: float = 0.5,
    ) -> List[SharedPattern]:
        """Get patterns that apply across verticals.

        Args:
            target_vertical: Optional target vertical to exclude from sources
            min_confidence: Minimum confidence for returned patterns

        Returns:
            List of SharedPattern instances
        """
        cursor = self.db.cursor()

        # Query for task types with consistent behavior across verticals
        cursor.execute(
            f"""
            SELECT
                task_type,
                AVG(quality_score) as avg_quality,
                COUNT(*) as sample_count,
                COUNT(DISTINCT vertical) as vertical_count,
                GROUP_CONCAT(DISTINCT vertical) as source_verticals
            FROM {Tables.RL_OUTCOME}
            WHERE quality_score IS NOT NULL
            GROUP BY task_type
            HAVING vertical_count >= ? AND sample_count >= ?
            ORDER BY avg_quality DESC
        """,
            (self._min_verticals, self._min_samples),
        )

        patterns = []
        for row in cursor.fetchall():
            task_type = row[0]
            avg_quality = row[1]
            sample_count = row[2]
            vertical_count = row[3]
            source_verticals = row[4].split(",") if row[4] else []

            # Skip if target vertical is the only source
            if target_vertical and target_vertical in source_verticals:
                if len(source_verticals) == 1:
                    continue

            # Calculate confidence based on sample count and vertical coverage
            confidence = min(
                sample_count / (self._min_samples * 2),  # Sample-based confidence
                vertical_count / 4,  # Vertical-based confidence
                1.0,
            )

            if confidence < min_confidence:
                continue

            # Get recommended mode for this task type
            recommended_mode = self._get_best_mode_for_task(task_type)

            pattern = SharedPattern(
                task_type=task_type,
                pattern_name=f"{task_type}_cross_vertical_pattern",
                avg_quality=avg_quality,
                confidence=confidence,
                source_verticals=source_verticals,
                recommended_mode=recommended_mode,
                recommendation=self._generate_recommendation(
                    task_type, avg_quality, recommended_mode
                ),
                sample_count=sample_count,
            )
            patterns.append(pattern)

        logger.debug(
            f"Found {len(patterns)} shared patterns "
            f"(target_vertical={target_vertical}, min_confidence={min_confidence})"
        )

        return patterns

    def _get_best_mode_for_task(self, task_type: str) -> Optional[str]:
        """Get the best performing mode for a task type.

        Args:
            task_type: Task type to analyze

        Returns:
            Best mode or None
        """
        cursor = self.db.cursor()

        # Get mode with highest quality for this task type
        # Mode is stored in metadata as JSON, so we look at outcomes that mention modes
        cursor.execute(
            f"""
            SELECT
                metadata,
                AVG(quality_score) as avg_quality,
                COUNT(*) as count
            FROM {Tables.RL_OUTCOME}
            WHERE task_type = ? AND quality_score IS NOT NULL
            GROUP BY metadata
            HAVING count >= 5
            ORDER BY avg_quality DESC
            LIMIT 1
        """,
            (task_type,),
        )

        row = cursor.fetchone()
        if row and row[0]:
            try:
                import json

                metadata = json_loads(row[0])
                return metadata.get("mode")
            except Exception:
                pass

        return None

    def _generate_recommendation(
        self,
        task_type: str,
        avg_quality: float,
        recommended_mode: Optional[str],
    ) -> str:
        """Generate human-readable recommendation.

        Args:
            task_type: Task type
            avg_quality: Average quality score
            recommended_mode: Recommended mode

        Returns:
            Recommendation text
        """
        quality_level = "high" if avg_quality > 0.7 else "moderate" if avg_quality > 0.5 else "low"

        if recommended_mode:
            return (
                f"For {task_type} tasks, {recommended_mode.upper()} mode typically yields "
                f"{quality_level} quality results (avg: {avg_quality:.2f}). "
                f"This pattern is consistent across multiple verticals."
            )
        else:
            return (
                f"For {task_type} tasks, expect {quality_level} quality results "
                f"(avg: {avg_quality:.2f}). Pattern observed across multiple verticals."
            )

    def get_recommendation(
        self,
        task_type: str,
        target_vertical: str,
    ) -> RLRecommendation:
        """Get recommendation for a task type in a target vertical.

        Uses cross-vertical patterns to provide recommendations, especially
        useful for cold-start scenarios in new verticals.

        Args:
            task_type: Type of task
            target_vertical: Target vertical

        Returns:
            RLRecommendation with cross-vertical insights
        """
        # Check if target vertical has enough data
        cursor = self.db.cursor()
        cursor.execute(
            f"""
            SELECT COUNT(*)
            FROM {Tables.RL_OUTCOME}
            WHERE vertical = ? AND task_type = ?
        """,
            (target_vertical, task_type),
        )

        local_count = cursor.fetchone()[0]

        # If local data is sufficient, return low-confidence baseline
        if local_count >= self._min_samples:
            return RLRecommendation(
                value=0.5,
                confidence=0.3,
                reason=f"Sufficient local data ({local_count} samples) - use local learner",
                sample_size=local_count,
                is_baseline=True,
            )

        # Get cross-vertical patterns
        patterns = self.get_shared_patterns(
            target_vertical=target_vertical,
            min_confidence=0.5,
        )

        # Find pattern for this task type
        matching_pattern = next(
            (p for p in patterns if p.task_type == task_type),
            None,
        )

        if matching_pattern:
            return RLRecommendation(
                value=matching_pattern.avg_quality,
                confidence=matching_pattern.confidence * 0.8,  # Slightly reduce for transfer
                reason=(
                    f"Cross-vertical pattern from {', '.join(matching_pattern.source_verticals)}: "
                    f"{matching_pattern.recommendation}"
                ),
                sample_size=matching_pattern.sample_count,
                is_baseline=False,
            )

        # No matching pattern found
        return RLRecommendation(
            value=0.5,
            confidence=0.2,
            reason=f"No cross-vertical pattern for task type '{task_type}'",
            sample_size=0,
            is_baseline=True,
        )

    def record_pattern_application(
        self,
        pattern_id: str,
        target_vertical: str,
        success: bool,
        quality_score: float,
        feedback: Optional[str] = None,
    ) -> None:
        """Record the application of a pattern to track transfer success.

        Args:
            pattern_id: ID of the applied pattern
            target_vertical: Vertical where pattern was applied
            success: Whether application was successful
            quality_score: Quality score of the result
            feedback: Optional feedback text
        """
        cursor = self.db.cursor()
        cursor.execute(
            f"""
            INSERT INTO {Tables.RL_PATTERN_USE}
            (pattern_id, target_vertical, applied_at, success, quality_score, feedback)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (
                pattern_id,
                target_vertical,
                datetime.now().isoformat(),
                1 if success else 0,
                quality_score,
                feedback,
            ),
        )
        self.db.commit()

        logger.debug(
            f"Recorded pattern application: pattern={pattern_id}, "
            f"vertical={target_vertical}, success={success}"
        )

    def get_transfer_success_rate(
        self,
        target_vertical: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get success rate of cross-vertical pattern transfers.

        Args:
            target_vertical: Optional filter by target vertical

        Returns:
            Dict with transfer statistics
        """
        cursor = self.db.cursor()

        if target_vertical:
            cursor.execute(
                f"""
                SELECT
                    COUNT(*) as total,
                    SUM(success) as successes,
                    AVG(quality_score) as avg_quality
                FROM {Tables.RL_PATTERN_USE}
                WHERE target_vertical = ?
            """,
                (target_vertical,),
            )
        else:
            cursor.execute(f"""
                SELECT
                    COUNT(*) as total,
                    SUM(success) as successes,
                    AVG(quality_score) as avg_quality
                FROM {Tables.RL_PATTERN_USE}
            """)

        row = cursor.fetchone()
        total = row[0] or 0
        successes = row[1] or 0
        avg_quality = row[2] or 0.0

        return {
            "total_applications": total,
            "successful_applications": successes,
            "success_rate": successes / total if total > 0 else 0.0,
            "avg_quality": avg_quality,
            "target_vertical": target_vertical,
        }

    def record_outcome(self, outcome: RLOutcome) -> None:
        """Record an outcome and update learned values.

        CrossVerticalLearner primarily learns from aggregate patterns,
        but this method allows recording individual outcomes for pattern discovery.

        Args:
            outcome: RL outcome to record
        """
        # Pattern discovery is done lazily via get_shared_patterns
        # Individual outcomes are already stored by the coordinator
        logger.debug(
            f"CrossVerticalLearner received outcome: "
            f"task={outcome.task_type}, vertical={outcome.vertical}"
        )

    def _compute_reward(self, outcome: RLOutcome) -> float:
        """Compute reward signal from outcome.

        CrossVerticalLearner doesn't use traditional rewards; it analyzes
        aggregate patterns instead.

        Args:
            outcome: RL outcome

        Returns:
            Quality score as reward (0.0-1.0)
        """
        return outcome.quality_score

    # ------------------------------------------------------------------
    # Priority 4 Phase 3: Transfer Learning — pattern export / import
    # ------------------------------------------------------------------

    def export_patterns(
        self, repo_id: Optional[str] = None, min_confidence: float = 0.5
    ) -> Dict[str, Any]:
        """Serialize learned cross-vertical patterns for cross-project transfer.

        Exports only patterns above min_confidence so that noise from sparse
        data is not transferred to other projects.

        Args:
            repo_id: Scope export to a specific repo (None = all)
            min_confidence: Minimum pattern confidence to include

        Returns:
            Dict ready for json_dumps, importable via import_patterns()
        """
        # Source 1: dynamically discovered patterns from rl_outcome
        try:
            discovered = self.get_shared_patterns(min_confidence=min_confidence)
        except Exception as e:
            logger.debug("cross_vertical: export_patterns — get_shared_patterns failed: %s", e)
            discovered = []

        # Source 2: explicitly stored patterns in rl_pattern table (includes imports)
        stored: List[SharedPattern] = []
        try:
            cursor = self.db.cursor()
            cursor.execute(
                f"""
                SELECT task_type, pattern_name, avg_quality, confidence,
                       source_verticals, recommended_mode, recommendation, sample_count
                FROM {Tables.RL_PATTERN}
                WHERE confidence >= ?
                """,
                (min_confidence,),
            )
            for row in cursor.fetchall():
                rd = dict(row)
                try:
                    src_verts = json_loads(rd.get("source_verticals") or "[]")
                except (JSONDecodeError, TypeError):
                    src_verts = []
                stored.append(
                    SharedPattern(
                        task_type=rd["task_type"],
                        pattern_name=rd.get("pattern_name", "stored"),
                        avg_quality=rd["avg_quality"],
                        confidence=rd["confidence"],
                        source_verticals=src_verts,
                        recommended_mode=rd.get("recommended_mode"),
                        recommendation=rd.get("recommendation", ""),
                        sample_count=rd.get("sample_count", 0),
                    )
                )
        except Exception as e:
            logger.debug("cross_vertical: export_patterns — rl_pattern query failed: %s", e)

        # Merge, deduplicating by (task_type, pattern_name)
        seen: set = set()
        patterns: List[SharedPattern] = []
        for p in discovered + stored:
            key = (p.task_type, p.pattern_name)
            if key not in seen:
                seen.add(key)
                patterns.append(p)

        exported = []
        for p in patterns:
            exported.append(
                {
                    "task_type": p.task_type,
                    "pattern_name": p.pattern_name,
                    "avg_quality": p.avg_quality,
                    "confidence": p.confidence,
                    "source_verticals": p.source_verticals,
                    "recommended_mode": p.recommended_mode,
                    "recommendation": p.recommendation,
                    "sample_count": p.sample_count,
                }
            )

        return {
            "schema_version": 1,
            "exported_at": datetime.now().isoformat(),
            "source_repo_id": repo_id,
            "pattern_count": len(exported),
            "patterns": exported,
        }

    def import_patterns(
        self,
        data: Dict[str, Any],
        source_repo_id: Optional[str] = None,
        confidence_decay: float = 0.8,
    ) -> int:
        """Load patterns exported from another project into this learner's DB.

        Applies confidence_decay so imported patterns have lower weight than
        locally-learned ones; they serve as warm-start priors, not ground truth.

        Args:
            data: Dict returned by export_patterns() (must have "patterns" key)
            source_repo_id: Identifier of the source project (for provenance)
            confidence_decay: Multiply imported confidence by this factor (default 0.8)

        Returns:
            Number of patterns imported
        """
        if data.get("schema_version") != 1:
            logger.warning("cross_vertical: unknown export schema version, skipping import")
            return 0

        patterns = data.get("patterns", [])
        if not patterns:
            return 0

        cursor = self.db.cursor()
        imported = 0
        now = datetime.now().isoformat()

        for p in patterns:
            try:
                pattern_id = (
                    f"imported_{source_repo_id or 'ext'}_{p['task_type']}_{p['pattern_name']}"
                )
                decayed_confidence = p["confidence"] * confidence_decay
                source_verticals = p.get("source_verticals", [])

                cursor.execute(
                    f"""
                    INSERT OR IGNORE INTO {Tables.RL_PATTERN}
                    (pattern_id, task_type, pattern_name, avg_quality, confidence,
                     source_verticals, recommended_mode, recommendation,
                     sample_count, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        pattern_id,
                        p["task_type"],
                        p.get("pattern_name", "imported"),
                        p["avg_quality"],
                        decayed_confidence,
                        json_dumps(source_verticals),
                        p.get("recommended_mode"),
                        p.get("recommendation", ""),
                        int(p.get("sample_count", 0) * confidence_decay),
                        now,
                        now,
                    ),
                )
                if cursor.rowcount > 0:
                    imported += 1
            except Exception as e:
                logger.debug("cross_vertical: import failed for pattern %s: %s", p, e)

        self.db.commit()
        logger.info(
            "cross_vertical: imported %d/%d patterns from %s",
            imported,
            len(patterns),
            source_repo_id or "external",
        )
        return imported

    def adapt_patterns(
        self,
        source_vertical: str,
        target_vertical: str,
        min_confidence: float = 0.6,
    ) -> List[RLRecommendation]:
        """Generate adaptation recommendations for target_vertical based on source_vertical.

        Queries patterns learned in source_vertical and re-emits them as
        recommendations for target_vertical with a confidence penalty for
        the domain shift.

        Args:
            source_vertical: Vertical to transfer from
            target_vertical: Vertical to transfer to
            min_confidence: Minimum source pattern confidence

        Returns:
            List of RLRecommendations adapted for target_vertical
        """
        try:
            cursor = self.db.cursor()
            cursor.execute(
                f"""
                SELECT task_type, AVG(quality_score) as avg_quality, COUNT(*) as cnt
                FROM {Tables.RL_OUTCOME}
                WHERE vertical = ?
                  AND quality_score IS NOT NULL
                GROUP BY task_type
                HAVING cnt >= ?
                """,
                (source_vertical, self._min_samples),
            )
            rows = cursor.fetchall()
        except Exception as e:
            logger.debug("cross_vertical: adapt_patterns query failed: %s", e)
            return []

        recommendations = []
        for row in rows:
            row_dict = dict(row)
            task_type = row_dict["task_type"]
            avg_quality = row_dict["avg_quality"]
            cnt = row_dict["cnt"]

            confidence = min(0.75, cnt / (cnt + 20))  # Bayesian shrinkage
            if confidence < min_confidence:
                continue

            # Apply domain-shift penalty
            adapted_confidence = confidence * 0.85

            recommendations.append(
                RLRecommendation(
                    value=avg_quality,
                    confidence=adapted_confidence,
                    reason=(
                        f"Adapted from {source_vertical} → {target_vertical}: "
                        f"avg quality {avg_quality:.2f} on {cnt} outcomes"
                    ),
                    sample_size=cnt,
                    is_baseline=False,
                    metadata={
                        "transfer_type": "domain_adaptation",
                        "source_vertical": source_vertical,
                        "target_vertical": target_vertical,
                        "task_type": task_type,
                        "confidence_decay": 0.85,
                    },
                )
            )

        return recommendations


__all__ = [
    "CrossVerticalLearner",
    "SharedPattern",
]
