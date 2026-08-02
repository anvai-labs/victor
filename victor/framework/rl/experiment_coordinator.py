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

"""Experiment coordinator for A/B testing RL policies.

This module provides infrastructure for safely testing new RL policies
against existing baselines before full deployment.

Key Features:
1. Consistent user/session assignment to variants
2. Statistical significance testing
3. Automatic rollout/rollback based on results
4. Shadow mode for risk-free evaluation

Experiment Lifecycle:
1. Define experiment with control/treatment variants
2. Configure traffic split and success metrics
3. Run experiment until statistical significance
4. Analyze results and decide on rollout
5. Gradually increase traffic or rollback

Sprint 5: Advanced RL Patterns
"""

import hashlib
from victor.core.json_utils import json_dumps, json_loads
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ExperimentStatus(str, Enum):
    """Experiment lifecycle status."""

    DRAFT = "draft"  # Not yet active
    RUNNING = "running"  # Actively collecting data
    PAUSED = "paused"  # Temporarily halted
    ANALYZING = "analyzing"  # Collecting complete, analyzing
    COMPLETED = "completed"  # Decision made
    ROLLED_OUT = "rolled_out"  # Treatment adopted
    ROLLED_BACK = "rolled_back"  # Control kept


class VariantType(str, Enum):
    """Experiment variant types."""

    CONTROL = "control"
    TREATMENT = "treatment"


@dataclass
class Variant:
    """Experiment variant configuration.

    Attributes:
        name: Variant name
        type: Control or treatment
        config: Variant-specific configuration
        description: Human-readable description
    """

    name: str
    type: VariantType
    config: Dict[str, Any] = field(default_factory=dict)
    description: str = ""


@dataclass
class ExperimentConfig:
    """Configuration for an A/B experiment.

    Attributes:
        experiment_id: Unique experiment identifier
        name: Human-readable name
        description: Experiment description
        control: Control variant
        treatment: Treatment variant
        traffic_split: Fraction of traffic to treatment (0-1)
        metrics: List of metrics to track
        min_samples_per_variant: Minimum samples before analysis
        significance_level: p-value threshold (e.g., 0.05)
        min_effect_size: Minimum detectable effect size
        max_duration_days: Maximum experiment duration
    """

    experiment_id: str
    name: str
    description: str
    control: Variant
    treatment: Variant
    traffic_split: float = 0.1  # 10% to treatment by default
    metrics: List[str] = field(default_factory=lambda: ["success_rate", "quality_score"])
    min_samples_per_variant: int = 100
    significance_level: float = 0.05
    min_effect_size: float = 0.05  # 5% improvement
    max_duration_days: int = 14


@dataclass
class VariantMetrics:
    """Accumulated metrics for a variant.

    Attributes:
        variant_name: Variant name
        sample_count: Number of samples
        success_count: Number of successes
        total_quality: Sum of quality scores
        total_latency_ms: Sum of latencies
        metric_sums: Sums for custom metrics
    """

    variant_name: str
    sample_count: int = 0
    success_count: int = 0
    total_quality: float = 0.0
    total_latency_ms: float = 0.0
    metric_sums: Dict[str, float] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Compute success rate."""
        if self.sample_count == 0:
            return 0.0
        return self.success_count / self.sample_count

    @property
    def avg_quality(self) -> float:
        """Compute average quality."""
        if self.sample_count == 0:
            return 0.0
        return self.total_quality / self.sample_count

    @property
    def avg_latency_ms(self) -> float:
        """Compute average latency."""
        if self.sample_count == 0:
            return 0.0
        return self.total_latency_ms / self.sample_count

    def get_metric(self, name: str) -> float:
        """Get average for a custom metric."""
        if self.sample_count == 0:
            return 0.0
        return self.metric_sums.get(name, 0.0) / self.sample_count


@dataclass
class ExperimentResult:
    """Result of experiment analysis.

    Attributes:
        experiment_id: Experiment identifier
        is_significant: Whether result is statistically significant
        treatment_better: Whether treatment outperforms control
        effect_size: Relative improvement (treatment - control) / control
        p_value: Statistical p-value
        confidence_interval: 95% confidence interval for effect
        recommendation: Recommended action
        details: Detailed analysis
    """

    experiment_id: str
    is_significant: bool
    treatment_better: bool
    effect_size: float
    p_value: float
    confidence_interval: Tuple[float, float]
    recommendation: str
    details: Dict[str, Any] = field(default_factory=dict)


class ExperimentCoordinator:
    """Coordinator for A/B testing of RL policies.

    Manages experiment lifecycle, variant assignment, metric collection,
    and statistical analysis.

    Usage:
        coordinator = ExperimentCoordinator(db_connection)

        # Define experiment
        config = ExperimentConfig(
            experiment_id="tool_selector_v2",
            name="Tool Selector V2",
            control=Variant("baseline", VariantType.CONTROL),
            treatment=Variant("rl_selector", VariantType.TREATMENT),
            traffic_split=0.1,
        )
        coordinator.create_experiment(config)

        # Assign variant for session
        variant = coordinator.assign_variant("tool_selector_v2", "session_123")

        # Record outcome
        coordinator.record_outcome(
            "tool_selector_v2",
            "session_123",
            success=True,
            quality_score=0.85,
            latency_ms=150.0,
        )

        # Check results
        result = coordinator.analyze_experiment("tool_selector_v2")
    """

    def __init__(self, db_connection: Optional[Any] = None):
        """Initialize experiment coordinator.

        Args:
            db_connection: Optional SQLite connection
        """
        self.db = db_connection

        # Active experiments
        self._experiments: Dict[str, ExperimentConfig] = {}

        # Experiment status
        self._status: Dict[str, ExperimentStatus] = {}

        # Variant metrics
        self._metrics: Dict[str, Dict[str, VariantMetrics]] = {}

        # Session assignments (for consistency)
        self._assignments: Dict[str, Dict[str, str]] = {}

        if db_connection:
            self._ensure_tables()
            self._load_state()

    def _ensure_tables(self) -> None:
        """Create tables for experiment state."""
        if not self.db:
            return

        cursor = self.db.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                config TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiment_metrics (
                experiment_id TEXT NOT NULL,
                variant_name TEXT NOT NULL,
                sample_count INTEGER NOT NULL DEFAULT 0,
                success_count INTEGER NOT NULL DEFAULT 0,
                total_quality REAL NOT NULL DEFAULT 0.0,
                total_latency_ms REAL NOT NULL DEFAULT 0.0,
                metric_sums TEXT NOT NULL DEFAULT '{}',
                updated_at TEXT NOT NULL,
                PRIMARY KEY (experiment_id, variant_name)
            )
            """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiment_assignments (
                experiment_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                variant_name TEXT NOT NULL,
                assigned_at TEXT NOT NULL,
                PRIMARY KEY (experiment_id, session_id)
            )
            """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiment_outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                variant_name TEXT NOT NULL,
                success INTEGER NOT NULL,
                quality_score REAL,
                latency_ms REAL,
                metrics TEXT,
                recorded_at TEXT NOT NULL
            )
            """)

        self.db.commit()

    def _load_state(self) -> None:
        """Load state from database."""
        if not self.db:
            return

        cursor = self.db.cursor()

        try:
            # Load experiments
            cursor.execute("SELECT * FROM experiments")
            for row in cursor.fetchall():
                row_dict = dict(row)
                config = json_loads(row_dict["config"])

                # Reconstruct config
                experiment_config = ExperimentConfig(
                    experiment_id=row_dict["experiment_id"],
                    name=row_dict["name"],
                    description=row_dict["description"] or "",
                    control=Variant(
                        name=config["control"]["name"],
                        type=VariantType(config["control"]["type"]),
                        config=config["control"].get("config", {}),
                    ),
                    treatment=Variant(
                        name=config["treatment"]["name"],
                        type=VariantType(config["treatment"]["type"]),
                        config=config["treatment"].get("config", {}),
                    ),
                    traffic_split=config.get("traffic_split", 0.1),
                    metrics=config.get("metrics", ["success_rate"]),
                    min_samples_per_variant=config.get("min_samples_per_variant", 100),
                    significance_level=config.get("significance_level", 0.05),
                )

                self._experiments[row_dict["experiment_id"]] = experiment_config
                self._status[row_dict["experiment_id"]] = ExperimentStatus(row_dict["status"])

            # Load metrics
            cursor.execute("SELECT * FROM experiment_metrics")
            for row in cursor.fetchall():
                row_dict = dict(row)
                exp_id = row_dict["experiment_id"]
                variant = row_dict["variant_name"]

                if exp_id not in self._metrics:
                    self._metrics[exp_id] = {}

                self._metrics[exp_id][variant] = VariantMetrics(
                    variant_name=variant,
                    sample_count=row_dict["sample_count"],
                    success_count=row_dict["success_count"],
                    total_quality=row_dict["total_quality"],
                    total_latency_ms=row_dict["total_latency_ms"],
                    metric_sums=json_loads(row_dict["metric_sums"]),
                )

            if self._experiments:
                logger.info(f"ExperimentCoordinator: Loaded {len(self._experiments)} experiments")

        except Exception as e:
            logger.debug(f"ExperimentCoordinator: Could not load state: {e}")

    def create_experiment(self, config: ExperimentConfig) -> bool:
        """Create a new experiment.

        Args:
            config: Experiment configuration

        Returns:
            True if created successfully
        """
        if config.experiment_id in self._experiments:
            logger.warning(f"Experiment {config.experiment_id} already exists, skipping")
            return False

        self._experiments[config.experiment_id] = config
        self._status[config.experiment_id] = ExperimentStatus.DRAFT

        # Initialize metrics
        self._metrics[config.experiment_id] = {
            config.control.name: VariantMetrics(variant_name=config.control.name),
            config.treatment.name: VariantMetrics(variant_name=config.treatment.name),
        }

        self._assignments[config.experiment_id] = {}

        # Save to database
        self._save_experiment(config)

        logger.info(f"ExperimentCoordinator: Created experiment {config.experiment_id}")
        return True

    def start_experiment(self, experiment_id: str) -> bool:
        """Start an experiment.

        Args:
            experiment_id: Experiment identifier

        Returns:
            True if started
        """
        if experiment_id not in self._experiments:
            return False

        if self._status[experiment_id] != ExperimentStatus.DRAFT:
            return False

        self._status[experiment_id] = ExperimentStatus.RUNNING
        self._update_status(experiment_id)

        logger.info(f"ExperimentCoordinator: Started experiment {experiment_id}")
        return True

    def pause_experiment(self, experiment_id: str) -> bool:
        """Pause an experiment.

        Args:
            experiment_id: Experiment identifier

        Returns:
            True if paused
        """
        if experiment_id not in self._experiments:
            return False

        if self._status[experiment_id] != ExperimentStatus.RUNNING:
            return False

        self._status[experiment_id] = ExperimentStatus.PAUSED
        self._update_status(experiment_id)

        logger.info(f"ExperimentCoordinator: Paused experiment {experiment_id}")
        return True

    def assign_variant(self, experiment_id: str, session_id: str) -> Optional[str]:
        """Assign a variant to a session.

        Uses consistent hashing to ensure the same session always
        gets the same variant.

        Args:
            experiment_id: Experiment identifier
            session_id: Session identifier

        Returns:
            Variant name or None if experiment not running
        """
        if experiment_id not in self._experiments:
            return None

        if self._status.get(experiment_id) != ExperimentStatus.RUNNING:
            return None

        # Check for existing assignment
        if experiment_id in self._assignments:
            if session_id in self._assignments[experiment_id]:
                return self._assignments[experiment_id][session_id]

        config = self._experiments[experiment_id]

        # Consistent hash for assignment
        hash_input = f"{experiment_id}:{session_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest()[:8], 16)
        hash_fraction = hash_value / 0xFFFFFFFF

        # Assign based on traffic split
        if hash_fraction < config.traffic_split:
            variant = config.treatment.name
        else:
            variant = config.control.name

        # Store assignment
        if experiment_id not in self._assignments:
            self._assignments[experiment_id] = {}
        self._assignments[experiment_id][session_id] = variant

        # Save to database
        self._save_assignment(experiment_id, session_id, variant)

        return variant

    def record_outcome(
        self,
        experiment_id: str,
        session_id: str,
        success: bool,
        quality_score: float = 0.5,
        latency_ms: float = 0.0,
        custom_metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """Record an outcome for an experiment.

        Args:
            experiment_id: Experiment identifier
            session_id: Session identifier
            success: Whether the task succeeded
            quality_score: Quality score (0-1)
            latency_ms: Latency in milliseconds
            custom_metrics: Additional metrics
        """
        if experiment_id not in self._experiments:
            return

        # Get variant assignment
        variant = self._assignments.get(experiment_id, {}).get(session_id)
        if not variant:
            return

        # Update metrics
        metrics = self._metrics[experiment_id][variant]
        metrics.sample_count += 1
        metrics.success_count += 1 if success else 0
        metrics.total_quality += quality_score
        metrics.total_latency_ms += latency_ms

        if custom_metrics:
            for name, value in custom_metrics.items():
                metrics.metric_sums[name] = metrics.metric_sums.get(name, 0.0) + value

        # Save to database
        self._save_outcome(experiment_id, session_id, variant, success, quality_score, latency_ms)
        self._save_metrics(experiment_id, variant, metrics)

    def analyze_experiment(self, experiment_id: str) -> Optional[ExperimentResult]:
        """Analyze experiment results.

        Performs statistical significance testing using a two-proportion
        z-test for success rates.

        Args:
            experiment_id: Experiment identifier

        Returns:
            ExperimentResult or None if insufficient data
        """
        if experiment_id not in self._experiments:
            return None

        config = self._experiments[experiment_id]
        control_metrics = self._metrics[experiment_id].get(config.control.name)
        treatment_metrics = self._metrics[experiment_id].get(config.treatment.name)

        if not control_metrics or not treatment_metrics:
            return None

        # Check minimum sample size
        if (
            control_metrics.sample_count < config.min_samples_per_variant
            or treatment_metrics.sample_count < config.min_samples_per_variant
        ):
            return ExperimentResult(
                experiment_id=experiment_id,
                is_significant=False,
                treatment_better=False,
                effect_size=0.0,
                p_value=1.0,
                confidence_interval=(0.0, 0.0),
                recommendation="Insufficient samples - continue experiment",
                details={
                    "control_samples": control_metrics.sample_count,
                    "treatment_samples": treatment_metrics.sample_count,
                    "min_required": config.min_samples_per_variant,
                },
            )

        # Two-proportion z-test
        p_c = control_metrics.success_rate
        p_t = treatment_metrics.success_rate
        n_c = control_metrics.sample_count
        n_t = treatment_metrics.sample_count

        # Pooled proportion
        p_pooled = (control_metrics.success_count + treatment_metrics.success_count) / (n_c + n_t)

        # Standard error
        se = math.sqrt(p_pooled * (1 - p_pooled) * (1 / n_c + 1 / n_t))

        if se == 0:
            z_score = 0.0
        else:
            z_score = (p_t - p_c) / se

        # P-value (two-tailed)
        p_value = 2 * (1 - self._norm_cdf(abs(z_score)))

        # Effect size
        effect_size = (p_t - p_c) / max(p_c, 0.001)

        # 95% confidence interval
        margin = 1.96 * se
        ci_low = (p_t - p_c) - margin
        ci_high = (p_t - p_c) + margin

        # Determine significance and recommendation
        is_significant = p_value < config.significance_level
        treatment_better = p_t > p_c

        if is_significant and treatment_better and effect_size >= config.min_effect_size:
            recommendation = "Roll out treatment - significant improvement detected"
        elif is_significant and not treatment_better:
            recommendation = "Keep control - treatment performed worse"
        elif not is_significant:
            recommendation = "Continue experiment - results not yet significant"
        else:
            recommendation = "Effect size too small - consider keeping control"

        return ExperimentResult(
            experiment_id=experiment_id,
            is_significant=is_significant,
            treatment_better=treatment_better,
            effect_size=effect_size,
            p_value=p_value,
            confidence_interval=(ci_low, ci_high),
            recommendation=recommendation,
            details={
                "control": {
                    "samples": n_c,
                    "success_rate": p_c,
                    "avg_quality": control_metrics.avg_quality,
                },
                "treatment": {
                    "samples": n_t,
                    "success_rate": p_t,
                    "avg_quality": treatment_metrics.avg_quality,
                },
                "z_score": z_score,
            },
        )

    def _norm_cdf(self, x: float) -> float:
        """Approximate cumulative distribution function for standard normal.

        Args:
            x: Z-score

        Returns:
            Probability P(Z <= x)
        """
        # Approximation using error function
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    def rollout_treatment(self, experiment_id: str) -> bool:
        """Mark experiment as rolled out (treatment adopted).

        Args:
            experiment_id: Experiment identifier

        Returns:
            True if successful
        """
        if experiment_id not in self._experiments:
            return False

        self._status[experiment_id] = ExperimentStatus.ROLLED_OUT
        self._update_status(experiment_id)

        logger.info(f"ExperimentCoordinator: Rolled out {experiment_id}")
        return True

    def rollback_experiment(self, experiment_id: str) -> bool:
        """Mark experiment as rolled back (control kept).

        Args:
            experiment_id: Experiment identifier

        Returns:
            True if successful
        """
        if experiment_id not in self._experiments:
            return False

        self._status[experiment_id] = ExperimentStatus.ROLLED_BACK
        self._update_status(experiment_id)

        logger.info(f"ExperimentCoordinator: Rolled back {experiment_id}")
        return True

    def get_experiment_status(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of an experiment.

        Args:
            experiment_id: Experiment identifier

        Returns:
            Status dictionary or None
        """
        if experiment_id not in self._experiments:
            return None

        config = self._experiments[experiment_id]
        status = self._status[experiment_id]
        metrics = self._metrics.get(experiment_id, {})
        control_config = config.control.config if isinstance(config.control.config, dict) else {}
        treatment_config = (
            config.treatment.config if isinstance(config.treatment.config, dict) else {}
        )
        section_name = control_config.get("section_name") or treatment_config.get("section_name")
        provider = control_config.get("provider") or treatment_config.get("provider")

        return {
            "experiment_id": experiment_id,
            "name": config.name,
            "status": status.value,
            "section_name": section_name,
            "provider": provider,
            "traffic_split": config.traffic_split,
            "control": {
                "name": config.control.name,
                "strategy_name": control_config.get("strategy_name"),
                "samples": metrics.get(
                    config.control.name, VariantMetrics(config.control.name)
                ).sample_count,
                "success_rate": metrics.get(
                    config.control.name, VariantMetrics(config.control.name)
                ).success_rate,
            },
            "treatment": {
                "name": config.treatment.name,
                "strategy_name": treatment_config.get("strategy_name"),
                "samples": metrics.get(
                    config.treatment.name, VariantMetrics(config.treatment.name)
                ).sample_count,
                "success_rate": metrics.get(
                    config.treatment.name, VariantMetrics(config.treatment.name)
                ).success_rate,
            },
        }

    def list_experiments(self) -> List[Dict[str, Any]]:
        """List all experiments.

        Returns:
            List of experiment summaries
        """
        return [self.get_experiment_status(exp_id) for exp_id in self._experiments]

    def _save_experiment(self, config: ExperimentConfig) -> None:
        """Save experiment to database."""
        if not self.db:
            return

        cursor = self.db.cursor()
        timestamp = datetime.now().isoformat()

        config_json = json_dumps(
            {
                "control": {
                    "name": config.control.name,
                    "type": config.control.type.value,
                    "config": config.control.config,
                },
                "treatment": {
                    "name": config.treatment.name,
                    "type": config.treatment.type.value,
                    "config": config.treatment.config,
                },
                "traffic_split": config.traffic_split,
                "metrics": config.metrics,
                "min_samples_per_variant": config.min_samples_per_variant,
                "significance_level": config.significance_level,
            }
        )

        cursor.execute(
            """
            INSERT OR REPLACE INTO experiments
            (experiment_id, name, description, config, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                config.experiment_id,
                config.name,
                config.description,
                config_json,
                self._status[config.experiment_id].value,
                timestamp,
                timestamp,
            ),
        )

        self.db.commit()

    def _update_status(self, experiment_id: str) -> None:
        """Update experiment status in database."""
        if not self.db:
            return

        cursor = self.db.cursor()
        timestamp = datetime.now().isoformat()

        cursor.execute(
            """
            UPDATE experiments
            SET status = ?, updated_at = ?
            WHERE experiment_id = ?
            """,
            (self._status[experiment_id].value, timestamp, experiment_id),
        )

        self.db.commit()

    def _save_assignment(self, experiment_id: str, session_id: str, variant: str) -> None:
        """Save variant assignment to database."""
        if not self.db:
            return

        cursor = self.db.cursor()
        timestamp = datetime.now().isoformat()

        cursor.execute(
            """
            INSERT OR REPLACE INTO experiment_assignments
            (experiment_id, session_id, variant_name, assigned_at)
            VALUES (?, ?, ?, ?)
            """,
            (experiment_id, session_id, variant, timestamp),
        )

        self.db.commit()

    def _save_outcome(
        self,
        experiment_id: str,
        session_id: str,
        variant: str,
        success: bool,
        quality_score: float,
        latency_ms: float,
    ) -> None:
        """Save outcome to database."""
        if not self.db:
            return

        cursor = self.db.cursor()
        timestamp = datetime.now().isoformat()

        cursor.execute(
            """
            INSERT INTO experiment_outcomes
            (experiment_id, session_id, variant_name, success,
             quality_score, latency_ms, recorded_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                experiment_id,
                session_id,
                variant,
                1 if success else 0,
                quality_score,
                latency_ms,
                timestamp,
            ),
        )

        self.db.commit()

    def _save_metrics(self, experiment_id: str, variant: str, metrics: VariantMetrics) -> None:
        """Save metrics to database."""
        if not self.db:
            return

        cursor = self.db.cursor()
        timestamp = datetime.now().isoformat()

        cursor.execute(
            """
            INSERT OR REPLACE INTO experiment_metrics
            (experiment_id, variant_name, sample_count, success_count,
             total_quality, total_latency_ms, metric_sums, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                experiment_id,
                variant,
                metrics.sample_count,
                metrics.success_count,
                metrics.total_quality,
                metrics.total_latency_ms,
                json_dumps(metrics.metric_sums),
                timestamp,
            ),
        )

        self.db.commit()

    def export_metrics(self) -> Dict[str, Any]:
        """Export coordinator metrics.

        Returns:
            Dictionary with metrics
        """
        return {
            "total_experiments": len(self._experiments),
            "by_status": {
                status.value: sum(1 for s in self._status.values() if s == status)
                for status in ExperimentStatus
            },
            "total_assignments": sum(len(assigns) for assigns in self._assignments.values()),
        }


# Global singleton
_experiment_coordinator: Optional[ExperimentCoordinator] = None


def get_experiment_coordinator(
    db_connection: Optional[Any] = None,
) -> ExperimentCoordinator:
    """Get global experiment coordinator (lazy init).

    Args:
        db_connection: Optional database connection

    Returns:
        ExperimentCoordinator singleton
    """
    global _experiment_coordinator
    if _experiment_coordinator is None:
        _experiment_coordinator = ExperimentCoordinator(db_connection)
    return _experiment_coordinator
