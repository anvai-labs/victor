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

"""RL learner for per-provider grounding/hallucination threshold optimization.

This learner uses Thompson Sampling (Bayesian optimization) to learn optimal
confidence thresholds for hallucination detection per provider.

Problem:
- Static thresholds cause false positives (overly strict) or missed
  hallucinations (too lenient)
- Different providers have different hallucination patterns

Strategy:
- Context: (provider, model, response_type)
- Action: Select threshold from discretized range [0.5, 0.95]
- Reward: +0.1 correct, -1.0 false positive, -2.0 false negative

Thompson Sampling is ideal here because:
1. Good exploration/exploitation balance for continuous optimization
2. Naturally handles uncertainty
3. Works well with sparse rewards

Sprint 3: Cache & Grounding Learners
"""

import logging
import math
import random
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from victor.framework.rl.base import BaseLearner, RLOutcome, RLRecommendation
from victor.core.schema import Tables
from victor.framework.rl.migration import RLTableMigrator

logger = logging.getLogger(__name__)


class GroundingThresholdLearner(BaseLearner):
    """Learn optimal grounding thresholds using Thompson Sampling.

    Uses Beta distribution conjugate prior for threshold selection.
    Each (provider, response_type) pair maintains its own Beta parameters.

    Attributes:
        name: Always "grounding_threshold"
        db: SQLite database connection
        learning_rate: Update rate for parameters, default 0.1
    """

    # Discretized threshold levels
    THRESHOLD_LEVELS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

    # Default threshold when no data available
    DEFAULT_THRESHOLD = 0.70

    # Response types for state categorization
    RESPONSE_TYPES = [
        "code_generation",
        "explanation",
        "analysis",
        "edit",
        "search",
        "general",
    ]

    # Minimum observations before confident recommendation
    MIN_SAMPLES_FOR_CONFIDENCE = 10

    # Beta prior parameters (weakly informative)
    PRIOR_ALPHA = 1.0
    PRIOR_BETA = 1.0

    def __init__(
        self,
        name: str,
        db_connection: Any,
        learning_rate: float = 0.1,
        provider_adapter: Optional[Any] = None,
    ):
        """Initialize grounding threshold learner.

        Args:
            name: Learner name (should be "grounding_threshold")
            db_connection: SQLite database connection
            learning_rate: Parameter update rate (default 0.1)
            provider_adapter: Optional provider adapter
        """
        super().__init__(name, db_connection, learning_rate, provider_adapter)

        # Beta distribution parameters: (alpha, beta) for each (provider, response_type, threshold)
        # Alpha = successes + prior, Beta = failures + prior
        self._beta_params: Dict[str, Dict[float, Tuple[float, float]]] = {}

        # Track outcomes for analysis
        self._fp_rates: Dict[str, float] = {}  # False positive rates per provider
        self._fn_rates: Dict[str, float] = {}  # False negative rates per provider
        self._total_decisions: int = 0

        # Load state from database
        self._load_state()

    def _ensure_tables(self) -> None:
        """Migrate legacy per-learner tables to unified RL tables."""
        RLTableMigrator(self.db).run_if_needed(
            self.name, RLTableMigrator.migrate_grounding_threshold
        )

    def _load_state(self) -> None:
        """Load state from database."""
        cursor = self.db.cursor()

        try:
            # Load Beta parameters from rl_param: param_key = "alpha:{ctx}:{thresh}" / "beta:{ctx}:{thresh}"
            cursor.execute(
                f"SELECT param_key, param_value, sample_count FROM {Tables.RL_PARAM}"
                f" WHERE learner_id = ?",
                (self.name,),
            )
            for row in cursor.fetchall():
                row_dict = dict(row)
                key = row_dict["param_key"]
                value = row_dict["param_value"]
                if value is None:
                    continue
                if key.startswith("alpha:"):
                    rest = key[len("alpha:") :]
                    # rest = "context_key:threshold"
                    last_colon = rest.rfind(":")
                    context_key = rest[:last_colon]
                    threshold = float(rest[last_colon + 1 :])
                    if context_key not in self._beta_params:
                        self._beta_params[context_key] = {}
                    existing = self._beta_params[context_key].get(
                        threshold, (self.PRIOR_ALPHA, self.PRIOR_BETA)
                    )
                    self._beta_params[context_key][threshold] = (value, existing[1])
                    self._total_decisions += row_dict.get("sample_count") or 0
                elif key.startswith("beta:"):
                    rest = key[len("beta:") :]
                    last_colon = rest.rfind(":")
                    context_key = rest[:last_colon]
                    threshold = float(rest[last_colon + 1 :])
                    if context_key not in self._beta_params:
                        self._beta_params[context_key] = {}
                    existing = self._beta_params[context_key].get(
                        threshold, (self.PRIOR_ALPHA, self.PRIOR_BETA)
                    )
                    self._beta_params[context_key][threshold] = (existing[0], value)

        except Exception as e:
            logger.debug(f"RL: Could not load Beta parameters: {e}")

        try:
            # Load error rates from rl_task_stat: task_type=provider, stat_key=tp/tn/fp/fn
            cursor.execute(
                f"SELECT task_type, stat_key, stat_value FROM {Tables.RL_TASK_STAT}"
                f" WHERE learner_id = ?",
                (self.name,),
            )
            provider_counts: Dict[str, Dict[str, float]] = {}
            for row in cursor.fetchall():
                row_dict = dict(row)
                provider = row_dict["task_type"]
                if provider not in provider_counts:
                    provider_counts[provider] = {}
                provider_counts[provider][row_dict["stat_key"]] = row_dict["stat_value"]

            for provider, counts in provider_counts.items():
                total = sum(counts.values())
                if total > 0:
                    self._fp_rates[provider] = counts.get("false_positives", 0) / total
                    self._fn_rates[provider] = counts.get("false_negatives", 0) / total

        except Exception as e:
            logger.debug(f"RL: Could not load error rates: {e}")

        if self._beta_params:
            logger.info(
                f"RL: Loaded {len(self._beta_params)} grounding threshold contexts from database"
            )

    def record_outcome(self, outcome: RLOutcome) -> None:
        """Record grounding verification outcome.

        Expected metadata:
        - provider: Provider name
        - response_type: Type of response (code_generation, explanation, etc.)
        - threshold_used: The threshold that was used
        - result_type: "tp" (true positive), "tn" (true negative),
                       "fp" (false positive), "fn" (false negative)
        - actual_hallucination: Whether there was actually a hallucination
        - detected_hallucination: Whether hallucination was detected

        Args:
            outcome: Outcome with grounding verification data
        """
        provider = outcome.provider
        response_type = outcome.metadata.get("response_type", "general")
        threshold_used = outcome.metadata.get("threshold_used", self.DEFAULT_THRESHOLD)
        result_type = outcome.metadata.get("result_type")

        if not result_type:
            # Infer from actual vs detected
            actual = outcome.metadata.get("actual_hallucination", False)
            detected = outcome.metadata.get("detected_hallucination", False)
            if actual and detected:
                result_type = "tp"  # True positive - correctly detected
            elif not actual and not detected:
                result_type = "tn"  # True negative - correctly passed
            elif not actual and detected:
                result_type = "fp"  # False positive - wrongly flagged
            elif actual and not detected:
                result_type = "fn"  # False negative - missed hallucination

        if not result_type:
            logger.debug("RL: grounding_threshold outcome missing result info, skipping")
            return

        # Build context key
        context_key = self._build_context_key(provider, response_type)

        # Compute reward
        reward = self._compute_reward(result_type)

        # Update Beta parameters (Thompson Sampling update)
        self._update_beta_params(context_key, threshold_used, result_type)

        # Update provider stats
        self._update_provider_stats(provider, result_type)

        # Save to database
        self._save_to_db(context_key, threshold_used, result_type, reward)

        self._total_decisions += 1

        logger.debug(
            f"RL: Grounding threshold for {provider}/{response_type} "
            f"threshold={threshold_used:.2f} result={result_type} reward={reward:.2f}"
        )

    def _build_context_key(self, provider: str, response_type: str) -> str:
        """Build context key for state lookup."""
        # Normalize response type
        if response_type not in self.RESPONSE_TYPES:
            response_type = "general"
        return f"{provider}:{response_type}"

    def _compute_reward(self, result_type: str) -> float:
        """Compute reward from verification result.

        Args:
            result_type: tp, tn, fp, or fn

        Returns:
            Reward value
        """
        rewards = {
            "tp": 0.1,  # True positive: correctly detected hallucination
            "tn": 0.1,  # True negative: correctly passed good response
            "fp": -1.0,  # False positive: wrongly flagged good response
            "fn": -2.0,  # False negative: missed hallucination (worst)
        }
        return rewards.get(result_type, 0.0)

    def _update_beta_params(self, context_key: str, threshold: float, result_type: str) -> None:
        """Update Beta distribution parameters for Thompson Sampling.

        For each threshold level, we track:
        - Alpha: proportional to successes (correct decisions at this threshold)
        - Beta: proportional to failures (incorrect decisions)

        Args:
            context_key: Provider + response type key
            threshold: Threshold value used
            result_type: Result of the decision
        """
        # Ensure context exists
        if context_key not in self._beta_params:
            self._beta_params[context_key] = {}
            for t in self.THRESHOLD_LEVELS:
                self._beta_params[context_key][t] = (self.PRIOR_ALPHA, self.PRIOR_BETA)

        # Find closest threshold level
        closest_threshold = min(self.THRESHOLD_LEVELS, key=lambda t: abs(t - threshold))

        # Get current parameters
        alpha, beta = self._beta_params[context_key][closest_threshold]

        # Update based on result
        if result_type in ("tp", "tn"):
            # Success - increase alpha
            alpha += self.learning_rate
        else:
            # Failure - increase beta
            beta += self.learning_rate

        self._beta_params[context_key][closest_threshold] = (alpha, beta)

    def _update_provider_stats(self, provider: str, result_type: str) -> None:
        """Update provider-level statistics."""
        cursor = self.db.cursor()
        stat_map = {
            "tp": "true_positives",
            "tn": "true_negatives",
            "fp": "false_positives",
            "fn": "false_negatives",
        }
        stat_key = stat_map.get(result_type)
        if not stat_key:
            return

        ts = datetime.now().isoformat()
        cursor.execute(
            f"""
            INSERT INTO {Tables.RL_TASK_STAT}
            (learner_id, task_type, stat_key, stat_value, sample_count, updated_at)
            VALUES (?, ?, ?, 1, 1, ?)
            ON CONFLICT(learner_id, task_type, stat_key) DO UPDATE SET
                stat_value = stat_value + 1,
                sample_count = sample_count + 1,
                updated_at = excluded.updated_at
            """,
            (self.name, provider, stat_key, ts),
        )
        self.db.commit()

    def _save_to_db(
        self, context_key: str, threshold: float, result_type: str, reward: float
    ) -> None:
        """Save parameters and history to database."""
        cursor = self.db.cursor()
        timestamp = datetime.now().isoformat()

        # Find closest threshold level
        closest_threshold = min(self.THRESHOLD_LEVELS, key=lambda t: abs(t - threshold))

        # Save Beta parameters to rl_param (explicit upsert — avoids NULL context UNIQUE gotcha)
        alpha, beta_val = self._beta_params[context_key][closest_threshold]
        for prefix, value in (("alpha", alpha), ("beta", beta_val)):
            param_key = f"{prefix}:{context_key}:{closest_threshold}"
            existing = cursor.execute(
                f"SELECT sample_count FROM {Tables.RL_PARAM} "
                f"WHERE learner_id = ? AND param_key = ? AND context IS NULL",
                (self.name, param_key),
            ).fetchone()
            if existing:
                cursor.execute(
                    f"UPDATE {Tables.RL_PARAM} SET param_value = ?, sample_count = ?, updated_at = ? "
                    f"WHERE learner_id = ? AND param_key = ? AND context IS NULL",
                    (value, existing[0] + 1, timestamp, self.name, param_key),
                )
            else:
                cursor.execute(
                    f"INSERT INTO {Tables.RL_PARAM} "
                    f"(learner_id, param_key, param_value, sample_count, updated_at) "
                    f"VALUES (?, ?, ?, 1, ?)",
                    (self.name, param_key, value, timestamp),
                )

        # Save decision history to rl_transition
        cursor.execute(
            f"""
            INSERT INTO {Tables.RL_TRANSITION}
            (learner_id, from_state, to_state, action, reward, metadata, created_at)
            VALUES (?, ?, ?, ?, ?, '', ?)
            """,
            (
                self.name,
                context_key,
                result_type,
                str(closest_threshold),
                reward,
                timestamp,
            ),
        )

        self.db.commit()

    def get_recommendation(
        self, provider: str, model: str, task_type: str
    ) -> Optional[RLRecommendation]:
        """Get recommended threshold using Thompson Sampling.

        Uses Beta distribution sampling for explore/exploit balance.

        Args:
            provider: Provider name
            model: Model name (not used directly, but part of context)
            task_type: Response type / task type

        Returns:
            Recommendation with optimal threshold
        """
        context_key = self._build_context_key(provider, task_type)

        # Check if we have data for this context
        if context_key not in self._beta_params:
            return RLRecommendation(
                value=self.DEFAULT_THRESHOLD,
                confidence=0.3,
                reason="No learned data, using default threshold",
                sample_size=0,
                is_baseline=True,
            )

        # Thompson Sampling: sample from Beta distribution for each threshold
        samples = {}
        total_samples = 0
        for threshold, (alpha, beta) in self._beta_params[context_key].items():
            # Sample from Beta(alpha, beta)
            sample = self._rng.betavariate(alpha, beta)
            samples[threshold] = sample
            total_samples += int(alpha + beta - 2 * self.PRIOR_ALPHA)

        # Select threshold with highest sampled value
        best_threshold = max(samples.keys(), key=lambda t: samples[t])
        best_sample = samples[best_threshold]

        # Compute confidence based on sample size and parameter certainty
        alpha, beta = self._beta_params[context_key][best_threshold]
        sample_count = int(alpha + beta - 2 * self.PRIOR_ALPHA)

        if sample_count < self.MIN_SAMPLES_FOR_CONFIDENCE:
            confidence = 0.3 + 0.2 * (sample_count / self.MIN_SAMPLES_FOR_CONFIDENCE)
            is_baseline = True
        else:
            # Confidence based on Beta distribution concentration
            variance = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
            confidence = min(0.95, 0.5 + 0.45 * (1 - math.sqrt(variance)))
            is_baseline = False

        return RLRecommendation(
            value=best_threshold,
            confidence=confidence,
            reason=f"Thompson sample={best_sample:.3f}, α={alpha:.1f}, β={beta:.1f}",
            sample_size=total_samples,
            is_baseline=is_baseline,
        )

    def get_optimal_threshold(
        self, provider: str, response_type: str = "general"
    ) -> Tuple[float, float]:
        """Get optimal threshold for a provider/response type.

        Convenience method that returns threshold and confidence.

        Args:
            provider: Provider name
            response_type: Type of response

        Returns:
            Tuple of (threshold, confidence)
        """
        rec = self.get_recommendation(provider, "", response_type)
        return (
            rec.value if rec else self.DEFAULT_THRESHOLD,
            rec.confidence if rec else 0.3,
        )

    def get_provider_error_rates(self, provider: str) -> Dict[str, float]:
        """Get error rates for a provider.

        Args:
            provider: Provider name

        Returns:
            Dictionary with fp_rate, fn_rate, precision, recall
        """
        cursor = self.db.cursor()
        cursor.execute(
            f"SELECT stat_key, stat_value FROM {Tables.RL_TASK_STAT}"
            f" WHERE learner_id = ? AND task_type = ?",
            (self.name, provider),
        )
        counts = {dict(r)["stat_key"]: int(dict(r)["stat_value"]) for r in cursor.fetchall()}

        if not counts:
            return {"fp_rate": 0.0, "fn_rate": 0.0, "precision": 0.0, "recall": 0.0}

        tp = counts.get("true_positives", 0)
        tn = counts.get("true_negatives", 0)
        fp = counts.get("false_positives", 0)
        fn = counts.get("false_negatives", 0)

        total = tp + tn + fp + fn
        if total == 0:
            return {"fp_rate": 0.0, "fn_rate": 0.0, "precision": 0.0, "recall": 0.0}

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        return {
            "fp_rate": fp / total,
            "fn_rate": fn / total,
            "precision": precision,
            "recall": recall,
            "total_samples": total,
        }

    def get_all_provider_stats(self) -> Dict[str, Dict[str, float]]:
        """Get error rates for all providers.

        Returns:
            Dictionary mapping provider to error rates
        """
        cursor = self.db.cursor()
        cursor.execute(
            f"SELECT DISTINCT task_type FROM {Tables.RL_TASK_STAT} WHERE learner_id = ?",
            (self.name,),
        )
        providers = [dict(r)["task_type"] for r in cursor.fetchall()]

        return {provider: self.get_provider_error_rates(provider) for provider in providers}

    def export_metrics(self) -> Dict[str, Any]:
        """Export learner metrics for monitoring.

        Returns:
            Dictionary with learner stats
        """
        # Compute average threshold per context
        avg_thresholds = {}
        for context_key, params in self._beta_params.items():
            # Weight thresholds by sample count
            weighted_sum = 0.0
            total_weight = 0.0
            for threshold, (alpha, beta) in params.items():
                weight = alpha + beta - 2 * self.PRIOR_ALPHA
                weighted_sum += threshold * weight
                total_weight += weight
            if total_weight > 0:
                avg_thresholds[context_key] = weighted_sum / total_weight

        return {
            "learner": self.name,
            "total_contexts": len(self._beta_params),
            "total_decisions": self._total_decisions,
            "threshold_levels": self.THRESHOLD_LEVELS,
            "default_threshold": self.DEFAULT_THRESHOLD,
            "learned_thresholds": avg_thresholds,
            "provider_stats": self.get_all_provider_stats(),
        }
