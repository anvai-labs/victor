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

"""Recommendation explainability for the RL framework.

Enriches RLRecommendation objects with structured "why" context so that
developers and users can understand which signals drove each recommendation.

Design constraints:
- Uses existing RLRecommendation.metadata field (no new dataclass fields)
- Reads from existing learner state (no additional DB queries)
- Returns plain dicts — no new serialization format
- Does NOT modify the learner; it reads and annotates

Usage:
    from victor.framework.rl.explainability import RecommendationExplainer
    from victor.framework.rl.coordinator import get_rl_coordinator

    explainer = RecommendationExplainer(get_rl_coordinator())

    # Explain a tool recommendation
    explanation = explainer.explain_tool_recommendation(
        tool_name="read",
        task_type="analysis",
    )
    # Returns: {"tool": "read", "signals": [...], "summary": "..."}

    # Explain a set of ranked tools
    explanations = explainer.explain_tool_rankings(
        rankings=[("read", 0.85, 0.9), ("search", 0.72, 0.8)],
        task_type="analysis",
    )
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from victor.framework.rl.base import RLRecommendation

logger = logging.getLogger(__name__)


class RecommendationExplainer:
    """Annotates RL recommendations with human-readable explanation signals.

    Reads from existing learner state via the coordinator — no new storage.
    All explanations are returned as plain dicts and optionally injected into
    RLRecommendation.metadata["explanation"] for downstream consumers.
    """

    CONFIDENCE_LABELS = {
        (0.0, 0.4): "low",
        (0.4, 0.7): "medium",
        (0.7, 1.0): "high",
    }

    def __init__(self, coordinator: Any) -> None:
        """Args:
        coordinator: RLCoordinator (or MetaLearningCoordinator) instance
        """
        self._coord = coordinator

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def explain_tool_recommendation(
        self, tool_name: str, task_type: str = "default"
    ) -> Dict[str, Any]:
        """Explain why a tool was recommended for a given task type.

        Reads from the tool_selector learner's Q-values and UsageAnalytics
        success rates to produce a multi-signal explanation.

        Args:
            tool_name: Name of the tool to explain
            task_type: Task context

        Returns:
            Explanation dict with "signals", "summary", "confidence_label"
        """
        signals: List[Dict[str, Any]] = []
        learner = self._coord.get_learner("tool_selector")

        if learner is not None:
            stats = learner.get_tool_stats(tool_name)
            q_value = stats.get("q_value", 0.5)
            count = stats.get("selection_count", 0)
            success_rate = stats.get("success_rate", 0.0)
            task_q = stats.get("task_q_values", {}).get(task_type)

            signals.append(
                {
                    "source": "rl_q_learning",
                    "value": round(q_value, 3),
                    "weight": 0.8,
                    "description": (
                        f"Global Q-value {q_value:.3f} from {count} executions "
                        f"(success rate {success_rate:.0%})"
                    ),
                }
            )

            if task_q is not None:
                signals.append(
                    {
                        "source": "task_specific_q",
                        "value": round(task_q, 3),
                        "weight": 0.7,
                        "description": f"Task-specific Q-value for '{task_type}': {task_q:.3f}",
                    }
                )

            # Analytics signal if wired
            analytics = getattr(learner, "_analytics", None)
            if analytics is not None:
                try:
                    insights = analytics.get_tool_insights(tool_name)
                    analytics_rate = insights.get("success_rate", None)
                    if analytics_rate is not None:
                        signals.append(
                            {
                                "source": "usage_analytics",
                                "value": round(analytics_rate, 3),
                                "weight": 0.2,
                                "description": (
                                    f"Current-session success rate from UsageAnalytics: "
                                    f"{analytics_rate:.0%}"
                                ),
                            }
                        )
                except Exception:
                    pass

            # Predictor signal if wired
            predictor = getattr(learner, "_predictor", None)
            if predictor is not None:
                signals.append(
                    {
                        "source": "tool_predictor",
                        "value": None,
                        "weight": None,
                        "description": "Priority 3 ToolPredictor available for sequence prediction",
                    }
                )

        if not signals:
            signals.append(
                {
                    "source": "default",
                    "value": 0.5,
                    "weight": 1.0,
                    "description": "No RL data yet — using default Q-value (optimistic init)",
                }
            )

        blended = self._blend_signals(signals)
        return {
            "tool": tool_name,
            "task_type": task_type,
            "signals": signals,
            "blended_score": round(blended, 3),
            "confidence_label": self._confidence_label(blended),
            "summary": self._tool_summary(tool_name, task_type, signals, blended),
        }

    def explain_tool_rankings(
        self,
        rankings: List[Tuple[str, float, float]],
        task_type: str = "default",
    ) -> List[Dict[str, Any]]:
        """Explain a full ranked tool list.

        Args:
            rankings: List of (tool_name, score, confidence) from get_tool_rankings()
            task_type: Task context

        Returns:
            List of explanation dicts, one per tool, preserving rank order
        """
        explained = []
        for rank, (tool_name, score, confidence) in enumerate(rankings, 1):
            entry = self.explain_tool_recommendation(tool_name, task_type)
            entry["rank"] = rank
            entry["ranking_score"] = round(score, 3)
            entry["ranking_confidence"] = round(confidence, 3)
            explained.append(entry)
        return explained

    def explain_model_recommendation(
        self, provider: str, task_type: str = "default"
    ) -> Dict[str, Any]:
        """Explain why a provider/model was recommended.

        Args:
            provider: Provider name (e.g. "anthropic")
            task_type: Task context

        Returns:
            Explanation dict
        """
        signals: List[Dict[str, Any]] = []
        learner = self._coord.get_learner("model_selector")

        if learner is not None:
            rec = learner.get_recommendation(provider, "", task_type)
            if rec is not None:
                signals.append(
                    {
                        "source": "rl_q_learning",
                        "value": (round(rec.value, 3) if isinstance(rec.value, float) else None),
                        "weight": 0.9,
                        "description": rec.reason,
                    }
                )

            # Learned threshold signal
            threshold = learner.get_optimal_threshold(task_type)
            if threshold is not None:
                signals.append(
                    {
                        "source": "learned_threshold",
                        "value": round(threshold, 3),
                        "weight": None,
                        "description": (
                            f"Learned confidence threshold for '{task_type}': {threshold:.2f} "
                            f"(heuristic used when confidence ≥ this value)"
                        ),
                    }
                )

        if not signals:
            signals.append(
                {
                    "source": "default",
                    "value": 0.5,
                    "weight": 1.0,
                    "description": "No RL data — using default selection",
                }
            )

        blended = self._blend_signals(signals)
        return {
            "provider": provider,
            "task_type": task_type,
            "signals": signals,
            "blended_score": round(blended, 3),
            "confidence_label": self._confidence_label(blended),
            "summary": (
                f"Provider '{provider}' recommended for '{task_type}' with "
                f"{self._confidence_label(blended)} confidence ({blended:.0%} score)"
            ),
        }

    def annotate_recommendation(
        self, rec: RLRecommendation, learner_name: str, context: Dict[str, Any]
    ) -> RLRecommendation:
        """Inject explanation into an existing RLRecommendation's metadata.

        Non-destructive: appends "explanation" key to metadata without modifying
        the value, confidence, or reason fields.

        Args:
            rec: Recommendation to annotate
            learner_name: Name of the learner that produced it
            context: Context dict (e.g. {"tool_name": "read", "task_type": "analysis"})

        Returns:
            Same RLRecommendation with metadata["explanation"] populated
        """
        explanation = {
            "learner": learner_name,
            "confidence_label": self._confidence_label(rec.confidence),
            "sample_size": rec.sample_size,
            "is_baseline": rec.is_baseline,
            "context": context,
            "signal_summary": rec.reason,
        }
        rec.metadata["explanation"] = explanation
        return rec

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _blend_signals(self, signals: List[Dict[str, Any]]) -> float:
        """Weighted average of numeric signal values."""
        total_weight = 0.0
        total_score = 0.0
        for s in signals:
            v = s.get("value")
            w = s.get("weight") or 0.0
            if v is not None and isinstance(v, (int, float)) and w > 0:
                total_score += v * w
                total_weight += w
        if total_weight == 0:
            return 0.5
        return total_score / total_weight

    def _confidence_label(self, score: float) -> str:
        for (lo, hi), label in self.CONFIDENCE_LABELS.items():
            if lo <= score < hi:
                return label
        return "high"

    def _tool_summary(
        self,
        tool_name: str,
        task_type: str,
        signals: List[Dict[str, Any]],
        blended: float,
    ) -> str:
        sources = [s["source"] for s in signals if s.get("value") is not None]
        label = self._confidence_label(blended)
        return (
            f"Tool '{tool_name}' for '{task_type}': {label} confidence ({blended:.0%}) "
            f"from {len(sources)} signal(s): {', '.join(sources)}"
        )


def get_recommendation_explainer(
    coordinator: Any = None,
) -> Optional["RecommendationExplainer"]:
    """Get a RecommendationExplainer if USE_LEARNING_FROM_EXECUTION is enabled.

    Returns None when the flag is disabled so callers can gracefully skip
    explanation without touching prod-critical paths.

    Args:
        coordinator: RLCoordinator instance; defaults to get_rl_coordinator()

    Returns:
        RecommendationExplainer, or None if feature flag is off
    """
    from victor.core.feature_flags import FeatureFlag, get_feature_flag_manager

    if not get_feature_flag_manager().is_enabled(FeatureFlag.USE_LEARNING_FROM_EXECUTION):
        return None

    if coordinator is None:
        from victor.framework.rl.coordinator import get_rl_coordinator

        coordinator = get_rl_coordinator()

    return RecommendationExplainer(coordinator)
