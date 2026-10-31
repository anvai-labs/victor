"""SDK-owned reinforcement-learning contracts for extracted verticals."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class LearnerType(str, Enum):
    """Available learner types in the RL system."""

    TOOL_SELECTOR = "tool_selector"
    CACHE_EVICTION = "cache_eviction"
    CONTINUATION_PATIENCE = "continuation_patience"
    CONTINUATION_PROMPTS = "continuation_prompts"
    GROUNDING_THRESHOLD = "grounding_threshold"
    SEMANTIC_THRESHOLD = "semantic_threshold"
    QUALITY_WEIGHTS = "quality_weights"
    MODEL_SELECTOR = "model_selector"
    MODE_TRANSITION = "mode_transition"
    PROMPT_TEMPLATE = "prompt_template"
    OPTION_FRAMEWORK = "option_framework"


DEFAULT_PATIENCE_MAP: Dict[str, int] = {
    "anthropic": 4,
    "openai": 4,
    "google": 4,
    "deepseek": 5,
    "xai": 4,
    "moonshot": 4,
    "kimi": 4,
    "ollama": 6,
    "lmstudio": 6,
    "vllm": 6,
}


DEFAULT_ACTIVE_LEARNERS: List[LearnerType] = [
    LearnerType.TOOL_SELECTOR,
    LearnerType.CONTINUATION_PATIENCE,
    LearnerType.GROUNDING_THRESHOLD,
]


@dataclass
class BaseRLConfig:
    """Base RL configuration with shared defaults for extracted verticals."""

    active_learners: List[LearnerType] = field(
        default_factory=lambda: list(DEFAULT_ACTIVE_LEARNERS)
    )
    task_type_mappings: Dict[str, List[str]] = field(default_factory=dict)
    quality_thresholds: Dict[str, float] = field(default_factory=dict)
    default_patience: Dict[str, int] = field(default_factory=lambda: dict(DEFAULT_PATIENCE_MAP))
    exploration_bonus: float = 0.15

    def get_tools_for_task(self, task_type: str) -> List[str]:
        return self.task_type_mappings.get(task_type.lower(), [])

    def get_quality_threshold(self, task_type: str) -> float:
        return self.quality_thresholds.get(task_type.lower(), 0.80)

    def get_patience(self, provider: str) -> int:
        return self.default_patience.get(provider.lower(), 4)

    def is_learner_active(self, learner: LearnerType) -> bool:
        return learner in self.active_learners

    def get_rl_config(self) -> Dict[str, Any]:
        return {
            "active_learners": [learner.value for learner in self.active_learners],
            "task_type_mappings": self.task_type_mappings,
            "quality_thresholds": self.quality_thresholds,
            "default_patience": self.default_patience,
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"learners={len(self.active_learners)}, "
            f"task_types={len(self.task_type_mappings)})"
        )


# ── Promoted from victor.framework.rl.base ─────────────────────────────
#
# These two dataclasses mirror the runtime shapes in
# ``victor.framework.rl.base`` (reached through the ``rl_runtime`` bridge)
# field for field: same names, same types, same defaults, same required
# arguments. They are deliberately NOT a host re-export — ``rl.py`` is a
# Stable, host-free module and the top-level package imports these names
# eagerly — so the shapes are pinned against the host by
# ``tests/unit/test_rl_contracts.py`` instead.


@dataclass
class RLOutcome:
    """Result of a model execution, used for RL learning.

    Promoted from victor.framework.rl.base for contract-only vertical
    development. Field-for-field compatible with the runtime ``RLOutcome``
    (which also carries the ``to_dict``/``from_dict`` storage round-trip);
    ``timestamp`` is an ISO-8601 string, not a POSIX float.
    """

    provider: str
    model: str
    task_type: str
    success: bool
    quality_score: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    vertical: str = ""
    segment_rewards: Optional[Dict[int, float]] = None


@dataclass
class RLRecommendation:
    """RL learner recommendation for decision-making.

    Promoted from victor.framework.rl.base for contract-only vertical
    development. Field-for-field compatible with the runtime
    ``RLRecommendation``: ``value``/``confidence``/``reason``/``sample_size``
    are required and ``is_baseline`` defaults to ``False``.
    """

    value: Any
    confidence: float
    reason: str
    sample_size: int
    is_baseline: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


__all__ = [
    "BaseRLConfig",
    "DEFAULT_ACTIVE_LEARNERS",
    "DEFAULT_PATIENCE_MAP",
    "LearnerType",
    "RLOutcome",
    "RLRecommendation",
]
