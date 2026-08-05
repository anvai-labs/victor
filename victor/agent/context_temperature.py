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

"""Context temperature classifier for relevance-aware message filtering.

Classifies conversation messages into temperature tiers (CRITICAL, HOT, WARM, COLD)
based on temporal recency and semantic relevance to the current task.

Temperature tiers control score multipliers in TurnBoundaryContextAssembler:
- CRITICAL: System prompt and current turn — always included
- HOT:      Recent turns and referenced tool results — full score
- WARM:     Moderately aged messages — reduced score (0.6x)
- COLD:     Stale, low-relevance messages — excluded (0.0x)

Based on: HYVE - Hybrid Views for LLM Context Engineering (arXiv 2604.05400)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Optional, Tuple


class ContextTemperature(str, Enum):
    """Temperature tier for a conversation message."""

    CRITICAL = "critical"
    HOT = "hot"
    WARM = "warm"
    COLD = "cold"


@dataclass
class TemperatureConfig:
    """Configuration for context temperature classification."""

    hot_max_age_turns: int = 2
    warm_max_age_turns: int = 5
    hot_max_items: int = 8
    warm_score_multiplier: float = 0.6
    cold_score_multiplier: float = 0.0


class ContextTemperatureClassifier:
    """Classifies conversation messages into temperature tiers.

    Provides score multipliers that compose with the existing scoring pipeline
    in TurnBoundaryContextAssembler — COLD messages score 0.0 (excluded),
    WARM messages score 0.6x, HOT and CRITICAL messages are unaffected (1.0x).
    """

    def __init__(self, config: Optional[TemperatureConfig] = None) -> None:
        self._config = config or TemperatureConfig()

    def classify(
        self,
        messages: List[Any],
        current_turn: int,
        recent_tool_names: FrozenSet[str],
    ) -> List[Tuple[Any, str]]:
        """Assign a temperature tier to each message.

        Args:
            messages: Older conversation messages (before the recent-turn window)
            current_turn: Number of recent turns already preserved in full
            recent_tool_names: Tool names used in the current/recent turns

        Returns:
            List of (message, tier_value) pairs
        """
        result: List[Tuple[Any, str]] = []
        hot_count = 0

        for i, msg in enumerate(messages):
            # Age relative to the oldest message in the list (0 = most recent older msg)
            age = len(messages) - 1 - i
            tier = self._compute_tier(msg, age, recent_tool_names, hot_count)
            if tier == ContextTemperature.HOT:
                hot_count += 1
            result.append((msg, tier.value))

        return result

    def _compute_tier(
        self,
        msg: Any,
        age: int,
        recent_tool_names: FrozenSet[str],
        hot_count: int,
    ) -> ContextTemperature:
        """Determine temperature tier for a single message."""
        # Critical: system prompt role
        role = getattr(msg, "role", "") or ""
        if role == "system":
            return ContextTemperature.CRITICAL

        cfg = self._config

        # HOT: recent enough and either a referenced tool result or just recent
        if age <= cfg.hot_max_age_turns and hot_count < cfg.hot_max_items:
            content = getattr(msg, "content", "") or ""
            is_tool_result = role == "tool"
            is_referenced = (
                any(name in content for name in recent_tool_names) if recent_tool_names else False
            )
            if is_tool_result or is_referenced or age <= 1:
                return ContextTemperature.HOT

        # WARM: moderately aged
        if age <= cfg.warm_max_age_turns:
            return ContextTemperature.WARM

        return ContextTemperature.COLD

    def get_score_multipliers(
        self,
        classified: List[Tuple[Any, str]],
    ) -> Dict[int, float]:
        """Return score multipliers keyed by id(message).

        HOT and CRITICAL messages are not included (implicit multiplier = 1.0).
        WARM messages get warm_score_multiplier.
        COLD messages get cold_score_multiplier (default 0.0 = excluded).
        """
        multipliers: Dict[int, float] = {}
        cfg = self._config

        for msg, tier in classified:
            if tier == ContextTemperature.COLD.value:
                multipliers[id(msg)] = cfg.cold_score_multiplier
            elif tier == ContextTemperature.WARM.value:
                multipliers[id(msg)] = cfg.warm_score_multiplier

        return multipliers
