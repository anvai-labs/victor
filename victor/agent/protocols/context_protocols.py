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

"""Context management protocols for conversation assembly."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable


@runtime_checkable
class ContextTemperatureClassifierProtocol(Protocol):
    """Classifies context messages by temporal/semantic relevance tiers.

    Enables temperature-aware context filtering in TurnBoundaryContextAssembler.
    HOT messages are kept fully, WARM messages get reduced score, COLD are excluded.

    Based on: HYVE - Hybrid Views for LLM Context Engineering (arXiv 2604.05400)
    """

    def classify(
        self,
        messages: List[Any],
        current_turn: int,
        recent_tool_names: frozenset,
    ) -> List[Tuple[Any, str]]:
        """Assign temperature tier to each message.

        Args:
            messages: List of conversation messages
            current_turn: Current turn index (0-based)
            recent_tool_names: Tool names used in recent turns

        Returns:
            List of (message, temperature_tier) pairs where tier is
            one of: "critical", "hot", "warm", "cold"
        """
        ...

    def get_score_multipliers(
        self,
        classified: List[Tuple[Any, str]],
    ) -> Dict[int, float]:
        """Return score multipliers keyed by message id().

        Args:
            classified: Output of classify()

        Returns:
            Dict mapping id(message) → float multiplier.
            HOT and CRITICAL messages have no entry (multiplier = 1.0).
            WARM messages get a reduction factor (e.g. 0.6).
            COLD messages get 0.0 (excluded from scoring).
        """
        ...
