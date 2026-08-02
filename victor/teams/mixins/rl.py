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

"""RL (Reinforcement Learning) mixin for team coordinators.

Adds RL integration for recording team execution outcomes and learning
optimal team compositions.

Features:
    - Outcome recording for team executions
    - Quality score computation
    - Integration with TeamCompositionLearner

Example:
    class MyCoordinator(RLMixin):
        def __init__(self):
            RLMixin.__init__(self)

        async def execute_team(self, config):
            result = await self._do_execution(config)
            self._record_team_rl_outcome(
                team_name=config.name,
                formation=config.formation.value,
                success=result.success,
                quality_score=self._compute_quality_score(result),
                metadata={"members": len(config.members)},
            )
            return result
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class RLMixin:
    """Mixin adding RL integration to team coordinators.

    This mixin implements IRLCoordinator protocol methods and provides
    helpers for recording outcomes and computing quality scores.

    Attributes:
        _rl_coordinator: RL coordinator instance for outcome recording
        _rl_enabled: Whether RL integration is enabled
    """

    _rl_coordinator: Optional[Any] = None
    _rl_enabled: bool = True

    def __init__(self, *, enable_rl: bool = True) -> None:
        """Initialize RL mixin.

        Args:
            enable_rl: Whether to enable RL integration
        """
        self._rl_enabled = enable_rl
        self._rl_coordinator = None

    def set_rl_coordinator(self, rl_coordinator: Any) -> None:
        """Set the RL coordinator for outcome recording.

        Args:
            rl_coordinator: RL coordinator instance
        """
        self._rl_coordinator = rl_coordinator

    def _record_team_rl_outcome(
        self,
        team_name: str,
        formation: str,
        success: bool,
        quality_score: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record RL outcome for team execution.

        This method is called after team execution to record the outcome
        for the RL learner to use in optimizing team compositions.

        Args:
            team_name: Name of the team
            formation: Formation pattern used
            success: Whether execution succeeded
            quality_score: Quality score (0.0 to 1.0)
            metadata: Additional metadata
        """
        if not self._rl_enabled:
            return

        try:
            # Try to use RL hooks if available
            from victor.framework.rl.hooks import RLEvent, RLEventType, get_rl_hooks

            hooks = get_rl_hooks()
            if hooks:
                hooks.emit(
                    RLEvent(
                        type=RLEventType.TEAM_COMPLETED,
                        success=success,
                        quality_score=quality_score,
                        team_formation=formation,
                        metadata={
                            "team_name": team_name,
                            "formation": formation,
                            "success": success,
                            "quality_score": quality_score,
                            **(metadata or {}),
                        },
                    )
                )
        except ImportError:
            logger.debug("RL hooks not available, skipping outcome recording")
        except Exception as e:
            logger.warning(f"Failed to record RL outcome: {e}")

        # Also try direct RL coordinator if set
        if self._rl_coordinator:
            try:
                self._rl_coordinator.record_outcome(
                    team_name=team_name,
                    formation=formation,
                    success=success,
                    quality_score=quality_score,
                    metadata=metadata,
                )
            except Exception as e:
                logger.warning(f"Failed to record via RL coordinator: {e}")

    def _compute_quality_score(
        self,
        success: bool,
        member_count: int,
        total_tool_calls: int,
        duration_seconds: float,
        failed_members: int = 0,
    ) -> float:
        """Compute quality score for team execution.

        Score computation:
            - Baseline: 0.5
            - Success bonus: +0.2
            - Efficiency bonus: +0.1 (if tool_calls < members * 10)
            - Speed bonus: +0.1 (if duration < 60s)
            - Member failure penalty: -0.1 per failure

        Args:
            success: Whether execution succeeded overall
            member_count: Number of team members
            total_tool_calls: Total tool calls across all members
            duration_seconds: Total execution duration
            failed_members: Number of members that failed

        Returns:
            Quality score between 0.0 and 1.0
        """
        score = 0.5  # Baseline

        # Success bonus
        if success:
            score += 0.2

        # Efficiency bonus (fewer tool calls per member)
        if member_count > 0 and total_tool_calls < member_count * 10:
            score += 0.1

        # Speed bonus
        if duration_seconds < 60.0:
            score += 0.1

        # Member failure penalty
        score -= 0.1 * failed_members

        # Clamp to valid range
        return max(0.0, min(1.0, score))
