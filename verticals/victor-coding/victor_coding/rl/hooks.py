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

"""RL recording hooks for coding vertical.

Provides hooks for recording outcomes and getting recommendations
from the RL system, specialized for coding tasks.
"""

from typing import Any, Dict, List, Optional

from victor_contracts.rl_runtime import RLManager, get_rl_coordinator
from victor_contracts.rl import LearnerType
from victor_coding.rl.config import CodingRLConfig, get_default_config


class CodingRLHooks:
    """RL recording hooks for coding middleware.

    This class provides methods for recording outcomes (success/failure)
    and getting recommendations from the RL system, all specialized
    for coding tasks.

    Example:
        hooks = CodingRLHooks()

        # Record tool success
        hooks.on_tool_success(
            tool_name="edit_files",
            task_type="refactoring",
            provider="anthropic",
            model="claude-3-opus",
            duration_ms=1500.0,
        )

        # Get tool recommendations
        tools = hooks.get_tool_recommendation(
            task_type="debugging",
            available_tools=["read", "grep", "shell"],
        )
    """

    def __init__(
        self,
        rl_manager: Optional[RLManager] = None,
        config: Optional[CodingRLConfig] = None,
    ):
        """Initialize hooks.

        Args:
            rl_manager: Optional RLManager instance. If not provided,
                creates one using the global coordinator.
            config: Optional CodingRLConfig. Uses default if not provided.
        """
        self._rl = rl_manager
        self._config = config or get_default_config()

    @property
    def rl(self) -> RLManager:
        """Get the RL manager, creating if needed."""
        if self._rl is None:
            self._rl = RLManager(get_rl_coordinator())
        return self._rl

    @property
    def config(self) -> CodingRLConfig:
        """Get the RL configuration."""
        return self._config

    # =========================================================================
    # Recording Outcomes
    # =========================================================================

    def on_tool_success(
        self,
        tool_name: str,
        task_type: str,
        provider: str,
        model: str,
        duration_ms: float,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record successful tool execution.

        Args:
            tool_name: Name of the tool that executed
            task_type: Type of task being performed
            provider: LLM provider name
            model: Model name
            duration_ms: Execution duration in milliseconds
            context: Additional context information
        """
        if not self._config.is_learner_active(LearnerType.TOOL_SELECTOR):
            return

        self.rl.record_success(
            learner=LearnerType.TOOL_SELECTOR,
            provider=provider,
            model=model,
            task_type=task_type,
            quality_score=1.0,
            metadata={
                "tool": tool_name,
                "duration_ms": duration_ms,
                **(context or {}),
            },
            vertical="coding",
        )

    def on_tool_failure(
        self,
        tool_name: str,
        task_type: str,
        provider: str,
        model: str,
        error: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record failed tool execution.

        Args:
            tool_name: Name of the tool that failed
            task_type: Type of task being performed
            provider: LLM provider name
            model: Model name
            error: Error message
            context: Additional context information
        """
        if not self._config.is_learner_active(LearnerType.TOOL_SELECTOR):
            return

        self.rl.record_failure(
            learner=LearnerType.TOOL_SELECTOR,
            provider=provider,
            model=model,
            task_type=task_type,
            error=error,
            metadata={
                "tool": tool_name,
                **(context or {}),
            },
            vertical="coding",
        )

    def on_continuation_success(
        self,
        provider: str,
        model: str,
        attempts: int,
        task_type: str = "continuation",
    ) -> None:
        """Record successful continuation.

        Args:
            provider: LLM provider name
            model: Model name
            attempts: Number of attempts it took
            task_type: Task type
        """
        if not self._config.is_learner_active(LearnerType.CONTINUATION_PATIENCE):
            return

        self.rl.record_success(
            learner=LearnerType.CONTINUATION_PATIENCE,
            provider=provider,
            model=model,
            task_type=task_type,
            metadata={"attempts": attempts},
            vertical="coding",
        )

    def on_continuation_failure(
        self,
        provider: str,
        model: str,
        attempts: int,
        error: str,
    ) -> None:
        """Record failed continuation.

        Args:
            provider: LLM provider name
            model: Model name
            attempts: Number of attempts before failure
            error: Error message
        """
        if not self._config.is_learner_active(LearnerType.CONTINUATION_PATIENCE):
            return

        self.rl.record_failure(
            learner=LearnerType.CONTINUATION_PATIENCE,
            provider=provider,
            model=model,
            task_type="continuation",
            error=error,
            metadata={"attempts": attempts},
            vertical="coding",
        )

    def on_mode_transition(
        self,
        from_mode: str,
        to_mode: str,
        success: bool,
        task_type: str,
        provider: str = "unknown",
        model: str = "unknown",
    ) -> None:
        """Record mode transition outcome.

        Args:
            from_mode: Mode transitioned from
            to_mode: Mode transitioned to
            success: Whether transition was effective
            task_type: Task type
            provider: LLM provider name
            model: Model name
        """
        if not self._config.is_learner_active(LearnerType.MODE_TRANSITION):
            return

        if success:
            self.rl.record_success(
                learner=LearnerType.MODE_TRANSITION,
                provider=provider,
                model=model,
                task_type=task_type,
                metadata={"from_mode": from_mode, "to_mode": to_mode},
                vertical="coding",
            )
        else:
            self.rl.record_failure(
                learner=LearnerType.MODE_TRANSITION,
                provider=provider,
                model=model,
                task_type=task_type,
                error="Mode transition ineffective",
                metadata={"from_mode": from_mode, "to_mode": to_mode},
                vertical="coding",
            )

    def on_grounding_result(
        self,
        score: float,
        threshold: float,
        passed: bool,
        task_type: str,
        provider: str = "unknown",
        model: str = "unknown",
    ) -> None:
        """Record grounding verification result.

        Args:
            score: Grounding score achieved
            threshold: Threshold used
            passed: Whether grounding passed
            task_type: Task type
            provider: LLM provider name
            model: Model name
        """
        if not self._config.is_learner_active(LearnerType.GROUNDING_THRESHOLD):
            return

        if passed:
            self.rl.record_success(
                learner=LearnerType.GROUNDING_THRESHOLD,
                provider=provider,
                model=model,
                task_type=task_type,
                quality_score=score,
                metadata={"threshold": threshold},
                vertical="coding",
            )
        else:
            self.rl.record_failure(
                learner=LearnerType.GROUNDING_THRESHOLD,
                provider=provider,
                model=model,
                task_type=task_type,
                error="Below grounding threshold",
                metadata={"score": score, "threshold": threshold},
                vertical="coding",
            )

    # =========================================================================
    # Getting Recommendations
    # =========================================================================

    def get_tool_recommendation(
        self,
        task_type: str,
        available_tools: Optional[List[str]] = None,
    ) -> List[str]:
        """Get RL-recommended tools for task.

        First checks if RL has learned recommendations, then falls back
        to config-based recommendations.

        Args:
            task_type: Type of task (debugging, refactoring, etc.)
            available_tools: List of available tool names

        Returns:
            List of recommended tool names
        """
        # Try RL recommendation first
        rl_rec = self.rl.get_tool_recommendation(
            task_type=task_type,
            available_tools=available_tools,
            vertical="coding",
        )
        if rl_rec:
            return rl_rec

        # Fall back to config-based recommendation
        config_tools = self._config.get_tools_for_task(task_type)
        if available_tools:
            return [t for t in config_tools if t in available_tools]
        return config_tools

    def get_patience_recommendation(
        self,
        provider: str,
        model: str,
    ) -> int:
        """Get recommended continuation patience.

        Args:
            provider: LLM provider name
            model: Model name

        Returns:
            Recommended patience value
        """
        # Try RL recommendation first
        rl_rec = self.rl.get_patience_recommendation(provider, model)
        if rl_rec is not None:
            return rl_rec

        # Fall back to config-based recommendation
        return self._config.get_patience(provider)

    def get_quality_threshold(
        self,
        task_type: str,
    ) -> float:
        """Get quality threshold for task type.

        Args:
            task_type: Type of task

        Returns:
            Quality threshold (0.0-1.0)
        """
        # For now, just use config threshold
        # In future, could be RL-tuned
        return self._config.get_quality_threshold(task_type)

    def __repr__(self) -> str:
        return f"CodingRLHooks(config={self._config})"


# Singleton instance
_hooks_instance: CodingRLHooks | None = None


def get_coding_rl_hooks() -> CodingRLHooks:
    """Get the singleton CodingRLHooks instance.

    Returns:
        CodingRLHooks instance
    """
    global _hooks_instance
    if _hooks_instance is None:
        _hooks_instance = CodingRLHooks()
    return _hooks_instance


__all__ = [
    "CodingRLHooks",
    "get_coding_rl_hooks",
]
