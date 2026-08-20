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

"""Recovery coordinator integrating with existing Victor framework components.

This module provides the main entry point for the recovery system, coordinating:
- Failure detection
- Strategy selection
- Recovery execution
- Learning feedback (via existing QLearningStore)
- Telemetry (via existing UsageAnalytics)
- Context management (via existing ContextCompactor)

FRAMEWORK INTEGRATION:
======================
This coordinator reuses existing Victor components instead of creating duplicates:

1. Q-Learning: victor.agent.adaptive_mode_controller.QLearningStore
   - SQLite-backed, mode transition learning
   - We extend it for recovery action learning

2. Telemetry: victor.agent.usage_analytics.UsageAnalytics
   - Singleton pattern for tool/provider metrics
   - We add recovery-specific metrics

3. Context: victor.agent.context_compactor.ContextCompactor
   - Proactive compaction at configurable threshold
   - Intelligent truncation strategies

4. Circuit Breaker: victor.providers.circuit_breaker.CircuitBreaker
   - Full CLOSED/OPEN/HALF_OPEN state machine
   - Registry for managing multiple breakers

5. Task Tracking: victor.agent.unified_task_tracker.UnifiedTaskTracker
   - Loop detection, milestone tracking
   - Tool budget enforcement

The coordinator follows the Facade pattern, providing a simple interface
to the complex recovery subsystem while leveraging existing infrastructure.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

from victor.agent.recovery.protocols import (
    FailureType,
    RecoveryAction,
    RecoveryContext,
    RecoveryResult,
)
from victor.agent.recovery.strategies import CompositeRecoveryStrategy
from victor.agent.recovery.temperature import ProgressiveTemperatureAdjuster
from victor.agent.recovery.prompts import ModelSpecificPromptRegistry

# Import existing framework components
if TYPE_CHECKING:
    from victor.agent.adaptive_mode_controller import QLearningStore
    from victor.agent.usage_analytics import UsageAnalytics
    from victor.agent.context_compactor import ContextCompactor
    from victor.providers.circuit_breaker import CircuitBreaker, CircuitBreakerRegistry

logger = logging.getLogger(__name__)


@dataclass
class RecoveryOutcome:
    """Complete outcome of a recovery operation."""

    result: RecoveryResult
    new_temperature: Optional[float] = None
    fallback_provider: Optional[str] = None
    fallback_model: Optional[str] = None
    compaction_performed: bool = False
    tokens_freed: int = 0

    @property
    def requires_model_switch(self) -> bool:
        return self.fallback_provider is not None

    @property
    def requires_retry(self) -> bool:
        return self.result.action in (
            RecoveryAction.PROMPT_TOOL_CALL,
            RecoveryAction.RETRY_WITH_TEMPLATE,
            RecoveryAction.ADJUST_TEMPERATURE,
        )


class RecoveryCoordinator:
    """Coordinator for the recovery system using existing framework components.

    INTEGRATION ARCHITECTURE:
    ========================

    This coordinator acts as a thin facade that:
    1. Detects failures using existing UnifiedTaskTracker patterns
    2. Selects recovery strategies (new, protocol-based)
    3. Records outcomes via existing UsageAnalytics
    4. Learns via existing QLearningStore
    5. Manages context via existing ContextCompactor

    Benefits of this approach:
    - No duplicate telemetry collection
    - Unified Q-learning across all components
    - Consistent circuit breaker behavior
    - Centralized context management

    Example:
        # Create with existing framework components
        from victor.agent.adaptive_mode_controller import QLearningStore
        from victor.agent.usage_analytics import UsageAnalytics

        coordinator = RecoveryCoordinator(
            q_store=QLearningStore(),  # Existing Q-learning
            analytics=UsageAnalytics.get_instance(),  # Existing singleton
        )

        # Detect and recover
        failure = coordinator.detect_failure(content, tool_calls, ...)
        if failure:
            outcome = await coordinator.recover(failure, provider, model, ...)
            # Apply outcome.result.message, outcome.new_temperature, etc.
            coordinator.record_outcome(success=True)
    """

    def __init__(
        self,
        q_store: Optional["QLearningStore"] = None,
        analytics: Optional["UsageAnalytics"] = None,
        context_compactor: Optional["ContextCompactor"] = None,
        circuit_breaker_registry: Optional["CircuitBreakerRegistry"] = None,
        data_dir: Optional[Path] = None,
    ):
        """Initialize coordinator with framework components.

        Args:
            q_store: Existing QLearningStore from adaptive_mode_controller
            analytics: Existing UsageAnalytics singleton
            context_compactor: Existing ContextCompactor instance
            circuit_breaker_registry: Existing CircuitBreakerRegistry
            data_dir: Directory for recovery-specific persistence
        """
        # Store framework component references
        self._q_store = q_store
        self._analytics = analytics
        self._context_compactor = context_compactor
        self._circuit_registry = circuit_breaker_registry

        # Initialize recovery-specific components (these extend framework)
        self._strategy = CompositeRecoveryStrategy(q_store=q_store)
        self._temperature = ProgressiveTemperatureAdjuster(q_store=q_store)

        # Prompt registry can use separate persistence
        prompts_db = data_dir / "prompt_templates.db" if data_dir else None
        self._prompts = ModelSpecificPromptRegistry(q_store=q_store, db_path=prompts_db)

        # Current recovery context for learning
        self._current_context: Optional[RecoveryContext] = None
        self._current_outcome: Optional[RecoveryOutcome] = None

        # Recovery-specific metrics (recorded to analytics)
        self._recovery_attempts = 0
        self._recovery_successes = 0

    @classmethod
    def create_with_framework(
        cls,
        settings: Any = None,  # VictorSettings
        data_dir: Optional[Path] = None,
    ) -> "RecoveryCoordinator":
        """Create coordinator using standard framework components.

        This factory method automatically retrieves framework singletons
        and creates a coordinator with proper integration.

        Args:
            settings: VictorSettings instance (for configuration)
            data_dir: Override data directory

        Returns:
            Configured RecoveryCoordinator
        """
        # Import framework components
        from victor.agent.adaptive_mode_controller import QLearningStore
        from victor.agent.usage_analytics import UsageAnalytics

        # Get existing singletons/instances
        q_store = QLearningStore()  # Uses default path
        analytics = UsageAnalytics.get_instance()

        # Data directory
        if data_dir is None and settings:
            from victor.config.settings import get_project_paths

            paths = get_project_paths()
            data_dir = paths.project_victor_dir / "recovery"

        return cls(
            q_store=q_store,
            analytics=analytics,
            data_dir=data_dir,
        )

    def set_context_compactor(self, compactor: "ContextCompactor") -> None:
        """Set the context compactor reference.

        Called by orchestrator after creating the compactor.
        """
        self._context_compactor = compactor

    def set_circuit_registry(self, registry: "CircuitBreakerRegistry") -> None:
        """Set the circuit breaker registry reference."""
        self._circuit_registry = registry

    def detect_failure(
        self,
        content: str = "",
        tool_calls: Optional[List[Dict]] = None,
        mentioned_tools: Optional[List[str]] = None,
        elapsed_time: float = 0.0,
        session_idle_timeout: float = 180.0,
        quality_score: float = 0.5,
        consecutive_failures: int = 0,
        recent_responses: Optional[List[str]] = None,
        context_utilization: Optional[float] = None,
    ) -> Optional[FailureType]:
        """Detect failure type from response characteristics.

        Uses existing framework patterns where possible:
        - Context utilization from ContextCompactor
        - Quality scoring from grounding_verifier

        Args:
            content: Response content
            tool_calls: Tool calls made
            mentioned_tools: Tools mentioned but not called
            elapsed_time: Session elapsed time
            session_idle_timeout: Session time limit
            quality_score: Response quality (from grounding_verifier)
            consecutive_failures: Count of consecutive failures
            recent_responses: Recent responses for loop detection
            context_utilization: Context usage ratio (from compactor)

        Returns:
            FailureType if failure detected, None otherwise
        """
        # Priority 1: Timeout approaching
        if elapsed_time > session_idle_timeout * 0.9:
            return FailureType.TIMEOUT_APPROACHING

        # Priority 2: Context overflow (use existing compactor metrics if available)
        if context_utilization is not None and context_utilization > 0.85:
            return FailureType.CONTEXT_OVERFLOW
        elif self._context_compactor:
            # Get utilization from existing compactor
            try:
                utilization = self._context_compactor.get_context_utilization()
                if utilization > 0.85:
                    return FailureType.CONTEXT_OVERFLOW
            except Exception:
                pass  # Compactor not ready

        # Priority 3: Empty response
        if not content and not tool_calls:
            return FailureType.EMPTY_RESPONSE

        # Priority 4: Hallucinated tool call
        if mentioned_tools and not tool_calls:
            return FailureType.HALLUCINATED_TOOL

        # Priority 5: Stuck loop (reuse existing pattern detection)
        if self._detect_stuck_loop(content, recent_responses):
            return FailureType.STUCK_LOOP

        # Priority 6: Repeated response
        if recent_responses and self._detect_repeated_response(content, recent_responses):
            return FailureType.REPEATED_RESPONSE

        # Priority 7: Low quality
        if quality_score < 0.3:
            return FailureType.LOW_QUALITY

        return None

    def _detect_stuck_loop(
        self,
        content: str,
        recent_responses: Optional[List[str]],
    ) -> bool:
        """Detect stuck planning loop pattern."""
        if not content:
            return False

        import re

        planning_patterns = [
            r"\bi[''`]?m\s+going\s+to\s+(read|examine|check|call|use)\b",
            r"\bi\s+will\s+now\s+(read|examine|check|call|use)\b",
            r"\blet\s+me\s+(read|check|examine|look\s+at)\b",
        ]

        planning_count = 0
        for pattern in planning_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                planning_count += 1

        if planning_count >= 2:
            return True

        if recent_responses:
            for response in recent_responses[-3:]:
                for pattern in planning_patterns:
                    if re.search(pattern, response, re.IGNORECASE):
                        planning_count += 1

        return planning_count >= 3

    def _detect_repeated_response(
        self,
        content: str,
        recent_responses: List[str],
    ) -> bool:
        """Detect repeated response pattern."""
        if not content or not recent_responses:
            return False

        content_lower = content.lower().strip()
        for response in recent_responses[-3:]:
            response_lower = response.lower().strip()
            if len(content_lower) > 50 and len(response_lower) > 50:
                min_len = min(len(content_lower), len(response_lower))
                prefix_len = min(200, min_len)
                if content_lower[:prefix_len] == response_lower[:prefix_len]:
                    return True

        return False

    async def recover(
        self,
        failure_type: FailureType,
        provider: str,
        model: str,
        content: str = "",
        tool_calls_made: int = 0,
        tool_budget: int = 10,
        iteration_count: int = 0,
        max_iterations: int = 50,
        elapsed_time: float = 0.0,
        session_idle_timeout: float = 180.0,
        current_temperature: float = 0.7,
        consecutive_failures: int = 0,
        mentioned_tools: Optional[List[str]] = None,
        recent_responses: Optional[List[str]] = None,
        quality_score: float = 0.5,
        task_type: str = "general",
        is_analysis_task: bool = False,
        is_action_task: bool = False,
        session_id: Optional[str] = None,
    ) -> RecoveryOutcome:
        """Attempt recovery using strategy selection and framework integration."""
        # Build recovery context
        context = RecoveryContext(
            failure_type=failure_type,
            content=content,
            tool_calls_made=tool_calls_made,
            tool_budget=tool_budget,
            iteration_count=iteration_count,
            max_iterations=max_iterations,
            elapsed_time_seconds=elapsed_time,
            session_idle_timeout=session_idle_timeout,
            provider_name=provider,
            model_name=model,
            current_temperature=current_temperature,
            consecutive_failures=consecutive_failures,
            mentioned_tools=tuple(mentioned_tools or []),
            recent_responses=tuple(recent_responses or []),
            last_quality_score=quality_score,
            task_type=task_type,
            is_analysis_task=is_analysis_task,
            is_action_task=is_action_task,
        )

        self._current_context = context
        self._recovery_attempts += 1

        # Record to existing analytics
        if self._analytics:
            self._analytics.record_tool_execution(
                tool_name=f"recovery:{failure_type.name}",
                success=False,  # Recording the failure
                execution_time_ms=0,
                error_type=failure_type.name,
            )

        # Get recovery result from strategy
        result = await self._strategy.recover(context)

        # Build outcome
        outcome = RecoveryOutcome(result=result)

        # Handle temperature adjustment
        if result.action == RecoveryAction.ADJUST_TEMPERATURE:
            new_temp, reason = self._temperature.get_adjusted_temperature(context, session_id)
            outcome.new_temperature = new_temp
            result.new_temperature = new_temp

        # Handle model fallback via circuit breaker
        if result.action == RecoveryAction.SWITCH_MODEL:
            fallback = self._get_fallback_model(provider, model, failure_type)
            if fallback:
                outcome.fallback_provider, outcome.fallback_model = fallback
                result.fallback_model = f"{fallback[0]}/{fallback[1]}"

        # Handle context compaction via existing compactor
        if failure_type == FailureType.CONTEXT_OVERFLOW and self._context_compactor:
            try:
                freed = self._context_compactor.compact()
                outcome.compaction_performed = True
                outcome.tokens_freed = freed
            except Exception as e:
                logger.warning(f"Context compaction failed: {e}")

        # Get appropriate prompt template
        if result.action in (
            RecoveryAction.PROMPT_TOOL_CALL,
            RecoveryAction.RETRY_WITH_TEMPLATE,
            RecoveryAction.FORCE_SUMMARY,
        ):
            template, kwargs = self._prompts.get_template(
                context, escalation_level=consecutive_failures
            )
            try:
                result.message = template.format(**kwargs)
            except KeyError:
                pass  # Use default message
            result.prompt_template_id = template.id

        self._current_outcome = outcome

        logger.info(
            f"Recovery for {failure_type.name}: "
            f"action={result.action.name}, "
            f"strategy={result.strategy_name}, "
            f"confidence={result.confidence:.2f}"
        )

        return outcome

    def _get_fallback_model(
        self,
        current_provider: str,
        current_model: str,
        failure_type: FailureType,
    ) -> Optional[Tuple[str, str]]:
        """Get fallback model using circuit breaker status if available."""
        # Default fallback chains
        fallback_chains = {
            "anthropic": [
                ("openai", "gpt-4o"),
                ("anthropic", "claude-3-5-haiku-latest"),
            ],
            "openai": [
                ("anthropic", "claude-3-5-sonnet-latest"),
                ("openai", "gpt-4o-mini"),
            ],
            "ollama": [("ollama", "qwen2.5-coder:14b"), ("ollama", "llama3.1:8b")],
            "lmstudio": [("ollama", "qwen2.5-coder:14b")],
        }

        chain = fallback_chains.get(current_provider.lower(), [])

        # Filter by circuit breaker if available
        if self._circuit_registry:
            available = []
            for p, m in chain:
                key = f"{p}:{m}"
                breaker = self._circuit_registry.get(key)
                if breaker is None or breaker.state.value == "closed":
                    available.append((p, m))
            chain = available

        # Return first available
        for p, m in chain:
            if not (p == current_provider and m == current_model):
                return (p, m)

        return None

    def record_outcome(
        self,
        success: bool,
        quality_improvement: float = 0.0,
    ) -> None:
        """Record recovery outcome for learning.

        Integrates with:
        - QLearningStore: Updates Q-values for recovery actions
        - UsageAnalytics: Records success/failure metrics
        - PromptRegistry: Updates template effectiveness

        Args:
            success: Whether recovery was successful
            quality_improvement: Change in quality score
        """
        if not self._current_context or not self._current_outcome:
            logger.warning("record_outcome called without prior recovery")
            return

        context = self._current_context
        outcome = self._current_outcome
        result = outcome.result

        if success:
            self._recovery_successes += 1

        # Record to existing analytics
        if self._analytics:
            self._analytics.record_tool_execution(
                tool_name=f"recovery:{context.failure_type.name}",
                success=success,
                execution_time_ms=0,
            )

        # Record to Q-learning store
        if self._q_store:
            state_key = f"recovery:{context.to_state_key()}"
            action_key = result.action.name
            reward = 1.0 if success else -0.5
            reward += quality_improvement * 0.5
            self._q_store.update_q_value(state_key, action_key, reward)

        # Record in strategy for local learning
        self._strategy.record_outcome(context, result, success)

        # Record temperature outcome
        if outcome.new_temperature is not None:
            self._temperature.record_outcome(
                context,
                outcome.new_temperature,
                success,
                quality_improvement,
            )

        # Record template usage
        if result.prompt_template_id:
            self._prompts.record_usage(
                result.prompt_template_id,
                success,
                quality_improvement,
                context,
            )

        # Clear current context
        self._current_context = None
        self._current_outcome = None

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information."""
        diagnostics = {
            "recovery_attempts": self._recovery_attempts,
            "recovery_successes": self._recovery_successes,
            "success_rate": (self._recovery_successes / max(self._recovery_attempts, 1)),
            "strategy_metrics": self._strategy.get_strategy_metrics(),
            "template_stats": self._prompts.get_template_stats(),
            "learned_temperatures": self._temperature.get_learned_optima(),
        }

        if self._context_compactor:
            try:
                diagnostics["context_utilization"] = (
                    self._context_compactor.get_context_utilization()
                )
            except Exception as e:
                logger.debug("Failed to get context utilization for diagnostics: %s", e)

        return diagnostics

    def reset_session(self, session_id: str) -> None:
        """Reset session state."""
        self._temperature.reset_session(session_id)
        self._current_context = None
        self._current_outcome = None
