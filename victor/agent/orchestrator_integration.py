# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
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

"""Orchestrator integration bridge for runtime-intelligence components.

This module provides a non-intrusive integration layer that connects the
RuntimeIntelligencePipeline to the AgentOrchestrator. It hooks into the
orchestrator's request/response flow to enable:

1. Resilient Provider Calls:
   - Circuit breaker for failing providers
   - Automatic retry with exponential backoff
   - Rate limiting for API protection

2. Response Quality Scoring:
   - Multi-dimensional quality assessment
   - Grounding verification (hallucination detection)
   - Learning from user feedback

3. Intelligent Mode Transitions:
   - Q-learning based mode optimization
   - Profile-specific learning
   - Optimal tool budget calculation

4. Prompt Optimization:
   - Embedding-based context selection
   - Task-type-specific prompt generation
   - Provider-aware prompt strategies

Architecture (Decorator Pattern):

    ┌─────────────────────────────────────────────────────────┐
    │                 OrchestratorIntegration                  │
    │     (Decorator/Wrapper around orchestrator methods)      │
    └─────────────────────────────────────────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │                   │
                    ▼                   ▼
    ┌───────────────────────┐  ┌────────────────────────────┐
    │  AgentOrchestrator    │  │ RuntimeIntelligencePipeline│
    │  (existing behavior)  │  │  (learning components)     │
    └───────────────────────┘  └────────────────────────────┘

Usage:
    orchestrator = AgentOrchestrator(settings, provider, model)
    integration = await OrchestratorIntegration.create(orchestrator)

    # The integration hooks into orchestrator callbacks
    # and enhances request/response processing

    # Check stats
    stats = integration.get_pipeline_stats()
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from victor.agent.orchestrator import AgentOrchestrator
    from victor.agent.runtime_intelligence_pipeline import (
        RuntimeIntelligencePipeline,
        RequestContext,
        ResponseResult,
    )

# Import protocols for runtime type hints (protocols.py has no heavy deps)
from victor.core.protocols import (
    OrchestratorProtocol,
    RuntimeIntelligencePipelineProtocol,
)

logger = logging.getLogger(__name__)


@dataclass
class IntegrationConfig:
    """Configuration for orchestrator integration."""

    # Enable/disable individual features
    enable_resilient_calls: bool = True
    enable_quality_scoring: bool = True
    enable_mode_learning: bool = True
    enable_prompt_optimization: bool = True

    # Quality thresholds
    min_quality_threshold: float = 0.5
    grounding_confidence_threshold: float = 0.7

    # Logging
    log_learning_events: bool = True
    log_quality_scores: bool = True


@dataclass
class IntegrationMetrics:
    """Metrics tracked by the integration."""

    total_requests: int = 0
    enhanced_requests: int = 0
    quality_validated_responses: int = 0
    grounding_checked_responses: int = 0
    mode_transitions: int = 0
    resilient_calls: int = 0
    circuit_breaker_trips: int = 0
    avg_quality_score: float = 0.0
    avg_grounding_score: float = 0.0
    total_learning_reward: float = 0.0


class OrchestratorIntegration:
    """Integration bridge connecting RuntimeIntelligencePipeline to AgentOrchestrator.

    This class wraps around an AgentOrchestrator instance and enhances its
    functionality with runtime-intelligence features. It uses a non-intrusive
    approach - the orchestrator continues to work normally, but gains:

    1. Pre-request optimization (prompt building, mode selection)
    2. Post-response validation (quality scoring, grounding verification)
    3. Continuous learning (feedback recording, Q-learning updates)
    """

    def __init__(
        self,
        orchestrator: "AgentOrchestrator",
        pipeline: "RuntimeIntelligencePipeline",
        config: Optional[IntegrationConfig] = None,
    ):
        """Initialize the integration bridge.

        Args:
            orchestrator: The AgentOrchestrator to enhance
            pipeline: The RuntimeIntelligencePipeline to use
            config: Optional configuration settings
        """
        self._orchestrator = orchestrator
        self._pipeline = pipeline
        self._config = config or IntegrationConfig()
        self._metrics = IntegrationMetrics()

        # Track current request context
        self._current_context: Optional["RequestContext"] = None
        self._session_start = datetime.now()

        # Observers for quality/grounding events
        self._quality_observers: List[Callable[[float, Dict[str, float]], None]] = []
        self._grounding_observers: List[Callable[[bool, List[str]], None]] = []

    @classmethod
    async def create(
        cls,
        orchestrator: "AgentOrchestrator",
        config: Optional[IntegrationConfig] = None,
    ) -> "OrchestratorIntegration":
        """Factory method to create and initialize the integration.

        Args:
            orchestrator: The orchestrator to enhance
            config: Optional configuration

        Returns:
            Initialized OrchestratorIntegration
        """
        from victor.agent.runtime_intelligence_pipeline import (
            RuntimeIntelligencePipeline,
        )

        # Create pipeline from orchestrator's current state
        pipeline = await RuntimeIntelligencePipeline.create(
            provider_name=orchestrator.provider_name,
            model=orchestrator.model,
            profile_name=f"{orchestrator.provider_name}:{orchestrator.model}",
            project_root=(
                str(orchestrator.project_context.context_file.parent)
                if orchestrator.project_context.context_file
                else None
            ),
        )

        return cls(orchestrator, pipeline, config)

    def _resolve_operating_mode(self) -> str:
        """Return the agent's operating mode (build/plan/review/delegate/explore).

        This is deliberately *not* the ``ConversationStage``. The two are different
        axes: a stage says where the turn is (INITIAL, READING, COMPLETION), while
        the operating mode says what the agent is permitted to do. Feeding a stage
        name in here meant ``IntelligentPromptBuilder._get_mode_hint`` never matched
        a known mode, silently dropped the mode hint, and generated a read-only
        "code analyst" prompt for a session running in BUILD mode.

        ``ModeAwareMixin`` already exposes the real value; use it.
        """
        mode_name = getattr(self._orchestrator, "current_mode_name", None)
        if isinstance(mode_name, str) and mode_name.strip():
            return mode_name.strip().lower()

        logger.debug("Operating mode unavailable on orchestrator; defaulting to 'explore'")
        return "explore"

    async def prepare_request(
        self,
        task: str,
        task_type: str = "general",
        current_mode: Optional[str] = None,
    ) -> "RequestContext":
        """Prepare an optimized request context.

        Uses the runtime-intelligence pipeline to:
        1. Build an optimized system prompt
        2. Get mode transition recommendation
        3. Calculate optimal tool budget

        Args:
            task: The current task/query
            task_type: Detected task type
            current_mode: Current agent mode

        Returns:
            RequestContext with optimized settings
        """
        self._metrics.total_requests += 1

        if not self._config.enable_prompt_optimization:
            # Return minimal context if optimization disabled
            from victor.agent.runtime_intelligence_pipeline import RequestContext

            return RequestContext(
                system_prompt="",
                recommended_tool_budget=self._orchestrator.tool_budget,
                recommended_mode=current_mode or "explore",
                should_continue=True,
            )

        if current_mode is None:
            current_mode = self._resolve_operating_mode()

        # Prepare context using pipeline
        context = await self._pipeline.prepare_request(
            task=task,
            task_type=task_type,
            current_mode=current_mode,
            tool_calls_made=self._orchestrator.tool_calls_used,
            tool_budget=self._orchestrator.tool_budget,
            iteration_count=self._orchestrator.unified_tracker.iteration_count,
            iteration_budget=20,  # Default iteration budget
            quality_score=0.5,  # Initial quality estimate
        )

        self._current_context = context
        self._metrics.enhanced_requests += 1

        if self._config.log_learning_events:
            logger.debug(
                f"[OrchestratorIntegration] Prepared request: "
                f"mode={context.recommended_mode}, budget={context.recommended_tool_budget}"
            )

        return context

    async def validate_response(
        self,
        response: str,
        query: str = "",
        tool_calls: int = 0,
        success: bool = True,
        task_type: str = "general",
    ) -> "ResponseResult":
        """Validate and score a response.

        Uses the runtime-intelligence pipeline to:
        1. Score response quality
        2. Verify grounding (hallucination detection)
        3. Record feedback for learning

        Args:
            response: The model's response
            query: Original query
            tool_calls: Number of tool calls made
            success: Whether the response was successful
            task_type: Task type

        Returns:
            ResponseResult with quality and grounding info
        """
        from victor.agent.runtime_intelligence_pipeline import ResponseResult

        if not self._config.enable_quality_scoring:
            return ResponseResult(
                is_valid=True,
                quality_score=0.5,
                grounding_score=1.0,
                is_grounded=True,
            )

        result = await self._pipeline.process_response(
            response=response,
            query=query,
            tool_calls=tool_calls,
            tool_budget=self._orchestrator.tool_budget,
            success=success,
            task_type=task_type,
        )

        # Batch metrics update
        self._metrics.quality_validated_responses += 1
        if result.grounding_score > 0:
            self._metrics.grounding_checked_responses += 1

        # Use faster exponential moving average
        alpha = 0.1
        self._metrics.avg_quality_score += alpha * (
            result.quality_score - self._metrics.avg_quality_score
        )
        self._metrics.avg_grounding_score += alpha * (
            result.grounding_score - self._metrics.avg_grounding_score
        )
        self._metrics.total_learning_reward += result.learning_reward

        # Single-pass observer notification
        if self._quality_observers:
            for observer in self._quality_observers:
                try:
                    observer(result.quality_score, result.quality_details)
                except Exception as e:
                    logger.warning("Quality observer failed: %s", e)
        if self._grounding_observers:
            for observer in self._grounding_observers:
                try:
                    observer(result.is_grounded, result.grounding_issues)
                except Exception as e:
                    logger.warning("Grounding observer failed: %s", e)

        if self._config.log_quality_scores:
            logger.debug(
                f"[OrchestratorIntegration] Response validated: "
                f"quality={result.quality_score:.2f}, grounded={result.is_grounded}"
            )

        return result

    def should_continue(self) -> tuple[bool, str]:
        """Determine if processing should continue using learned behaviors.

        Returns:
            Tuple of (should_continue, reason)
        """
        if not self._config.enable_mode_learning:
            return True, "Mode learning disabled"

        return self._pipeline.should_continue(
            tool_calls_made=self._orchestrator.tool_calls_used,
            tool_budget=self._orchestrator.tool_budget,
            quality_score=self._metrics.avg_quality_score,
            iteration_count=self._orchestrator.unified_tracker.iteration_count,
            iteration_budget=20,
        )

    def get_optimal_tool_budget(self, task_type: str) -> int:
        """Get learned optimal tool budget for a task type.

        Args:
            task_type: The task type

        Returns:
            Optimal tool budget
        """
        if self._config.enable_mode_learning:
            learned_budget = self._pipeline.get_optimal_tool_budget(task_type)
            if learned_budget > 0:
                return learned_budget
        return self._orchestrator.tool_budget

    def add_quality_observer(self, observer: Callable[[float, Dict[str, float]], None]) -> None:
        """Add observer for quality score events.

        Args:
            observer: Callback receiving (quality_score, quality_details)
        """
        self._quality_observers.append(observer)

    def add_grounding_observer(self, observer: Callable[[bool, List[str]], None]) -> None:
        """Add observer for grounding verification events.

        Args:
            observer: Callback receiving (is_grounded, grounding_issues)
        """
        self._grounding_observers.append(observer)

    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics.

        Returns:
            Dictionary with pipeline stats
        """
        pipeline_stats = self._pipeline.get_stats()

        return {
            "integration": {
                "total_requests": self._metrics.total_requests,
                "enhanced_requests": self._metrics.enhanced_requests,
                "quality_validated": self._metrics.quality_validated_responses,
                "grounding_checked": self._metrics.grounding_checked_responses,
                "avg_quality_score": round(self._metrics.avg_quality_score, 3),
                "avg_grounding_score": round(self._metrics.avg_grounding_score, 3),
                "total_learning_reward": round(self._metrics.total_learning_reward, 3),
            },
            "pipeline": {
                "total_requests": pipeline_stats.total_requests,
                "successful_requests": pipeline_stats.successful_requests,
                "circuit_breaker_trips": pipeline_stats.circuit_breaker_trips,
                "cache_state": pipeline_stats.cache_state,
            },
            "learning": self._pipeline.get_learning_summary(),
            "config": {
                "resilient_calls": self._config.enable_resilient_calls,
                "quality_scoring": self._config.enable_quality_scoring,
                "mode_learning": self._config.enable_mode_learning,
                "prompt_optimization": self._config.enable_prompt_optimization,
            },
        }

    def get_quality_threshold_status(self) -> Dict[str, Any]:
        """Check if quality metrics meet configured thresholds.

        Returns:
            Dictionary with threshold status
        """
        quality_score = round(self._metrics.avg_quality_score, 3)
        grounding_score = round(self._metrics.avg_grounding_score, 3)

        return {
            "quality_meets_threshold": quality_score >= self._config.min_quality_threshold,
            "quality_score": quality_score,
            "quality_threshold": self._config.min_quality_threshold,
            "grounding_meets_threshold": grounding_score
            >= self._config.grounding_confidence_threshold,
            "grounding_score": grounding_score,
            "grounding_threshold": self._config.grounding_confidence_threshold,
        }

    def reset_session(self) -> None:
        """Reset session tracking."""
        self._session_start = datetime.now()
        self._current_context = None
        self._metrics = IntegrationMetrics()
        self._pipeline.reset_session()

    @property
    def orchestrator(self) -> "AgentOrchestrator":
        """Get the wrapped orchestrator."""
        return self._orchestrator

    @property
    def pipeline(self) -> "RuntimeIntelligencePipeline":
        """Get the runtime-intelligence pipeline."""
        return self._pipeline

    # =========================================================================
    # Runtime-intelligence methods (moved from AgentOrchestrator)
    # =========================================================================

    async def prepare_runtime_intelligence_request(
        self,
        task: str,
        task_type: str,
        conversation_state: Any,
        unified_tracker: Any,
    ) -> Optional[Dict[str, Any]]:
        """Pre-request hook for runtime-intelligence integration.

        Called at the start of stream_chat to:
        - Get mode transition recommendations (Q-learning)
        - Get optimal tool budget for task type
        - Enable prompt optimization if configured

        Args:
            task: The user's task/query
            task_type: Detected task type (analysis, edit, etc.)
            conversation_state: Orchestrator's conversation state
            unified_tracker: Orchestrator's unified tracker

        Returns:
            Dictionary with recommendations, or None if pipeline disabled
        """
        try:
            current_mode = self._resolve_operating_mode()

            # Prepare request context (async call to pipeline)
            context = await self.prepare_request(
                task=task,
                task_type=task_type,
                current_mode=current_mode,
            )

            # Apply recommended tool budget if available (skip if user made a sticky override)
            # NOTE: We no longer reduce the budget based on pipeline recommendations.
            # The pipeline may suggest a smaller budget based on Q-learning, but this
            # caused premature stopping (e.g., 50 -> 5). The user's budget is authoritative.
            # We only log the recommendation for debugging purposes.
            if context.recommended_tool_budget:
                sticky_budget = getattr(unified_tracker, "_sticky_user_budget", False)
                if not sticky_budget:
                    current_budget = unified_tracker.progress.tool_budget
                    # Only log significant differences, but don't reduce the budget
                    if abs(context.recommended_tool_budget - current_budget) > 10:
                        logger.debug(
                            "RuntimeIntelligencePipeline recommended budget "
                            f"{context.recommended_tool_budget} "
                            f"differs from current {current_budget}, keeping current budget"
                        )

            return {
                "recommended_mode": context.recommended_mode,
                "recommended_tool_budget": context.recommended_tool_budget,
                "should_continue": context.should_continue,
                "system_prompt_addition": (
                    context.system_prompt if context.system_prompt else None
                ),
            }
        except (AttributeError, KeyError) as e:
            logger.debug(
                f"RuntimeIntelligencePipeline prepare_request skipped (not configured): {e}"
            )
            return None
        except (ValueError, TypeError) as e:
            logger.debug(f"RuntimeIntelligencePipeline prepare_request failed (data error): {e}")
            return None

    async def validate_runtime_intelligence_response(
        self,
        response: str,
        query: str,
        tool_calls: int,
        task_type: str,
    ) -> Optional[Dict[str, Any]]:
        """Post-response hook for runtime-intelligence integration.

        Called after each streaming iteration to:
        - Score response quality (coherence, completeness, relevance)
        - Verify grounding (detect hallucinations)
        - Record feedback for Q-learning

        Args:
            response: The model's response content
            query: Original user query
            tool_calls: Number of tool calls made so far
            task_type: Task type for context

        Returns:
            Dictionary with quality/grounding scores, or None if pipeline disabled
        """
        # Skip validation for empty or very short responses
        if not response or len(response.strip()) < 50:
            return None

        try:
            result = await self.validate_response(
                response=response,
                query=query,
                tool_calls=tool_calls,
                success=True,
                task_type=task_type,
            )

            # Log quality info (debug-level; internal pipeline detail)
            if not result.is_valid:
                logger.debug(
                    "RuntimeIntelligencePipeline: Response below quality threshold "
                    "(quality=%.2f, grounded=%s)",
                    result.quality_score,
                    result.is_grounded,
                )

            return {
                "quality_score": result.quality_score,
                "grounding_score": result.grounding_score,
                "is_grounded": result.is_grounded,
                "is_valid": result.is_valid,
                "grounding_issues": result.grounding_issues,
                # Grounding failure handling - force finalize after max retries
                "should_finalize": getattr(result, "should_finalize", False),
                "should_retry": getattr(result, "should_retry", False),
                "finalize_reason": getattr(result, "finalize_reason", ""),
                # Actionable feedback for grounding correction on retry
                "grounding_feedback": getattr(result, "grounding_feedback", ""),
                "grounding_verified_references": getattr(
                    result, "grounding_verified_references", []
                ),
                "grounding_unverified_references": getattr(
                    result, "grounding_unverified_references", []
                ),
                "grounding_evidence_summary": getattr(result, "grounding_evidence_summary", ""),
            }
        except (AttributeError, KeyError) as e:
            logger.debug(
                f"RuntimeIntelligencePipeline validate_response skipped (not configured): {e}"
            )
            return None
        except (ValueError, TypeError) as e:
            logger.debug(f"RuntimeIntelligencePipeline validate_response failed (data error): {e}")
            return None

    def record_runtime_intelligence_outcome(
        self,
        success: bool,
        quality_score: float,
        user_satisfied: bool,
        completed: bool,
        rl_coordinator: Any,
        stream_context: Any,
        vertical_context: Any,
        provider_name: str,
        model: str,
        tool_calls_used: int,
        continuation_prompts: int,
        max_continuation_prompts_used: int,
        stuck_loop_detected: bool,
    ) -> None:
        """Record outcome for Q-learning feedback.

        Called at the end of a conversation to record the outcome
        for reinforcement learning. This helps the system learn
        optimal mode transitions and tool budgets.

        Also records continuation prompt learning outcomes if RL learner enabled.

        Args:
            success: Whether the task was completed successfully
            quality_score: Final quality score (0.0-1.0)
            user_satisfied: Whether user seemed satisfied
            completed: Whether task reached completion
            rl_coordinator: The RL coordinator for recording outcomes
            stream_context: Current stream context
            vertical_context: Vertical context for vertical name
            provider_name: Name of the provider
            model: Model name
            tool_calls_used: Total tool calls used
            continuation_prompts: Number of continuation prompts used
            max_continuation_prompts_used: Max continuation prompts configured
            stuck_loop_detected: Whether a stuck loop was detected
        """
        # Record RL outcomes for all learners
        if rl_coordinator and stream_context:
            try:
                from victor.framework.rl.base import RLOutcome

                ctx = stream_context
                # Determine task type from context
                task_type = "default"
                if ctx.is_analysis_task:
                    task_type = "analysis"
                elif ctx.is_action_task:
                    task_type = "action"

                # Get vertical name from context (avoid hardcoded "coding")
                vertical_name = getattr(vertical_context, "vertical_name", None) or "default"

                # Vision P6: derive segment-level process reward from this session's tool trajectory
                # (first production use of the credit-assignment framework) and blend it into the
                # quality score so every learner consumes a process-aware signal. No tools → no change.
                segment_rewards: Optional[Dict[int, float]] = None
                try:
                    from victor.core.feature_flags import (
                        FeatureFlag,
                        is_feature_enabled,
                    )

                    if is_feature_enabled(FeatureFlag.USE_LEARNING_FROM_EXECUTION):
                        from victor.framework.rl.trace_to_credit import (
                            blend_quality_with_segments,
                            compute_segment_rewards_from_trace,
                        )

                        segment_rewards = compute_segment_rewards_from_trace(ctx) or None
                        if segment_rewards:
                            quality_score = blend_quality_with_segments(
                                quality_score, segment_rewards
                            )
                except Exception as exc:  # never break outcome recording on the P6 blend
                    logger.debug("P6 segment-reward blend skipped: %s", exc)

                # Record outcome for continuation_prompts learner
                outcome = RLOutcome(
                    provider=provider_name,
                    model=model,
                    task_type=task_type,
                    success=success and completed,
                    quality_score=quality_score,
                    segment_rewards=segment_rewards,
                    metadata={
                        "continuation_prompts_used": continuation_prompts,
                        "max_prompts_configured": max_continuation_prompts_used,
                        "stuck_loop_detected": stuck_loop_detected,
                        "forced_completion": ctx.force_completion,
                        "tool_calls_total": tool_calls_used,
                    },
                    vertical=vertical_name,
                )
                rl_coordinator.record_outcome("continuation_prompts", outcome, vertical_name)

                # Emit RL hook for continuation prompt
                self.emit_continuation_event(
                    event_type="prompt",
                    success=success and completed,
                    quality_score=quality_score,
                    task_type=task_type,
                    prompts_used=continuation_prompts,
                    provider_name=provider_name,
                    model=model,
                )

                # Also record for continuation_patience learner if we have stuck loop data
                if continuation_prompts > 0:
                    patience_outcome = RLOutcome(
                        provider=provider_name,
                        model=model,
                        task_type=task_type,
                        success=success and completed,
                        quality_score=quality_score,
                        metadata={
                            "flagged_as_stuck": stuck_loop_detected,
                            "actually_stuck": stuck_loop_detected and not success,
                            "eventually_made_progress": not stuck_loop_detected and success,
                        },
                        vertical=vertical_name,
                    )
                    rl_coordinator.record_outcome(
                        "continuation_patience", patience_outcome, vertical_name
                    )

                    # Emit RL hook for continuation patience
                    self.emit_continuation_event(
                        event_type="patience",
                        success=success and completed,
                        quality_score=quality_score,
                        task_type=task_type,
                        prompts_used=continuation_prompts,
                        stuck_detected=stuck_loop_detected,
                        provider_name=provider_name,
                        model=model,
                    )

            except Exception as e:
                logger.warning(f"RL: Failed to record RL outcomes: {e}")

        try:
            # Access the mode controller through the pipeline
            pipeline = self._pipeline
            if hasattr(pipeline, "_mode_controller") and pipeline._mode_controller:
                pipeline._mode_controller.record_outcome(
                    success=success,
                    quality_score=quality_score,
                    user_satisfied=user_satisfied,
                    completed=completed,
                )
                logger.debug(
                    "RuntimeIntelligencePipeline recorded outcome: "
                    f"success={success}, quality={quality_score:.2f}"
                )
        except Exception as e:
            logger.debug(f"RuntimeIntelligencePipeline record_outcome failed: {e}")

    def should_continue_runtime_intelligence(self) -> tuple[bool, str]:
        """Check if processing should continue using learned behaviors.

        Uses Q-learning based decisions to determine if the agent
        should continue processing or transition to completion.

        Returns:
            Tuple of (should_continue, reason)
        """
        try:
            return self.should_continue()
        except Exception as e:
            logger.debug(f"RuntimeIntelligencePipeline should_continue failed: {e}")
            return True, "Fallback to continue"

    # Backward-compatible aliases while callers migrate to runtime-intelligence naming.
    async def prepare_intelligent_request(
        self,
        task: str,
        task_type: str,
        conversation_state: Any,
        unified_tracker: Any,
    ) -> Optional[Dict[str, Any]]:
        return await self.prepare_runtime_intelligence_request(
            task=task,
            task_type=task_type,
            conversation_state=conversation_state,
            unified_tracker=unified_tracker,
        )

    async def validate_intelligent_response(
        self,
        response: str,
        query: str,
        tool_calls: int,
        task_type: str,
    ) -> Optional[Dict[str, Any]]:
        return await self.validate_runtime_intelligence_response(
            response=response,
            query=query,
            tool_calls=tool_calls,
            task_type=task_type,
        )

    def record_intelligent_outcome(
        self,
        success: bool,
        quality_score: float,
        user_satisfied: bool,
        completed: bool,
        rl_coordinator: Any,
        stream_context: Any,
        vertical_context: Any,
        provider_name: str,
        model: str,
        tool_calls_used: int,
        continuation_prompts: int,
        max_continuation_prompts_used: int,
        stuck_loop_detected: bool,
    ) -> None:
        self.record_runtime_intelligence_outcome(
            success=success,
            quality_score=quality_score,
            user_satisfied=user_satisfied,
            completed=completed,
            rl_coordinator=rl_coordinator,
            stream_context=stream_context,
            vertical_context=vertical_context,
            provider_name=provider_name,
            model=model,
            tool_calls_used=tool_calls_used,
            continuation_prompts=continuation_prompts,
            max_continuation_prompts_used=max_continuation_prompts_used,
            stuck_loop_detected=stuck_loop_detected,
        )

    def should_continue_intelligent(self) -> tuple[bool, str]:
        return self.should_continue_runtime_intelligence()

    def emit_continuation_event(
        self,
        event_type: str,
        success: bool,
        quality_score: float,
        task_type: str,
        prompts_used: int,
        provider_name: str,
        model: str,
        stuck_detected: bool = False,
    ) -> None:
        """Emit RL event for continuation attempt.

        Args:
            event_type: Either "prompt" or "patience"
            success: Whether the continuation was successful
            quality_score: Quality score of the result
            task_type: Type of task
            prompts_used: Number of continuation prompts used
            provider_name: Provider name for the event
            model: Model name for the event
            stuck_detected: Whether a stuck loop was detected
        """
        try:
            from victor.framework.rl.hooks import get_rl_hooks, RLEvent, RLEventType

            hooks = get_rl_hooks()
            if hooks is None:
                return

            if event_type == "prompt":
                event = RLEvent(
                    type=RLEventType.CONTINUATION_PROMPT,
                    success=success,
                    quality_score=quality_score,
                    provider=provider_name,
                    model=model,
                    task_type=task_type,
                    metadata={
                        "prompts_used": prompts_used,
                    },
                )
            else:  # patience
                event = RLEvent(
                    type=RLEventType.CONTINUATION_ATTEMPT,
                    success=success,
                    quality_score=quality_score,
                    provider=provider_name,
                    model=model,
                    task_type=task_type,
                    metadata={
                        "prompts_used": prompts_used,
                        "stuck_detected": stuck_detected,
                    },
                )

            hooks.emit(event)

        except Exception as e:
            logger.debug(f"Continuation event emission failed: {e}")


# Convenience function for quick integration
async def enhance_orchestrator(
    orchestrator: "AgentOrchestrator",
    config: Optional[IntegrationConfig] = None,
) -> OrchestratorIntegration:
    """Enhance an orchestrator with runtime-intelligence capabilities.

    This is the main entry point for adding intelligent features to
    an existing orchestrator.

    Args:
        orchestrator: The AgentOrchestrator to enhance
        config: Optional configuration

    Returns:
        OrchestratorIntegration instance

    Example:
        orchestrator = AgentOrchestrator(settings, provider, model)
        integration = await enhance_orchestrator(orchestrator)

        # Use the enhanced flow
        context = await integration.prepare_request(task, task_type)
        # ... orchestrator does its work ...
        result = await integration.validate_response(response)
    """
    return await OrchestratorIntegration.create(orchestrator, config)
