"""Runtime-intelligence hooks for the orchestrator facade.

Moved verbatim from ``victor/agent/orchestrator.py`` (facade decomposition).
All methods delegate to ``OrchestratorIntegration`` via the
``runtime_intelligence_integration`` property installed on ``AgentOrchestrator``
by ``orchestrator_properties``; the mixin itself holds no state.
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class OrchestratorRuntimeIntelligenceMixin:
    """Runtime-intelligence pre/post hooks and back-compat aliases.

    The orchestrator facade exposes these so services and the agentic loop
    can drive runtime-intelligence (recommendations, validation, Q-learning
    outcome recording, continuation decisions) without importing the
    integration directly.
    """

    async def _prepare_runtime_intelligence_request(
        self, task: str, task_type: str
    ) -> Optional[Dict[str, Any]]:
        """Pre-request hook for runtime-intelligence integration.

        Delegates to OrchestratorIntegration.prepare_runtime_intelligence_request().

        Args:
            task: The user's task/query
            task_type: Detected task type (analysis, edit, etc.)

        Returns:
            Dictionary with recommendations, or None if pipeline disabled
        """
        integration = self.runtime_intelligence_integration
        if not integration:
            return None

        return await integration.prepare_runtime_intelligence_request(
            task=task,
            task_type=task_type,
            conversation_state=self.conversation_state,
            unified_tracker=self.unified_tracker,
        )

    async def _validate_runtime_intelligence_response(
        self,
        response: str,
        query: str,
        tool_calls: int,
        task_type: str,
    ) -> Optional[Dict[str, Any]]:
        """Post-response hook for runtime-intelligence integration.

        Delegates to OrchestratorIntegration.validate_runtime_intelligence_response().

        Args:
            response: The model's response content
            query: Original user query
            tool_calls: Number of tool calls made so far
            task_type: Task type for context

        Returns:
            Dictionary with quality/grounding scores, or None if pipeline disabled
        """
        integration = self.runtime_intelligence_integration
        if not integration:
            return None

        return await integration.validate_runtime_intelligence_response(
            response=response,
            query=query,
            tool_calls=tool_calls,
            task_type=task_type,
        )

    def _record_runtime_intelligence_outcome(
        self,
        success: bool,
        quality_score: float = 0.5,
        user_satisfied: bool = True,
        completed: bool = True,
    ) -> None:
        """Record outcome for Q-learning feedback.

        Delegates to OrchestratorIntegration.record_runtime_intelligence_outcome().

        Args:
            success: Whether the task was completed successfully
            quality_score: Final quality score (0.0-1.0)
            user_satisfied: Whether user seemed satisfied
            completed: Whether task reached completion
        """
        integration = self.runtime_intelligence_integration
        if not integration:
            return

        # Get orchestrator state to pass to integration
        stream_context = getattr(self, "_current_stream_context", None)
        continuation_prompts = getattr(self, "_continuation_prompts", 0)
        max_continuation_prompts_used = getattr(self, "_max_continuation_prompts_used", 6)
        stuck_loop_detected = getattr(self, "_stuck_loop_detected", False)

        try:
            integration.record_runtime_intelligence_outcome(
                success=success,
                quality_score=quality_score,
                user_satisfied=user_satisfied,
                completed=completed,
                rl_coordinator=self._rl_coordinator,
                stream_context=stream_context,
                vertical_context=self._vertical_context,
                provider_name=self.provider.name,
                model=self.model,
                tool_calls_used=self.tool_calls_used,
                continuation_prompts=continuation_prompts,
                max_continuation_prompts_used=max_continuation_prompts_used,
                stuck_loop_detected=stuck_loop_detected,
            )
        except Exception as e:
            logger.debug(f"RuntimeIntelligencePipeline record_outcome failed: {e}")

        # EvoTest: trigger session-end GEPA evolution (E12)
        if self.tool_calls_used >= 3:
            try:
                rl_coord = getattr(self, "_rl_coordinator", None)
                if rl_coord and hasattr(rl_coord, "try_evolve_on_session_end"):
                    result = rl_coord.try_evolve_on_session_end(
                        getattr(self.provider, "name", "unknown"), self.model
                    )
                    if result and isinstance(result, dict):
                        # Display evolution report via debug logger
                        debug_log = getattr(self, "debug_logger", None)
                        if debug_log and hasattr(debug_log, "log_evolution_report"):
                            debug_log.log_evolution_report(result)
            except Exception as e:
                logger.debug(f"[evotest] Session-end evolution failed: {e}")

    def _should_continue_runtime_intelligence(self) -> tuple[bool, str]:
        """Check if processing should continue using learned behaviors.

        Delegates to OrchestratorIntegration.should_continue_runtime_intelligence().

        Returns:
            Tuple of (should_continue, reason)
        """
        integration = self.runtime_intelligence_integration
        if not integration:
            return True, "Pipeline disabled"

        return integration.should_continue_runtime_intelligence()

    # Backward-compatible aliases while runtime-intelligence naming propagates.
    async def _prepare_intelligent_request(
        self, task: str, task_type: str
    ) -> Optional[Dict[str, Any]]:
        return await self._prepare_runtime_intelligence_request(task, task_type)

    async def _validate_intelligent_response(
        self,
        response: str,
        query: str,
        tool_calls: int,
        task_type: str,
    ) -> Optional[Dict[str, Any]]:
        return await self._validate_runtime_intelligence_response(
            response=response,
            query=query,
            tool_calls=tool_calls,
            task_type=task_type,
        )

    def _record_intelligent_outcome(
        self,
        success: bool,
        quality_score: float = 0.5,
        user_satisfied: bool = True,
        completed: bool = True,
    ) -> None:
        self._record_runtime_intelligence_outcome(
            success=success,
            quality_score=quality_score,
            user_satisfied=user_satisfied,
            completed=completed,
        )

    def _should_continue_intelligent(self) -> tuple[bool, str]:
        return self._should_continue_runtime_intelligence()
