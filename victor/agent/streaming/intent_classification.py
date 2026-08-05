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

"""Intent classification handler for streaming chat.

This module provides the IntentClassificationHandler class which encapsulates
the intent classification and continuation action determination logic that
previously lived inside ``ChatCoordinator._stream_chat_impl`` and is now
orchestrated by ``StreamingChatExecutor``.

The handler manages:
- Content yielding when no tool calls
- Intent text extraction and classification (with caching)
- Tracking variable management
- Response loop detection
- Task completion signal building
- Continuation action determination via ContinuationStrategy
- Action override logic (repeated response, grounding failure)

Design Pattern: Strategy + Facade
=================================
The handler acts as a facade over intent classification and continuation
strategy, providing a single entry point for determining what action
to take when the model responds without tool calls.

Usage:
    handler = IntentClassificationHandler(...)

    result = handler.classify_and_determine_action(
        stream_ctx=stream_ctx,
        full_content=full_content,
        content_length=content_length,
        mentioned_tools=mentioned_tools,
    )

    for chunk in result.chunks:
        yield chunk

    # Use result.action_result for continuation handling
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Set,
    TYPE_CHECKING,
)

from victor.agent.continuation_contract import ContinuationDirective
from victor.agent.continuation_strategy import (
    ContinuationActionType,
    coerce_continuation_action,
)
from victor.agent.services.protocols.streaming_runtime import (
    StreamingChunkRuntimeProtocol,
    StreamingConversationStateProtocol,
    StreamingIntentClassifierRuntimeProtocol,
    StreamingRLRuntimeProtocol,
    StreamingSanitizerRuntimeProtocol,
    StreamingTrackerRuntimeProtocol,
)
from victor.agent.streaming.context import StreamingChatContext
from victor.providers.base import StreamChunk

if TYPE_CHECKING:
    from victor.agent.orchestrator import AgentOrchestrator
    from victor.config.settings import Settings

logger = logging.getLogger(__name__)


# =============================================================================
# Result Types
# =============================================================================


@dataclass
class IntentClassificationResult:
    """Result of intent classification and action determination.

    Attributes:
        chunks: List of chunks to yield (content yielded to UI).
        action_result: The result from ContinuationStrategy with final action.
        action: The final action after overrides.
        state_updates: Dict of state updates to apply to orchestrator.
        content_cleared: Whether full_content was cleared after yielding.
    """

    chunks: List[StreamChunk] = field(default_factory=list)
    action_result: ContinuationDirective = field(
        default_factory=lambda: ContinuationDirective.from_legacy(
            action=ContinuationActionType.FINISH,
            reason="default_finish",
        )
    )
    action: ContinuationActionType = ContinuationActionType.FINISH
    state_updates: Dict[str, Any] = field(default_factory=dict)
    content_cleared: bool = False

    def __post_init__(self) -> None:
        """Coerce legacy stub payloads into the typed continuation contract."""

        self.action = coerce_continuation_action(self.action)
        if not isinstance(self.action_result, ContinuationDirective):
            legacy_payload = (
                dict(self.action_result) if isinstance(self.action_result, dict) else {}
            )
            self.action_result = ContinuationDirective.from_legacy(
                action=legacy_payload.get("action", self.action),
                reason=legacy_payload.get("reason", "legacy_action_result"),
                message=legacy_payload.get("message"),
                updates=legacy_payload.get("updates"),
                extracted_call=legacy_payload.get("extracted_call"),
                mentioned_tools=legacy_payload.get("mentioned_tools"),
                set_final_summary_requested=bool(
                    legacy_payload.get("set_final_summary_requested", False)
                ),
                set_max_prompts_summary_requested=bool(
                    legacy_payload.get("set_max_prompts_summary_requested", False)
                ),
            )

    def add_chunk(self, chunk: StreamChunk) -> None:
        """Add a chunk to yield."""
        self.chunks.append(chunk)


# =============================================================================
# Tracking State Container
# =============================================================================


@dataclass
class TrackingState:
    """Container for tracking variables used in intent classification.

    These are typically stored on the orchestrator but are passed here
    for cleaner dependency management.
    """

    continuation_prompts: int = 0
    asking_input_prompts: int = 0
    consecutive_blocked_attempts: int = 0
    cumulative_prompt_interventions: int = 0
    synthesis_nudge_count: int = 0
    max_prompts_summary_requested: bool = False
    final_summary_requested: bool = False
    force_finalize: bool = False
    required_files: Set[str] = field(default_factory=set)
    read_files_session: Set[str] = field(default_factory=set)
    required_outputs: Set[str] = field(default_factory=set)


# =============================================================================
# Intent Classification Handler
# =============================================================================


class IntentClassificationHandler:
    """Handler for intent classification and continuation action determination.

    This class encapsulates the logic for classifying model intent when
    no tool calls are made, and determining the appropriate continuation
    action via ContinuationStrategy.

    The handler manages:
    - Content yielding to UI when no tool calls
    - Intent classification with caching
    - Response loop detection
    - Task completion signal building
    - Continuation action determination
    - Action override logic

    Example:
        handler = IntentClassificationHandler(...)
        result = handler.classify_and_determine_action(stream_ctx, content, ...)

        for chunk in result.chunks:
            yield chunk
        # Process result.action_result with ContinuationHandler
    """

    def __init__(
        self,
        intent_classifier: StreamingIntentClassifierRuntimeProtocol,
        unified_tracker: StreamingTrackerRuntimeProtocol,
        sanitizer: StreamingSanitizerRuntimeProtocol,
        chunk_generator: StreamingChunkRuntimeProtocol,
        settings: "Settings",
        rl_coordinator: Optional[StreamingRLRuntimeProtocol] = None,
        conversation_state: Optional[StreamingConversationStateProtocol] = None,
        provider_name: str = "",
        model: str = "",
        tool_budget: int = 20,
        runtime_intelligence: Optional[Any] = None,
    ):
        """Initialize the intent classification handler.

        Args:
            intent_classifier: Classifier for determining model intent.
            unified_tracker: Tracker for response loops and config.
            sanitizer: Content sanitizer.
            chunk_generator: Generator for stream chunks.
            settings: Application settings.
            rl_coordinator: Optional RL coordinator for action determination.
            conversation_state: Optional conversation state for cycle detection.
            provider_name: Name of the LLM provider.
            model: Model name.
            tool_budget: Tool call budget.
            runtime_intelligence: Optional canonical runtime-intelligence service.
        """
        self._intent_classifier = intent_classifier
        self._unified_tracker = unified_tracker
        self._sanitizer = sanitizer
        self._chunk_generator = chunk_generator
        self._settings = settings
        self._rl_coordinator = rl_coordinator
        self._conversation_state = conversation_state
        self._provider_name = provider_name
        self._model = model
        self._tool_budget = tool_budget
        self._runtime_intelligence = runtime_intelligence

        # Loop termination evaluator — HybridLoopEvaluator routes to legacy or
        # agentic backend based on USE_STATEGRAPH_AGENTIC_LOOP feature flag.
        # One instance per handler so AgenticLoopEvaluator's score history
        # persists across iterations of the same streaming session.
        from victor.agent.loop_evaluation.hybrid import HybridLoopEvaluator

        self._loop_evaluator = HybridLoopEvaluator()

        # Intent cache for reducing embedding calls
        self._intent_cache: Dict[int, Any] = {}
        self._max_cache_size = 100

    def classify_and_determine_action(
        self,
        stream_ctx: StreamingChatContext,
        full_content: str,
        content_length: int,
        mentioned_tools: List[str],
        tracking_state: TrackingState,
    ) -> IntentClassificationResult:
        """Classify intent and determine continuation action.

        This is the main entry point that handles the complete intent
        classification phase when no tool calls are present.

        Args:
            stream_ctx: The streaming context.
            full_content: Full content from model response.
            content_length: Length of content.
            mentioned_tools: Tools mentioned but not called.
            tracking_state: Current tracking state from orchestrator.

        Returns:
            IntentClassificationResult with chunks, action_result, and updates.
        """
        result = IntentClassificationResult()
        tool_format_confusion = False

        # Step 1: Yield content to UI
        content_for_intent = full_content or ""
        if full_content:
            sanitized = self._sanitizer.sanitize(full_content)
            detect_confusion = getattr(self._sanitizer, "has_tool_format_confusion", None)
            if callable(detect_confusion):
                tool_format_confusion = bool(detect_confusion(full_content))
            if sanitized:
                logger.debug(f"Yielding content to UI: {len(sanitized)} chars")
                result.add_chunk(self._chunk_generator.generate_content_chunk(sanitized))
                stream_ctx.accumulate_content(sanitized)
                logger.debug(
                    f"Total accumulated content: {stream_ctx.total_accumulated_chars} chars"
                )
                result.content_cleared = True

        if tool_format_confusion:
            logger.warning(
                "Detected malformed tool-style plaintext in model response; "
                "finishing early to prevent a continuation loop"
            )
            result.add_chunk(
                self._chunk_generator.generate_content_chunk(
                    "\n\n[Model emitted malformed tool-style text instead of valid tool "
                    "calls. Stopping to prevent a loop. Try rerunning or switching provider.]",
                    is_final=True,
                )
            )
            result.action_result = ContinuationDirective.from_legacy(
                action=ContinuationActionType.FINISH,
                reason="Malformed tool-style plaintext detected instead of valid tool calls",
            )
            result.action = ContinuationActionType.FINISH
            return result

        # Step 2: Extract intent text (last 500 chars for pattern matching)
        intent_text = content_for_intent
        if len(intent_text) > 500:
            intent_text = intent_text[-500:]

        # Step 3: Classify intent (with caching)
        intent_result = self._classify_intent_cached(intent_text)

        # Step 4: Check for response loop
        is_repeated_response = self._unified_tracker.check_response_loop(full_content or "")

        # Step 5: Build task completion signals
        task_completion_signals = self._build_task_completion_signals(tracking_state, stream_ctx)

        # Step 6: Determine continuation action
        action_result = self._determine_action(
            intent_result=intent_result,
            stream_ctx=stream_ctx,
            content_length=content_length,
            full_content=full_content if not result.content_cleared else "",
            mentioned_tools=mentioned_tools,
            tracking_state=tracking_state,
            task_completion_signals=task_completion_signals,
        )

        # Step 7: Apply state updates
        self._apply_state_updates(action_result, result)

        # Step 8: Determine final action with overrides
        action = coerce_continuation_action(action_result.get("action"))
        action = self._apply_action_overrides(action, is_repeated_response, tracking_state)
        result.action_result = action_result.with_action(action)
        result.action = action

        return result

    def _classify_intent_cached(self, intent_text: str) -> Any:
        """Classify intent with caching for performance."""
        cache_key = hash(intent_text)

        if cache_key in self._intent_cache:
            intent_result = self._intent_cache[cache_key]
            logger.debug(f"Intent classification (cached): {intent_result.intent.name}")
            return intent_result

        intent_result = self._intent_classifier.classify_intent_sync(intent_text)

        # Cache result (limit size to prevent memory bloat)
        if len(self._intent_cache) < self._max_cache_size:
            self._intent_cache[cache_key] = intent_result

        logger.debug(
            f"Intent classification: {intent_result.intent.name} "
            f"(confidence={intent_result.confidence:.3f}, "
            f"text_len={len(intent_text)}, "
            f"top_matches={intent_result.top_matches[:3]})"
        )

        return intent_result

    def _build_task_completion_signals(
        self, tracking_state: TrackingState, stream_ctx: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Build task completion signals for early termination detection.

        Args:
            tracking_state: Current tracking state
            stream_ctx: Optional streaming context for iteration count

        Returns:
            Dictionary with task completion signals
        """
        # Get cycle count from conversation state
        cycle_count = 0
        if self._conversation_state:
            try:
                state_summary = self._conversation_state.get_state_summary()
                if isinstance(state_summary, dict):
                    cycle_count = state_summary.get("transition_count", 0)
                    # Also check stage visit counts
                    if hasattr(self._conversation_state, "_history"):
                        history = self._conversation_state._history
                        if hasattr(history, "get_max_visit_count"):
                            cycle_count = max(cycle_count, history.get_max_visit_count())
            except Exception:
                pass  # Don't fail on state access errors

        # AGENTIC LOOP FIX: Include current iteration for loop detection
        current_iteration = 0
        if stream_ctx and hasattr(stream_ctx, "total_iterations"):
            current_iteration = stream_ctx.total_iterations

        explicit_database_query_requested = False
        executed_tool_names: Set[str] = set()
        database_query_satisfied = False
        direct_response_requested = False
        original_user_message = ""
        if stream_ctx:
            try:
                from victor.agent.action_authorizer import (
                    has_explicit_readonly_shell_request,
                )
                from victor.framework.task.direct_response import (
                    classify_direct_response_prompt,
                )

                original_user_message = getattr(stream_ctx, "user_message", "") or ""
                explicit_database_query_requested = has_explicit_readonly_shell_request(
                    original_user_message
                )
                direct_response_requested = classify_direct_response_prompt(
                    original_user_message
                ).is_direct_response
            except Exception:
                explicit_database_query_requested = False
                direct_response_requested = False

            executed_tool_names = set(getattr(stream_ctx, "executed_tool_names", set()) or set())
            database_query_satisfied = bool(executed_tool_names & {"shell", "db", "database"})

        return {
            "required_files": tracking_state.required_files,
            "read_files": tracking_state.read_files_session,
            "required_outputs": tracking_state.required_outputs,
            "all_files_read": (
                len(tracking_state.required_files) > 0
                and tracking_state.read_files_session.issuperset(tracking_state.required_files)
            ),
            "cycle_count": cycle_count,
            "synthesis_nudge_count": tracking_state.synthesis_nudge_count,
            "cumulative_prompt_interventions": tracking_state.cumulative_prompt_interventions,
            "current_iteration": current_iteration,  # For algorithmic loop detection
            # Budget-aware loop detection: sum of tool calls in last 5 iterations
            # (precise productivity signal, falls back to cumulative unique tools
            # if rolling window not maintained on stream_ctx)
            "recent_tool_call_count": (
                getattr(stream_ctx, "recent_tool_call_count", None)
                if stream_ctx is not None and hasattr(stream_ctx, "recent_tool_call_count")
                else len(executed_tool_names)
            ),
            "original_user_message": original_user_message,
            "executed_tool_names": executed_tool_names,
            "explicit_database_query_requested": explicit_database_query_requested,
            "database_query_satisfied": database_query_satisfied,
            "is_qa_task": (bool(getattr(stream_ctx, "is_qa_task", False)) if stream_ctx else False),
            "direct_response_requested": direct_response_requested,
        }

    def _resolve_one_shot_mode(self) -> bool:
        """Resolve one-shot mode from the canonical nested settings group.

        The headless/automation flags now live under ``settings.automation``.
        Keep a flat fallback for compatibility with older test doubles or
        transitional callers.
        """
        automation = getattr(self._settings, "automation", None)
        nested_value = getattr(automation, "one_shot_mode", None)
        if nested_value is not None:
            return bool(nested_value)
        return bool(getattr(self._settings, "one_shot_mode", False))

    def _determine_action(
        self,
        intent_result: Any,
        stream_ctx: StreamingChatContext,
        content_length: int,
        full_content: str,
        mentioned_tools: List[str],
        tracking_state: TrackingState,
        task_completion_signals: Dict[str, Any],
    ) -> "ContinuationDirective":
        """Determine continuation action via the HybridLoopEvaluator.

        Builds a ``LoopContext`` from all available turn information and
        delegates to ``self._loop_evaluator`` (either ``LegacyEvaluator``
        wrapping ``ContinuationStrategy``, or ``AgenticLoopEvaluator`` using
        PERCEIVE→EVALUATE→DECIDE, based on ``USE_STATEGRAPH_AGENTIC_LOOP``).

        The returned ``LoopDecision`` is converted back to a
        ``ContinuationDirective`` for downstream compatibility.
        """
        from victor.agent.loop_evaluation.protocol import LoopContext
        from victor.framework.task.direct_response import (
            classify_direct_response_prompt,
        )

        one_shot_mode = self._resolve_one_shot_mode()

        # Determine if the original user message is a direct-response prompt.
        original_user_message = task_completion_signals.get("original_user_message", "") or ""
        is_direct_response = task_completion_signals.get(
            "direct_response_requested",
            (
                classify_direct_response_prompt(original_user_message).is_direct_response
                if original_user_message
                else False
            ),
        )

        ctx = LoopContext(
            user_message=original_user_message,
            task_type=getattr(stream_ctx, "coarse_task_type", "default"),
            is_analysis_task=stream_ctx.is_analysis_task,
            is_action_task=stream_ctx.is_action_task,
            is_direct_response=bool(is_direct_response),
            full_content=full_content or "",
            content_length=content_length,
            mentioned_tools=mentioned_tools or [],
            intent_result=intent_result,
            iteration=stream_ctx.total_iterations,
            continuation_prompts=tracking_state.continuation_prompts,
            asking_input_prompts=tracking_state.asking_input_prompts,
            max_prompts_summary_requested=tracking_state.max_prompts_summary_requested,
            force_tool_execution_attempts=getattr(stream_ctx, "force_tool_execution_attempts", 0),
            synthesis_nudge_count=tracking_state.synthesis_nudge_count,
            quality_score=getattr(stream_ctx, "last_quality_score", 0.0) or 0.0,
            task_completion_signals=task_completion_signals,
            one_shot_mode=one_shot_mode,
            compaction_occurred=getattr(stream_ctx, "compaction_occurred", False),
            compaction_messages_removed=getattr(stream_ctx, "compaction_message_removed_count", 0),
            degraded_resume_state=bool(getattr(stream_ctx, "degraded_resume_state", False)),
            resume_summary=str(getattr(stream_ctx, "resume_summary", "") or ""),
            settings=self._settings,
            rl_coordinator=self._rl_coordinator,
            provider_name=self._provider_name,
            model=self._model,
            tool_budget=self._tool_budget,
            unified_tracker_config=self._unified_tracker.config,
            runtime_intelligence=self._runtime_intelligence,
        )

        decision = self._loop_evaluator.evaluate(ctx)
        return decision.to_directive()

    def _apply_state_updates(
        self, action_result: ContinuationDirective, result: IntentClassificationResult
    ) -> None:
        """Apply state updates from action result."""
        updates = action_result.state_patch.to_updates_dict()

        if "continuation_prompts" in updates:
            result.state_updates["continuation_prompts"] = updates["continuation_prompts"]
        if "asking_input_prompts" in updates:
            result.state_updates["asking_input_prompts"] = updates["asking_input_prompts"]
        if "synthesis_nudge_count" in updates:
            result.state_updates["synthesis_nudge_count"] = updates["synthesis_nudge_count"]

        if action_result.state_patch.final_summary_requested:
            result.state_updates["final_summary_requested"] = True
        if action_result.state_patch.max_prompts_summary_requested:
            result.state_updates["max_prompts_summary_requested"] = True

    def _apply_action_overrides(
        self,
        action: ContinuationActionType,
        is_repeated_response: bool,
        tracking_state: TrackingState,
    ) -> ContinuationActionType:
        """Apply action overrides based on conditions."""
        # Override: Repeated response detected
        if is_repeated_response and action in (
            ContinuationActionType.PROMPT_TOOL_CALL,
            ContinuationActionType.REQUEST_SUMMARY,
        ):
            logger.info(
                "Continuation action: finish - " "Overriding to finish due to repeated response"
            )
            return ContinuationActionType.FINISH

        # Override: Force finalize from grounding failure
        if tracking_state.force_finalize:
            logger.info(
                "Continuation action: finish - "
                "Overriding to finish due to grounding failure limit"
            )
            return ContinuationActionType.FINISH

        return action


# =============================================================================
# Factory Function
# =============================================================================


def create_intent_classification_handler(
    orchestrator: "AgentOrchestrator",
) -> IntentClassificationHandler:
    """Factory function to create an IntentClassificationHandler from an orchestrator.

    Args:
        orchestrator: The AgentOrchestrator instance.

    Returns:
        Configured IntentClassificationHandler.
    """
    runtime_intelligence = getattr(
        getattr(orchestrator, "__dict__", {}),
        "get",
        lambda *_args, **_kwargs: None,
    )("_runtime_intelligence")

    return IntentClassificationHandler(
        intent_classifier=orchestrator.intent_classifier,
        unified_tracker=orchestrator.unified_tracker,
        sanitizer=orchestrator.sanitizer,
        chunk_generator=orchestrator._chunk_generator,
        settings=orchestrator.settings,
        rl_coordinator=orchestrator._rl_coordinator,
        conversation_state=getattr(orchestrator, "conversation_state", None),
        provider_name=orchestrator.provider.name,
        model=orchestrator.model,
        tool_budget=orchestrator.tool_budget,
        runtime_intelligence=runtime_intelligence,
    )


def create_tracking_state(orchestrator: "AgentOrchestrator") -> TrackingState:
    """Create a TrackingState from orchestrator state.

    Args:
        orchestrator: The AgentOrchestrator instance.

    Returns:
        TrackingState populated from orchestrator.
    """
    return TrackingState(
        continuation_prompts=getattr(orchestrator, "_continuation_prompts", 0),
        asking_input_prompts=getattr(orchestrator, "_asking_input_prompts", 0),
        consecutive_blocked_attempts=getattr(orchestrator, "_consecutive_blocked_attempts", 0),
        cumulative_prompt_interventions=getattr(
            orchestrator, "_cumulative_prompt_interventions", 0
        ),
        synthesis_nudge_count=getattr(orchestrator, "_synthesis_nudge_count", 0),
        max_prompts_summary_requested=getattr(
            orchestrator, "_max_prompts_summary_requested", False
        ),
        final_summary_requested=getattr(orchestrator, "_final_summary_requested", False),
        force_finalize=getattr(orchestrator, "_force_finalize", False),
        required_files=set(getattr(orchestrator, "_required_files", [])),
        read_files_session=getattr(orchestrator, "_read_files_session", set()),
        required_outputs=set(getattr(orchestrator, "_required_outputs", [])),
    )


def apply_tracking_state_updates(
    orchestrator: "AgentOrchestrator",
    updates: Dict[str, Any],
    force_finalize_used: bool = False,
) -> None:
    """Apply tracking state updates back to orchestrator.

    Args:
        orchestrator: The AgentOrchestrator instance.
        updates: Dict of state updates from IntentClassificationResult.
        force_finalize_used: Whether force_finalize was used (reset it).
    """
    if "continuation_prompts" in updates:
        orchestrator._continuation_prompts = updates["continuation_prompts"]
    if "asking_input_prompts" in updates:
        orchestrator._asking_input_prompts = updates["asking_input_prompts"]
    if "synthesis_nudge_count" in updates:
        orchestrator._synthesis_nudge_count = updates["synthesis_nudge_count"]
    if updates.get("final_summary_requested"):
        orchestrator._final_summary_requested = True
    if updates.get("max_prompts_summary_requested"):
        orchestrator._max_prompts_summary_requested = True
    if force_finalize_used:
        orchestrator._force_finalize = False
