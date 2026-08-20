# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Shared streaming helper implementations for the canonical chat service path."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, List, Optional

from victor.agent.conversation.history_metadata import build_internal_history_metadata
from victor.agent.paradigm_router import get_paradigm_router
from victor.agent.prompt_requirement_extractor import extract_prompt_requirements
from victor.agent.topology_contract import TopologyAction
from victor.agent.topology_grounder import TopologyGrounder
from victor.agent.topology_selector import TopologySelector
from victor.agent.topology_telemetry import (
    build_topology_telemetry_event,
    emit_topology_telemetry_event,
)
from victor.agent.runtime.context import AgentRuntimeContext
from victor.agent.services.context_service import compact_context_if_recommended
from victor.agent.unified_task_tracker import TrackerTaskType
from victor.core.loop_thresholds import DEFAULT_BLOCKED_CONSECUTIVE_THRESHOLD
from victor.core.errors import (
    ProviderAuthError,
    ProviderConnectionError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from victor.framework.task.direct_response import classify_direct_response_prompt
from victor.framework.request_scope_heuristics import (
    is_ambiguous_write_followup_request,
)
from victor.framework.topology_runtime import prepare_topology_runtime_contract
from victor.framework.task import TaskComplexity
from victor.providers.base import Message, StreamChunk
from victor.providers.openai_compat import consume_last_tool_message_cleanup_stats

if TYPE_CHECKING:
    from victor.agent.streaming.context import StreamingChatContext

logger = logging.getLogger(__name__)
_MISSING = object()


class ChatStreamHelperMixin:
    """Shared streaming helper methods reused by service and compatibility shims."""

    @staticmethod
    def _get_runtime_state_host(runtime_host: Any) -> Any:
        """Return the concrete object that owns raw runtime instance state."""
        resolver = (
            getattr(runtime_host, "_resolve_runtime_state_host", None)
            if hasattr(type(runtime_host), "_resolve_runtime_state_host")
            else None
        )
        if callable(resolver):
            return resolver(runtime_host)
        instance_dict = getattr(runtime_host, "__dict__", {})
        if isinstance(instance_dict, dict) and "_orchestrator" in instance_dict:
            return instance_dict["_orchestrator"]
        return runtime_host

    @classmethod
    def _get_runtime_state_dict(cls, runtime_host: Any) -> Dict[str, Any]:
        """Return the instance dictionary for raw runtime state lookups."""
        resolver = (
            getattr(runtime_host, "_resolve_runtime_state_dict", None)
            if hasattr(type(runtime_host), "_resolve_runtime_state_dict")
            else None
        )
        if callable(resolver):
            resolved = resolver(runtime_host)
            if isinstance(resolved, dict):
                return resolved
        state_host = cls._get_runtime_state_host(runtime_host)
        instance_dict = getattr(state_host, "__dict__", {})
        return instance_dict if isinstance(instance_dict, dict) else {}

    def _has_runtime_capability(self, name: str) -> bool:
        """Return whether the active runtime exposes a named capability."""
        resolver = getattr(self, "_resolve_runtime_capability_presence", None)
        if callable(resolver):
            return bool(resolver(name))

        orch = self._orchestrator
        checker = getattr(orch, "has_capability", None)
        if callable(checker):
            try:
                return bool(checker(name))
            except Exception:
                logger.debug("Capability probe failed for %s", name, exc_info=True)

        state_dict = self._get_runtime_state_dict(orch)
        return name in state_dict or f"_{name}" in state_dict

    def _get_runtime_capability_value(self, name: str, default: Any = None) -> Any:
        """Return a named runtime capability or state-backed value."""
        resolver = getattr(self, "_resolve_runtime_capability_value", None)
        if callable(resolver):
            return resolver(name, default)

        orch = self._orchestrator
        getter = getattr(orch, "get_capability_value", None)
        if callable(getter):
            try:
                value = getter(name)
                if value is not None:
                    return value
            except Exception:
                logger.debug("Capability read failed for %s", name, exc_info=True)

        state_dict = self._get_runtime_state_dict(orch)
        if name in state_dict:
            return state_dict.get(name)
        return state_dict.get(f"_{name}", default)

    def _get_last_stream_task_context(self) -> Optional[Dict[str, Any]]:
        """Return the last persisted streaming task context snapshot, if any."""
        context = self._get_runtime_capability_value("last_stream_task_context")
        return context if isinstance(context, dict) else None

    @staticmethod
    def _should_promote_general_task_to_edit(user_message: str) -> bool:
        """Return whether an underspecified write follow-up should be treated as edit work."""
        from victor.agent.action_authorizer import ActionIntent, detect_intent

        intent_result = detect_intent(user_message)
        return (
            intent_result.intent == ActionIntent.WRITE_ALLOWED
            and is_ambiguous_write_followup_request(user_message)
        )

    def _resolve_continuation_task_context(
        self,
        user_message: str,
        detected_task_type: TrackerTaskType,
    ) -> Optional[Dict[str, Any]]:
        """Return prior task context when a continuation should inherit task shape."""
        from victor.agent.action_authorizer import split_continuation_request

        is_continuation, continuation_payload = split_continuation_request(user_message)
        if not is_continuation:
            return None

        last_context = self._get_last_stream_task_context()
        if not last_context:
            return None

        prior_task_type = last_context.get("unified_task_type")
        if isinstance(prior_task_type, str):
            try:
                prior_task_type = TrackerTaskType(prior_task_type)
            except ValueError:
                prior_task_type = None
        if not isinstance(prior_task_type, TrackerTaskType):
            return None

        payload = continuation_payload.strip()
        bare_continuation = not payload
        carry_forward_task_shape = bare_continuation or (
            detected_task_type == TrackerTaskType.GENERAL
            and prior_task_type != TrackerTaskType.GENERAL
        )
        carry_forward_resume_context = carry_forward_task_shape or (
            bool(payload) and is_ambiguous_write_followup_request(payload)
        )
        if not carry_forward_resume_context:
            return None

        resolved = dict(last_context)
        resolved["unified_task_type"] = prior_task_type
        resolved["continuation_payload"] = payload
        resolved["bare_continuation"] = bare_continuation
        resolved["carry_forward_task_shape"] = carry_forward_task_shape
        resolved["carry_forward_resume_context"] = carry_forward_resume_context
        return resolved

    async def _handle_context_and_iteration_limits(
        self,
        user_message: str,
        max_total_iterations: int,
        max_context: int,
        total_iterations: int,
        last_quality_score: float,
    ) -> tuple[bool, Optional[StreamChunk]]:
        """Compatibility delegate for context and iteration limit handling."""
        orch = self._orchestrator

        state_dict = self._get_runtime_state_dict(orch)
        chat_service = state_dict.get("_chat_service")
        if chat_service is None:
            chat_service = getattr(orch, "_chat_service", None)
        service_handler = getattr(chat_service, "handle_context_and_iteration_limits", None)
        if callable(service_handler):
            return await service_handler(
                user_message,
                max_total_iterations,
                max_context,
                total_iterations,
                last_quality_score,
            )

        state_host = self._get_runtime_state_host(orch)
        runtime_getter = getattr(state_host, "_get_context_limit_runtime", None)
        if callable(runtime_getter):
            runtime = runtime_getter()
            runtime_handler = getattr(runtime, "handle_limits", None)
            if callable(runtime_handler):
                return await runtime_handler(
                    user_message,
                    max_total_iterations,
                    max_context,
                    total_iterations,
                    last_quality_score,
                )

        return False, None

    async def _prepare_stream(self, user_message: str, **kwargs: Any) -> tuple[
        Any,
        float,
        float,
        Dict[str, int],
        int,
        int,
        int,
        bool,
        TrackerTaskType,
        Any,
        int,
    ]:
        """Prepare streaming state and return commonly used values."""
        orch = self._orchestrator

        orch._cancel_event = asyncio.Event()
        orch._is_streaming = True

        stream_metrics = orch._metrics_collector.init_stream_metrics()
        start_time = stream_metrics.start_time
        estimated_content_tokens: float = 0

        cumulative_usage: Dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        }

        orch.conversation.ensure_system_prompt()
        orch._system_added = True
        orch._session_state.reset_for_new_turn()
        orch.unified_tracker.reset()
        orch.reminder_manager.reset()

        usage_analytics = self._get_runtime_capability_value("usage_analytics")
        if self._has_runtime_capability("usage_analytics") and usage_analytics:
            usage_analytics.start_session()

        tool_sequence_tracker = self._get_runtime_capability_value("tool_sequence_tracker")
        if self._has_runtime_capability("tool_sequence_tracker") and tool_sequence_tracker:
            tool_sequence_tracker.clear_history()

        if orch._context_manager and hasattr(orch._context_manager, "start_background_compaction"):
            await orch._context_manager.start_background_compaction(interval_seconds=15.0)

        max_total_iterations = orch.unified_tracker.config.get("max_total_iterations", 50)

        fallback_iteration = kwargs.get("_fallback_iteration", 0)
        if fallback_iteration > 0:
            total_iterations = fallback_iteration
            logger.info(
                f"[Fallback] Preserving iteration count: continuing from iteration {total_iterations} "
                f"(preserved {total_iterations} iterations of progress)"
            )
        else:
            total_iterations = 0

        force_completion = False

        from victor.agent.conversation.types import (
            MESSAGE_SOURCE_METADATA_KEY,
            MessageSource,
        )

        orch.add_message(
            "user",
            user_message,
            metadata={MESSAGE_SOURCE_METADATA_KEY: MessageSource.USER_TYPED.value},
        )

        if self._has_runtime_capability("usage_analytics") and usage_analytics:
            usage_analytics.record_turn()

        unified_task_type = orch.unified_tracker.detect_task_type(user_message)
        continuation_task_context = self._resolve_continuation_task_context(
            user_message, unified_task_type
        )
        orch._pending_continuation_task_context = continuation_task_context
        if continuation_task_context is not None and continuation_task_context.get(
            "carry_forward_task_shape"
        ):
            prior_task_type = continuation_task_context["unified_task_type"]
            if prior_task_type != unified_task_type:
                orch.unified_tracker.set_task_type(prior_task_type)
                unified_task_type = prior_task_type
            logger.info(
                "Continuation request detected; carrying forward prior task type: %s",
                unified_task_type.value,
            )
        elif continuation_task_context is not None:
            logger.info(
                "Continuation request detected; reusing prior resume context without "
                "overriding current task shape"
            )
        elif (
            unified_task_type == TrackerTaskType.GENERAL
            and self._should_promote_general_task_to_edit(user_message)
        ):
            orch.unified_tracker.set_task_type(TrackerTaskType.EDIT)
            unified_task_type = TrackerTaskType.EDIT
            logger.info(
                "Promoted general task type to edit for explicit write-authorized follow-up request"
            )
        logger.info(f"Task type detected: {unified_task_type.value}")

        # Publish the settled type. Detection, continuation carry-forward, and the
        # edit promotion above have all had their say by this point, so this is
        # the first moment the turn's task type is final.
        #
        # Every consumer downstream read an attribute nobody assigned:
        # ``TurnContext.task_type`` fell back to its "default" literal, that value
        # rode into every RLEvent, and RL_OUTCOME therefore recorded "default" for
        # every row ever written. Prompt evolution reads task_type back off those
        # rows to scope candidates to the work they were learned from, so with one
        # value for everything it could not distinguish an edit turn from a search
        # — the population dimension existed in the schema and carried no
        # information. The tracker was likewise only updated on the continuation
        # branch, leaving the tool-selection guard reading GENERAL for real edits.
        orch.unified_tracker.set_task_type(unified_task_type)
        orch._current_task_type = unified_task_type.value

        prompt_requirements = extract_prompt_requirements(user_message)
        if prompt_requirements.has_explicit_requirements():
            orch.unified_tracker._progress.has_prompt_requirements = True

            if (
                prompt_requirements.tool_budget
                and prompt_requirements.tool_budget > orch.unified_tracker._progress.tool_budget
            ):
                orch.unified_tracker.set_tool_budget(prompt_requirements.tool_budget)
                logger.info(
                    f"Dynamic budget from prompt: {prompt_requirements.tool_budget} "
                    f"(files={prompt_requirements.file_count}, fixes={prompt_requirements.fix_count})"
                )

            if (
                prompt_requirements.iteration_budget
                and prompt_requirements.iteration_budget
                > orch.unified_tracker._task_config.max_exploration_iterations
            ):
                orch.unified_tracker.set_max_iterations(prompt_requirements.iteration_budget)
                logger.info(
                    f"Dynamic iterations from prompt: {prompt_requirements.iteration_budget}"
                )

        direct_response = classify_direct_response_prompt(user_message)
        intelligent_task = None
        if not direct_response.is_direct_response:
            intelligent_task = asyncio.create_task(
                orch._prepare_runtime_intelligence_request(
                    task=user_message,
                    task_type=unified_task_type.value,
                )
            )

        max_exploration_iterations = orch.unified_tracker.max_exploration_iterations

        task_classification, complexity_tool_budget = self._prepare_task(
            user_message, unified_task_type
        )
        if continuation_task_context is not None:
            prior_task_classification = continuation_task_context.get("task_classification")
            prior_budget = continuation_task_context.get("complexity_tool_budget")
            current_complexity = getattr(task_classification, "complexity", None)
            if (
                continuation_task_context.get("carry_forward_task_shape")
                and prior_task_classification is not None
                and (
                    continuation_task_context.get("bare_continuation")
                    or current_complexity == TaskComplexity.SIMPLE
                )
            ):
                task_classification = prior_task_classification
                prior_complexity = getattr(prior_task_classification, "complexity", None)
                if prior_budget is not None:
                    complexity_tool_budget = int(prior_budget)
                    orch.unified_tracker.set_tool_budget(complexity_tool_budget)
                logger.info(
                    "Continuation request detected; carrying forward prior task complexity: %s",
                    getattr(prior_complexity, "value", prior_complexity),
                )

        intelligent_context = await intelligent_task if intelligent_task is not None else None
        if intelligent_context:
            if intelligent_context.get("system_prompt_addition"):
                from victor.agent.conversation.types import MessageSource

                orch.add_message(
                    "user",
                    # No "[SYSTEM-REMINDER:" prefix: the authenticated envelope
                    # (FEP-0026) already marks this as runtime guidance, and it is
                    # the only marker the model can verify. An inner unkeyed label
                    # would teach it that unverifiable provenance markers are normal.
                    intelligent_context["system_prompt_addition"],
                    metadata=build_internal_history_metadata(
                        "system_reminder", source=MessageSource.AGENT_GUIDANCE
                    ),
                )
                logger.debug("Injected intelligent pipeline optimized prompt")
        elif direct_response.is_direct_response:
            logger.debug("Skipping intelligent prompt injection for direct-response prompt")

        return (
            stream_metrics,
            start_time,
            estimated_content_tokens,
            cumulative_usage,
            max_total_iterations,
            max_exploration_iterations,
            total_iterations,
            force_completion,
            unified_task_type,
            task_classification,
            complexity_tool_budget,
        )

    async def _create_stream_context(
        self, user_message: str, **kwargs: Any
    ) -> "StreamingChatContext":
        """Create a StreamingChatContext with all prepared data."""
        from victor.agent.streaming import create_stream_context

        orch = self._orchestrator
        (
            stream_metrics,
            start_time,
            estimated_content_tokens,
            cumulative_usage,
            max_total_iterations,
            max_exploration_iterations,
            total_iterations,
            force_completion,
            unified_task_type,
            task_classification,
            complexity_tool_budget,
        ) = await self._prepare_stream(user_message, **kwargs)

        task_keywords = orch._classify_task_keywords(user_message)
        continuation_task_context = getattr(orch, "_pending_continuation_task_context", None)
        if isinstance(continuation_task_context, dict) and continuation_task_context.get(
            "carry_forward_task_shape"
        ):
            prior_coarse = continuation_task_context.get("coarse_task_type")
            if prior_coarse and task_keywords.get("coarse_task_type") in (
                None,
                "default",
                "general",
            ):
                task_keywords["coarse_task_type"] = prior_coarse
            for key in ("is_analysis_task", "is_action_task", "needs_execution"):
                if key in continuation_task_context and not task_keywords.get(key):
                    task_keywords[key] = bool(continuation_task_context.get(key))

        ctx = create_stream_context(
            user_message=user_message,
            max_iterations=max_total_iterations,
            max_exploration=max_exploration_iterations,
            tool_budget=complexity_tool_budget,
            max_blocked_before_force=getattr(
                orch.settings,
                "recovery_blocked_consecutive_threshold",
                DEFAULT_BLOCKED_CONSECUTIVE_THRESHOLD,
            ),
        )

        ctx.stream_metrics = stream_metrics
        ctx.start_time = start_time
        ctx.estimated_content_tokens = estimated_content_tokens
        ctx.cumulative_usage = cumulative_usage
        ctx.total_iterations = total_iterations
        ctx.force_completion = force_completion
        ctx.unified_task_type = unified_task_type
        ctx.task_classification = task_classification
        ctx.complexity_tool_budget = complexity_tool_budget

        task_type_val = task_keywords.get("task_type", "default")
        unified_task_type_val = getattr(unified_task_type, "value", "")
        unified_is_action = unified_task_type_val in ("edit", "create", "create_simple")
        classification_complexity = getattr(task_classification, "complexity", None)
        classification_is_action = classification_complexity == TaskComplexity.ACTION
        classification_is_analysis = classification_complexity == TaskComplexity.ANALYSIS
        # Use keyword-based classification as fallback when embedding classifier returns GENERAL
        # but keywords suggest analysis (e.g., "structural analysis", "framework analysis")
        keyword_is_analysis = task_keywords.get(
            "is_analysis_task",
            task_type_val in ("analysis", "analyze"),
        )
        unified_is_analysis = unified_task_type.value in ("analyze", "analysis")

        # Fallback: if unified says GENERAL but keywords say analysis, trust keywords
        ctx.is_analysis_task = (
            keyword_is_analysis
            or unified_is_analysis
            or classification_is_analysis
            or (unified_task_type.value == "general" and keyword_is_analysis)
        )
        ctx.is_action_task = (
            bool(task_keywords.get("is_action_task"))
            or unified_is_action
            or classification_is_action
        )
        ctx.needs_execution = (
            bool(task_keywords.get("needs_execution"))
            or task_type_val in ("execution", "action")
            or classification_is_action
        )
        coarse_task_type = task_keywords.get("coarse_task_type", task_type_val)
        if coarse_task_type in (None, "default", "general") and (
            unified_is_action or classification_is_action
        ):
            coarse_task_type = "action"
        ctx.coarse_task_type = coarse_task_type

        if task_classification and hasattr(task_classification, "complexity"):
            ctx.is_complex_task = task_classification.complexity in (
                TaskComplexity.COMPLEX,
                TaskComplexity.ANALYSIS,
            )

        from victor.agent.services.turn_execution_runtime import TurnExecutor

        ctx.is_qa_task = TurnExecutor._is_question_only(user_message)
        ctx.goals = orch._tool_planner.infer_goals_from_message(user_message)
        ctx.tool_budget = orch.tool_budget
        ctx.tool_calls_used = orch.tool_calls_used
        ctx.task_completion_detector = orch._task_completion_detector
        if isinstance(continuation_task_context, dict):
            ctx.degraded_resume_state = bool(
                continuation_task_context.get("degraded_resume_state", False)
            )
            resume_summary = continuation_task_context.get("resume_summary")
            if resume_summary:
                ctx.resume_summary = str(resume_summary)
            recent_resources = continuation_task_context.get("resume_recent_resources") or []
            if isinstance(recent_resources, list):
                ctx.resume_recent_resources = [str(item) for item in recent_resources if item]
            recent_tools = continuation_task_context.get("resume_recent_tools") or []
            if isinstance(recent_tools, list):
                ctx.resume_recent_tools = [str(item) for item in recent_tools if item]
            task_intent = continuation_task_context.get("task_intent")
            if task_intent:
                ctx.task_intent = str(task_intent)
            plan_steps = continuation_task_context.get("plan_steps") or []
            if isinstance(plan_steps, list):
                ctx.plan_steps = [str(item) for item in plan_steps if item][:8]
            intent_log = continuation_task_context.get("intent_log") or []
            if isinstance(intent_log, list):
                ctx.intent_log = [item for item in intent_log if isinstance(item, dict)][-12:]
            last_compaction_policy_reason = continuation_task_context.get(
                "last_compaction_policy_reason"
            )
            if last_compaction_policy_reason:
                ctx.last_compaction_policy_reason = str(last_compaction_policy_reason)
        await self._initialize_stream_topology_context(ctx, user_message)
        orch._pending_continuation_task_context = None

        return ctx

    async def _initialize_stream_topology_context(
        self,
        stream_ctx: "StreamingChatContext",
        user_message: str,
    ) -> None:
        """Build and ground a topology plan for the streaming runtime."""
        if not self._is_stream_topology_enabled() or stream_ctx.topology_plan is not None:
            return

        orch = self._orchestrator
        task_classification = stream_ctx.task_classification
        task_type = str(
            getattr(task_classification, "task_type", None)
            or stream_ctx.coarse_task_type
            or getattr(stream_ctx.unified_task_type, "value", "default")
            or "default"
        )
        tool_budget = (
            stream_ctx.complexity_tool_budget
            if stream_ctx.complexity_tool_budget is not None
            else stream_ctx.tool_budget
        )
        if tool_budget is None:
            tool_budget = getattr(orch, "tool_budget", 10) or 10

        routing_context: Dict[str, Any] = {
            "iteration_budget": stream_ctx.max_total_iterations,
            "tool_budget": int(tool_budget),
            "available_team_formations": self._default_team_formations(),
        }

        provider_hints = await self._get_stream_topology_provider_hints(
            task_type=task_type,
            context=routing_context,
        )
        if provider_hints:
            routing_context.update(provider_hints)
            provider_candidates: List[str] = []
            primary_provider = provider_hints.get("provider_hint")
            if isinstance(primary_provider, str) and primary_provider:
                provider_candidates.append(primary_provider)
            for fallback in provider_hints.get("fallback_chain", []):
                if isinstance(fallback, str) and fallback:
                    provider_candidates.append(fallback)
            if provider_candidates:
                routing_context["provider_candidates"] = list(dict.fromkeys(provider_candidates))
        runtime_intelligence = self._get_runtime_state_dict(orch).get("_runtime_intelligence")
        structured_routing_policy = None
        if runtime_intelligence is not None and hasattr(
            runtime_intelligence, "get_structured_routing_policy"
        ):
            learned_scope_context = dict(routing_context)
            learned_scope_context.setdefault("task_type", task_type)
            try:
                structured_routing_policy = runtime_intelligence.get_structured_routing_policy(
                    query=user_message,
                    scope_context=learned_scope_context,
                )
            except Exception as exc:
                logger.debug("Streaming structured routing policy unavailable: %s", exc)
            else:
                if structured_routing_policy is not None:
                    if hasattr(structured_routing_policy, "to_dict"):
                        serialized_policy = structured_routing_policy.to_dict()
                        if isinstance(serialized_policy, dict):
                            stream_ctx.structured_routing_policy = serialized_policy
                    learned_topology_context = structured_routing_policy.selector_context()
                    if isinstance(learned_topology_context, dict) and learned_topology_context:
                        routing_context.update(learned_topology_context)
        elif runtime_intelligence is not None and hasattr(
            runtime_intelligence, "get_topology_routing_context"
        ):
            learned_scope_context = dict(routing_context)
            learned_scope_context.setdefault("task_type", task_type)
            try:
                learned_topology_context = runtime_intelligence.get_topology_routing_context(
                    query=user_message, scope_context=learned_scope_context
                )
            except Exception as exc:
                logger.debug("Streaming topology feedback hints unavailable: %s", exc)
            else:
                if learned_topology_context:
                    routing_context.update(learned_topology_context)

        paradigm_router = getattr(self, "_paradigm_router", None)
        if paradigm_router is None:
            paradigm_router = get_paradigm_router()
            self._paradigm_router = paradigm_router

        topology_selector = getattr(self, "_topology_selector", None)
        if topology_selector is None:
            topology_selector = TopologySelector()
            self._topology_selector = topology_selector

        topology_grounder = getattr(self, "_topology_grounder", None)
        if topology_grounder is None:
            topology_grounder = TopologyGrounder()
            self._topology_grounder = topology_grounder

        topology_input = paradigm_router.build_topology_input(
            task_type=task_type,
            query=user_message,
            history_length=len(getattr(orch, "messages", []) or []),
            query_complexity=self._stream_query_complexity(stream_ctx),
            tool_budget=int(tool_budget),
            context=routing_context,
        )
        topology_decision = topology_selector.select(topology_input)
        topology_plan = topology_grounder.ground(topology_decision)
        prepared_runtime = prepare_topology_runtime_contract(
            topology_plan,
            orchestrator=orch,
            task_type=task_type,
            complexity=str(
                getattr(
                    getattr(task_classification, "complexity", None),
                    "value",
                    getattr(task_classification, "complexity", None),
                )
                or "medium"
            ),
        )
        topology_overrides = dict(prepared_runtime.runtime_context_overrides)

        stream_ctx.topology_input = topology_input.to_dict()
        stream_ctx.topology_decision = topology_decision.to_dict()
        stream_ctx.topology_plan = topology_plan.to_dict()
        stream_ctx.topology_preparation = prepared_runtime.to_result(
            prepared=prepared_runtime.team_plan is not None
        )
        if prepared_runtime.team_plan is not None:
            stream_ctx.topology_plan["team_name"] = prepared_runtime.team_plan.team_name
            stream_ctx.topology_plan["team_display_name"] = prepared_runtime.team_plan.display_name
            stream_ctx.topology_plan["member_count"] = prepared_runtime.team_plan.member_count
        stream_ctx.runtime_context_overrides = dict(topology_overrides)
        stream_ctx.provider_kwargs = self._stream_provider_call_overrides(topology_overrides)

        if topology_plan.tool_budget is not None:
            adjusted_tool_budget = max(0, int(topology_plan.tool_budget))
            stream_ctx.tool_budget = adjusted_tool_budget
            stream_ctx.complexity_tool_budget = adjusted_tool_budget
        if topology_plan.iteration_budget is not None:
            adjusted_iteration_budget = max(1, int(topology_plan.iteration_budget))
            stream_ctx.max_total_iterations = adjusted_iteration_budget
            stream_ctx.max_exploration_iterations = min(
                max(1, stream_ctx.max_exploration_iterations),
                adjusted_iteration_budget,
            )
        if topology_decision.action == TopologyAction.DIRECT_RESPONSE:
            stream_ctx.is_qa_task = True

        stream_ctx.runtime_override_snapshot = self._apply_stream_runtime_overrides(
            topology_overrides
        )

        topology_event = build_topology_telemetry_event(
            topology_input,
            topology_decision,
            grounded_plan=topology_plan,
            outcome={"status": "planned", "runtime": "streaming"},
        )
        stream_ctx.topology_events.append(topology_event.to_dict())
        await emit_topology_telemetry_event(topology_event)

        logger.info(
            "[StreamingTopology] action=%s topology=%s provider=%s formation=%s",
            topology_decision.action.value,
            topology_decision.topology.value,
            topology_plan.provider,
            topology_plan.formation,
        )

    def _is_stream_topology_enabled(self) -> bool:
        """Return whether streaming topology routing is enabled."""
        explicit = getattr(self, "_topology_enabled", None)
        if explicit is not None:
            return bool(explicit)
        settings = getattr(self._orchestrator, "settings", None)
        if settings is not None and hasattr(settings, "enable_topology_routing"):
            return bool(settings.enable_topology_routing)
        return True

    async def _get_stream_topology_provider_hints(
        self,
        task_type: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Fetch provider-routing hints when the active provider exposes them."""
        orch = self._orchestrator
        provider = getattr(orch, "provider", None)
        model_hint = getattr(orch, "model", None)
        preferred_providers = context.get("preferred_providers")

        hint_sources = [
            getattr(provider, "get_topology_provider_hints", None),
            getattr(getattr(provider, "engine", None), "get_topology_provider_hints", None),
        ]
        for get_hints in hint_sources:
            if not callable(get_hints):
                continue
            try:
                hints = await get_hints(
                    task_type=task_type,
                    model_hint=model_hint,
                    preferred_providers=preferred_providers,
                )
            except Exception as exc:
                logger.debug("Streaming topology provider hints unavailable: %s", exc)
                continue
            if isinstance(hints, dict):
                return hints
        return {}

    @staticmethod
    def _stream_query_complexity(stream_ctx: "StreamingChatContext") -> Optional[float]:
        """Convert streaming task complexity into a stable numeric score."""
        task_classification = stream_ctx.task_classification
        complexity = getattr(task_classification, "complexity", None)
        if complexity is not None:
            complexity_value = getattr(complexity, "value", str(complexity)).lower()
            if (
                "analysis" in complexity_value
                or "complex" in complexity_value
                or "high" in complexity_value
            ):
                return 0.8
            if "medium" in complexity_value or "moderate" in complexity_value:
                return 0.5
            return 0.2

        if stream_ctx.is_analysis_task or stream_ctx.is_complex_task:
            return 0.8
        if stream_ctx.is_action_task:
            return 0.5
        return 0.2

    @staticmethod
    def _default_team_formations() -> List[str]:
        """Return canonical team formation hints for topology grounding."""
        from victor.teams.types import TeamFormation

        return [formation.value for formation in TeamFormation]

    @staticmethod
    def _stream_provider_call_overrides(overrides: Dict[str, Any]) -> Dict[str, Any]:
        """Extract provider-facing runtime override hints."""
        provider_keys = (
            "provider_hint",
            "execution_mode",
            "escalation_target",
            "topology_action",
            "topology_kind",
            "topology_metadata",
        )
        return {
            key: value
            for key, value in overrides.items()
            if key in provider_keys and value is not None
        }

    def _apply_stream_runtime_overrides(
        self,
        overrides: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Apply temporary runtime overrides for the streaming turn."""
        if not overrides:
            return None

        orch = self._orchestrator
        state_dict = self._get_runtime_state_dict(orch)
        snapshot: Dict[str, Any] = {
            "orchestrator_runtime_context": state_dict.get(
                "_runtime_tool_context_overrides",
                _MISSING,
            ),
        }

        merged_context: Dict[str, Any] = {}
        previous_runtime_context = snapshot["orchestrator_runtime_context"]
        if isinstance(previous_runtime_context, dict):
            merged_context.update(previous_runtime_context)
        merged_context.update(overrides)
        orch._runtime_tool_context_overrides = merged_context

        tool_budget = self._coerce_stream_int_override(overrides.get("tool_budget"))
        if tool_budget is not None:
            if hasattr(orch, "tool_budget"):
                snapshot["orchestrator_tool_budget"] = getattr(orch, "tool_budget", _MISSING)
                try:
                    orch.tool_budget = max(0, tool_budget)
                except Exception:
                    pass

            task_coordinator = getattr(orch, "task_coordinator", None)
            if task_coordinator is not None and hasattr(task_coordinator, "tool_budget"):
                snapshot["task_coordinator_tool_budget"] = getattr(
                    task_coordinator,
                    "tool_budget",
                    _MISSING,
                )
                try:
                    task_coordinator.tool_budget = max(0, tool_budget)
                except Exception:
                    pass

            tool_service = getattr(orch, "_tool_service", None)
            if tool_service is not None and hasattr(tool_service, "get_tool_budget"):
                snapshot["tool_service_budget"] = getattr(
                    tool_service,
                    "budget",
                    (
                        tool_service.get_budget_info().get("max")
                        if hasattr(tool_service, "get_budget_info")
                        else tool_service.get_tool_budget()
                    ),
                )
                if hasattr(tool_service, "set_tool_budget"):
                    try:
                        tool_service.set_tool_budget(max(0, tool_budget))
                    except Exception:
                        pass

            tool_pipeline = getattr(orch, "_tool_pipeline", None)
            pipeline_config = getattr(tool_pipeline, "config", None)
            if pipeline_config is not None and hasattr(pipeline_config, "tool_budget"):
                snapshot["pipeline_tool_budget"] = getattr(
                    pipeline_config,
                    "tool_budget",
                    _MISSING,
                )
                try:
                    pipeline_config.tool_budget = max(0, tool_budget)
                except Exception:
                    pass

        return snapshot

    def _restore_stream_runtime_overrides(
        self,
        snapshot: Optional[Dict[str, Any]],
    ) -> None:
        """Restore runtime state after one streaming turn completes."""
        if not snapshot:
            return

        orch = self._orchestrator
        state_dict = self._get_runtime_state_dict(orch)
        previous_runtime_context = snapshot.get("orchestrator_runtime_context", _MISSING)
        if previous_runtime_context is _MISSING:
            if "_runtime_tool_context_overrides" in state_dict:
                delattr(orch, "_runtime_tool_context_overrides")
        else:
            orch._runtime_tool_context_overrides = previous_runtime_context

        previous_orchestrator_budget = snapshot.get("orchestrator_tool_budget", _MISSING)
        if previous_orchestrator_budget is not _MISSING and hasattr(orch, "tool_budget"):
            try:
                orch.tool_budget = previous_orchestrator_budget
            except Exception:
                pass

        task_coordinator = getattr(orch, "task_coordinator", None)
        previous_task_coordinator_budget = snapshot.get("task_coordinator_tool_budget", _MISSING)
        if (
            previous_task_coordinator_budget is not _MISSING
            and task_coordinator is not None
            and hasattr(task_coordinator, "tool_budget")
        ):
            try:
                task_coordinator.tool_budget = previous_task_coordinator_budget
            except Exception:
                pass

        tool_service = getattr(orch, "_tool_service", None)
        previous_service_budget = snapshot.get("tool_service_budget", _MISSING)
        if (
            previous_service_budget is not _MISSING
            and tool_service is not None
            and hasattr(tool_service, "set_tool_budget")
        ):
            try:
                tool_service.set_tool_budget(previous_service_budget)
            except Exception:
                pass

        tool_pipeline = getattr(orch, "_tool_pipeline", None)
        pipeline_config = getattr(tool_pipeline, "config", None)
        previous_pipeline_budget = snapshot.get("pipeline_tool_budget", _MISSING)
        if (
            previous_pipeline_budget is not _MISSING
            and pipeline_config is not None
            and hasattr(pipeline_config, "tool_budget")
        ):
            try:
                pipeline_config.tool_budget = previous_pipeline_budget
            except Exception:
                pass

    @staticmethod
    def _coerce_stream_int_override(value: Any) -> Optional[int]:
        """Convert runtime override values into integers when possible."""
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _prepare_task(
        self, user_message: str, unified_task_type: TrackerTaskType
    ) -> tuple[Any, int]:
        """Prepare task-specific guidance and budget adjustments."""
        orch = self._orchestrator

        if orch.task_coordinator._reminder_manager is None:
            orch.task_coordinator.set_reminder_manager(orch.reminder_manager)

        return orch.task_coordinator.prepare_task(
            user_message, unified_task_type, orch.conversation_controller
        )

    async def _run_iteration_pre_checks(
        self,
        stream_ctx: "StreamingChatContext",
        user_message: str,
    ) -> AsyncIterator[StreamChunk]:
        """Run pre-iteration checks: cancellation, compaction, time limit."""
        orch = self._orchestrator

        if orch._check_cancellation():
            logger.info("Stream cancelled by user request")
            orch._is_streaming = False
            orch._record_runtime_intelligence_outcome(
                success=False,
                quality_score=stream_ctx.last_quality_score,
                user_satisfied=False,
                completed=False,
            )
            yield StreamChunk(
                content="\n\n[Cancelled by user]\n",
                is_final=True,
            )
            return

        lifecycle_handled = await self._run_lifecycle_pre_iteration_compaction(
            stream_ctx,
            user_message,
        )
        context_service_handled = False
        if not lifecycle_handled:
            context_service_handled = await self._run_context_service_pre_iteration_compaction(
                stream_ctx,
            )

        if not lifecycle_handled and not context_service_handled and orch._context_compactor:
            compaction_action = orch._context_compactor.check_and_compact(
                current_query=user_message,
                force=False,
                tool_call_count=orch.tool_calls_used,
                task_complexity=TaskComplexity.COMPLEX.value,
            )
            if compaction_action.action_taken:
                logger.info(
                    f"Compacted context: {compaction_action.messages_removed} messages removed, "
                    f"{compaction_action.tokens_freed} tokens freed"
                )
                compaction_summary = ""
                if hasattr(orch, "conversation_controller") and orch.conversation_controller:
                    summaries = orch.conversation_controller.get_compaction_summaries()
                    if summaries:
                        compaction_summary = summaries[-1]
                if hasattr(stream_ctx, "record_compaction_event"):
                    stream_ctx.record_compaction_event(
                        summary=compaction_summary,
                        messages_removed=compaction_action.messages_removed,
                        strategy=getattr(orch.settings, "context_compaction_strategy", "tiered"),
                        reason="pre_iteration",
                    )
                else:
                    stream_ctx.compaction_occurred = True
                    stream_ctx.last_compaction_turn = stream_ctx.total_iterations
                    stream_ctx.compaction_message_removed_count = compaction_action.messages_removed
                    stream_ctx.compaction_summary = compaction_summary
                logger.info(
                    f"Post-compaction continuation enabled at turn {stream_ctx.total_iterations}"
                )

        time_limit = getattr(orch.settings, "stream_idle_timeout_seconds", 300)
        if stream_ctx.is_over_time_limit(time_limit):
            logger.warning(f"Stream time limit exceeded: {stream_ctx.elapsed_time():.1f}s")
            yield StreamChunk(
                content=f"\n\n[Session exceeded {time_limit}s idle timeout - providing summary]\n",
                is_final=False,
            )
            stream_ctx.force_completion = True

        stream_ctx.increment_iteration()

        if stream_ctx.pending_grounding_feedback:
            logger.info("Injecting pending grounding feedback as system message")
            from victor.agent.conversation.types import MessageSource

            orch.add_message(
                "user",
                f"[GROUNDING-FEEDBACK: {stream_ctx.pending_grounding_feedback}]",
                metadata=build_internal_history_metadata(
                    "grounding_feedback", source=MessageSource.AGENT_GROUNDING
                ),
            )
            stream_ctx.pending_grounding_feedback = ""

    async def _run_context_service_pre_iteration_compaction(
        self,
        stream_ctx: "StreamingChatContext",
    ) -> bool:
        """Run context-service compaction before legacy compactor fallback."""
        orch = self._orchestrator
        context_service = getattr(orch, "_context_service", None)
        if context_service is None:
            return False

        strategy = str(
            getattr(getattr(orch, "settings", None), "context_compaction_strategy", "tiered")
            or "tiered"
        )
        result = await compact_context_if_recommended(
            context_service,
            strategy=strategy,
            min_messages=6,
        )
        if not result.handled:
            return False
        if result.messages_removed <= 0:
            return True

        logger.info(
            "ContextService compacted root context: %s messages removed",
            result.messages_removed,
        )
        if hasattr(stream_ctx, "record_compaction_event"):
            stream_ctx.record_compaction_event(
                summary=f"Compacted {result.messages_removed} messages via ContextService",
                messages_removed=result.messages_removed,
                strategy=strategy,
                reason="pre_iteration",
                policy_reason="context_service",
            )
        else:
            stream_ctx.compaction_occurred = True
            stream_ctx.last_compaction_turn = stream_ctx.total_iterations
            stream_ctx.compaction_message_removed_count = result.messages_removed
            stream_ctx.compaction_summary = (
                f"Compacted {result.messages_removed} messages via ContextService"
            )
        return True

    async def _run_lifecycle_pre_iteration_compaction(
        self,
        stream_ctx: "StreamingChatContext",
        user_message: str,
    ) -> bool:
        """Run service-owned root context compaction before legacy compactor fallback."""
        orch = self._orchestrator
        lifecycle = getattr(orch, "_context_lifecycle_service", None)
        if lifecycle is None:
            return False
        after_agent_turn = getattr(lifecycle, "after_agent_turn", None)
        if not callable(after_agent_turn):
            return False

        runtime_context = self._root_runtime_context(orch)
        result = await after_agent_turn(
            runtime_context,
            messages=self._root_runtime_messages(orch),
            min_messages=6,
        )
        if not isinstance(result, dict) or not result.get("compacted"):
            return True

        removed = int(result.get("messages_removed", 0) or 0)
        summary = str(
            result.get("summary")
            or f"Compacted {removed} messages for {runtime_context.display_name}"
        )
        strategy = str(
            result.get("strategy")
            or getattr(orch.settings, "context_compaction_strategy", "tiered")
        )
        logger.info(
            "Lifecycle compacted root context: %s messages removed, %s tokens freed",
            removed,
            int(result.get("tokens_freed", 0) or 0),
        )
        if hasattr(stream_ctx, "record_compaction_event"):
            stream_ctx.record_compaction_event(
                summary=summary,
                messages_removed=removed,
                strategy=strategy,
                reason="pre_iteration",
                policy_reason="context_lifecycle",
            )
        else:
            stream_ctx.compaction_occurred = True
            stream_ctx.last_compaction_turn = stream_ctx.total_iterations
            stream_ctx.compaction_message_removed_count = removed
            stream_ctx.compaction_summary = summary
        return True

    @staticmethod
    def _root_runtime_context(orch: Any) -> AgentRuntimeContext:
        existing = getattr(orch, "_agent_runtime_context", None) or getattr(
            orch,
            "agent_runtime_context",
            None,
        )
        if isinstance(existing, AgentRuntimeContext):
            return existing
        session_id = (
            getattr(orch, "active_session_id", None)
            or getattr(orch, "session_id", None)
            or getattr(orch, "_memory_session_id", None)
            or "session_root"
        )
        return AgentRuntimeContext(
            agent_id=str(getattr(orch, "agent_id", None) or "root_agent"),
            display_name=str(getattr(orch, "display_name", None) or "Root Agent"),
            role=str(getattr(orch, "role", None) or "manager"),
            session_id=str(session_id),
        )

    @staticmethod
    def _root_runtime_messages(orch: Any) -> List[Any]:
        get_messages = getattr(orch, "get_messages", None)
        if callable(get_messages):
            try:
                return list(get_messages() or [])
            except Exception as exc:
                logger.debug("Failed to collect root messages for lifecycle: %s", exc)
        controller = getattr(orch, "conversation_controller", None) or getattr(
            orch,
            "_conversation_controller",
            None,
        )
        return list(getattr(controller, "messages", None) or [])

    async def _stream_provider_response(
        self,
        tools: Any,
        provider_kwargs: Dict[str, Any],
        stream_ctx: "StreamingChatContext",
    ) -> tuple[str, Any, float, bool]:
        """Stream response from provider with rate limit retry."""
        return await self._stream_with_rate_limit_retry(tools, provider_kwargs, stream_ctx)

    def _get_rate_limit_wait_time(self, exc: Exception, attempt: int) -> float:
        """Get wait time for rate limit retry."""
        orch = self._orchestrator
        base_wait = orch._provider_service.get_rate_limit_wait_time(exc)
        backoff_multiplier = 2**attempt
        wait_time = base_wait * backoff_multiplier
        return min(wait_time, 300.0)

    async def _stream_with_rate_limit_retry(
        self,
        tools: Any,
        provider_kwargs: Dict[str, Any],
        stream_ctx: "StreamingChatContext",
        max_retries: int = 3,
    ) -> tuple[str, Any, float, bool]:
        """Stream provider response with automatic rate limit retry."""
        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                return await self._stream_provider_response_inner(
                    tools, provider_kwargs, stream_ctx
                )
            except ProviderRateLimitError as exc:
                last_exception = exc
                if attempt < max_retries:
                    wait_time = self._get_rate_limit_wait_time(exc, attempt)
                    endpoint_info = (
                        f"{self.provider_name}:{self.model}"
                        if hasattr(self, "provider_name")
                        else "API"
                    )
                    logger.warning(
                        f"[yellow]⚠ Rate limit hit for {endpoint_info}[/] "
                        f"(attempt {attempt + 1}/{max_retries + 1}). "
                        f"Waiting {wait_time:.0f}s before retry..."
                    )
                    logger.debug(f"Rate limit detail: {str(exc)[:300]}")

                    if attempt == 0:
                        logger.info(
                            "[dim]Tip: To avoid rate limits, consider:[/]\n"
                            "[dim]  • Using an API key instead of free tier[/]\n"
                            "[dim]  • Adding a small delay between requests[/]\n"
                            "[dim]  • Reducing request frequency[/]"
                        )

                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Rate limit persisted after {max_retries + 1} attempts")
            except ProviderConnectionError as exc:
                # A mid-stream disconnect (broken SSE) cannot be resumed at the
                # HTTP level, but the turn can be safely re-issued: all prior
                # iterations' work is already in conversation history, so a
                # bounded retry here prevents a single transient drop from
                # aborting the whole task and discarding accumulated progress.
                last_exception = exc
                if attempt < max_retries:
                    wait_time = min(2.0**attempt, 8.0)
                    logger.warning(
                        f"Provider connection dropped mid-stream "
                        f"(attempt {attempt + 1}/{max_retries + 1}): "
                        f"{str(exc)[:200]}. Re-issuing turn in {wait_time:.0f}s..."
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(
                        f"Provider connection failed after {max_retries + 1} attempts; "
                        "giving up on this turn."
                    )
            except Exception as exc:
                exc_str = str(exc).lower()
                if "rate_limit" in exc_str or "429" in exc_str or "rate limit" in exc_str:
                    last_exception = exc
                    if attempt < max_retries:
                        wait_time = self._get_rate_limit_wait_time(exc, attempt)
                        logger.warning(
                            f"Rate limit hit (attempt {attempt + 1}/{max_retries + 1}). "
                            f"Waiting {wait_time:.1f}s before retry..."
                        )
                        logger.debug(f"Rate limit detail: {type(exc).__name__}: {str(exc)[:200]}")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"Rate limit persisted after {max_retries + 1} attempts")
                else:
                    raise

        if last_exception:
            raise last_exception
        raise RuntimeError("Rate limit retry exhausted without exception")

    async def _stream_provider_response_inner(
        self,
        tools: Any,
        provider_kwargs: Dict[str, Any],
        stream_ctx: "StreamingChatContext",
    ) -> tuple[str, Any, float, bool]:
        """Inner implementation of stream_provider_response without retry logic."""
        orch = self._orchestrator

        full_content = ""
        # Intra-turn repetition guard (Tool Reliability P5): stops a degenerate
        # generation (the model looping one sentence/paragraph) instead of
        # burning tokens to max_tokens. Between-turn detection cannot see this.
        repetition_detector = None
        if getattr(orch.settings, "stream_repetition_enabled", True):
            from victor.agent.streaming.repetition_guard import IntraTurnRepetitionDetector

            repetition_detector = IntraTurnRepetitionDetector(
                window_chars=getattr(orch.settings, "stream_repetition_window_chars", 4000),
                min_segment_chars=getattr(orch.settings, "stream_repetition_min_segment_chars", 24),
                max_repeats=getattr(orch.settings, "stream_repetition_max_repeats", 6),
            )
        tool_calls = None
        garbage_detected = False
        consecutive_garbage_chunks = 0
        max_garbage_chunks = 3
        estimated_content_tokens: float = 0

        assembled = orch.get_assembled_messages(
            current_query=stream_ctx.user_message if stream_ctx else None,
            goals=(stream_ctx.goals if stream_ctx else None),
            selected_tools=tools,
            planned_tools=(stream_ctx.planned_tools if stream_ctx else None),
        )
        # Resolve per-turn temperature through the SAME unified seam as the buffered path (ADR-013),
        # so streaming no longer ignores task-hint/profile-per-task temperature. Falls back to
        # orch.temperature when no resolver is available.
        from victor.agent.services.temperature_resolution import (
            normalize_task_type,
            resolve_effective_temperature,
        )

        stream_task_type = (
            normalize_task_type(
                getattr(stream_ctx, "unified_task_type", None)
                or getattr(stream_ctx, "coarse_task_type", None)
            )
            if stream_ctx
            else None
        )
        stream_temperature = resolve_effective_temperature(
            orch,
            task_type=stream_task_type,
            model=orch.model,
            base_temperature=orch.temperature,
        )
        provider_stream = orch.provider.stream(
            messages=assembled,
            model=orch.model,
            temperature=stream_temperature,
            max_tokens=orch.max_tokens,
            tools=tools,
            **provider_kwargs,
        )
        heartbeat_interval = max(
            1.0,
            float(getattr(orch.settings, "stream_provider_wait_heartbeat_seconds", 15.0)),
        )
        stall_timeout = max(
            heartbeat_interval,
            float(
                getattr(
                    orch.settings,
                    "stream_provider_stall_timeout_seconds",
                    getattr(orch.settings, "stream_idle_timeout_seconds", 300.0),
                )
            ),
        )
        loop_stall_grace = max(
            0.0,
            float(
                getattr(
                    orch.settings,
                    "stream_provider_loop_stall_grace_seconds",
                    max(5.0, heartbeat_interval * 2.0),
                )
            ),
        )
        waiting_since = time.monotonic()
        first_chunk_received = False
        # Reasoning-only stream diagnostics: a thinking model (kimi-k3, DeepSeek-R,
        # o-series) can burn the entire completion budget on reasoning deltas and
        # finish with zero content — which otherwise presents as an opaque "empty
        # response" that identical-parameter retries can never fix.
        reasoning_chars = 0
        refusal_chars = 0
        last_stop_reason: Optional[str] = None

        # Producer/consumer split for deterministic, on-task stream cleanup.
        #
        # The provider's httpx SSE generator must be entered, iterated, AND closed in a
        # single asyncio task — otherwise httpcore/anyio raises "Attempted to exit cancel
        # scope in a different task" / "async generator ignored GeneratorExit". The previous
        # code spawned a fresh `create_task(provider_iterator.__anext__())` per chunk, so the
        # generator's anyio cancel scope was entered in one child task and resumed/exited in
        # later ones. Here a single producer task owns the whole generator lifecycle (via
        # `aclosing`, so every exit path — normal end, tool-call break, stall, cancellation —
        # closes it in the producer's own task). The consumer applies the heartbeat/stall
        # timeouts against a bounded queue, which never cancels the in-flight read.
        _DONE = object()
        chunk_queue: asyncio.Queue = asyncio.Queue(maxsize=1)

        async def _drain_provider_stream() -> None:
            try:
                async with contextlib.aclosing(provider_stream):
                    async for produced in provider_stream:
                        await chunk_queue.put(("chunk", produced))
                await chunk_queue.put(("done", _DONE))
            except asyncio.CancelledError:
                raise  # cleanup happens via aclosing.__aexit__ in THIS task
            except Exception as exc:  # forward provider errors to the consumer in order
                await chunk_queue.put(("error", exc))

        producer_task = asyncio.create_task(_drain_provider_stream())
        # Persistent queue-get task, re-awaited across heartbeats under asyncio.shield so a
        # heartbeat timeout never cancels (and drops) an in-flight dequeue. Using a Task
        # (not a bare coroutine) also keeps it driven independently of the wait_for wrapper.
        pending_get: Optional[asyncio.Future] = None
        try:
            while True:
                if pending_get is None:
                    pending_get = asyncio.ensure_future(chunk_queue.get())
                try:
                    wait_started_at = time.monotonic()
                    kind, payload = await asyncio.wait_for(
                        asyncio.shield(pending_get), timeout=heartbeat_interval
                    )
                    pending_get = None
                except asyncio.TimeoutError:
                    now = time.monotonic()
                    wait_overrun_seconds = max(0.0, now - wait_started_at - heartbeat_interval)
                    if wait_overrun_seconds > loop_stall_grace:
                        # The heartbeat wait overran by more than the grace window: the local
                        # asyncio event loop was blocked (host-side GC/CPU), not the provider.
                        # Attribute it to the host and reset the timer so a responsive provider
                        # (including cloud) is not penalized for local blocking.
                        self._record_provider_status_event(
                            stream_ctx,
                            "local_runtime_stall",
                            waited_seconds=round(now - waiting_since, 3),
                            overrun_seconds=round(wait_overrun_seconds, 3),
                            model=getattr(orch, "model", None),
                        )
                        logger.warning(
                            "[provider-stream] Local event loop blocked for %.1fs "
                            "(host-side GC/CPU, not the provider); not charging it "
                            "against the provider stall timer",
                            wait_overrun_seconds,
                        )
                        waiting_since = now
                        continue

                    waited_seconds = now - waiting_since
                    event_kind = (
                        "provider_waiting" if not first_chunk_received else "still_generating"
                    )
                    self._record_provider_status_event(
                        stream_ctx,
                        event_kind,
                        waited_seconds=round(waited_seconds, 3),
                        model=getattr(orch, "model", None),
                    )
                    logger.info(
                        "[provider-stream] %s after %.1fs (model=%s)",
                        event_kind,
                        waited_seconds,
                        getattr(orch, "model", "unknown"),
                    )
                    if waited_seconds >= stall_timeout:
                        logger.error(
                            "[provider-stream] Stall timeout after %.1fs without provider chunk",
                            waited_seconds,
                        )
                        self._record_provider_status_event(
                            stream_ctx,
                            "provider_stall_timeout",
                            waited_seconds=round(waited_seconds, 3),
                            model=getattr(orch, "model", None),
                        )
                        # The finally below cancels the producer; its aclosing finalizes the
                        # httpx stream in the producer task.
                        raise ProviderTimeoutError(
                            f"Provider stream stalled for {waited_seconds:.1f}s without chunks"
                        )
                    continue

                if kind == "error":
                    self._log_client_error_payload_shape(payload, assembled)
                    raise payload
                if kind == "done":
                    self._record_provider_status_event(
                        stream_ctx,
                        "completion_detected",
                        waited_seconds=round(time.monotonic() - waiting_since, 3),
                        model=getattr(orch, "model", None),
                        reason="stream_end",
                        content_length=len(full_content),
                        tool_call_count=len(tool_calls) if tool_calls else 0,
                    )
                    break

                chunk = payload

                if hasattr(stream_ctx, "reset_activity_timer"):
                    stream_ctx.reset_activity_timer()

                if not first_chunk_received:
                    first_chunk_received = True
                    self._record_provider_status_event(
                        stream_ctx,
                        "first_token_received",
                        waited_seconds=round(time.monotonic() - waiting_since, 3),
                        model=getattr(orch, "model", None),
                        has_content=bool(getattr(chunk, "content", "")),
                        has_tool_calls=bool(getattr(chunk, "tool_calls", None)),
                    )

                waiting_since = time.monotonic()
                # Accumulate token usage from the RAW chunk BEFORE garbage-filtering.
                # Providers routinely attach the turn's usage to the final, empty-content
                # chunk (Ollama's `done` chunk, OpenAI-compat's terminal usage chunk). That
                # chunk has no content, so `_handle_stream_chunk` drops it as garbage — and
                # with it the turn's real token accounting, leaving the api_*_tokens at 0.
                # Reading usage here makes the dominant cost term observable regardless of
                # whether the chunk survives the filter.
                chunk_stop_reason = getattr(chunk, "stop_reason", None)
                if chunk_stop_reason:
                    last_stop_reason = str(chunk_stop_reason)
                chunk_meta = getattr(chunk, "metadata", None)
                if isinstance(chunk_meta, dict):
                    reasoning_delta = chunk_meta.get("reasoning_content")
                    if reasoning_delta:
                        reasoning_chars += len(str(reasoning_delta))
                    refusal_delta = chunk_meta.get("refusal")
                    if refusal_delta:
                        refusal_chars += len(str(refusal_delta))

                raw_usage = getattr(chunk, "usage", None)
                if raw_usage:
                    for key in stream_ctx.cumulative_usage:
                        stream_ctx.cumulative_usage[key] += raw_usage.get(key, 0)
                    logger.debug(
                        f"Chunk usage: in={raw_usage.get('prompt_tokens', 0)} "
                        f"out={raw_usage.get('completion_tokens', 0)} "
                        f"cache_read={raw_usage.get('cache_read_input_tokens', 0)}"
                    )

                # Sandhi attaches transport diagnostics (retry attempts, usage
                # completeness, upstream request id) to the same terminal chunk
                # that carries usage. Fold them into the turn so finalize can
                # surface them — previously recorded by the transport and read
                # by nothing.
                chunk_meta = getattr(chunk, "metadata", None)
                if isinstance(chunk_meta, dict):
                    sandhi_diag = chunk_meta.get("sandhi_usage")
                    if isinstance(sandhi_diag, dict):
                        if hasattr(stream_ctx, "record_provider_diagnostics"):
                            stream_ctx.record_provider_diagnostics(sandhi_diag)
                        # Wire-truth latency (W3b): prefer sandhi's boundary
                        # measurement over client wall-clock in stream metrics.
                        stream_ctx.stream_metrics.record_wire_latency(sandhi_diag)

                chunk, consecutive_garbage_chunks, garbage_detected = self._handle_stream_chunk(
                    chunk,
                    consecutive_garbage_chunks,
                    max_garbage_chunks,
                    garbage_detected,
                )
                if chunk is None:
                    continue

                full_content += chunk.content
                stream_ctx.stream_metrics.total_chunks += 1
                if chunk.content:
                    orch._metrics_collector.record_first_token()
                    # Rough chars/4 ESTIMATE for the live "~N tokens (est.)" readout and
                    # confidence monitoring only — NOT billing. The authoritative counts are
                    # the provider's usage accumulated into stream_ctx.cumulative_usage above
                    # (surfaced as api_*_tokens on the task report / cost footer).
                    estimated_content_tokens += len(chunk.content) / 4
                    stream_ctx.stream_metrics.total_content_length += len(chunk.content)

                if repetition_detector is not None and chunk.content:
                    repeated = repetition_detector.feed(chunk.content)
                    if repeated is not None:
                        # Degenerate loop: truncate at the second occurrence (keep
                        # one instance) and break — the shared `finally` tears the
                        # provider stream down exactly like the tool-call break.
                        full_content = full_content[
                            : repetition_detector.truncation_point(full_content)
                        ]
                        # A loop with no tool call is UNUSABLE output, not an
                        # answer. Without this flag the truncated narration was
                        # emitted as the assistant turn and the loop simply ran
                        # again, so the model could spend turn after turn
                        # narrating ("I'll call the tool. Now. Executing.")
                        # while the tool it meant to call never ran. Reuse the
                        # garbage-detection contract: the handler turns this into
                        # force_completion, which ends the spin and makes the
                        # model deliver a final answer from what it already has.
                        if not tool_calls:
                            garbage_detected = True
                        logger.warning(
                            "Intra-turn repetition detected — stopping generation " "(segment: %r)",
                            repeated[:120],
                        )
                        self._record_provider_status_event(
                            stream_ctx,
                            "completion_detected",
                            waited_seconds=0.0,
                            model=getattr(orch, "model", None),
                            reason="content_repetition_intraturn",
                            content_length=len(full_content),
                        )
                        break

                if chunk.tool_calls:
                    logger.debug(f"Received tool_calls in chunk: {chunk.tool_calls}")
                    tool_calls = chunk.tool_calls
                    stream_ctx.stream_metrics.tool_calls_count += len(chunk.tool_calls)

                if tool_calls:
                    # Tool calls signal the end of this turn — the SSE stream is done.
                    # Break immediately; do NOT start a new stream call with the same
                    # messages (that would duplicate full_content).
                    self._record_provider_status_event(
                        stream_ctx,
                        "completion_detected",
                        waited_seconds=0.0,
                        model=getattr(orch, "model", None),
                        reason="tool_calls",
                        content_length=len(full_content),
                        tool_call_count=len(tool_calls),
                    )
                    logger.debug("Tool calls received, breaking stream loop")
                    break
        finally:
            # Deterministic, on-task cleanup. Cancelling the producer raises CancelledError
            # inside _drain_provider_stream, whose `aclosing` __aexit__ then closes the httpx
            # SSE generator in the producer's OWN task — never via GC and never across tasks.
            # Covers every exit: stream end, tool-call break, stall timeout, provider error,
            # and an outer cancellation (e.g. the UI's on_chat_end draining the turn).
            if pending_get is not None and not pending_get.done():
                pending_get.cancel()
            if not producer_task.done():
                producer_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await producer_task

        if garbage_detected and not tool_calls:
            logger.info("Setting force_completion due to garbage detection")

        if not tool_calls and not stream_ctx.force_completion:
            content_length = len(full_content.strip()) if full_content else 0

            if content_length == 0:
                stream_ctx.empty_stream_diagnostics = {
                    "reasoning_chars": reasoning_chars,
                    "refusal_chars": refusal_chars,
                    "stop_reason": last_stop_reason,
                }
                self._record_provider_status_event(
                    stream_ctx,
                    "empty_stream_completed",
                    model=getattr(orch, "model", None),
                    reasoning_chars=reasoning_chars,
                    stop_reason=last_stop_reason,
                )
                if reasoning_chars:
                    logger.warning(
                        "Stream completed without content or tool calls after %d reasoning "
                        "chars (stop_reason=%s) — reasoning likely consumed the completion "
                        "budget (max_tokens=%s)",
                        reasoning_chars,
                        last_stop_reason,
                        getattr(orch, "max_tokens", None),
                    )
                else:
                    logger.warning("Stream completed without content or tool calls")
            elif content_length < 50:
                self._record_provider_status_event(
                    stream_ctx,
                    "short_content_completed",
                    model=getattr(orch, "model", None),
                    content_length=content_length,
                )
                logger.warning(f"Stream completed with very short content ({content_length} chars)")
                logger.debug(f"Short content: {full_content}")

        cleanup_stats = consume_last_tool_message_cleanup_stats()
        if cleanup_stats.get("history_repaired"):
            self._record_provider_status_event(
                stream_ctx,
                "tool_history_repaired",
                model=getattr(orch, "model", None),
                stripped_assistant_tool_calls=int(
                    cleanup_stats.get("stripped_assistant_tool_calls", 0) or 0
                ),
                removed_orphaned_tool_responses=int(
                    cleanup_stats.get("removed_orphaned_tool_responses", 0) or 0
                ),
                repair_id=cleanup_stats.get("repair_id"),
                skipped_tool_messages_without_id=int(
                    cleanup_stats.get("skipped_tool_messages_without_id", 0) or 0
                ),
            )
            logger.warning(
                "Provider payload required tool-history repair: repair_id=%s stripped=%s removed=%s "
                "skipped_without_id=%s",
                cleanup_stats.get("repair_id"),
                cleanup_stats.get("stripped_assistant_tool_calls", 0),
                cleanup_stats.get("removed_orphaned_tool_responses", 0),
                cleanup_stats.get("skipped_tool_messages_without_id", 0),
            )

        logger.debug(
            "Stream completion summary - content: %s chars, tool_calls: %s",
            len(full_content) if full_content else 0,
            len(tool_calls) if tool_calls else 0,
        )

        stream_ctx.estimated_content_tokens = estimated_content_tokens
        return full_content, tool_calls, estimated_content_tokens, garbage_detected

    @staticmethod
    def _log_client_error_payload_shape(error: BaseException, messages: Any) -> None:
        """On a 4xx provider rejection, log the request's message shape.

        A 4xx on a chat request is almost always a payload-structure problem
        (broken tool_calls/tool pairing, invalid param) rather than a transient
        fault, and the surfaced error body is often opaque ("upstream status
        400"). Log a compact per-message summary — role, content length,
        tool-call ids, tool_call_id — so the defect is attributable from the
        session log alone (session modality-doc-review-fixes-b4e87728).
        """
        status = getattr(error, "status_code", None)
        if not isinstance(status, int) or not (400 <= status < 500) or status == 429:
            return
        try:
            parts = []
            for i, message in enumerate(messages or []):
                if not isinstance(message, dict):
                    message = getattr(message, "__dict__", None) or {}
                role = message.get("role", "?")
                content = message.get("content")
                entry = f"{i}:{role} len={len(content) if isinstance(content, str) else 0}"
                tool_calls = message.get("tool_calls") or []
                tc_ids = [tc.get("id", "?") for tc in tool_calls if isinstance(tc, dict)]
                if tc_ids:
                    entry += f" tool_calls={tc_ids}"
                if message.get("tool_call_id"):
                    entry += f" tool_call_id={message['tool_call_id']}"
                parts.append(entry)
            logger.warning(
                "[provider-4xx] status=%s payload shape (%d messages): %s",
                status,
                len(parts),
                " | ".join(parts),
            )
        except Exception:  # diagnostic must never mask the original error
            logger.debug("[provider-4xx] shape summary failed", exc_info=True)

    @staticmethod
    def _record_provider_status_event(
        stream_ctx: "StreamingChatContext",
        kind: str,
        **payload: Any,
    ) -> None:
        """Record structured provider-wait telemetry on the mutable stream context."""
        events = getattr(stream_ctx, "provider_status_events", None)
        if not isinstance(events, list):
            events = []
            stream_ctx.provider_status_events = events

        events.append(
            {
                "source": "streaming_provider",
                "kind": kind,
                "timestamp": time.time(),
                "iteration": getattr(stream_ctx, "total_iterations", 0),
                **payload,
            }
        )

    def _handle_stream_chunk(
        self,
        chunk: Any,
        consecutive_garbage_chunks: int,
        max_garbage_chunks: int,
        garbage_detected: bool,
    ) -> tuple[Any, int, bool]:
        """Handle garbage detection for a streaming chunk."""
        orch = self._orchestrator
        if chunk.content and orch.sanitizer.is_garbage_content(chunk.content):
            consecutive_garbage_chunks += 1
            if consecutive_garbage_chunks >= max_garbage_chunks:
                if not garbage_detected:
                    garbage_detected = True
                    logger.warning(
                        f"Garbage content detected after {len(chunk.content)} chars - stopping stream early"
                    )
                return None, consecutive_garbage_chunks, garbage_detected
        else:
            consecutive_garbage_chunks = 0
        return chunk, consecutive_garbage_chunks, garbage_detected

    async def _handle_empty_response_recovery(
        self,
        stream_ctx: "StreamingChatContext",
        tools: Any,
    ) -> tuple[bool, Any, Optional[StreamChunk]]:
        """Handle empty response recovery with multi-strategy retry."""
        orch = self._orchestrator

        # Reasoning-token exhaustion: when the empty stream carried reasoning
        # deltas but no content, the completion budget was consumed by thinking.
        # Retrying with identical parameters is deterministic failure (observed:
        # two consecutive ~112s reasoning-only streams, session
        # modality-doc-review-fixes-b4e87728) — raise max_tokens for the retry
        # so the model has budget left after reasoning.
        #
        # The escalation rule itself lives in victor.agent.recovery.empty_response
        # so the planning path (which calls provider.chat directly and never
        # reaches this streaming-only method) applies the same policy.
        from victor.agent.recovery.empty_response import (
            DEFAULT_TEMPERATURE_LADDER,
            diagnose_empty_response,
            next_retry_parameters,
        )

        diagnostics = getattr(stream_ctx, "empty_stream_diagnostics", None) or {}
        diagnosis = diagnose_empty_response(diagnostics)
        escalation = next_retry_parameters(
            attempt=0,
            base_max_tokens=orch.max_tokens,
            diagnosis=diagnosis,
            # The streaming retry walks the temperature ladder itself below.
            temperature_ladder=(),
        )
        recovery_max_tokens = escalation.max_tokens
        if recovery_max_tokens != orch.max_tokens:
            logger.info("Empty stream: %s (was max_tokens=%s)", escalation.reason, orch.max_tokens)

        recovery_temps = list(DEFAULT_TEMPERATURE_LADDER)
        for temp in recovery_temps:
            try:
                from victor.agent.conversation.types import (
                    MESSAGE_SOURCE_METADATA_KEY,
                    MessageSource,
                )

                orch.add_message(
                    "system",
                    "Please provide a response to the user's question. "
                    "If you need to use tools, go ahead. Otherwise, provide a text answer.",
                    metadata={MESSAGE_SOURCE_METADATA_KEY: MessageSource.SYSTEM_INJECTED.value},
                )

                provider_kwargs: Dict[str, Any] = dict(
                    getattr(stream_ctx, "provider_kwargs", {}) or {}
                )
                if (
                    orch.thinking
                    or provider_kwargs.get("execution_mode") == "escalated_single_agent"
                ):
                    provider_kwargs["thinking"] = {
                        "type": "enabled",
                        "budget_tokens": 10000,
                    }

                full_content = ""
                recovered_tool_calls = None

                retry_assembled = orch.get_assembled_messages(
                    current_query=stream_ctx.user_message if stream_ctx else None,
                    goals=(stream_ctx.goals if stream_ctx else None),
                    selected_tools=tools,
                    planned_tools=(stream_ctx.planned_tools if stream_ctx else None),
                )
                # aclosing guarantees the recovery stream is finalized in THIS task on the
                # tool-call `break` below, rather than left to GC (which would close it
                # off-task and raise "ignored GeneratorExit" / "no running event loop").
                async with contextlib.aclosing(
                    orch.provider.stream(
                        messages=retry_assembled,
                        model=orch.model,
                        temperature=temp,
                        max_tokens=recovery_max_tokens,
                        tools=tools,
                        **provider_kwargs,
                    )
                ) as recovery_stream:
                    async for chunk in recovery_stream:
                        if hasattr(stream_ctx, "reset_activity_timer"):
                            stream_ctx.reset_activity_timer()
                        if chunk.content:
                            full_content += chunk.content
                        if chunk.tool_calls:
                            recovered_tool_calls = chunk.tool_calls
                            break

                if recovered_tool_calls:
                    logger.info(f"Recovery at temperature {temp} produced tool calls")
                    # Record the assistant tool_calls message BEFORE the main loop
                    # executes the tools. Without it the tool results land in history
                    # with no matching assistant tool_calls entry, the pairing repairs
                    # strand one side or the other on every subsequent turn, and
                    # strict providers (Moonshot/OpenAI-compat) reject the payload
                    # with a non-retryable 400 (session modality-doc-review-fixes-
                    # b4e87728). Mirrors the normal tool-call path in
                    # chat_stream_executor and the content path below.
                    from victor.agent.conversation.types import (
                        MESSAGE_SOURCE_METADATA_KEY,
                        MessageSource,
                    )

                    orch.add_message(
                        "assistant",
                        "",
                        tool_calls=recovered_tool_calls,
                        metadata={MESSAGE_SOURCE_METADATA_KEY: MessageSource.AGENT_RESPONSE.value},
                    )
                    return True, recovered_tool_calls, None

                if full_content.strip():
                    logger.info(
                        f"Recovery at temperature {temp} produced content "
                        f"({len(full_content)} chars)"
                    )
                    sanitized = orch.sanitizer.sanitize(full_content)
                    if sanitized:
                        from victor.agent.conversation.types import (
                            MESSAGE_SOURCE_METADATA_KEY,
                            MessageSource,
                        )

                        orch.add_message(
                            "assistant",
                            sanitized,
                            metadata={
                                MESSAGE_SOURCE_METADATA_KEY: MessageSource.AGENT_RESPONSE.value
                            },
                        )
                    final_chunk = orch._chunk_generator.generate_content_chunk(
                        sanitized or full_content, is_final=True
                    )
                    return True, None, final_chunk

            except Exception as exc:
                logger.warning(f"Recovery attempt at temperature {temp} failed: {exc}")
                continue

        return False, None, None
