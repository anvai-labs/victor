# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
# Licensed under the Apache License, Version 2.0 (the "License").

"""Composition boundary for legacy chat-runtime construction signatures."""

import asyncio
from typing import Any
import weakref

from victor.agent.services.chat_delivery import ChatDelivery
from victor.agent.services.chat_planning import ChatPlanning

from victor.agent.services.chat_runtime_services import (
    ChatCompletion,
    ChatConversation,
    ChatFeedback,
    ChatGovernance,
    ChatRuntimeServices,
    ChatStreamLifecycle,
    ChatStreamMetrics,
    ChatTaskState,
    ChatToolCalls,
    SessionTaskRequirementState,
)
from victor.agent.services.chat_turn_runtime import ChatTurnRuntime, TaskReportMetrics
from victor.agent.services.orchestrator_protocol_adapter import OrchestratorProtocolAdapter
from victor.agent.services.task_guidance_runtime import TaskGuidanceRuntime
from victor.agent.services.tool_selection_runtime import ToolSelectionRuntime
from victor.agent.session_state_accessor import SessionStateAccessor


class _WeakOwner:
    """Resolve a runtime owner without extending the facade's lifetime.

    Runtime owners must support weak references. A strong-reference fallback
    would make this capability view keep the facade alive after its session is
    closed, which is the ownership bug this boundary prevents.
    """

    __slots__ = ("_owner_ref",)

    def __init__(self, owner: Any) -> None:
        try:
            owner_ref = weakref.ref(owner)
        except TypeError as exc:
            raise TypeError("Chat runtime owner must support weak references") from exc
        object.__setattr__(self, "_owner_ref", owner_ref)

    def _owner(self) -> Any:
        owner = self._owner_ref()
        if owner is None:
            raise RuntimeError("Chat runtime owner is no longer available")
        return owner


class _WeakRuntimeHost(_WeakOwner):
    """Forward compatibility runtime state through a weak owner."""

    __slots__ = ()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._owner(), name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._owner(), name, value)


class _ChatTurnStateView(_WeakOwner):
    """Enumerate the live facade state required by ``ChatTurnRuntime``."""

    __slots__ = ()

    @property
    def provider_name(self) -> str:
        owner = self._owner()
        provider = getattr(owner, "provider", None)
        return str(getattr(provider, "name", getattr(owner, "provider_name", "unknown")))

    @property
    def model(self) -> str:
        return str(getattr(self._owner(), "model", "unknown"))

    @property
    def stream_context(self) -> Any:
        return getattr(self._owner(), "_current_stream_context", None)

    @property
    def unified_tracker(self) -> Any:
        return getattr(self._owner(), "unified_tracker", None)

    @property
    def task_type_candidates(self) -> tuple[Any, Any]:
        owner = self._owner()
        return (
            getattr(owner, "_current_task_type", None),
            getattr(owner, "_task_type", None),
        )

    @property
    def context_service(self) -> Any:
        return getattr(self._owner(), "_context_service", None)

    @property
    def metrics(self) -> TaskReportMetrics:
        metrics = getattr(self._owner(), "_metrics_coordinator", None)
        if metrics is None:
            raise TypeError("Chat turn runtime requires task-report metrics")
        return metrics

    def apply_skill_for_turn(self, user_message: str) -> None:
        self._owner()._get_skill_runtime().apply_skill_for_turn(user_message)

    def activate_constraints(self, constraints: Any, vertical: str | None) -> None:
        owner = self._owner()
        owner._constraint_activator.activate_constraints(
            constraints=constraints,
            vertical=vertical or getattr(owner, "vertical", "coding"),
        )

    def deactivate_constraints(self) -> None:
        self._owner()._constraint_activator.deactivate_constraints()

    def assign_turn_credit(self) -> None:
        service = getattr(self._owner(), "_credit_tracking_service", None)
        if service is not None:
            service.assign_turn_credit_at_boundary()


class _ChatOutcomeView(_WeakOwner):
    """Forward the remaining compatibility feedback hook without retention."""

    __slots__ = ()

    def record_outcome(
        self,
        *,
        success: bool,
        quality_score: float,
        user_satisfied: bool,
        completed: bool,
    ) -> None:
        self._owner()._record_runtime_intelligence_outcome(
            success=success,
            quality_score=quality_score,
            user_satisfied=user_satisfied,
            completed=completed,
        )


class _ChatStreamLifecycleView(_WeakOwner):
    """Coordinate live stream state without exposing facade fields to consumers."""

    __slots__ = ()

    def begin(self) -> None:
        owner = self._owner()
        owner._cancel_event = asyncio.Event()
        owner._is_streaming = True

    def is_cancelled(self) -> bool:
        event = getattr(self._owner(), "_cancel_event", None)
        return bool(event is not None and event.is_set())

    def finish(self) -> None:
        owner = self._owner()
        owner._is_streaming = False
        owner._cancel_event = None


class _ChatStreamMetricsView(_WeakOwner):
    """Resolve the two metrics components behind one streaming capability."""

    __slots__ = ()

    def _collector(self) -> Any:
        collector = getattr(self._owner(), "_metrics_collector", None)
        if collector is None:
            raise TypeError("Chat streaming metrics require a collector")
        return collector

    def begin(self) -> Any:
        return self._collector().init_stream_metrics()

    def record_first_token(self) -> None:
        self._collector().record_first_token()

    def finalize(
        self,
        usage_data: dict[str, int],
        *,
        provider_diagnostics: dict[str, Any] | None = None,
    ) -> Any:
        coordinator = getattr(self._owner(), "_metrics_coordinator", None)
        if coordinator is None:
            raise TypeError("Chat streaming metrics require a coordinator")
        return coordinator.finalize_stream_metrics(
            usage_data,
            provider_diagnostics=provider_diagnostics,
        )


class _ChatTaskStateView(_WeakOwner):
    """Coordinate the live task tracker without exposing facade-owned state."""

    __slots__ = ()

    def _tracker(self) -> Any:
        tracker = getattr(self._owner(), "unified_tracker", None)
        if tracker is None:
            raise TypeError("Chat task state requires a unified tracker")
        return tracker

    def reset_turn(self) -> None:
        owner = self._owner()
        session_state = getattr(owner, "_session_state", None)
        reminder_manager = getattr(owner, "reminder_manager", None)
        if session_state is None or reminder_manager is None:
            raise TypeError("Chat task state requires session and reminder state")
        session_state.reset_for_new_turn()
        self._tracker().reset()
        reminder_manager.reset()

    def max_total_iterations(self) -> int:
        return int(self._tracker().config.get("max_total_iterations", 50))

    def detect_task_type(self, user_message: str) -> Any:
        return self._tracker().detect_task_type(user_message)

    def set_task_type(self, task_type: Any) -> None:
        self._tracker().set_task_type(task_type)

    def publish_task_type(self, task_type: Any) -> None:
        self._tracker().set_task_type(task_type)
        self._owner()._current_task_type = task_type.value

    def set_continuation_context(self, context: dict[str, Any] | None) -> None:
        self._owner()._pending_continuation_task_context = context

    def continuation_context(self) -> dict[str, Any] | None:
        context = getattr(self._owner(), "_pending_continuation_task_context", None)
        return context if isinstance(context, dict) else None

    def apply_prompt_requirements(
        self,
        *,
        tool_budget: int | None,
        iteration_budget: int | None,
    ) -> tuple[bool, bool]:
        tracker = self._tracker()
        tracker.progress.has_prompt_requirements = True
        tool_budget_updated = bool(tool_budget and tool_budget > tracker.progress.tool_budget)
        if tool_budget_updated:
            tracker.set_tool_budget(tool_budget)

        task_config = getattr(tracker, "_task_config", None)
        configured_iterations = int(getattr(task_config, "max_exploration_iterations", 0) or 0)
        iteration_budget_updated = bool(
            iteration_budget and iteration_budget > configured_iterations
        )
        if iteration_budget_updated:
            tracker.set_max_iterations(iteration_budget)
        return tool_budget_updated, iteration_budget_updated

    def max_exploration_iterations(self) -> int:
        return int(self._tracker().max_exploration_iterations)

    def set_tool_budget(self, budget: int) -> None:
        self._tracker().set_tool_budget(budget)


class _ChatToolCallView(_WeakOwner):
    """Resolve tool collaborators at call time without retaining the facade."""

    __slots__ = ()

    def reset(self) -> None:
        pipeline = getattr(self._owner(), "_tool_pipeline", None)
        if pipeline is not None:
            pipeline.reset()

    def parse_and_validate(
        self,
        tool_calls: list[dict[str, Any]] | None,
        full_content: str,
    ) -> tuple[list[dict[str, Any]] | None, str]:
        owner = self._owner()
        parser = getattr(owner, "_tool_service", None)
        adapter = getattr(owner, "tool_adapter", None)
        if parser is None or adapter is None:
            raise TypeError("Chat tool-call processing requires a parser and adapter")
        return parser.parse_and_validate_tool_calls(tool_calls, full_content, adapter)


class _ChatConversationView(_WeakOwner):
    """Resolve conversation operations without retaining the facade."""

    __slots__ = ()

    def _controller(self) -> Any:
        controller = getattr(self._owner(), "_conversation_controller", None)
        if controller is None:
            raise TypeError("Chat conversation processing requires a controller")
        return controller

    def messages(self) -> list[Any]:
        return list(getattr(self._controller(), "messages", None) or [])

    def record_actual_usage(self, prompt_tokens: int) -> None:
        controller = self._controller()
        total_chars = sum(len(message.content) for message in controller.messages)
        controller.record_actual_usage(prompt_tokens, total_chars)

    def persist_terminal_summary(self, summary: str) -> None:
        controller = self._controller()
        controller.persist_compaction_summary(summary, [])
        controller.inject_compaction_context()


def _runtime_owner(runtime_owner: Any) -> Any:
    return (
        runtime_owner._orchestrator
        if isinstance(runtime_owner, OrchestratorProtocolAdapter)
        else runtime_owner
    )


def bind_chat_turn_runtime(runtime_owner: Any) -> ChatTurnRuntime:
    """Build the complete turn frame over an enumerated weak state view."""
    return ChatTurnRuntime(_ChatTurnStateView(_runtime_owner(runtime_owner)))


def bind_chat_runtime_services(runtime_owner: Any) -> ChatRuntimeServices:
    """Bind existing session ownership, without retaining the facade in the view.

    An explicit view is required for standalone runtimes without a session owner;
    manufacturing fallback state would disconnect requirement tracking.
    """
    owner = _runtime_owner(runtime_owner)
    accessor = getattr(owner, "_session_accessor", None)
    if not isinstance(accessor, SessionStateAccessor):
        raise TypeError(
            "Chat runtime requires SessionStateAccessor or explicit ChatRuntimeServices"
        )
    runtime_host = _WeakRuntimeHost(owner)
    return ChatRuntimeServices(
        session=SessionTaskRequirementState(accessor),
        stream_turn_lock=accessor.stream_turn_lock,
        stream_lifecycle=ChatStreamLifecycle(_ChatStreamLifecycleView(owner)),
        metrics=ChatStreamMetrics(_ChatStreamMetricsView(owner)),
        task_state=ChatTaskState(_ChatTaskStateView(owner)),
        delivery=ChatDelivery(
            chunks=getattr(owner, "_chunk_generator", None),
            sanitizer=getattr(owner, "sanitizer", None),
        ),
        planning=ChatPlanning(
            guidance=TaskGuidanceRuntime(runtime_host),
            planner=getattr(owner, "_tool_planner", None),
            selection=ToolSelectionRuntime(runtime_host),
        ),
        governance=ChatGovernance(gate=getattr(owner, "_message_policy_gate", None)),
        completion=ChatCompletion(detector=getattr(owner, "_task_completion_detector", None)),
        conversation=ChatConversation(runtime=_ChatConversationView(owner)),
        tool_calls=ChatToolCalls(runtime=_ChatToolCallView(owner)),
        feedback=ChatFeedback(recorder=_ChatOutcomeView(owner)),
        recovery=getattr(owner, "_recovery_service", None)
        or getattr(owner, "_recovery_coordinator", None),
    )
