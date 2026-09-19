# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
# Licensed under the Apache License, Version 2.0 (the "License").

"""Enumerated service capabilities consumed by the chat runtime.

FEP-0031 phase 1 is incremental: task requirements, response delivery,
planning/guidance, stream execution controls, lifecycle, metrics, task
classification, context lifecycle, and runtime intelligence have migrated. The
view keeps an explicit capability shape while resolving mutable state at its
canonical owners.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

from victor.agent.session_state_accessor import SessionStateAccessor
from victor.agent.services.chat_delivery import ChatDelivery
from victor.agent.services.chat_planning import ChatPlanning
from victor.agent.unified_task_tracker import TrackerTaskType
from victor.framework.policies.gate import GateResult

logger = logging.getLogger(__name__)


class TaskRequirementState(Protocol):
    """The session state needed to prepare a turn's task requirements."""

    @property
    def required_files(self) -> list[str]: ...

    @required_files.setter
    def required_files(self, value: list[str]) -> None: ...

    @property
    def required_outputs(self) -> list[str]: ...

    @required_outputs.setter
    def required_outputs(self, value: list[str]) -> None: ...

    @property
    def read_files(self) -> set[str]: ...

    @property
    def all_files_read_nudge_sent(self) -> bool: ...

    @all_files_read_nudge_sent.setter
    def all_files_read_nudge_sent(self, value: bool) -> None: ...


class SessionTaskRequirementState:
    """Resolve state through the existing accessor after resets and restores."""

    __slots__ = ("_accessor",)

    def __init__(self, accessor: SessionStateAccessor) -> None:
        self._accessor = accessor

    @property
    def required_files(self) -> list[str]:
        return self._accessor.required_files

    @required_files.setter
    def required_files(self, value: list[str]) -> None:
        self._accessor.required_files = value

    @property
    def required_outputs(self) -> list[str]:
        return self._accessor.required_outputs

    @required_outputs.setter
    def required_outputs(self, value: list[str]) -> None:
        self._accessor.required_outputs = value

    @property
    def read_files(self) -> set[str]:
        return self._accessor.read_files_session

    @property
    def all_files_read_nudge_sent(self) -> bool:
        return self._accessor.all_files_read_nudge_sent

    @all_files_read_nudge_sent.setter
    def all_files_read_nudge_sent(self, value: bool) -> None:
        self._accessor.all_files_read_nudge_sent = value


class MessagePolicyGate(Protocol):
    """Request/response policy checks used by chat delivery."""

    async def gate_request(self, content: str) -> GateResult: ...

    async def gate_response(self, content: str) -> GateResult: ...


@dataclass(frozen=True, slots=True)
class ChatGovernance:
    """Optional message governance without exposing its composition owner."""

    gate: MessagePolicyGate | None = None

    @staticmethod
    def _require_result(result: Any, phase: str) -> GateResult:
        if not isinstance(result, GateResult):
            raise TypeError(f"Configured chat governance gate returned an invalid {phase} result")
        return result

    async def check_request(self, content: str) -> GateResult | None:
        if self.gate is None:
            return None
        return self._require_result(await self.gate.gate_request(content), "request")

    async def check_response(self, content: str) -> GateResult | None:
        if self.gate is None:
            return None
        return self._require_result(await self.gate.gate_response(content), "response")


class CompletionDetector(Protocol):
    """Completion signals consumed by the streaming execution path."""

    def analyze_response(self, content: str) -> Any: ...

    def get_completion_confidence(self) -> Any: ...

    def get_state(self) -> Any: ...

    def clear_active_signal(self) -> None: ...

    def reset(self) -> None: ...


@dataclass(frozen=True, slots=True)
class ChatCompletion:
    """Detect terminal responses and expose their sanitized summary."""

    detector: CompletionDetector | None = None

    def reset(self) -> None:
        if self.detector is not None:
            self.detector.reset()

    def terminal_summary(self) -> str:
        if self.detector is None:
            return ""
        from victor.core.completion_markers import strip_active_completion_markers

        summary = getattr(self.detector.get_state(), "last_summary", "")
        return strip_active_completion_markers(summary).strip()

    def detect_high_confidence(self, content: str, *, has_pending_tools: bool) -> bool:
        if self.detector is None or not content:
            return False

        from victor.agent.task_completion import CompletionConfidence

        self.detector.analyze_response(content)
        if self.detector.get_completion_confidence() != CompletionConfidence.HIGH:
            return False
        if has_pending_tools:
            self.detector.clear_active_signal()
            return False

        return True


class ConversationRuntime(Protocol):
    """Conversation history and accounting operations used by streaming chat."""

    def messages(self) -> list[Any]: ...

    def record_actual_usage(self, prompt_tokens: int) -> None: ...

    def persist_terminal_summary(self, summary: str) -> None: ...


@dataclass(frozen=True, slots=True)
class ChatConversation:
    """Best-effort conversation history, accounting, and summary capability."""

    runtime: ConversationRuntime | None = None

    def messages(self) -> list[Any]:
        if self.runtime is None:
            return []
        try:
            return self.runtime.messages()
        except Exception as exc:
            logger.debug("Failed to read chat conversation messages: %s", exc)
            return []

    def record_actual_usage(self, prompt_tokens: int) -> None:
        if self.runtime is None or prompt_tokens <= 0:
            return
        try:
            self.runtime.record_actual_usage(prompt_tokens)
        except Exception as exc:
            logger.debug("Failed to record actual chat usage: %s", exc)

    def persist_terminal_summary(self, summary: str) -> None:
        if self.runtime is None or not summary:
            return
        try:
            self.runtime.persist_terminal_summary(summary)
            logger.info("VICTOR_SUMMARY persisted for next-turn context injection")
        except Exception as exc:
            logger.debug("Failed to persist VICTOR_SUMMARY: %s", exc)


class ToolCallRuntime(Protocol):
    """Tool parsing and reset operations required by streaming chat."""

    def reset(self) -> None: ...

    def parse_and_validate(
        self,
        tool_calls: list[dict[str, Any]] | None,
        full_content: str,
    ) -> tuple[list[dict[str, Any]] | None, str]: ...


@dataclass(frozen=True, slots=True)
class ChatToolCalls:
    """Tool-call parsing and per-turn pipeline reset capability."""

    runtime: ToolCallRuntime | None = None

    def reset(self) -> None:
        if self.runtime is not None:
            self.runtime.reset()

    def parse_and_validate(
        self,
        tool_calls: list[dict[str, Any]] | None,
        full_content: str,
    ) -> tuple[list[dict[str, Any]] | None, str]:
        if self.runtime is None:
            raise TypeError("Chat tool-call processing requires a runtime")
        return self.runtime.parse_and_validate(tool_calls, full_content)


@dataclass(frozen=True, slots=True)
class ChatRoutingIntelligence:
    """Learned routing additions and their optional serialized policy."""

    context: dict[str, Any] = field(default_factory=dict)
    structured_policy: dict[str, Any] | None = None


class RuntimeIntelligenceRuntime(Protocol):
    """Learning and routing operations consumed by streaming chat."""

    def executor_runtime(self) -> Any: ...

    async def prepare_request(
        self,
        *,
        task: str,
        task_type: str,
    ) -> dict[str, Any] | None: ...

    def routing_context(
        self,
        *,
        query: str,
        scope_context: dict[str, Any],
    ) -> ChatRoutingIntelligence: ...

    def record_topology_outcome(self, payload: dict[str, Any]) -> None: ...

    def record_outcome(
        self,
        *,
        success: bool,
        quality_score: float,
        user_satisfied: bool,
        completed: bool,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class ChatRuntimeIntelligence:
    """Optional request guidance, routing policy, and outcome feedback."""

    runtime: RuntimeIntelligenceRuntime | None = None

    def executor_runtime(self) -> Any:
        if self.runtime is None:
            return None
        return self.runtime.executor_runtime()

    async def prepare_request(
        self,
        *,
        task: str,
        task_type: str,
    ) -> dict[str, Any] | None:
        if self.runtime is None:
            return None
        return await self.runtime.prepare_request(task=task, task_type=task_type)

    def routing_context(
        self,
        *,
        query: str,
        scope_context: dict[str, Any],
    ) -> ChatRoutingIntelligence:
        if self.runtime is None:
            return ChatRoutingIntelligence()
        return self.runtime.routing_context(query=query, scope_context=scope_context)

    def record_topology_outcome(self, payload: dict[str, Any]) -> None:
        if self.runtime is not None:
            self.runtime.record_topology_outcome(payload)

    def record_outcome(
        self,
        *,
        success: bool,
        quality_score: float = 0.5,
        user_satisfied: bool = True,
        completed: bool = True,
    ) -> None:
        if self.runtime is not None:
            self.runtime.record_outcome(
                success=success,
                quality_score=quality_score,
                user_satisfied=user_satisfied,
                completed=completed,
            )


class StreamLifecycleRuntime(Protocol):
    """Mutable stream state exposed at the chat composition boundary."""

    def begin(self) -> None: ...

    def bind_context(self, context: Any) -> None: ...

    def current_context(self) -> Any | None: ...

    def clear_context(self, context: Any) -> None: ...

    def is_cancelled(self) -> bool: ...

    def finish(self) -> None: ...


@dataclass(frozen=True, slots=True)
class ChatStreamLifecycle:
    """Own the start, cancellation, and terminal state of one stream."""

    runtime: StreamLifecycleRuntime

    def begin(self) -> None:
        self.runtime.begin()

    def bind_context(self, context: Any) -> None:
        self.runtime.bind_context(context)

    def current_context(self) -> Any | None:
        return self.runtime.current_context()

    def clear_context(self, context: Any) -> None:
        self.runtime.clear_context(context)

    def is_cancelled(self) -> bool:
        return self.runtime.is_cancelled()

    def finish(self) -> None:
        self.runtime.finish()


class StreamMetricsRuntime(Protocol):
    """Metrics operations needed by the streaming path."""

    def begin(self) -> Any: ...

    def record_first_token(self) -> None: ...

    def finalize(
        self,
        usage_data: dict[str, int],
        *,
        provider_diagnostics: dict[str, Any] | None = None,
    ) -> Any: ...


@dataclass(frozen=True, slots=True)
class ChatStreamMetrics:
    """Own stream metric initialization, first-token timing, and finalization."""

    runtime: StreamMetricsRuntime | None = None

    def _require_runtime(self) -> StreamMetricsRuntime:
        if self.runtime is None:
            raise TypeError("Chat streaming metrics require a runtime")
        return self.runtime

    def begin(self) -> Any:
        return self._require_runtime().begin()

    def record_first_token(self) -> None:
        self._require_runtime().record_first_token()

    def finalize(
        self,
        usage_data: dict[str, int],
        *,
        provider_diagnostics: dict[str, Any] | None = None,
    ) -> Any:
        return self._require_runtime().finalize(
            usage_data,
            provider_diagnostics=provider_diagnostics,
        )


class TaskStateRuntime(Protocol):
    """Per-turn task classification and budget state used by streaming chat."""

    def reset_turn(self) -> None: ...

    def max_total_iterations(self) -> int: ...

    def detect_task_type(self, user_message: str) -> TrackerTaskType: ...

    def set_task_type(self, task_type: TrackerTaskType) -> None: ...

    def publish_task_type(self, task_type: TrackerTaskType) -> None: ...

    def set_continuation_context(self, context: dict[str, Any] | None) -> None: ...

    def continuation_context(self) -> dict[str, Any] | None: ...

    def apply_prompt_requirements(
        self,
        *,
        tool_budget: int | None,
        iteration_budget: int | None,
    ) -> tuple[bool, bool]: ...

    def max_exploration_iterations(self) -> int: ...

    def set_tool_budget(self, budget: int) -> None: ...


@dataclass(frozen=True, slots=True)
class ChatTaskState:
    """Expose task classification and budget updates without facade state access."""

    runtime: TaskStateRuntime | None = None

    def _require_runtime(self) -> TaskStateRuntime:
        if self.runtime is None:
            raise TypeError("Chat task state requires a runtime")
        return self.runtime

    def reset_turn(self) -> None:
        self._require_runtime().reset_turn()

    def max_total_iterations(self) -> int:
        return self._require_runtime().max_total_iterations()

    def detect_task_type(self, user_message: str) -> TrackerTaskType:
        return self._require_runtime().detect_task_type(user_message)

    def set_task_type(self, task_type: TrackerTaskType) -> None:
        self._require_runtime().set_task_type(task_type)

    def publish_task_type(self, task_type: TrackerTaskType) -> None:
        self._require_runtime().publish_task_type(task_type)

    def set_continuation_context(self, context: dict[str, Any] | None) -> None:
        self._require_runtime().set_continuation_context(context)

    def continuation_context(self) -> dict[str, Any] | None:
        return self._require_runtime().continuation_context()

    def apply_prompt_requirements(
        self,
        *,
        tool_budget: int | None,
        iteration_budget: int | None,
    ) -> tuple[bool, bool]:
        return self._require_runtime().apply_prompt_requirements(
            tool_budget=tool_budget,
            iteration_budget=iteration_budget,
        )

    def max_exploration_iterations(self) -> int:
        return self._require_runtime().max_exploration_iterations()

    def set_tool_budget(self, budget: int) -> None:
        self._require_runtime().set_tool_budget(budget)


@dataclass(frozen=True, slots=True)
class ChatCompactionEvent:
    """Normalized compaction result consumed by the streaming turn."""

    messages_removed: int
    tokens_freed: int = 0
    summary: str = ""
    strategy: str = "tiered"
    policy_reason: str = ""


class ContextLifecycleRuntime(Protocol):
    """Context startup and pre-iteration compaction operations."""

    async def start_background_compaction(self) -> None: ...

    async def compact_before_iteration(
        self,
        user_message: str,
    ) -> ChatCompactionEvent | None: ...


@dataclass(frozen=True, slots=True)
class ChatContextLifecycle:
    """Expose context lifecycle policy without facade-owned collaborators."""

    runtime: ContextLifecycleRuntime | None = None

    async def start_background_compaction(self) -> None:
        if self.runtime is not None:
            await self.runtime.start_background_compaction()

    async def compact_before_iteration(
        self,
        user_message: str,
    ) -> ChatCompactionEvent | None:
        if self.runtime is None:
            return None
        return await self.runtime.compact_before_iteration(user_message)


@dataclass(frozen=True, slots=True)
class ChatRuntimeServices:
    """Explicit capabilities already migrated from the chat runtime facade."""

    session: TaskRequirementState
    stream_lifecycle: ChatStreamLifecycle
    stream_turn_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False, compare=False)
    metrics: ChatStreamMetrics = field(default_factory=ChatStreamMetrics)
    task_state: ChatTaskState = field(default_factory=ChatTaskState)
    context_lifecycle: ChatContextLifecycle = field(default_factory=ChatContextLifecycle)
    delivery: ChatDelivery = field(default_factory=ChatDelivery)
    planning: ChatPlanning = field(default_factory=ChatPlanning)
    governance: ChatGovernance = field(default_factory=ChatGovernance)
    completion: ChatCompletion = field(default_factory=ChatCompletion)
    conversation: ChatConversation = field(default_factory=ChatConversation)
    tool_calls: ChatToolCalls = field(default_factory=ChatToolCalls)
    intelligence: ChatRuntimeIntelligence = field(default_factory=ChatRuntimeIntelligence)
    # Recovery is a turn capability, not a property of the orchestrator facade.
    recovery: object | None = None
