# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
# Licensed under the Apache License, Version 2.0 (the "License").

"""Enumerated service capabilities consumed by the chat runtime.

FEP-0031 phase 1 is incremental: task requirements, response delivery,
planning/guidance, and stream execution controls have migrated. The view keeps
an explicit capability shape while resolving mutable session-owned state at its
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


class CompletionSummaryStore(Protocol):
    """Conversation operations needed to carry a completion summary forward."""

    def persist_compaction_summary(self, summary: str, message_ids: list[str]) -> None: ...

    def inject_compaction_context(self) -> bool: ...


@dataclass(frozen=True, slots=True)
class ChatCompletion:
    """Detect terminal responses and persist their continuation summary."""

    detector: CompletionDetector | None = None
    summary_store: CompletionSummaryStore | None = None

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

    def persist_terminal_summary(self) -> None:
        """Persist the detector summary after the caller marks the turn complete."""
        summary = self.terminal_summary()
        if summary and self.summary_store is not None:
            try:
                self.summary_store.persist_compaction_summary(summary, [])
                self.summary_store.inject_compaction_context()
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


class OutcomeRecorder(Protocol):
    """Runtime feedback sink without exposing its composition owner."""

    def record_outcome(
        self,
        *,
        success: bool,
        quality_score: float,
        user_satisfied: bool,
        completed: bool,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class ChatFeedback:
    """Best-effort outcome feedback for chat execution."""

    recorder: OutcomeRecorder | None = None

    def record_outcome(
        self,
        *,
        success: bool,
        quality_score: float = 0.5,
        user_satisfied: bool = True,
        completed: bool = True,
    ) -> None:
        if self.recorder is not None:
            self.recorder.record_outcome(
                success=success,
                quality_score=quality_score,
                user_satisfied=user_satisfied,
                completed=completed,
            )


@dataclass(frozen=True, slots=True)
class ChatRuntimeServices:
    """Explicit capabilities already migrated from the chat runtime facade."""

    session: TaskRequirementState
    stream_turn_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False, compare=False)
    delivery: ChatDelivery = field(default_factory=ChatDelivery)
    planning: ChatPlanning = field(default_factory=ChatPlanning)
    governance: ChatGovernance = field(default_factory=ChatGovernance)
    completion: ChatCompletion = field(default_factory=ChatCompletion)
    tool_calls: ChatToolCalls = field(default_factory=ChatToolCalls)
    feedback: ChatFeedback = field(default_factory=ChatFeedback)
    # Recovery is a turn capability, not a property of the orchestrator facade.
    recovery: object | None = None
