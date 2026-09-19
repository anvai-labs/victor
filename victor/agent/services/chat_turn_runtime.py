# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
# Licensed under the Apache License, Version 2.0 (the "License").

"""Service-owned setup, reporting, and teardown for one chat turn."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from victor.agent.task_report_metadata import (
    build_compaction_metadata,
    build_task_report_finish_metadata,
    build_task_report_start_metadata,
    resolve_task_type,
)


class TaskReportMetrics(Protocol):
    """Metrics operations needed to frame one chat task report."""

    def start_task_report(
        self,
        description: str,
        task_type: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str: ...

    def finish_task_report(
        self,
        success: bool,
        *,
        task_type: str | None = None,
        metadata: dict[str, Any] | None = None,
        error: str | None = None,
        tool_schema_tokens: int | None = None,
        compaction: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...

    def get_last_tool_strategy_event(self) -> dict[str, Any] | None: ...


class ChatTurnState(Protocol):
    """Live state and collaborators required by the turn frame."""

    @property
    def provider_name(self) -> str: ...

    @property
    def model(self) -> str: ...

    @property
    def stream_context(self) -> Any: ...

    @property
    def unified_tracker(self) -> Any: ...

    @property
    def task_type_candidates(self) -> tuple[Any, Any]: ...

    @property
    def context_service(self) -> Any: ...

    @property
    def metrics(self) -> TaskReportMetrics: ...

    def apply_skill_for_turn(self, user_message: str) -> None: ...

    def activate_constraints(self, constraints: Any, vertical: str | None) -> None: ...

    def deactivate_constraints(self) -> None: ...

    def assign_turn_credit(self) -> None: ...


@dataclass(frozen=True, slots=True)
class ChatTurnRuntime:
    """Own the complete per-turn frame without retaining the facade."""

    state: ChatTurnState

    def enter(
        self,
        user_message: str,
        *,
        stream: bool,
        constraints: Any = None,
        vertical: str | None = None,
    ) -> None:
        """Activate stream skills and temporary constraints for a turn."""
        if stream:
            self.state.apply_skill_for_turn(user_message)
        if constraints:
            self.state.activate_constraints(constraints, vertical)

    def exit(
        self,
        _user_message: str,
        *,
        stream: bool,
        constraints: Any = None,
        vertical: str | None = None,
    ) -> None:
        """Release temporary state and close the credit-assignment loop."""
        del stream, vertical
        if constraints:
            self.state.deactivate_constraints()
        self.state.assign_turn_credit()

    def start_task_report(
        self,
        user_message: str,
        *,
        stream: bool = False,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        """Start the canonical task report for this turn."""
        return self.state.metrics.start_task_report(
            description=user_message,
            task_type=self._resolve_task_type(),
            metadata=build_task_report_start_metadata(
                stream=stream,
                provider_name=self.state.provider_name,
                model=self.state.model,
                metadata=metadata,
            ),
        )

    def finish_task_report(
        self,
        success: bool,
        *,
        user_message: str,
        stream: bool = False,
        response: Any = None,
        error: BaseException | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Finish the canonical task report with continuation and compaction data."""
        stream_context = self.state.stream_context
        metrics = self.state.metrics
        last_tool_event = metrics.get_last_tool_strategy_event() or {}
        return metrics.finish_task_report(
            success,
            task_type=self._resolve_task_type(),
            metadata=build_task_report_finish_metadata(
                stream=stream,
                provider_name=self.state.provider_name,
                model=self.state.model,
                user_message=user_message,
                response=response,
                stream_ctx=stream_context,
                metadata=metadata,
            ),
            error=str(error) if error else None,
            tool_schema_tokens=int(last_tool_event.get("tool_tokens", 0) or 0),
            compaction=build_compaction_metadata(
                stream_context,
                self.state.context_service,
            ),
        )

    def _resolve_task_type(self) -> str:
        return resolve_task_type(
            self.state.stream_context,
            self.state.unified_tracker,
            self.state.task_type_candidates,
        )
