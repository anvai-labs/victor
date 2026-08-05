# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Coordination runtime boundaries for AgentOrchestrator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from victor.agent.runtime.provider_runtime import LazyRuntimeProxy


@dataclass(frozen=True)
class CoordinationRuntimeComponents:
    """Coordination runtime handles exposed to the orchestrator facade.

    Note: recovery_coordinator is now RecoveryService with native streaming runtime.
    The name is retained for compatibility but will be renamed in a future update.
    """

    recovery_coordinator: LazyRuntimeProxy[Any]
    chunk_generator: LazyRuntimeProxy[Any]
    tool_planner: LazyRuntimeProxy[Any]
    task_coordinator: LazyRuntimeProxy[Any]
    coordination_advisor_runtime: LazyRuntimeProxy[Any]


def create_coordination_runtime_components(
    *,
    factory: Any,
    get_recovery_service: Optional[Callable[[], Any]] = None,
) -> CoordinationRuntimeComponents:
    """Create lazy coordination components for orchestrator wiring.

    The recovery_coordinator is now RecoveryService with native streaming runtime
    enabled. When a recovery service accessor is provided, bind it lazily on
    first recovery coordinator materialization for backward compatibility.
    """

    def _create_recovery_coordinator() -> Any:
        recovery_coordinator = factory.create_recovery_coordinator()
        if get_recovery_service is not None and hasattr(
            recovery_coordinator, "bind_recovery_service"
        ):
            recovery_coordinator.bind_recovery_service(get_recovery_service())
        return recovery_coordinator

    return CoordinationRuntimeComponents(
        recovery_coordinator=LazyRuntimeProxy(
            factory=_create_recovery_coordinator,
            name="recovery_coordinator",
        ),
        chunk_generator=LazyRuntimeProxy(
            factory=lambda: factory.create_chunk_generator(),
            name="chunk_generator",
        ),
        tool_planner=LazyRuntimeProxy(
            factory=lambda: factory.create_tool_planner(),
            name="tool_planner",
        ),
        task_coordinator=LazyRuntimeProxy(
            factory=lambda: factory.create_task_coordinator(),
            name="task_coordinator",
        ),
        coordination_advisor_runtime=LazyRuntimeProxy(
            factory=lambda: factory.create_coordination_advisor_runtime(),
            name="coordination_advisor_runtime",
        ),
    )
