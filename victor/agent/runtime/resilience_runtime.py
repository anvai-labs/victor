# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Resilience runtime boundaries for AgentOrchestrator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from victor.agent.runtime.provider_runtime import LazyRuntimeProxy


@dataclass(frozen=True)
class ResilienceRuntimeComponents:
    """Resilience runtime handles exposed to the orchestrator facade."""

    recovery_handler: LazyRuntimeProxy[Any]
    recovery_integration: LazyRuntimeProxy[Any]


def create_resilience_runtime_components(
    *,
    factory: Any,
    context_compactor: Any,
) -> ResilienceRuntimeComponents:
    """Create lazy resilience components for orchestrator wiring."""

    def _build_recovery_handler() -> Any:
        handler = factory.create_recovery_handler()
        if handler is not None:
            set_context_compactor = getattr(handler, "set_context_compactor", None)
            if callable(set_context_compactor):
                set_context_compactor(context_compactor)
        return handler

    recovery_handler = LazyRuntimeProxy(
        factory=_build_recovery_handler,
        name="recovery_handler",
    )

    def _build_recovery_integration() -> Any:
        return factory.create_recovery_integration(recovery_handler.get_instance())

    return ResilienceRuntimeComponents(
        recovery_handler=recovery_handler,
        recovery_integration=LazyRuntimeProxy(
            factory=_build_recovery_integration,
            name="recovery_integration",
        ),
    )
