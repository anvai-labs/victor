# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Interaction runtime boundaries for AgentOrchestrator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from victor.runtime.context import ResolvedRuntimeServices


@dataclass(frozen=True)
class InteractionRuntimeComponents:
    """Interaction runtime handles exposed to the orchestrator facade."""

    chat_service: Any
    tool_service: Any
    session_service: Any
    context_service: Any
    recovery_service: Any


def create_interaction_runtime_components(
    *,
    enabled_tools: Optional[Any],
    factory: Any,
    context_compactor: Any,
    tool_pipeline: Any,
    tool_registry: Any,
    tool_executor: Any,
    tool_cache: Any,
    tool_budget: int,
    tool_selector: Any,
    tool_access_controller: Any,
    mode_controller: Any,
    argument_normalizer: Any,
    session_state_manager: Any,
    lifecycle_manager: Any,
    memory_manager: Any,
    memory_session_id: Optional[str],
    checkpoint_manager: Any,
    cost_tracker: Any,
    conversation_controller: Any,
    streaming_coordinator: Any,
    runtime_services: Optional[ResolvedRuntimeServices] = None,
) -> InteractionRuntimeComponents:
    """Create service-first interaction/runtime components for orchestrator wiring."""
    from victor.agent.services.chat_service import ChatService, ChatServiceConfig
    from victor.agent.services.recovery_service import RecoveryService
    from victor.agent.services.session_service import SessionService
    from victor.agent.services.tool_service import ToolService, ToolServiceConfig

    resolved_services = runtime_services or ResolvedRuntimeServices()

    resolved_tool_service = resolved_services.tool
    if resolved_tool_service is None:
        resolved_tool_service = ToolService(
            config=ToolServiceConfig(default_tool_budget=tool_budget),
            tool_selector=tool_selector,
            tool_executor=tool_executor,
            tool_registrar=tool_registry,
        )

    if hasattr(resolved_tool_service, "bind_runtime_components"):
        resolved_tool_service.bind_runtime_components(
            tool_registry=tool_registry,
            tool_pipeline=tool_pipeline,
            tool_cache=tool_cache,
            mode_controller=mode_controller,
            argument_normalizer=argument_normalizer,
        )

    if enabled_tools and hasattr(resolved_tool_service, "set_enabled_tools"):
        resolved_tool_service.set_enabled_tools(enabled_tools)

    resolved_session_service = resolved_services.session
    if resolved_session_service is None:
        resolved_session_service = SessionService(
            session_state_manager=session_state_manager,
            lifecycle_manager=lifecycle_manager,
            memory_manager=memory_manager,
            checkpoint_manager=checkpoint_manager,
            cost_tracker=cost_tracker,
        )

    if hasattr(resolved_session_service, "bind_runtime_components"):
        resolved_session_service.bind_runtime_components(
            lifecycle_manager=lifecycle_manager,
            memory_manager=memory_manager,
            checkpoint_manager=checkpoint_manager,
            cost_tracker=cost_tracker,
            memory_session_id=memory_session_id,
        )

    resolved_context_service = resolved_services.context
    if resolved_context_service is None:
        from victor.agent.services.adapters.context_adapter import ContextServiceAdapter

        resolved_context_service = ContextServiceAdapter(
            conversation_controller=conversation_controller,
            context_compactor=context_compactor,
        )

    resolved_recovery_service = resolved_services.recovery
    if resolved_recovery_service is None:
        resolved_recovery_service = RecoveryService()

    resolved_chat_service = resolved_services.chat
    if resolved_chat_service is None:
        resolved_chat_service = ChatService(
            config=ChatServiceConfig(
                max_iterations=200,
                stream_chunk_size=100,
            ),
            provider_service=resolved_services.provider,
            tool_service=resolved_tool_service,
            context_service=resolved_context_service,
            recovery_service=resolved_recovery_service,
            conversation_controller=conversation_controller,
            streaming_coordinator=streaming_coordinator,
        )

    return InteractionRuntimeComponents(
        chat_service=resolved_chat_service,
        tool_service=resolved_tool_service,
        session_service=resolved_session_service,
        context_service=resolved_context_service,
        recovery_service=resolved_recovery_service,
    )
