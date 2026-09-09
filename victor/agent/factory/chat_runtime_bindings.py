# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
# Licensed under the Apache License, Version 2.0 (the "License").

"""Composition boundary for legacy chat-runtime construction signatures."""

from typing import Any

from victor.agent.services.chat_runtime_services import (
    ChatRuntimeServices,
    SessionTaskRequirementState,
)
from victor.agent.services.orchestrator_protocol_adapter import OrchestratorProtocolAdapter
from victor.agent.session_state_accessor import SessionStateAccessor


def bind_chat_runtime_services(runtime_owner: Any) -> ChatRuntimeServices:
    """Bind existing session ownership, without retaining the facade in the view.

    An explicit view is required for standalone runtimes without a session owner;
    manufacturing fallback state would disconnect requirement tracking.
    """
    owner = (
        runtime_owner._orchestrator
        if isinstance(runtime_owner, OrchestratorProtocolAdapter)
        else runtime_owner
    )
    accessor = getattr(owner, "_session_accessor", None)
    if not isinstance(accessor, SessionStateAccessor):
        raise TypeError(
            "Chat runtime requires SessionStateAccessor or explicit ChatRuntimeServices"
        )
    return ChatRuntimeServices(session=SessionTaskRequirementState(accessor))
