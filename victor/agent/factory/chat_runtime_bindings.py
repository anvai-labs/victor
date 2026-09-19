# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
# Licensed under the Apache License, Version 2.0 (the "License").

"""Composition boundary for legacy chat-runtime construction signatures."""

from typing import Any
import weakref

from victor.agent.services.chat_delivery import ChatDelivery
from victor.agent.services.chat_planning import ChatPlanning

from victor.agent.services.chat_runtime_services import (
    ChatRuntimeServices,
    SessionTaskRequirementState,
)
from victor.agent.services.orchestrator_protocol_adapter import OrchestratorProtocolAdapter
from victor.agent.services.task_guidance_runtime import TaskGuidanceRuntime
from victor.agent.services.tool_selection_runtime import ToolSelectionRuntime
from victor.agent.session_state_accessor import SessionStateAccessor


class _WeakRuntimeHost:
    """Forward runtime state without extending the facade's lifetime.

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

    def __getattr__(self, name: str) -> Any:
        return getattr(self._owner(), name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._owner(), name, value)


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
    runtime_host = _WeakRuntimeHost(owner)
    return ChatRuntimeServices(
        session=SessionTaskRequirementState(accessor),
        stream_turn_lock=accessor.stream_turn_lock,
        delivery=ChatDelivery(
            chunks=getattr(owner, "_chunk_generator", None),
            sanitizer=getattr(owner, "sanitizer", None),
        ),
        planning=ChatPlanning(
            guidance=TaskGuidanceRuntime(runtime_host),
            planner=getattr(owner, "_tool_planner", None),
            selection=ToolSelectionRuntime(runtime_host),
        ),
        recovery=getattr(owner, "_recovery_service", None)
        or getattr(owner, "_recovery_coordinator", None),
    )
