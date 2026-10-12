"""Runtime identity context for root agents and subagents."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, Iterator, Optional, Protocol
from uuid import uuid4


def derive_scoped_session_id(parent_session_id: Optional[str], team_id: str, member_id: str) -> str:
    """Colon-scoped child session id for team/worktree-isolated members.

    Shared by AgentRuntimeContext.derive_child and TeamExecution so the
    format has exactly one definition.
    """
    parent = parent_session_id or "session"
    return f"{parent}:{team_id}:{member_id}"


@dataclass(frozen=True)
class AgentRuntimeContext:
    """Identity and session scope for one agent runtime instance."""

    agent_id: str
    display_name: str
    role: str
    session_id: str
    parent_session_id: Optional[str] = None
    team_id: Optional[str] = None
    plan_id: Optional[str] = None
    plan_step_id: Optional[str] = None
    member_id: Optional[str] = None

    def identity_metadata(self) -> Dict[str, Any]:
        """Return stable identity metadata for persistence and observability."""
        return {
            "agent_id": self.agent_id,
            "display_name": self.display_name,
            "role": self.role,
            "session_id": self.session_id,
            "parent_session_id": self.parent_session_id,
            "team_id": self.team_id,
            "plan_id": self.plan_id,
            "plan_step_id": self.plan_step_id,
            "member_id": self.member_id,
        }

    def derive_child(
        self,
        *,
        agent_id: str,
        display_name: str,
        role: str,
        member_id: str,
        team_id: Optional[str] = None,
        plan_id: Optional[str] = None,
        plan_step_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> "AgentRuntimeContext":
        """Create a child runtime context with its own session scope."""
        resolved_team_id = team_id or self.team_id or "team"
        child_session_id = session_id or derive_scoped_session_id(
            self.session_id, resolved_team_id, member_id
        )
        return AgentRuntimeContext(
            agent_id=agent_id,
            display_name=display_name,
            role=role,
            session_id=child_session_id,
            parent_session_id=self.session_id,
            team_id=resolved_team_id,
            plan_id=plan_id or self.plan_id,
            plan_step_id=plan_step_id,
            member_id=member_id,
        )


RuntimeContextResolver = Callable[[], AgentRuntimeContext]


def require_runtime_context(resolver: Optional[RuntimeContextResolver]) -> AgentRuntimeContext:
    """Validate identity before an effect or lifecycle boundary; never invent one."""
    if resolver is None:
        raise RuntimeError("Runtime identity resolver is not initialized")
    context = resolver()
    if not isinstance(context, AgentRuntimeContext):
        raise TypeError("Runtime identity resolver must return AgentRuntimeContext")
    for field, value in (
        ("agent_id", context.agent_id),
        ("session_id", context.session_id),
        ("role", context.role),
        ("display_name", context.display_name),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Runtime identity requires a nonempty {field}")
    return context


class RuntimeSessionState(Protocol):
    """Only session state needed to bind current identity at assembly."""

    active_session_id: Optional[str]
    _memory_session_id: Optional[str]


class RuntimeIdentity:
    """Own current root identity or an explicitly bound immutable member identity.

    Roots receive an explicit ephemeral session at assembly, including when memory
    is disabled. Session selection belongs here; consumers resolve at operation
    boundaries so resume cannot leave a constructor-time snapshot behind.
    """

    def __init__(self, state: RuntimeSessionState) -> None:
        self._state = state
        self._initial = AgentRuntimeContext(
            agent_id="root_agent",
            display_name="Root Agent",
            role="supervisor",
            session_id=f"session_{uuid4().hex}",
        )
        self._context = self._initial
        self._member: Optional[AgentRuntimeContext] = None

    def resolve(self) -> AgentRuntimeContext:
        active_session_id = self._state.active_session_id
        memory_session_id = self._state._memory_session_id
        if self._member is not None:
            if active_session_id is not None and active_session_id != self._member.session_id:
                raise ValueError("Bound member session identity cannot be replaced")
            return self._member
        session = active_session_id if active_session_id is not None else memory_session_id
        if session is None:
            session = self._initial.session_id
        if not isinstance(session, str) or not session.strip():
            raise ValueError("Runtime identity requires a nonempty session_id")
        if self._context.session_id != session:
            self._context = replace(self._initial, session_id=session)
        return self._context

    def validate_session_transition(self, session_id: str) -> None:
        """Reject invalid/member-crossing recovery before hydration mutates state."""
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("Runtime identity requires a nonempty session_id")
        if self._member is not None and session_id != self._member.session_id:
            raise ValueError("Bound member session identity cannot be replaced")

    def bind_member(self, context: AgentRuntimeContext) -> None:
        self._member = require_runtime_context(lambda: context)
        self._state.active_session_id = context.session_id


@contextmanager
def bind_runtime_session(context: AgentRuntimeContext) -> Iterator[None]:
    """Bind a resolved identity to transport; never hold this scope across a yield."""
    from victor.core.context import session_id

    token = session_id.set(context.session_id)
    try:
        yield
    finally:
        session_id.reset(token)
