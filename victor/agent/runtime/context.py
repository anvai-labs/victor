"""Runtime identity context for root agents and subagents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


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
