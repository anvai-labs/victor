# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Context-scoped policy ASK approval handler for team members (ADR-023 pillar 2a).

A team member (SubAgent) runs its own :class:`AgentOrchestrator`, whose policy engine
middleware resolves the ASK approval handler from the shared DI container at build time.
To route a member's ASK to the session terminal modal *tagged with member_id* without
threading a parameter through the orchestrator/factory constructors (which would regrow the
decomposed orchestrator hotspot), the SubAgent publishes a member-tagging handler on this
``ContextVar`` around the synchronous member-orchestrator construction; the policy-engine
builder reads it and prefers it over the container-resolved handler.

Set-and-reset spans only the (synchronous) `AgentOrchestrator(...)` call — the middleware
captures the handler at build time, so the var is `None` again before the member ever runs.
Absent a value (every top-level agent, every non-team build) behavior is byte-identical.
"""

from __future__ import annotations

import contextvars
from typing import TYPE_CHECKING, Any, Optional

from victor.framework.approval_pause import ApprovalPause

if TYPE_CHECKING:
    from victor.framework.hitl import ApprovalRequest

#: The member-tagging ASK approval handler in effect while a member orchestrator is being
#: constructed. ``None`` everywhere else (zero overhead on the top-level/single-agent path).
current_member_approval_handler: "contextvars.ContextVar[Optional[Any]]" = contextvars.ContextVar(
    "current_member_approval_handler", default=None
)


#: ADR-023 pillar 2b: durable pause is armed (a checkpointer + thread_id are configured, i.e. the
#: coordinator set ``TeamContext.pause_hook``) while a team member executes. When set, a member's
#: policy ``ASK`` raises :class:`MemberApprovalPause` — durably pausing the team — instead of blocking
#: on the terminal modal (slice 2a). ``None`` everywhere else: non-team single-agent runs, and team
#: runs without a checkpointer, keep slice-2a inline approval (byte-identical).
current_member_durable_pause_enabled: "contextvars.ContextVar[Optional[bool]]" = (
    contextvars.ContextVar("current_member_durable_pause_enabled", default=None)
)


class MemberApprovalPause(ApprovalPause):
    """Control-flow signal: a durable team member paused awaiting human approval (ADR-023 2b).

    A subclass of :class:`~victor.agent.approval_pause.ApprovalPause` (FEP-0029) — the shared
    ``BaseException`` pause signal — so it rides *through* every ``except Exception:`` on the
    policy-ASK → tool-pipeline → orchestrator → AgenticLoop → ``SubAgent._execute_with_retry`` path
    untouched (audited: no ``except BaseException`` on that path), and is caught only at
    :meth:`SubAgent.execute`, which converts it into an ``awaiting_approval`` member result. Team
    code that catches ``MemberApprovalPause`` still catches exactly the member variant; the
    single-agent turn boundary catches the base ``ApprovalPause``. Distinct from
    :class:`asyncio.CancelledError`.
    """

    def __init__(self, request: "ApprovalRequest") -> None:
        super().__init__(request)
        # Preserve the member-specific message for existing logs/tests.
        title = getattr(request, "title", "")
        self.args = (f"Member paused awaiting approval: {title}",)
