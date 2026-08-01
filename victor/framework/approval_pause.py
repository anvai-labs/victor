# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
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

"""Durable approval-pause control-flow signal, shared by single-agent and team runs (FEP-0029).

When durable pause is armed, a policy ``ASK`` raises :class:`ApprovalPause` instead of blocking on
the terminal modal, so the run can park and resume later (headless/API/deferred approval). This is
the single-agent generalization of ADR-023's team member pause; :class:`MemberApprovalPause`
(``victor.agent.member_approval_context``) is a subclass so team behavior is unchanged.

``ApprovalPause`` is deliberately a :class:`BaseException` (not :class:`Exception`) so it rides
*through* every ``except Exception:`` on the policy-ASK → tool-pipeline → orchestrator → AgenticLoop
path untouched (audited: no ``except BaseException``/bare ``except:`` on that path), and is caught
only at the turn boundary (``victor.framework.message_execution``), which converts it into an
``awaiting_approval`` result. Distinct from :class:`asyncio.CancelledError`.
"""

from __future__ import annotations

import contextvars
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from victor.framework.hitl import ApprovalRequest


@dataclass
class ApprovalDecision:
    """A human's decision on a durably-paused approval request (FEP-0029 resume).

    Passed to ``VictorClient.resume(run_id, decision)``: ``approved=True`` replays the persisted
    gated tool call; ``approved=False`` skips it with a tool-error result. ``response`` is the
    human's note and ``responder`` their identity (both surfaced to the model / audit).
    """

    approved: bool
    response: Optional[str] = None
    responder: Optional[str] = None


#: ADR-028 / FEP-0029: durable pause is armed for the duration of a single-agent turn (a
#: checkpointer-free opt-in via ``SessionConfig.tool_approval.durable`` → ``governance.durable``).
#: While set, the policy ``ASK`` handler raises :class:`ApprovalPause` — durably pausing the run —
#: instead of blocking on the terminal modal (ADR-021). ``False`` everywhere else: a normal
#: interactive turn keeps the inline modal (byte-identical). The team path uses its own peer var
#: (``current_member_durable_pause_enabled``); both raise the same signal family.
current_durable_pause_enabled: "contextvars.ContextVar[bool]" = contextvars.ContextVar(
    "current_durable_pause_enabled", default=False
)


class ApprovalPause(BaseException):
    """Control-flow signal: a durable run paused awaiting human approval (FEP-0029).

    Carries the pending :class:`~victor.framework.hitl.ApprovalRequest`. Caught at the turn
    boundary and converted into an ``awaiting_approval`` result + a resumable ``run_id``.
    """

    def __init__(self, request: "ApprovalRequest") -> None:
        self.request = request
        title = getattr(request, "title", "")
        super().__init__(f"Run paused awaiting approval: {title}")
