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
from typing import Any, Optional

#: The member-tagging ASK approval handler in effect while a member orchestrator is being
#: constructed. ``None`` everywhere else (zero overhead on the top-level/single-agent path).
current_member_approval_handler: "contextvars.ContextVar[Optional[Any]]" = contextvars.ContextVar(
    "current_member_approval_handler", default=None
)
