# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
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

"""Core data types for the governance policy engine.

A :class:`Policy` evaluates a :class:`PolicyEvent` (a phase + the tool/result
under inspection plus a :class:`PolicyContext` snapshot) and returns a
:class:`PolicyVerdict` (ALLOW / DENY / ASK). The :class:`~victor.framework.
policies.engine.PolicyEngine` composes an ordered list of policies into a
single verdict.

These types are intentionally framework-internal and dependency-free so the
engine can be unit-tested without booting services.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

# Sentinel marking "no result replacement requested" so that a policy can
# legitimately replace a tool result with ``None`` and still be distinguished
# from "left the result untouched".
UNSET: Any = object()


class Phase(str, Enum):
    """Lifecycle points where policies may intercept agent actions.

    Tool phases gate individual tool calls (wired via the middleware chain);
    message phases gate the user message before the LLM call (REQUEST) and the
    assistant's final output after it (RESPONSE), wired at the turn boundary.
    The enum stays extensible (further LLM/streaming phases can be added without
    changing the engine contract).
    """

    REQUEST = "request"
    RESPONSE = "response"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"


class PolicyAction(str, Enum):
    """The three verdicts a policy can return."""

    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"


@dataclass(frozen=True)
class PolicyContext:
    """Immutable snapshot of session state available to a policy.

    Populated by the middleware's ``context_provider`` at evaluation time.
    Defaults are chosen so that an absent provider degrades gracefully
    (e.g. ``cost_usd=0.0`` means a cost budget simply never trips).
    """

    session_id: str = ""
    usage: Dict[str, Any] = field(default_factory=dict)
    cost_usd: float = 0.0
    model: Optional[str] = None
    labels: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PolicyEvent:
    """The unit a policy evaluates.

    Attributes:
        phase: Lifecycle point (TOOL_CALL before execution, TOOL_RESULT after,
            REQUEST before the LLM call, RESPONSE after it).
        tool_name: Name of the tool being called. Empty for message phases
            (REQUEST/RESPONSE), which are not tool-scoped.
        arguments: Tool arguments (the engine threads modifications between
            policies, so a policy sees prior policies' modifications).
        result: Tool result (only meaningful on TOOL_RESULT).
        content: Message text being gated (the user message on REQUEST, the
            assistant output on RESPONSE). The engine threads modifications
            between policies, mirroring ``arguments``/``result``.
        success: Whether the tool execution succeeded (TOOL_RESULT only).
        context: Session snapshot (cost, model, labels, usage).
    """

    phase: Phase
    tool_name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    result: Any = None
    content: Optional[str] = None
    success: bool = True
    context: PolicyContext = field(default_factory=PolicyContext)


@dataclass(frozen=True)
class PolicyVerdict:
    """The decision a policy returns for a :class:`PolicyEvent`.

    Use the :meth:`allow`, :meth:`deny`, and :meth:`ask` constructors rather
    than building instances directly.

    Attributes:
        action: ALLOW / DENY / ASK.
        reason: Human-readable explanation (surfaced on DENY/ASK).
        policy_name: Name of the deciding policy (for audit/elicitation).
        modified_arguments: Replacement tool arguments, or None to leave as-is.
        modified_result: Replacement tool result, or :data:`UNSET` to leave
            as-is (TOOL_RESULT redaction).
        modified_content: Replacement message text, or :data:`UNSET` to leave
            as-is (REQUEST/RESPONSE redaction).
    """

    action: PolicyAction
    reason: str = ""
    policy_name: str = ""
    modified_arguments: Optional[Dict[str, Any]] = None
    modified_result: Any = UNSET
    modified_content: Any = UNSET

    @classmethod
    def allow(
        cls,
        *,
        policy_name: str = "",
        modified_arguments: Optional[Dict[str, Any]] = None,
        modified_result: Any = UNSET,
        modified_content: Any = UNSET,
    ) -> "PolicyVerdict":
        """Allow the action (optionally modifying arguments, result, or content)."""
        return cls(
            action=PolicyAction.ALLOW,
            policy_name=policy_name,
            modified_arguments=modified_arguments,
            modified_result=modified_result,
            modified_content=modified_content,
        )

    @classmethod
    def deny(cls, reason: str, *, policy_name: str = "") -> "PolicyVerdict":
        """Block the action with a reason."""
        return cls(action=PolicyAction.DENY, reason=reason, policy_name=policy_name)

    @classmethod
    def ask(
        cls,
        reason: str,
        *,
        policy_name: str = "",
        modified_arguments: Optional[Dict[str, Any]] = None,
    ) -> "PolicyVerdict":
        """Request human approval before proceeding."""
        return cls(
            action=PolicyAction.ASK,
            reason=reason,
            policy_name=policy_name,
            modified_arguments=modified_arguments,
        )

    @property
    def is_allow(self) -> bool:
        """Whether the verdict allows the action."""
        return self.action is PolicyAction.ALLOW

    @property
    def is_deny(self) -> bool:
        """Whether the verdict blocks the action."""
        return self.action is PolicyAction.DENY

    @property
    def is_ask(self) -> bool:
        """Whether the verdict requires human approval."""
        return self.action is PolicyAction.ASK


__all__ = [
    "UNSET",
    "Phase",
    "PolicyAction",
    "PolicyContext",
    "PolicyEvent",
    "PolicyVerdict",
]
