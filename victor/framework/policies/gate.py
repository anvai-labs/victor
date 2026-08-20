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

"""Message-phase adapter exposing the :class:`PolicyEngine` to the turn path.

The tool pipeline gates tool calls via :class:`~victor.framework.policies.
middleware.PolicyEngineMiddleware`. Message phases (REQUEST/RESPONSE) are not
tool-scoped and have no middleware-chain seam, so this thin adapter gates the
user message before the LLM call and the assistant output after it.

Unlike the tool middleware (whose ``MiddlewareResult`` is bool-only), a message
gate returns a small :class:`GateResult` carrying the possibly-redacted text so
the caller can substitute it. ASK is resolved synchronously via the shared
:func:`~victor.framework.policies.middleware.resolve_policy_ask` helper.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from victor.framework.policies.engine import PolicyEngine
from victor.framework.policies.middleware import ContextProvider, resolve_policy_ask
from victor.framework.policies.types import (
    UNSET,
    Phase,
    PolicyContext,
    PolicyEvent,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GateResult:
    """Outcome of gating a message through the policy engine.

    Attributes:
        allowed: Whether the message may proceed (False on DENY or a declined
            ASK — the caller should substitute a refusal).
        content: The text to use going forward: the original text when allowed
            and untouched, the redacted text when a policy modified it, or a
            refusal/empty string when blocked (the caller decides the wording).
        reason: Human-readable explanation, surfaced on block.
        blocked_by: Name of the deciding policy when blocked (audit/UX).
    """

    allowed: bool
    content: str
    reason: str = ""
    blocked_by: str = ""


class MessagePolicyGate:
    """Gates REQUEST/RESPONSE message text through a :class:`PolicyEngine`."""

    def __init__(
        self,
        engine: PolicyEngine,
        context_provider: Optional[ContextProvider] = None,
        *,
        approval_handler: Optional[Any] = None,
        ask_fallback: str = "deny",
        ask_timeout_seconds: int = 300,
    ) -> None:
        """Initialize the gate.

        Args:
            engine: The composed policy engine (shared with tool gating or a
                dedicated message-phase engine).
            context_provider: Callable returning a fresh :class:`PolicyContext`
                per evaluation. If None, an empty context is used.
            approval_handler: Optional async HITL approval handler for ASK.
            ask_fallback: ``"deny"`` (default, fail safe) or ``"allow"`` —
                outcome of an ASK when no approval handler is configured.
            ask_timeout_seconds: Timeout passed to the approval request.
        """
        self._engine = engine
        self._context_provider = context_provider
        self._approval_handler = approval_handler
        self._ask_fallback = ask_fallback.lower().strip()
        self._ask_timeout_seconds = ask_timeout_seconds

    async def gate_request(self, text: str) -> GateResult:
        """Gate the user message before it reaches the LLM (REQUEST phase)."""
        return await self._gate(Phase.REQUEST, text, "user message")

    async def gate_response(self, text: str) -> GateResult:
        """Gate the assistant's final output after the LLM (RESPONSE phase)."""
        return await self._gate(Phase.RESPONSE, text, "assistant response")

    # -- internals ----------------------------------------------------------

    async def _gate(self, phase: Phase, text: str, label: str) -> GateResult:
        event = PolicyEvent(
            phase=phase,
            tool_name="",
            content=text,
            context=self._safe_context(),
        )
        verdict = await self._engine.evaluate(event)

        if verdict.is_deny:
            return GateResult(
                allowed=False,
                content="",
                reason=verdict.reason or f"Blocked {label} by policy.",
                blocked_by=verdict.policy_name,
            )

        # Redaction (from any ALLOW/ASK verdict) is applied regardless of ASK.
        new_text = verdict.modified_content if verdict.modified_content is not UNSET else text

        if verdict.is_ask:
            approved = await resolve_policy_ask(
                self._approval_handler,
                ask_fallback=self._ask_fallback,
                ask_timeout_seconds=self._ask_timeout_seconds,
                title=f"Approve {label}",
                description=verdict.reason or f"Policy requests approval for this {label}.",
                context={"phase": phase.value, "policy": verdict.policy_name},
            )
            if not approved:
                return GateResult(
                    allowed=False,
                    content="",
                    reason=verdict.reason or f"Approval declined for {label}.",
                    blocked_by=verdict.policy_name,
                )

        return GateResult(allowed=True, content=new_text)

    def _safe_context(self) -> PolicyContext:
        """Resolve the session snapshot, degrading to empty on any failure."""
        if self._context_provider is None:
            return PolicyContext()
        try:
            return self._context_provider()
        except Exception:  # pragma: no cover - provider must not break the gate
            logger.debug("Policy context provider failed; using empty context", exc_info=True)
            return PolicyContext()


__all__ = ["MessagePolicyGate", "GateResult"]
