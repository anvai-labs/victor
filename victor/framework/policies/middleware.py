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

"""Adapter exposing the :class:`PolicyEngine` to the tool execution pipeline.

Victor's tool pipeline already runs a :class:`MiddlewareChain` with
``before_tool_call`` / ``after_tool_call`` hooks (see
``victor/agent/middleware_chain.py``). This adapter implements that protocol
so the policy engine plugs in with no pipeline changes, at ``CRITICAL``
priority so it runs before other middleware.

Because :class:`MiddlewareResult` only carries a boolean ``proceed``, the
three-way ALLOW/DENY/ASK verdict is collapsed here: an ASK is resolved
*synchronously inside* ``before_tool_call`` by awaiting a human approval via
:class:`~victor.framework.hitl.HITLController`. If no approval handler is
configured, the ASK falls back per ``ask_fallback`` (default ``"deny"`` — fail
safe) rather than silently proceeding.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional, Set

from victor.core.verticals.protocols import (
    MiddlewarePriority,
    MiddlewareProtocol,
    MiddlewareResult,
)
from victor.framework.policies.engine import PolicyEngine
from victor.framework.policies.types import (
    UNSET,
    Phase,
    PolicyContext,
    PolicyEvent,
    PolicyVerdict,
)

logger = logging.getLogger(__name__)

# Returns a fresh snapshot of session state for each evaluation.
ContextProvider = Callable[[], PolicyContext]


async def resolve_policy_ask(
    approval_handler: Optional[Any],
    *,
    ask_fallback: str,
    ask_timeout_seconds: int,
    title: str,
    description: str,
    context: Dict[str, Any],
) -> bool:
    """Resolve an ASK verdict to a boolean approval decision.

    Shared by :class:`PolicyEngineMiddleware` (tool gating) and
    :class:`~victor.framework.policies.gate.MessagePolicyGate` (message gating)
    so both honour the same HITL lifecycle and fail-safe fallback.

    Args:
        approval_handler: Async HITL ``ApprovalHandler`` or None. When None, the
            decision falls back to ``ask_fallback``.
        ask_fallback: ``"allow"`` or ``"deny"`` (fail safe) — outcome when no
            handler is configured or the handler errors.
        ask_timeout_seconds: Timeout passed to the approval request.
        title: Short approval title (e.g. ``"Approve tool: run_command"``).
        description: Longer human-readable reason.
        context: Structured context attached to the approval request.

    Returns:
        True to proceed, False to block.
    """
    if approval_handler is None:
        logger.warning(
            "Policy requested approval (%s) but no approval handler is "
            "configured; applying ask_fallback=%s",
            title,
            ask_fallback,
        )
        return ask_fallback == "allow"

    # Reuse the existing HITL machinery for the request lifecycle.
    from victor.framework.hitl import HITLController

    controller = HITLController(approval_handler=approval_handler)
    request = controller.request_approval(
        title=title,
        description=description,
        context=context,
        timeout_seconds=ask_timeout_seconds,
    )
    try:
        resolved = await controller.process_approval(request.id)
    except Exception:  # pragma: no cover - defensive
        logger.exception("Approval handler failed for '%s'; failing safe", title)
        return ask_fallback == "allow"
    return resolved.is_approved


class PolicyEngineMiddleware(MiddlewareProtocol):
    """Bridges :class:`PolicyEngine` verdicts onto the middleware protocol."""

    def __init__(
        self,
        engine: PolicyEngine,
        context_provider: Optional[ContextProvider] = None,
        *,
        approval_handler: Optional[Any] = None,
        ask_fallback: str = "deny",
        ask_timeout_seconds: int = 300,
    ) -> None:
        """Initialize the middleware.

        Args:
            engine: The composed policy engine.
            context_provider: Callable returning a fresh :class:`PolicyContext`
                per evaluation (cost, model, labels). If None, an empty context
                is used (cost budgets never trip).
            approval_handler: Optional async HITL approval handler
                (``ApprovalHandler``) used to resolve ASK verdicts. If None,
                ASK resolves via ``ask_fallback``.
            ask_fallback: ``"deny"`` (default, fail safe) or ``"allow"`` —
                outcome of an ASK when no approval handler is configured.
            ask_timeout_seconds: Timeout passed to the approval request.
        """
        self._engine = engine
        self._context_provider = context_provider
        self._approval_handler = approval_handler
        self._ask_fallback = ask_fallback.lower().strip()
        self._ask_timeout_seconds = ask_timeout_seconds

    # -- MiddlewareProtocol -------------------------------------------------

    def get_priority(self) -> MiddlewarePriority:
        """Run first — governance gates before any other middleware."""
        return MiddlewarePriority.CRITICAL

    def get_applicable_tools(self) -> Optional[Set[str]]:
        """Apply to all tools; per-policy scoping happens inside the engine."""
        return None

    async def before_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> MiddlewareResult:
        """Gate a tool call through the policy engine (TOOL_CALL phase)."""
        event = PolicyEvent(
            phase=Phase.TOOL_CALL,
            tool_name=tool_name,
            arguments=arguments,
            context=self._safe_context(),
        )
        verdict = await self._engine.evaluate(event)

        if verdict.is_deny:
            return MiddlewareResult(
                proceed=False,
                error_message=self._block_message(tool_name, verdict, asked=False),
            )

        if verdict.is_ask:
            approved = await self._resolve_ask(tool_name, verdict, arguments)
            if not approved:
                return MiddlewareResult(
                    proceed=False,
                    error_message=self._block_message(tool_name, verdict, asked=True),
                )

        # ALLOW, or an approved ASK: proceed, applying any argument modifications.
        return MiddlewareResult(proceed=True, modified_arguments=verdict.modified_arguments)

    async def after_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        result: Any,
        success: bool,
    ) -> Optional[Any]:
        """Inspect/transform a tool result (TOOL_RESULT phase).

        Returns the (possibly redacted) result, or None to leave it unchanged
        per the middleware-chain contract.
        """
        event = PolicyEvent(
            phase=Phase.TOOL_RESULT,
            tool_name=tool_name,
            arguments=arguments,
            result=result,
            success=success,
            context=self._safe_context(),
        )
        verdict = await self._engine.evaluate(event)
        if verdict.modified_result is not UNSET:
            return verdict.modified_result
        return None

    # -- internals ----------------------------------------------------------

    def _safe_context(self) -> PolicyContext:
        """Resolve the session snapshot, degrading to empty on any failure."""
        if self._context_provider is None:
            return PolicyContext()
        try:
            return self._context_provider()
        except Exception:  # pragma: no cover - provider must not break the gate
            logger.debug("Policy context provider failed; using empty context", exc_info=True)
            return PolicyContext()

    async def _resolve_ask(
        self,
        tool_name: str,
        verdict: PolicyVerdict,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Resolve an ASK verdict to a boolean approval decision.

        Includes the tool ``arguments`` in the approval context so surfaces can show the
        exact command/diff being approved (informed approval) rather than just the name.
        """
        return await resolve_policy_ask(
            self._approval_handler,
            ask_fallback=self._ask_fallback,
            ask_timeout_seconds=self._ask_timeout_seconds,
            title=f"Approve tool: {tool_name}",
            description=verdict.reason or f"Policy requests approval to run '{tool_name}'.",
            context={
                "tool_name": tool_name,
                "policy": verdict.policy_name,
                "arguments": arguments or {},
            },
        )

    @staticmethod
    def _block_message(tool_name: str, verdict: PolicyVerdict, *, asked: bool) -> str:
        """Build the user-facing block message."""
        prefix = "Approval declined" if asked else "Blocked by policy"
        policy = f" [{verdict.policy_name}]" if verdict.policy_name else ""
        reason = f": {verdict.reason}" if verdict.reason else ""
        return f"{prefix}{policy} for '{tool_name}'{reason}"


__all__ = ["PolicyEngineMiddleware", "ContextProvider", "resolve_policy_ask"]
