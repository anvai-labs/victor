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

"""Approval handlers for resolving policy ASK verdicts.

An approval handler is the standard HITL ``ApprovalHandler`` callable
(``victor.framework.hitl``): ``async (ApprovalRequest) -> (status, response,
responder)``. The :class:`~victor.framework.policies.middleware.
PolicyEngineMiddleware` uses one to turn an ASK verdict into an approve/reject
decision.

This module provides an interactive console handler for CLI sessions. Library
and server callers should supply their own handler (e.g. one that elicits over
a websocket) rather than the console one.
"""

from __future__ import annotations

import logging
from typing import Awaitable, Callable, Optional, Tuple

from victor.framework.hitl import ApprovalRequest, ApprovalStatus

logger = logging.getLogger(__name__)

# (approved: bool) decision function, injectable for testing.
ConfirmFn = Callable[[str], bool]

_ApprovalResult = Tuple[ApprovalStatus, Optional[str], Optional[str]]
ApprovalHandlerFn = Callable[[ApprovalRequest], Awaitable[_ApprovalResult]]


def _default_confirm(prompt: str) -> bool:
    """Prompt the user on the console, returning True if approved.

    Uses Rich's ``Confirm`` when available, falling back to ``input()``.
    Defaults to ``False`` (reject) on any error or non-interactive input — a
    governance prompt should fail safe.
    """
    try:
        from rich.prompt import Confirm

        return bool(Confirm.ask(prompt, default=False))
    except Exception:
        pass
    try:
        answer = input(f"{prompt} [y/N] ").strip().lower()
        return answer in {"y", "yes"}
    except (EOFError, KeyboardInterrupt, Exception):  # pragma: no cover - defensive
        return False


def make_console_approval_handler(
    confirm_fn: Optional[ConfirmFn] = None,
) -> ApprovalHandlerFn:
    """Build an interactive console approval handler.

    Args:
        confirm_fn: Optional ``(prompt) -> bool`` used to ask the user. Defaults
            to a Rich/``input()``-based prompt. Injectable for testing.

    Returns:
        An async ``ApprovalHandler`` suitable for HITLController / the policy
        middleware.

    Note:
        The confirm call is synchronous and blocks until the user answers; that
        is intentional — nothing should proceed while awaiting approval.
    """
    confirm = confirm_fn or _default_confirm

    async def handler(request: ApprovalRequest) -> _ApprovalResult:
        prompt = request.title
        if request.description:
            prompt = f"{request.title} — {request.description}"
        try:
            approved = confirm(prompt)
        except Exception:  # pragma: no cover - defensive: fail safe on prompt error
            logger.debug("Console approval prompt failed; rejecting", exc_info=True)
            approved = False
        status = ApprovalStatus.APPROVED if approved else ApprovalStatus.REJECTED
        return status, "console decision", "console_user"

    return handler


#: Default interactive console approval handler.
console_approval_handler: ApprovalHandlerFn = make_console_approval_handler()


class PolicyApprovalHandler:
    """Container-registerable holder for a policy ASK approval handler.

    The console handler is TTY-only. Non-interactive surfaces (HTTP API, TUI,
    websocket) that want ASK verdicts resolved interactively register one of
    these in the DI container::

        from victor.framework.policies import PolicyApprovalHandler
        container.register(PolicyApprovalHandler, PolicyApprovalHandler(my_handler))

    The policy wiring then resolves it (taking precedence over the TTY console
    handler) so ASK works on any surface that supplies its own elicitation.
    """

    def __init__(self, handler: ApprovalHandlerFn) -> None:
        """Wrap an async ``ApprovalHandler`` callable."""
        self.handler = handler


def register_policy_approval_handler(
    handler: ApprovalHandlerFn, container: Optional[object] = None
) -> bool:
    """Register *handler* as the DI container's policy ASK approval handler.

    This is the framework seam any surface (web chat, TUI, API) uses to answer ASK
    verdicts without reaching into the container itself. Must be called before the agent
    builds its middleware (i.e. before the first turn), while the container is mutable.

    Idempotent and defensive: returns ``True`` when a handler is registered (or already
    present), ``False`` when registration isn't possible (e.g. the container is frozen and
    none is registered yet). When *container* is ``None`` the global container is used.
    """
    if container is None:
        from victor.core import get_container

        container = get_container()

    try:
        if container.get_optional(PolicyApprovalHandler) is not None:
            return True  # already registered (e.g. a prior session in this process)
    except Exception:
        pass

    try:
        # Register a pre-built instance — NOT register(), whose second argument is a
        # factory Callable[[container], T]. PolicyApprovalHandler is not callable, so
        # register() would store it as a factory and raise TypeError on resolution (then
        # get swallowed to None), silently defeating the handler. register_instance binds
        # the instance directly.
        container.register_instance(PolicyApprovalHandler, PolicyApprovalHandler(handler))
        return True
    except Exception:
        logger.debug("Could not register policy approval handler", exc_info=True)
        return False


__all__ = [
    "ConfirmFn",
    "ApprovalHandlerFn",
    "make_console_approval_handler",
    "console_approval_handler",
    "PolicyApprovalHandler",
    "register_policy_approval_handler",
]
