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

"""Human-in-the-loop tool approval for the Chainlit chat UI.

When ``SessionConfig.tool_approval.enabled`` is set, the governance policy engine routes
the configured tools through the ASK path and calls a ``PolicyApprovalHandler`` registered
in the DI container. This module provides that handler for the web UI: it renders a
Chainlit ``AskActionMessage`` (Approve / Reject) and maps the click back to an
``ApprovalStatus``.

The decision mapping (:func:`decision_from_action`) is Chainlit-free so it can be unit
tested without the optional ``chat-ui`` extra installed; only :func:`chainlit_approval_handler`
imports ``chainlit`` (lazily, inside the coroutine).
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

from victor.framework.hitl import ApprovalRequest, ApprovalStatus
from victor.ui.rendering.markdown_presenters import tool_call_summary

logger = logging.getLogger(__name__)

_RESPONDER = "chat_ui_user"

# The handler returns the framework's ApprovalHandler tuple:
# (status, response_message, responder_identity).
ApprovalResult = Tuple[ApprovalStatus, Optional[str], Optional[str]]

_APPROVE_VALUE = "approve"
_REJECT_VALUE = "reject"


def _decision_token(action: Any) -> Optional[str]:
    """Pull the approve/reject token from an AskActionMessage response, version-robustly.

    Chainlit 2.x returns the chosen action as a dict with ``name`` + ``payload`` (``value``
    was renamed to ``payload``); 1.x used a flat ``value``. Prefer the stable action
    ``name`` ("approve"/"reject"), then ``payload['value']``, then legacy ``value``.
    """
    if action is None:
        return None
    if hasattr(action, "get"):
        name = action.get("name")
        if name:
            return name
        payload = action.get("payload")
        if isinstance(payload, dict) and payload.get("value"):
            return payload.get("value")
        return action.get("value")
    return getattr(action, "name", None) or getattr(action, "value", None)


def decision_from_action(action: Optional[Any]) -> ApprovalResult:
    """Map a Chainlit AskActionMessage response (or ``None`` on timeout) to an approval result.

    Anything that isn't an explicit approve maps to a non-approval (fail-safe).
    """
    if action is None:
        return ApprovalStatus.TIMEOUT, "No response before timeout", _RESPONDER

    if _decision_token(action) == _APPROVE_VALUE:
        return ApprovalStatus.APPROVED, "Approved in chat UI", _RESPONDER
    return ApprovalStatus.REJECTED, "Rejected in chat UI", _RESPONDER


def _argument_preview(tool: str, args: Any) -> str:
    """Render the consequential argument (command / content / diff) as a fenced block.

    Informed approval: the user should see *what* will run, not just the tool name.
    """
    if not isinstance(args, dict) or not args:
        return ""
    name = (tool or "").lower()
    command = args.get("command") or args.get("cmd") or args.get("script")
    if command:
        return f"```bash\n{str(command)[:2000]}\n```"
    diff = args.get("diff") or args.get("patch")
    if diff:
        return f"```diff\n{str(diff)[:2000]}\n```"
    content = args.get("content") or args.get("text")
    if content and ("write" in name or "create" in name or "edit" in name):
        return f"```\n{str(content)[:2000]}\n```"
    # Generic tools: a compact one-line `tool(args)` summary.
    return f"`{tool_call_summary(tool, args)}`"


def _format_prompt(request: ApprovalRequest) -> str:
    """Build the approval prompt markdown — with the exact command/diff being approved."""
    ctx = request.context or {}
    tool = ctx.get("tool_name") or "tool"
    lines = [f"**Approve tool: `{tool}`?**"]
    if request.description:
        lines.append("")
        lines.append(request.description)
    preview = _argument_preview(tool, ctx.get("arguments"))
    if preview:
        lines.append("")
        lines.append(preview)
    return "\n".join(lines)


async def chainlit_approval_handler(request: ApprovalRequest) -> ApprovalResult:
    """Prompt the user to approve/reject a tool via Chainlit; return the decision.

    Registered in the DI container as a ``PolicyApprovalHandler`` so the policy engine
    invokes it for ASK verdicts. Blocks the tool call until the user answers (or the
    request times out).
    """
    import chainlit as cl

    try:
        action = await cl.AskActionMessage(
            content=_format_prompt(request),
            actions=[
                # Chainlit 2.x: payload (dict) replaced the 1.x `value` kwarg.
                cl.Action(
                    name="approve",
                    payload={"value": _APPROVE_VALUE},
                    label="✅ Approve",
                ),
                cl.Action(name="reject", payload={"value": _REJECT_VALUE}, label="🚫 Reject"),
            ],
            timeout=request.timeout_seconds or 120,
        ).send()
    except Exception:  # elicitation failed -> fail safe (reject)
        logger.exception("Chat UI approval prompt failed; rejecting")
        return ApprovalStatus.REJECTED, "Approval prompt failed", _RESPONDER

    return decision_from_action(action)
