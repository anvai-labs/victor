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

"""Bound single-action durable resume through the canonical execution runtime.

Missing results do not prove a sibling never executed. Ambiguous batches and
legacy unbound approvals require reconciliation/new approval, never guessed replay.
The store's single-use claim is not a durable external-effect receipt (G62).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_MAX_CONTINUATION_TURNS = 10


@dataclass
class ResumeResult:
    """Outcome of replaying a paused turn (FEP-0029)."""

    final_content: str
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    approved: bool = True
    gated_tool: Optional[str] = None
    continuation_turns: int = 0
    executed_siblings: int = 0
    # Chained pause (FEP-0029): a *new* ASK fired during the continuation → the run parked again.
    # ``awaiting_run_id`` is the fresh resume token; ``awaiting_approval_request`` its request dict.
    awaiting_run_id: Optional[str] = None
    awaiting_approval_request: Optional[Dict[str, Any]] = None


class ResumeError(RuntimeError):
    """A paused run could not be replayed (no gated call found, missing runtime surface, …)."""


def _tool_call_id(tc: Dict[str, Any]) -> Optional[str]:
    return tc.get("id") or tc.get("tool_call_id")


def _msg_attr(msg: Any, name: str) -> Any:
    """Read a field off a conversation Message, looking through its metadata."""
    value = getattr(msg, name, None)
    if value is None:
        meta = getattr(msg, "metadata", None) or {}
        value = meta.get(name)
    return value


def _find_unresolved_calls(messages: List[Any]) -> List[Tuple[Dict[str, Any], str]]:
    """Return every ``(tool_call, id)`` in the last assistant message that has no result yet.

    Transcript absence is not evidence of nonexecution. The caller rejects ambiguous
    unresolved batches. Raises if none is found or an ID is missing/duplicated.
    """
    resolved_ids = {
        _msg_attr(m, "tool_call_id") for m in messages if getattr(m, "role", None) == "tool"
    }
    resolved_ids.discard(None)

    assistant_calls: List[Dict[str, Any]] = []
    for m in reversed(messages):
        if getattr(m, "role", None) == "assistant":
            calls = _msg_attr(m, "tool_calls")
            if calls:
                assistant_calls = list(calls)
                break

    ids = [_tool_call_id(tc) for tc in assistant_calls]
    if len(ids) != len(set(ids)):
        raise ResumeError("duplicate tool_call IDs cannot be approved")
    unresolved: List[Tuple[Dict[str, Any], str]] = []
    for tc in assistant_calls:
        tc_id = _tool_call_id(tc)
        if tc_id in resolved_ids:
            continue
        if not tc_id:
            raise ResumeError("an unresolved tool_call has no id; cannot link its result")
        unresolved.append((tc, tc_id))

    if not unresolved:
        raise ResumeError("no unresolved gated tool_call found in the paused conversation")
    return unresolved


def _last_user_message(messages: List[Any]) -> str:
    for m in reversed(messages):
        if getattr(m, "role", None) == "user":
            return str(getattr(m, "content", "") or "")
    return ""


async def resume_paused_run(orchestrator: Any, paused_run: Any, decision: Any) -> ResumeResult:
    """Resolve one bound pending call; never replay unresolved siblings."""
    from copy import deepcopy
    import math
    import time

    from victor.framework.approval_binding import (
        ApprovalBindingError,
        ApprovalGrant,
        current_approval_grant,
        digest,
        proposal,
    )

    controller = getattr(orchestrator, "_conversation_controller", None)
    runtime_factory = getattr(orchestrator, "_get_tool_execution_runtime", None)
    turn_executor = getattr(orchestrator, "turn_executor", None)
    if controller is None or not callable(runtime_factory) or turn_executor is None:
        raise ResumeError("orchestrator is missing the canonical conversation/tool/turn runtime")
    approved = getattr(decision, "approved", None)
    if type(approved) is not bool:
        raise ResumeError("approval decision must be a boolean")
    session_id = getattr(paused_run, "session_id", None)
    if not session_id or getattr(orchestrator, "active_session_id", None) != session_id:
        raise ResumeError("approval requires the restored original session")
    messages = list(controller.messages)
    unresolved = _find_unresolved_calls(messages)
    if len(unresolved) != 1:
        raise ResumeError("unresolved siblings require reconciliation before approval resume")
    tc, tc_id = unresolved[0]
    try:
        request = paused_run.approval_request
        pending = paused_run.pending_tool
        binding = deepcopy(pending["binding"])
        ctx = request["context"]
        if ctx.get("member_id") is not None or ctx.get("member_role") is not None:
            raise ApprovalBindingError("Member approval requires member-owned resume")
        timeout = request["timeout_seconds"]
        created = request["created_at"]
        if (
            type(timeout) not in (int, float)
            or type(created) not in (int, float)
            or not math.isfinite(timeout)
            or not math.isfinite(created)
            or timeout <= 0
            or created <= 0
            or created > time.time()
        ):
            raise ApprovalBindingError("Invalid approval lifetime")
        expected = {
            **ctx["action_binding"],
            "session_id": session_id,
            "agent_id": getattr(paused_run, "agent_id", None),
            "request_id": request["id"],
            "expires_at": created + timeout,
        }
        call = proposal(tc)
        if (
            binding != expected
            or type(binding["version"]) is not int
            or binding["version"] != 1
            or not request["id"]
            or binding["call_id"] != tc_id
            or binding["proposal"] != digest(call)
            or pending["tool_name"] != binding["tool_name"]
            or ctx["tool_name"] != binding["tool_name"]
            or digest(pending["arguments"]) != binding["payload"]
            or digest(ctx["arguments"]) != binding["payload"]
            or time.time() >= binding["expires_at"]
        ):
            raise ApprovalBindingError("Approval payload, identity or expiry changed")
    except (KeyError, TypeError, ValueError, AttributeError, ApprovalBindingError) as exc:
        raise ResumeError(
            "Approval is unbound, changed or expired; request a new approval"
        ) from exc

    gated_tool = binding["tool_name"]
    sibling_count = 0
    if approved:
        grant = ApprovalGrant(
            binding, binding["expires_at"], lambda: orchestrator.active_session_id == session_id
        )
        token = current_approval_grant.set(grant)
        try:
            await runtime_factory().execute_tool_calls([call])
        finally:
            current_approval_grant.reset(token)
        if not grant.dispatched:
            raise ResumeError("Approved action did not pass current policy and dispatch checks")
        results = [
            m
            for m in controller.messages
            if getattr(m, "role", None) == "tool" and _msg_attr(m, "tool_call_id") == tc_id
        ]
        if len(results) != 1:
            raise ResumeError(
                "Dispatched action has no unique recorded result; reconcile without replay"
            )
    else:
        note = getattr(decision, "response", None) or ""
        controller.add_tool_result(tc_id, f"Tool call rejected by human approval: {note}".strip())

    # Continuation: drive the turn primitive (adds no user message) until the model stops calling
    # tools. Durable pause is ARMED here (FEP-0029 chained pauses): a *new* ASK during the
    # continuation raises ApprovalPause (the Phase-1 mechanism), which we catch and record as a fresh
    # paused_run — the run parks again with a new run_id that the surfaces already render.
    from victor.framework.approval_pause import ApprovalPause, current_durable_pause_enabled

    tool_name = gated_tool or "unknown"
    user_message = _last_user_message(messages)
    final_content = ""
    final_tool_calls: List[Dict[str, Any]] = []
    turns = 0
    awaiting_run_id: Optional[str] = None
    awaiting_request: Optional[Dict[str, Any]] = None

    _token = current_durable_pause_enabled.set(True)
    try:
        for _ in range(_MAX_CONTINUATION_TURNS):
            try:
                turn = await turn_executor.execute_turn(user_message)
            except ApprovalPause as pause:
                # Chained pause: a further ASK fired mid-continuation — park again.
                import time

                from victor.agent.paused_run_store import record_pause_from_approval

                awaiting_run_id, awaiting_request = record_pause_from_approval(
                    getattr(pause, "request", None),
                    session_id=getattr(paused_run, "session_id", None),
                    agent_id=getattr(paused_run, "agent_id", None),
                    created_at=time.time(),
                    metadata={"chained_from": getattr(paused_run, "run_id", None)},
                )
                break
            turns += 1
            response = getattr(turn, "response", None)
            final_content = str(getattr(response, "content", "") or "")
            final_tool_calls = list(getattr(response, "tool_calls", None) or [])
            if not getattr(turn, "has_tool_calls", False):
                break
    finally:
        current_durable_pause_enabled.reset(_token)

    return ResumeResult(
        final_content=final_content,
        tool_calls=final_tool_calls,
        approved=approved,
        gated_tool=tool_name,
        continuation_turns=turns,
        executed_siblings=sibling_count,
        awaiting_run_id=awaiting_run_id,
        awaiting_approval_request=awaiting_request,
    )
