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

"""Faithful replay of a durably-paused single-agent turn (FEP-0029 Phase 3a).

When a turn paused on a policy ASK, the assistant had already produced a message with ``tool_calls``;
one of those calls was gated and never ran. On resume with a human decision this module — *without
re-calling the LLM for that message* — executes (approve) or skips (reject) the **exact** persisted
gated call, appends its ``role=tool`` result to the rehydrated conversation, and drives one or more
continuation turns so the model sees the result and proceeds.

Scope (3a): a single gated tool per paused turn. Multi-tool batch partiality, streaming resume, and
chained pauses are deferred. Approved execution goes through ``ToolService.execute_tool`` (the raw
tool executor — it does *not* re-run the ASK middleware, since the human already decided); the
continuation reuses the turn executor's ``execute_turn`` primitive (which adds no user message), so
there is no spurious user turn.
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


class ResumeError(RuntimeError):
    """A paused run could not be replayed (no gated call found, missing runtime surface, …)."""


def _tool_call_id(tc: Dict[str, Any]) -> Optional[str]:
    return tc.get("id") or tc.get("tool_call_id")


def _tool_call_name(tc: Dict[str, Any]) -> Optional[str]:
    return tc.get("name") or (tc.get("function") or {}).get("name")


def _tool_call_args(tc: Dict[str, Any]) -> Any:
    return tc.get("arguments") if "arguments" in tc else (tc.get("function") or {}).get("arguments")


def _msg_attr(msg: Any, name: str) -> Any:
    """Read a field off a conversation Message, looking through its metadata."""
    value = getattr(msg, name, None)
    if value is None:
        meta = getattr(msg, "metadata", None) or {}
        value = meta.get(name)
    return value


def _find_gated_call(
    messages: List[Any], pending_tool: Optional[Dict[str, Any]]
) -> Tuple[Dict[str, Any], str]:
    """Locate the gated tool_call in the last assistant message that has no result yet.

    Returns ``(tool_call, tool_call_id)``. Raises :class:`ResumeError` if none is found. The gated
    call is the one whose ``id`` has no matching ``role=tool`` result; ``pending_tool`` (name/args
    recorded at pause time) disambiguates when several are unresolved.
    """
    # Collect tool_call_ids that already have a result.
    resolved_ids = {
        _msg_attr(m, "tool_call_id") for m in messages if getattr(m, "role", None) == "tool"
    }
    resolved_ids.discard(None)

    # Last assistant message carrying tool_calls.
    assistant_calls: List[Dict[str, Any]] = []
    for m in reversed(messages):
        if getattr(m, "role", None) == "assistant":
            calls = _msg_attr(m, "tool_calls")
            if calls:
                assistant_calls = list(calls)
                break

    unresolved = [tc for tc in assistant_calls if _tool_call_id(tc) not in resolved_ids]
    if not unresolved:
        raise ResumeError("no unresolved gated tool_call found in the paused conversation")

    chosen = unresolved[0]
    if pending_tool and len(unresolved) > 1:
        want = pending_tool.get("tool_name")
        chosen = next((tc for tc in unresolved if _tool_call_name(tc) == want), unresolved[0])

    tc_id = _tool_call_id(chosen)
    if not tc_id:
        raise ResumeError("gated tool_call has no id; cannot link its result")
    return chosen, tc_id


def _format_tool_result(result: Any) -> str:
    """Render a ToolService.execute_tool result into tool-message content."""
    if result is None:
        return ""
    for attr in ("output", "result", "content"):
        val = getattr(result, attr, None)
        if val is not None:
            return str(val)
    return str(result)


def _last_user_message(messages: List[Any]) -> str:
    for m in reversed(messages):
        if getattr(m, "role", None) == "user":
            return str(getattr(m, "content", "") or "")
    return ""


async def resume_paused_run(orchestrator: Any, paused_run: Any, decision: Any) -> ResumeResult:
    """Replay a paused turn's gated tool call under ``decision`` and continue (FEP-0029 Phase 3a).

    ``orchestrator`` must expose ``_conversation_controller`` (``.messages`` + ``.add_tool_result``),
    ``_tool_service`` (``.execute_tool``), and ``turn_executor`` (``.execute_turn``). Assumes the
    conversation was already rehydrated (``VictorClient.resume_session``).
    """
    controller = getattr(orchestrator, "_conversation_controller", None)
    tool_service = getattr(orchestrator, "_tool_service", None)
    turn_executor = getattr(orchestrator, "turn_executor", None)
    if controller is None or tool_service is None or turn_executor is None:
        raise ResumeError("orchestrator is missing the conversation/tool/turn runtime surface")

    messages = list(controller.messages)
    pending_tool = getattr(paused_run, "pending_tool", None)
    gated_call, tc_id = _find_gated_call(messages, pending_tool)
    tool_name = _tool_call_name(gated_call) or (pending_tool or {}).get("tool_name") or "unknown"

    approved = bool(getattr(decision, "approved", False))
    if approved:
        # Raw executor: the human already approved, so bypass the ASK middleware (no re-prompt).
        arguments = _tool_call_args(gated_call)
        if isinstance(arguments, str):
            import json

            try:
                arguments = json.loads(arguments)
            except Exception:
                arguments = {}
        result = await tool_service.execute_tool(tool_name, arguments or {})
        content = _format_tool_result(result)
    else:
        note = getattr(decision, "response", None) or ""
        content = f"Tool call rejected by human approval: {note}".strip()

    controller.add_tool_result(tc_id, content)

    # Continuation: drive the turn primitive (adds no user message) until the model stops calling
    # tools. Durable pause is not armed here, so a *new* ASK during continuation follows the normal
    # inline path (chained durable pauses are deferred to a later phase).
    user_message = _last_user_message(messages)
    final_content = ""
    final_tool_calls: List[Dict[str, Any]] = []
    turns = 0
    for _ in range(_MAX_CONTINUATION_TURNS):
        turn = await turn_executor.execute_turn(user_message)
        turns += 1
        response = getattr(turn, "response", None)
        final_content = str(getattr(response, "content", "") or "")
        final_tool_calls = list(getattr(response, "tool_calls", None) or [])
        if not getattr(turn, "has_tool_calls", False):
            break

    return ResumeResult(
        final_content=final_content,
        tool_calls=final_tool_calls,
        approved=approved,
        gated_tool=tool_name,
        continuation_turns=turns,
    )
