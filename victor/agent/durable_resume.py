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

"""Faithful replay of a durably-paused single-agent turn (FEP-0029 Phase 3a/3b).

When a turn paused on a policy ASK, the assistant had already produced a message with ``tool_calls``;
one of those calls was gated and never ran. On resume with a human decision this module — *without
re-calling the LLM for that message* — resolves **every** unresolved tool_call in that message and
drives one or more continuation turns so the model sees the results and proceeds.

The **gated** call (matched by the pause's ``pending_tool``) is handled per the human's decision:
approve → ``ToolService.execute_tool`` (the raw tool executor, which does *not* re-run the ASK
middleware since the human already decided); reject → a tool-error result. A parallel-tool pause
aborts the *whole* batch, so any **sibling** calls that never ran are also unresolved; those are
executed through the normal ``ToolService.execute_tool_call`` pipeline (reused, policy-honoring) so
the conversation has a result for every tool_call before continuing (providers reject a dangling
tool_call). The continuation reuses the turn executor's ``execute_turn`` primitive (which adds no
user message), so there is no spurious user turn.

This is the single shared replay used by every surface (``VictorClient.resume``, the HTTP
``/chat/resume`` route, and the ``victor session resume`` CLI) — hardening here improves all of them.

Deferred: streaming resume; chained pauses (a *new* ASK during continuation follows the inline path
here, since durable pause is not re-armed).
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


def _find_unresolved_calls(messages: List[Any]) -> List[Tuple[Dict[str, Any], str]]:
    """Return every ``(tool_call, id)`` in the last assistant message that has no result yet.

    When a paused turn requested several tool_calls in parallel, the ``ApprovalPause`` aborts the
    whole batch — so *all* of them are unresolved on resume (not just the gated one). Order is
    preserved. Raises :class:`ResumeError` if none is found or a call lacks an id.
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


def _pick_gated_index(
    unresolved: List[Tuple[Dict[str, Any], str]], pending_tool: Optional[Dict[str, Any]]
) -> int:
    """Index of the gated call among the unresolved ones — the one the human decided on.

    Disambiguated by the ``pending_tool`` (name recorded at pause time); defaults to the first.
    """
    if pending_tool and len(unresolved) > 1:
        want = pending_tool.get("tool_name")
        for i, (tc, _id) in enumerate(unresolved):
            if _tool_call_name(tc) == want:
                return i
    return 0


def _format_tool_result(result: Any) -> str:
    """Render a tool result (ToolService.execute_tool object *or* execute_tool_call dict) to text."""
    if result is None:
        return ""
    if isinstance(result, dict):
        for key in ("result", "output", "content", "error"):
            if result.get(key) is not None:
                return str(result[key])
        return str(result)
    for attr in ("output", "result", "content"):
        val = getattr(result, attr, None)
        if val is not None:
            return str(val)
    return str(result)


def _parse_args(arguments: Any) -> Dict[str, Any]:
    """Coerce tool_call arguments (dict or JSON string) into a dict."""
    if isinstance(arguments, str):
        import json

        try:
            arguments = json.loads(arguments)
        except Exception:
            return {}
    return arguments or {}


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
    unresolved = _find_unresolved_calls(messages)
    gated_idx = _pick_gated_index(unresolved, pending_tool)
    gated_tool = _tool_call_name(unresolved[gated_idx][0]) or (pending_tool or {}).get("tool_name")
    approved = bool(getattr(decision, "approved", False))

    # Resolve EVERY unresolved call so the conversation is consistent before continuing (a
    # parallel-tool pause aborts the whole batch, so siblings need results too — else providers
    # reject an assistant tool_call with no matching result). The gated call is handled per the
    # human's decision; siblings run through the normal tool pipeline (reused, policy-honoring).
    sibling_count = 0
    for i, (tc, tc_id) in enumerate(unresolved):
        name = _tool_call_name(tc) or (pending_tool or {}).get("tool_name") or "unknown"
        args = _parse_args(_tool_call_args(tc))
        if i == gated_idx:
            if approved:
                # Raw executor: the human already approved, so bypass the ASK middleware.
                content = _format_tool_result(await tool_service.execute_tool(name, args))
            else:
                note = getattr(decision, "response", None) or ""
                content = f"Tool call rejected by human approval: {note}".strip()
        else:
            # Sibling that never ran (batch aborted at the gate) — execute via the existing
            # pipeline path so its policy is honored, and record its result.
            sibling_count += 1
            res = await tool_service.execute_tool_call(
                {"id": tc_id, "name": name, "arguments": args}
            )
            content = _format_tool_result(res)
        controller.add_tool_result(tc_id, content)

    # Continuation: drive the turn primitive (adds no user message) until the model stops calling
    # tools. Durable pause is not armed here, so a *new* ASK during continuation follows the normal
    # inline path (chained durable pauses are deferred to a later phase).
    tool_name = gated_tool or "unknown"
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
        executed_siblings=sibling_count,
    )
