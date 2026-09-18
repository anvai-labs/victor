"""Shared conversation runner; concrete strategies remain in the canonical registry."""

from __future__ import annotations

import inspect
import json
import logging
from dataclasses import asdict
from typing import Any, List

from victor.coordination.formations.base import BaseFormationStrategy, TeamContext
from victor.framework.member_event_sink import (
    MEMBER_HANDOFF,
    MEMBER_SPOKE,
    MemberEvent,
    current_member_sink,
)
from victor.teams.transcript import PeerHandoff, TeamTranscript
from victor.teams.types import AgentMessage, MemberResult, MessageType

logger = logging.getLogger(__name__)


def _object(output: str, fields: set[str]) -> dict:
    value = json.loads(output)
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"Expected JSON object with exactly {sorted(fields)}")
    return value


async def _call(function, *args):
    result = function(*args)
    return await result if inspect.isawaitable(result) else result


class ConversationFormation(BaseFormationStrategy):
    """Bounded transcript execution with explicit selection and termination."""

    mode = "group_chat"

    def validate_context(self, context: TeamContext) -> bool:
        return isinstance(context, TeamContext)

    def supports_early_termination(self) -> bool:
        return True

    def supports_durable_pause(self) -> bool:
        return False

    async def execute(
        self, agents: List[Any], context: TeamContext, task: AgentMessage
    ) -> List[MemberResult]:
        if context.checkpoint_hook is not None or context.resume_completed:
            raise ValueError("Conversation formations do not support durable partial resume")
        max_turns = context.get("conversation_max_turns", 6)
        if type(max_turns) is not int or max_turns < 1:
            raise ValueError("conversation_max_turns must be a positive integer")
        transcript = TeamTranscript(context.get("transcript_max_chars", 32768))
        by_id = {agent.id: agent for agent in agents}
        if not agents or len(by_id) != len(agents):
            raise ValueError("Conversation requires configured members with unique IDs")
        router = context.get("conversation_router_id")
        judge = context.get("conversation_judge_id")
        selector = context.get("selector_func")
        candidate_func = context.get("candidate_func")
        termination = context.get("termination_func")
        for name, callback in (
            ("selector_func", selector),
            ("candidate_func", candidate_func),
            ("termination_func", termination),
        ):
            if callback is not None and not callable(callback):
                raise ValueError(f"{name} must be callable")
        if router is not None and (
            router not in by_id or selector is not None or self.mode != "group_chat"
        ):
            raise ValueError(
                "Router must be a configured member and cannot combine with a selector"
            )
        if self.mode == "debate" and judge not in by_id:
            raise ValueError("Debate requires a configured judge member")
        speakers = [identifier for identifier in by_id if identifier not in {router, judge}]
        if not speakers:
            raise ValueError("Conversation requires at least one speaking member")
        if self.mode != "group_chat" and (selector is not None or candidate_func is not None):
            raise ValueError("Programmatic speaker selection is supported only for GROUP_CHAT")
        current = context.get("handoff_start_id", speakers[0])
        if self.mode == "handoff" and current not in speakers:
            raise ValueError("Handoff start must be a configured speaking member")
        results: dict[str, MemberResult] = {}
        final_output = ""
        success = False
        reason = "max_turns"
        sink = current_member_sink.get()

        def record(result: MemberResult):
            prior = results.get(result.member_id)
            if prior is not None:
                result.tool_calls_used += prior.tool_calls_used
                result.duration_seconds += prior.duration_seconds
                result.success = result.success and prior.success
            result.metadata["conversation_turns"] = (
                prior.metadata.get("conversation_turns", 0) if prior else 0
            ) + 1
            results[result.member_id] = result

        async def invoke(identifier, payload, index):
            message = AgentMessage(
                sender_id="coordinator", message_type=MessageType.TASK, content=json.dumps(payload)
            )
            result = (
                await self._execute_members_concurrently(
                    [by_id[identifier]], message, context, [context], indices=[index]
                )
            )[0]
            record(result)
            if not result.success:
                raise ValueError(result.error or f"Member {identifier} failed")
            return result

        async def emit(entry, handoff=None):
            if sink is None:
                return
            await sink.emit(
                MemberEvent(
                    kind=MEMBER_SPOKE,
                    member_id=entry.member_id,
                    formation=self.mode,
                    index=entry.sequence,
                    metadata={"transcript_sequence": entry.sequence},
                )
            )
            if handoff is not None:
                await sink.emit(
                    MemberEvent(
                        kind=MEMBER_HANDOFF,
                        member_id=entry.member_id,
                        formation=self.mode,
                        index=entry.sequence,
                        metadata=asdict(handoff),
                    )
                )

        last_result = None
        try:
            for turn in range(max_turns):
                last_result = None
                eligible = list(speakers)
                if candidate_func is not None:
                    eligible = await _call(candidate_func, transcript.snapshot(), tuple(speakers))
                    if (
                        not isinstance(eligible, (list, tuple))
                        or not eligible
                        or not all(isinstance(item, str) for item in eligible)
                        or len(set(eligible)) != len(eligible)
                        or not set(eligible) <= set(speakers)
                    ):
                        raise ValueError(
                            "candidate_func must return a nonempty unique subset of speaking IDs"
                        )
                if self.mode == "handoff":
                    selected = current
                elif selector is not None:
                    selected = await _call(selector, transcript.snapshot(), tuple(eligible))
                elif router is not None:
                    selection = await invoke(
                        router,
                        {
                            "task": task.content,
                            "transcript": transcript.to_list(),
                            "eligible_member_ids": eligible,
                            "response_contract": {"speaker_id": "one eligible member ID"},
                            "example": {"speaker_id": eligible[0]},
                        },
                        turn,
                    )
                    last_result = selection
                    selected = _object(selection.output, {"speaker_id"})["speaker_id"]
                else:
                    selected = eligible[turn % len(eligible)]
                if not isinstance(selected, str) or selected not in eligible:
                    raise ValueError("Speaker selection returned an ineligible member ID")
                result = await invoke(
                    selected,
                    {
                        "task": task.content,
                        "transcript": transcript.to_list(),
                        "member_id": selected,
                        "peer_ids": speakers,
                        "mode": self.mode,
                        "instructions": "Contribute to the shared task. Return only this JSON contract. Use references for large artifacts.",
                        "response_contract": {
                            "content": "nonempty message",
                            "done": "boolean",
                            "handoff_to": "peer ID or null",
                        },
                        "example": {
                            "content": "Artifact: result.md",
                            "done": True,
                            "handoff_to": None,
                        },
                    },
                    turn,
                )
                last_result = result
                message = _object(result.output, {"content", "done", "handoff_to"})
                if type(message["done"]) is not bool:
                    raise ValueError("done must be a boolean")
                target = message["handoff_to"]
                if target is not None and (
                    not isinstance(target, str) or target not in speakers or target == selected
                ):
                    raise ValueError("handoff_to must identify a different configured peer")
                if target is not None and (self.mode != "handoff" or message["done"]):
                    raise ValueError("Handoff destination is incompatible with this turn")
                if self.mode == "handoff" and not message["done"] and target is None:
                    raise ValueError("An unfinished handoff turn requires a peer destination")
                entry = transcript.append(selected, message["content"], target)
                result.output = entry.content
                handoff = (
                    PeerHandoff(selected, target, entry.sequence) if target is not None else None
                )
                await emit(entry, handoff)
                final_output = entry.content
                stop = False
                if termination is not None:
                    stop = await _call(termination, transcript.snapshot())
                    if type(stop) is not bool:
                        raise ValueError("termination_func must return a boolean")
                if self.mode != "debate" and (message["done"] or stop):
                    success, reason = True, "predicate" if stop else "done"
                    break
                if self.mode == "debate" and stop:
                    break
                if target is not None:
                    current = target
            if self.mode == "debate":
                verdict = await invoke(
                    judge,
                    {
                        "task": task.content,
                        "transcript": transcript.to_list(),
                        "role": "judge",
                        "response_contract": {
                            "selected_member_id": "member that spoke",
                            "verdict": "nonempty verdict",
                        },
                        "example": {
                            "selected_member_id": transcript.snapshot()[0].member_id,
                            "verdict": "Selected the supported proposal",
                        },
                    },
                    max_turns,
                )
                last_result = verdict
                parsed = _object(verdict.output, {"selected_member_id", "verdict"})
                if parsed["selected_member_id"] not in {
                    entry.member_id for entry in transcript.snapshot()
                }:
                    raise ValueError("Judge selected a member that did not speak")
                entry = transcript.append(judge, parsed["verdict"])
                verdict.output = entry.content
                await emit(entry)
                success, reason, final_output = True, "judge", entry.content
        except Exception as exc:
            logger.warning("Conversation %s failed: %s", self.mode, exc)
            success, reason, final_output = False, "error", str(exc)
            if last_result is not None:
                last_result.success = False
                last_result.error = str(exc)
            if not results:
                raise
        if not success and reason == "max_turns":
            logger.warning("Conversation %s exhausted max_turns=%d", self.mode, max_turns)
        context.set("conversation_transcript", transcript.to_list())
        context.set("conversation_termination", reason)
        for result in results.values():
            result.metadata.update(
                conversation_success=success,
                conversation_output=final_output,
                conversation_termination=reason,
            )
        return list(results.values())
