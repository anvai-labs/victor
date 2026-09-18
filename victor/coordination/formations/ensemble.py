"""Opt-in ensemble aggregation shared by PARALLEL and CONSENSUS dispatch.

This is an aggregation policy, not another formation registry. Candidates sample
one task independently. All decisions consume validated JSON contracts.
"""

from __future__ import annotations

import copy
import json
import logging
from collections import Counter
from typing import Any, List

from victor.coordination.formations.base import BaseFormationStrategy, TeamContext
from victor.teams.types import AgentMessage, MemberResult, MessageType

logger = logging.getLogger(__name__)
MODES = frozenset({"vote", "judge", "synthesizer"})


def _contract(output: str, fields: set[str]) -> dict[str, str]:
    value = json.loads(output)
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"Expected JSON object with exactly {sorted(fields)}")
    if any(not isinstance(item, str) or not item.strip() for item in value.values()):
        raise ValueError("Contract values must be nonempty strings")
    return value


async def execute_ensemble(
    strategy: BaseFormationStrategy, agents: List[Any], context: TeamContext, task: AgentMessage
) -> List[MemberResult]:
    """Run one proposal wave, then one deterministic or model aggregation pass.

    Durable partial resume is explicitly unsupported for this mode. Reject a
    checkpointer before any member executes; normal formation durability remains
    unchanged when ensemble_mode is absent.
    """
    mode = context.get("ensemble_mode")
    if mode not in MODES:
        raise ValueError(f"ensemble_mode must be one of {sorted(MODES)}")
    if context.checkpoint_hook is not None or context.resume_completed:
        raise ValueError("Ensemble aggregation does not support durable partial resume")
    aggregator_id = context.get("ensemble_aggregator_id")
    aggregators = [agent for agent in agents if agent.id == aggregator_id]
    if mode != "vote" and len(aggregators) != 1:
        raise ValueError(f"{mode} requires one configured ensemble_aggregator_id")
    if mode == "vote" and aggregator_id is not None:
        raise ValueError("Vote mode does not accept an aggregator member")
    candidates = [agent for agent in agents if agent.id != aggregator_id]
    if len(candidates) < 2 or len({agent.id for agent in agents}) != len(agents):
        raise ValueError("Ensemble requires at least two candidates with distinct member IDs")
    prompt = AgentMessage(
        sender_id="coordinator",
        message_type=MessageType.TASK,
        content=json.dumps(
            {
                "task": task.content,
                "instructions": "Independently solve this task. Return only the JSON contract. Use artifact references for large deliverables.",
                "response_contract": {
                    "vote_key": "canonical answer used for equality voting",
                    "answer": "answer or artifact reference",
                },
                "example": {"vote_key": "42", "answer": "42"},
            }
        ),
    )
    contexts = [
        TeamContext(
            team_id=context.team_id,
            formation=context.formation,
            shared_state=copy.deepcopy(context.shared_state),
        )
        for _ in candidates
    ]
    results = await strategy._execute_members_concurrently(candidates, prompt, context, contexts)
    proposals = []
    for result in results:
        if not result.success:
            continue
        try:
            proposal = _contract(result.output, {"vote_key", "answer"})
            if len(proposal["answer"]) > 8192:
                raise ValueError(
                    "Proposal answer exceeds 8192 characters; return an artifact reference"
                )
            proposals.append({"member_id": result.member_id, **proposal})
        except (ValueError, TypeError) as exc:
            logger.warning("Invalid ensemble proposal from %s: %s", result.member_id, exc)
            result.success = False
            result.error = str(exc)
    decision = ""
    error = None
    if len(proposals) != len(candidates):
        error = "Every ensemble candidate must deliver a valid proposal"
    elif mode == "vote":
        counts = Counter(proposal["vote_key"] for proposal in proposals)
        winner, count = counts.most_common(1)[0]
        if count * 2 <= len(candidates):
            error = "No strict majority; vote ties do not select an arbitrary winner"
        else:
            decision = next(
                proposal["answer"] for proposal in proposals if proposal["vote_key"] == winner
            )
    else:
        fields = {"selected_member_id"} if mode == "judge" else {"answer"}
        aggregate_task = AgentMessage(
            sender_id="coordinator",
            message_type=MessageType.TASK,
            content=json.dumps(
                {
                    "task": task.content,
                    "candidates": proposals,
                    "role": mode,
                    "instructions": "Return only the JSON contract. Judge selects one listed member; synthesizer combines the proposals once.",
                    "response_contract": dict.fromkeys(fields, "nonempty string"),
                    "example": (
                        {"selected_member_id": proposals[0]["member_id"]}
                        if mode == "judge"
                        else {"answer": "combined answer or artifact reference"}
                    ),
                }
            ),
        )
        aggregate = (
            await strategy._execute_members_concurrently(
                aggregators, aggregate_task, context, [context], indices=[len(candidates)]
            )
        )[0]
        results.append(aggregate)
        try:
            if not aggregate.success:
                raise ValueError(aggregate.error or "Aggregator execution failed")
            response = _contract(aggregate.output, fields)
            if mode == "judge":
                selected = [
                    proposal
                    for proposal in proposals
                    if proposal["member_id"] == response["selected_member_id"]
                ]
                if not selected:
                    raise ValueError("Judge selected an unknown candidate member")
                decision = selected[0]["answer"]
            else:
                decision = response["answer"]
        except (ValueError, TypeError) as exc:
            aggregate.success = False
            aggregate.error = str(exc)
            error = str(exc)
    if error:
        logger.warning("Ensemble %s failed: %s", mode, error)
    for result in results:
        result.metadata.update(
            ensemble_mode=mode,
            ensemble_success=error is None,
            ensemble_decision=decision,
            ensemble_error=error,
        )
    return results
