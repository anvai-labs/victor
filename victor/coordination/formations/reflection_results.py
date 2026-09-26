"""Opt-in retention of reflection member results across refinement iterations."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from victor.coordination.formations.member_attempts import aggregate_attempts
from victor.teams.types import AgentMessage, MemberResult, MessageType


class ReflectionResults:
    """Keep the legacy text loop while retaining canonical participant results."""

    def __init__(self, agents: list[Any], context: Any, saved: dict[str, Any]):
        self.enabled = bool(context.get("capture_member_usage", False))
        self.context = context
        self.agents = {agent.id: agent for agent in agents} if self.enabled else {}
        if saved and bool(saved.get("capture_member_usage", False)) != self.enabled:
            raise ValueError("Cannot change reflection member capture during resume")
        self.attempts = [
            MemberResult.from_dict(deepcopy(item)) for item in saved.get("member_results", [])
        ]

    async def execute(self, agent: Any, prompt: str) -> str:
        if not self.enabled:
            return await agent.execute(prompt, context=self.context.shared_state)
        if agent.id not in self.agents:
            raise ValueError("Reflection capture requires the canonical member participant")
        result = await self.agents[agent.id].execute(
            AgentMessage(
                sender_id="reflection_formation", content=prompt, message_type=MessageType.TASK
            ),
            self.context,
        )
        self.attempts.append(deepcopy(result))
        if not result.success:
            raise RuntimeError(result.error or "Reflection member execution failed")
        return result.output

    def checkpoint(self) -> dict[str, Any]:
        if not self.enabled:
            return {}
        return {
            "capture_member_usage": True,
            "member_results": [deepcopy(item.to_dict()) for item in self.attempts],
        }

    def finish(self, aggregate: MemberResult) -> list[MemberResult]:
        if not self.enabled:
            return [aggregate]
        if not self.attempts:
            raise ValueError("Reflection capture has no captured member results")
        grouped: dict[str, list[MemberResult]] = {}
        for attempt in self.attempts:
            grouped.setdefault(attempt.member_id, []).append(attempt)
        results = []
        for attempts in grouped.values():
            result = aggregate_attempts(attempts, "reflection")
            if aggregate.member_id == result.member_id and not aggregate.success:
                result.success = False
                result.error = aggregate.error
            result.metadata.update(
                reflection_success=aggregate.success
                and aggregate.metadata.get("satisfied") is True,
                reflection_output=aggregate.output,
                reflection_summary=deepcopy(aggregate.metadata),
            )
            results.append(result)
        return results
