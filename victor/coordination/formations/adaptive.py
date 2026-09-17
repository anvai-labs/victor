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

"""Bounded adaptive execution over the canonical formation registry.

The default cycle is SequentialFormation → HierarchicalFormation →
ConsensusFormation. Low performance, member failure, or structured quality feedback
can trigger another attempt during this invocation. Members may execute more than
once: use idempotent tasks. All switching state is local to an invocation.
"""

from __future__ import annotations

import logging
import time
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from victor.coordination.formations.base import BaseFormationStrategy, TeamContext
from victor.teams.types import AgentMessage, MemberResult, TeamFormation

logger = logging.getLogger(__name__)


class AdaptationStrategy(str, Enum):
    """Signal used to decide whether another formation should be attempted."""

    PERFORMANCE = "performance"
    ERROR_RATE = "error_rate"
    FEEDBACK = "feedback"


class AdaptiveFormation(BaseFormationStrategy):
    """Try SequentialFormation, HierarchicalFormation, then ConsensusFormation.

    Options can be supplied per run through ``shared_state['adaptive_options']``.
    Results retain real member IDs, outcomes and metrics; adaptation diagnostics
    are structured metadata. Durable mid-loop resume is deliberately unsupported.
    """

    def __init__(
        self,
        adaptation_strategy: str = "performance",
        max_switches: int = 3,
        performance_threshold: float = 0.5,
        max_duration_seconds: float = 60.0,
        formation_cycle: Optional[List[str]] = None,
        *,
        resolve_formation: Optional[Callable[[TeamFormation], BaseFormationStrategy]] = None,
    ):
        self.adaptation_strategy = AdaptationStrategy(adaptation_strategy)
        if max_switches < 0 or max_duration_seconds <= 0 or not 0 <= performance_threshold <= 1:
            raise ValueError("Invalid adaptive bounds")
        self.max_switches = max_switches
        self.performance_threshold = performance_threshold
        self.max_duration_seconds = max_duration_seconds
        self.formation_cycle = [
            TeamFormation(name)
            for name in (
                formation_cycle
                if formation_cycle is not None
                else ["sequential", "hierarchical", "consensus"]
            )
        ]
        # Only participant-driven strategies are allowed. Reflection requires role
        # binding by the coordinator; adaptive recursion is never meaningful.
        if not self.formation_cycle or any(
            name in (TeamFormation.ADAPTIVE, TeamFormation.REFLECTION)
            for name in self.formation_cycle
        ):
            raise ValueError("Adaptive cycle must contain non-recursive participant formations")
        self._resolve_formation = resolve_formation

    async def execute(
        self, agents: List[Any], context: TeamContext, task: AgentMessage
    ) -> List[MemberResult]:
        options = context.get("adaptive_options")
        if options is not None:
            if not isinstance(options, dict) or "resolve_formation" in options:
                raise ValueError("adaptive_options must contain only adaptive configuration")
            configured = AdaptiveFormation(**options, resolve_formation=self._resolve_formation)
        else:
            configured = self
        return await configured._execute_run(agents, context, task)

    async def _execute_run(self, agents, context, task):
        if not agents:
            return [
                MemberResult(
                    member_id="adaptive", success=False, output="", error="No members available"
                )
            ]
        resolve = self._resolve_formation
        if resolve is None:
            from victor.coordination.formations import create_formation_registry

            resolve = create_formation_registry().__getitem__
        hint = context.get("initial_formation_hint")
        if hint is not None:
            initial = TeamFormation(hint)
            if initial not in self.formation_cycle:
                raise ValueError("initial_formation_hint must be in adaptive formation_cycle")
            index = self.formation_cycle.index(initial)
        else:
            index = (
                self.formation_cycle.index(TeamFormation.HIERARCHICAL)
                if len(task.content) > 1000 and TeamFormation.HIERARCHICAL in self.formation_cycle
                else 0
            )
        history = []
        totals: Dict[str, int] = {}
        durations: Dict[str, float] = {}
        latest: Dict[str, MemberResult] = {}
        # Never reuse an inner formation's checkpoint across topology changes.
        inner_context = TeamContext(context.team_id, context.formation, dict(context.shared_state))
        inner_context.member_event_hook = context.member_event_hook
        for switches in range(self.max_switches + 1):
            name = self.formation_cycle[index]
            started = time.monotonic()
            try:
                results = await resolve(name).execute(agents, inner_context, task)
            except Exception as exc:
                logger.warning("Adaptive formation %s failed: %s", name.value, exc)
                results = [
                    MemberResult(member_id="adaptive", success=False, output="", error=str(exc))
                ]
            elapsed = time.monotonic() - started
            score = self._evaluate_performance(results, elapsed)
            history.append(
                {"formation": name.value, "duration": elapsed, "performance_score": score}
            )
            for result in results:
                latest[result.member_id] = result
                totals[result.member_id] = totals.get(result.member_id, 0) + result.tool_calls_used
                durations[result.member_id] = (
                    durations.get(result.member_id, 0) + result.duration_seconds
                )
            if not results:
                results = [
                    MemberResult(
                        member_id="adaptive",
                        success=False,
                        output="",
                        error="Formation returned no member results",
                    )
                ]
            retry = (
                any(not result.success for result in results) or score < self.performance_threshold
            )
            if not retry or switches == self.max_switches:
                if retry:
                    logger.warning(
                        "Adaptive formation exhausted its switch budget (score %.3f)", score
                    )
                # A narrower final topology must not erase earlier members' costs
                # or failed deliverables. Later attempts replace the same ID only.
                for result in results:
                    latest[result.member_id] = result
                results = list(latest.values())
                for result in results:
                    result.tool_calls_used = totals.get(result.member_id, 0)
                    result.duration_seconds = durations.get(result.member_id, 0)
                    result.metadata.update(
                        current_formation=name.value,
                        performance_score=score,
                        formation_switches=switches,
                        formation_history=list(history),
                        adaptation_strategy=self.adaptation_strategy.value,
                        formation="adaptive",
                    )
                return results
            logger.warning(
                "Adaptive formation %s degraded (score %.3f); switching topology", name.value, score
            )
            index = (index + 1) % len(self.formation_cycle)
        raise AssertionError("Adaptive loop must return within its switch bound")

    def _evaluate_performance(self, results, duration):
        if self.adaptation_strategy == AdaptationStrategy.PERFORMANCE:
            return max(0.0, 1 - duration / self.max_duration_seconds)
        if self.adaptation_strategy == AdaptationStrategy.ERROR_RATE:
            return sum(result.success for result in results) / len(results) if results else 0.0
        scores = [result.metadata.get("quality_score") for result in results]
        if not scores or any(
            not isinstance(score, (int, float)) or not 0 <= score <= 1 for score in scores
        ):
            raise ValueError(
                "Feedback adaptation requires a numeric quality_score in [0, 1] from every member"
            )
        return sum(scores) / len(scores)

    def validate_context(self, context):
        return any(
            hasattr(value, "execute") and hasattr(value, "id")
            for value in context.shared_state.values()
        )

    def supports_durable_pause(self) -> bool:
        """False: adaptive attempts have no persisted topology/iteration cursor."""
        return False

    def supports_early_termination(self) -> bool:
        return True

    def get_performance_summary(self):
        """Run diagnostics live in result metadata, never mutable strategy state."""
        return {
            "formations_tried": [],
            "total_switches": 0,
            "current_formation": None,
            "adaptation_strategy": self.adaptation_strategy.value,
        }


__all__ = ["AdaptiveFormation", "AdaptationStrategy"]
