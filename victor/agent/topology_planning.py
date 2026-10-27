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

"""Topology planning integration for the AgenticLoop (extracted).

NOTE for tests: monkeypatch module-level functions in THIS module
(victor.agent.topology_planning), not in agentic_loop.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, Dict, List, Optional

from victor.agent.perception_integration import Perception
from victor.agent.topology_contract import TopologyAction
from victor.agent.topology_grounder import GroundedTopologyPlan
from victor.agent.topology_telemetry import (
    build_topology_telemetry_event,
    emit_topology_telemetry_event,
)

logger = logging.getLogger(__name__)


class TopologyPlanningMixin:
    """Topology plan initialization, runtime preparation, and provider hints."""

    async def _initialize_topology_plan(
        self,
        query: str,
        perception: Perception,
        state: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[GroundedTopologyPlan]:
        """Build and ground a topology plan for the first runtime iteration."""
        if not self._topology_enabled or state.get("_topology_plan_obj") is not None:
            return None

        task_type = str(
            state.get("task_type")
            or getattr(getattr(perception, "task_analysis", None), "task_type", "unknown")
            or "unknown"
        )
        task_classification = state.get("_task_classification")
        default_tool_budget = getattr(task_classification, "tool_budget", None)
        tool_budget = state.get("tool_budget", default_tool_budget if default_tool_budget else 10)
        routing_context = dict(context or {})
        routing_context.setdefault(
            "iteration_budget", state.get("iteration_budget", self.max_iterations)
        )
        routing_context.setdefault("tool_budget", tool_budget)
        routing_context.setdefault("available_team_formations", self._default_team_formations())

        provider_hints = await self._get_topology_provider_hints(
            task_type=task_type,
            context=routing_context,
        )
        if provider_hints:
            state["_topology_provider_hints"] = dict(provider_hints)
            routing_context.update(provider_hints)
            provider_candidates = []
            primary_provider = provider_hints.get("provider_hint")
            if isinstance(primary_provider, str) and primary_provider:
                provider_candidates.append(primary_provider)
            for fallback in provider_hints.get("fallback_chain", []):
                if isinstance(fallback, str) and fallback:
                    provider_candidates.append(fallback)
            if provider_candidates:
                routing_context["provider_candidates"] = list(dict.fromkeys(provider_candidates))
        structured_routing_policy = state.get("_structured_routing_policy_obj")
        if structured_routing_policy is not None and hasattr(
            structured_routing_policy, "selector_context"
        ):
            learned_topology_context = structured_routing_policy.selector_context()
            if isinstance(learned_topology_context, dict) and learned_topology_context:
                routing_context.update(learned_topology_context)
        elif hasattr(self.runtime_intelligence, "get_structured_routing_policy"):
            learned_scope_context = dict(routing_context)
            learned_scope_context.setdefault("task_type", task_type)
            try:
                structured_routing_policy = self.runtime_intelligence.get_structured_routing_policy(
                    query=query,
                    scope_context=learned_scope_context,
                )
            except Exception as exc:
                logger.debug(
                    "Runtime intelligence structured routing policy unavailable: %s",
                    exc,
                )
            else:
                if structured_routing_policy is not None:
                    state["_structured_routing_policy_obj"] = structured_routing_policy
                    if hasattr(structured_routing_policy, "to_dict"):
                        serialized_policy = structured_routing_policy.to_dict()
                        if isinstance(serialized_policy, dict):
                            state["structured_routing_policy"] = serialized_policy
                    learned_topology_context = structured_routing_policy.selector_context()
                    if isinstance(learned_topology_context, dict) and learned_topology_context:
                        routing_context.update(learned_topology_context)
        if hasattr(self.runtime_intelligence, "get_topology_routing_context"):
            learned_scope_context = dict(routing_context)
            learned_scope_context.setdefault("task_type", task_type)
            try:
                learned_topology_context = self.runtime_intelligence.get_topology_routing_context(
                    query=query, scope_context=learned_scope_context
                )
            except Exception as exc:
                logger.debug("Runtime intelligence topology routing hints unavailable: %s", exc)
            else:
                if isinstance(learned_topology_context, dict) and learned_topology_context:
                    routing_context.update(learned_topology_context)

        topology_input = self.paradigm_router.build_topology_input(
            task_type=task_type,
            query=query,
            history_length=len(conversation_history) if conversation_history else 0,
            query_complexity=self._perception_complexity_score(perception),
            tool_budget=int(tool_budget) if tool_budget is not None else None,
            context=routing_context,
        )
        topology_decision = self._topology_selector.select(topology_input)
        topology_plan = self._topology_grounder.ground(topology_decision)
        topology_overrides = topology_plan.to_context_overrides()
        runtime_context_overrides = self._merge_runtime_context_overrides(
            state,
            topology_overrides,
        )

        state["_topology_input_obj"] = topology_input
        state["_topology_decision_obj"] = topology_decision
        state["_topology_plan_obj"] = topology_plan
        state["topology_input"] = topology_input.to_dict()
        state["topology_decision"] = topology_decision.to_dict()
        state["topology_plan"] = topology_plan.to_dict()
        self._set_runtime_context_overrides(state, runtime_context_overrides)
        state["topology_iteration"] = state.get("iteration", 1)
        if topology_plan.tool_budget is not None:
            state["tool_budget"] = topology_plan.tool_budget
        if topology_plan.iteration_budget is not None:
            state["iteration_budget"] = topology_plan.iteration_budget
        if topology_decision.action == TopologyAction.DIRECT_RESPONSE:
            state["_is_qa_task"] = True

        logger.info(
            "[Topology] action=%s topology=%s provider=%s formation=%s",
            topology_decision.action.value,
            topology_decision.topology.value,
            topology_plan.provider,
            topology_plan.formation,
        )
        return topology_plan

    async def _prepare_topology_runtime(
        self,
        topology_plan: GroundedTopologyPlan,
        query: str,
        state: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Let the runtime prepare topology-specific execution state once selected."""
        if self.turn_executor is None:
            return None

        prepare_runtime_topology = getattr(self.turn_executor, "prepare_runtime_topology", None)
        if (
            prepare_runtime_topology is None
            or not callable(prepare_runtime_topology)
            or not inspect.iscoroutinefunction(prepare_runtime_topology)
        ):
            return None

        try:
            result = await prepare_runtime_topology(
                topology_plan,
                user_message=query,
                task_classification=state.get("_task_classification"),
            )
        except Exception as exc:
            logger.debug("Topology runtime preparation unavailable: %s", exc)
            return None

        if isinstance(result, dict):
            prepared = dict(result)
            prepared_overrides = prepared.get("runtime_context_overrides")
            if isinstance(prepared_overrides, dict):
                merged_overrides = self._merge_runtime_context_overrides(
                    state,
                    prepared_overrides,
                )
                self._set_runtime_context_overrides(state, merged_overrides)
            return prepared
        return None

    async def _get_topology_provider_hints(
        self,
        task_type: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Fetch provider-routing hints when the active provider exposes them."""
        if self.turn_executor is None:
            return {}

        provider_context = getattr(self.turn_executor, "_provider_context", None)
        if provider_context is None:
            return {}

        provider = getattr(provider_context, "provider", None)
        model_hint = getattr(provider_context, "model", None)
        preferred_providers = context.get("preferred_providers")

        hint_sources = [
            getattr(provider, "get_topology_provider_hints", None),
            getattr(getattr(provider, "engine", None), "get_topology_provider_hints", None),
        ]
        for get_hints in hint_sources:
            if not callable(get_hints):
                continue
            try:
                hints = await get_hints(
                    task_type=task_type,
                    model_hint=model_hint,
                    preferred_providers=preferred_providers,
                )
            except Exception as exc:
                logger.debug("Topology provider hints unavailable: %s", exc)
                continue
            if isinstance(hints, dict):
                return hints
        return {}
