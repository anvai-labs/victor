"""Planning-to-team execution adapter.

This module keeps planning mode wired to the reusable team formation runtime
instead of creating a separate multi-agent execution path.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, Optional

from victor.agent.planning.base import (
    ExecutionPlan,
    PlanStep,
    StepResult,
    StepType,
    get_step_allowed_tools,
)
from victor.agent.planning.readable_schema import ReadableTaskPlan, TaskComplexity
from victor.agent.runtime.naming import build_display_name
from victor.agent.subagents import SubAgentOrchestrator, SubAgentRole
from victor.teams.types import (
    TeamAgentCategory,
    TeamFormation,
    TeamMember,
    TeamParticipant,
    TeamResult,
)
from victor.tools.core_tool_aliases import canonicalize_core_tool_name


class PlanningTeamExecutionAdapter:
    """Execute complex planned work through ``UnifiedTeamCoordinator`` formations."""

    def __init__(
        self,
        orchestrator: Any,
        sub_agent_orchestrator: Optional[SubAgentOrchestrator] = None,
        coordinator_factory: Optional[Callable[[Any], Any]] = None,
        approval_callback: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.orchestrator = orchestrator
        self.sub_agent_orchestrator = sub_agent_orchestrator
        self._coordinator_factory = coordinator_factory
        # Optional async callback ``(step, context) -> (approved: bool, feedback: str)``.
        # When None, approval nodes auto-approve and record a checkpoint marker.
        self._approval_callback = approval_callback

    # Node execution types that require the team-adapter path.
    # AutonomousPlanner (the fallback) has no awareness of these types.
    _TEAM_REQUIRED_EXECUTION_TYPES: frozenset = frozenset(
        {"compute", "tool", "loop", "conditional", "approval", "checkpoint"}
    )

    def should_use_team(self, plan: ReadableTaskPlan) -> bool:
        """Return whether this plan requires team-adapter execution.

        Always True for COMPLEX plans.  Also True when any step declares an
        advanced execution type (compute, loop, conditional, approval) because
        AutonomousPlanner — the fallback — has no dispatch logic for these.
        Falls back to exploratory-step counting for SIMPLE/MODERATE plans that
        use only the default agent execution path.
        """
        if plan.complexity == TaskComplexity.COMPLEX:
            return True

        exploratory_steps = 0
        for step_data in plan.steps:
            if isinstance(step_data, dict):
                exec_type = str(step_data.get("exec", step_data.get("execution", ""))).lower()
                step_type = str(step_data.get("type", "")).lower()
            elif len(step_data) >= 2:
                exec_type = str(step_data[5]).lower() if len(step_data) > 5 else ""
                step_type = str(step_data[1]).lower()
            else:
                continue
            if exec_type in self._TEAM_REQUIRED_EXECUTION_TYPES:
                return True
            if step_type in {"research", "analyze", "analysis", "review", "doc"}:
                exploratory_steps += 1
        return exploratory_steps >= 2

    async def execute_step(
        self,
        *,
        plan: ReadableTaskPlan,
        execution_plan: ExecutionPlan,
        step: PlanStep,
        root_session_id: Optional[str] = None,
        plan_state: Optional[Dict[str, Any]] = None,
    ) -> StepResult:
        """Dispatch a plan step to the appropriate execution node type.

        Dispatch order:
          1. compute      — deterministic function, no model call
          2. tool         — deterministic tool-only subagent, no reasoning overhead
          3. loop         — iterate over plan_state collection, one subagent per item
          4. conditional  — evaluate condition on plan_state, branch route downstream steps
          5. approval     — user checkpoint before continuing
          6. team         — UnifiedTeamCoordinator formation
          7. agent        — default single-worker path
        """
        resolved_plan_state = plan_state or {}

        # 1. Compute node — no model, no spawn; receives plan_state so it can
        # read outputs produced by earlier tool/compute steps.
        compute_node = self._compute_node_for_step(step)
        if compute_node is not None:
            return self._execute_compute_node(step, compute_node, resolved_plan_state)

        execution = (step.execution or step.context.get("execution", "")).lower()

        team_id = self._team_id(execution_plan.id, step.id)
        root_session_id = root_session_id or self._root_session_id()
        context = self._step_context(
            plan=plan,
            execution_plan=execution_plan,
            step=step,
            team_id=team_id,
            root_session_id=root_session_id,
        )

        # 2. Tool node — single subagent, tool-only
        if execution == "tool":
            return await self._execute_tool_node(
                step, execution_plan, team_id, context, resolved_plan_state
            )

        # 3. Loop node — iterate over a plan-state collection.  Inferred loop
        # steps occasionally miss the collection key; those must still get real
        # worker execution instead of becoming a zero-item no-op.
        if execution == "loop" and (step.context.get("items") or step.context.get("loop_over")):
            return await self._execute_loop_node(
                step, execution_plan, team_id, context, resolved_plan_state
            )

        # 4. Conditional node — evaluate condition, record branch decision in metadata
        if execution == "conditional":
            return self._execute_conditional_node(step, resolved_plan_state)

        # 5. Approval node — user checkpoint
        if execution in ("approval", "checkpoint"):
            return await self._execute_approval_node(step, context)

        # 6. Team node — explicit multi-agent formation
        if execution == "team":
            return await self._execute_team_node(
                step, execution_plan, team_id, context, resolved_plan_state
            )

        # 7. Default: single worker (agent)
        task = self._task_description_for_step(step, resolved_plan_state)
        if self._should_execute_step_directly(execution_plan, step):
            members = self._build_members(execution_plan, team_id, current_step=step)
            worker = next(
                member for member_id, member in members.items() if member_id != "plan_manager"
            )
            payload = await worker.execute_task(task, context)
            return self._member_payload_to_step_result(payload, worker.id)

        coordinator = self._create_coordinator()
        members = self._build_members(execution_plan, team_id, current_step=step)
        supervisor = members["plan_manager"]

        coordinator.set_formation(TeamFormation.HIERARCHICAL)
        coordinator.set_supervisor(supervisor)
        for member_id, member in members.items():
            if member_id != supervisor.id:
                coordinator.add_member(member)

        result = await coordinator.execute_task(task, context)
        return self._team_result_to_step_result(result)

    async def _execute_tool_node(
        self,
        step: PlanStep,
        execution_plan: "ExecutionPlan",
        team_id: str,
        context: Dict[str, Any],
        plan_state: Optional[Dict[str, Any]] = None,
    ) -> StepResult:
        """Run a deterministic tool-only step via a single subagent worker."""
        members = self._build_members(execution_plan, team_id, current_step=step)
        worker = next(m for mid, m in members.items() if mid != "plan_manager")
        task = self._task_description_for_step(step, plan_state)
        payload = await worker.execute_task(task, context)
        result = self._member_payload_to_step_result(payload, worker.id)
        result.metadata["execution_mode"] = "tool_node"
        return result

    async def _execute_team_node(
        self,
        step: PlanStep,
        execution_plan: "ExecutionPlan",
        team_id: str,
        context: Dict[str, Any],
        plan_state: Optional[Dict[str, Any]] = None,
    ) -> StepResult:
        """Run a step as an explicit team formation."""
        coordinator = self._create_coordinator()
        members = self._build_members(execution_plan, team_id, current_step=step)
        supervisor = members["plan_manager"]
        coordinator.set_formation(TeamFormation.HIERARCHICAL)
        coordinator.set_supervisor(supervisor)
        for member_id, member in members.items():
            if member_id != supervisor.id:
                coordinator.add_member(member)
        task = self._task_description_for_step(step, plan_state)
        result = await coordinator.execute_task(task, context)
        step_result = self._team_result_to_step_result(result)
        step_result.metadata["execution_mode"] = "team_node"
        return step_result

    async def _execute_loop_node(
        self,
        step: PlanStep,
        execution_plan: "ExecutionPlan",
        team_id: str,
        context: Dict[str, Any],
        plan_state: Dict[str, Any],
    ) -> StepResult:
        """Iterate over a plan-state collection, spawning one subagent per item.

        Items are resolved from (in priority order):
        1. ``step.context["items"]`` — static list embedded in the step
        2. ``plan_state[step.context["loop_over"]]`` — dynamic list from a prior step

        Execution is sequential by default to avoid exhausting provider retry budget.
        Set ``step.context["parallel"] = True`` to use concurrent execution.
        """
        items = self._resolve_loop_items(step, plan_state)
        if not items:
            return StepResult(
                success=True,
                output=f"Loop '{step.description}': no items to iterate — skipping.",
                tool_calls_used=0,
                metadata={
                    "execution_mode": "loop_node",
                    "loop_items_count": 0,
                    "loop_over": step.context.get("loop_over", ""),
                },
            )

        parallel = bool(step.context.get("parallel", False))
        subagents = self._subagents()
        parent_session_id = context.get("parent_session_id") or context.get("root_session_id")

        async def _run_item(index: int, item: str) -> tuple[str, StepResult]:
            item_task = f"{step.description} — [{item}]"
            evidence_guidance = self._evidence_guidance_for_step(step)
            if evidence_guidance:
                item_task = f"{item_task}\n\n{evidence_guidance}"
            # Append per-step exit criteria so the sub-agent has explicit completion
            # requirements and cannot self-terminate after a shallow scan.
            if step.exit_criteria:
                criteria_str = "; ".join(step.exit_criteria)
                item_task = f"{item_task}\n\nVerification criteria (must satisfy before completing): {criteria_str}"
            agent_id = f"{team_id}_{self._slug(step.id)}_loop_{index}"
            child_session_id = self._child_session_id(
                parent_session_id, team_id, f"{step.id}_loop_{index}"
            )
            spawn_result = await subagents.spawn(
                role=self._role_for_step(step),
                task=item_task,
                tool_budget=step.estimated_tool_calls,
                allowed_tools=get_step_allowed_tools(step),
                member_id=f"loop_{self._slug(step.id)}_{index}",
                agent_id=agent_id,
                display_name=f"Loop {index + 1}/{len(items)}: {item}",
                team_id=team_id,
                plan_id=execution_plan.id,
                plan_step_id=step.id,
                parent_session_id=parent_session_id,
                child_session_id=child_session_id,
            )
            item_result = StepResult(
                success=spawn_result.success,
                output=spawn_result.summary or "",
                error=spawn_result.error,
                tool_calls_used=spawn_result.tool_calls_used,
                duration_seconds=spawn_result.duration_seconds,
                metadata=dict(spawn_result.details),
            )
            return item, item_result

        if parallel:
            import asyncio as _asyncio

            pairs = await _asyncio.gather(*[_run_item(i, item) for i, item in enumerate(items)])
        else:
            pairs = []
            early_stopped = False
            for i, item in enumerate(items):
                pair = await _run_item(i, item)
                pairs.append(pair)
                # Exit criteria check after each iteration (sequential only)
                if step.exit_criteria:
                    accumulated = "\n".join(p[1].output for p in pairs)
                    if self._exit_criteria_met(step.exit_criteria, accumulated):
                        early_stopped = True
                        break
            else:
                early_stopped = False

        outputs: list[str] = []
        total_tool_calls = 0
        failed_items: list[str] = []
        for item, item_result in pairs:
            total_tool_calls += item_result.tool_calls_used
            if item_result.success:
                outputs.append(f"[{item}]\n{item_result.output}")
            else:
                failed_items.append(item)
                outputs.append(f"[{item}] FAILED: {item_result.error}")

        success = len(failed_items) == 0
        meta: Dict[str, Any] = {
            "execution_mode": "loop_node",
            "loop_items": items,
            "loop_items_count": len(items),
            "items_executed": len(pairs),
            "failed_items": failed_items,
            "parallel": parallel,
        }
        if not parallel:
            meta["early_stopped"] = early_stopped
        return StepResult(
            success=success,
            output="\n\n".join(outputs),
            error=f"Failed items: {failed_items}" if failed_items else None,
            tool_calls_used=total_tool_calls,
            metadata=meta,
        )

    @staticmethod
    def _resolve_loop_items(step: PlanStep, plan_state: Dict[str, Any]) -> list[str]:
        """Resolve the item collection for a loop node."""
        static_items = step.context.get("items", [])
        if static_items:
            return [str(i) for i in static_items]

        loop_over = step.context.get("loop_over", "")
        if loop_over and plan_state:
            raw = plan_state.get(loop_over)
            if isinstance(raw, list):
                return [str(i) for i in raw if str(i).strip()]
            if isinstance(raw, str):
                return [line.strip() for line in raw.splitlines() if line.strip()]

            # Fallback: find the plan_state key whose name overlaps most with
            # loop_over, then try to fall back to any non-empty list value.
            if raw is None:
                lo_words = {
                    w.rstrip("s") for w in loop_over.replace("_", " ").split() if len(w) > 3
                }
                best_val = None
                best_score = 0
                for k, v in plan_state.items():
                    if k.startswith("step_") or not isinstance(v, list) or not v:
                        continue
                    k_words = {w.rstrip("s") for w in k.replace("_", " ").split()}
                    score = len(lo_words & k_words)
                    if score > best_score:
                        best_score = score
                        best_val = v
                if best_val is not None:
                    return [str(i) for i in best_val if str(i).strip()]
        return []

    @staticmethod
    def _exit_criteria_met(criteria: list[str], output: str) -> bool:
        """Return True when all exit criteria appear in accumulated output."""
        output_lower = output.lower()
        return all(c.lower() in output_lower for c in criteria)

    # ---------------------------------------------------------------------------
    # Conditional node
    # ---------------------------------------------------------------------------

    def _execute_conditional_node(
        self,
        step: PlanStep,
        plan_state: Dict[str, Any],
    ) -> StepResult:
        """Evaluate a plan-state condition and record which branch to skip.

        The ``skip_step_ids`` list in the result metadata is read by the runtime
        to mark inactive branch steps as SKIPPED before the next iteration.

        Context keys:
          ``condition_on`` — plan_state key to evaluate
          ``condition``    — ``"non_empty"`` (default), ``"multiple"``,
                             ``"single"``, ``"empty"``, or ``"truthy"``
          ``produces``     — optional plan_state key to store the bool result
          ``branches``     — ``{"true": [step_ids], "false": [step_ids]}``
        """
        import logging as _logging

        _log = _logging.getLogger(__name__)

        condition_on = step.context.get("condition_on", "")
        condition = step.context.get("condition", "non_empty")
        branches: Dict[str, Any] = step.context.get("branches", {})
        value = plan_state.get(condition_on) if condition_on else None
        value_source = (
            f"plan_state['{condition_on}']" if condition_on and value is not None else "none"
        )
        # Fallback A: key name mismatch — find a plan_state key whose words
        # overlap with condition_on (handles LLM naming variation).
        if condition_on and value is None:
            _log.info(
                "Conditional step %s: key '%s' missing from plan_state %s; trying word-overlap fallback",
                step.id,
                condition_on,
                [k for k in plan_state if not k.startswith("step_")],
            )
            cond_words = {
                w.rstrip("s") for w in condition_on.replace("_", " ").split() if len(w) > 3
            }
            for k, v in plan_state.items():
                if not k.startswith("step_") and isinstance(v, list) and v:
                    key_words = {w.rstrip("s") for w in k.replace("_", " ").split()}
                    if cond_words & key_words:
                        value = v
                        value_source = f"fallback_word_overlap['{k}']"
                        _log.info(
                            "Conditional step %s: resolved '%s' via word-overlap key '%s' (%d items)",
                            step.id,
                            condition_on,
                            k,
                            len(v),
                        )
                        break

        # Fallback B: condition_on is empty or still unresolved — pick the first
        # non-empty list in plan_state that isn't a raw step dump (step_N keys).
        # This handles planning schemas where the LLM never sets condition_on.
        if value is None:
            for k, v in plan_state.items():
                if not k.startswith("step_") and isinstance(v, list) and v:
                    value = v
                    value_source = f"fallback_first_list['{k}']"
                    _log.info(
                        "Conditional step %s: condition_on='%s' unresolved; "
                        "using first plan_state list '%s' (%d items) as fallback",
                        step.id,
                        condition_on or "<empty>",
                        k,
                        len(v),
                    )
                    break

        result = self._evaluate_condition(condition, value)

        inactive = "false" if result else "true"
        skip_ids = [str(s) for s in (branches.get(inactive) or [])]

        produces = step.context.get("produces", "")
        if produces:
            plan_state[produces] = result

        label = f"'{condition}' on '{condition_on}'" if condition_on else f"'{condition}'"
        _log.info(
            "Conditional step %s: %s = %s (value_source=%s, value=%r) → active=%s, skip=%s",
            step.id,
            label,
            result,
            value_source,
            value,
            "true" if result else "false",
            skip_ids,
        )
        return StepResult(
            success=True,
            output=f"Condition {label}: {result} — skipping {skip_ids if skip_ids else 'nothing'}.",
            tool_calls_used=0,
            metadata={
                "execution_mode": "conditional_node",
                "condition": condition,
                "condition_on": condition_on,
                "condition_result": result,
                "value_source": value_source,
                "skip_step_ids": skip_ids,
                "active_branch": "true" if result else "false",
            },
        )

    @staticmethod
    def _collection_size(value: Any) -> int:
        if isinstance(value, (list, tuple, set, dict, str)):
            return len(value)
        return 1 if value is not None else 0

    @staticmethod
    def _review_targets_value_looks_unreliable(value: Any) -> bool:
        """Detect prose/list-extraction artifacts masquerading as review targets.

        A plausible target identifier is a slash-separated path or a single
        identifier token (snake_case, kebab-case, or a directory name).  Prose
        sentences ("the workspace contains three packages...") fail this check.
        """
        if not isinstance(value, list) or not value:
            return False
        cleaned = [str(item).strip() for item in value if str(item).strip()]
        if not cleaned:
            return True
        plausible_re = re.compile(r"^[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*$")
        plausible = [item for item in cleaned if plausible_re.match(item)]
        prose_markers = (
            "workspace",
            "contains",
            "identified",
            "found",
            "members",
            "packages",
            "modules",
            "components",
            "targets",
            "source files",
        )
        if len(cleaned) == 1 and any(marker in cleaned[0].lower() for marker in prose_markers):
            return True
        return len(plausible) < len(cleaned)

    @staticmethod
    def _evaluate_condition(condition: str, value: Any) -> bool:
        """Evaluate a named condition against a plan-state value."""
        if condition == "non_empty" or condition == "truthy":
            return bool(value)
        if condition == "empty":
            return not bool(value)
        if condition == "multiple":
            return isinstance(value, (list, tuple)) and len(value) > 1
        if condition == "single":
            return isinstance(value, (list, tuple)) and len(value) == 1
        return bool(value)

    # ---------------------------------------------------------------------------
    # Approval node
    # ---------------------------------------------------------------------------

    async def _execute_approval_node(
        self,
        step: PlanStep,
        context: Dict[str, Any],
    ) -> StepResult:
        """Pause execution for a user approval checkpoint.

        When ``_approval_callback`` is set (injected at construction), it is called
        as ``await callback(step, context)`` and must return ``(approved: bool,
        feedback: str)``.  When unset, the checkpoint auto-approves so the plan
        continues without interruption — suitable for non-interactive contexts.
        """
        if self._approval_callback is not None:
            try:
                approved, feedback = await self._approval_callback(step, context)
            except Exception as exc:
                return StepResult(
                    success=False,
                    output="",
                    error=f"Approval callback raised: {exc}",
                    metadata={"execution_mode": "approval_node", "approved": False},
                )
            return StepResult(
                success=approved,
                output=feedback
                or (
                    f"Approved: {step.description}" if approved else f"Rejected: {step.description}"
                ),
                error=None if approved else f"User rejected: {feedback}",
                tool_calls_used=0,
                metadata={
                    "execution_mode": "approval_node",
                    "approved": approved,
                    "feedback": feedback,
                },
            )

        # No callback — auto-approve with a checkpoint marker
        return StepResult(
            success=True,
            output=f"Approval checkpoint (auto-approved): {step.description}",
            tool_calls_used=0,
            metadata={
                "execution_mode": "approval_node",
                "approved": True,
                "auto_approved": True,
            },
        )

    _SYNTHESIS_KEYWORDS_RE = re.compile(
        r"\b(synthesize|summarize|compile|report|document|write up|present)\b",
        re.IGNORECASE,
    )
    _ARTIFACT_PRODUCES_RE = re.compile(
        r"(?:^|_)(?:checklist|report|summary|findings)(?:_|$)",
        re.IGNORECASE,
    )

    @staticmethod
    def _is_synthesis_step(step: PlanStep) -> bool:
        """Return True when this step assembles/writes a report from prior findings."""
        if "write" in (step.allowed_tools or []):
            return True
        return bool(
            PlanningTeamExecutionAdapter._SYNTHESIS_KEYWORDS_RE.search(step.description or "")
        )

    @staticmethod
    def _produces_output_contract(produces_key: str) -> str:
        """Return the final-answer contract for a step that writes plan_state.

        List-like producers feed loops and conditionals, so they need compact
        line-oriented items.  Artifact producers such as checklist/report/findings
        feed humans or synthesis steps and must return the actual content, not a
        progress note or a single label.
        """
        key = str(produces_key or "").strip()
        if PlanningTeamExecutionAdapter._ARTIFACT_PRODUCES_RE.search(key):
            return (
                "OUTPUT FORMAT (required): return the concrete artifact for "
                f"plan_state key '{key}' as Markdown. Include clear section headings "
                "and bullet or numbered items with enough detail for the next step to "
                "use directly. Do not return a status update, progress note, intent "
                "statement, or a single-line placeholder. If there is genuinely "
                "nothing to report, output exactly: (none)"
            )
        return (
            "OUTPUT FORMAT (required): respond with a plain list — one item "
            "per line, no bullet symbols, no prose introduction, no trailing "
            f"commentary. Each line must be a single '{key}' item "
            "(e.g. a file path, identifier, or short label). If you have "
            "nothing to list, output exactly: (none)"
        )

    @staticmethod
    def _task_description_for_step(
        step: PlanStep,
        plan_state: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Return the task string for a worker, augmenting with prior-step context
        and output-format instructions when the step must produce a named list.

        Context injection priority:
          1. step.inputs declared — inject only those specific named outputs (precise)
          2. Synthesis/write step — inject all named outputs + truncated step_N dumps
          3. Default — inject all named outputs (non-step_N keys) only
        """
        task = step.description
        is_synthesis = PlanningTeamExecutionAdapter._is_synthesis_step(step)
        partial_failed_dependencies = list(
            dict.fromkeys(
                str(dep)
                for dep in (step.context.get("partial_failed_dependencies", []) or [])
                if str(dep).strip()
            )
        )
        if partial_failed_dependencies:
            failed_list = ", ".join(partial_failed_dependencies)
            task = (
                f"{task}\n\n"
                "Partial execution context: the following prerequisite step(s) failed "
                f"or produced insufficient evidence: {failed_list}. Continue using the "
                "available evidence, do not invent missing findings, and explicitly "
                "call out any coverage gaps in your output."
            )

        if plan_state:
            declared_inputs: list[str] = getattr(step, "inputs", []) or []
            context_lines: list[str] = []

            if declared_inputs:
                # Precise injection: only what this step declared it needs.
                for key in declared_inputs:
                    value = plan_state.get(key)
                    if value is None:
                        continue
                    if isinstance(value, list) and value:
                        items_str = ", ".join(str(v) for v in value[:20])
                        context_lines.append(f"- {key}: {items_str}")
                    elif str(value).strip():
                        context_lines.append(f"- {key}: {str(value).strip()[:400]}")
            else:
                # Fallback: inject all named outputs; for synthesis steps also include
                # truncated step_N content so the sub-agent can synthesize findings.
                for key, value in plan_state.items():
                    is_step_key = key.startswith("step_")
                    if is_step_key and not is_synthesis:
                        continue
                    if isinstance(value, list) and value:
                        items_str = ", ".join(str(v) for v in value[:20])
                        context_lines.append(f"- {key}: {items_str}")
                    elif isinstance(value, (str, bool, int, float)) and str(value).strip():
                        short = str(value).strip()
                        short = short[:600] if is_step_key else short[:200]
                        context_lines.append(f"- {key}: {short}")

            if context_lines:
                task = task + "\n\nContext from prior steps:\n" + "\n".join(context_lines)

        produces_key = step.context.get("produces", "")
        if produces_key:
            contract = PlanningTeamExecutionAdapter._produces_output_contract(produces_key)
            task = f"{task}\n\n{contract}"

        # Compute steps (execution="compute") that produce named output are dispatched
        # to the agent path when no registered handler exists.  Without guidance,
        # the model sometimes hallucinates tool names ("FirstResponderTool") or enters
        # a halted state instead of generating the structured content from knowledge.
        # Adding an explicit note suppresses that behaviour.
        step_execution = (step.execution or step.context.get("execution", "")).lower()
        if step_execution == "compute" and produces_key:
            task = (
                f"{task}\n\n"
                f"NOTE: This is a knowledge generation step — no tool calls are required. "
                f"Produce the content directly from your knowledge and reasoning. "
                f"Do not attempt to call any tools."
            )

        evidence_guidance = PlanningTeamExecutionAdapter._evidence_guidance_for_step(step)
        if evidence_guidance:
            task = f"{task}\n\n{evidence_guidance}"

        # Append exit criteria so the sub-agent has explicit completion requirements
        # and cannot self-terminate before satisfying them.  Loop iterations receive
        # these via _execute_loop_node; all other steps get them here.
        exit_criteria = getattr(step, "exit_criteria", []) or []
        if exit_criteria and step_execution != "loop":
            criteria_str = "; ".join(exit_criteria)
            task = (
                f"{task}\n\n"
                f"Verification criteria (must satisfy before completing): {criteria_str}"
            )

        return task

    @staticmethod
    def _evidence_guidance_for_step(step: PlanStep) -> str:
        """Return pre-run evidence guidance for analysis/review sub-agents.

        This mirrors the post-run evidence contract so agents know the success
        bar before they start. It intentionally avoids inventory/discovery steps,
        where listing files is the work.
        """
        description = str(step.description or "")
        desc = description.lower()
        step_type = getattr(step, "step_type", None)
        execution = (step.execution or step.context.get("execution", "")).lower()
        if execution in {"approval", "compute", "conditional", "tool", "checkpoint"}:
            return ""
        if step_type not in {StepType.RESEARCH, StepType.REVIEW}:
            return ""
        if re.search(r"\b(map|inventory|discover|enumerate|list|present|show)\b", desc):
            return ""
        if not re.search(r"\b(review|analy[sz]e|analysis|audit|inspect|scan)\b", desc):
            return ""

        minimum = (
            "at least 3 concrete findings"
            if re.search(
                r"\b(workspace|cross-(?:target|crate|module|package|component)|"
                r"codebase|crate|module|package|component|target)\b",
                desc,
            )
            else "concrete findings"
        )
        return (
            "Evidence requirements (must satisfy before completing): "
            "read the relevant source files, not only directory listings; "
            f"return {minimum} with file:line references when findings exist; "
            "if there are no findings, state `No findings` and name the files inspected; "
            "do not stop after inventory, planning, or an intent statement."
        )

    @staticmethod
    def _should_execute_step_directly(execution_plan: ExecutionPlan, step: PlanStep) -> bool:
        """Avoid a manager/worker hierarchy when a step has exactly one owner."""
        return step in execution_plan.steps

    @staticmethod
    def _step_context(
        *,
        plan: ReadableTaskPlan,
        execution_plan: ExecutionPlan,
        step: PlanStep,
        team_id: str,
        root_session_id: Optional[str],
    ) -> Dict[str, Any]:
        return {
            "root_session_id": root_session_id,
            "parent_session_id": root_session_id,
            "plan_id": execution_plan.id,
            "plan_step_id": step.id,
            "team_id": team_id,
            "formation": "direct_step_worker",
            "plan_name": plan.name,
            "plan_complexity": plan.complexity.value,
            "task_description": plan.desc,
            "step_description": step.description,
        }

    def _create_coordinator(self) -> Any:
        if self._coordinator_factory is not None:
            return self._coordinator_factory(self.orchestrator)

        from victor.teams import UnifiedTeamCoordinator

        return UnifiedTeamCoordinator(self.orchestrator)

    def _build_members(
        self,
        execution_plan: ExecutionPlan,
        team_id: str,
        *,
        current_step: Optional[PlanStep] = None,
    ) -> Dict[str, Any]:
        members: Dict[str, Any] = {}
        manager_allowed_tools = self._runtime_allowed_tools(current_step)
        manager = TeamMember(
            id="plan_manager",
            role=SubAgentRole.PLANNER,
            name="Plan Manager",
            goal=f"Coordinate and synthesize plan {execution_plan.id}",
            tool_budget=10,
            allowed_tools=manager_allowed_tools,
            agent_category=TeamAgentCategory.SUPERVISOR,
            is_manager=True,
            can_delegate=True,
        )
        members[manager.id] = self._adapt_member(manager, team_id, execution_plan.id)

        step_members = [current_step] if current_step is not None else execution_plan.steps
        for step in step_members:
            role = self._role_for_step(step)
            member_id = f"step_{self._slug(step.id)}_{role.value}"
            member = TeamMember(
                id=member_id,
                role=role,
                name=self._display_name(step, role),
                goal=step.description,
                tool_budget=step.estimated_tool_calls or 10,
                allowed_tools=self._runtime_allowed_tools(step),
                reports_to=manager.id,
            )
            members[member.id] = self._adapt_member(
                member,
                team_id,
                execution_plan.id,
                plan_step_id=step.id,
            )

        return members

    def _runtime_allowed_tools(self, step: Optional[PlanStep]) -> Optional[list[str]]:
        """Return step tools filtered to tools the parent runtime actually exposes.

        Plan generation may include useful-but-optional tools such as ``shell`` even
        when the active profile or runtime has not registered them.  Passing such
        tools into a constrained sub-agent advertises an unusable tool schema and
        causes ``Unknown or disabled tool`` skips.  When we can inspect the parent
        runtime, keep the step's order but drop unavailable tools.  If runtime
        availability cannot be inspected (tests, lightweight adapters), preserve the
        original plan hints.
        """
        if step is None:
            return None

        requested = get_step_allowed_tools(step)
        if not requested:
            return None

        available = self._available_parent_tools()
        if available is None:
            return requested

        filtered: list[str] = []
        dropped: list[str] = []
        for tool_name in requested:
            canonical = canonicalize_core_tool_name(str(tool_name))
            if canonical in available:
                if canonical not in filtered:
                    filtered.append(canonical)
            else:
                dropped.append(str(tool_name))

        if dropped:
            import logging as _logging

            _logging.getLogger(__name__).info(
                "Planning step %s: dropped unavailable tool hint(s): %s " "(available=%s)",
                getattr(step, "id", "?"),
                dropped,
                sorted(available)[:12],
            )

        return filtered

    def _available_parent_tools(self) -> Optional[set[str]]:
        """Best-effort snapshot of tool names enabled in the parent runtime."""
        candidates: list[Any] = []

        getter = getattr(self.orchestrator, "get_enabled_tools", None)
        if callable(getter):
            try:
                enabled = getter()
                if enabled is not None:
                    return {canonicalize_core_tool_name(str(name)) for name in enabled}
                raise RuntimeError("Parent enabled-tool lookup returned no result")
            except Exception as exc:
                raise RuntimeError("Cannot determine parent tool permissions") from exc

        tools = getattr(self.orchestrator, "tools", None) or getattr(
            self.orchestrator, "tool_registry", None
        )
        if tools is not None:
            candidates.append(tools)

        for registry in candidates:
            try:
                if hasattr(registry, "list_tools"):
                    names: set[str] = set()
                    for item in registry.list_tools():
                        name = item if isinstance(item, str) else getattr(item, "name", None)
                        if name:
                            names.add(canonicalize_core_tool_name(str(name)))
                    return names
                if hasattr(registry, "get_tool_names"):
                    names = registry.get_tool_names()
                    if names is not None:
                        return {canonicalize_core_tool_name(str(name)) for name in names}
                    raise RuntimeError("Parent registry returned no tool list")
            except Exception as exc:
                raise RuntimeError("Cannot determine parent tool permissions") from exc

        return None

    def _adapt_member(
        self,
        member: TeamMember,
        team_id: str,
        plan_id: str,
        *,
        plan_step_id: Optional[str] = None,
    ) -> TeamParticipant:
        async def _execute(task: str, context: Dict[str, Any]) -> Dict[str, Any]:
            subagents = self._subagents()
            parent_session_id = context.get("parent_session_id") or context.get("root_session_id")
            child_session_id = self._child_session_id(parent_session_id, team_id, member.id)
            agent_id = f"{team_id}_{member.id}"
            result = await subagents.spawn(
                role=member.role,
                task=task,
                tool_budget=member.tool_budget,
                allowed_tools=member.allowed_tools,
                member_id=member.id,
                agent_id=agent_id,
                display_name=member.name,
                team_id=team_id,
                plan_id=plan_id,
                plan_step_id=plan_step_id or context.get("plan_step_id"),
                parent_session_id=parent_session_id,
                child_session_id=child_session_id,
                timeout_seconds=max(300, min(1800, member.tool_budget * 30)),
            )
            metadata = dict(result.details)
            return {
                "success": result.success,
                "output": self._subagent_output_for_parent(result),
                "error": result.error,
                "metadata": metadata,
                "tool_calls_used": result.tool_calls_used,
                "duration_seconds": result.duration_seconds,
            }

        return TeamParticipant(member=member, executor=_execute)

    @staticmethod
    def _subagent_output_for_parent(result: Any) -> str:
        """Choose the most useful child output for plan-state extraction.

        The final assistant message is normally the right payload. When the agentic
        loop exits with a thin status line after substantial tool use, preserve that
        line but append a bounded digest of the tool evidence so generic ``produces``
        extraction and synthesis steps have concrete material to consume.
        """
        summary = str(getattr(result, "summary", "") or "").strip()
        details = dict(getattr(result, "details", {}) or {})
        full_response = str(details.get("full_response") or "").strip()
        output = summary or full_response
        evidence = details.get("tool_evidence") or {}
        evidence_summary = str(evidence.get("summary") or "").strip()
        tool_calls_used = int(getattr(result, "tool_calls_used", 0) or 0)

        if not evidence_summary or tool_calls_used < 3:
            return output

        weak_final = len(output) < 240 or bool(
            re.match(
                r"(?is)^\s*(?:done|completed|i will|i'll|let me|now let me|"
                r"sub-agent failed|insufficient progress|task failed)\b",
                output,
            )
        )
        if not weak_final:
            return output

        header = "Tool-backed evidence digest from this sub-agent:"
        if output:
            return f"{output}\n\n{header}\n{evidence_summary}"
        return f"{header}\n{evidence_summary}"

    # ---------------------------------------------------------------------------
    # Compute-node registry
    # ---------------------------------------------------------------------------

    #: Named deterministic compute nodes.  Each entry maps a node name to a
    #: callable that receives the PlanStep and returns a StepResult.  New
    #: deterministic plan actions should be added here rather than as heuristics.
    _COMPUTE_NODES: Dict[str, Callable[["PlanStep"], StepResult]] = {}

    #: No built-in language-specific nodes.  Language/domain checklists belong in
    #: verticals (e.g. victor-coding) and are registered via register_compute_node()
    #: inside VictorPlugin.register(context) at plugin init.
    _BUILTIN_NODES: frozenset = frozenset()

    # ---------------------------------------------------------------------------
    # Built-in compute node: review checklist artifact
    # ---------------------------------------------------------------------------
    #
    # Language-specific deterministic parsing (e.g. Cargo workspace member
    # extraction, npm/pnpm workspaces, Gradle multi-module) belongs in verticals,
    # which register their own compute nodes via ``register_compute_node()``.
    # The framework intentionally ships zero domain-specific scanners so that
    # behaviour stays the same across ecosystems and the planning prompt remains
    # language-agnostic.

    @staticmethod
    def _builtin_checklist_artifact(step: "PlanStep", plan_state: Dict[str, Any]) -> "StepResult":
        """Generate a deterministic, language-agnostic review checklist artifact.

        Categories are extracted from the step description (the planner must
        include the requested categories verbatim, e.g.
        ``"... covering: concurrency, error handling, allocation efficiency"``).
        When no categories can be extracted, a domain-agnostic default of generic
        evaluation dimensions is used.  Verticals that want language-specific
        categories should register a richer compute node and reference it via
        ``step.context["node"]`` in the plan.
        """
        del plan_state

        description = (step.description or "").strip()
        topic = description or "Review checklist"
        categories = PlanningTeamExecutionAdapter._extract_checklist_categories(description)
        category_source = "extracted_from_description"
        if not categories:
            categories = [
                "Correctness and behavioural invariants",
                "Performance and resource efficiency",
                "Concurrency and ordering",
                "Error handling and failure modes",
                "Maintainability and readability",
                "Security and trust boundaries",
                "Testability and observability",
                "Interface design and boundaries",
                "Dependency and configuration hygiene",
            ]
            category_source = "generic_default"

        lines = [
            f"# {topic}",
            "",
            "Use this checklist as the evaluation framework for downstream analysis.",
            "Record file:line evidence for every actionable finding.",
            "",
        ]
        for index, category in enumerate(categories, start=1):
            clean = category.strip(" .")
            if not clean:
                continue
            lines.extend(
                [
                    f"## {index}. {clean}",
                    f"- Check whether {clean.lower()} is relevant in each review target.",
                    "- Capture concrete evidence — code locations, observed behaviour, measurements.",
                    "- Record any performance, resource, correctness, or maintainability risk.",
                    "- Note concrete file paths and suggested fixes for actionable items.",
                    "",
                ]
            )

        return StepResult(
            success=True,
            output="\n".join(lines).strip(),
            tool_calls_used=0,
            metadata={
                "execution_mode": "builtin_compute",
                "compute_node": "_checklist_artifact",
                "node_type": "deterministic_planning_step",
                "category_count": len(categories),
                "category_source": category_source,
            },
        )

    @staticmethod
    def _extract_checklist_categories(description: str) -> list[str]:
        """Extract comma-separated checklist categories from a plan-step description."""
        if not description:
            return []
        match = re.search(r"\bcovering:\s*(.+)$", description, flags=re.IGNORECASE)
        raw = match.group(1) if match else description
        raw = re.sub(r"^create\s+(?:comprehensive\s+)?", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\b(?:best practices|checklist)\b", "", raw, flags=re.IGNORECASE)
        parts = re.split(r",|\band\b", raw)
        categories: list[str] = []
        for part in parts:
            item = re.sub(r"\([^)]*\)", "", part).strip(" -.;:")
            item = re.sub(r"\s+", " ", item)
            if len(item) < 4:
                continue
            if item.lower() not in {c.lower() for c in categories}:
                categories.append(item)
        return categories[:18]

    # ---------------------------------------------------------------------------
    # Built-in compute node: generic findings aggregation
    # ---------------------------------------------------------------------------
    #
    # ``_aggregate_target_findings`` is the framework's only deterministic
    # synthesis node.  It composes ``per_target_findings`` from plan_state into
    # a markdown digest without making any language-specific assumptions about
    # file extensions, package managers, manifest formats, or naming
    # conventions.  Anything richer (per-language hotspot scans, dependency
    # graph extraction, semantic recommendations) must live in a vertical and
    # register itself via ``register_compute_node()``.

    @staticmethod
    def _builtin_aggregate_target_findings(
        step: "PlanStep", plan_state: Dict[str, Any]
    ) -> "StepResult":
        """Aggregate ``per_target_findings`` (or compatible keys) into markdown."""
        del step

        candidate_keys = (
            "per_target_findings",
            "per_crate_findings",
            "per_module_findings",
            "per_package_findings",
            "per_component_findings",
        )
        items: list[str] = []
        chosen_key = ""
        for key in candidate_keys:
            raw = plan_state.get(key)
            if isinstance(raw, list) and raw:
                items = [str(item).strip() for item in raw if str(item).strip()]
                chosen_key = key
                break
            if isinstance(raw, str) and raw.strip():
                items = [
                    line.strip()
                    for line in raw.splitlines()
                    if line.strip() and line.strip() != "(none)"
                ]
                chosen_key = key
                break

        targets_raw = plan_state.get("review_targets") or plan_state.get("workspace_members") or []
        targets: list[str]
        if isinstance(targets_raw, list):
            targets = [str(t).strip() for t in targets_raw if str(t).strip()]
        elif isinstance(targets_raw, str):
            targets = [
                line.strip()
                for line in targets_raw.splitlines()
                if line.strip() and line.strip() != "(none)"
            ]
        else:
            targets = []

        lines = ["# Aggregated Review Findings", ""]
        if targets:
            lines.append(f"Reviewed {len(targets)} target(s).")
            lines.append("")
        if items:
            lines.append(f"## Per-target findings ({len(items)} item(s) from `{chosen_key}`)")
            lines.append("")
            for item in items[:200]:
                if item.startswith(("-", "*", "#")):
                    lines.append(item)
                else:
                    lines.append(f"- {item}")
            if len(items) > 200:
                lines.append(f"- … and {len(items) - 200} more item(s) truncated.")
        else:
            lines.append("## Per-target findings")
            lines.append("")
            lines.append(
                "- No per-target findings were available in plan_state; "
                "downstream synthesis steps should rely on their own evidence."
            )

        return StepResult(
            success=True,
            output="\n".join(lines).strip(),
            tool_calls_used=0,
            metadata={
                "execution_mode": "builtin_compute",
                "compute_node": "_aggregate_target_findings",
                "node_type": "deterministic_planning_step",
                "target_count": len(targets),
                "findings_count": len(items),
                "findings_source_key": chosen_key,
            },
        )

    @classmethod
    def register_compute_node(
        cls,
        name: str,
        fn: "Callable[..., StepResult]",
    ) -> None:
        """Register a named deterministic compute node.

        ``fn`` receives ``(step: PlanStep, plan_state: Dict[str, Any])`` so it
        can read outputs from prior tool or compute steps.  Single-argument
        callables ``fn(step)`` are also accepted for backward compatibility.
        """
        cls._COMPUTE_NODES[name] = fn

    # ---------------------------------------------------------------------------
    # Execution dispatch helpers
    # ---------------------------------------------------------------------------

    @classmethod
    def _compute_node_for_step(cls, step: PlanStep) -> Optional[str]:
        """Return a deterministic compute node name when one applies.

        Resolution is intentionally narrow and language-agnostic:

        1. Coordinative steps (``approval``, ``checkpoint``, ``conditional``)
           never use a compute node.
        2. ``execution == "compute"`` steps honour an explicit ``step.context["node"]``
           when that name is registered.  Without an explicit node, a compute step
           with a ``produces`` contract falls through to the agent path so the
           runtime can gather real evidence; non-producing compute steps fall back
           to the generic ``_generic_compute`` placeholder.
        3. As a small ergonomic exception, steps whose description clearly asks
           the planner to *create* a checklist artifact route to the built-in
           ``_checklist_artifact`` generator.  This is the only description-based
           routing the framework retains; all other domain-specific routing must
           be expressed by the planner via ``context.node``.

        Plugins that need language-specific deterministic behaviour (Rust crate
        scans, npm workspace parsing, Gradle module review, etc.) must register
        their nodes via ``register_compute_node`` and the planner must reference
        them explicitly via ``step.context["node"]``.  Substring matching on the
        step description is *not* used — it produced brittle routing that broke
        for paraphrased prompts and non-English descriptions.
        """
        execution = (step.execution or step.context.get("execution", "")).lower()
        produces_key = step.context.get("produces", "")
        desc = (step.description or "").lower()
        is_checklist_artifact = cls._is_checklist_artifact_step(step, produces_key, desc)
        if execution in {"approval", "checkpoint", "conditional"}:
            return None
        if execution == "compute":
            explicit_node = step.context.get("node", "")
            if explicit_node and (
                explicit_node in cls._COMPUTE_NODES or explicit_node in cls._BUILTIN_NODES
            ):
                return explicit_node
            if "_checklist_artifact" in cls._COMPUTE_NODES and is_checklist_artifact:
                return "_checklist_artifact"
            # Compute steps that must produce structured output (list/data) for downstream
            # steps cannot use the generic no-op placeholder — they need real tool execution.
            # Return None to let the step fall through to the agent execution path.
            if step.context.get("produces", ""):
                return None
            # Fallback: non-producing compute steps use the generic placeholder
            return "_generic_compute"

        # Non-compute steps: the ONLY description-based routing the framework
        # keeps is checklist generation, because checklist creation is intrinsic
        # to the planning loop (the planner uses it for self-orientation before
        # gathering evidence) and is fully language-agnostic.
        if "_checklist_artifact" in cls._COMPUTE_NODES and is_checklist_artifact:
            return "_checklist_artifact"

        return None

    @staticmethod
    def _is_checklist_artifact_step(step: PlanStep, produces_key: str, desc: str) -> bool:
        """Return True only for steps that create a checklist artifact.

        Analysis steps often say "evaluate against the checklist"; those must use
        tools and source evidence, not the deterministic checklist generator.
        """
        if "checklist" in produces_key.lower():
            return True
        if "checklist" not in desc:
            return False
        step_type = getattr(step, "step_type", None)
        if step_type not in {StepType.RESEARCH, StepType.IMPLEMENTATION}:
            return True
        return bool(
            re.search(r"\b(create|build|generate|write|draft|present)\b", desc)
            and not re.search(r"\b(against|using|with|apply|evaluate)\b.{0,40}\bchecklist\b", desc)
        )

    @classmethod
    def _execute_compute_node(
        cls,
        step: PlanStep,
        compute_node: str,
        plan_state: Optional[Dict[str, Any]] = None,
    ) -> StepResult:
        """Dispatch to a registered compute node or return a generic placeholder.

        Language/domain-specific nodes must be registered by verticals via
        register_compute_node().  Unknown node names produce a generic result
        so deterministic steps never block execution on missing domain content.

        ``plan_state`` is forwarded to the registered function so compute nodes
        can read outputs produced by earlier tool or compute steps.  The
        registered function signature should be ``fn(step, plan_state) -> StepResult``
        but callables that only accept ``step`` are also supported (plan_state
        is passed as a keyword argument and ignored if the function doesn't
        declare it).
        """
        if compute_node in cls._COMPUTE_NODES:
            fn = cls._COMPUTE_NODES[compute_node]
            try:
                return fn(step, plan_state or {})
            except TypeError:
                # Registered function uses the old single-argument signature.
                return fn(step)
        # Generic fallback — no domain content; the agent step that follows can
        # elaborate if needed, or the vertical can register a named handler.
        description = (step.description or "").strip()
        output = f"Compute step: {description}"
        if "present" in description.lower():
            output = f"Step ready for review: {description}"
        return StepResult(
            success=True,
            output=output,
            tool_calls_used=0,
            metadata={
                "execution_mode": "compute_node",
                "compute_node": compute_node,
                "node_type": "deterministic_planning_step",
                "registered": compute_node in cls._COMPUTE_NODES,
            },
        )

    def _subagents(self) -> SubAgentOrchestrator:
        if self.sub_agent_orchestrator is None:
            self.sub_agent_orchestrator = SubAgentOrchestrator(self.orchestrator)
        return self.sub_agent_orchestrator

    @staticmethod
    def _team_result_to_step_result(result: Any) -> StepResult:
        if isinstance(result, TeamResult):
            success = result.success
            output = result.final_output
            error = result.error
            tool_calls = result.total_tool_calls
            metadata = {
                "execution_mode": "hierarchical_team",
                "formation": getattr(result.formation, "value", str(result.formation)),
                "member_results": {
                    member_id: {
                        "success": member.success,
                        "error": member.error,
                        "tool_calls_used": member.tool_calls_used,
                    }
                    for member_id, member in result.member_results.items()
                },
            }
            fallback = PlanningTeamExecutionAdapter._successful_worker_fallback(
                result.member_results
            )
            if not success and fallback is not None:
                success = True
                output = fallback
                error = None
                metadata["used_successful_worker_fallback"] = True
        else:
            success = bool(result.get("success"))
            output = str(result.get("final_output", ""))
            error = result.get("error")
            tool_calls = int(result.get("total_tool_calls", 0) or 0)
            metadata = {
                "execution_mode": "hierarchical_team",
                "member_results": result.get("member_results", {}),
            }
            fallback = PlanningTeamExecutionAdapter._successful_worker_fallback(
                result.get("member_results", {})
            )
            if not success and fallback is not None:
                success = True
                output = fallback
                error = None
                metadata["used_successful_worker_fallback"] = True

        return StepResult(
            success=success,
            output=output,
            error=error,
            tool_calls_used=tool_calls,
            artifacts=[],
            metadata=metadata,
        )

    @staticmethod
    def _member_payload_to_step_result(payload: Any, member_id: str) -> StepResult:
        if not isinstance(payload, dict):
            return StepResult(
                success=True,
                output=str(payload),
                metadata={
                    "execution_mode": "direct_step_worker",
                    "member_id": member_id,
                },
            )

        metadata = dict(payload.get("metadata") or {})
        metadata.setdefault("execution_mode", "direct_step_worker")
        metadata.setdefault("member_id", member_id)
        tool_evidence = metadata.get("tool_evidence") or {}
        if isinstance(tool_evidence, dict) and tool_evidence.get("tool_names"):
            metadata.setdefault("tool_names_used", list(tool_evidence.get("tool_names") or []))
        artifacts = metadata.get("artifacts") or metadata.get("changed_files") or []
        if not isinstance(artifacts, list):
            artifacts = [str(artifacts)]
        return StepResult(
            success=bool(payload.get("success")),
            output=str(payload.get("output", "") or ""),
            error=payload.get("error"),
            tool_calls_used=int(payload.get("tool_calls_used", 0) or 0),
            duration_seconds=float(payload.get("duration_seconds", 0.0) or 0.0),
            artifacts=[str(artifact) for artifact in artifacts],
            metadata=metadata,
        )

    @staticmethod
    def _successful_worker_fallback(member_results: Any) -> Optional[str]:
        """Return worker output when manager failed but every worker succeeded."""
        if not member_results:
            return None
        items = dict(member_results).items()
        worker_results = [
            member_result for member_id, member_result in items if member_id != "plan_manager"
        ]
        if not worker_results:
            return None
        if not all(
            PlanningTeamExecutionAdapter._member_success(member) for member in worker_results
        ):
            return None
        return "\n\n".join(
            output
            for output in (
                PlanningTeamExecutionAdapter._member_output(member) for member in worker_results
            )
            if output
        )

    @staticmethod
    def _member_success(member_result: Any) -> bool:
        if isinstance(member_result, dict):
            return bool(member_result.get("success"))
        return bool(getattr(member_result, "success", False))

    @staticmethod
    def _member_output(member_result: Any) -> str:
        if isinstance(member_result, dict):
            return str(member_result.get("output", "") or "")
        return str(getattr(member_result, "output", "") or "")

    @staticmethod
    def _role_for_step(step: PlanStep) -> SubAgentRole:
        role = (step.sub_agent_role or "").lower()
        role_map = {
            "researcher": SubAgentRole.RESEARCHER,
            "planner": SubAgentRole.PLANNER,
            "executor": SubAgentRole.EXECUTOR,
            "reviewer": SubAgentRole.REVIEWER,
            "tester": SubAgentRole.TESTER,
        }
        if role in role_map:
            return role_map[role]
        step_type_map = {
            StepType.RESEARCH: SubAgentRole.RESEARCHER,
            StepType.PLANNING: SubAgentRole.PLANNER,
            StepType.IMPLEMENTATION: SubAgentRole.EXECUTOR,
            StepType.TESTING: SubAgentRole.TESTER,
            StepType.REVIEW: SubAgentRole.REVIEWER,
            StepType.DEPLOYMENT: SubAgentRole.EXECUTOR,
        }
        return step_type_map.get(step.step_type, SubAgentRole.EXECUTOR)

    @staticmethod
    def _display_name(step: PlanStep, role: SubAgentRole) -> str:
        return build_display_name(role, task=step.description, ordinal=step.id)

    @staticmethod
    def _slug(value: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", str(value)).strip("_").lower()
        return slug or "step"

    @classmethod
    def _team_id(cls, plan_id: str, step_id: str) -> str:
        return f"team_{cls._slug(plan_id)}_{cls._slug(step_id)}"

    @staticmethod
    def _child_session_id(parent_session_id: Optional[str], team_id: str, member_id: str) -> str:
        parent = parent_session_id or "session"
        return f"{parent}:{team_id}:{member_id}"

    def _root_session_id(self) -> Optional[str]:
        return (
            getattr(self.orchestrator, "active_session_id", None)
            or getattr(self.orchestrator, "session_id", None)
            or getattr(self.orchestrator, "_memory_session_id", None)
        )


# Register the two domain-agnostic compute nodes the framework ships.  Both are
# safe to use across any language or ecosystem.  Verticals (e.g. victor-coding)
# register language-specific scanners by calling ``register_compute_node()``
# inside ``VictorPlugin.register(context)`` at plugin init.
#
# Intentionally NOT registered here:
#   * Cargo / Rust workspace parsers and hotspot scanners (live in victor-coding)
#   * Any other language-specific deterministic node
PlanningTeamExecutionAdapter.register_compute_node(
    "_checklist_artifact",
    PlanningTeamExecutionAdapter._builtin_checklist_artifact,
)
PlanningTeamExecutionAdapter.register_compute_node(
    "_aggregate_target_findings",
    PlanningTeamExecutionAdapter._builtin_aggregate_target_findings,
)
