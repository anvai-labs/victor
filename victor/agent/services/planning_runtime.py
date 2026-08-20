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

"""Canonical service-owned planning runtime implementation.

This module provides `PlanningRuntimeService` as the active implementation for
structured planning during chat execution. `PlanningCoordinator` remains as a
compatibility alias for older imports and attributes.

Key Features:
- Auto-detects complex multi-step tasks that benefit from planning
- Generates structured plans using ReadableTaskPlan schema
- Executes plans step-by-step with context-aware tools
- Provides progress reporting and plan state management
- Falls back to regular chat for simple tasks

Design:
--------
The coordinator uses a complexity threshold to decide when to use planning:
- SIMPLE tasks: Direct chat (fast, minimal overhead)
- MODERATE/COMPLEX tasks: Plan-based execution (structured, reliable)

For plan-based tasks:
1. Generate plan using generate_task_plan() with token-efficient schema
2. Show plan to user
3. Execute steps sequentially with StepAwareToolSelector
4. Report progress and handle failures
5. Complete with summary

Example:
    service = PlanningRuntimeService(orchestrator)

    # Will auto-detect and plan if complex
    response = await service.chat_with_planning(
        "Analyze the Victor codebase architecture and provide SOLID evaluation"
    )
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, TYPE_CHECKING

from victor.agent.planning.constants import (
    COMPLEXITY_KEYWORDS,
    DEFAULT_MIN_PLANNING_COMPLEXITY,
    DEFAULT_MIN_STEPS_THRESHOLD,
    DEFAULT_MIN_KEYWORD_MATCHES,
    STEP_INDICATORS,
)
from victor.agent.planning.readable_schema import (
    ReadableTaskPlan,
    TaskComplexity as PlanningTaskComplexity,
    generate_task_plan,
    precompute_plan_inference_embeddings,
)
from victor.agent.task_analyzer import TaskAnalysis
from victor.agent.services.context_service import compact_context_if_recommended
from victor.framework.execution_checkpoint import ApprovalState
from victor.framework.task import TaskComplexity
from victor.providers.base import CompletionResponse

if TYPE_CHECKING:
    from victor.agent.services.protocols.chat_runtime import PlanningContextProtocol
    from victor.agent.planning.base import ExecutionPlan, PlanResult
    from victor.agent.planning.team_execution import PlanningTeamExecutionAdapter
    from victor.ui.rendering.protocol import StreamRenderer

logger = logging.getLogger(__name__)

__all__ = [
    "PlanningRuntimeService",
    "PlanningCoordinator",
    "PlanningConfig",
    "PlanningMode",
    "PlanningResult",
]


class PlanningMode(Enum):
    """Planning mode for chat requests."""

    AUTO = "auto"  # Automatically decide based on task complexity
    ALWAYS = "always"  # Always use planning
    NEVER = "never"  # Never use planning (direct chat only)


@dataclass
class PlanningConfig:
    """Configuration for planning behavior."""

    # Minimum complexity level to trigger planning
    # Use framework TaskComplexity (simple/medium/complex)
    min_planning_complexity: TaskComplexity = TaskComplexity.MEDIUM

    # Thresholds for detecting multi-step tasks
    min_steps_threshold: int = DEFAULT_MIN_STEPS_THRESHOLD
    min_keyword_matches: int = DEFAULT_MIN_KEYWORD_MATCHES
    complexity_keywords: List[str] = field(default_factory=lambda: list(COMPLEXITY_KEYWORDS))
    step_indicators: List[str] = field(default_factory=lambda: list(STEP_INDICATORS))

    # Planning behavior
    show_plan_before_execution: bool = True  # Require user to see plan first
    # Planning displays the proposed work, then starts execution. Tool/runtime
    # approval gates still protect escalated or destructive actions.
    auto_approve: bool = True
    allow_plan_modification: bool = False  # Allow user to modify plan (future)
    max_parallel_steps: int = 3  # Max independent plan steps to execute concurrently

    # Fallback behavior
    fallback_on_planning_failure: bool = True  # Fall back to direct chat if planning fails
    max_planning_retries: int = 1  # Number of retries for plan generation


@dataclass
class PlanningResult:
    """Result from planning-based chat execution."""

    mode: str  # "planned" or "direct"
    plan: Optional[ReadableTaskPlan] = None
    execution_result: Optional["PlanResult"] = None
    response: Optional[CompletionResponse] = None
    steps_completed: int = 0
    steps_total: int = 0
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        """Check if execution was successful."""
        if self.mode == "direct":
            return self.response is not None
        return self.execution_result is not None and self.execution_result.success


@dataclass(frozen=True)
class PlanApprovalDecision:
    """Decision after displaying a plan to the user."""

    proceed: bool
    user_approved_execution: bool = False
    reason: str = ""
    approval_state: ApprovalState = ApprovalState.NOT_REQUIRED


class _PlanProgressDisplay:
    """Best-effort Rich display for live plan step status."""

    _STATUS_STYLE = {
        "pending": "dim",
        "in_progress": "bright_cyan",
        "completed": "white",
        "failed": "red",
        "skipped": "yellow",
        "blocked": "yellow",
    }

    _STATUS_LABEL = {
        "pending": "pending",
        "in_progress": "running",
        "completed": "done",
        "failed": "failed",
        "skipped": "skipped",
        "blocked": "blocked",
    }

    def __init__(self, plan: ReadableTaskPlan, execution_plan: Any, console: Any = None) -> None:
        self._plan = plan
        self._execution_plan = execution_plan
        self._console = console
        self._last_statuses: dict[str, str] = {}
        self._step_started_at: dict[str, float] = {}
        self._step_elapsed: dict[str, float] = {}
        self._live = None
        self._started = False

    def start(self) -> None:
        try:
            from rich.console import Console
            from rich.live import Live

            self._console = self._console or Console()
            self._last_statuses = {
                str(getattr(step, "id", "")): getattr(
                    getattr(step, "status", None), "value", "pending"
                )
                for step in list(getattr(self._execution_plan, "steps", []) or [])
            }
            self._live = Live(
                self._render_graph(title="Execution Graph"),
                console=self._console,
                auto_refresh=False,
                transient=False,
            )
            self._live.start(refresh=True)
            self._started = True
        except Exception as exc:
            logger.debug("Plan progress display disabled: %s", exc)
            self._live = None
            self._started = False

    def update(self) -> None:
        if not self._started or self._console is None:
            return
        try:
            for step in list(getattr(self._execution_plan, "steps", []) or []):
                step_id = str(getattr(step, "id", ""))
                status = getattr(getattr(step, "status", None), "value", "pending")
                if self._last_statuses.get(step_id) == status:
                    continue
                previous_status = self._last_statuses.get(step_id)
                self._last_statuses[step_id] = status
                if status == "pending":
                    continue
                if status == "in_progress":
                    self._step_started_at.setdefault(step_id, time.monotonic())
                    continue
                elif status in {"completed", "failed", "skipped", "blocked"}:
                    started_at = self._step_started_at.get(step_id)
                    if started_at is not None:
                        self._step_elapsed[step_id] = max(0.0, time.monotonic() - started_at)
                    elif previous_status == "pending":
                        self._step_elapsed.setdefault(step_id, 0.0)
                    if self._live is not None:
                        self._live.update(
                            self._render_graph(title=f"Step {step_id} {status}"),
                            refresh=True,
                        )
        except Exception as exc:
            logger.debug("Plan progress display update failed: %s", exc)

    async def tick_until(self, awaitable: Any, interval: float = 1.0) -> Any:
        """Await work while refreshing live elapsed time for running steps."""
        task = asyncio.ensure_future(awaitable)
        try:
            while not task.done():
                await asyncio.sleep(interval)
                self.update()
            return await task
        finally:
            if not task.done():
                task.cancel()

    def stop(self) -> None:
        if not self._started or self._console is None:
            return
        try:
            final_graph = self._render_graph(title="Final Execution Graph")
            if self._live is not None:
                self._live.update(final_graph, refresh=True)
                self._live.stop()
            else:
                self._console.print(final_graph)
        except Exception as exc:
            logger.debug("Plan progress display stop failed: %s", exc)
        finally:
            self._live = None
            self._started = False

    def _render_graph(self, *, title: str) -> Any:
        from rich.console import Group
        from rich.markup import escape
        from rich.panel import Panel
        from rich.text import Text

        lines: list[Any] = []
        steps = list(getattr(self._execution_plan, "steps", []) or [])
        by_id = {str(getattr(step, "id", "")): step for step in steps}
        successors = self._build_reduced_successors(steps, by_id)
        reduced_predecessors = self._invert_successors(successors)
        levels = self._compute_dag_levels(steps, reduced_predecessors)
        order = {str(getattr(step, "id", "")): index for index, step in enumerate(steps)}

        status_counts = self._count_statuses(steps)
        completed = status_counts.get("completed", 0)
        failed = status_counts.get("failed", 0)
        skipped = status_counts.get("skipped", 0)
        blocked = status_counts.get("blocked", 0)
        running = status_counts.get("in_progress", 0)
        pending = status_counts.get("pending", 0)
        terminal = completed + skipped
        done_label = (
            f"{completed}/{len(steps)} done"
            if skipped == 0
            else f"{terminal}/{len(steps)} terminal ({completed} done, {skipped} skipped)"
        )
        lines.append(
            Text(
                (
                    f"{done_label}"
                    f"  {running} running"
                    f"  {pending} pending"
                    f"  {blocked} blocked"
                    f"  {failed} failed"
                ),
                style="dim",
            )
        )
        slowest = self._format_slowest_steps(by_id)
        if slowest:
            lines.append(Text("slowest: " + slowest, style="dim"))
        lines.append(Text(""))

        for level in sorted(levels):
            step_ids = sorted(levels[level], key=lambda step_id: order.get(step_id, 10_000))
            if len(step_ids) == 1:
                step_id = step_ids[0]
                lines.append(
                    self._format_progress_bullet(
                        by_id.get(step_id),
                        deps_override=reduced_predecessors.get(step_id, []),
                    )
                )
                continue

            lines.append(Text(f"• parallel group ({len(step_ids)} steps)", style="bold dim"))
            for step_id in step_ids:
                lines.append(
                    self._format_progress_bullet(
                        by_id.get(step_id),
                        prefix="  - ",
                        deps_override=reduced_predecessors.get(step_id, []),
                    )
                )

        if lines and isinstance(lines[-1], Text) and not lines[-1].plain:
            lines.pop()
        return Panel(
            Group(*lines),
            title=f"{title}: {escape(self._plan.name)}",
            border_style="cyan",
            expand=True,
        )

    @staticmethod
    def _build_reduced_successors(
        steps: list[Any],
        by_id: dict[str, Any],
    ) -> dict[str, list[str]]:
        successors: dict[str, list[str]] = {step_id: [] for step_id in by_id}
        for step in steps:
            step_id = str(getattr(step, "id", ""))
            for dep in list(getattr(step, "depends_on", []) or []):
                dep_id = str(dep)
                if dep_id in by_id and step_id not in successors.setdefault(dep_id, []):
                    successors[dep_id].append(step_id)

        reduced: dict[str, list[str]] = {}
        for parent_id, child_ids in successors.items():
            reduced[parent_id] = [
                child_id
                for child_id in child_ids
                if not _PlanProgressDisplay._has_alternate_path(
                    successors,
                    parent_id,
                    child_id,
                )
            ]
        return reduced

    @staticmethod
    def _has_alternate_path(
        successors: dict[str, list[str]],
        parent_id: str,
        target_id: str,
    ) -> bool:
        stack = [child for child in successors.get(parent_id, []) if child != target_id]
        seen: set[str] = set()
        while stack:
            current = stack.pop()
            if current == target_id:
                return True
            if current in seen:
                continue
            seen.add(current)
            stack.extend(successors.get(current, []))
        return False

    @staticmethod
    def _invert_successors(successors: dict[str, list[str]]) -> dict[str, list[str]]:
        predecessors: dict[str, list[str]] = {}
        for parent_id, child_ids in successors.items():
            for child_id in child_ids:
                predecessors.setdefault(child_id, []).append(parent_id)
        return predecessors

    @staticmethod
    def _compute_dag_levels(
        steps: list[Any],
        predecessors: dict[str, list[str]],
    ) -> dict[int, list[str]]:
        step_ids = [str(getattr(step, "id", "")) for step in steps]
        step_id_set = set(step_ids)
        level_by_id: dict[str, int] = {}

        for _ in range(max(1, len(step_ids))):
            changed = False
            for step_id in step_ids:
                preds = [pred for pred in predecessors.get(step_id, []) if pred in step_id_set]
                if not preds:
                    candidate_level = 0
                elif all(pred in level_by_id for pred in preds):
                    candidate_level = max(level_by_id[pred] for pred in preds) + 1
                else:
                    continue
                if level_by_id.get(step_id) != candidate_level:
                    level_by_id[step_id] = candidate_level
                    changed = True
            if not changed:
                break

        for step_id in step_ids:
            level_by_id.setdefault(step_id, 0)

        levels: dict[int, list[str]] = {}
        for step_id in step_ids:
            levels.setdefault(level_by_id[step_id], []).append(step_id)
        return levels

    @staticmethod
    def _count_statuses(steps: list[Any]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for step in steps:
            status = getattr(getattr(step, "status", None), "value", "pending")
            counts[status] = counts.get(status, 0) + 1
        return counts

    def _format_slowest_steps(self, by_id: dict[str, Any]) -> str:
        completed: list[tuple[float, str]] = []
        for step_id, elapsed in self._step_elapsed.items():
            step = by_id.get(step_id)
            status = getattr(getattr(step, "status", None), "value", "pending")
            if step is not None and status == "completed":
                completed.append((elapsed, step_id))
        completed.sort(reverse=True)
        return ", ".join(
            f"{step_id} {self._format_elapsed_value(elapsed)}" for elapsed, step_id in completed[:3]
        )

    def _format_progress_bullet(
        self,
        step: Any,
        prefix: str = "• ",
        deps_override: list[str] | None = None,
        max_description: int = 88,
    ) -> Any:
        from rich.text import Text

        if step is None:
            return Text(f"{prefix}[? missing]", style="red")

        step_id = str(getattr(step, "id", ""))
        status = getattr(getattr(step, "status", None), "value", "pending")
        label = self._STATUS_LABEL.get(status, status)
        style = self._STATUS_STYLE.get(status, "white")
        result = getattr(step, "result", None)
        tools = ""
        if result is not None:
            tools = f" tools={int(getattr(result, 'tool_calls_used', 0) or 0)}"
        elapsed_value = self._step_elapsed.get(step_id)
        if elapsed_value is None and status == "in_progress":
            started_at = self._step_started_at.get(step_id)
            if started_at is not None:
                elapsed_value = max(0.0, time.monotonic() - started_at)
        elapsed = self._format_elapsed(elapsed_value)
        deps = (
            deps_override
            if deps_override is not None
            else list(getattr(step, "depends_on", []) or [])
        )
        dep_text = f" after {','.join(str(dep) for dep in deps)}" if deps else ""
        description = str(getattr(step, "description", "") or "")
        if len(description) > max_description:
            description = description[: max(0, max_description - 3)] + "..."

        text = Text(prefix, style="dim")
        text.append(f"{step_id} {label}", style=style)
        if tools or dep_text or elapsed:
            meta = " ".join(part for part in (tools.strip(), elapsed.strip(), dep_text) if part)
            text.append(f" ({meta})", style="dim")
        text.append(f" {description}", style="white" if status != "pending" else "dim")
        return text

    def _format_graph_node(
        self,
        step: Any,
        prefix: str = "",
        deps_override: list[str] | None = None,
        repeated: bool = False,
        max_description: int = 96,
    ) -> Any:
        text = self._format_progress_bullet(
            step,
            prefix=prefix,
            deps_override=deps_override,
            max_description=max_description,
        )
        if repeated:
            text.append(" (shown above)", style="dim")
        return text

    @staticmethod
    def _format_elapsed(elapsed: float | None) -> str:
        if elapsed is None:
            return ""
        return f" elapsed={_PlanProgressDisplay._format_elapsed_value(elapsed)}"

    @staticmethod
    def _format_elapsed_value(elapsed: float) -> str:
        if elapsed < 1:
            return f"{elapsed * 1000:.0f}ms"
        if elapsed < 60:
            return f"{elapsed:.1f}s"
        minutes, seconds = divmod(elapsed, 60)
        return f"{int(minutes)}m{seconds:04.1f}s"


class PlanningRuntimeService:
    """Service-owned runtime helper for integrating chat with autonomous planning.

    This helper sits between chat execution and the AutonomousPlanner,
    automatically deciding when to use structured planning vs direct chat.

    Attributes:
        orchestrator: Parent orchestrator for LLM calls and tool execution
        config: Configuration for planning behavior
        active_plan: Currently active plan (if any)
    """

    def __init__(
        self,
        orchestrator: "PlanningContextProtocol",
        config: Optional[PlanningConfig] = None,
        renderer: Optional["StreamRenderer"] = None,  # NEW: Inject renderer for consistent UI
    ):
        """Initialize the planning coordinator.

        Args:
            orchestrator: Any object satisfying PlanningContextProtocol
            config: Optional configuration (uses defaults if not provided)
            renderer: Optional renderer for consistent plan display (falls back to Rich console)
        """
        self.orchestrator = orchestrator
        self.config = config or PlanningConfig()
        self.renderer = renderer  # NEW: Store renderer for consistent plan display
        self.active_plan: Optional[ReadableTaskPlan] = None
        self._planning_mode = PlanningMode.AUTO

        # Pre-compute archetype embeddings for exec-type inference so the hot
        # path never pays embedding latency.  This is a no-op when the embedding
        # service is not yet initialized.
        precompute_plan_inference_embeddings()

        logger.info(
            f"PlanningRuntimeService initialized with "
            f"min_complexity={self.config.min_planning_complexity.value}, "
            f"renderer={'injected' if renderer else 'console fallback'}"
        )

    async def chat_with_planning(
        self,
        user_message: str,
        task_analysis: Optional[TaskAnalysis] = None,
        mode: PlanningMode = PlanningMode.AUTO,
    ) -> CompletionResponse:
        """Chat with automatic planning for complex tasks.

        This is the main entry point. It analyzes the task and decides:
        - Simple tasks → direct chat (fast)
        - Complex tasks → plan → execute → summarize (reliable)

        Args:
            user_message: User's message
            task_analysis: Optional pre-computed task analysis
            mode: Planning mode (auto/always/never)

        Returns:
            CompletionResponse from the model
        """
        self._planning_mode = mode

        # Step 1: Analyze if planning is needed
        should_plan = self._should_use_planning(user_message, task_analysis)

        if not should_plan or mode == PlanningMode.NEVER:
            logger.info("Using direct chat mode")
            return await self._direct_chat(user_message)

        # Step 2: Generate plan
        logger.info(f"Using planning mode for: {user_message[:100]}...")
        try:
            # Compact context before plan generation to avoid overflow
            await self._compact_context_if_needed()
            plan = await self._generate_plan(user_message, task_analysis)
            self.active_plan = plan
        except Exception as e:
            logger.error(f"Plan generation failed: {e}")
            if self.config.fallback_on_planning_failure:
                logger.info("Falling back to direct chat")
                return await self._direct_chat(user_message)
            raise

        # Step 3: Show plan and potentially wait for approval
        user_approved = True
        if self.config.show_plan_before_execution:
            approval = await self._show_plan_to_user(plan)
            if not approval.proceed:
                # User rejected the plan
                logger.info("Plan rejected by user")
                # Return a response explaining the plan was rejected
                return await self._generate_plan_rejected_response(plan)
            user_approved = approval.user_approved_execution

        # Step 4: Execute plan (only if approved)
        result = await self._execute_plan(plan, user_approved=user_approved)

        # Step 5: Generate final response
        # Compact context before generating final summary to avoid overflow
        await self._compact_context_if_needed()
        response = await self._generate_final_response(plan, result)
        self._attach_planning_response_metadata(response, plan, result)

        return response

    def _map_complexity(self, framework_complexity: TaskComplexity) -> PlanningTaskComplexity:
        """Map framework TaskComplexity to planning TaskComplexity.

        Args:
            framework_complexity: Framework complexity level

        Returns:
            Planning TaskComplexity
        """
        # Map framework complexity to planning complexity
        # Framework: simple/medium/complex/generation/action/analysis
        # Planning: simple/moderate/complex
        complexity_map = {
            TaskComplexity.SIMPLE: PlanningTaskComplexity.SIMPLE,
            TaskComplexity.MEDIUM: PlanningTaskComplexity.MODERATE,
            TaskComplexity.COMPLEX: PlanningTaskComplexity.COMPLEX,
            TaskComplexity.GENERATION: PlanningTaskComplexity.COMPLEX,
            TaskComplexity.ACTION: PlanningTaskComplexity.MODERATE,
            TaskComplexity.ANALYSIS: PlanningTaskComplexity.COMPLEX,
        }
        return complexity_map.get(framework_complexity, PlanningTaskComplexity.MODERATE)

    def _should_use_planning(
        self,
        user_message: str,
        task_analysis: Optional[TaskAnalysis] = None,
    ) -> bool:
        """Determine if planning should be used for this task.

        Uses multiple signals:
        1. Task complexity from TaskAnalyzer
        2. Keyword analysis for multi-step indicators
        3. User's explicit planning mode

        Args:
            user_message: User's message
            task_analysis: Optional task analysis

        Returns:
            True if planning should be used
        """
        if self._planning_mode == PlanningMode.ALWAYS:
            return True

        if self._planning_mode == PlanningMode.NEVER:
            return False

        from victor.agent.planning.intent import is_explicit_planning_request

        if is_explicit_planning_request(user_message):
            logger.info("Planning triggered by explicit planning/checklist request")
            return True

        # Check task complexity if available
        if task_analysis:
            # Map framework complexity to planning complexity for threshold comparison
            planning_complexity = self._map_complexity(task_analysis.complexity)
            min_planning_complexity = self._map_complexity(self.config.min_planning_complexity)

            if planning_complexity.value >= min_planning_complexity.value:
                logger.info(f"Planning triggered by complexity: {planning_complexity.value}")
                return True

        # Check for multi-step keywords
        message_lower = user_message.lower()
        keyword_matches = sum(1 for kw in self.config.complexity_keywords if kw in message_lower)
        if keyword_matches >= self.config.min_keyword_matches:
            logger.info(f"Planning triggered by keywords: {keyword_matches} matches")
            return True

        # Check for explicit "step" language
        step_count = sum(
            1 for indicator in self.config.step_indicators if indicator in message_lower
        )
        if step_count >= self.config.min_steps_threshold:
            logger.info(f"Planning triggered by step indicators: {step_count} matches")
            return True

        return False

    async def _generate_plan(
        self,
        user_message: str,
        task_analysis: Optional[TaskAnalysis] = None,
    ) -> ReadableTaskPlan:
        """Generate a structured plan for the task.

        Args:
            user_message: User's message
            task_analysis: Optional task analysis

        Returns:
            ReadableTaskPlan with structured steps
        """
        # Determine complexity
        if task_analysis:
            complexity = self._map_complexity(task_analysis.complexity)
        else:
            complexity = PlanningTaskComplexity.MODERATE

        # Check for planning-specific provider/model override (priority: CLI > profile > default)
        planning_provider = self.orchestrator.provider
        planning_model = self.orchestrator.model

        # Try to get CLI override first (public attribute, set by CLI arg parsing)
        cli_planning_model = getattr(self.orchestrator, "planning_model_override", None)
        if cli_planning_model:
            logger.info(f"Using CLI planning model override: {cli_planning_model}")
            # Parse planning model (format: "model" or "provider:model")
            if ":" in cli_planning_model:
                planning_provider_name, planning_model = cli_planning_model.split(":", 1)
                from victor.providers.provider_factory import get_provider

                try:
                    planning_provider = get_provider(planning_provider_name)
                    logger.info(
                        f"Using planning provider from CLI override: {planning_provider_name}"
                    )
                except Exception as e:
                    logger.warning(f"Failed to get planning provider {planning_provider_name}: {e}")
            else:
                planning_model = cli_planning_model
        else:
            # Try profile override if no CLI override
            profile = getattr(self.orchestrator, "profile", None)
            if profile:
                planning_provider_override = getattr(profile, "planning_provider", None)
                planning_model_override = getattr(profile, "planning_model", None)

                if planning_provider_override:
                    logger.info(
                        f"Using planning provider override from profile: {planning_provider_override}"
                    )
                    from victor.providers.provider_factory import get_provider

                    try:
                        planning_provider = get_provider(planning_provider_override)
                    except Exception as e:
                        logger.warning(
                            f"Failed to get planning provider {planning_provider_override}: {e}"
                        )
                        logger.info("Falling back to default provider for planning")

                if planning_model_override:
                    logger.info(
                        f"Using planning model override from profile: {planning_model_override}"
                    )
                    planning_model = planning_model_override

        # Build skill-aware prompt from the shared framework skill matcher/capability.
        enriched_request = user_message
        try:
            matcher = getattr(self.orchestrator, "_skill_matcher", None) or getattr(
                self.orchestrator, "skill_matcher", None
            )
            if matcher and getattr(matcher, "initialized", getattr(matcher, "_initialized", False)):
                from victor.framework.skill_planner import (
                    build_skill_aware_plan_prompt,
                    build_skill_decomposition,
                    coerce_skill_catalog,
                )

                skills = coerce_skill_catalog(getattr(matcher, "skills", None))
                if skills:
                    decomposition = build_skill_decomposition(user_message, matcher)
                    enriched_request = build_skill_aware_plan_prompt(
                        user_message,
                        skills,
                        selected_skills=decomposition.skills if decomposition else None,
                        decomposition_confidence=(
                            decomposition.confidence if decomposition else None
                        ),
                    )
                    logger.debug(
                        "Plan generation enriched with %d skills%s",
                        len(skills),
                        (
                            f" and decomposition {' -> '.join(decomposition.skills)}"
                            if decomposition and decomposition.skills
                            else ""
                        ),
                    )
        except Exception:
            logger.debug("Skill-aware planning enrichment skipped", exc_info=True)

        # Extract repository and conversation context so the plan is grounded
        # in actual project structure instead of language-specific assumptions.
        prior_context = self._build_plan_generation_context()

        # Generate plan using readable schema
        plan = await generate_task_plan(
            provider=planning_provider,
            user_request=enriched_request,
            complexity=complexity,
            model=planning_model,
            conversation_context=prior_context or None,
        )

        logger.info(
            f"Generated plan: {plan.name} with {len(plan.steps)} steps, "
            f"complexity={plan.complexity.value}"
        )

        return plan

    def _build_plan_generation_context(self) -> str:
        """Build compact context for plan generation."""
        parts = []
        repository_context = self._extract_repository_profile_context()
        if repository_context:
            parts.append(repository_context)
        prior_context = self._extract_prior_context()
        if prior_context:
            parts.append("Prior assistant context:\n" f"{prior_context}")
        return "\n\n".join(parts)

    def _extract_repository_profile_context(self) -> str:
        """Return language-aware repository inventory guidance for the planner."""
        try:
            from pathlib import Path

            from victor.agent.planning.repository_profile import (
                detect_repository_profile,
            )
            from victor.config.settings import get_project_paths

            root = Path(get_project_paths().project_root)
            profile = detect_repository_profile(root)
            return profile.to_planning_context()
        except Exception as exc:
            logger.debug("_extract_repository_profile_context failed: %s", exc)
            return ""

    def _extract_prior_context(self) -> str:
        """Return the most recent substantive assistant response for plan seeding.

        Walks the conversation history in reverse and returns the first assistant
        message with meaningful content (>200 chars), capped at 3000 chars so the
        plan-generation prompt stays within token budget.  Returns "" when no
        suitable message is found.
        """
        orch = self.orchestrator
        if not (hasattr(orch, "conversation") and orch.conversation):
            return ""
        try:
            messages = orch.conversation.messages
            for msg in reversed(messages):
                if msg.role == "assistant" and msg.content and len(msg.content) > 200:
                    return msg.content[:3000]
        except Exception as exc:
            logger.debug("_extract_prior_context failed: %s", exc)
        return ""

    async def _show_plan_to_user(self, plan: ReadableTaskPlan) -> PlanApprovalDecision:
        """Display the plan to user and request approval when execution is effectful.

        Args:
            plan: Plan to display

        Returns:
            Plan approval decision.
        """
        # CRITICAL FIX: Use injected renderer for consistent display
        if self.renderer:
            return await self._show_plan_with_renderer(plan)
        else:
            return await self._show_plan_with_console(plan)

    async def _show_plan_with_renderer(self, plan: ReadableTaskPlan) -> PlanApprovalDecision:
        """Display plan using injected renderer (consistent UI).

        Args:
            plan: Plan to display

        Returns:
            Plan approval decision.
        """
        from rich.table import Table

        # Pause renderer to show plan
        self.renderer.pause()

        try:
            # Build plan table
            table = Table(
                title=f"\U0001f4cb {plan.name} ({plan.complexity.value})",
                show_lines=False,
            )
            table.add_column("#", style="dim", width=3)
            table.add_column("Type", style="cyan", width=12)
            table.add_column("Description")
            table.add_column("Tools", style="dim")

            enriched_steps = plan._enrich_step_dicts(plan.steps)
            for step in enriched_steps:
                step_id, step_type, step_desc, step_tools_raw = self._unpack_step(step)
                exec_type = str(step.get("exec", "") if isinstance(step, dict) else "")
                if exec_type and exec_type not in step_type:
                    step_type = f"{step_type}[{exec_type[:4]}.]"
                step_tools = str(step_tools_raw) if step_tools_raw else ""
                table.add_row(step_id, step_type, step_desc, step_tools)

            # Get console from renderer (if available)
            console = getattr(self.renderer, "console", None)
            if console:
                console.print(table)
                if plan.duration:
                    console.print(f"[dim]Estimated: {plan.duration}[/]")

            # Save plan to disk
            self._save_plan_to_disk(plan, console)

            if not self._plan_requires_execution_approval(plan):
                if console:
                    console.print(
                        "[dim green]Read-only exploration plan: continuing without approval.[/]"
                    )
                logger.info("Plan auto-continued without approval: read-only exploration")
                return PlanApprovalDecision(
                    proceed=True,
                    user_approved_execution=False,
                    reason="read_only_exploration",
                    approval_state=ApprovalState.NOT_REQUIRED,
                )

            if self.config.auto_approve:
                if console:
                    console.print(
                        "[dim green]Executing plan without a separate approval prompt.[/]"
                    )
                logger.info("Plan auto-approved for execution")
                return PlanApprovalDecision(
                    proceed=True,
                    user_approved_execution=True,
                    reason="config_auto_approve",
                    approval_state=ApprovalState.APPROVED,
                )

            # Request approval only when explicitly configured.
            if not self.config.auto_approve:
                approved = await self._request_plan_approval(plan, console)
                return PlanApprovalDecision(
                    proceed=approved,
                    user_approved_execution=approved,
                    reason="user_prompt",
                    approval_state=(ApprovalState.APPROVED if approved else ApprovalState.REJECTED),
                )

        finally:
            # Always resume renderer
            self.renderer.resume()

    async def _show_plan_with_console(self, plan: ReadableTaskPlan) -> PlanApprovalDecision:
        """Fallback: Display plan using separate Rich console.

        Args:
            plan: Plan to display

        Returns:
            Plan approval decision.
        """
        from rich.console import Console
        from rich.table import Table

        console = Console()

        table = Table(
            title=f"\U0001f4cb {plan.name} ({plan.complexity.value})",
            show_lines=False,
        )
        table.add_column("#", style="dim", width=3)
        table.add_column("Type", style="cyan", width=12)
        table.add_column("Description")
        table.add_column("Tools", style="dim")

        enriched_steps = plan._enrich_step_dicts(plan.steps)
        for step in enriched_steps:
            step_id, step_type, step_desc, step_tools_raw = self._unpack_step(step)
            exec_type = str(step.get("exec", "") if isinstance(step, dict) else "")
            if exec_type and exec_type not in step_type:
                step_type = f"{step_type}[{exec_type[:4]}.]"
            step_tools = str(step_tools_raw) if step_tools_raw else ""
            table.add_row(step_id, step_type, step_desc, step_tools)

        console.print(table)

        if plan.duration:
            console.print(f"[dim]Estimated: {plan.duration}[/]")

        # Save plan to disk
        self._save_plan_to_disk(plan, console)

        if not self._plan_requires_execution_approval(plan):
            console.print("[dim green]Read-only exploration plan: continuing without approval.[/]")
            logger.info("Plan auto-continued without approval: read-only exploration")
            return PlanApprovalDecision(
                proceed=True,
                user_approved_execution=False,
                reason="read_only_exploration",
                approval_state=ApprovalState.NOT_REQUIRED,
            )

        if self.config.auto_approve:
            console.print("[dim green]Executing plan without a separate approval prompt.[/]")
            logger.info("Plan auto-approved for execution")
            return PlanApprovalDecision(
                proceed=True,
                user_approved_execution=True,
                reason="config_auto_approve",
                approval_state=ApprovalState.APPROVED,
            )

        # Request approval only when explicitly configured.
        if not self.config.auto_approve:
            approved = await self._request_plan_approval(plan, console)
            return PlanApprovalDecision(
                proceed=approved,
                user_approved_execution=approved,
                reason="user_prompt",
                approval_state=(ApprovalState.APPROVED if approved else ApprovalState.REJECTED),
            )

    def _plan_requires_execution_approval(self, plan: ReadableTaskPlan) -> bool:
        """Return True when a displayed plan includes effectful execution."""
        if bool(getattr(plan, "approval", False)):
            return True

        for step in plan.steps:
            _, step_type, step_desc, _ = self._unpack_step(step)
            step_type = step_type.strip().lower()
            step_desc = step_desc.strip().lower()
            step_tools = self._extract_plan_step_tools(step)
            if self._step_requires_execution_approval(step_type, step_desc, step_tools):
                return True
        return False

    @staticmethod
    def _unpack_step(step: Any) -> tuple[str, str, str, Any]:
        """Return (id, type, desc, tools_raw) from either list or dict step format."""
        if isinstance(step, dict):
            tools_raw = step.get("tools", "")
            return (
                str(step.get("id", "")),
                str(step.get("type", "")),
                str(step.get("desc", step.get("description", ""))),
                tools_raw,
            )
        # Compact list format: [id, type, desc, tools, deps?, exec?]
        return (
            str(step[0]) if len(step) > 0 else "",
            str(step[1]) if len(step) > 1 else "",
            str(step[2]) if len(step) > 2 else "",
            step[3] if len(step) > 3 else "",
        )

    @classmethod
    def _serializable_plan_step(cls, step: Any) -> dict[str, Any]:
        """Return a JSON-safe persisted form for a readable-plan step."""
        sid, stype, sdesc, stools = cls._unpack_step(step)
        payload: dict[str, Any] = {
            "id": sid,
            "type": stype,
            "description": sdesc,
            "tools": (list(stools) if isinstance(stools, (list, set, tuple)) else (stools or [])),
        }
        if isinstance(step, dict):
            optional_keys = (
                "deps",
                "depends_on",
                "inputs",
                "consumes",
                "exec",
                "execution",
                "produces",
                "loop_over",
                "items",
                "condition_on",
                "condition",
                "branches",
                "tool_calls",
                "exit",
                "exit_criteria",
                "node",
            )
            for key in optional_keys:
                if key in step and step[key] not in (None, "", [], {}):
                    payload[key] = step[key]
            return payload

        if isinstance(step, list):
            if len(step) > 4 and step[4]:
                payload["deps"] = step[4]
            if len(step) > 5 and step[5]:
                payload["exec"] = step[5]
        return payload

    @classmethod
    def _extract_plan_step_tools(cls, step: Any) -> set[str]:
        """Extract normalized tool names from readable step data."""
        _, _, _, raw_tools = cls._unpack_step(step)
        if isinstance(raw_tools, str):
            return {
                tool.strip().lower()
                for tool in raw_tools.replace(";", ",").split(",")
                if tool.strip()
            }
        if isinstance(raw_tools, list):
            return {str(tool).strip().lower() for tool in raw_tools if str(tool).strip()}
        return set()

    @classmethod
    def _step_requires_execution_approval(
        cls,
        step_type: str,
        step_desc: str,
        tools: set[str],
    ) -> bool:
        """Classify whether a plan step can mutate state or run commands."""
        effectful_types = {
            "bug",
            "bugfix",
            "deploy",
            "deployment",
            "feature",
            "implementation",
            "implement",
            "refactor",
            "test",
            "testing",
        }
        if step_type in effectful_types:
            return True

        effectful_tools = {
            "apply_patch",
            "bash",
            "command",
            "edit",
            "execute",
            "patch",
            "run",
            "shell",
            "test",
            "write",
        }
        if tools & effectful_tools:
            return True

        effectful_desc_markers = (
            "apply patch",
            "create file",
            "edit file",
            "modify file",
            "run command",
            "run tests",
            "shell command",
            "write file",
        )
        return any(marker in step_desc for marker in effectful_desc_markers)

    async def _request_plan_approval(self, plan: ReadableTaskPlan, console) -> bool:
        """Request user approval for plan execution.

        Runs the blocking Rich prompt in a thread so the asyncio event loop
        is not stalled while waiting for stdin — avoids conflicts with the
        victor CLI's own input handler.

        Default answer is YES so that pressing Enter (the natural "looks good,
        proceed" gesture) executes the plan rather than silently rejecting it.

        Falls back to auto-approve when stdin is non-interactive (e.g. pipes,
        EOF on Ctrl-D) so the agent never hangs in headless mode.

        Args:
            plan: Plan to approve
            console: Rich console for prompt

        Returns:
            True if approved, False otherwise
        """
        from rich.prompt import Confirm

        console.print()

        # Non-interactive stdin (pipe / redirect / headless): auto-approve and proceed.
        if not sys.stdin.isatty():
            console.print("[dim yellow]Non-interactive stdin: auto-approving plan[/]")
            logger.info("Plan auto-approved (non-interactive stdin)")
            return True

        try:
            # Off-load blocking stdin read to a thread pool so the event loop
            # can continue processing (e.g. CLI keep-alive tasks, signal handlers).
            console.print(
                "[yellow]This plan includes file changes, shell commands, tests, "
                "deployment, or another effectful step.[/]"
            )
            console.print(
                "[dim]Press Enter to execute, or type y then Enter. "
                "Type n then Enter to reject.[/]"
            )
            approved = await asyncio.to_thread(
                Confirm.ask,
                "[bold yellow]Execute this plan? [Y/n][/]",
                default=True,  # Enter = yes; matches user intent when asking to "plan then implement"
                console=console,
            )
        except EOFError:
            # Ctrl-D or stdin closed: treat as implicit approval to avoid silent failure.
            console.print("[dim yellow]stdin closed: auto-approving plan[/]")
            logger.info("Plan auto-approved (stdin EOF)")
            approved = True
        except Exception as exc:
            logger.warning("Plan approval prompt failed (%s): auto-approving", exc)
            approved = True

        if approved:
            console.print("[green]✓ Plan approved. Executing...[/]")
            logger.info(f"Plan approved by user: {plan.name}")
        else:
            console.print("[red]✗ Plan rejected.[/]")
            logger.info(f"Plan rejected by user: {plan.name}")

        return approved

    def _save_plan_to_disk(self, plan: ReadableTaskPlan, console) -> None:
        """Save plan to .victor/plans/ directory.

        Args:
            plan: Plan to save
            console: Rich console for status messages
        """
        import os
        import json
        from datetime import datetime

        try:
            # Ensure plans directory exists
            plans_dir = os.path.expanduser("~/.victor/plans")
            os.makedirs(plans_dir, exist_ok=True)

            # Generate filename from plan name and timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_name = "".join(
                c if c.isalnum() or c in (" ", "-", "_") else "_" for c in plan.name
            )
            safe_name = safe_name[:50]  # Truncate long names
            filename = f"{timestamp}_{safe_name}.json"
            filepath = os.path.join(plans_dir, filename)

            enriched_steps = plan._enrich_step_dicts(plan.steps)

            # Convert plan to dict for JSON serialization. Persist the enriched
            # runtime-facing fields as well as the display fields so plan-mode
            # postmortems can see data-flow and execution-shape decisions.
            plan_dict = {
                "name": plan.name,
                "complexity": plan.complexity.value,
                "description": plan.description if hasattr(plan, "description") else "",
                "duration": plan.duration if hasattr(plan, "duration") else "",
                "approval": bool(getattr(plan, "approval", False)),
                "steps": [self._serializable_plan_step(step) for step in enriched_steps],
                "generated_at": timestamp,
                "step_count": len(enriched_steps),
            }

            # Save to file
            with open(filepath, "w") as f:
                json.dump(plan_dict, f, indent=2)

            logger.info(f"Plan saved to {filepath}")
            if console:
                console.print(f"[dim]Plan saved to: {filepath}[/]")

        except Exception as e:
            logger.warning(f"Failed to save plan to disk: {e}")
            if console:
                console.print(f"[dim yellow]⚠ Failed to save plan: {e}[/]")

    async def _execute_plan(
        self, plan: ReadableTaskPlan, user_approved: bool = False
    ) -> "PlanResult":
        """Execute the plan step by step.

        Args:
            plan: Plan to execute
            user_approved: True if user explicitly approved this plan (should auto-approve steps)

        Returns:
            PlanResult with execution summary
        """
        # Import here to avoid circular dependency
        from victor.agent.planning.autonomous import AutonomousPlanner
        from victor.agent.planning.team_execution import PlanningTeamExecutionAdapter

        team_adapter = PlanningTeamExecutionAdapter(self.orchestrator)
        if team_adapter.should_use_team(plan):
            return await self._execute_plan_via_team_adapter(plan, team_adapter)

        planner = AutonomousPlanner(self.orchestrator)

        # Convert to execution plan
        execution_plan = plan.to_execution_plan()

        # If user explicitly approved the plan, auto-approve all steps
        # Otherwise, use config setting (which defaults to False for safety)
        auto_approve = user_approved or self.config.auto_approve

        progress_display = self._create_plan_progress_display(plan, execution_plan)

        def _progress_callback(_step: Any, _status: Any) -> None:
            progress_display.update()

        # Execute with auto-approval based on user's plan approval
        progress_display.start()
        try:
            result = await planner.execute_plan(
                execution_plan,
                auto_approve=auto_approve,
                progress_callback=_progress_callback,
            )
            progress_display.update()
        finally:
            progress_display.stop()
        self._apply_plan_evidence_contracts(execution_plan, result)
        self._attach_plan_execution_state(
            execution_plan,
            result,
            execution_mode="autonomous_planner",
        )

        logger.info(
            f"Plan execution: success={result.success}, "
            f"steps_completed={result.steps_completed}/{result.total_steps}"
        )

        return result

    def _create_plan_progress_display(
        self,
        plan: ReadableTaskPlan,
        execution_plan: Any,
    ) -> _PlanProgressDisplay:
        """Create a live plan progress display using the active renderer console when possible."""
        console = getattr(self.renderer, "console", None) if self.renderer else None
        return _PlanProgressDisplay(plan, execution_plan, console=console)

    def _apply_plan_evidence_contracts(self, execution_plan: Any, result: Any) -> None:
        """Recompute plan success after validating each step's execution evidence."""
        from victor.agent.planning.base import StepStatus

        for step in getattr(execution_plan, "steps", []) or []:
            step_result = getattr(step, "result", None) or result.step_results.get(step.id)
            if step_result is None:
                continue
            validated = self._apply_step_evidence_contract(step, step_result)
            step.result = validated
            result.step_results[step.id] = validated
            step.status = StepStatus.COMPLETED if validated.success else StepStatus.FAILED

        result.steps_completed = sum(
            1
            for step in getattr(execution_plan, "steps", [])
            if step.status == StepStatus.COMPLETED
        )
        result.steps_failed = sum(
            1 for step in getattr(execution_plan, "steps", []) if step.status == StepStatus.FAILED
        )
        result.success = result.steps_failed == 0 and result.steps_completed == len(
            getattr(execution_plan, "steps", []) or []
        )
        result.final_output = "\n\n".join(
            step_result.output for step_result in result.step_results.values() if step_result.output
        )

    async def _execute_plan_via_team_adapter(
        self,
        plan: ReadableTaskPlan,
        team_adapter: "PlanningTeamExecutionAdapter",
    ) -> "PlanResult":
        """Execute complex planned work through reusable team formations.

        Maintains a ``plan_state`` dict that flows between steps:
        - Each step's output is stored as ``plan_state["step_{id}"]``.
        - Steps with a ``produces`` context key additionally store their output
          under that named key so downstream loop nodes can reference it.
        - Loop nodes read from ``plan_state[step.context["loop_over"]]``.
        """
        from victor.agent.planning.base import PlanResult, StepStatus

        execution_plan = plan.to_execution_plan()
        progress_display = self._create_plan_progress_display(plan, execution_plan)
        max_concurrent = self._effective_team_plan_concurrency()
        root_session_id = (
            getattr(self.orchestrator, "active_session_id", None)
            or getattr(self.orchestrator, "session_id", None)
            or getattr(self.orchestrator, "_memory_session_id", None)
        )
        result = PlanResult(
            plan_id=execution_plan.id,
            success=True,
            total_steps=len(execution_plan.steps),
        )

        # Shared mutable state flowing between steps (StateGraph-style).
        plan_state: Dict[str, Any] = {}

        # Do NOT use is_failed() in the while condition — it would halt the plan the
        # moment ANY step fails evidence contract, killing independent follow-on steps.
        # Termination is handled by is_complete() + the "no ready steps → BLOCKED" guard.
        progress_display.start()
        try:
            while not execution_plan.is_complete():
                ready_steps = execution_plan.get_ready_steps()
                if not ready_steps:
                    pending_steps = [
                        step for step in execution_plan.steps if step.status == StepStatus.PENDING
                    ]
                    satisfied_ids = {
                        s.id
                        for s in execution_plan.steps
                        if s.status in (StepStatus.COMPLETED, StepStatus.SKIPPED)
                    }
                    for step in pending_steps:
                        unmet = [dep for dep in step.depends_on if dep not in satisfied_ids]
                        logger.info(
                            "  BLOCKED step %s: depends_on=%s, satisfied=%s, unmet=%s",
                            step.id,
                            step.depends_on,
                            sorted(satisfied_ids),
                            unmet,
                        )
                        step.status = StepStatus.BLOCKED
                    progress_display.update()
                    logger.info(
                        "Team plan: no ready steps remaining (%d pending → BLOCKED, %d failed, %d skipped).",
                        len(pending_steps),
                        sum(1 for s in execution_plan.steps if s.status == StepStatus.FAILED),
                        sum(1 for s in execution_plan.steps if s.status == StepStatus.SKIPPED),
                    )
                    break

                batch = ready_steps[:max_concurrent]
                for step in batch:
                    step.status = StepStatus.IN_PROGRESS
                progress_display.update()

                is_parallel = len(batch) > 1
                logger.info(
                    "Team plan: dispatching %s batch of %d step(s): %s  [plan_state keys: %s]",
                    "PARALLEL" if is_parallel else "sequential",
                    len(batch),
                    [s.id for s in batch],
                    [k for k in plan_state if not k.startswith("step_")],
                )

                # Snapshot plan_state before dispatch: all parallel steps receive the
                # same consistent view of what was produced by prior completed steps.
                # Parallel steps write to different produces keys so there is no conflict;
                # the live dict is updated after asyncio.gather returns.
                dispatch_state = dict(plan_state) if is_parallel else plan_state

                batch_work = asyncio.gather(
                    *[
                        team_adapter.execute_step(
                            plan=plan,
                            execution_plan=execution_plan,
                            step=step,
                            root_session_id=root_session_id,
                            plan_state=dispatch_state,
                        )
                        for step in batch
                    ],
                    return_exceptions=True,
                )
                if isinstance(progress_display, _PlanProgressDisplay):
                    step_results = await progress_display.tick_until(batch_work)
                else:
                    step_results = await batch_work

                failed_step_ids: list[str] = []
                for step, step_result in zip(batch, step_results):
                    if isinstance(step_result, Exception):
                        from victor.agent.planning.base import StepResult

                        step_result = StepResult(
                            success=False,
                            output="",
                            error=f"{type(step_result).__name__}: {step_result}",
                        )

                    step_result = self._persist_step_artifact_if_needed(
                        execution_plan,
                        step,
                        step_result,
                    )
                    step_result = self._apply_step_evidence_contract(step, step_result, plan_state)
                    step_result = await self._retry_step_if_needed(
                        step=step,
                        step_result=step_result,
                        plan=plan,
                        execution_plan=execution_plan,
                        team_adapter=team_adapter,
                        root_session_id=root_session_id,
                        plan_state=plan_state,
                        progress_display=progress_display,
                    )
                    step.result = step_result
                    step.status = StepStatus.COMPLETED if step_result.success else StepStatus.FAILED
                    result.step_results[step.id] = step_result
                    result.total_tool_calls += step_result.tool_calls_used

                    exec_mode = str(step.context.get("execution", "") or step.execution or "agent")
                    logger.info(
                        "Team plan step %s [%s/%s] %s  (tools=%d, chars=%d)",
                        step.id,
                        exec_mode,
                        (
                            step.step_type.value
                            if hasattr(step.step_type, "value")
                            else step.step_type
                        ),
                        (
                            "COMPLETED"
                            if step_result.success
                            else f"FAILED: {step_result.error or '?'}"
                        ),
                        step_result.tool_calls_used,
                        len(step_result.output or ""),
                    )

                    # Accumulate plan state so downstream steps (e.g. loop nodes) can
                    # reference this step's output by step ID or by its "produces" key.
                    plan_state[f"step_{step.id}"] = step_result.output
                    produces_key = step.context.get("produces", "")
                    if produces_key and step_result.output:
                        extracted = self._extract_list_from_output(step_result.output)
                        extracted = self._coerce_required_produces_items(
                            step,
                            step_result,
                            produces_key,
                            extracted,
                        )
                        requires_nonempty_output = self._requires_nonempty_produces(
                            step, produces_key
                        ) or self._produces_consumed_downstream(
                            execution_plan,
                            getattr(step, "id", ""),
                            produces_key,
                        )

                        if step_result.success and not extracted and requires_nonempty_output:
                            from victor.agent.planning.base import (
                                StepResult as _SProduces,
                            )

                            step_result = _SProduces(
                                success=False,
                                output=step_result.output,
                                error=(
                                    f"Step {step.id} produced no structured items for "
                                    f"required plan_state key '{produces_key}'"
                                ),
                                tool_calls_used=step_result.tool_calls_used,
                                duration_seconds=getattr(step_result, "duration_seconds", 0.0),
                                artifacts=list(getattr(step_result, "artifacts", []) or []),
                                metadata={
                                    **dict(getattr(step_result, "metadata", {}) or {}),
                                    "empty_required_produces": {
                                        "key": produces_key,
                                        "output_chars": len(step_result.output or ""),
                                    },
                                },
                            )
                            step.result = step_result
                            step.status = StepStatus.FAILED
                            result.step_results[step.id] = step_result
                            logger.warning(
                                "Team plan step %s produced empty required key '%s'",
                                step.id,
                                produces_key,
                            )
                            step_result = await self._retry_step_if_needed(
                                step=step,
                                step_result=step_result,
                                plan=plan,
                                execution_plan=execution_plan,
                                team_adapter=team_adapter,
                                root_session_id=root_session_id,
                                plan_state=plan_state,
                                progress_display=progress_display,
                            )
                            if step_result.success:
                                extracted = self._extract_list_from_output(step_result.output)
                                extracted = self._coerce_required_produces_items(
                                    step,
                                    step_result,
                                    produces_key,
                                    extracted,
                                )
                                step.result = step_result
                                step.status = StepStatus.COMPLETED
                                result.step_results[step.id] = step_result

                        # Store produces_key in plan_state only when the step succeeded.
                        # A failed step may have non-empty output (the original text before
                        # the evidence contract rejected it), but storing empty extracted
                        # findings would allow downstream synthesis steps to proceed with
                        # no real data.  The rescue path below may promote a failed step to
                        # COMPLETED, at which point we store produces_key explicitly.
                        if step_result.success:
                            plan_state[produces_key] = extracted
                            logger.info(
                                "Team plan step %s produced '%s' → %d item(s): %s",
                                step.id,
                                produces_key,
                                len(extracted),
                                extracted[:5],
                            )

                        # Rescue: if the agentic loop reported failure but the step
                        # produced valid output, promote it to COMPLETED.  Covers known
                        # false-positive exit reasons:
                        #   (a) "Clarification required" — PerceptionIntegration
                        #       misidentified a self-contained knowledge generation task.
                        #   (b) "Agent stuck: N turns without tool calls" — spin detector
                        #       fired on a knowledge task that correctly made 0 tool calls.
                        #   (c) "Insufficient progress" — plateau detector on synthesis.
                        # Note: "Insufficient execution evidence" was removed — steps with
                        # ≥5 tools + ≥100-char output now pass the evidence contract directly
                        # (see _assess_step_evidence), so they never reach this rescue path.
                        # Guard: only when produces has items AND output is substantial
                        # (>= 100 chars), so legitimately empty-output steps are not rescued.
                        _AGENTIC_LOOP_FP_PATTERNS = (
                            "Clarification required",
                            "Agent stuck",
                            "Insufficient progress",
                        )
                        _is_agentic_loop_fp = any(
                            pat in (step_result.error or "") for pat in _AGENTIC_LOOP_FP_PATTERNS
                        )
                        if (
                            not step_result.success
                            and len(extracted) > 0
                            and len(step_result.output or "") >= 100
                            and _is_agentic_loop_fp
                        ):
                            from victor.agent.planning.base import (
                                StepResult as _SRescue,
                            )

                            rescued = _SRescue(
                                success=True,
                                output=step_result.output,
                                tool_calls_used=step_result.tool_calls_used,
                                metadata={
                                    **step_result.metadata,
                                    "rescued_clarification_fp": True,
                                },
                            )
                            step.result = rescued
                            step.status = StepStatus.COMPLETED
                            result.step_results[step.id] = rescued
                            step_result = rescued
                            # Step is now rescued → store produces_key
                            plan_state[produces_key] = extracted
                            logger.info(
                                "Team plan step %s produced '%s' → %d item(s): %s",
                                step.id,
                                produces_key,
                                len(extracted),
                                extracted[:5],
                            )
                            logger.info(
                                "Team plan step %s: rescued from clarification false-positive "
                                "(produces='%s', %d items, %d chars)",
                                step.id,
                                produces_key,
                                len(extracted),
                                len(rescued.output or ""),
                            )

                    # Conditional node: apply branch routing immediately so the
                    # next get_ready_steps() call sees the correct PENDING/SKIPPED state.
                    skip_ids = step_result.metadata.get("skip_step_ids", [])
                    if skip_ids:
                        self._skip_specific_steps(execution_plan, skip_ids)
                        logger.info(
                            "Team plan step %s (conditional): skipping branch steps %s",
                            step.id,
                            skip_ids,
                        )

                    if not step_result.success:
                        failed_step_ids.append(step.id)
                    progress_display.update()

                if failed_step_ids:
                    # Block effectful dependents of failed steps, but let read-only
                    # analysis/reporting continue with explicit partial-context notes.
                    self._skip_team_plan_dependents(execution_plan, failed_step_ids)
                    progress_display.update()
                    logger.info(
                        "Team plan: step(s) %s failed — continuing eligible partial-analysis "
                        "dependents and blocking effectful dependents.",
                        failed_step_ids,
                    )
        finally:
            progress_display.stop()

        # Attach failure results for steps still PENDING or BLOCKED after the loop exits
        # (unmet dependencies, failed required predecessors, or missing plan_state keys).
        # This makes the failure surface actionable instead of silently leaving steps
        # without step_results in the final summary.
        from victor.agent.planning.base import StepResult as _StepResult

        satisfied_ids = {
            s.id
            for s in execution_plan.steps
            if s.status in (StepStatus.COMPLETED, StepStatus.SKIPPED)
        }
        for stuck_step in execution_plan.steps:
            if stuck_step.status not in (StepStatus.PENDING, StepStatus.BLOCKED):
                continue
            unmet = [
                dep for dep in getattr(stuck_step, "depends_on", []) if dep not in satisfied_ids
            ]
            reason = getattr(getattr(stuck_step, "result", None), "error", None) or (
                f"Step blocked: unmet dependencies {unmet}"
                if unmet
                else "Step blocked: no ready predecessor produced required plan_state keys"
            )
            stuck_step.status = StepStatus.BLOCKED
            blocked_result = getattr(stuck_step, "result", None) or _StepResult(
                success=False,
                output="",
                error=reason,
                tool_calls_used=0,
            )
            stuck_step.result = blocked_result
            result.step_results[stuck_step.id] = blocked_result
            logger.warning(
                "Team plan step %s → BLOCKED: %s",
                stuck_step.id,
                reason,
            )

        result.steps_completed = sum(
            1 for step in execution_plan.steps if step.status == StepStatus.COMPLETED
        )
        result.steps_failed = sum(
            1 for step in execution_plan.steps if step.status == StepStatus.FAILED
        )
        # Treat SKIPPED as terminal — intentionally-skipped branch steps count as done.
        _terminal = (StepStatus.COMPLETED, StepStatus.SKIPPED)
        result.success = result.steps_failed == 0 and all(
            step.status in _terminal for step in execution_plan.steps
        )
        result.final_output = "\n\n".join(
            step_result.output for step_result in result.step_results.values() if step_result.output
        )
        self._attach_plan_execution_state(
            execution_plan,
            result,
            execution_mode="team_adapter",
        )

        logger.info(
            "Team plan execution: success=%s, steps_completed=%s/%s",
            result.success,
            result.steps_completed,
            result.total_steps,
        )
        return result

    @staticmethod
    def _requires_nonempty_produces(step: Any, produces_key: str) -> bool:
        """Return True when an empty produced value should fail the step.

        Most produced values are collections consumed by downstream loop or synthesis nodes.
        Empty extraction for those keys usually means the agent narrated intent instead of
        returning the requested artifact, so treating it as success hides the real failure.
        """
        key = str(produces_key or "").strip().lower()
        if not key:
            return False
        required_key_suffixes = (
            "findings",
            "inventory",
            "items",
            "members",
            "modules",
            "packages",
            "paths",
            "report",
            "checklist",
            "summary",
            "targets",
        )
        if key.endswith(required_key_suffixes):
            return True

        context = getattr(step, "context", {}) or {}
        if context.get("loop_over") == produces_key:
            return True
        inputs = getattr(step, "inputs", None) or context.get("inputs", []) or []
        return bool(inputs)

    @staticmethod
    def _produces_consumed_downstream(
        execution_plan: Any,
        producer_step_id: str,
        produces_key: str,
    ) -> bool:
        """Return True if later plan nodes declare that they consume ``produces_key``."""
        key = str(produces_key or "").strip()
        if not key:
            return False
        for candidate in getattr(execution_plan, "steps", []) or []:
            if str(getattr(candidate, "id", "")) == str(producer_step_id):
                continue
            context = dict(getattr(candidate, "context", {}) or {})
            inputs = [
                str(item).strip()
                for item in (
                    getattr(candidate, "inputs", None)
                    or context.get("inputs", [])
                    or context.get("consumes", [])
                    or []
                )
                if str(item).strip()
            ]
            if key in inputs:
                return True
            if context.get("loop_over") == key or context.get("condition_on") == key:
                return True
        return False

    @classmethod
    def _should_retry_after_evidence_failure(cls, step: Any, step_result: Any) -> bool:
        """Return True when a failed produced artifact deserves one agentic retry."""
        if getattr(step_result, "success", False):
            return False
        metadata = dict(getattr(step_result, "metadata", {}) or {})
        validation = metadata.get("evidence_validation") or {}
        context = dict(getattr(step, "context", {}) or {})
        produces_key = str(context.get("produces", "") or "")
        if not produces_key or not cls._requires_nonempty_produces(step, produces_key):
            return False
        execution = str(getattr(step, "execution", "") or context.get("execution", "") or "")
        if execution in {"compute", "approval", "checkpoint", "conditional"}:
            return False
        empty_required = metadata.get("empty_required_produces") or {}
        if empty_required:
            return True
        if not validation or validation.get("passed"):
            return False
        return bool(
            validation.get("is_directory_listing_only")
            or validation.get("has_unresolved_tool_markup")
            or "required produced artifact" in str(getattr(step_result, "error", "") or "").lower()
            or "insufficient execution evidence"
            in str(getattr(step_result, "error", "") or "").lower()
            or "produced no structured items"
            in str(getattr(step_result, "error", "") or "").lower()
        )

    async def _retry_step_if_needed(
        self,
        *,
        step: Any,
        step_result: Any,
        plan: ReadableTaskPlan,
        execution_plan: "ExecutionPlan",
        team_adapter: "PlanningTeamExecutionAdapter",
        root_session_id: Optional[str],
        plan_state: dict,
        progress_display: _PlanProgressDisplay,
    ) -> Any:
        """Retry hollow/unfinished produced artifacts through a stricter agentic pass."""
        if not self._should_retry_after_evidence_failure(step, step_result):
            return step_result

        retry_step = self._agentic_retry_step(step, step_result)
        retry_work = team_adapter.execute_step(
            plan=plan,
            execution_plan=execution_plan,
            step=retry_step,
            root_session_id=root_session_id,
            plan_state=plan_state,
        )
        if isinstance(progress_display, _PlanProgressDisplay):
            retry_result = await progress_display.tick_until(retry_work)
        else:
            retry_result = await retry_work
        retry_result = self._persist_step_artifact_if_needed(
            execution_plan,
            step,
            retry_result,
        )
        retry_result = self._apply_step_evidence_contract(
            step,
            retry_result,
            plan_state,
        )
        if retry_result.success:
            metadata = dict(getattr(retry_result, "metadata", {}) or {})
            metadata["agentic_retry"] = {
                "original_error": step_result.error,
                "original_output_chars": len(step_result.output or ""),
                "retry_tool_budget": retry_step.estimated_tool_calls,
            }
            retry_result.metadata = metadata
            logger.info(
                "Team plan step %s recovered via agentic evidence retry",
                step.id,
            )
            return retry_result
        return retry_result

    @staticmethod
    def _agentic_retry_step(step: Any, step_result: Any) -> Any:
        """Create a stricter retry step without changing the original plan shape."""
        from dataclasses import replace

        context = dict(getattr(step, "context", {}) or {})
        produces_key = str(context.get("produces", "") or "produced_artifact")
        original_output = str(getattr(step_result, "output", "") or "").strip()
        retry_criteria = [
            f"produce the `{produces_key}` artifact in the requested final-answer format",
            "inspect the relevant project materials and cite concrete evidence when available",
            "cover the full requested scope before completing; do not stop after inventory",
        ]
        if original_output:
            retry_criteria.append(
                "the previous attempt was rejected as hollow evidence; do not repeat that shape"
            )
        existing = [str(item) for item in getattr(step, "exit_criteria", []) or []]
        budget = max(int(getattr(step, "estimated_tool_calls", 0) or 0), 25)
        try:
            return replace(
                step,
                estimated_tool_calls=budget,
                exit_criteria=existing + retry_criteria,
            )
        except TypeError:
            step.estimated_tool_calls = budget
            step.exit_criteria = existing + retry_criteria
            return step

    @classmethod
    def _persist_step_artifact_if_needed(
        cls,
        execution_plan: Any,
        step: Any,
        step_result: Any,
    ) -> Any:
        """Persist long/generated plan outputs so summaries cannot truncate them away.

        Quality gating: never persist failed steps, degenerate model output, or
        pure file/path listings — those artifacts would mislead later synthesis
        steps and pollute the .victor/plans/artifacts directory with hollow files.
        """
        output = str(getattr(step_result, "output", "") or "")
        if not output.strip() or not getattr(step_result, "success", False):
            return step_result

        metadata_pre = dict(getattr(step_result, "metadata", {}) or {})
        evidence_validation = metadata_pre.get("evidence_validation") or {}
        if isinstance(evidence_validation, dict) and not evidence_validation.get("passed", True):
            logger.info(
                "Skipping artifact persistence for step %s — evidence validation failed: %s",
                getattr(step, "id", "?"),
                evidence_validation.get("reason", "?"),
            )
            return step_result
        is_degenerate, degenerate_reason = cls._is_degenerate_output(output)
        if is_degenerate:
            logger.warning(
                "Skipping artifact persistence for step %s — degenerate output: %s",
                getattr(step, "id", "?"),
                degenerate_reason,
            )
            return step_result
        if cls._is_directory_listing_only(output):
            logger.info(
                "Skipping artifact persistence for step %s — output is a directory listing only",
                getattr(step, "id", "?"),
            )
            return step_result

        context = dict(getattr(step, "context", {}) or {})
        produces_key = str(context.get("produces", "") or "").strip()
        description = str(getattr(step, "description", "") or "")
        should_persist = len(output) >= 1200 or cls._is_artifact_producing_step(
            produces_key,
            description,
        )
        if not should_persist:
            return step_result

        try:
            plan_id = str(getattr(execution_plan, "id", "") or "plan")
            artifact_root = Path(".victor") / "plans" / "artifacts" / cls._artifact_slug(plan_id)
            artifact_root.mkdir(parents=True, exist_ok=True)
            step_id = cls._artifact_slug(str(getattr(step, "id", "") or "step"))
            key = cls._artifact_slug(produces_key or "output")
            suffix = ".md" if cls._looks_like_markdown_artifact(produces_key, output) else ".txt"
            artifact_path = artifact_root / f"step_{step_id}_{key}{suffix}"
            artifact_path.write_text(output, encoding="utf-8")
        except Exception as exc:
            logger.debug("Failed to persist plan step artifact: %s", exc)
            return step_result

        artifacts = list(getattr(step_result, "artifacts", []) or [])
        artifact_str = str(artifact_path)
        if artifact_str not in artifacts:
            artifacts.append(artifact_str)
        metadata = dict(getattr(step_result, "metadata", {}) or {})
        metadata.setdefault("plan_artifact_path", artifact_str)
        metadata.setdefault("plan_artifact_bytes", len(output.encode("utf-8")))

        from victor.agent.planning.base import StepResult as _StepResult

        return _StepResult(
            success=bool(getattr(step_result, "success", False)),
            output=output,
            error=getattr(step_result, "error", None),
            tool_calls_used=int(getattr(step_result, "tool_calls_used", 0) or 0),
            duration_seconds=float(getattr(step_result, "duration_seconds", 0.0) or 0.0),
            artifacts=artifacts,
            metadata=metadata,
        )

    @staticmethod
    def _is_artifact_producing_step(produces_key: str, description: str) -> bool:
        text = f"{produces_key} {description}".lower()
        return bool(
            re.search(
                r"(?:^|[\s_-])(report|summary|checklist|findings|recommendations|scorecard)(?:$|[\s_-])",
                text,
            )
        )

    @staticmethod
    def _looks_like_markdown_artifact(produces_key: str, output: str) -> bool:
        if re.search(r"(?:^|_)(?:report|summary|checklist|findings)(?:_|$)", produces_key):
            return True
        return output.lstrip().startswith("#") or "\n- " in output or "\n1. " in output

    @staticmethod
    def _artifact_slug(value: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value)).strip("._-")
        return slug or "artifact"

    @classmethod
    def _coerce_required_produces_items(
        cls,
        step: Any,
        step_result: Any,
        produces_key: str,
        extracted: list[str],
    ) -> list[str]:
        """Preserve substantive prose outputs as a single plan-state item.

        Some providers return cross-crate or synthesis findings as paragraphs instead of a
        bullet list.  If the evidence contract already accepted a substantial tool-backed
        output, keep that prose as one downstream synthesis input.  Short intent statements
        still fail through the required-produces guard.
        """
        if extracted or not getattr(step_result, "success", False):
            output = str(getattr(step_result, "output", "") or "").strip()
            if (
                extracted
                and output
                and cls._is_artifact_producing_step(produces_key, getattr(step, "description", ""))
                and cls._looks_like_markdown_artifact(produces_key, output)
            ):
                return [output]
            return extracted
        if not cls._requires_nonempty_produces(step, produces_key):
            return extracted

        output = str(getattr(step_result, "output", "") or "").strip()
        metadata = dict(getattr(step_result, "metadata", {}) or {})
        validation = metadata.get("evidence_validation") or {}
        validation_passed = bool(validation.get("passed"))
        reason = str(validation.get("reason", "") or "")
        tool_calls = int(getattr(step_result, "tool_calls_used", 0) or 0)
        key = str(produces_key or "").lower()
        step_type = str(
            getattr(
                getattr(step, "step_type", None),
                "value",
                getattr(step, "step_type", ""),
            )
            or ""
        ).lower()
        context = dict(getattr(step, "context", {}) or {})
        execution = str(context.get("execution", getattr(step, "execution", "")) or "").lower()
        description = str(getattr(step, "description", "") or "").lower()
        is_generated_artifact = (
            key.endswith(("checklist", "report", "summary", "findings"))
            or any(word in key for word in ("checklist", "report", "summary", "findings"))
            or any(word in description for word in ("checklist", "report", "summary", "findings"))
        )
        is_doc_or_compute = (
            step_type in {"research", "documentation", "review"}
            or execution in {"compute", "approval"}
            or "doc" in step_type
        )

        if (
            validation_passed
            and len(output) >= 500
            and tool_calls >= 3
            and "intent statement" not in reason.lower()
        ):
            return [output]
        if (
            validation_passed
            and is_generated_artifact
            and is_doc_or_compute
            and len(output) >= 1000
            and "intent statement" not in reason.lower()
        ):
            return [output]
        return extracted

    @staticmethod
    def _extract_list_from_output(output: str) -> list[str]:
        """Best-effort extraction of a newline/bullet list from step output.

        Used to populate plan_state when a step declares ``produces``.  The
        result feeds loop node item resolution via ``loop_over``.

        Extraction order:
          1. JSON array (structured sub-agent output preferred)
          2. Bullet / numbered / plain newline list
          3. Prose guard: single-item result that looks like a sentence → empty list
             (prevents prose fallback from mis-routing conditional nodes)
        """
        if not output:
            return []

        # Sentinel emitted by _task_description_for_step when the agent found nothing.
        if output.strip() == "(none)":
            return []

        # Sentinel emitted by _execute_compute_node generic fallback when no registered
        # handler exists.  This is not real step output — returning early avoids spurious
        # prose-guard warnings and mis-routing of downstream conditional steps.
        if output.lstrip().startswith("Compute step:") or output.lstrip().startswith(
            "Step ready for review:"
        ):
            logger.warning(
                "_extract_list_from_output: output is a _generic_compute placeholder "
                "(step lacked a registered handler) — produces key will be empty. Raw: %.80s",
                output,
            )
            return []

        # Strip XML thinking/tool blocks emitted by reasoning models (e.g. GLM-5.x
        # outputs <thinking>...</thinking> and <tool_call> markup inline in content).
        # These blocks contaminate line-by-line extraction with XML tag strings.
        output = re.sub(
            r"<thinking\b[^>]*>.*?</thinking\s*>",
            "",
            output,
            flags=re.DOTALL | re.IGNORECASE,
        )
        output = re.sub(
            r"<tool_call\b[^>]*>.*?</tool_call\s*>",
            "",
            output,
            flags=re.DOTALL | re.IGNORECASE,
        )
        output = re.sub(
            r"<tool_call\b[^>]*>.*?(?:</tool_call[^\n>]*>?|$)",
            "",
            output,
            flags=re.DOTALL | re.IGNORECASE,
        )
        # Remove residual lone XML open/close tags on their own lines
        output = re.sub(r"^\s*</?\w[\w_\-]*[^>]*>\s*$", "", output, flags=re.MULTILINE)
        # Strip ZAI/GLM-5.x ∂-delimited tool-call fragments (format: tool∂param∂value∂...).
        # The ∂ character (U+2202) is used internally by GLM as a field separator in its
        # tool invocation syntax; it should never appear in extracted plan-state items.
        output = re.sub(r"[^\S\n]*∂[^\n]*", "", output)  # rest of line after ∂
        output = re.sub(r"[^\S\n]*\S*∂", "", output)  # prefix∂ fragments mid-line
        if not output.strip():
            return []

        # 1. Try JSON array first (structured output from sub-agent)
        stripped = output.strip()
        if stripped.startswith("["):
            try:
                import json as _json

                parsed = _json.loads(stripped)
                if isinstance(parsed, list):
                    return [str(i).strip() for i in parsed if str(i).strip()]
            except (ValueError, TypeError):
                pass

        # 2. Line-by-line extraction
        lines = output.splitlines()
        items: list[str] = []
        for line in lines:
            # Strip leading bullets, numbers, whitespace
            cleaned = re.sub(r"^[\s\-\*•\d\.]+", "", line).strip()
            # Strip trailing annotations (e.g. "crates/state - State management")
            cleaned = re.split(r"\s+[-—–]\s+", cleaned)[0].strip()
            # Discard empty lines or suspiciously long prose (not a list item)
            if cleaned and 1 <= len(cleaned) <= 150:
                items.append(cleaned)

        # Post-filter: strip agent narration lines that contaminate a real list.
        # Examples: "The file was truncated. Let me read the middle portion..."
        #           "FirstResponderTool halted, entering general response mode."
        # Guard: only apply when the filtered result is non-empty so that single-line
        # prose outputs that embed real path tokens can still reach the prose guard's
        # secondary embedded-token scan below.
        _CONTINUATION_RE = re.compile(
            r"(?:the file was truncated|let me read|let me re-read|let me continue|"
            r"entering general response mode|firstresponder.*halted|"
            r"let me now|reading the|i(?:'ll| will) read|"
            r"let me examine|let me check|let me look|to read more|"
            r"^actually,?\s+let me|^now let me|^let me also|"
            r"^now i (?:have|see|understand|know|can)\b|"
            r"^i need to (?:see|read|re-read|check|look|examine|review|get)\b)",
            re.IGNORECASE,
        )
        filtered = [i for i in items if not _CONTINUATION_RE.search(i)]
        if filtered:
            items = filtered

        # 3. Prose guard: if the only extracted item is a sentence, it is not a structured
        #    list. Storing it would cause conditional nodes to take the wrong branch.
        if len(items) == 1:
            sole = items[0]
            is_prose = (
                sole.endswith(":")
                or re.search(
                    r"\b(let|now|will|going|please|here|next|reading|listing)\b",
                    sole,
                    re.IGNORECASE,
                )
                or (len(sole.split()) > 6 and sole[0].isupper())
            )
            if is_prose:
                # 3b. Secondary scan: look for embedded path-like or technical identifier
                # tokens within the prose. Models like GLM-5.x sometimes narrate a plan
                # ("Now let me examine crates/state and crates/tools...") while embedding
                # the actual list items inline. Only extract tokens that are unambiguously
                # non-prose: slash-paths, snake_case, or kebab-case identifiers.
                #
                # Prefer tokens that contain "/" (genuine paths). Hyphenated words
                # (e.g. "read-only" from deterministic execution summaries) must contain
                # at least one "/" to qualify — they are common English adjectives, not paths.
                all_tokens = re.findall(
                    r"\b[\w][\w]*(?:[/_\-][\w]+)+\b",
                    output,
                )
                # Prefer slash-paths; fall back to all tokens only when no paths are found
                path_tokens = [t for t in all_tokens if "/" in t]
                path_like = (
                    path_tokens
                    if path_tokens
                    else [t for t in all_tokens if "_" in t]  # snake_case but not kebab-only
                )
                # Deduplicate while preserving order
                seen: set[str] = set()
                path_like = [t for t in path_like if not (t in seen or seen.add(t))]  # type: ignore[func-returns-value]
                if path_like:
                    logger.debug(
                        "_extract_list_from_output: sole item looks like prose — "
                        "extracted %d embedded path/identifier token(s) as fallback. Raw: %.80s",
                        len(path_like),
                        sole,
                    )
                    return path_like
                logger.debug(
                    "_extract_list_from_output: sole item looks like prose, not a list — "
                    "returning empty list. Raw: %.80s",
                    sole,
                )
                return []

        return items

    def _attach_plan_execution_state(
        self,
        execution_plan: "ExecutionPlan",
        result: "PlanResult",
        *,
        execution_mode: str,
    ) -> None:
        """Attach graph/checkpoint-friendly plan execution state to the result."""
        metadata = dict(getattr(result, "metadata", {}) or {})
        metadata["plan_execution_state"] = self._build_plan_execution_state(
            execution_plan,
            result,
            execution_mode=execution_mode,
        )
        result.metadata = metadata

    @staticmethod
    def _attach_planning_response_metadata(
        response: CompletionResponse,
        plan: ReadableTaskPlan,
        result: "PlanResult",
    ) -> None:
        """Expose plan execution state on the final response metadata."""
        metadata = dict(response.metadata or {})
        result_metadata = getattr(result, "metadata", {}) or {}
        plan_execution_state = result_metadata.get("plan_execution_state")
        if plan_execution_state:
            metadata["plan_execution_state"] = plan_execution_state

        metadata["planning"] = {
            **dict(metadata.get("planning", {}) or {}),
            "mode": "planned",
            "plan_name": plan.name,
            "plan_complexity": plan.complexity.value,
            "steps_completed": int(getattr(result, "steps_completed", 0) or 0),
            "steps_total": int(getattr(result, "total_steps", len(plan.steps)) or 0),
            "success": bool(getattr(result, "success", False)),
        }
        response.metadata = metadata

    @staticmethod
    def _build_plan_execution_state(
        execution_plan: "ExecutionPlan",
        result: "PlanResult",
        *,
        execution_mode: str,
    ) -> Dict[str, Any]:
        """Build a compact serializable snapshot of plan execution state."""
        from victor.agent.planning.base import StepStatus

        steps = list(getattr(execution_plan, "steps", []) or [])
        step_statuses = {step.id: step.status.value for step in steps}

        def step_ids_with_status(status: StepStatus) -> list[str]:
            return [step.id for step in steps if step.status == status]

        return {
            "plan_id": getattr(execution_plan, "id", getattr(result, "plan_id", "")),
            "goal": getattr(execution_plan, "goal", ""),
            "execution_mode": execution_mode,
            "success": bool(getattr(result, "success", False)),
            "total_steps": int(getattr(result, "total_steps", len(steps)) or 0),
            "steps_completed": int(getattr(result, "steps_completed", 0) or 0),
            "steps_failed": int(getattr(result, "steps_failed", 0) or 0),
            "total_tool_calls": int(getattr(result, "total_tool_calls", 0) or 0),
            "progress_percent": float(execution_plan.progress_percentage()),
            "step_statuses": step_statuses,
            "ready_step_ids": [step.id for step in execution_plan.get_ready_steps()],
            "completed_step_ids": step_ids_with_status(StepStatus.COMPLETED),
            "failed_step_ids": step_ids_with_status(StepStatus.FAILED),
            "skipped_step_ids": step_ids_with_status(StepStatus.SKIPPED),
            "blocked_step_ids": step_ids_with_status(StepStatus.BLOCKED),
        }

    _CONCRETE_FILE_REF_RE = re.compile(
        r"(?:(?:^|[\s`'\"(])[\w./-]+\.(?:rs|toml|lock|py|md|json|ya?ml)(?::\d+)?)",
        re.IGNORECASE,
    )
    _CONCRETE_DIR_REF_RE = re.compile(
        r"(?:(?:^|[\s`'\"(])(?:[\w.-]+/)+(?:src|benches|tests|examples|crates)?/?)",
        re.IGNORECASE,
    )
    _WEAK_STEP_OUTPUTS = {
        "done",
        "complete",
        "completed",
        "mapped",
        "reviewed",
        "team complete",
        "inventory complete",
        "review complete",
        "analysis complete",
        "report complete",
    }

    @staticmethod
    def _mark_step_exempt(step_result: Any, reason: str) -> Any:
        """Stamp evidence_validation on exempt steps so summary counts are accurate.

        Without this, exempted steps are counted as 'missing' in the validation summary
        because _evidence_validation_metadata returns None when the key is absent.
        """
        metadata = dict(getattr(step_result, "metadata", {}) or {})
        if "evidence_validation" not in metadata:
            metadata["evidence_validation"] = {
                "passed": True,
                "exempt": True,
                "reason": reason,
            }
            step_result.metadata = metadata
        return step_result

    def _apply_step_evidence_contract(
        self,
        step: Any,
        step_result: Any,
        plan_state: dict | None = None,
    ) -> Any:
        """Fail read-heavy plan steps that completed without concrete evidence."""
        if not getattr(step_result, "success", False):
            return step_result

        # Approval, conditional, and compute nodes never produce tool-backed evidence
        # by design — they coordinate/evaluate rather than gather data.
        metadata = dict(getattr(step_result, "metadata", {}) or {})
        result_exec = str(metadata.get("execution_mode", "") or "").lower()
        if result_exec in {"builtin_compute", "compute_node"} or metadata.get("compute_node"):
            logger.debug(
                "Evidence contract exempt for %s step (result execution_mode=%s)",
                getattr(step, "id", "?"),
                result_exec or "compute_node",
            )
            return self._mark_step_exempt(
                step_result,
                f"result-exec={result_exec or 'compute_node'}",
            )

        step_exec = str(
            getattr(step, "execution", "")
            or (step.context.get("execution", "") if hasattr(step, "context") else "")
        ).lower()
        if step_exec in ("conditional", "approval", "checkpoint", "compute", "tool"):
            logger.debug(
                "Evidence contract exempt for %s step (exec=%s)",
                getattr(step, "id", "?"),
                step_exec,
            )
            return self._mark_step_exempt(step_result, f"exec={step_exec}")

        # Synthesis steps — those that declare inputs which are all present as keys
        # in plan_state — are exempt from the evidence contract.  Their "evidence"
        # is the plan_state data itself (collected by prior steps); requiring
        # file-reference or counted-scope patterns in the prose output is a
        # false-positive for aggregation/report steps.
        #
        # IMPORTANT: data-gathering research steps (allowed_tools contains read/grep/
        # code_search/shell) are NOT exempt even when their declared inputs are present.
        # These steps must produce new evidence from the codebase, not just aggregate
        # what prior steps already collected.  Exempting them caused intent statements
        # ("Let me grep for patterns") with 2 tool calls to pass silently, yielding
        # empty cross_crate_findings=[].
        if plan_state is not None:
            declared_inputs = [
                str(k).strip()
                for k in (
                    getattr(step, "inputs", None)
                    or (step.context.get("inputs", []) if hasattr(step, "context") else [])
                    or []
                )
                if str(k).strip()
            ]
            if declared_inputs and all(inp in plan_state for inp in declared_inputs):
                _GATHERING_TOOLS = frozenset({"read", "grep", "code_search", "shell"})
                _allowed = set(getattr(step, "allowed_tools", []) or [])
                if not (_allowed & _GATHERING_TOOLS):
                    logger.debug(
                        "Evidence contract exempt for step %s — all declared inputs "
                        "(%s) are present in plan_state (synthesis step)",
                        getattr(step, "id", "?"),
                        declared_inputs,
                    )
                    return self._mark_step_exempt(step_result, "synthesis:inputs-in-plan-state")

        if not self._step_requires_evidence_contract(step):
            return self._mark_step_exempt(step_result, "not-required:step-type")

        passed, reason, evidence = self._assess_step_evidence(step, step_result, metadata)
        metadata["evidence_validation"] = {
            "passed": passed,
            "reason": reason,
            **evidence,
        }
        step_result.metadata = metadata
        logger.info(
            "Evidence contract step %s: %s — %s  (tools=%d, chars=%d, file_ref=%s, counted=%s)",
            getattr(step, "id", "?"),
            "PASS" if passed else "FAIL",
            reason,
            evidence.get("tool_calls_used", 0),
            evidence.get("output_chars", 0),
            evidence.get("has_file_reference", False),
            evidence.get("has_counted_scope", False),
        )
        if passed:
            return step_result

        from victor.agent.planning.base import StepResult

        return StepResult(
            success=False,
            output=getattr(step_result, "output", "") or "",
            error=f"Insufficient execution evidence for step {getattr(step, 'id', '?')}: {reason}",
            tool_calls_used=int(getattr(step_result, "tool_calls_used", 0) or 0),
            duration_seconds=float(getattr(step_result, "duration_seconds", 0.0) or 0.0),
            artifacts=list(getattr(step_result, "artifacts", []) or []),
            metadata=metadata,
        )

    @staticmethod
    def _step_requires_evidence_contract(step: Any) -> bool:
        from victor.agent.planning.base import StepType

        step_type = getattr(step, "step_type", None)
        description = str(getattr(step, "description", "") or "").lower()
        allowed_tools = set(getattr(step, "allowed_tools", []) or [])
        if step_type == StepType.REVIEW and re.search(
            r"\b(present|show|discuss)\b.*\b(user|feedback|next steps|remediation)\b",
            description,
        ):
            return False
        if "checklist" in description and (
            "write" in allowed_tools
            or "build" in description
            or "create" in description
            or "present" in description
        ):
            return False
        if step_type in {StepType.RESEARCH, StepType.REVIEW}:
            return True
        return bool(
            re.search(
                r"\b(analyze|audit|check|enumerate|map|review|scan|search)\b"
                r"|summarize findings",
                description,
            )
        )

    @staticmethod
    def _is_directory_listing_only(output: str) -> bool:
        """Return True when output consists solely of file/directory path lines.

        Distinguishes `ls`/`find` path listings (which contain file extensions but no
        semantic content) from actual file reads.  A pure listing never satisfies the
        evidence contract even if `_CONCRETE_FILE_REF_RE` matches.

        Heuristic: ≥60% of non-empty lines are ≤120 chars and either contain a '/'
        separator, end with a recognised source extension, or are bracketed `[path]`
        headers (a common subagent format).  Outputs longer than 500 lines are almost
        certainly real content, not just a directory listing.

        The 60% threshold (lowered from 70%) plus the bracketed-header pattern catches
        subagent outputs of the form (illustrative example)::

            [path/to/manifest]
            path/to/manifest

            [path/to/another-manifest]
            path/to/another-manifest
        """
        lines = [ln.strip() for ln in output.splitlines() if ln.strip()]
        if not lines or len(lines) > 500:
            return False
        bracketed_header_re = re.compile(r"^\[[^\]\s]+\]$")
        ext_re = re.compile(
            r"\.(rs|toml|py|md|json|ya?ml|lock|txt|sh|ts|tsx|js|jsx|"
            r"go|java|kt|rb|gemspec|gradle|gradle\.kts|php|swift|cs|cpp|c|h|hpp)$",
            re.IGNORECASE,
        )
        path_like = sum(
            1
            for ln in lines
            if len(ln) <= 120
            and (
                bracketed_header_re.match(ln)
                or (not re.search(r"\s", ln) and ("/" in ln or ext_re.search(ln)))
            )
        )
        return (path_like / len(lines)) >= 0.60

    @staticmethod
    def _is_degenerate_output(output: str) -> tuple[bool, str]:
        """Detect model degenerate-sampling output (token loops, single-symbol spew).

        Returns ``(is_degenerate, reason)``.  Triggers when:
          * Output ≥ 200 chars yet collapses to ≤3 unique whitespace-delimited tokens.
          * A single short token (≤40 chars) accounts for ≥40% of the output's byte length.
          * Any 2-token bigram repeats >50 times consecutively.
          * Output ≥ 200 chars and ≥99% of characters belong to a single contiguous run
            of word characters (no whitespace at all in a long span).

        These patterns covered the ``EventHandlerExecutorAction``-style loop seen in
        plan step 8 of plan_e53a5ce4 (~50KB single-token spew that previously passed
        the evidence contract).
        """
        if not output:
            return False, ""
        stripped = output.strip()
        if len(stripped) < 200:
            return False, ""

        # Single-token run: no whitespace at all in a 200+ char span
        no_ws = re.sub(r"\s+", "", stripped)
        if len(no_ws) >= 200 and not re.search(r"\s", stripped):
            return (
                True,
                f"output is a single unbroken {len(no_ws)}-char token (model loop)",
            )

        tokens = stripped.split()
        if not tokens:
            return False, ""

        unique_tokens = set(tokens)
        if len(tokens) >= 20 and len(unique_tokens) <= 3:
            sample = next(iter(unique_tokens))
            return (
                True,
                f"output has {len(tokens)} tokens but only {len(unique_tokens)} "
                f"unique (sample={sample!r:.40s})",
            )

        # Single short token dominates byte length
        from collections import Counter

        counts = Counter(tokens)
        top_token, top_count = counts.most_common(1)[0]
        if (
            len(top_token) <= 40
            and top_count >= 10
            and (top_count * len(top_token)) / max(len(stripped), 1) >= 0.40
        ):
            return (
                True,
                f"token {top_token!r:.40s} consumes "
                f"{(top_count * len(top_token)) / len(stripped):.0%} of output "
                f"({top_count} repetitions)",
            )

        # Consecutive bigram repetition
        run_len = 1
        for i in range(1, len(tokens) - 1):
            if tokens[i] == tokens[i - 1]:
                run_len += 1
                if run_len > 50:
                    return (
                        True,
                        f"token {tokens[i]!r:.40s} repeated {run_len}+ times consecutively",
                    )
            else:
                run_len = 1

        return False, ""

    def _assess_step_evidence(
        self,
        step: Any,
        step_result: Any,
        metadata: Dict[str, Any],
    ) -> tuple[bool, str, Dict[str, Any]]:
        output = str(getattr(step_result, "output", "") or "").strip()
        description = str(getattr(step, "description", "") or "")
        full_text = self._step_evidence_text(output, metadata, description)
        normalized = re.sub(r"\s+", " ", output.lower()).strip(" .")
        tool_calls = int(getattr(step_result, "tool_calls_used", 0) or 0)
        artifacts = list(getattr(step_result, "artifacts", []) or [])
        source_count = self._metadata_int(metadata, "source_count")
        produces_key = str((getattr(step, "context", {}) or {}).get("produces", "") or "")
        requires_produced_artifact = bool(
            produces_key and self._requires_nonempty_produces(step, produces_key)
        )

        has_file_ref = bool(self._CONCRETE_FILE_REF_RE.search(full_text))
        is_listing_only = has_file_ref and self._is_directory_listing_only(output)
        if is_listing_only:
            # Path listing matches the file-extension regex but contains no content
            has_file_ref = False

        has_unresolved_tool_markup = bool(
            re.search(
                r"</?tool_call\b|</?parameter\b|</?thinking\b",
                output,
                flags=re.IGNORECASE,
            )
        )

        # Detect content-tool use from output characteristics when explicit tool_names are
        # unavailable (SubAgent.execute_task() returns only a summary string, losing details).
        # A content tool (read, grep, code_search) leaves behind code-like patterns:
        # keywords, braces, function signatures — things that never appear in pure ls output.
        has_content_tool = (
            tool_calls >= 1
            and not is_listing_only
            and bool(
                re.search(
                    r"(?:fn |def |class |impl |pub |use |import |from |const |let |var |async )"
                    r"|\b(?:return|if|for|while|match|switch)\b"
                    r"|[{};](?:\s|$)",
                    output,
                )
            )
        )
        # Manifest dependency-audit output contains TOML/YAML/JSON syntax but no
        # code keywords (e.g. pyproject.toml, package.json, build.gradle).  Recognise
        # manifest reads separately so dependency-audit steps pass the evidence
        # contract without needing code-specific patterns.
        if not has_content_tool and tool_calls >= 1 and not is_listing_only:
            has_content_tool = bool(
                re.search(
                    r"\[(?:dependencies|dev-dependencies|build-dependencies|workspace|package|features)\]"
                    r"|version\s*=\s*[\"'][\d.]"
                    r"|crate-type\s*=\s*\["
                    r"|edition\s*=\s*[\"']\d{4}",
                    output,
                )
            )
        # Also honour explicit tool_names_used if a caller populates it in metadata.
        explicit_tools = set(metadata.get("tool_names_used", []))
        _CONTENT_TOOLS = {"read", "grep", "code_search", "search", "shell", "cat"}
        if explicit_tools & _CONTENT_TOOLS:
            has_content_tool = True

        # Synthesis/write steps: evidence is the act of writing, not file reads.
        # "write" in allowed_tools signals a doc/synthesis step whose output may be
        # a short summary of what was written rather than file content.
        allowed_step_tools = set(getattr(step, "allowed_tools", []) or [])
        _WRITE_TOOLS = {"write", "write_file", "write_to_file"}
        is_write_step = bool(allowed_step_tools & _WRITE_TOOLS)

        evidence = {
            "tool_calls_used": tool_calls,
            "output_chars": len(output),
            "has_file_reference": has_file_ref,
            "has_directory_scope": bool(self._CONCRETE_DIR_REF_RE.search(full_text)),
            "has_counted_scope": self._has_counted_scope(full_text),
            "has_artifacts": bool(artifacts),
            "source_count": source_count,
            "is_directory_listing_only": is_listing_only,
            "has_content_tool": has_content_tool,
            "is_write_step": is_write_step,
            "has_unresolved_tool_markup": has_unresolved_tool_markup,
            "requires_produced_artifact": requires_produced_artifact,
        }

        if not output:
            return False, "step returned no output", evidence
        if normalized in self._WEAK_STEP_OUTPUTS or normalized.startswith("done "):
            return False, "step output is only a generic completion marker", evidence
        # Degenerate model output (token loops, single-symbol spew).  This check runs
        # before any "tools ran, so trust it" branches because such outputs frequently
        # accompany many tool calls and a long byte count, yet contain no information.
        is_degenerate, degenerate_reason = self._is_degenerate_output(output)
        evidence["is_degenerate_output"] = is_degenerate
        if is_degenerate:
            return (
                False,
                f"degenerate model output rejected: {degenerate_reason}",
                evidence,
            )
        if requires_produced_artifact and has_unresolved_tool_markup:
            return (
                False,
                "required produced artifact contains unresolved tool-call markup",
                evidence,
            )
        if requires_produced_artifact and is_listing_only:
            return (
                False,
                "required produced artifact is only a file/path listing",
                evidence,
            )
        if artifacts or source_count > 0:
            return True, "durable artifacts or sources recorded", evidence
        if tool_calls <= 0:
            return False, "no tool-backed execution was recorded", evidence
        # Short outputs that open with an intent phrase ("Let me use grep...",
        # "I will analyze...", etc.) are planning statements, not execution evidence.
        # The code-keyword heuristic false-positives on natural-language "use" or
        # "from" in such sentences, so we gate this check before has_content_tool.
        #
        # Exception: ZAI/GLM-5.1 sometimes emits an intent phrase as its final
        # spawn summary after running many tool calls (the real findings are in the
        # tool results, not the text message). When >= 5 tools ran, trust the execution
        # count — the intent phrase is a spawn-summary artifact, not evidence of no work.
        _stripped = output.strip()
        _is_intent_phrase = (
            len(_stripped) < 200
            and not is_write_step
            and bool(
                re.match(
                    r"(?i)^(?:let me\b|i will\b|i'll\b|i'm going to\b"
                    r"|let's\b|now let me\b|i can\b|i should\b)",
                    _stripped,
                )
            )
        )
        if _is_intent_phrase:
            if requires_produced_artifact:
                return (
                    False,
                    "output is an intent statement, not the required produced artifact",
                    evidence,
                )
            if tool_calls >= 5:
                return (
                    True,
                    "agent ran ≥5 tool calls with intent-phrase summary (spawn artifact) — "
                    "substantive analysis confirmed",
                    evidence,
                )
            return (
                False,
                "output is an intent statement without concrete findings",
                evidence,
            )
        # Multi-tool substantive analysis: 5+ tool calls with 100+ chars confirms real work
        # was done even when the output lacks explicit file:line references.  This covers
        # cross-crate analysis, large search sweeps, and batch code reviews where the
        # sub-agent's summary describes patterns rather than listing source paths.
        # The 100-char floor keeps thin outputs (generic markers, short summaries) still
        # subject to the heuristic checks below.
        if tool_calls >= 5 and len(output) >= 100:
            return (
                True,
                "≥5 tool calls with substantive output — analysis confirmed",
                evidence,
            )
        # Write/synthesis steps: any tool usage + non-trivial output satisfies evidence.
        # The write step's "output" is typically a short summary or confirmation, not the
        # document itself, so the 240-char threshold is inappropriate here.
        if is_write_step and tool_calls >= 1 and len(output) >= 20:
            return True, "write tool used in synthesis step", evidence
        if (
            evidence["has_file_reference"]
            or evidence["has_counted_scope"]
            or evidence["has_content_tool"]
            or (
                evidence["has_directory_scope"]
                and not is_listing_only  # pure path listings don't satisfy the scope check
                and self._is_directory_mapping_step(description, full_text)
            )
        ):
            return True, "concrete file or scope evidence found", evidence
        if tool_calls >= 3 and len(output) >= 240:
            return True, "multi-tool analysis produced a substantive summary", evidence
        return (
            False,
            "missing concrete file references, counts, artifacts, or scoped findings",
            evidence,
        )

    @classmethod
    def _step_evidence_text(
        cls,
        output: str,
        metadata: Dict[str, Any],
        step_description: str = "",
    ) -> str:
        parts = [output]
        if step_description:
            parts.append(step_description)
        for key in ("full_response", "evidence", "sources", "files", "changed_files"):
            value = metadata.get(key)
            if value:
                parts.append(str(value))
        return "\n".join(parts)

    @staticmethod
    def _metadata_int(metadata: Dict[str, Any], key: str) -> int:
        try:
            return int(metadata.get(key, 0) or 0)
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _has_counted_scope(text: str) -> bool:
        lower = text.lower()
        if not re.search(r"\b\d+\b", lower):
            return False
        return any(
            term in lower
            for term in (
                "file",
                "files",
                "line",
                "lines",
                "match",
                "matches",
                "module",
                "modules",
                "workspace",
                "workspaces",
                "crate",
                "crates",
                "usage",
                "usages",
                "scanned",
                "searched",
            )
        )

    @staticmethod
    def _is_directory_mapping_step(description: str, text: str) -> bool:
        lower = f"{description}\n{text}".lower()
        return any(term in lower for term in ("directory tree", "dir tree", "src/", "benches/"))

    def _effective_team_plan_concurrency(self) -> int:
        """Return the bounded concurrency for independent team-plan steps."""
        env_value = os.getenv("VICTOR_PLAN_MAX_CONCURRENT_AGENTS") or os.getenv(
            "VICTOR_PLAN_MAX_PARALLEL_STEPS"
        )
        if env_value:
            try:
                return max(1, int(env_value))
            except ValueError:
                logger.warning(
                    "Ignoring invalid plan concurrency override: %s",
                    env_value,
                )
        return max(1, int(self.config.max_parallel_steps or 1))

    @staticmethod
    def _skip_specific_steps(
        execution_plan: "ExecutionPlan",
        step_ids: list[str],
    ) -> None:
        """Mark the given PENDING step IDs as SKIPPED (conditional branch routing)."""
        from victor.agent.planning.base import StepStatus

        skip_set = {str(s) for s in step_ids}
        for step in execution_plan.steps:
            if step.id in skip_set and step.status == StepStatus.PENDING:
                step.status = StepStatus.SKIPPED

    @staticmethod
    def _step_can_continue_after_failed_deps(step: Any) -> bool:
        """Return whether a pending step may run with partial upstream evidence.

        Read-only analysis/review/documentation can still produce useful findings when an
        upstream branch fails. Effectful steps must not run on missing prerequisites.
        """
        from victor.agent.planning.base import StepType
        from victor.agent.planning.team_execution import PlanningTeamExecutionAdapter

        description = str(getattr(step, "description", "") or "").lower()
        step_type = getattr(step, "step_type", None)
        step_type_value = str(getattr(step_type, "value", step_type) or "").lower()
        allowed_tools = {str(tool).lower() for tool in getattr(step, "allowed_tools", []) or []}
        execution = str(
            getattr(step, "execution", "")
            or (getattr(step, "context", {}) or {}).get("execution", "")
            or ""
        ).lower()

        effectful_words = {
            "apply",
            "change",
            "commit",
            "deploy",
            "edit",
            "fix",
            "format",
            "install",
            "merge",
            "migrate",
            "modify",
            "patch",
            "publish",
            "release",
            "remove",
            "run tests",
            "start server",
            "update files",
            "write code",
        }
        effectful_tools = {
            "apply_patch",
            "edit",
            "write_file",
            "delete_file",
            "git",
            "cargo",
            "npm",
            "pytest",
            "make",
            "maturin",
        }
        if (
            step_type in {StepType.IMPLEMENTATION, StepType.TESTING, StepType.DEPLOYMENT}
            or step_type_value in {"implementation", "testing", "deployment"}
            or execution in {"tool", "deploy"}
            or allowed_tools & effectful_tools
            or any(word in description for word in effectful_words)
        ):
            # Report/synthesis artifacts are allowed to run because their purpose is
            # to explain partial completion, not to mutate source or external state.
            if not PlanningTeamExecutionAdapter._is_synthesis_step(step):
                return False

        read_only_types = {
            StepType.RESEARCH,
            StepType.REVIEW,
            StepType.PLANNING,
        }
        read_only_words = (
            "analy",
            "audit",
            "catalog",
            "checklist",
            "cross-crate",
            "document",
            "evaluate",
            "findings",
            "inspect",
            "inventory",
            "map",
            "profile",
            "read",
            "report",
            "review",
            "scan",
            "summar",
            "synthesize",
            "verify",
        )
        if PlanningTeamExecutionAdapter._is_synthesis_step(step):
            return True
        return bool(
            step_type in read_only_types
            or step_type_value in {"research", "review", "planning", "documentation"}
            or any(word in description for word in read_only_words)
        )

    @staticmethod
    def _skip_team_plan_dependents(
        execution_plan: "ExecutionPlan",
        failed_step_ids: list[str],
    ) -> None:
        """Mark pending transitive dependents blocked after a failed team-plan step.

        Read-only analysis/review/report steps are exempt: they aggregate partial
        plan_state and can still produce useful findings when upstream analysis steps
        partially failed. Effectful dependents remain blocked.
        """
        from victor.agent.planning.base import StepStatus
        from victor.agent.planning.base import StepResult

        failed = set(failed_step_ids)
        changed = True
        while changed:
            changed = False
            for step in execution_plan.steps:
                if step.status != StepStatus.PENDING:
                    continue
                if any(dep in failed for dep in step.depends_on):
                    if PlanningRuntimeService._step_can_continue_after_failed_deps(step):
                        failed_deps = [dep for dep in step.depends_on if dep in failed]
                        step.depends_on = [dep for dep in step.depends_on if dep not in failed]
                        step.context.setdefault("partial_failed_dependencies", []).extend(
                            failed_deps
                        )
                        step.context["partial_execution"] = True
                        continue
                    failed_deps = [dep for dep in step.depends_on if dep in failed]
                    step.status = StepStatus.BLOCKED
                    step.result = StepResult(
                        success=False,
                        output="",
                        error=f"Step blocked: failed required dependencies {failed_deps}",
                        metadata={"blocked_by_failed_dependencies": failed_deps},
                    )
                    failed.add(step.id)
                    changed = True

    async def _generate_final_response(
        self,
        plan: ReadableTaskPlan,
        result: "PlanResult",
    ) -> CompletionResponse:
        """Generate final response summarizing plan execution.

        Args:
            plan: Original plan
            result: Execution result

        Returns:
            CompletionResponse with summary
        """
        # Build summary prompt
        summary_prompt = self._build_summary_prompt(plan, result)

        # Get response from provider
        response = await self.orchestrator.provider.chat(
            messages=[{"role": "user", "content": summary_prompt}],
            model=self.orchestrator.model,
            temperature=0.7,  # Lower temperature for summaries
            max_tokens=self.orchestrator.max_tokens,
        )

        self._append_artifact_paths_to_response(response, result)
        return response

    async def _generate_plan_rejected_response(
        self,
        plan: ReadableTaskPlan,
    ) -> CompletionResponse:
        """Generate response when user rejects plan.

        Args:
            plan: The rejected plan

        Returns:
            CompletionResponse explaining the rejection
        """
        rejection_prompt = f"""The user has rejected the following plan:

Plan: {plan.name}
Complexity: {plan.complexity.value}
Steps: {len(plan.steps)}

Please acknowledge that the plan was rejected and ask if the user would like to:
1. Modify the plan
2. Try a different approach
3. Answer the question without a structured plan

Keep your response concise and helpful.
"""

        response = await self.orchestrator.provider.chat(
            messages=[{"role": "user", "content": rejection_prompt}],
            model=self.orchestrator.model,
            temperature=0.7,
            max_tokens=self.orchestrator.max_tokens,
        )

        return response

    def _build_summary_prompt(
        self,
        plan: ReadableTaskPlan,
        result: "PlanResult",
    ) -> str:
        """Build a prompt for summarizing plan execution.

        Args:
            plan: Original plan
            result: Execution result

        Returns:
            Summary prompt
        """
        parts = [
            "Summarize the planned task execution using only the evidence below.",
            f"Task: {plan.name}",
            f"Complexity: {plan.complexity.value}",
            f"Steps completed: {result.steps_completed}/{result.total_steps}",
            f"Overall success: {result.success}",
            "",
            "Steps executed:",
        ]

        for i, step in enumerate(plan.steps):
            step_id, step_type, step_desc, _ = self._unpack_step(step)
            step_result = result.step_results.get(str(step_id)) or result.step_results.get(step_id)
            if step_result is not None:
                status = "completed" if step_result.success else "failed"
                parts.append(f"  - {step_id}. [{step_type}] {status}: {step_desc}")
                if step_result.output:
                    parts.append(f"    Evidence: {step_result.output[:2000]}")
                artifacts = list(getattr(step_result, "artifacts", []) or [])
                if artifacts:
                    parts.append(f"    Artifacts: {', '.join(str(a) for a in artifacts[:5])}")
                if step_result.error:
                    parts.append(f"    Error: {step_result.error}")
                evidence_line = self._format_evidence_validation_for_summary(step_result)
                if evidence_line:
                    parts.append(f"    {evidence_line}")
            else:
                status = "not run" if i >= result.steps_completed else "unknown"
                parts.append(f"  - {step_id}. [{step_type}] {status}: {step_desc}")

        evidence_summary = self._format_evidence_validation_summary_for_summary(plan, result)
        if evidence_summary:
            parts.extend(["", evidence_summary])

        provider_retry_lines = self._format_provider_retry_diagnostics_for_summary(result)
        if provider_retry_lines:
            parts.extend(["", "Provider retry diagnostics:", *provider_retry_lines])

        error_message = getattr(result, "error_message", "")
        if error_message:
            parts.extend(
                [
                    "",
                    f"Error: {error_message}",
                ]
            )
        if result.final_output:
            parts.extend(
                [
                    "",
                    "Aggregated step output:",
                    result.final_output[:6000],
                ]
            )

        parts.extend(
            [
                "",
                "Produce a concise user-facing summary. If a step failed or lacks evidence, "
                "state that plainly and do not invent repository findings.",
                "Do not report failed evidence-validation steps as completed repository analysis.",
                "Treat steps missing evidence validation as unverified when summarizing "
                "repository findings.",
                "Mention provider retry diagnostics separately from repository findings.",
            ]
        )

        return "\n".join(parts)

    @classmethod
    def _append_artifact_paths_to_response(cls, response: CompletionResponse, result: Any) -> None:
        """Guarantee durable artifact paths appear in the final user-visible response."""
        artifacts = cls._collect_step_artifacts(result)
        if not artifacts:
            return
        content = response.content or ""
        missing = [artifact for artifact in artifacts if artifact not in content]
        if not missing:
            return
        artifact_lines = "\n".join(f"- `{artifact}`" for artifact in missing[:10])
        suffix = f"\n\nFull artifacts:\n{artifact_lines}"
        response.content = content.rstrip() + suffix

    @staticmethod
    def _collect_step_artifacts(result: Any) -> list[str]:
        artifacts: list[str] = []
        seen: set[str] = set()
        for step_result in dict(getattr(result, "step_results", {}) or {}).values():
            for artifact in list(getattr(step_result, "artifacts", []) or []):
                artifact_str = str(artifact)
                if artifact_str and artifact_str not in seen:
                    seen.add(artifact_str)
                    artifacts.append(artifact_str)
        return artifacts

    @classmethod
    def _format_evidence_validation_summary_for_summary(
        cls,
        plan: ReadableTaskPlan,
        result: Any,
    ) -> str:
        step_results = getattr(result, "step_results", None)
        if not isinstance(step_results, dict):
            return ""

        counts = {"passed": 0, "failed": 0, "exempt": 0, "missing": 0, "not_run": 0}
        for step in plan.steps:
            step_id = cls._unpack_step(step)[0] if step else ""
            step_result = step_results.get(str(step_id)) or step_results.get(step_id)
            if step_result is None:
                counts["not_run"] += 1
                continue

            validation = cls._evidence_validation_metadata(step_result)
            if validation is None:
                counts["missing"] += 1
            elif validation.get("exempt"):
                counts["exempt"] += 1
            elif validation.get("passed"):
                counts["passed"] += 1
            else:
                counts["failed"] += 1

        if not any(counts.values()):
            return ""
        return (
            "Evidence validation summary: "
            f"passed={counts['passed']}; failed={counts['failed']}; "
            f"exempt={counts['exempt']}; missing={counts['missing']}; not_run={counts['not_run']}"
        )

    @staticmethod
    def _format_evidence_validation_for_summary(step_result: Any) -> str:
        validation = PlanningRuntimeService._evidence_validation_metadata(step_result)
        if validation is None:
            return ""

        status = "passed" if validation.get("passed") else "failed"
        reason = str(validation.get("reason") or "not specified")
        tool_calls = validation.get("tool_calls_used", getattr(step_result, "tool_calls_used", 0))
        file_ref = bool(validation.get("has_file_reference"))
        counted_scope = bool(validation.get("has_counted_scope"))
        artifacts = bool(validation.get("has_artifacts"))
        source_count = validation.get("source_count", 0)
        return (
            f"Evidence validation: {status}; reason={reason}; "
            f"tool_calls={tool_calls}; file_ref={file_ref}; counted_scope={counted_scope}; "
            f"artifacts={artifacts}; source_count={source_count}"
        )

    @staticmethod
    def _evidence_validation_metadata(step_result: Any) -> Optional[dict[str, Any]]:
        metadata = getattr(step_result, "metadata", None)
        if not isinstance(metadata, dict):
            return None
        validation = metadata.get("evidence_validation")
        if not isinstance(validation, dict):
            return None
        return validation

    @classmethod
    def _format_provider_retry_diagnostics_for_summary(cls, result: Any) -> list[str]:
        diagnostics = cls._collect_provider_retry_diagnostics(result)
        if not diagnostics:
            return []

        lines = []
        for diagnostic in diagnostics:
            provider = cls._diagnostic_value(diagnostic, "provider", "unknown")
            model = cls._diagnostic_value(diagnostic, "model", "unknown")
            retry_count = cls._diagnostic_value(diagnostic, "retry_count", "unknown")
            last_error = cls._diagnostic_value(
                diagnostic,
                "last_error",
                cls._diagnostic_value(diagnostic, "error", ""),
            )
            line = f"  - provider={provider}; model={model}; retry_count={retry_count}"
            if last_error:
                line = f"{line}; last_error={str(last_error)[:500]}"
            lines.append(line)
        return lines

    @classmethod
    def _collect_provider_retry_diagnostics(cls, result: Any) -> list[Any]:
        diagnostics: list[Any] = []
        cls._extend_retry_diagnostics(diagnostics, getattr(result, "metadata", None))

        step_results = getattr(result, "step_results", None)
        if isinstance(step_results, dict):
            for step_result in step_results.values():
                cls._extend_retry_diagnostics(
                    diagnostics,
                    getattr(step_result, "metadata", None),
                )
        return diagnostics

    @classmethod
    def _extend_retry_diagnostics(cls, diagnostics: list[Any], metadata: Any) -> None:
        if not isinstance(metadata, dict):
            return
        for key in ("provider_retry_diagnostics", "provider_retries", "retry_events"):
            value = metadata.get(key)
            if value:
                diagnostics.extend(cls._coerce_diagnostic_items(value))

    @staticmethod
    def _coerce_diagnostic_items(value: Any) -> list[Any]:
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        return [value]

    @staticmethod
    def _diagnostic_value(diagnostic: Any, key: str, default: Any = "") -> Any:
        if isinstance(diagnostic, dict):
            return diagnostic.get(key, default)
        return getattr(diagnostic, key, default)

    async def _direct_chat(self, user_message: str) -> CompletionResponse:
        """Fallback to direct chat without planning.

        Args:
            user_message: User's message

        Returns:
            CompletionResponse from chat
        """
        # Delegate to orchestrator's chat
        return await self.orchestrator.chat(user_message)

    def set_planning_mode(self, mode: PlanningMode) -> None:
        """Set the planning mode.

        Args:
            mode: Planning mode to use
        """
        self._planning_mode = mode
        logger.info(f"Planning mode set to: {mode.value}")

    def get_active_plan(self) -> Optional[ReadableTaskPlan]:
        """Get the currently active plan.

        Returns:
            Active plan or None
        """
        return self.active_plan

    def clear_active_plan(self) -> None:
        """Clear the active plan."""
        self.active_plan = None
        logger.info("Active plan cleared")

    async def _compact_context_if_needed(self) -> None:
        """Compact conversation context if it's getting too large.

        This helps prevent context overflow during long planning sessions.
        """
        orch = self.orchestrator
        if await self._compact_with_context_service():
            return

        has_capability = getattr(orch, "has_capability", None)
        get_capability_value = getattr(orch, "get_capability_value", None)
        if (
            callable(has_capability)
            and callable(get_capability_value)
            and has_capability("context_compactor")
            and get_capability_value("context_compactor")
        ):
            try:
                # Get current query from public conversation API
                current_query = ""
                if hasattr(orch, "conversation") and orch.conversation:
                    current_query = orch.conversation.get_latest_user_message() or ""

                # Check and compact
                compactor = get_capability_value("context_compactor")
                compaction_result = compactor.check_and_compact(
                    current_query=current_query,
                    force=False,
                    tool_call_count=orch.tool_calls_used,
                    task_complexity="complex",  # Planning tasks are complex
                )

                if compaction_result.action_taken:
                    logger.info(
                        f"Context compacted: {compaction_result.messages_removed} messages removed, "
                        f"{compaction_result.tokens_freed} tokens freed"
                    )
            except Exception as e:
                logger.warning(f"Context compaction failed: {e}")

    async def _compact_with_context_service(self) -> bool:
        """Use the canonical context service before legacy compactor fallback."""
        context_service = getattr(self.orchestrator, "_context_service", None)
        if context_service is None:
            return False

        try:
            result = await compact_context_if_recommended(
                context_service,
                strategy=self._context_compaction_strategy(),
                min_messages=6,
            )
            if not result.handled:
                return False
            if result.messages_removed > 0:
                logger.info(
                    "ContextService compacted planning context: %s messages removed",
                    result.messages_removed,
                )
            return True
        except Exception as e:
            logger.warning(f"Context service compaction failed: {e}")
            return True

    def _context_compaction_strategy(self) -> str:
        settings = getattr(self.orchestrator, "settings", None)
        return str(getattr(settings, "context_compaction_strategy", "tiered") or "tiered")


# Compatibility alias. New service-owned call sites should import
# PlanningRuntimeService; legacy coordinator paths continue to resolve.
PlanningCoordinator = PlanningRuntimeService
