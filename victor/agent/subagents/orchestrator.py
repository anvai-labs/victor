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

"""Sub-agent orchestrator for spawning and managing sub-agents.

This module provides the SubAgentOrchestrator class that enables:
- Spawning sub-agents with role-specific configurations
- Parallel execution of multiple sub-agents (fan-out)
- Sequential task delegation
- Result aggregation

Example:
    orchestrator = SubAgentOrchestrator(parent_orchestrator)

    # Spawn a single sub-agent
    result = await orchestrator.spawn(
        SubAgentRole.RESEARCHER,
        "Find all API endpoints in the codebase",
        tool_budget=15,
    )

    # Fan out to multiple sub-agents in parallel
    results = await orchestrator.fan_out([
        SubAgentTask(SubAgentRole.RESEARCHER, "Research auth patterns"),
        SubAgentTask(SubAgentRole.RESEARCHER, "Research database patterns"),
        SubAgentTask(SubAgentRole.PLANNER, "Plan auth implementation"),
    ], max_concurrent=3)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, List, Optional, Set

from victor.agent.runtime.naming import build_display_name, generate_agent_id
from victor.agent.subagents.base import (
    SubAgent,
    SubAgentConfig,
    SubAgentResult,
    SubAgentRole,
)

if TYPE_CHECKING:
    from victor.agent.orchestrator import AgentOrchestrator
    from victor.agent.services.context_lifecycle_service import ContextLifecycleService
    from victor.agent.subagents.protocols import RoleToolProvider
    from victor.providers.base import StreamChunk
    from victor.workflows.definition import ConstraintsProtocol

logger = logging.getLogger(__name__)


# Default tool sets for each role
ROLE_DEFAULT_TOOLS: Dict[SubAgentRole, List[str]] = {
    SubAgentRole.RESEARCHER: [
        "read",
        "ls",
        "grep",
        "search",
        "code_search",
        "semantic_code_search",
        "web_search",
        "web_fetch",
    ],
    SubAgentRole.PLANNER: [
        "read",
        "ls",
        "grep",
        "search",
        "plan_files",
    ],
    SubAgentRole.EXECUTOR: [
        "read",
        "write",
        "edit",
        "ls",
        "grep",
        "search",
        "shell",
        "test",
        "git",
    ],
    SubAgentRole.REVIEWER: [
        "read",
        "ls",
        "grep",
        "search",
        "git",
        "test",
        "shell",
    ],
    SubAgentRole.TESTER: [
        "read",
        "write",
        "ls",
        "grep",
        "search",
        "test",
        "shell",
    ],
}

# Default budgets for each role
ROLE_DEFAULT_BUDGETS: Dict[SubAgentRole, int] = {
    SubAgentRole.RESEARCHER: 15,
    SubAgentRole.PLANNER: 10,
    SubAgentRole.EXECUTOR: 30,
    SubAgentRole.REVIEWER: 15,
    SubAgentRole.TESTER: 20,
}

# Default context limits for each role (chars, used when provider context unknown)
ROLE_DEFAULT_CONTEXT: Dict[SubAgentRole, int] = {
    SubAgentRole.RESEARCHER: 50000,
    SubAgentRole.PLANNER: 30000,
    SubAgentRole.EXECUTOR: 80000,
    SubAgentRole.REVIEWER: 40000,
    SubAgentRole.TESTER: 50000,
}

# Role context fractions — percentage of model's context window per role
ROLE_CONTEXT_FRACTIONS: Dict[SubAgentRole, float] = {
    SubAgentRole.RESEARCHER: 0.25,
    SubAgentRole.PLANNER: 0.15,
    SubAgentRole.EXECUTOR: 0.50,
    SubAgentRole.REVIEWER: 0.20,
    SubAgentRole.TESTER: 0.25,
}


def get_context_for_role(
    role: SubAgentRole,
    provider: str = "ollama",
    model: Optional[str] = None,
) -> int:
    """Calculate context limit for a subagent role based on provider capabilities.

    Queries provider_context_limits.yaml for the model's actual context window,
    then allocates a fraction based on role. Falls back to ROLE_DEFAULT_CONTEXT
    if provider limits are unavailable.

    Args:
        role: SubAgent role
        provider: Provider name (e.g., "ollama", "deepseek", "anthropic")
        model: Optional model name for model-specific limits

    Returns:
        Context limit in characters
    """
    try:
        from victor.config.config_loaders import get_provider_limits

        limits = get_provider_limits(provider, model)
        model_context_tokens = limits.effective_context
        model_context_chars = model_context_tokens * 4  # ~4 chars/token

        fraction = ROLE_CONTEXT_FRACTIONS.get(role, 0.25)
        scaled = int(model_context_chars * fraction)

        # Cap at 200K chars to prevent memory issues
        result = min(scaled, 200000)

        logger.debug(
            "Context for %s: %d chars (model=%d tokens, fraction=%.0f%%)",
            role.value,
            result,
            model_context_tokens,
            fraction * 100,
        )
        return result

    except Exception:
        return ROLE_DEFAULT_CONTEXT.get(role, 50000)


@dataclass
class SubAgentTask:
    """Represents a task to be executed by a sub-agent.

    Used with fan_out() to specify multiple tasks for parallel execution.

    Attributes:
        role: Sub-agent role
        task: Task description
        tool_budget: Optional budget override
        allowed_tools: Optional tool list override
        context_limit: Optional context limit override
    """

    role: SubAgentRole
    task: str
    tool_budget: Optional[int] = None
    allowed_tools: Optional[List[str]] = None
    context_limit: Optional[int] = None
    member_id: Optional[str] = None
    agent_id: Optional[str] = None
    display_name: Optional[str] = None
    team_id: Optional[str] = None
    plan_id: Optional[str] = None
    plan_step_id: Optional[str] = None
    parent_session_id: Optional[str] = None
    child_session_id: Optional[str] = None


@dataclass
class FanOutResult:
    """Result from fan_out execution.

    Contains results from all sub-agents and aggregate statistics.

    Attributes:
        results: List of individual sub-agent results
        all_success: True if all sub-agents succeeded
        total_tool_calls: Sum of tool calls across all sub-agents
        total_duration: Total execution time (max of parallel tasks)
        errors: List of error messages from failed sub-agents
    """

    results: List[SubAgentResult]
    all_success: bool
    total_tool_calls: int
    total_duration: float
    errors: List[str] = field(default_factory=list)


class SubAgentOrchestrator:
    """Orchestrates spawning and execution of sub-agents.

    Provides high-level API for creating sub-agents with sensible defaults
    and managing parallel execution.

    Attributes:
        parent: Parent orchestrator that owns this sub-agent orchestrator
        active_subagents: Set of currently running sub-agents

    Example:
        orchestrator = SubAgentOrchestrator(parent)

        # Simple spawn with role defaults
        result = await orchestrator.spawn(
            SubAgentRole.RESEARCHER,
            "Find authentication patterns",
        )

        # Spawn with custom configuration
        result = await orchestrator.spawn(
            SubAgentRole.EXECUTOR,
            "Implement JWT authentication",
            tool_budget=40,
            allowed_tools=["read", "write", "edit", "shell"],
        )
    """

    def __init__(
        self,
        parent_orchestrator: "AgentOrchestrator",
        role_provider: Optional["RoleToolProvider"] = None,
        vertical: Optional[str] = None,
        context_lifecycle: Optional["ContextLifecycleService"] = None,
    ):
        """Initialize sub-agent orchestrator.

        Args:
            parent_orchestrator: Parent orchestrator that spawns sub-agents
            role_provider: Optional custom role provider for vertical-specific tools.
                          If None, uses global get_role_tool_provider().
            vertical: Optional vertical name for context-aware tool selection.
        """
        from victor.agent.subagents.protocols import get_role_tool_provider

        self.parent = parent_orchestrator
        self.active_subagents: Set[SubAgent] = set()
        self._role_provider = role_provider or get_role_tool_provider()
        self._vertical = vertical
        self._context_lifecycle = context_lifecycle or getattr(
            parent_orchestrator,
            "_context_lifecycle_service",
            None,
        )

        logger.info(
            f"SubAgentOrchestrator initialized"
            f"{f' for vertical: {vertical}' if vertical else ''}"
        )

    async def spawn(
        self,
        role: SubAgentRole,
        task: str,
        tool_budget: Optional[int] = None,
        allowed_tools: Optional[List[str]] = None,
        context_limit: Optional[int] = None,
        can_spawn_subagents: bool = False,
        timeout_seconds: int = 300,
        disable_embeddings: bool = False,
        constraints: Optional["ConstraintsProtocol"] = None,
        member_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        display_name: Optional[str] = None,
        team_id: Optional[str] = None,
        plan_id: Optional[str] = None,
        plan_step_id: Optional[str] = None,
        parent_session_id: Optional[str] = None,
        child_session_id: Optional[str] = None,
        result_summary_max_chars: Optional[int] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        reasoning_effort: Optional[str] = None,
    ) -> SubAgentResult:
        """Spawn a sub-agent to execute a task.

        Creates a sub-agent with the specified role and task, using sensible
        defaults for unspecified parameters based on the role.

        Args:
            role: Sub-agent role specialization
            task: Task description for the sub-agent
            tool_budget: Maximum tool calls (default: role-specific)
            allowed_tools: List of allowed tools (default: role-specific)
            context_limit: Maximum context size (default: role-specific)
            can_spawn_subagents: Whether sub-agent can spawn children
            timeout_seconds: Maximum execution time
            disable_embeddings: Disable codebase embeddings (workflow service mode)
            constraints: Optional task constraints for write policy activation
            provider: Optional provider name override for heterogeneous teams
                (e.g. "openai", "anthropic"). None inherits the parent provider.
            model: Optional model override paired with ``provider``.
            temperature: Optional per-member sampling temperature override
                (None inherits the parent's temperature).
            reasoning_effort: Optional per-member reasoning effort ("low"/
                "medium"/"high") for reasoning-capable models (None inherits/omits;
                forwarded only to providers/models that support it).

        Returns:
            SubAgentResult with execution outcome

        Example:
            result = await orchestrator.spawn(
                SubAgentRole.RESEARCHER,
                "Find all database models in the codebase",
            )
            if result.success:
                print(result.summary)
        """
        # Activate constraints if provided
        if constraints:
            from victor.agent.constraint_activation_service import (
                get_constraint_activator,
            )

            activator = get_constraint_activator()
            result = activator.activate_constraints(constraints, self._vertical)
            if not result.success:
                logger.warning(f"SubAgent constraint activation failed: {result.error}")
            else:
                logger.debug(f"SubAgent constraints activated for role: {role.value}")

        # Apply role-specific defaults using the role provider (OCP-compliant)
        role_name = role.value if hasattr(role, "value") else str(role)

        if allowed_tools is not None:
            effective_tools = allowed_tools
        else:
            # Use role provider for vertical-aware tool selection
            effective_tools = self._role_provider.get_tools_for_role(role_name, self._vertical)

        effective_budget = tool_budget or self._role_provider.get_budget_for_role(role_name)
        effective_agent_id = agent_id or generate_agent_id(role_name)
        effective_display_name = display_name or build_display_name(role_name, task=task)
        if context_limit:
            effective_context = context_limit
        else:
            # Try provider-aware context sizing first, fallback to role defaults
            provider_name = getattr(self.parent, "provider_name", "ollama")
            model_name = getattr(self.parent, "model", None)
            effective_context = get_context_for_role(role, provider_name, model_name)

        # Heterogeneous execution: resolve a per-member provider override when
        # requested (cross-vendor teams). Fail-open — inherit the parent provider
        # on any resolution error so a misconfigured override never blocks the run.
        provider_override = None
        model_override = None
        if provider:
            provider_override = await self._resolve_override_provider(provider, model)
            if provider_override is not None:
                model_override = model

        # Create configuration
        config = SubAgentConfig(
            role=role,
            task=task,
            allowed_tools=effective_tools,
            tool_budget=effective_budget,
            context_limit=effective_context,
            can_spawn_subagents=can_spawn_subagents,
            timeout_seconds=timeout_seconds,
            disable_embeddings=disable_embeddings,
            member_id=member_id,
            agent_id=effective_agent_id,
            display_name=effective_display_name,
            team_id=team_id,
            plan_id=plan_id,
            plan_step_id=plan_step_id,
            parent_session_id=parent_session_id,
            child_session_id=child_session_id,
            result_summary_max_chars=result_summary_max_chars,
            provider_override=provider_override,
            model_override=model_override,
            temperature_override=temperature,
            reasoning_effort_override=reasoning_effort,
        )

        # Create and execute sub-agent
        subagent = SubAgent(config, self.parent, context_lifecycle=self._context_lifecycle)
        self.active_subagents.add(subagent)

        try:
            result = await asyncio.wait_for(
                subagent.execute(),
                timeout=timeout_seconds,
            )
            self._attach_identity_metadata(result, config)
            return result
        except asyncio.TimeoutError:
            logger.warning(f"{role.value} sub-agent timed out after {timeout_seconds}s")
            timeout_result = SubAgentResult(
                success=False,
                summary=f"Sub-agent timed out after {timeout_seconds} seconds",
                details={"role": role.value, "task": task[:200]},
                tool_calls_used=0,
                context_size=0,
                duration_seconds=float(timeout_seconds),
                error=f"Timeout after {timeout_seconds}s",
            )
            self._attach_identity_metadata(timeout_result, config)
            return timeout_result
        finally:
            self.active_subagents.discard(subagent)
            # Deactivate constraints after spawn completes
            if constraints:
                from victor.agent.constraint_activation_service import (
                    get_constraint_activator,
                )

                activator = get_constraint_activator()
                activator.deactivate_constraints()
                logger.debug(f"SubAgent constraints deactivated for role: {role.value}")

    async def _resolve_override_provider(
        self, provider: str, model: Optional[str]
    ) -> Optional[Any]:
        """Resolve a provider instance for a per-member override (fail-open).

        Builds a managed provider for ``provider``/``model`` via the shared
        factory, resolving the API key the same way the main provider does.
        Returns None on any failure so the caller inherits the parent provider
        rather than blocking the run.

        Args:
            provider: Provider name (e.g. "openai", "anthropic").
            model: Optional model id; falls back to the provider's default.

        Returns:
            A provider instance, or None to inherit the parent provider.
        """
        try:
            from victor.config.api_keys import get_api_key
            from victor.providers.factory import ManagedProviderFactory

            effective_model = model or getattr(self.parent, "model", None) or ""
            api_key = get_api_key(provider)
            return await ManagedProviderFactory.create(
                provider_name=provider,
                model=effective_model,
                api_key=api_key,
            )
        except Exception as exc:
            logger.warning(
                "Per-member provider override '%s' could not be resolved (%s); "
                "inheriting parent provider.",
                provider,
                exc,
            )
            return None

    @staticmethod
    def _attach_identity_metadata(result: SubAgentResult, config: SubAgentConfig) -> None:
        """Attach stable team/session identity metadata to a sub-agent result."""
        result.details.setdefault("member_id", config.member_id)
        result.details.setdefault("agent_id", config.agent_id)
        result.details.setdefault("display_name", config.display_name)
        result.details.setdefault("team_id", config.team_id)
        result.details.setdefault("plan_id", config.plan_id)
        result.details.setdefault("plan_step_id", config.plan_step_id)
        result.details.setdefault("parent_session_id", config.parent_session_id)
        result.details.setdefault("child_session_id", config.child_session_id)

    async def fan_out(
        self,
        tasks: List[SubAgentTask],
        max_concurrent: int = 3,
    ) -> FanOutResult:
        """Execute multiple sub-agent tasks in parallel.

        Spawns multiple sub-agents concurrently, respecting the concurrency
        limit. Results are returned in the same order as input tasks.

        Args:
            tasks: List of tasks to execute
            max_concurrent: Maximum concurrent sub-agents

        Returns:
            FanOutResult with all results and aggregate statistics

        Example:
            results = await orchestrator.fan_out([
                SubAgentTask(SubAgentRole.RESEARCHER, "Find auth code"),
                SubAgentTask(SubAgentRole.RESEARCHER, "Find database models"),
                SubAgentTask(SubAgentRole.PLANNER, "Plan refactoring"),
            ])

            for i, result in enumerate(results.results):
                print(f"Task {i}: {'Success' if result.success else 'Failed'}")
        """
        start_time = time.time()
        semaphore = asyncio.Semaphore(max_concurrent)

        async def run_with_semaphore(task: SubAgentTask) -> SubAgentResult:
            async with semaphore:
                return await self.spawn(
                    role=task.role,
                    task=task.task,
                    tool_budget=task.tool_budget,
                    allowed_tools=task.allowed_tools,
                    context_limit=task.context_limit,
                    member_id=task.member_id,
                    agent_id=task.agent_id,
                    display_name=task.display_name,
                    team_id=task.team_id,
                    plan_id=task.plan_id,
                    plan_step_id=task.plan_step_id,
                    parent_session_id=task.parent_session_id,
                    child_session_id=task.child_session_id,
                )

        logger.info(
            f"Fan-out: spawning {len(tasks)} sub-agents " f"(max concurrent: {max_concurrent})"
        )

        # Execute all tasks
        results = await asyncio.gather(
            *[run_with_semaphore(task) for task in tasks],
            return_exceptions=True,
        )

        # Process results
        processed_results: List[SubAgentResult] = []
        errors: List[str] = []
        total_tool_calls = 0

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Convert exception to failed result
                error_msg = (
                    f"Task {i} ({tasks[i].role.value}): {type(result).__name__}: {str(result)}"
                )
                errors.append(error_msg)
                processed_results.append(
                    SubAgentResult(
                        success=False,
                        summary=f"Exception: {str(result)[:450]}",
                        details={"exception": str(result)},
                        tool_calls_used=0,
                        context_size=0,
                        duration_seconds=0,
                        error=error_msg,
                    )
                )
            else:
                processed_results.append(result)
                total_tool_calls += result.tool_calls_used
                if not result.success and result.error:
                    errors.append(f"Task {i} ({tasks[i].role.value}): {result.error}")

        total_duration = time.time() - start_time
        all_success = all(r.success for r in processed_results)

        logger.info(
            f"Fan-out complete: {sum(1 for r in processed_results if r.success)}/{len(tasks)} "
            f"succeeded, {total_tool_calls} total tool calls, {total_duration:.1f}s"
        )

        return FanOutResult(
            results=processed_results,
            all_success=all_success,
            total_tool_calls=total_tool_calls,
            total_duration=total_duration,
            errors=errors,
        )

    async def stream_spawn(
        self,
        role: SubAgentRole,
        task: str,
        *,
        tool_budget: Optional[int] = None,
        allowed_tools: Optional[List[str]] = None,
        context_limit: Optional[int] = None,
        can_spawn_subagents: bool = False,
        timeout_seconds: int = 300,
    ) -> AsyncIterator["StreamChunk"]:
        """Spawn a sub-agent with streaming output.

        Like spawn() but yields StreamChunk as agent generates content.
        Useful for real-time progress display during long-running tasks.

        Args:
            role: Sub-agent role specialization
            task: Task description for the sub-agent
            tool_budget: Maximum tool calls (default: role-specific)
            allowed_tools: List of allowed tools (default: role-specific)
            context_limit: Maximum context size (default: role-specific)
            can_spawn_subagents: Whether sub-agent can spawn children
            timeout_seconds: Maximum execution time

        Yields:
            StreamChunk with incremental content and tool calls

        Example:
            async for chunk in orchestrator.stream_spawn(
                SubAgentRole.RESEARCHER,
                "Find all API endpoints",
            ):
                print(chunk.content, end="", flush=True)
                if chunk.is_final:
                    print(f"\\nDone: {chunk.metadata}")
        """
        from victor.providers.base import StreamChunk

        # Apply role-specific defaults
        effective_budget = tool_budget or ROLE_DEFAULT_BUDGETS.get(role, 15)
        effective_tools = allowed_tools or ROLE_DEFAULT_TOOLS.get(role, ["read", "ls"])
        effective_context = context_limit or ROLE_DEFAULT_CONTEXT.get(role, 50000)

        # Create configuration
        config = SubAgentConfig(
            role=role,
            task=task,
            allowed_tools=effective_tools,
            tool_budget=effective_budget,
            context_limit=effective_context,
            can_spawn_subagents=can_spawn_subagents,
            timeout_seconds=timeout_seconds,
        )

        # Create sub-agent
        subagent = SubAgent(config, self.parent)
        self.active_subagents.add(subagent)

        start_time = time.time()

        try:
            # Stream with manual timeout checking per chunk
            async for chunk in subagent.stream_execute():
                # Check timeout before yielding each chunk
                elapsed = time.time() - start_time
                if elapsed > timeout_seconds:
                    logger.warning(
                        f"{role.value} sub-agent stream timed out after {timeout_seconds}s"
                    )
                    yield StreamChunk(
                        content="",
                        is_final=True,
                        metadata={
                            "error": f"Timeout after {timeout_seconds}s",
                            "timeout": True,
                            "role": role.value,
                            "task": task[:200],
                            "duration_seconds": float(timeout_seconds),
                            "success": False,
                        },
                    )
                    return

                yield chunk

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            logger.error(
                f"{role.value} sub-agent stream spawn failed: {error_msg}",
                exc_info=True,
            )
            yield StreamChunk(
                content="",
                is_final=True,
                metadata={
                    "error": error_msg,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "role": role.value,
                    "duration_seconds": time.time() - start_time,
                    "success": False,
                },
            )

        finally:
            self.active_subagents.discard(subagent)

    def get_active_count(self) -> int:
        """Get number of currently active sub-agents.

        Returns:
            Count of active sub-agents
        """
        return len(self.active_subagents)


__all__ = [
    "SubAgentOrchestrator",
    "SubAgentTask",
    "FanOutResult",
    "ROLE_DEFAULT_TOOLS",
    "ROLE_DEFAULT_BUDGETS",
    "ROLE_DEFAULT_CONTEXT",
]
