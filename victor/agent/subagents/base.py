# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
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

"""Sub-agent base classes for hierarchical task delegation.

This module provides the core infrastructure for sub-agents - specialized,
isolated instances of AgentOrchestrator with constrained scopes. Sub-agents
enable hierarchical task delegation and parallel execution.

Design Principles:
- Context Isolation: Each sub-agent has independent message history
- Resource Constraints: Limited tool access and budget
- Role Specialization: Specific roles (researcher, planner, executor, etc.)
- Backward Compatible: Existing code works without sub-agents

Example Usage:
    from victor.agent.subagents.base import SubAgent, SubAgentRole, SubAgentConfig

    # Create sub-agent configuration
    config = SubAgentConfig(
        role=SubAgentRole.RESEARCHER,
        task="Research authentication patterns in codebase",
        allowed_tools=["read", "search", "code_search"],
        tool_budget=15,
        context_limit=50000,
    )

    # Create and execute sub-agent
    subagent = SubAgent(config, parent_orchestrator)
    result = await subagent.execute()

    print(f"Success: {result.success}")
    print(f"Summary: {result.summary}")
    print(f"Tool calls used: {result.tool_calls_used}")
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Dict,
    List,
    Optional,
    Union,
)

from victor.agent.subagents.protocols import SubAgentContext, SubAgentContextAdapter
from victor.agent.runtime.naming import build_display_name, generate_agent_id
from victor.core.retry import compute_backoff_delay

if TYPE_CHECKING:
    from victor.agent.runtime.context import AgentRuntimeContext
    from victor.agent.presentation import PresentationProtocol
    from victor.agent.orchestrator import AgentOrchestrator
    from victor.agent.services.context_lifecycle_service import ContextLifecycleService
    from victor.core.container import ServiceContainer
    from victor.framework.hitl import ApprovalHandler, ApprovalRequest
    from victor.providers.base import StreamChunk
    from victor.teams.types import AgentMessage

logger = logging.getLogger(__name__)


# Re-export from canonical location for backward compatibility
from victor.core.shared_types import SubAgentRole  # noqa: F401


from victor.protocols.team import IAgent


@dataclass
class SubAgentConfig:
    """Configuration for a sub-agent instance.

    Defines the constraints and capabilities for a sub-agent. All fields
    except role and task have sensible defaults based on the role.

    Attributes:
        role: Sub-agent role specialization
        task: Task description for the sub-agent to execute
        allowed_tools: List of tool names the sub-agent can use
        tool_budget: Maximum number of tool calls allowed
        context_limit: Maximum context size in characters
        can_spawn_subagents: Whether this sub-agent can spawn child sub-agents
        working_directory: Optional working directory override
        timeout_seconds: Maximum execution time in seconds
        system_prompt_override: Optional custom system prompt (overrides role default)
        disable_embeddings: Disable codebase embeddings for this sub-agent (workflow service mode)
        result_summary_max_chars: Maximum chars retained in ``SubAgentResult.summary``.
            ``None`` preserves the full response for parent handoff.
    """

    role: SubAgentRole
    task: str
    allowed_tools: List[str]
    tool_budget: int
    context_limit: int
    can_spawn_subagents: bool = False
    working_directory: Optional[str] = None
    timeout_seconds: int = 300
    system_prompt_override: Optional[str] = None
    disable_embeddings: bool = False
    member_id: Optional[str] = None
    agent_id: Optional[str] = None
    display_name: Optional[str] = None
    team_id: Optional[str] = None
    plan_id: Optional[str] = None
    plan_step_id: Optional[str] = None
    parent_session_id: Optional[str] = None
    child_session_id: Optional[str] = None
    result_summary_max_chars: Optional[int] = None
    # Heterogeneous execution: a pre-resolved provider instance, model, and
    # temperature to use instead of inheriting the parent's. None = inherit.
    provider_override: Optional[Any] = None
    model_override: Optional[str] = None
    temperature_override: Optional[float] = None
    reasoning_effort_override: Optional[str] = None

    def to_runtime_context(self) -> "AgentRuntimeContext":
        """Build the common per-agent runtime context from this config."""
        from victor.agent.runtime.context import AgentRuntimeContext

        agent_id = self.agent_id or generate_agent_id(self.role)
        return AgentRuntimeContext(
            agent_id=agent_id,
            display_name=self.display_name or build_display_name(self.role, task=self.task),
            role=self.role.value,
            session_id=self.child_session_id or agent_id,
            parent_session_id=self.parent_session_id,
            team_id=self.team_id,
            plan_id=self.plan_id,
            plan_step_id=self.plan_step_id,
            member_id=self.member_id,
        )


@dataclass
class SubAgentResult:
    """Result from sub-agent execution.

    Contains both the outcome (success/failure) and detailed metrics about
    the execution. The summary provides a concise overview while details
    contain the full response and tool calls.

    Attributes:
        success: Whether the sub-agent completed successfully
        summary: Brief summary of the result (first 500 chars)
        details: Full details including response and tool calls
        tool_calls_used: Number of tool calls made
        context_size: Final context size in characters
        duration_seconds: Execution time in seconds
        error: Error message if execution failed
    """

    success: bool
    summary: str
    details: Dict[str, Any]
    tool_calls_used: int
    context_size: int
    duration_seconds: float
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "success": self.success,
            "summary": self.summary,
            "details": self.details,
            "tool_calls_used": self.tool_calls_used,
            "context_size": self.context_size,
            "duration_seconds": self.duration_seconds,
            "error": self.error,
        }


def _bounded_result_summary(content: Optional[str], max_chars: Optional[int]) -> str:
    """Return sub-agent summary content, preserving full handoff by default."""
    if not content:
        return ""
    if max_chars is None or max_chars <= 0:
        return content
    return content[:max_chars]


class SubAgent(IAgent):  # type: ignore[misc]
    """Represents a spawned sub-agent instance.

    A sub-agent is a wrapper around AgentOrchestrator with:
    - Constrained tool access (only allowed_tools)
    - Independent context (isolated message history)
    - Limited budget (max tool calls)
    - Role-specific system prompt

    Sub-agents execute a single task and return a structured result.
    They cannot interact directly with the user - all results are
    returned to the parent orchestrator.

    Attributes:
        config: Configuration defining constraints and capabilities
        parent: Parent orchestrator that spawned this sub-agent
        orchestrator: Constrained orchestrator instance for execution

    Example:
        config = SubAgentConfig(
            role=SubAgentRole.PLANNER,
            task="Create implementation plan for JWT auth",
            allowed_tools=["read", "ls", "search", "plan_files"],
            tool_budget=10,
            context_limit=30000,
        )

        subagent = SubAgent(config, parent_orchestrator)
        result = await subagent.execute()
    """

    def __init__(
        self,
        config: SubAgentConfig,
        parent: Union["AgentOrchestrator", SubAgentContext],
        presentation: Optional["PresentationProtocol"] = None,
        context_lifecycle: Optional["ContextLifecycleService"] = None,
    ):
        """Initialize sub-agent with configuration and parent context.

        Args:
            config: Sub-agent configuration
            parent: Parent context providing settings, provider, model, and tools.
                Can be either a full AgentOrchestrator (for backward compatibility)
                or any object implementing the SubAgentContext protocol.
            presentation: Optional presentation adapter for icons (creates default if None)

        Note:
            If a full AgentOrchestrator is passed, it will be automatically
            adapted to SubAgentContext for ISP compliance. This maintains
            backward compatibility with existing code while enabling cleaner
            testing through protocol-based dependency injection.
        """
        self.config = config
        self._id = config.agent_id or generate_agent_id(config.role)
        if self.config.agent_id is None:
            self.config.agent_id = self._id
        if self.config.display_name is None:
            self.config.display_name = build_display_name(self.config.role, task=self.config.task)

        # Lazy init for backward compatibility
        if presentation is None:
            from victor.agent.presentation import create_presentation_adapter

            self._presentation = create_presentation_adapter()
        else:
            self._presentation = presentation

        # Auto-adapt full orchestrator to SubAgentContext for ISP compliance
        # Check if it's already a SubAgentContext (including adapters)
        if isinstance(parent, SubAgentContext):
            self._context: SubAgentContext = parent
        else:
            # Wrap the orchestrator with the adapter
            self._context = SubAgentContextAdapter(parent)

        # Keep reference to parent for backward compatibility
        # (some code may access self.parent directly)
        self.parent = parent
        self.orchestrator: Optional["AgentOrchestrator"] = None
        self._context_lifecycle = context_lifecycle
        self._owned_context_lifecycle: Optional["ContextLifecycleService"] = None

        logger.info(
            f"Created {config.role.value} sub-agent: {config.task[:50]}... "
            f"(budget={config.tool_budget}, tools={len(config.allowed_tools)})"
        )

    @property
    def id(self) -> str:
        """Unique identifier for this agent."""
        return self._id

    @property
    def role(self):
        """Role of this agent (SubAgentRole)."""
        return self.config.role

    @property
    def persona(self):
        """Persona of this agent (None for SubAgent)."""
        return None

    async def execute_task(self, task: str, context: Dict[str, Any]) -> str:
        """Execute a task using this sub-agent.

        Note: The SubAgent is configured with a specific task at creation.
        This method executes that configured task, ignoring the passed task
        parameter. The context parameter is also ignored as SubAgent has
        its own isolated context.

        Returns:
            String summary of execution outcome.
        """
        # Execute the configured task (ignore passed task parameter)
        result = await self.execute()
        return result.summary if result.summary else ""

    async def receive_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Receive a message (SubAgents don't support direct messaging).

        SubAgents are isolated execution units that don't participate in
        direct agent-to-agent communication. This method exists for protocol
        compatibility.

        Returns:
            None, as SubAgents don't respond to messages
        """
        return None

    def _build_member_approval_handler(self) -> Optional["ApprovalHandler"]:
        """Wrap the session's policy ASK approval handler to tag this member (ADR-023).

        Team members share the session's DI container (the global container the client
        registers the terminal approval handler into), so a member's ASK-gated tool already
        resolves that handler. This wrapper stamps the member's identity onto each
        :class:`ApprovalRequest` (``context["member_id"]`` / ``["member_role"]``) so the
        shared terminal modal can show *which* member is asking (surfaces read the context;
        the framework policy site stays member-agnostic).

        Returns ``None`` when no session handler is registered — the member then keeps the
        default ``ask_fallback`` behavior, byte-identical to before.
        """
        try:
            from victor.core import get_container
            from victor.framework.policies import PolicyApprovalHandler

            holder = get_container().get_optional(PolicyApprovalHandler)
        except Exception:  # pragma: no cover - defensive; approval is best-effort
            holder = None
        inner = getattr(holder, "handler", None) if holder is not None else None
        if inner is None:
            return None

        member_id = self.id
        member_role = self.config.role.value

        async def _tagged_handler(request: "ApprovalRequest") -> Any:
            request.context.setdefault("member_id", member_id)
            request.context.setdefault("member_role", member_role)
            return await inner(request)

        return _tagged_handler

    def _create_constrained_orchestrator(self) -> "AgentOrchestrator":
        """Create orchestrator with role-specific constraints.

        Creates a new orchestrator instance with:
        - Limited tool access (only allowed_tools)
        - Constrained budget and context
        - Role-specific system prompt
        - Shared provider and settings from parent context
        - disable_embeddings flag for workflow service mode

        Returns:
            Constrained orchestrator instance
        """
        from copy import deepcopy
        from victor.agent.orchestrator import AgentOrchestrator

        # Copy settings from parent context but apply constraints
        settings = deepcopy(self._context.settings)
        settings.tool_budget = self.config.tool_budget
        settings.max_context_chars = self.config.context_limit

        # Heterogeneous execution: use a per-member provider override when one
        # was resolved (cross-vendor teams); otherwise inherit the parent's
        # provider/model so behavior is unchanged for members without overrides.
        provider = self.config.provider_override or self._context.provider
        model = self.config.model_override or self._context.model
        if self.config.provider_override is not None:
            provider_name = getattr(
                self.config.provider_override,
                "provider_name",
                self._context.provider_name,
            )
        else:
            provider_name = self._context.provider_name
        temperature = (
            self.config.temperature_override
            if self.config.temperature_override is not None
            else self._context.temperature
        )
        # Per-member reasoning_effort override; inherit the parent's when unset.
        reasoning_effort = (
            self.config.reasoning_effort_override
            if self.config.reasoning_effort_override is not None
            else getattr(self._context, "reasoning_effort", None)
        )

        # Create new orchestrator instance.
        # Use the actual provider object (not just the name) for proper initialization
        orchestrator = AgentOrchestrator(
            settings=settings,
            provider=provider,
            model=model,
            temperature=temperature,
            provider_name=provider_name,
            system_prompt_override=self._get_role_prompt(),
            reasoning_effort=reasoning_effort,
            # ADR-023 pillar 2: route this member's policy ASK verdicts to the session's
            # terminal approval handler, tagged with member_id so the shared modal shows
            # which member is asking. None when no handler is registered (unchanged).
            approval_handler=self._build_member_approval_handler(),
            # Note: We'll share the parent's DI container for now
            # In production, we might want isolated scoped containers
        )

        # Inherit parent's vertical context via flyweight pattern
        parent_vc = self._context.vertical_context
        if parent_vc is not None and hasattr(parent_vc, "create_child_context"):
            child_vc = parent_vc.create_child_context(
                enabled_tools=set(self.config.allowed_tools),
            )
            orchestrator._vertical_context = child_vc

        # Set disable_embeddings flag for workflow service mode
        if self.config.disable_embeddings:
            gear_icon = self._presentation.icon("gear", with_color=False)
            logger.debug(
                f"   {gear_icon}  Setting disable_embeddings=True for {self.config.role.value} sub-agent"
            )
            if hasattr(orchestrator, "_session_state_manager"):
                orchestrator._session_state_manager.execution_state.disable_embeddings = True

        # Register only allowed tools
        self._configure_allowed_tools(orchestrator)

        logger.debug(
            f"Created constrained orchestrator for {self.config.role.value}: "
            f"{len(self.config.allowed_tools)} tools, budget={self.config.tool_budget}"
        )

        return orchestrator

    def _configure_allowed_tools(self, orchestrator: "AgentOrchestrator") -> None:
        """Configure orchestrator with only allowed tools.

        Args:
            orchestrator: Orchestrator to configure
        """
        # Clear existing tool registrations
        orchestrator.tool_registry.clear()

        # Register only allowed tools from parent context
        missing_tools = []
        for tool_name in self.config.allowed_tools:
            tool = self._context.tool_registry.get(tool_name)
            if tool:
                orchestrator.tool_registry.register(tool)
            else:
                missing_tools.append(tool_name)

        # Log missing tools at debug level (tools are optional/role-based)
        if missing_tools:
            logger.debug(
                f"Tools not found in parent registry for {self.config.role.value} sub-agent: "
                f"{', '.join(missing_tools[:5])}"
                f"{'...' if len(missing_tools) > 5 else ''}"
            )

    def _get_role_prompt(self) -> str:
        """Get system prompt for this role.

        If a custom system_prompt_override is provided in config, use that.
        Otherwise, use the role-specific default prompt.

        Returns:
            System prompt string
        """
        if self.config.system_prompt_override:
            return self.config.system_prompt_override

        # Import here to avoid circular dependency
        from victor.agent.subagents.prompts import get_role_prompt

        return get_role_prompt(self.config.role)

    async def _execute_with_retry(self) -> Any:
        """Execute orchestrator.chat() with exponential backoff retry.

        Implements retry logic with exponential backoff for handling transient errors
        like rate limits, timeouts, and connection issues.

        Returns:
            Response from orchestrator.chat()

        Raises:
            Exception: If all retries are exhausted
        """
        from victor.core.errors import (
            ProviderConnectionError,
            ProviderError,
            ProviderRateLimitError,
            ProviderTimeoutError,
        )

        max_attempts = 3
        base_delay = 1.0  # Start with 1 second
        max_delay = 60.0  # Cap at 60 seconds

        last_exception = None

        refresh_icon = self._presentation.icon("refresh", with_color=False)
        success_icon = self._presentation.icon("success", with_color=False)
        error_icon = self._presentation.icon("error", with_color=False)

        logger.debug(
            f"   {refresh_icon} Starting execution with retry logic (max {max_attempts} attempts)"
        )

        for attempt in range(1, max_attempts + 1):
            try:
                if attempt > 1:
                    logger.debug(f"   {refresh_icon} Retry attempt {attempt}/{max_attempts}...")

                # Try to execute the chat
                response = await self.orchestrator.chat(self.config.task)

                if attempt > 1:
                    logger.debug(
                        f"   {success_icon} Retry attempt {attempt}/{max_attempts} succeeded!"
                    )

                return response

            except (
                ProviderRateLimitError,
                ProviderTimeoutError,
                ProviderConnectionError,
                ProviderError,  # Generic provider errors (e.g., server disconnects)
                ConnectionError,
                TimeoutError,
                OSError,  # Network-level errors
            ) as e:
                last_exception = e

                if attempt >= max_attempts:
                    # Final attempt failed, will raise after loop
                    logger.warning(
                        f"Sub-agent {self.config.role.value}: All {max_attempts} attempts failed. Last error: {type(e).__name__}: {e}"
                    )
                    break

                retry_after = getattr(e, "retry_after", None)
                if isinstance(retry_after, (int, float)) and retry_after > 0:
                    delay = min(float(retry_after), max_delay)
                else:
                    # Calculate exponential backoff delay: 2^(attempt-1) * base_delay.
                    delay = compute_backoff_delay(attempt - 1, base_delay, max_delay=max_delay)

                logger.info(
                    f"Sub-agent {self.config.role.value}: Attempt {attempt}/{max_attempts} failed with "
                    f"{type(e).__name__}. Retrying in {delay:.1f}s..."
                )

                await asyncio.sleep(delay)

            except Exception as e:
                # Non-retriable error, raise immediately
                logger.error(
                    f"Sub-agent {self.config.role.value}: Non-retriable error: {type(e).__name__}: {e}"
                )
                raise

        # All retries exhausted
        logger.error(f"   {error_icon} All retry attempts exhausted for {self.config.role.value}")
        raise last_exception

    async def execute(self) -> SubAgentResult:
        """Execute the sub-agent task.

        Runs the task in the constrained orchestrator and returns a
        structured result. Handles exceptions gracefully and always
        returns a SubAgentResult (never raises).

        Returns:
            SubAgentResult with execution outcome and metrics
        """
        start_time = time.time()

        try:
            # Create constrained orchestrator lazily
            if self.orchestrator is None:
                self.orchestrator = self._create_constrained_orchestrator()

            logger.info(f"Executing {self.config.role.value} sub-agent: {self.config.task[:50]}...")

            # Run the task with retry on rate limits
            response = await self._execute_with_retry()
            response_metadata = getattr(response, "metadata", None) or {}
            execution_success = response_metadata.get("agentic_loop_success") is not False
            execution_error = (
                response_metadata.get("agentic_loop_error") if not execution_success else None
            )

            # Extract metrics
            tool_calls_used = getattr(self.orchestrator, "tool_calls_used", 0)
            context_size = len(str(self.orchestrator.get_messages()))

            # Create structured result. A failed agentic loop is a real sub-agent
            # failure even when the provider returned a final response object.
            runtime_context = self.config.to_runtime_context()
            lifecycle_report = await self._run_context_lifecycle(runtime_context)
            status = "success" if execution_success else "failed"
            result = SubAgentResult(
                success=execution_success,
                summary=_bounded_result_summary(
                    response.content,
                    self.config.result_summary_max_chars,
                ),
                details={
                    "full_response": response.content,
                    "tool_calls": getattr(response, "tool_calls", []) or [],
                    "tool_evidence": self._build_tool_evidence_handoff(),
                    "role": self.config.role.value,
                    "agentic_loop_success": execution_success,
                    "context_lifecycle": lifecycle_report,
                    "parent_handoff": self._build_parent_handoff(
                        runtime_context,
                        summary=response.content or "",
                        status=status,
                        metadata={
                            "tool_calls_used": tool_calls_used,
                            "agentic_loop_success": execution_success,
                            **({"agentic_loop_error": execution_error} if execution_error else {}),
                        },
                    ),
                    **self._identity_metadata(),
                },
                tool_calls_used=tool_calls_used,
                context_size=context_size,
                duration_seconds=time.time() - start_time,
                error=execution_error,
            )
            if execution_error:
                result.details["agentic_loop_error"] = execution_error

            logger.info(
                f"{self.config.role.value} sub-agent completed: "
                f"{tool_calls_used}/{self.config.tool_budget} tool calls, "
                f"{result.duration_seconds:.1f}s"
            )

            return result

        except Exception as e:
            # Create error result
            error_msg = f"{type(e).__name__}: {str(e)}"
            logger.error(
                f"{self.config.role.value} sub-agent failed: {error_msg}",
                exc_info=True,
            )

            tool_calls_used = 0
            context_size = 0
            if self.orchestrator:
                tool_calls_used = getattr(self.orchestrator, "tool_calls_used", 0)
                try:
                    context_size = len(str(self.orchestrator.get_messages()))
                except Exception as e:
                    logger.debug("Failed to compute sub-agent context size: %s", e)

            runtime_context = self.config.to_runtime_context()
            return SubAgentResult(
                success=False,
                summary=f"Sub-agent failed: {error_msg[:450]}",
                details={
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "role": self.config.role.value,
                    "parent_handoff": self._build_parent_handoff(
                        runtime_context,
                        summary=error_msg,
                        status="failed",
                        metadata={"tool_calls_used": tool_calls_used},
                    ),
                    **self._identity_metadata(),
                },
                tool_calls_used=tool_calls_used,
                context_size=context_size,
                duration_seconds=time.time() - start_time,
                error=error_msg,
            )

    def _identity_metadata(self) -> Dict[str, Any]:
        """Return stable team/session identity metadata for this runtime."""
        return {
            "member_id": self.config.member_id,
            "agent_id": self.id,
            "display_name": self.config.display_name,
            "team_id": self.config.team_id,
            "plan_id": self.config.plan_id,
            "plan_step_id": self.config.plan_step_id,
            "parent_session_id": self.config.parent_session_id,
            "child_session_id": self.config.child_session_id,
        }

    async def _run_context_lifecycle(
        self, runtime_context: "AgentRuntimeContext"
    ) -> Dict[str, Any]:
        """Run common per-agent context lifecycle hooks for this sub-agent."""
        lifecycle = self._context_lifecycle or self._default_context_lifecycle()
        messages = self._get_orchestrator_messages()
        return await lifecycle.after_agent_turn(
            runtime_context,
            messages=messages,
            min_messages=6,
        )

    def _build_parent_handoff(
        self,
        runtime_context: "AgentRuntimeContext",
        *,
        summary: str,
        status: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        lifecycle = self._context_lifecycle or self._default_context_lifecycle()
        return lifecycle.build_parent_handoff(
            runtime_context,
            summary=summary,
            status=status,
            metadata=metadata,
        )

    def _default_context_lifecycle(self) -> "ContextLifecycleService":
        """Create a local lifecycle service when no parent-scoped service is injected."""
        from victor.agent.services.context_lifecycle_service import (
            ContextLifecycleService,
        )

        if self._owned_context_lifecycle is not None:
            return self._owned_context_lifecycle
        max_tokens = max(1, int(self.config.context_limit or 50000) // 4)
        self._owned_context_lifecycle = ContextLifecycleService.with_defaults(max_tokens=max_tokens)
        return self._owned_context_lifecycle

    def _get_orchestrator_messages(self) -> List[Any]:
        if self.orchestrator is None:
            return []
        get_messages = getattr(self.orchestrator, "get_messages", None)
        if not callable(get_messages):
            return []
        try:
            messages = get_messages()
        except Exception as exc:
            logger.debug("Failed to collect sub-agent messages for lifecycle: %s", exc)
            return []
        return list(messages or [])

    def _build_tool_evidence_handoff(self) -> Dict[str, Any]:
        """Return a bounded digest of tool outputs for parent plan-state extraction.

        Some providers finish a long tool-backed turn with a very short final message.
        Planning mode still needs enough evidence to populate named ``produces`` keys,
        so sub-agents hand back a compact, tool-output-derived digest alongside the
        final response. The digest is metadata only; the parent execution adapter
        decides when to use it.
        """
        messages = self._get_orchestrator_messages()
        entries: List[Dict[str, str]] = []
        tool_names: List[str] = []
        total_chars = 0

        for message in messages:
            role = getattr(message, "role", None)
            if hasattr(role, "value"):
                role = role.value
            if isinstance(message, dict):
                role = message.get("role", role)
            if str(role or "").lower() != "tool":
                continue

            name = ""
            if isinstance(message, dict):
                name = str(
                    message.get("name")
                    or (message.get("metadata") or {}).get("tool_name")
                    or (message.get("metadata") or {}).get("name")
                    or "tool"
                )
                content = str(message.get("content") or "")
            else:
                metadata = getattr(message, "metadata", None) or {}
                name = str(
                    getattr(message, "name", None)
                    or metadata.get("tool_name")
                    or metadata.get("name")
                    or "tool"
                )
                content = str(getattr(message, "content", "") or "")

            content = content.strip()
            if not content:
                continue
            if name not in tool_names:
                tool_names.append(name)

            snippet = re.sub(r"\s+", " ", content)
            snippet = snippet[:700]
            total_chars += len(content)
            entries.append({"tool": name, "snippet": snippet})
            if len(entries) >= 24:
                break

        lines = [f"{entry['tool']}: {entry['snippet']}" for entry in entries]
        return {
            "count": len(entries),
            "tool_names": tool_names,
            "total_chars": total_chars,
            "summary": "\n".join(lines)[:12000],
        }

    async def stream_execute(self) -> AsyncIterator["StreamChunk"]:
        """Execute sub-agent task with streaming output.

        Like execute() but yields chunks as content is generated.
        Uses orchestrator.stream_chat() instead of .chat().

        Yields:
            StreamChunk with incremental content and tool calls

        Example:
            subagent = SubAgent(config, parent_orchestrator)
            async for chunk in subagent.stream_execute():
                print(chunk.content, end="", flush=True)
                if chunk.is_final:
                    print(f"\\nCompleted: {chunk.metadata}")
        """
        from victor.providers.base import StreamChunk

        start_time = time.time()

        try:
            # Create constrained orchestrator lazily
            if self.orchestrator is None:
                self.orchestrator = self._create_constrained_orchestrator()

            logger.info(
                f"Stream executing {self.config.role.value} sub-agent: "
                f"{self.config.task[:50]}..."
            )

            # Stream the task using orchestrator.stream_chat()
            async for chunk in self.orchestrator.stream_chat(self.config.task):
                yield chunk

                # If this was the final chunk from the orchestrator, we'll add metadata
                if chunk.is_final:
                    # Extract metrics
                    tool_calls_used = getattr(self.orchestrator, "tool_calls_used", 0)
                    context_size = len(str(self.orchestrator.get_messages()))
                    duration = time.time() - start_time

                    # Enhance the final chunk with execution metadata
                    enhanced_metadata = chunk.metadata.copy() if chunk.metadata else {}
                    enhanced_metadata.update(
                        {
                            "tool_calls_used": tool_calls_used,
                            "context_size": context_size,
                            "duration_seconds": duration,
                            "role": self.config.role.value,
                            "success": True,
                        }
                    )

                    # Yield a final chunk with metadata (if not already included)
                    yield StreamChunk(
                        content="",
                        is_final=True,
                        metadata=enhanced_metadata,
                    )

                    logger.info(
                        f"{self.config.role.value} sub-agent stream completed: "
                        f"{tool_calls_used}/{self.config.tool_budget} tool calls, "
                        f"{duration:.1f}s"
                    )
                    return

        except Exception as e:
            # Create error chunk
            error_msg = f"{type(e).__name__}: {str(e)}"
            logger.error(
                f"{self.config.role.value} sub-agent stream failed: {error_msg}",
                exc_info=True,
            )

            tool_calls_used = 0
            context_size = 0
            if self.orchestrator:
                tool_calls_used = getattr(self.orchestrator, "tool_calls_used", 0)
                try:
                    context_size = len(str(self.orchestrator.get_messages()))
                except Exception as e:
                    logger.debug("Failed to compute sub-agent stream context size: %s", e)

            # Yield error chunk with is_final=True
            yield StreamChunk(
                content="",
                is_final=True,
                metadata={
                    "error": error_msg,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "tool_calls_used": tool_calls_used,
                    "context_size": context_size,
                    "duration_seconds": time.time() - start_time,
                    "role": self.config.role.value,
                    "success": False,
                },
            )


__all__ = [
    "SubAgentRole",
    "SubAgentConfig",
    "SubAgentResult",
    "SubAgent",
]
