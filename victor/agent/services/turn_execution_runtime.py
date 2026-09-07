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

"""Canonical service-owned turn execution runtime implementation.

This module provides `TurnExecutor` as the active implementation for the
agentic loop runtime. The historical
`victor.agent.coordinators.turn_executor` module is now only a
compatibility import path that re-exports these definitions.

The TurnExecutor handles:
- Multi-turn agentic loop (model → tools → model → tools → ...)
- Iteration limit enforcement
- Tool call execution coordination
- Response completion handling
- Error recovery and retry logic

Architecture:
------------
The TurnExecutor depends on protocol-based abstractions rather than
concrete classes, enabling the Dependency Inversion Principle (DIP):

- ChatContextProtocol: For message/conversation access
- ToolContextProtocol: For tool execution
- ProviderContextProtocol: For LLM calls
- ExecutionProvider: For executing model turns

Phase 1: Extract TurnExecutor
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import logging
import re
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

from dataclasses import dataclass, field, replace

from victor.agent.conversation.types import MESSAGE_SOURCE_METADATA_KEY, MessageSource
from victor.agent.runtime.context import AgentRuntimeContext
from victor.agent.response_completer import ToolFailureContext
from victor.agent.services.context_service import compact_context_if_recommended
from victor.providers.base import CompletionResponse

if TYPE_CHECKING:
    from victor.agent.services.protocols.chat_runtime import (
        ChatContextProtocol,
        OrchestratorRuntimeProtocol,
        ParallelExplorationProtocol,
        ProviderContextProtocol,
        StatePassedExplorationProtocol,
        ToolContextProtocol,
    )


@dataclass
class TurnResult:
    """Result of a single execution turn (one LLM call + tool execution).

    This is the primitive unit returned by execute_turn().
    AgenticLoop uses this to make per-turn evaluation decisions.

    Attributes:
        response: Raw model response
        tool_results: Results from tool execution (empty if no tools called)
        follow_up_suggestions: Aggregated recovery suggestions from blocked/failed tools
        has_tool_calls: Whether the model requested tools
        tool_calls_count: Number of tool calls in this turn
        all_tools_blocked: Whether all tool calls were blocked by dedup
        is_qa_response: Whether this was a Q&A shortcut
        content: Response text content (convenience)
    """

    response: CompletionResponse
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    follow_up_suggestions: List[Dict[str, Any]] = field(default_factory=list)
    has_tool_calls: bool = False
    tool_calls_count: int = 0
    all_tools_blocked: bool = False
    is_qa_response: bool = False

    @property
    def content(self) -> str:
        """Response text content."""
        return self.response.content or ""

    @property
    def has_content(self) -> bool:
        """Whether the response has text content."""
        return bool(self.response.content)

    @property
    def successful_tool_count(self) -> int:
        """Number of tools that succeeded."""
        return sum(1 for r in self.tool_results if r.get("success"))

    @property
    def failed_tool_count(self) -> int:
        """Number of tools that failed."""
        return sum(1 for r in self.tool_results if not r.get("success"))

    @property
    def tool_signatures(self) -> Set[str]:
        """Compute stable signatures for all tool calls in this turn."""
        signatures = set()
        if not self.response.tool_calls:
            return signatures

        for tc in self.response.tool_calls:
            name = tc.get("name", "unknown")
            args = tc.get("arguments", {})
            # Normalize args for stable hashing
            if isinstance(args, dict):
                # Sort keys and convert values to string
                sorted_args = sorted(args.items())
                args_str = str([(k, str(v)) for k, v in sorted_args])
            else:
                args_str = str(args)
            signatures.add(f"{name}:{args_str}")
        return signatures


logger = logging.getLogger(__name__)
_MISSING = object()

__all__ = ["TurnExecutor", "TurnResult"]

# Module-level guard to prevent recursive subagent exploration
_EXPLORATION_IN_PROGRESS = False


class TurnExecutor:
    """Service-owned runtime helper for the agentic execution loop.

    This helper contains the core agentic loop logic used by the canonical
    service-first runtime path while remaining reusable from legacy shims.

    The agentic loop implements the following flow:
    1. Initialize tracking (tool budget, iteration budget)
    2. Get model response
    3. Execute any tool calls
    4. Continue until model provides final response (no tool calls)
    5. Ensure non-empty response on tool failures

    Args:
        chat_context: Protocol providing conversation/message access
        tool_context: Protocol providing tool selection/execution
        provider_context: Protocol providing LLM provider access
        execution_provider: Protocol for executing model turns
    """

    def __init__(
        self,
        chat_context: "ChatContextProtocol",
        tool_context: "ToolContextProtocol",
        provider_context: "ProviderContextProtocol",
        execution_provider: Any,  # ExecutionProvider protocol
        exploration_coordinator: Optional[Any] = None,
        message_policy_gate: Optional[Any] = None,
    ) -> None:
        """Initialize the TurnExecutor.

        Args:
            chat_context: Chat context protocol implementation
            tool_context: Tool context protocol implementation
            provider_context: Provider context protocol implementation
            execution_provider: Execution provider protocol implementation
            exploration_coordinator: Optional shared exploration runtime or
                state-passed coordinator. When omitted, TurnExecutor prefers
                the shared exploration_state_passed surface from the
                orchestrator facade and falls back to the direct
                ExplorationCoordinator helper only when no orchestrator-backed
                state-passed surface is available.
            message_policy_gate: Optional governance
                :class:`~victor.framework.policies.gate.MessagePolicyGate` that
                gates the REQUEST (user message pre-LLM) and RESPONSE (assistant
                output) phases. None disables message-phase governance (default).
        """
        self._chat_context = chat_context
        self._tool_context = tool_context
        self._provider_context = provider_context
        self._execution_provider = execution_provider
        self._exploration_coordinator = exploration_coordinator
        self._message_policy_gate = message_policy_gate
        self._exploration_done = False  # Instance-level: fires once per conversation
        self._last_tool_follow_up_guidance_signature: Optional[tuple[str, ...]] = None

    def _resolve_orchestrator(self) -> "Optional[OrchestratorRuntimeProtocol]":
        """Return the explicit orchestrator owner for this runtime, if any."""
        orchestrator = getattr(self, "_orchestrator", None)
        if orchestrator is not None:
            return orchestrator
        return getattr(self._chat_context, "_orchestrator", None)

    @staticmethod
    def _is_state_passed_explorer(explorer: Any) -> bool:
        """Return whether an exploration dependency exposes the state-passed surface."""
        return callable(getattr(explorer, "explore", None)) and not callable(
            getattr(explorer, "explore_parallel", None)
        )

    def _resolve_parallel_explorer(
        self,
    ) -> "tuple[ParallelExplorationProtocol | StatePassedExplorationProtocol, bool]":
        """Resolve the exploration implementation for the current runtime."""
        explorer = self._exploration_coordinator
        if explorer is not None:
            return explorer, self._is_state_passed_explorer(explorer)

        orchestrator = self._resolve_orchestrator()
        if orchestrator is not None:
            facade = getattr(orchestrator, "_orchestration_facade", None)
            if facade is None:
                try:
                    facade = getattr(orchestrator, "orchestration_facade", None)
                except Exception:
                    facade = None

            state_passed = getattr(facade, "exploration_state_passed", None) if facade else None
            if state_passed is not None:
                self._exploration_coordinator = state_passed
                return state_passed, True

        from victor.agent.services.exploration_runtime import ExplorationCoordinator

        explorer = ExplorationCoordinator()
        self._exploration_coordinator = explorer
        return explorer, False

    @staticmethod
    def _normalize_exploration_payload(findings: Any) -> Dict[str, Any]:
        """Normalize exploration findings to a shared payload shape."""
        return {
            "summary": getattr(findings, "summary", "") or "",
            "file_paths": list(getattr(findings, "file_paths", []) or []),
            "duration_seconds": float(getattr(findings, "duration_seconds", 0.0) or 0.0),
            "tool_calls": int(getattr(findings, "tool_calls", 0) or 0),
        }

    async def _run_state_passed_parallel_exploration(
        self,
        explorer: Any,
        *,
        orchestrator: Any,
        user_message: str,
        project_root: Any,
        complexity: str,
        max_results: int,
    ) -> Dict[str, Any]:
        """Run exploration through the shared state-passed coordinator."""
        from victor.agent.coordinators.state_context import (
            TransitionApplier,
            create_snapshot,
        )

        snapshot = create_snapshot(orchestrator)
        snapshot = replace(
            snapshot,
            capabilities={**snapshot.capabilities, "task_complexity": complexity},
        )

        result = await explorer.explore(
            snapshot,
            user_message,
            project_root=project_root,
            max_results=max_results,
        )

        if not result.transitions.is_empty():
            await TransitionApplier(orchestrator).apply_batch(result.transitions)

        metrics = getattr(orchestrator, "conversation_state", {}).get("exploration_metrics", {})
        file_paths = result.metadata.get("file_paths") or getattr(
            orchestrator, "conversation_state", {}
        ).get("explored_files", [])
        summary = result.metadata.get("summary") or getattr(
            orchestrator, "conversation_state", {}
        ).get("exploration_summary", "")
        tool_calls = result.metadata.get("tool_calls")
        duration_seconds = result.metadata.get("duration_seconds")

        if tool_calls is None and isinstance(metrics, dict):
            tool_calls = metrics.get("tool_calls", 0)
        if duration_seconds is None and isinstance(metrics, dict):
            duration_seconds = metrics.get("duration_seconds", 0.0)

        return {
            "summary": summary or "",
            "file_paths": list(file_paths or []),
            "tool_calls": int(tool_calls or 0),
            "duration_seconds": float(duration_seconds or 0.0),
        }

    @staticmethod
    def _serialize_conversation_message(message: Any) -> Optional[Dict[str, Any]]:
        """Normalize a conversation message into a serializable mapping."""
        if isinstance(message, dict):
            role = message.get("role")
            content = message.get("content")
        elif hasattr(message, "model_dump"):
            payload = message.model_dump()
            if not isinstance(payload, dict):
                return None
            role = payload.get("role")
            content = payload.get("content")
        else:
            role = getattr(message, "role", None)
            content = getattr(message, "content", None)

        if role is None and content is None:
            return None
        return {"role": role, "content": content}

    def _get_agentic_loop_conversation_history(
        self,
        user_message: str,
    ) -> Optional[List[Dict[str, Any]]]:
        """Return serialized conversation history excluding the current user turn."""
        messages = getattr(self._chat_context, "messages", None)
        if messages is None:
            conversation = getattr(self._chat_context, "conversation", None)
            messages = getattr(conversation, "messages", None)
        if not messages:
            return None

        history = [
            payload
            for message in messages
            if (payload := self._serialize_conversation_message(message)) is not None
        ]
        if (
            history
            and history[-1].get("role") == "user"
            and history[-1].get("content") == user_message
        ):
            history = history[:-1]
        return history or None

    @staticmethod
    def _agentic_loop_accepts_conversation_history(loop: Any) -> bool:
        """Return whether the loop run method supports conversation history."""
        run = getattr(loop, "run", None)
        if run is None:
            return False

        candidate = getattr(run, "side_effect", None)
        if not callable(candidate):
            candidate = run

        try:
            parameters = inspect.signature(candidate).parameters.values()
        except (TypeError, ValueError):
            return True

        return any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            or parameter.name == "conversation_history"
            for parameter in parameters
        )

    def _snapshot_agentic_loop_state(self) -> Dict[str, Any]:
        """Capture mutable state before delegated AgenticLoop execution."""
        conversation = getattr(self._chat_context, "conversation", None)
        messages = getattr(conversation, "messages", None) if conversation is not None else None
        if messages is None:
            messages = getattr(self._chat_context, "messages", None)

        return {
            "messages": list(messages) if messages is not None else None,
            "conversation_system_added": (
                getattr(conversation, "_system_added", _MISSING)
                if conversation is not None
                else _MISSING
            ),
            "chat_context_system_added": getattr(self._chat_context, "_system_added", _MISSING),
            "tool_calls_used": getattr(self._tool_context, "tool_calls_used", _MISSING),
        }

    def _restore_agentic_loop_state(self, snapshot: Dict[str, Any]) -> None:
        """Restore state before resuming the legacy execution loop."""
        conversation = getattr(self._chat_context, "conversation", None)
        if conversation is not None and snapshot.get("messages") is not None:
            restored_messages = list(snapshot["messages"])
            if hasattr(conversation, "_messages"):
                conversation._messages = restored_messages
            else:
                current_messages = getattr(conversation, "messages", None)
                if isinstance(current_messages, list):
                    current_messages[:] = restored_messages

        conversation_system_added = snapshot.get("conversation_system_added", _MISSING)
        if conversation is not None and conversation_system_added is not _MISSING:
            conversation._system_added = conversation_system_added

        chat_context_system_added = snapshot.get("chat_context_system_added", _MISSING)
        if chat_context_system_added is _MISSING:
            if hasattr(self._chat_context, "_system_added"):
                delattr(self._chat_context, "_system_added")
        else:
            self._chat_context._system_added = chat_context_system_added

        tool_calls_used = snapshot.get("tool_calls_used", _MISSING)
        if tool_calls_used is not _MISSING and hasattr(self._tool_context, "tool_calls_used"):
            self._tool_context.tool_calls_used = tool_calls_used

    async def _execute_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute tool calls through the canonical tool-context surface."""
        from victor.agent.background_tasks import BackgroundTaskDef
        import asyncio

        tool_results = await self._tool_context.execute_tool_calls(tool_calls)

        for i, res in enumerate(tool_results):
            raw_result = res.get("result")
            if isinstance(raw_result, BackgroundTaskDef):
                task_id = raw_result.task_id

                # Callback to inject a message when the background task completes
                def on_background_task_done(fut: asyncio.Task):
                    try:
                        task_res = fut.result()
                        # Format the result if needed
                        if isinstance(task_res, dict):
                            stdout = task_res.get("stdout", "").strip()
                            stderr = task_res.get("stderr", "").strip()
                            out = []
                            if stdout:
                                out.append(f"### STDOUT\n```text\n{stdout}\n```")
                            if stderr:
                                out.append(f"### STDERR\n```text\n{stderr}\n```")
                            msg = (
                                "\n\n".join(out)
                                if out
                                else "Command executed successfully with no output."
                            )
                        else:
                            msg = str(task_res)
                    except asyncio.CancelledError:
                        msg = "### ❌ ERROR\nBackground task was cancelled."
                    except Exception as e:
                        msg = f"### ❌ ERROR\nBackground task failed: {e}"

                    content = f"Background task {task_id} ({raw_result.context}) finished with result:\n\n{msg}"
                    self._chat_context.add_message(role="system", content=content)

                # The task is already executing in the background, just attach the callback
                if isinstance(raw_result.task, asyncio.Task):
                    raw_result.task.add_done_callback(on_background_task_done)

                tool_results[i][
                    "result"
                ] = f"### 💡 SYSTEM HINT\nTask is running in the background.\nTask ID: {task_id}\n\nThe watcher will notify you when it completes."

        return tool_results

    # =====================================================================
    # Public API
    # =====================================================================

    async def execute_agentic_loop(
        self,
        user_message: str,
        max_iterations: int = 25,
        runtime_context_overrides: Optional[Dict[str, Any]] = None,
    ) -> CompletionResponse:
        """Execute the full agentic loop via the canonical AgenticLoop runtime.

        Args:
            user_message: Initial user message
            max_iterations: Maximum model turns
            runtime_context_overrides: Scoped runtime hints to apply to turns in this loop

        Returns:
            CompletionResponse with complete response
        """
        # Governance REQUEST phase: gate/redact the user message before it
        # enters history or reaches the LLM. A block short-circuits the turn
        # (no LLM call, nothing stored); a redaction substitutes the text.
        if self._message_policy_gate is not None:
            gate_result = await self._message_policy_gate.gate_request(user_message)
            if not gate_result.allowed:
                return CompletionResponse(
                    content=gate_result.reason or "Your message was blocked by policy.",
                    role="assistant",
                    tool_calls=None,
                )
            user_message = gate_result.content

        # Ensure system prompt is included once at start of conversation
        self._chat_context.conversation.ensure_system_prompt()
        self._chat_context._system_added = True

        # Add user message to history
        self._chat_context.add_message(
            "user",
            user_message,
            metadata={MESSAGE_SOURCE_METADATA_KEY: MessageSource.USER_TYPED.value},
        )
        agentic_loop_state = self._snapshot_agentic_loop_state()

        try:
            response = await self._execute_via_agentic_loop(
                user_message,
                max_iterations,
                runtime_context_overrides=runtime_context_overrides,
            )
        except Exception:
            self._restore_agentic_loop_state(agentic_loop_state)
            raise

        # Governance RESPONSE phase: gate/redact the assistant's final output.
        if self._message_policy_gate is not None and response is not None:
            response = await self._gate_final_response(response)
        return response

    async def _gate_final_response(self, response: CompletionResponse) -> CompletionResponse:
        """Apply the RESPONSE-phase policy gate to a final assistant response."""
        original = response.content or ""
        gate_result = await self._message_policy_gate.gate_response(original)
        if not gate_result.allowed:
            return response.model_copy(
                update={
                    "content": gate_result.reason or "The response was withheld by policy.",
                    "tool_calls": None,
                }
            )
        if gate_result.content != original:
            return response.model_copy(update={"content": gate_result.content})
        return response

    async def prepare_runtime_topology(
        self,
        topology_plan: Any,
        *,
        user_message: str,
        task_classification: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Apply topology-selected runtime preparation steps.

        AgenticLoop owns topology selection. TurnExecutor owns the concrete
        runtime machinery needed to realize those choices.
        """
        from victor.framework.topology_runtime import (
            derive_topology_task_context,
            prepare_topology_runtime_contract,
        )

        orchestrator = self._resolve_orchestrator()
        task_type, complexity = derive_topology_task_context(task_classification)
        prepared_runtime = prepare_topology_runtime_contract(
            topology_plan,
            orchestrator=orchestrator,
            task_type=task_type,
            complexity=complexity,
        )

        if prepared_runtime.parallel_exploration is not None:
            applied = await self._run_parallel_exploration(
                user_message,
                task_classification,
                force=bool(prepared_runtime.parallel_exploration.get("force", True)),
                max_results_override=prepared_runtime.parallel_exploration.get(
                    "max_results_override"
                ),
            )
            return prepared_runtime.to_result(prepared=bool(applied))

        return prepared_runtime.to_result(prepared=prepared_runtime.team_plan is not None)

    async def execute_single_turn(
        self,
        messages: List[Any],
        tools: Optional[List[Any]] = None,
    ) -> CompletionResponse:
        """Execute a single model turn (raw LLM call, no tool execution).

        This is the lowest-level execution path — a single LLM API call.
        For a full turn with tool execution, use execute_turn().

        Args:
            messages: Conversation history
            tools: Optional tool definitions

        Returns:
            CompletionResponse from model
        """
        return await self._execution_provider.execute_turn(
            messages=messages,
            model=self._provider_context.model,
            temperature=self._provider_context.temperature,
            max_tokens=self._provider_context.max_tokens,
            tools=tools,
        )

    async def execute_turn(
        self,
        user_message: str,
        task_classification: Optional[Any] = None,
        is_qa_task: bool = False,
        enable_thinking: bool = False,
        intent: Optional[str] = None,
        temperature_override: Optional[float] = None,
        task_type: Optional[str] = None,
        runtime_context_overrides: Optional[Dict[str, Any]] = None,
    ) -> TurnResult:
        """Execute a single complete turn: LLM call + tool execution.

        This is the primitive unit for AgenticLoop. It performs exactly
        one model call, executes any requested tools, and returns
        structured results without looping.

        The caller (AgenticLoop) owns the iteration loop and decides
        whether to continue, retry, or complete based on TurnResult.

        Args:
            user_message: User message (for tool selection context)
            task_classification: Task classification for tool selection
            is_qa_task: Whether to skip tool selection for Q&A
            enable_thinking: Whether to enable model thinking

        Returns:
            TurnResult with response, tool results, and metadata
        """
        overrides = dict(runtime_context_overrides or {})
        runtime_snapshot = self._apply_runtime_context_overrides(overrides)
        try:
            # PHASE 16: Begin turn for stage transition batching
            # This batches tool executions and applies Phase 1 optimizations consistently
            _orch = self._resolve_orchestrator()
            if _orch and hasattr(_orch, "transition_coordinator") and _orch.transition_coordinator:
                _orch.transition_coordinator.begin_turn()

            # Correlation spine: stamp a fresh turn_id so capture records emitted during
            # this turn (tool.supply, tool.intent, rl_outcome) share one id. Best-effort.
            try:
                from victor.core.context import begin_turn as _begin_turn

                _begin_turn()
            except Exception:  # correlation is non-critical
                pass

            effective_tool_budget = self._coerce_int_override(overrides.get("tool_budget"))
            if effective_tool_budget is None:
                effective_tool_budget = self._tool_context.tool_budget

            # Deterministic read-only plan steps do not need a model turn to
            # choose the obvious tool. Callers that explicitly want the model
            # to drive the loop (e.g. init.md synthesis via `--agentic`) opt
            # out via `runtime_context_overrides["disable_deterministic_fast_path"]`
            # — the fast-path otherwise bypasses the LLM entirely and reports
            # success on a single iteration even though the caller asked for
            # written output that only the model can produce.
            fast_path_disabled = bool(overrides.get("disable_deterministic_fast_path"))
            if (
                not fast_path_disabled
                and not is_qa_task
                and self._tool_context.tool_calls_used < effective_tool_budget
            ):
                deterministic_turn = await self._maybe_execute_deterministic_tool_turn(
                    user_message,
                    task_classification=task_classification,
                )
                if deterministic_turn is not None:
                    return deterministic_turn

            # Select tools (unless Q&A task)
            tools = None
            if (
                not is_qa_task
                and self._provider_context.provider.supports_tools()
                and self._tool_context.tool_calls_used < effective_tool_budget
            ):
                tools = await self._select_tools_for_turn(user_message, intent=intent)

            # Prepare thinking parameter
            provider_kwargs: Dict[str, Any] = {}
            if enable_thinking or self._provider_context.thinking:
                provider_kwargs["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": 10000,
                }

            # Check context compaction before API call
            if task_classification:
                await self._check_context_compaction(user_message, task_classification)

            # Route per-turn temperature through the unified resolver (ADR-013): pass the task type
            # so it applies profile/settings/task-hint precedence + model bounds. An explicit
            # temperature_override (heterogeneous teams / recovery) still wins downstream.
            provider_kwargs["task_type"] = task_type or self._derive_task_type(task_classification)
            if temperature_override is not None:
                provider_kwargs["temperature_override"] = temperature_override
            provider_kwargs.update(self._provider_call_overrides(overrides))

            # Execute model turn
            response = await self._execute_model_turn(tools=tools, **provider_kwargs)

            # Track tokens
            self._accumulate_token_usage(response)

            # Add assistant response/tool-call envelope to conversation history.
            # OpenAI-compatible providers require every subsequent tool message
            # to be paired with the assistant message that declared its
            # tool_calls. Tool-only assistant turns often have empty content.
            if response.content or response.tool_calls:
                self._chat_context.add_message(
                    "assistant",
                    response.content or "",
                    tool_calls=response.tool_calls,
                    metadata={MESSAGE_SOURCE_METADATA_KEY: MessageSource.AGENT_RESPONSE.value},
                )
                if task_classification:
                    await self._check_context_compaction(user_message, task_classification)

            # Execute tool calls if present
            tool_results: List[Dict[str, Any]] = []
            follow_up_suggestions: List[Dict[str, Any]] = []
            all_blocked = False
            if response.tool_calls:
                # Resolve orchestrator/container once for all per-turn services
                _orch = self._resolve_orchestrator()

                # Sort tool calls by dependency order when multiple tools are requested
                if len(response.tool_calls) > 1:
                    try:
                        from victor.agent.protocols.tool_protocols import (
                            ToolDependencyGraphProtocol,
                        )

                        _container = getattr(_orch, "_container", None)
                        _dep_graph = (
                            _container.get_optional(ToolDependencyGraphProtocol)
                            if _container
                            else None
                        )
                        if _dep_graph:
                            _names = [tc.get("name", "") for tc in response.tool_calls]
                            _ordered = _dep_graph.get_execution_order(_names)
                            _name_rank = {n: i for i, n in enumerate(_ordered)}
                            response.tool_calls.sort(
                                key=lambda tc: _name_rank.get(tc.get("name", ""), 999)
                            )
                    except Exception:
                        pass  # Never block execution on ordering failure

                tool_results = await self._execute_tool_calls(response.tool_calls)

                # Record tool→tool transitions for trajectory learning
                try:
                    from victor.agent.protocols.tool_protocols import (
                        ToolDependencyGraphProtocol,
                    )

                    _container = getattr(_orch, "_container", None)
                    _dep_graph = (
                        _container.get_optional(ToolDependencyGraphProtocol) if _container else None
                    )
                    if _dep_graph and len(tool_results) > 1:
                        _task_type = (
                            getattr(task_classification, "task_type", "general")
                            if task_classification
                            else "general"
                        )
                        _tool_names = [
                            r.get("tool_name", "") for r in tool_results if r.get("tool_name")
                        ]
                        for i in range(len(_tool_names) - 1):
                            _dep_graph.record_transition(
                                _tool_names[i],
                                _tool_names[i + 1],
                                _task_type,
                            )
                except Exception:
                    pass  # Trajectory learning is best-effort

                # Check for dedup-blocked spin
                _pipeline = getattr(self._tool_context, "_tool_pipeline", None)
                if _pipeline is None:
                    if _orch:
                        _pipeline = getattr(_orch, "_tool_pipeline", None)
                all_blocked = bool(
                    _pipeline
                    and getattr(
                        _pipeline,
                        "last_batch_effectively_blocked",
                        getattr(_pipeline, "last_batch_all_skipped", False),
                    )
                )

                follow_up_suggestions = self._collect_follow_up_suggestions(tool_results)
                self._inject_tool_follow_up_guidance(
                    follow_up_suggestions,
                    tool_results,
                    all_tools_blocked=all_blocked,
                )

                # Record failures for optimization hints
                for result in tool_results:
                    if not result.get("success"):
                        _injector = (
                            getattr(_orch, "_optimization_injector", None) if _orch else None
                        )
                        if _injector and result.get("error"):
                            _injector.record_failure(
                                result.get("tool_name", "unknown"),
                                result["error"],
                            )

            # PHASE 16: End turn for stage transition batching
            # Processes all batched tools and applies Phase 1 optimizations
            new_stage = None
            if _orch and hasattr(_orch, "transition_coordinator") and _orch.transition_coordinator:
                new_stage = _orch.transition_coordinator.end_turn()
                if new_stage:
                    logger.debug(f"[TurnExecutor] Stage transitioned to {new_stage.name}")

            return TurnResult(
                response=response,
                tool_results=tool_results,
                follow_up_suggestions=follow_up_suggestions,
                has_tool_calls=bool(response.tool_calls),
                tool_calls_count=len(response.tool_calls) if response.tool_calls else 0,
                all_tools_blocked=all_blocked,
                is_qa_response=is_qa_task and bool(response.content),
            )
        finally:
            self._restore_runtime_context_overrides(runtime_snapshot)

    async def _maybe_execute_deterministic_tool_turn(
        self,
        user_message: str,
        task_classification: Optional[Any] = None,
    ) -> Optional[TurnResult]:
        """Execute obvious read-only plan steps without a model tool-selection turn."""
        tool_calls = self._deterministic_tool_calls(user_message)
        if not tool_calls:
            return None

        self._chat_context.add_message(
            "assistant",
            "",
            tool_calls=tool_calls,
            metadata={MESSAGE_SOURCE_METADATA_KEY: MessageSource.AGENT_RESPONSE.value},
        )
        if task_classification:
            await self._check_context_compaction(user_message, task_classification)

        tool_results = await self._execute_tool_calls(tool_calls)
        response_content = self._summarize_deterministic_tool_results(tool_calls, tool_results)
        return TurnResult(
            response=CompletionResponse(
                content=response_content,
                role="assistant",
                tool_calls=tool_calls,
                metadata={"deterministic_tool_execution": True},
            ),
            tool_results=tool_results,
            has_tool_calls=True,
            tool_calls_count=len(tool_calls),
        )

    @staticmethod
    def _deterministic_tool_calls(user_message: str) -> List[Dict[str, Any]]:
        """Return tool calls for explicit read-only plan-step phrasing.

        Generic across ecosystems — the framework no longer hardcodes Rust/Cargo
        dispatch.  Manifest discovery delegates to
        ``language_manifests.LanguageManifestHandler`` instances, which already
        cover Rust, Python, JavaScript, TypeScript, Go, Java, Kotlin, Ruby, and
        PHP.  Verticals add new ecosystems via ``register_manifest_handler()``.
        """
        from pathlib import Path

        normalized = " ".join(user_message.strip().split())
        if not normalized:
            return []

        tool_calls: List[Dict[str, Any]] = []
        lowered = normalized.lower()

        # Long instruction sets (synthesis prompts, init.md generation, etc.)
        # frequently mention "manifest" or "package layout" in passing while
        # actually asking for a written deliverable. The fast-path is only
        # appropriate for short, focused plan steps like "review the build
        # manifest", not multi-paragraph instructions. If the prompt asks to
        # *produce* content (write/generate/synthesize/return markdown) or is
        # simply too long to be a single read step, fall through to the real
        # agent loop so the model gets to actually compose the output.
        _OUTPUT_PRODUCTION_TOKENS = (
            "generate ",
            " write ",
            "write a ",
            "write the ",
            "synthesize",
            "compose ",
            "produce ",
            "return only",
            "return the markdown",
            "init.md",
            "begin with `#",
            "begin with #",
        )
        if len(normalized) > 400 or any(tok in lowered for tok in _OUTPUT_PRODUCTION_TOKENS):
            return []

        paths: List[str] = []

        # Generic manifest-aware discovery: pick a language hint from the message
        # only if it's unambiguous (the language name is present AND the step
        # mentions a workspace/manifest/component review intent), then delegate
        # to the language's manifest handler.  Without a clear language hint we
        # skip — the step falls through to the regular agent path.
        _has_manifest_review_intent = (
            "manifest" in lowered
            or "workspace" in lowered
            or "package layout" in lowered
            or "module layout" in lowered
            or "project layout" in lowered
            or "review targets" in lowered
        )
        if _has_manifest_review_intent:
            try:
                from victor.agent.planning.language_manifests import (
                    _HANDLERS as _LANG_HANDLERS,
                )
                from victor.agent.planning.language_manifests import (
                    select_language_manifests,
                )

                # Match on whole-word language names so "java" doesn't pick up
                # "javascript" tokens — the registry already separates them.
                detected_languages = [
                    lang
                    for lang in _LANG_HANDLERS.keys()
                    if re.search(rf"\b{re.escape(lang)}\b", lowered)
                ]
                for lang in detected_languages:
                    selection = select_language_manifests(lang, normalized, root=Path.cwd())
                    paths.extend(selection.paths)
            except Exception:
                logger.debug("Language manifest selection failed; no paths added")

        # Generic ``read <path/to/manifest-or-source-file>`` extractor.  The
        # extension allowlist mirrors the multi-ecosystem set used by
        # ``_is_directory_listing_only`` in planning_runtime.
        _READ_PATH_RE = re.compile(
            r"^read\s+(?:the\s+)?(?:root\s+)?(?P<path>"
            r"[\w./-]*\.(?:toml|json|ya?ml|lock|mod|gradle|gradle\.kts|"
            r"xml|gemspec|rs|py|js|jsx|ts|tsx|go|java|kt|rb|php|swift|cs|cpp|c|h|hpp))"
            r"\b",
            re.IGNORECASE,
        )
        match = _READ_PATH_RE.search(normalized)
        if match:
            extracted = match.group("path")
            if extracted and Path(extracted).is_file():
                paths.append(extracted)

        deduped_paths = list(dict.fromkeys(paths))
        for path in deduped_paths:
            clean_path = path[2:] if path.startswith("./") else path
            if ".." in clean_path.split("/"):
                continue
            tool_calls.append(
                TurnExecutor._deterministic_tool_call(
                    "read",
                    {"path": clean_path, "offset": 0, "limit": 2000},
                )
            )
        return tool_calls

    @staticmethod
    def _deterministic_tool_call(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Build a stable OpenAI-compatible tool call envelope."""
        digest = hashlib.sha1(f"{name}:{arguments!r}".encode("utf-8")).hexdigest()[:12]
        return {
            "id": f"call_deterministic_{digest}",
            "name": name,
            "arguments": arguments,
        }

    @staticmethod
    def _summarize_deterministic_tool_results(
        tool_calls: List[Dict[str, Any]],
        tool_results: List[Dict[str, Any]],
    ) -> str:
        """Create a small deterministic completion message for read-only tool batches.

        For single shell calls the actual stdout is appended.
        For multi-read batches the successfully-read file paths are appended as a
        plain list so that downstream ``produces``-key extraction via
        ``_extract_list_from_output`` gets a structured list rather than prose.
        """
        if not tool_results:
            return ""

        ok = sum(1 for result in tool_results if result.get("success"))
        failed = len(tool_results) - ok
        tool_names = ", ".join(
            dict.fromkeys(str(call.get("name", "tool")) for call in tool_calls).keys()
        )
        header = (
            f"Deterministic read-only execution completed for {tool_names}: "
            f"{ok} succeeded, {failed} failed."
        )

        if len(tool_calls) == 1 and tool_calls[0].get("name") == "shell":
            for result in tool_results:
                content = (
                    result.get("stdout")
                    or result.get("content")
                    or result.get("full_result")
                    or result.get("result")
                )
                if content:
                    text = str(content).strip()
                    if text:
                        return f"{header}\n\n{text[:8000]}"
            return header

        # For multi-call read/ls batches: emit the file paths as a plain list so
        # _extract_list_from_output can populate plan_state[produces_key] correctly.
        paths = [
            str((call.get("arguments") or {}).get("path", ""))
            for call, res in zip(tool_calls, tool_results)
            if res.get("success") and (call.get("arguments") or {}).get("path")
        ]
        if paths:
            return "\n".join(paths)

        return header

    # =====================================================================
    # AgenticLoop Integration (Phase 10)
    # =====================================================================

    async def _execute_via_agentic_loop(
        self,
        user_message: str,
        max_iterations: int,
        runtime_context_overrides: Optional[Dict[str, Any]] = None,
    ) -> CompletionResponse:
        """Delegate execution to AgenticLoop for enhanced evaluation.

        AgenticLoop runs the PERCEIVE -> PLAN -> ACT -> EVALUATE -> DECIDE
        cycle, calling self.execute_turn() for each turn.

        Args:
            user_message: User message (already added to conversation)
            max_iterations: Maximum iterations

        Returns:
            CompletionResponse from the final turn
        """
        from victor.framework.agentic_loop import AgenticLoop

        # Classify task for tool selection
        task_classification = self._provider_context.task_classifier.classify(user_message)
        is_qa = self._is_question_only(user_message)

        # Calculate iteration budget. A per-call ``iteration_budget`` override
        # (from runtime_context_overrides) is read directly here rather than via
        # a mutate-then-restore of the shared settings.chat_max_iterations.
        iter_override = self._coerce_int_override(
            (runtime_context_overrides or {}).get("iteration_budget")
        )
        max_iterations_setting = (
            iter_override
            if iter_override is not None
            else getattr(self._chat_context.settings, "chat_max_iterations", 10)
        )
        task_budget = max(task_classification.tool_budget * 2, 1)
        iteration_budget = min(task_budget, max_iterations_setting, max_iterations)

        # Create AgenticLoop with self as turn_executor
        # Use a lightweight mock orchestrator since AgenticLoop only needs
        # turn_executor for single-turn execution
        # ADR-009 strategy + ADR-011 judge pinning: env override, settings default, then
        # downgrade rubric/hybrid unless the session (judge) model passed calibration.
        from victor.agent.services.judge_calibration_gate import resolve_completion_strategy

        _settings = getattr(self._chat_context, "settings", None)
        _strategy = resolve_completion_strategy(
            _settings, getattr(self._provider_context, "model", None)
        )
        _rubric_fn = self._build_rubric_complete_fn() if _strategy in ("rubric", "hybrid") else None
        from victor.framework.effect_gate import resolve_effect_gate_enabled

        _effect_gate = resolve_effect_gate_enabled(_settings)
        loop = AgenticLoop(
            orchestrator=None,
            turn_executor=self,
            max_iterations=iteration_budget,
            enable_fulfillment_check=True,
            enable_adaptive_iterations=True,
            exploration_settings=getattr(self._chat_context.settings, "exploration", None),
            config={"completion_strategy": _strategy, "enable_effect_gate": _effect_gate},
            rubric_complete_fn=_rubric_fn,
            verifier=getattr(self, "_verifier", None),
            max_verify_retries=getattr(self, "_max_verify_retries", 0),
        )

        # Inject task classification into state for execute_turn()
        context = {
            "_task_classification": task_classification,
            "_is_qa_task": is_qa,
        }
        # Seed the initial tool_budget from the service so _select_topology
        # doesn't fall back to the hardcoded default of 10.  Without this seed,
        # sub-agents whose ToolService starts at 50 (or any value > 10) have
        # their topology plan compute tool_budget=10 and then apply it via
        # _apply_tool_budget_override, which collapses every sub-agent to 10
        # tool calls regardless of the configured step budget.
        _service_budget = getattr(self._tool_context, "tool_budget", None)
        if isinstance(_service_budget, int) and _service_budget > 10:
            context["tool_budget"] = _service_budget
        if runtime_context_overrides:
            context["runtime_context_overrides"] = dict(runtime_context_overrides)
        conversation_history = self._get_agentic_loop_conversation_history(user_message)

        loop_run_kwargs = {"context": context}
        if conversation_history is not None and self._agentic_loop_accepts_conversation_history(
            loop
        ):
            loop_run_kwargs["conversation_history"] = conversation_history

        loop_result = await loop.run(user_message, **loop_run_kwargs)
        loop_success = getattr(loop_result, "success", True)
        # Real agentic-loop iteration count (excludes rubric-judge / recovery sub-calls) — surfaced on
        # the response metadata so it can reach TaskResult.metadata for observability + A/B harnesses.
        loop_iterations = len(getattr(loop_result, "iterations", []) or [])
        loop_error = None
        if not loop_success:
            loop_error = "Agentic loop ended before satisfying the task"
            if loop_result.iterations and loop_result.iterations[-1].evaluation:
                loop_error = loop_result.iterations[-1].evaluation.reason or loop_error
        has_tool_evidence = self._loop_has_successful_tool_evidence(loop_result)

        # Extract the last TurnResult's response
        if loop_result.iterations:
            last = loop_result.iterations[-1]
            if last.action_result is not None:
                turn_result = last.action_result
                if hasattr(turn_result, "response"):
                    response = turn_result.response
                    if not loop_success and has_tool_evidence and not response.content:
                        synthesized = await self._ensure_complete_response(
                            None,
                            ToolFailureContext(),
                        )
                        if synthesized.content and not synthesized.content.startswith(
                            "I was unable to generate a complete response."
                        ):
                            metadata = dict(getattr(synthesized, "metadata", None) or {})
                            metadata["agentic_loop_success"] = True
                            metadata["agentic_loop_recovered"] = True
                            metadata["agentic_loop_recovery_reason"] = loop_error
                            metadata["agentic_loop_iterations"] = loop_iterations
                            synthesized.metadata = metadata
                            return synthesized

                    metadata = dict(getattr(response, "metadata", None) or {})
                    metadata["agentic_loop_success"] = loop_success
                    metadata["agentic_loop_iterations"] = loop_iterations
                    if loop_error:
                        metadata["agentic_loop_error"] = loop_error
                    response.metadata = metadata
                    return response

        # Fallback: ensure we have a response
        failure_context = ToolFailureContext()
        response = await self._ensure_complete_response(None, failure_context)
        metadata = dict(getattr(response, "metadata", None) or {})
        metadata["agentic_loop_success"] = loop_success
        metadata["agentic_loop_iterations"] = loop_iterations
        if loop_error:
            metadata["agentic_loop_error"] = loop_error
        response.metadata = metadata
        return response

    async def synthesize_from_tool_evidence(
        self,
        *,
        recovery_reason: Optional[str] = None,
    ) -> TurnResult:
        """Produce a final response from existing tool evidence without another tool turn."""
        response = await self._ensure_complete_response(None, ToolFailureContext())
        metadata = dict(getattr(response, "metadata", None) or {})
        metadata["agentic_loop_synthesis"] = True
        if recovery_reason:
            metadata["agentic_loop_synthesis_reason"] = recovery_reason
        response.metadata = metadata
        return TurnResult(
            response=response,
            tool_results=[],
            has_tool_calls=False,
            tool_calls_count=0,
        )

    @staticmethod
    def _loop_has_successful_tool_evidence(loop_result: Any) -> bool:
        """Return True when a loop gathered successful tool output before stopping."""
        for iteration in getattr(loop_result, "iterations", []) or []:
            action_result = getattr(iteration, "action_result", None)
            if action_result is None:
                continue
            successful_tool_count = getattr(action_result, "successful_tool_count", 0)
            if successful_tool_count:
                return True
            for result in getattr(action_result, "tool_results", []) or []:
                if isinstance(result, dict) and result.get("success"):
                    return True
        return False

    # =====================================================================
    # Q&A Detection
    # =====================================================================

    @staticmethod
    def _is_question_only(message: str) -> bool:
        """Detect messages that are pure Q&A and unlikely to need tools.

        Enhanced to detect more Q&A patterns:
        - Direct questions
        - Explanation requests
        - Definition requests
        - Concept explanations

        Returns True for conversational/knowledge questions like:
        - "What is 2+2?"
        - "Explain Python decorators"
        - "How does X work?"

        Returns False for messages that imply code/file work:
        - "Fix the bug in main.py"
        - "Create a new file"
        - "Refactor the database module"
        """
        import re

        msg = message.strip().lower()

        # Short messages ending with ? are almost always Q&A
        if len(msg) < 120 and msg.endswith("?"):
            # Unless they contain action words implying code work
            action_words = (
                "fix",
                "create",
                "write",
                "edit",
                "refactor",
                "add",
                "implement",
                "update",
                "delete",
                "remove",
                "change",
                "modify",
                "build",
                "deploy",
                "run",
                "test",
                "debug",
                "install",
                "configure",
                "setup",
                "migrate",
            )
            words = set(msg.split())
            if not words.intersection(action_words):
                return True

        # Explicit "explain", "what is", "how does" patterns
        qa_prefixes = (
            "what is",
            "what are",
            "what's",
            "who is",
            "who are",
            "how does",
            "how do",
            "how is",
            "how can",
            "why does",
            "why do",
            "why is",
            "when does",
            "when is",
            "explain",
            "describe",
            "define",
            "tell me about",
            "can you explain",
            "what does",
            "reply with",
            "answer",
            "say ",
        )
        if any(msg.startswith(p) for p in qa_prefixes):
            return True

        # NEW: Add more Q&A patterns with regex
        qa_patterns = [
            r"^(what|how|why|when|where|who|which|whose|explain|describe|define)",
            r"tell me (about|how|why)",
            r"what('s| is| are) the (difference|definition)",
            r"how do (you|you think|I)",
            r"can you (explain|describe|tell me)",
            r"^(pros and cons|advantages|disadvantages)",
        ]

        combined_pattern = "|".join(qa_patterns)
        if re.match(combined_pattern, msg):
            # Additional check: exclude if it contains implementation keywords
            implementation_keywords = ["implement", "code", "write", "create", "build"]
            if not any(kw in msg for kw in implementation_keywords):
                return True

        # Check for explanation/definition keywords
        explanation_keywords = ["explain", "describe", "define", "what is", "how does"]
        if any(keyword in msg for keyword in explanation_keywords):
            # But exclude if it also includes implementation keywords
            implementation_keywords = ["implement", "code", "write", "create", "build"]
            if not any(kw in msg for kw in implementation_keywords):
                return True

        return False

    # =====================================================================
    # Parallel Exploration
    # =====================================================================

    async def _run_parallel_exploration(
        self,
        user_message: str,
        task_classification: Any,
        *,
        force: bool = False,
        max_results_override: Optional[int] = None,
    ) -> bool:
        """Run parallel exploration subagents for complex tasks.

        Spawns concurrent RESEARCHER subagents to explore the codebase
        before the main agentic loop starts. Findings are injected into
        the conversation context as a user message.

        Only fires for COMPLEX/ACTION tasks when parallel_exploration is enabled,
        unless a topology-selected runtime explicitly forces it.
        """
        # Only explore once per conversation, not on continuations
        if self._exploration_done:
            return False
        if user_message.startswith("You have not edited") or user_message == "Continue.":
            return False

        global _EXPLORATION_IN_PROGRESS
        if _EXPLORATION_IN_PROGRESS:
            return False  # Prevent recursive subagent exploration

        # Check if exploration is enabled and task warrants it
        try:
            from victor.config.settings import load_settings

            settings = load_settings()
            pipeline = getattr(settings, "pipeline", None)
            if pipeline and not getattr(pipeline, "parallel_exploration", True):
                return False

            from victor.framework.task.protocols import TaskComplexity

            complexity = getattr(task_classification, "complexity", None)
            if not force and complexity not in {
                TaskComplexity.COMPLEX,
                TaskComplexity.ACTION,
                TaskComplexity.ANALYSIS,
            }:
                return False
        except Exception:
            return False

        try:
            from pathlib import Path

            from victor.config.settings import get_project_paths

            project_root = Path(get_project_paths().project_root)
            explorer, uses_state_passed = self._resolve_parallel_explorer()

            # Calculate resource-aware exploration budget
            from victor.agent.budget.resource_calculator import (
                calculate_exploration_budget,
            )

            provider_name = getattr(self._provider_context, "provider_name", "ollama")
            if callable(provider_name):
                provider_name = "ollama"  # Mock fallback
            model_name = getattr(self._provider_context, "model", None)
            complexity_str = getattr(task_classification, "complexity", "action")
            if hasattr(complexity_str, "value"):
                complexity_str = complexity_str.value

            budget = calculate_exploration_budget(
                complexity=complexity_str,
                provider=provider_name,
                model=model_name,
            )

            if budget.max_parallel_agents == 0:
                self._exploration_done = True
                return False

            max_results = max_results_override or budget.tool_budget_per_agent
            if isinstance(max_results, int):
                max_results = max(1, max_results)
            else:
                max_results = budget.tool_budget_per_agent

            _EXPLORATION_IN_PROGRESS = True
            try:
                if uses_state_passed:
                    orchestrator = self._resolve_orchestrator()
                    if orchestrator is None:
                        raise RuntimeError("state-passed exploration requires orchestrator context")
                    findings = await asyncio.wait_for(
                        self._run_state_passed_parallel_exploration(
                            explorer,
                            orchestrator=orchestrator,
                            user_message=user_message,
                            project_root=project_root,
                            complexity=complexity_str,
                            max_results=max_results,
                        ),
                        timeout=budget.exploration_timeout,
                    )
                else:
                    findings = self._normalize_exploration_payload(
                        await asyncio.wait_for(
                            explorer.explore_parallel(
                                task_description=user_message,
                                project_root=project_root,
                                max_results=max_results,
                            ),
                            timeout=budget.exploration_timeout,
                        )
                    )
            finally:
                _EXPLORATION_IN_PROGRESS = False
                self._exploration_done = True  # Never explore again this conversation

            if findings["summary"]:
                from victor.agent.conversation.types import (
                    MESSAGE_SOURCE_METADATA_KEY,
                    MessageSource,
                )

                self._chat_context.add_message(
                    "user",
                    f"[Parallel exploration results]\n{findings['summary']}",
                    metadata={MESSAGE_SOURCE_METADATA_KEY: MessageSource.AGENT_GUIDANCE.value},
                )
                logger.info(
                    "Parallel exploration: %d files, %d tool calls, %.1fs",
                    len(findings["file_paths"]),
                    findings["tool_calls"],
                    findings["duration_seconds"],
                )

            return True

        except asyncio.TimeoutError:
            logger.debug("Parallel exploration timed out (90s), skipping")
            return False
        except Exception as e:
            logger.debug("Parallel exploration skipped: %s", e)
            return False

    # =====================================================================
    # Private Methods
    # =====================================================================

    @staticmethod
    def _collect_follow_up_suggestions(
        tool_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Collect unique follow-up suggestions from blocked or failed tool results."""
        suggestions: List[Dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()

        for result in tool_results:
            if result.get("success"):
                continue
            raw_suggestions = result.get("follow_up_suggestions")
            if not isinstance(raw_suggestions, list):
                continue

            for suggestion in raw_suggestions:
                if not isinstance(suggestion, dict):
                    continue
                command = suggestion.get("command")
                if not isinstance(command, str) or not command.strip():
                    continue
                tool_name = str(suggestion.get("tool") or "")
                description = str(
                    suggestion.get("description")
                    or suggestion.get("reason")
                    or "Use this alternative tool path."
                ).strip()
                signature = (tool_name, command.strip(), description)
                if signature in seen:
                    continue
                seen.add(signature)

                normalized = dict(suggestion)
                normalized["tool"] = tool_name
                normalized["command"] = command.strip()
                normalized["description"] = description
                normalized["reason"] = (
                    str(suggestion.get("reason") or description).strip() or description
                )
                suggestions.append(normalized)
                if len(suggestions) >= 4:
                    return suggestions

        return suggestions

    @staticmethod
    def _format_tool_follow_up_guidance(suggestions: List[Dict[str, Any]]) -> str:
        """Format a compact recovery nudge for the next model turn."""
        lines = [
            "[Tool recovery guidance]",
            "The previous tool batch was blocked or unproductive. Do not repeat the same tool call.",
            "Choose one of these next actions instead:",
        ]
        for suggestion in suggestions[:2]:
            command = suggestion.get("command", "").strip()
            description = str(
                suggestion.get("description")
                or suggestion.get("reason")
                or "Try this alternative next."
            ).strip()
            if command:
                lines.append(f"- {command}: {description}")
        return "\n".join(lines)

    def _inject_tool_follow_up_guidance(
        self,
        follow_up_suggestions: List[Dict[str, Any]],
        tool_results: List[Dict[str, Any]],
        *,
        all_tools_blocked: bool,
    ) -> None:
        """Inject actionable recovery guidance into conversation state for the next turn."""
        has_success = any(result.get("success") for result in tool_results)
        requires_recovery_nudge = all_tools_blocked or (tool_results and not has_success)
        if not requires_recovery_nudge:
            self._last_tool_follow_up_guidance_signature = None
            return

        if not follow_up_suggestions or not hasattr(self._chat_context, "add_message"):
            return

        signature = tuple(
            suggestion.get("command", "").strip() for suggestion in follow_up_suggestions[:2]
        )
        if not any(signature) or signature == self._last_tool_follow_up_guidance_signature:
            return

        from victor.agent.conversation.types import (
            MESSAGE_SOURCE_METADATA_KEY,
            MessageSource,
        )

        self._chat_context.add_message(
            "user",
            self._format_tool_follow_up_guidance(follow_up_suggestions),
            metadata={MESSAGE_SOURCE_METADATA_KEY: MessageSource.AGENT_GUIDANCE.value},
        )
        self._last_tool_follow_up_guidance_signature = signature

    async def _select_tools_for_turn(
        self,
        user_message: str,
        intent: Optional[str] = None,
    ) -> Optional[List[Any]]:
        """Select tools for the current iteration.

        Args:
            user_message: Original user message
            intent: Serialized ActionIntent value from perception (e.g. "read_only").
                    When provided, tools blocked for that intent are removed.

        Returns:
            List of tool definitions or None
        """
        conversation_depth = self._chat_context.conversation.message_count()
        from victor.agent.tool_selection.history_projection import _selector_history_projection

        conversation_history = _selector_history_projection(self._chat_context.messages)

        tools = await self._tool_context.tool_selector.select_tools(
            user_message,
            use_semantic=self._tool_context.use_semantic_selection,
            conversation_history=conversation_history,
            conversation_depth=conversation_depth,
        )

        # When the caller curated the toolset (_enabled_tools is a real non-empty
        # collection), the schema must reach the LLM UNCHANGED — no stage
        # prioritization or intent filtering that drops curated tools (code/graph).
        # select_tools() already short-circuits to the stable curated set (#368);
        # skip the downstream gates too so the full 6-tool set survives.
        _curated = getattr(self._tool_context.tool_selector, "_enabled_tools", None)
        if isinstance(_curated, (set, frozenset, list)) and _curated:
            return tools

        # Prioritize by stage
        tools = self._tool_context.tool_selector.prioritize_by_stage(user_message, tools)

        # Delegate intent filtering to the canonical planner when available.
        planner = getattr(self._tool_context, "_tool_planner", None)
        if tools and intent and planner and hasattr(planner, "filter_tools_by_intent"):
            try:
                from victor.agent.action_authorizer import ActionIntent

                tools = planner.filter_tools_by_intent(
                    tools,
                    current_intent=ActionIntent(intent),
                    user_message=user_message,
                )
            except (ValueError, ImportError, AttributeError):
                pass
        # Backward-compatible fallback for shim contexts that have not yet been wired
        # through the canonical planner service.
        elif tools and intent:
            try:
                from victor.agent.action_authorizer import (
                    ActionIntent,
                    is_tool_blocked_for_intent,
                )

                action_intent = ActionIntent(intent)
                tools = [
                    t
                    for t in tools
                    if not is_tool_blocked_for_intent(
                        (t.get("name") if isinstance(t, dict) else getattr(t, "name", None)) or "",
                        action_intent,
                        user_message,
                    )
                ]
            except (ValueError, ImportError, AttributeError):
                pass

        return tools

    def _build_rubric_complete_fn(self) -> Optional[Any]:
        """The LLM rubric judge's ``complete_fn(prompt)->text`` (ADR-009, FEP-0030).

        Delegates to the ``agent.completion_judge`` backend resolver: the judge is
        the session model by default, or a decoupled side-LLM (``llm:<model>@<url>``).
        Returns None when no LLM judge applies (the loop then uses the heuristic judge).
        Used only when completion_strategy is rubric/hybrid.
        """
        from victor.agent.services.judge_calibration_gate import build_judge_complete_fn

        settings = getattr(self._chat_context, "settings", None)
        return build_judge_complete_fn(settings, self._provider_context)

    @staticmethod
    def _derive_task_type(task_classification: Any) -> Optional[str]:
        """Best-effort task_type string from a classification (str / object / dict)."""
        if task_classification is None:
            return None
        if isinstance(task_classification, str):
            return task_classification
        if isinstance(task_classification, dict):
            return task_classification.get("task_type")
        for attr in ("task_type", "unified_task_type", "category"):
            val = getattr(task_classification, attr, None)
            if val is not None:
                return val if isinstance(val, str) else getattr(val, "value", None)
        return None

    def _resolve_turn_temperature(
        self, task_type: Optional[str], model: str, explicit_override: Optional[float]
    ) -> Optional[float]:
        """Resolve this turn's temperature via the unified resolver (ADR-013).

        An explicit caller override (heterogeneous teams / recovery) wins. Otherwise delegate to the
        orchestrator-owned :class:`TemperatureResolver`; if unavailable (state-passed contexts), fall
        back to the provider-context temperature so behaviour is never worse than before.
        """
        from victor.agent.services.temperature_resolution import resolve_effective_temperature

        return resolve_effective_temperature(
            self._resolve_orchestrator(),
            task_type=task_type,
            model=model,
            base_temperature=self._provider_context.temperature,
            explicit_override=explicit_override,
        )

    async def _execute_model_turn(
        self,
        tools: Optional[List[Any]] = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Execute a single model turn.

        Args:
            tools: Optional tool definitions
            **kwargs: Additional provider parameters; temperature_override overrides
                the provider-default temperature for this task type.

        Returns:
            CompletionResponse from model
        """
        model = self._provider_context.model
        # Per-turn temperature via the unified resolver (ADR-013): task_type drives
        # profile/settings/task-hint precedence + model-bounds. An explicit override
        # (heterogeneous teams / recovery) still wins.
        task_type = kwargs.pop("task_type", None)
        explicit_override = kwargs.pop("temperature_override", None)
        temperature = self._resolve_turn_temperature(task_type, model, explicit_override)

        # Forward a configured per-member reasoning_effort only when the
        # provider+model report support, so it is never sent to a model that
        # would reject it (the provider also strips it defensively).
        reasoning_effort = getattr(self._provider_context, "reasoning_effort", None)
        if (
            reasoning_effort
            and "reasoning_effort" not in kwargs
            and getattr(self._provider_context, "supports_reasoning_effort", None) is not None
            and self._provider_context.supports_reasoning_effort(model)
        ):
            kwargs["reasoning_effort"] = reasoning_effort

        return await self._execution_provider.execute_turn(
            messages=self._chat_context.messages,
            model=model,
            temperature=temperature,
            max_tokens=self._provider_context.max_tokens,
            tools=tools,
            **kwargs,
        )

    def _provider_call_overrides(self, overrides: Dict[str, Any]) -> Dict[str, Any]:
        """Extract provider-facing runtime override hints."""
        provider_keys = (
            "provider_hint",
            "execution_mode",
            "escalation_target",
            "topology_action",
            "topology_kind",
            "topology_metadata",
        )
        return {
            key: value
            for key, value in overrides.items()
            if key in provider_keys and value is not None
        }

    def _apply_runtime_context_overrides(
        self,
        overrides: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Apply temporary runtime overrides for the duration of one turn."""
        if not overrides:
            return None

        snapshot: Dict[str, Any] = {}
        orchestrator = self._resolve_orchestrator()
        snapshot["orchestrator"] = orchestrator
        snapshot["chat_runtime_context"] = getattr(
            self._chat_context,
            "_runtime_context_overrides",
            _MISSING,
        )
        self._chat_context._runtime_context_overrides = dict(overrides)

        if orchestrator is not None:
            snapshot["orchestrator_runtime_context"] = getattr(
                orchestrator,
                "_runtime_tool_context_overrides",
                _MISSING,
            )
            merged_context = {}
            previous_runtime_context = snapshot["orchestrator_runtime_context"]
            if isinstance(previous_runtime_context, dict):
                merged_context.update(previous_runtime_context)
            merged_context.update(overrides)
            orchestrator._runtime_tool_context_overrides = merged_context

        tool_budget = self._coerce_int_override(overrides.get("tool_budget"))
        if tool_budget is not None:
            self._apply_tool_budget_override(tool_budget, snapshot, orchestrator)

        # NOTE: ``iteration_budget`` is no longer applied here by mutating the
        # shared ``settings.chat_max_iterations``; it is read directly in
        # ``_execute_via_agentic_loop`` from ``runtime_context_overrides``.

        return snapshot

    def _restore_runtime_context_overrides(self, snapshot: Optional[Dict[str, Any]]) -> None:
        """Restore runtime state after one turn completes."""
        if not snapshot:
            return

        orchestrator = snapshot.get("orchestrator")
        if orchestrator is not None:
            previous_context = snapshot.get("orchestrator_runtime_context", _MISSING)
            if previous_context is _MISSING:
                if hasattr(orchestrator, "_runtime_tool_context_overrides"):
                    delattr(orchestrator, "_runtime_tool_context_overrides")
            else:
                orchestrator._runtime_tool_context_overrides = previous_context

        previous_chat_context = snapshot.get("chat_runtime_context", _MISSING)
        if previous_chat_context is _MISSING:
            if hasattr(self._chat_context, "_runtime_context_overrides"):
                delattr(self._chat_context, "_runtime_context_overrides")
        else:
            self._chat_context._runtime_context_overrides = previous_chat_context

        self._restore_tool_budget_override(snapshot, orchestrator)

    def _apply_tool_budget_override(
        self,
        tool_budget: int,
        snapshot: Dict[str, Any],
        orchestrator: Any,
    ) -> None:
        """Apply a temporary tool budget override to known runtime owners."""
        if orchestrator is not None and hasattr(orchestrator, "tool_budget"):
            snapshot["orchestrator_tool_budget"] = getattr(orchestrator, "tool_budget", _MISSING)
            try:
                orchestrator.tool_budget = max(0, tool_budget)
            except Exception:
                pass

        tool_service = (
            getattr(orchestrator, "_tool_service", None) if orchestrator is not None else None
        )
        if tool_service is None:
            tool_service = getattr(self._tool_context, "_tool_service", None)
        if tool_service is not None and hasattr(tool_service, "get_tool_budget"):
            snapshot["tool_service_budget"] = getattr(
                tool_service,
                "budget",
                (
                    tool_service.get_budget_info().get("max")
                    if hasattr(tool_service, "get_budget_info")
                    else tool_service.get_tool_budget()
                ),
            )
            if hasattr(tool_service, "set_tool_budget"):
                try:
                    tool_service.set_tool_budget(max(0, tool_budget))
                except Exception:
                    pass

        tool_pipeline = (
            getattr(orchestrator, "_tool_pipeline", None) if orchestrator is not None else None
        )
        if tool_pipeline is None:
            tool_pipeline = getattr(self._tool_context, "_tool_pipeline", None)
        pipeline_config = getattr(tool_pipeline, "config", None)
        if pipeline_config is not None and hasattr(pipeline_config, "tool_budget"):
            snapshot["pipeline_tool_budget"] = getattr(pipeline_config, "tool_budget", _MISSING)
            try:
                pipeline_config.tool_budget = max(0, tool_budget)
            except Exception:
                pass

    def _restore_tool_budget_override(
        self,
        snapshot: Dict[str, Any],
        orchestrator: Any,
    ) -> None:
        """Restore prior tool budget state after a temporary override."""
        previous_orchestrator_budget = snapshot.get("orchestrator_tool_budget", _MISSING)
        if orchestrator is not None and previous_orchestrator_budget is not _MISSING:
            try:
                orchestrator.tool_budget = previous_orchestrator_budget
            except Exception:
                pass

        tool_service = (
            getattr(orchestrator, "_tool_service", None) if orchestrator is not None else None
        )
        if tool_service is None:
            tool_service = getattr(self._tool_context, "_tool_service", None)
        previous_service_budget = snapshot.get("tool_service_budget", _MISSING)
        if (
            tool_service is not None
            and previous_service_budget is not _MISSING
            and hasattr(tool_service, "set_tool_budget")
        ):
            try:
                tool_service.set_tool_budget(previous_service_budget)
            except Exception:
                pass

        tool_pipeline = (
            getattr(orchestrator, "_tool_pipeline", None) if orchestrator is not None else None
        )
        if tool_pipeline is None:
            tool_pipeline = getattr(self._tool_context, "_tool_pipeline", None)
        pipeline_config = getattr(tool_pipeline, "config", None)
        previous_pipeline_budget = snapshot.get("pipeline_tool_budget", _MISSING)
        if pipeline_config is not None and previous_pipeline_budget is not _MISSING:
            try:
                pipeline_config.tool_budget = previous_pipeline_budget
            except Exception:
                pass

    @staticmethod
    def _coerce_int_override(value: Any) -> Optional[int]:
        """Best-effort coercion for integer runtime overrides."""
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _accumulate_token_usage(self, response: CompletionResponse) -> None:
        """Accumulate token usage for evaluation tracking.

        Token usage is accumulated through chat_context._cumulative_token_usage.

        Args:
            response: Response from model
        """
        if response.usage:
            cum = self._chat_context._cumulative_token_usage
            cum["prompt_tokens"] += response.usage.get("prompt_tokens", 0)
            cum["completion_tokens"] += response.usage.get("completion_tokens", 0)
            cum["total_tokens"] += response.usage.get("total_tokens", 0)

            # Feed actual prompt_tokens back to ConversationController so that
            # get_context_metrics() uses real counts instead of char estimation.
            # Also persist cumulative token usage to ConversationStore (session analytics).
            prompt_tokens = response.usage.get("prompt_tokens", 0)
            if prompt_tokens > 0:
                try:
                    ctrl = self._chat_context.conversation
                    total_chars = sum(len(m.content) for m in ctrl.messages)
                    ctrl.record_actual_usage(prompt_tokens, total_chars)
                    # Persist to DB if a conversation store is wired
                    store = getattr(ctrl, "_conversation_store", None)
                    session_id = getattr(ctrl, "_session_id", None)
                    if store is not None and session_id:
                        store.update_session_token_usage(
                            session_id=session_id,
                            prompt_tokens=prompt_tokens,
                            completion_tokens=response.usage.get("completion_tokens", 0),
                        )
                except Exception:
                    pass  # Never break the hot path over metrics

            # Extended token fields: neutral usage first, transport
            # diagnostics (metadata["sandhi_usage"]) for the fields with no
            # neutral home yet, and the raw native body only as a legacy
            # fallback for non-typed providers. The native body must never be
            # load-bearing (foundations strategy F1) — typed transports work
            # identically with extensions absent.
            usage = response.usage or {}
            meta = getattr(response, "metadata", None) or {}
            sandhi_usage = meta.get("sandhi_usage") or {}
            raw = getattr(response, "raw_response", None)
            raw_usage = (raw.get("usage") or {}) if isinstance(raw, dict) else {}
            prompt_details = raw_usage.get("prompt_tokens_details", {}) or {}
            comp_details = raw_usage.get("completion_tokens_details", {}) or {}

            cum = self._chat_context._cumulative_token_usage
            cum["cached_tokens"] = cum.get("cached_tokens", 0) + (
                usage.get("cache_read_input_tokens", 0)
                or prompt_details.get("cached_tokens", 0)
                or raw_usage.get("prompt_cache_hit_tokens", 0)
                or raw_usage.get("cache_read_input_tokens", 0)
            )
            cum["cache_miss_tokens"] = cum.get("cache_miss_tokens", 0) + (
                sandhi_usage.get("cache_miss_tokens", 0)
                or raw_usage.get("prompt_cache_miss_tokens", 0)
            )
            cum["reasoning_tokens"] = cum.get("reasoning_tokens", 0) + (
                usage.get("reasoning_tokens", 0) or comp_details.get("reasoning_tokens", 0)
            )
            cum["cost_usd_micros"] = cum.get("cost_usd_micros", 0) + (
                sandhi_usage.get("cost_in_usd_ticks", 0) or raw_usage.get("cost_in_usd_ticks", 0)
            )

    async def _check_context_compaction(
        self,
        user_message: str,
        task_classification: Any,
    ) -> None:
        """Check and perform context compaction if needed.

        Args:
            user_message: Current query
            task_classification: Task complexity classification
        """
        if await self._check_lifecycle_context_compaction(user_message):
            return

        complexity_value = getattr(getattr(task_classification, "complexity", None), "value", None)
        if await self._check_context_service_compaction(task_complexity=complexity_value):
            return

        if self._chat_context._context_compactor:
            compaction_action = self._chat_context._context_compactor.check_and_compact(
                current_query=user_message,
                force=False,
                tool_call_count=self._tool_context.tool_calls_used,
                task_complexity=task_classification.complexity.value,
            )
            if compaction_action.action_taken:
                logger.info(
                    f"Compacted context: {compaction_action.messages_removed} messages removed, "
                    f"{compaction_action.tokens_freed} tokens freed"
                )

    async def _check_context_service_compaction(
        self, task_complexity: Optional[str] = None
    ) -> bool:
        """Use the canonical context service before legacy compactor fallback."""
        context_service = self._resolve_context_service()
        if context_service is None:
            return False

        result = await compact_context_if_recommended(
            context_service,
            strategy=self._resolve_context_compaction_strategy(),
            min_messages=6,
            task_complexity=task_complexity,
        )
        if not result.handled:
            return False

        if result.messages_removed > 0:
            logger.info(
                "ContextService compacted context before non-streaming turn: "
                "%s messages removed",
                result.messages_removed,
            )
        return True

    def _resolve_context_service(self) -> Any:
        """Return the context service wired to the runtime owner, if available."""
        orchestrator = self._resolve_orchestrator()
        if orchestrator is not None:
            context_service = getattr(orchestrator, "_context_service", None)
            if context_service is not None:
                return context_service
        return getattr(self._chat_context, "_context_service", None)

    def _resolve_context_compaction_strategy(self) -> str:
        """Return the configured context compaction strategy."""
        orchestrator = self._resolve_orchestrator()
        settings = (
            getattr(orchestrator, "settings", None)
            if orchestrator is not None
            else getattr(self._chat_context, "settings", None)
        )
        return str(getattr(settings, "context_compaction_strategy", "tiered") or "tiered")

    async def _check_lifecycle_context_compaction(self, user_message: str) -> bool:
        """Prefer service-owned context lifecycle compaction when available."""
        orchestrator = self._resolve_orchestrator()
        lifecycle = (
            getattr(orchestrator, "_context_lifecycle_service", None) if orchestrator else None
        )
        if lifecycle is None:
            return False
        after_agent_turn = getattr(lifecycle, "after_agent_turn", None)
        if not callable(after_agent_turn):
            return False

        runtime_context = self._root_runtime_context(orchestrator)
        result = after_agent_turn(
            runtime_context,
            messages=self._root_runtime_messages(orchestrator),
            min_messages=6,
        )
        if inspect.isawaitable(result):
            result = await result
        if isinstance(result, dict) and result.get("compacted"):
            logger.info(
                "Lifecycle compacted context before non-streaming turn: %s messages removed",
                result.get("messages_removed", 0),
            )
        return True

    @staticmethod
    def _root_runtime_context(orchestrator: Any) -> AgentRuntimeContext:
        existing = getattr(orchestrator, "_agent_runtime_context", None) or getattr(
            orchestrator,
            "agent_runtime_context",
            None,
        )
        if isinstance(existing, AgentRuntimeContext):
            return existing
        session_id = (
            getattr(orchestrator, "active_session_id", None)
            or getattr(orchestrator, "session_id", None)
            or getattr(orchestrator, "_memory_session_id", None)
            or "session_root"
        )
        return AgentRuntimeContext(
            agent_id=str(getattr(orchestrator, "agent_id", None) or "root_agent"),
            display_name=str(getattr(orchestrator, "display_name", None) or "Root Agent"),
            role=str(getattr(orchestrator, "role", None) or "manager"),
            session_id=str(session_id),
        )

    def _root_runtime_messages(self, orchestrator: Any) -> List[Any]:
        get_messages = getattr(orchestrator, "get_messages", None)
        if callable(get_messages):
            try:
                return list(get_messages() or [])
            except Exception as exc:
                logger.debug("Failed to collect non-streaming root messages: %s", exc)
        messages = getattr(self._chat_context, "messages", None)
        if messages is not None:
            return list(messages or [])
        controller = getattr(orchestrator, "conversation_controller", None) or getattr(
            orchestrator,
            "_conversation_controller",
            None,
        )
        return list(getattr(controller, "messages", None) or [])

    async def _ensure_complete_response(
        self,
        final_response: Optional[CompletionResponse],
        failure_context: ToolFailureContext,
    ) -> CompletionResponse:
        """Ensure we have a complete response.

        If the response is empty or None, use the response completer
        to generate a response or use a fallback.

        Args:
            final_response: Response from agentic loop (may be None)
            failure_context: Tool failure context for fallback messages

        Returns:
            CompletionResponse with content
        """
        if final_response is not None and final_response.content:
            return final_response

        # Use response completer to generate a response
        completion_result = await self._provider_context.response_completer.ensure_response(
            messages=self._chat_context.messages,
            model=self._provider_context.model,
            temperature=self._provider_context.temperature,
            max_tokens=self._provider_context.max_tokens,
            failure_context=(failure_context if failure_context.failed_tools else None),
        )

        if completion_result.content:
            from victor.agent.conversation.types import (
                MESSAGE_SOURCE_METADATA_KEY,
                MessageSource,
            )

            self._chat_context.add_message(
                "assistant",
                completion_result.content,
                metadata={MESSAGE_SOURCE_METADATA_KEY: MessageSource.AGENT_RESPONSE.value},
            )
            return CompletionResponse(
                content=completion_result.content,
                role="assistant",
                tool_calls=None,
            )

        # Last resort fallback
        fallback_content = (
            "I was unable to generate a complete response. " "Please try rephrasing your request."
        )
        if failure_context.failed_tools:
            fallback_content = (
                self._provider_context.response_completer.format_tool_failure_message(
                    failure_context
                )
            )

        from victor.agent.conversation.types import (
            MESSAGE_SOURCE_METADATA_KEY,
            MessageSource,
        )

        self._chat_context.add_message(
            "assistant",
            fallback_content,
            metadata={MESSAGE_SOURCE_METADATA_KEY: MessageSource.AGENT_RESPONSE.value},
        )
        return CompletionResponse(
            content=fallback_content,
            role="assistant",
            tool_calls=None,
        )

    # =====================================================================
    # Static Heuristics
    # =====================================================================

    @staticmethod
    def _is_question_only_scored(message: str) -> tuple:
        """Detect Q&A messages with confidence score (HDPO-inspired).

        Returns (is_qa: bool, confidence: float) where confidence indicates
        how certain the heuristic is. Low confidence (<0.7) suggests the
        edge LLM should be consulted for refinement.
        """
        from victor.framework.task.direct_response import (
            classify_direct_response_prompt,
            has_codebase_context,
        )

        direct_response = classify_direct_response_prompt(message)
        if direct_response.is_direct_response:
            return (True, direct_response.confidence)

        msg = message.strip().lower()
        codebase_context = has_codebase_context(message)

        # Short messages ending with ? — high confidence Q&A
        if len(msg) < 120 and msg.endswith("?"):
            action_words = (
                "fix",
                "create",
                "write",
                "edit",
                "refactor",
                "add",
                "implement",
                "update",
                "delete",
                "remove",
                "change",
                "modify",
                "build",
                "deploy",
                "run",
                "test",
                "debug",
                "install",
                "configure",
                "setup",
                "migrate",
            )
            words = set(msg.split())
            action_matches = words.intersection(action_words)
            if not action_matches:
                if codebase_context:
                    return (False, 0.5)
                return (True, 0.95)  # Strong Q&A signal
            else:
                # Question with action words — ambiguous
                return (False, 0.5)  # Low confidence

        # Explicit QA prefixes — high confidence
        qa_prefixes = (
            "what is",
            "what are",
            "what's",
            "who is",
            "who are",
            "how does",
            "how do",
            "how is",
            "how can",
            "why does",
            "why do",
            "why is",
            "when does",
            "when is",
            "explain",
            "describe",
            "define",
            "tell me about",
            "can you explain",
            "what does",
            "show me",
            "reply with",
            "answer",
            "say ",
        )
        for prefix in qa_prefixes:
            if msg.startswith(prefix):
                # Check if action words follow the QA prefix
                rest = msg[len(prefix) :].strip()
                action_words = (
                    "fix",
                    "create",
                    "write",
                    "edit",
                    "refactor",
                    "implement",
                    "update",
                    "delete",
                    "build",
                    "deploy",
                    "run",
                    "test",
                    "debug",
                )
                rest_words = set(rest.split())
                if rest_words.intersection(action_words):
                    return (True, 0.5)  # QA prefix but action words — ambiguous
                if codebase_context:
                    return (False, 0.5)
                return (True, 0.9)

        # Action-heavy messages — high confidence NOT Q&A
        action_indicators = (
            "fix",
            "create",
            "write",
            "edit",
            "refactor",
            "implement",
            "update",
            "delete",
            "build",
            "deploy",
            "run",
            "test",
            "debug",
            "install",
            "configure",
            "setup",
            "migrate",
        )
        words = set(msg.split())
        action_count = len(words.intersection(action_indicators))
        if action_count >= 2:
            return (False, 0.95)
        if action_count == 1:
            return (False, 0.8)

        # No strong signals — default to needs-tools with low confidence
        return (False, 0.6)

    @staticmethod
    def _is_question_only(message: str) -> bool:
        """Detect messages that are pure Q&A and unlikely to need tools.

        Backward-compatible wrapper around _is_question_only_scored().
        """
        is_qa, _ = TurnExecutor._is_question_only_scored(message)
        return is_qa


__all__ = [
    "TurnExecutor",
]
