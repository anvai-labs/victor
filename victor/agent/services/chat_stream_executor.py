# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Canonical service-owned executor for streaming chat turns.

This module owns the live streaming loop used by the canonical chat service
path. Deprecated pipeline surfaces should reuse this executor instead of
owning a parallel implementation.
"""

from __future__ import annotations

import hashlib
import logging

from victor.core.async_utils import aclosing_if_supported
from victor.agent.services.tool_selection_runtime import (
    hydrate_demand_tools,
)
import re
from types import SimpleNamespace
from dataclasses import dataclass
from typing import Any, AsyncIterator, List, Optional, Protocol, Tuple

from victor.agent.services.chat_runtime_services import (
    ChatRuntimeServices,
    TaskRequirementState,
    _StreamingCancelled,
    _raise_if_stream_cancelled,
)
from victor.agent.output_deduplicator import OutputDeduplicator
from victor.agent.safety import get_write_tool_names
from victor.agent.services.protocols.streaming_runtime import (
    StreamingExecutionRuntimeProtocol,
)
from victor.tools.core_tool_aliases import canonicalize_core_tool_name
from victor.tools.tool_names import get_canonical_name
from victor.core.completion_markers import (
    FILE_DONE_MARKER,
    strip_active_completion_markers,
)
from victor.framework.runtime_evaluation_policy import RuntimeEvaluationPolicy
from victor.providers.base import CompletionResponse, StreamChunk
from victor.agent.services.turn_execution_runtime import TurnDisposition, TurnResult
from victor.agent.streaming.tool_execution import ToolExecutionResult

logger = logging.getLogger(__name__)


def _cancelled_stream_chunk(services: ChatRuntimeServices, stream_ctx: Any) -> StreamChunk:
    """Close live cancellation state and return its terminal failure signal."""
    services.stream_lifecycle.finish()
    services.intelligence.record_outcome(
        success=False,
        quality_score=float(getattr(stream_ctx, "last_quality_score", 0.0) or 0.0),
        user_satisfied=False,
        completed=False,
    )
    return StreamChunk(
        content="\n\n[Cancelled by user]\n",
        is_final=True,
        metadata={"agentic_loop_success": False, "cancelled": True},
    )


def _tool_call_signatures(tool_calls: Optional[List[dict]]) -> set:
    """Stable per-turn tool-call signatures for spin detection.

    Mirrors ``TurnToolExecutionResult.tool_signatures`` (name + sorted, stringified args)
    so the streaming path can feed the SpinDetector the same signature scheme the
    single-turn primitive uses. Repeated identical signatures across turns (e.g. the same
    ``code_search(query=...)`` issued every iteration) are how a tool-call loop is detected
    early, instead of waiting for assistant content to become near-identical.
    """
    signatures: set = set()
    for tc in tool_calls or []:
        name = tc.get("name", "unknown")
        args = tc.get("arguments", {})
        if isinstance(args, dict):
            args_str = str([(k, str(v)) for k, v in sorted(args.items())])
        else:
            args_str = str(args)
        signatures.add(f"{name}:{args_str}")
    return signatures


def _count_productive_tools(tool_exec_result: Any) -> int:
    """Count tool calls that succeeded with output for progress/plateau accounting.

    The streaming ``ToolExecutionResult`` (victor/agent/streaming/tool_execution.py) exposes
    ``tool_results`` (a list of dicts) and ``tool_calls_executed`` — it does NOT have the
    ``successful_tool_count`` property of the single-turn ``TurnToolExecutionResult``. Derive
    the count from ``tool_results`` so a turn whose tools all failed/were blocked/returned
    nothing does not count as progress (and still gets nudged). When no per-result detail is
    available, fall back to whether any tool executed at all.
    """
    if not tool_exec_result:
        return 0
    results = getattr(tool_exec_result, "tool_results", None)
    if isinstance(results, list) and results:
        return sum(1 for r in results if isinstance(r, dict) and r.get("success"))
    return int(getattr(tool_exec_result, "tool_calls_executed", 0) or 0)


def _extract_required_files_from_prompt(user_message: str) -> List[str]:
    """Extract required file paths from the user message."""
    pattern = r'(?:^|[\s"\'`])(/[\w./\-]+\.\w+|\.{1,2}/[\w./\-]+\.\w+)'
    matches = re.findall(pattern, user_message)
    return list(set(matches))


def _extract_required_outputs_from_prompt(user_message: str) -> List[str]:
    """Extract required outputs from the user message."""
    return []


def _get_decision_service(orchestrator: object) -> object | None:
    """Get the LLM decision service from the container if available.

    Resolution is centralized in :func:`get_decision_service`; the service is
    registered per the ``decision_backend`` enum (FEP-0012), so absence (None)
    means "disabled". The legacy ``USE_LLM_DECISION_SERVICE`` gate was removed
    so a registered local-classifier is honored here regardless of that flag.
    """
    from victor.agent.services.protocols.decision_service import get_decision_service

    return get_decision_service(getattr(orchestrator, "_container", None))


@dataclass
class _ToolTurnOutcome:
    """Single-slot mutable holder for the tool-execution ACT sub-step's result.

    An async generator can't ``return`` a value, so ``_execute_tools_turn`` yields tool
    chunks and writes its ``ToolExecutionResult`` here for ``run()`` to read — both for
    loop control (``should_return``) and downstream evaluation (fulfillment / plateau /
    search-novelty). Part of FEP-0007 Phase 2's move toward a shared ``stream_turn()``.
    """

    result: Optional[ToolExecutionResult] = None


@dataclass
class _EmitDecision:
    """Single-slot mutable holder for the assistant-content emit / empty-response band.

    ``_emit_assistant_turn`` emits the assistant response (or handles tool-call-only and
    empty-response recovery), yielding content/recovery chunks. Because it yields, it can't
    return its outcome: the typed ``disposition`` derives the compatibility stop/retry views.
    The remaining fields propagate loop-local state the ACT mapper needs afterward —
    ``assistant_content_yielded`` (gates terminal-delivery fallback in continuation),
    ``prev_iteration_had_content`` (drives the inter-iteration blank-line separator), and
    ``tool_calls`` (empty-response recovery may replace it). Part of FEP-0007 Phase 2's move
    toward a shared ``stream_turn()``.
    """

    disposition: TurnDisposition = TurnDisposition.CONTINUE
    recovery_content: Optional[str] = None
    assistant_content_yielded: bool = False
    prev_iteration_had_content: bool = False
    tool_calls: Any = None

    @property
    def should_return(self) -> bool:
        return self.disposition.stops_run

    @property
    def should_continue(self) -> bool:
        return self.disposition is TurnDisposition.RECOVERY_RETRY


@dataclass
class StreamingActResult:
    """ACT-phase output of one streaming turn (FEP-0007 Addendum A, the streaming-ACT seam).

    ``execute_turn_streaming`` runs ONE turn's ACT sub-steps (provider response → assistant
    emit → tool execution) and yields that turn's ``StreamChunk``s. Because an async generator
    can't ``return`` a value, it writes its outcome here for the caller — the buffered shape the
    shared EVALUATE/DECIDE phases consume:

    - ``turn_result``: the ``TurnResult`` (same primitive ``turn_executor.execute_turn`` returns
      for the buffered path) built from this turn's content + tool results.
    - ``full_content`` / ``tool_calls`` / ``tools``: provider outputs (no-tool content is
      response-governed before publication), surfaced for the caller's evaluation
      (``tools`` is needed for empty-response recovery).
    - ``tool_exec_result``: the streaming ``ToolExecutionResult`` (``None`` when no tools ran).
    - ``garbage_detected`` / ``assistant_content_yielded``: per-turn signals the streaming
      EVALUATE band reads (garbage→force completion; whether visible content was streamed).
    The canonical ``turn_result.disposition`` carries source-classified recovery and stop
    outcomes. There is no separate mutable boolean stop channel.

    This carries ACT output only; the EVALUATE/DECIDE bands live in the shared ``AgenticLoop``
    phases (``run_streaming``), which is the canonical streaming loop — the legacy ``run()`` /
    ``_stream_turn`` were removed at the FEP-0007 cutover.
    """

    turn_result: Optional[TurnResult] = None
    full_content: str = ""
    tool_calls: Any = None
    tools: Any = None
    tool_exec_result: Any = None
    garbage_detected: bool = False
    assistant_content_yielded: bool = False
    # HIGH-confidence task-completion active signal (no pending tool calls). Surfaced onto the
    # TurnResult so the shared EVALUATE phase stops promptly — restoring the streaming loop's
    # immediate completion the unified loop would otherwise lose to the under-scoring
    # EnhancedCompletionEvaluator (FEP-0007 cutover tune-up).
    forced_completion: bool = False


class StreamingActPort(Protocol):
    """The streaming ACT seam consumed by the unified loop (FEP-0007 Addendum A).

    Symmetric with the buffered ACT (``turn_executor.execute_turn() -> TurnResult``): a single
    turn's provider + emit + tool sub-steps, yielding ``StreamChunk``s for live delivery while
    producing the same ``TurnResult`` the shared EVALUATE phase consumes. Lets ``AgenticLoop``
    own PERCEIVE/PLAN/EVALUATE/DECIDE once and differ only in the ACT call between modes.
    """

    def execute_turn_streaming(
        self,
        orch: Any,
        runtime_owner: Any,
        stream_ctx: Any,
        *,
        user_message: str,
        goals: Any,
        recovery: Any,
        create_recovery_context: Any,
        forced_task_completion: bool = False,
        result: StreamingActResult,
    ) -> AsyncIterator[StreamChunk]:
        """Run one turn's ACT, yielding chunks and writing the outcome to ``result``."""
        ...


class StreamingChatExecutor:
    """Canonical streaming chat executor bound to a runtime owner."""

    # Canonical write/mutation tool names sourced from the unified registry
    # (victor.tools.metadata_registry decorator-driven, with a static fallback in
    # victor.agent.safety). Resolved once at class-definition time and canonicalized
    # through core_tool_aliases + tool_names so legacy aliases (edit_files,
    # str_replace_editor, apply_patch, ...) collapse onto the canonical surface.
    _WRITE_MUTATION_TOOL_NAMES = frozenset(
        get_canonical_name(canonicalize_core_tool_name(name)) for name in get_write_tool_names()
    )

    # Read-only exploration tools that nudge a write-intent turn toward action.
    # These are the compact read surface; aliases (read_file, list_directory,
    # grep/rg, cat) collapse via canonicalize_core_tool_name below.
    _WRITE_EXPLORATION_TOOL_NAMES = frozenset(
        {
            get_canonical_name(canonicalize_core_tool_name(name))
            for name in (
                "cat",
                "code_search",
                "find",
                "grep",
                "ls",
                "read",
                "rg",
                "search",
                "shell",
            )
        }
    )
    _WRITE_ACTION_GUARD_THRESHOLD = 8
    _WRITE_ACTION_GUARD_ESCALATE_THRESHOLD = 12

    def __init__(
        self,
        runtime_owner: StreamingExecutionRuntimeProtocol,
        perception: Optional[Any] = None,
        fulfillment: Optional[Any] = None,
        confidence_monitor: Optional[Any] = None,
    ) -> None:
        self._runtime_owner = runtime_owner
        runtime_intelligence = self._get_runtime_intelligence()
        resolved_policy = getattr(runtime_intelligence, "evaluation_policy", None)
        if not isinstance(resolved_policy, RuntimeEvaluationPolicy):
            resolved_policy = RuntimeEvaluationPolicy()
        self._evaluation_policy = resolved_policy
        self._perception = perception
        self._fulfillment = fulfillment
        self._confidence_monitor = confidence_monitor
        self._last_tool_context: Optional[str] = None
        self._last_tools: Optional[Any] = None
        self._visible_output_deduplicator = OutputDeduplicator(min_block_length=40)
        self._prev_visible_content: str = ""

    @property
    def services(self) -> ChatRuntimeServices:
        """Return the runtime's explicitly bound service capabilities."""
        return self._runtime_owner.services

    def _get_runtime_intelligence(self) -> Any:
        """Resolve the optional learned runtime through its typed capability."""
        intelligence = getattr(self.services, "intelligence", None)
        if intelligence is None:
            return None
        return intelligence.executor_runtime()

    @staticmethod
    def _normalize_visible_content_key(content: str) -> str:
        """Return a stable comparison key for visible content suppression."""
        return re.sub(r"\s+", " ", content).strip().lower()

    @staticmethod
    def _normalize_visible_candidate(content: str, *, user_message: str = "") -> str:
        """Normalize raw model output into a display candidate without deduping."""
        if not content or not content.strip():
            return ""

        display_content = strip_active_completion_markers(content)
        if not display_content or not display_content.strip():
            return ""

        if user_message:
            from victor.framework.task.direct_response import (
                normalize_direct_response_output,
            )

            display_content = normalize_direct_response_output(
                user_message, display_content
            ).strip()
            if not display_content:
                return ""

        return display_content

    def _prepare_visible_content(self, content: str, *, user_message: str = "") -> str:
        """Normalize model output for display and conversation history."""
        display_content = self._normalize_visible_candidate(content, user_message=user_message)
        if not display_content:
            return ""

        display_content = self._visible_output_deduplicator.process(display_content).strip()
        if not display_content:
            return ""

        normalized_key = self._normalize_visible_content_key(display_content)
        if not normalized_key:
            return ""
        if normalized_key == self._prev_visible_content:
            logger.info("Suppressing repeated visible output block")
            return ""

        self._prev_visible_content = normalized_key
        return display_content

    def _resolve_terminal_visible_output(
        self,
        stream_ctx: Any,
        *,
        full_content: str,
        user_message: str,
    ) -> tuple[str, Optional[str]]:
        """Pick the best visible terminal response from provider and summary fallbacks."""
        candidates = (
            ("provider_response", full_content),
            ("completion_summary", self.services.completion.terminal_summary()),
            ("compaction_summary", getattr(stream_ctx, "compaction_summary", "")),
        )

        for source, candidate in candidates:
            normalized = self._normalize_visible_candidate(candidate, user_message=user_message)
            if not normalized:
                continue

            sanitized = self.services.delivery.sanitize(normalized, optional=True)
            if isinstance(sanitized, str) and sanitized.strip():
                return sanitized.strip(), source

            plain_text = self.services.delivery.strip_markup(normalized, optional=True)
            if isinstance(plain_text, str) and plain_text.strip():
                return plain_text.strip(), source

        return "", None

    async def _build_terminal_delivery_chunk(
        self,
        orch: Any,
        stream_ctx: Any,
        *,
        full_content: str,
        user_message: str,
    ) -> StreamChunk:
        """Publish already-governed provider text or govern a distinct summary fallback."""
        terminal_content, content_source = self._resolve_terminal_visible_output(
            stream_ctx,
            full_content=full_content,
            user_message=user_message,
        )

        if terminal_content:
            # ACT already governed provider text; summaries are distinct candidates.
            if content_source != "provider_response":
                terminal_content = await self.services.governance.govern_response(terminal_content)
            _raise_if_stream_cancelled(self.services)

            from victor.agent.conversation.types import (
                MESSAGE_SOURCE_METADATA_KEY,
                MessageSource,
            )

            orch.add_message(
                "assistant",
                terminal_content,
                persist_synchronously=True,
                metadata={MESSAGE_SOURCE_METADATA_KEY: MessageSource.AGENT_RESPONSE.value},
            )
            self._append_stream_event(
                stream_ctx,
                "provider_status_events",
                {
                    "source": "streaming_executor",
                    "kind": "final_chunk_emitted",
                    "response_source": content_source,
                    "iteration": getattr(stream_ctx, "total_iterations", 0),
                    "content_length": len(terminal_content),
                },
            )
            if content_source and content_source != "provider_response":
                logger.warning(
                    "Terminal response content recovered from %s fallback (provider output was not visible)",
                    content_source,
                )
            return self.services.delivery.content_chunk(terminal_content, is_final=True)

        self._append_stream_event(
            stream_ctx,
            "provider_status_events",
            {
                "source": "streaming_executor",
                "kind": "final_marker_emitted",
                "iteration": getattr(stream_ctx, "total_iterations", 0),
            },
        )
        logger.warning(
            "No visible terminal assistant content available; emitting final marker only"
        )
        return self.services.delivery.final_marker_chunk()

    @staticmethod
    def _resolve_provider_identity(orch: Any) -> tuple[str, str]:
        """Resolve provider and model names for model-aware continuation policy."""
        provider = getattr(orch, "provider", None)
        provider_name = getattr(provider, "name", None) or getattr(orch, "provider_name", "")
        model = getattr(orch, "model", "") or ""
        return str(provider_name or "unknown"), str(model or "unknown")

    def _build_post_compaction_continuation_prompt(
        self,
        orch: Any,
        stream_ctx: Any,
    ) -> str:
        """Build a model-aware post-compaction continuation prompt."""
        from victor.agent.compaction_continuation_bonus import get_compaction_bonus
        from victor.agent.context_reminder_templates import get_post_compaction_reminder

        task_type = (
            getattr(stream_ctx, "coarse_task_type", None)
            or getattr(getattr(stream_ctx, "unified_task_type", None), "value", None)
            or "default"
        )
        provider_name, model_name = self._resolve_provider_identity(orch)
        messages_removed = int(getattr(stream_ctx, "compaction_message_removed_count", 0) or 0)
        reminder = get_post_compaction_reminder(
            task_type=task_type,
            compaction_summary=str(getattr(stream_ctx, "compaction_summary", "") or ""),
            messages_removed=messages_removed,
        )

        try:
            compaction_bonus = get_compaction_bonus().get_bonus(
                provider=provider_name,
                model=model_name,
                compaction_occurred=True,
                messages_removed=messages_removed,
            )
        except Exception:
            compaction_bonus = 1

        ledger_text = ""
        if hasattr(stream_ctx, "build_continuation_ledger"):
            ledger_text = stream_ctx.build_continuation_ledger(
                max_events=max(3, min(6, compaction_bonus + 2)),
                max_plan_steps=max(3, min(5, compaction_bonus + 1)),
                max_chars=900 if compaction_bonus >= 3 else 600,
            )

        parts = ["[CONTEXT COMPACTED]", reminder]
        if ledger_text:
            parts.append("Continuation ledger:\n" + ledger_text)
        if compaction_bonus >= 3:
            parts.append(
                "Resume from the recorded plan before you summarize or ask the user for input."
            )
        else:
            parts.append(
                "Continue from the recorded plan and use tools if more evidence is needed."
            )
        return " ".join(part for part in parts[:2]) + (
            "\n\n" + "\n\n".join(parts[2:]) if len(parts) > 2 else ""
        )

    @staticmethod
    def _append_stream_event(stream_ctx: Any, field_name: str, event: dict[str, Any]) -> None:
        """Append a structured event list onto the mutable stream context."""
        events = getattr(stream_ctx, field_name, None)
        if not isinstance(events, list):
            events = []
            setattr(stream_ctx, field_name, events)
        events.append(event)

    @staticmethod
    def _tool_name_value(tool_name: Any) -> str:
        return str(tool_name or "").split(".")[-1].strip().lower()

    def _is_write_action_turn(self, stream_ctx: Any) -> bool:
        """Return True for turns where continued read-only exploration is risky."""
        if not bool(getattr(stream_ctx, "is_action_task", False)):
            return False

        intent = self.services.planning.current_intent()
        intent_value = str(getattr(intent, "value", intent) or "").lower()
        if intent_value in {"write_allowed", "edit", "write"}:
            return True

        task_type = str(getattr(getattr(stream_ctx, "unified_task_type", None), "value", "") or "")
        coarse_type = str(getattr(stream_ctx, "coarse_task_type", "") or "")
        task_text = f"{task_type} {coarse_type}".lower()
        return any(token in task_text for token in ("edit", "write", "code", "coding"))

    def _has_mutation_tool_executed(self, stream_ctx: Any) -> bool:
        tool_names = {
            get_canonical_name(canonicalize_core_tool_name(self._tool_name_value(name)))
            for name in (getattr(stream_ctx, "executed_tool_names", set()) or set())
            if name
        }
        return bool(tool_names & self._WRITE_MUTATION_TOOL_NAMES)

    def _maybe_inject_write_action_guard(
        self,
        orch: Any,
        stream_ctx: Any,
        *,
        user_message: str,
        tool_calls_used: int,
    ) -> bool:
        """Nudge write-intent turns out of repeated read-only exploration."""
        if not self._is_write_action_turn(stream_ctx):
            return False
        if self._has_mutation_tool_executed(stream_ctx):
            return False

        executed_names = {
            get_canonical_name(canonicalize_core_tool_name(self._tool_name_value(name)))
            for name in (getattr(stream_ctx, "executed_tool_names", set()) or set())
            if name
        }
        if not executed_names:
            return False
        if not (executed_names & self._WRITE_EXPLORATION_TOOL_NAMES):
            return False

        threshold = self._WRITE_ACTION_GUARD_THRESHOLD
        level = "initial"
        if tool_calls_used >= self._WRITE_ACTION_GUARD_ESCALATE_THRESHOLD:
            threshold = self._WRITE_ACTION_GUARD_ESCALATE_THRESHOLD
            level = "escalated"
        if tool_calls_used < threshold:
            return False

        marker = f"_write_action_guard_{level}_injected"
        if getattr(stream_ctx, marker, False):
            return False
        setattr(stream_ctx, marker, True)

        from victor.agent.conversation.types import (
            MESSAGE_SOURCE_METADATA_KEY,
            MessageSource,
        )

        if level == "escalated":
            message = (
                "[SYSTEM: This is still an edit/action task. You have already used "
                f"{tool_calls_used} tool call(s) and no mutation tool has run. Do not call "
                "read, grep, code_search, or shell again unless it directly executes the edit. "
                "Next response must either call edit/write/apply_patch with a concrete change, "
                "or return a concise blocker explaining the exact file/location that prevents "
                "a safe edit.]"
            )
        else:
            message = (
                "[SYSTEM: This is an edit/action task, and enough repository context has been "
                f"gathered ({tool_calls_used} tool call(s)) without any mutation. Stop broad "
                "read-only exploration. On the next turn, apply the smallest concrete code "
                "change with an edit/write/apply_patch-style tool, or state the precise blocker "
                "that makes a safe edit impossible.]"
            )

        orch.add_message(
            "system",
            message,
            metadata={MESSAGE_SOURCE_METADATA_KEY: MessageSource.SYSTEM_INJECTED.value},
        )
        self._append_stream_event(
            stream_ctx,
            "provider_status_events",
            {
                "source": "streaming_executor",
                "kind": "write_action_guard_injected",
                "level": level,
                "tool_calls_used": tool_calls_used,
                "executed_tool_names": sorted(executed_names),
                "user_message_preview": user_message[:160],
            },
        )
        logger.info(
            "Injected write-action guard after %s read-only tool calls (level=%s, tools=%s)",
            tool_calls_used,
            level,
            sorted(executed_names),
        )
        return True

    @staticmethod
    def _normalize_optional_text(value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(getattr(value, "value", value)).strip()
        return text or None

    def _record_confidence_early_stop(self, stream_ctx: Any) -> None:
        """Persist an early-stop signal into the shared degradation event path."""
        task_type = self._normalize_optional_text(
            getattr(getattr(stream_ctx, "unified_task_type", None), "value", None)
            or getattr(stream_ctx, "coarse_task_type", None)
        )
        event = {
            "source": "streaming_confidence",
            "kind": "confidence_early_stop",
            "task_type": task_type,
            "iteration": getattr(stream_ctx, "total_iterations", 0),
            "pre_degraded": False,
            "post_degraded": False,
            "recovered": False,
            "adaptation_cost": 0.0,
            "degradation_reasons": ["confidence_threshold_reached"],
            "total_tokens": getattr(stream_ctx, "total_tokens", 0.0),
        }
        self._append_stream_event(stream_ctx, "degradation_events", event)

    def _record_recovery_action(self, stream_ctx: Any, recovery_action: Any) -> None:
        """Persist structured recovery metadata for stream teardown normalization."""
        action = self._normalize_optional_text(getattr(recovery_action, "action", None))
        if action is None or action == "continue":
            return

        task_type = self._normalize_optional_text(
            getattr(getattr(stream_ctx, "unified_task_type", None), "value", None)
            or getattr(stream_ctx, "coarse_task_type", None)
        )
        confidence = getattr(recovery_action, "confidence", None)
        try:
            confidence_value = float(confidence) if confidence is not None else None
        except (TypeError, ValueError):
            confidence_value = None
        event = {
            "source": "streaming_recovery",
            "kind": "recovery_action",
            "action": action,
            "failure_type": self._normalize_optional_text(
                getattr(recovery_action, "failure_type", None)
            ),
            "task_type": task_type,
            "iteration": getattr(stream_ctx, "total_iterations", 0),
            "reason": self._normalize_optional_text(getattr(recovery_action, "reason", None)),
            "strategy_name": self._normalize_optional_text(
                getattr(recovery_action, "strategy_name", None)
            ),
            "confidence": confidence_value,
            "fallback_provider": self._normalize_optional_text(
                getattr(recovery_action, "fallback_provider", None)
            ),
            "fallback_model": self._normalize_optional_text(
                getattr(recovery_action, "fallback_model", None)
            ),
        }
        self._append_stream_event(stream_ctx, "recovery_events", event)

    def _reset_streaming_turn_state(self, orch: Any) -> None:
        """Reset per-turn state shared across streaming iterations."""
        self._last_tool_context = None
        self._last_tools = None

        if hasattr(orch, "tool_calls_used"):
            orch.tool_calls_used = 0

        self.services.tool_calls.reset()
        self.services.completion.reset()

    @staticmethod
    def _serialize_conversation_message(message: Any) -> dict[str, Any] | None:
        """Normalize a conversation message into a mapping for perception services."""
        if isinstance(message, dict):
            payload = dict(message)
        elif hasattr(message, "model_dump"):
            payload = message.model_dump()
        else:
            role = getattr(message, "role", None)
            content = getattr(message, "content", None)
            if role is None and content is None:
                return None
            payload = {"role": role, "content": content}

        if not isinstance(payload, dict):
            return None
        if payload.get("role") is None and payload.get("content") is None:
            return None
        return payload

    def _get_conversation_history(
        self,
        runtime_owner: Any,
        orch: Any,
        user_message: str,
    ) -> Optional[List[dict[str, Any]]]:
        """Return serialized conversation history excluding the current user turn."""
        messages = None
        assembled_getter = getattr(orch, "get_assembled_messages", None)
        if callable(assembled_getter):
            try:
                messages = assembled_getter(current_query=user_message)
            except Exception as exc:
                logger.debug("Falling back to raw conversation history: %s", exc)

        if messages is None:
            chat_context = getattr(runtime_owner, "_chat_context", None)
            messages = getattr(chat_context, "messages", None)
        if messages is None:
            messages = getattr(orch, "messages", None)
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

    async def _get_tools_cached(
        self,
        context_msg: str,
        goals: Any,
        planned_tools: Any = None,
    ) -> Any:
        """Select tools with per-turn caching."""
        if self._last_tool_context == context_msg and self._last_tools is not None:
            return self._last_tools

        tools = await self.services.planning.select_tools(
            context_msg,
            goals,
            planned_tools=planned_tools,
        )
        if tools is not None:
            self._last_tool_context = context_msg
            self._last_tools = tools
        return tools

    @staticmethod
    async def _extract_task_requirements(session: TaskRequirementState, user_message: str) -> None:
        """Update the session owner's requirements and emit their state event."""
        session.required_files = _extract_required_files_from_prompt(user_message)
        session.required_outputs = _extract_required_outputs_from_prompt(user_message)
        session.read_files.clear()
        session.all_files_read_nudge_sent = False
        logger.debug(
            "Task requirements extracted - files: %s, outputs: %s",
            session.required_files,
            session.required_outputs,
        )
        if session.required_files or session.required_outputs:
            from victor.core.events import get_observability_bus

            event_bus = get_observability_bus()
            await event_bus.emit(
                topic="state.task.requirements_extracted",
                data={
                    "required_files": session.required_files,
                    "required_outputs": session.required_outputs,
                    "file_count": len(session.required_files),
                    "output_count": len(session.required_outputs),
                    "category": "state",
                },
            )

    def _apply_run_guidance(
        self,
        orch: Any,
        stream_ctx: Any,
        user_message: str,
        max_exploration_iterations: int,
    ) -> None:
        """Apply intent guard + task-type guidance + action-task guidance for one run.

        A cohesive piece of run()'s preamble (FEP-0007 Phase 2 decomposition). Mutates orch
        (guards + guidance messages); yields nothing.
        """
        self.services.planning.apply_intent_guard(user_message)

        if stream_ctx.is_analysis_task and stream_ctx.unified_task_type.value in (
            "edit",
            "create",
        ):
            logger.info(
                "Compound task detected (analysis+%s): unified_tracker will use appropriate "
                "exploration limits",
                stream_ctx.unified_task_type.value,
            )

        logger.info(
            "Task type classification: coarse=%s, unified=%s, is_analysis=%s, is_action=%s",
            stream_ctx.coarse_task_type,
            stream_ctx.unified_task_type.value,
            stream_ctx.is_analysis_task,
            stream_ctx.is_action_task,
        )

        self.services.planning.apply_task_guidance(
            user_message=user_message,
            unified_task_type=stream_ctx.unified_task_type,
            is_analysis_task=stream_ctx.is_analysis_task,
            is_action_task=stream_ctx.is_action_task,
            needs_execution=stream_ctx.needs_execution,
            max_exploration_iterations=max_exploration_iterations,
        )

        if stream_ctx.is_action_task:
            logger.info(
                "Detected action-oriented task - allowing up to %s exploration iterations",
                max_exploration_iterations,
            )

            from victor.agent.conversation.history_metadata import (
                build_internal_history_metadata,
            )
            from victor.agent.conversation.types import MessageSource

            _guidance_meta = build_internal_history_metadata(
                "action_guidance", source=MessageSource.AGENT_GUIDANCE
            )
            if stream_ctx.needs_execution:
                orch.add_message(
                    "user",
                    "[ACTION-GUIDANCE: This is an action-oriented task requiring execution. "
                    "Follow this workflow: "
                    "1. CREATE the file/script with write or edit "
                    "2. EXECUTE it immediately with shell (don't skip this step!) "
                    "3. SHOW the output to the user. "
                    "Minimize exploration and proceed directly to create→execute→show results.]",
                    metadata=_guidance_meta,
                )
            else:
                orch.add_message(
                    "user",
                    "[ACTION-GUIDANCE: This is an action-oriented task (create/write/build). "
                    "Minimize exploration and proceed directly to creating what was requested. "
                    "Only explore if absolutely necessary to complete the task.]",
                    metadata=_guidance_meta,
                )

    def _initialize_task_intent(self, stream_ctx: Any, user_message: str) -> Any:
        """Seed task intent on the stream context and return the inferred goals.

        Final cohesive piece of run()'s preamble (FEP-0007 Phase 2 decomposition). Mutates
        stream_ctx (task intent, plan steps, task-start event); yields nothing. Returns the
        inferred ``goals`` so the loop can use them for tool planning.
        """
        goals = self.services.planning.infer_goals(user_message)
        if hasattr(stream_ctx, "set_task_intent"):
            stream_ctx.set_task_intent(user_message)
        if hasattr(stream_ctx, "extend_plan_steps"):
            stream_ctx.extend_plan_steps(goals)
        if hasattr(stream_ctx, "record_intent_event"):
            stream_ctx.record_intent_event(
                "task_start",
                f"task start ({stream_ctx.coarse_task_type})",
                task_type=stream_ctx.coarse_task_type,
            )
        return goals

    async def _stream_provider_turn(
        self,
        orch: Any,
        runtime_owner: Any,
        stream_ctx: Any,
        goals: Any,
    ) -> tuple[Any, str, Any, bool]:
        """Resolve this turn's tools and stream the provider response.

        The ACT provider sub-step of run()'s per-turn body (FEP-0007 Phase 2, toward a
        shared stream_turn() boundary). Plans tools, resolves the active tool set, builds
        provider kwargs, and streams the provider response for ONE turn. Sets
        stream_ctx.planned_tools and returns the ``(tools, full_content, tool_calls,
        garbage_detected)`` tuple (tools is needed by run() for empty-response recovery).
        Token streaming happens inside ``_stream_provider_response``; this helper yields
        nothing.
        """
        planned_tools = None
        if goals:
            available_inputs = ["query"]
            if orch.observed_files:
                available_inputs.append("file_contents")
            planned_tools = self.services.planning.plan_tools(goals, available_inputs)
        stream_ctx.planned_tools = planned_tools

        # Stage 1 — demand hydration (FEP-0034). Must precede BOTH the Q&A
        # branch and the session-tool freeze below: turn 1 can be
        # Q&A-classified while turn 2 carries the mention, and the frozen set
        # is computed on the first get_session_tools() call — a tool that
        # misses that first registration never reaches this session's schema.
        hydrate_demand_tools(orch, getattr(stream_ctx, "context_msg", "") or "")

        if getattr(stream_ctx, "is_qa_task", False):
            tools = None
        elif orch.get_session_tools() is not None:
            tools = orch.get_session_tools()
        else:
            tools = await self._get_tools_cached(
                stream_ctx.context_msg,
                goals,
                planned_tools=planned_tools,
            )

        provider_kwargs = dict(getattr(stream_ctx, "provider_kwargs", {}) or {})
        if orch.thinking or provider_kwargs.get("execution_mode") == "escalated_single_agent":
            provider_kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": 10000,
            }

        full_content, tool_calls, _, garbage_detected = (
            await runtime_owner._stream_provider_response(
                tools=tools,
                provider_kwargs=provider_kwargs,
                stream_ctx=stream_ctx,
            )
        )
        return tools, full_content, tool_calls, garbage_detected

    async def _execute_tools_turn(
        self,
        orch: Any,
        runtime_owner: Any,
        stream_ctx: Any,
        *,
        user_message: str,
        tool_calls: Any,
        full_content: str,
        result_holder: _ToolTurnOutcome,
    ) -> AsyncIterator[StreamChunk]:
        """Execute this turn's tool calls, streaming chunks; record the result on the holder.

        The ACT tool-execution sub-step of run()'s per-turn body (FEP-0007 Phase 2, toward a
        shared stream_turn() boundary). Lazily builds the tool-execution handler, streams (or
        buffers) tool-result chunks, then applies post-execution bookkeeping (tool-call
        accounting + write-action guard). Because an async generator can't return a value, the
        produced ``ToolExecutionResult`` is written to ``result_holder.result`` for run() to read
        (``should_return`` loop control + downstream fulfillment/plateau/novelty evaluation).
        """
        _raise_if_stream_cancelled(self.services)
        if not runtime_owner._tool_execution_handler:
            from victor.agent.streaming import create_tool_execution_handler

            runtime_owner._tool_execution_handler = create_tool_execution_handler(orch)

        runtime_owner._tool_execution_handler.update_observed_files(
            set(orch.observed_files) if orch.observed_files else set()
        )

        tool_exec_result = None
        accounted = False

        def account_completed_tools() -> None:
            nonlocal accounted
            if accounted or tool_exec_result is None:
                return
            orch.tool_calls_used += tool_exec_result.tool_calls_executed
            stream_ctx.tool_calls_used = orch.tool_calls_used
            record_count = getattr(stream_ctx, "record_iteration_tool_count", None)
            if callable(record_count):
                record_count(tool_exec_result.tool_calls_executed)
            result_holder.result = tool_exec_result
            accounted = True

        try:
            if hasattr(runtime_owner._tool_execution_handler, "execute_tools_streaming"):
                from victor.agent.streaming.tool_execution import (
                    ToolExecutionResult,
                )

                tool_exec_result = ToolExecutionResult()
                async with aclosing_if_supported(
                    runtime_owner._tool_execution_handler.execute_tools_streaming(
                        stream_ctx=stream_ctx,
                        tool_calls=tool_calls,
                        user_message=user_message,
                        full_content=full_content,
                        tool_calls_used=orch.tool_calls_used,
                        tool_budget=orch.tool_budget,
                        result=tool_exec_result,
                    )
                ) as stream:
                    iterator = aiter(stream)
                    while True:
                        _raise_if_stream_cancelled(self.services)
                        try:
                            chunk = await anext(iterator)
                        except StopAsyncIteration:
                            break
                        if self.services.stream_lifecycle.is_cancelled():
                            break
                        yield chunk
            else:
                _raise_if_stream_cancelled(self.services)
                tool_exec_result = await runtime_owner._tool_execution_handler.execute_tools(
                    stream_ctx=stream_ctx,
                    tool_calls=tool_calls,
                    user_message=user_message,
                    full_content=full_content,
                    tool_calls_used=orch.tool_calls_used,
                    tool_budget=orch.tool_budget,
                )
                if not self.services.stream_lifecycle.is_cancelled():
                    for chunk in tool_exec_result.chunks:
                        yield chunk

            cancelled = self.services.stream_lifecycle.is_cancelled()
            if cancelled and tool_exec_result.tool_calls_executed <= 0:
                raise _StreamingCancelled
            account_completed_tools()
            if cancelled:
                raise _StreamingCancelled

            self._maybe_inject_write_action_guard(
                orch,
                stream_ctx,
                user_message=user_message,
                tool_calls_used=orch.tool_calls_used,
            )
        finally:
            if (
                tool_exec_result is not None
                and tool_exec_result.tool_calls_executed > 0
                and not accounted
            ):
                account_completed_tools()

    async def _emit_assistant_turn(
        self,
        orch: Any,
        runtime_owner: Any,
        stream_ctx: Any,
        *,
        recovery: Any,
        create_recovery_context: Any,
        full_content: str,
        tool_calls: Any,
        forced_task_completion: bool,
        user_message: str,
        tools: Any,
        decision: _EmitDecision,
    ) -> AsyncIterator[StreamChunk]:
        """Emit already-governed provider content or handle tools/empty-response recovery.

        The sole production caller, execute_turn_streaming, governs every nonempty
        provider narrative before entering this private emitter. Newly produced
        recovery and summary candidates are governed at their own source boundary.

        The emit + empty-response-recovery band of run()'s per-turn body (FEP-0007 Phase 2, toward
        a shared stream_turn() boundary). Sanitizes and yields visible content (a forced-completion
        turn with no pending tool calls is the final emit), records a tool-call-only assistant
        message, or runs the empty-response recovery ladder. Because it yields, the loop control and
        carried loop-local state are written to ``decision`` (``should_return`` / ``should_continue``,
        plus ``assistant_content_yielded`` / ``prev_iteration_had_content`` / possibly-replaced
        ``tool_calls``) for run() to read.
        """
        decision.tool_calls = tool_calls

        if full_content:
            visible_content = self._prepare_visible_content(
                full_content,
                user_message=getattr(stream_ctx, "user_message", ""),
            )
            if visible_content:
                decision.prev_iteration_had_content = True
            sanitized = self.services.delivery.sanitize(visible_content)
            if sanitized:
                from victor.agent.conversation.types import (
                    MESSAGE_SOURCE_METADATA_KEY,
                    MessageSource,
                )

                _is_final_emit = forced_task_completion and not tool_calls
                orch.add_message(
                    "assistant",
                    sanitized,
                    tool_calls=tool_calls,
                    persist_synchronously=_is_final_emit,
                    metadata={MESSAGE_SOURCE_METADATA_KEY: MessageSource.AGENT_RESPONSE.value},
                )
                decision.assistant_content_yielded = True
                yield self.services.delivery.content_chunk(
                    sanitized,
                    is_final=_is_final_emit,
                )
                if _is_final_emit:
                    decision.disposition = TurnDisposition.COMPLETION_CANDIDATE
                    return
            else:
                plain_text = self.services.delivery.strip_markup(
                    visible_content
                    or self._normalize_visible_candidate(
                        full_content,
                        user_message=getattr(stream_ctx, "user_message", ""),
                    )
                )
                if plain_text:
                    from victor.agent.conversation.types import (
                        MESSAGE_SOURCE_METADATA_KEY,
                        MessageSource,
                    )

                    _is_final_emit = forced_task_completion and not tool_calls
                    orch.add_message(
                        "assistant",
                        plain_text,
                        tool_calls=tool_calls,
                        persist_synchronously=_is_final_emit,
                        metadata={MESSAGE_SOURCE_METADATA_KEY: MessageSource.AGENT_RESPONSE.value},
                    )
                    decision.assistant_content_yielded = True
                    yield self.services.delivery.content_chunk(
                        plain_text,
                        is_final=_is_final_emit,
                    )
                    if _is_final_emit:
                        decision.disposition = TurnDisposition.COMPLETION_CANDIDATE
                        return
                elif forced_task_completion and not tool_calls:
                    final_chunk = await self._build_terminal_delivery_chunk(
                        orch,
                        stream_ctx,
                        full_content=full_content,
                        user_message=user_message,
                    )
                    decision.recovery_content = final_chunk.content
                    yield final_chunk
                    decision.disposition = TurnDisposition.COMPLETION_CANDIDATE
                    return
        elif tool_calls:
            from victor.agent.conversation.types import (
                MESSAGE_SOURCE_METADATA_KEY,
                MessageSource,
            )

            orch.add_message(
                "assistant",
                "",
                tool_calls=tool_calls,
                metadata={MESSAGE_SOURCE_METADATA_KEY: MessageSource.AGENT_RESPONSE.value},
            )
        else:
            recovery_ctx = create_recovery_context(stream_ctx)
            final_chunk = recovery.check_natural_completion(
                recovery_ctx, has_tool_calls=False, content_length=0
            )
            if final_chunk:
                final_chunk.content = await self.services.governance.govern_response(
                    final_chunk.content
                )
                decision.recovery_content = final_chunk.content
                yield final_chunk
                decision.disposition = TurnDisposition.COMPLETION_CANDIDATE
                return

            logger.warning("Model returned empty response - attempting aggressive recovery")

            recovery_ctx = create_recovery_context(stream_ctx)
            recovery_chunk, should_force = recovery.handle_empty_response(recovery_ctx)
            _ = should_force
            if recovery_chunk:
                yield recovery_chunk
                decision.disposition = TurnDisposition.RECOVERY_RETRY
                return

            recovery_success, recovered_tool_calls, final_chunk = (
                await runtime_owner._handle_empty_response_recovery(stream_ctx, tools)
            )

            if recovery_success:
                if bool(final_chunk) == bool(recovered_tool_calls):
                    raise ValueError("Successful recovery must produce either content or tools")
                if final_chunk:
                    decision.recovery_content = final_chunk.content
                    yield final_chunk
                    decision.disposition = TurnDisposition.COMPLETION_CANDIDATE
                    return
                if recovered_tool_calls:
                    decision.tool_calls = recovered_tool_calls
                    logger.info(
                        "Recovery produced %s tool call(s) - continuing main loop",
                        len(recovered_tool_calls),
                    )
            else:
                recovery_ctx = create_recovery_context(stream_ctx)
                fallback_msg = await self.services.governance.govern_response(
                    recovery.get_recovery_fallback_message(recovery_ctx)
                )
                self.services.intelligence.record_outcome(
                    success=False,
                    quality_score=0.3,
                    user_satisfied=False,
                    completed=False,
                )
                decision.recovery_content = fallback_msg
                yield self.services.delivery.content_chunk(fallback_msg, is_final=True)
                decision.disposition = TurnDisposition.RECOVERY_EXHAUSTED
                return

    def _detect_high_confidence_completion(
        self,
        stream_ctx: Any,
        *,
        full_content: str,
        tool_calls: Any,
    ) -> bool:
        """Return True when this turn's answer is a HIGH-confidence completion with no pending tools.

        Runs the completion capability over the assistant content; on a HIGH-confidence
        active signal with no outstanding tool calls it persists a VICTOR_SUMMARY for any next-turn
        context and returns True. This is the streaming loop's prompt-completion signal (formerly in
        ``_detect_task_completion_and_mentions``) — the unified loop surfaces it onto the TurnResult
        so EVALUATE stops immediately instead of restating to the iteration cap (FEP-0007 tune-up).
        """
        completed = self.services.completion.detect_high_confidence(
            full_content,
            has_pending_tools=bool(tool_calls),
        )
        if not completed:
            return False

        logger.info(
            "Task completion: HIGH confidence detected (active signal), "
            "forcing completion NOW (immediate stop, skip continuation)"
        )
        stream_ctx.force_completion = True
        stream_ctx.skip_continuation = True
        self.services.conversation.persist_terminal_summary(
            self.services.completion.terminal_summary()
        )
        return True

    @staticmethod
    def _build_streaming_turn_result(
        stream_ctx: Any,
        *,
        full_content: str,
        tool_calls: Any,
        tool_exec_result: Optional[ToolExecutionResult],
        forced_completion: bool = False,
        disposition: TurnDisposition = TurnDisposition.CONTINUE,
    ) -> TurnResult:
        """Assemble a ``TurnResult`` from one streaming turn's ACT output.

        Produces the same primitive ``turn_executor.execute_turn`` returns for the buffered
        path, so the shared EVALUATE phase is mode-agnostic: the streamed ``full_content`` /
        ``tool_calls`` become the ``CompletionResponse``, and the streaming
        ``ToolExecutionResult.tool_results`` (if any) become ``TurnResult.tool_results``. A
        ``forced_completion`` HIGH-confidence signal is stamped onto ``response.metadata`` so the
        shared EVALUATE phase stops promptly (FEP-0007 cutover tune-up).
        """
        if tool_exec_result is not None:
            if disposition is not TurnDisposition.CONTINUE:
                raise ValueError("Tool execution cannot follow a stopped or recovering emit")
            disposition = tool_exec_result.disposition

        response = CompletionResponse(
            content=full_content or "",
            tool_calls=list(tool_calls) if tool_calls else None,
            metadata={"forced_task_completion": True} if forced_completion else None,
        )
        tool_results = list(tool_exec_result.tool_results) if tool_exec_result is not None else []
        return TurnResult(
            response=response,
            tool_results=tool_results,
            has_tool_calls=bool(tool_calls),
            tool_calls_count=len(tool_calls) if tool_calls else 0,
            is_qa_response=bool(getattr(stream_ctx, "is_qa_task", False)),
            disposition=disposition,
        )

    async def execute_turn_streaming(
        self,
        orch: Any,
        runtime_owner: Any,
        stream_ctx: Any,
        *,
        user_message: str,
        goals: Any,
        recovery: Any,
        create_recovery_context: Any,
        forced_task_completion: bool = False,
        result: StreamingActResult,
    ) -> AsyncIterator[StreamChunk]:
        """Run ONE turn's ACT as a contiguous streaming primitive (FEP-0007 Addendum A).

        The streaming counterpart of the buffered ACT (``turn_executor.execute_turn``): it runs
        this turn's provider response, assistant-content emit, and tool execution in sequence —
        reusing the existing ``_stream_provider_turn`` / ``_emit_assistant_turn`` /
        ``_execute_tools_turn`` helpers — yielding every ``StreamChunk`` for live delivery and
        writing the produced ``TurnResult`` (plus the raw signals the shared EVALUATE/DECIDE
        phases need) onto ``result``.

        This carries ACT only: the per-turn EVALUATE/DECIDE bands (early-stop, task-completion,
        continuation, nudge, quality, post-tool fulfillment/plateau/novelty) live in the shared
        ``AgenticLoop`` phases that drive this primitive via ``run_streaming``. It is the live
        streaming ACT — the legacy ``run()`` / ``_stream_turn`` loop was removed at the cutover.
        """
        # Correlation spine: fresh turn_id so this streaming turn's capture records
        # (tool.supply, tool.intent, rl_outcome) share one id. Best-effort.
        try:
            from victor.core.context import begin_turn as _begin_turn

            _begin_turn()
        except Exception:  # correlation is non-critical
            pass

        # ACT — provider response (token streaming happens inside _stream_provider_turn).
        _raise_if_stream_cancelled(self.services)
        tools, full_content, tool_calls, garbage_detected = await self._stream_provider_turn(
            orch, runtime_owner, stream_ctx, goals
        )
        _raise_if_stream_cancelled(self.services)
        tool_calls, full_content = self.services.tool_calls.parse_and_validate(
            tool_calls,
            full_content,
        )
        result.tools = tools
        result.garbage_detected = garbage_detected

        # ACT — emit the assistant response (handles tool-call-only / empty-response recovery).
        _emit = _EmitDecision()
        if full_content:
            full_content = await self.services.governance.govern_response(full_content)
        async with aclosing_if_supported(
            self._emit_assistant_turn(
                orch,
                runtime_owner,
                stream_ctx,
                recovery=recovery,
                create_recovery_context=create_recovery_context,
                full_content=full_content,
                tool_calls=tool_calls,
                forced_task_completion=forced_task_completion,
                user_message=user_message,
                tools=tools,
                decision=_emit,
            )
        ) as stream:
            iterator = aiter(stream)
            while True:
                _raise_if_stream_cancelled(self.services)
                try:
                    chunk = await anext(iterator)
                except StopAsyncIteration:
                    break
                _raise_if_stream_cancelled(self.services)
                yield chunk
        _raise_if_stream_cancelled(self.services)
        result.assistant_content_yielded = _emit.assistant_content_yielded
        if _emit.recovery_content is not None:
            full_content = _emit.recovery_content
        # Empty-response recovery may have replaced the tool calls.
        tool_calls = _emit.tool_calls

        # ACT — execute tools (only when this turn produced tool calls and emit did not exit).
        tool_exec_result = None
        if tool_calls and not (_emit.should_return or _emit.should_continue):
            _raise_if_stream_cancelled(self.services)
            _tool_outcome = _ToolTurnOutcome()
            async with aclosing_if_supported(
                self._execute_tools_turn(
                    orch,
                    runtime_owner,
                    stream_ctx,
                    user_message=user_message,
                    tool_calls=tool_calls,
                    full_content=full_content,
                    result_holder=_tool_outcome,
                )
            ) as stream:
                async for chunk in stream:
                    yield chunk
            tool_exec_result = _tool_outcome.result

        # Restore the streaming loop's prompt completion: a HIGH-confidence answer with no pending
        # tools stops the unified loop immediately (signalled via the TurnResult to EVALUATE),
        # rather than relying solely on the under-scoring EnhancedCompletionEvaluator.
        result.forced_completion = (
            _emit.disposition is TurnDisposition.CONTINUE
            and (tool_exec_result is None or not tool_exec_result.should_return)
            and self._detect_high_confidence_completion(
                stream_ctx,
                full_content=full_content,
                tool_calls=tool_calls,
            )
        )

        result.full_content = full_content
        result.tool_calls = tool_calls
        result.tool_exec_result = tool_exec_result
        result.turn_result = self._build_streaming_turn_result(
            stream_ctx,
            full_content=full_content,
            tool_calls=tool_calls,
            tool_exec_result=tool_exec_result,
            forced_completion=result.forced_completion,
            disposition=_emit.disposition,
        )

    async def run_unified(self, user_message: str, **kwargs: Any) -> AsyncIterator[StreamChunk]:
        """Unified streaming run — drive ``AgenticLoop.run_streaming`` (FEP-0007 step-3 cutover).

        Makes the streaming UI path run the SAME research-rooted PERCEIVE -> PLAN -> ACT ->
        EVALUATE -> DECIDE loop as the buffered path, emitting ``StreamChunk``s via the streaming
        ACT adapter. This is the step-3 cutover entry point: it runs the per-run governance gate,
        prepares the streaming session/adapter, builds the ``AgenticLoop`` over the orchestrator's
        own collaborators (the same ``turn_executor`` + ``runtime_intelligence`` the buffered loop
        uses), and yields ``run_streaming``'s chunks.

        This is the SOLE live streaming entry point: ``ServiceStreamingRuntime``'s stream path calls
        ``run_unified`` directly. The temporary ``run()`` alias has been removed.
        The UI path uses the unified loop's EVALUATE phase, including
        its requirement-driven completion (EnhancedCompletion / fulfillment) —
        so it may complete earlier on multi-step tasks than the old streaming
        loop, which simply followed the model's tool calls. That convergence is
        the whole point of the unification; whether the completion threshold is
        well-tuned for real tasks is verified live (zai), not on the scripted
        parity battery. See the run/stream parity battery.
        """
        from victor.agent.services.streaming_act_adapter import StreamingActAdapter
        from victor.framework.agentic_loop import AgenticLoop

        runtime_owner = self._runtime_owner
        orch = runtime_owner._orchestrator

        # Governance REQUEST phase (per-run): a block short-circuits the WHOLE run with a single
        # refusal chunk; a redaction substitutes the message used downstream. Unlike run(), this
        # lives in the run wrapper (not the per-turn ACT), since run_streaming owns the turn loop.
        req = await self.services.governance.check_request(user_message)
        if req is not None:
            if not req.allowed:
                reason = req.reason or "Your message was blocked by policy."
                chunk = self.services.delivery.content_chunk(reason, is_final=True)
                chunk.metadata = {
                    **(chunk.metadata or {}),
                    "agentic_loop_success": False,
                    "agentic_loop_iterations": 0,
                    "agentic_loop_error": reason,
                }
                yield chunk
                return
            user_message = req.content

        # Per-run setup (stream context, goals, recovery) captured as a session + ACT adapter.
        adapter = await StreamingActAdapter.prepare(self, user_message, **kwargs)
        stream_ctx = adapter.session.stream_ctx

        # Completion strategy (ADR-009): thread from settings; build the provider-backed rubric judge
        # for rubric/hybrid (default "enhanced" → no rubric, no behavior change). Reuses the buffered
        # path's helper so both modes resolve completion identically.
        from victor.agent.services.judge_calibration_gate import resolve_completion_strategy

        _te = getattr(orch, "turn_executor", None)
        # Env override → settings default → ADR-011 judge pinning; same gate as buffered path.
        _strategy = resolve_completion_strategy(
            getattr(orch, "settings", None),
            getattr(getattr(_te, "_provider_context", None), "model", None),
        )
        _rubric_fn = (
            _te._build_rubric_complete_fn()
            if (_strategy in ("rubric", "hybrid") and _te is not None)
            else None
        )
        from victor.framework.effect_gate import resolve_effect_gate_enabled
        from victor.framework.per_turn_auditor import resolve_per_turn_auditor_enabled

        _settings = getattr(orch, "settings", None)
        _effect_gate = resolve_effect_gate_enabled(_settings)
        _auditor = resolve_per_turn_auditor_enabled(_settings)
        # EVR-6: when the auditor is on, back it with the edge-model prefix judge (Ollama,
        # token-free). build_edge_turn_judge() returns None when the edge model is unavailable,
        # so the auditor degrades to its deterministic heuristic.
        _turn_judge = None
        if _auditor:
            from victor.agent.edge_turn_judge import build_edge_turn_judge

            _turn_judge = build_edge_turn_judge()
        loop = AgenticLoop(
            orchestrator=None,
            turn_executor=_te,
            runtime_intelligence=self._get_runtime_intelligence(),
            max_iterations=getattr(stream_ctx, "max_total_iterations", 10),
            enable_fulfillment_check=True,
            enable_adaptive_iterations=True,
            exploration_settings=getattr(getattr(orch, "settings", None), "exploration", None),
            streaming_act_port=adapter,
            config={
                "completion_strategy": _strategy,
                "enable_effect_gate": _effect_gate,
                "enable_per_turn_auditor": _auditor,
            },
            rubric_complete_fn=_rubric_fn,
            verifier=getattr(_te, "_verifier", None),
            max_verify_retries=getattr(_te, "_max_verify_retries", 0),
            per_turn_judge=_turn_judge,
        )

        conversation_history = self._get_conversation_history(runtime_owner, orch, user_message)
        async with aclosing_if_supported(
            loop.run_streaming(user_message, conversation_history=conversation_history)
        ) as stream:
            iterator = aiter(stream)
            cancelled = False
            while True:
                if self.services.stream_lifecycle.is_cancelled():
                    cancelled = True
                    break
                try:
                    chunk = await anext(iterator)
                except _StreamingCancelled:
                    cancelled = True
                    break
                except StopAsyncIteration:
                    cancelled = self.services.stream_lifecycle.is_cancelled()
                    break
                if self.services.stream_lifecycle.is_cancelled():
                    cancelled = True
                    break
                yield chunk
        if cancelled:
            yield _cancelled_stream_chunk(self.services, stream_ctx)


def create_streaming_chat_executor(
    runtime_owner: StreamingExecutionRuntimeProtocol,
    perception: Optional[Any] = None,
    fulfillment: Optional[Any] = None,
) -> StreamingChatExecutor:
    """Factory helper for creating the canonical service-owned executor."""
    return StreamingChatExecutor(
        runtime_owner,
        perception=perception,
        fulfillment=fulfillment,
    )
