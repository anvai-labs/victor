# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
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
import re
from types import SimpleNamespace
from dataclasses import dataclass
from typing import Any, AsyncIterator, List, Optional, Protocol, Tuple

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

logger = logging.getLogger(__name__)


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

    result: Any = None


@dataclass
class _EmitDecision:
    """Single-slot mutable holder for the assistant-content emit / empty-response band.

    ``_emit_assistant_turn`` emits the assistant response (or handles tool-call-only and
    empty-response recovery), yielding content/recovery chunks. Because it yields, it can't
    return its outcome: ``should_return`` ends the stream, ``should_continue`` restarts the loop,
    and the remaining fields propagate loop-local state run() needs afterward —
    ``assistant_content_yielded`` (gates terminal-delivery fallback in continuation),
    ``prev_iteration_had_content`` (drives the inter-iteration blank-line separator), and
    ``tool_calls`` (empty-response recovery may replace it). Part of FEP-0007 Phase 2's move
    toward a shared ``stream_turn()``.
    """

    should_return: bool = False
    should_continue: bool = False
    assistant_content_yielded: bool = False
    prev_iteration_had_content: bool = False
    tool_calls: Any = None


@dataclass
class StreamingActResult:
    """ACT-phase output of one streaming turn (FEP-0007 Addendum A, the streaming-ACT seam).

    ``execute_turn_streaming`` runs ONE turn's ACT sub-steps (provider response → assistant
    emit → tool execution) and yields that turn's ``StreamChunk``s. Because an async generator
    can't ``return`` a value, it writes its outcome here for the caller — the buffered shape the
    shared EVALUATE/DECIDE phases consume:

    - ``turn_result``: the ``TurnResult`` (same primitive ``turn_executor.execute_turn`` returns
      for the buffered path) built from this turn's content + tool results.
    - ``full_content`` / ``tool_calls`` / ``tools``: the raw provider outputs, surfaced for the
      caller's evaluation (``tools`` is needed for empty-response recovery).
    - ``tool_exec_result``: the streaming ``ToolExecutionResult`` (``None`` when no tools ran).
    - ``garbage_detected`` / ``assistant_content_yielded``: per-turn signals the streaming
      EVALUATE band reads (garbage→force completion; whether visible content was streamed).
    - ``emit_should_return`` / ``emit_should_continue``: loop-control hints surfaced by the
      assistant-emit / empty-response sub-step, for the future ``run_streaming`` driver.

    This carries ACT output only; the EVALUATE/DECIDE bands live in the shared ``AgenticLoop``
    phases (``run_streaming``), which is the canonical streaming loop — the legacy ``run()`` /
    ``_stream_turn`` were removed at the FEP-0007 cutover.
    """

    turn_result: Optional[Any] = None
    full_content: str = ""
    tool_calls: Any = None
    tools: Any = None
    tool_exec_result: Any = None
    garbage_detected: bool = False
    assistant_content_yielded: bool = False
    emit_should_return: bool = False
    emit_should_continue: bool = False
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
        runtime_intelligence: Optional[Any] = None,
        perception: Optional[Any] = None,
        fulfillment: Optional[Any] = None,
        confidence_monitor: Optional[Any] = None,
    ) -> None:
        self._runtime_owner = runtime_owner
        self._runtime_intelligence = runtime_intelligence
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

    @staticmethod
    def _get_task_completion_summary(detector: Any) -> str:
        """Return the detector's last completion summary without active markers."""
        state = getattr(detector, "_state", None)
        summary = getattr(state, "last_summary", "") if state is not None else ""
        return strip_active_completion_markers(summary).strip()

    @staticmethod
    def _build_final_marker_chunk(orch: Any) -> StreamChunk:
        """Return a terminal marker chunk even when no text is available."""
        generator = getattr(orch, "_chunk_generator", None)
        marker_factory = getattr(generator, "generate_final_marker_chunk", None)
        if callable(marker_factory):
            return marker_factory()
        return StreamChunk(content="", is_final=True)

    def _resolve_terminal_visible_output(
        self,
        orch: Any,
        stream_ctx: Any,
        *,
        full_content: str,
        user_message: str,
    ) -> tuple[str, Optional[str]]:
        """Pick the best visible terminal response from provider and summary fallbacks."""
        detector = getattr(orch, "_task_completion_detector", None)
        candidates = (
            ("provider_response", full_content),
            ("completion_summary", self._get_task_completion_summary(detector)),
            ("compaction_summary", getattr(stream_ctx, "compaction_summary", "")),
        )

        sanitizer = getattr(orch, "sanitizer", None)
        for source, candidate in candidates:
            normalized = self._normalize_visible_candidate(candidate, user_message=user_message)
            if not normalized:
                continue

            sanitized = sanitizer.sanitize(normalized) if sanitizer else normalized
            if isinstance(sanitized, str) and sanitized.strip():
                return sanitized.strip(), source

            plain_text = sanitizer.strip_markup(normalized) if sanitizer else normalized
            if isinstance(plain_text, str) and plain_text.strip():
                return plain_text.strip(), source

        return "", None

    async def _govern_final_response(self, orch: Any, text: str) -> tuple[str, bool]:
        """RESPONSE-phase gate for the streaming path's final assistant output.

        Returns ``(text_to_use, blocked)``. When no message policy gate is
        configured the text is returned unchanged. Streaming cannot un-send
        tokens that were already emitted incrementally, so this governs the
        persisted history copy and the final delivered chunk (the documented
        limitation of message governance on the streaming path).
        """
        gate = getattr(orch, "_message_policy_gate", None)
        if gate is None or not text:
            return text, False
        result = await gate.gate_response(text)
        if not result.allowed:
            return result.reason or "The response was withheld by policy.", True
        return result.content, False

    async def _build_terminal_delivery_chunk(
        self,
        orch: Any,
        stream_ctx: Any,
        *,
        full_content: str,
        user_message: str,
    ) -> StreamChunk:
        """Guarantee a visible terminal assistant chunk or an explicit final marker."""
        terminal_content, content_source = self._resolve_terminal_visible_output(
            orch,
            stream_ctx,
            full_content=full_content,
            user_message=user_message,
        )

        if terminal_content:
            # RESPONSE-phase governance on the terminal (final) assistant output.
            terminal_content, _blocked = await self._govern_final_response(orch, terminal_content)

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
            return orch._chunk_generator.generate_content_chunk(terminal_content, is_final=True)

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
        return self._build_final_marker_chunk(orch)

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

    def _is_write_action_turn(self, orch: Any, stream_ctx: Any) -> bool:
        """Return True for turns where continued read-only exploration is risky."""
        if not bool(getattr(stream_ctx, "is_action_task", False)):
            return False

        intent = getattr(orch, "_current_intent", None)
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
        if not self._is_write_action_turn(orch, stream_ctx):
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

        tool_pipeline = getattr(orch, "_tool_pipeline", None)
        reset_pipeline = getattr(tool_pipeline, "reset", None)
        if callable(reset_pipeline):
            reset_pipeline()

        detector = getattr(orch, "_task_completion_detector", None)
        reset_detector = getattr(detector, "reset", None)
        if callable(reset_detector):
            reset_detector()

    @staticmethod
    def _clear_deferred_active_completion_signal(detector: Any) -> None:
        """Clear a completion marker that arrived alongside additional tool calls."""
        clear_active_signal = getattr(detector, "clear_active_signal", None)
        if callable(clear_active_signal):
            clear_active_signal()
            return

        state = getattr(detector, "_state", None)
        if state is None:
            return

        if hasattr(state, "active_signal_detected"):
            state.active_signal_detected = False

        signals = getattr(state, "completion_signals", None)
        if isinstance(signals, set):
            signals_to_keep = {
                signal for signal in signals if not str(signal).startswith("active:")
            }
            state.completion_signals = signals_to_keep

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
        orch: Any,
        context_msg: str,
        goals: Any,
        planned_tools: Any = None,
    ) -> Any:
        """Select tools with per-turn caching."""
        if self._last_tool_context == context_msg and self._last_tools is not None:
            return self._last_tools

        tools = await orch._select_tools_for_turn(
            context_msg,
            goals,
            planned_tools=planned_tools,
        )
        if tools is not None:
            self._last_tool_context = context_msg
            self._last_tools = tools
        return tools

    @staticmethod
    def _should_execute_prepared_team(stream_ctx: Any) -> bool:
        """Check whether streaming topology resolved a concrete team execution."""
        runtime_context_overrides = getattr(stream_ctx, "runtime_context_overrides", None)
        return bool(
            isinstance(runtime_context_overrides, dict)
            and runtime_context_overrides.get("execution_mode") == "team_execution"
            and runtime_context_overrides.get("team_name")
        )

    @staticmethod
    def _build_stream_team_context(
        user_message: str,
        stream_ctx: Any,
    ) -> dict[str, Any]:
        """Build shared context for framework team execution in streaming mode."""
        runtime_context_overrides = getattr(stream_ctx, "runtime_context_overrides", {}) or {}
        context = {
            "query": user_message,
            "task_type": getattr(stream_ctx, "coarse_task_type", None),
            "task_complexity": getattr(
                getattr(getattr(stream_ctx, "task_classification", None), "complexity", None),
                "value",
                getattr(getattr(stream_ctx, "task_classification", None), "complexity", None),
            ),
            "topology_plan": getattr(stream_ctx, "topology_plan", None),
            "topology_decision": getattr(stream_ctx, "topology_decision", None),
            "perception": getattr(stream_ctx, "perception", None),
        }
        for key in (
            "team_name",
            "team_display_name",
            "formation_hint",
            "topology_action",
            "topology_kind",
            "topology_metadata",
            "provider_hint",
            "max_workers",
            "worktree_isolation",
            "materialize_worktrees",
            "dry_run_worktrees",
            "cleanup_worktrees",
        ):
            value = runtime_context_overrides.get(key)
            if value is not None:
                context[key] = value
        return context

    async def _execute_prepared_team(
        self,
        orch: Any,
        user_message: str,
        stream_ctx: Any,
    ) -> Optional[StreamChunk]:
        """Execute a prepared framework team and convert it into a final stream chunk."""
        runtime_context_overrides = getattr(stream_ctx, "runtime_context_overrides", None)
        if not isinstance(runtime_context_overrides, dict):
            return None

        from victor.framework.team_runtime import run_configured_team

        complexity_value = getattr(
            getattr(stream_ctx, "task_classification", None), "complexity", None
        )
        if hasattr(complexity_value, "value"):
            complexity_value = complexity_value.value
        if complexity_value is None:
            complexity_value = "medium"

        team_execution = await run_configured_team(
            orch,
            goal=user_message,
            task_type=str(getattr(stream_ctx, "coarse_task_type", None) or "unknown"),
            complexity=str(complexity_value),
            preferred_team=runtime_context_overrides.get("team_name"),
            preferred_formation=runtime_context_overrides.get("formation_hint"),
            max_workers=runtime_context_overrides.get("max_workers"),
            tool_budget=runtime_context_overrides.get("tool_budget"),
            context=self._build_stream_team_context(user_message, stream_ctx),
        )
        if team_execution is None:
            return None

        resolved_team, team_result = team_execution
        final_output = (
            team_result.final_output.strip() or team_result.error or "Team execution completed."
        )
        if final_output:
            from victor.agent.conversation.types import (
                MESSAGE_SOURCE_METADATA_KEY,
                MessageSource,
            )

            orch.add_message(
                "assistant",
                final_output,
                metadata={MESSAGE_SOURCE_METADATA_KEY: MessageSource.AGENT_RESPONSE.value},
            )
            if hasattr(stream_ctx, "accumulate_content"):
                stream_ctx.accumulate_content(final_output)
            if hasattr(stream_ctx, "update_context_message"):
                stream_ctx.update_context_message(final_output)
        if hasattr(stream_ctx, "reset_activity_timer"):
            stream_ctx.reset_activity_timer()
        if hasattr(stream_ctx, "update_quality_score"):
            stream_ctx.update_quality_score(0.92 if team_result.success else 0.2)

        stream_ctx.full_content = final_output
        stream_ctx.force_completion = True
        stream_ctx.total_iterations = max(1, getattr(stream_ctx, "total_iterations", 0))
        stream_ctx.tool_calls_used = team_result.total_tool_calls
        if hasattr(orch, "tool_calls_used"):
            orch.tool_calls_used = team_result.total_tool_calls

        metadata = getattr(stream_ctx, "topology_plan", None)
        if isinstance(metadata, dict):
            metadata["team_name"] = resolved_team.team_name
            metadata["team_display_name"] = resolved_team.display_name
            metadata["member_count"] = resolved_team.member_count

        return orch._chunk_generator.generate_content_chunk(final_output, is_final=True)

    @staticmethod
    async def _extract_task_requirements(orch: Any, user_message: str) -> None:
        """Extract required files/outputs from the prompt onto orch + emit a state event.

        A cohesive piece of run()'s preamble (FEP-0007 Phase 2 decomposition). Mutates orch
        run-state; yields nothing.
        """
        orch._required_files = _extract_required_files_from_prompt(user_message)
        orch._required_outputs = _extract_required_outputs_from_prompt(user_message)
        orch._read_files_session.clear()
        orch._all_files_read_nudge_sent = False
        logger.debug(
            "Task requirements extracted - files: %s, outputs: %s",
            orch._required_files,
            orch._required_outputs,
        )
        if orch._required_files or orch._required_outputs:
            from victor.core.events import get_observability_bus

            event_bus = get_observability_bus()
            await event_bus.emit(
                topic="state.task.requirements_extracted",
                data={
                    "required_files": orch._required_files,
                    "required_outputs": orch._required_outputs,
                    "file_count": len(orch._required_files),
                    "output_count": len(orch._required_outputs),
                    "category": "state",
                },
            )

    @staticmethod
    def _apply_run_guidance(
        orch: Any, stream_ctx: Any, user_message: str, max_exploration_iterations: int
    ) -> None:
        """Apply intent guard + task-type guidance + action-task guidance for one run.

        A cohesive piece of run()'s preamble (FEP-0007 Phase 2 decomposition). Mutates orch
        (guards + guidance messages); yields nothing.
        """
        orch._apply_intent_guard(user_message)

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

        orch._apply_task_guidance(
            user_message,
            stream_ctx.unified_task_type,
            stream_ctx.is_analysis_task,
            stream_ctx.is_action_task,
            stream_ctx.needs_execution,
            max_exploration_iterations,
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

    @staticmethod
    def _initialize_task_intent(orch: Any, stream_ctx: Any, user_message: str) -> Any:
        """Seed task intent on the stream context and return the inferred goals.

        Final cohesive piece of run()'s preamble (FEP-0007 Phase 2 decomposition). Mutates
        stream_ctx (task intent, plan steps, task-start event); yields nothing. Returns the
        inferred ``goals`` so the loop can use them for tool planning.
        """
        goals = orch._tool_planner.infer_goals_from_message(user_message)
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
            planned_tools = orch._tool_planner.plan_tools(goals, available_inputs)
        stream_ctx.planned_tools = planned_tools

        if getattr(stream_ctx, "is_qa_task", False):
            tools = None
        elif orch.get_session_tools() is not None:
            tools = orch.get_session_tools()
        else:
            tools = await self._get_tools_cached(
                orch,
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
        if not runtime_owner._tool_execution_handler:
            from victor.agent.streaming import create_tool_execution_handler

            runtime_owner._tool_execution_handler = create_tool_execution_handler(orch)

        runtime_owner._tool_execution_handler.update_observed_files(
            set(orch.observed_files) if orch.observed_files else set()
        )

        if hasattr(runtime_owner._tool_execution_handler, "execute_tools_streaming"):
            from victor.agent.streaming.tool_execution import (
                ToolExecutionResult,
            )

            tool_exec_result = ToolExecutionResult()
            async for chunk in runtime_owner._tool_execution_handler.execute_tools_streaming(
                stream_ctx=stream_ctx,
                tool_calls=tool_calls,
                user_message=user_message,
                full_content=full_content,
                tool_calls_used=orch.tool_calls_used,
                tool_budget=orch.tool_budget,
                result=tool_exec_result,
            ):
                yield chunk
        else:
            tool_exec_result = await runtime_owner._tool_execution_handler.execute_tools(
                stream_ctx=stream_ctx,
                tool_calls=tool_calls,
                user_message=user_message,
                full_content=full_content,
                tool_calls_used=orch.tool_calls_used,
                tool_budget=orch.tool_budget,
            )
            for chunk in tool_exec_result.chunks:
                yield chunk

        orch.tool_calls_used += tool_exec_result.tool_calls_executed
        stream_ctx.tool_calls_used = orch.tool_calls_used
        _record = getattr(stream_ctx, "record_iteration_tool_count", None)
        if callable(_record):
            _record(tool_exec_result.tool_calls_executed)
        self._maybe_inject_write_action_guard(
            orch,
            stream_ctx,
            user_message=user_message,
            tool_calls_used=orch.tool_calls_used,
        )

        result_holder.result = tool_exec_result

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
        """Emit the assistant response (or handle tool-call-only / empty-response recovery).

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
            sanitized = orch.sanitizer.sanitize(visible_content)
            if sanitized:
                from victor.agent.conversation.types import (
                    MESSAGE_SOURCE_METADATA_KEY,
                    MessageSource,
                )

                # RESPONSE-phase governance only on the final emit; intermediate
                # content turns continue the loop and must not be blocked.
                _is_final_emit = forced_task_completion and not tool_calls
                if _is_final_emit:
                    sanitized, _ = await self._govern_final_response(orch, sanitized)
                orch.add_message(
                    "assistant",
                    sanitized,
                    tool_calls=tool_calls,
                    persist_synchronously=_is_final_emit,
                    metadata={MESSAGE_SOURCE_METADATA_KEY: MessageSource.AGENT_RESPONSE.value},
                )
                decision.assistant_content_yielded = True
                yield orch._chunk_generator.generate_content_chunk(
                    sanitized,
                    is_final=_is_final_emit,
                )
                if _is_final_emit:
                    decision.should_return = True
                    return
            else:
                plain_text = orch.sanitizer.strip_markup(
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
                    if _is_final_emit:
                        plain_text, _ = await self._govern_final_response(orch, plain_text)
                    orch.add_message(
                        "assistant",
                        plain_text,
                        tool_calls=tool_calls,
                        persist_synchronously=_is_final_emit,
                        metadata={MESSAGE_SOURCE_METADATA_KEY: MessageSource.AGENT_RESPONSE.value},
                    )
                    decision.assistant_content_yielded = True
                    yield orch._chunk_generator.generate_content_chunk(
                        plain_text,
                        is_final=_is_final_emit,
                    )
                    if _is_final_emit:
                        decision.should_return = True
                        return
                elif forced_task_completion and not tool_calls:
                    yield await self._build_terminal_delivery_chunk(
                        orch,
                        stream_ctx,
                        full_content=full_content,
                        user_message=user_message,
                    )
                    decision.should_return = True
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
                yield final_chunk
                decision.should_return = True
                return

            logger.warning("Model returned empty response - attempting aggressive recovery")

            recovery_ctx = create_recovery_context(stream_ctx)
            recovery_chunk, should_force = recovery.handle_empty_response(recovery_ctx)
            _ = should_force
            if recovery_chunk:
                yield recovery_chunk
                decision.should_continue = True
                return

            recovery_success, recovered_tool_calls, final_chunk = (
                await runtime_owner._handle_empty_response_recovery(stream_ctx, tools)
            )

            if recovery_success:
                if final_chunk:
                    yield final_chunk
                    decision.should_return = True
                    return
                if recovered_tool_calls:
                    decision.tool_calls = recovered_tool_calls
                    logger.info(
                        "Recovery produced %s tool call(s) - continuing main loop",
                        len(recovered_tool_calls),
                    )
            else:
                recovery_ctx = create_recovery_context(stream_ctx)
                fallback_msg = recovery.get_recovery_fallback_message(recovery_ctx)
                orch._record_runtime_intelligence_outcome(
                    success=False,
                    quality_score=0.3,
                    user_satisfied=False,
                    completed=False,
                )
                yield orch._chunk_generator.generate_content_chunk(fallback_msg, is_final=True)
                decision.should_return = True
                return

    def _detect_high_confidence_completion(
        self,
        orch: Any,
        stream_ctx: Any,
        *,
        full_content: str,
        tool_calls: Any,
    ) -> bool:
        """Return True when this turn's answer is a HIGH-confidence completion with no pending tools.

        Runs ``orch._task_completion_detector`` over the assistant content; on a HIGH-confidence
        active signal with no outstanding tool calls it persists a VICTOR_SUMMARY for any next-turn
        context and returns True. This is the streaming loop's prompt-completion signal (formerly in
        ``_detect_task_completion_and_mentions``) — the unified loop surfaces it onto the TurnResult
        so EVALUATE stops immediately instead of restating to the iteration cap (FEP-0007 tune-up).
        """
        detector = getattr(orch, "_task_completion_detector", None)
        if not detector or not full_content:
            return False

        from victor.agent.task_completion import CompletionConfidence

        detector.analyze_response(full_content)
        if detector.get_completion_confidence() != CompletionConfidence.HIGH:
            return False
        if tool_calls:
            # Defer: a HIGH marker alongside pending tool calls isn't a real completion.
            self._clear_deferred_active_completion_signal(detector)
            return False

        logger.info(
            "Task completion: HIGH confidence detected (active signal), "
            "forcing completion NOW (immediate stop, skip continuation)"
        )
        stream_ctx.force_completion = True
        stream_ctx.skip_continuation = True
        last_summary = getattr(getattr(detector, "_state", None), "last_summary", "")
        sanitized_summary = strip_active_completion_markers(last_summary).strip()
        if sanitized_summary and hasattr(orch, "_conversation_controller"):
            try:
                orch._conversation_controller.persist_compaction_summary(sanitized_summary, [])
                orch._conversation_controller.inject_compaction_context()
                logger.info("VICTOR_SUMMARY persisted for next-turn context injection")
            except Exception as exc:
                logger.debug("Failed to persist VICTOR_SUMMARY: %s", exc)
        return True

    @staticmethod
    def _build_streaming_turn_result(
        stream_ctx: Any,
        *,
        full_content: str,
        tool_calls: Any,
        tool_exec_result: Any,
        forced_completion: bool = False,
    ) -> Any:
        """Assemble a ``TurnResult`` from one streaming turn's ACT output.

        Produces the same primitive ``turn_executor.execute_turn`` returns for the buffered
        path, so the shared EVALUATE phase is mode-agnostic: the streamed ``full_content`` /
        ``tool_calls`` become the ``CompletionResponse``, and the streaming
        ``ToolExecutionResult.tool_results`` (if any) become ``TurnResult.tool_results``. A
        ``forced_completion`` HIGH-confidence signal is stamped onto ``response.metadata`` so the
        shared EVALUATE phase stops promptly (FEP-0007 cutover tune-up).
        """
        from victor.agent.services.turn_execution_runtime import TurnResult

        response = CompletionResponse(
            content=full_content or "",
            tool_calls=list(tool_calls) if tool_calls else None,
            metadata={"forced_task_completion": True} if forced_completion else None,
        )
        tool_results = (
            list(getattr(tool_exec_result, "tool_results", None) or [])
            if tool_exec_result is not None
            else []
        )
        return TurnResult(
            response=response,
            tool_results=tool_results,
            has_tool_calls=bool(tool_calls),
            tool_calls_count=len(tool_calls) if tool_calls else 0,
            is_qa_response=bool(getattr(stream_ctx, "is_qa_task", False)),
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
        tools, full_content, tool_calls, garbage_detected = await self._stream_provider_turn(
            orch, runtime_owner, stream_ctx, goals
        )
        tool_calls, full_content = orch._parse_and_validate_tool_calls(tool_calls, full_content)
        result.tools = tools
        result.garbage_detected = garbage_detected

        # ACT — emit the assistant response (handles tool-call-only / empty-response recovery).
        _emit = _EmitDecision()
        async for chunk in self._emit_assistant_turn(
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
        ):
            yield chunk
        result.assistant_content_yielded = _emit.assistant_content_yielded
        result.emit_should_return = _emit.should_return
        result.emit_should_continue = _emit.should_continue
        # Empty-response recovery may have replaced the tool calls.
        tool_calls = _emit.tool_calls

        # ACT — execute tools (only when this turn produced tool calls and emit did not exit).
        tool_exec_result = None
        if tool_calls and not (_emit.should_return or _emit.should_continue):
            _tool_outcome = _ToolTurnOutcome()
            async for chunk in self._execute_tools_turn(
                orch,
                runtime_owner,
                stream_ctx,
                user_message=user_message,
                tool_calls=tool_calls,
                full_content=full_content,
                result_holder=_tool_outcome,
            ):
                yield chunk
            tool_exec_result = _tool_outcome.result

        # Restore the streaming loop's prompt completion: a HIGH-confidence answer with no pending
        # tools stops the unified loop immediately (signalled via the TurnResult to EVALUATE),
        # rather than relying solely on the under-scoring EnhancedCompletionEvaluator.
        result.forced_completion = self._detect_high_confidence_completion(
            orch, stream_ctx, full_content=full_content, tool_calls=tool_calls
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
        )

    async def run_unified(self, user_message: str, **kwargs: Any) -> AsyncIterator[StreamChunk]:
        """Unified streaming run — drive ``AgenticLoop.run_streaming`` (FEP-0007 cutover, DRAFT).

        Makes the streaming UI path run the SAME research-rooted PERCEIVE -> PLAN -> ACT ->
        EVALUATE -> DECIDE loop as the buffered path, emitting ``StreamChunk``s via the streaming
        ACT adapter. This is the step-3 cutover entry point: it runs the per-run governance gate,
        prepares the streaming session/adapter, builds the ``AgenticLoop`` over the orchestrator's
        own collaborators (the same ``turn_executor`` + ``runtime_intelligence`` the buffered loop
        uses), and yields ``run_streaming``'s chunks.

        DRAFT / not yet wired as the sole live path into ``ChatStreamRuntime``
        (``chat_stream_runtime.py`` already calls ``run_unified``; the legacy
        ``run()`` entry point is preserved as a thin alias delegating here so the
        callers that still reference it — notably ``AgenticLoop.run_streaming`` —
        keep working during the cutover). **Behavior change vs the removed legacy
        run() body:** the UI path adopts the unified loop's EVALUATE — including
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
        gate = getattr(orch, "_message_policy_gate", None)
        if gate is not None:
            req = await gate.gate_request(user_message)
            if not req.allowed:
                yield orch._chunk_generator.generate_content_chunk(
                    req.reason or "Your message was blocked by policy.",
                    is_final=True,
                )
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
        loop = AgenticLoop(
            orchestrator=None,
            turn_executor=_te,
            runtime_intelligence=self._runtime_intelligence,
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
        )

        conversation_history = self._get_conversation_history(runtime_owner, orch, user_message)
        async for chunk in loop.run_streaming(
            user_message, conversation_history=conversation_history
        ):
            yield chunk

    async def run(self, user_message: str, **kwargs: Any) -> AsyncIterator[StreamChunk]:
        """Backward-compatible streaming entry point (LTS-deprecated).

        .. deprecated:: 0.8.0
            Use :meth:`run_unified` instead. ``run()`` is a thin alias retained
            only for the FEP-0007 cutover period so the one remaining internal
            caller (``AgenticLoop.run_streaming``) and the streaming parity test
            battery keep working. It delegates directly to ``run_unified`` so
            there is a single live code path, and will be removed once the last
            caller migrates. Do NOT add new callers of ``run()`` — that re-opens
            the wrong (legacy) streaming seam the unification closed.
        """
        import warnings

        warnings.warn(
            "StreamingChatExecutor.run() is deprecated; use run_unified() instead. "
            "run() is a temporary FEP-0007 cutover alias and will be removed.",
            DeprecationWarning,
            stacklevel=2,
        )
        async for chunk in self.run_unified(user_message, **kwargs):
            yield chunk


def create_streaming_chat_executor(
    runtime_owner: StreamingExecutionRuntimeProtocol,
    runtime_intelligence: Optional[Any] = None,
    perception: Optional[Any] = None,
    fulfillment: Optional[Any] = None,
) -> StreamingChatExecutor:
    """Factory helper for creating the canonical service-owned executor."""
    return StreamingChatExecutor(
        runtime_owner,
        runtime_intelligence=runtime_intelligence,
        perception=perception,
        fulfillment=fulfillment,
    )
