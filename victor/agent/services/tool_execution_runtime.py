# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Service-owned tool execution runtime helper."""

from __future__ import annotations

import inspect
import logging
from typing import Any, Dict, List, Optional

from victor.agent.runtime.context import AgentRuntimeContext
from victor.agent.tool_execution.categorization import (
    ToolCategory,
    categorize_tool_call,
)
from victor.agent.tool_output_formatter import FormattingContext
from victor.framework.execution_checkpoint import ExecutionCheckpoint

logger = logging.getLogger(__name__)


def _tool_result_matches_checkpoint(
    result: Dict[str, Any],
    checkpoint: Any,
) -> bool:
    """Return whether a tool result belongs to the triggering checkpoint call."""
    if checkpoint is None:
        return False

    tool_call = getattr(checkpoint, "triggering_tool_call", None) or {}
    checkpoint_tool_name = str(tool_call.get("name") or "")
    result_tool_name = str(result.get("name") or "")
    if checkpoint_tool_name and result_tool_name and checkpoint_tool_name != result_tool_name:
        return False

    checkpoint_tool_call_id = tool_call.get("id")
    result_tool_call_id = result.get("tool_call_id")
    if checkpoint_tool_call_id and result_tool_call_id:
        return checkpoint_tool_call_id == result_tool_call_id

    return bool(checkpoint_tool_name and result_tool_name)


class ToolExecutionRuntime:
    """Bridge orchestrator tool pipeline state into the canonical tool service."""

    def __init__(self, runtime_host: Any) -> None:
        self._runtime = runtime_host

    async def execute_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute tool calls and post-process results through the canonical services."""
        runtime = self._runtime
        if not tool_calls:
            return []

        tool_calls = [tool_call for tool_call in tool_calls if isinstance(tool_call, dict)]
        if not tool_calls:
            return []

        await self._maybe_create_execution_checkpoint(runtime, tool_calls)

        pipeline_result = await runtime._tool_pipeline.execute_tool_calls(
            tool_calls=tool_calls,
            context=runtime._get_tool_context(),
        )
        runtime.tool_calls_used = runtime._tool_pipeline.calls_used
        self._record_tool_intents(runtime, tool_calls)
        await self._compact_before_tool_result_injection(runtime, pipeline_result)

        from victor.agent.services.tool_service import ToolResultContext

        ctx = ToolResultContext(
            executed_tools=runtime.executed_tools,
            observed_files=runtime.observed_files,
            failed_tool_signatures=runtime.failed_tool_signatures,
            shown_tool_errors=runtime._shown_tool_errors,
            continuation_prompts=runtime._continuation_prompts,
            asking_input_prompts=runtime._asking_input_prompts,
            tool_calls_used=runtime.tool_calls_used,
            record_tool_execution=runtime._record_tool_execution,
            conversation_state=runtime.conversation_state,
            unified_tracker=runtime.unified_tracker,
            usage_logger=runtime.usage_logger,
            add_message=runtime.add_message,
            format_tool_output=self.format_tool_output,
            console=runtime.console,
            presentation=runtime._presentation,
            stream_context=(
                runtime._current_stream_context
                if hasattr(runtime, "_current_stream_context")
                else None
            ),
            task_type=getattr(
                runtime, "_current_task_type", getattr(runtime, "_task_type", "unknown")
            ),
        )

        results = runtime._tool_service.process_tool_results(pipeline_result, ctx)
        results = self._ensure_tool_response_coverage(runtime, tool_calls, results)
        self._annotate_tool_results_with_execution_checkpoint(runtime, results)
        self._record_tool_results(runtime, results)
        runtime._continuation_prompts = ctx.continuation_prompts
        runtime._asking_input_prompts = ctx.asking_input_prompts
        return results

    def format_tool_output(self, tool_name: str, args: Dict[str, Any], output: Any) -> str:
        """Format tool output with provider/context-aware boundaries."""
        runtime = self._runtime
        controller = getattr(runtime, "_conversation_controller", None)
        context_metrics = controller.get_context_metrics()
        provider = getattr(runtime, "provider", None)
        settings = getattr(runtime, "settings", None)
        context = FormattingContext(
            provider_name=getattr(provider, "name", None),
            model=getattr(settings, "model", None),
            remaining_tokens=context_metrics.remaining_tokens,
            max_tokens=context_metrics.max_tokens,
            response_token_reserve=getattr(settings, "response_token_reserve", 4096),
        )

        return runtime._tool_output_formatter.format_tool_output(
            tool_name=tool_name,
            args=args,
            output=output,
            context=context,
        )

    async def _maybe_create_execution_checkpoint(
        self,
        runtime: Any,
        tool_calls: List[Dict[str, Any]],
    ) -> Optional[ExecutionCheckpoint]:
        """Create a unified checkpoint envelope before a file-changing tool batch."""
        triggering_tool_call = self._first_write_tool_call(tool_calls)
        if triggering_tool_call is None:
            return None

        tool_name = str(triggering_tool_call.get("name") or "tool")
        conversation_checkpoint_id = await self._save_conversation_checkpoint(
            runtime,
            tool_name,
        )
        filesystem_checkpoint_id = await self._create_filesystem_checkpoint(
            runtime,
            tool_name,
        )
        checkpoint = ExecutionCheckpoint.create(
            session_id=self._resolve_runtime_context(runtime).session_id,
            graph_checkpoint_id=self._resolve_graph_checkpoint_id(runtime),
            conversation_checkpoint_id=conversation_checkpoint_id,
            filesystem_checkpoint_id=filesystem_checkpoint_id,
            triggering_tool_call=triggering_tool_call,
            metadata={
                "source": "tool_execution_runtime",
                "tool_batch_size": len(tool_calls),
            },
        )
        self._record_execution_checkpoint(runtime, checkpoint, tool_name)
        return checkpoint

    @staticmethod
    def _first_write_tool_call(
        tool_calls: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        for tool_call in tool_calls:
            tool_name = str(tool_call.get("name") or "")
            arguments = tool_call.get("arguments", {}) or {}
            if not isinstance(arguments, dict):
                arguments = {}
            if categorize_tool_call(tool_name, arguments) == ToolCategory.WRITE:
                return dict(tool_call)
        return None

    @staticmethod
    async def _save_conversation_checkpoint(runtime: Any, tool_name: str) -> Optional[str]:
        save_checkpoint = getattr(runtime, "save_checkpoint", None)
        if not callable(save_checkpoint):
            session_service = getattr(runtime, "_session_service", None)
            save_checkpoint = getattr(session_service, "save_checkpoint", None)
        if not callable(save_checkpoint):
            return None

        try:
            checkpoint_id = save_checkpoint(
                description=f"Before tool {tool_name} modifies files",
                tags=["execution", "pre_tool", f"tool:{tool_name}"],
            )
            if inspect.isawaitable(checkpoint_id):
                checkpoint_id = await checkpoint_id
            return str(checkpoint_id) if checkpoint_id else None
        except Exception as exc:
            logger.debug("Conversation checkpoint before %s failed: %s", tool_name, exc)
            return None

    @staticmethod
    async def _create_filesystem_checkpoint(runtime: Any, tool_name: str) -> Optional[str]:
        checkpoint_owner = (
            getattr(runtime, "git_checkpoint_manager", None)
            or getattr(runtime, "_git_checkpoint_manager", None)
            or getattr(runtime, "filesystem_checkpoint_manager", None)
            or getattr(runtime, "_filesystem_checkpoint_manager", None)
        )
        if checkpoint_owner is None:
            return None

        creator = (
            getattr(checkpoint_owner, "create_checkpoint", None)
            or getattr(
                checkpoint_owner,
                "checkpoint",
                None,
            )
            or getattr(
                checkpoint_owner,
                "create",
                None,
            )
        )
        if not callable(creator):
            return None

        try:
            checkpoint = creator(description=f"Before tool {tool_name} modifies files")
            if inspect.isawaitable(checkpoint):
                checkpoint = await checkpoint
            checkpoint_id = getattr(checkpoint, "id", checkpoint)
            return str(checkpoint_id) if checkpoint_id else None
        except Exception as exc:
            logger.debug("Filesystem checkpoint before %s failed: %s", tool_name, exc)
            return None

    @staticmethod
    def _resolve_graph_checkpoint_id(runtime: Any) -> Optional[str]:
        for attr_name in (
            "_current_graph_checkpoint_id",
            "current_graph_checkpoint_id",
            "_graph_checkpoint_id",
            "graph_checkpoint_id",
        ):
            checkpoint_id = getattr(runtime, attr_name, None)
            if checkpoint_id:
                return str(checkpoint_id)
        return None

    @staticmethod
    def _record_execution_checkpoint(
        runtime: Any,
        checkpoint: ExecutionCheckpoint,
        tool_name: str,
    ) -> None:
        checkpoints = getattr(runtime, "_execution_checkpoints", None)
        if not isinstance(checkpoints, list):
            checkpoints = []
            runtime._execution_checkpoints = checkpoints
        checkpoints.append(checkpoint)
        runtime._last_execution_checkpoint = checkpoint

        stream_ctx = getattr(runtime, "_current_stream_context", None)
        if stream_ctx is not None and hasattr(stream_ctx, "record_intent_event"):
            stream_ctx.record_intent_event(
                "execution_checkpoint",
                f"checkpoint before {tool_name}",
                tool=tool_name,
                execution_checkpoint_id=checkpoint.id,
                conversation_checkpoint_id=checkpoint.conversation_checkpoint_id,
                filesystem_checkpoint_id=checkpoint.filesystem_checkpoint_id,
            )

    @staticmethod
    def _annotate_tool_results_with_execution_checkpoint(
        runtime: Any,
        results: List[Dict[str, Any]],
    ) -> None:
        """Attach trace-safe checkpoint identifiers to matching tool result dicts."""
        execution_checkpoint = getattr(runtime, "_last_execution_checkpoint", None)
        if execution_checkpoint is None:
            return

        checkpoint_metadata = execution_checkpoint.to_trace_metadata()
        for result in results:
            if not isinstance(result, dict):
                continue
            if not _tool_result_matches_checkpoint(result, execution_checkpoint):
                continue
            for key, value in checkpoint_metadata.items():
                result.setdefault(key, value)

    async def _compact_before_tool_result_injection(
        self,
        runtime: Any,
        pipeline_result: Any,
    ) -> None:
        """Compact context before injecting unusually large tool output blocks."""
        estimated_output_tokens = self._estimate_pipeline_result_tokens(pipeline_result)
        if estimated_output_tokens <= 0:
            return

        lifecycle_decision = await self._compact_with_context_lifecycle(
            runtime,
            estimated_output_tokens,
        )
        if lifecycle_decision is not None:
            self._record_compaction_decision(runtime, lifecycle_decision, estimated_output_tokens)
            return

        context_service = getattr(runtime, "_context_service", None)
        if context_service is None:
            return

        prepare = getattr(context_service, "prepare_for_tool_output_injection", None)
        if not callable(prepare):
            return

        decision = prepare(
            estimated_output_tokens,
            provider_name=self._resolve_provider_name(runtime),
            model_name=str(getattr(runtime, "model", "") or ""),
            task_type=str(
                getattr(
                    runtime,
                    "_current_task_type",
                    getattr(runtime, "_task_type", "unknown"),
                )
                or "unknown"
            ),
            min_messages=6,
            default_strategy=str(
                getattr(
                    getattr(runtime, "settings", None),
                    "context_compaction_strategy",
                    "tiered",
                )
                or "tiered"
            ),
        )
        if inspect.isawaitable(decision):
            decision = await decision
        if not isinstance(decision, dict):
            return

        self._record_compaction_decision(runtime, decision, estimated_output_tokens)

    async def _compact_with_context_lifecycle(
        self,
        runtime: Any,
        estimated_output_tokens: int,
    ) -> Dict[str, Any] | None:
        lifecycle = getattr(runtime, "_context_lifecycle_service", None)
        if lifecycle is None:
            return None
        before_tool_output = getattr(lifecycle, "before_tool_output", None)
        if not callable(before_tool_output):
            return None
        runtime_context = self._resolve_runtime_context(runtime)
        messages = self._get_runtime_messages(runtime)
        decision = before_tool_output(
            runtime_context,
            estimated_output_tokens=estimated_output_tokens,
            messages=messages,
            provider_name=self._resolve_provider_name(runtime),
            model_name=str(getattr(runtime, "model", "") or ""),
            task_type=str(
                getattr(
                    runtime,
                    "_current_task_type",
                    getattr(runtime, "_task_type", "unknown"),
                )
                or "unknown"
            ),
            min_messages=6,
            default_strategy=str(
                getattr(
                    getattr(runtime, "settings", None),
                    "context_compaction_strategy",
                    "tiered",
                )
                or "tiered"
            ),
        )
        if inspect.isawaitable(decision):
            decision = await decision
        return decision if isinstance(decision, dict) else None

    def _record_compaction_decision(
        self,
        runtime: Any,
        decision: Dict[str, Any],
        estimated_output_tokens: int,
    ) -> None:
        removed = int(decision.get("messages_removed", 0) or 0)
        if removed <= 0:
            return

        strategy = str(decision.get("strategy", "") or "")
        reason = str(decision.get("reason", "") or "pre_tool_output")
        policy_reason = str(decision.get("policy_reason", "") or "")
        compaction_summary = str(
            decision.get("summary") or self._get_latest_compaction_summary(runtime)
        )
        logger.info(
            "Compacted context before tool-result injection: strategy=%s policy_reason=%s estimated_output_tokens=%s removed=%s",
            strategy,
            policy_reason or "n/a",
            estimated_output_tokens,
            removed,
        )

        stream_ctx = getattr(runtime, "_current_stream_context", None)
        if stream_ctx is not None:
            if hasattr(stream_ctx, "record_compaction_event"):
                stream_ctx.record_compaction_event(
                    summary=compaction_summary,
                    messages_removed=removed,
                    strategy=strategy,
                    reason=reason,
                    policy_reason=policy_reason,
                )
            else:
                stream_ctx.compaction_occurred = True
                stream_ctx.last_compaction_turn = getattr(stream_ctx, "total_iterations", 0)
                stream_ctx.compaction_message_removed_count = removed
                stream_ctx.compaction_summary = compaction_summary

    def _resolve_runtime_context(self, runtime: Any) -> AgentRuntimeContext:
        existing = getattr(runtime, "_agent_runtime_context", None) or getattr(
            runtime,
            "agent_runtime_context",
            None,
        )
        if isinstance(existing, AgentRuntimeContext):
            return existing
        session_id = (
            getattr(runtime, "active_session_id", None)
            or getattr(runtime, "session_id", None)
            or getattr(runtime, "_memory_session_id", None)
            or "session_root"
        )
        return AgentRuntimeContext(
            agent_id=str(getattr(runtime, "agent_id", None) or "root_agent"),
            display_name=str(getattr(runtime, "display_name", None) or "Root Agent"),
            role=str(getattr(runtime, "role", None) or "manager"),
            session_id=str(session_id),
        )

    @staticmethod
    def _get_runtime_messages(runtime: Any) -> List[Any]:
        get_messages = getattr(runtime, "get_messages", None)
        if callable(get_messages):
            try:
                return list(get_messages() or [])
            except Exception as exc:
                logger.debug("Failed to collect runtime messages for lifecycle: %s", exc)
        controller = getattr(runtime, "_conversation_controller", None)
        messages = getattr(controller, "messages", None)
        return list(messages or [])

    @staticmethod
    def _resolve_provider_name(runtime: Any) -> str:
        """Resolve the active provider identifier for context-service policy decisions."""
        provider = getattr(runtime, "provider", None)
        return str(getattr(provider, "name", getattr(runtime, "provider_name", "")) or "")

    @staticmethod
    def _iter_pipeline_results(pipeline_result: Any) -> List[Any]:
        """Return raw pipeline call results regardless of wrapper shape."""
        results = getattr(pipeline_result, "results", pipeline_result)
        return results if isinstance(results, list) else []

    def _estimate_pipeline_result_tokens(self, pipeline_result: Any) -> int:
        """Estimate the token cost of injecting raw tool outputs into context."""
        total_chars = 0
        for result in self._iter_pipeline_results(pipeline_result):
            if isinstance(result, dict):
                payload = result.get("result")
                error = result.get("error")
            else:
                payload = getattr(result, "result", None)
                error = getattr(result, "error", None)
            if payload is not None:
                total_chars += len(str(payload))
            if error:
                total_chars += len(str(error))
        return max(0, total_chars // 4)

    @staticmethod
    def _get_latest_compaction_summary(runtime: Any) -> str:
        """Read the latest compaction summary from the conversation controller."""
        controller = getattr(runtime, "_conversation_controller", None) or getattr(
            runtime, "conversation_controller", None
        )
        getter = getattr(controller, "get_compaction_summaries", None)
        if not callable(getter):
            return ""
        try:
            summaries = getter() or []
        except Exception as exc:
            logger.debug("Failed to read latest compaction summary: %s", exc)
            return ""
        if not summaries:
            return ""
        return str(summaries[-1])

    @staticmethod
    def _record_tool_intents(runtime: Any, tool_calls: List[Dict[str, Any]]) -> None:
        """Record planned tool actions in the stream continuation ledger."""
        stream_ctx = getattr(runtime, "_current_stream_context", None)
        if stream_ctx is None or not hasattr(stream_ctx, "record_intent_event"):
            return

        for tool_call in tool_calls[:6]:
            if not isinstance(tool_call, dict):
                continue
            tool_name = str(tool_call.get("name", "") or "")
            if not tool_name:
                continue
            arguments = tool_call.get("arguments", {}) or {}
            arg_preview = ""
            if isinstance(arguments, dict):
                for key in ("path", "command", "query", "pattern", "file"):
                    if key in arguments:
                        arg_preview = f"{key}={arguments[key]}"
                        break
            summary = f"planned {tool_name}"
            if arg_preview:
                summary += f" ({arg_preview})"
            stream_ctx.record_intent_event("tool_intent", summary, tool=tool_name)

    @staticmethod
    def _record_tool_results(runtime: Any, results: List[Dict[str, Any]]) -> None:
        """Record completed tool outcomes in the continuation ledger."""
        stream_ctx = getattr(runtime, "_current_stream_context", None)
        if stream_ctx is None or not hasattr(stream_ctx, "record_intent_event"):
            return
        execution_checkpoint = getattr(runtime, "_last_execution_checkpoint", None)

        for result in results[:8]:
            if not isinstance(result, dict):
                continue
            tool_name = str(result.get("name", "") or "")
            if not tool_name:
                continue
            status = "ok" if result.get("success") else "failed"
            summary = f"{tool_name} {status}"
            if isinstance(result.get("args"), dict):
                path = result["args"].get("path")
                if path:
                    summary += f" ({path})"
            metadata: Dict[str, Any] = {"tool": tool_name}
            if _tool_result_matches_checkpoint(result, execution_checkpoint):
                metadata.update(execution_checkpoint.to_trace_metadata())
            stream_ctx.record_intent_event("tool_result", summary, **metadata)

    @staticmethod
    def _ensure_tool_response_coverage(
        runtime: Any,
        tool_calls: List[Dict[str, Any]],
        results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Backfill missing provider-visible tool responses for dropped call IDs."""
        expected_by_id = {
            tool_call.get("id"): tool_call
            for tool_call in tool_calls
            if isinstance(tool_call, dict) and tool_call.get("id")
        }
        if not expected_by_id:
            return results

        observed_result_ids = {
            result.get("tool_call_id")
            for result in results
            if isinstance(result, dict) and result.get("tool_call_id")
        }

        conversation = getattr(runtime, "conversation", None)
        conversation_messages = getattr(conversation, "_messages", None)
        message_by_id: Dict[str, Any] = {}
        if conversation_messages:
            for message in conversation_messages:
                if getattr(message, "role", None) != "tool":
                    continue
                tool_call_id = getattr(message, "tool_call_id", None)
                if tool_call_id:
                    message_by_id[tool_call_id] = message

        missing_ids = [
            tool_call_id
            for tool_call_id in expected_by_id
            if tool_call_id not in observed_result_ids and tool_call_id not in message_by_id
        ]
        if missing_ids:
            logger.error(
                "Tool response coverage gap detected for %d call(s): %s",
                len(missing_ids),
                missing_ids,
            )

        for tool_call_id in missing_ids:
            tool_call = expected_by_id[tool_call_id]
            tool_name = tool_call.get("name", "tool")
            tool_args = tool_call.get("arguments", {}) or {}
            fallback_content = (
                f"Tool result unavailable for '{tool_name}'. Victor did not complete "
                "post-processing for this tool call, so treat it as failed and continue "
                "with the available context."
            )
            try:
                runtime.add_message(
                    "tool",
                    fallback_content,
                    name=tool_name,
                    tool_call_id=tool_call_id,
                    persist_synchronously=True,
                )
            except TypeError:
                try:
                    runtime.add_message(
                        "tool",
                        fallback_content,
                        name=tool_name,
                        tool_call_id=tool_call_id,
                    )
                except Exception:
                    logger.exception(
                        "Failed to persist fallback tool response for %s (tool_call_id=%s)",
                        tool_name,
                        tool_call_id,
                    )
            except Exception:
                logger.exception(
                    "Failed to persist fallback tool response for %s (tool_call_id=%s)",
                    tool_name,
                    tool_call_id,
                )
            results.append(
                {
                    "name": tool_name,
                    "success": False,
                    "elapsed": 0.0,
                    "args": tool_args,
                    "error": "Tool response coverage gap",
                    "result": fallback_content,
                    "full_result": fallback_content,
                    "follow_up_suggestions": None,
                    "was_pruned": False,
                    "tool_call_id": tool_call_id,
                    "content": fallback_content,
                    "skipped": True,
                    "outcome_kind": "tool_response_missing",
                    "block_source": "tool_result_postprocessing",
                    "retryable": False,
                    "user_message": fallback_content,
                }
            )

        for tool_call_id, message in message_by_id.items():
            if tool_call_id in observed_result_ids or tool_call_id not in expected_by_id:
                continue
            message_content = getattr(message, "content", "")
            results.append(
                {
                    "name": getattr(message, "tool_name", None)
                    or getattr(message, "name", None)
                    or expected_by_id[tool_call_id].get("name", "tool"),
                    "success": False,
                    "elapsed": 0.0,
                    "args": expected_by_id[tool_call_id].get("arguments", {}) or {},
                    "error": "Tool response recovered from conversation history",
                    "result": message_content,
                    "full_result": message_content,
                    "follow_up_suggestions": None,
                    "was_pruned": False,
                    "tool_call_id": tool_call_id,
                    "content": message_content,
                    "skipped": True,
                    "outcome_kind": "tool_response_recovered",
                    "block_source": "conversation_history",
                    "retryable": False,
                    "user_message": message_content,
                }
            )

        return results
