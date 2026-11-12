"""Shared framework message execution helpers.

This module centralizes the framework-first execution contract for a single
message turn:
- prepare prompt/context payloads
- resolve the canonical service-backed chat runtime
- normalize runtime responses into TaskResult
- expose a consistent streaming-event iterator

The goal is to keep framework entry points such as Agent and VictorClient
aligned on one execution shape while the lower-level agent/services runtime
continues to own the actual orchestration work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, Mapping, Optional

from victor.core.errors import CancellationError
from victor.framework._internal import _supports_keyword_argument, format_context_message
from victor.framework.task import DirectResponseOutputState, TaskResult
from victor.providers.base import CompletionResponse
from victor.runtime.chat_runtime import resolve_chat_runtime as _resolve_chat_runtime


@dataclass(frozen=True)
class PreparedMessage:
    """Canonical prompt payload for a single framework message turn."""

    runtime_message: str
    response_message: str


def prepare_message(
    user_message: str,
    context: Optional[Mapping[str, Any]] = None,
) -> PreparedMessage:
    """Prepare runtime and response messages from user input plus optional context."""
    runtime_message = user_message

    if context:
        context_message = format_context_message(dict(context))
        if context_message:
            runtime_message = f"{context_message}\n\n{user_message}"

    return PreparedMessage(
        runtime_message=runtime_message,
        response_message=user_message,
    )


def _coerce_completion_response(
    result: Any,
    *,
    default_model: Optional[str],
) -> CompletionResponse:
    """Normalize runtime chat results into CompletionResponse."""
    if isinstance(result, CompletionResponse):
        return result

    if isinstance(result, Mapping) or hasattr(result, "content"):
        content = (
            result.get("content", "")
            if isinstance(result, Mapping)
            else getattr(result, "content", "")
        )
        role = (
            result.get("role", "assistant")
            if isinstance(result, Mapping)
            else getattr(result, "role", "assistant")
        )
        tool_calls = (
            result.get("tool_calls", [])
            if isinstance(result, Mapping)
            else getattr(result, "tool_calls", [])
        )
        usage = (
            result.get("usage") if isinstance(result, Mapping) else getattr(result, "usage", None)
        )
        stop_reason = (
            result.get("stop_reason", "stop")
            if isinstance(result, Mapping)
            else getattr(result, "stop_reason", "stop")
        )
        model = (
            result.get("model", default_model)
            if isinstance(result, Mapping)
            else getattr(result, "model", default_model)
        )
        metadata = (
            result.get("metadata")
            if isinstance(result, Mapping)
            else getattr(result, "metadata", None)
        )

        return CompletionResponse(
            content=content or "",
            role=role or "assistant",
            tool_calls=tool_calls or [],
            usage=usage,
            stop_reason=stop_reason or "stop",
            model=model,
            metadata=dict(metadata) if isinstance(metadata, Mapping) else None,
        )

    return CompletionResponse(
        content=str(result),
        model=default_model,
        tool_calls=[],
        usage=None,
        stop_reason="stop",
    )


def _resolve_stage_value(orchestrator: Any) -> str:
    """Best-effort stage value extraction for TaskResult metadata."""
    get_stage = getattr(orchestrator, "get_stage", None)
    if callable(get_stage):
        try:
            stage = get_stage()
            return getattr(stage, "value", str(stage))
        except Exception:
            pass

    stage = getattr(orchestrator, "current_stage", None)
    if stage is not None:
        return getattr(stage, "value", str(stage))

    return "unknown"


def _durable_pause_enabled(orchestrator: Any) -> bool:
    """Whether durable pause (FEP-0029) is opted in for this run (governance.durable)."""
    settings = getattr(orchestrator, "settings", None)
    governance = getattr(settings, "governance", None)
    return bool(getattr(governance, "durable", False))


async def _build_awaiting_task_result(orchestrator: Any, pause: Any) -> TaskResult:
    """Convert an :class:`ApprovalPause` into an ``awaiting_approval`` TaskResult (FEP-0029).

    Persists a resumable pause before returning the
    ``run_id`` + pending ``approval_request`` on the result so the caller can resume later.
    """
    import time

    from victor.agent.paused_run_store import record_pause_from_approval_async

    owner_fields = ("active_session_id", "agent_id", "model")
    owner = tuple(getattr(orchestrator, name, None) for name in owner_fields)
    run_id, req_dict = await record_pause_from_approval_async(
        getattr(pause, "request", None),
        session_id=owner[0],
        agent_id=owner[1] or owner[2],
        created_at=time.time(),
        metadata={"stage": _resolve_stage_value(orchestrator)},
    )
    if owner != tuple(getattr(orchestrator, name, None) for name in owner_fields):
        raise PermissionError("Approval pause ownership changed during persistence")
    return TaskResult(
        content="",
        tool_calls=[],
        success=False,
        error=None,
        metadata={"stage": _resolve_stage_value(orchestrator)},
        status="awaiting_approval",
        run_id=run_id,
        approval_request=req_dict,
    )


def _coalesce_value(*values: Any) -> Any:
    """Return the first value that is not ``None``."""
    for value in values:
        if value is not None:
            return value
    return None


def _safe_int(value: Any, default: int = 0) -> int:
    """Coerce a value to int without raising."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Coerce a value to float without raising."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _get_last_task_report(orchestrator: Any) -> Optional[dict[str, Any]]:
    """Best-effort retrieval of the canonical per-task report."""
    getter = getattr(orchestrator, "get_last_task_report", None)
    if not callable(getter):
        return None

    try:
        report = getter()
    except Exception:
        return None

    if not isinstance(report, Mapping):
        return None

    return dict(report)


def _build_task_result_metadata(
    orchestrator: Any,
    response: CompletionResponse,
) -> dict[str, Any]:
    """Build framework TaskResult metadata from provider usage plus task report data."""
    usage = response.usage if isinstance(response.usage, Mapping) else {}
    prompt_details = (
        usage.get("prompt_tokens_details", {})
        if isinstance(usage.get("prompt_tokens_details", {}), Mapping)
        else {}
    )
    completion_details = (
        usage.get("completion_tokens_details", {})
        if isinstance(usage.get("completion_tokens_details", {}), Mapping)
        else {}
    )
    task_report = _get_last_task_report(orchestrator)
    resp_meta = getattr(response, "metadata", None) or {}

    def report_value(key: str) -> Any:
        if task_report is None or key not in task_report:
            return None
        return task_report.get(key)

    total_cost_usd = report_value("total_cost_usd")
    cost_usd_micros = _coalesce_value(
        (
            _safe_int(round(_safe_float(total_cost_usd) * 1_000_000))
            if total_cost_usd is not None
            else None
        ),
        usage.get("cost_usd_micros"),
        usage.get("cost_in_usd_ticks"),
    )

    metadata = {
        "stage": _resolve_stage_value(orchestrator),
        "model": response.model,
        "usage": response.usage,
        "stop_reason": response.stop_reason,
        "tokens_input": _safe_int(
            _coalesce_value(report_value("api_prompt_tokens"), usage.get("prompt_tokens"))
        ),
        "tokens_output": _safe_int(
            _coalesce_value(report_value("api_completion_tokens"), usage.get("completion_tokens"))
        ),
        "tokens_used": _safe_int(
            _coalesce_value(report_value("api_total_tokens"), usage.get("total_tokens"))
        ),
        "turns": _safe_int(report_value("request_count")),
        # Real agentic-loop iteration count (excludes rubric-judge/recovery sub-calls); see
        # TurnExecutor.execute_agentic_loop. Used by the A/B harnesses for an honest turn metric.
        "agentic_loop_iterations": _safe_int(resp_meta.get("agentic_loop_iterations")),
        "cached_tokens": _safe_int(
            _coalesce_value(
                report_value("cache_read_tokens"),
                usage.get("cached_tokens"),
                prompt_details.get("cached_tokens"),
            )
        ),
        "cache_hit_rate": _safe_float(report_value("cache_hit_rate")),
        "reasoning_tokens": _safe_int(
            _coalesce_value(
                usage.get("reasoning_tokens"),
                completion_details.get("reasoning_tokens"),
            )
        ),
        "cost_usd_micros": _safe_int(cost_usd_micros),
        "tool_schema_tokens": _safe_int(report_value("tool_schema_tokens")),
        "compaction_saved_tokens": _safe_int(report_value("compaction_saved_tokens")),
        "compaction_messages_removed": _safe_int(report_value("compaction_messages_removed")),
    }

    if task_report is not None:
        metadata["task_report"] = task_report

    return metadata


async def _invoke_chat(
    runtime: Any,
    message: str,
    *,
    stream: bool = False,
    forward_stream_option: bool = False,
    runtime_context_overrides: Optional[Mapping[str, Any]] = None,
    run_context: Optional[Mapping[str, Any]] = None,
    caller_run_context: Optional[Any] = None,
) -> Any:
    """Invoke the runtime chat entrypoint with compatibility-aware kwargs."""
    chat_callable = runtime.chat
    kwargs = {}

    # Consolidation PR 4: forward the structured CallerRunContext object to
    # runtimes that accept it; legacy runtimes drop it silently. The legacy
    # dict kwargs below keep forwarding for the same compatibility reason.
    if caller_run_context is not None and _supports_keyword_argument(
        chat_callable, "caller_run_context"
    ):
        kwargs["caller_run_context"] = caller_run_context

    if runtime_context_overrides and _supports_keyword_argument(
        chat_callable, "runtime_context_overrides"
    ):
        kwargs["runtime_context_overrides"] = dict(runtime_context_overrides)

    if run_context and _supports_keyword_argument(chat_callable, "run_context"):
        # FEP-0043: forward the structured caller context alongside the
        # flattened prompt text so the loop's state can seed it (criteria
        # gating). Runtimes that predate the keyword are untouched.
        kwargs["run_context"] = dict(run_context)

    if forward_stream_option and _supports_keyword_argument(chat_callable, "stream"):
        kwargs["stream"] = stream

    return await chat_callable(message, **kwargs)


async def execute_message(
    *,
    orchestrator: Any,
    execution_context: Any = None,
    user_message: str,
    context: Optional[Mapping[str, Any]] = None,
    stream: bool = False,
    forward_stream_option: bool = False,
    runtime_context_overrides: Optional[Mapping[str, Any]] = None,
    caller_run_context: Optional[Any] = None,
) -> TaskResult:
    """Execute a single message turn via the canonical service-backed runtime.

    FEP-0029: when the session opts into durable approval (``governance.durable``), durable pause is
    armed for the turn — a policy ASK raises :class:`ApprovalPause`, which is caught here and
    converted into an ``awaiting_approval`` :class:`TaskResult` (with a resumable ``run_id``) instead
    of blocking the inline modal. Disarmed, this is byte-identical to before.
    """
    from victor.framework.approval_pause import ApprovalPause, current_durable_pause_enabled

    prepared = prepare_message(user_message, context)

    _durable = _durable_pause_enabled(orchestrator)
    _token = current_durable_pause_enabled.set(True) if _durable else None
    try:
        chat_runtime = _resolve_chat_runtime(orchestrator, execution_context)
        output_state = DirectResponseOutputState(prepared.response_message)
        response = _coerce_completion_response(
            await _invoke_chat(
                chat_runtime,
                prepared.runtime_message,
                stream=stream,
                forward_stream_option=forward_stream_option,
                runtime_context_overrides=runtime_context_overrides,
                run_context=context,
                caller_run_context=caller_run_context,
            ),
            default_model=getattr(orchestrator, "model", "unknown"),
        )
        response_content = output_state.normalize_final_response(response.content or "")
        response_metadata = response.metadata or {}
        success = response_metadata.get("agentic_loop_success", True) is True
        error = (
            None
            if success
            else str(
                response_metadata.get("agentic_loop_error")
                or "Agentic loop did not complete successfully"
            )
        )
        calls = response_metadata.get("agentic_loop_tool_calls", response.tool_calls or [])
        tool_calls = []
        if "agentic_loop_tool_calls" in response_metadata:
            invalid = not isinstance(calls, list)
            for call in calls if isinstance(calls, list) else []:
                if (
                    not isinstance(call, dict)
                    or not isinstance(call.get("tool"), str)
                    or not isinstance(call.get("arguments"), Mapping)
                ):
                    invalid = True
                    continue
                fact = dict(call)
                fact["arguments"] = dict(call["arguments"])
                if type(fact.get("success")) is not bool:
                    invalid = True
                fact["success"] = fact.get("success") is True
                tool_calls.append(fact)
            if invalid:
                success, error = False, "Invalid run tool-call evidence"
        else:
            tool_calls = [dict(call) for call in calls]

        return TaskResult(
            content=response_content,
            tool_calls=tool_calls,
            success=success,
            error=error,
            metadata=_build_task_result_metadata(orchestrator, response),
        )

    except ApprovalPause as pause:
        # FEP-0029: the turn parked on a policy ASK. Record a resumable pause and surface it.
        return await _build_awaiting_task_result(orchestrator, pause)
    except CancellationError:
        return TaskResult(
            content="",
            tool_calls=[],
            success=False,
            error="Operation cancelled",
            metadata={"stage": _resolve_stage_value(orchestrator)},
        )
    except Exception as exc:
        return TaskResult(
            content="",
            tool_calls=[],
            success=False,
            error=str(exc),
            metadata={"stage": _resolve_stage_value(orchestrator)},
        )
    finally:
        if _token is not None:
            current_durable_pause_enabled.reset(_token)


async def stream_message_events(
    *,
    orchestrator: Any,
    execution_context: Any = None,
    user_message: str,
    context: Optional[Mapping[str, Any]] = None,
    runtime_context_overrides: Optional[Mapping[str, Any]] = None,
    caller_run_context: Optional[Any] = None,
) -> AsyncIterator[Any]:
    """Yield framework stream events for a single message turn.

    FEP-0029: when durable approval is armed (``governance.durable``), a policy ASK mid-stream raises
    :class:`ApprovalPause`, which is caught here — the pause is recorded (shared
    ``record_pause_from_approval_async``) and an ``AWAITING_APPROVAL`` event carrying the resume ``run_id`` +
    ``approval_request`` is yielded before the stream ends, so streaming callers (TUI, SSE) can render
    a paused lane and resume. Disarmed, this is byte-identical to before.

    FEP-0043 streaming parity: ``context`` is forwarded as the structured
    ``run_context`` (buffered ``execute_message`` contract) and
    ``runtime_context_overrides`` as the scoped runtime hints; both reach the
    chat-stream seam only when the runtime accepts them.
    """
    from victor.framework._internal import stream_with_events
    from victor.framework.approval_pause import ApprovalPause, current_durable_pause_enabled
    from victor.framework.events import awaiting_approval_event

    prepared = prepare_message(user_message, context)
    chat_runtime = _resolve_chat_runtime(orchestrator, execution_context)

    _durable = _durable_pause_enabled(orchestrator)
    _token = current_durable_pause_enabled.set(True) if _durable else None
    try:
        async for event in stream_with_events(
            chat_runtime,
            prepared.runtime_message,
            response_prompt=prepared.response_message,
            runtime_context_overrides=runtime_context_overrides,
            run_context=context,
            caller_run_context=caller_run_context,
        ):
            yield event
    except ApprovalPause as pause:
        result = await _build_awaiting_task_result(orchestrator, pause)
        yield awaiting_approval_event(result.run_id, result.approval_request)
    finally:
        if _token is not None:
            current_durable_pause_enabled.reset(_token)


async def iter_runtime_stream_events(
    runtime: Any,
    message: str,
    *,
    runtime_context_overrides: Optional[Mapping[str, Any]] = None,
    run_context: Optional[Mapping[str, Any]] = None,
    caller_run_context: Optional[Any] = None,
) -> AsyncIterator[Any]:
    """Yield framework stream events from either a chat runtime or agent wrapper.

    FEP-0043 streaming parity: the caller overrides/context are forwarded when
    the underlying runtime accepts them; legacy runtimes drop both silently
    (signature sniffing).
    """
    from victor.framework._internal import stream_with_events

    if hasattr(runtime, "stream_chat"):
        async for event in stream_with_events(
            runtime,
            message,
            response_prompt=message,
            runtime_context_overrides=runtime_context_overrides,
            run_context=run_context,
            caller_run_context=caller_run_context,
        ):
            yield event
        return

    stream_callable = runtime.stream
    forward: Dict[str, Any] = {}
    if runtime_context_overrides and _supports_keyword_argument(
        stream_callable, "runtime_context_overrides"
    ):
        forward["runtime_context_overrides"] = dict(runtime_context_overrides)
    if caller_run_context is not None and _supports_keyword_argument(
        stream_callable, "caller_run_context"
    ):
        forward["caller_run_context"] = caller_run_context
    async for event in stream_callable(message, **forward):
        yield event


__all__ = [
    "PreparedMessage",
    "prepare_message",
    "execute_message",
    "stream_message_events",
    "iter_runtime_stream_events",
]
