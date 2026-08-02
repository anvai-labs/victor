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
import inspect
from typing import Any, AsyncIterator, Mapping, Optional
from unittest.mock import Mock

from victor.core.errors import CancellationError
from victor.framework._internal import format_context_message
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

        return CompletionResponse(
            content=content or "",
            role=role or "assistant",
            tool_calls=tool_calls or [],
            usage=usage,
            stop_reason=stop_reason or "stop",
            model=model,
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


def _build_awaiting_task_result(orchestrator: Any, pause: Any) -> TaskResult:
    """Convert an :class:`ApprovalPause` into an ``awaiting_approval`` TaskResult (FEP-0029).

    Records a resumable pause in the process-local paused-run store (Phase 1) and returns the
    ``run_id`` + pending ``approval_request`` on the result so the caller can resume later.
    """
    import time

    from victor.agent.paused_run_store import record_pause_from_approval

    run_id, req_dict = record_pause_from_approval(
        getattr(pause, "request", None),
        session_id=getattr(orchestrator, "active_session_id", None),
        agent_id=getattr(orchestrator, "agent_id", None) or getattr(orchestrator, "model", None),
        created_at=time.time(),
        metadata={"stage": _resolve_stage_value(orchestrator)},
    )
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


def _supports_keyword_argument(callable_obj: Any, argument: str) -> bool:
    """Return whether a callable accepts a given keyword or arbitrary kwargs."""
    if isinstance(callable_obj, Mock):
        return True

    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return False

    for parameter in signature.parameters.values():
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            return True

    return argument in signature.parameters


async def _invoke_chat(
    runtime: Any,
    message: str,
    *,
    stream: bool = False,
    forward_stream_option: bool = False,
    runtime_context_overrides: Optional[Mapping[str, Any]] = None,
) -> Any:
    """Invoke the runtime chat entrypoint with compatibility-aware kwargs."""
    chat_callable = runtime.chat
    kwargs = {}

    if runtime_context_overrides and _supports_keyword_argument(
        chat_callable, "runtime_context_overrides"
    ):
        kwargs["runtime_context_overrides"] = dict(runtime_context_overrides)

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
            ),
            default_model=getattr(orchestrator, "model", "unknown"),
        )
        response_content = output_state.normalize_final_response(response.content or "")

        return TaskResult(
            content=response_content,
            tool_calls=response.tool_calls or [],
            success=True,
            error=None,
            metadata=_build_task_result_metadata(orchestrator, response),
        )

    except ApprovalPause as pause:
        # FEP-0029: the turn parked on a policy ASK. Record a resumable pause and surface it.
        return _build_awaiting_task_result(orchestrator, pause)
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
) -> AsyncIterator[Any]:
    """Yield framework stream events for a single message turn.

    FEP-0029: when durable approval is armed (``governance.durable``), a policy ASK mid-stream raises
    :class:`ApprovalPause`, which is caught here — the pause is recorded (shared
    ``record_pause_from_approval``) and an ``AWAITING_APPROVAL`` event carrying the resume ``run_id`` +
    ``approval_request`` is yielded before the stream ends, so streaming callers (TUI, SSE) can render
    a paused lane and resume. Disarmed, this is byte-identical to before.
    """
    import time

    from victor.agent.paused_run_store import record_pause_from_approval
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
        ):
            yield event
    except ApprovalPause as pause:
        run_id, req_dict = record_pause_from_approval(
            getattr(pause, "request", None),
            session_id=getattr(orchestrator, "active_session_id", None),
            agent_id=getattr(orchestrator, "agent_id", None)
            or getattr(orchestrator, "model", None),
            created_at=time.time(),
            metadata={"stage": _resolve_stage_value(orchestrator)},
        )
        yield awaiting_approval_event(run_id, req_dict)
    finally:
        if _token is not None:
            current_durable_pause_enabled.reset(_token)


async def iter_runtime_stream_events(
    runtime: Any,
    message: str,
) -> AsyncIterator[Any]:
    """Yield framework stream events from either a chat runtime or agent wrapper."""
    from victor.framework._internal import stream_with_events

    if hasattr(runtime, "stream_chat"):
        async for event in stream_with_events(runtime, message, response_prompt=message):
            yield event
        return

    async for event in runtime.stream(message):
        yield event


__all__ = [
    "PreparedMessage",
    "prepare_message",
    "execute_message",
    "stream_message_events",
    "iter_runtime_stream_events",
]
