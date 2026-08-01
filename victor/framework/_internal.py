"""Internal helpers mapping framework API to orchestrator.

This module contains implementation details that bridge the simplified
framework API to the existing AgentOrchestrator internals. Users should
not import from this module directly.

Phase 7.6 Refactoring:
- configure_tools() now delegates to ToolConfigurator
- stream_with_events() uses EventRegistry for conversion
- setup_observability_integration() uses AgentBridge patterns

Phase 8.0 - Vertical Integration Unification:
- Added vertical parameter to create_orchestrator_from_options()
- Uses VerticalIntegrationPipeline for consistent vertical application
- Achieves parity with FrameworkShim CLI path

Phase 4 - Agent Creation Unification:
- create_orchestrator_from_options() now delegates to OrchestratorFactory.create_agent()
- Ensures consistent code maintenance and eliminates code proliferation
- All agent creation paths use same factory infrastructure (SOLID SRP, DIP)
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, List, Optional, Tuple, Type, Union

from victor.agent.config import FrameworkCompatibleAgentConfig, normalize_agent_config
from victor.core.completion_markers import strip_active_completion_markers
from victor.core.errors import VictorError
from victor.framework.event_registry import EventTarget, get_event_registry
from victor.framework.events import (
    AgentExecutionEvent,
    EventType,
    error_event,
    member_event,
    milestone_event,
    stream_end_event,
    stream_start_event,
)
from victor.framework.member_event_sink import (
    MemberEvent,
    MemberEventSink,
    current_member_sink,
    is_close_sentinel,
)
from victor.framework.task import DirectResponseOutputState
from victor.framework.tools import ToolSet
from victor.framework.protocols import ObservabilityPortProtocol

# Import shared capability helpers for protocol-based access
from victor.framework.capability_runtime import (
    check_capability as _check_capability,
    invoke_capability as _invoke_capability,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from victor.core.verticals.base import VerticalBase


def _sanitize_stream_error_message(error: Exception) -> str:
    """Return a bounded error string suitable for streaming UI surfaces."""
    # Structured Victor errors already carry a clean, user-facing message plus an
    # actionable recovery hint — prefer them over the raw HTTP response body
    # (which for e.g. a missing model is an opaque JSON 404 payload).
    if isinstance(error, VictorError):
        parts = [error.message.strip()] if error.message and error.message.strip() else []
        hint = getattr(error, "recovery_hint", None)
        if isinstance(hint, str) and hint.strip():
            parts.append(f"💡 {hint.strip()}")
        if parts:
            combined = "\n".join(parts)
            return combined if len(combined) <= 1200 else f"{combined[:1200].rstrip()}..."

    response = getattr(error, "response", None)
    status_code = getattr(response, "status_code", None)
    response_text = getattr(response, "text", None)
    raw_message = response_text if isinstance(response_text, str) and response_text else str(error)
    message = raw_message.strip() or f"{type(error).__name__}: {error!r}"
    lowered = message.lower()

    if "<html" in lowered or "<body" in lowered or "<!doctype html" in lowered:
        status = f" (HTTP {status_code})" if isinstance(status_code, int) else ""
        return (
            "Provider returned an HTML authentication page"
            f"{status}. Credentials may be expired, missing, or not accepted by the endpoint."
        )

    max_len = 1000
    if len(message) > max_len:
        return f"{message[:max_len].rstrip()}..."
    return message


async def create_orchestrator_from_options(
    provider: str,
    model: Optional[str],
    temperature: float,
    max_tokens: int,
    tools: Union[ToolSet, List[str], None],
    thinking: bool,
    airgapped: bool,
    profile: Optional[str],
    workspace: Optional[str],
    config: Optional[FrameworkCompatibleAgentConfig],
    system_prompt: Optional[str] = None,
    enable_observability: bool = True,
    session_id: Optional[str] = None,
    vertical: Optional[Union[Type["VerticalBase"], str]] = None,
) -> Any:
    """Create an AgentOrchestrator from framework options.

    Phase 4 Refactoring:
    Now delegates to OrchestratorFactory.create_agent() for unified
    agent creation, ensuring consistent code maintenance and eliminating
    code proliferation (SOLID SRP, DIP).

    This function bridges the simplified framework API to the
    internal orchestrator creation logic.

    Args:
        provider: LLM provider name
        model: Model identifier
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        tools: Tool configuration
        thinking: Enable extended thinking
        airgapped: Air-gapped mode
        profile: Profile name from profiles.yaml
        workspace: Working directory
        config: Advanced configuration
        system_prompt: Optional custom system prompt (e.g., from a vertical)
        enable_observability: Whether to auto-initialize ObservabilityIntegration
        session_id: Optional session ID for event correlation
        vertical: Optional vertical class or name to apply

    Returns:
        Configured AgentOrchestrator instance
    """
    from victor.agent.orchestrator import AgentOrchestrator
    from victor.agent.orchestrator_factory import OrchestratorFactory
    from victor.config.settings import load_settings
    from victor.core.bootstrap import ensure_bootstrapped
    from victor.providers.registry import ProviderRegistry

    # Load settings
    settings = load_settings()

    config = normalize_agent_config(config)

    # Apply config overrides
    if config:
        for key, value in config.to_settings_dict().items():
            if hasattr(settings, key):
                setattr(settings, key, value)

    # Apply airgapped mode
    if airgapped:
        settings.security.airgapped_mode = True

    if getattr(settings, "framework_private_fallback_strict_mode", False):
        os.environ.setdefault("VICTOR_STRICT_FRAMEWORK_PRIVATE_FALLBACKS", "1")
    if getattr(settings, "framework_protocol_fallback_strict_mode", False):
        os.environ.setdefault("VICTOR_STRICT_FRAMEWORK_PROTOCOL_FALLBACKS", "1")

    # Resolve vertical name for bootstrap
    vertical_name = None
    if vertical:
        if isinstance(vertical, str):
            vertical_name = vertical
        elif hasattr(vertical, "name"):
            vertical_name = vertical.name

    # Bootstrap with vertical context BEFORE orchestrator creation
    # This ensures vertical services are registered with correct vertical name
    ensure_bootstrapped(settings, vertical=vertical_name)

    # Create OrchestratorFactory with provider
    provider_class = ProviderRegistry.get(provider)
    provider_instance = provider_class(
        model=model or settings.provider.default_model,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    factory = OrchestratorFactory(
        settings=settings,
        provider=provider_instance,
        model=model or settings.provider.default_model,
        temperature=temperature,
        max_tokens=max_tokens,
        profile_name=profile or "default",
        thinking=thinking,
    )

    # Use factory's create_agent() method for unified agent creation
    # This delegates to existing factory infrastructure (SOLID DIP)
    agent = await factory.create_agent(
        mode="foreground",
        provider=provider,
        model=model or settings.provider.default_model,
        tools=tools,
        airgapped=airgapped,
        vertical=vertical,
        system_prompt=system_prompt,
        enable_observability=enable_observability,
        session_id=session_id,
    )

    # Apply vertical configuration using unified pipeline
    # This achieves parity with FrameworkShim CLI path
    if vertical:
        apply_vertical_to_orchestrator(agent._orchestrator, vertical)

    # Auto-initialize observability integration
    if enable_observability:
        setup_observability_integration(agent._orchestrator, session_id)

    # Return the agent's orchestrator for backward compatibility
    # Note: This maintains the existing API while delegating to factory
    return agent._orchestrator


def apply_vertical_to_orchestrator(
    orchestrator: Any,
    vertical: Union[Type["VerticalBase"], str],
) -> None:
    """Apply vertical configuration to orchestrator using integration pipeline.

    This provides parity with FrameworkShim by using the same
    VerticalIntegrationPipeline for both SDK and CLI paths.

    Args:
        orchestrator: AgentOrchestrator instance
        vertical: Vertical class or name string
    """
    from victor.framework.vertical_service import apply_vertical_configuration

    apply_vertical_configuration(orchestrator, vertical, source="sdk")


def setup_observability_integration(
    orchestrator: Any,
    session_id: Optional[str] = None,
) -> Any:
    """Set up ObservabilityIntegration for the orchestrator.

    This wires the EventBus into the orchestrator for unified event handling.
    Used by both the SDK path (Agent.create) and the CLI path (FrameworkShim).

    Args:
        orchestrator: AgentOrchestrator instance
        session_id: Optional session ID for event correlation

    Returns:
        ObservabilityIntegration instance
    """
    from victor.observability.integration import ObservabilityIntegration

    # Create integration with optional session ID
    integration = ObservabilityIntegration(
        session_id=session_id,
    )

    # Wire into orchestrator
    integration.wire_orchestrator(orchestrator)

    # Wire provider resilience notifications to the same bus so dashboard
    # and other subscribers see fallback events.  Follows the same pattern
    # as CircuitBreakerRegistry.wire_observability().
    try:
        from victor.providers.resilience import ResilientProvider

        ResilientProvider.wire_observability(integration.event_bus)
    except Exception:
        pass  # Non-critical — resilience works without observability

    # Ensure reference is stored through public observability ports.
    if isinstance(orchestrator, ObservabilityPortProtocol):
        orchestrator.set_observability(integration)
    elif hasattr(orchestrator, "set_observability") and callable(orchestrator.set_observability):
        orchestrator.set_observability(integration)
    elif hasattr(orchestrator, "observability"):
        orchestrator.observability = integration

    return integration


def apply_system_prompt(orchestrator: Any, system_prompt: str) -> None:
    """Apply a custom system prompt to the orchestrator.

    SOLID Compliance (DIP): This function only uses public methods.
    It never writes to private attributes to maintain proper encapsulation
    and dependency inversion.

    Args:
        orchestrator: AgentOrchestrator instance
        system_prompt: Custom system prompt text
    """
    import logging

    logger = logging.getLogger(__name__)

    # Use capability-based approach (protocol-first, fallback to hasattr)
    # SOLID Compliance (DIP): Only use public methods, never write to private attributes
    if _check_capability(orchestrator, "custom_prompt"):
        _invoke_capability(orchestrator, "custom_prompt", system_prompt)
        logger.debug("Applied system prompt via custom_prompt capability")
    elif _check_capability(orchestrator, "prompt_builder"):
        # Fallback: try direct prompt builder access via public method only
        prompt_builder = getattr(orchestrator, "prompt_builder", None)
        if prompt_builder:
            if hasattr(prompt_builder, "set_custom_prompt"):
                prompt_builder.set_custom_prompt(system_prompt)
                logger.debug("Applied system prompt via prompt_builder.set_custom_prompt")
            else:
                logger.warning(
                    "Cannot set custom prompt: prompt_builder lacks set_custom_prompt method. "
                    "Consider implementing CapabilityRegistryProtocol."
                )
    else:
        logger.warning(
            "Cannot set custom prompt: orchestrator lacks custom_prompt capability "
            "and prompt_builder. Consider implementing CapabilityRegistryProtocol."
        )


def configure_tools(
    orchestrator: Any,
    tools: Union[ToolSet, List[str]],
    airgapped: bool = False,
) -> None:
    """Configure tools on the orchestrator using ToolConfigurator.

    This delegates to the ToolConfigurator from Phase 7.5 for proper
    tool configuration with support for filters and hooks.

    Args:
        orchestrator: AgentOrchestrator instance
        tools: Tool configuration (ToolSet or list of tool names)
        airgapped: Whether to apply airgapped filter
    """
    from victor.framework.tool_config import (
        AirgappedFilter,
        ToolConfigurator,
        ToolConfigMode,
        get_tool_configurator,
    )

    # Get or create configurator
    configurator = get_tool_configurator()

    # Add airgapped filter if in airgapped mode
    if airgapped:
        configurator.add_filter(AirgappedFilter())

    # Configure using ToolSet or list
    if isinstance(tools, ToolSet):
        # Use configure_from_toolset for ToolSet instances
        configurator.configure_from_toolset(orchestrator, tools)
    else:
        # Convert list to set and configure
        configurator.configure(orchestrator, set(tools), ToolConfigMode.REPLACE)


async def _merge_orchestrator_and_members(
    chunk_aiter: Any,
    sink: MemberEventSink,
) -> AsyncIterator[Tuple[str, Any]]:
    """Interleave orchestrator stream chunks with team member lifecycle events.

    Yields ``("chunk", chunk)`` for each orchestrator chunk and ``("member", MemberEvent)``
    for each team member event pushed onto ``sink`` (ADR-023). The orchestrator stream is the
    sole termination driver: when it is exhausted we close the sink, flush any member events
    still queued, and stop. Member emits are bounded/lossy (they never block the team), and on
    any error both pending waits are cancelled — so this can neither deadlock nor reorder the
    chunk sequence (a new chunk is requested only after the previous one is yielded).
    """
    ci = chunk_aiter.__aiter__()
    chunk_task: Optional["asyncio.Future[Any]"] = asyncio.ensure_future(ci.__anext__())
    member_task: Optional["asyncio.Future[Any]"] = asyncio.ensure_future(sink.get())
    try:
        while chunk_task is not None or member_task is not None:
            pending = {t for t in (chunk_task, member_task) if t is not None}
            done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)

            if member_task is not None and member_task in done:
                item = member_task.result()
                if not is_close_sentinel(item):
                    yield ("member", item)
                # Keep listening for more member events while the primary is live.
                member_task = asyncio.ensure_future(sink.get()) if chunk_task is not None else None

            if chunk_task is not None and chunk_task in done:
                try:
                    chunk = chunk_task.result()
                except StopAsyncIteration:
                    # Primary ended: close the sink, drain remaining member events, stop.
                    chunk_task = None
                    await sink.close()
                    while True:
                        if member_task is None:
                            member_task = asyncio.ensure_future(sink.get())
                        item = await member_task
                        member_task = None
                        if is_close_sentinel(item):
                            break
                        yield ("member", item)
                    break
                else:
                    yield ("chunk", chunk)
                    chunk_task = asyncio.ensure_future(ci.__anext__())
    finally:
        for task in (chunk_task, member_task):
            if task is not None and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task


async def stream_with_events(
    orchestrator: Any,
    prompt: str,
    *,
    response_prompt: Optional[str] = None,
) -> AsyncIterator[AgentExecutionEvent]:
    """Stream orchestrator response as framework AgentExecutionEvents.

    This function wraps the orchestrator's stream_chat method
    and converts internal stream chunks to framework AgentExecutionEvents.

    A per-turn :class:`MemberEventSink` is published on ``current_member_sink`` for the
    duration of the stream so a team running deep inside the turn can push member lifecycle
    events (ADR-023 / FEP-0028); they are interleaved with the orchestrator's own chunks by
    :func:`_merge_orchestrator_and_members`. With no team running, the sink stays empty and
    the emitted event sequence is byte-identical to the single-agent path.

    Args:
        orchestrator: AgentOrchestrator instance
        prompt: Prompt sent to the runtime
        response_prompt: Original user prompt for output normalization

    Yields:
        AgentExecutionEvent objects representing agent actions
    """
    registry = get_event_registry()
    output_state = DirectResponseOutputState(response_prompt or prompt)
    saw_content_event = False

    sink = MemberEventSink()
    sink_token = current_member_sink.set(sink)

    # Emit stream start
    yield stream_start_event()

    try:
        async for tag, item in _merge_orchestrator_and_members(
            orchestrator.stream_chat(prompt), sink
        ):
            if tag == "member":
                member_ev: MemberEvent = item
                yield member_event(
                    member_ev.kind,
                    member_ev.member_id,
                    content=member_ev.content,
                    success=member_ev.success,
                    index=member_ev.index,
                    formation=member_ev.formation,
                )
                continue

            chunk = item
            metadata = getattr(chunk, "metadata", None)
            chunk_metadata = metadata if isinstance(metadata, dict) else {}

            status_message = chunk_metadata.get("status") or chunk_metadata.get("status_update")
            if status_message:
                yield milestone_event(str(status_message), metadata=chunk_metadata)

            refusal_text = chunk_metadata.get("refusal")
            if refusal_text:
                yield milestone_event(f"Model refusal: {refusal_text}", metadata=chunk_metadata)

            reasoning_content = chunk_metadata.get("reasoning_content")
            if reasoning_content:
                yield registry.from_external(
                    {"reasoning_content": reasoning_content},
                    "reasoning_content",
                    EventTarget.STREAM_CHUNK,
                    metadata=chunk_metadata,
                )

            content = output_state.consume_stream_content(
                getattr(chunk, "content", ""),
                metadata=chunk_metadata,
            )
            if content:
                saw_content_event = True
                yield registry.from_external(
                    {"content": content},
                    "content",
                    EventTarget.STREAM_CHUNK,
                    metadata=chunk_metadata,
                )

            tool_start = chunk_metadata.get("tool_start")
            if isinstance(tool_start, dict):
                yield registry.from_external(
                    {
                        "tool_name": tool_start.get("name", "unknown"),
                        "tool_id": tool_start.get("id"),
                        "arguments": tool_start.get("arguments", {}),
                    },
                    "tool_start",
                    EventTarget.STREAM_CHUNK,
                    metadata=chunk_metadata,
                )

            tool_result = chunk_metadata.get("tool_result")
            if isinstance(tool_result, dict):
                yield registry.from_external(
                    {
                        "tool_name": tool_result.get("name", "unknown"),
                        "tool_id": tool_result.get("id"),
                        "result": tool_result.get("result", ""),
                        "success": tool_result.get("success", True),
                    },
                    "tool_result",
                    EventTarget.STREAM_CHUNK,
                    metadata=chunk_metadata,
                )

            tool_calls = getattr(chunk, "tool_calls", None) or []
            for tc in tool_calls:
                if not isinstance(tc, dict):
                    continue
                yield registry.from_external(
                    {
                        "tool_name": tc.get("name", "unknown"),
                        "tool_id": tc.get("id"),
                        "arguments": tc.get("arguments", {}),
                    },
                    "tool_call",
                    EventTarget.STREAM_CHUNK,
                    metadata=chunk_metadata,
                )

        final_content, final_metadata = output_state.flush_stream_content()
        if final_content:
            saw_content_event = True
            yield registry.from_external(
                {"content": final_content},
                "content",
                EventTarget.STREAM_CHUNK,
                metadata=final_metadata,
            )
        elif not saw_content_event:
            fallback_content = _get_stream_completion_fallback(orchestrator, output_state)
            if fallback_content:
                yield registry.from_external(
                    {"content": fallback_content},
                    "content",
                    EventTarget.STREAM_CHUNK,
                    metadata={"source": "completion_summary_fallback"},
                )

        # Emit stream end
        yield stream_end_event(success=True)

    except Exception as e:
        error_message = _sanitize_stream_error_message(e)
        logger.exception("stream_with_events failed: %s", error_message)
        yield error_event(error_message, recoverable=False)
        yield stream_end_event(success=False, error=error_message)
    finally:
        current_member_sink.reset(sink_token)


def format_context_message(context: Dict[str, Any]) -> Optional[str]:
    """Format context dict into a message string.

    Args:
        context: Context dictionary with file, error, code, etc.

    Returns:
        Formatted string for prepending to conversation, or None
    """
    if not context:
        return None

    parts = []

    if "file" in context:
        parts.append(f"File: {context['file']}")

    if "files" in context:
        parts.append(f"Files: {', '.join(context['files'])}")

    if "error" in context:
        parts.append(f"Error: {context['error']}")

    if "code" in context:
        parts.append(f"```\n{context['code']}\n```")

    # Add any other context as key-value pairs
    for key, value in context.items():
        if key not in ("file", "files", "error", "code"):
            parts.append(f"{key}: {value}")

    return "\n".join(parts) if parts else None


def _get_stream_completion_fallback(
    orchestrator: Any,
    output_state: DirectResponseOutputState,
) -> str:
    """Return a best-effort terminal content fallback for swallowed streams."""
    detector = getattr(orchestrator, "_task_completion_detector", None)
    state = getattr(detector, "_state", None) if detector is not None else None
    last_summary = getattr(state, "last_summary", "") if state is not None else ""
    normalized = strip_active_completion_markers(last_summary or "").strip()
    if not normalized:
        return ""
    return output_state.normalize_final_response(normalized).strip()


def collect_tool_calls(events: List[AgentExecutionEvent]) -> List[Dict[str, Any]]:
    """Collect tool calls from a list of events.

    Args:
        events: List of AgentExecutionEvent objects

    Returns:
        List of tool call dictionaries
    """
    tool_calls = []

    for event in events:
        if event.type == EventType.TOOL_RESULT:
            tool_calls.append(
                {
                    "tool": event.tool_name,
                    "tool_id": event.tool_id,
                    "arguments": event.arguments,
                    "result": event.result,
                    "success": event.success,
                }
            )

    return tool_calls
