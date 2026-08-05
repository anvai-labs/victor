# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Private streaming runtime adapter for the canonical chat service path."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, Mapping, Optional

from victor.agent.services.chat_stream_helpers import ChatStreamHelperMixin

if TYPE_CHECKING:
    from victor.providers.base import StreamChunk
    from victor.agent.services.chat_stream_executor import StreamingChatExecutor
    from victor.agent.streaming.intent_classification import IntentClassificationHandler
    from victor.agent.streaming.continuation import ContinuationHandler
    from victor.agent.streaming.tool_execution import ToolExecutionHandler

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StreamingRuntimeBindings:
    """Resolved runtime state and capability accessors for a streaming owner."""

    runtime_owner: Any
    state_host: Any
    instance_dicts: tuple[dict[str, Any], ...]

    @classmethod
    def resolve(cls, runtime_owner: Any) -> "StreamingRuntimeBindings":
        """Resolve runtime ownership across direct owners and adapters."""
        owner_dict = getattr(runtime_owner, "__dict__", {})
        instance_dicts: list[dict[str, Any]] = []
        state_host = runtime_owner
        if isinstance(owner_dict, dict):
            instance_dicts.append(owner_dict)
            host = owner_dict.get("_orchestrator")
            if host is not None:
                state_host = host

        host_dict = getattr(state_host, "__dict__", None)
        if isinstance(host_dict, dict) and host_dict not in instance_dicts:
            instance_dicts.append(host_dict)

        return cls(
            runtime_owner=runtime_owner,
            state_host=state_host,
            instance_dicts=tuple(instance_dicts),
        )

    @property
    def state_dict(self) -> dict[str, Any]:
        """Return the mutable runtime state dictionary."""
        host_dict = getattr(self.state_host, "__dict__", {})
        return host_dict if isinstance(host_dict, dict) else {}

    def has_capability(self, name: str) -> bool:
        """Return whether a capability is exposed by the runtime owner."""
        checker = getattr(self.runtime_owner, "has_capability", None)
        if callable(checker):
            try:
                if bool(checker(name)):
                    return True
            except Exception:
                logger.debug("Capability probe failed for %s", name, exc_info=True)
        for instance_dict in self.instance_dicts:
            if name in instance_dict or f"_{name}" in instance_dict:
                return True
        return False

    def get_capability_value(self, name: str, default: Any = None) -> Any:
        """Return a capability value, falling back to runtime state when needed."""
        getter = getattr(self.runtime_owner, "get_capability_value", None)
        if callable(getter):
            try:
                value = getter(name)
                if value is not None:
                    return value
            except Exception:
                logger.debug("Capability read failed for %s", name, exc_info=True)
        for instance_dict in self.instance_dicts:
            if name in instance_dict:
                return instance_dict.get(name)
            underscored = f"_{name}"
            if underscored in instance_dict:
                return instance_dict.get(underscored)
        return default


class ServiceStreamingRuntime(ChatStreamHelperMixin):
    """Service-owned runtime adapter for the canonical streaming executor.

    This adapter keeps the live streaming path off the deprecated ChatCoordinator
    shim while reusing the shared service-owned streaming helper implementations.
    """

    def __init__(self, orchestrator: Any) -> None:
        self._orchestrator = orchestrator
        self._intent_classification_handler: Optional["IntentClassificationHandler"] = None
        self._continuation_handler: Optional["ContinuationHandler"] = None
        self._tool_execution_handler: Optional["ToolExecutionHandler"] = None
        self._streaming_executor: Optional["StreamingChatExecutor"] = None
        self._runtime_bindings: Optional[StreamingRuntimeBindings] = None

    def _get_runtime_bindings(
        self, runtime_owner: Optional[Any] = None
    ) -> StreamingRuntimeBindings:
        """Return resolved runtime state/capability bindings for the active owner."""
        owner = runtime_owner if runtime_owner is not None else self._orchestrator
        if self._runtime_bindings is None or self._runtime_bindings.runtime_owner is not owner:
            self._runtime_bindings = StreamingRuntimeBindings.resolve(owner)
        return self._runtime_bindings

    def _resolve_runtime_state_host(self, runtime_host: Any) -> Any:
        """Hook used by shared helpers to resolve the canonical state owner."""
        return self._get_runtime_bindings(runtime_host).state_host

    def _resolve_runtime_state_dict(self, runtime_host: Any) -> dict[str, Any]:
        """Hook used by shared helpers to resolve mutable runtime state."""
        return self._get_runtime_bindings(runtime_host).state_dict

    def _resolve_runtime_capability_presence(self, name: str) -> bool:
        """Hook used by shared helpers to resolve capability presence."""
        return self._get_runtime_bindings().has_capability(name)

    def _resolve_runtime_capability_value(self, name: str, default: Any = None) -> Any:
        """Hook used by shared helpers to resolve capability values."""
        return self._get_runtime_bindings().get_capability_value(name, default)

    def get_executor(self) -> "StreamingChatExecutor":
        """Get or create the canonical service-owned streaming executor."""
        if self._streaming_executor is None:
            from victor.agent.services.chat_stream_executor import (
                create_streaming_chat_executor,
            )
            from victor.agent.services.runtime_intelligence import (
                RuntimeIntelligenceService,
            )

            bindings = self._get_runtime_bindings()
            orch = bindings.runtime_owner
            state_host = bindings.state_host
            state_dict = bindings.state_dict
            perception = bindings.get_capability_value("perception_integration")
            fulfillment = bindings.get_capability_value("fulfillment_detector")
            runtime_intelligence = state_dict.get("_runtime_intelligence")
            if runtime_intelligence is None:
                runtime_intelligence = RuntimeIntelligenceService.from_orchestrator(
                    orch,
                    perception_integration=perception,
                    optimization_injector=state_dict.get("_optimization_injector"),
                )
                state_host._runtime_intelligence = runtime_intelligence
            self._streaming_executor = create_streaming_chat_executor(
                self,
                runtime_intelligence=runtime_intelligence,
                perception=perception,
                fulfillment=fulfillment,
            )
        return self._streaming_executor

    @staticmethod
    def _coerce_unit_float(value: Any, default: float = 0.0) -> float:
        """Normalize optional quality values into the canonical [0, 1] range."""
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return default
        return max(0.0, min(1.0, numeric))

    @staticmethod
    def _coerce_optional_text(value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(getattr(value, "value", value)).strip()
        return text or None

    def _classify_stream_outcome(self, ctx: Any, *, failed: bool = False) -> Dict[str, Any]:
        """Derive a shared stream outcome snapshot for telemetry finalizers."""
        quality_score = self._coerce_unit_float(getattr(ctx, "last_quality_score", 0.0))
        has_substantial_content = False
        if hasattr(ctx, "has_substantial_content"):
            try:
                has_substantial_content = bool(ctx.has_substantial_content())
            except Exception:
                has_substantial_content = False

        if failed:
            status = "error"
        elif quality_score >= 0.7:
            status = "completed"
        elif has_substantial_content or quality_score >= 0.45:
            status = "resolved"
        else:
            status = "failed"

        return {
            "status": status,
            "quality_score": quality_score,
            "has_substantial_content": has_substantial_content,
        }

    def _normalize_stream_recovery_event(
        self,
        event: Any,
        *,
        recovered: bool,
        failed: bool,
    ) -> Optional[Dict[str, Any]]:
        """Translate raw stream recovery telemetry into degradation-event schema."""
        if hasattr(event, "to_dict"):
            event = event.to_dict()
        if not isinstance(event, Mapping):
            return None

        action = self._coerce_optional_text(event.get("action"))
        reason = self._coerce_optional_text(event.get("reason"))
        strategy_name = self._coerce_optional_text(event.get("strategy_name"))
        failure_type = self._coerce_optional_text(event.get("failure_type")) or "STREAMING_RECOVERY"
        reasons = []
        for value in (action, strategy_name, reason):
            if value and value not in reasons:
                reasons.append(value)

        adaptation_cost = event.get("adaptation_cost")
        try:
            adaptation_cost_value = float(adaptation_cost) if adaptation_cost is not None else 1.0
        except (TypeError, ValueError):
            adaptation_cost_value = 1.0

        confidence = event.get("confidence")
        try:
            confidence_value = float(confidence) if confidence is not None else None
        except (TypeError, ValueError):
            confidence_value = None

        normalized = {
            "source": self._coerce_optional_text(event.get("source")) or "streaming_recovery",
            "kind": self._coerce_optional_text(event.get("kind")) or "recovery_action",
            "failure_type": failure_type,
            "provider": self._coerce_optional_text(event.get("provider")),
            "model": self._coerce_optional_text(event.get("model")),
            "task_type": self._coerce_optional_text(event.get("task_type")),
            "pre_degraded": True,
            "post_degraded": bool(failed),
            "recovered": bool(recovered),
            "adaptation_cost": adaptation_cost_value,
            "iteration": event.get("iteration"),
            "degradation_reasons": reasons,
            "action": action,
            "reason": reason,
            "strategy_name": strategy_name,
            "confidence": confidence_value,
            "fallback_provider": self._coerce_optional_text(event.get("fallback_provider")),
            "fallback_model": self._coerce_optional_text(event.get("fallback_model")),
        }
        return normalized

    def _build_stream_degradation_feedback_payload(
        self,
        ctx: Any,
        *,
        failed: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Finalize stream degradation/recovery telemetry into evaluation-ready events."""
        degradation_events = list(getattr(ctx, "degradation_events", []) or [])
        recovery_events = list(getattr(ctx, "recovery_events", []) or [])
        if not degradation_events and not recovery_events:
            return None

        outcome = self._classify_stream_outcome(ctx, failed=failed)
        recovered = outcome["status"] in {"completed", "resolved"} and not failed
        for recovery_event in recovery_events:
            normalized = self._normalize_stream_recovery_event(
                recovery_event,
                recovered=recovered,
                failed=failed,
            )
            if normalized is not None:
                degradation_events.append(normalized)

        return {
            "status": outcome["status"],
            "completion_score": outcome["quality_score"],
            "degradation_events": degradation_events,
            "recovery_events": recovery_events,
        }

    def _build_stream_topology_feedback_payload(
        self,
        ctx: Any,
        *,
        failed: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Build a live topology-feedback payload from the completed stream context."""
        topology_events = list(getattr(ctx, "topology_events", []) or [])
        if not topology_events:
            return None

        outcome_summary = self._classify_stream_outcome(ctx, failed=failed)
        status = outcome_summary["status"]
        quality_score = outcome_summary["quality_score"]
        has_substantial_content = outcome_summary["has_substantial_content"]

        outcome = {
            "status": status,
            "completion_score": quality_score,
            "tool_calls": getattr(ctx, "tool_calls_used", 0),
            "turns": getattr(ctx, "total_iterations", 0),
            "runtime": "streaming",
            "force_completion": bool(getattr(ctx, "force_completion", False)),
            "has_substantial_content": has_substantial_content,
        }
        topology_events[-1]["outcome"] = dict(outcome)

        return {
            "status": status,
            "completion_score": quality_score,
            "tool_calls": outcome["tool_calls"],
            "turns": outcome["turns"],
            "topology_events": topology_events,
        }

    @staticmethod
    def _has_degraded_resume_state(
        ctx: Any,
        *,
        failed: bool = False,
    ) -> bool:
        """Return whether the completed stream should resume from a degraded state."""
        if failed:
            return True
        provider_status_events = list(getattr(ctx, "provider_status_events", []) or [])
        degraded_kinds = {
            "tool_history_repaired",
            "empty_stream_completed",
            "provider_stall_timeout",
        }
        if any(event.get("kind") in degraded_kinds for event in provider_status_events):
            return True
        degradation_events = list(getattr(ctx, "degradation_events", []) or [])
        recovery_events = list(getattr(ctx, "recovery_events", []) or [])
        return bool(degradation_events or recovery_events)

    @classmethod
    def _build_resume_summary(
        cls,
        ctx: Any,
        *,
        failed: bool = False,
    ) -> str:
        """Build a compact resume summary for bare continuation follow-ups."""
        parts: list[str] = []
        tool_calls_used = int(getattr(ctx, "tool_calls_used", 0) or 0)
        if tool_calls_used > 0:
            parts.append(f"{tool_calls_used} tool call(s) used")

        recent_resources = sorted(
            str(item) for item in (getattr(ctx, "unique_resources", set()) or set())
        )
        if recent_resources:
            preview = ", ".join(recent_resources[:4])
            parts.append(f"files examined: {preview}")

        recent_tools = sorted(
            str(item) for item in (getattr(ctx, "executed_tool_names", set()) or set()) if item
        )
        if recent_tools:
            parts.append(f"tools used: {', '.join(recent_tools[:4])}")

        if cls._has_degraded_resume_state(ctx, failed=failed):
            provider_status_events = list(getattr(ctx, "provider_status_events", []) or [])
            if any(
                event.get("kind") == "tool_history_repaired" for event in provider_status_events
            ):
                parts.append("previous provider request required tool-history repair")
            elif failed:
                parts.append("previous turn ended before completion")
            else:
                parts.append("previous turn recovered from degraded stream state")

        return "; ".join(parts)

    async def stream_chat(self, user_message: str, **kwargs: Any) -> AsyncIterator["StreamChunk"]:
        """Stream a response through the canonical service-owned executor."""
        _ = kwargs.pop("_preserve_iteration", None)
        fallback_iteration = kwargs.pop("_fallback_iteration", 0)

        if fallback_iteration > 0:
            logger.info(
                "[ServiceStreamingRuntime] Using fallback iteration: %s",
                fallback_iteration,
            )
            kwargs["_fallback_iteration"] = fallback_iteration

        bindings = self._get_runtime_bindings()
        state_host = bindings.state_host
        executor = self.get_executor()
        stream_failed = False

        try:
            # FEP-0007 cutover: drive the unified loop (AgenticLoop.run_streaming via run_unified)
            # so the streaming UI path runs the same PERCEIVE/PLAN/ACT/EVALUATE/DECIDE loop as the
            # buffered path. The legacy run() is now dead and removed in the follow-up step.
            async for chunk in executor.run_unified(user_message, **kwargs):
                yield chunk
        except Exception:
            stream_failed = True
            raise
        finally:
            ctx = None
            current_stream_context = bindings.get_capability_value("current_stream_context")
            if current_stream_context is not None:
                ctx = current_stream_context
            else:
                ctx = bindings.state_dict.get("_current_stream_context")

            if ctx is not None:
                state_dict = bindings.state_dict
                if hasattr(ctx, "cumulative_usage"):
                    # Fold this turn's usage into the orchestrator's session-cumulative dict —
                    # the SAME object the metrics service snapshots for task-report token deltas
                    # (orchestrator passes it as `cumulative_token_usage=`). The prior code
                    # accumulated into `state_dict["_cumulative_token_usage"]`, which is absent
                    # (None) on the service path, so the loop was a silent no-op and every task
                    # report read total_tokens=0 despite real usage on `ctx.cumulative_usage`.
                    cumulative_usage = getattr(self._orchestrator, "_cumulative_token_usage", None)
                    if not isinstance(cumulative_usage, dict):
                        cumulative_usage = state_dict.get("_cumulative_token_usage")
                    if isinstance(cumulative_usage, dict):
                        for key, value in ctx.cumulative_usage.items():
                            if key in cumulative_usage:
                                cumulative_usage[key] += value
                        if cumulative_usage.get("total_tokens", 0) == 0:
                            cumulative_usage["total_tokens"] = cumulative_usage.get(
                                "prompt_tokens", 0
                            ) + cumulative_usage.get("completion_tokens", 0)

                    # Close the cost-measurement wire (C0): the service streaming runtime
                    # never finalized stream metrics, so per-turn tokens/cost stayed 0 and the
                    # canonical cost record (SessionCostTracker + usage.jsonl "stream_completed")
                    # was never emitted — the *dominant* cost term went unmeasured. Finalize
                    # once here (this finally runs once per turn; the continuation runtime owns
                    # its own finalize on the legacy path), reusing the existing pipeline.
                    try:
                        # Route through the metrics service directly so the turn's
                        # aggregated Sandhi diagnostics ride along (the orchestrator
                        # wrapper predates that parameter); fall back to the wrapper.
                        diagnostics = getattr(ctx, "provider_diagnostics", None) or None
                        metrics_owner = getattr(self._orchestrator, "_metrics_coordinator", None)
                        if metrics_owner is not None:
                            metrics_owner.finalize_stream_metrics(
                                ctx.cumulative_usage, provider_diagnostics=diagnostics
                            )
                        else:
                            self._orchestrator._finalize_stream_metrics(ctx.cumulative_usage)
                    except Exception:
                        logger.debug("C0 stream-metrics finalize failed", exc_info=True)

                    prompt_tokens = ctx.cumulative_usage.get("prompt_tokens", 0)
                    if prompt_tokens > 0:
                        try:
                            ctrl = state_dict.get("_conversation_controller")
                            total_chars = sum(len(m.content) for m in ctrl.messages)
                            ctrl.record_actual_usage(prompt_tokens, total_chars)
                        except Exception:
                            pass

                topology_feedback_payload = self._build_stream_topology_feedback_payload(
                    ctx,
                    failed=stream_failed,
                )
                if topology_feedback_payload is not None:
                    ctx.topology_events = list(topology_feedback_payload["topology_events"])
                    runtime_intelligence = state_dict.get("_runtime_intelligence")
                    if runtime_intelligence is not None and hasattr(
                        runtime_intelligence, "record_topology_outcome"
                    ):
                        try:
                            runtime_intelligence.record_topology_outcome(topology_feedback_payload)
                        except Exception as exc:
                            logger.debug(
                                "Failed to record streaming topology runtime outcome: %s",
                                exc,
                            )

                degradation_feedback_payload = self._build_stream_degradation_feedback_payload(
                    ctx,
                    failed=stream_failed,
                )
                if degradation_feedback_payload is not None:
                    ctx.degradation_events = list(
                        degradation_feedback_payload["degradation_events"]
                    )
                    ctx.recovery_events = list(degradation_feedback_payload["recovery_events"])

                degraded_resume_state = self._has_degraded_resume_state(
                    ctx,
                    failed=stream_failed,
                )
                resume_recent_resources = sorted(
                    str(item) for item in (getattr(ctx, "unique_resources", set()) or set()) if item
                )[:8]
                resume_recent_tools = sorted(
                    str(item)
                    for item in (getattr(ctx, "executed_tool_names", set()) or set())
                    if item
                )[:8]
                resume_summary = self._build_resume_summary(
                    ctx,
                    failed=stream_failed,
                )
                provider_status_events = list(getattr(ctx, "provider_status_events", []) or [])

                state_host._last_stream_task_context = {
                    "unified_task_type": getattr(ctx, "unified_task_type", None),
                    "task_classification": getattr(ctx, "task_classification", None),
                    "complexity_tool_budget": getattr(ctx, "complexity_tool_budget", None),
                    "coarse_task_type": getattr(ctx, "coarse_task_type", None),
                    "is_analysis_task": bool(getattr(ctx, "is_analysis_task", False)),
                    "is_action_task": bool(getattr(ctx, "is_action_task", False)),
                    "needs_execution": bool(getattr(ctx, "needs_execution", False)),
                    "tool_calls_used": int(getattr(ctx, "tool_calls_used", 0) or 0),
                    "task_intent": str(getattr(ctx, "task_intent", "") or ""),
                    "plan_steps": list(getattr(ctx, "plan_steps", []) or [])[:8],
                    "intent_log": list(getattr(ctx, "intent_log", []) or [])[-12:],
                    "last_compaction_policy_reason": str(
                        getattr(ctx, "last_compaction_policy_reason", "") or ""
                    ),
                    "resume_recent_resources": resume_recent_resources,
                    "resume_recent_tools": resume_recent_tools,
                    "provider_status_events": provider_status_events[-6:],
                    "degraded_resume_state": degraded_resume_state,
                    "resume_summary": resume_summary,
                }

                runtime_snapshot = getattr(ctx, "runtime_override_snapshot", None)
                self._restore_stream_runtime_overrides(runtime_snapshot)
                ctx.runtime_override_snapshot = None

            if "_current_stream_context" in bindings.state_dict:
                state_host._current_stream_context = None
