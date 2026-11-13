"""Victor runtime module — execution context, lifecycle, cache, and tracing."""

from victor.runtime.cache_registry import CacheCategory, CacheRegistry
from victor.runtime.chat_runtime import resolve_chat_runtime, resolve_chat_service
from victor.runtime.context import (
    ExecutionContext,  # Backward compatibility alias
    ResolvedRuntimeServices,
    RuntimeExecutionContext,
    ServiceAccessor,
    register_runtime_services,
    resolve_execution_context,
    resolve_runtime_services,
)
from victor.runtime.policy_snapshot import PolicySnapshotError, RunPolicySnapshot
from victor.runtime.trace_context import TraceContext, current_trace, get_correlation_id

__all__ = [
    "CacheCategory",
    "CacheRegistry",
    "RuntimeExecutionContext",
    "ExecutionContext",  # Backward compatibility alias
    "ResolvedRuntimeServices",
    "ServiceAccessor",
    "register_runtime_services",
    "resolve_execution_context",
    "resolve_runtime_services",
    "resolve_chat_runtime",
    "resolve_chat_service",
    "PolicySnapshotError",
    "RunPolicySnapshot",
    "TraceContext",
    "current_trace",
    "get_correlation_id",
]
