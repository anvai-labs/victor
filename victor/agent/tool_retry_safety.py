"""Shared automatic-replay guard for tool execution, not backend reconciliation.

Tool metadata is a trusted adapter declaration. Neither a tool name, scheduling
category, error string nor a model proposal proves that replay is safe.
"""

from typing import Any

from victor.core.errors import ErrorInfo
from victor.tools.enums import AccessMode


def allows_tool_retry(tool: Any) -> bool:
    """Allow automatic replay only for a declared effect-free operation.

    Explicit access mode takes precedence over the legacy idempotency property.
    A mixed/network tool needs operation-specific reconciliation; a generic
    idempotency flag is not a backend same-key/receipt guarantee.
    """
    if tool is None:
        return False
    try:
        mode = getattr(tool, "access_mode", None)
        if mode is not None:
            return mode is AccessMode.READONLY
        return getattr(tool, "is_idempotent", False) is True
    except Exception:
        # Broken capability metadata must not authorize another effect.
        return False


def mark_unknown_tool_outcome(error_info: ErrorInfo) -> None:
    """Retain failure classification while explicitly withholding replay permission."""
    error_info.details.update(
        execution_outcome="unknown", retryable=False, reconciliation_required=True
    )
    error_info.recovery_hint = (
        "The operation may have committed. Reconcile its outcome before another attempt; "
        "automatic replay is blocked."
    )
    if "suggestion" in error_info.details:
        error_info.details["suggestion"] = error_info.recovery_hint


def blocks_tool_retry(result: Any) -> bool:
    """Read a result or ErrorInfo replay veto without interpreting error prose."""
    if getattr(result, "retryable", None) is False:
        return True
    info = result if isinstance(result, ErrorInfo) else getattr(result, "error_info", None)
    return isinstance(info, ErrorInfo) and info.details.get("retryable") is False
