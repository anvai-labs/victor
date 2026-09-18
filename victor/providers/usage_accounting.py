"""Numeric usage accounting shared by streaming and cost consumers."""

from typing import Any, Mapping, MutableMapping


def billable_completion_tokens(usage: Mapping[str, Any]) -> int:
    """Fold reasoning once per call, or reuse a sum already folded per call.

    Missing inclusion metadata retains the historical compatibility heuristic.
    Never apply that heuristic to mixed-call aggregates.
    """
    if "billable_completion_tokens" in usage:
        return usage["billable_completion_tokens"]
    completion = usage.get("completion_tokens", 0)
    reasoning = usage.get("reasoning_tokens", 0)
    included = usage.get("reasoning_included")
    if included is False or (included is None and reasoning > completion):
        return completion + reasoning
    return completion


def usage_total_tokens(usage: Mapping[str, Any]) -> int:
    """Preserve a reported total, deriving absent/zero totals from billable usage."""
    return usage.get("total_tokens", 0) or (
        usage.get("prompt_tokens", 0) + billable_completion_tokens(usage)
    )


def accumulate_usage(total: MutableMapping[str, int], usage: Mapping[str, Any]) -> None:
    """Sum usage without losing each call's reasoning inclusion convention."""
    for key in (
        "prompt_tokens",
        "completion_tokens",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
        "reasoning_tokens",
    ):
        total[key] = total.get(key, 0) + usage.get(key, 0)
    total["billable_completion_tokens"] = total.get(
        "billable_completion_tokens", 0
    ) + billable_completion_tokens(usage)
    total["total_tokens"] = total.get("total_tokens", 0) + usage_total_tokens(usage)
