# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0
"""Post-loop presentation recovery and recorded outcome evidence.

When an agentic run ends, its caller may recover the user-facing answer without
changing the run's verdict. The buffered epilogue (``TurnExecutor``) and the
streaming loop (``AgenticLoop.run_streaming`` → ``StreamingChatExecutor``) both
decide with these functions (FEP-0007 parity); ``TurnExecutor`` exposes them as
``post_loop_recovery_allowed`` / ``post_loop_recovery_kind`` /
``turn_has_successful_tool_evidence`` so the framework loop reaches them through
its turn executor.
"""

from __future__ import annotations

from copy import deepcopy
import json
from typing import Any, Optional

# Response-metadata key carrying the summed provider usage of a recovery
# synthesis (already accounted at the invocation boundary); the streaming
# consumer folds it into the stream's own usage totals.
RECOVERY_USAGE_KEY = "recovery_usage"


def recovery_allowed(
    *, spin_guard_stopped: bool, action_result: object, last_decision: Any
) -> bool:
    """Whether a finished loop may recover its presentation.

    Structured terminal stops are honored: the spin guard, a turn disposition
    that stops the run (e.g. admission denial) and an ESCALATE verdict.
    """
    from victor.agent.services.turn_execution_runtime import TurnResult
    from victor.framework.evaluation_nodes import EvaluationDecision

    if spin_guard_stopped:
        return False
    if isinstance(action_result, TurnResult) and action_result.disposition.stops_run:
        return False
    if last_decision in (EvaluationDecision.ESCALATE, "escalate"):
        return False
    return True


def recovery_kind(
    *,
    recovery_allowed: bool,
    success: bool,
    response: Optional[Any],
    has_tool_evidence: bool,
) -> Optional[str]:
    """Which presentation recovery a finished loop needs, if any.

    ``"complete"`` when the last recorded turn produced no response at all;
    ``"synthesize"`` when a failed run's last turn has no text but earlier tools
    succeeded (answer from that evidence); otherwise ``None``. Recovery never
    changes the run verdict.
    """
    if not recovery_allowed:
        return None
    if response is None:
        return "complete"
    if not success and not getattr(response, "content", None) and has_tool_evidence:
        return "synthesize"
    return None


def turn_has_tool_evidence(action_result: Any) -> bool:
    """Return True when one turn produced successful tool output."""
    if action_result is None:
        return False
    if getattr(action_result, "successful_tool_count", 0):
        return True
    return any(
        isinstance(result, dict) and result.get("success")
        for result in getattr(action_result, "tool_results", []) or []
    )


def loop_has_tool_evidence(loop_result: Any) -> bool:
    """Return True when a loop gathered successful tool output before stopping."""
    return any(
        turn_has_tool_evidence(getattr(iteration, "action_result", None))
        for iteration in getattr(loop_result, "iterations", []) or []
    )


def response_metadata(
    existing: Optional[dict[str, Any]], loop_result: Any, *, iterations: int, error: Optional[str]
) -> dict[str, Any]:
    """Publish the verdict and detached current-run receipts, apart from pending calls."""
    facts: list[dict[str, Any]] = []
    seen: set[int] = set()
    actions = [getattr(i, "action_result", None) for i in loop_result.iterations]
    actions.append(loop_result.final_state.get("action_result"))
    for action in actions:
        if action is None or id(action) in seen:
            continue
        seen.add(id(action))
        requested = getattr(getattr(action, "response", None), "tool_calls", None) or []
        for result in getattr(action, "tool_results", None) or []:
            if not isinstance(result, dict):
                continue
            call_id = result.get("tool_call_id")
            matches = [call for call in requested if call_id and call.get("id") == call_id]
            call = matches[0] if len(matches) == 1 else {}
            arguments = result.get("args", call.get("arguments", {}))
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except (ValueError, TypeError):
                    arguments = {}
            # Receipts must not expose mutable execution arguments or raw output.
            snapshot_error = False
            try:
                snapshot = deepcopy(arguments) if isinstance(arguments, dict) else {}
            except Exception:
                # An observability snapshot cannot undo or replay a completed tool.
                snapshot, snapshot_error = {}, True
            name = result.get("name") or result.get("tool_name") or call.get("name") or "unknown"
            fact = {
                "id": call_id,
                "tool": name,
                "arguments": snapshot,
                "success": result.get("success") is True,
            }
            if snapshot_error:
                fact["arguments_unavailable"] = True
            for key in ("error", "skipped", "outcome_kind", "block_source", "retryable"):
                if key in result:
                    fact[key] = result[key]
            facts.append(fact)
    metadata = dict(existing or {})
    metadata.update(
        agentic_loop_success=loop_result.success is True,
        agentic_loop_iterations=iterations,
        agentic_loop_tool_calls=facts,
    )
    metadata.pop("agentic_loop_error", None)
    if error:
        metadata["agentic_loop_error"] = error
    return metadata
