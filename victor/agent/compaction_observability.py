# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compaction decision observability.

Extracted from ``context_compactor`` (hotspot ratchet TD-R1 → TD-14 caps that
file's size; new behavior belongs in owning modules, not the hotspot).

Every compaction decision — skip or trigger — leaves one INFO trace line so a
wedged session's log always answers: what did we think the window was, where
did it come from, how full was it (tokens AND chars), and why did we (not)
compact. Before this, skip decisions logged nothing and no line ever recorded
which window source was in use (2026-11-01..04 incident sessions).

All helpers are defensive against metrics/controller stand-ins: a non-numeric
or mock field must never break the compaction path.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional, Tuple


class CompactionTrigger(Enum):
    """What triggered the compaction."""

    NONE = "none"  # No compaction needed
    THRESHOLD = "threshold"  # Hit utilization threshold
    OVERFLOW = "overflow"  # Actually overflowing
    BUDGET = "budget"  # Cumulative session prefill volume over budget
    MANUAL = "manual"  # Manually requested
    SCHEDULED = "scheduled"  # Background scheduled compaction


@dataclass
class CompactionAction:
    """Result of a compaction check/action.

    Attributes:
        trigger: What triggered the action
        messages_removed: Number of messages removed
        chars_freed: Approximate characters freed
        tokens_freed: Approximate tokens freed
        truncations_applied: Number of truncations applied
        new_utilization: Context utilization after action
        details: Additional details about what was done
    """

    trigger: CompactionTrigger
    messages_removed: int = 0
    chars_freed: int = 0
    tokens_freed: int = 0
    truncations_applied: int = 0
    new_utilization: float = 0.0
    details: List[str] = field(default_factory=list)

    @property
    def action_taken(self) -> bool:
        """Whether any action was taken."""
        return self.messages_removed > 0 or self.truncations_applied > 0


def coerce_int(value: Any) -> Optional[int]:
    """Coerce a metrics field to int; None for mocks/non-numeric values."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def coerce_float(value: Any) -> Optional[float]:
    """Coerce a metrics field to float; None for mocks/non-numeric values."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def probe_budget_state(controller: Any) -> Tuple[bool, int, Optional[int]]:
    """Read a controller's session-prefill budget state (best-effort).

    The budget trigger must be evaluable independently of utilization: it
    used to live only inside ``smart_compact_history`` (i.e. behind the
    compactor's utilization gate), so mis-estimated low utilization kept it
    silent while cumulative prefill ran 10-32x over its limit.

    Returns:
        Tuple of (due, session_prompt_tokens, limit).
    """
    if controller is None or not hasattr(controller, "budget_compaction_due"):
        return False, 0, None
    try:
        result = controller.budget_compaction_due()
        if isinstance(result, tuple) and len(result) == 3:
            return result
    except Exception:  # noqa: BLE001 - probe must never raise
        pass
    return False, 0, None


def threshold_origin_label(
    adaptive_enabled: bool,
    messages: Optional[Any],
    phase: Optional[Any],
    phase_aware: bool,
) -> str:
    """Name where the effective compaction threshold came from."""
    if adaptive_enabled and messages:
        return "adaptive"
    if phase and phase_aware:
        return f"phase:{getattr(phase, 'value', phase)}"
    return "default"


def skip_reason(metrics: Any) -> str:
    """Human reason for a below-threshold skip."""
    window = coerce_int(getattr(metrics, "max_context_tokens", None))
    est = coerce_int(getattr(metrics, "estimated_tokens", None))
    if window and est is not None:
        return f"utilization below threshold (headroom {max(0, window - est):,} tok)"
    return "utilization below threshold (char-space)"


def trigger_reason(trigger_value: str, budget: Tuple[bool, int, Optional[int]], force: bool) -> str:
    """Human reason for a compaction trigger."""
    if force:
        return "forced"
    if trigger_value == "overflow":
        return "overflow risk"
    if trigger_value == "budget":
        return f"session prefill {budget[1]:,} >= limit {budget[2]:,}"
    return "utilization crossed threshold"


def evaluate_trigger(
    logger: Any,
    metrics: Any,
    *,
    force: bool,
    enable_proactive: bool,
    utilization: float,
    threshold: float,
    origin: str,
    budget_state: Tuple[bool, int, Optional[int]],
) -> CompactionTrigger:
    """Decide the compaction trigger and emit its decision trace.

    The budget condition is evaluated independently of utilization: it used
    to live only inside ``smart_compact_history`` (i.e. behind the utilization
    gate), so mis-estimated low utilization kept it silent while cumulative
    prefill ran 10-32x over its limit.

    Returns:
        The decided CompactionTrigger (NONE when no compaction is warranted).
    """
    if force:
        trigger = CompactionTrigger.MANUAL
    elif metrics.is_overflow_risk:
        trigger = CompactionTrigger.OVERFLOW
    elif budget_state[0]:
        trigger = CompactionTrigger.BUDGET
    elif enable_proactive and utilization >= threshold:
        trigger = CompactionTrigger.THRESHOLD
    else:
        trigger = CompactionTrigger.NONE

    trace_compaction_check(
        logger,
        metrics,
        threshold=threshold,
        origin=origin,
        decision=f"trigger:{trigger.value}" if trigger is not CompactionTrigger.NONE else "skip",
        reason=(
            trigger_reason(trigger.value, budget_state, force)
            if trigger is not CompactionTrigger.NONE
            else skip_reason(metrics)
        ),
        budget=budget_state,
    )
    return trigger


def trace_compaction_check(
    logger: Any,
    metrics: Any,
    *,
    threshold: float,
    origin: str,
    decision: str,
    reason: str,
    budget: Tuple[bool, int, Optional[int]],
) -> None:
    """Emit the one-line ``[compaction-check]`` decision trace."""
    budget_due, session_prefill, prefill_limit = budget
    window = coerce_int(getattr(metrics, "max_context_tokens", None))
    window_source = getattr(metrics, "context_window_source", None)
    window_repr = (
        f"{window:,} ({window_source if isinstance(window_source, str) and window_source else 'unknown'})"
        if window
        else "unset(charspace)"
    )
    est = coerce_int(getattr(metrics, "estimated_tokens", None))
    chars = coerce_int(getattr(metrics, "char_count", None))
    last_actual = coerce_int(getattr(metrics, "last_actual_prompt_tokens", None))
    util_token = coerce_float(getattr(metrics, "utilization", None)) if window else None
    util_chars = coerce_float(getattr(metrics, "char_utilization", None))
    threshold_value = coerce_float(threshold)
    parts = [
        f"est_tokens={est:,}" if est is not None else "est_tokens=?",
        f"chars={chars:,}" if chars is not None else "chars=?",
        (
            f"last_actual_prompt_tokens={last_actual:,}"
            if last_actual is not None
            else "last_actual_prompt_tokens=none"
        ),
        f"window={window_repr}",
        f"util_token={util_token:.1%}" if util_token is not None else "util_token=n/a",
        f"util_chars={util_chars:.1%}" if util_chars is not None else "util_chars=n/a",
        (
            f"threshold={threshold_value:.0%}({origin})"
            if threshold_value is not None
            else f"threshold=?({origin})"
        ),
        (
            f"budget={session_prefill:,}/{prefill_limit:,}{' DUE' if budget_due else ''}"
            if prefill_limit
            else "budget=disabled"
        ),
    ]
    logger.info(
        "[compaction-check] %s | decision=%s reason=%s",
        " ".join(parts),
        decision,
        reason,
    )


def log_compaction_result(
    logger: Any,
    *,
    action_taken: bool,
    trigger_value: str,
    messages_removed: int,
    truncations_applied: int,
    tokens_freed: int,
    tokens_before: int,
    utilization: float,
    metrics_after: Any,
    started_at: float,
) -> None:
    """Emit the ``[compaction-result]`` outcome line (INFO when action taken)."""
    tokens_after = coerce_int(getattr(metrics_after, "estimated_tokens", None))
    util_after = coerce_float(getattr(metrics_after, "utilization", None))
    window_after = coerce_int(getattr(metrics_after, "max_context_tokens", None))
    window_source = getattr(metrics_after, "context_window_source", "")
    util_before = coerce_float(utilization)
    log = logger.info if action_taken else logger.debug
    log(
        "[compaction-result] trigger=%s removed=%d messages truncated=%d outputs "
        "freed~%d tokens est_tokens=%s->%s util=%s->%s window=%s took=%.0fms",
        trigger_value,
        messages_removed,
        truncations_applied,
        tokens_freed,
        f"{tokens_before:,}",
        f"{tokens_after:,}" if tokens_after is not None else "?",
        f"{util_before:.1%}" if util_before is not None else "?",
        f"{util_after:.1%}" if util_after is not None else "?",
        (
            f"{window_after:,}/{window_source if isinstance(window_source, str) else 'unknown'}"
            if window_after
            else "charspace"
        ),
        (time.monotonic() - started_at) * 1000,
    )


def duplicate_suppression_age(controller: Any) -> Optional[float]:
    """Age (seconds) of the last compaction if a duplicate trigger is suppressed.

    Returns None when suppression does not apply: no controller, no recent
    compaction, or a non-numeric age (mock stand-ins never suppress).
    """
    if controller is None or not hasattr(controller, "last_compaction_age_seconds"):
        return None
    interval = coerce_float(getattr(controller, "MIN_COMPACTION_INTERVAL_SECONDS", 60.0))
    if interval is None:
        return None
    try:
        age = controller.last_compaction_age_seconds
    except Exception:  # noqa: BLE001 - suppression is best-effort
        return None
    age_value = coerce_float(age)
    if age_value is None or age_value >= interval:
        return None
    return age_value
