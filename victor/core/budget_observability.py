"""Structured budget-refusal observability.

Victor has ~10 independent tool-budget owners/envelopes (see
``docs/analysis/2026-11-09-seam-duplication-audit.md``, Cluster 1) whose
counters can diverge, so a run can be refused well before the caller's
funded budget is spent. This module gives every enforcement site a single
structured log line carrying the (owner, ceiling, used) triad read from
that owner's own accounting at refusal time, so budget divergence is
diagnosable from logs alone.

Deployment note: capture these triads by enabling INFO for the
``victor.budget`` logger - default WARNING root configs drop them.

Pure observability: recording a refusal never changes refusal behavior,
strings, or outcome kinds. Lines are INFO (these are informational
refusals, not errors) on the ``victor.budget`` logger with a stable
machine-parsable shape::

    BUDGET_REFUSAL owner=<owner> ceiling=<n> used=<n> tool=<name> outcome=<kind>
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger("victor.budget")


def record_budget_refusal(
    owner: str,
    ceiling: int,
    used: int,
    *,
    tool_name: Optional[str] = None,
    outcome_kind: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Record a structured triad for a tool-call budget refusal/skip.

    Args:
        owner: Which budget owner refused (e.g. ``pipeline``,
            ``tool_service``, ``streaming``).
        ceiling: The owner's own budget ceiling at refusal time.
        used: The owner's own usage counter at refusal time.
        tool_name: The refused tool, when the refusal names a single call.
        outcome_kind: The refusal's outcome kind (e.g.
            ``budget_exhausted``), when one applies.
        extra: Optional additional key/value context appended after the
            stable fields.
    """
    parts = [
        "BUDGET_REFUSAL",
        f"owner={owner}",
        f"ceiling={ceiling}",
        f"used={used}",
        f"tool={tool_name if tool_name else 'unknown'}",
        f"outcome={outcome_kind if outcome_kind else 'none'}",
    ]
    if extra:
        parts.extend(f"{key}={extra[key]!r}" for key in sorted(extra))
    logger.info(" ".join(parts))
