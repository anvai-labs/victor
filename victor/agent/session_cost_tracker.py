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

"""Session cost tracking for LLM API usage.

This module provides per-request and session-cumulative cost tracking
with export capabilities for analytics and cost management.

Example usage:
    tracker = SessionCostTracker(provider="anthropic", model="claude-3-5-sonnet")

    # Record requests
    tracker.record_request(prompt_tokens=1000, completion_tokens=500)
    tracker.record_request(prompt_tokens=2000, completion_tokens=800)

    # Get session summary
    summary = tracker.get_summary()
    print(f"Total cost: ${summary['total_cost']:.4f}")

    # Export for analytics
    tracker.export_json(Path("./session_costs.json"))
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional

from victor.core.context import get_auth_group_id, get_auth_subject_id, get_turn_id
from victor.providers.usage_accounting import billable_completion_tokens, usage_total_tokens

logger = logging.getLogger(__name__)

RequestOutcome = Literal["success", "error", "cancelled", "closed"]
UsageStatus = Literal["unknown", "partial", "reported"]


@dataclass
class RequestCost:
    """Cost data for a single request."""

    request_id: str
    timestamp: float
    model: str

    # Token counts
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    total_tokens: int = 0

    # Cost breakdown (USD)
    input_cost: float = 0.0
    output_cost: float = 0.0
    cache_cost: float = 0.0
    total_cost: float = 0.0

    # Metadata
    duration_seconds: float = 0.0
    tool_calls: int = 0
    # The agentic-loop turn this call served (sandhi x-sandhi-step-id; None when
    # unbound — bare provider usage outside a run). Populated from the correlation
    # spine so per-step cost rows join the sandhi run cost tree.
    step_id: Optional[str] = None
    # Preserve provider completion counts while exposing the output actually priced.
    billable_completion_tokens: int = 0
    provider: str = "unknown"
    outcome: Literal["pending", "success", "error", "cancelled", "closed"] = "success"
    usage_status: UsageStatus = "reported"
    transport_attempts: Optional[int] = None
    subject_id: Optional[str] = field(default=None, repr=False)
    group_id: Optional[str] = field(default=None, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "request_id": self.request_id,
            "timestamp": self.timestamp,
            "timestamp_iso": datetime.fromtimestamp(self.timestamp).isoformat(),
            "model": self.model,
            "provider": self.provider,
            "outcome": self.outcome,
            "usage_status": self.usage_status,
            "tokens": (
                {
                    "prompt": self.prompt_tokens,
                    "completion": self.completion_tokens,
                    "billable_completion": self.billable_completion_tokens,
                    "cache_read": self.cache_read_tokens,
                    "cache_write": self.cache_write_tokens,
                    "total": self.total_tokens,
                }
                if self.usage_status != "unknown"
                else None
            ),
            "cost": (
                {
                    "input": self.input_cost,
                    "output": self.output_cost,
                    "cache": self.cache_cost,
                    "total": self.total_cost,
                }
                if self.usage_status != "unknown"
                else None
            ),
            "metadata": {
                "duration_seconds": self.duration_seconds,
                "tool_calls": self.tool_calls,
                "transport_attempts": self.transport_attempts,
                **({"step_id": self.step_id} if self.step_id else {}),
            },
        }


@dataclass
class SessionCostTracker:
    """Tracks costs across a chat session.

    Provides:
    - Per-request cost tracking
    - Session-cumulative totals
    - Export to JSON/CSV formats
    - Cost summary statistics
    """

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    provider: str = "unknown"
    model: str = "unknown"
    start_time: float = field(default_factory=time.time)

    # Attribution (FEP-0020 Phase 1) — carried from the API-server auth seam so
    # cost/usage records answer "which user / which team". Default None keeps the
    # historical session-scoped, unattributed behavior.
    subject_id: Optional[str] = None
    group_id: Optional[str] = None

    # Per-request tracking
    requests: List[RequestCost] = field(default_factory=list)
    _pending_requests: Dict[str, RequestCost] = field(default_factory=dict, init=False, repr=False)

    # Cumulative totals
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cache_read_tokens: int = 0
    total_cache_write_tokens: int = 0
    total_tokens: int = 0

    # Cumulative costs
    total_input_cost: float = 0.0
    total_output_cost: float = 0.0
    total_cache_cost: float = 0.0
    total_cost: float = 0.0

    # Capabilities reference
    _capabilities: Optional[Any] = field(default=None, repr=False)

    # Sandhi usage-gateway bridge (FEP-0020 Phase 2). Optional; when attached, each
    # recorded request also emits one neutral usage event (per-subject attribution)
    # to sandhi's local sink. None ⇒ no gateway metering (default). Best-effort —
    # never fails the request path.
    _sandhi: Optional[Any] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Initialize capabilities from config."""
        if self._capabilities is None and self.provider != "unknown":
            try:
                from victor.config.metrics_capabilities import get_metrics_capabilities

                self._capabilities = get_metrics_capabilities(self.provider, self.model)
            except Exception as e:
                logger.debug(f"Failed to load metrics capabilities: {e}")

    def record_request(
        self,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        duration_seconds: float = 0.0,
        tool_calls: int = 0,
        model: Optional[str] = None,
        *,
        billable_completion_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
    ) -> RequestCost:
        """Record a single request's cost.

        Args:
            prompt_tokens: Number of input tokens
            completion_tokens: Number of output tokens
            cache_read_tokens: Number of cache read tokens
            cache_write_tokens: Number of cache write tokens
            duration_seconds: Request duration
            tool_calls: Number of tool calls in response
            model: Optional model override (for multi-model sessions)
            billable_completion_tokens: Output after per-call reasoning folding;
                defaults to completion_tokens for legacy callers.
            total_tokens: Provider total (or a per-call sum); derived when absent.

        Returns:
            RequestCost object with calculated costs
        """
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cache_read_input_tokens": cache_read_tokens,
            "cache_creation_input_tokens": cache_write_tokens,
        }
        if billable_completion_tokens is not None:
            usage["billable_completion_tokens"] = billable_completion_tokens
        if total_tokens is not None:
            usage["total_tokens"] = total_tokens
        request = self.begin_request(model=model)
        self.settle_request(
            request,
            outcome="success",
            usage=usage,
            duration_seconds=duration_seconds,
            tool_calls=tool_calls,
        )
        return request

    def begin_request(
        self, *, provider: Optional[str] = None, model: Optional[str] = None
    ) -> RequestCost:
        """Count a dispatched invocation and capture its caller before any await/yield."""
        request = RequestCost(
            request_id=str(uuid.uuid4()),
            timestamp=time.time(),
            provider=provider or self.provider,
            model=model or self.model,
            step_id=get_turn_id() or None,
            subject_id=get_auth_subject_id() or self.subject_id,
            group_id=get_auth_group_id() or self.group_id,
            outcome="pending",
            usage_status="unknown",
        )
        self.requests.append(request)
        self._pending_requests[request.request_id] = request
        return request

    def settle_request(
        self,
        request: RequestCost,
        *,
        outcome: RequestOutcome,
        usage: Optional[Mapping[str, Any]],
        usage_complete: bool = True,
        duration_seconds: float = 0.0,
        tool_calls: int = 0,
        transport_attempts: Optional[int] = None,
    ) -> None:
        """Settle an owned invocation once; absent usage is never observed zero.

        Partial observations contribute to known totals, but are not mirrored to
        the legacy meter, whose contract cannot represent incomplete usage.
        Duration is invocation lifetime, including consumer pauses, not wire latency.
        """
        if self._pending_requests.get(request.request_id) is not request:
            raise ValueError("Request is not pending on this tracker")
        capabilities = self._capabilities
        if (request.provider, request.model) != (self.provider, self.model):
            from victor.config.metrics_capabilities import get_metrics_capabilities

            capabilities = get_metrics_capabilities(request.provider, request.model)

        if usage is not None:
            request.prompt_tokens = usage.get("prompt_tokens", 0)
            request.completion_tokens = usage.get("completion_tokens", 0)
            request.billable_completion_tokens = billable_completion_tokens(usage)
            request.cache_read_tokens = usage.get("cache_read_input_tokens", 0)
            request.cache_write_tokens = usage.get("cache_creation_input_tokens", 0)
            request.total_tokens = usage_total_tokens(usage)
            request.usage_status = "reported" if usage_complete else "partial"
            if capabilities and capabilities.cost_enabled:
                costs = capabilities.calculate_cost(
                    request.prompt_tokens,
                    request.billable_completion_tokens,
                    request.cache_read_tokens,
                    request.cache_write_tokens,
                )
                request.input_cost = costs["input_cost"]
                request.output_cost = costs["output_cost"]
                request.cache_cost = costs["cache_cost"]
                request.total_cost = sum(
                    (request.input_cost, request.output_cost, request.cache_cost)
                )

        request.outcome = outcome
        request.duration_seconds = duration_seconds
        request.tool_calls = tool_calls
        request.transport_attempts = transport_attempts
        del self._pending_requests[request.request_id]
        self.total_prompt_tokens += request.prompt_tokens
        self.total_completion_tokens += request.completion_tokens
        self.total_cache_read_tokens += request.cache_read_tokens
        self.total_cache_write_tokens += request.cache_write_tokens
        self.total_tokens += request.total_tokens
        self.total_input_cost += request.input_cost
        self.total_output_cost += request.output_cost
        self.total_cache_cost += request.cache_cost
        self.total_cost += request.total_cost

        if self._sandhi is not None and request.usage_status == "reported":
            try:
                self._sandhi.record(
                    provider=request.provider,
                    model=request.model,
                    prompt_tokens=request.prompt_tokens,
                    completion_tokens=request.billable_completion_tokens,
                    cache_read_tokens=request.cache_read_tokens,
                    cache_write_tokens=request.cache_write_tokens,
                    subject_id=request.subject_id,
                    group_id=request.group_id,
                    session_id=self.session_id,
                )
            except Exception:
                logger.exception("Usage meter failed for settled request %s", request.request_id)

    def get_summary(self) -> Dict[str, Any]:
        """Get session cost summary.

        Returns:
            Dictionary with session statistics
        """
        elapsed = time.time() - self.start_time
        request_count = len(self.requests)

        return {
            "session_id": self.session_id,
            "provider": self.provider,
            "model": self.model,
            "start_time": self.start_time,
            "start_time_iso": datetime.fromtimestamp(self.start_time).isoformat(),
            "elapsed_seconds": elapsed,
            "request_count": request_count,
            "usage_coverage": {
                status: sum(request.usage_status == status for request in self.requests)
                for status in ("unknown", "partial", "reported")
            },
            "pending_requests": len(self._pending_requests),
            "tokens": {
                "prompt": self.total_prompt_tokens,
                "completion": self.total_completion_tokens,
                "cache_read": self.total_cache_read_tokens,
                "cache_write": self.total_cache_write_tokens,
                "total": self.total_tokens,
            },
            "cost": {
                "input": self.total_input_cost,
                "output": self.total_output_cost,
                "cache": self.total_cache_cost,
                "total": self.total_cost,
            },
            "averages": {
                "tokens_per_request": (
                    self.total_tokens / request_count if request_count > 0 else 0
                ),
                "cost_per_request": (self.total_cost / request_count if request_count > 0 else 0),
            },
            "cost_enabled": (self._capabilities.cost_enabled if self._capabilities else False),
        }

    def get_formatted_summary(self) -> str:
        """Get human-readable session summary.

        Returns:
            Formatted string for display
        """
        summary = self.get_summary()
        lines = [
            f"Session Cost Summary ({self.provider}/{self.model})",
            f"{'=' * 50}",
            f"Requests: {summary['request_count']}",
            f"Duration: {summary['elapsed_seconds']:.1f}s",
            "",
            "Tokens:",
            f"  Input:      {summary['tokens']['prompt']:,}",
            f"  Output:     {summary['tokens']['completion']:,}",
            f"  Cache Read: {summary['tokens']['cache_read']:,}",
            f"  Total:      {summary['tokens']['total']:,}",
        ]

        if summary["cost_enabled"]:
            lines.extend(
                [
                    "",
                    "Cost (USD):",
                    f"  Input:  ${summary['cost']['input']:.4f}",
                    f"  Output: ${summary['cost']['output']:.4f}",
                    f"  Cache:  ${summary['cost']['cache']:.4f}",
                    f"  Total:  ${summary['cost']['total']:.4f}",
                ]
            )
        else:
            lines.extend(["", "Cost tracking not enabled for this provider"])

        return "\n".join(lines)

    def format_inline_cost(self, request_cost: Optional[RequestCost] = None) -> str:
        """Format cost for inline display.

        Args:
            request_cost: Specific request to format, or None for session total

        Returns:
            Formatted string like "$0.015" or "cost n/a"
        """
        if not self._capabilities or not self._capabilities.cost_enabled:
            return "cost n/a"

        if request_cost:
            return f"${request_cost.total_cost:.4f}"
        else:
            return f"${self.total_cost:.4f}"

    def export_json(self, path: Path) -> None:
        """Export session cost data to JSON.

        Args:
            path: Output file path
        """
        data = {
            "summary": self.get_summary(),
            "requests": [r.to_dict() for r in self.requests],
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Exported session costs to {path}")

    def export_csv(self, path: Path) -> None:
        """Export per-request breakdown to CSV.

        Args:
            path: Output file path
        """
        import csv

        headers = [
            "request_id",
            "timestamp",
            "timestamp_iso",
            "model",
            "prompt_tokens",
            "completion_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
            "total_tokens",
            "input_cost",
            "output_cost",
            "cache_cost",
            "total_cost",
            "duration_seconds",
            "tool_calls",
            "provider",
            "outcome",
            "usage_status",
            "transport_attempts",
        ]

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)

            for req in self.requests:
                writer.writerow(
                    [
                        req.request_id,
                        req.timestamp,
                        datetime.fromtimestamp(req.timestamp).isoformat(),
                        req.model,
                        req.prompt_tokens if req.usage_status != "unknown" else None,
                        req.completion_tokens if req.usage_status != "unknown" else None,
                        req.cache_read_tokens if req.usage_status != "unknown" else None,
                        req.cache_write_tokens if req.usage_status != "unknown" else None,
                        req.total_tokens if req.usage_status != "unknown" else None,
                        req.input_cost if req.usage_status != "unknown" else None,
                        req.output_cost if req.usage_status != "unknown" else None,
                        req.cache_cost if req.usage_status != "unknown" else None,
                        req.total_cost if req.usage_status != "unknown" else None,
                        req.duration_seconds,
                        req.tool_calls,
                        req.provider,
                        req.outcome,
                        req.usage_status,
                        req.transport_attempts,
                    ]
                )

        logger.info(f"Exported request costs to {path}")

    def reset(self) -> None:
        """Reset session tracking (start fresh)."""
        if self._pending_requests:
            raise ValueError("Cannot reset a session with pending invocations")
        self.requests.clear()
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cache_read_tokens = 0
        self.total_cache_write_tokens = 0
        self.total_tokens = 0
        self.total_input_cost = 0.0
        self.total_output_cost = 0.0
        self.total_cache_cost = 0.0
        self.total_cost = 0.0
        self.start_time = time.time()
        self.session_id = str(uuid.uuid4())
        logger.debug("Session cost tracker reset")
