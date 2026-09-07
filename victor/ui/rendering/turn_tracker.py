"""Conversation turn tracking for UX display.

Tracks conversation turns with metadata including tool usage, token counts,
cost estimates, and context window awareness, for visual turn boundaries in
the streaming display.

Features:
- **Turn numbering**: Sequential turn tracking with start/end lifecycle
- **Tool usage per turn**: Count and categorize tools used in each turn
- **Context window awareness**: Estimate and warn when approaching limits
- **Cost estimation**: Token and cost tracking per turn
- **Visual boundaries**: Section separators with turn metadata
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

logger = logging.getLogger(__name__)


# Default context window sizes by provider (in tokens)
DEFAULT_CONTEXT_LIMITS: Dict[str, int] = {
    "gpt-4": 8192,
    "gpt-4-turbo": 128000,
    "gpt-4o": 128000,
    "claude-3-opus": 200000,
    "claude-3-sonnet": 200000,
    "claude-3-haiku": 200000,
    "claude-3-5-sonnet": 200000,
    "gemini-pro": 32768,
    "gemini-1.5-pro": 1048576,
    "llama-3": 8192,
    "llama-3.1": 131072,
    "mixtral": 32768,
    "default": 128000,
}

# Cost per 1K tokens (USD) for common models
ESTIMATED_COST_PER_1K: Dict[str, Tuple[float, float]] = {
    "gpt-4": (0.03, 0.06),
    "gpt-4-turbo": (0.01, 0.03),
    "gpt-4o": (0.0025, 0.01),
    "claude-3-opus": (0.015, 0.075),
    "claude-3-sonnet": (0.003, 0.015),
    "claude-3-haiku": (0.00025, 0.00125),
    "claude-3-5-sonnet": (0.003, 0.015),
    "gemini-pro": (0.0005, 0.0015),
    "llama-3": (0.0, 0.0),  # Local models are free
    "default": (0.003, 0.015),
}


@dataclass
class TurnMetrics:
    """Metrics collected for a single conversation turn.

    Attributes:
        turn_number: Sequential turn number (1-based)
        tool_calls: Number of tool invocations in this turn
        tool_categories: Tool categories used in this turn
        input_tokens: Estimated input tokens for this turn
        output_tokens: Estimated output tokens for this turn
        cost_estimate_usd: Estimated cost in USD
        duration_ms: Duration of the turn in milliseconds
        start_time: Wall clock time when the turn started
        end_time: Wall clock time when the turn ended
    """

    turn_number: int = 0
    tool_calls: int = 0
    tool_categories: List[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cost_estimate_usd: float = 0.0
    duration_ms: float = 0.0
    start_time: float = 0.0
    end_time: float = 0.0
    tool_durations_ms: Dict[str, float] = field(default_factory=dict)
    # Wire truth from the sandhi run cost tree (ADR-0005 D7), when reachable.
    # Neutral units (ADR-0001): the cache split and billable come straight from
    # the metered events; None means no tree was available this turn.
    wire_cache_read_tokens: Optional[int] = None
    wire_billable_tokens: Optional[int] = None


@dataclass
class ContextUsage:
    """Current context window usage information.

    Attributes:
        estimated_tokens: Estimated total tokens in context
        max_tokens: Maximum context window size
        usage_ratio: Ratio of used to max (0.0 to 1.0)
        is_approaching_limit: True if usage > 80% of max
        is_at_limit: True if usage > 95% of max
        model: Model name used for context limit
    """

    estimated_tokens: int = 0
    max_tokens: int = 128000
    usage_ratio: float = 0.0
    is_approaching_limit: bool = False
    is_at_limit: bool = False
    model: str = "default"


class TurnTracker:
    """Tracks conversation turns with metadata.

    Provides structured turn lifecycle management for the streaming display.
    Each turn captures tool usage, token estimates, cost, and duration.

    Usage:
        tracker = TurnTracker()
        tracker.start_turn()
        tracker.record_tool_call("code_search")
        tracker.record_tool_call("read")
        tracker.end_turn(input_tokens=500, output_tokens=200)
        metrics = tracker.get_current_turn_metrics()
        summary = tracker.get_session_summary()
    """

    def __init__(self, model: str = "default"):
        """Initialize TurnTracker.

        Args:
            model: Model name for context limit and cost estimation
        """
        self._model = model
        self._current_turn: Optional[TurnMetrics] = None
        self._completed_turns: List[TurnMetrics] = []
        self._turn_number: int = 0
        self._accumulated_input_tokens: int = 0
        self._accumulated_output_tokens: int = 0
        self._accumulated_cost: float = 0.0
        self._accumulated_tool_calls: int = 0
        self._session_start_time: float = time.monotonic()

    # ── Turn Lifecycle ────────────────────────────────────────────

    def _overlay_wire_truth(self) -> None:
        """Best-effort: stamp this turn's metrics from the sandhi run cost tree.

        Reads the step row the proxy folded for this turn's step id (own, not
        rollup) and records the neutral split — cache read and billable — next
        to the price-table estimate. Never raises, never blocks: every failure
        path (no gateway route, no admin token, unknown run, fetch error)
        leaves the estimate standing and the wire fields None.
        """
        if self._current_turn is None:
            return
        try:
            from victor.core.context import get_session_id, get_turn_id

            run_id = str(get_session_id() or "")
            step_id = str(get_turn_id() or "")
            if not run_id or not step_id:
                return
            from victor.observability.sandhi_admin import fetch_run_cost_tree, find_step

            # The proxy persists through a buffered writer when SANDHI_STORE is
            # set, so the just-finished turn's row can land a beat after the
            # response returns. Two short retries cover the common backlog;
            # after that the estimate stands.
            row = None
            for attempt in range(3):
                tree = fetch_run_cost_tree(run_id)
                if tree is not None:
                    row = find_step(tree, step_id)
                    if row is not None:
                        break
                if attempt < 2:
                    time.sleep(0.05)
            if row is None:
                return
            self._current_turn.wire_cache_read_tokens = int(row["cache_read_tokens"])
            self._current_turn.wire_billable_tokens = int(row["billable_tokens"])
        except Exception:  # pragma: no cover - display must never block on metering
            return

    def start_turn(self) -> int:
        """Begin a new conversation turn.

        Returns:
            The new turn number (1-based)
        """
        # Finalize any in-progress turn
        if self._current_turn is not None:
            self._finalize_current_turn()

        self._turn_number += 1
        self._current_turn = TurnMetrics(
            turn_number=self._turn_number,
            start_time=time.monotonic(),
        )
        logger.debug("Turn %d started", self._turn_number)
        return self._turn_number

    def end_turn(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> TurnMetrics:
        """End the current conversation turn and record metrics.

        Args:
            input_tokens: Estimated input tokens for this turn
            output_tokens: Estimated output tokens for this turn

        Returns:
            TurnMetrics for the completed turn

        Raises:
            RuntimeError: If no turn is active
        """
        if self._current_turn is None:
            raise RuntimeError("No active turn to end")

        self._current_turn.input_tokens = input_tokens
        self._current_turn.output_tokens = output_tokens
        self._current_turn.end_time = time.monotonic()
        self._current_turn.duration_ms = (
            self._current_turn.end_time - self._current_turn.start_time
        ) * 1000

        # Estimate cost (fallback — replaced by wire truth below when a tree exists)
        input_cost, output_cost = ESTIMATED_COST_PER_1K.get(
            self._model, ESTIMATED_COST_PER_1K["default"]
        )
        self._current_turn.cost_estimate_usd = (input_tokens / 1000) * input_cost + (
            output_tokens / 1000
        ) * output_cost
        self._overlay_wire_truth()

        metrics = self._finalize_current_turn()
        logger.debug(
            "Turn %d ended: %d tools, %d+%d tokens, $%.4f, %.1fs",
            metrics.turn_number,
            metrics.tool_calls,
            metrics.input_tokens,
            metrics.output_tokens,
            metrics.cost_estimate_usd,
            metrics.duration_ms / 1000,
        )
        return metrics

    def _finalize_current_turn(self) -> TurnMetrics:
        """Finalize and store the current turn metrics."""
        assert self._current_turn is not None
        metrics = self._current_turn
        self._completed_turns.append(metrics)
        self._accumulated_input_tokens += metrics.input_tokens
        self._accumulated_output_tokens += metrics.output_tokens
        self._accumulated_cost += metrics.cost_estimate_usd
        self._accumulated_tool_calls += metrics.tool_calls
        self._current_turn = None
        return metrics

    # ── Tool Recording ────────────────────────────────────────────

    def record_tool_call(self, tool_name: str) -> None:
        """Record a tool invocation in the current turn.

        Args:
            tool_name: Name of the tool that was called
        """
        if self._current_turn is None:
            logger.debug("No active turn, starting one for tool call")
            self.start_turn()

        self._current_turn.tool_calls += 1

        # Extract category from tool name (e.g., "code_search" -> "search")
        category = self._categorize_tool(tool_name)
        if category and category not in self._current_turn.tool_categories:
            self._current_turn.tool_categories.append(category)

    def record_tool_result(self, tool_name: str, duration_ms: float) -> None:
        """Record the result of a tool execution with its duration.

        Accumulates per-tool timing data for the current turn, enabling
        data-driven identification of slow tools. Logged at INFO so the
        timing is visible in victor.log for offline analysis.
        """
        if self._current_turn is None:
            return
        # Track per-tool cumulative duration for this turn.
        prev = self._current_turn.tool_durations_ms.get(tool_name, 0.0)
        self._current_turn.tool_durations_ms[tool_name] = prev + duration_ms
        if duration_ms > 100:
            logger.info(
                "tool_timing: %s took %.0fms (turn %d)",
                tool_name,
                duration_ms,
                self._current_turn.turn_number,
            )

    @staticmethod
    def _categorize_tool(tool_name: str) -> Optional[str]:
        """Categorize a tool by its name."""
        tool_lower = tool_name.lower()
        categories = {
            "filesystem": ["read", "write", "ls", "edit", "file_info"],
            "search": ["code_search", "semantic_code_search", "search", "grep"],
            "git": ["git_status", "git_diff", "git_log", "git_blame", "git"],
            "analysis": ["overview", "analyze", "inspect", "metrics"],
            "execution": ["shell", "bash", "run", "code_exec"],
            "web": ["web_search", "fetch", "http", "web_fetch"],
            "testing": ["test", "pytest", "run_tests"],
            "build": ["build", "compile", "make"],
        }
        for category, prefixes in categories.items():
            for prefix in prefixes:
                if tool_lower.startswith(prefix):
                    return category
        return "other"

    # ── Context Window ────────────────────────────────────────────

    def get_context_usage(self) -> ContextUsage:
        """Return current context window usage.

        Estimates total tokens in context based on accumulated input
        and output tokens across all completed turns plus the current turn.

        Returns:
            ContextUsage with estimated usage and warnings
        """
        current_input = self._current_turn.input_tokens if self._current_turn else 0
        current_output = self._current_turn.output_tokens if self._current_turn else 0

        estimated_tokens = (
            self._accumulated_input_tokens
            + self._accumulated_output_tokens
            + current_input
            + current_output
        )

        max_tokens = DEFAULT_CONTEXT_LIMITS.get(self._model, DEFAULT_CONTEXT_LIMITS["default"])
        usage_ratio = estimated_tokens / max_tokens if max_tokens > 0 else 0.0

        return ContextUsage(
            estimated_tokens=estimated_tokens,
            max_tokens=max_tokens,
            usage_ratio=usage_ratio,
            is_approaching_limit=usage_ratio > 0.8,
            is_at_limit=usage_ratio > 0.95,
            model=self._model,
        )

    # ── Query Methods ─────────────────────────────────────────────

    def get_current_turn_metrics(self) -> Optional[TurnMetrics]:
        """Return metrics for the current (in-progress) turn.

        Returns:
            TurnMetrics if a turn is active, None otherwise
        """
        return self._current_turn

    def get_completed_turns(self) -> List[TurnMetrics]:
        """Return metrics for all completed turns.

        Returns:
            List of completed TurnMetrics objects
        """
        return list(self._completed_turns)

    def get_turn_count(self) -> int:
        """Return the total number of turns (completed + current).

        Returns:
            Total turn count
        """
        count = len(self._completed_turns)
        if self._current_turn is not None:
            count += 1
        return count

    def get_session_summary(self) -> Dict[str, Any]:
        """Return a summary of the entire session.

        Returns:
            Dict with session-level metrics
        """
        session_duration = time.monotonic() - self._session_start_time
        context = self.get_context_usage()

        return {
            "total_turns": self.get_turn_count(),
            "completed_turns": len(self._completed_turns),
            "total_tool_calls": self._accumulated_tool_calls,
            "total_input_tokens": self._accumulated_input_tokens,
            "total_output_tokens": self._accumulated_output_tokens,
            "total_cost_usd": round(self._accumulated_cost, 6),
            "session_duration_s": round(session_duration, 1),
            "context_usage": {
                "estimated_tokens": context.estimated_tokens,
                "max_tokens": context.max_tokens,
                "usage_ratio": round(context.usage_ratio, 3),
                "approaching_limit": context.is_approaching_limit,
                "at_limit": context.is_at_limit,
            },
            "model": self._model,
        }

    # ── Display Helpers ───────────────────────────────────────────

    def format_turn_header(self, turn_number: Optional[int] = None) -> str:
        """Format a turn header for display.

        Args:
            turn_number: Turn number (uses current turn if None)

        Returns:
            Formatted turn header string
        """
        tn = turn_number or self._turn_number
        return f"--- Turn {tn} ---"

    def format_turn_metadata(self, metrics: Optional[TurnMetrics] = None) -> str:
        """Format turn metadata for display.

        Args:
            metrics: TurnMetrics to format (uses current turn if None)

        Returns:
            Formatted metadata string
        """
        m = metrics or self._current_turn
        if m is None:
            return ""

        parts = []
        if m.tool_categories:
            parts.append(" | ".join(c.capitalize() for c in m.tool_categories))
        if m.input_tokens or m.output_tokens:
            parts.append(f"Tokens: {m.input_tokens} in / {m.output_tokens} out")
        if m.wire_billable_tokens is not None:
            # Wire truth from the sandhi run cost tree: billable (fresh +
            # cache-read + out, ADR-0005 D4) and the cache share. Shown in
            # preference to the dollar estimate, which stays as the fallback.
            cache = m.wire_cache_read_tokens or 0
            share = f", {cache} cached" if cache else ""
            parts.append(f"Metered: {m.wire_billable_tokens} billable{share}")
        if m.cost_estimate_usd > 0:
            parts.append(f"Cost: ~${m.cost_estimate_usd:.4f}")
        if m.duration_ms > 0:
            parts.append(f"Duration: {m.duration_ms / 1000:.1f}s")

        return " | ".join(parts) if parts else ""

    def format_context_warning(self) -> Optional[str]:
        """Format a context window warning if approaching limits.

        Returns:
            Warning string if approaching limits, None otherwise
        """
        context = self.get_context_usage()
        if context.is_at_limit:
            return (
                f"[red]⚠ Context window nearly full "
                f"({context.estimated_tokens:,} / {context.max_tokens:,} tokens)[/]"
            )
        if context.is_approaching_limit:
            return (
                f"[yellow]Context window at "
                f"{context.usage_ratio:.0%} "
                f"({context.estimated_tokens:,} / {context.max_tokens:,} tokens)[/]"
            )
        return None
