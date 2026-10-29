"""Runtime credit tracking service for automatic tool-level credit assignment.

Bridges the gap between ToolPipeline execution results and the credit
assignment system. Accumulates per-turn tool results, assigns credit at
turn boundaries, and emits credit signals to the ObservabilityBus.

This is the Phase 3 integration layer (FEP-0001) that makes credit
assignment automatic rather than opt-in.

Usage:
    # Created by InitializationPhaseManager (phase 9: credit_runtime)
    service = CreditTrackingService(settings, observability_bus)

    # Called by ToolPipeline.on_tool_complete callback
    service.record_tool_result(tool_call_result)

    # Called at turn boundary (after LLM response)
    signals = service.assign_turn_credit()
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from victor.framework.rl.credit_assignment import (
    ActionMetadata,
    CreditAssignmentConfig,
    CreditAssignmentIntegration,
    CreditGranularity,
    CreditMethodology,
    CreditSignal,
    compute_credit_metrics,
)
from victor.framework.tool_naming import get_canonical_name

logger = logging.getLogger(__name__)


# ============================================================================
# Reward extraction from tool results
# ============================================================================


@dataclass
class ToolRewardSignal:
    """Extracted reward signal from a single tool execution."""

    tool_name: str
    success: bool
    reward: float
    execution_time_ms: float
    agent_id: str = "default"
    team_id: Optional[str] = None
    session_id: Optional[str] = None
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    arguments_summary: str = ""


def extract_reward_from_tool_result(
    tool_name: str,
    success: bool,
    execution_time_ms: float,
    error: Optional[str] = None,
) -> float:
    """Extract a scalar reward from a tool execution result.

    Reward heuristic:
    - Success: +1.0, scaled down slightly for slow tools (>5s)
    - Failure: -1.0, with extra penalty for timeouts
    - Skipped/cached: +0.5 (still useful, less novel)

    Args:
        tool_name: Name of the executed tool
        success: Whether execution succeeded
        execution_time_ms: Execution duration in milliseconds
        error: Optional error message

    Returns:
        Scalar reward in [-2.0, 1.0]
    """
    if success:
        # Base reward for success
        reward = 1.0
        # Slight penalty for very slow tools (>5s) — prefer efficient executions
        if execution_time_ms > 5000:
            reward -= min(0.3, (execution_time_ms - 5000) / 30000)
        return reward
    else:
        # Base penalty for failure
        reward = -1.0
        # Extra penalty for timeouts (wasted time budget)
        if error and "timed out" in error.lower():
            reward = -2.0
        return reward


# ============================================================================
# Credit Tracking Service
# ============================================================================


class CreditTrackingService:
    """Runtime service that tracks tool executions and assigns credit.

    Lifecycle:
    1. Created during initialization (phase 9: credit_runtime)
    2. Attached to ToolPipeline via on_tool_complete callback
    3. Accumulates ToolRewardSignals during each turn
    4. At turn boundary, assigns credit via CreditAssignmentIntegration
    5. Emits credit signals to ObservabilityBus
    6. Optionally persists to SQLite for historical analysis
    """

    def __init__(
        self,
        methodology: CreditMethodology = CreditMethodology.GAE,
        config: Optional[CreditAssignmentConfig] = None,
        observability_bus: Optional[Any] = None,
        emit_events: bool = True,
        persist: bool = False,
        auto_assign_at_turn_boundary: bool = True,
    ):
        """Initialize the credit tracking service.

        Args:
            methodology: Default credit assignment methodology
            config: Credit assignment configuration
            observability_bus: ObservabilityBus for event emission
            emit_events: Whether to emit credit.* events
            persist: Whether to persist credit data to SQLite
            auto_assign_at_turn_boundary: Whether the runtime should call
                :meth:`assign_turn_credit_at_boundary` automatically each turn
        """
        self._methodology = methodology
        self._config = config or CreditAssignmentConfig(methodology=methodology)
        self._observability_bus = observability_bus
        self._emit_events = emit_events
        self._persist = persist
        self._auto_assign_at_turn_boundary = auto_assign_at_turn_boundary

        # Per-turn accumulator
        self._current_turn_signals: List[ToolRewardSignal] = []
        self._turn_count: int = 0

        # Cross-turn integration
        self._integration = CreditAssignmentIntegration(default_config=self._config)

        # History for GEPA enrichment
        self._recent_credit_signals: List[CreditSignal] = []
        self._max_history = 200

        logger.debug(
            "CreditTrackingService initialized (methodology=%s, emit=%s, persist=%s)",
            methodology.value,
            emit_events,
            persist,
        )

    @classmethod
    def from_settings(
        cls, settings: Any, observability_bus: Optional[Any] = None
    ) -> "CreditTrackingService":
        """Create from Victor settings.

        Args:
            settings: Victor Settings object
            observability_bus: Optional ObservabilityBus instance

        Returns:
            Configured CreditTrackingService
        """
        ca_settings = getattr(settings, "credit_assignment", None)
        if ca_settings is None:
            # Default settings
            return cls(observability_bus=observability_bus)

        # Map string methodology to enum
        methodology_map = {m.value: m for m in CreditMethodology}
        methodology = methodology_map.get(ca_settings.default_methodology, CreditMethodology.GAE)

        config = CreditAssignmentConfig(
            methodology=methodology,
            gamma=ca_settings.gamma,
            lambda_gae=ca_settings.lambda_gae,
            shapley_sampling_count=ca_settings.shapley_sampling_count,
        )

        return cls(
            methodology=methodology,
            config=config,
            observability_bus=observability_bus,
            emit_events=ca_settings.emit_observability_events,
            persist=ca_settings.persist_to_db,
            auto_assign_at_turn_boundary=ca_settings.auto_assign_at_turn_boundary,
        )

    # ----------------------------------------------------------------
    # Tool result recording (called by ToolPipeline callback)
    # ----------------------------------------------------------------

    def record_tool_result(
        self,
        tool_name: str,
        success: bool,
        execution_time_ms: float,
        error: Optional[str] = None,
        arguments: Optional[Dict[str, Any]] = None,
        agent_id: str = "default",
        team_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> ToolRewardSignal:
        """Record a tool execution result for the current turn.

        Called by the ToolPipeline's on_tool_complete callback after
        each tool execution.

        Args:
            tool_name: Name of the executed tool
            success: Whether execution succeeded
            execution_time_ms: Execution duration
            error: Optional error message
            arguments: Optional tool arguments (for context)
            agent_id: Agent that initiated the call
            team_id: Optional team identifier for multi-agent runs
            session_id: Optional execution/session identifier for trace alignment

        Returns:
            The extracted ToolRewardSignal
        """
        tool_name = get_canonical_name(tool_name)
        reward = extract_reward_from_tool_result(tool_name, success, execution_time_ms, error)

        signal = ToolRewardSignal(
            tool_name=tool_name,
            agent_id=agent_id,
            team_id=team_id,
            session_id=session_id,
            success=success,
            reward=reward,
            execution_time_ms=execution_time_ms,
            error=error,
            arguments_summary=self._summarize_args(arguments) if arguments else "",
        )

        self._current_turn_signals.append(signal)

        logger.debug(
            "Recorded tool result: %s (success=%s, reward=%.2f)",
            tool_name,
            success,
            reward,
        )

        return signal

    # ----------------------------------------------------------------
    # Turn boundary credit assignment
    # ----------------------------------------------------------------

    def assign_turn_credit(
        self,
        agent_id: str = "default",
        methodology: Optional[CreditMethodology] = None,
        team_id: Optional[str] = None,
    ) -> List[CreditSignal]:
        """Assign credit for all tool executions in the current turn.

        Called at the turn boundary (after LLM response is complete).
        Converts accumulated ToolRewardSignals into ActionMetadata,
        runs credit assignment, emits events, and resets for next turn.

        Args:
            agent_id: Agent ID for this turn
            methodology: Override methodology for this turn
            team_id: Optional team ID for this turn

        Returns:
            List of CreditSignal for this turn's tools
        """
        if not self._current_turn_signals:
            return []

        method = methodology or self._methodology
        # MONTE_CARLO maps to SegmentLevelCreditAssigner which expects
        # TrajectorySegment, not ActionMetadata. Remap to GAE for step-level.
        if method == CreditMethodology.MONTE_CARLO:
            method = CreditMethodology.GAE
        self._turn_count += 1

        # Build trajectory from accumulated signals
        trajectory: List[ActionMetadata] = []
        rewards: List[float] = []

        for i, signal in enumerate(self._current_turn_signals):
            signal_agent_id = signal.agent_id if signal.agent_id != "default" else agent_id
            metadata = ActionMetadata(
                agent_id=signal_agent_id,
                team_id=signal.team_id or team_id,
                action_id=f"turn{self._turn_count}_tool{i}_{signal.tool_name}",
                turn_index=self._turn_count,
                step_index=i,
                tool_name=signal.tool_name,
                timestamp=signal.timestamp,
                duration_ms=int(signal.execution_time_ms),
                session_id=signal.session_id,
            )
            trajectory.append(metadata)
            rewards.append(signal.reward)

        # Assign credit
        try:
            credit_signals = self._integration.assign_credit(
                trajectory, rewards, method, self._config
            )
        except Exception as e:
            logger.warning("Credit assignment failed for turn %d: %s", self._turn_count, e)
            credit_signals = []

        # Emit observability events
        if self._emit_events and credit_signals:
            self._emit_credit_events(credit_signals)

        # Persist if configured
        if self._persist and credit_signals:
            self._persist_signals(credit_signals)

        # Store in history for GEPA enrichment
        self._recent_credit_signals.extend(credit_signals)
        if len(self._recent_credit_signals) > self._max_history:
            self._recent_credit_signals = self._recent_credit_signals[-self._max_history :]

        # Reset for next turn
        tool_count = len(self._current_turn_signals)
        self._current_turn_signals = []

        logger.debug(
            "Turn %d credit assigned: %d tools → %d signals",
            self._turn_count,
            tool_count,
            len(credit_signals),
        )

        return credit_signals

    def assign_turn_credit_at_boundary(self) -> None:
        """Run per-turn credit assignment from the runtime turn boundary.

        Thin, exception-safe wrapper the orchestrator calls once per completed turn
        (streaming, non-streaming, and planning alike). Respects the
        ``auto_assign_at_turn_boundary`` setting and never propagates failures — a
        credit-assignment error must not break the turn. :meth:`assign_turn_credit`
        itself no-ops when no tools ran this turn.
        """
        if not self._auto_assign_at_turn_boundary:
            return
        try:
            self.assign_turn_credit()
        except Exception:  # pragma: no cover - feedback loop must never break a turn
            logger.debug("assign_turn_credit at turn boundary failed", exc_info=True)

    # ----------------------------------------------------------------
    # GEPA enrichment interface
    # ----------------------------------------------------------------

    def get_recent_credit_signals(self, limit: int = 50) -> List[CreditSignal]:
        """Get recent credit signals for GEPA trace enrichment.

        Args:
            limit: Maximum number of signals to return

        Returns:
            Recent credit signals, newest first
        """
        return list(reversed(self._recent_credit_signals[-limit:]))

    def get_tool_credit_summary(
        self,
        *,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        team_id: Optional[str] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Get aggregate credit summary by tool name.

        Returns:
            Dict mapping tool_name → {total_credit, avg_credit, call_count, success_rate}
        """
        tool_stats: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"total_credit": 0.0, "call_count": 0, "successes": 0}
        )

        for signal in self._recent_credit_signals:
            if not self._matches_filters(
                signal,
                session_id=session_id,
                agent_id=agent_id,
                team_id=team_id,
            ):
                continue
            if signal.metadata and signal.metadata.tool_name:
                tool = signal.metadata.tool_name
                tool_stats[tool]["total_credit"] += signal.credit
                tool_stats[tool]["call_count"] += 1
                if signal.credit > 0:
                    tool_stats[tool]["successes"] += 1

        result: Dict[str, Dict[str, float]] = {}
        for tool, stats in tool_stats.items():
            count = stats["call_count"]
            result[tool] = {
                "total_credit": stats["total_credit"],
                "avg_credit": stats["total_credit"] / count if count > 0 else 0.0,
                "call_count": float(count),
                "success_rate": stats["successes"] / count if count > 0 else 0.0,
            }

        return result

    def get_agent_credit_summary(
        self,
        *,
        session_id: Optional[str] = None,
        team_id: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Get aggregate credit summary by agent ID.

        Returns:
            Dict mapping agent_id → {total_credit, avg_credit, call_count, success_rate, team_id}
        """
        agent_stats: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "total_credit": 0.0,
                "call_count": 0,
                "successes": 0,
                "team_id": None,
            }
        )

        for signal in self._recent_credit_signals:
            if not self._matches_filters(
                signal,
                session_id=session_id,
                team_id=team_id,
            ):
                continue
            if signal.metadata and signal.metadata.agent_id:
                agent_id = signal.metadata.agent_id
                agent_stats[agent_id]["total_credit"] += signal.credit
                agent_stats[agent_id]["call_count"] += 1
                if signal.credit > 0:
                    agent_stats[agent_id]["successes"] += 1
                if signal.metadata.team_id:
                    agent_stats[agent_id]["team_id"] = signal.metadata.team_id

        result: Dict[str, Dict[str, Any]] = {}
        for agent_id, stats in agent_stats.items():
            count = stats["call_count"]
            result[agent_id] = {
                "total_credit": stats["total_credit"],
                "avg_credit": stats["total_credit"] / count if count > 0 else 0.0,
                "call_count": float(count),
                "success_rate": stats["successes"] / count if count > 0 else 0.0,
                "team_id": stats["team_id"],
            }

        return result

    # ----------------------------------------------------------------
    # Feedback loop: credit-driven guidance for prompt injection
    # ----------------------------------------------------------------

    def generate_tool_guidance(
        self,
        max_hints: int = 5,
        *,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        team_id: Optional[str] = None,
    ) -> Optional[str]:
        """Generate concise tool effectiveness guidance from credit history.

        This is the core feedback loop: credit signals flow back into the
        agent's next turn via prompt injection. Returns a short guidance
        string (under 200 tokens) that can be added to the system prompt
        or injected into the user message for KV-stable providers.

        Only generates guidance after sufficient data (>= 3 turns with
        >= 5 total tool calls). Returns None if insufficient data.

        Args:
            max_hints: Maximum number of tool-specific hints to include

        Returns:
            Guidance string or None if insufficient data
        """
        summary = self.get_tool_credit_summary(
            session_id=session_id,
            agent_id=agent_id,
            team_id=team_id,
        )
        if not summary:
            return None

        # Require minimum data before giving guidance
        total_calls = sum(int(s["call_count"]) for s in summary.values())
        if self._turn_count < 3 or total_calls < 5:
            return None

        # Rank tools by average credit
        ranked = sorted(summary.items(), key=lambda x: x[1]["avg_credit"])

        lines = ["Tool effectiveness (from recent execution credit):"]

        # Flag underperforming tools (negative avg credit)
        underperforming = [(t, s) for t, s in ranked if s["avg_credit"] < -0.3]
        for tool, stats in underperforming[:max_hints]:
            calls = int(stats["call_count"])
            lines.append(
                f"- {tool}: low effectiveness (avg credit {stats['avg_credit']:+.1f} "
                f"over {calls} calls). Verify arguments before using."
            )

        # Highlight high-value tools (positive avg credit)
        high_value = [(t, s) for t, s in reversed(ranked) if s["avg_credit"] > 0.5]
        for tool, stats in high_value[:max_hints]:
            calls = int(stats["call_count"])
            lines.append(
                f"- {tool}: high effectiveness (avg credit {stats['avg_credit']:+.1f} "
                f"over {calls} calls)."
            )

        if len(lines) == 1:
            return None  # No actionable hints

        return "\n".join(lines)

    def generate_agent_guidance(
        self,
        max_agents: int = 4,
        *,
        session_id: Optional[str] = None,
        team_id: Optional[str] = None,
    ) -> Optional[str]:
        """Generate concise agent-specific guidance from recent team credit.

        Returns a short textual summary that can be fed into prompt optimization
        for offline reflection in multi-agent scenarios.
        """
        summary = self.get_agent_credit_summary(session_id=session_id, team_id=team_id)
        if not summary:
            return None

        total_calls = sum(int(stats["call_count"]) for stats in summary.values())
        if total_calls < 4:
            return None

        ranked = sorted(summary.items(), key=lambda item: item[1]["avg_credit"])
        lines = ["Agent execution credit (from recent team runs):"]

        underperforming = [(agent, stats) for agent, stats in ranked if stats["avg_credit"] < -0.3]
        for agent_id, stats in underperforming[:max_agents]:
            calls = int(stats["call_count"])
            lines.append(
                f"- {agent_id}: low effectiveness (avg credit {stats['avg_credit']:+.1f} "
                f"over {calls} calls). Tighten tool-choice and handoff discipline."
            )

        high_value = [
            (agent, stats) for agent, stats in reversed(ranked) if stats["avg_credit"] > 0.0
        ]
        for agent_id, stats in high_value[:max_agents]:
            calls = int(stats["call_count"])
            lines.append(
                f"- {agent_id}: high effectiveness (avg credit {stats['avg_credit']:+.1f} "
                f"over {calls} calls). Preserve this execution pattern."
            )

        if len(lines) == 1:
            return None

        return "\n".join(lines)

    # ----------------------------------------------------------------
    # State
    # ----------------------------------------------------------------

    @property
    def turn_count(self) -> int:
        """Number of completed turns."""
        return self._turn_count

    @property
    def pending_signals(self) -> int:
        """Number of tool results pending credit assignment."""
        return len(self._current_turn_signals)

    def reset(self) -> None:
        """Reset all state."""
        self._current_turn_signals.clear()
        self._recent_credit_signals.clear()
        self._integration.reset()
        self._turn_count = 0

    # ----------------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------------

    def _emit_credit_events(self, signals: List[CreditSignal]) -> None:
        """Emit credit signals to ObservabilityBus."""
        if self._observability_bus is None:
            return

        try:
            from victor.core.events.emit_helper import emit_event_sync

            # Emit per-tool signals
            for signal in signals:
                emit_event_sync(
                    self._observability_bus,
                    "credit.tool_signal",
                    {
                        "action_id": signal.action_id,
                        "tool_name": (signal.metadata.tool_name if signal.metadata else None),
                        "raw_reward": signal.raw_reward,
                        "credit": signal.credit,
                        "confidence": signal.confidence,
                        "methodology": (signal.methodology.value if signal.methodology else None),
                        "agent_id": (signal.metadata.agent_id if signal.metadata else None),
                        "turn_index": (signal.metadata.turn_index if signal.metadata else None),
                    },
                )

            # Emit turn summary
            metrics = compute_credit_metrics(signals)
            emit_event_sync(
                self._observability_bus,
                "credit.turn_summary",
                {
                    "turn": self._turn_count,
                    "tool_count": len(signals),
                    "total_credit": metrics.get("total_credit", 0.0),
                    "avg_credit": metrics.get("avg_credit", 0.0),
                    "positive_ratio": metrics.get("positive_ratio", 0.0),
                },
            )
        except Exception as e:
            logger.debug("Failed to emit credit events: %s", e)

    def _persist_signals(self, signals: List[CreditSignal]) -> None:
        """Persist credit signals to SQLite."""
        try:
            from victor.framework.rl.credit_persistence import get_persistent_db

            db = get_persistent_db()
            db.save_session(
                session_id=f"turn_{self._turn_count}_{int(time.time())}",
                methodology=self._methodology,
                granularity=CreditGranularity.STEP,
                signals=signals,
                success=any(s.credit > 0 for s in signals),
                duration=sum(s.metadata.duration_ms for s in signals if s.metadata) / 1000.0,
            )
        except Exception as e:
            logger.debug("Failed to persist credit signals: %s", e)

    def _summarize_args(self, arguments: Optional[Dict[str, Any]]) -> str:
        """Create a short summary of tool arguments for context."""
        if not arguments:
            return ""
        # Just capture the keys and any file paths
        parts = []
        for key, val in arguments.items():
            if key in ("path", "file_path", "filename"):
                parts.append(f"{key}={val}")
            else:
                parts.append(key)
        return ", ".join(parts[:5])

    @staticmethod
    def _matches_filters(
        signal: CreditSignal,
        *,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        team_id: Optional[str] = None,
    ) -> bool:
        """Return whether a stored credit signal matches the requested scope."""
        metadata = signal.metadata
        if metadata is None:
            return False
        if session_id is not None and metadata.session_id != session_id:
            return False
        if agent_id is not None and metadata.agent_id != agent_id:
            return False
        if team_id is not None and metadata.team_id != team_id:
            return False
        return True


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "CreditTrackingService",
    "ToolRewardSignal",
    "extract_reward_from_tool_result",
]
