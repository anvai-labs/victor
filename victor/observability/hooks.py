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

"""State machine hooks implementing the Observer pattern.

Provides a decoupled way to observe state machine transitions without
modifying the state machine itself.

Design Patterns:
    - Observer: StateHookManager notifies registered hooks
    - Strategy: Each hook implements its own handling strategy

Example:
    from victor.observability import StateHookManager, StateTransitionHook

    manager = StateHookManager()

    # Add a logging hook
    @manager.on_transition
    def log_transition(old_stage, new_stage, context):
        print(f"Transition: {old_stage} -> {new_stage}")

    # Add a hook for specific stages
    @manager.on_enter("EXECUTION")
    def on_execution_start(stage, context):
        print("Starting execution phase!")
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Deque, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)


# Type aliases for hook callbacks
TransitionCallback = Callable[[str, str, Dict[str, Any]], None]
StageCallback = Callable[[str, Dict[str, Any]], None]
# Enhanced callback with history access
HistoryAwareCallback = Callable[[str, str, Dict[str, Any], "TransitionHistory"], None]


@dataclass(frozen=True)
class TransitionRecord:
    """Immutable record of a single state transition.

    Captures all details of a transition for history tracking and analysis.

    Attributes:
        old_stage: The stage being exited.
        new_stage: The stage being entered.
        timestamp: When the transition occurred (UTC).
        context: Copy of the transition context.
        duration_ms: Time spent in old_stage since entering it (if known).
        sequence_number: Monotonically increasing sequence number.
    """

    old_stage: str
    new_stage: str
    timestamp: datetime
    context: Dict[str, Any]
    duration_ms: Optional[float] = None
    sequence_number: int = 0

    def __str__(self) -> str:
        """Human-readable representation."""
        ts = self.timestamp.strftime("%H:%M:%S.%f")[:-3]
        dur = f" ({self.duration_ms:.1f}ms)" if self.duration_ms else ""
        return f"[{ts}] {self.old_stage} -> {self.new_stage}{dur}"


class TransitionHistory:
    """Queryable history of state transitions.

    Provides analytics and query capabilities over transition records.
    Implements a fixed-size circular buffer for memory efficiency.

    Attributes:
        max_size: Maximum number of records to keep.
        records: Deque of transition records.
    """

    def __init__(self, max_size: int = 1000) -> None:
        """Initialize the history.

        Args:
            max_size: Maximum number of records to keep.
        """
        self.max_size = max_size
        self.records: Deque[TransitionRecord] = deque(maxlen=max_size)
        self._sequence: int = 0
        self._stage_enter_times: Dict[str, float] = {}

    def add(self, record: TransitionRecord) -> None:
        """Add a transition record.

        Args:
            record: Record to add.
        """
        self.records.append(record)

    def create_record(
        self,
        old_stage: str,
        new_stage: str,
        context: Dict[str, Any],
    ) -> TransitionRecord:
        """Create and add a new transition record.

        Args:
            old_stage: Stage being exited.
            new_stage: Stage being entered.
            context: Transition context.

        Returns:
            The created record.
        """
        self._sequence += 1
        now = time.time()

        # Calculate duration in old stage
        duration_ms = None
        if old_stage in self._stage_enter_times:
            duration_ms = (now - self._stage_enter_times[old_stage]) * 1000

        # Track when we enter the new stage
        self._stage_enter_times[new_stage] = now

        record = TransitionRecord(
            old_stage=old_stage,
            new_stage=new_stage,
            timestamp=datetime.now(timezone.utc),
            context=dict(context),  # Copy to prevent mutation
            duration_ms=duration_ms,
            sequence_number=self._sequence,
        )
        self.add(record)
        return record

    @property
    def current_stage(self) -> Optional[str]:
        """Get the current stage (last transition's new_stage)."""
        if self.records:
            return self.records[-1].new_stage
        return None

    @property
    def previous_stage(self) -> Optional[str]:
        """Get the previous stage (last transition's old_stage)."""
        if self.records:
            return self.records[-1].old_stage
        return None

    def get_last(self, n: int = 1) -> List[TransitionRecord]:
        """Get the last N transitions.

        Args:
            n: Number of records to return.

        Returns:
            List of most recent records (newest last).
        """
        return list(self.records)[-n:]

    def get_transitions_to(self, stage: str) -> List[TransitionRecord]:
        """Get all transitions into a specific stage.

        Args:
            stage: Target stage name.

        Returns:
            List of matching records.
        """
        return [r for r in self.records if r.new_stage == stage]

    def get_transitions_from(self, stage: str) -> List[TransitionRecord]:
        """Get all transitions from a specific stage.

        Args:
            stage: Source stage name.

        Returns:
            List of matching records.
        """
        return [r for r in self.records if r.old_stage == stage]

    def get_transitions_between(
        self,
        from_stage: str,
        to_stage: str,
    ) -> List[TransitionRecord]:
        """Get all transitions between two specific stages.

        Args:
            from_stage: Source stage name.
            to_stage: Target stage name.

        Returns:
            List of matching records.
        """
        return [r for r in self.records if r.old_stage == from_stage and r.new_stage == to_stage]

    def get_stage_visit_count(self, stage: str) -> int:
        """Count how many times a stage was entered.

        Args:
            stage: Stage name to count.

        Returns:
            Number of times stage was entered.
        """
        return sum(1 for r in self.records if r.new_stage == stage)

    def get_average_duration(self, stage: str) -> Optional[float]:
        """Get average time spent in a stage.

        Args:
            stage: Stage name.

        Returns:
            Average duration in milliseconds, or None if no data.
        """
        durations = [r.duration_ms for r in self.records if r.old_stage == stage and r.duration_ms]
        if durations:
            return sum(durations) / len(durations)
        return None

    def get_transition_pattern(self) -> List[Tuple[str, str]]:
        """Get the sequence of transitions as (from, to) tuples.

        Returns:
            List of (old_stage, new_stage) tuples.
        """
        return [(r.old_stage, r.new_stage) for r in self.records]

    def has_visited(self, stage: str) -> bool:
        """Check if a stage has been visited.

        Args:
            stage: Stage name.

        Returns:
            True if stage appears in history.
        """
        return any(r.new_stage == stage for r in self.records)

    def has_cycle(self) -> bool:
        """Check if the history contains any cycles (revisiting stages).

        Returns:
            True if any stage was visited more than once.
        """
        visited = set()
        for r in self.records:
            if r.new_stage in visited:
                return True
            visited.add(r.new_stage)
        return False

    def get_stage_sequence(self) -> List[str]:
        """Get the sequence of stages visited.

        Returns:
            List of stage names in order (including current).
        """
        if not self.records:
            return []
        stages = [self.records[0].old_stage]
        for r in self.records:
            stages.append(r.new_stage)
        return stages

    def clear(self) -> None:
        """Clear all history."""
        self.records.clear()
        self._sequence = 0
        self._stage_enter_times.clear()

    def __len__(self) -> int:
        """Get number of records."""
        return len(self.records)

    def __bool__(self) -> bool:
        """Check if history is non-empty."""
        return bool(self.records)


@dataclass
class StateTransitionHook:
    """Hook for state machine transitions.

    Represents a single hook that can be triggered on state transitions.
    Supports filtering by stage names and priority ordering.

    Attributes:
        callback: Function to call on transition
        on_enter: Set of stage names to trigger on enter
        on_exit: Set of stage names to trigger on exit
        on_transition: Whether to trigger on any transition
        priority: Hook priority (higher executes first)
        name: Optional hook name for debugging
    """

    callback: Union[TransitionCallback, StageCallback, HistoryAwareCallback]
    on_enter: Set[str] = field(default_factory=set)
    on_exit: Set[str] = field(default_factory=set)
    on_transition: bool = False
    on_transition_with_history: bool = False
    priority: int = 0
    name: Optional[str] = None

    def should_fire_on_enter(self, stage: str) -> bool:
        """Check if hook should fire on entering stage.

        Args:
            stage: Stage name being entered.

        Returns:
            True if hook should fire.
        """
        if not self.on_enter:
            return False
        return stage in self.on_enter or "*" in self.on_enter

    def should_fire_on_exit(self, stage: str) -> bool:
        """Check if hook should fire on exiting stage.

        Args:
            stage: Stage name being exited.

        Returns:
            True if hook should fire.
        """
        if not self.on_exit:
            return False
        return stage in self.on_exit or "*" in self.on_exit


class StateHookManager:
    """Manager for state machine hooks implementing Observer pattern.

    Provides a clean API for registering and triggering hooks on
    state machine transitions. Hooks are decoupled from the state
    machine implementation.

    Features:
        - Priority-based hook ordering
        - Stage-specific filtering
        - Decorator-based registration
        - Error isolation (one hook failure doesn't affect others)
        - Transition history tracking with analytics

    Example:
        manager = StateHookManager()

        # Decorator registration
        @manager.on_transition
        def log_all(old, new, ctx):
            print(f"{old} -> {new}")

        @manager.on_enter("EXECUTION")
        def on_exec(stage, ctx):
            print("Starting execution")

        # History-aware hooks
        @manager.on_transition_with_history
        def analyze(old, new, ctx, history):
            if history.has_cycle():
                print(f"Cycle detected! Pattern: {history.get_stage_sequence()}")

        # Manual registration
        manager.add_hook(StateTransitionHook(
            callback=my_callback,
            on_enter={"PLANNING", "EXECUTION"},
            priority=10,
        ))

        # Trigger hooks
        manager.fire_transition("INITIAL", "PLANNING", {"tool": "read"})

        # Query history
        print(manager.history.get_last(5))
    """

    def __init__(
        self,
        history_max_size: int = 1000,
        enable_history: bool = True,
    ) -> None:
        """Initialize the hook manager.

        Args:
            history_max_size: Maximum transition records to keep.
            enable_history: Whether to track transition history.
        """
        self._hooks: List[StateTransitionHook] = []
        self._enabled = True
        self._enable_history = enable_history
        self._history = TransitionHistory(max_size=history_max_size) if enable_history else None

    def add_hook(self, hook: StateTransitionHook) -> Callable[[], None]:
        """Add a state transition hook.

        Args:
            hook: Hook to add.

        Returns:
            Function to remove the hook.
        """
        self._hooks.append(hook)
        # Sort by priority (highest first)
        self._hooks.sort(key=lambda h: h.priority, reverse=True)

        def remove() -> None:
            if hook in self._hooks:
                self._hooks.remove(hook)

        return remove

    def remove_hook(self, hook: StateTransitionHook) -> None:
        """Remove a hook.

        Args:
            hook: Hook to remove.
        """
        if hook in self._hooks:
            self._hooks.remove(hook)

    def clear_hooks(self) -> None:
        """Remove all hooks."""
        self._hooks.clear()

    def enable(self) -> None:
        """Enable hook firing."""
        self._enabled = True

    def disable(self) -> None:
        """Disable hook firing (hooks are still registered)."""
        self._enabled = False

    @property
    def hook_count(self) -> int:
        """Get number of registered hooks."""
        return len(self._hooks)

    @property
    def history(self) -> Optional[TransitionHistory]:
        """Get the transition history (None if disabled).

        Returns:
            TransitionHistory instance or None.
        """
        return self._history

    @property
    def history_enabled(self) -> bool:
        """Check if history tracking is enabled."""
        return self._enable_history

    def clear_history(self) -> None:
        """Clear the transition history."""
        if self._history:
            self._history.clear()

    # =========================================================================
    # Decorator API
    # =========================================================================

    def on_transition(
        self,
        callback: Optional[TransitionCallback] = None,
        *,
        priority: int = 0,
        name: Optional[str] = None,
    ) -> Union[TransitionCallback, Callable[[TransitionCallback], TransitionCallback]]:
        """Decorator to register a transition hook.

        Can be used with or without arguments:

            @manager.on_transition
            def hook(old, new, ctx): ...

            @manager.on_transition(priority=10)
            def hook(old, new, ctx): ...

        Args:
            callback: Optional callback function.
            priority: Hook priority.
            name: Optional hook name.

        Returns:
            Decorator or decorated function.
        """

        def decorator(fn: TransitionCallback) -> TransitionCallback:
            hook = StateTransitionHook(
                callback=fn,
                on_transition=True,
                priority=priority,
                name=name or fn.__name__,
            )
            self.add_hook(hook)
            return fn

        if callback is not None:
            return decorator(callback)
        return decorator

    def on_transition_with_history(
        self,
        callback: Optional[HistoryAwareCallback] = None,
        *,
        priority: int = 0,
        name: Optional[str] = None,
    ) -> Union[HistoryAwareCallback, Callable[[HistoryAwareCallback], HistoryAwareCallback]]:
        """Decorator to register a history-aware transition hook.

        Similar to on_transition, but the callback receives the TransitionHistory
        as a fourth argument, enabling analytics over the transition sequence.

        Can be used with or without arguments:

            @manager.on_transition_with_history
            def analyze(old, new, ctx, history):
                if history.has_cycle():
                    print(f"Cycle detected at transition #{len(history)}")

            @manager.on_transition_with_history(priority=10)
            def debug(old, new, ctx, history):
                print(f"Pattern: {history.get_stage_sequence()}")

        Args:
            callback: Optional callback function.
            priority: Hook priority.
            name: Optional hook name.

        Returns:
            Decorator or decorated function.

        Note:
            If history tracking is disabled, the callback receives an empty
            TransitionHistory instance.
        """

        def decorator(fn: HistoryAwareCallback) -> HistoryAwareCallback:
            hook = StateTransitionHook(
                callback=fn,
                on_transition_with_history=True,
                priority=priority,
                name=name or fn.__name__,
            )
            self.add_hook(hook)
            return fn

        if callback is not None:
            return decorator(callback)
        return decorator

    def on_enter(
        self,
        *stages: str,
        priority: int = 0,
        name: Optional[str] = None,
    ) -> Callable[[StageCallback], StageCallback]:
        """Decorator to register an on_enter hook.

        Args:
            stages: Stage names to trigger on (or "*" for all).
            priority: Hook priority.
            name: Optional hook name.

        Returns:
            Decorator function.

        Example:
            @manager.on_enter("EXECUTION", "VERIFICATION")
            def on_start(stage, ctx):
                print(f"Entering {stage}")
        """

        def decorator(fn: StageCallback) -> StageCallback:
            hook = StateTransitionHook(
                callback=fn,
                on_enter=set(stages) if stages else {"*"},
                priority=priority,
                name=name or fn.__name__,
            )
            self.add_hook(hook)
            return fn

        return decorator

    def on_exit(
        self,
        *stages: str,
        priority: int = 0,
        name: Optional[str] = None,
    ) -> Callable[[StageCallback], StageCallback]:
        """Decorator to register an on_exit hook.

        Args:
            stages: Stage names to trigger on (or "*" for all).
            priority: Hook priority.
            name: Optional hook name.

        Returns:
            Decorator function.
        """

        def decorator(fn: StageCallback) -> StageCallback:
            hook = StateTransitionHook(
                callback=fn,
                on_exit=set(stages) if stages else {"*"},
                priority=priority,
                name=name or fn.__name__,
            )
            self.add_hook(hook)
            return fn

        return decorator

    # =========================================================================
    # Firing Methods
    # =========================================================================

    def fire_enter(self, stage: str, context: Dict[str, Any]) -> None:
        """Fire hooks for entering a stage.

        Args:
            stage: Stage being entered.
            context: Transition context.
        """
        if not self._enabled:
            return

        for hook in self._hooks:
            if hook.should_fire_on_enter(stage):
                try:
                    hook.callback(stage, context)  # type: ignore
                except Exception as e:
                    logger.warning(f"Hook '{hook.name or 'unnamed'}' error on enter {stage}: {e}")

    def fire_exit(self, stage: str, context: Dict[str, Any]) -> None:
        """Fire hooks for exiting a stage.

        Args:
            stage: Stage being exited.
            context: Transition context.
        """
        if not self._enabled:
            return

        for hook in self._hooks:
            if hook.should_fire_on_exit(stage):
                try:
                    hook.callback(stage, context)  # type: ignore
                except Exception as e:
                    logger.warning(f"Hook '{hook.name or 'unnamed'}' error on exit {stage}: {e}")

    def fire_transition(
        self,
        old_stage: str,
        new_stage: str,
        context: Dict[str, Any],
    ) -> Optional[TransitionRecord]:
        """Fire all hooks for a complete transition.

        Fires in order:
        1. Record transition in history (if enabled)
        2. on_exit hooks for old_stage
        3. on_transition hooks
        4. on_transition_with_history hooks
        5. on_enter hooks for new_stage

        Args:
            old_stage: Stage being exited.
            new_stage: Stage being entered.
            context: Transition context.

        Returns:
            The TransitionRecord if history is enabled, None otherwise.
        """
        if not self._enabled:
            return None

        # Record in history first (so hooks can access it)
        record: Optional[TransitionRecord] = None
        if self._history is not None:
            record = self._history.create_record(old_stage, new_stage, context)

        # Fire exit hooks
        self.fire_exit(old_stage, context)

        # Fire transition hooks
        for hook in self._hooks:
            if hook.on_transition:
                try:
                    hook.callback(old_stage, new_stage, context)  # type: ignore
                except Exception as e:
                    logger.warning(f"Hook '{hook.name or 'unnamed'}' error on transition: {e}")

        # Fire history-aware transition hooks
        # Provide empty history if disabled, so hooks don't need to handle None
        history = self._history if self._history is not None else TransitionHistory(max_size=0)
        for hook in self._hooks:
            if hook.on_transition_with_history:
                try:
                    hook.callback(old_stage, new_stage, context, history)  # type: ignore
                except Exception as e:
                    logger.warning(
                        f"Hook '{hook.name or 'unnamed'}' error on transition (with history): {e}"
                    )

        # Fire enter hooks
        self.fire_enter(new_stage, context)

        return record

    def get_last_transition(self) -> Optional[TransitionRecord]:
        """Get the most recent transition record.

        Returns:
            The last TransitionRecord, or None if history is empty/disabled.
        """
        if self._history and self._history.records:
            return self._history.records[-1]
        return None


class LoggingHook:
    """Pre-built logging hook for state transitions.

    Example:
        manager = StateHookManager()
        logging_hook = LoggingHook(logger=my_logger)
        manager.add_hook(logging_hook.create_hook())
    """

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        level: int = logging.INFO,
        include_context: bool = False,
    ) -> None:
        """Initialize logging hook.

        Args:
            logger: Logger to use (default: module logger).
            level: Log level.
            include_context: Whether to include context in log.
        """
        self._logger = logger or logging.getLogger(__name__)
        self._level = level
        self._include_context = include_context

    def create_hook(self, priority: int = -100) -> StateTransitionHook:
        """Create the hook instance.

        Args:
            priority: Hook priority (default low to log after other hooks).

        Returns:
            Configured StateTransitionHook.
        """
        return StateTransitionHook(
            callback=self._log_transition,
            on_transition=True,
            priority=priority,
            name="LoggingHook",
        )

    def _log_transition(
        self,
        old_stage: str,
        new_stage: str,
        context: Dict[str, Any],
    ) -> None:
        """Log a transition.

        Args:
            old_stage: Previous stage.
            new_stage: New stage.
            context: Transition context.
        """
        msg = f"State transition: {old_stage} -> {new_stage}"
        if self._include_context and context:
            msg += f" (context: {context})"
        self._logger.log(self._level, msg)


class MetricsHook:
    """Pre-built metrics hook for state transitions.

    Tracks transition counts and time spent in each stage.

    Example:
        manager = StateHookManager()
        metrics = MetricsHook()
        manager.add_hook(metrics.create_hook())

        # Later
        print(metrics.get_stats())
    """

    def __init__(self) -> None:
        """Initialize metrics hook."""
        self._transition_count: Dict[str, int] = {}
        self._stage_entries: Dict[str, int] = {}
        self._total_transitions = 0

    def create_hook(self, priority: int = 100) -> StateTransitionHook:
        """Create the hook instance.

        Args:
            priority: Hook priority (default high to track before others).

        Returns:
            Configured StateTransitionHook.
        """
        return StateTransitionHook(
            callback=self._record_transition,
            on_transition=True,
            priority=priority,
            name="MetricsHook",
        )

    def _record_transition(
        self,
        old_stage: str,
        new_stage: str,
        context: Dict[str, Any],
    ) -> None:
        """Record a transition.

        Args:
            old_stage: Previous stage.
            new_stage: New stage.
            context: Transition context.
        """
        key = f"{old_stage}->{new_stage}"
        self._transition_count[key] = self._transition_count.get(key, 0) + 1
        self._stage_entries[new_stage] = self._stage_entries.get(new_stage, 0) + 1
        self._total_transitions += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get collected statistics.

        Returns:
            Dictionary with metrics.
        """
        return {
            "total_transitions": self._total_transitions,
            "transitions": dict(self._transition_count),
            "stage_entries": dict(self._stage_entries),
        }

    def reset(self) -> None:
        """Reset all metrics."""
        self._transition_count.clear()
        self._stage_entries.clear()
        self._total_transitions = 0
