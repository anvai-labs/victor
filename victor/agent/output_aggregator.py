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

"""
Output Aggregation Service (Observer Pattern).

This module implements an Observer pattern to aggregate tool outputs and detect
task completion states. It addresses the need for synthesizing multiple tool
results into coherent responses, especially for providers like DeepSeek that
make many tool calls without synthesizing findings.

SOLID Principles Applied:
- Single Responsibility: Each class has one purpose (state tracking, observation, aggregation)
- Open/Closed: New observers can be added without modifying existing code
- Liskov Substitution: All observers are interchangeable
- Interface Segregation: Observer interface is minimal
- Dependency Inversion: Aggregator depends on abstractions, not concrete observers

Addresses GAP-6: Missing output consolidation
Addresses GAP-8: Missing task completion signal
"""

from victor.core.json_utils import json_dumps
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Protocol, List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import time
import logging
import hashlib

if TYPE_CHECKING:
    from victor.agent.presentation import PresentationProtocol

logger = logging.getLogger(__name__)


class AggregationState(Enum):
    """States of the output aggregation process."""

    COLLECTING = "collecting"  # Still gathering tool results
    READY_TO_SYNTHESIZE = "ready_to_synthesize"  # Ready for synthesis
    COMPLETE = "complete"  # Task completed
    STUCK = "stuck"  # No progress being made
    LOOPING = "looping"  # Redundant operations detected


@dataclass
class ToolOutput:
    """Represents a single tool execution output."""

    tool_name: str
    result: Any
    success: bool = True
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    args_hash: str = ""  # Hash of arguments for deduplication

    def __post_init__(self) -> None:
        if not self.args_hash and self.metadata.get("args"):
            self.args_hash = self._compute_hash(self.metadata["args"])

    @staticmethod
    def _compute_hash(args: Dict[str, Any]) -> str:
        """Compute a hash of tool arguments for deduplication."""
        import json

        try:
            args_str = json_dumps(args, sort_keys=True, default=str)
            return hashlib.md5(args_str.encode()).hexdigest()[:12]
        except Exception:
            return ""


@dataclass
class AggregatedResult:
    """Result of output aggregation."""

    state: AggregationState
    results: List[ToolOutput] = field(default_factory=list)
    synthesis_prompt: str = ""
    confidence: float = 0.0
    summary: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def tool_count(self) -> int:
        """Number of tool outputs collected."""
        return len(self.results)

    @property
    def unique_tools(self) -> Set[str]:
        """Set of unique tool names used."""
        return {r.tool_name for r in self.results}

    @property
    def success_rate(self) -> float:
        """Percentage of successful tool executions."""
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.success) / len(self.results)


class OutputAggregatorObserver(Protocol):
    """Observer interface for aggregation state changes."""

    def on_state_change(self, new_state: AggregationState) -> None:
        """Called when aggregation state changes."""
        ...

    def on_result_added(self, result: ToolOutput) -> None:
        """Called when a new result is added."""
        ...

    def on_synthesis_ready(self, aggregated: AggregatedResult) -> None:
        """Called when synthesis is ready."""
        ...


class LoggingObserver:
    """Observer that logs aggregation events."""

    def __init__(self) -> None:
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def on_state_change(self, new_state: AggregationState) -> None:
        self._logger.info(f"Aggregation state changed to: {new_state.value}")

    def on_result_added(self, result: ToolOutput) -> None:
        self._logger.debug(f"Tool result added: {result.tool_name} (success={result.success})")

    def on_synthesis_ready(self, aggregated: AggregatedResult) -> None:
        self._logger.info(
            f"Synthesis ready: {aggregated.tool_count} results, " f"state={aggregated.state.value}"
        )


class MetricsObserver:
    """Observer that collects aggregation metrics."""

    def __init__(self) -> None:
        self.state_changes: List[tuple[AggregationState, float]] = []
        self.results_by_tool: Dict[str, int] = {}
        self.synthesis_count: int = 0

    def on_state_change(self, new_state: AggregationState) -> None:
        self.state_changes.append((new_state, time.time()))

    def on_result_added(self, result: ToolOutput) -> None:
        self.results_by_tool[result.tool_name] = self.results_by_tool.get(result.tool_name, 0) + 1

    def on_synthesis_ready(self, aggregated: AggregatedResult) -> None:
        self.synthesis_count += 1

    def get_metrics(self) -> Dict[str, Any]:
        """Get collected metrics."""
        return {
            "state_transitions": len(self.state_changes),
            "results_by_tool": dict(self.results_by_tool),
            "total_results": sum(self.results_by_tool.values()),
            "synthesis_count": self.synthesis_count,
        }


class OutputAggregator:
    """
    Aggregates tool outputs and detects completion state.

    Implements the Observer pattern to notify interested parties of
    state changes and synthesis readiness.
    """

    def __init__(
        self,
        max_results: int = 10,
        stale_threshold_seconds: float = 30.0,
        loop_detection_window: int = 5,
        presentation: Optional["PresentationProtocol"] = None,
    ) -> None:
        """
        Initialize the output aggregator.

        Args:
            max_results: Max results before forcing synthesis
            stale_threshold_seconds: Seconds of inactivity before marking stuck
            loop_detection_window: Number of recent results to check for loops
            presentation: Optional presentation adapter for icons (creates default if None)
        """
        self._results: List[ToolOutput] = []
        self._observers: List[OutputAggregatorObserver] = []
        self._state = AggregationState.COLLECTING
        self._last_result_time: float = time.time()
        self._max_results = max_results
        self._stale_threshold = stale_threshold_seconds
        self._loop_detection_window = loop_detection_window
        self._seen_hashes: Set[str] = set()
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Lazy init for backward compatibility
        if presentation is None:
            from victor.agent.presentation import create_presentation_adapter

            self._presentation = create_presentation_adapter()
        else:
            self._presentation = presentation

    @property
    def state(self) -> AggregationState:
        """Current aggregation state."""
        return self._state

    @property
    def results(self) -> List[ToolOutput]:
        """List of collected results."""
        return list(self._results)

    def add_observer(self, observer: OutputAggregatorObserver) -> None:
        """Add an observer for state changes."""
        self._observers.append(observer)

    def remove_observer(self, observer: OutputAggregatorObserver) -> None:
        """Remove an observer."""
        if observer in self._observers:
            self._observers.remove(observer)

    def add_result(
        self,
        tool_name: str,
        result: Any,
        success: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add a tool result to the aggregator.

        Args:
            tool_name: Name of the tool
            result: The result output
            success: Whether the tool succeeded
            metadata: Additional metadata (including args)
        """
        output = ToolOutput(
            tool_name=tool_name,
            result=result,
            success=success,
            metadata=metadata or {},
        )

        # Check for duplicate operations
        if output.args_hash:
            if output.args_hash in self._seen_hashes:
                self._logger.debug(f"Duplicate tool call detected: {tool_name} with same args")
                output.metadata["is_duplicate"] = True
            else:
                self._seen_hashes.add(output.args_hash)

        self._results.append(output)
        self._last_result_time = time.time()

        # Notify observers
        for obs in self._observers:
            try:
                obs.on_result_added(output)
            except Exception as e:
                self._logger.warning(f"Observer error on result added: {e}")

        self._update_state()

    def _update_state(self) -> None:
        """Update aggregation state based on current results."""
        old_state = self._state
        current_time = time.time()

        # Check for stuck state
        if current_time - self._last_result_time > self._stale_threshold:
            self._state = AggregationState.STUCK
        # Check for looping (redundant operations)
        elif self._detect_loop():
            self._state = AggregationState.LOOPING
        # Check for synthesis threshold
        elif len(self._results) >= self._max_results:
            self._state = AggregationState.READY_TO_SYNTHESIZE
        # Check for completion patterns
        elif self._detect_completion_pattern():
            self._state = AggregationState.COMPLETE

        if old_state != self._state:
            self._logger.info(f"State transition: {old_state.value} -> {self._state.value}")
            for obs in self._observers:
                try:
                    obs.on_state_change(self._state)
                except Exception as e:
                    self._logger.warning(f"Observer error on state change: {e}")

            # Notify synthesis ready if applicable
            if self._state in (
                AggregationState.READY_TO_SYNTHESIZE,
                AggregationState.LOOPING,
            ):
                aggregated = self.get_aggregated_result()
                for obs in self._observers:
                    try:
                        obs.on_synthesis_ready(aggregated)
                    except Exception as e:
                        self._logger.warning(f"Observer error on synthesis ready: {e}")

    def _detect_loop(self) -> bool:
        """Detect if recent tool calls form a loop."""
        if len(self._results) < self._loop_detection_window:
            return False

        recent = self._results[-self._loop_detection_window :]

        # Check for same tool called repeatedly
        tool_names = [r.tool_name for r in recent]
        if len(set(tool_names)) == 1:
            return True

        # Check for duplicate args hashes
        hashes = [r.args_hash for r in recent if r.args_hash]
        if hashes and len(hashes) != len(set(hashes)):
            # More than half are duplicates
            if len(hashes) - len(set(hashes)) > len(hashes) // 2:
                return True

        return False

    def _detect_completion_pattern(self) -> bool:
        """Detect if results form a complete answer."""
        if not self._results:
            return False

        # Pattern: Last result is successful and contains substantial content
        last = self._results[-1]
        if not last.success:
            return False

        # Check if result looks like a final answer
        if isinstance(last.result, str):
            # Large content likely means synthesis happened
            return len(last.result) > 500

        # Check for specific completion signals in metadata
        if last.metadata.get("is_final_answer"):
            return True

        return False

    def get_synthesis_prompt(self) -> str:
        """Generate prompt to synthesize collected results."""
        if not self._results:
            return ""

        result_summaries = []
        for i, r in enumerate(self._results, 1):
            status = self._presentation.icon("success" if r.success else "error", with_color=False)
            result_str = str(r.result)
            if len(result_str) > 200:
                result_str = result_str[:200] + "..."
            result_summaries.append(f"{i}. [{status}] {r.tool_name}: {result_str}")

        unique_tools = ", ".join(sorted({r.tool_name for r in self._results}))

        return f"""Based on the following {len(self._results)} tool results (using {unique_tools}), provide a synthesized answer:

{chr(10).join(result_summaries)}

Instructions:
1. Synthesize these findings into a coherent, actionable response
2. Highlight key insights from multiple sources
3. Note any inconsistencies or gaps in the gathered information
4. Provide clear next steps or recommendations if applicable
"""

    def get_aggregated_result(self) -> AggregatedResult:
        """Get the aggregated result with current state."""
        return AggregatedResult(
            state=self._state,
            results=list(self._results),
            synthesis_prompt=self.get_synthesis_prompt(),
            confidence=self._calculate_confidence(),
            summary=self._generate_summary(),
            metadata={
                "unique_tools": list({r.tool_name for r in self._results}),
                "total_results": len(self._results),
                "success_rate": sum(1 for r in self._results if r.success)
                / max(len(self._results), 1),
                "has_duplicates": any(r.metadata.get("is_duplicate") for r in self._results),
            },
        )

    def _calculate_confidence(self) -> float:
        """Calculate confidence score for aggregated results."""
        if not self._results:
            return 0.0

        factors = []

        # Factor 1: Success rate
        success_rate = sum(1 for r in self._results if r.success) / len(self._results)
        factors.append(success_rate)

        # Factor 2: Tool diversity (more tools = higher confidence for complex tasks)
        unique_ratio = len({r.tool_name for r in self._results}) / len(self._results)
        factors.append(min(unique_ratio * 1.5, 1.0))

        # Factor 3: No duplicates penalty
        duplicate_count = sum(1 for r in self._results if r.metadata.get("is_duplicate"))
        duplicate_penalty = max(0, 1 - (duplicate_count / max(len(self._results), 1)))
        factors.append(duplicate_penalty)

        # Factor 4: State penalty
        state_penalties = {
            AggregationState.COMPLETE: 1.0,
            AggregationState.READY_TO_SYNTHESIZE: 0.9,
            AggregationState.COLLECTING: 0.7,
            AggregationState.LOOPING: 0.5,
            AggregationState.STUCK: 0.3,
        }
        factors.append(state_penalties.get(self._state, 0.5))

        return sum(factors) / len(factors)

    def _generate_summary(self) -> str:
        """Generate a brief summary of aggregated results."""
        if not self._results:
            return "No results collected"

        successful = sum(1 for r in self._results if r.success)
        unique_tools = {r.tool_name for r in self._results}

        return (
            f"Collected {len(self._results)} results from {len(unique_tools)} tools "
            f"({successful} successful). State: {self._state.value}"
        )

    def reset(self) -> None:
        """Reset the aggregator for a new task."""
        self._results.clear()
        self._seen_hashes.clear()
        self._state = AggregationState.COLLECTING
        self._last_result_time = time.time()

    def check_stale(self) -> bool:
        """Check if aggregator is stale and update state if needed."""
        if time.time() - self._last_result_time > self._stale_threshold:
            if self._state != AggregationState.STUCK:
                self._state = AggregationState.STUCK
                for obs in self._observers:
                    try:
                        obs.on_state_change(self._state)
                    except Exception as e:
                        self._logger.warning(f"Observer error on stale check: {e}")
            return True
        return False


# Factory functions for common configurations
def create_default_aggregator() -> OutputAggregator:
    """Create an aggregator with default settings."""
    aggregator = OutputAggregator()
    aggregator.add_observer(LoggingObserver())
    return aggregator


def create_monitored_aggregator() -> tuple[OutputAggregator, MetricsObserver]:
    """Create an aggregator with metrics collection."""
    aggregator = OutputAggregator()
    metrics = MetricsObserver()
    aggregator.add_observer(LoggingObserver())
    aggregator.add_observer(metrics)
    return aggregator, metrics
