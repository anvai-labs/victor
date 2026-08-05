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

"""Option framework for hierarchical RL.

This module provides the Option abstraction for temporal abstraction in
hierarchical reinforcement learning. Options are reusable skills that
encapsulate multi-step behaviors.

An Option consists of:
- Initiation set: States where the option can be started
- Policy: How to behave while executing the option
- Termination condition: When the option completes

Example Options:
- "explore_codebase": Read files, search code, understand structure
- "implement_feature": Write code, run tests, fix errors
- "debug_issue": Analyze error, identify cause, apply fix

Sprint 5-6: Advanced Patterns - Hierarchical RL
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from victor.framework.tool_naming import ToolNames, get_canonical_name

logger = logging.getLogger(__name__)


class OptionStatus(str, Enum):
    """Status of an option execution."""

    INACTIVE = "inactive"  # Option not started
    RUNNING = "running"  # Option currently executing
    COMPLETED = "completed"  # Option finished successfully
    TERMINATED = "terminated"  # Option terminated early
    FAILED = "failed"  # Option failed


@dataclass
class OptionState:
    """State representation for option execution.

    Attributes:
        current_mode: Current agent mode (explore, plan, build, etc.)
        tools_used: Tools used in current option
        iterations: Iterations in current option
        context_size: Current context size
        task_progress: Estimated task progress (0-1)
        last_tool_success: Whether last tool succeeded
        custom_features: Additional state features
    """

    current_mode: str = "build"
    tools_used: List[str] = field(default_factory=list)
    iterations: int = 0
    context_size: int = 0
    task_progress: float = 0.0
    last_tool_success: bool = True
    custom_features: Dict[str, Any] = field(default_factory=dict)

    def to_tuple(self) -> tuple:
        """Convert to hashable tuple for Q-table lookup."""
        return (
            self.current_mode,
            len(self.tools_used),
            min(self.iterations // 5, 10),  # Bucketed
            min(self.context_size // 10000, 10),  # Bucketed
            int(self.task_progress * 10),  # Discretized
            self.last_tool_success,
        )


@dataclass
class OptionResult:
    """Result of option execution.

    Attributes:
        status: Final status
        reward: Cumulative reward
        steps: Number of steps taken
        final_state: State at termination
        metadata: Additional result data
    """

    status: OptionStatus
    reward: float = 0.0
    steps: int = 0
    final_state: Optional[OptionState] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def _canonicalize_recent_tools(tools_used: List[str], limit: int) -> List[str]:
    """Normalize recent tool history for policy decisions."""
    return [get_canonical_name(tool) for tool in tools_used[-limit:]]


class Option(ABC):
    """Abstract base class for options (temporal abstractions).

    Options are reusable skills that encapsulate multi-step behaviors.
    Each option defines:
    - When it can start (initiation set)
    - How to behave (internal policy)
    - When it terminates (termination condition)
    """

    def __init__(self, name: str, description: str = ""):
        """Initialize option.

        Args:
            name: Option name (e.g., "explore_codebase")
            description: Human-readable description
        """
        self.name = name
        self.description = description
        self._status = OptionStatus.INACTIVE
        self._start_time: Optional[datetime] = None
        self._steps = 0
        self._cumulative_reward = 0.0

    @property
    def status(self) -> OptionStatus:
        """Get current status."""
        return self._status

    @property
    def is_active(self) -> bool:
        """Check if option is currently running."""
        return self._status == OptionStatus.RUNNING

    @abstractmethod
    def can_initiate(self, state: OptionState) -> bool:
        """Check if option can be initiated in given state.

        Args:
            state: Current state

        Returns:
            True if option can start
        """
        pass

    @abstractmethod
    def should_terminate(self, state: OptionState) -> bool:
        """Check if option should terminate.

        Args:
            state: Current state

        Returns:
            True if option should end
        """
        pass

    @abstractmethod
    def get_action(self, state: OptionState) -> str:
        """Get action from option's internal policy.

        Args:
            state: Current state

        Returns:
            Action to take (tool name or mode)
        """
        pass

    def start(self, state: OptionState) -> bool:
        """Start the option.

        Args:
            state: Starting state

        Returns:
            True if started successfully
        """
        if not self.can_initiate(state):
            return False

        self._status = OptionStatus.RUNNING
        self._start_time = datetime.now()
        self._steps = 0
        self._cumulative_reward = 0.0

        logger.debug(f"Option '{self.name}' started")
        return True

    def step(self, state: OptionState, reward: float = 0.0) -> Optional[str]:
        """Execute one step of the option.

        Args:
            state: Current state
            reward: Reward from previous action

        Returns:
            Next action or None if terminated
        """
        if self._status != OptionStatus.RUNNING:
            return None

        self._steps += 1
        self._cumulative_reward += reward

        if self.should_terminate(state):
            self._status = OptionStatus.COMPLETED
            logger.debug(f"Option '{self.name}' completed after {self._steps} steps")
            return None

        return self.get_action(state)

    def terminate(self, success: bool = True) -> OptionResult:
        """Terminate the option.

        Args:
            success: Whether termination is due to success

        Returns:
            OptionResult with execution summary
        """
        self._status = OptionStatus.COMPLETED if success else OptionStatus.TERMINATED

        return OptionResult(
            status=self._status,
            reward=self._cumulative_reward,
            steps=self._steps,
            metadata={
                "name": self.name,
                "duration": (
                    (datetime.now() - self._start_time).total_seconds() if self._start_time else 0
                ),
            },
        )


class ExploreOption(Option):
    """Option for exploring and understanding the codebase.

    Initiation: Any state where understanding is needed
    Policy: Prioritize read, search, and list operations
    Termination: Sufficient context gathered or max steps reached
    """

    MAX_EXPLORE_STEPS = 15
    MIN_CONTEXT_FOR_COMPLETION = 5000

    def __init__(self):
        super().__init__(
            name="explore_codebase",
            description="Explore and understand the codebase structure",
        )
        self._explore_tools = {
            ToolNames.READ,
            "code_search",
            "semantic_code_search",
            ToolNames.LS,
            "find_definition",
        }

    def can_initiate(self, state: OptionState) -> bool:
        """Can initiate when context is low or in explore mode."""
        return (
            state.context_size < self.MIN_CONTEXT_FOR_COMPLETION
            or state.current_mode == "explore"
            or state.task_progress < 0.2
        )

    def should_terminate(self, state: OptionState) -> bool:
        """Terminate when enough context or max steps reached."""
        return (
            self._steps >= self.MAX_EXPLORE_STEPS
            or state.context_size >= self.MIN_CONTEXT_FOR_COMPLETION * 2
            or state.task_progress >= 0.5
        )

    def get_action(self, state: OptionState) -> str:
        """Get exploration action based on state."""
        # Prioritize based on what we've done
        used_set = set(_canonicalize_recent_tools(state.tools_used, limit=5))

        # If we haven't searched yet, search first
        if "code_search" not in used_set and "semantic_code_search" not in used_set:
            return "semantic_code_search"

        # If we haven't read files, read
        if ToolNames.READ not in used_set:
            return ToolNames.READ

        # Default to listing directory
        if ToolNames.LS not in used_set:
            return ToolNames.LS

        # Cycle through explore tools
        return "code_search"


class ImplementOption(Option):
    """Option for implementing features or fixes.

    Initiation: After exploration, when ready to write code
    Policy: Write code, then verify with tests
    Termination: Tests pass or max iterations reached
    """

    MAX_IMPLEMENT_STEPS = 20
    MIN_PROGRESS_FOR_COMPLETION = 0.8

    def __init__(self):
        super().__init__(
            name="implement_feature",
            description="Implement code changes for the task",
        )
        self._implement_tools = {
            ToolNames.WRITE,
            ToolNames.EDIT,
            "execute_code",
            "run_tests",
            ToolNames.SHELL,
        }

    def can_initiate(self, state: OptionState) -> bool:
        """Can initiate when we have enough context."""
        return (
            state.context_size >= 3000
            or state.task_progress >= 0.3
            or state.current_mode == "build"
        )

    def should_terminate(self, state: OptionState) -> bool:
        """Terminate when task complete or stuck."""
        return (
            self._steps >= self.MAX_IMPLEMENT_STEPS
            or state.task_progress >= self.MIN_PROGRESS_FOR_COMPLETION
            or (self._steps >= 5 and not state.last_tool_success)
        )

    def get_action(self, state: OptionState) -> str:
        """Get implementation action based on state."""
        recent_tools = _canonicalize_recent_tools(state.tools_used, limit=3)

        # If we just wrote code, run tests
        if any(t in {ToolNames.WRITE, ToolNames.EDIT} for t in recent_tools):
            return "run_tests"

        # If tests failed, edit to fix
        if not state.last_tool_success and "run_tests" in recent_tools:
            return ToolNames.EDIT

        # Default to editing
        return ToolNames.EDIT


class DebugOption(Option):
    """Option for debugging issues.

    Initiation: After error or test failure
    Policy: Analyze error, identify cause, apply fix
    Termination: Issue resolved or max attempts
    """

    MAX_DEBUG_STEPS = 10

    def __init__(self):
        super().__init__(
            name="debug_issue",
            description="Debug and fix issues",
        )

    def can_initiate(self, state: OptionState) -> bool:
        """Can initiate after failure."""
        return not state.last_tool_success or state.current_mode == "debug"

    def should_terminate(self, state: OptionState) -> bool:
        """Terminate when fixed or max attempts."""
        return self._steps >= self.MAX_DEBUG_STEPS or (state.last_tool_success and self._steps >= 2)

    def get_action(self, state: OptionState) -> str:
        """Get debug action based on state."""
        recent_tools = _canonicalize_recent_tools(state.tools_used, limit=3)

        # First, analyze the error
        if self._steps == 0:
            return ToolNames.READ

        # Then try to fix
        if ToolNames.READ in recent_tools:
            return ToolNames.EDIT

        # Verify fix
        return "run_tests"


class ReviewOption(Option):
    """Option for reviewing and validating work.

    Initiation: After implementation, before completion
    Policy: Check code quality, run tests, verify behavior
    Termination: Review complete
    """

    MAX_REVIEW_STEPS = 8

    def __init__(self):
        super().__init__(
            name="review_work",
            description="Review and validate completed work",
        )

    def can_initiate(self, state: OptionState) -> bool:
        """Can initiate when task nearly complete."""
        return state.task_progress >= 0.7 or state.current_mode == "review"

    def should_terminate(self, state: OptionState) -> bool:
        """Terminate after review steps."""
        return self._steps >= self.MAX_REVIEW_STEPS or state.task_progress >= 0.95

    def get_action(self, state: OptionState) -> str:
        """Get review action."""
        if self._steps == 0:
            return "run_tests"
        elif self._steps == 1:
            return ToolNames.READ
        else:
            return "code_search"


class OptionRegistry:
    """Registry for available options.

    Manages option lifecycle and selection based on state.
    """

    def __init__(self):
        """Initialize registry with default options."""
        self._options: Dict[str, Option] = {}
        self._active_option: Optional[Option] = None

        # Register default options
        self.register(ExploreOption())
        self.register(ImplementOption())
        self.register(DebugOption())
        self.register(ReviewOption())

    def register(self, option: Option) -> None:
        """Register an option.

        Args:
            option: Option to register
        """
        self._options[option.name] = option
        logger.debug(f"Registered option: {option.name}")

    def get_available_options(self, state: OptionState) -> List[Option]:
        """Get options that can be initiated in current state.

        Args:
            state: Current state

        Returns:
            List of available options
        """
        return [opt for opt in self._options.values() if opt.can_initiate(state)]

    def get_option(self, name: str) -> Optional[Option]:
        """Get option by name.

        Args:
            name: Option name

        Returns:
            Option or None
        """
        return self._options.get(name)

    @property
    def active_option(self) -> Optional[Option]:
        """Get currently active option."""
        return self._active_option

    def start_option(self, name: str, state: OptionState) -> bool:
        """Start an option.

        Args:
            name: Option name
            state: Starting state

        Returns:
            True if started
        """
        option = self._options.get(name)
        if not option:
            return False

        if option.start(state):
            self._active_option = option
            return True
        return False

    def step_active_option(self, state: OptionState, reward: float = 0.0) -> Optional[str]:
        """Step the active option.

        Args:
            state: Current state
            reward: Reward from previous action

        Returns:
            Next action or None if terminated
        """
        if not self._active_option:
            return None

        action = self._active_option.step(state, reward)

        if action is None:
            # Option terminated
            self._active_option = None

        return action

    def terminate_active_option(self, success: bool = True) -> Optional[OptionResult]:
        """Terminate the active option.

        Args:
            success: Whether successful

        Returns:
            OptionResult or None
        """
        if not self._active_option:
            return None

        result = self._active_option.terminate(success)
        self._active_option = None
        return result

    def export_metrics(self) -> Dict[str, Any]:
        """Export registry metrics.

        Returns:
            Dictionary with metrics
        """
        return {
            "total_options": len(self._options),
            "option_names": list(self._options.keys()),
            "active_option": self._active_option.name if self._active_option else None,
        }
