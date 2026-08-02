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

"""Debug Adapter Protocol (DAP) data types.

Based on Microsoft's Debug Adapter Protocol specification:
https://microsoft.github.io/debug-adapter-protocol/

These types provide a language-agnostic interface for debugging operations,
enabling Victor to work with any debugger that implements DAP.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class DebugState(Enum):
    """Current state of a debug session."""

    INITIALIZING = "initializing"  # Adapter starting up
    RUNNING = "running"  # Program executing
    STOPPED = "stopped"  # Hit breakpoint or step
    PAUSED = "paused"  # User requested pause
    TERMINATED = "terminated"  # Program finished
    DISCONNECTED = "disconnected"  # Adapter disconnected


class DebugStopReason(Enum):
    """Reason why debugger stopped execution.

    Renamed from StopReason to be semantically distinct:
    - DebugStopReason (here): Debugger stop reasons (breakpoint, step, exception)
    - TrackerStopReason (victor.agent.unified_task_tracker): Task tracker stop reasons enum
    - LoopStopRecommendation (victor.agent.loop_detector): Loop detection recommendation dataclass
    """

    BREAKPOINT = "breakpoint"  # Hit a breakpoint
    STEP = "step"  # Completed a step
    EXCEPTION = "exception"  # Exception/error occurred
    PAUSE = "pause"  # User requested pause
    ENTRY = "entry"  # Program entry point
    GOTO = "goto"  # Jumped to location
    FUNCTION_BREAKPOINT = "function breakpoint"
    DATA_BREAKPOINT = "data breakpoint"
    INSTRUCTION_BREAKPOINT = "instruction breakpoint"


# Backward compatibility alias
StopReason = DebugStopReason


@dataclass
class SourceLocation:
    """A location in source code."""

    path: Path
    line: int  # 1-indexed
    column: Optional[int] = None  # 1-indexed
    end_line: Optional[int] = None
    end_column: Optional[int] = None

    def __str__(self) -> str:
        loc = f"{self.path}:{self.line}"
        if self.column:
            loc += f":{self.column}"
        return loc


@dataclass
class Breakpoint:
    """A breakpoint in the program.

    Breakpoints can be:
    - Line breakpoints: Stop at a specific line
    - Conditional breakpoints: Stop when condition is true
    - Hit count breakpoints: Stop after N hits
    - Log points: Log a message instead of stopping
    """

    id: int
    verified: bool  # Whether debugger confirmed breakpoint can be hit
    source: SourceLocation
    condition: Optional[str] = None  # Expression that must be true
    hit_condition: Optional[str] = None  # e.g., ">= 5" for hit count
    log_message: Optional[str] = None  # Log instead of stop
    enabled: bool = True
    hit_count: int = 0  # Number of times hit

    # Additional metadata
    message: Optional[str] = None  # Error/warning message
    instruction_reference: Optional[str] = None


@dataclass
class Thread:
    """A thread in the debugged program."""

    id: int
    name: str
    state: DebugState = DebugState.RUNNING


@dataclass
class StackFrame:
    """A stack frame in the call stack.

    Represents a function/method call on the stack with its
    source location and local variable scope.
    """

    id: int
    name: str  # Function/method name
    source: Optional[SourceLocation] = None
    module_id: Optional[str] = None
    presentation_hint: str = "normal"  # "normal", "label", "subtle"

    # Computed properties
    can_restart: bool = False  # Can restart execution from here
    instruction_pointer: Optional[str] = None


@dataclass
class Scope:
    """A variable scope (locals, globals, etc.)."""

    name: str  # "Locals", "Globals", "Arguments", etc.
    variables_reference: int  # Reference for fetching variables
    named_variables: Optional[int] = None
    indexed_variables: Optional[int] = None
    expensive: bool = False  # Whether fetching variables is expensive
    source: Optional[SourceLocation] = None


@dataclass
class Variable:
    """A variable or property in a scope.

    Variables can be nested (objects, arrays, dicts) with
    children accessible via variables_reference.
    """

    name: str
    value: str  # String representation of value
    type: Optional[str] = None  # Type name
    variables_reference: int = 0  # >0 means has children
    named_variables: Optional[int] = None
    indexed_variables: Optional[int] = None

    # Memory reference for low-level inspection
    memory_reference: Optional[str] = None

    # Evaluation metadata
    evaluate_name: Optional[str] = None  # Expression to get this value
    presentation_hint: Optional[Dict[str, Any]] = None


@dataclass
class ExceptionInfo:
    """Information about an exception/error."""

    exception_id: str
    description: Optional[str] = None
    break_mode: str = "always"  # "never", "always", "unhandled", "userUnhandled"
    details: Optional[Dict[str, Any]] = None


@dataclass
class EvaluateResult:
    """Result of evaluating an expression."""

    result: str  # String representation
    type: Optional[str] = None
    variables_reference: int = 0  # For complex values
    named_variables: Optional[int] = None
    indexed_variables: Optional[int] = None
    memory_reference: Optional[str] = None


@dataclass
class DebugSession:
    """Represents an active debug session.

    Tracks the current state, breakpoints, threads, and provides
    methods for common debug operations.
    """

    id: str
    name: str
    language: str
    state: DebugState = DebugState.INITIALIZING

    # Program info
    program: Optional[Path] = None
    working_directory: Optional[Path] = None
    arguments: List[str] = field(default_factory=list)
    environment: Dict[str, str] = field(default_factory=dict)

    # Debug state
    breakpoints: Dict[str, List[Breakpoint]] = field(default_factory=dict)  # path -> breakpoints
    threads: List[Thread] = field(default_factory=list)
    current_thread_id: Optional[int] = None
    stop_reason: Optional[StopReason] = None
    exception_info: Optional[ExceptionInfo] = None

    # Capabilities (set by adapter)
    supports_conditional_breakpoints: bool = False
    supports_hit_conditional_breakpoints: bool = False
    supports_log_points: bool = False
    supports_restart_frame: bool = False
    supports_stepping_granularity: bool = False
    supports_exception_options: bool = False
    supports_data_breakpoints: bool = False

    def get_breakpoints_for_file(self, path: Union[str, Path]) -> List[Breakpoint]:
        """Get all breakpoints for a file."""
        key = str(Path(path).resolve())
        return self.breakpoints.get(key, [])

    def get_active_thread(self) -> Optional[Thread]:
        """Get the currently active thread."""
        if self.current_thread_id is None:
            return None
        for thread in self.threads:
            if thread.id == self.current_thread_id:
                return thread
        return None


@dataclass
class LaunchConfiguration:
    """Configuration for launching a debug session."""

    program: Path
    language: str
    arguments: List[str] = field(default_factory=list)
    working_directory: Optional[Path] = None
    environment: Dict[str, str] = field(default_factory=dict)

    # Debug options
    stop_on_entry: bool = False
    no_debug: bool = False  # Run without debugging

    # Language-specific options
    extra_options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AttachConfiguration:
    """Configuration for attaching to a running process."""

    process_id: Optional[int] = None
    host: Optional[str] = None
    port: Optional[int] = None

    # Language-specific options
    language: str = "python"
    extra_options: Dict[str, Any] = field(default_factory=dict)
