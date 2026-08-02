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

"""Session state management for agent orchestrator.

This module extracts session state management from the AgentOrchestrator god class
(TD-002) into a dedicated, focused component. It handles:

- Tool call tracking and budget management
- File observation and read tracking
- Failed tool signature tracking (loop prevention)
- Session flags for conversation flow control
- Token usage accumulation
- Task requirement tracking
- Checkpoint serialization/deserialization

Design:
- Standalone (no orchestrator imports)
- Clear separation between execution state and session flags
- Serializable for checkpoint/restore operations
- Thread-safe state updates where needed

Usage:
    manager = SessionStateManager(tool_budget=200)

    # Track tool execution
    manager.record_tool_call("read_file", {"path": "/src/main.py"})
    manager.increment_tool_calls()

    # Check budget
    if manager.is_budget_exhausted():
        # Force completion

    # Track file reads
    manager.record_file_read("/src/main.py")

    # Check for task completion
    manager.set_task_requirements(
        required_files=["main.py", "utils.py"],
        required_outputs=["analysis"]
    )
    if manager.check_all_files_read():
        # All required files have been read

    # Checkpoint/restore
    state = manager.get_checkpoint_state()
    # ... later ...
    manager.apply_checkpoint_state(state)
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ExecutionState:
    """Tracks tool execution state during a session.

    Attributes:
        tool_calls_used: Number of tool calls made in this session
        observed_files: Set of file paths that have been observed/read
        executed_tools: List of tool names that have been executed (with order)
        failed_tool_signatures: Set of (tool_name, args_hash) tuples for failed calls
        read_files_session: Files actually read (vs just observed) in this session
        required_files: Files required to complete the current task
        required_outputs: Outputs required to complete the current task
        token_usage: Cumulative token usage tracking for evaluation
        disable_embeddings: Disable codebase embeddings for this session (workflow service mode)
    """

    tool_calls_used: int = 0
    observed_files: Set[str] = field(default_factory=set)
    executed_tools: List[str] = field(default_factory=list)
    failed_tool_signatures: Set[Tuple[str, str]] = field(default_factory=set)
    read_files_session: Set[str] = field(default_factory=set)
    required_files: List[str] = field(default_factory=list)
    required_outputs: List[str] = field(default_factory=list)
    token_usage: Dict[str, int] = field(
        default_factory=lambda: {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        }
    )
    disable_embeddings: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize execution state to dictionary.

        Returns:
            Dictionary representation for checkpoint storage
        """
        return {
            "tool_calls_used": self.tool_calls_used,
            "observed_files": list(self.observed_files),
            "executed_tools": list(self.executed_tools),
            "failed_tool_signatures": [list(sig) for sig in self.failed_tool_signatures],
            "read_files_session": list(self.read_files_session),
            "required_files": list(self.required_files),
            "required_outputs": list(self.required_outputs),
            "token_usage": dict(self.token_usage),
            "disable_embeddings": self.disable_embeddings,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionState":
        """Deserialize execution state from dictionary.

        Args:
            data: Dictionary from checkpoint storage

        Returns:
            ExecutionState instance
        """
        return cls(
            tool_calls_used=data.get("tool_calls_used", 0),
            observed_files=set(data.get("observed_files", [])),
            executed_tools=list(data.get("executed_tools", [])),
            failed_tool_signatures={tuple(sig) for sig in data.get("failed_tool_signatures", [])},
            read_files_session=set(data.get("read_files_session", [])),
            required_files=list(data.get("required_files", [])),
            required_outputs=list(data.get("required_outputs", [])),
            token_usage=data.get(
                "token_usage",
                {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                },
            ),
            disable_embeddings=data.get("disable_embeddings", False),
        )


@dataclass
class SessionFlags:
    """Boolean flags controlling session behavior.

    Attributes:
        system_added: Whether system prompt has been added to conversation
        all_files_read_nudge_sent: Whether we've sent a nudge that all required files are read
        tool_capability_warned: Whether we've warned about tool capability limitations
        consecutive_blocked_attempts: Count of consecutive blocked tool attempts
        total_blocked_attempts: Total blocked attempts in the session
    """

    system_added: bool = False
    all_files_read_nudge_sent: bool = False
    tool_capability_warned: bool = False
    consecutive_blocked_attempts: int = 0
    total_blocked_attempts: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize session flags to dictionary.

        Returns:
            Dictionary representation for checkpoint storage
        """
        return {
            "system_added": self.system_added,
            "all_files_read_nudge_sent": self.all_files_read_nudge_sent,
            "tool_capability_warned": self.tool_capability_warned,
            "consecutive_blocked_attempts": self.consecutive_blocked_attempts,
            "total_blocked_attempts": self.total_blocked_attempts,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionFlags":
        """Deserialize session flags from dictionary.

        Args:
            data: Dictionary from checkpoint storage

        Returns:
            SessionFlags instance
        """
        return cls(
            system_added=data.get("system_added", False),
            all_files_read_nudge_sent=data.get("all_files_read_nudge_sent", False),
            tool_capability_warned=data.get("tool_capability_warned", False),
            consecutive_blocked_attempts=data.get("consecutive_blocked_attempts", 0),
            total_blocked_attempts=data.get("total_blocked_attempts", 0),
        )


class SessionStateManager:
    """Manages session state for agent orchestration.

    Centralizes all session state management that was previously scattered
    across the AgentOrchestrator class. Provides clean APIs for:

    - Tool call tracking and budget enforcement
    - File observation and read tracking
    - Failed tool detection (loop prevention)
    - Session flag management
    - Token usage tracking
    - Checkpoint/restore operations

    Thread-safety: This class is NOT thread-safe. Use external locking if
    accessing from multiple threads.

    Example:
        manager = SessionStateManager(tool_budget=200)

        # Record tool execution
        manager.record_tool_call("read_file", {"path": "/src/main.py"})

        # Check if we should stop
        if manager.is_budget_exhausted():
            print("Tool budget exhausted")

        # Get session summary
        summary = manager.get_session_summary()
        print(f"Used {summary['tool_calls_used']}/{summary['tool_budget']} tools")
    """

    def __init__(self, tool_budget: int = 200):
        """Initialize session state manager.

        Args:
            tool_budget: Maximum number of tool calls allowed (default: 200)
        """
        self._tool_budget = tool_budget
        self._execution_state = ExecutionState()
        self._session_flags = SessionFlags()

        logger.debug(f"SessionStateManager initialized with tool_budget={tool_budget}")

    @property
    def tool_budget(self) -> int:
        """Get the tool budget for this session."""
        return self._tool_budget

    @tool_budget.setter
    def tool_budget(self, value: int) -> None:
        """Set the tool budget for this session."""
        self._tool_budget = max(1, value)

    @property
    def tool_calls_used(self) -> int:
        """Get the number of tool calls used."""
        return self._execution_state.tool_calls_used

    @property
    def observed_files(self) -> Set[str]:
        """Get the set of observed files."""
        return self._execution_state.observed_files

    @property
    def executed_tools(self) -> List[str]:
        """Get the list of executed tools."""
        return self._execution_state.executed_tools

    @property
    def execution_state(self) -> ExecutionState:
        """Get the execution state (read-only access)."""
        return self._execution_state

    @property
    def session_flags(self) -> SessionFlags:
        """Get the session flags (read-only access)."""
        return self._session_flags

    # =========================================================================
    # Tool Call Tracking
    # =========================================================================

    def record_tool_call(self, tool_name: str, args: Dict[str, Any]) -> None:
        """Record a tool call execution.

        Tracks the tool name in the executed tools list and observes any
        file paths in the arguments.

        Args:
            tool_name: Name of the tool being called
            args: Tool arguments dictionary
        """
        self._execution_state.executed_tools.append(tool_name)

        # Track file paths from common argument names
        for key in ("path", "file_path", "filepath", "file"):
            if key in args and isinstance(args[key], str):
                self._execution_state.observed_files.add(args[key])

        logger.debug(
            f"Recorded tool call: {tool_name}, "
            f"total_calls={len(self._execution_state.executed_tools)}"
        )

    def increment_tool_calls(self, count: int = 1) -> int:
        """Increment the tool calls counter.

        Args:
            count: Number of calls to add (default: 1)

        Returns:
            New total tool calls count
        """
        self._execution_state.tool_calls_used += count
        return self._execution_state.tool_calls_used

    def is_budget_exhausted(self) -> bool:
        """Check if the tool budget has been exhausted.

        Returns:
            True if tool_calls_used >= tool_budget
        """
        return self._execution_state.tool_calls_used >= self._tool_budget

    def get_remaining_budget(self) -> int:
        """Get the remaining tool budget.

        Returns:
            Number of tool calls remaining
        """
        return max(0, self._tool_budget - self._execution_state.tool_calls_used)

    # =========================================================================
    # Failed Tool Signature Tracking
    # =========================================================================

    @staticmethod
    def _hash_args(args: Dict[str, Any]) -> str:
        """Create a hash of tool arguments for signature matching.

        Args:
            args: Tool arguments dictionary

        Returns:
            MD5 hash string of the arguments
        """
        # Sort keys for consistent hashing
        sorted_args = sorted(args.items())
        args_str = str(sorted_args)
        return hashlib.md5(args_str.encode()).hexdigest()

    def check_failed_signature(self, name: str, args_hash: str) -> bool:
        """Check if a tool call signature has already failed.

        Used to prevent infinite loops of retrying the same failed call.

        Args:
            name: Tool name
            args_hash: Hash of the tool arguments

        Returns:
            True if this exact call has failed before
        """
        signature = (name, args_hash)
        return signature in self._execution_state.failed_tool_signatures

    def add_failed_signature(self, name: str, args_hash: str) -> None:
        """Record a failed tool call signature.

        Args:
            name: Tool name
            args_hash: Hash of the tool arguments
        """
        signature = (name, args_hash)
        self._execution_state.failed_tool_signatures.add(signature)
        logger.debug(f"Added failed signature: {name}:{args_hash[:8]}...")

    def check_and_record_failed(self, tool_name: str, args: Dict[str, Any]) -> bool:
        """Check if a tool call has failed before, and if not, mark it as failed.

        Convenience method combining check and add operations.

        Args:
            tool_name: Tool name
            args: Tool arguments

        Returns:
            True if this call had already failed (should skip)
        """
        args_hash = self._hash_args(args)
        if self.check_failed_signature(tool_name, args_hash):
            return True
        self.add_failed_signature(tool_name, args_hash)
        return False

    # =========================================================================
    # File Read Tracking
    # =========================================================================

    def record_file_read(self, filepath: str) -> None:
        """Record that a file has been read.

        Args:
            filepath: Path to the file that was read
        """
        self._execution_state.read_files_session.add(filepath)
        self._execution_state.observed_files.add(filepath)
        logger.debug(
            f"Recorded file read: {filepath}, "
            f"total_read={len(self._execution_state.read_files_session)}"
        )

    def get_read_files(self) -> Set[str]:
        """Get the set of files read in this session.

        Returns:
            Set of file paths
        """
        return self._execution_state.read_files_session.copy()

    # =========================================================================
    # Task Requirements
    # =========================================================================

    def set_task_requirements(
        self,
        required_files: Optional[List[str]] = None,
        required_outputs: Optional[List[str]] = None,
    ) -> None:
        """Set the requirements for the current task.

        Args:
            required_files: Files that must be read to complete the task
            required_outputs: Outputs that must be generated
        """
        if required_files is not None:
            self._execution_state.required_files = list(required_files)
        if required_outputs is not None:
            self._execution_state.required_outputs = list(required_outputs)

        logger.debug(
            f"Task requirements set: {len(self._execution_state.required_files)} files, "
            f"{len(self._execution_state.required_outputs)} outputs"
        )

    def check_all_files_read(self) -> bool:
        """Check if all required files have been read.

        Returns:
            True if all required files are in read_files_session
        """
        if not self._execution_state.required_files:
            return False

        return self._execution_state.read_files_session.issuperset(
            set(self._execution_state.required_files)
        )

    # =========================================================================
    # Session Flags
    # =========================================================================

    def mark_system_added(self) -> None:
        """Mark that the system prompt has been added."""
        self._session_flags.system_added = True

    def is_system_added(self) -> bool:
        """Check if system prompt has been added."""
        return self._session_flags.system_added

    def mark_all_files_read_nudge_sent(self) -> None:
        """Mark that the 'all files read' nudge has been sent."""
        self._session_flags.all_files_read_nudge_sent = True

    def should_send_all_files_read_nudge(self) -> bool:
        """Check if we should send the 'all files read' nudge.

        Returns:
            True if all files are read and nudge hasn't been sent
        """
        return self.check_all_files_read() and not self._session_flags.all_files_read_nudge_sent

    def mark_tool_capability_warned(self) -> None:
        """Mark that tool capability warning has been shown."""
        self._session_flags.tool_capability_warned = True

    def is_tool_capability_warned(self) -> bool:
        """Check if tool capability warning has been shown."""
        return self._session_flags.tool_capability_warned

    def record_blocked_attempt(self) -> int:
        """Record a blocked tool attempt.

        Returns:
            Current consecutive blocked attempts count
        """
        self._session_flags.consecutive_blocked_attempts += 1
        self._session_flags.total_blocked_attempts += 1
        return self._session_flags.consecutive_blocked_attempts

    def reset_blocked_attempts(self) -> None:
        """Reset the consecutive blocked attempts counter."""
        self._session_flags.consecutive_blocked_attempts = 0

    def get_blocked_attempts(self) -> Tuple[int, int]:
        """Get blocked attempt counts.

        Returns:
            Tuple of (consecutive_blocked, total_blocked)
        """
        return (
            self._session_flags.consecutive_blocked_attempts,
            self._session_flags.total_blocked_attempts,
        )

    # =========================================================================
    # Token Usage Tracking
    # =========================================================================

    def get_token_usage(self) -> Dict[str, int]:
        """Get cumulative token usage.

        Returns:
            Dictionary with prompt_tokens, completion_tokens, total_tokens, etc.
        """
        return self._execution_state.token_usage.copy()

    def reset_token_usage(self) -> None:
        """Reset cumulative token usage tracking."""
        for key in self._execution_state.token_usage:
            self._execution_state.token_usage[key] = 0

    def update_token_usage(
        self,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cache_creation_input_tokens: int = 0,
        cache_read_input_tokens: int = 0,
    ) -> None:
        """Update cumulative token usage.

        Args:
            prompt_tokens: Input tokens used
            completion_tokens: Output tokens generated
            cache_creation_input_tokens: Tokens used for cache creation
            cache_read_input_tokens: Tokens read from cache
        """
        self._execution_state.token_usage["prompt_tokens"] += prompt_tokens
        self._execution_state.token_usage["completion_tokens"] += completion_tokens
        self._execution_state.token_usage["total_tokens"] += prompt_tokens + completion_tokens
        self._execution_state.token_usage[
            "cache_creation_input_tokens"
        ] += cache_creation_input_tokens
        self._execution_state.token_usage["cache_read_input_tokens"] += cache_read_input_tokens

    # =========================================================================
    # Checkpoint/Restore
    # =========================================================================

    def get_checkpoint_state(self) -> Dict[str, Any]:
        """Serialize session state for checkpointing.

        Returns:
            Dictionary containing all session state for storage
        """
        return {
            "tool_budget": self._tool_budget,
            "execution_state": self._execution_state.to_dict(),
            "session_flags": self._session_flags.to_dict(),
        }

    def apply_checkpoint_state(self, state: Dict[str, Any]) -> None:
        """Restore session state from a checkpoint.

        Args:
            state: Dictionary from get_checkpoint_state()
        """
        self._tool_budget = state.get("tool_budget", self._tool_budget)
        if "execution_state" in state:
            self._execution_state = ExecutionState.from_dict(state["execution_state"])
        if "session_flags" in state:
            self._session_flags = SessionFlags.from_dict(state["session_flags"])

        logger.debug(
            f"Applied checkpoint state: tool_calls={self._execution_state.tool_calls_used}"
        )

    # =========================================================================
    # Reset
    # =========================================================================

    def reset(self, preserve_token_usage: bool = False) -> None:
        """Reset session state for a new session.

        Args:
            preserve_token_usage: If True, keep accumulated token usage
        """
        saved_tokens = None
        if preserve_token_usage:
            saved_tokens = self._execution_state.token_usage.copy()

        self._execution_state = ExecutionState()
        self._session_flags = SessionFlags()

        if saved_tokens:
            self._execution_state.token_usage = saved_tokens

        logger.debug("Session state reset")

    def reset_for_new_turn(self) -> None:
        """Partial reset for a new conversation turn.

        Preserves some state (like token usage and observed files) but
        resets turn-specific state.
        """
        self._session_flags.consecutive_blocked_attempts = 0
        self._execution_state.read_files_session.clear()
        self._session_flags.all_files_read_nudge_sent = False
        self._execution_state.tool_calls_used = 0  # Reset tool counter for new turn

        logger.debug("Session state reset for new turn")

    # =========================================================================
    # Summary/Stats
    # =========================================================================

    def get_session_summary(self) -> Dict[str, Any]:
        """Get a summary of the session state.

        Returns:
            Dictionary with session statistics
        """
        return {
            "tool_budget": self._tool_budget,
            "tool_calls_used": self._execution_state.tool_calls_used,
            "tool_calls_remaining": self.get_remaining_budget(),
            "budget_exhausted": self.is_budget_exhausted(),
            "files_observed": len(self._execution_state.observed_files),
            "files_read": len(self._execution_state.read_files_session),
            "unique_tools_used": len(set(self._execution_state.executed_tools)),
            "total_tool_executions": len(self._execution_state.executed_tools),
            "failed_signatures": len(self._execution_state.failed_tool_signatures),
            "required_files_count": len(self._execution_state.required_files),
            "all_files_read": self.check_all_files_read(),
            "system_added": self._session_flags.system_added,
            "total_blocked_attempts": self._session_flags.total_blocked_attempts,
            "token_usage": self._execution_state.token_usage.copy(),
        }

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"SessionStateManager("
            f"tool_calls={self._execution_state.tool_calls_used}/{self._tool_budget}, "
            f"files_read={len(self._execution_state.read_files_session)}, "
            f"tools_executed={len(self._execution_state.executed_tools)})"
        )


# =============================================================================
# Factory Function
# =============================================================================


def create_session_state_manager(
    tool_budget: int = 200,
    initial_state: Optional[Dict[str, Any]] = None,
) -> SessionStateManager:
    """Create a SessionStateManager instance.

    Factory function for dependency injection.

    Args:
        tool_budget: Maximum tool calls allowed
        initial_state: Optional checkpoint state to restore from

    Returns:
        Configured SessionStateManager instance
    """
    manager = SessionStateManager(tool_budget=tool_budget)
    if initial_state:
        manager.apply_checkpoint_state(initial_state)
    return manager
