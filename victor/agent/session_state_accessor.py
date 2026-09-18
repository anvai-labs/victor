"""Session state accessor — centralizes session state delegation.

Extracted from Orchestrator to reduce property count and improve SRP.
The orchestrator delegates session state access through this class.
"""

from __future__ import annotations

import asyncio
from typing import Dict, List, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from victor.agent.session_state_manager import SessionStateManager


class SessionStateAccessor:
    """Provides property-based access to SessionStateManager fields.

    Centralizes the 11 session state delegation properties that were
    previously on the Orchestrator class. Each property delegates to
    the underlying SessionStateManager, maintaining the same interface.
    """

    def __init__(self, session_state: SessionStateManager) -> None:
        self._session_state = session_state
        self._stream_turn_lock = asyncio.Lock()

    @property
    def stream_turn_lock(self) -> asyncio.Lock:
        """Return the concurrency gate shared by streaming views of this session."""
        return self._stream_turn_lock

    @property
    def session_state(self) -> SessionStateManager:
        """Get the session state manager.

        Returns:
            SessionStateManager instance for consolidated state tracking
        """
        return self._session_state

    @property
    def tool_calls_used(self) -> int:
        """Get the number of tool calls used in this session.

        Delegates to SessionStateManager.
        """
        return self._session_state.tool_calls_used

    @tool_calls_used.setter
    def tool_calls_used(self, value: int) -> None:
        """Set the number of tool calls used (for backward compatibility)."""
        self._session_state.execution_state.tool_calls_used = value

    @property
    def observed_files(self) -> Set[str]:
        """Get set of files observed/read during this session.

        Delegates to SessionStateManager.
        """
        return self._session_state.execution_state.observed_files

    @observed_files.setter
    def observed_files(self, value: Set[str]) -> None:
        """Set observed files (for checkpoint restore)."""
        self._session_state.execution_state.observed_files = set(value) if value else set()

    @property
    def executed_tools(self) -> List[str]:
        """Get list of executed tool names in order.

        Delegates to SessionStateManager.
        """
        return self._session_state.execution_state.executed_tools

    @executed_tools.setter
    def executed_tools(self, value: List[str]) -> None:
        """Set executed tools (for checkpoint restore)."""
        self._session_state.execution_state.executed_tools = list(value) if value else []

    @property
    def failed_tool_signatures(self) -> Set[Tuple[str, str]]:
        """Get set of (tool_name, args_hash) tuples for failed calls.

        Delegates to SessionStateManager.
        """
        return self._session_state.execution_state.failed_tool_signatures

    @failed_tool_signatures.setter
    def failed_tool_signatures(self, value: Set[Tuple[str, str]]) -> None:
        """Set failed tool signatures (for checkpoint restore)."""
        self._session_state.execution_state.failed_tool_signatures = set(value) if value else set()

    @property
    def tool_capability_warned(self) -> bool:
        """Get whether we've warned about tool capability limitations.

        Delegates to SessionStateManager.
        """
        return self._session_state.session_flags.tool_capability_warned

    @tool_capability_warned.setter
    def tool_capability_warned(self, value: bool) -> None:
        """Set tool capability warning flag."""
        self._session_state.session_flags.tool_capability_warned = value

    @property
    def read_files_session(self) -> Set[str]:
        """Get files read during this session for task completion detection.

        Delegates to SessionStateManager.
        """
        return self._session_state.execution_state.read_files_session

    @property
    def required_files(self) -> List[str]:
        """Get required files extracted from user prompts.

        Delegates to SessionStateManager.
        """
        return self._session_state.execution_state.required_files

    @required_files.setter
    def required_files(self, value: List[str]) -> None:
        """Set required files list."""
        self._session_state.execution_state.required_files = list(value)

    @property
    def required_outputs(self) -> List[str]:
        """Get required outputs extracted from user prompts.

        Delegates to SessionStateManager.
        """
        return self._session_state.execution_state.required_outputs

    @required_outputs.setter
    def required_outputs(self, value: List[str]) -> None:
        """Set required outputs list."""
        self._session_state.execution_state.required_outputs = list(value)

    @property
    def all_files_read_nudge_sent(self) -> bool:
        """Get whether we've sent a nudge that all required files are read.

        Delegates to SessionStateManager.
        """
        return self._session_state.session_flags.all_files_read_nudge_sent

    @all_files_read_nudge_sent.setter
    def all_files_read_nudge_sent(self, value: bool) -> None:
        """Set all files read nudge flag."""
        self._session_state.session_flags.all_files_read_nudge_sent = value

    @property
    def cumulative_token_usage(self) -> Dict[str, int]:
        """Get cumulative token usage for evaluation/benchmarking.

        Delegates to SessionStateManager.
        """
        return self._session_state.get_token_usage()

    @cumulative_token_usage.setter
    def cumulative_token_usage(self, value: Dict[str, int]) -> None:
        """Set cumulative token usage (for backward compatibility)."""
        self._session_state.execution_state.token_usage = dict(value)
