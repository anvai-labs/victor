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

"""Service-owned runtime protocols for chat migration boundaries.

These protocols describe the coordinator-era runtime state that still exists
around chat execution, planning, and streaming, but they are now hosted in the
service protocol package so internal code does not depend on coordinator-named
modules.

They remain useful for:
- Legacy chat runtime compatibility tests
- Debug-time conformance checks around orchestrator runtime state
- Incremental migration of planning/streaming helper ownership

New high-level application code should still prefer `ChatServiceProtocol`,
`ProviderServiceProtocol`, and `ToolServiceProtocol` where possible.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional, Protocol, runtime_checkable

__all__ = [
    "ChatRuntimeHelperAccessProtocol",
    "ExecutionMode",
    "ChatContextProtocol",
    "ToolContextProtocol",
    "ProviderContextProtocol",
    "ChatOrchestratorProtocol",
    "PlanningContextProtocol",
    "SyncChatRuntimeProtocol",
    "OrchestratorRuntimeProtocol",
    "ParallelExplorationProtocol",
    "StatePassedExplorationProtocol",
]


class ExecutionMode(Enum):
    """Execution mode for chat operations."""

    SYNC = "sync"
    STREAMING = "streaming"
    AUTO = "auto"


@runtime_checkable
class ChatRuntimeHelperAccessProtocol(Protocol):
    """Canonical helper getter surface for planning and context-limit runtimes."""

    def _get_planning_chat_runtime(self) -> Any:
        """Return the canonical planning runtime helper."""
        ...

    def _get_context_limit_runtime(self) -> Any:
        """Return the canonical context-limit runtime helper."""
        ...


class ChatContextProtocol(Protocol):
    """Legacy conversation/message runtime boundary for the chat shim path."""

    conversation: Any
    messages: Any

    def add_message(self, role: str, content: str, **metadata: Any) -> None:
        """Add a message to conversation history."""
        ...

    _system_added: bool
    conversation_controller: Any
    _conversation_controller: Any
    _context_compactor: Any
    _context_manager: Any
    settings: Any
    _session_state: Any

    def _check_context_overflow(self, max_context: int) -> bool:
        """Check if conversation exceeds context window limit."""
        ...

    def _get_max_context_chars(self) -> int:
        """Get maximum context length in characters for current provider/model."""
        ...

    def _get_thinking_disabled_prompt(self, prompt: str) -> str:
        """Wrap prompt to disable thinking for completion requests."""
        ...

    _cumulative_token_usage: Dict[str, int]


class ToolContextProtocol(Protocol):
    """Tool runtime boundary for the chat shim path."""

    tool_selector: Any
    tool_adapter: Any
    _tool_planner: Any
    tool_budget: int
    tool_calls_used: int
    use_semantic_selection: bool
    observed_files: Any

    async def execute_tool_calls(self, tool_calls: Any) -> List[Dict[str, Any]]:
        """Execute tool calls via the canonical tool runtime surface."""
        ...

    def _model_supports_tool_calls(self) -> bool:
        """Check if current model supports native tool calling."""
        ...


class ProviderContextProtocol(Protocol):
    """Legacy provider runtime boundary for the chat shim path."""

    provider: Any
    model: str
    temperature: float
    max_tokens: int
    thinking: Any
    _provider_service: Any
    _cancel_event: Any
    _is_streaming: bool

    def _check_cancellation(self) -> bool:
        """Check if cancellation has been requested."""
        ...


@runtime_checkable
class ChatOrchestratorProtocol(
    ChatContextProtocol,
    ToolContextProtocol,
    ProviderContextProtocol,
    Protocol,
):
    """Complete legacy runtime protocol for chat shim compatibility."""

    task_classifier: Any
    task_coordinator: Any
    _task_analyzer: Any
    unified_tracker: Any
    _task_completion_detector: Any

    _recovery_coordinator: Any
    _recovery_integration: Any

    def create_recovery_context(self, stream_ctx: Any) -> Any:
        """Create StreamingRecoveryContext via the canonical runtime surface."""
        ...

    async def _handle_recovery_with_integration(
        self,
        stream_ctx: Any,
        full_content: str,
        tool_calls: Any,
        mentioned_tools: Optional[List[str]] = None,
    ) -> Any:
        """Handle recovery using the recovery integration system."""
        ...

    def _apply_recovery_action(self, recovery_action: Any, stream_ctx: Any) -> Any:
        """Apply a recovery action and return optional StreamChunk."""
        ...

    def _record_runtime_intelligence_outcome(
        self,
        success: bool,
        quality_score: float,
        user_satisfied: bool,
        completed: bool,
    ) -> None:
        """Record outcome for intelligent pipeline metrics."""
        ...

    async def _validate_runtime_intelligence_response(
        self,
        response: str,
        query: str,
        tool_calls: int,
        task_type: str,
    ) -> Optional[Dict[str, Any]]:
        """Validate response quality using intelligent pipeline."""
        ...

    async def _prepare_runtime_intelligence_request(
        self,
        task: str,
        task_type: str,
    ) -> Any:
        """Prepare intelligent pipeline context before provider call."""
        ...

    sanitizer: Any
    response_completer: Any
    debug_logger: Any
    reminder_manager: Any
    _chunk_generator: Any
    _presentation: Any
    _metrics_collector: Any
    _streaming_handler: Any

    _required_files: List[str]
    _required_outputs: List[str]
    _read_files_session: Any
    _all_files_read_nudge_sent: bool
    _usage_analytics: Any
    _sequence_tracker: Any


@runtime_checkable
class PlanningContextProtocol(Protocol):
    """Legacy planning runtime boundary while planning migrates service-side."""

    @property
    def provider(self) -> Any:
        """LLM provider instance."""
        ...

    @property
    def model(self) -> str:
        """Current model name."""
        ...

    @property
    def max_tokens(self) -> int:
        """Maximum tokens for completion."""
        ...

    @property
    def profile(self) -> Any:
        """Optional profile carrying planning model/provider overrides."""
        ...

    @property
    def planning_model_override(self) -> Any:
        """Optional CLI override for planning model/provider routing."""
        ...

    @property
    def _system_prompt_override(self) -> Any:
        """Current temporary system prompt override, if any."""
        ...

    @property
    def skill_matcher(self) -> Any:
        """Shared framework skill matcher when available."""
        ...

    async def chat(self, user_message: str, use_planning: bool = False) -> Any:
        """Execute a non-streaming chat turn."""
        ...

    def set_system_prompt(self, prompt: str) -> None:
        """Set a temporary system prompt override for planning."""
        ...


@runtime_checkable
class SyncChatRuntimeProtocol(Protocol):
    """Fallback runtime surface for deprecated sync chat compatibility."""

    async def chat(self, user_message: str, use_planning: bool = False) -> Any:
        """Execute a non-streaming chat turn."""
        ...

    def get_last_skill_match_info(self) -> Optional[Dict[str, Any]]:
        """Return skill metadata for the last completed chat turn."""
        ...


# =============================================================================
# Typed contracts for TurnExecutor (Wave 1 — duck-typing reduction)
# =============================================================================


@runtime_checkable
class OrchestratorRuntimeProtocol(Protocol):
    """Minimum typed surface TurnExecutor needs from the orchestrator.

    Replaces ``Any`` return type of ``_resolve_orchestrator()`` so mypy can
    validate attribute access instead of silently accepting Any.  Only methods
    are checked by ``isinstance()`` at runtime; attribute annotations serve as
    static documentation for the static type checker.
    """

    async def get_messages(self) -> List[Any]:
        """Return the current conversation message list."""
        ...


@runtime_checkable
class ParallelExplorationProtocol(Protocol):
    """Classic parallel exploration path (ExplorationCoordinator)."""

    async def explore_parallel(
        self,
        task_description: str,
        project_root: Any,
        max_results: int = 5,
        **kwargs: Any,
    ) -> Any:
        """Run parallel file/codebase exploration and return findings."""
        ...


@runtime_checkable
class StatePassedExplorationProtocol(Protocol):
    """State-passed exploration path (ExplorationStatePassedCoordinator)."""

    async def explore(
        self,
        context: Any,
        user_message: str,
        **kwargs: Any,
    ) -> Any:
        """Run state-passed exploration using a context snapshot."""
        ...
