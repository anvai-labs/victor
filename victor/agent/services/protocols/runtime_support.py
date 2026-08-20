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

"""Service-owned runtime protocol surfaces for migrated agent seams.

These protocols are the canonical host for active service/runtime imports under
``victor.agent.services.protocols``.

Legacy imports from ``victor.agent.protocols`` remain supported for backward
compatibility, but those names now alias back to these service-owned protocol
definitions.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

from victor.core.constants import DEFAULT_VERTICAL


@runtime_checkable
class CoordinationAdvisorRuntimeProtocol(Protocol):
    """Protocol for service-owned coordination recommendation runtime access."""

    def suggest_for_task(
        self,
        *,
        task_type: str,
        complexity: str,
        mode: str = "build",
        runtime_subject: Optional[Any] = None,
        coordination_advisor: Optional[Any] = None,
        vertical_context: Optional[Any] = None,
    ) -> Any:
        """Build coordination suggestions using shared framework logic."""
        ...

    def serialize_suggestion(
        self,
        suggestion: Any,
        *,
        vertical: Optional[str] = None,
        available_teams: Optional[tuple[str, ...]] = None,
        available_workflows: Optional[tuple[str, ...]] = None,
        default_workflow: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Serialize coordination suggestions for transport or state output."""
        ...


@runtime_checkable
class ToolPlanningRuntimeProtocol(Protocol):
    """Canonical service-owned protocol for tool planning runtime access."""

    def plan_tools(
        self, goals: List[str], available_inputs: Optional[List[str]] = None
    ) -> List[Any]:
        """Plan a sequence of tools to satisfy goals."""
        ...

    def infer_goals_from_message(self, user_message: str) -> List[str]:
        """Infer planning goals from user request."""
        ...

    def filter_tools_by_intent(
        self,
        tools: List[Any],
        current_intent: Optional[Any] = None,
        user_message: Optional[str] = None,
    ) -> List[Any]:
        """Filter tools based on detected user intent."""
        ...


@runtime_checkable
class TaskRuntimeProtocol(Protocol):
    """Canonical service-owned protocol for task coordination runtime access."""

    def prepare_task(
        self, user_message: str, unified_task_type: Any, conversation_controller: Any
    ) -> tuple[Any, int]:
        """Prepare task-specific guidance and budget adjustments."""
        ...

    def apply_intent_guard(self, user_message: str, conversation_controller: Any) -> None:
        """Detect intent and inject prompt guards for read-only tasks."""
        ...

    def apply_task_guidance(
        self,
        user_message: str,
        unified_task_type: Any,
        is_analysis_task: bool,
        is_action_task: bool,
        needs_execution: bool,
        max_exploration_iterations: int,
        conversation_controller: Any,
    ) -> None:
        """Apply guidance and budget tweaks for analysis/action tasks."""
        ...

    @property
    def current_intent(self) -> Any:
        """Get the current detected intent."""
        ...

    @property
    def temperature(self) -> float:
        """Get the current temperature setting."""
        ...

    @property
    def tool_budget(self) -> int:
        """Get the current tool budget."""
        ...

    @property
    def observed_files(self) -> list:
        """Get the list of observed files."""
        ...


@runtime_checkable
class StateRuntimeProtocol(Protocol):
    """Canonical service-owned protocol for live conversation-stage state."""

    def get_current_stage(self) -> Any:
        """Get the current conversation stage."""
        ...

    def transition_to(
        self,
        stage: Any,
        reason: str = "",
        tool_name: Optional[str] = None,
    ) -> bool:
        """Transition to a new conversation stage."""
        ...

    def get_message_history(self) -> List[Any]:
        """Get the full message history."""
        ...

    def get_recent_messages(
        self,
        limit: int = 10,
        include_system: bool = False,
    ) -> List[Any]:
        """Get recent messages from history."""
        ...

    def is_in_exploration_phase(self) -> bool:
        """Check if currently in exploration phase."""
        ...

    def is_in_execution_phase(self) -> bool:
        """Check if currently in execution phase."""
        ...


@runtime_checkable
class PromptRuntimeProtocol(Protocol):
    """Canonical service-owned protocol for mutable prompt runtime access."""

    def build_system_prompt(
        self,
        context: Any,
        include_hints: bool = True,
    ) -> str:
        """Build the complete system prompt."""
        ...

    def add_task_hint(self, task_type: str, hint: str) -> None:
        """Add or update a task-type hint."""
        ...

    def get_task_hint(self, task_type: str) -> Optional[str]:
        """Get the hint for a task type."""
        ...

    def add_section(
        self,
        name: str,
        content: str,
        priority: Optional[int] = None,
    ) -> None:
        """Add a runtime section to be included in prompts."""
        ...

    def set_grounding_mode(self, mode: str) -> None:
        """Set the grounding rules mode."""
        ...


@runtime_checkable
class StreamingRecoveryRuntimeProtocol(Protocol):
    """Canonical service-owned protocol for streaming recovery coordination."""

    def check_time_limit(self, ctx: Any) -> Optional[Any]:
        """Check if session has exceeded time limit."""
        ...

    def check_iteration_limit(self, ctx: Any) -> Optional[Any]:
        """Check if session has exceeded iteration limit."""
        ...

    def check_natural_completion(
        self, ctx: Any, has_tool_calls: bool, content_length: int
    ) -> Optional[Any]:
        """Check for natural completion."""
        ...

    def check_tool_budget(self, ctx: Any) -> bool:
        """Check if tool budget has been exhausted."""
        ...

    def check_progress(self, ctx: Any) -> bool:
        """Check if session is making progress."""
        ...

    def check_blocked_threshold(self, ctx: Any, all_blocked: bool) -> Optional[Tuple[Any, bool]]:
        """Check if too many tools have been blocked."""
        ...

    def check_force_action(self, ctx: Any) -> Tuple[bool, Optional[str]]:
        """Check if recovery handler recommends force action."""
        ...

    def handle_empty_response(self, ctx: Any) -> Tuple[Optional[Any], bool]:
        """Handle empty model response."""
        ...

    def handle_blocked_tool(
        self, ctx: Any, tool_name: str, tool_args: Dict[str, Any], block_reason: str
    ) -> Any:
        """Handle blocked tool call."""
        ...

    def handle_force_tool_execution(self, ctx: Any) -> Tuple[bool, Optional[List[Any]]]:
        """Handle forced tool execution."""
        ...

    def handle_force_completion(self, ctx: Any) -> Optional[List[Any]]:
        """Handle forced completion."""
        ...

    def handle_loop_warning(self, ctx: Any) -> Optional[List[Any]]:
        """Handle loop detection warning."""
        ...

    async def handle_recovery_with_integration(
        self,
        ctx: Any,
        full_content: str,
        tool_calls: Optional[List[Dict[str, Any]]],
        mentioned_tools: Optional[List[str]],
        message_adder: Any,
    ) -> Any:
        """Handle response using the recovery integration."""
        ...

    def apply_recovery_action(
        self, recovery_action: Any, ctx: Any, message_adder: Any
    ) -> Optional[Any]:
        """Apply a recovery action from the recovery integration."""
        ...

    def filter_blocked_tool_calls(
        self, ctx: Any, tool_calls: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Any], int]:
        """Filter out blocked tool calls."""
        ...

    def truncate_tool_calls(
        self, ctx: Any, tool_calls: List[Dict[str, Any]], max_calls: int
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """Truncate tool calls to budget limit."""
        ...

    def get_recovery_prompts(self, ctx: Any) -> List[str]:
        """Get recovery prompts for current context."""
        ...

    def get_recovery_fallback_message(self, ctx: Any) -> str:
        """Get fallback message when recovery fails."""
        ...

    def should_use_tools_for_recovery(self, ctx: Any) -> bool:
        """Determine if tools should be used during recovery."""
        ...

    def format_completion_metrics(self, ctx: Any) -> Dict[str, Any]:
        """Format completion metrics for display."""
        ...

    def format_budget_exhausted_metrics(self, ctx: Any) -> Dict[str, Any]:
        """Format budget exhausted metrics."""
        ...

    def generate_tool_result_chunks(self, results: List[Any], ctx: Any) -> List[Any]:
        """Generate stream chunks from tool results."""
        ...


@runtime_checkable
class ChunkRuntimeProtocol(Protocol):
    """Canonical service-owned protocol for streaming chunk generation."""

    def generate_tool_start_chunk(
        self, tool_name: str, tool_args: Dict[str, Any], status_msg: str
    ) -> Any:
        """Generate chunk indicating tool execution start."""
        ...

    def generate_tool_result_chunks(self, result: Dict[str, Any]) -> List[Any]:
        """Generate chunks for tool execution result."""
        ...

    def generate_thinking_status_chunk(self) -> Any:
        """Generate chunk indicating thinking/processing status."""
        ...

    def generate_budget_error_chunk(self) -> Any:
        """Generate chunk for budget limit error."""
        ...

    def generate_force_response_error_chunk(self) -> Any:
        """Generate chunk for forced response error."""
        ...

    def generate_final_marker_chunk(self) -> Any:
        """Generate final marker chunk to signal stream completion."""
        ...

    def generate_metrics_chunk(
        self, metrics_line: str, is_final: bool = False, prefix: str = "\n\n"
    ) -> Any:
        """Generate chunk for metrics display."""
        ...

    def generate_content_chunk(self, content: str, is_final: bool = False, suffix: str = "") -> Any:
        """Generate chunk for content display."""
        ...

    def get_budget_exhausted_chunks(self, stream_ctx: Any) -> List[Any]:
        """Get chunks for budget exhaustion warning."""
        ...


@runtime_checkable
class RLLearningRuntimeProtocol(Protocol):
    """Canonical service-owned protocol for reinforcement learning runtime access."""

    def record_outcome(
        self,
        learner_name: str,
        outcome: Any,
        vertical: str = DEFAULT_VERTICAL,
    ) -> None:
        """Record an outcome for a specific learner."""
        ...

    def get_recommendation(
        self,
        learner_name: str,
        provider: str,
        model: str,
        task_type: str,
    ) -> Optional[Any]:
        """Get recommendation from a learner."""
        ...

    def export_metrics(self) -> Dict[str, Any]:
        """Export all learned values and metrics for monitoring."""
        ...

    def create_prompt_rollout_experiment(
        self,
        *,
        section_name: str,
        provider: str,
        treatment_hash: str,
        control_hash: Optional[str] = None,
        traffic_split: float = 0.1,
        min_samples_per_variant: int = 50,
    ) -> Optional[str]:
        """Create a prompt rollout experiment for an approved prompt candidate."""
        ...

    async def create_prompt_rollout_experiment_async(
        self,
        *,
        section_name: str,
        provider: str,
        treatment_hash: str,
        control_hash: Optional[str] = None,
        traffic_split: float = 0.1,
        min_samples_per_variant: int = 50,
    ) -> Optional[str]:
        """Async version of create_prompt_rollout_experiment."""
        ...

    def analyze_prompt_rollout_experiment(
        self,
        *,
        section_name: str,
        provider: str,
        treatment_hash: str,
    ) -> Optional[Dict[str, Any]]:
        """Analyze a prompt rollout experiment for a candidate."""
        ...

    async def analyze_prompt_rollout_experiment_async(
        self,
        *,
        section_name: str,
        provider: str,
        treatment_hash: str,
    ) -> Optional[Dict[str, Any]]:
        """Async version of analyze_prompt_rollout_experiment."""
        ...

    def apply_prompt_rollout_recommendation(
        self,
        *,
        section_name: str,
        provider: str,
        treatment_hash: str,
        dry_run: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Apply the recommended rollout/rollback action for a prompt candidate."""
        ...

    async def apply_prompt_rollout_recommendation_async(
        self,
        *,
        section_name: str,
        provider: str,
        treatment_hash: str,
        dry_run: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Async version of apply_prompt_rollout_recommendation."""
        ...

    def process_prompt_candidate_evaluation_suite(
        self,
        suite: Any,
        *,
        min_pass_rate: float = 0.5,
        promote_best: bool = False,
        create_rollout: bool = False,
        rollout_control_hash: Optional[str] = None,
        rollout_traffic_split: float = 0.1,
        rollout_min_samples_per_variant: int = 100,
        analyze_rollout: bool = False,
        apply_rollout_decision: bool = False,
        rollout_decision_dry_run: bool = False,
    ) -> Any:
        """Process a prompt-candidate benchmark suite through rollout stages."""
        ...

    async def process_prompt_candidate_evaluation_suite_async(
        self,
        suite: Any,
        *,
        min_pass_rate: float = 0.5,
        promote_best: bool = False,
        create_rollout: bool = False,
        rollout_control_hash: Optional[str] = None,
        rollout_traffic_split: float = 0.1,
        rollout_min_samples_per_variant: int = 100,
        analyze_rollout: bool = False,
        apply_rollout_decision: bool = False,
        rollout_decision_dry_run: bool = False,
    ) -> Any:
        """Async version of process_prompt_candidate_evaluation_suite."""
        ...

    def close(self) -> None:
        """Close database connection."""
        ...


__all__ = [
    "ChunkRuntimeProtocol",
    "CoordinationAdvisorRuntimeProtocol",
    "PromptRuntimeProtocol",
    "RLLearningRuntimeProtocol",
    "StateRuntimeProtocol",
    "StreamingRecoveryRuntimeProtocol",
    "TaskRuntimeProtocol",
    "ToolPlanningRuntimeProtocol",
]
