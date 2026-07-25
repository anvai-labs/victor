# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
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

"""Streaming chat context dataclass.

Contains all state needed for a streaming chat session, enabling
clean dependency injection and easier testing.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from victor.core.loop_thresholds import DEFAULT_BLOCKED_CONSECUTIVE_THRESHOLD
from victor.agent.stream_handler import StreamMetrics
from victor.agent.unified_classifier import ClassifierTaskType


@dataclass
class StreamingChatContext:
    """Context holding all state for a streaming chat session.

    This dataclass encapsulates all the mutable state that was previously
    scattered across local variables in stream_chat. By centralizing state,
    we enable:
    - Easier testing with controlled initial states
    - Clear documentation of what state exists
    - Simpler parameter passing between methods
    """

    # User input
    user_message: str

    # Timing
    start_time: float = field(default_factory=time.time)
    last_activity_time: float = field(default_factory=time.time)

    # Metrics
    stream_metrics: StreamMetrics = field(default_factory=StreamMetrics)
    # Rough content-length/4 token ESTIMATE, used only for the live "~N tokens (est.)"
    # streaming readout and confidence monitoring. NOT the billed token count — the
    # authoritative per-turn counts are api_prompt/completion/total_tokens on the
    # TaskExecutionReport, sourced from the provider's reported usage. See cumulative_usage
    # below for the real provider numbers accumulated during the stream.
    estimated_content_tokens: float = 0.0
    cumulative_usage: Dict[str, int] = field(
        default_factory=lambda: {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        }
    )
    # Aggregated Sandhi transport diagnostics (metadata["sandhi_usage"] on each
    # provider response): retry attempts, usage completeness, upstream request
    # ids. Accumulated across the turn's provider calls via
    # record_provider_diagnostics(); surfaced on the stream_completed record.
    provider_diagnostics: Dict[str, Any] = field(default_factory=dict)

    # Iteration control
    max_total_iterations: int = 30
    max_exploration_iterations: int = 10
    total_iterations: int = 0
    force_completion: bool = False
    force_completion_warning_shown: bool = False  # Prevent duplicate warnings
    skip_continuation: bool = (
        False  # Skip continuation strategy when completion is forced (agentic loop fix)
    )

    # Task classification
    unified_task_type: ClassifierTaskType = ClassifierTaskType.DEFAULT
    task_classification: Optional[Dict[str, Any]] = None
    complexity_tool_budget: Optional[int] = None
    is_analysis_task: bool = False
    is_action_task: bool = False
    is_complex_task: bool = False  # GAP-16: Track COMPLEX complexity for lenient progress checking
    needs_execution: bool = False
    is_qa_task: bool = False  # Pure Q&A task — skip tools entirely
    coarse_task_type: str = "default"

    # Content accumulation
    total_accumulated_chars: int = 0
    context_msg: str = ""

    # Goals for tool selection
    goals: List[str] = field(default_factory=list)
    planned_tools: Optional[List[Any]] = None

    # Quality tracking
    last_quality_score: float = 0.5

    # Blocked attempts tracking
    consecutive_blocked_attempts: int = 0
    max_blocked_before_force: int = DEFAULT_BLOCKED_CONSECUTIVE_THRESHOLD

    # Garbage detection
    garbage_detected: bool = False

    # Continuation tracking
    continuation_prompts: int = 0
    asking_input_prompts: int = 0

    # Tool execution tracking
    tool_calls: Optional[List[Dict[str, Any]]] = None
    full_content: str = ""
    mentioned_tools_detected: List[str] = field(default_factory=list)

    # Thinking/reasoning content accumulation
    reasoning_buffer: str = (
        ""  # Accumulate LLM reasoning/thinking content separately from regular content
    )

    # Recovery tracking
    consecutive_empty_responses: int = 0
    total_blocked_attempts: int = 0
    force_tool_execution_attempts: int = 0

    # Grounding feedback for retry
    pending_grounding_feedback: str = ""
    """Actionable feedback to inject when grounding fails and retry is allowed."""

    # Substantial content threshold for recovery decisions
    substantial_content_threshold: int = 500

    # Pre-execution intent log (LogAct-inspired)
    intent_log: List[Dict[str, Any]] = field(default_factory=list)
    task_intent: str = ""
    plan_steps: List[str] = field(default_factory=list)

    # Tool execution tracking (for budget and progress checks)
    tool_budget: int = 200  # Default tool budget
    tool_calls_used: int = 0
    unique_resources: Set[str] = field(default_factory=set)
    executed_tool_names: Set[str] = field(default_factory=set)
    budget_warning_shown: bool = False
    budget_relief_uses: int = 0
    # Rolling window of tool call counts per iteration (last 5 iterations).
    # Used by budget-aware loop detection to distinguish productive agents
    # from spinning ones — see continuation_strategy.py is_productive check.
    recent_iteration_tool_counts: List[int] = field(default_factory=list)

    # Compaction tracking (P0 fix for post-compaction continuation)
    compaction_occurred: bool = False
    compaction_summary: str = ""
    last_compaction_turn: int = -1
    compaction_message_removed_count: int = 0
    last_compaction_strategy: str = ""
    last_compaction_reason: str = ""
    last_compaction_policy_reason: str = ""

    # Topology/runtime override tracking
    runtime_context_overrides: Dict[str, Any] = field(default_factory=dict)
    provider_kwargs: Dict[str, Any] = field(default_factory=dict)
    topology_input: Optional[Dict[str, Any]] = None
    topology_decision: Optional[Dict[str, Any]] = None
    topology_plan: Optional[Dict[str, Any]] = None
    topology_preparation: Optional[Dict[str, Any]] = None
    structured_routing_policy: Optional[Dict[str, Any]] = None
    topology_events: List[Dict[str, Any]] = field(default_factory=list)
    degradation_events: List[Dict[str, Any]] = field(default_factory=list)
    recovery_events: List[Dict[str, Any]] = field(default_factory=list)
    provider_status_events: List[Dict[str, Any]] = field(default_factory=list)
    runtime_override_snapshot: Optional[Dict[str, Any]] = None
    degraded_resume_state: bool = False
    resume_summary: str = ""
    resume_recent_resources: List[str] = field(default_factory=list)
    resume_recent_tools: List[str] = field(default_factory=list)

    def elapsed_time(self) -> float:
        """Get idle time since last activity (provider response or tool execution)."""
        return time.time() - self.last_activity_time

    def total_session_time(self) -> float:
        """Get total elapsed time since session start."""
        return time.time() - self.start_time

    def record_provider_diagnostics(self, sandhi_diag: Dict[str, Any]) -> None:
        """Fold one provider call's ``sandhi_usage`` diagnostics into the turn.

        Aggregation: total attempts and provider-call count sum; incomplete
        usage reports are counted (a non-"complete" completeness means the
        billed token numbers may be partial); the last outcome and upstream
        request id win (most recent call is the one a debugger wants).
        """
        diag = self.provider_diagnostics
        diag["provider_calls"] = diag.get("provider_calls", 0) + 1
        try:
            attempts = int(sandhi_diag.get("attempts") or 1)
        except (TypeError, ValueError):
            attempts = 1
        diag["attempts_total"] = diag.get("attempts_total", 0) + attempts
        completeness = sandhi_diag.get("completeness")
        if completeness and str(completeness).lower() != "complete":
            diag["incomplete_usage_count"] = diag.get("incomplete_usage_count", 0) + 1
        if sandhi_diag.get("outcome"):
            diag["last_outcome"] = sandhi_diag["outcome"]
        if sandhi_diag.get("upstream_request_id"):
            diag["last_upstream_request_id"] = sandhi_diag["upstream_request_id"]

    def reset_activity_timer(self) -> None:
        """Reset the activity timer to current time.

        Should be called whenever the provider/model responds or a tool executes,
        to prevent idle timeout during active work.
        """
        self.last_activity_time = time.time()

    def increment_iteration(self) -> int:
        """Increment and return the iteration count."""
        self.total_iterations += 1
        return self.total_iterations

    def record_iteration_tool_count(self, count: int) -> None:
        """Append per-iteration tool call count to the rolling window (size 5)."""
        try:
            self.recent_iteration_tool_counts.append(int(count))
        except (TypeError, ValueError):
            self.recent_iteration_tool_counts.append(0)
        # Keep only the last 5 iterations
        if len(self.recent_iteration_tool_counts) > 5:
            self.recent_iteration_tool_counts = self.recent_iteration_tool_counts[-5:]

    @property
    def recent_tool_call_count(self) -> int:
        """Sum of tool calls across the last 5 iterations.

        Returns 0 if no iterations have been recorded yet. Used by
        budget-aware loop detection to decide if the agent is still productive.
        """
        return sum(self.recent_iteration_tool_counts)

    def is_over_time_limit(self, limit_seconds: float) -> bool:
        """Check if session has exceeded idle time limit.

        This checks time since last activity (provider response or tool execution),
        not total session time. This prevents timeout during active work.
        """
        return self.elapsed_time() > limit_seconds

    def is_over_iteration_limit(self) -> bool:
        """Check if we've exceeded max iterations."""
        return self.total_iterations >= self.max_total_iterations

    def should_force_completion(self) -> bool:
        """Check if we should force completion based on various conditions."""
        return (
            self.force_completion
            or self.consecutive_blocked_attempts >= self.max_blocked_before_force
        )

    def record_blocked_attempt(self) -> bool:
        """Record a blocked tool attempt and return True if force threshold reached."""
        self.consecutive_blocked_attempts += 1
        return self.consecutive_blocked_attempts >= self.max_blocked_before_force

    def reset_blocked_attempts(self) -> None:
        """Reset the blocked attempts counter."""
        self.consecutive_blocked_attempts = 0

    def accumulate_content(self, content: str) -> None:
        """Add content to accumulated total."""
        self.total_accumulated_chars += len(content)

    def update_context_message(self, content: str) -> None:
        """Update the context message for next iteration.

        Deduplicates content to prevent the LLM feedback loop: if the new
        content is largely a repeat of the current context_msg, we keep the
        existing context_msg unchanged. This prevents the model from seeing
        its own repeated output and continuing/regenerating it.

        The strategy:
        - If content is empty, fall back to user_message
        - If >60% of words in new content already appear in context_msg,
          strip the overlapping portion and only append genuinely new text
        - Otherwise, replace context_msg with the new content
        """
        if not content:
            self.context_msg = self.user_message
            return

        # If context_msg is still the original user_message, always update
        if self.context_msg == self.user_message:
            self.context_msg = content
            return

        # Compute word-level overlap between new content and current context_msg
        new_words = content.split()
        existing_words = set(self.context_msg.split())

        if not existing_words or not new_words:
            self.context_msg = content
            return

        overlap_count = sum(1 for w in new_words if w in existing_words)
        overlap_ratio = overlap_count / len(new_words) if new_words else 0

        if overlap_ratio > 0.6:
            # High overlap — extract only the genuinely new portion
            # Find the suffix of content that doesn't overlap with context_msg
            for split_point in range(min(len(new_words), 20), 0, -1):
                candidate = " ".join(new_words[:split_point])
                if candidate in self.context_msg:
                    # Everything after this split point is new
                    remaining = " ".join(new_words[split_point:])
                    if remaining.strip():
                        self.context_msg = self.context_msg + " " + remaining
                    # else: entirely duplicate — don't update at all
                    return

            # Couldn't find a clean split — don't update (break the feedback loop)
            return
        else:
            # Low overlap — genuinely new content, safe to replace
            self.context_msg = content

    def record_empty_response(self) -> bool:
        """Record an empty response and return True if threshold exceeded."""
        self.consecutive_empty_responses += 1
        return self.consecutive_empty_responses >= 3  # Default threshold

    def reset_empty_responses(self) -> None:
        """Reset the empty response counter."""
        self.consecutive_empty_responses = 0

    def has_substantial_content(self) -> bool:
        """Check if we've accumulated substantial content."""
        return self.total_accumulated_chars >= self.substantial_content_threshold

    def record_tool_blocked(self) -> None:
        """Record a blocked tool attempt."""
        self.total_blocked_attempts += 1

    def record_force_tool_attempt(self) -> int:
        """Record a forced tool execution attempt and return count."""
        self.force_tool_execution_attempts += 1
        return self.force_tool_execution_attempts

    def reset_force_tool_attempts(self) -> None:
        """Reset the forced tool execution counter."""
        self.force_tool_execution_attempts = 0

    def get_remaining_budget(self) -> int:
        """Get remaining tool budget."""
        return max(0, self.tool_budget - self.tool_calls_used)

    def is_budget_exhausted(self) -> bool:
        """Check if tool budget is exhausted."""
        return self.get_remaining_budget() <= 0

    def is_approaching_budget_limit(
        self,
        warning_threshold: int = 250,
        warning_pct: float = 0.8,
        warning_remaining: int = 5,
    ) -> bool:
        """Check if approaching budget limit."""
        remaining = self.get_remaining_budget()
        if remaining <= 0 or self.tool_budget <= 0:
            return False

        dynamic_remaining = min(max(1, self.tool_budget // 5), max(1, warning_remaining))
        over_absolute_threshold = self.tool_calls_used >= max(1, warning_threshold)
        over_ratio_threshold = (self.tool_calls_used / self.tool_budget) >= warning_pct
        low_remaining = remaining <= dynamic_remaining
        return over_absolute_threshold or over_ratio_threshold or low_remaining

    def record_tool_execution(self, count: int = 1) -> None:
        """Record tool calls used."""
        self.tool_calls_used += count

    def record_executed_tool_name(self, tool_name: str) -> None:
        """Track a successfully issued tool name for completion policy checks."""
        if tool_name:
            self.executed_tool_names.add(tool_name)

    def add_unique_resource(self, resource: str) -> None:
        """Track a unique resource accessed by tools."""
        self.unique_resources.add(resource)

    def check_progress(self, base_max_consecutive: int = 8) -> bool:
        """Check if progress is being made relative to tool calls.

        Returns True if progress is adequate, False if stuck.

        GAP-16: Also considers is_complex_task for lenient treatment, not just
        is_analysis_task and is_action_task. COMPLEX tasks from ComplexityClassifier
        need more exploration before forcing completion.
        """
        max_consecutive = base_max_consecutive
        if self.is_analysis_task:
            max_consecutive = 50
        elif self.is_action_task:
            max_consecutive = 30
        elif self.is_complex_task:
            # GAP-16: COMPLEX tasks get moderate leniency (between action and default)
            max_consecutive = 20

        if self.tool_calls_used < max_consecutive:
            return True  # Haven't hit limit yet

        # Check unique resources accessed
        # GAP-16: Include is_complex_task for lenient threshold calculation
        requires_lenient = self.is_analysis_task or self.is_action_task or self.is_complex_task
        threshold = self.tool_calls_used // 4 if requires_lenient else self.tool_calls_used // 2
        return len(self.unique_resources) >= threshold

    def update_quality_score(self, score: float) -> None:
        """Update the last quality score."""
        self.last_quality_score = score

    @staticmethod
    def _normalize_ledger_text(value: Any, limit: int = 220) -> str:
        """Normalize free-form ledger text into a compact single line."""
        if value is None:
            return ""
        text = " ".join(str(value).split()).strip()
        if len(text) <= limit:
            return text
        return text[: limit - 3].rstrip() + "..."

    def set_task_intent(self, intent: str) -> None:
        """Set the current task intent for continuation recovery."""
        normalized = self._normalize_ledger_text(intent, limit=260)
        if normalized:
            self.task_intent = normalized

    def extend_plan_steps(self, steps: List[str]) -> None:
        """Add concise plan steps to the continuation ledger."""
        for step in steps:
            normalized = self._normalize_ledger_text(step, limit=140)
            if normalized and normalized not in self.plan_steps:
                self.plan_steps.append(normalized)
        if len(self.plan_steps) > 8:
            self.plan_steps = self.plan_steps[:8]

    def record_intent_event(
        self,
        kind: str,
        summary: str,
        **metadata: Any,
    ) -> None:
        """Record a bounded continuation-ledger event for post-compaction recovery."""
        normalized_summary = self._normalize_ledger_text(summary, limit=180)
        if not normalized_summary:
            return

        event: Dict[str, Any] = {
            "timestamp": time.time(),
            "kind": str(kind or "event"),
            "summary": normalized_summary,
        }
        for key, value in metadata.items():
            if value in (None, "", [], {}, set()):
                continue
            if isinstance(value, (list, tuple, set)):
                event[key] = [self._normalize_ledger_text(item, limit=80) for item in value][:6]
            else:
                event[key] = self._normalize_ledger_text(value, limit=120)

        self.intent_log.append(event)
        if len(self.intent_log) > 20:
            self.intent_log = self.intent_log[-20:]

    def record_compaction_event(
        self,
        *,
        summary: str = "",
        messages_removed: int = 0,
        strategy: str = "",
        reason: str = "",
        policy_reason: str = "",
    ) -> None:
        """Persist compaction state and ledger metadata for recovery prompts."""
        self.compaction_occurred = True
        self.last_compaction_turn = self.total_iterations
        self.compaction_message_removed_count = max(0, int(messages_removed or 0))
        if summary:
            self.compaction_summary = self._normalize_ledger_text(summary, limit=240)
        if strategy:
            self.last_compaction_strategy = str(strategy)
        if reason:
            self.last_compaction_reason = str(reason)
        if policy_reason:
            self.last_compaction_policy_reason = str(policy_reason)

        detail = f"{self.compaction_message_removed_count} messages removed"
        if self.last_compaction_strategy:
            detail += f" via {self.last_compaction_strategy}"
        if self.last_compaction_reason:
            detail += f" ({self.last_compaction_reason})"
        self.record_intent_event(
            "compaction",
            detail,
            compaction_summary=self.compaction_summary,
            compaction_policy_reason=self.last_compaction_policy_reason,
        )

    def build_continuation_ledger(
        self,
        *,
        max_events: int = 4,
        max_plan_steps: int = 4,
        max_chars: int = 700,
    ) -> str:
        """Render a compact task/plan ledger for continuation after compaction."""
        lines: List[str] = []
        if self.task_intent:
            lines.append(f"Intent: {self.task_intent}")
        if self.plan_steps:
            lines.append("Plan: " + "; ".join(self.plan_steps[:max_plan_steps]))
        if self.resume_summary:
            lines.append("Resume: " + self._normalize_ledger_text(self.resume_summary, limit=180))
        if self.resume_recent_tools:
            lines.append("Recent tools: " + ", ".join(self.resume_recent_tools[:4]))
        if self.resume_recent_resources:
            lines.append("Recent files: " + ", ".join(self.resume_recent_resources[:4]))

        recent_events = self.intent_log[-max_events:]
        if recent_events:
            lines.append(
                "Recent actions: "
                + "; ".join(
                    self._normalize_ledger_text(event.get("summary", ""), limit=120)
                    for event in recent_events
                    if event.get("summary")
                )
            )

        if not lines:
            return ""

        rendered = "\n".join(f"- {line}" for line in lines if line.strip()).strip()
        if len(rendered) <= max_chars:
            return rendered
        return rendered[: max_chars - 3].rstrip() + "..."


@dataclass
class PreparedStreamContext:
    """Result of preparing a stream context.

    Returned by the preparation phase, containing everything needed
    to start the streaming loop.
    """

    context: StreamingChatContext
    tools: List[Dict[str, Any]]
    provider_kwargs: Dict[str, Any]


def create_stream_context(
    user_message: str,
    max_iterations: int = 30,
    max_exploration: int = 10,
    tool_budget: Optional[int] = None,
    max_blocked_before_force: Optional[int] = None,
) -> StreamingChatContext:
    """Factory function to create a StreamingChatContext.

    Args:
        user_message: The user's input message
        max_iterations: Maximum total iterations
        max_exploration: Maximum exploration iterations
        tool_budget: Optional tool budget override
        max_blocked_before_force: Optional blocked-attempt threshold override

    Returns:
        Initialized StreamingChatContext
    """
    ctx = StreamingChatContext(
        user_message=user_message,
        max_total_iterations=max_iterations,
        max_exploration_iterations=max_exploration,
        context_msg=user_message,
    )
    if tool_budget is not None:
        ctx.complexity_tool_budget = tool_budget
    if max_blocked_before_force is not None:
        ctx.max_blocked_before_force = max_blocked_before_force
    return ctx
