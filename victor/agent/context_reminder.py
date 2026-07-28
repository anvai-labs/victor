from __future__ import annotations

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

"""Context Reminder Manager for intelligent system message injection.

This module provides a scalable, flexible system for managing context reminders
that are injected into conversations. It reduces token waste by:
- Consolidating multiple reminders into single messages
- Only injecting when context has changed
- Adapting reminder frequency based on task type and provider
- Supporting provider-specific formatting

Design Principles:
- Minimal token overhead: consolidate, don't repeat
- Smart injection: only when context changes
- Provider-aware: different strategies for different LLMs
- Extensible: easy to add new reminder types
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set

if TYPE_CHECKING:
    from victor.agent.presentation import PresentationProtocol

logger = logging.getLogger(__name__)


class ReminderType(Enum):
    """Types of context reminders."""

    EVIDENCE = "evidence"  # Files read, resources accessed
    BUDGET = "budget"  # Tool call budget status
    TASK_HINT = "task_hint"  # Task type guidance
    GROUNDING = "grounding"  # Grounding rules reminder
    PROGRESS = "progress"  # Progress tracking
    COMPACTION = "compaction"  # Context was compacted — summary of lost content
    CUSTOM = "custom"  # User-defined reminders


def _declare_reminder_decisions() -> Dict[ReminderType, str]:
    """Bind each emitted reminder to the decision it informs. See REMINDER_DECISIONS."""
    return {
        ReminderType.EVIDENCE: (
            "whether a file still needs reading — files read in earlier turns can "
            "fall out of a compacted context, so this is not derivable from history"
        ),
        ReminderType.BUDGET: (
            "whether to start wrapping up — the enforced tool budget lives on the "
            "runtime side and is never visible to the model"
        ),
        ReminderType.TASK_HINT: (
            "how much exploration the task warrants, keyed on measured complexity "
            "rather than the task-type guidance already in the system prompt"
        ),
        ReminderType.COMPACTION: (
            "whether to re-read something that was summarised away — by definition "
            "the model cannot see what was removed from its own context"
        ),
        # Deliberately absent, and must stay absent:
        #   PROGRESS  — every tool_call and result is already serialised onto the
        #               wire (openai_compat.build_openai_messages), so a count
        #               restates visible information. It also undercounted.
        #   GROUNDING — GROUNDING_RULES already ships in the system prompt every
        #               turn; repeating it per-turn buys nothing.
    }


#: What decision the model makes with each reminder — the co-design contract for
#: this boundary. A signal earns its place only if it informs a decision the model
#: could not already make from what is in its context.
#:
#: Restating something already on the wire is not neutral: it costs tokens every
#: turn and creates a second copy that can disagree with the first. In session
#: sandhi-cdfbc589 the progress reminder reported "3 tools used" after eight calls
#: (it appended one name per batch while the counter advanced per call). The model
#: compared it with the tool calls in its own context, found it false, and treated
#: the mismatch as evidence of tampering — a redundant signal could only agree
#: (adding nothing) or disagree (actively misleading).
#:
#: Types absent from this mapping are not emitted. Adding one means answering:
#: what does the model do with this that it could not do without it?
REMINDER_DECISIONS: Dict["ReminderType", str] = {}


def _declare_retired_reminders() -> Dict[ReminderType, str]:
    """Types deliberately not emitted, and why. See REMINDER_DECISIONS."""
    return {
        ReminderType.PROGRESS: (
            "every tool_call and its result is already serialised onto the wire "
            "(openai_compat.build_openai_messages), so a per-turn count restates "
            "what the model can see — and in sandhi-cdfbc589 it restated it wrongly"
        ),
        ReminderType.GROUNDING: (
            "GROUNDING_RULES already ships in the system prompt on every turn; "
            "repeating it per-turn costs tokens and buys no new decision"
        ),
    }


REMINDER_DECISIONS = _declare_reminder_decisions()
RETIRED_REMINDERS = _declare_retired_reminders()


def _is_emittable(reminder_type: "ReminderType") -> bool:
    """Whether this type may reach the model.

    CUSTOM is exempt: it carries a caller-supplied formatter, so the caller owns
    the justification for it. Every built-in type must appear in
    REMINDER_DECISIONS, or it is one we deliberately retired.
    """
    if reminder_type is ReminderType.CUSTOM:
        return True
    return reminder_type in REMINDER_DECISIONS


@dataclass
class ReminderConfig:
    """Configuration for reminder injection behavior.

    Attributes:
        enabled: Whether this reminder type is enabled
        frequency: How often to inject (1 = every time, 3 = every 3rd tool call)
        priority: Higher priority reminders are included first (0-100)
        max_tokens: Maximum tokens for this reminder (0 = unlimited)
        provider_overrides: Provider-specific frequency overrides
    """

    enabled: bool = True
    frequency: int = 1  # Inject every N tool calls
    priority: int = 50  # Higher = more important
    max_tokens: int = 0  # 0 = no limit
    provider_overrides: Dict[str, int] = field(default_factory=dict)

    def get_frequency_for_provider(self, provider: str) -> int:
        """Get frequency for a specific provider."""
        return self.provider_overrides.get(provider.lower(), self.frequency)


@dataclass
class ContextState:
    """Current context state for tracking changes.

    Attributes:
        observed_files: Files that have been read
        executed_tools: Tools that have been executed
        tool_calls_made: Total tool calls in this turn
        tool_budget: Maximum tool calls allowed, or None if the runtime has not
            supplied one yet (budget reminders stay silent while it is None)
        task_complexity: Current task complexity level
        last_reminder_at: Tool call count at last reminder
        reminder_history: Hash of last reminder content to detect changes
    """

    observed_files: Set[str] = field(default_factory=set)
    executed_tools: List[str] = field(default_factory=list)
    tool_calls_made: int = 0
    # None means "not yet told by the runtime". Budget reminders are suppressed
    # entirely in that state — never invent a limit and count down against it.
    tool_budget: Optional[int] = None
    task_complexity: str = "medium"
    task_hint: str = ""
    last_reminder_at: int = 0
    reminder_history: Dict[ReminderType, str] = field(default_factory=dict)
    compaction_summary: str = ""  # Summary of compacted context, set after compaction


class ContextReminderManager:
    """Manages intelligent context reminder injection.

    This manager consolidates system messages to reduce token waste while
    maintaining model grounding. It tracks context state and only injects
    reminders when they provide new information.

    Example:
        manager = ContextReminderManager(provider="google")
        manager.update_state(observed_files={"main.py"}, tool_calls=1)

        # Only returns reminder if context has changed
        reminder = manager.get_consolidated_reminder()
        if reminder:
            conversation.add_system_message(reminder)
    """

    # Default configurations per reminder type
    DEFAULT_CONFIGS: Dict[ReminderType, ReminderConfig] = {
        ReminderType.EVIDENCE: ReminderConfig(
            enabled=True,
            frequency=3,  # Every 3rd tool call
            priority=80,
            provider_overrides={
                "google": 4,  # Google handles context well
                "anthropic": 4,
                "openai": 3,
                "ollama": 2,  # Local models need more frequent reminders
                "lmstudio": 2,
            },
        ),
        ReminderType.BUDGET: ReminderConfig(
            enabled=True,
            frequency=5,  # Only when running low
            priority=90,  # High priority when budget is low
            provider_overrides={
                "google": 6,
                "anthropic": 6,
                "ollama": 4,
            },
        ),
        ReminderType.TASK_HINT: ReminderConfig(
            enabled=True,
            frequency=1,  # Only inject once at start
            priority=70,
        ),
        ReminderType.GROUNDING: ReminderConfig(
            enabled=True,
            frequency=10,  # Rarely needed after initial prompt
            priority=60,
        ),
        ReminderType.PROGRESS: ReminderConfig(
            enabled=True,
            frequency=5,
            priority=40,
        ),
        ReminderType.COMPACTION: ReminderConfig(
            enabled=True,
            frequency=1,  # Always show after compaction
            priority=95,  # High priority — agent needs to know what was lost
        ),
    }

    def __init__(
        self,
        provider: str = "unknown",
        configs: Optional[Dict[ReminderType, ReminderConfig]] = None,
        custom_formatters: Optional[Dict[ReminderType, Callable[[ContextState], str]]] = None,
        presentation: Optional["PresentationProtocol"] = None,
    ):
        """Initialize the context reminder manager.

        Args:
            provider: The LLM provider name (affects reminder frequency)
            configs: Custom reminder configurations
            custom_formatters: Custom formatter functions per reminder type
            presentation: Optional presentation adapter for icons (creates default if None)
        """
        self.provider = str(provider).lower()
        self.configs = configs or self.DEFAULT_CONFIGS.copy()
        self.custom_formatters = custom_formatters or {}
        self.state = ContextState()
        self._reminder_count = 0
        # Lazy init for backward compatibility
        if presentation is None:
            from victor.agent.presentation import create_presentation_adapter

            self._presentation = create_presentation_adapter()
        else:
            self._presentation = presentation

    def reset(self) -> None:
        """Reset state for a new conversation turn."""
        self.state = ContextState()
        self._reminder_count = 0

    def update_state(
        self,
        observed_files: Optional[Set[str]] = None,
        executed_tool: Optional[str] = None,
        tool_calls: Optional[int] = None,
        tool_budget: Optional[int] = None,
        task_complexity: Optional[str] = None,
        task_hint: Optional[str] = None,
        executed_tools: Optional[List[str]] = None,
    ) -> None:
        """Update the current context state.

        Args:
            observed_files: Files that have been read (replaces existing)
            executed_tool: Tool that was just executed (appends)
            tool_calls: Current tool call count
            tool_budget: Maximum tool calls allowed
            task_complexity: Current task complexity level
            task_hint: Task type hint to inject
            executed_tools: Every tool executed in this batch (appends all). Prefer
                this over ``executed_tool`` when a turn executes tools in parallel —
                the progress reminder reports ``len(executed_tools)``, so appending
                one name per batch understates the work done.
        """
        if observed_files is not None:
            self.state.observed_files = observed_files
        if executed_tools:
            self.state.executed_tools.extend(executed_tools)
        elif executed_tool:
            self.state.executed_tools.append(executed_tool)
        if tool_calls is not None:
            self.state.tool_calls_made = tool_calls
        if tool_budget is not None:
            self.state.tool_budget = tool_budget
        if task_complexity is not None:
            self.state.task_complexity = task_complexity
        if task_hint is not None:
            self.state.task_hint = task_hint

    def add_observed_file(self, file_path: str) -> None:
        """Add a file to the observed files set.

        Args:
            file_path: Path of the file that was read
        """
        self.state.observed_files.add(file_path)

    def should_inject_reminder(self, reminder_type: ReminderType) -> bool:
        """Check if a reminder should be injected now.

        Args:
            reminder_type: The type of reminder to check

        Returns:
            True if the reminder should be injected
        """
        # A type with no declared consumer decision is not emitted. This is the
        # gate, not the config: disabling by config alone would leave the
        # formatter reachable and let the signal creep back in via `force=True`.
        if not _is_emittable(reminder_type):
            return False

        config = self.configs.get(reminder_type)
        if not config or not config.enabled:
            return False

        frequency = config.get_frequency_for_provider(self.provider)

        # Special case: task hint only injected once
        if reminder_type == ReminderType.TASK_HINT:
            return (
                self.state.task_hint and ReminderType.TASK_HINT not in self.state.reminder_history
            )

        # Special case: budget reminder only when running low — and only when the
        # runtime has actually told us the budget. Guessing one and counting down
        # against it tells the model something it can measure as false.
        if reminder_type == ReminderType.BUDGET:
            if self.state.tool_budget is None:
                return False
            remaining = self.state.tool_budget - self.state.tool_calls_made
            return remaining <= 5 and remaining > 0

        # Check frequency
        calls_since_last = self.state.tool_calls_made - self.state.last_reminder_at
        return calls_since_last >= frequency

    def _format_evidence_reminder(self) -> str:
        """Format the evidence/files reminder."""
        if self.state.observed_files:
            files = sorted(self.state.observed_files)
            # Truncate if too many files
            if len(files) > 10:
                files = files[:10] + [f"... and {len(files) - 10} more"]
            return f"[FILES: {', '.join(files)}]"
        # Phrase as an actionable status, not a bare flag. Small models read a
        # terse "[NO FILES READ]" as a prohibition ("I am not allowed to read
        # files") and refuse, instead of as "you have not read any files yet".
        # Stay tool-name-agnostic: the tool surface varies by config/vertical
        # (discrete read/ls vs domain-based `fs ls`/`fs cat`), so naming specific
        # tools risks pointing at one that is not registered.
        return "[No files read yet — gather evidence with the available tools before answering.]"

    def _format_budget_reminder(self) -> str:
        """Format the budget reminder (empty when no budget has been supplied)."""
        if self.state.tool_budget is None:
            return ""
        remaining = self.state.tool_budget - self.state.tool_calls_made
        if remaining <= 3:
            warning_icon = self._presentation.icon("warning", with_color=False)
            return f"[{warning_icon} {remaining} tool calls remaining - wrap up soon]"
        elif remaining <= 5:
            return f"[{remaining} tool calls remaining]"
        return ""

    def _format_task_hint(self) -> str:
        """Format the task hint reminder."""
        if self.state.task_hint:
            return self.state.task_hint.strip()
        return ""

    def _format_progress_reminder(self) -> str:
        """Format the progress reminder."""
        if self.state.executed_tools:
            recent = self.state.executed_tools[-3:]  # Last 3 tools
            return f"[Progress: {len(self.state.executed_tools)} tools used, recent: {', '.join(recent)}]"
        return ""

    def _format_grounding_reminder(self) -> str:
        """Format the grounding rules reminder."""
        return "[Ground responses in tool output only. Do not invent file paths or content.]"

    def _format_compaction_reminder(self) -> str:
        """Format the compaction context reminder.

        Injected after context was compacted to inform the agent what context
        was lost and what key information was preserved.
        """
        if not self.state.compaction_summary:
            return ""
        return (
            f"[Context compacted: {self.state.compaction_summary}. "
            "Earlier conversation details were summarized. "
            "Use tools to re-read files if you need specific content.]"
        )

    def get_reminder(self, reminder_type: ReminderType) -> Optional[str]:
        """Get a specific reminder if it should be injected.

        Args:
            reminder_type: The type of reminder to get

        Returns:
            The formatted reminder string, or None if not needed
        """
        if not _is_emittable(reminder_type):
            return None
        if not self.should_inject_reminder(reminder_type):
            return None

        # Use custom formatter if provided
        if reminder_type in self.custom_formatters:
            content = self.custom_formatters[reminder_type](self.state)
        else:
            # Use built-in formatters
            formatters = {
                ReminderType.EVIDENCE: self._format_evidence_reminder,
                ReminderType.BUDGET: self._format_budget_reminder,
                ReminderType.TASK_HINT: self._format_task_hint,
                ReminderType.PROGRESS: self._format_progress_reminder,
                ReminderType.GROUNDING: self._format_grounding_reminder,
                ReminderType.COMPACTION: self._format_compaction_reminder,
            }
            formatter = formatters.get(reminder_type)
            content = formatter() if formatter else ""

        if not content:
            return None

        # Check if content has changed
        prev_content = self.state.reminder_history.get(reminder_type, "")
        if content == prev_content and reminder_type != ReminderType.BUDGET:
            return None  # No change, don't repeat

        # Update history
        self.state.reminder_history[reminder_type] = content
        return content

    def get_consolidated_reminder(self, force: bool = False) -> Optional[str]:
        """Get a consolidated reminder combining all active reminders.

        This is the main method to call after each tool execution. It returns
        a single, consolidated system message that includes all relevant
        context reminders while minimizing redundancy.

        Args:
            force: Force injection even if conditions aren't met

        Returns:
            Consolidated reminder string, or None if no reminder needed
        """
        parts = []

        # Collect reminders by priority
        reminder_items: List[tuple[int, str]] = []

        for reminder_type in ReminderType:
            if reminder_type == ReminderType.CUSTOM:
                continue

            if not _is_emittable(reminder_type):
                continue
            if force or self.should_inject_reminder(reminder_type):
                content = self.get_reminder(reminder_type)
                if content:
                    config = self.configs.get(reminder_type, ReminderConfig())
                    reminder_items.append((config.priority, content))

        if not reminder_items:
            return None

        # Sort by priority (highest first) and combine
        reminder_items.sort(key=lambda x: x[0], reverse=True)
        parts = [item[1] for item in reminder_items]

        # Update tracking
        self.state.last_reminder_at = self.state.tool_calls_made
        self._reminder_count += 1

        # Combine into single message
        return " | ".join(parts)

    def get_user_message_prefix(self, force: bool = False) -> str:
        """Get reminder text formatted for user message injection.

        Returns formatted prefix string for prepending to user messages.
        Used when cache_optimization is enabled to keep system messages
        byte-stable (enabling provider prefix caching at 90% discount).

        Args:
            force: Force injection even if conditions aren't met

        Returns:
            Formatted prefix string, or empty string if no reminders needed
        """
        reminder = self.get_consolidated_reminder(force=force)
        if not reminder:
            return ""
        return f"[Context: {reminder}]\n\n"

    def get_minimal_reminder(self) -> Optional[str]:
        """Get a minimal reminder for token-constrained situations.

        Returns only the most critical information:
        - Budget warning if low
        - File count if any files read

        Returns:
            Minimal reminder string, or None if nothing critical
        """
        parts = []

        # Budget warning is critical — but only when a budget is actually known.
        if self.state.tool_budget is not None:
            remaining = self.state.tool_budget - self.state.tool_calls_made
            if remaining <= 3:
                warning_icon = self._presentation.icon("warning", with_color=False)
                parts.append(f"{warning_icon} {remaining} calls left")

        # File count
        if self.state.observed_files:
            parts.append(f"{len(self.state.observed_files)} files read")

        return " | ".join(parts) if parts else None

    def configure_for_provider(self, provider: str) -> None:
        """Reconfigure manager for a specific provider.

        Args:
            provider: The provider name
        """
        self.provider = str(provider).lower()

        # Adjust configs based on provider characteristics
        if provider in {"google", "anthropic", "openai"}:
            # Cloud providers handle context well - less frequent reminders
            self.configs[ReminderType.EVIDENCE].frequency = 4
            self.configs[ReminderType.GROUNDING].frequency = 15
        elif provider in {"ollama", "lmstudio"}:
            # Local models need more frequent grounding
            self.configs[ReminderType.EVIDENCE].frequency = 2
            self.configs[ReminderType.GROUNDING].frequency = 5

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about reminder injection.

        Returns:
            Dictionary with reminder statistics
        """
        return {
            "total_reminders": self._reminder_count,
            "tool_calls_made": self.state.tool_calls_made,
            "files_observed": len(self.state.observed_files),
            "tools_executed": len(self.state.executed_tools),
            "provider": self.provider,
        }


# Convenience functions for common use cases


def create_reminder_manager(
    provider: str,
    task_complexity: str = "medium",
    tool_budget: Optional[int] = None,
) -> ContextReminderManager:
    """Create a configured reminder manager for a task.

    Args:
        provider: The LLM provider name
        task_complexity: Task complexity level
        tool_budget: Maximum tool calls allowed. Leave as None at construction time
            and let the runtime supply the enforced budget via ``update_state`` —
            a hardcoded default here is stated to the model as fact.

    Returns:
        Configured ContextReminderManager instance
    """
    manager = ContextReminderManager(provider=provider)
    manager.update_state(
        task_complexity=task_complexity,
        tool_budget=tool_budget,
    )
    return manager


def get_evidence_reminder(files: Set[str], provider: str = "unknown") -> str:
    """Quick helper to format an evidence reminder.

    Args:
        files: Set of file paths that have been read
        provider: Provider name for formatting

    Returns:
        Formatted evidence reminder string
    """
    manager = ContextReminderManager(provider=provider)
    manager.update_state(observed_files=files)
    return manager._format_evidence_reminder()
