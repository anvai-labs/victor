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


from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, TYPE_CHECKING

from victor.agent.message_history import MessageHistory
from victor.agent.conversation.state_machine import (
    ConversationStateMachine,
    ConversationStage,
)
from victor.agent.prompt_normalizer import PromptNormalizer, get_prompt_normalizer
from victor.config.orchestrator_constants import (
    COMPACTION_CONFIG,
    CONTEXT_LIMITS,
    SEMANTIC_THRESHOLDS,
    TASK_COMPACTION_CONFIGS,
)
from victor.core.completion_markers import strip_active_completion_markers
from victor.providers.base import Message

if TYPE_CHECKING:
    from victor.storage.embeddings.service import EmbeddingService
    from victor.agent.conversation.store import ConversationStore
    from victor.agent.compaction_summarizer import CompactionSummaryStrategy
    from victor.agent.context_reminder import ContextReminderManager
    from victor.agent.compaction_hierarchy import HierarchicalCompactionManager


class CompactionStrategy(Enum):
    SIMPLE = "simple"
    TIERED = "tiered"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"


@dataclass
class MessageImportance:
    message: Message
    index: int
    score: float
    reason: str = ""


logger = logging.getLogger(__name__)


# Patterns for pinned output requirements (never compacted)
# These patterns indicate user-specified deliverable format requirements
# that must survive compaction to prevent the agent from losing focus
PINNED_REQUIREMENT_PATTERNS = [
    re.compile(r"\bmust\s+output\b", re.IGNORECASE),
    re.compile(r"\brequired\s+format\b", re.IGNORECASE),
    re.compile(r"\bfindings\s+table\b", re.IGNORECASE),
    re.compile(r"\btop[-\s]?\d+\s+fixes\b", re.IGNORECASE),
    re.compile(r"\bdeliverables?\s*:", re.IGNORECASE),
    re.compile(r"\bcreate\s+a\s+findings\s+table\b", re.IGNORECASE),
    re.compile(r"\bprovide\s+top[-\s]?\d+\b", re.IGNORECASE),
    re.compile(r"\boutput\s+must\s+include\b", re.IGNORECASE),
    re.compile(r"\brequired\s+outputs?\s*:", re.IGNORECASE),
    re.compile(r"\b\d+[-–]\d+\s+findings\b", re.IGNORECASE),  # "6-10 findings"
    re.compile(r"\bprioritize\s+\d+[-–]\d+\s+findings\b", re.IGNORECASE),
]


def _is_pinned_content(content: str) -> bool:
    """Check if content contains pinned output requirement patterns.

    Args:
        content: Message content to check

    Returns:
        True if content contains pinned requirement patterns
    """
    for pattern in PINNED_REQUIREMENT_PATTERNS:
        if pattern.search(content):
            return True
    return False


def _sanitize_compaction_summary(summary: str) -> str:
    """Remove runtime completion markers from stored and injected summaries."""
    if not summary:
        return ""
    return strip_active_completion_markers(summary).strip()


@dataclass
class ContextMetrics:
    char_count: int
    estimated_tokens: int
    message_count: int
    is_overflow_risk: bool = False
    max_context_chars: int = CONTEXT_LIMITS.max_context_chars

    @property
    def total_chars(self) -> int:
        """Backward compatibility alias for char_count."""
        return self.char_count

    @property
    def utilization(self) -> float:
        return min(1.0, self.char_count / self.max_context_chars) if self.max_context_chars else 0.0

    @property
    def max_tokens(self) -> int:
        return int(self.max_context_chars / 2.8)

    @property
    def remaining_tokens(self) -> int:
        return max(0, self.max_tokens - self.estimated_tokens)


@dataclass
class ConversationConfig:
    max_context_chars: int = CONTEXT_LIMITS.max_context_chars
    chars_per_token_estimate: int = CONTEXT_LIMITS.chars_per_token_estimate
    enable_stage_tracking: bool = True
    enable_context_monitoring: bool = True
    compaction_strategy: CompactionStrategy = CompactionStrategy.TIERED
    compaction_threshold: float = (
        0.85  # Phase 4: Utilization threshold for triggering compaction (85%)
    )
    min_messages_to_keep: int = 6
    tool_result_retention_weight: float = COMPACTION_CONFIG.tool_result_retention_weight
    recent_message_weight: float = COMPACTION_CONFIG.recent_message_weight
    semantic_relevance_threshold: float = SEMANTIC_THRESHOLDS.compaction_relevance

    def __post_init__(self):
        """Ensure compaction_strategy is a CompactionStrategy enum.

        Allows passing strategy as string (e.g., "hybrid") and converts to enum.
        """
        if isinstance(self.compaction_strategy, str):
            self.compaction_strategy = CompactionStrategy(self.compaction_strategy.lower())


class ConversationController:
    """Controls conversation state, history, and context management.

    Phase 4 Enhancement: Added minimum interval between compactions to prevent
    aggressive context loss. Compaction now requires both threshold overflow AND
    minimum time elapsed (MIN_COMPACTION_INTERVAL_SECONDS).
    """

    # Phase 4: Minimum interval between compactions (prevents aggressive compaction)
    MIN_COMPACTION_INTERVAL_SECONDS: float = 60.0

    def __init__(
        self,
        config: Optional[ConversationConfig] = None,
        message_history: Optional[MessageHistory] = None,
        state_machine: Optional[ConversationStateMachine] = None,
        embedding_service: Optional["EmbeddingService"] = None,
        conversation_store: Optional["ConversationStore"] = None,
        session_id: Optional[str] = None,
        prompt_normalizer: Optional[PromptNormalizer] = None,
        compaction_summarizer: Optional["CompactionSummaryStrategy"] = None,
        context_reminder_manager: Optional["ContextReminderManager"] = None,
        hierarchical_manager: Optional["HierarchicalCompactionManager"] = None,
    ):
        self.config = config or ConversationConfig()
        self._history = message_history or MessageHistory()
        self._state_machine = state_machine or ConversationStateMachine()
        self._embedding_service = embedding_service
        self._conversation_store = conversation_store
        self._session_id = session_id
        self._system_prompt: Optional[str] = None
        self._system_added = False
        self._context_callbacks: List[Callable[[ContextMetrics], None]] = []
        self._compaction_summarizer = compaction_summarizer
        self._compaction_summaries: List[str] = []
        self._current_plan: Optional[Any] = None
        self._context_reminder_manager = context_reminder_manager
        self._hierarchical_manager = hierarchical_manager
        # Phase 4: Track last compaction time for minimum interval enforcement
        self._last_compaction_time: float = 0.0
        # PromptNormalizer for input deduplication (DIP compliance)
        self._normalizer = prompt_normalizer or get_prompt_normalizer()
        # Token accounting: actual API-returned token counts replace char estimation
        self._last_actual_prompt_tokens: Optional[int] = None
        self._observed_chars_per_token: float = float(self.config.chars_per_token_estimate)

    def get_session_cost_usd(self) -> float:
        """Return the live cumulative cost (USD) for the active session.

        Reads actual API-returned token costs from the conversation store.
        Returns 0.0 when no store/session is bound or pricing is unavailable,
        so callers (e.g. the governance cost-budget policy) degrade gracefully.
        """
        if not self._conversation_store or not self._session_id:
            return 0.0
        try:
            stats = self._conversation_store.get_session_token_stats(self._session_id)
        except Exception:  # pragma: no cover - defensive: cost must not break flow
            return 0.0
        if not stats:
            return 0.0
        try:
            return float(stats.get("cost_usd", 0.0) or 0.0)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return 0.0

    @property
    def messages(self) -> List[Message]:
        return self._history.messages

    @property
    def message_count(self) -> int:
        return len(self._history.messages)

    @property
    def stage(self) -> ConversationStage:
        return self._state_machine.get_stage()

    @property
    def system_prompt(self) -> Optional[str]:
        return self._system_prompt

    @property
    def current_plan(self) -> Optional[Any]:
        """Get the current execution plan.

        Returns:
            The current ExecutionPlan or None if no plan is active.
        """
        return self._current_plan

    def set_current_plan(self, plan: Optional[Any]) -> None:
        """Set the current execution plan.

        Args:
            plan: The ExecutionPlan to set, or None to clear.
        """
        self._current_plan = plan

    def set_system_prompt(self, prompt: str) -> None:
        self._system_prompt = prompt
        self._system_added = False

    def ensure_system_message(self) -> None:
        if self._system_added or not self._system_prompt:
            return
        if self._history.messages and self._history.messages[0].role == "system":
            self._system_added = True
            return
        self._history._messages.insert(0, Message(role="system", content=self._system_prompt))
        self._system_added = True

    def add_user_message(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Message:
        """Add a user message with optional normalization.

        Uses PromptNormalizer to:
        1. Normalize action verbs (view→read, check→read)
        2. Detect and log duplicate messages
        3. Collapse repeated continuation requests

        Args:
            content: User message content
            metadata: Optional metadata dict (e.g. {"source": "u"} for USER_TYPED)
            **kwargs: Unused extra kwargs (for forward compat)

        Returns:
            The added Message
        """
        self.ensure_system_message()

        # Normalize input using PromptNormalizer
        result = self._normalizer.normalize(content)
        normalized_content = result.normalized

        # Log normalizations if any were made
        if result.changes:
            logger.debug(f"Normalized prompt: {result.changes}")
        if result.is_duplicate:
            logger.debug("Detected duplicate message (will still add for conversational flow)")

        message = self._history.add_user_message(normalized_content, metadata=metadata)
        if self.config.enable_stage_tracking:
            self._state_machine.record_message(normalized_content, is_user=True)
        if (
            self.config.enable_context_monitoring
            and (metrics := self.get_context_metrics()).is_overflow_risk
        ):
            self._notify_context_callbacks(metrics)
        return message

    def add_assistant_message(
        self,
        content: str,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Message:
        message = self._history.add_assistant_message(
            content, tool_calls=tool_calls, metadata=metadata
        )
        if self.config.enable_stage_tracking:
            self._state_machine.record_message(content, is_user=False)
        if (
            self.config.enable_context_monitoring
            and (metrics := self.get_context_metrics()).is_overflow_risk
        ):
            self._notify_context_callbacks(metrics)
        return message

    def add_tool_result(
        self,
        tool_call_id: str,
        tool_name: str,
        result: str,
    ) -> Message:
        """Add a tool result message to history.

        Args:
            tool_call_id: ID of the tool call
            tool_name: Name of the tool
            result: Tool execution result

        Returns:
            The created message
        """
        message = self._history.add_tool_result(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            content=result,
        )
        return message

    def add_message(self, role: str, content: str, **kwargs: Any) -> Message:
        """Add a message with specified role (backward compatibility).

        Args:
            role: Message role (user, assistant, system)
            content: Message content
            **kwargs: Additional message metadata (e.g., tool_calls for assistant, metadata dict)

        Returns:
            The created message
        """
        if role == "user":
            return self.add_user_message(content, metadata=kwargs.get("metadata"))
        elif role == "assistant":
            return self.add_assistant_message(content, **kwargs)
        elif role == "system":
            self.set_system_prompt(content)
            self.ensure_system_message()
            return self.messages[0]
        else:
            # For other roles, use add_message from history
            return self._history.add_message(role, content, **kwargs)

    def record_actual_usage(self, prompt_tokens: int, total_chars: int) -> None:
        """Update token accounting with actual API-returned prompt token count.

        Replaces char-based estimation for the next get_context_metrics() call
        and calibrates the per-session chars-per-token ratio (rolling update
        clamped to [2, 8] to guard against noise from very short messages).

        Args:
            prompt_tokens: Actual prompt_tokens from the provider API response
            total_chars: Total characters in conversation at the time of the call
        """
        if prompt_tokens <= 0:
            return
        self._last_actual_prompt_tokens = prompt_tokens
        if total_chars > 0:
            observed = total_chars / prompt_tokens
            # Rolling average: 80% old, 20% new observation
            self._observed_chars_per_token = 0.8 * self._observed_chars_per_token + 0.2 * observed
            # Clamp to sane range: 2–8 chars per token
            self._observed_chars_per_token = max(2.0, min(8.0, self._observed_chars_per_token))
            logger.debug(
                "Token accounting calibrated: %.2f chars/token (prompt_tokens=%d, chars=%d)",
                self._observed_chars_per_token,
                prompt_tokens,
                total_chars,
            )

    def get_context_metrics(self) -> ContextMetrics:
        """Calculate current context metrics.

        Uses actual API-returned prompt_tokens when available (set via
        record_actual_usage()), falling back to calibrated char estimation.

        Returns:
            ContextMetrics with size and overflow information
        """
        total_chars = sum(len(m.content) for m in self.messages)

        if self._last_actual_prompt_tokens is not None:
            # Project current size using the calibrated chars-per-token ratio
            estimated_tokens = int(total_chars / self._observed_chars_per_token)
        else:
            estimated_tokens = total_chars // self.config.chars_per_token_estimate

        return ContextMetrics(
            char_count=total_chars,
            estimated_tokens=estimated_tokens,
            message_count=len(self.messages),
            is_overflow_risk=total_chars
            > self.config.max_context_chars * CONTEXT_LIMITS.overflow_threshold,
            max_context_chars=self.config.max_context_chars,
        )

    def check_context_overflow(self) -> bool:
        """Check if context is at risk of overflow.

        Returns:
            True if context is dangerously large
        """
        metrics = self.get_context_metrics()
        if metrics.is_overflow_risk:
            logger.warning(
                "Context overflow risk: %d chars (~%d tokens). Max: %d chars",
                metrics.char_count,
                metrics.estimated_tokens,
                metrics.max_context_chars,
            )
        return metrics.is_overflow_risk

    def get_stage_recommended_tools(self) -> Set[str]:
        """Get tools recommended for current conversation stage.

        Returns:
            Set of recommended tool names
        """
        return self._state_machine.get_stage_tools()

    def reset(self) -> None:
        """Reset conversation to initial state."""
        self._history.clear()
        self._state_machine.reset()
        self._system_added = False
        logger.info("Conversation reset")

    def load_history(self, history: MessageHistory) -> None:
        """Replace the live history in place (cross-visit resume).

        Repoints the controller at a previously stored session's messages so
        the turn loop — which reads this controller's ``_history`` directly —
        recalls the prior turns. The system message is re-established so the
        history stays well-formed.
        """
        self._history = history
        self._system_added = bool(history.messages) and history.messages[0].role == "system"
        self.ensure_system_message()
        logger.info("Conversation history loaded (%d messages)", len(history.messages))

    def compact_history(self, keep_recent: int = 10) -> int:
        """Compact history by removing old messages (simple strategy).

        Keeps system message and most recent messages.
        For smarter compaction, use smart_compact_history().

        Args:
            keep_recent: Number of recent messages to keep

        Returns:
            Number of messages removed
        """
        if len(self.messages) <= keep_recent + 1:  # +1 for system message
            return 0

        # Keep system message and recent messages
        system_msg = None
        if self.messages and self.messages[0].role == "system":
            system_msg = self.messages[0]

        recent = self.messages[-keep_recent:]
        removed = len(self.messages) - keep_recent - (1 if system_msg else 0)

        self._history.clear()
        if system_msg:
            self._history.append_message(system_msg)
        for msg in recent:
            self._history.append_message(msg)

        logger.info(f"Compacted history: removed {removed} messages")
        return removed

    def _should_compact(self, current_utilization: float) -> bool:
        """Check if compaction should be triggered based on threshold and minimum interval.

        Phase 4 Enhancement: Compaction now requires BOTH:
        1. Context utilization exceeds threshold
        2. Minimum time elapsed since last compaction (MIN_COMPACTION_INTERVAL_SECONDS)

        This prevents aggressive compaction that causes context loss and maintains
        conversation continuity.

        Args:
            current_utilization: Current context utilization (0.0 to 1.0)

        Returns:
            True if compaction should proceed, False otherwise
        """
        import time

        # Check threshold first (cheap check)
        if current_utilization < self.config.compaction_threshold:
            return False

        # Check minimum interval (prevents aggressive compaction)
        time_since_last = time.time() - self._last_compaction_time
        if time_since_last < self.MIN_COMPACTION_INTERVAL_SECONDS:
            logger.debug(
                f"Compaction blocked: only {time_since_last:.1f}s since last compaction "
                f"(need {self.MIN_COMPACTION_INTERVAL_SECONDS}s). "
                f"Utilization={current_utilization:.2f}, threshold={self.config.compaction_threshold:.2f}"
            )
            return False

        return True

    def smart_compact_history(
        self,
        target_messages: Optional[int] = None,
        current_query: Optional[str] = None,
        task_type: Optional[str] = None,
        force: bool = False,
    ) -> int:
        """Smart context compaction using configured strategy.

        Uses the configured compaction strategy to intelligently reduce
        context size while preserving the most important information.

        Strategies:
        - SIMPLE: Same as compact_history (keep N recent)
        - TIERED: Prioritize tool results over discussion messages
        - SEMANTIC: Use embeddings to keep task-relevant messages
        - HYBRID: Combine tiered and semantic approaches

        Args:
            target_messages: Target number of messages to keep (default: min_messages_to_keep)
            current_query: Current user query for semantic relevance scoring
            task_type: Optional task type for task-specific compaction configuration
            force: If True, bypass threshold and interval checks (useful for testing/explicit calls)

        Returns:
            Number of messages removed
        """
        # Phase 4: Check if compaction should proceed (threshold + minimum interval)
        # Skip checks if force=True (for explicit calls/testing)
        if not force:
            # Calculate current context utilization
            metrics = self.get_context_metrics()
            current_utilization = metrics.utilization
            if not self._should_compact(current_utilization):
                return 0

        import time

        self._last_compaction_time = time.time()  # Update last compaction time

        # Use task-specific config if available
        if task_type and task_type in TASK_COMPACTION_CONFIGS:
            task_config = TASK_COMPACTION_CONFIGS[task_type]
            target = target_messages or task_config.min_messages_after_compact
            logger.info(f"Using task-specific compaction config for '{task_type}'")
            pass  # task_config values used inline below via task_config.*
        else:
            target = target_messages or self.config.min_messages_to_keep

        strategy = self.config.compaction_strategy

        if len(self.messages) <= target + 1:  # +1 for system message
            return 0

        logger.info(f"Smart compacting with strategy: {strategy.value}")

        if strategy == CompactionStrategy.SIMPLE:
            return self.compact_history(keep_recent=target)

        # Normalize non-Message items IN-PLACE before compaction.
        # Prevents HTTP 400 from providers when compacted history contains
        # raw strings or dicts instead of Message objects.
        non_msg_count = sum(1 for m in self._history._messages if not isinstance(m, Message))
        if non_msg_count > 0:
            logger.warning(
                "Normalizing %d non-Message objects in history before compaction",
                non_msg_count,
            )
            normalized = []
            for m in self._history._messages:
                if isinstance(m, Message):
                    normalized.append(m)
                elif isinstance(m, dict) and "role" in m:
                    safe = {
                        k: v
                        for k, v in m.items()
                        if k in ("role", "content", "name", "tool_calls", "tool_call_id")
                    }
                    normalized.append(Message(**safe))
                elif isinstance(m, str):
                    normalized.append(Message(role="assistant", content=m))
            from victor.agent.message_history import _TrackedList

            self._history._messages = _TrackedList(normalized)

        # Score all messages for importance
        scored_messages = self._score_messages(current_query=current_query)

        # Sort by score (descending) and keep top N
        scored_messages.sort(key=lambda x: x.score, reverse=True)

        # ALWAYS keep root system message (at index 0) for caching stability
        # AND keep the most recent system message if it differs from the root.
        system_msgs_to_keep = []
        if (
            self.messages
            and hasattr(self.messages[0], "role")
            and self.messages[0].role == "system"
        ):
            system_msgs_to_keep.append(self.messages[0])
            # Remove from scored list to handle separately
            scored_messages = [sm for sm in scored_messages if sm.index != 0]

        # Find any other system messages (e.g. dynamic instructions)
        # We only want to keep the MOST RECENT one if there are many.
        other_system_msgs = [sm for sm in scored_messages if sm.message.role == "system"]
        if other_system_msgs:
            # Sort by index to find the last one
            other_system_msgs.sort(key=lambda x: x.index)
            recent_system = other_system_msgs[-1]
            # If it's different from root, we might want to keep it
            if recent_system.message.content != system_msgs_to_keep[0].content:
                system_msgs_to_keep.append(recent_system.message)

            # Remove all other system messages from the candidates
            system_indices = {sm.index for sm in other_system_msgs}
            scored_messages = [sm for sm in scored_messages if sm.index not in system_indices]

        # Keep top messages (up to target)
        messages_to_keep = scored_messages[:target]

        # Ensure balanced representation: Guaranteed Role Minimums
        # We need at least 1 USER and 1 ASSISTANT (as requested by user fix)
        # but also ensure they don't get buried if target is small.
        kept_roles = {sm.message.role for sm in messages_to_keep}
        # Build once and maintain — reused for tool-call pair preservation below.
        kept_indices = {sm.index for sm in messages_to_keep}

        # 1. Ensure USER message survives (at least one)
        if "user" not in kept_roles:
            for sm in scored_messages[target:]:
                if sm.message.role == "user":
                    messages_to_keep.append(sm)
                    kept_indices.add(sm.index)
                    break

        # 2. Protect ALL USER_TYPED messages — they are inviolable ground truth.
        # Nudges and synthetic injections may be dropped, but never actual user input.
        from victor.agent.conversation.types import (
            MESSAGE_SOURCE_METADATA_KEY,
            MessageSource,
        )

        for sm in scored_messages[target:]:
            src_raw = getattr(sm.message, "metadata", {}) or {}
            if (
                isinstance(src_raw, dict)
                and src_raw.get(MESSAGE_SOURCE_METADATA_KEY) == MessageSource.USER_TYPED.value
            ):
                if sm.index not in kept_indices:
                    messages_to_keep.append(sm)
                    kept_indices.add(sm.index)

        # 3. Ensure ASSISTANT message survives (the most recent one)
        if "assistant" not in kept_roles:
            # Look backwards from all scored messages for the latest assistant turn
            for sm in reversed(scored_messages):
                if sm.message.role == "assistant":
                    messages_to_keep.append(sm)
                    kept_indices.add(sm.index)
                    break

        # Preserve tool-call pairs: if an assistant message with tool_calls
        # is kept, ensure its corresponding tool responses are also kept.
        # Likewise, if a tool response is kept, retain the assistant message
        # that declared its tool_call_id. This keeps provider history valid
        # after compaction instead of relying on destructive request-time repair.
        # This prevents orphaned tool_calls that cause HTTP 400 on Z.AI/OpenAI.
        all_messages = self._history._messages  # Use normalized list
        tool_call_owners = {}
        for i, msg in enumerate(all_messages):
            if getattr(msg, "role", "") != "assistant" or not getattr(msg, "tool_calls", None):
                continue
            for tc in msg.tool_calls:
                tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                if tc_id:
                    tool_call_owners[tc_id] = i

        for sm in list(messages_to_keep):
            if sm.message.role == "assistant" and getattr(sm.message, "tool_calls", None):
                for tc in sm.message.tool_calls:
                    tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                    if tc_id:
                        for i, msg in enumerate(all_messages):
                            if (
                                getattr(msg, "role", "") == "tool"
                                and getattr(msg, "tool_call_id", "") == tc_id
                                and i not in kept_indices
                            ):
                                messages_to_keep.append(
                                    MessageImportance(msg, i, 999.0, "paired_tool_response")
                                )
                                kept_indices.add(i)
            elif sm.message.role == "tool" and getattr(sm.message, "tool_call_id", None):
                owner_index = tool_call_owners.get(sm.message.tool_call_id)
                if owner_index is not None and owner_index not in kept_indices:
                    owner_msg = all_messages[owner_index]
                    messages_to_keep.append(
                        MessageImportance(owner_msg, owner_index, 999.0, "paired_tool_call")
                    )
                    kept_indices.add(owner_index)

        for sm in list(messages_to_keep):
            if sm.message.role != "assistant" or not getattr(sm.message, "tool_calls", None):
                continue
            for tc in sm.message.tool_calls:
                tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                if not tc_id:
                    continue
                for i, msg in enumerate(all_messages):
                    if (
                        getattr(msg, "role", "") == "tool"
                        and getattr(msg, "tool_call_id", "") == tc_id
                        and i not in kept_indices
                    ):
                        messages_to_keep.append(
                            MessageImportance(msg, i, 999.0, "paired_tool_response")
                        )
                        kept_indices.add(i)

        # Sort by original index to maintain conversation order
        messages_to_keep.sort(key=lambda x: x.index)

        # Generate summary of removed messages
        removed_indices = {sm.index for sm in scored_messages if sm.index not in kept_indices}
        removed_messages = [m for i, m in enumerate(self.messages) if i in removed_indices]
        if removed_messages:
            summary = _sanitize_compaction_summary(
                self._generate_compaction_summary(removed_messages)
            )
            if summary:
                self._compaction_summaries.append(summary)
                # Feed into hierarchical manager if available
                if self._hierarchical_manager:
                    turn_index = len(self.messages)
                    self._hierarchical_manager.add_summary(summary, turn_index)
                # Persist to SQLite for future semantic retrieval
                message_ids = [f"msg_{i}" for i in removed_indices]
                self.persist_compaction_summary(summary, message_ids)

        # Rebuild message list
        removed_count = len(self.messages) - len(messages_to_keep) - len(system_msgs_to_keep)
        self._history.clear()

        # Add preserved system messages first (Root + Recent Dynamic)
        for msg in system_msgs_to_keep:
            self._history.append_message(msg)

        # Add conversation messages in their original chronological order
        messages_to_keep.sort(key=lambda x: x.index)
        for scored_msg in messages_to_keep:
            self._history.append_message(scored_msg.message)

        logger.info(
            f"Smart compacted: removed {removed_count} messages, "
            f"kept {len(messages_to_keep)} (strategy: {strategy.value})"
        )
        return removed_count

    def _score_messages(
        self,
        messages: Optional[List[Message]] = None,
        current_query: Optional[str] = None,
    ) -> List[MessageImportance]:
        """Score messages for importance during compaction.

        Delegates to canonical score_messages() from conversation.scoring,
        with pre-processing for system messages and pinned content.

        Args:
            messages: Messages to score (defaults to self.messages)
            current_query: Optional query for semantic relevance

        Returns:
            List of MessageImportance objects
        """
        msgs_to_score = messages if messages is not None else self.messages

        # DEBUG: trace source of msgs_to_score
        if msgs_to_score:
            import traceback as _tb

            sample_types = [type(m).__name__ for m in msgs_to_score[:5]]
            history_id = id(self._history) if hasattr(self, "_history") else "N/A"
            history_msgs_types = (
                [type(m).__name__ for m in self._history._messages[:5]]
                if hasattr(self, "_history") and hasattr(self._history, "_messages")
                else ["N/A"]
            )
            logger.info(
                "SCORE_TRACE: msgs_to_score len=%d types=%s | "
                "messages_param=%s | history_id=%s | history._messages types=%s | "
                "self.messages id=%s | history.messages id=%s",
                len(msgs_to_score),
                sample_types,
                "provided" if messages is not None else "self.messages",
                history_id,
                history_msgs_types,
                id(self.messages),
                id(self._history.messages) if hasattr(self, "_history") else "N/A",
            )

        # Pre-process: handle system messages and pinned content before canonical scoring
        scored: List[MessageImportance] = []
        scorable_msgs: List[Message] = []
        scorable_indices: List[int] = []

        # Defensive: convert non-Message objects (dicts, strings) to Message objects.
        # This can happen when session restore or external callers inject dicts
        # into the history, or after serialization round-trips.
        non_message_count = sum(1 for m in msgs_to_score if not isinstance(m, Message))
        if non_message_count > 0:
            # Log type info for debugging — helps identify where non-Message items come from
            type_counts: dict = {}
            for m in msgs_to_score:
                t = type(m).__name__
                type_counts[t] = type_counts.get(t, 0) + 1
            logger.info(
                "Normalizing %d non-Message items in history (%d total), types: %s",
                non_message_count,
                len(msgs_to_score),
                type_counts,
            )
            converted = []
            for m in msgs_to_score:
                if isinstance(m, Message):
                    converted.append(m)
                elif isinstance(m, dict) and "role" in m:
                    try:
                        converted.append(Message(**m))
                    except Exception:
                        converted.append(
                            Message(
                                role=m.get("role", "assistant"),
                                content=str(m.get("content", "")),
                            )
                        )
                elif isinstance(m, str):
                    converted.append(Message(role="assistant", content=m))
                else:
                    converted.append(Message(role="assistant", content=str(m)))
            msgs_to_score = converted

        for i, msg in enumerate(msgs_to_score):
            if msg.role == "system":
                scored.append(MessageImportance(msg, i, 1000.0, "system"))
            elif _is_pinned_content(msg.content):
                scored.append(MessageImportance(msg, i, 999.0, "pinned_requirement"))
                logger.debug(f"Message {i} marked as pinned (output requirement)")
            # P0 FIX: Always preserve the FIRST (original) user message - highest priority
            # This ensures the agent never loses track of the original task intent
            elif msg.role == "user" and not any(sm.message.role == "user" for sm in scored):
                # This is the first user message we've encountered - preserve it with highest priority
                logger.debug(f"Original user message at index {i} preserved with highest priority")
                scored.append(MessageImportance(msg, i, 2000.0, "original_user_intent"))
            else:
                scorable_msgs.append(msg)
                scorable_indices.append(i)

        if not scorable_msgs:
            return scored

        # Lazy imports to avoid circular dependency
        from victor.agent.conversation.types import ConversationMessage, MessagePriority
        from victor.agent.conversation.scoring import score_messages, CONTROLLER_WEIGHTS

        # Convert to ConversationMessage for canonical scorer
        conv_messages = [
            ConversationMessage.from_provider_message(
                msg,
                priority=(MessagePriority.HIGH if msg.role == "tool" else MessagePriority.MEDIUM),
            )
            for msg in scorable_msgs
        ]

        # Build embedding function if semantic scoring is enabled
        embedding_fn = None
        if self._embedding_service and self.config.compaction_strategy in (
            CompactionStrategy.SEMANTIC,
            CompactionStrategy.HYBRID,
        ):

            def _embedding_fn(texts: List[str], query: str) -> List[float]:
                similarities = []
                for text in texts:
                    sim = self._compute_semantic_similarity(text, query)
                    similarities.append(sim)
                return similarities

            embedding_fn = _embedding_fn

        # Delegate to canonical scorer
        canonical_scored = score_messages(
            conv_messages,
            current_query=current_query,
            weights=CONTROLLER_WEIGHTS,
            embedding_fn=embedding_fn,
        )

        # Convert back to MessageImportance, mapping to original messages
        conv_to_idx = {id(cm): idx for cm, idx in zip(conv_messages, scorable_indices)}
        conv_to_msg = {id(cm): msg for cm, msg in zip(conv_messages, scorable_msgs)}

        for conv_msg, canonical_score in canonical_scored:
            orig_idx = conv_to_idx[id(conv_msg)]
            orig_msg = conv_to_msg[id(conv_msg)]
            scored.append(MessageImportance(orig_msg, orig_idx, canonical_score, "canonical"))

        return scored

    def _compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two texts using embeddings.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score between 0 and 1
        """
        if not self._embedding_service:
            return 0.0

        try:
            # Truncate long texts to avoid embedding issues
            t1 = text1[:2000] if len(text1) > 2000 else text1
            t2 = text2[:2000] if len(text2) > 2000 else text2

            embeddings = self._embedding_service.embed_batch([t1, t2])
            if len(embeddings) == 2:
                from victor.processing.native import cosine_similarity

                emb1, emb2 = embeddings[0], embeddings[1]
                # Use Rust-accelerated cosine similarity (with NumPy fallback)
                return cosine_similarity(list(emb1), list(emb2))
        except ImportError:
            logger.debug("cosine_similarity not available, returning 0.0")
        except (ValueError, TypeError) as e:
            logger.warning(f"Semantic similarity computation failed (bad embeddings): {e}")

        return 0.0

    def _generate_compaction_summary(
        self,
        removed_messages: List[Message],
        current_query: Optional[str] = None,
    ) -> str:
        """Generate a summary of removed messages for context preservation.

        Uses the legacy compaction summarizer strategy if available,
        otherwise falls back to keyword-based extraction.

        Args:
            removed_messages: Messages being removed
            current_query: Optional current query for relevance scoring

        Returns:
            Summary string, or empty string if nothing noteworthy
        """
        if not removed_messages:
            return ""

        # Legacy: Delegate to strategy if available
        if self._compaction_summarizer:
            try:
                return self._compaction_summarizer.summarize(removed_messages)
            except Exception as e:
                logger.warning(f"Compaction summarizer failed, using fallback: {e}")

        # Fallback: keyword extraction
        user_msgs = [m for m in removed_messages if m.role == "user"]
        tool_msgs = [m for m in removed_messages if m.role == "tool"]

        all_content = " ".join(m.content[:200] for m in removed_messages[:10])
        topics = self._extract_key_topics(all_content)

        parts = []
        if user_msgs:
            parts.append(f"{len(user_msgs)} user messages")
        if tool_msgs:
            parts.append(f"{len(tool_msgs)} tool results")
        if topics:
            parts.append(f"topics: {', '.join(topics[:3])}")

        if parts:
            return f"[Earlier conversation: {'; '.join(parts)}]"
        return ""

    def _extract_key_topics(self, text: str) -> List[str]:
        """Extract key topics from text using simple heuristics.

        Args:
            text: Text to extract topics from

        Returns:
            List of key topic words/phrases
        """
        import re

        # Simple keyword extraction - find capitalized words and technical terms
        words = re.findall(r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)*\b", text)  # CamelCase
        words += re.findall(r"\b[a-z]+_[a-z_]+\b", text)  # snake_case
        words += re.findall(r"\b(?:def|class|function|file|error|test|api)\s+\w+", text.lower())

        # Deduplicate and limit
        seen = set()
        topics = []
        for w in words:
            w_lower = w.lower()
            if w_lower not in seen and len(w) > 3:
                seen.add(w_lower)
                topics.append(w)
                if len(topics) >= 5:
                    break

        return topics

    def get_compaction_summaries(self) -> List[str]:
        """Get summaries from previous compaction operations.

        Returns:
            List of summary strings
        """
        return self._compaction_summaries.copy()

    def inject_compaction_context(self) -> bool:
        """Update compaction context for the next LLM call.

        Two injection modes:
        - With ContextReminderManager: Sets compaction_summary on manager state.
          Actual injection happens during the next get_consolidated_reminder() call.
        - Without (fallback): Inserts a user reminder message directly at position 1.

        P0 FIX: Uses user-role reminders for higher salience (OpenDev finding).

        Returns:
            True if compaction context was updated or injected.
        """
        if not self._compaction_summaries:
            return False

        # Use hierarchical manager for compressed context if available
        if self._hierarchical_manager:
            combined = self._hierarchical_manager.get_active_context()
        else:
            combined = " | ".join(self._compaction_summaries[-3:])
        combined = _sanitize_compaction_summary(combined)

        if not combined:
            return False

        # Primary: delegate to reminder framework
        if self._context_reminder_manager:
            self._context_reminder_manager.state.compaction_summary = combined
            return True

        # Fallback: direct insertion (backward compat)
        # P0 FIX: Use user role for higher salience instead of system role (OpenDev finding)
        if len(self.messages) > 1:
            from victor.agent.conversation.types import (
                MESSAGE_SOURCE_METADATA_KEY,
                MessageSource,
            )

            reminder_msg = Message(
                role="user",
                content=f"[CONTEXT COMPACTED: Previous context was compacted. Summary: {combined}]",
                metadata={MESSAGE_SOURCE_METADATA_KEY: MessageSource.COMPACTION_SUMMARY.value},
            )
            self._history._messages.insert(1, reminder_msg)
            logger.debug(f"Injected compaction context (user role): {combined[:100]}...")
            return True

        return False

    def set_embedding_service(self, service: "EmbeddingService") -> None:
        """Set the embedding service for semantic compaction.

        Args:
            service: EmbeddingService instance
        """
        self._embedding_service = service
        # Also set on conversation store if available
        if self._conversation_store:
            self._conversation_store.set_embedding_service(service)

    def set_conversation_store(
        self,
        store: "ConversationStore",
        session_id: str,
    ) -> None:
        """Set the persistent conversation store for enhanced compaction.

        Enables semantic retrieval from full conversation history stored in SQLite.

        Args:
            store: ConversationStore instance
            session_id: Session ID for store operations
        """
        self._conversation_store = store
        self._session_id = session_id
        # Share embedding service with store
        if self._embedding_service:
            store.set_embedding_service(self._embedding_service)

    def retrieve_relevant_history(
        self,
        query: str,
        limit: int = 5,
    ) -> List[str]:
        """Retrieve relevant historical context from persistent store.

        Uses semantic similarity to find relevant messages and summaries
        from the full conversation history in SQLite.

        Args:
            query: Current query to find relevant history for
            limit: Maximum items to retrieve

        Returns:
            List of relevant historical context strings
        """
        if not self._conversation_store or not self._session_id:
            return []

        relevant_context: List[str] = []

        try:
            if hasattr(self._conversation_store, "get_dual_trace_relevant_messages"):
                execution_limit = max(1, limit // 3)
                semantic_limit = max(1, limit - execution_limit)
                relevant_traces = self._conversation_store.get_dual_trace_relevant_messages(
                    self._session_id,
                    query,
                    semantic_limit=semantic_limit,
                    execution_limit=execution_limit,
                    min_similarity=self.config.semantic_relevance_threshold,
                )

                for msg, score in relevant_traces.get("semantic", []):
                    role_label = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
                    context = f"[Semantic memory {role_label} (relevance: {score:.2f})]: {msg.content[:500]}"
                    relevant_context.append(context)

                for msg, score in relevant_traces.get("execution", []):
                    trace_text = (
                        self._conversation_store.get_message_trace_text(msg, "execution")
                        if hasattr(self._conversation_store, "get_message_trace_text")
                        else msg.content
                    )
                    context = f"[Execution trace (relevance: {score:.2f})]: {trace_text[:500]}"
                    relevant_context.append(context)
            else:
                # Get semantically similar historical messages
                relevant_msgs = self._conversation_store.get_semantically_relevant_messages(
                    self._session_id,
                    query,
                    limit=limit,
                    min_similarity=self.config.semantic_relevance_threshold,
                )

                for msg, score in relevant_msgs:
                    # Format as context reminder
                    role_label = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
                    context = (
                        f"[Historical {role_label} (relevance: {score:.2f})]: {msg.content[:500]}"
                    )
                    relevant_context.append(context)

            # Get relevant summaries
            relevant_summaries = self._conversation_store.get_relevant_summaries(
                self._session_id,
                query,
                limit=2,
            )

            for summary, score in relevant_summaries:
                sanitized_summary = _sanitize_compaction_summary(summary)
                if not sanitized_summary:
                    continue
                context = f"[Prior context summary (relevance: {score:.2f})]: {sanitized_summary}"
                relevant_context.append(context)

        except (OSError, IOError) as e:
            logger.warning(f"Failed to retrieve relevant history (I/O error): {e}")
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to retrieve relevant history (data error): {e}")

        return relevant_context[:limit]

    def persist_compaction_summary(self, summary: str, message_ids: List[str]) -> None:
        """Persist a compaction summary to the SQLite store.

        Args:
            summary: Summary of compacted content
            message_ids: IDs of messages that were summarized
        """
        summary = _sanitize_compaction_summary(summary)
        if not summary:
            return
        if not self._conversation_store or not self._session_id:
            return

        try:
            self._conversation_store.store_compaction_summary(
                self._session_id,
                summary,
                message_ids,
            )
        except (OSError, IOError) as e:
            logger.warning(f"Failed to persist compaction summary (I/O error): {e}")
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to persist compaction summary (data error): {e}")

    def on_context_overflow(self, callback: Callable[[ContextMetrics], None]) -> None:
        """Register callback for context overflow events.

        Args:
            callback: Function to call when overflow is detected
        """
        self._context_callbacks.append(callback)

    def _notify_context_callbacks(self, metrics: ContextMetrics) -> None:
        """Notify registered callbacks about context state."""
        for callback in self._context_callbacks:
            try:
                callback(metrics)
            except (TypeError, ValueError, AttributeError) as e:
                logger.warning(f"Context callback failed (invalid callback): {e}")
            except Exception as e:
                # Log with traceback for unexpected callback failures
                logger.exception(f"Context callback failed unexpectedly: {e}")

    def get_last_user_message(self) -> Optional[str]:
        """Get the content of the last user message.

        Returns:
            Last user message content or None
        """
        for msg in reversed(self.messages):
            if msg.role == "user":
                return msg.content
        return None

    def get_last_assistant_message(self) -> Optional[str]:
        """Get the content of the last assistant message.

        Returns:
            Last assistant message content or None
        """
        for msg in reversed(self.messages):
            if msg.role == "assistant":
                return msg.content
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Export conversation state as dictionary.

        Returns:
            Dictionary representation of conversation
        """
        return {
            "messages": [{"role": m.role, "content": m.content} for m in self.messages],
            "stage": self.stage.value,
            "metrics": {
                "char_count": self.get_context_metrics().char_count,
                "message_count": self.message_count,
            },
        }

    @classmethod
    def from_messages(
        cls,
        messages: List[Message],
        config: Optional[ConversationConfig] = None,
    ) -> "ConversationController":
        """Create controller from existing messages.

        Args:
            messages: List of messages to initialize with
            config: Optional configuration

        Returns:
            New ConversationController with messages
        """
        controller = cls(config=config)
        for msg in messages:
            controller._history.append_message(msg)
        if messages and messages[0].role == "system":
            controller._system_prompt = messages[0].content
            controller._system_added = True
        return controller
