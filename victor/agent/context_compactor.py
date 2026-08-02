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

"""Context Compactor - Proactive context management and optimization.

This module provides intelligent context compaction that goes beyond
simple message removal:

- **Proactive compaction**: Triggers before overflow (at configurable threshold)
- **Tool result truncation**: Intelligently truncates large tool outputs
- **Token-accurate estimation**: Uses model-specific tokenization estimates
- **Content-aware pruning**: Preserves important content patterns

Design Pattern: Strategy + Decorator
====================================
ContextCompactor wraps ConversationController and adds proactive
compaction policies that trigger before context overflow occurs.

Usage:
    compactor = ContextCompactor(conversation_controller)

    # After each message, check if compaction is needed
    action = compactor.check_and_compact(current_query)
    # Returns: CompactionAction with what was done

    # Truncate tool result before adding to conversation
    truncated = compactor.truncate_tool_result(result, max_tokens=2000)
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from victor.config.orchestrator_constants import (
    COMPACTION_CONFIG,
    CONTEXT_LIMITS,
    RL_LEARNER_CONFIG,
)

# Lazy import to avoid circular dependency
AdaptiveCompactionThreshold = None
from victor.core.shared_types import TaskPhase
from victor.providers.base import Message

if TYPE_CHECKING:
    from victor.agent.conversation.controller import ConversationController
    from victor.framework.rl.learners.context_pruning import ContextPruningLearner
    from victor.providers.base import BaseProvider

logger = logging.getLogger(__name__)


class MessagePriority(int, Enum):
    """Priority levels for messages during compaction.

    Higher values = more important = preserved longer during compaction.
    PINNED messages are never compacted away.
    """

    LOW = 25  # Can be compacted first
    MEDIUM = 50  # Standard messages
    HIGH = 75  # Important context
    CRITICAL = 100  # Error messages, recent context
    PINNED = 150  # Output requirements - never compacted


class CompactionTrigger(Enum):
    """What triggered the compaction."""

    NONE = "none"  # No compaction needed
    THRESHOLD = "threshold"  # Hit utilization threshold
    OVERFLOW = "overflow"  # Actually overflowing
    MANUAL = "manual"  # Manually requested
    SCHEDULED = "scheduled"  # Background scheduled compaction


class TruncationStrategy(Enum):
    """Strategy for truncating large content."""

    HEAD = "head"  # Keep beginning
    TAIL = "tail"  # Keep end
    BOTH = "both"  # Keep beginning and end
    SMART = "smart"  # Use content-aware truncation


@dataclass
class ParallelReadBudget:
    """Budget information for parallel file reads.

    This helps the LLM plan efficient parallel reads by knowing:
    - How many files can be read in parallel
    - Maximum chars per file to stay within context limits
    - Total budget available for reads

    Example:
        budget = calculate_parallel_read_budget(context_window=65536)
        # Returns: ParallelReadBudget(max_parallel_files=10, chars_per_file=8192, ...)
    """

    context_window: int  # Model's context window in tokens
    usable_tokens: int  # Tokens available for reads (after reserving for output)
    usable_chars: int  # Approximate chars available (~3 chars/token)
    max_parallel_files: int  # Recommended number of parallel reads
    chars_per_file: int  # Max chars per file to stay within budget
    total_read_budget: int  # Total chars budget for all reads

    def to_prompt_hint(self) -> str:
        """Generate a hint string for the system prompt."""
        return (
            f"PARALLEL READ BUDGET: You can read up to {self.max_parallel_files} files "
            f"in parallel, each limited to ~{self.chars_per_file:,} chars "
            f"(~{self.chars_per_file // 35} lines). "
            f"Total read budget: {self.total_read_budget:,} chars."
        )


def calculate_parallel_read_budget(
    context_window: int = 65536,
    output_reserve: float = COMPACTION_CONFIG.output_reserve_pct,
    chars_per_token: float = COMPACTION_CONFIG.chars_per_token,
    target_parallel_files: int = COMPACTION_CONFIG.parallel_read_target_files,
) -> ParallelReadBudget:
    """Calculate optimal budget for parallel file reads.

    This function helps plan parallel reads by calculating how much content
    can be read from multiple files while staying within context limits.

    Args:
        context_window: Model's context window in tokens (default: 64K)
        output_reserve: Fraction of context to reserve for output (default: 0.5)
        chars_per_token: Approximate characters per token (default: 3.0)
        target_parallel_files: Target number of parallel reads (default: 10)

    Returns:
        ParallelReadBudget with calculated limits

    Example:
        >>> budget = calculate_parallel_read_budget(65536)
        >>> budget.max_parallel_files
        10
        >>> budget.chars_per_file
        9830
    """
    usable_tokens = int(context_window * (1 - output_reserve))
    usable_chars = int(usable_tokens * chars_per_token)
    chars_per_file = usable_chars // target_parallel_files

    # Round down to nice number (multiple of 1024)
    chars_per_file = (chars_per_file // 1024) * 1024
    if chars_per_file < 4096:
        chars_per_file = 4096  # Minimum useful read size

    return ParallelReadBudget(
        context_window=context_window,
        usable_tokens=usable_tokens,
        usable_chars=usable_chars,
        max_parallel_files=target_parallel_files,
        chars_per_file=chars_per_file,
        total_read_budget=chars_per_file * target_parallel_files,
    )


@dataclass
class CompactorConfig:
    """Configuration for the context compactor.

    Attributes:
        proactive_threshold: Utilization % to trigger proactive compaction (0.0-1.0)
        min_messages_after_compact: Minimum messages to keep after compaction
        tool_result_max_chars: Max characters for tool results
        tool_result_max_lines: Max lines for tool results
        truncation_strategy: Strategy for truncating large content
        preserve_code_blocks: Try to preserve code blocks when truncating
        preserve_json_structure: Try to preserve JSON structure when truncating
        enable_proactive: Enable proactive (pre-overflow) compaction
        enable_tool_truncation: Enable automatic tool result truncation
        enable_phase_aware: Enable phase-aware compaction thresholds
        enable_fast_pruning: Enable fast pruning before LLM compaction (P1: OpenDev finding)
        phase_thresholds: Phase-specific compaction thresholds (overrides proactive_threshold)
    """

    proactive_threshold: float = CONTEXT_LIMITS.proactive_compaction_threshold
    min_messages_after_compact: int = COMPACTION_CONFIG.min_messages_after_compact
    # 8192 chars allows ~10-12 parallel file reads within 32K usable tokens (64K context / 2)
    # Dynamic formula: (context_window * 0.5 * 3) / expected_parallel_files
    # Example: (65536 * 0.5 * 3) / 10 = ~9,830 chars per file
    tool_result_max_chars: int = COMPACTION_CONFIG.tool_result_max_chars
    tool_result_max_lines: int = COMPACTION_CONFIG.tool_result_max_lines
    truncation_strategy: TruncationStrategy = TruncationStrategy.SMART
    preserve_code_blocks: bool = True
    preserve_json_structure: bool = True
    enable_proactive: bool = True
    enable_tool_truncation: bool = True
    enable_phase_aware: bool = True
    enable_fast_pruning: bool = COMPACTION_CONFIG.enable_fast_pruning
    # L1: reference-aware tool-result pruning (default OFF). When on, fast pruning preserves
    # tool results still cited by later assistant turns and prunes only the unreferenced rest.
    enable_reference_aware_pruning: bool = COMPACTION_CONFIG.enable_reference_aware_pruning
    reference_window_turns: int = COMPACTION_CONFIG.reference_window_turns
    # Wave 6: shrink threshold by this factor when task complexity is "high"
    high_complexity_threshold_factor: float = 0.85
    # Phase-specific thresholds (75% conservative to avoid overflows due to token estimation inaccuracy)
    # Reduced from 85-90% to 75% to provide safety buffer for inconsistent chars_per_token values
    # across different modules (ContextLimits: 3.0, CompactionConfig: 3.5, ConversationStore: 4.0)
    phase_thresholds: Dict[TaskPhase, float] = field(
        default_factory=lambda: {
            TaskPhase.EXPLORATION: 0.75,  # Reduced from 0.85 - safety buffer for token estimation
            TaskPhase.PLANNING: 0.75,  # Reduced from 0.90 - avoid premature compaction
            TaskPhase.EXECUTION: 0.75,  # Reduced from 0.85 - prioritize recent context
            TaskPhase.REVIEW: 0.75,  # Reduced from 0.85 - full context for review
        }
    )


@dataclass
class CompactionAction:
    """Result of a compaction check/action.

    Attributes:
        trigger: What triggered the action
        messages_removed: Number of messages removed
        chars_freed: Approximate characters freed
        tokens_freed: Approximate tokens freed
        truncations_applied: Number of truncations applied
        new_utilization: Context utilization after action
        details: Additional details about what was done
    """

    trigger: CompactionTrigger
    messages_removed: int = 0
    chars_freed: int = 0
    tokens_freed: int = 0
    truncations_applied: int = 0
    new_utilization: float = 0.0
    details: List[str] = field(default_factory=list)

    @property
    def action_taken(self) -> bool:
        """Whether any action was taken."""
        return self.messages_removed > 0 or self.truncations_applied > 0


@dataclass
class TruncationResult:
    """Result of truncating content.

    Attributes:
        content: The truncated content
        original_chars: Original character count
        truncated_chars: Characters removed
        truncated: Whether truncation occurred
        indicator: Truncation indicator added (e.g., "... [truncated]")
    """

    content: str
    original_chars: int
    truncated_chars: int
    truncated: bool
    indicator: str = ""


class ContextCompactor:
    """Proactive context compaction and optimization.

    Wraps ConversationController to add:
    - Proactive compaction before overflow
    - Intelligent tool result truncation
    - Content-aware pruning
    - Token-accurate estimation

    Thread-safe for concurrent access.
    """

    # Token estimation factors by content type
    TOKEN_FACTORS = {
        "code": 3.0,  # Code is more token-dense
        "json": 2.8,  # JSON has structure overhead
        "prose": 4.0,  # English prose ~4 chars/token
        "mixed": 3.5,  # Default mixed content
    }

    # Patterns for preserving important content
    CODE_BLOCK_PATTERN = re.compile(r"```[\s\S]*?```", re.MULTILINE)
    ERROR_PATTERN = re.compile(r"(?:error|exception|traceback|failed):", re.IGNORECASE)
    PATH_PATTERN = re.compile(r"(?:/[\w./]+|\w:\\[\w\\]+)")

    # Patterns for pinned output requirements (never compacted)
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
    ]

    def __init__(
        self,
        controller: Optional["ConversationController"] = None,
        config: Optional[CompactorConfig] = None,
        pruning_learner: Optional["ContextPruningLearner"] = None,
        provider_type: str = "cloud",
        event_bus: Optional[Any] = None,
        decision_service: Optional[Any] = None,
        runtime_intelligence: Optional[Any] = None,
        adaptive_threshold: Optional["AdaptiveCompactionThreshold"] = None,
    ):
        """Initialize the context compactor.

        Args:
            controller: The ConversationController to wrap (optional for testing)
            config: Optional compactor configuration
            pruning_learner: Optional RL learner for adaptive pruning decisions
            provider_type: Provider type for RL state (cloud, local)
            event_bus: Optional event bus for compaction events (ObservabilityBus)
            decision_service: Optional TieredDecisionService for tier-based compaction routing
            runtime_intelligence: Optional canonical runtime-intelligence service
            adaptive_threshold: Optional adaptive threshold system for pattern-based thresholds
        """
        self.controller = controller
        self.config = config or CompactorConfig()
        self.pruning_learner = pruning_learner
        self.provider_type = provider_type
        self._event_bus = event_bus
        self._decision_service = decision_service
        self._runtime_intelligence = runtime_intelligence
        self._last_compaction_turn: int = 0
        self._total_chars_freed: int = 0
        self._total_tokens_freed: int = 0
        self._compaction_count: int = 0

        # Track RL learner recommendations
        self._rl_enabled = RL_LEARNER_CONFIG.enabled and pruning_learner is not None
        self._last_rl_action: Optional[str] = None
        self._last_task_success: bool = True

        # Adaptive threshold system
        self._adaptive_threshold = adaptive_threshold
        self._adaptive_enabled = adaptive_threshold is not None

        logger.debug(
            f"ContextCompactor initialized (threshold: {self.config.proactive_threshold:.0%}, "
            f"rl_enabled: {self._rl_enabled}, "
            f"adaptive_enabled: {self._adaptive_enabled}, "
            f"decision_service: {decision_service is not None or runtime_intelligence is not None})"
        )

    def _has_decision_support(self) -> bool:
        """Return whether compaction decisions can use the canonical runtime boundary."""
        if self._runtime_intelligence is not None:
            return True
        return self._decision_service is not None

    def _decide_sync(
        self,
        decision_type: Any,
        context: Dict[str, Any],
        *,
        heuristic_result: Any = None,
        heuristic_confidence: float = 0.0,
    ) -> Optional[Any]:
        """Delegate bounded compaction decisions via runtime intelligence when available."""
        if self._runtime_intelligence is not None:
            return self._runtime_intelligence.decide_sync(
                decision_type,
                context,
                heuristic_result=heuristic_result,
                heuristic_confidence=heuristic_confidence,
            )
        if self._decision_service is not None:
            return self._decision_service.decide_sync(
                decision_type,
                context,
                heuristic_result=heuristic_result,
                heuristic_confidence=heuristic_confidence,
            )
        return None

    # ========================================================================
    # Tier-Based Provider Selection
    # ========================================================================

    def _get_provider_for_tier(self, tier: str) -> Optional["BaseProvider"]:
        """Get a provider instance for the specified tier.

        Args:
            tier: One of 'edge', 'balanced', 'performance'

        Returns:
            Provider instance for the tier, or None if unavailable
        """
        try:
            from victor.config.decision_settings import DecisionServiceSettings
            from victor.providers.registry import ProviderRegistry

            # Get tier spec from settings
            settings = DecisionServiceSettings()
            spec = getattr(settings, tier, None)
            if spec is None:
                logger.debug(f"No spec found for tier '{tier}'")
                return None

            # Build provider kwargs
            provider_kwargs: Dict[str, Any] = {"timeout": spec.timeout_ms // 1000}

            if spec.provider == "ollama":
                provider_kwargs["base_url"] = "http://localhost:11434"
            else:
                try:
                    from victor.config.api_keys import get_api_key

                    api_key = get_api_key(spec.provider)
                    if api_key:
                        provider_kwargs["api_key"] = api_key
                except Exception:
                    pass

            provider = ProviderRegistry.create(spec.provider, **provider_kwargs)
            logger.debug(f"Created provider for tier '{tier}': {spec.provider}/{spec.model}")
            return provider

        except Exception as e:
            logger.warning(f"Failed to create provider for tier '{tier}': {e}")
            return None

    def _get_default_provider(self) -> Optional["BaseProvider"]:
        """Get the default provider from the orchestrator.

        Returns:
            The active provider from ProviderManager, or None if unavailable
        """
        try:
            if self.controller and hasattr(self.controller, "_orchestrator"):
                provider_manager = self.controller._orchestrator.provider_manager
                return provider_manager.get_active_provider()
        except Exception as e:
            logger.debug(f"Failed to get default provider: {e}")
        return None

    def _has_eligible_compaction_payload(
        self,
        min_messages_keep: int,
        metrics: Optional[Any] = None,
    ) -> bool:
        """Return True when compaction has something meaningful to remove or trim."""
        try:
            raw_messages = self.controller.get_messages()
        except Exception as exc:
            logger.debug("Could not inspect messages for compaction eligibility: %s", exc)
            return True

        try:
            messages = list(raw_messages)
        except Exception:
            messages = []

        if len(messages) == 0:
            if metrics is not None:
                try:
                    if metrics.message_count > min_messages_keep:
                        return True
                except Exception:
                    pass
            else:
                try:
                    metrics = self.controller.get_context_metrics()
                    if metrics.message_count > min_messages_keep:
                        return True
                except Exception:
                    pass

            fallback_messages = getattr(self.controller, "messages", None)
            if fallback_messages:
                try:
                    messages = list(fallback_messages)
                except Exception:
                    messages = []

        if len(messages) == 0:
            return True

        non_pinned = [
            message
            for message in messages
            if self._assign_priority(message) < MessagePriority.PINNED
        ]
        if len(non_pinned) > min_messages_keep:
            return True

        max_chars = max(1024, self.config.tool_result_max_chars)
        for message in messages:
            if (
                getattr(message, "role", None) == "tool"
                and len(getattr(message, "content", "") or "") > max_chars
            ):
                return True
        return False

    def get_prompt_optimization_decision(
        self,
        current_query: Optional[str] = None,
        recent_failures: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Get a system prompt optimization decision for dynamic prompts.

        When system_prompt_strategy is 'dynamic', this method uses the decision
        service to determine which prompt sections to include based on current
        context and recent failures.

        Args:
            current_query: Current user query for context
            recent_failures: Optional list of recent error messages for failure hints

        Returns:
            Dict with optimization decision, or None if decision service unavailable
        """
        if not self._has_decision_support():
            return None

        try:
            # Build context for optimization decision
            context = {
                "current_query": current_query or "",
                "recent_failures": recent_failures or [],
                "compaction_count": self._compaction_count,
                "utilization": (
                    self.controller.get_context_metrics().utilization if self.controller else 0.0
                ),
            }

            # Use decision service for optimization
            # Note: We'll use a simple heuristic for now since we don't have a dedicated
            # SYSTEM_PROMPT_OPTIMIZATION decision type yet
            # This can be extended later with a proper decision type

            # Simple heuristic-based optimization
            all_sections = [
                "concise_mode",
                "task_guidance",
                "tool_constraint",
                "completion",
                "tool_guidance",
                "few_shot_examples",
            ]

            # Include context reminder if we've compacted recently
            add_context_reminder = self._compaction_count > 0

            # Add failure hints if there are recent failures
            add_failure_hints = bool(recent_failures)

            # Adjust for complexity based on utilization
            adjust_for_complexity = context["utilization"] > 0.7

            # For dynamic mode, include all sections by default
            # (can be refined later with edge model decision)
            include_sections = all_sections

            return {
                "include_sections": include_sections,
                "add_context_reminder": add_context_reminder,
                "add_failure_hints": add_failure_hints,
                "adjust_for_complexity": adjust_for_complexity,
                "confidence": 0.8,
                "reason": "Heuristic-based prompt optimization",
            }

        except Exception as e:
            logger.debug(f"Prompt optimization decision failed: {e}")
            return None

    # ========================================================================
    # Message Priority Assignment
    # ========================================================================

    def _is_pinned_requirement(self, content: str) -> bool:
        """Check if content contains pinned output requirement patterns.

        Pinned requirements are output format specifications that must
        survive compaction so the agent produces the required output.

        Args:
            content: Message content to check

        Returns:
            True if content contains pinned requirement patterns
        """
        for pattern in self.PINNED_REQUIREMENT_PATTERNS:
            if pattern.search(content):
                return True
        return False

    def _assign_priority(self, message: Any) -> MessagePriority:
        """Assign priority level to a message for compaction decisions.

        Priority levels determine which messages are compacted first:
        - PINNED: Output requirements - never compacted
        - CRITICAL: Errors, recent context - compacted last
        - HIGH: Important context
        - MEDIUM: Standard messages
        - LOW: Can be compacted first

        Args:
            message: Message dict with 'role' and 'content'

        Returns:
            MessagePriority level for this message
        """
        if isinstance(message, dict):
            role = message.get("role", "")
            content = message.get("content", "")
        else:
            role = getattr(message, "role", "")
            content = getattr(message, "content", "")

        role = role or ""
        content = content or ""

        # Check for pinned output requirements first
        if self._is_pinned_requirement(content):
            return MessagePriority.PINNED

        # System messages are critical
        if role == "system":
            return MessagePriority.CRITICAL

        # Check for error patterns
        if self.ERROR_PATTERN.search(content):
            return MessagePriority.CRITICAL

        # User messages with questions are high priority
        if role == "user" and "?" in content:
            return MessagePriority.HIGH

        # Default priority based on role
        if role == "user":
            return MessagePriority.MEDIUM
        elif role == "assistant":
            return MessagePriority.MEDIUM

        return MessagePriority.LOW

    # ========================================================================
    # Main Compaction Interface
    # ========================================================================

    def _maybe_summarize_turns(self) -> Optional[CompactionAction]:
        """Summarize old turns if conversation exceeds turn threshold.

        Uses LLMCompactionSummarizer to produce a compact summary of the
        oldest turns. This preserves intent and decisions better than simple
        pruning. Summary replaces original messages in the active context.

        Uses decision service for tier-based routing:
        - Simple compaction (≤8 messages) → Edge tier (fast, free)
        - Complex compaction (>8 messages) → Performance tier (high quality)
        - Falls back to main LLM if decision service unavailable

        Returns:
            CompactionAction if summarization occurred, None otherwise.
        """
        from victor.config.orchestrator_constants import SUMMARIZATION_CONFIG
        from victor.agent.llm_compaction_summarizer import LLMCompactionSummarizer
        from victor.agent.decisions.schemas import DecisionType

        if not SUMMARIZATION_CONFIG.enabled or self.controller is None:
            return None

        messages = self.controller.messages
        # Count user messages as turn indicators
        turn_count = sum(1 for m in messages if m.role == "user")
        if turn_count <= SUMMARIZATION_CONFIG.trigger_turn_count:
            return None

        # Already summarized recently? Skip to avoid excessive LLM calls
        if self._last_compaction_turn > 0:
            turns_since_last = turn_count - self._last_compaction_turn
            if turns_since_last < SUMMARIZATION_CONFIG.max_turns_to_summarize:
                return None

        # Find the oldest turns to summarize
        turn_boundaries = [i for i, m in enumerate(messages) if m.role == "user"]
        if len(turn_boundaries) <= SUMMARIZATION_CONFIG.trigger_turn_count:
            return None

        # Always preserve the root system prompt (index 0)
        start_idx = 1 if messages and messages[0].role == "system" else 0
        # Summarize up to N oldest turns
        end_turn_idx = min(SUMMARIZATION_CONFIG.max_turns_to_summarize, len(turn_boundaries) - 3)
        if end_turn_idx <= 0:
            return None

        end_msg_idx = turn_boundaries[end_turn_idx]
        turns_to_summarize = messages[start_idx:end_msg_idx]

        if not turns_to_summarize:
            return None

        # Determine complexity and select provider
        provider = None
        complexity = "complex" if len(turns_to_summarize) > 8 else "simple"
        estimated_tokens = sum(self.estimate_message_tokens(m) for m in turns_to_summarize)

        # Try decision service for tier-based routing
        if self._has_decision_support():
            try:
                decision_result = self._decide_sync(
                    DecisionType.COMPACTION,
                    context={
                        "message_count": len(turns_to_summarize),
                        "estimated_tokens": estimated_tokens,
                        "complexity": complexity,
                    },
                    heuristic_result=None,
                )

                if decision_result and decision_result.result:
                    decision = decision_result.result
                    recommended_tier = getattr(decision, "recommended_tier", None)
                    if recommended_tier:
                        provider = self._get_provider_for_tier(recommended_tier)
                        logger.debug(
                            f"Compaction decision: tier={recommended_tier}, "
                            f"complexity={complexity}, messages={len(turns_to_summarize)}"
                        )
            except Exception as e:
                logger.debug(f"Decision service failed, using fallback: {e}")

        # Fallback to default provider if decision service didn't return one
        if provider is None:
            provider = self._get_default_provider()
            if provider is None:
                logger.debug("No provider available for summarization")
                return None

        # Attempt LLM-based summarization
        summary_text = ""
        try:
            summarizer = LLMCompactionSummarizer(provider=provider)
            summary_text = summarizer.summarize(turns_to_summarize)
        except Exception as e:
            logger.debug(f"LLM summarization failed, falling back to heuristic: {e}")
            summary_text = self._heuristic_summarize(turns_to_summarize)

        if not summary_text:
            return None

        # Wrap summary in a message
        from victor.providers.base import Message

        summary_msg = Message(
            role="assistant",
            content=f"[Context summary of {len(turns_to_summarize)} earlier messages]\n{summary_text}",
        )

        # Replace old turns with summary message
        chars_before = sum(len(m.content) for m in turns_to_summarize)
        del messages[start_idx:end_msg_idx]
        messages.insert(start_idx, summary_msg)

        chars_after = len(summary_msg.content)
        chars_freed = chars_before - chars_after
        self._last_compaction_turn = sum(1 for m in messages if m.role == "user")

        logger.info(
            "Summarized %d messages (%dK -> %dK chars, freed %dK, complexity=%s)",
            len(turns_to_summarize),
            chars_before // 1024,
            chars_after // 1024,
            chars_freed // 1024,
            complexity,
        )

        return CompactionAction(
            trigger=CompactionTrigger.THRESHOLD,
            messages_removed=len(turns_to_summarize) - 1,
            tokens_freed=self._estimate_tokens(chars_freed),
            new_utilization=0.0,
            details=[
                f"LLM summarized {len(turns_to_summarize)} messages",
                f"Freed ~{chars_freed // 4} tokens",
                f"Complexity: {complexity}",
            ],
        )

    def _heuristic_summarize(self, turns: List[Any]) -> str:
        """Fallback heuristic summarization if LLM call fails."""
        tool_names = set()
        files_mentioned = set()
        key_decisions = []
        for msg in turns:
            tool_name = getattr(msg, "tool_name", None)
            if tool_name:
                tool_names.add(tool_name)

            # Extract file paths
            import re

            paths = re.findall(r"[\w./]+\.(?:py|js|ts|rs|go|java|yaml|json)", msg.content[:500])
            files_mentioned.update(paths[:5])

            # Extract decisions from assistant
            if msg.role == "assistant" and len(msg.content) > 50:
                first_line = msg.content.split("\n")[0][:150]
                if first_line:
                    key_decisions.append(first_line)

        summary_parts = []
        if tool_names:
            summary_parts.append(f"Tools used: {', '.join(sorted(tool_names)[:10])}")
        if files_mentioned:
            summary_parts.append(f"Files touched: {', '.join(sorted(files_mentioned)[:8])}")
        if key_decisions:
            summary_parts.append("Key points:")
            for d in key_decisions[:5]:
                summary_parts.append(f"  - {d}")

        return "\n".join(summary_parts) if summary_parts else "Prior history summarized."

    def check_and_compact(
        self,
        current_query: Optional[str] = None,
        force: bool = False,
        tool_call_count: int = 0,
        task_complexity: str = "medium",
        phase: Optional[TaskPhase] = None,
    ) -> CompactionAction:
        """Check context utilization and compact if needed.

        Two-phase compaction:
        1. Turn-based summarization (early, at turn threshold)
        2. Utilization-based compaction (late, at phase-aware threshold)

        Uses RL learner for adaptive pruning when available.

        Args:
            current_query: Current user query for semantic relevance
            force: Force compaction regardless of utilization
            tool_call_count: Number of tool calls made (for RL state)
            task_complexity: Task complexity level (for RL state)
            phase: Current task phase for phase-aware thresholds

        Returns:
            CompactionAction describing what was done
        """
        # Phase 1: Turn-based summarization (fires early, before utilization threshold)
        if not force:
            summary_action = self._maybe_summarize_turns()
            if summary_action and summary_action.action_taken:
                return summary_action

        metrics = self.controller.get_context_metrics()
        utilization = metrics.utilization

        # Get messages for adaptive threshold calculation
        messages = None
        if self._adaptive_enabled:
            try:
                messages = self.controller.get_messages()
            except Exception as e:
                logger.debug(f"Could not get messages for adaptive threshold: {e}")

        # Get RL recommendation if learner is available
        rl_config = None
        rl_action = None
        if self._rl_enabled and self.pruning_learner is not None:
            try:
                recommendation = self.pruning_learner.get_recommendation(
                    context_utilization=utilization,
                    tool_call_count=tool_call_count,
                    task_complexity=task_complexity,
                    provider_type=self.provider_type,
                )
                rl_action = recommendation.value
                rl_config = recommendation.metadata.get("config", {})
                self._last_rl_action = rl_action
                logger.debug(
                    f"RL recommendation: action={rl_action}, confidence={recommendation.confidence:.2f}"
                )
            except Exception as e:
                logger.warning(f"RL recommendation failed: {e}")

        # Get base threshold (adaptive > phase-aware > default)
        base_threshold = self._get_compaction_threshold(phase, messages)

        # Get effective thresholds (RL config overrides phase-aware)
        effective_threshold = (
            rl_config.get("compaction_threshold", base_threshold) if rl_config else base_threshold
        )
        effective_min_messages = (
            rl_config.get("min_messages_keep", self.config.min_messages_after_compact)
            if rl_config
            else self.config.min_messages_after_compact
        )

        # Log phase-aware threshold
        if phase and self.config.enable_phase_aware:
            logger.debug(
                f"Phase-aware compaction: phase={phase.value}, threshold={effective_threshold:.0%}"
            )

        # Determine trigger
        if force:
            trigger = CompactionTrigger.MANUAL
        elif metrics.is_overflow_risk:
            trigger = CompactionTrigger.OVERFLOW
        elif self.config.enable_proactive and utilization >= effective_threshold:
            trigger = CompactionTrigger.THRESHOLD
        else:
            return CompactionAction(
                trigger=CompactionTrigger.NONE,
                new_utilization=utilization,
            )

        if not force and not self._has_eligible_compaction_payload(
            effective_min_messages,
            metrics,
        ):
            logger.debug(
                "Compaction skipped: utilization %.1f%% exceeded threshold %.0f%% but no "
                "eligible messages or large tool payloads were available",
                utilization * 100,
                effective_threshold * 100,
            )
            return CompactionAction(
                trigger=CompactionTrigger.NONE,
                new_utilization=utilization,
                details=["No eligible compaction payload"],
            )

        logger.info(
            f"Compaction triggered: {trigger.value} (utilization: {utilization:.1%}, "
            f"threshold: {effective_threshold:.0%})"
        )

        # P1 FIX: Fast pruning before LLM compaction (OpenDev finding)
        # Replace large tool results with [pruned] markers before expensive LLM compaction
        # This reduces LLM compaction cost by 30-40%
        if self.config.enable_fast_pruning:
            try:
                from victor.agent.fast_pruning import get_fast_pruner

                # L1: when reference-aware pruning is enabled, use a config-bearing pruner that
                # preserves still-cited tool results. Default off → identical to the singleton.
                if self.config.enable_reference_aware_pruning:
                    from victor.agent.fast_pruning import FastPruner, FastPruningConfig

                    fast_pruner = FastPruner(
                        FastPruningConfig(
                            enable_reference_tracking=True,
                            reference_window_turns=self.config.reference_window_turns,
                        )
                    )
                else:
                    fast_pruner = get_fast_pruner()
                messages_before = self.controller.get_messages()

                # Estimate if fast pruning would help
                original_size, estimated_size = fast_pruner.estimate_size_reduction(
                    messages_before, current_turn=len(messages_before)
                )

                # Only perform fast pruning if we'd save at least 20%
                if original_size > 0 and (original_size - estimated_size) / original_size > 0.2:
                    pruned_messages = fast_pruner.prune_old_tool_results(
                        messages_before, current_turn=len(messages_before)
                    )

                    if fast_pruner.get_pruned_count() > 0:
                        # Update controller messages with pruned version
                        # This is done by replacing the message list
                        from victor.agent.message_history import _TrackedList

                        self.controller._history._messages = _TrackedList(pruned_messages)
                        logger.info(
                            f"[fast-pruning] Pruned {fast_pruner.get_pruned_count()} messages "
                            f"before LLM compaction (saved {original_size - estimated_size} chars)"
                        )
            except Exception as e:
                logger.debug(f"Fast pruning failed (continuing with LLM compaction): {e}")

        # Perform compaction
        chars_before = metrics.char_count
        messages_removed = self.controller.smart_compact_history(
            target_messages=effective_min_messages,
            current_query=current_query,
        )

        # Calculate results
        metrics_after = self.controller.get_context_metrics()
        chars_freed = chars_before - metrics_after.char_count
        tokens_freed = self._estimate_tokens(chars_freed)
        truncations_applied = 0

        if messages_removed == 0 and chars_freed <= 0:
            truncations_applied, fallback_chars_freed = self._truncate_large_tool_messages()
            if truncations_applied:
                metrics_after = self.controller.get_context_metrics()
                chars_freed = max(fallback_chars_freed, chars_before - metrics_after.char_count)
                tokens_freed = self._estimate_tokens(chars_freed)

        action_taken = messages_removed > 0 or truncations_applied > 0 or chars_freed > 0
        if action_taken:
            self._total_chars_freed += chars_freed
            self._total_tokens_freed += tokens_freed
            self._compaction_count += 1
            self._last_compaction_turn = len(self.controller.messages)

        details = [
            f"Removed {messages_removed} messages",
            f"Freed ~{tokens_freed} tokens",
            f"New utilization: {metrics_after.utilization:.1%}",
        ]
        if truncations_applied:
            details.append(f"Truncated {truncations_applied} large tool outputs")
        if rl_action:
            details.append(f"RL action: {rl_action}")

        action = CompactionAction(
            trigger=trigger,
            messages_removed=messages_removed,
            chars_freed=chars_freed,
            tokens_freed=tokens_freed,
            truncations_applied=truncations_applied,
            new_utilization=metrics_after.utilization,
            details=details,
        )

        # Emit compaction event if event bus available
        if self._event_bus and messages_removed > 0:
            try:
                from victor.core.events.emit_helper import emit_event_sync

                emit_event_sync(
                    self._event_bus,
                    "context.compacted",
                    {
                        "messages_removed": messages_removed,
                        "chars_freed": chars_freed,
                        "trigger": trigger.value,
                        "summary": "; ".join(details),
                    },
                )
            except Exception as e:
                logger.debug(f"Failed to emit compaction event: {e}")

        log_method = logger.info if action_taken else logger.debug
        log_method(
            f"Compaction complete: {messages_removed} messages removed, "
            f"{truncations_applied} tool outputs truncated, ~{tokens_freed} tokens freed"
        )

        return action

    def _truncate_large_tool_messages(self) -> Tuple[int, int]:
        """Fallback compaction for short histories dominated by large tool outputs."""
        try:
            messages = self.controller.get_messages()
        except Exception as exc:
            logger.debug("Could not fetch messages for fallback truncation: %s", exc)
            return 0, 0

        truncations = 0
        chars_freed = 0
        max_chars = max(1024, self.config.tool_result_max_chars)
        max_lines = max(40, self.config.tool_result_max_lines)

        for message in messages:
            if getattr(message, "role", None) != "tool":
                continue
            content = getattr(message, "content", "")
            if not isinstance(content, str):
                continue
            if len(content) <= max_chars and content.count("\n") + 1 <= max_lines:
                continue

            truncated = self.truncate_tool_result(
                content,
                max_chars=max_chars,
                max_lines=max_lines,
                content_type="mixed",
            )
            if not truncated.truncated:
                continue
            message.content = truncated.content
            truncations += 1
            chars_freed += truncated.truncated_chars

        if truncations:
            logger.info(
                "Fallback compaction truncated %d large tool outputs, freed ~%d chars",
                truncations,
                chars_freed,
            )
        return truncations, chars_freed

    def record_task_outcome(self, task_success: bool, tokens_saved: int = 0) -> None:
        """Record task outcome for RL learning.

        Call this after a task completes to update the RL learner with
        the outcome of the most recent pruning decision.

        Args:
            task_success: Whether the task completed successfully
            tokens_saved: Estimated tokens saved by pruning
        """
        if not self._rl_enabled or self.pruning_learner is None:
            return

        if self._last_rl_action is None:
            return  # No pruning decision was made

        try:
            # Get current state for recording
            metrics = self.controller.get_context_metrics() if self.controller else None
            utilization = metrics.utilization if metrics else 0.5

            self.pruning_learner.record_outcome(
                context_utilization=utilization,
                tool_call_count=0,  # Will be refined later
                action=self._last_rl_action,
                task_success=task_success,
                tokens_saved=tokens_saved,
                task_complexity="medium",
                provider_type=self.provider_type,
            )
            self._last_task_success = task_success
            logger.debug(
                f"Recorded RL outcome: action={self._last_rl_action}, "
                f"success={task_success}, tokens_saved={tokens_saved}"
            )
        except Exception as e:
            logger.warning(f"Failed to record RL outcome: {e}")
        finally:
            self._last_rl_action = None

    def should_compact(
        self,
        task_complexity: Optional[str] = None,
    ) -> Tuple[bool, CompactionTrigger]:
        """Check if compaction should be performed.

        Args:
            task_complexity: Optional task complexity string ("high", "medium", "low").
                When "high", the effective proactive threshold is reduced by
                ``config.high_complexity_threshold_factor`` so high-complexity
                tasks compact earlier (their context fills faster).

        Returns:
            Tuple of (should_compact, trigger_reason)
        """
        metrics = self.controller.get_context_metrics()

        if metrics.is_overflow_risk:
            return True, CompactionTrigger.OVERFLOW

        if self.config.enable_proactive:
            effective_threshold = self.config.proactive_threshold
            if task_complexity == "high":
                effective_threshold *= self.config.high_complexity_threshold_factor
            if metrics.utilization >= effective_threshold:
                return True, CompactionTrigger.THRESHOLD

        return False, CompactionTrigger.NONE

    # ========================================================================
    # Tool Result Truncation
    # ========================================================================

    def truncate_tool_result(
        self,
        content: str,
        max_chars: Optional[int] = None,
        max_lines: Optional[int] = None,
        content_type: str = "mixed",
    ) -> TruncationResult:
        """Intelligently truncate a tool result.

        Uses content-aware truncation that preserves important patterns
        like error messages, file paths, and code structure.

        Args:
            content: The tool result content
            max_chars: Maximum characters (default: config value)
            max_lines: Maximum lines (default: config value)
            content_type: Type of content (code, json, prose, mixed)

        Returns:
            TruncationResult with the (possibly) truncated content
        """
        if not self.config.enable_tool_truncation:
            return TruncationResult(
                content=content,
                original_chars=len(content),
                truncated_chars=0,
                truncated=False,
            )

        max_chars = max_chars or self.config.tool_result_max_chars
        max_lines = max_lines or self.config.tool_result_max_lines
        original_len = len(content)

        # Check if truncation needed
        lines = content.split("\n")
        needs_truncation = len(content) > max_chars or len(lines) > max_lines

        if not needs_truncation:
            return TruncationResult(
                content=content,
                original_chars=original_len,
                truncated_chars=0,
                truncated=False,
            )

        # Apply truncation strategy
        strategy = self.config.truncation_strategy

        if strategy == TruncationStrategy.SMART:
            truncated = self._smart_truncate(content, max_chars, max_lines, content_type)
        elif strategy == TruncationStrategy.HEAD:
            truncated = self._head_truncate(content, max_chars, max_lines)
        elif strategy == TruncationStrategy.TAIL:
            truncated = self._tail_truncate(content, max_chars, max_lines)
        else:  # BOTH
            truncated = self._both_truncate(content, max_chars, max_lines)

        return TruncationResult(
            content=truncated,
            original_chars=original_len,
            truncated_chars=original_len - len(truncated),
            truncated=True,
            indicator="[output truncated]",
        )

    def _smart_truncate(
        self,
        content: str,
        max_chars: int,
        max_lines: int,
        content_type: str,
    ) -> str:
        """Content-aware smart truncation.

        Preserves:
        - Error messages and tracebacks
        - File paths mentioned in the content
        - Code block structure
        - JSON structure (if applicable)
        """
        lines = content.split("\n")

        # Handle single-line content specially
        if len(lines) <= 2 and len(content) > max_chars:
            # For single/double line content, just truncate by chars
            half = max_chars // 2 - 20  # Leave room for marker
            return f"{content[:half]}\n... [content truncated] ...\n{content[-half:]}"

        # Priority lines (errors, paths, important markers)
        priority_indices: set = set()
        for i, line in enumerate(lines):
            if self.ERROR_PATTERN.search(line):
                priority_indices.add(i)
                # Also include lines around errors
                if i > 0:
                    priority_indices.add(i - 1)
                if i < len(lines) - 1:
                    priority_indices.add(i + 1)
            elif self.PATH_PATTERN.search(line):
                priority_indices.add(i)

        # Detect code blocks to preserve
        if self.config.preserve_code_blocks:
            code_blocks = list(self.CODE_BLOCK_PATTERN.finditer(content))
            if code_blocks:
                # Mark first code block lines as priority
                first_block = code_blocks[0]
                start_line = content[: first_block.start()].count("\n")
                end_line = content[: first_block.end()].count("\n")
                for i in range(start_line, min(end_line + 1, start_line + 20)):
                    priority_indices.add(i)

        # Build truncated output
        result_lines: List[str] = []
        char_count = 0
        half_lines = max(1, max_lines // 2)

        # Keep first portion
        for _i, line in enumerate(lines[:half_lines]):
            if char_count + len(line) > max_chars // 2:
                # Truncate this line if it's too long
                remaining = max_chars // 2 - char_count
                if remaining > 50:
                    result_lines.append(line[:remaining] + "...")
                break
            result_lines.append(line)
            char_count += len(line) + 1

        # Add truncation marker
        truncated_count = len(lines) - half_lines * 2
        if truncated_count > 0:
            result_lines.append(f"\n... [{truncated_count} lines truncated] ...\n")

        # Keep last portion and any priority lines in between
        remaining_chars = max_chars - char_count - 50  # Leave room for marker
        remaining_lines: List[str] = []

        for i, line in enumerate(lines[-half_lines:]):
            actual_idx = len(lines) - half_lines + i
            if remaining_chars <= 0 and actual_idx not in priority_indices:
                continue
            remaining_lines.append(line)
            remaining_chars -= len(line) + 1

        result_lines.extend(remaining_lines)

        return "\n".join(result_lines)

    def _head_truncate(self, content: str, max_chars: int, max_lines: int) -> str:
        """Keep the beginning of the content."""
        lines = content.split("\n")
        result_lines = lines[:max_lines]
        result = "\n".join(result_lines)

        if len(result) > max_chars:
            result = result[:max_chars]

        return result + "\n... [output truncated]"

    def _tail_truncate(self, content: str, max_chars: int, max_lines: int) -> str:
        """Keep the end of the content."""
        lines = content.split("\n")
        result_lines = lines[-max_lines:]
        result = "\n".join(result_lines)

        if len(result) > max_chars:
            result = result[-max_chars:]

        return "[output truncated] ...\n" + result

    def _both_truncate(self, content: str, max_chars: int, max_lines: int) -> str:
        """Keep beginning and end."""
        lines = content.split("\n")
        half = max_lines // 2
        half_chars = max_chars // 2

        head = "\n".join(lines[:half])
        if len(head) > half_chars:
            head = head[:half_chars]

        tail = "\n".join(lines[-half:])
        if len(tail) > half_chars:
            tail = tail[-half_chars:]

        truncated = len(lines) - max_lines
        return f"{head}\n\n... [{truncated} lines truncated] ...\n\n{tail}"

    # ========================================================================
    # Phase-Aware Compaction
    # ========================================================================

    def _get_compaction_threshold(
        self,
        phase: Optional[TaskPhase] = None,
        messages: Optional[List[Any]] = None,
    ) -> float:
        """Get the compaction threshold for the current phase.

        Priority order:
        1. Adaptive threshold (if enabled and messages available)
        2. Phase-aware threshold (if enabled and phase provided)
        3. Default threshold from config

        Args:
            phase: Current task phase
            messages: Optional list of messages for adaptive threshold calculation

        Returns:
            Compaction threshold (0.0-1.0)
        """
        # Try adaptive threshold first (if enabled and messages available)
        if self._adaptive_enabled and messages:
            try:
                adaptive_threshold = self._adaptive_threshold.calculate_threshold(messages)
                logger.debug(
                    f"Using adaptive threshold: {adaptive_threshold:.0%} "
                    f"(pattern: {self._adaptive_threshold.get_current_analysis().pattern.value if self._adaptive_threshold.get_current_analysis() else 'unknown'})"
                )
                return adaptive_threshold
            except Exception as e:
                logger.warning(f"Adaptive threshold calculation failed, falling back: {e}")

        # Use phase-aware threshold if enabled and phase is provided
        if self.config.enable_phase_aware and phase and phase in self.config.phase_thresholds:
            return self.config.phase_thresholds[phase]

        # Fall back to default threshold
        return self.config.proactive_threshold

    # ========================================================================
    # Token Estimation
    # ========================================================================

    def _estimate_tokens(
        self,
        chars: int,
        content_type: str = "mixed",
    ) -> int:
        """Estimate token count from character count.

        Uses content-type-specific factors for more accurate estimation.

        Args:
            chars: Character count
            content_type: Type of content

        Returns:
            Estimated token count
        """
        factor = self.TOKEN_FACTORS.get(content_type, self.TOKEN_FACTORS["mixed"])
        return int(chars / factor)

    def estimate_message_tokens(self, message: Message) -> int:
        """Estimate tokens for a message.

        Args:
            message: The message to estimate

        Returns:
            Estimated token count
        """
        content = message.content

        # Detect content type
        if "```" in content:
            content_type = "code"
        elif content.strip().startswith("{") or content.strip().startswith("["):
            content_type = "json"
        else:
            content_type = "prose"

        # Add overhead for role/structure
        overhead = 4  # Typical message overhead

        return self._estimate_tokens(len(content), content_type) + overhead

    # ========================================================================
    # Statistics and Metrics
    # ========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Get compactor statistics.

        Returns:
            Dictionary with compaction statistics
        """
        metrics = self.controller.get_context_metrics()

        stats = {
            "current_utilization": metrics.utilization,
            "current_chars": metrics.char_count,
            "current_messages": metrics.message_count,
            "compaction_count": self._compaction_count,
            "total_chars_freed": self._total_chars_freed,
            "total_tokens_freed": self._total_tokens_freed,
            "last_compaction_turn": self._last_compaction_turn,
            "proactive_threshold": self.config.proactive_threshold,
            "proactive_enabled": self.config.enable_proactive,
            "truncation_enabled": self.config.enable_tool_truncation,
            "adaptive_enabled": self._adaptive_enabled,
        }

        # Add adaptive threshold info if enabled
        if self._adaptive_enabled and self._adaptive_threshold:
            analysis = self._adaptive_threshold.get_current_analysis()
            if analysis:
                stats.update(
                    {
                        "adaptive_pattern": analysis.pattern.value,
                        "adaptive_confidence": analysis.confidence,
                        "adaptive_topic_coherence": analysis.topic_coherence,
                        "adaptive_context_dependency": analysis.context_dependency,
                        "adaptive_threshold": analysis.recommended_threshold,
                        "adaptive_reasoning": analysis.reasoning,
                    }
                )

        return stats

    def get_compaction_history(self) -> List[str]:
        """Get summaries of past compactions.

        Returns:
            List of compaction summary strings
        """
        return self.controller.get_compaction_summaries()

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def reset_statistics(self) -> None:
        """Reset compaction statistics."""
        self._total_chars_freed = 0
        self._total_tokens_freed = 0
        self._compaction_count = 0
        self._last_compaction_turn = 0
        logger.debug("Compactor statistics reset")

    # ========================================================================
    # Adaptive Threshold Management
    # ========================================================================

    def set_adaptive_threshold(self, adaptive_threshold: "AdaptiveCompactionThreshold") -> None:
        """Set or update the adaptive threshold system.

        If a ``DomainKeywordsProviderProtocol`` is registered in the capability
        registry, its keywords are automatically forwarded to the threshold so
        plugin-contributed domains participate in topic-switch detection.

        Args:
            adaptive_threshold: AdaptiveCompactionThreshold instance
        """
        global AdaptiveCompactionThreshold
        if AdaptiveCompactionThreshold is None:
            from victor.agent.adaptive_compaction import (
                AdaptiveCompactionThreshold as ACT,
            )

            AdaptiveCompactionThreshold = ACT

        # Forward any plugin-contributed domain keywords to the threshold.
        try:
            from victor.core.capability_registry import CapabilityRegistry
            from victor.framework.protocols import DomainKeywordsProviderProtocol

            registry = CapabilityRegistry.get_instance()
            provider = registry.get(DomainKeywordsProviderProtocol)
            if provider is not None:
                extra = provider.get_domain_keywords()
                adaptive_threshold.set_extra_domain_keywords(extra)
                logger.debug(
                    "Forwarded domain keywords from provider '%s' to adaptive threshold",
                    getattr(provider, "get_provider_name", lambda: repr(provider))(),
                )
        except Exception as exc:
            logger.debug("Domain keyword provider lookup failed (non-fatal): %s", exc)

        self._adaptive_threshold = adaptive_threshold
        self._adaptive_enabled = True
        logger.info("Adaptive threshold system enabled")

    def disable_adaptive_threshold(self) -> None:
        """Disable adaptive threshold and use static thresholds."""
        self._adaptive_enabled = False
        logger.info("Adaptive threshold system disabled")

    def get_adaptive_analysis(self) -> Optional[Any]:
        """Get the most recent adaptive threshold analysis.

        Returns:
            PatternAnalysis from adaptive system, or None if not enabled
        """
        if self._adaptive_enabled and self._adaptive_threshold:
            return self._adaptive_threshold.get_current_analysis()
        return None

    def get_adaptive_threshold_history(self) -> List[Dict[str, Any]]:
        """Get history of adaptive threshold calculations.

        Returns:
            List of dicts with timestamp, threshold, and pattern
        """
        if self._adaptive_enabled and self._adaptive_threshold:
            return self._adaptive_threshold.get_threshold_history()
        return []

    # =========================================================================
    # Async API
    # =========================================================================

    def _ensure_async_state(self) -> None:
        """Initialize async-related state if not already done."""
        if not hasattr(self, "_monitor_task"):
            import asyncio

            self._monitor_task: Optional[asyncio.Task] = None
            self._async_running = False
            self._check_interval_seconds = 30.0
            self._last_check_time = 0.0
            self._async_compactions = 0
            self._background_checks = 0

    async def check_and_compact_async(
        self,
        current_query: Optional[str] = None,
        force: bool = False,
        tool_call_count: int = 0,
        task_complexity: str = "medium",
        phase: Optional[TaskPhase] = None,
    ) -> CompactionAction:
        """Check and compact asynchronously.

        Runs the sync compaction in a thread pool to avoid blocking.

        Args:
            current_query: Current user query for relevance
            force: Force compaction
            tool_call_count: Number of tool calls made (for RL state)
            task_complexity: Task complexity level (for RL state)
            phase: Current task phase for phase-aware thresholds

        Returns:
            CompactionAction describing what was done
        """
        import asyncio

        if self._adaptive_enabled:
            try:
                self.controller.get_messages()
            except Exception as e:
                logger.debug(f"Could not get messages for adaptive threshold: {e}")

        self._ensure_async_state()
        import asyncio

        self._ensure_async_state()
        action = await asyncio.to_thread(
            self.check_and_compact,
            current_query,
            force,
            tool_call_count,
            task_complexity,
            phase,
        )

        if action.action_taken:
            self._async_compactions += 1

        return action

    async def should_compact_async(self) -> Tuple[bool, CompactionTrigger]:
        """Check if compaction should be performed (async).

        Returns:
            Tuple of (should_compact, trigger_reason)
        """
        import asyncio

        return await asyncio.to_thread(self.should_compact)

    async def truncate_tool_result_async(
        self,
        content: str,
        max_chars: Optional[int] = None,
        max_lines: Optional[int] = None,
        content_type: str = "mixed",
    ) -> TruncationResult:
        """Truncate tool result asynchronously.

        Args:
            content: Tool result content
            max_chars: Max characters
            max_lines: Max lines
            content_type: Content type

        Returns:
            TruncationResult
        """
        import asyncio

        return await asyncio.to_thread(
            self.truncate_tool_result,
            content,
            max_chars,
            max_lines,
            content_type,
        )

    async def start_background_compaction(
        self,
        interval_seconds: Optional[float] = None,
    ) -> None:
        """Start background task that monitors and compacts proactively.

        Args:
            interval_seconds: Override check interval (default 30.0)
        """
        import asyncio
        import time

        self._ensure_async_state()

        if self._async_running:
            return

        if interval_seconds is not None:
            self._check_interval_seconds = interval_seconds

        self._async_running = True

        async def _monitor_loop() -> None:
            while self._async_running:
                await asyncio.sleep(self._check_interval_seconds)
                if not self._async_running:
                    break

                try:
                    self._background_checks += 1
                    should, trigger = await self.should_compact_async()

                    if should and trigger != CompactionTrigger.NONE:
                        logger.debug(f"Background compaction triggered: {trigger.value}")
                        await self.check_and_compact_async()

                    self._last_check_time = time.time()

                except Exception as e:
                    logger.warning(f"Background compaction error: {e}")

        self._monitor_task = asyncio.create_task(_monitor_loop())
        logger.debug(
            f"ContextCompactor: Started background monitoring "
            f"(interval={self._check_interval_seconds}s)"
        )

    async def stop_background_compaction(self) -> None:
        """Stop background compaction monitoring."""
        import asyncio

        self._ensure_async_state()
        self._async_running = False

        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None

        logger.debug("ContextCompactor: Stopped background monitoring")

    def get_async_statistics(self) -> Dict[str, Any]:
        """Get async compactor statistics.

        Returns:
            Dict with both sync and async statistics
        """
        self._ensure_async_state()
        base_stats = self.get_statistics()
        base_stats.update(
            {
                "async_compactions": self._async_compactions,
                "background_checks": self._background_checks,
                "background_running": self._async_running,
                "last_check_time": self._last_check_time,
            }
        )
        return base_stats


def create_context_compactor(
    controller: "ConversationController",
    proactive_threshold: float = 0.50,  # Changed from 0.90 - balanced default
    min_messages_after_compact: int = 8,
    tool_result_max_chars: int = 8192,
    tool_result_max_lines: int = 200,
    truncation_strategy: TruncationStrategy = TruncationStrategy.SMART,
    preserve_code_blocks: bool = True,
    enable_proactive: bool = True,
    enable_tool_truncation: bool = True,
    pruning_learner: Optional["ContextPruningLearner"] = None,
    provider_type: str = "cloud",
    runtime_intelligence: Optional[Any] = None,
) -> ContextCompactor:
    """Factory function to create a configured ContextCompactor.

    Args:
        controller: The ConversationController to wrap
        proactive_threshold: Utilization % to trigger proactive compaction (default: 0.90)
        min_messages_after_compact: Minimum messages to keep after compaction (default: 8)
        tool_result_max_chars: Maximum characters for tool results (default: 8192)
        tool_result_max_lines: Maximum lines for tool results (default: 200)
        truncation_strategy: Strategy for truncating tool results (default: SMART)
        preserve_code_blocks: Preserve code blocks during truncation (default: True)
        enable_proactive: Enable proactive compaction (default: True)
        enable_tool_truncation: Enable tool result truncation (default: True)
        pruning_learner: Optional RL learner for adaptive pruning decisions
        provider_type: Provider type for RL state (cloud, local)
        runtime_intelligence: Optional canonical runtime-intelligence service

    Returns:
        Configured ContextCompactor instance
    """
    config = CompactorConfig(
        proactive_threshold=proactive_threshold,
        min_messages_after_compact=min_messages_after_compact,
        tool_result_max_chars=tool_result_max_chars,
        tool_result_max_lines=tool_result_max_lines,
        truncation_strategy=truncation_strategy,
        preserve_code_blocks=preserve_code_blocks,
        enable_proactive=enable_proactive,
        enable_tool_truncation=enable_tool_truncation,
    )
    return ContextCompactor(
        controller=controller,
        config=config,
        pruning_learner=pruning_learner,
        provider_type=provider_type,
        runtime_intelligence=runtime_intelligence,
    )
