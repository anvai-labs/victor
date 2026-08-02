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

"""Service-owned runtime for task coordination.

This module hosts the canonical TaskCoordinator implementation on the
service-first side of the architecture.

The legacy ``victor.agent.task_coordinator`` module now re-exports this
implementation for compatibility.
"""

import logging
from typing import Any, Optional, TYPE_CHECKING

from victor.agent.conversation.history_metadata import build_internal_history_metadata
from victor.core.events import ObservabilityBus
from victor.core.events.emit_helper import emit_event_sync

if TYPE_CHECKING:
    from victor.agent.task_analyzer import TaskAnalyzer
    from victor.agent.unified_task_tracker import UnifiedTaskTracker
    from victor.agent.conversation.controller import ConversationController
    from victor.agent.prompt_builder import SystemPromptBuilder
    from victor.config.settings import Settings
    from victor.agent.task_analyzer import TaskType

logger = logging.getLogger(__name__)


class TaskCoordinator:
    """Coordinates task preparation, intent detection, and guidance.

    This component provides a semantic interface for task coordination,
    consolidating methods that were previously in AgentOrchestrator.

    Architecture:
    - TaskCoordinator: High-level coordinator for task operations
    - TaskAnalyzer: Complexity classification and intent detection
    - UnifiedTaskTracker: Tool budget management
    - ConversationController: Message injection
    - PromptBuilder: Task hint retrieval

    Responsibilities:
    - Prepare tasks with complexity detection
    - Apply intent-based prompt guards
    - Provide task-specific guidance
    - Adjust tool budgets based on complexity

    Design Pattern:
    - Coordinator/Facade: Simplifies task coordination interface
    - Delegation: Delegates to TaskAnalyzer, UnifiedTaskTracker, etc.

    Extracted from CRITICAL-001 Phase 2D.
    """

    def __init__(
        self,
        task_analyzer: "TaskAnalyzer",
        unified_tracker: "UnifiedTaskTracker",
        prompt_builder: "SystemPromptBuilder",
        settings: "Settings",
        event_bus: Optional[ObservabilityBus] = None,
    ):
        """Initialize TaskCoordinator.

        Args:
            task_analyzer: Task complexity classifier and intent detector
            unified_tracker: Tool budget and iteration tracker
            prompt_builder: System prompt builder for task hints
            settings: Application settings
            event_bus: Optional ObservabilityBus instance. If None, uses DI container.
        """
        self.task_analyzer = task_analyzer
        self._event_bus = event_bus or self._get_default_bus()
        self.unified_tracker = unified_tracker
        self.prompt_builder = prompt_builder
        self.settings = settings
        self._current_intent = None
        self._current_task_classification = None
        self._temperature = getattr(settings, "temperature", 0.7)
        self._tool_budget = getattr(settings, "tool_budget", 15)
        self._observed_files = []
        self._reminder_manager = None

    def _get_default_bus(self) -> Optional[ObservabilityBus]:
        """Get default ObservabilityBus from DI container.

        Returns:
            ObservabilityBus instance or None if unavailable
        """
        try:
            from victor.core.events import get_observability_bus

            return get_observability_bus()
        except Exception:
            return None

    # =====================================================================
    # Task Preparation
    # =====================================================================

    @staticmethod
    def _is_write_remediation_followup(
        user_message: str,
        *,
        intent_result: Any,
    ) -> bool:
        """Return whether a write-authorized follow-up is asking to address prior findings."""
        from victor.agent.action_authorizer import (
            ActionIntent,
            split_continuation_request,
        )

        if getattr(intent_result, "intent", None) != ActionIntent.WRITE_ALLOWED:
            return False

        is_continuation, continuation_payload = split_continuation_request(user_message)
        candidate = (continuation_payload if is_continuation else user_message).strip().lower()
        if not candidate:
            return False

        remediation_terms = (
            "adress",
            "address",
            "apply",
            "fix",
            "handle",
            "implement",
            "improve",
            "remediate",
            "resolve",
            "tackle",
        )
        target_terms = (
            "feedback",
            "finding",
            "findings",
            "gap",
            "gaps",
            "bug",
            "bugs",
            "issue",
            "issues",
            "recommendation",
            "recommendations",
            "suggestion",
            "suggestions",
        )

        return any(term in candidate for term in remediation_terms) and any(
            term in candidate for term in target_terms
        )

    def prepare_task(
        self,
        user_message: str,
        unified_task_type: "TaskType",
        conversation_controller: "ConversationController",
    ) -> tuple[Any, int]:
        """Prepare task-specific guidance and budget adjustments.

        Uses granular task classification for hint lookup, then falls back
        to unified task type if needed. Adjusts tool budget based on
        task complexity.

        Args:
            user_message: The user's input message
            unified_task_type: Unified task type classification
            conversation_controller: Conversation controller for message injection

        Returns:
            Tuple of (task_classification, complexity_tool_budget)
        """
        from victor.agent.prompt_builder import get_task_type_hint
        from victor.storage.embeddings.task_classifier import TaskTypeClassifier
        from victor.agent.action_authorizer import ActionIntent, split_continuation_request
        from victor.framework.task import (
            TaskClassification,
            TaskComplexity,
            DEFAULT_BUDGETS,
        )
        from victor.framework.task.direct_response import (
            classify_direct_response_prompt,
        )

        direct_response = classify_direct_response_prompt(user_message)

        # Get granular task type for more specific hints
        granular_task_type = None
        try:
            classifier = TaskTypeClassifier.get_instance()
            if classifier._initialized:
                result = classifier.classify_sync(user_message)
                granular_task_type = result.task_type.value
        except Exception:
            pass  # Fall back to unified type

        # Try granular type first (e.g., "refactor", "bug_fix", "documentation")
        task_hint = None
        hint_source = None
        if granular_task_type:
            task_hint = get_task_type_hint(
                granular_task_type,
                prompt_contributors=self.prompt_builder.prompt_contributors,
            )
            if task_hint:
                hint_source = granular_task_type

        # Fall back to unified type if no granular hint found
        if not task_hint:
            task_hint = get_task_type_hint(
                unified_task_type.value,
                prompt_contributors=self.prompt_builder.prompt_contributors,
            )
            if task_hint:
                hint_source = unified_task_type.value

        if task_hint and not direct_response.is_direct_response:
            conversation_controller.add_message(
                "user",
                f"[TASK-HINT: {task_hint.strip()}]",
                metadata=build_internal_history_metadata("task_hint"),
            )
            logger.debug(f"Injected task hint for task type: {hint_source}")
        elif task_hint:
            logger.debug("Skipping task hint injection for direct-response prompt")

        # Classify task complexity and adjust tool budget
        task_classification = self.task_analyzer.classify_complexity(user_message)
        complexity_tool_budget = DEFAULT_BUDGETS.get(task_classification.complexity, 15)

        # Continuation carry-forward: a bare "continue"-style message describes no
        # task of its own, so classifying its text (usually → SIMPLE) would slash
        # the budget of the very work being continued (seen live: 200 → 20 on
        # "continue next steps", starving the follow-up turn). Mirror the intent
        # carry-forward in apply_intent_guard: a continuation may keep or raise
        # the prior turn's budget, never shrink it.
        is_continuation, _continuation_payload = split_continuation_request(user_message)
        previous_classification = self._current_task_classification
        if is_continuation and previous_classification is not None:
            prior_budget = DEFAULT_BUDGETS.get(
                previous_classification.complexity, complexity_tool_budget
            )
            if prior_budget > complexity_tool_budget:
                matched_patterns = list(getattr(task_classification, "matched_patterns", []) or [])
                task_classification = TaskClassification(
                    complexity=previous_classification.complexity,
                    tool_budget=prior_budget,
                    confidence=max(
                        float(getattr(task_classification, "confidence", 0.0)),
                        float(getattr(previous_classification, "confidence", 0.0)),
                    ),
                    matched_patterns=[
                        "continuation_complexity_carry_forward",
                        *matched_patterns,
                    ],
                )
                complexity_tool_budget = prior_budget
                logger.info(
                    "Continuation request detected; carrying forward prior task "
                    "complexity %s (tool budget %d)",
                    previous_classification.complexity.value,
                    prior_budget,
                )

        task_type_value = str(getattr(unified_task_type, "value", unified_task_type))
        if task_classification.complexity == TaskComplexity.SIMPLE:
            intent_result = self.task_analyzer.detect_intent(user_message)
            remediation_followup = self._is_write_remediation_followup(
                user_message,
                intent_result=intent_result,
            )
            should_promote_for_action = (
                task_type_value in {"edit", "create", "create_simple"} or remediation_followup
            )
            if intent_result.intent == ActionIntent.WRITE_ALLOWED and should_promote_for_action:
                matched_patterns = list(getattr(task_classification, "matched_patterns", []) or [])
                if task_type_value in {"edit", "create", "create_simple"}:
                    matched_patterns.append("write_intent_task_shape")
                if remediation_followup:
                    matched_patterns.append("write_intent_remediation_followup")
                task_classification = TaskClassification(
                    complexity=TaskComplexity.ACTION,
                    tool_budget=DEFAULT_BUDGETS.get(TaskComplexity.ACTION, 50),
                    confidence=max(
                        float(getattr(task_classification, "confidence", 0.0)),
                        float(getattr(intent_result, "confidence", 0.0)),
                        0.85,
                    ),
                    matched_patterns=matched_patterns,
                )
                complexity_tool_budget = task_classification.tool_budget
                logger.info(
                    "Promoted simple task complexity to action based on explicit write intent "
                    "for a write-oriented implementation follow-up"
                )

        if task_classification.complexity == TaskComplexity.SIMPLE:
            # Override with simpler budget for simple tasks
            current_max = self.unified_tracker.config.get("max_total_iterations", 50)
            new_max = min(complexity_tool_budget, current_max)
            self.unified_tracker.set_tool_budget(new_max)
            logger.info(
                f"Task complexity: {task_classification.complexity.value}, "
                f"adjusted max_iterations to {complexity_tool_budget}"
            )
        elif task_classification.complexity == TaskComplexity.GENERATION:
            # Generation tasks should complete in 1-2 tool calls
            current_max = self.unified_tracker.config.get("max_total_iterations", 50)
            new_max = min(complexity_tool_budget + 1, current_max)
            self.unified_tracker.set_tool_budget(new_max)
            logger.info(
                f"Generation task detected, limiting iterations to {complexity_tool_budget + 1}"
            )
        elif task_classification.complexity in (
            TaskComplexity.ACTION,
            TaskComplexity.COMPLEX,
        ):
            # Action/complex tasks (edit, refactor, multi-step) need extra iterations
            # for read-modify-test cycles. Bump iteration cap by 1.5x without
            # touching tool_budget (user-set budgets remain sticky).
            try:
                current_max = int(self.unified_tracker.config.get("max_total_iterations", 50) or 50)
            except (TypeError, ValueError):
                current_max = 50
            bumped_max = int(current_max * 1.5)
            set_max = getattr(self.unified_tracker, "set_max_iterations", None)
            if bumped_max > current_max and callable(set_max):
                try:
                    set_max(bumped_max, user_override=False)
                    logger.info(
                        f"Action/complex task: raised max_iterations from {current_max} "
                        f"to {bumped_max} for read-modify-test cycles"
                    )
                except Exception as exc:
                    logger.warning(
                        f"Failed to raise max_iterations ({type(exc).__name__}: {exc}); "
                        f"continuing with current_max={current_max}"
                    )
            else:
                logger.info(
                    f"Task complexity: {task_classification.complexity.value}, "
                    f"max_iterations unchanged ({current_max})"
                )
        else:
            logger.info(
                f"Task complexity: {task_classification.complexity.value}, "
                f"confidence: {task_classification.confidence:.2f}"
            )

        # Update reminder manager with task complexity and hint
        if self._reminder_manager:
            from victor.framework.enrichment.strategies import get_complexity_hint

            self._reminder_manager.update_state(
                task_complexity=task_classification.complexity.value,
                task_hint=get_complexity_hint(task_classification.complexity),
                tool_budget=complexity_tool_budget,
            )

        # Emit STATE event for task preparation
        if self._event_bus:
            emit_event_sync(
                self._event_bus,
                topic="task.prepared",
                data={
                    "unified_task_type": unified_task_type.value,
                    "complexity": task_classification.complexity.value,
                    "confidence": task_classification.confidence,
                    "tool_budget": complexity_tool_budget,
                    "category": "state",  # Preserve for observability
                },
                source="TaskCoordinator",
            )

        # Remember the classification so a follow-up continuation turn can carry
        # it forward instead of re-classifying the bare "continue" text.
        self._current_task_classification = task_classification

        return task_classification, complexity_tool_budget

    # =====================================================================
    # Intent Detection
    # =====================================================================

    def apply_intent_guard(
        self, user_message: str, conversation_controller: "ConversationController"
    ) -> None:
        """Detect intent and inject prompt guards for read-only tasks.

        Analyzes the user's message to determine their intent (DISPLAY_ONLY,
        READ_ONLY, WRITE_ALLOWED, etc.) and injects appropriate prompt guards
        to prevent unwanted write operations.

        Args:
            user_message: The user's input message
            conversation_controller: Conversation controller for message injection
        """
        from victor.agent.action_authorizer import (
            ActionIntent,
            IntentClassification,
            PROMPT_GUARDS,
            SAFE_ACTIONS,
            has_explicit_readonly_shell_request,
            split_continuation_request,
        )
        from victor.framework.task.direct_response import (
            classify_direct_response_prompt,
        )

        # Reset per-turn guidance up front: the prompt builder is session-scoped, so
        # without this the intent/db guidance from prior turns would accumulate via the
        # `existing + db_hint` aggregation below and leak across turns.
        self.prompt_builder.contextual_guidance = None

        previous_intent = self._current_intent
        intent_result = self.task_analyzer.detect_intent(user_message)
        is_continuation, continuation_payload = split_continuation_request(user_message)
        direct_response = classify_direct_response_prompt(user_message)

        if (
            is_continuation
            and isinstance(previous_intent, ActionIntent)
            # Never carry forward DISPLAY_ONLY on a bare "continue" — the original task
            # may include action steps (commit, write, etc.) and the guard would block them.
            # DISPLAY_ONLY is a per-turn signal, not a session-level constraint.
            and previous_intent != ActionIntent.DISPLAY_ONLY
            and (
                not continuation_payload
                or (
                    intent_result.intent in (ActionIntent.DISPLAY_ONLY, ActionIntent.AMBIGUOUS)
                    and getattr(intent_result, "confidence", 0.0) <= 0.3
                    and all(
                        signal == "continuation_with_payload"
                        for signal in list(getattr(intent_result, "matched_signals", []) or [])
                    )
                )
            )
        ):
            carry_forward_signal = "continuation_carry_forward"
            prior_matches = list(getattr(intent_result, "matched_signals", []) or [])
            intent_result = IntentClassification(
                intent=previous_intent,
                confidence=max(getattr(intent_result, "confidence", 0.0), 0.85),
                matched_signals=[carry_forward_signal, *prior_matches],
                safe_actions=SAFE_ACTIONS[previous_intent].copy(),
                prompt_guard=PROMPT_GUARDS[previous_intent],
            )
            logger.info(
                "Continuation request detected; carrying forward prior intent: %s",
                previous_intent.value,
            )

        self._current_intent = intent_result.intent

        if intent_result.intent in (ActionIntent.DISPLAY_ONLY, ActionIntent.READ_ONLY):
            if direct_response.is_direct_response:
                logger.info(
                    "Intent: %s, skipping prompt guard for direct-response prompt",
                    intent_result.intent.value,
                )
            elif intent_result.prompt_guard:
                # NEW: Inject via system prompt instead of user message (Phase 3D Optimization)
                self.prompt_builder.contextual_guidance = (
                    f"INTENT CONSTRAINT: {intent_result.prompt_guard.strip()}"
                )
                logger.info(f"Intent: {intent_result.intent.value}, applied contextual guidance")
        elif intent_result.intent == ActionIntent.WRITE_ALLOWED:
            logger.info("Intent: write_allowed, no prompt guard needed")

        if has_explicit_readonly_shell_request(user_message):
            # NEW: Aggregate with existing guidance
            existing = self.prompt_builder.contextual_guidance or ""
            db_hint = (
                "DATABASE INSPECTION: The user explicitly wants direct database/SQLite inspection. "
                "Prefer the db tool first. If SQLite CLI inspection is more direct, use a "
                "focused readonly shell command. Do not substitute repo/code inspection for "
                "the requested database query step."
            )
            self.prompt_builder.contextual_guidance = f"{existing}\n\n{db_hint}".strip()
            logger.info("Applied database inspection contextual guidance")

    # =====================================================================
    # Task Guidance
    # =====================================================================

    def apply_task_guidance(
        self,
        user_message: str,
        unified_task_type: "TaskType",
        is_analysis_task: bool,
        is_action_task: bool,
        needs_execution: bool,
        max_exploration_iterations: int,
        conversation_controller: "ConversationController",
    ) -> None:
        """Apply guidance and budget tweaks for analysis/action tasks.

        Provides task-specific system prompts and adjusts agent parameters
        (temperature, tool budget) based on task characteristics.

        Args:
            user_message: The user's input message
            unified_task_type: Unified task type classification
            is_analysis_task: Whether this is an analysis task
            is_action_task: Whether this is an action-oriented task
            needs_execution: Whether the task requires execution
            max_exploration_iterations: Maximum exploration iterations allowed
            conversation_controller: Conversation controller for message injection
        """
        if is_analysis_task:
            # Temperature for analysis is owned by the unified resolver (ADR-013): the "analyze"
            # task-hint constant is applied at the provider seam — no scalar mutation here.
            conversation_controller.add_message(
                "user",
                "[ANALYSIS-APPROACH: Work through the codebase one module at a time. "
                "For each module: 1) List its files, 2) Read 2-3 key files, 3) Note observations. "
                "After examining 3-4 modules, provide your summary. Keep responses concise.]",
            )

            # Increase tool budget for comprehensive analysis,
            # but only if the user didn't explicitly set it via --tool-budget
            if not self.unified_tracker._sticky_user_budget:
                original_budget = self._tool_budget
                self._tool_budget = max(self._tool_budget, 200)
                if self._tool_budget != original_budget:
                    logger.info(
                        f"Analysis task: increased tool_budget from {original_budget} to {self._tool_budget}"
                    )
            else:
                logger.info(f"Analysis task: respecting user-set tool_budget={self._tool_budget}")

            conversation_controller.add_message(
                "user",
                "[ANALYSIS-GUIDANCE: This is an ANALYSIS task requiring thorough exploration of the codebase. "
                "You MUST systematically examine multiple modules and files using tools like "
                "read, ls, and code_search. "
                "DO NOT stop after examining just a few files. "
                "Continue using tools until you have gathered comprehensive information about "
                "all major components of the codebase. "
                "Only provide your final analysis AFTER you have examined all relevant modules.]",
            )

        if is_action_task:
            logger.info(
                f"Detected action-oriented task - allowing up to {max_exploration_iterations} exploration iterations"
            )

            if needs_execution:
                conversation_controller.add_message(
                    "user",
                    "[ACTION-GUIDANCE: This is an action-oriented task requiring execution. "
                    "Follow this workflow: "
                    "1. CREATE the file/script with write or edit "
                    "2. EXECUTE it immediately with shell (don't skip this step!) "
                    "3. SHOW the output to the user. "
                    "Minimize exploration and proceed directly to create→execute→show results.]",
                )
            else:
                conversation_controller.add_message(
                    "user",
                    "[ACTION-GUIDANCE: This is an action-oriented task (create/write/build). "
                    "Minimize exploration and proceed directly to creating what was requested. "
                    "Only explore if absolutely necessary to complete the task.]",
                )

    # =====================================================================
    # Property Accessors
    # =====================================================================

    @property
    def current_intent(self):
        """Get the current detected intent."""
        return self._current_intent

    @property
    def temperature(self) -> float:
        """Get the current temperature setting."""
        return self._temperature

    @temperature.setter
    def temperature(self, value: float):
        """Set the temperature setting."""
        self._temperature = value

    @property
    def tool_budget(self) -> int:
        """Get the current tool budget."""
        return self._tool_budget

    @tool_budget.setter
    def tool_budget(self, value: int):
        """Set the tool budget."""
        self._tool_budget = value

    @property
    def observed_files(self) -> list:
        """Get the list of observed files."""
        return self._observed_files

    @observed_files.setter
    def observed_files(self, value: list):
        """Set the list of observed files."""
        self._observed_files = value

    def set_reminder_manager(self, reminder_manager: Any):
        """Set the reminder manager reference.

        Args:
            reminder_manager: ReminderManager instance
        """
        self._reminder_manager = reminder_manager
