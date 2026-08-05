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

"""Conversation state machine for intelligent stage detection.

MIGRATION NOTICE: State storage is migrating to canonical system.

For state storage, use the canonical state management system:
    - victor.state.ConversationStateManager - Conversation scope state
    - victor.state.get_global_manager() - Unified access to all scopes

This module is kept for its stage detection business logic.
The ConversationStateMachine now uses ConversationStateManager internally
for state storage.

---

Legacy Documentation:

This module provides automatic detection of conversation stages to improve
tool selection accuracy. Instead of manual stage management, it infers the
current stage from:
- Tool execution history
- Message content patterns
- Observed file/resource access
- Conversation depth

Stages:
- INITIAL: First interaction, exploring the request
- PLANNING: Understanding scope, searching for files
- READING: Examining files, gathering context
- ANALYSIS: Reviewing code, analyzing structure
- EXECUTION: Making changes, running commands
- VERIFICATION: Testing, validating changes
- COMPLETION: Summarizing, wrapping up

Hook Integration (Observer Pattern):
The state machine supports hooks for observing transitions:
- on_enter: Called when entering a stage
- on_exit: Called when exiting a stage
- on_transition: Called on any stage change

Example:
    from victor.observability import StateHookManager

    hooks = StateHookManager()

    @hooks.on_transition
    def log_transition(old, new, ctx):
        print(f"{old} -> {new}")

    machine = ConversationStateMachine(hooks=hooks)

Migration Example:
    # OLD (using ConversationStateMachine for state storage):
    machine = ConversationStateMachine()
    machine.record_tool_execution("read", {"file": "test.py"})
    stage = machine.get_stage()

    # NEW (using canonical state management):
    from victor.state import ConversationStateManager, StateScope

    # For stage detection, still use ConversationStateMachine:
    machine = ConversationStateMachine()
    machine.record_tool_execution("read", {"file": "test.py"})
    stage = machine.get_stage()

    # For state storage, use ConversationStateManager:
    mgr = ConversationStateManager()
    await mgr.set("tool_history", ["read"])
    await mgr.set("observed_files", {"test.py"})

    # OR for unified access:
    from victor.state import get_global_manager
    state = get_global_manager()
    await state.set("tool_history", ["read"], scope=StateScope.CONVERSATION)
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple

from victor.tools.metadata_registry import (
    get_tools_by_stage as registry_get_tools_by_stage,
)
from victor.core.events import ObservabilityBus

if TYPE_CHECKING:
    from victor.observability.hooks import StateHookManager

logger = logging.getLogger(__name__)


# Re-export from canonical location for backward compatibility
from victor.core.shared_types import ConversationStage  # noqa: F401

# Stage ordering for adjacency calculations (since values are strings, not ints)
STAGE_ORDER: Dict[ConversationStage, int] = {
    stage: idx for idx, stage in enumerate(ConversationStage)
}


# Stage-to-tool mapping is now fully decorator-driven.
# Tools define their stages via @tool(stages=["reading", "execution"]) decorator.
# Use registry_get_tools_by_stage() to get tools for a stage.

# Keywords that suggest specific stages
# Weak EXECUTION keywords — common in bug descriptions but don't imply edit intent.
# Scored at 0.5 instead of 1.0 to prevent premature EXECUTION stage detection.
WEAK_EXECUTION_KEYWORDS = {"fix", "add", "change", "update"}

# Explicit allowlist of natural backward stage transitions.
# These get the lower threshold (0.50) instead of BACKWARD_TRANSITION_THRESHOLD (0.85).
# All other backward transitions require high confidence to prevent regressions.
NATURAL_BACKWARD_TRANSITIONS = {
    (ConversationStage.EXECUTION, ConversationStage.READING),  # verify-after-edit
    (ConversationStage.EXECUTION, ConversationStage.ANALYSIS),  # re-analyze after edit
    (ConversationStage.ANALYSIS, ConversationStage.READING),  # need more file context
    (
        ConversationStage.VERIFICATION,
        ConversationStage.EXECUTION,
    ),  # fix after failed test
    (
        ConversationStage.VERIFICATION,
        ConversationStage.ANALYSIS,
    ),  # re-analyze after test
    (ConversationStage.COMPLETION, ConversationStage.VERIFICATION),  # re-verify
}

# Empirical stage transition probabilities (AutonAgenticAI-inspired).
# Used by predict_next_stage() for predictive context pruning.
STAGE_TRANSITION_PROBS: Dict[str, Dict[str, float]] = {
    "initial": {"reading": 0.7, "planning": 0.3},
    "planning": {"reading": 0.6, "execution": 0.3, "analysis": 0.1},
    "reading": {"execution": 0.5, "analysis": 0.3, "reading": 0.2},
    "analysis": {"execution": 0.6, "reading": 0.3, "planning": 0.1},
    "execution": {"verification": 0.5, "reading": 0.3, "execution": 0.2},
    "verification": {"completion": 0.4, "execution": 0.4, "reading": 0.2},
    "completion": {"completion": 1.0},
}

STAGE_KEYWORDS: Dict[ConversationStage, List[str]] = {
    ConversationStage.INITIAL: ["what", "how", "where", "explain", "help", "can you"],
    ConversationStage.PLANNING: ["plan", "approach", "strategy", "design", "architect"],
    ConversationStage.READING: ["show", "read", "look at", "check", "find"],
    ConversationStage.ANALYSIS: ["analyze", "review", "examine", "understand", "why"],
    ConversationStage.EXECUTION: [
        "change",
        "modify",
        "create",
        "add",
        "remove",
        "fix",
        "implement",
        "write",
        "update",
    ],
    ConversationStage.VERIFICATION: ["test", "verify", "check", "validate", "run"],
    ConversationStage.COMPLETION: ["done", "finish", "complete", "summarize", "commit"],
}


@dataclass
class ConversationState:
    """Tracks the current state of a conversation.

    Attributes:
        stage: Current conversation stage
        tool_history: List of tools executed in order
        observed_files: Files that have been read
        modified_files: Files that have been modified
        message_count: Number of messages in conversation
        last_tools: Last N tools executed (for pattern detection)
    """

    stage: ConversationStage = ConversationStage.INITIAL
    tool_history: List[str] = field(default_factory=list)
    observed_files: Set[str] = field(default_factory=set)
    modified_files: Set[str] = field(default_factory=set)
    message_count: int = 0
    last_tools: List[str] = field(default_factory=list)
    _stage_confidence: float = 0.5

    def record_tool_execution(self, tool_name: str, args: Dict[str, Any]) -> None:
        """Record a tool execution and update state.

        Args:
            tool_name: Name of the executed tool
            args: Tool arguments
        """
        self.tool_history.append(tool_name)
        self.last_tools.append(tool_name)
        if len(self.last_tools) > 5:
            self.last_tools.pop(0)

        # Track file access (use canonical tool names)
        file_arg = args.get("file") or args.get("path") or args.get("file_path")
        if file_arg:
            if tool_name in {"read", "ls", "search", "overview"}:
                self.observed_files.add(str(file_arg))
            elif tool_name in {"write", "edit"}:
                self.modified_files.add(str(file_arg))

    def record_message(self) -> None:
        """Record a new message in the conversation."""
        self.message_count += 1

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the conversation state to a dictionary.

        Returns:
            Dictionary representation of the state.
        """
        return {
            "stage": self.stage.name,
            "tool_history": self.tool_history,
            "observed_files": list(self.observed_files),
            "modified_files": list(self.modified_files),
            "message_count": self.message_count,
            "last_tools": self.last_tools,
            "stage_confidence": self._stage_confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationState":
        """Restore conversation state from a dictionary.

        Args:
            data: Dictionary containing serialized state.

        Returns:
            Restored ConversationState instance.
        """
        state = cls()
        state.stage = ConversationStage[data.get("stage", "INITIAL")]
        state.tool_history = data.get("tool_history", [])
        state.observed_files = set(data.get("observed_files", []))
        state.modified_files = set(data.get("modified_files", []))
        state.message_count = data.get("message_count", 0)
        state.last_tools = data.get("last_tools", [])
        state._stage_confidence = data.get("stage_confidence", 0.5)
        return state


class ConversationStateMachine:
    """State machine for detecting and managing conversation stages.

    Automatically detects the current stage based on:
    - Tool execution patterns
    - Message content analysis
    - Conversation progression

    Includes cooldown mechanism to prevent rapid stage thrashing.

    Supports hooks via the Observer pattern for decoupled state observation:
    - on_enter: Called when entering a stage
    - on_exit: Called when exiting a stage
    - on_transition: Called on any stage change

    Example:
        from victor.observability import StateHookManager

        hooks = StateHookManager()

        @hooks.on_enter("EXECUTION")
        def on_exec_start(stage, context):
            print("Starting execution phase!")

        machine = ConversationStateMachine(hooks=hooks)
    """

    # Minimum seconds between stage transitions (prevents thrashing)
    # Reduced from 5.0 to 2.0 to allow faster progression for SWE-bench tasks
    # while still preventing thrashing
    TRANSITION_COOLDOWN_SECONDS: float = 2.0

    # Stage detection order is now managed by unified decision chain config.
    # See: victor.agent.decisions.chain.get_decision_chain("stage_detection")
    # Configure via settings.pipeline.decision_chain or decision_chain_default

    # Thrashing detection: if >MAX transitions in WINDOW seconds, apply longer cooldown.
    # Tuned conservatively — natural EXECUTION↔READING cycling (read-edit-read) is
    # expected in agentic workflows. Only flag true oscillation (>8 in 2 min).
    MAX_TRANSITIONS_PER_WINDOW: int = 8
    THRASHING_WINDOW_SECONDS: float = 120.0  # 2 min window
    THRASHING_COOLDOWN_SECONDS: float = 10.0  # Brief pause, not hard lock

    # Minimum tools required to trigger stage transition (increased from 3 to 5)
    # Allows more work per turn while still skipping edge model when confident
    # Session analysis shows this balances work-per-turn with optimization benefits
    MIN_TOOLS_FOR_TRANSITION: int = 5

    # Confidence threshold for backward stage transitions
    BACKWARD_TRANSITION_THRESHOLD: float = 0.85

    # Maximum reads without edit before forcing READING → EXECUTION transition
    # Prevents infinite exploration in SWE-bench style bug fix tasks
    MAX_READS_WITHOUT_EDIT: int = 5

    def __init__(
        self,
        hooks: Optional["StateHookManager"] = None,
        track_history: bool = True,
        max_history_size: int = 100,
        event_bus: Optional[ObservabilityBus] = None,
        state_manager: Optional[Any] = None,
        runtime_intelligence: Optional[Any] = None,
        action_intent: Optional[Any] = None,
        transition_coordinator: Optional[Any] = None,
        llm_decision_service: Optional[Any] = None,
    ) -> None:
        """Initialize the state machine.

        Args:
            hooks: Optional StateHookManager for transition callbacks.
            track_history: Whether to track transition history.
            max_history_size: Maximum number of transitions to keep in history.
            event_bus: Optional ObservabilityBus instance. If None, uses DI container.
            state_manager: Optional ConversationStateManager for canonical state storage.
                          If provided, state will be synced to the manager.
            runtime_intelligence: Optional canonical runtime-intelligence service for
                stage-detection augmentation.
            action_intent: Optional ActionIntent for context-aware calibration.
            transition_coordinator: Optional StageTransitionCoordinator for batching
                tool executions and applying Phase 1 optimizations consistently.
            llm_decision_service: Optional LLM decision service for stage decisions.
        """
        self.state = ConversationState()
        self._last_transition_time: float = 0.0
        self._transition_count: int = 0
        self._hooks = hooks
        self._track_history = track_history
        self._max_history_size = max_history_size
        self._transition_history: List[Dict[str, Any]] = []

        # Stage detection order managed by decisions.chain module
        self._event_bus = event_bus or self._get_default_bus()
        self._state_manager = state_manager  # Optional canonical state manager
        self._runtime_intelligence = runtime_intelligence
        self._llm_decision_service = llm_decision_service or self._resolve_llm_decision_service()
        self._action_intent = action_intent  # PHASE 2: For context-aware calibration
        self._transition_coordinator = transition_coordinator  # PHASE 16: For batching

        # Edge model decision cache — avoid repeated calls with same input
        # Key: stage_value:message_prefix → (result_stage, confidence, timestamp)
        self._edge_stage_cache: Dict[str, tuple] = {}

        # Sync initial state to manager if provided
        if self._state_manager:
            self._sync_state_to_manager()

    @staticmethod
    def _resolve_llm_decision_service() -> Optional[Any]:
        """Resolve the optional decision service from DI for compatibility callers."""
        try:
            from victor.agent.services.protocols.decision_service import (
                LLMDecisionServiceProtocol,
            )
            from victor.core.service_resolution import resolve_optional_service

            return resolve_optional_service(
                LLMDecisionServiceProtocol,
                legacy_key="llm_decision_service",
                required_attrs=("decide_sync",),
            )
        except Exception:
            return None

    def predict_next_stage(self) -> tuple:
        """Predict the most likely next stage (AutonAgenticAI-inspired).

        Uses empirical transition probabilities to predict where
        the conversation will move next. Used by context pruning to
        proactively preserve messages relevant to the predicted phase.

        Returns:
            (predicted_stage: str, confidence: float)
        """
        current = self.state.stage.value
        probs = STAGE_TRANSITION_PROBS.get(current, {})
        if not probs:
            return (current, 0.5)
        best_stage = max(probs, key=probs.get)
        return (best_stage, probs[best_stage])

    def reset(self) -> None:
        """Reset state for a new conversation."""
        self.state = ConversationState()
        self._transition_history.clear()
        self._transition_count = 0
        self._last_transition_time = 0.0

        # Reset manager state if provided
        if self._state_manager:
            self._sync_state_to_manager()

    def _is_thrashing(self) -> bool:
        """Detect stage thrashing — rapid oscillation between stages.

        Returns True if more than MAX_TRANSITIONS_PER_WINDOW transitions
        occurred in the last THRASHING_WINDOW_SECONDS.
        """
        import time

        if len(self._transition_history) < self.MAX_TRANSITIONS_PER_WINDOW:
            return False
        now = time.time()
        recent = [
            t
            for t in self._transition_history
            if now - t.get("timestamp", 0) < self.THRASHING_WINDOW_SECONDS
        ]
        return len(recent) >= self.MAX_TRANSITIONS_PER_WINDOW

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

    def _sync_state_to_manager(self) -> None:
        """Sync current state to the canonical state manager.

        This is called internally when state changes to keep the
        ConversationStateManager in sync.
        """
        if not self._state_manager:
            return

        try:
            # Sync state to manager (non-blocking)
            # We do this synchronously for compatibility
            state_dict = self.state.to_dict()

            # Store in manager (using internal _state for direct access)
            self._state_manager._state.update(state_dict)
        except Exception as e:
            logger.warning(f"Failed to sync state to manager: {e}")

    def set_transition_coordinator(self, coordinator: Optional[Any]) -> None:
        """Set or update the transition coordinator.

        Args:
            coordinator: Optional StageTransitionCoordinator for batching
                tool executions and applying Phase 1 optimizations.
        """
        self._transition_coordinator = coordinator
        logger.debug(
            f"Transition coordinator {'enabled' if coordinator else 'disabled'} "
            f"for ConversationStateMachine"
        )

    def record_tool_execution(self, tool_name: str, args: Dict[str, Any]) -> None:
        """Record tool execution and potentially transition stage.

        PHASE 16: When transition_coordinator is attached, tool executions
        are batched and processed once per turn instead of per-tool.
        This prevents Phase 1 optimization bypassing in the streaming path.

        Args:
            tool_name: Name of the executed tool
            args: Tool arguments
        """
        self.state.record_tool_execution(tool_name, args)

        # Sync to canonical state manager if provided
        if self._state_manager:
            self._sync_state_to_manager()

        # PHASE 16: Use coordinator if available (batches tools, processes once per turn)
        if self._transition_coordinator is not None:
            # Just record the tool - coordinator will process batch at end_turn()
            self._transition_coordinator.record_tool(tool_name, args)
        else:
            # Legacy behavior: check transition immediately for each tool
            self._maybe_transition()

    def record_message(self, content: str, is_user: bool = True) -> None:
        """Record a message and potentially transition stage.

        Args:
            content: Message content
            is_user: Whether the message is from the user
        """
        self.state.record_message()

        if is_user:
            # Detect stage from user message content
            detected_stage = self._detect_stage_from_content(content)
            if detected_stage:
                self._transition_to(detected_stage, confidence=0.7)

    def get_stage(self) -> ConversationStage:
        """Get the current conversation stage.

        Returns:
            Current ConversationStage
        """
        return self.state.stage

    def get_current_stage(self) -> ConversationStage:
        """Backward-compatible alias for orchestrator integrations."""
        return self.get_stage()

    def get_stage_tools(self) -> Set[str]:
        """Get tools relevant to the current stage.

        Tools define their stages via @tool(stages=["reading", "execution"]) decorator.
        The metadata registry indexes tools by stage for efficient lookup.

        Returns:
            Set of tool names relevant to current stage
        """
        return self._get_tools_for_stage(self.state.stage)

    def _get_tools_for_stage(self, stage: ConversationStage) -> Set[str]:
        """Get tools for a specific stage from the metadata registry.

        Tools define their stages via @tool(stages=["reading", "execution"]) decorator.
        The registry indexes tools by stage for efficient lookup.

        Args:
            stage: The conversation stage to get tools for

        Returns:
            Set of tool names relevant to the stage
        """
        stage_name = stage.name.lower()
        return registry_get_tools_by_stage(stage_name)

    def get_state_summary(self) -> Dict[str, Any]:
        """Get a summary of the current conversation state.

        Returns:
            Dictionary with state information
        """
        return {
            "stage": self.state.stage.name,
            "confidence": self.state._stage_confidence,
            "message_count": self.state.message_count,
            "tools_executed": len(self.state.tool_history),
            "files_observed": len(self.state.observed_files),
            "files_modified": len(self.state.modified_files),
            "last_tools": self.state.last_tools,
            "recommended_tools": list(self.get_stage_tools()),
        }

    def set_action_intent(self, intent: Any) -> None:
        """Update the action intent for context-aware calibration.

        Phase 2 Enhancement: Allows dynamic intent updates for edge model calibration.
        The intent is used to bias-correct edge model decisions (e.g., READ_ONLY tasks
        should not transition to EXECUTION even if edge model suggests it).

        Args:
            intent: ActionIntent enum value (WRITE_ALLOWED, READ_ONLY, DISPLAY_ONLY, etc.)
        """
        from victor.agent.action_authorizer import ActionIntent

        self._action_intent = intent
        logger.debug(f"ActionIntent updated: {intent.value if intent else None}")

    def _detect_stage_from_content(self, content: str) -> Optional[ConversationStage]:
        """Detect stage from message content using configurable strategy order.

        The detection order is configurable via STAGE_DETECTION_ORDER:
        - "heuristic": Fast keyword-based detection (default first)
        - "llm": LLM decision service with full context (default tiebreaker)

        Args:
            content: Message content to analyze

        Returns:
            Detected stage or None
        """
        from victor.agent.decisions.chain import get_decision_chain

        for strategy in get_decision_chain("stage_detection"):
            result = None
            if strategy == "heuristic":
                result = self._detect_stage_heuristic(content)
            elif strategy == "llm":
                edge_stage, _ = self._detect_stage_with_edge_model(content)
                result = edge_stage

            if result is not None:
                # Guard: suppress EXECUTION before any files are read
                if (
                    result == ConversationStage.EXECUTION
                    and self.state.message_count <= 1
                    and not self.state.observed_files
                ):
                    return ConversationStage.READING
                return result

        return None

    def _detect_stage_heuristic(self, content: str) -> Optional[ConversationStage]:
        """Keyword-based stage detection with weighted scoring.

        Returns:
            Detected stage if confidence >= 2, else None (falls to next strategy)
        """
        content_lower = content.lower()
        scores: Dict[ConversationStage, float] = {}

        for stage, keywords in STAGE_KEYWORDS.items():
            for kw in keywords:
                if kw in content_lower:
                    weight = 1.0
                    if stage == ConversationStage.EXECUTION and kw in WEAK_EXECUTION_KEYWORDS:
                        weight = 0.5
                    scores[stage] = scores.get(stage, 0) + weight

        if scores:
            best_stage = max(scores, key=scores.get)  # type: ignore
            if scores[best_stage] >= 2:  # High confidence
                return best_stage
            # Low confidence — return best match but let next strategy override
            return best_stage

        return None

    def _detect_stage_with_edge_model(
        self,
        content: str,
        tool_context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[ConversationStage], float]:
        """Use edge model for stage detection when heuristics are ambiguous.

        Called from two paths:
        1. Content-based detection (keywords ambiguous) — tool_context is None
        2. Tool-based transition fallback (overlap too low) — tool_context has
           last_tools, current_stage, detected_stage_heuristic

        Args:
            content: Message content to analyze (may be empty for tool path)
            tool_context: Optional dict with tool-based context for richer prompt

        Returns:
            Tuple of (detected stage, confidence) or (None, 0.0)
        """
        try:
            from victor.agent.decisions.schemas import DecisionType

            # Always provide full state to edge model — prevents cold-calling
            # that disagrees with framework's richer context.
            context: Dict[str, Any] = {
                "message_excerpt": content[:200] if content else "",
                "current_stage": (
                    tool_context.get("current_stage", "")
                    if tool_context
                    else self.state.stage.value
                ),
                "last_tools": (
                    tool_context.get("last_tools", "")
                    if tool_context
                    else ",".join(self.state.last_tools[-5:])
                ),
                "detected_stage_heuristic": (
                    tool_context.get("detected_stage_heuristic", "") if tool_context else ""
                ),
                "files_observed": len(self.state.observed_files),
                "files_modified": len(self.state.modified_files),
                "tool_call_count": len(self.state.last_tools),
            }

            logger.debug(
                "Edge model input: stage=%s, tools=%s, files_read=%d, "
                "files_modified=%d, tool_calls=%d",
                context.get("current_stage"),
                context.get("last_tools"),
                context.get("files_observed", 0),
                context.get("files_modified", 0),
                context.get("tool_call_count", 0),
            )

            # Use predict_next_stage() as heuristic for the decision service
            # (AutonAgenticAI-inspired: static prediction as fast-path)
            predicted_stage, pred_confidence = self.predict_next_stage()
            heuristic_conf = pred_confidence if pred_confidence >= 0.6 else 0.0

            decision = self._decide_stage_sync(
                DecisionType.STAGE_DETECTION,
                context=context,
                heuristic_confidence=heuristic_conf,
            )
            if decision is None:
                return None, 0.0

            if decision.source in ("heuristic", "budget_exhausted", "timeout_fallback"):
                return None, 0.0

            if not hasattr(decision.result, "stage"):
                return None, 0.0

            stage_name = decision.result.stage
            confidence = decision.confidence

            logger.debug(
                "Edge model output: stage=%s, confidence=%.2f, source=%s",
                stage_name,
                confidence,
                decision.source,
            )

            if confidence < 0.6:
                return None, 0.0

            # Map string to ConversationStage enum
            stage_map = {
                "initial": ConversationStage.INITIAL,
                "planning": ConversationStage.PLANNING,
                "reading": ConversationStage.READING,
                "analysis": ConversationStage.ANALYSIS,
                "execution": ConversationStage.EXECUTION,
                "verification": ConversationStage.VERIFICATION,
                "completion": ConversationStage.COMPLETION,
            }
            result = stage_map.get(stage_name)

            # ============================================================================
            # PHASE 2: Context-Aware Calibration
            # Calibrate edge model results based on actual behavior, not counting
            # ============================================================================
            if result == ConversationStage.EXECUTION and confidence >= 0.95:
                from victor.agent.action_authorizer import ActionIntent

                files_read = len(self.state.observed_files)
                files_modified = len(self.state.modified_files)

                # If reading many files without editing, distrust EXECUTION
                # This happens when edge model is overconfident or biased
                if (
                    self._action_intent != ActionIntent.WRITE_ALLOWED
                    and files_read > 10
                    and files_modified == 0
                ):
                    logger.warning(
                        f"Edge model calibration: {stage_name} ({confidence:.2f}) → ANALYSIS. "
                        f"Reason: Agent has read {files_read} files without any edits. "
                        f"High confidence EXECUTION is likely biased/overconfident."
                    )
                    return ConversationStage.ANALYSIS, 0.7

            # Correct for intent bias
            if self._action_intent and result:
                from victor.agent.action_authorizer import ActionIntent

                # Read-only tasks shouldn't be in EXECUTION stage
                if (
                    self._action_intent == ActionIntent.READ_ONLY
                    and result == ConversationStage.EXECUTION
                ):
                    logger.warning(
                        f"Edge model calibration: {stage_name} ({confidence:.2f}) → ANALYSIS. "
                        f"Reason: Task intent is READ_ONLY, EXECUTION is inappropriate."
                    )
                    return ConversationStage.ANALYSIS, confidence * 0.8

                # Display-only tasks shouldn't be in EXECUTION stage
                if (
                    self._action_intent == ActionIntent.DISPLAY_ONLY
                    and result == ConversationStage.EXECUTION
                ):
                    logger.warning(
                        f"Edge model calibration: {stage_name} ({confidence:.2f}) → ANALYSIS. "
                        f"Reason: Task intent is DISPLAY_ONLY, EXECUTION is inappropriate."
                    )
                    return ConversationStage.ANALYSIS, confidence * 0.8

            if result:
                logger.info(f"Edge stage detection: {stage_name} (confidence={confidence:.2f})")
            return result, confidence

        except Exception as e:
            logger.debug(f"Edge stage detection unavailable: {e}")
            return None, 0.0

    def _decide_stage_sync(
        self,
        decision_type: Any,
        context: Dict[str, Any],
        *,
        heuristic_confidence: float = 0.0,
    ) -> Optional[Any]:
        """Use the canonical runtime-intelligence boundary for stage decisions."""
        if self._runtime_intelligence is not None:
            decision = self._runtime_intelligence.decide_sync(
                decision_type,
                context,
                heuristic_confidence=heuristic_confidence,
            )
            if decision is not None:
                return decision

        service = self._llm_decision_service
        if service is None:
            service = self._resolve_llm_decision_service()
            self._llm_decision_service = service
        if service is None:
            return None

        return service.decide_sync(
            decision_type,
            context=context,
            heuristic_confidence=heuristic_confidence,
        )

    def _detect_stage_from_tools(self) -> Optional[ConversationStage]:
        """Detect stage from recent tool execution patterns.

        Returns:
            Detected stage or None
        """
        if not self.state.last_tools:
            logger.debug("_detect_stage_from_tools: No recent tools, returning None")
            return None

        # Score stages based on tool overlap (uses registry + static fallback)
        scores: Dict[ConversationStage, int] = {}

        for stage in ConversationStage:
            stage_tools = self._get_tools_for_stage(stage)
            overlap = len(set(self.state.last_tools) & stage_tools)
            if overlap > 0:
                scores[stage] = overlap

        if scores:
            max_score = max(scores.values())
            tied_stages = [s for s, v in scores.items() if v == max_score]

            # Tie-breaking logic to prevent oscillation:
            # 1. Prefer current stage if it's in the tied group (stability)
            # 2. Otherwise, prefer the most advanced stage (forward progress)
            if len(tied_stages) == 1:
                detected = tied_stages[0]
            elif self.state.stage in tied_stages:
                # Current stage is tied - stay to avoid oscillation
                detected = self.state.stage
                logger.debug("_detect_stage_from_tools: Tie resolved by staying at current stage")
            else:
                # Pick the most advanced (highest in workflow order)
                detected = max(tied_stages, key=lambda s: STAGE_ORDER[s])
                logger.debug(
                    "_detect_stage_from_tools: Tie resolved by picking most advanced stage"
                )

            scores_str = ", ".join(f"{k.name}={v}" for k, v in scores.items())
            logger.debug(
                f"_detect_stage_from_tools: last_tools={self.state.last_tools}, "
                f"scores=[{scores_str}], detected={detected.name}"
            )
            return detected

        logger.debug(
            f"_detect_stage_from_tools: No stage overlap for tools={self.state.last_tools}"
        )
        return None

    def _maybe_transition(self) -> None:
        """Check if we should transition to a new stage.

        Refined optimization (Phase 1):
        - Check cooldown FIRST before any expensive operations
        - Only call edge model when heuristic is uncertain
        - Skip edge model calls that won't change outcome
        """
        import time

        # ============================================================================
        # FIX 1.1: Check cooldown FIRST before ANY expensive operations
        # Prevents 80%+ of edge model calls by skipping during cooldown period
        # ============================================================================
        current_time = time.time()
        time_since_last = current_time - self._last_transition_time

        if time_since_last < self.TRANSITION_COOLDOWN_SECONDS:
            logger.debug(
                f"_maybe_transition: In cooldown ({time_since_last:.1f}s < "
                f"{self.TRANSITION_COOLDOWN_SECONDS}s), skipping all checks including edge model"
            )
            return  # Early exit - don't call edge model, don't check anything

        # DISABLED: Trust UnifiedTaskTracker's sophisticated loop detection instead
        # UnifiedTaskTracker has:
        # - Offset-aware file read detection (distinguishes different offsets)
        # - Task-type aware thresholds (different for EDIT vs. ANALYZE tasks)
        # - Mode-aware exploration limits (adaptive mode support)
        # - Signature-based deduplication with permanent blocking
        # - Read-only loop detection (>20 files read, 0 modified)
        # Primitive counting is redundant and inferior.
        if self._should_force_execution_transition():
            logger.info(
                f"Forcing READING→EXECUTION: {len(self.state.observed_files)} files read, "
                f"{len(self.state.modified_files)} files modified"
            )
            self._transition_to(ConversationStage.EXECUTION, confidence=0.8)
            return

        # ============================================================================
        # FIX 1.2: Only call edge model when heuristic is uncertain
        # Reduces edge model calls by 90%+ by skipping when heuristic is confident
        # ============================================================================
        detected = self._detect_stage_from_tools()
        if detected and detected != self.state.stage:
            stage_tools = self._get_tools_for_stage(detected)
            recent_overlap = len(set(self.state.last_tools) & stage_tools)

            logger.debug(
                f"_maybe_transition: current={self.state.stage.name}, detected={detected.name}, "
                f"overlap={recent_overlap}, min_threshold={self.MIN_TOOLS_FOR_TRANSITION}, "
                f"stage_tools_count={len(stage_tools)}"
            )

            # HIGH CONFIDENCE: Heuristic is certain, skip edge model
            # When overlap >= MIN_TOOLS_FOR_TRANSITION (3), we have strong evidence
            # No need to call expensive edge model
            if recent_overlap >= self.MIN_TOOLS_FOR_TRANSITION:
                confidence = 0.6 + (recent_overlap * 0.1)
                logger.debug(
                    f"_maybe_transition: High heuristic confidence ({confidence:.2f}), "
                    f"skipping edge model call and transitioning directly"
                )
                self._transition_to(detected, confidence=confidence)
                return

            # LOW CONFIDENCE: Consult edge model only when uncertain
            # When overlap < MIN_TOOLS_FOR_TRANSITION, heuristic is uncertain
            # Edge model provides valuable tiebreaker in this case
            logger.debug(
                f"_maybe_transition: Low heuristic confidence (overlap={recent_overlap} < "
                f"{self.MIN_TOOLS_FOR_TRANSITION}), consulting edge model for tiebreaker"
            )

            # Check if edge model is in decision chain
            from victor.agent.decisions.chain import should_use_llm

            if not should_use_llm("stage_detection"):
                logger.debug("_maybe_transition: Edge model not in chain, keeping current stage")
                return

            edge_stage, edge_confidence = self._try_edge_model_transition(
                detected, heuristic_confidence=0.6 + (recent_overlap * 0.1)
            )
            if edge_stage is not None and edge_confidence > 0.6:
                logger.info(
                    f"Edge model override: {detected.name}→{edge_stage.name} "
                    f"(edge={edge_confidence:.2f} > heuristic=0.6)"
                )
                self._transition_to(edge_stage, confidence=edge_confidence)
            else:
                logger.debug(
                    f"_maybe_transition: Edge model declined (confidence={edge_confidence:.2f}), "
                    f"keeping current stage {self.state.stage.name}"
                )

    def _try_edge_model_transition(
        self,
        heuristic_stage: ConversationStage,
        heuristic_confidence: float,
    ) -> Tuple[Optional[ConversationStage], float]:
        """Try edge model as fallback when heuristic confidence is low.

        Only called when USE_EDGE_MODEL feature flag is enabled.
        Returns (None, 0.0) if edge model is unavailable or disabled.
        """
        try:
            from victor.core.feature_flags import get_feature_flag_manager, FeatureFlag

            if not get_feature_flag_manager().is_enabled(FeatureFlag.USE_EDGE_MODEL):
                return None, 0.0
        except Exception:
            return None, 0.0

        return self._detect_stage_with_edge_model(
            content="",
            tool_context={
                "last_tools": (",".join(self.state.last_tools) if self.state.last_tools else ""),
                "current_stage": self.state.stage.name.lower(),
                "detected_stage_heuristic": heuristic_stage.name.lower(),
                "message_excerpt": "",
            },
        )

    def _should_force_execution_transition(self) -> bool:
        """Check if we should force transition from READING to EXECUTION.

        DISABLED: Trust UnifiedTaskTracker's sophisticated loop detection instead.

        UnifiedTaskTracker already handles:
        - Read-only loop detection (>20 files read, 0 modified)
        - File read overlap detection (same offset/limit repeated)
        - Search query repetition (similar searches)
        - Signature-based deduplication (permanent blocking after warning)
        - Task-type aware thresholds (different for EDIT vs. ANALYZE)
        - Mode-aware limits (exploration multiplier)

        Primitive counting (consecutive reads, unique files) is redundant and inferior.

        Returns:
            False (disabled - trust UnifiedTaskTracker)
        """
        # DISABLED: Trust UnifiedTaskTracker's sophisticated loop detection
        # This primitive heuristic is redundant with UnifiedTaskTracker's superior detection
        return False

    def _transition_to(self, new_stage: ConversationStage, confidence: float = 0.5) -> None:
        """Transition to a new stage.

        Args:
            new_stage: Stage to transition to
            confidence: Confidence in this transition

        Note: Transitions are rate-limited by TRANSITION_COOLDOWN_SECONDS to
        prevent rapid stage thrashing observed in analysis tasks.

        Hook invocation order (Observer Pattern):
        1. on_exit hooks for old_stage
        2. Stage update
        3. on_transition hooks (old_stage, new_stage)
        4. on_enter hooks for new_stage
        """
        import time

        old_stage = self.state.stage

        # Don't transition backwards unless confidence is sufficient.
        # Natural backward transitions (explicit allowlist) use lower threshold (0.50).
        # All other backward transitions require BACKWARD_TRANSITION_THRESHOLD (0.85).
        if STAGE_ORDER[new_stage] < STAGE_ORDER[old_stage]:
            is_natural_backward = (old_stage, new_stage) in NATURAL_BACKWARD_TRANSITIONS
            threshold = 0.50 if is_natural_backward else self.BACKWARD_TRANSITION_THRESHOLD
            if confidence < threshold:
                logger.debug(
                    f"_transition_to: Backward transition blocked {old_stage.name} -> {new_stage.name}, "
                    f"confidence={confidence:.2f} < threshold={threshold}"
                )
                return

        if new_stage != old_stage:
            current_time = time.time()
            time_since_last = current_time - self._last_transition_time

            # Check for thrashing (rapid oscillation)
            if self._is_thrashing():
                if time_since_last < self.THRASHING_COOLDOWN_SECONDS:
                    logger.debug(
                        "Stage thrashing dampened: %s->%s (cooldown %ds)",
                        old_stage.name,
                        new_stage.name,
                        int(self.THRASHING_COOLDOWN_SECONDS),
                    )
                    return

            # Enforce normal cooldown
            if time_since_last < self.TRANSITION_COOLDOWN_SECONDS:
                logger.debug(
                    f"Stage transition blocked by cooldown: {old_stage.name} -> {new_stage.name} "
                    f"(waited {time_since_last:.1f}s, need {self.TRANSITION_COOLDOWN_SECONDS}s)"
                )
                return

            logger.info(
                f"Stage transition: {old_stage.name} -> {new_stage.name} "
                f"(confidence: {confidence:.2f})"
            )

            # Build context for hooks
            hook_context = self._build_hook_context(confidence)

            # Fire hooks if registered (Observer Pattern)
            if self._hooks:
                self._hooks.fire_transition(
                    old_stage.name,
                    new_stage.name,
                    hook_context,
                )

            # Update state
            self.state.stage = new_stage
            self.state._stage_confidence = confidence
            self._last_transition_time = current_time
            self._transition_count += 1

            # Sync to canonical state manager if provided
            if self._state_manager:
                self._sync_state_to_manager()

            # Emit STATE event for stage transition
            if self._event_bus:
                try:
                    from victor.core.events.emit_helper import emit_event_sync

                    emit_event_sync(
                        self._event_bus,
                        topic="state.stage_changed",
                        data={
                            "old_stage": old_stage.name,
                            "new_stage": new_stage.name,
                            "confidence": confidence,
                            "transition_count": self._transition_count,
                            "message_count": self.state.message_count,
                            "tools_executed": len(self.state.tool_history),
                            "files_observed": len(self.state.observed_files),
                            "files_modified": len(self.state.modified_files),
                            "category": "state",  # Preserve for observability
                        },
                        source="ConversationStateMachine",
                    )
                except Exception as e:
                    logger.debug(f"Failed to emit stage change event: {e}")

            # Record transition history
            if self._track_history:
                self._record_transition(
                    old_stage, new_stage, confidence, current_time, hook_context
                )

    def _build_hook_context(self, confidence: float) -> Dict[str, Any]:
        """Build context dictionary for hook callbacks.

        Args:
            confidence: Transition confidence.

        Returns:
            Context dictionary with state information.
        """
        return {
            "confidence": confidence,
            "tool_history": list(self.state.tool_history[-10:]),  # Last 10 tools
            "last_tools": list(self.state.last_tools),
            "message_count": self.state.message_count,
            "files_observed": len(self.state.observed_files),
            "files_modified": len(self.state.modified_files),
            "transition_count": self._transition_count,
        }

    def _record_transition(
        self,
        old_stage: ConversationStage,
        new_stage: ConversationStage,
        confidence: float,
        timestamp: float,
        context: Dict[str, Any],
    ) -> None:
        """Record a transition in history.

        Args:
            old_stage: Previous stage.
            new_stage: New stage.
            confidence: Transition confidence.
            timestamp: Unix timestamp.
            context: Transition context.
        """
        from datetime import datetime

        record = {
            "from_stage": old_stage.name,
            "to_stage": new_stage.name,
            "confidence": confidence,
            "timestamp": timestamp,
            "datetime": datetime.fromtimestamp(timestamp).isoformat(),
            "transition_number": self._transition_count,
            "message_count": context.get("message_count", 0),
            "tool_count": len(context.get("tool_history", [])),
        }

        self._transition_history.append(record)

        # Enforce max history size
        if len(self._transition_history) > self._max_history_size:
            self._transition_history.pop(0)

    @property
    def transition_history(self) -> List[Dict[str, Any]]:
        """Get the transition history.

        Returns:
            List of transition records.
        """
        return list(self._transition_history)

    @property
    def transition_count(self) -> int:
        """Get the total number of transitions.

        Returns:
            Number of transitions.
        """
        return self._transition_count

    def get_transitions_summary(self) -> Dict[str, Any]:
        """Get a summary of all transitions.

        Returns:
            Dictionary with transition statistics.
        """
        if not self._transition_history:
            return {
                "total_transitions": 0,
                "unique_paths": 0,
                "transitions_by_stage": {},
                "average_confidence": 0.0,
            }

        # Count transitions by path
        path_counts: Dict[str, int] = {}
        stage_entries: Dict[str, int] = {}
        total_confidence = 0.0

        for record in self._transition_history:
            path = f"{record['from_stage']}->{record['to_stage']}"
            path_counts[path] = path_counts.get(path, 0) + 1
            stage_entries[record["to_stage"]] = stage_entries.get(record["to_stage"], 0) + 1
            total_confidence += record["confidence"]

        return {
            "total_transitions": len(self._transition_history),
            "unique_paths": len(path_counts),
            "transitions_by_path": path_counts,
            "transitions_by_stage": stage_entries,
            "average_confidence": total_confidence / len(self._transition_history),
        }

    def set_hooks(self, hooks: "StateHookManager") -> None:
        """Set or replace the hook manager.

        Args:
            hooks: StateHookManager instance.
        """
        self._hooks = hooks

    def clear_hooks(self) -> None:
        """Remove all hooks."""
        self._hooks = None

    def should_include_tool(self, tool_name: str) -> bool:
        """Check if a tool should be included based on current stage.

        This provides a soft recommendation - tools outside the current
        stage are not excluded, just deprioritized.

        Args:
            tool_name: Name of the tool

        Returns:
            True if tool is recommended for current stage
        """
        stage_tools = self.get_stage_tools()

        # Always include if in current stage's tools
        if tool_name in stage_tools:
            return True

        # Also include if in adjacent stages (flexible)
        current_idx = STAGE_ORDER[self.state.stage]
        for stage in ConversationStage:
            if abs(STAGE_ORDER[stage] - current_idx) <= 1:
                if tool_name in self._get_tools_for_stage(stage):
                    return True

        return False

    def get_tool_priority_boost(self, tool_name: str) -> float:
        """Get priority boost for a tool based on current stage.

        Args:
            tool_name: Name of the tool

        Returns:
            Boost value (0.0 to 0.2) to add to similarity score
        """
        if tool_name in self.get_stage_tools():
            return 0.15  # High boost for stage-relevant tools

        # Check adjacent stages
        current_idx = STAGE_ORDER[self.state.stage]
        for stage in ConversationStage:
            if abs(STAGE_ORDER[stage] - current_idx) == 1:
                if tool_name in self._get_tools_for_stage(stage):
                    return 0.08  # Medium boost for adjacent stage tools

        return 0.0  # No boost

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the state machine to a dictionary.

        Returns:
            Dictionary representation suitable for JSON serialization.
        """
        return {
            "state": self.state.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationStateMachine":
        """Restore state machine from a dictionary.

        Args:
            data: Dictionary containing serialized state machine.

        Returns:
            Restored ConversationStateMachine instance.
        """
        machine = cls()
        if "state" in data:
            machine.state = ConversationState.from_dict(data["state"])
        return machine
