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

"""Task completion detection for agent workflow optimization.

This module provides mechanisms to detect when a task's objectives have been
achieved, preventing unnecessary continuation loops and improving efficiency.

Issue Reference: workflow-test-issues-v2.md Issue #1
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Set,
    runtime_checkable,
)

from victor.core.completion_markers import (
    SUMMARY_MARKER,
    detect_active_completion_marker,
    strip_active_completion_markers,
)
from victor.agent.action_authorizer import (
    build_write_tool_set,
    normalize_tool_name_for_policy,
)
from victor.tools.core_tool_aliases import canonicalize_core_tool_name
from victor.tools.tool_names import get_canonical_name

if TYPE_CHECKING:
    from victor.agent.presentation import PresentationProtocol

logger = logging.getLogger(__name__)


def _classifier_completion_signal_enabled() -> bool:
    """FEP-0012: whether the shipped local classifier may signal task completion.

    The task_completion head predicts reward buckets (pass/partial/fail). Trusting
    a confident "pass" to STOP the agentic loop is a behavioral change, so it is
    gated off by default and surfaced for Phase-7 A/B validation before it ships
    as a default. See ``DecisionServiceSettings.local_classifier_completion_signal``.
    """
    try:
        from victor.config.decision_settings import DecisionServiceSettings

        return DecisionServiceSettings().local_classifier_completion_signal
    except Exception:
        return False


class DeliverableType(Enum):
    """Types of deliverables an agent can produce."""

    FILE_CREATED = "file_created"
    FILE_MODIFIED = "file_modified"
    FILE_DELETED = "file_deleted"
    PLAN_PROVIDED = "plan_provided"
    ANSWER_PROVIDED = "answer_provided"
    CODE_EXECUTED = "code_executed"
    ANALYSIS_PROVIDED = "analysis_provided"
    ERROR_REPORTED = "error_reported"


# ---------------------------------------------------------------------------
# Deliverable inference helpers
# ---------------------------------------------------------------------------

# Map task_type → default deliverables when LLM omits them
_TASK_TYPE_DELIVERABLE_MAP: Dict[str, List[DeliverableType]] = {
    "action": [DeliverableType.FILE_MODIFIED],
    "edit": [DeliverableType.FILE_MODIFIED],
    "generation": [DeliverableType.FILE_CREATED],
    "analysis": [DeliverableType.ANALYSIS_PROVIDED],
    "search": [DeliverableType.ANSWER_PROVIDED],
}

# Map deliverable string → DeliverableType enum
_DELIVERABLE_STRING_MAP: Dict[str, DeliverableType] = {
    "file_created": DeliverableType.FILE_CREATED,
    "file_modified": DeliverableType.FILE_MODIFIED,
    "analysis_provided": DeliverableType.ANALYSIS_PROVIDED,
    "answer_provided": DeliverableType.ANSWER_PROVIDED,
    "plan_provided": DeliverableType.PLAN_PROVIDED,
    "code_executed": DeliverableType.CODE_EXECUTED,
}


def _infer_deliverables_from_task_type(task_type: str) -> List[DeliverableType]:
    """Infer expected deliverables from a task_type classification.

    Used when the LLM returns a task_type but no explicit deliverables.

    Args:
        task_type: One of "action", "edit", "generation", "analysis", "search"

    Returns:
        List of inferred DeliverableType values
    """
    return list(_TASK_TYPE_DELIVERABLE_MAP.get(task_type, []))


def _parse_deliverables(deliverable_strings: List[str]) -> List[DeliverableType]:
    """Convert deliverable strings from LLM response to DeliverableType enums.

    Args:
        deliverable_strings: List of string deliverable names from LLM

    Returns:
        List of matching DeliverableType values (unrecognized strings are skipped)
    """
    result = []
    for s in deliverable_strings:
        dt = _DELIVERABLE_STRING_MAP.get(s)
        if dt is not None:
            result.append(dt)
    return result


class ResponsePhase(Enum):
    """Phases of agent response for completion detection."""

    EXPLORATION = "exploration"  # Reading files, searching codebase
    SYNTHESIS = "synthesis"  # Summarizing, planning, preparing output
    FINAL_OUTPUT = "final_output"  # Delivering answer, completed work
    BLOCKED = "blocked"  # Cannot complete, needs user input


class CompletionConfidence(Enum):
    """Confidence levels for task completion detection."""

    HIGH = "high"  # Active rare marker detected - deterministic
    MEDIUM = "medium"  # File modifications + passive signal
    LOW = "low"  # Only passive phrase detected
    NONE = "none"  # No completion signal detected


@dataclass
class TaskDeliverable:
    """Represents a completed deliverable."""

    type: DeliverableType
    description: str
    artifact_path: Optional[str] = None
    verified: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskCompletionState:
    """Tracks the completion state of a task.

    Note: max_continuation_requests defaults to MEDIUM complexity budget.
    Use configure_for_complexity() to set based on task complexity.
    """

    expected_deliverables: List[DeliverableType] = field(default_factory=list)
    completed_deliverables: List[TaskDeliverable] = field(default_factory=list)
    completion_signals: Set[str] = field(default_factory=set)
    continuation_requests: int = 0
    # Default to MEDIUM complexity budget (5) - was 2 which caused premature termination
    max_continuation_requests: int = 5
    # Active signal detection flag - set when explicit completion signal detected
    active_signal_detected: bool = False
    # Text extracted from the last VICTOR_SUMMARY:: marker; persisted for next-turn injection
    last_summary: str = ""

    @property
    def is_complete(self) -> bool:
        """Check if all expected deliverables are met."""
        # If we've had too many continuation requests, force completion
        if self.continuation_requests >= self.max_continuation_requests:
            return True

        # If no expected deliverables, check for completion signals
        if not self.expected_deliverables:
            return bool(self.completion_signals) or bool(self.completed_deliverables)

        # Check if all expected types are in completed
        completed_types = {d.type for d in self.completed_deliverables}
        return all(dt in completed_types for dt in self.expected_deliverables)

    @property
    def completion_percentage(self) -> float:
        """Calculate completion percentage."""
        if not self.expected_deliverables:
            return 100.0 if self.is_complete else 0.0

        completed_types = {d.type for d in self.completed_deliverables}
        matched = sum(1 for dt in self.expected_deliverables if dt in completed_types)
        return (matched / len(self.expected_deliverables)) * 100


@runtime_checkable
class ITaskCompletionDetector(Protocol):
    """Protocol for task completion detection."""

    def analyze_intent(self, user_message: str) -> List[DeliverableType]:
        """Infer expected deliverables from user request."""
        ...

    def record_tool_result(self, tool_name: str, result: Dict[str, Any]) -> None:
        """Record deliverables from tool execution."""
        ...

    def analyze_response(self, response_text: str) -> None:
        """Detect completion signals in response text."""
        ...

    def should_stop(self) -> bool:
        """Check if task objectives are met and agent should stop."""
        ...

    def get_completion_summary(self) -> str:
        """Generate summary of completed deliverables."""
        ...

    def reset(self) -> None:
        """Reset state for new task."""
        ...


class TaskCompletionDetector:
    """Detects when a task's objectives have been achieved.

    This detector uses a priority-based approach:
    1. ACTIVE SIGNALS (Priority 1): Explicit completion phrases instructed in system prompt
    2. TOOL EVIDENCE (Priority 2): File modifications recorded from tool results
    3. PASSIVE PHRASES (Priority 3): Fallback phrase detection for models that ignore instructions

    This detector analyzes:
    1. User intent to determine expected deliverables
    2. Tool results to track completed deliverables
    3. Response text for completion signals
    4. Continuation request patterns to detect loops

    Usage:
        detector = TaskCompletionDetector()
        detector.analyze_intent("Create a cache_manager.py file")
        # ... agent executes tools ...
        detector.record_tool_result("write", {"success": True, "path": "cache.py"})
        if detector.should_stop():
            print(detector.get_completion_summary())
    """

    # Priority 1: Active signals - deterministic, instructed in system prompt.
    # Rare terminal markers avoid collisions with normal status prose.

    # Priority 3: Passive phrases indicating task completion (fallback)
    COMPLETION_PHRASES: frozenset = frozenset(
        {
            # File operations
            "successfully created",
            "has been created",
            "file created",
            "has been implemented",
            "implementation complete",
            "successfully written",
            "has been written",
            "file saved",
            # Task completion
            "task complete",
            "task completed",
            "done",
            "finished",
            "completed successfully",
            # Delivery phrases
            "here's the",
            "here is the",
            "i've created",
            "i have created",
            "the file is now",
            "the implementation is",
            # Summary phrases
            "in summary",
            "to summarize",
            "that covers",
            "this completes",
            # Bug fix / SWE-bench completion phrases
            "fix applied",
            "fix has been applied",
            "bug fixed",
            "bug has been fixed",
            "issue resolved",
            "issue has been resolved",
            "the fix is",
            "i've fixed",
            "i have fixed",
            "successfully fixed",
            "has been fixed",
            "patch applied",
            "change applied",
            "modification complete",
            "edit complete",
            # Active completion signaling (instructed in system prompt)
            "task complete:",
            "task complete.",
            # Additional patterns for broader LLM coverage
            "has been modified",
            "has been updated",
            "has been changed",
            "modification is complete",
            "update is complete",
            "changes have been made",
            "file updated",
            "code updated",
            "the changes are",
            "i've updated",
            "i have updated",
            "i've modified",
            "i have modified",
        }
    )

    # Phrases indicating continuation loop (agent asking to continue)
    CONTINUATION_PHRASES: frozenset = frozenset(
        {
            "continue with the implementation",
            "what would you like me to",
            "how would you like to proceed",
            "if you need further",
            "let me know if you",
            "would you like me to",
            "please specify",
            "please provide more details",
            "if there's anything else",
            "any other",
            # Additional patterns for broader LLM coverage
            "shall i proceed",
            "do you want me to",
            "should i continue",
            "need me to",
            "want me to add",
            "want me to make",
            "anything else you'd like",
            "any changes you'd like",
            "any modifications",
        }
    )

    # Tools that produce file deliverables
    WRITE_TOOLS: frozenset = build_write_tool_set("create_file", "delete_file", "rename_file")

    # Tools that execute code
    EXECUTE_TOOLS: frozenset = frozenset(
        {
            "shell",
            "test",
        }
    )

    def __init__(
        self,
        presentation: Optional["PresentationProtocol"] = None,
        decision_service: Optional[Any] = None,
        runtime_intelligence: Optional[Any] = None,
    ):
        """Initialize the task completion detector.

        Args:
            presentation: Optional presentation adapter for icons (creates default if None)
            decision_service: Optional LLMDecisionService for low-confidence augmentation
            runtime_intelligence: Optional canonical runtime-intelligence service
        """
        self._state = TaskCompletionState()
        self._decision_service = decision_service
        self._runtime_intelligence = runtime_intelligence
        self._framework_keyword_detector: Any = None
        self._last_framework_signal: Any = None
        # Lazy init for backward compatibility
        if presentation is None:
            from victor.agent.presentation import create_presentation_adapter

            self._presentation = create_presentation_adapter()
        else:
            self._presentation = presentation
        self._intent_keywords: Dict[str, List[DeliverableType]] = {
            # File creation keywords
            "create": [DeliverableType.FILE_CREATED],
            "add": [DeliverableType.FILE_CREATED],
            "write": [DeliverableType.FILE_CREATED],
            "implement": [DeliverableType.FILE_CREATED, DeliverableType.CODE_EXECUTED],
            "build": [DeliverableType.FILE_CREATED],
            # File modification keywords
            "update": [DeliverableType.FILE_MODIFIED],
            "modify": [DeliverableType.FILE_MODIFIED],
            "change": [DeliverableType.FILE_MODIFIED],
            "fix": [DeliverableType.FILE_MODIFIED],
            "refactor": [DeliverableType.FILE_MODIFIED],
            # Planning keywords
            "plan": [DeliverableType.PLAN_PROVIDED],
            "design": [DeliverableType.PLAN_PROVIDED],
            "how to": [DeliverableType.PLAN_PROVIDED],
            "strategy": [DeliverableType.PLAN_PROVIDED],
            # Analysis keywords
            "what": [DeliverableType.ANSWER_PROVIDED],
            "explain": [DeliverableType.ANSWER_PROVIDED],
            "describe": [DeliverableType.ANSWER_PROVIDED],
            "analyze": [DeliverableType.ANALYSIS_PROVIDED],
            "review": [DeliverableType.ANALYSIS_PROVIDED],
            # Execution keywords
            "run": [DeliverableType.CODE_EXECUTED],
            "execute": [DeliverableType.CODE_EXECUTED],
            "test": [DeliverableType.CODE_EXECUTED],
        }

    def _has_decision_support(self) -> bool:
        """Return whether LLM decision augmentation is available."""
        if self._runtime_intelligence is not None:
            return True
        return self._decision_service is not None

    def _get_framework_keyword_detector(self) -> Any:
        """Lazily resolve the shared framework completion signal detector."""
        if self._framework_keyword_detector is not None:
            return self._framework_keyword_detector

        try:
            from victor.framework.context_aware_keyword_detector import (
                ContextAwareKeywordDetector,
            )

            self._framework_keyword_detector = ContextAwareKeywordDetector()
        except Exception:
            self._framework_keyword_detector = False
        return self._framework_keyword_detector

    def _resolve_framework_task_type(self) -> str:
        """Map expected deliverables into the framework completion task taxonomy."""
        deliverables = list(self._state.expected_deliverables)
        if not deliverables:
            deliverables = [deliverable.type for deliverable in self._state.completed_deliverables]

        if any(
            deliverable in {DeliverableType.FILE_CREATED, DeliverableType.FILE_MODIFIED}
            for deliverable in deliverables
        ):
            return "code_generation"
        if DeliverableType.CODE_EXECUTED in deliverables:
            return "testing"
        if DeliverableType.PLAN_PROVIDED in deliverables:
            return "analysis"
        if DeliverableType.ANALYSIS_PROVIDED in deliverables:
            return "analysis"
        if DeliverableType.ANSWER_PROVIDED in deliverables:
            return "search"
        return "unknown"

    def _should_use_framework_completion_detection(self, response_text: str) -> bool:
        """Gate framework completion signals to contexts with enough structure."""
        task_type = self._resolve_framework_task_type()
        if task_type != "unknown":
            return True
        return "```" in response_text or "##" in response_text

    def _apply_framework_completion_signal(self, response_text: str) -> None:
        """Augment detector state with the shared framework completion signal path."""
        if not self._should_use_framework_completion_detection(response_text):
            self._last_framework_signal = None
            return

        detector = self._get_framework_keyword_detector()
        if not detector:
            self._last_framework_signal = None
            return

        try:
            signal = detector.detect_completion(
                response=response_text,
                task_type=self._resolve_framework_task_type(),
                requirements=None,
            )
        except Exception:
            logger.debug("Framework completion signal detection failed", exc_info=True)
            self._last_framework_signal = None
            return

        self._last_framework_signal = signal

        if signal.is_continuation_request:
            self._state.continuation_requests += 1
            logger.debug("Framework continuation signal detected")
            return

        if signal.has_completion_indicator and not self._state.completion_signals:
            self._state.completion_signals.add("framework:completion")
            logger.debug(
                "Framework completion signal detected (confidence=%.2f)",
                signal.confidence,
            )

    def _decide_sync(
        self,
        decision_type: Any,
        context: Dict[str, Any],
        *,
        heuristic_result: Any = None,
        heuristic_confidence: float = 0.0,
    ) -> Optional[Any]:
        """Route LLM-backed decisions through the canonical runtime service when present."""
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

    def analyze_intent(self, user_message: str) -> List[DeliverableType]:
        """Infer expected deliverables from user request.

        Uses a priority chain:
        1. LLM classification (via decision_service) — understands complex prompts
        2. Regex keyword fallback — fast, no API cost

        Args:
            user_message: The user's request message

        Returns:
            List of expected deliverable types
        """
        # Priority 1: LLM classification via decision service
        if self._has_decision_support():
            try:
                from victor.agent.decisions.chain import should_use_llm

                if not should_use_llm("task_type_classification"):
                    return

                from victor.agent.decisions.schemas import DecisionType

                decision = self._decide_sync(
                    DecisionType.TASK_TYPE_CLASSIFICATION,
                    context={"message_excerpt": user_message[:500]},
                    heuristic_confidence=0.0,
                )
                if (
                    decision is not None
                    and decision.result is not None
                    and decision.source == "llm"
                    and decision.confidence >= 0.6
                    and hasattr(decision.result, "deliverables")
                ):
                    deliverables = _parse_deliverables(decision.result.deliverables)
                    # If LLM returned task_type but no deliverables, infer them
                    if not deliverables and hasattr(decision.result, "task_type"):
                        deliverables = _infer_deliverables_from_task_type(decision.result.task_type)
                    if deliverables:
                        self._state.expected_deliverables = list(deliverables)
                        logger.info(
                            f"LLM intent classification: task_type={decision.result.task_type}, "
                            f"deliverables={self._state.expected_deliverables}, "
                            f"confidence={decision.confidence:.2f}"
                        )
                        return self._state.expected_deliverables
            except Exception as e:
                logger.debug(f"LLM intent classification failed, using regex fallback: {e}")

        # Priority 2: Regex keyword fallback
        deliverables = self._regex_classify_intent(user_message)
        self._state.expected_deliverables = list(deliverables)
        logger.debug(f"Regex intent classification: {self._state.expected_deliverables}")
        return self._state.expected_deliverables

    def _regex_classify_intent(self, user_message: str) -> Set[DeliverableType]:
        """Regex-based intent classification (fallback).

        Args:
            user_message: The user's request message

        Returns:
            Set of expected deliverable types
        """
        message_lower = user_message.lower()
        deliverables: Set[DeliverableType] = set()

        # Check for file extension mentions (strong signal for file creation)
        if re.search(r"\.\w{2,4}\b", user_message):
            deliverables.add(DeliverableType.FILE_CREATED)

        # Check for intent keywords
        for keyword, types in self._intent_keywords.items():
            if keyword in message_lower:
                deliverables.update(types)

        return deliverables

    def record_tool_result(self, tool_name: str, result: Dict[str, Any]) -> None:
        """Record deliverables from tool execution.

        Args:
            tool_name: Name of the tool that was executed
            result: Result dictionary from tool execution
        """
        tool_lower = get_canonical_name(canonicalize_core_tool_name(tool_name.lower()))
        success = result.get("success", True)  # Assume success if not specified

        if not success:
            return

        # Check for file write operations
        if tool_lower in self.WRITE_TOOLS:
            path = result.get("path", result.get("file_path", "unknown"))
            deliverable = TaskDeliverable(
                type=DeliverableType.FILE_CREATED,
                description=f"Created/modified {path}",
                artifact_path=path,
                verified=True,
                metadata={"tool": tool_name},
            )
            self._state.completed_deliverables.append(deliverable)
            logger.debug(f"Recorded file deliverable: {path}")

        # Check for code execution
        elif tool_lower in self.EXECUTE_TOOLS:
            deliverable = TaskDeliverable(
                type=DeliverableType.CODE_EXECUTED,
                description=f"Executed {result.get('command', 'code')[:50]}",
                verified=True,
                metadata={"tool": tool_name, "exit_code": result.get("exit_code", 0)},
            )
            self._state.completed_deliverables.append(deliverable)
            logger.debug("Recorded code execution deliverable")

    def analyze_response(self, response_text: str) -> None:
        """Detect completion signals in response text with priority ordering.

        Priority:
        1. ACTIVE SIGNALS: Explicit completion signals instructed in system prompt
        2. PASSIVE PHRASES: Fallback detection for models ignoring instructions

        Args:
            response_text: The agent's response text
        """
        response_lower = response_text.lower()

        # Priority 1: Check rare, line-anchored active markers.
        active_marker = detect_active_completion_marker(response_text)
        if active_marker:
            signal = active_marker.lower()
            self._state.completion_signals.add(f"active:{signal}")
            self._state.active_signal_detected = True
            logger.info(f"Active completion signal detected: {signal}")
            # Extract and persist VICTOR_SUMMARY content for next-turn context injection.
            # Without this, compaction drops the assistant message and the model loses
            # all pending task state when the user says "continue".
            if active_marker == SUMMARY_MARKER:
                marker_pos = response_text.find(SUMMARY_MARKER)
                if marker_pos != -1:
                    summary_text = strip_active_completion_markers(
                        response_text[marker_pos + len(SUMMARY_MARKER) :]
                    ).strip()
                    if summary_text:
                        self._state.last_summary = summary_text
                        logger.info(
                            f"Persisted VICTOR_SUMMARY for next-turn injection ({len(summary_text)} chars)"
                        )
            return

        # Priority 3: Passive phrase detection (fallback)
        for phrase in self.COMPLETION_PHRASES:
            if phrase in response_lower:
                self._state.completion_signals.add(f"passive:{phrase}")
                logger.debug(f"Passive completion signal: {phrase}")

        # Check for continuation loop patterns
        for phrase in self.CONTINUATION_PHRASES:
            if phrase in response_lower:
                self._state.continuation_requests += 1
                logger.debug(
                    f"Detected continuation request ({self._state.continuation_requests}): {phrase}"
                )
                break

        # Shared framework completion detector augmentation. This keeps legacy
        # callers on TaskCompletionDetector while reusing framework-level
        # completion heuristics when task context is available.
        self._apply_framework_completion_signal(response_text)

        # LLM augmentation: if no active signal found and service available,
        # consult LLM for completion detection (gated by decision chain)
        if (
            not self._state.active_signal_detected
            and not self._state.completion_signals
            and self._has_decision_support()
        ):
            try:
                from victor.agent.decisions.chain import should_use_llm

                if not should_use_llm("task_completion"):
                    return

                from victor.agent.decisions.schemas import DecisionType

                decision = self._decide_sync(
                    DecisionType.TASK_COMPLETION,
                    context={
                        "response_tail": response_text[-500:],
                        "deliverable_count": len(self._state.completed_deliverables),
                        "signal_count": len(self._state.completion_signals),
                    },
                    heuristic_confidence=0.0,
                )
                if (
                    decision is not None
                    and decision.result is not None
                    and (
                        decision.source == "llm"
                        or (
                            decision.source == "local_classifier"
                            and _classifier_completion_signal_enabled()
                        )
                    )
                    and hasattr(decision.result, "is_complete")
                ):
                    if decision.result.is_complete and decision.confidence >= 0.7:
                        self._state.completion_signals.add("llm:task_complete")
                        logger.debug("LLM detected task completion")
                    if decision.result.phase == "stuck":
                        self._state.continuation_requests += 1
                        logger.debug("LLM detected stuck phase")
            except Exception:
                logger.debug(
                    "LLM decision augmentation failed in analyze_response",
                    exc_info=True,
                )

        # Infer deliverables from response content
        # Check both when no expected deliverables AND when expected ones aren't yet met
        completed_types = {d.type for d in self._state.completed_deliverables}

        # If we see plan-like content, record as plan provided
        if DeliverableType.PLAN_PROVIDED not in completed_types and any(
            marker in response_lower
            for marker in ["steps:", "step 1", "## implementation", "here's how"]
        ):
            self._state.completed_deliverables.append(
                TaskDeliverable(
                    type=DeliverableType.PLAN_PROVIDED,
                    description="Implementation plan provided",
                    verified=True,
                )
            )

        # If we see explanation content
        if DeliverableType.ANSWER_PROVIDED not in completed_types and any(
            marker in response_lower
            for marker in ["this file", "the function", "it handles", "this class"]
        ):
            self._state.completed_deliverables.append(
                TaskDeliverable(
                    type=DeliverableType.ANSWER_PROVIDED,
                    description="Explanation provided",
                    verified=True,
                )
            )

        # If we see structured analysis content (tables, bullet lists, findings)
        # record as analysis provided - this handles "review" tasks that expect
        # ANALYSIS_PROVIDED but never trigger it through tool results alone
        if DeliverableType.ANALYSIS_PROVIDED not in completed_types and any(
            marker in response_lower
            for marker in [
                "summary",
                "in summary",
                "findings",
                "analysis",
                "key points",
                "conclusion",
                "exit criteria",
                "## ",  # markdown headers indicate structured output
            ]
        ):
            self._state.completed_deliverables.append(
                TaskDeliverable(
                    type=DeliverableType.ANALYSIS_PROVIDED,
                    description="Analysis provided in response",
                    verified=True,
                )
            )

    def should_stop(self) -> bool:
        """Check if task objectives are met and agent should stop.

        Uses priority-based completion detection:
        1. ACTIVE SIGNAL: Explicit completion signal detected (deterministic)
        2. FILE MODS + SIGNAL: File edits + any completion signal
        3. FILE MODS + CONTINUATION: File edits + continuation requests
        4. STANDARD: All expected deliverables met

        Returns:
            True if agent should stop, False otherwise
        """
        # Priority 1: Active signal detected (deterministic, highest confidence)
        if self._state.active_signal_detected:
            logger.info("Stopping: Active completion signal detected")
            return True

        is_complete = self._state.is_complete

        # Priority 2: File modifications + signal (existing SWE-bench logic)
        # For bug_fix tasks: if we've made file edits, that's a strong completion signal
        # This handles SWE-bench style tasks where the agent edits but doesn't say "done"
        if not is_complete and self._has_file_modifications():
            # If we have file edits + any completion signal, we're done
            if self._state.completion_signals:
                is_complete = True
                logger.info("Bug fix completion: file edits + completion signal detected")
            # If we have file edits + 2+ continuation requests, likely done
            elif self._state.continuation_requests >= 2:
                is_complete = True
                logger.info("Bug fix completion: file edits + continuation requests detected")

        if is_complete:
            logger.info(
                f"Task completion detected: {len(self._state.completed_deliverables)} deliverables, "
                f"{len(self._state.completion_signals)} signals, "
                f"{self._state.continuation_requests} continuation requests"
            )

        return is_complete

    def _has_file_modifications(self) -> bool:
        """Check if any file modification deliverables have been recorded.

        Returns:
            True if file edits/creations detected
        """
        file_types = {DeliverableType.FILE_CREATED, DeliverableType.FILE_MODIFIED}
        return any(d.type in file_types for d in self._state.completed_deliverables)

    def get_completion_summary(self) -> str:
        """Generate summary of completed deliverables.

        Returns:
            Markdown-formatted summary string
        """
        if not self._state.completed_deliverables and not self._state.completion_signals:
            return "No deliverables recorded."

        lines = ["## Task Completion Summary", ""]

        if self._state.completed_deliverables:
            lines.append("### Deliverables")
            for d in self._state.completed_deliverables:
                status = self._presentation.icon(
                    "success" if d.verified else "pending", with_color=False
                )
                path_info = f" ({d.artifact_path})" if d.artifact_path else ""
                lines.append(f"- {status} {d.type.value}: {d.description}{path_info}")
            lines.append("")

        if self._state.completion_signals:
            lines.append(f"### Completion Signals: {len(self._state.completion_signals)}")
            lines.append("")

        lines.append(f"**Completion: {self._state.completion_percentage:.0f}%**")

        return "\n".join(lines)

    def get_state(self) -> TaskCompletionState:
        """Get the current completion state.

        Returns:
            Current TaskCompletionState
        """
        return self._state

    def reset(self) -> None:
        """Reset state for new task."""
        self._state = TaskCompletionState()
        self._last_framework_signal = None
        logger.debug("Task completion detector reset")

    def clear_active_signal(self) -> None:
        """Clear an active completion marker while preserving other task state."""
        if not self._state.active_signal_detected and not self._state.completion_signals:
            return

        self._state.active_signal_detected = False
        self._state.completion_signals = {
            signal for signal in self._state.completion_signals if not signal.startswith("active:")
        }
        logger.debug("Task completion detector active signal cleared")

    def configure_for_complexity(self, complexity: str) -> None:
        """Configure completion limits based on task complexity.

        Args:
            complexity: Complexity level (simple, medium, complex, generation, action, analysis)

        Uses ComplexityBudget from victor.framework.task for consolidated limits.
        """
        try:
            from victor.framework.task.complexity import ComplexityBudget
            from victor.framework.task.protocols import TaskComplexity

            complexity_enum = TaskComplexity(complexity.lower())
            budget = ComplexityBudget.for_complexity(complexity_enum)
            self._state.max_continuation_requests = budget.max_continuation_requests
            logger.debug(
                f"Configured completion for {complexity}: "
                f"max_continuation_requests={budget.max_continuation_requests}"
            )
        except (ValueError, ImportError) as e:
            logger.warning(f"Could not configure for complexity '{complexity}': {e}")

    def force_complete(self, reason: str) -> None:
        """Force task completion with a reason.

        Args:
            reason: Reason for forcing completion
        """
        self._state.completion_signals.add(f"forced:{reason}")
        logger.info(f"Forced task completion: {reason}")

    def detect_response_phase(self, response_text: str) -> ResponsePhase:
        """Detect the current phase of the agent's response.

        This helps distinguish between "thinking" (exploration/synthesis) and "output" (final delivery).

        Args:
            response_text: The agent's response text

        Returns:
            Detected ResponsePhase
        """
        response_lower = response_text.lower()
        # Strip markdown formatting for reliable signal detection
        stripped_lower = self._MARKDOWN_STRIP_RE.sub("", response_lower)

        # Check for blocked signals first
        blocked_signals = [
            "blocked:",
            "cannot_complete:",
            "_blocked_",
            "_cannot_complete_",
            "i cannot",
            "i'm unable to",
            "unable to complete",
        ]
        if any(signal in stripped_lower for signal in blocked_signals):
            return ResponsePhase.BLOCKED

        # Check for final output signals (active completion signals + delivery phrases)
        final_output_indicators = [
            "done:",
            "task_done:",
            "task done:",
            "summary:",
            "task complete:",
            "_done_",
            "_task_done_",
            "_summary_",
            "here's the",
            "here is the",
            "i've created",
            "i have created",
            "the file is now",
            "the implementation is",
            "successfully created",
        ]
        if any(indicator in stripped_lower for indicator in final_output_indicators):
            return ResponsePhase.FINAL_OUTPUT

        # Check for synthesis phase (summarizing, preparing output)
        synthesis_indicators = [
            "in summary",
            "to summarize",
            "in conclusion",
            "let me summarize",
            "here's what i found",
            "here is what i found",
            "based on my analysis",
            "after reviewing",
            "having examined",
        ]
        if any(indicator in response_lower for indicator in synthesis_indicators):
            return ResponsePhase.SYNTHESIS

        # Check for exploration phase (reading, searching)
        exploration_indicators = [
            "let me read",
            "let me check",
            "let me search",
            "let me look",
            "i'll read",
            "i'll check",
            "i'll search",
            "i'll examine",
            "reading",
            "searching",
            "looking at",
            "examining",
            "first, i need to",
            "let me start by",
            "i'll start by",
        ]
        if any(indicator in response_lower for indicator in exploration_indicators):
            return ResponsePhase.EXPLORATION

        # Default: if no clear phase indicators, assume exploration
        # (conservative default - better to continue than stop prematurely)
        return ResponsePhase.EXPLORATION

    def get_completion_confidence(self) -> CompletionConfidence:
        """Get the current completion confidence level.

        This provides a graded assessment of completion likelihood:
        - HIGH: Active signal detected (deterministic)
        - MEDIUM: File modifications + passive completion signal
        - LOW: Only passive completion phrase detected
        - NONE: No completion signals detected

        Returns:
            Current CompletionConfidence level
        """
        # Priority 1: Active signal detected (HIGH confidence - deterministic)
        if self._state.active_signal_detected:
            return CompletionConfidence.HIGH

        # Priority 2: File modifications + any completion signal (MEDIUM confidence)
        if self._has_file_modifications() and self._state.completion_signals:
            return CompletionConfidence.MEDIUM

        # Diagnostic: signals present but no file mods — waiting for edits
        if self._state.completion_signals and not self._has_file_modifications():
            logger.debug(
                "Completion signals detected (%d) but no file modifications — "
                "waiting for edits before marking complete",
                len(self._state.completion_signals),
            )

        # Priority 3: Only passive completion signals (LOW confidence)
        if self._state.completion_signals:
            # Check if signals are only passive phrases (not active)
            has_only_passive = all(
                signal.startswith(("passive:", "framework:"))
                for signal in self._state.completion_signals
            )
            if has_only_passive:
                return CompletionConfidence.LOW

        # Priority 4: No completion signals (NONE confidence)
        heuristic_confidence = CompletionConfidence.NONE

        # LLM augmentation: if confidence is LOW or NONE and service is available
        if (
            heuristic_confidence in (CompletionConfidence.LOW, CompletionConfidence.NONE)
            and self._has_decision_support()
        ):
            try:
                from victor.agent.decisions.chain import should_use_llm

                if not should_use_llm("task_completion"):
                    return heuristic_confidence

                from victor.agent.decisions.schemas import DecisionType

                conf_value = 0.3 if heuristic_confidence == CompletionConfidence.LOW else 0.0
                decision = self._decide_sync(
                    DecisionType.TASK_COMPLETION,
                    context={
                        "response_tail": "",  # Caller can set via analyze_response
                        "deliverable_count": len(self._state.completed_deliverables),
                        "signal_count": len(self._state.completion_signals),
                    },
                    heuristic_result=heuristic_confidence,
                    heuristic_confidence=conf_value,
                )
                if (
                    decision is not None
                    and decision.result is not None
                    and (
                        decision.source == "llm"
                        or (
                            decision.source == "local_classifier"
                            and _classifier_completion_signal_enabled()
                        )
                    )
                    and hasattr(decision.result, "is_complete")
                ):
                    if decision.result.is_complete and decision.confidence >= 0.7:
                        return CompletionConfidence.MEDIUM
            except Exception:
                logger.debug("LLM decision augmentation failed", exc_info=True)

        return heuristic_confidence


def create_task_completion_detector(
    runtime_intelligence: Optional[Any] = None,
) -> TaskCompletionDetector:
    """Factory function for creating TaskCompletionDetector.

    Returns:
        Configured TaskCompletionDetector instance
    """
    return TaskCompletionDetector(runtime_intelligence=runtime_intelligence)
