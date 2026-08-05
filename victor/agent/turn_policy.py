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

"""Shared turn policy for batch and streaming execution paths.

This module provides unified decision logic for both
TurnExecutor (batch) and StreamingChatExecutor (streaming):

- SpinDetector: detects blocked/stuck agent loops
- NudgePolicy: determines when/what nudge messages to inject
- FulfillmentCriteriaBuilder: auto-derives file-level criteria from tool results

Both execution paths import from this module to ensure consistent
behavior. No path-specific logic belongs here — only shared decisions.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

from victor.core.loop_thresholds import DEFAULT_BLOCKED_CONSECUTIVE_THRESHOLD
from victor.tools.tool_names import get_canonical_name

if TYPE_CHECKING:
    from victor.framework.search_novelty import SearchNoveltyTracker

logger = logging.getLogger(__name__)


# ============================================================================
# Shared Constants
# ============================================================================

MAX_NO_TOOL_TURNS = 3
"""Maximum consecutive turns without tool calls before termination."""

MAX_ALL_BLOCKED = DEFAULT_BLOCKED_CONSECUTIVE_THRESHOLD
"""Maximum consecutive turns where all tool calls are dedup-blocked."""

NUDGE_THRESHOLD = 2
"""Inject nudge message after this many no-tool turns."""

READ_ONLY_TOOLS = frozenset({"read", "ls", "grep"})
"""Tools considered read-only for code_search escalation nudge."""

READ_ONLY_ESCALATION_THRESHOLD = 5
"""Consecutive read-only turns before suggesting code_search."""


# ============================================================================
# SpinDetectorConfig
# ============================================================================


@dataclass
class SpinDetectorConfig:
    """Configurable thresholds for SpinDetector and NudgePolicy.

    Default values reproduce the legacy module-level constant behavior.
    Verticals or sessions can override by passing a custom config instance.
    """

    max_no_tool_turns: int = MAX_NO_TOOL_TURNS
    repetition_threshold: int = 3
    read_only_escalation_threshold: int = READ_ONLY_ESCALATION_THRESHOLD
    read_only_tools: frozenset = field(default_factory=lambda: frozenset({"read", "ls", "grep"}))


# ============================================================================
# SpinDetector
# ============================================================================


class SpinState(Enum):
    """Current spin detection state."""

    NORMAL = "normal"
    WARNING = "warning"  # Approaching limit
    BLOCKED = "blocked"  # All tools blocked by dedup
    STUCK = "stuck"  # No tools used for too long
    TERMINATED = "terminated"  # Limit exceeded


@dataclass
class SpinDetector:
    """Detects stuck/blocked agent loops.

    Tracks consecutive turns without tool calls and consecutive turns
    where all tool calls are blocked by dedup. Also tracks repetitive
    tool calls across turns to break loops.

    Example:
        detector = SpinDetector()

        for each turn:
            detector.record_turn(has_tools=True, all_blocked=False)
            state = detector.state
            if state == SpinState.TERMINATED:
                break
    """

    consecutive_no_tool_turns: int = 0
    consecutive_all_blocked: int = 0
    consecutive_read_only_turns: int = 0
    total_tool_calls: int = 0
    has_used_code_search: bool = False

    # Repetition tracking
    _turn_signatures: List[Set[str]] = field(default_factory=list)
    _repetition_count: int = 0
    REPETITION_THRESHOLD: int = 3  # kept for backward compat; config takes precedence

    # Configurable thresholds (defaults reproduce legacy behavior)
    config: SpinDetectorConfig = field(default_factory=SpinDetectorConfig)

    @property
    def state(self) -> SpinState:
        """Current spin state based on tracking counters."""
        if self.consecutive_all_blocked >= MAX_ALL_BLOCKED:
            return SpinState.TERMINATED
        if self.consecutive_no_tool_turns >= self.config.max_no_tool_turns:
            return SpinState.TERMINATED
        if self._repetition_count >= self.config.repetition_threshold - 1:
            return SpinState.TERMINATED
        if self.consecutive_all_blocked >= 2:
            return SpinState.BLOCKED
        if self.consecutive_no_tool_turns >= NUDGE_THRESHOLD:
            return SpinState.WARNING
        return SpinState.NORMAL

    def record_turn(
        self,
        has_tool_calls: bool,
        all_blocked: bool = False,
        tool_names: Optional[Set[str]] = None,
        tool_count: int = 0,
        tool_signatures: Optional[Set[str]] = None,
    ) -> SpinState:
        """Record a turn and return updated state.

        Args:
            has_tool_calls: Whether model requested tool calls
            all_blocked: Whether all tool calls were blocked by dedup
            tool_names: Set of tool names used (for read-only tracking)
            tool_count: Number of tool calls in this turn
            tool_signatures: Set of call signatures (tool:args) for this turn

        Returns:
            Updated SpinState
        """
        if has_tool_calls:
            self.consecutive_no_tool_turns = 0
            self.total_tool_calls += tool_count

            if all_blocked:
                self.consecutive_all_blocked += 1
            else:
                self.consecutive_all_blocked = 0

            # Track read-only turns for code_search escalation
            if tool_names:
                canonical_tool_names = {get_canonical_name(tool) for tool in tool_names}
                if "code_search" in canonical_tool_names:
                    self.has_used_code_search = True
                if canonical_tool_names.issubset(READ_ONLY_TOOLS):
                    self.consecutive_read_only_turns += 1
                else:
                    self.consecutive_read_only_turns = 0

            # Check for repetition across turns
            if tool_signatures:
                # Compare with previous turns
                is_repetitive = False
                for prev_signatures in self._turn_signatures[-2:]:  # Check last 2 turns
                    if tool_signatures == prev_signatures:
                        is_repetitive = True
                        break

                if is_repetitive:
                    self._repetition_count += 1
                else:
                    self._repetition_count = 0

                self._turn_signatures.append(tool_signatures)
                if len(self._turn_signatures) > 10:
                    self._turn_signatures.pop(0)
        else:
            self.consecutive_no_tool_turns += 1

        return self.state

    def reset(self) -> None:
        """Reset all counters for a new conversation."""
        self.consecutive_no_tool_turns = 0
        self.consecutive_all_blocked = 0
        self.consecutive_read_only_turns = 0
        self.total_tool_calls = 0
        self.has_used_code_search = False
        self._turn_signatures.clear()
        self._repetition_count = 0


# ============================================================================
# NudgePolicy
# ============================================================================


class NudgeType(Enum):
    """Types of nudge messages."""

    NONE = "none"
    USE_TOOLS = "use_tools"  # Agent not using tools
    DIFFERENT_TOOLS = "different_tools"  # All tools blocked by dedup
    CODE_SEARCH = "code_search"  # Too many read-only turns
    BUDGET_WARNING = "budget_warning"  # Past halfway on iteration budget
    REPETITION_BREAK = "repetition_break"  # Break repetition loop immediately


@dataclass
class NudgeDecision:
    """Decision about whether and what to nudge.

    Attributes:
        nudge_type: Type of nudge needed
        message: Nudge message text (user or system role)
        role: Message role ("user" or "system")
        should_inject: Whether to inject the nudge
    """

    nudge_type: NudgeType = NudgeType.NONE
    message: str = ""
    role: str = "user"

    @property
    def should_inject(self) -> bool:
        return self.nudge_type != NudgeType.NONE


class NudgePolicy:
    """Determines when and what nudge messages to inject.

    Uses SpinDetector state to decide. Both batch and streaming paths
    call this to get consistent nudge behavior.

    Example:
        policy = NudgePolicy()
        decision = policy.evaluate(detector, iteration=5, max_iterations=10)
        if decision.should_inject:
            chat_context.add_message(decision.role, decision.message)
    """

    def __init__(self, config: Optional[SpinDetectorConfig] = None) -> None:
        self._config = config or SpinDetectorConfig()

    def evaluate(
        self,
        detector: SpinDetector,
        iteration: int = 0,
        max_iterations: int = 10,
        intent: Optional[Any] = None,
    ) -> NudgeDecision:
        """Evaluate whether a nudge is needed.

        Args:
            detector: Current spin detector state
            iteration: Current iteration number
            max_iterations: Maximum iterations
            intent: Current ActionIntent (used to tailor nudge for write tasks)

        Returns:
            NudgeDecision with nudge type and message
        """
        state = detector.state

        # All tools blocked by dedup
        if state == SpinState.BLOCKED:
            return NudgeDecision(
                nudge_type=NudgeType.DIFFERENT_TOOLS,
                message=(
                    "Your last tool calls were blocked because you already "
                    "called them with the same arguments. Try a DIFFERENT "
                    "tool or different arguments. If you've made your fix, "
                    "provide your final answer."
                ),
                role="user",
            )

        # Agent not using tools
        if state == SpinState.WARNING:
            # After 2+ turns without tools, use stronger language
            if detector.consecutive_no_tool_turns >= 2:
                nudge = NudgeDecision(
                    nudge_type=NudgeType.REPETITION_BREAK,
                    message=(
                        f"STOP. You have not called any tools in {detector.consecutive_no_tool_turns} turns. "
                        f"This is a repetition loop. You MUST:\n"
                        f"1. Use a tool NOW (read, edit, write, shell, code_search)\n"
                        f"2. OR provide your final answer if the task is complete\n"
                        f"Do NOT respond with more text analysis. Take action or conclude."
                    ),
                    role="user",
                )
                return nudge
            else:
                nudge = NudgeDecision(
                    nudge_type=NudgeType.USE_TOOLS,
                    message=(
                        f"You have not called any tools in the last "
                        f"{detector.consecutive_no_tool_turns} turns. You MUST "
                        f"use a tool now (read, edit, write, shell) to make "
                        f"progress on the task. Do not respond with text only."
                    ),
                    role="user",
                )
                return nudge

        # Too many read-only turns: tailor based on intent
        if detector.consecutive_read_only_turns >= self._config.read_only_escalation_threshold:
            _is_write_task = (
                intent is not None and hasattr(intent, "value") and intent.value == "write_allowed"
            )
            if _is_write_task:
                # Model has gathered enough context — push it to edit, not search
                return NudgeDecision(
                    nudge_type=NudgeType.CODE_SEARCH,
                    message=(
                        "You've been reading files for several turns without making changes. "
                        "You have enough context. Stop reading and apply the fix now using:\n"
                        'edit(ops=[{"type": "replace", "path": "file.py", '
                        '"old_str": "exact text from file", "new_str": "replacement"}])\n'
                        "Use the exact text you already read as old_str."
                    ),
                    role="user",
                )
            if not detector.has_used_code_search:
                return NudgeDecision(
                    nudge_type=NudgeType.CODE_SEARCH,
                    message=(
                        "You have been browsing files for several turns. "
                        "Consider using code_search(query='...') to find "
                        "relevant code more efficiently."
                    ),
                    role="user",
                )

        return NudgeDecision()

    def budget_warning(
        self,
        iteration: int,
        max_iterations: int,
    ) -> NudgeDecision:
        """Check if a budget warning should be issued.

        Args:
            iteration: Current iteration
            max_iterations: Maximum iterations

        Returns:
            NudgeDecision with budget warning if past halfway
        """
        if iteration > max_iterations // 2:
            remaining = max_iterations - iteration
            return NudgeDecision(
                nudge_type=NudgeType.BUDGET_WARNING,
                message=(
                    f"WARNING: {remaining} turns remaining out of "
                    f"{max_iterations}. Make your edits NOW."
                ),
                role="user",
            )
        return NudgeDecision()


# ============================================================================
# FulfillmentCriteriaBuilder
# ============================================================================


@dataclass
class FulfillmentCriteria:
    """Auto-derived fulfillment criteria from tool execution results.

    Built by analyzing tool calls to determine what files were created/modified,
    what tests were run, etc. Used by FulfillmentDetector for completion checking.
    """

    file_paths: List[str] = field(default_factory=list)
    test_files: List[str] = field(default_factory=list)
    original_error: Optional[str] = None
    required_patterns: List[str] = field(default_factory=list)
    doc_files: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to criteria dict for FulfillmentDetector."""
        criteria: Dict[str, Any] = {}
        if self.file_paths:
            criteria["file_path"] = self.file_paths[0]  # Primary file
        if self.test_files:
            criteria["test_files"] = self.test_files
        if self.original_error:
            criteria["original_error"] = self.original_error
        if self.required_patterns:
            criteria["required_patterns"] = self.required_patterns
        if self.doc_files:
            criteria["doc_files"] = self.doc_files
        return criteria


class FulfillmentCriteriaBuilder:
    """Builds fulfillment criteria from tool execution results.

    Analyzes tool calls to determine what files were created, modified,
    or tested. This enables auto-derived fulfillment checking without
    requiring explicit criteria from the user.

    Example:
        builder = FulfillmentCriteriaBuilder()
        for tool_result in turn.tool_results:
            builder.record_tool_result(tool_result)
        criteria = builder.build()
    """

    def __init__(self) -> None:
        self._written_files: List[str] = []
        self._edited_files: List[str] = []
        self._test_files: List[str] = []
        self._doc_files: List[str] = []
        self._errors: List[str] = []

    def record_tool_result(self, result: Dict[str, Any]) -> None:
        """Record a tool execution result.

        Args:
            result: Tool result dict with tool_name, args, success, etc.
        """
        tool_name = result.get("tool_name", "")
        args = result.get("args", {})
        success = result.get("success", False)
        canonical_tool_name = get_canonical_name(tool_name)

        if not success:
            error = result.get("error", "")
            if error:
                self._errors.append(error)
            return

        # Track file operations
        file_path = args.get("file_path", "") or args.get("path", "")

        if canonical_tool_name in ("write", "create_file"):
            if file_path:
                self._written_files.append(file_path)
                if file_path.endswith(".md") or file_path.endswith(".rst"):
                    self._doc_files.append(file_path)

        elif canonical_tool_name in ("edit", "replace_in_file"):
            if file_path:
                self._edited_files.append(file_path)

        elif canonical_tool_name in ("shell", "run_command"):
            command = args.get("cmd") or args.get("command", "")
            if "pytest" in command or "test" in command:
                # Extract test file if present
                parts = command.split()
                for part in parts:
                    if part.endswith(".py") and "test" in part:
                        self._test_files.append(part)

    def build(self) -> FulfillmentCriteria:
        """Build fulfillment criteria from recorded tool results.

        Returns:
            FulfillmentCriteria with auto-derived fields
        """
        all_files = self._written_files + self._edited_files
        return FulfillmentCriteria(
            file_paths=all_files,
            test_files=self._test_files,
            original_error=self._errors[0] if self._errors else None,
            doc_files=self._doc_files,
        )

    def reset(self) -> None:
        """Reset builder for next conversation."""
        self._written_files.clear()
        self._edited_files.clear()
        self._test_files.clear()
        self._doc_files.clear()
        self._errors.clear()


# =============================================================================
# Shared per-turn evaluation: content-repetition, plateau, and the controller
# that both the headless (AgenticLoop) and streaming (StreamingChatExecutor)
# loops call so the "continue / nudge / stop" decision lives in ONE place.
# =============================================================================


def evaluate_overlap_repetition(overlap: float, repetition_count: int) -> Tuple[int, str]:
    """Decide repetition handling from word-overlap between consecutive turns.

    Returns ``(updated_repetition_count, action)`` where ``action`` is one of:
      - ``"near_duplicate"`` — ``overlap >= 0.8``: a single strong repeat is enough to
        force completion (breaks a narration spin immediately).
      - ``"high_overlap"``  — ``overlap > 0.5`` and the running count has reached 2.
      - ``"accumulating"``  — ``overlap > 0.5`` but only the first such turn so far.
      - ``"reset"``         — ``overlap <= 0.3``: genuinely distinct, clear the count.
      - ``"hold"``          — ``0.3 < overlap <= 0.5``: ambiguous; keep the count without
        decaying it, so an oscillating loop still converges to the threshold.
    """
    if overlap >= 0.8:
        return repetition_count + 1, "near_duplicate"
    if overlap > 0.5:
        new_count = repetition_count + 1
        return new_count, ("high_overlap" if new_count >= 2 else "accumulating")
    if overlap <= 0.3:
        return 0, "reset"
    return repetition_count, "hold"


# Actions that mean "stop now — the agent is repeating itself."
_TERMINAL_REPETITION_ACTIONS = frozenset({"exact_repeat", "near_duplicate", "high_overlap"})


class ContentRepetitionDetector:
    """Detect a narration/output spin from consecutive assistant content.

    Canonical implementation (previously copy-pasted into both loops, with the headless
    loop using a cruder content-length heuristic). Uses an exact MD5-hash match over the
    last 3 turns plus a Jaccard word-overlap check via :func:`evaluate_overlap_repetition`.
    Stateful per conversation; call :meth:`record` once per turn with the turn's full
    assistant content and read ``action`` — the caller decides how to stop (the headless
    loop marks the turn FAILED, the streaming loop force-completes and emits a chunk).
    """

    def __init__(self, min_content_len: int = 20, history: int = 5) -> None:
        self._min_content_len = min_content_len
        self._history = history
        self._content_hashes: List[str] = []
        self._repetition_count = 0
        self._prev_full_content = ""

    def record(self, content: Optional[str]) -> str:
        """Record one turn's content; return a repetition action.

        Actions: ``"exact_repeat"`` / ``"near_duplicate"`` / ``"high_overlap"`` are terminal
        (see :data:`_TERMINAL_REPETITION_ACTIONS`); ``"accumulating"`` / ``"hold"`` /
        ``"reset"`` / ``"none"`` are non-terminal.
        """
        if not content or len(content.strip()) <= self._min_content_len:
            return "none"

        normalized = re.sub(r"\s+", " ", content.strip().lower())
        content_hash = hashlib.md5(normalized.encode()).hexdigest()
        self._content_hashes.append(content_hash)
        if len(self._content_hashes) > self._history:
            self._content_hashes.pop(0)

        # Exact repeat: the same content three turns running.
        if len(self._content_hashes) >= 3 and len(set(self._content_hashes[-3:])) == 1:
            self._repetition_count += 1
            return "exact_repeat"

        action = "none"
        if self._prev_full_content and len(self._prev_full_content) > 50:
            prev_words = set(re.sub(r"\s+", " ", self._prev_full_content.strip().lower()).split())
            curr_words = set(normalized.split())
            if prev_words and curr_words:
                overlap = len(prev_words & curr_words) / len(prev_words | curr_words)
                self._repetition_count, action = evaluate_overlap_repetition(
                    overlap, self._repetition_count
                )
        else:
            self._repetition_count = 0

        self._prev_full_content = content
        return action

    @property
    def repetition_count(self) -> int:
        return self._repetition_count

    def reset(self) -> None:
        self._content_hashes.clear()
        self._repetition_count = 0
        self._prev_full_content = ""


@dataclass
class PlateauResult:
    """Outcome of a plateau check for one turn."""

    is_plateau: bool = False
    should_nudge: bool = False
    score: float = 0.0
    recent_scores: List[float] = field(default_factory=list)


class PlateauDetector:
    """Detect a progress plateau from a productivity-weighted score history.

    Canonical implementation using the streaming loop's formula (more informative than the
    headless loop's bare evaluation score): ``productive_count * 0.3 + min(content_len/2000,
    0.7)``. A plateau is the last ``window`` scores spanning < ``tolerance`` while still
    below ``ceiling``. A plateau only warrants a nudge when the turn was *unproductive*
    (productive_count == 0) — real work (read → search → edit) flattens the score legitimately.
    """

    def __init__(self, window: int = 3, tolerance: float = 0.05, ceiling: float = 0.8) -> None:
        self._window = window
        self._tolerance = tolerance
        self._ceiling = ceiling
        self._scores: List[float] = []

    def record(self, productive_count: int, content_len: int) -> PlateauResult:
        score = min(1.0, (productive_count * 0.3 + min(content_len / 2000, 0.7)))
        self._scores.append(score)
        if len(self._scores) < self._window:
            return PlateauResult(score=score, recent_scores=list(self._scores))
        recent = self._scores[-self._window :]
        is_plateau = (max(recent) - min(recent) < self._tolerance) and recent[-1] < self._ceiling
        # Productive turns flatten the score legitimately — don't nudge those.
        should_nudge = is_plateau and productive_count == 0
        return PlateauResult(
            is_plateau=is_plateau,
            should_nudge=should_nudge,
            score=score,
            recent_scores=recent,
        )

    @property
    def scores(self) -> List[float]:
        return self._scores

    def reset(self) -> None:
        self._scores.clear()


@dataclass(frozen=True)
class TurnObservation:
    """Normalized per-turn input both loops adapt to from their native turn output.

    Lets one :class:`TurnEvaluationController` serve a buffered ``TurnResult`` (headless) and
    the streaming ``ToolExecutionResult`` (streaming) without either loop's execution code
    leaking into the shared decision logic.
    """

    content: str = ""
    productive_count: int = 0
    has_tool_calls: bool = False
    all_blocked: bool = False
    tool_names: Set[str] = field(default_factory=set)
    tool_count: int = 0
    tool_signatures: Optional[Set[str]] = None
    iteration: int = 1
    max_iterations: int = 1
    intent_is_write: bool = False
    intent: Optional[Any] = None
    tool_results: Optional[List[Dict[str, Any]]] = None


@dataclass(frozen=True)
class TurnDecision:
    """The shared continue/nudge/stop verdict for one turn; the loop applies it.

    ``stop`` ends the loop. ``terminal_success`` distinguishes a clean finish (map to
    COMPLETE / force-completion) from a degraded stop (map to FAIL). ``stop_message`` is the
    user-facing chunk text for the streaming loop. ``nudge_message`` (if set) is injected by
    the loop via its own message channel (headless ``chat_ctx.add_message`` vs streaming
    ``orch.add_message``), so injection mechanics stay loop-local.
    """

    stop: bool = False
    terminal_success: bool = False
    stop_reason: Optional[str] = None
    stop_message: Optional[str] = None
    nudge_message: Optional[str] = None
    nudge_role: str = "system"
    nudge_kind: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


_REPETITION_STOP_MESSAGE = (
    "\n\n[Content repetition detected — stopping to prevent infinite output loop.]"
)


class TurnEvaluationController:
    """One per-turn decision point shared by the headless and streaming loops.

    Composes the already-shared guards (:class:`SpinDetector`, :class:`NudgePolicy`) with the
    now-shared :class:`ContentRepetitionDetector` and :class:`PlateauDetector`. Both loops
    feed a :class:`TurnObservation` and apply the returned :class:`TurnDecision`, so spin /
    repetition / plateau / nudge logic exists once. Turn *execution* (streamed vs buffered)
    stays loop-local. Convergence (fulfillment-completion + search-novelty) plugs in here in a
    follow-up so both loops gain it at once.
    """

    def __init__(
        self,
        spin_detector: Optional[SpinDetector] = None,
        nudge_policy: Optional[NudgePolicy] = None,
        content_repetition: Optional[ContentRepetitionDetector] = None,
        plateau: Optional[PlateauDetector] = None,
        search_novelty: Optional["SearchNoveltyTracker"] = None,
        *,
        enable_budget_warning: bool = True,
        enable_plateau_nudge: bool = True,
        enable_search_novelty: bool = True,
        min_iterations_before_force_complete: int = 2,
        enable_fulfillment_complete: bool = True,
        fulfillment_summary_min_chars: int = 800,
        fulfillment_min_findings: int = 3,
        min_iterations_before_fulfillment: int = 4,
    ) -> None:
        from victor.framework.search_novelty import SearchNoveltyTracker

        self.spin_detector = spin_detector or SpinDetector()
        self.nudge_policy = nudge_policy or NudgePolicy()
        self.content_repetition = content_repetition or ContentRepetitionDetector()
        self.plateau = plateau or PlateauDetector()
        self.search_novelty = search_novelty or SearchNoveltyTracker()
        # Per-loop preservation of genuine current differences: the headless loop owns plateau
        # via its own adaptive-termination (FAIL/extend) and emits budget warnings; the streaming
        # loop nudges on plateau and never warned on budget. Flags keep each exact until a
        # deliberate convergence pass.
        self._enable_budget_warning = enable_budget_warning
        self._enable_plateau_nudge = enable_plateau_nudge
        self._enable_search_novelty = enable_search_novelty
        self._min_iterations_before_force_complete = min_iterations_before_force_complete
        # Fulfillment-complete: stop the redundant turns once a substantial answer exists AND
        # enough findings were gathered AND the latest search is no longer adding much. The
        # large summary threshold + findings floor + low-novelty gate keep it conservative.
        self._enable_fulfillment_complete = enable_fulfillment_complete
        self._fulfillment_summary_min_chars = fulfillment_summary_min_chars
        self._fulfillment_min_findings = fulfillment_min_findings
        self._min_iterations_before_fulfillment = min_iterations_before_fulfillment
        self._best_content_len = 0

    def reset(self) -> None:
        self.spin_detector.reset()
        self.content_repetition.reset()
        self.plateau.reset()
        self.search_novelty.reset()
        self._best_content_len = 0

    @classmethod
    def from_exploration_settings(
        cls,
        exploration: Any = None,
        *,
        spin_detector: Optional[SpinDetector] = None,
        nudge_policy: Optional[NudgePolicy] = None,
        enable_plateau_nudge: bool = True,
        enable_budget_warning: bool = True,
        enable_search_novelty: bool = True,
    ) -> "TurnEvaluationController":
        """Build a controller with convergence thresholds from ``ExplorationSettings``.

        ``exploration`` is an ``ExplorationSettings`` (or ``None`` for defaults). Read
        defensively via getattr so a partial/old settings object still works. Per-loop flags
        (plateau/budget/novelty) stay caller-controlled; the numeric thresholds and the
        guard on/off switches come from settings.
        """
        from victor.framework.search_novelty import NoveltyConfig, SearchNoveltyTracker

        def _g(name: str, default: Any) -> Any:
            return getattr(exploration, name, default) if exploration is not None else default

        novelty_cfg = NoveltyConfig.from_exploration(exploration)
        return cls(
            spin_detector=spin_detector,
            nudge_policy=nudge_policy,
            search_novelty=SearchNoveltyTracker(novelty_cfg),
            enable_plateau_nudge=enable_plateau_nudge,
            enable_budget_warning=enable_budget_warning,
            enable_search_novelty=enable_search_novelty
            and bool(_g("search_novelty_guard_enabled", True)),
            min_iterations_before_force_complete=_g("min_iterations_before_force_complete", 2),
            enable_fulfillment_complete=bool(_g("fulfillment_completion_enabled", True)),
            fulfillment_summary_min_chars=_g("fulfillment_summary_min_chars", 800),
            fulfillment_min_findings=_g("fulfillment_min_findings", 3),
            min_iterations_before_fulfillment=_g("min_iterations_before_fulfillment", 4),
        )

    def evaluate(self, observation: TurnObservation, *, record_spin: bool = True) -> TurnDecision:
        """Run the shared per-turn guards and return a single decision.

        ``record_spin`` lets a caller that already feeds its own ``SpinDetector`` (both loops do
        today) skip the re-recording and have the controller read the current spin state — so
        wiring the controller in doesn't double-count turns.
        """
        # 1. Spin detection (shared component; already fed by both loops today).
        if record_spin:
            self.spin_detector.record_turn(
                has_tool_calls=observation.has_tool_calls,
                all_blocked=observation.all_blocked,
                tool_names=observation.tool_names,
                tool_count=observation.tool_count,
                tool_signatures=observation.tool_signatures,
            )

        # 2. Content repetition — a hard stop (degraded) when the agent repeats itself.
        action = self.content_repetition.record(observation.content)
        if action in _TERMINAL_REPETITION_ACTIONS:
            logger.warning(
                "[content-repetition] action=%s (consecutive=%s, content_len=%s) — stopping.",
                action,
                self.content_repetition.repetition_count,
                len(observation.content or ""),
            )
            return TurnDecision(
                stop=True,
                terminal_success=False,
                stop_reason="content_repetition",
                stop_message=_REPETITION_STOP_MESSAGE,
                metadata={"repetition_action": action},
            )

        # 2.5 Search novelty — diminishing returns on successive searches. After persistent
        # saturation, force-complete so the agent SYNTHESIZES the gathered context (a clean
        # finish, not a failure) instead of thrashing distinct queries to the iteration cap.
        novelty = self.search_novelty.record_turn(observation.tool_results)
        _editing = bool(
            {get_canonical_name(t) for t in (observation.tool_names or set())}
            & {"edit", "write", "create_file", "replace_in_file"}
        )
        if (
            self._enable_search_novelty
            and novelty.should_force_complete
            and not _editing  # never cut short a turn that is actively making edits
            and observation.iteration >= self._min_iterations_before_force_complete
        ):
            logger.info(
                "[search-novelty] %d consecutive low-novelty searches (ratio=%.2f) — "
                "force-completing to synthesize.",
                novelty.consecutive_low_novelty,
                novelty.novelty_ratio,
            )
            return TurnDecision(
                stop=True,
                terminal_success=True,
                stop_reason="search_saturated",
                stop_message="\n\n[Enough context gathered — synthesizing the answer.]",
                metadata={"novelty_ratio": novelty.novelty_ratio},
            )

        # 2.6 Fulfillment-complete — stop redundant turns once a SUBSTANTIAL answer has been
        # produced AND enough findings gathered AND the latest search is no longer adding much.
        # Fires earlier than pure saturation (1 low-novelty turn vs N) but only when a real
        # answer already exists, so it trims over-verification without cutting analysis short.
        self._best_content_len = max(self._best_content_len, len(observation.content or ""))
        if (
            self._enable_fulfillment_complete
            and not _editing
            and self._best_content_len >= self._fulfillment_summary_min_chars
            and novelty.total_distinct_hits >= self._fulfillment_min_findings
            and novelty.had_search
            and novelty.consecutive_low_novelty >= 1
            and observation.iteration >= self._min_iterations_before_fulfillment
        ):
            logger.info(
                "[fulfillment] answer produced (%d chars) + %d findings + low-novelty search "
                "— finalizing.",
                self._best_content_len,
                novelty.total_distinct_hits,
            )
            return TurnDecision(
                stop=True,
                terminal_success=True,
                stop_reason="fulfilled",
                stop_message="\n\n[Sufficient findings and an answer produced — finalizing.]",
                metadata={"findings": novelty.total_distinct_hits},
            )

        # 3. Plateau (productivity-weighted) — may warrant a nudge (when enabled for this loop).
        plateau = self.plateau.record(observation.productive_count, len(observation.content or ""))
        plateau_should_nudge = plateau.should_nudge and self._enable_plateau_nudge

        # 4. Nudge selection: spin > synthesize (search saturation) > plateau > budget warning.
        nudge_message: Optional[str] = None
        nudge_role = "system"
        nudge_kind: Optional[str] = None

        spin_nudge = self.nudge_policy.evaluate(
            self.spin_detector,
            iteration=observation.iteration,
            max_iterations=observation.max_iterations,
            intent=observation.intent,
        )
        if spin_nudge.should_inject:
            nudge_message, nudge_role = spin_nudge.message, spin_nudge.role
            nudge_kind = spin_nudge.nudge_type.value
        elif self._enable_search_novelty and novelty.should_nudge:
            from victor.framework.search_novelty import synthesize_nudge_message

            nudge_message = synthesize_nudge_message()
            nudge_kind = "synthesize"
        elif plateau_should_nudge:
            nudge_message = self._plateau_message(observation.intent_is_write)
            nudge_kind = "plateau"
        elif self._enable_budget_warning:
            budget_nudge = self.nudge_policy.budget_warning(
                observation.iteration, observation.max_iterations
            )
            if budget_nudge.should_inject:
                nudge_message, nudge_role = budget_nudge.message, budget_nudge.role
                nudge_kind = budget_nudge.nudge_type.value

        return TurnDecision(
            nudge_message=nudge_message,
            nudge_role=nudge_role,
            nudge_kind=nudge_kind,
            metadata={"plateau": plateau.is_plateau, "score": plateau.score},
        )

    @staticmethod
    def _plateau_message(intent_is_write: bool) -> str:
        return plateau_nudge_message(intent_is_write)


def plateau_nudge_message(intent_is_write: bool) -> str:
    """The shared plateau nudge text (write-intent aware) used by both loops."""
    if intent_is_write:
        return (
            "Progress stalled. You have enough context — stop reading and apply the change "
            'now with edit(ops=[{"type": "replace", "path": "file", "old_str": "exact text", '
            '"new_str": "replacement"}]).'
        )
    return (
        "Progress seems stalled. Try a different approach or summarize what you've found " "so far."
    )
