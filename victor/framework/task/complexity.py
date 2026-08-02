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

"""Task complexity classification for intelligent tool budgeting.

This module provides framework-level task classification services.
It determines task complexity and appropriate tool budgets.

Design Principles:
- SRP: Only handles classification and budgets (no prompt hints)
- Prompt hints are handled by enrichment/strategies/complexity_hints.py
- Framework-level service usable by all verticals

SOLID Compliance (Phase 2 Refactoring):
- Uses TASK_TYPE_TO_COMPLEXITY from victor.classification (Single Source of Truth)
- Optionally uses PatternMatcher for fast pattern-based classification
- Reduces pattern duplication with classification module

Usage:
    from victor.framework.task import TaskComplexityService, TaskComplexity

    service = TaskComplexityService()
    result = service.classify("refactor the authentication module")
    print(f"Complexity: {result.complexity}, Budget: {result.tool_budget}")
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

from .protocols import TaskClassification, TaskClassifierProtocol, TaskComplexity

# TASK_TYPE_TO_COMPLEXITY / get_pattern_matcher come from victor.classification
# (Single Source of Truth) but are imported at call sites: a module-scope import
# closes the cycle victor.classification -> victor.framework -> ... -> here and
# breaks any direct `import victor.classification` entry.

# Per-turn tool budgets are sourced from the framework-wide budget policy so the
# complexity-derived budgets cannot drift away from BUDGET_LIMITS again.
from victor.config.orchestrator_constants import BUDGET_LIMITS

logger = logging.getLogger(__name__)


# Consolidated pattern definitions for classification
PATTERNS: Dict[TaskComplexity, List[Tuple[str, float, str]]] = {
    TaskComplexity.SIMPLE: [
        (
            r"\b(list|show|display)\s+.*?(files?|directories?|folders?)\b",
            1.0,
            "list_files",
        ),
        (r"\bwhat\s+files\s+(are\s+)?(in|at)\b", 1.0, "what_files"),
        (r"\bgit\s+(status|log|branch)\b", 1.0, "git_status"),
        (
            r"\b(show|what|get)\s+(the\s+)?(current\s+)?(git\s+)?status\b",
            0.9,
            "status_query",
        ),
        (r"\bpwd\b|\bcurrent\s+(directory|dir|folder)\b", 1.0, "pwd"),
        # ls command - exclude ls= (Python kwarg like linestyle) and ls| (pipe)
        (r"(?:^|\s)ls\s+(?![\=\'])", 0.9, "ls_command"),
        (r"\bcount\s+(files?|lines?|words?)\b", 0.9, "count_query"),
    ],
    TaskComplexity.MEDIUM: [
        (
            r"\b(explain|describe|summarize)\s+(the\s+)?(file|code|function|class)\b",
            0.9,
            "explain_code",
        ),
        (r"\bread\s+(and\s+)?(explain|summarize)\b", 0.9, "read_explain"),
        (
            r"\bfind\s+(all\s+)?(classes?|functions?|methods?)\b",
            0.8,
            "find_definitions",
        ),
        (r"\bwhere\s+(is|are|does)\b", 0.8, "where_query"),
        (r"\bhow\s+(does|do|is)\s+.+\s+(work|implemented)\b", 0.9, "how_works"),
    ],
    TaskComplexity.COMPLEX: [
        (r"\brefactor\b", 0.9, "refactor"),
        (
            r"\b(implement|add|create)\s+(a\s+)?(new\s+)?(feature|system|module)\b",
            0.9,
            "implement_feature",
        ),
        (r"\b(migrate|upgrade)\s+", 0.9, "migration"),
        (
            r"\b(restructure|reorganize)\s+(the\s+)?(project|code|module|layout)\b",
            0.9,
            "restructure",
        ),
        (r"\bconvert\s+.+\s+(to|into)\s+", 0.9, "convert_code"),
        (r"\bconsolidate\s+(the\s+)?(code|duplicate|files?)\b", 0.9, "consolidate"),
        # Bug fix patterns - common in issue reports
        (r"\b(failed|fails|failing)\s+to\s+\w+", 1.0, "bug_fix"),
        (
            r"\b(unexpected|wrong|incorrect)\s+(behavior|result|output|error)\b",
            0.95,
            "bug_behavior",
        ),
        (r"\braise[sd]?\s+\w*Error\b", 0.9, "raises_error"),
        (r"\bfix\s+(the\s+)?(bug|issue|error|problem)\b", 1.0, "fix_bug"),
        (r"\b(bug|issue)\s*(report|#\d+)?\s*:", 0.9, "bug_report"),
        # SWE-bench narrative patterns (GitHub issue style)
        (r"\bthe\s+issue\s+is\s+(that\s+)?", 1.0, "swe_issue_narrative"),
        (
            r"\bwhen\s+\w+.+\s+(fails?|breaks?|raises?|throws?)\b",
            0.95,
            "swe_when_fails",
        ),
        (r"\b(test|tests?)\s+(fails?|failing|broken)\b", 1.0, "swe_test_fails"),
        (
            r"\bTraceback\b|\bAttributeError\b|\bTypeError\b|\bValueError\b",
            1.0,
            "swe_traceback",
        ),
        (
            r"\bexpected\s+.+\s+but\s+(got|received|returns?)\b",
            0.95,
            "swe_expected_but",
        ),
        (r"\bdoes\s+not\s+(work|function|import|load|return)\b", 0.95, "swe_does_not"),
        (r"\b(import|importing)\s+.+\s+(fails?|error)\b", 0.95, "swe_import_fails"),
        (
            r"\bmodule\s+.+\s+(not\s+found|missing|unavailable)\b",
            0.95,
            "swe_module_missing",
        ),
    ],
    TaskComplexity.GENERATION: [
        (
            r"\b(create|write|generate)\s+(a\s+)?(simple\s+)?(function|script|code)\b",
            1.0,
            "generate_code",
        ),
        (
            r"\bwrite\s+(a\s+)?(python|javascript|bash|ruby|go|rust)\s+(script|program|code)\b",
            1.0,
            "write_lang_script",
        ),
        (
            r"\bcomplete\s+(this|the)\s+(function|code|implementation)\b",
            1.0,
            "complete_function",
        ),
        (
            r"\bshow\s+(me\s+)?(a\s+)?code\s+(example|sample)\b",
            0.95,
            "show_code_example",
        ),
        (r"\bshow\s+me\s+code\s+for\b", 0.95, "show_code_for"),
        (r"\bimplement\s+(the\s+)?function\s+to\s+pass\b", 0.95, "implement_function"),
        # Lower weight - often appears in issue reports as code examples
        (r"\bdef\s+\w+\s*\([^)]*\)\s*:", 0.6, "function_definition"),
        (r'"""\s*\n.*?>>>', 0.7, "doctest_pattern"),
    ],
    TaskComplexity.ACTION: [
        (r"\bgit\s+(add|commit|push|pull|merge|rebase|stash)\b", 1.0, "git_command"),
        (
            r"\b(run|execute)\s+(the\s+)?(tests?|pytest|unittest|jest|mocha)\b",
            1.0,
            "run_tests",
        ),
        (r"\bpytest\b|\bnpm\s+test\b|\bcargo\s+test\b", 1.0, "test_command"),
        (
            r"\b(build|compile|deploy)\s+(the\s+)?(project|app|code)\b",
            0.9,
            "build_deploy",
        ),
        (
            r"\b(perform|do|run)\s+(a\s+)?(web\s*search|websearch)\b",
            1.0,
            "web_search_action",
        ),
        # SWE-bench / GitHub issue format (multi-step issue resolution)
        (
            r"###\s*(Description|Expected\s+behavior|Actual\s+behavior|Steps)",
            1.0,
            "github_issue_format",
        ),
        (r"\bSteps\s+to\s+Reproduce\b", 1.0, "steps_to_reproduce"),
        (
            r"\bPlease\s+(fix|resolve|address)\s+(this|the)\s+(issue|bug)\b",
            0.95,
            "issue_request",
        ),
    ],
    TaskComplexity.ANALYSIS: [
        (
            r"\b(comprehensive|thorough|detailed|full)\s+(analysis|review|audit)\b",
            1.0,
            "detailed_analysis",
        ),
        (
            r"\barchitecture\s+(review|analysis|overview)\b",
            1.0,
            "architecture_analysis",
        ),
        (
            r"\b(analyze|review|audit)\s+(the\s+)?(entire\s+)?(codebase|project|code)\b",
            1.0,
            "analyze_codebase",
        ),
        (
            r"\b(explain|describe)\s+(the\s+)?(entire|whole|full)\s+(codebase|project|system)\b",
            1.0,
            "explain_codebase",
        ),
        (r"\b(security|vulnerability)\s+(audit|scan|review)\b", 0.9, "security_audit"),
    ],
}


# Consolidated budget configuration per complexity level
# All limits are derived from complexity - single source of truth
@dataclass
class ComplexityBudget:
    """Consolidated budget configuration for a task complexity level.

    All components (adapter, completion detector, orchestrator) should use
    this instead of having their own hardcoded limits.
    """

    tool_budget: int  # Max tool calls
    max_turns: int  # Max conversation turns
    max_continuation_requests: int  # Max "need more info" before force stop
    timeout_seconds: int  # Total task timeout
    per_tool_timeout_seconds: int = 30  # Per-tool-call timeout

    @classmethod
    def for_complexity(cls, complexity: "TaskComplexity") -> "ComplexityBudget":
        """Get budget config for a complexity level."""
        return COMPLEXITY_BUDGETS.get(complexity, COMPLEXITY_BUDGETS[TaskComplexity.MEDIUM])


# Single source of truth for all budget-related configs.
#
# Tool budgets are derived from BUDGET_LIMITS (victor.config.orchestrator_constants)
# so that the per-turn budget enforced by the chat/streaming path stays aligned with
# the framework-wide budget policy. Previously these values were hardcoded small
# numbers (MEDIUM=15, ANALYSIS=60) that had drifted away from BUDGET_LIMITS
# (medium=50, analysis=500). A real multi-file analysis task classified as MEDIUM
# would then exhaust its 15-call budget after a handful of `ls`/`read` calls and stop
# before producing anything. The framework's design intent (documented on
# BUDGET_LIMITS) is generous exploration bounded by task-completion detection, not by
# tiny hard caps — so budgets are sourced from BUDGET_LIMITS to prevent re-drift.
COMPLEXITY_BUDGETS: Dict[TaskComplexity, ComplexityBudget] = {
    TaskComplexity.SIMPLE: ComplexityBudget(
        tool_budget=BUDGET_LIMITS.simple_task,  # 20
        max_turns=10,
        max_continuation_requests=4,
        timeout_seconds=120,
        per_tool_timeout_seconds=15,
    ),
    TaskComplexity.MEDIUM: ComplexityBudget(
        tool_budget=BUDGET_LIMITS.medium_task,  # 50
        max_turns=20,
        max_continuation_requests=10,
        timeout_seconds=300,
        per_tool_timeout_seconds=30,
    ),
    TaskComplexity.COMPLEX: ComplexityBudget(
        tool_budget=BUDGET_LIMITS.complex_task,  # 100
        max_turns=40,
        max_continuation_requests=20,
        timeout_seconds=600,
        per_tool_timeout_seconds=60,
    ),
    TaskComplexity.GENERATION: ComplexityBudget(
        tool_budget=BUDGET_LIMITS.simple_task,  # 20
        max_turns=10,
        max_continuation_requests=4,
        timeout_seconds=120,
        per_tool_timeout_seconds=15,
    ),
    TaskComplexity.ACTION: ComplexityBudget(
        tool_budget=BUDGET_LIMITS.action_task,  # 200
        max_turns=60,
        max_continuation_requests=30,
        timeout_seconds=900,
        per_tool_timeout_seconds=60,
    ),
    TaskComplexity.ANALYSIS: ComplexityBudget(
        tool_budget=BUDGET_LIMITS.analysis_task,  # 500
        max_turns=80,
        max_continuation_requests=40,
        timeout_seconds=1200,
        per_tool_timeout_seconds=60,
    ),
}

# Legacy: Simple tool budgets for backward compatibility
DEFAULT_BUDGETS: Dict[TaskComplexity, int] = {
    complexity: budget.tool_budget for complexity, budget in COMPLEXITY_BUDGETS.items()
}

# LEGACY: String-based mapping for backwards compatibility
# Prefer using CLASSIFICATION_TASK_TYPE_TO_COMPLEXITY from victor.classification
# which uses TaskType enum keys directly (Single Source of Truth)
TASK_TYPE_TO_COMPLEXITY: Dict[str, TaskComplexity] = {
    # Core task types
    "edit": TaskComplexity.MEDIUM,
    "search": TaskComplexity.MEDIUM,
    "create": TaskComplexity.COMPLEX,
    "create_simple": TaskComplexity.GENERATION,
    "analyze": TaskComplexity.MEDIUM,
    "design": TaskComplexity.ANALYSIS,
    "general": TaskComplexity.MEDIUM,
    "action": TaskComplexity.ACTION,
    "analysis_deep": TaskComplexity.COMPLEX,
    # Bug fix / issue resolution (SWE-bench style)
    "bug_fix": TaskComplexity.ACTION,
    "issue_resolution": TaskComplexity.ACTION,
    # Research vertical task types
    "literature_review": TaskComplexity.COMPLEX,
    "trend_research": TaskComplexity.COMPLEX,
    "technical_research": TaskComplexity.COMPLEX,
    "competitive_analysis": TaskComplexity.COMPLEX,
    "fact_check": TaskComplexity.MEDIUM,
    "general_query": TaskComplexity.MEDIUM,
    # Coding vertical task types
    "code_generation": TaskComplexity.GENERATION,
    "refactor": TaskComplexity.COMPLEX,
    "debug": TaskComplexity.MEDIUM,
    "test": TaskComplexity.MEDIUM,
    # DevOps vertical task types
    "infrastructure": TaskComplexity.COMPLEX,
    "ci_cd": TaskComplexity.COMPLEX,
    "dockerfile": TaskComplexity.MEDIUM,
    "docker_compose": TaskComplexity.MEDIUM,
    "kubernetes": TaskComplexity.COMPLEX,
    "terraform": TaskComplexity.COMPLEX,
    "monitoring": TaskComplexity.COMPLEX,
    # Data Analysis vertical task types
    "data_analysis": TaskComplexity.COMPLEX,
    "data_profiling": TaskComplexity.MEDIUM,
    "statistical_analysis": TaskComplexity.COMPLEX,
    "correlation_analysis": TaskComplexity.MEDIUM,
    "regression": TaskComplexity.COMPLEX,
    "clustering": TaskComplexity.COMPLEX,
    "time_series": TaskComplexity.COMPLEX,
    "visualization": TaskComplexity.MEDIUM,
}


class TaskComplexityService:
    """Framework service for task complexity classification.

    This service classifies user messages into complexity levels and
    provides appropriate tool budgets. It does NOT provide prompt hints -
    those are handled by the enrichment pipeline.

    Example:
        service = TaskComplexityService()
        result = service.classify("refactor authentication")
        print(f"Budget: {result.tool_budget}")
    """

    def __init__(
        self,
        budgets: Optional[Dict[TaskComplexity, int]] = None,
        custom_patterns: Optional[Dict[TaskComplexity, List[Tuple[str, float, str]]]] = None,
        custom_classifiers: Optional[List[Callable[[str], Optional[TaskClassification]]]] = None,
        use_semantic: bool = True,
        semantic_threshold: float = 0.65,
        use_rl: bool = True,
        llm_decision_service: Optional[Any] = None,
    ) -> None:
        """Initialize the complexity service.

        Inference chain (opinionated defaults, all optional):
        1. Custom classifiers (extensibility)
        2. High-confidence regex patterns (fast, deterministic)
        3. Semantic embeddings (meaning-based classification)
        4. RL-adjusted patterns (learned from outcomes)
        5. Score-based regex (aggregate pattern weights)
        6. Conservative default (MEDIUM)

        Args:
            budgets: Custom budget overrides per complexity level
            custom_patterns: Additional regex patterns for classification
            custom_classifiers: Custom classifier functions to try first
            use_semantic: Whether to use semantic classification (embeddings)
            semantic_threshold: Confidence threshold for semantic classification
            use_rl: Whether to use RL-based adjustments (learns from outcomes)
            llm_decision_service: Optional LLM decision service for edge model classification
        """
        self.budgets = budgets or DEFAULT_BUDGETS.copy()
        self.custom_classifiers = custom_classifiers or []
        self.use_semantic = use_semantic
        self.semantic_threshold = semantic_threshold
        self.use_rl = use_rl
        self._llm_decision_service = llm_decision_service
        self._semantic_classifier = None
        self._rl_learner = None

        # Compile regex patterns
        self._patterns: Dict[TaskComplexity, List[Tuple[re.Pattern, float, str]]] = {
            complexity: [] for complexity in TaskComplexity
        }
        for complexity, patterns in PATTERNS.items():
            self._add_patterns(complexity, patterns)
        if custom_patterns:
            for complexity, patterns in custom_patterns.items():
                self._add_patterns(complexity, patterns)

    def _add_patterns(
        self, complexity: TaskComplexity, patterns: List[Tuple[str, float, str]]
    ) -> None:
        """Add regex patterns for a complexity level."""
        for pattern_str, weight, name in patterns:
            try:
                self._patterns[complexity].append(
                    (re.compile(pattern_str, re.IGNORECASE), weight, name)
                )
            except re.error as e:
                logger.warning(f"Invalid pattern '{pattern_str}': {e}")

    def _classify_fast(self, message: str) -> Optional[TaskClassification]:
        """Fast classification using PatternMatcher from classification module.

        Uses the unified pattern registry for quick pattern-based classification.
        Falls back to None if no high-confidence match is found.
        """
        try:
            from victor.classification import (
                TASK_TYPE_TO_COMPLEXITY as CLASSIFICATION_TASK_TYPE_TO_COMPLEXITY,
                get_pattern_matcher,
            )

            matcher = get_pattern_matcher()
            pattern = matcher.match(message)
            if pattern and pattern.confidence >= 0.9:
                # Map TaskType to TaskComplexity using classification module's mapping
                complexity = CLASSIFICATION_TASK_TYPE_TO_COMPLEXITY.get(
                    pattern.task_type, TaskComplexity.MEDIUM
                )
                return TaskClassification(
                    complexity=complexity,
                    tool_budget=self.budgets[complexity],
                    confidence=pattern.confidence,
                    matched_patterns=[f"fast:{pattern.name}"],
                )
        except Exception as e:
            logger.debug(f"Fast classification failed: {e}")
        return None

    def _get_semantic_classifier(self):
        """Lazy-load the semantic classifier."""
        if self._semantic_classifier is None and self.use_semantic:
            try:
                from victor.storage.embeddings.task_classifier import TaskTypeClassifier

                self._semantic_classifier = TaskTypeClassifier.get_instance(
                    threshold=self.semantic_threshold
                )
            except ImportError:
                logger.warning("TaskTypeClassifier not available, using regex only")
                self.use_semantic = False
        return self._semantic_classifier

    def _get_rl_learner(self):
        """Lazy-load the RL complexity learner.

        Uses the RL coordinator's learner registry if available.
        Learns from task outcomes to adjust complexity predictions.
        """
        if self._rl_learner is None and self.use_rl:
            try:
                from victor.framework.rl.coordinator import RLCoordinator

                coordinator = RLCoordinator.get_instance()
                # Check if complexity learner is registered
                self._rl_learner = coordinator.get_learner("complexity")
                if self._rl_learner is None:
                    logger.debug("Complexity RL learner not registered, using inference only")
                    self.use_rl = False
            except (ImportError, Exception) as e:
                logger.debug(f"RL integration not available: {e}")
                self.use_rl = False
        return self._rl_learner

    def record_outcome(
        self,
        message: str,
        predicted: TaskComplexity,
        success: bool,
        actual_tools_used: int,
        actual_turns: int,
    ) -> None:
        """Record classification outcome for RL learning.

        Call this after task completion to help the classifier learn.
        The RL learner adjusts future predictions based on outcomes.

        Args:
            message: Original task message
            predicted: Complexity that was predicted/used
            success: Whether task completed successfully
            actual_tools_used: How many tools were actually used
            actual_turns: How many conversation turns occurred

        Example:
            # After task completion
            service.record_outcome(
                message="fix the authentication bug",
                predicted=TaskComplexity.MEDIUM,
                success=True,
                actual_tools_used=8,
                actual_turns=5,
            )
        """
        learner = self._get_rl_learner()
        if not learner:
            return

        try:
            # Calculate reward based on outcome
            # Positive: completed with reasonable resource usage
            # Negative: timeout, too many tools, or failure
            budget = COMPLEXITY_BUDGETS[predicted]
            tool_ratio = actual_tools_used / budget.tool_budget
            turn_ratio = actual_turns / budget.max_turns

            if success and tool_ratio <= 1.0 and turn_ratio <= 1.0:
                reward = 1.0 - (tool_ratio + turn_ratio) / 4  # Reward efficiency
            elif success:
                reward = 0.2  # Completed but over budget
            else:
                reward = -0.5  # Failed

            learner.record(
                state={"message_hash": hash(message[:100])},
                action=predicted.value,
                reward=reward,
                context={
                    "message_preview": message[:200],
                    "actual_tools": actual_tools_used,
                    "actual_turns": actual_turns,
                },
            )
            logger.debug(
                f"RL: Recorded complexity outcome: {predicted.value}, "
                f"reward={reward:.2f}, success={success}"
            )
        except Exception as e:
            logger.debug(f"RL outcome recording failed: {e}")

    def _classify_with_edge_model(self, message: str) -> Optional[TaskClassification]:
        """Attempt classification using edge model for ambiguous tasks.

        Handles complex prompts (e.g., SWE-bench issue text) that regex
        patterns can't reliably classify.
        """
        service = self._llm_decision_service
        if service is None:
            return None

        try:
            from victor.agent.decisions.chain import should_use_llm

            if not should_use_llm("task_type_classification"):
                return None

            from victor.agent.decisions.schemas import DecisionType

            decision = service.decide_sync(
                DecisionType.TASK_TYPE_CLASSIFICATION,
                context={"message_excerpt": message[:300]},
                heuristic_confidence=0.0,
            )

            if decision.source != "llm" or decision.confidence < 0.6:
                return None

            task_type = getattr(decision.result, "task_type", None)
            if not task_type:
                return None

            # Map task_type to TaskComplexity
            type_to_complexity = {
                "action": TaskComplexity.ACTION,
                "edit": TaskComplexity.ACTION,
                "analysis": TaskComplexity.ANALYSIS,
                "generation": TaskComplexity.GENERATION,
                "search": TaskComplexity.SIMPLE,
            }
            complexity = type_to_complexity.get(task_type, TaskComplexity.MEDIUM)

            result = TaskClassification(
                complexity=complexity,
                tool_budget=self.budgets[complexity],
                confidence=decision.confidence,
                matched_patterns=[f"edge:{task_type}"],
            )
            logger.info(
                f"Edge complexity classification: {complexity.value} "
                f"(task_type={task_type}, confidence={decision.confidence:.2f})"
            )
            return result

        except Exception as e:
            logger.debug(f"Edge complexity classification failed: {e}")
            return None

    def _classify_semantic(self, message: str) -> Optional[TaskClassification]:
        """Attempt semantic classification using embeddings."""
        classifier = self._get_semantic_classifier()
        if not classifier:
            return None

        try:
            from victor.classification import (
                TASK_TYPE_TO_COMPLEXITY as CLASSIFICATION_TASK_TYPE_TO_COMPLEXITY,
            )

            result = classifier.classify_sync(message)
            # Use CLASSIFICATION_TASK_TYPE_TO_COMPLEXITY from classification module
            # This uses TaskType enum keys directly (Single Source of Truth)
            complexity = CLASSIFICATION_TASK_TYPE_TO_COMPLEXITY.get(
                result.task_type, TaskComplexity.MEDIUM
            )
            if result.confidence >= self.semantic_threshold:
                return TaskClassification(
                    complexity=complexity,
                    tool_budget=self.budgets[complexity],
                    confidence=result.confidence,
                    matched_patterns=[f"semantic:{result.task_type.value}"]
                    + [m[0] for m in result.top_matches[:3]],
                )
        except Exception as e:
            logger.debug(f"Semantic classification failed: {e}")
        return None

    def classify(self, message: str) -> TaskClassification:
        """Classify a message's task complexity.

        Args:
            message: User message to classify

        Returns:
            TaskClassification with complexity, budget, and metadata
        """
        # Empty messages default to SIMPLE (no semantic meaning, no patterns match)
        if not message or not message.strip():
            return TaskClassification(
                complexity=TaskComplexity.SIMPLE,
                tool_budget=self.budgets[TaskComplexity.SIMPLE],
                confidence=0.3,
                matched_patterns=[],
            )

        from victor.framework.task.direct_response import (
            classify_direct_response_prompt,
        )

        direct_response = classify_direct_response_prompt(message)
        if direct_response.is_direct_response:
            tool_budget = 1 if direct_response.is_exact_response else 2
            return TaskClassification(
                complexity=TaskComplexity.SIMPLE,
                tool_budget=tool_budget,
                confidence=direct_response.confidence,
                matched_patterns=[f"direct_response:{direct_response.reason}"],
            )

        # Try custom classifiers first
        for classifier in self.custom_classifiers:
            result = classifier(message)
            if result is not None:
                return result

        # Check high-confidence patterns (order matters: specific before general)
        for complexity in [
            TaskComplexity.SIMPLE,
            TaskComplexity.GENERATION,
            TaskComplexity.ACTION,
            TaskComplexity.MEDIUM,
            TaskComplexity.COMPLEX,
            TaskComplexity.ANALYSIS,
        ]:
            for pattern, weight, name in self._patterns[complexity]:
                if weight >= 0.9 and pattern.search(message):
                    return TaskClassification(
                        complexity=complexity,
                        tool_budget=self.budgets[complexity],
                        confidence=0.95,
                        matched_patterns=[name],
                    )

        # Try semantic classification
        if self.use_semantic:
            semantic_result = self._classify_semantic(message)
            if semantic_result:
                return semantic_result

        # Try edge model classification (handles complex prompts like SWE-bench issues)
        edge_result = self._classify_with_edge_model(message)
        if edge_result:
            return edge_result

        # Score all patterns
        scores: Dict[TaskComplexity, Tuple[float, List[str]]] = {}
        for complexity, patterns in self._patterns.items():
            total_score = 0.0
            matched: List[str] = []
            for pattern, weight, name in patterns:
                if pattern.search(message):
                    total_score += weight
                    matched.append(name)
            if matched:
                scores[complexity] = (total_score, matched)

        # Default to MEDIUM if no patterns match
        if not scores:
            return TaskClassification(
                complexity=TaskComplexity.MEDIUM,
                tool_budget=self.budgets[TaskComplexity.MEDIUM],
                confidence=0.3,
                matched_patterns=[],
            )

        # Return highest scoring complexity
        winner, (score, matched) = max(scores.items(), key=lambda x: x[1][0])
        return TaskClassification(
            complexity=winner,
            tool_budget=self.budgets[winner],
            confidence=min(1.0, score / 2.0 + 0.3),
            matched_patterns=matched,
        )

    def get_budget(self, complexity: TaskComplexity) -> int:
        """Get tool budget for a complexity level."""
        return self.budgets.get(complexity, DEFAULT_BUDGETS[TaskComplexity.MEDIUM])

    def update_budget(self, complexity: TaskComplexity, budget: int) -> None:
        """Update budget for a complexity level."""
        self.budgets[complexity] = budget


# Convenience functions for common operations
def classify_task(message: str) -> TaskClassification:
    """Classify a task message using default service."""
    return TaskComplexityService().classify(message)


def get_budget_for_task(message: str, base_multiplier: float = 1.0) -> int:
    """Get tool budget for a task message."""
    return int(classify_task(message).tool_budget * base_multiplier)


def is_action_task(message: str) -> bool:
    """Check if a message represents an action task."""
    return classify_task(message).complexity == TaskComplexity.ACTION


def is_analysis_task(message: str) -> bool:
    """Check if a message represents an analysis task."""
    return classify_task(message).complexity == TaskComplexity.ANALYSIS
