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

"""Fulfillment Detection - Task completion detection for agentic loops.

This module provides task-specific fulfillment detection strategies:
- Code generation: AST analysis, syntax validation
- Testing: Test execution and result validation
- Debugging: Bug fix verification
- Information retrieval: Relevance and completeness checks
- System setup: Health checks and service validation

Design Principle: Reuse validation patterns from victor/evaluation/baseline_validator.py

Based on research from:
- arXiv:2601.21268 - Meta-evaluation without ground truth
- victor/evaluation/baseline_validator.py - Test validation patterns

Example:
    from victor.framework.fulfillment import FulfillmentDetector, TaskType

    detector = FulfillmentDetector()

    # Check if code generation task is complete
    result = await detector.check_fulfillment(
        task_type=TaskType.CODE_GENERATION,
        criteria={"file_path": "/path/to/file.py"},
        context={"syntax_check": True}
    )

    if result.is_fulfilled:
        print("Task complete!")
    else:
        print(f"Missing: {result.missing_criteria}")
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from victor.framework.perception_integration import Perception


@dataclass
class FulfillmentConfig:
    """Configurable thresholds and weights for fulfillment scoring.

    Based on SubSearch (arXiv:2604.07415) intermediate reward design:
    scores should reflect meaningful progress checkpoints, not just
    binary pass/fail.
    """

    fulfilled_threshold: float = 0.8
    partial_threshold: float = 0.4
    # Per-criterion weights (sum to ~1.0 per strategy)
    file_exists_weight: float = 0.3
    syntax_valid_weight: float = 0.3
    non_empty_weight: float = 0.2
    pattern_weight: float = 0.2
    # Test strategy weights
    test_pass_weight: float = 0.5
    test_error_weight: float = 0.3
    test_files_weight: float = 0.2

    def __post_init__(self) -> None:
        for name, val in [
            ("fulfilled_threshold", self.fulfilled_threshold),
            ("partial_threshold", self.partial_threshold),
        ]:
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"{name} must be in [0.0, 1.0], got {val}")

        code_gen_sum = (
            self.file_exists_weight
            + self.syntax_valid_weight
            + self.non_empty_weight
            + self.pattern_weight
        )
        if abs(code_gen_sum - 1.0) > 0.01:
            raise ValueError(
                f"Code-generation weights (file_exists + syntax_valid + non_empty + pattern) "
                f"must sum to 1.0 (±0.01), got {code_gen_sum:.4f}"
            )

        test_sum = self.test_pass_weight + self.test_error_weight + self.test_files_weight
        if abs(test_sum - 1.0) > 0.01:
            raise ValueError(
                f"Test weights (test_pass + test_error + test_files) "
                f"must sum to 1.0 (±0.01), got {test_sum:.4f}"
            )


DEFAULT_CONFIG = FulfillmentConfig()


class TaskType(Enum):
    """Task types for fulfillment detection."""

    CODE_GENERATION = "code_generation"
    CODE_MODIFICATION = "code_modification"
    DEBUGGING = "debugging"
    TESTING = "testing"
    ANALYSIS = "analysis"
    DOCUMENTATION = "documentation"
    SEARCH = "search"
    SETUP = "setup"
    DEPLOYMENT = "deployment"
    UNKNOWN = "unknown"


class FulfillmentStatus(Enum):
    """Status of task fulfillment."""

    FULFILLED = "fulfilled"  # Task complete
    PARTIAL = "partial"  # Partially complete
    NOT_FULFILLED = "not_fulfilled"  # Not complete
    ERROR = "error"  # Error during check
    UNKNOWN = "unknown"  # Cannot determine


@dataclass
class FulfillmentResult:
    """Result of fulfillment check.

    Attributes:
        status: Overall fulfillment status
        score: Confidence score (0.0-1.0)
        fulfilled_criteria: Criteria that were met
        missing_criteria: Criteria that were not met
        reason: Human-readable explanation
        metadata: Additional metadata
    """

    status: FulfillmentStatus
    score: float
    fulfilled_criteria: List[str] = field(default_factory=list)
    missing_criteria: List[str] = field(default_factory=list)
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_fulfilled(self) -> bool:
        """Check if task is fulfilled."""
        return self.status == FulfillmentStatus.FULFILLED

    @property
    def is_partial(self) -> bool:
        """Check if task is partially complete."""
        return self.status == FulfillmentStatus.PARTIAL

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "score": self.score,
            "fulfilled_count": len(self.fulfilled_criteria),
            "missing_count": len(self.missing_criteria),
            "fulfilled_criteria": self.fulfilled_criteria,
            "missing_criteria": self.missing_criteria,
            "reason": self.reason,
        }


class FulfillmentStrategy:
    """Base class for fulfillment detection strategies.

    Subclasses implement task-specific fulfillment logic.
    """

    async def check(
        self,
        criteria: Dict[str, Any],
        context: Dict[str, Any],
        config: Optional["FulfillmentConfig"] = None,
    ) -> FulfillmentResult:
        """Check if task is fulfilled.

        Args:
            criteria: Task acceptance criteria
            context: Execution context (files, test results, etc.)
            config: Optional FulfillmentConfig with per-criterion weight overrides.
                    Defaults to DEFAULT_CONFIG when not provided.

        Returns:
            FulfillmentResult with status and details
        """
        raise NotImplementedError("Subclasses must implement check()")


class CodeGenerationFulfillment(FulfillmentStrategy):
    """Fulfillment detection for code generation tasks.

    Checks:
    - File exists
    - Valid syntax
    - Code compiles/runs
    - Basic quality metrics
    """

    async def check(
        self,
        criteria: Dict[str, Any],
        context: Dict[str, Any],
        config: Optional["FulfillmentConfig"] = None,
    ) -> FulfillmentResult:
        """Check if code generation is complete."""
        cfg = config if config is not None else DEFAULT_CONFIG
        fulfilled = []
        missing = []
        score = 0.0

        file_path = criteria.get("file_path")
        if not file_path:
            return FulfillmentResult(
                status=FulfillmentStatus.ERROR,
                score=0.0,
                reason="No file_path in criteria",
            )

        path = Path(file_path)

        # Check file exists
        if path.exists():
            fulfilled.append("file_exists")
            score += cfg.file_exists_weight
        else:
            missing.append("file_exists")

        # Check syntax
        if path.exists():
            try:
                with open(path, "r") as f:
                    code = f.read()
                ast.parse(code)
                fulfilled.append("valid_syntax")
                score += cfg.syntax_valid_weight
            except SyntaxError as e:
                missing.append(f"valid_syntax: {e}")

        # Check if file is non-empty
        if path.exists():
            if path.stat().st_size > 0:
                fulfilled.append("non_empty")
                score += cfg.non_empty_weight
            else:
                missing.append("non_empty")

        # Check for required patterns (if specified)
        required_patterns = criteria.get("required_patterns", [])
        if required_patterns and path.exists():
            with open(path, "r") as f:
                content = f.read()
            for pattern in required_patterns:
                if pattern in content:
                    fulfilled.append(f"pattern_{pattern}")
                    score += cfg.pattern_weight / len(required_patterns)
                else:
                    missing.append(f"pattern_{pattern}")

        # Determine status
        if score >= cfg.fulfilled_threshold:
            status = FulfillmentStatus.FULFILLED
        elif score >= cfg.partial_threshold:
            status = FulfillmentStatus.PARTIAL
        else:
            status = FulfillmentStatus.NOT_FULFILLED

        return FulfillmentResult(
            status=status,
            score=score,
            fulfilled_criteria=fulfilled,
            missing_criteria=missing,
            reason=f"Code generation {status.value} (score: {score:.2f})",
        )


class TestingFulfillment(FulfillmentStrategy):
    """Fulfillment detection for testing tasks.

    Reuses patterns from victor/evaluation/baseline_validator.py

    Checks:
    - Tests exist
    - Tests pass
    - Coverage meets requirements
    """

    async def check(
        self,
        criteria: Dict[str, Any],
        context: Dict[str, Any],
        config: Optional["FulfillmentConfig"] = None,
    ) -> FulfillmentResult:
        """Check if testing task is complete."""
        fulfilled = []
        missing = []
        score = 0.0

        # Check test results in context
        test_results = context.get("test_results")
        if test_results:
            # Reuse TestStatus logic from baseline_validator.py
            passed = sum(1 for r in test_results if getattr(r, "passed", False))
            total = len(test_results)

            if total > 0:
                pass_rate = passed / total
                score += pass_rate * 0.5

                if pass_rate >= 0.8:
                    fulfilled.append("tests_passing")
                else:
                    missing.append(f"tests_passing: {passed}/{total} passed")

                # Check for no errors
                errors = sum(
                    1
                    for r in test_results
                    if getattr(r, "error_message", None) and not getattr(r, "passed", False)
                )
                if errors == 0:
                    fulfilled.append("no_test_errors")
                    score += 0.3
                else:
                    missing.append(f"no_test_errors: {errors} errors")
            else:
                missing.append("tests_exist")
        else:
            missing.append("tests_exist")

        # Check test files exist
        test_files = criteria.get("test_files", [])
        if test_files:
            existing_tests = sum(1 for f in test_files if Path(f).exists())
            if existing_tests > 0:
                fulfilled.append(f"test_files_exist: {existing_tests}/{len(test_files)}")
                score += 0.2 * (existing_tests / len(test_files))
            else:
                missing.append("test_files_exist")

        # Determine status
        if score >= 0.8:
            status = FulfillmentStatus.FULFILLED
        elif score >= 0.4:
            status = FulfillmentStatus.PARTIAL
        else:
            status = FulfillmentStatus.NOT_FULFILLED

        return FulfillmentResult(
            status=status,
            score=score,
            fulfilled_criteria=fulfilled,
            missing_criteria=missing,
            reason=f"Testing {status.value} (score: {score:.2f})",
        )


class DebuggingFulfillment(FulfillmentStrategy):
    """Fulfillment detection for debugging tasks.

    Checks:
    - Bug is fixed (tests pass)
    - No regressions (other tests still pass)
    - Error messages gone
    """

    async def check(
        self,
        criteria: Dict[str, Any],
        context: Dict[str, Any],
        config: Optional["FulfillmentConfig"] = None,
    ) -> FulfillmentResult:
        """Check if debugging task is complete."""
        fulfilled = []
        missing = []
        score = 0.0

        # Check if original error is fixed
        original_error = criteria.get("original_error")
        current_errors = context.get("errors", [])

        if original_error:
            if original_error not in str(current_errors):
                fulfilled.append("original_error_fixed")
                score += 0.5
            else:
                missing.append("original_error_fixed")

        # Check test results
        test_results = context.get("test_results")
        if test_results:
            passed = sum(1 for r in test_results if getattr(r, "passed", False))
            total = len(test_results)
            if total > 0 and passed == total:
                fulfilled.append("all_tests_pass")
                score += 0.3
            elif total > 0:
                missing.append(f"all_tests_pass: {passed}/{total}")

        # Check for no new errors
        error_count = len(context.get("errors", []))
        if error_count == 0:
            fulfilled.append("no_errors")
            score += 0.2
        else:
            missing.append(f"no_errors: {error_count} errors")

        # Determine status
        if score >= 0.8:
            status = FulfillmentStatus.FULFILLED
        elif score >= 0.4:
            status = FulfillmentStatus.PARTIAL
        else:
            status = FulfillmentStatus.NOT_FULFILLED

        return FulfillmentResult(
            status=status,
            score=score,
            fulfilled_criteria=fulfilled,
            missing_criteria=missing,
            reason=f"Debugging {status.value} (score: {score:.2f})",
        )


class SearchFulfillment(FulfillmentStrategy):
    """Fulfillment detection for search/retrieval tasks.

    Checks:
    - Results found
    - Relevance score
    - Completeness
    """

    async def check(
        self,
        criteria: Dict[str, Any],
        context: Dict[str, Any],
        config: Optional["FulfillmentConfig"] = None,
    ) -> FulfillmentResult:
        """Check if search task is complete."""
        fulfilled = []
        missing = []
        score = 0.0

        # Check search results
        results = context.get("search_results", [])
        min_results = criteria.get("min_results", 1)

        if len(results) >= min_results:
            fulfilled.append(f"results_found: {len(results)}")
            score += 0.5
        else:
            missing.append(f"results_found: {len(results)}/{min_results}")

        # Check relevance
        avg_relevance = context.get("avg_relevance", 0.0)
        min_relevance = criteria.get("min_relevance", 0.5)

        if avg_relevance >= min_relevance:
            fulfilled.append(f"relevance: {avg_relevance:.2f}")
            score += 0.3
        else:
            missing.append(f"relevance: {avg_relevance:.2f}/{min_relevance}")

        # Check completeness
        required_info = criteria.get("required_info", [])
        found_info = context.get("found_info", [])

        if all(info in found_info for info in required_info):
            fulfilled.append("complete_info")
            score += 0.2
        else:
            missing_required = [info for info in required_info if info not in found_info]
            missing.append(f"complete_info: missing {missing_required}")

        # Determine status
        if score >= 0.8:
            status = FulfillmentStatus.FULFILLED
        elif score >= 0.4:
            status = FulfillmentStatus.PARTIAL
        else:
            status = FulfillmentStatus.NOT_FULFILLED

        return FulfillmentResult(
            status=status,
            score=score,
            fulfilled_criteria=fulfilled,
            missing_criteria=missing,
            reason=f"Search {status.value} (score: {score:.2f})",
        )


class SetupFulfillment(FulfillmentStrategy):
    """Fulfillment detection for setup/installation tasks.

    Checks:
    - Services running
    - Dependencies installed
    - Configuration complete
    """

    async def check(
        self,
        criteria: Dict[str, Any],
        context: Dict[str, Any],
        config: Optional["FulfillmentConfig"] = None,
    ) -> FulfillmentResult:
        """Check if setup task is complete."""
        fulfilled = []
        missing = []
        score = 0.0

        # Check services
        services = criteria.get("services", [])
        running_services = context.get("running_services", [])

        for service in services:
            if service in running_services:
                fulfilled.append(f"service_{service}")
                score += 0.3 / len(services)
            else:
                missing.append(f"service_{service}")

        # Check dependencies
        dependencies = criteria.get("dependencies", [])
        installed_deps = context.get("installed_dependencies", [])

        for dep in dependencies:
            if dep in installed_deps:
                fulfilled.append(f"dependency_{dep}")
                score += 0.2 / len(dependencies)
            else:
                missing.append(f"dependency_{dep}")

        # Check configuration
        config_files = criteria.get("config_files", [])
        for config_file in config_files:
            if Path(config_file).exists():
                fulfilled.append(f"config_{config_file}")
                score += 0.1 / len(config_files)
            else:
                missing.append(f"config_{config_file}")

        # Determine status
        if score >= 0.8:
            status = FulfillmentStatus.FULFILLED
        elif score >= 0.4:
            status = FulfillmentStatus.PARTIAL
        else:
            status = FulfillmentStatus.NOT_FULFILLED

        return FulfillmentResult(
            status=status,
            score=score,
            fulfilled_criteria=fulfilled,
            missing_criteria=missing,
            reason=f"Setup {status.value} (score: {score:.2f})",
        )


def _score_to_status(score: float, config: FulfillmentConfig = DEFAULT_CONFIG) -> FulfillmentStatus:
    """Convert score to status using configurable thresholds."""
    if score >= config.fulfilled_threshold:
        return FulfillmentStatus.FULFILLED
    elif score >= config.partial_threshold:
        return FulfillmentStatus.PARTIAL
    return FulfillmentStatus.NOT_FULFILLED


class AnalysisFulfillment(FulfillmentStrategy):
    """Fulfillment detection for analysis/investigation tasks.

    Checks:
    - Analysis artifacts produced (findings, summaries)
    - Key questions answered
    - Evidence gathered
    """

    async def check(
        self,
        criteria: Dict[str, Any],
        context: Dict[str, Any],
        config: Optional["FulfillmentConfig"] = None,
    ) -> FulfillmentResult:
        """Check if analysis task is complete."""
        fulfilled: List[str] = []
        missing: List[str] = []
        score = 0.0

        # Check if findings were produced
        findings = context.get("findings", [])
        min_findings = criteria.get("min_findings", 1)
        if len(findings) >= min_findings:
            fulfilled.append(f"findings: {len(findings)}")
            score += 0.4
        else:
            missing.append(f"findings: {len(findings)}/{min_findings}")

        # Check if key questions were addressed
        questions = criteria.get("questions", [])
        answered = context.get("answered_questions", [])
        if questions:
            answer_rate = sum(1 for q in questions if q in answered) / len(questions)
            if answer_rate >= 0.8:
                fulfilled.append(f"questions_answered: {answer_rate:.0%}")
                score += 0.3
            else:
                missing.append(f"questions_answered: {answer_rate:.0%}")
            score += answer_rate * 0.1  # Partial credit
        else:
            # No specific questions = lower bar
            if findings:
                score += 0.2

        # Check if summary/conclusion exists
        has_summary = bool(context.get("summary"))
        if has_summary:
            fulfilled.append("summary_produced")
            score += 0.2
        else:
            missing.append("summary_produced")

        status = _score_to_status(score)
        return FulfillmentResult(
            status=status,
            score=score,
            fulfilled_criteria=fulfilled,
            missing_criteria=missing,
            reason=f"Analysis {status.value} (score: {score:.2f})",
        )


class DocumentationFulfillment(FulfillmentStrategy):
    """Fulfillment detection for documentation tasks.

    Checks:
    - Doc files exist and are non-empty
    - Required sections present
    - Examples included
    """

    async def check(
        self,
        criteria: Dict[str, Any],
        context: Dict[str, Any],
        config: Optional["FulfillmentConfig"] = None,
    ) -> FulfillmentResult:
        """Check if documentation task is complete."""
        fulfilled: List[str] = []
        missing: List[str] = []
        score = 0.0

        # Check doc files
        doc_files = criteria.get("doc_files", [])
        if doc_files:
            existing = 0
            for doc_file in doc_files:
                p = Path(doc_file)
                if p.exists() and p.stat().st_size > 0:
                    existing += 1
            if existing == len(doc_files):
                fulfilled.append(f"doc_files: {existing}/{len(doc_files)}")
                score += 0.4
            elif existing > 0:
                fulfilled.append(f"doc_files: {existing}/{len(doc_files)}")
                score += 0.2
            else:
                missing.append("doc_files")

        # Check content was produced (context-based)
        content = context.get("doc_content", "")
        if content:
            fulfilled.append("content_produced")
            score += 0.2
            content_lower = content.lower()

            # Check for required sections
            required_sections = criteria.get("required_sections", [])
            if required_sections:
                found = sum(1 for s in required_sections if s.lower() in content_lower)
                if found == len(required_sections):
                    fulfilled.append("all_sections_present")
                    score += 0.2
                elif found > 0:
                    fulfilled.append(f"sections: {found}/{len(required_sections)}")
                    score += 0.1
                else:
                    missing.append(f"sections: 0/{len(required_sections)}")

            # Check for examples
            has_examples = criteria.get("require_examples", False)
            if has_examples:
                if "example" in content_lower or "```" in content:
                    fulfilled.append("examples_included")
                    score += 0.1
                else:
                    missing.append("examples_included")
        else:
            missing.append("content_produced")

        # Fallback: if no doc_files specified, check context outputs
        if not doc_files and not content:
            outputs = context.get("outputs", [])
            if outputs:
                fulfilled.append(f"outputs: {len(outputs)}")
                score += 0.3
            else:
                missing.append("outputs")

        status = _score_to_status(score)
        return FulfillmentResult(
            status=status,
            score=score,
            fulfilled_criteria=fulfilled,
            missing_criteria=missing,
            reason=f"Documentation {status.value} (score: {score:.2f})",
        )


class DeploymentFulfillment(FulfillmentStrategy):
    """Fulfillment detection for deployment tasks.

    Checks:
    - Deployment artifacts created
    - Services healthy post-deploy
    - Configuration validated
    """

    async def check(
        self,
        criteria: Dict[str, Any],
        context: Dict[str, Any],
        config: Optional["FulfillmentConfig"] = None,
    ) -> FulfillmentResult:
        """Check if deployment task is complete."""
        fulfilled: List[str] = []
        missing: List[str] = []
        score = 0.0

        # Check deployment artifacts
        artifacts = criteria.get("artifacts", [])
        if artifacts:
            existing = sum(1 for a in artifacts if Path(a).exists())
            if existing == len(artifacts):
                fulfilled.append(f"artifacts: {existing}/{len(artifacts)}")
                score += 0.3
            elif existing > 0:
                fulfilled.append(f"artifacts: {existing}/{len(artifacts)}")
                score += 0.15
            else:
                missing.append("artifacts")

        # Check service health
        health_checks = context.get("health_checks", {})
        if health_checks:
            healthy = sum(1 for v in health_checks.values() if v)
            total = len(health_checks)
            if healthy == total:
                fulfilled.append(f"health: {healthy}/{total}")
                score += 0.4
            elif healthy > 0:
                fulfilled.append(f"health: {healthy}/{total}")
                score += 0.2
            else:
                missing.append(f"health: 0/{total}")
        elif criteria.get("require_health_check", False):
            missing.append("health_checks")

        # Check deploy status
        deploy_status = context.get("deploy_status")
        if deploy_status == "success":
            fulfilled.append("deploy_success")
            score += 0.3
        elif deploy_status == "partial":
            fulfilled.append("deploy_partial")
            score += 0.15
        elif deploy_status:
            missing.append(f"deploy_status: {deploy_status}")

        status = _score_to_status(score)
        return FulfillmentResult(
            status=status,
            score=score,
            fulfilled_criteria=fulfilled,
            missing_criteria=missing,
            reason=f"Deployment {status.value} (score: {score:.2f})",
        )


class FulfillmentDetector:
    """Task fulfillment detection system.

    Reuses validation patterns from victor/evaluation/baseline_validator.py

    Example:
        detector = FulfillmentDetector()

        result = await detector.check_fulfillment(
            task_type=TaskType.CODE_GENERATION,
            criteria={"file_path": "main.py"},
            context={}
        )
    """

    def __init__(self, config: Optional[FulfillmentConfig] = None):
        """Initialize fulfillment detector with strategies."""
        self.config = config or DEFAULT_CONFIG
        self.strategies: Dict[TaskType, FulfillmentStrategy] = {
            TaskType.CODE_GENERATION: CodeGenerationFulfillment(),
            TaskType.CODE_MODIFICATION: CodeGenerationFulfillment(),
            TaskType.TESTING: TestingFulfillment(),
            TaskType.DEBUGGING: DebuggingFulfillment(),
            TaskType.SEARCH: SearchFulfillment(),
            TaskType.SETUP: SetupFulfillment(),
            TaskType.ANALYSIS: AnalysisFulfillment(),
            TaskType.DOCUMENTATION: DocumentationFulfillment(),
            TaskType.DEPLOYMENT: DeploymentFulfillment(),
        }

    def _normalize_task_type(self, task_type: Any) -> Optional[TaskType]:
        """Normalize compatible task type inputs to the shared fulfillment enum."""
        if isinstance(task_type, TaskType):
            return task_type

        candidates: List[str] = []
        if isinstance(task_type, str):
            candidates.append(task_type)
        else:
            raw_value = getattr(task_type, "value", None)
            if isinstance(raw_value, str):
                candidates.append(raw_value)
            raw_name = getattr(task_type, "name", None)
            if isinstance(raw_name, str):
                candidates.append(raw_name)
            candidates.append(str(task_type))

        for candidate in candidates:
            normalized = candidate.strip().lower()
            if normalized.startswith("tasktype."):
                normalized = normalized.split(".", 1)[1]
            for enum_value in TaskType:
                if normalized in {enum_value.value, enum_value.name.lower()}:
                    return enum_value

        return None

    async def check_fulfillment(
        self,
        task_type: Any,
        criteria: Dict[str, Any],
        context: Dict[str, Any],
    ) -> FulfillmentResult:
        """Check if task is fulfilled.

        Args:
            task_type: Type of task to check
            criteria: Acceptance criteria
            context: Execution context (test results, files, etc.)

        Returns:
            FulfillmentResult with status and details
        """
        normalized_task_type = self._normalize_task_type(task_type)
        strategy = self.strategies.get(normalized_task_type) if normalized_task_type else None

        if not strategy:
            log = logger.debug if normalized_task_type == TaskType.UNKNOWN else logger.warning
            log(f"No fulfillment strategy for {task_type}")
            return FulfillmentResult(
                status=FulfillmentStatus.UNKNOWN,
                score=0.0,
                reason=f"No fulfillment strategy for {getattr(task_type, 'value', task_type)}",
            )

        try:
            return await strategy.check(criteria, context, config=self.config)
        except Exception as e:
            logger.error(f"Fulfillment check failed for {normalized_task_type}: {e}")
            return FulfillmentResult(
                status=FulfillmentStatus.ERROR,
                score=0.0,
                reason=f"Error during fulfillment check: {e}",
            )

    def register_strategy(
        self,
        task_type: TaskType,
        strategy: FulfillmentStrategy,
    ) -> None:
        """Register a custom fulfillment strategy.

        Args:
            task_type: Task type for this strategy
            strategy: FulfillmentStrategy instance
        """
        self.strategies[task_type] = strategy
