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

"""Focused protocols for emergent collaboration patterns.

This module defines ISP-compliant protocols for pattern discovery,
recommendation, and evolution. Each protocol has a single, well-defined
responsibility.

Protocol Separation:
- PatternMinerProtocol: Pattern mining - single responsibility
- PatternValidatorProtocol: Pattern validation - separate concern
- PatternRecommenderProtocol: Pattern recommendation - separate concern

Following the refinement plan, these protocols extend existing Victor
infrastructure (experiment tracking, team formations, template library).
"""

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from victor.framework.protocols import (
    FrameworkOrchestratorProtocol as OrchestratorProtocol,
)
from victor.teams import TeamFormation

# =============================================================================
# Data Classes
# =============================================================================


class PatternValidationResult:
    """Result of pattern validation."""

    def __init__(
        self,
        is_valid: bool,
        errors: Optional[List[str]] = None,
        warnings: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize validation result.

        Args:
            is_valid: Whether validation passed
            errors: List of error messages
            warnings: List of warning messages
            metadata: Optional additional metadata
        """
        self.is_valid = is_valid
        self.errors = errors or []
        self.warnings = warnings or []
        self.metadata = metadata or {}


class CollaborationPattern:
    """A discovered or defined collaboration pattern.

    Reuses existing TeamFormation enum from victor/teams/.
    """

    def __init__(
        self,
        pattern_id: str,
        name: str,
        description: str,
        formation: TeamFormation,
        participants: List[Dict[str, Any]],
        success_rate: float = 0.0,
        usage_count: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize collaboration pattern.

        Args:
            pattern_id: Unique pattern identifier
            name: Human-readable pattern name
            description: Pattern description
            formation: Team formation type (reuses existing enum)
            participants: Participant specifications
            success_rate: Historical success rate (0-1)
            usage_count: Number of times pattern used
            metadata: Optional additional metadata
        """
        self.pattern_id = pattern_id
        self.name = name
        self.description = description
        self.formation = formation
        self.participants = participants
        self.success_rate = success_rate
        self.usage_count = usage_count
        self.metadata = metadata or {}

    def to_team_coordinator(
        self,
        orchestrator: OrchestratorProtocol,
    ):
        """Convert pattern to executable team using existing coordinator.

        This method demonstrates code reuse - it uses the existing
        create_coordinator() function from victor/teams/__init__.py

        Args:
            orchestrator: Orchestrator protocol instance

        Returns:
            ITeamCoordinator instance from existing infrastructure
        """
        # Import here to avoid circular dependency
        from victor.teams import create_coordinator

        # Map participants to agent IDs
        agent_ids = [p.get("agent_id") for p in self.participants]

        # Use existing team coordinator factory
        return create_coordinator(
            orchestrator,
            formation=self.formation,
            participants=agent_ids,
        )


class TaskContext:
    """Context for pattern recommendation."""

    def __init__(
        self,
        task_description: str,
        required_capabilities: List[str],
        vertical: str,
        complexity: str = "medium",
        constraints: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize task context.

        Args:
            task_description: Task description
            required_capabilities: Required agent capabilities
            vertical: Vertical domain
            complexity: Task complexity (low, medium, high)
            constraints: Optional constraints (timeout, budget, etc.)
        """
        self.task_description = task_description
        self.required_capabilities = required_capabilities
        self.vertical = vertical
        self.complexity = complexity
        self.constraints = constraints or {}


class PatternRecommendation:
    """A recommended collaboration pattern."""

    def __init__(
        self,
        pattern: CollaborationPattern,
        score: float,
        rationale: str,
        expected_benefits: List[str],
        potential_risks: List[str],
    ) -> None:
        """Initialize pattern recommendation.

        Args:
            pattern: Recommended pattern
            score: Recommendation score (0-1)
            rationale: Explanation for recommendation
            expected_benefits: Expected benefits of using pattern
            potential_risks: Potential risks of using pattern
        """
        self.pattern = pattern
        self.score = score
        self.rationale = rationale
        self.expected_benefits = expected_benefits
        self.potential_risks = potential_risks


class WorkflowExecutionTrace:
    """Execution trace for pattern mining.

    Reuses existing experiment tracking infrastructure from
    victor/experiments/sqlite_store.py.
    """

    def __init__(
        self,
        workflow_id: str,
        execution_id: str,
        nodes_executed: List[str],
        execution_order: List[str],
        duration_ms: int,
        success: bool,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize workflow execution trace.

        Args:
            workflow_id: Workflow identifier
            execution_id: Unique execution identifier
            nodes_executed: List of node IDs executed
            execution_order: Order of node execution
            duration_ms: Execution duration in milliseconds
            success: Whether execution succeeded
            metadata: Optional additional metadata
        """
        self.workflow_id = workflow_id
        self.execution_id = execution_id
        self.nodes_executed = nodes_executed
        self.execution_order = execution_order
        self.duration_ms = duration_ms
        self.success = success
        self.metadata = metadata or {}


# =============================================================================
# Pattern Mining Protocol
# =============================================================================


@runtime_checkable
class PatternMinerProtocol(Protocol):
    """Pattern mining - single responsibility.

    Discovers collaboration patterns from workflow execution traces.

    Single Responsibility: Mining patterns from traces ONLY.

    Key Methods:
    - mine_from_traces: Extract patterns from execution traces
    - analyze_execution_order: Analyze execution patterns
    - detect_formations: Detect team formations from traces

    This protocol is SEPARATE from PatternValidatorProtocol because:
    - Mining is independent of validation
    - Can mine patterns without validating them
    - Different mining algorithms can be swapped
    - Follows Single Responsibility Principle

    Reuse Strategy:
    - Extend victor.experiments.sqlite_store.ExperimentTracker
    - Reuse existing trace storage and retrieval
    - Leverage existing experiment metadata
    """

    async def mine_from_traces(
        self,
        traces: List[WorkflowExecutionTrace],
    ) -> List[CollaborationPattern]:
        """Extract patterns from execution traces.

        Args:
            traces: List of workflow execution traces

        Returns:
            List of discovered collaboration patterns
        """
        ...

    async def analyze_execution_order(
        self,
        trace: WorkflowExecutionTrace,
    ) -> Dict[str, Any]:
        """Analyze execution order patterns.

        Args:
            trace: Single execution trace

        Returns:
            Dictionary with execution pattern analysis
        """
        ...

    async def detect_formations(
        self,
        traces: List[WorkflowExecutionTrace],
    ) -> Dict[TeamFormation, int]:
        """Detect team formations from traces.

        Args:
            traces: List of execution traces

        Returns:
            Dictionary mapping formation to frequency
        """
        ...


# =============================================================================
# Pattern Validation Protocol
# =============================================================================


@runtime_checkable
class PatternValidatorProtocol(Protocol):
    """Pattern validation - separate concern.

    Validates discovered or defined collaboration patterns.

    Single Responsibility: Validation ONLY (separate from mining).

    Key Methods:
    - validate: Validate pattern structure and correctness
    - test_pattern: Test pattern against sample tasks
    - estimate_success: Estimate success probability

    This protocol is SEPARATE from PatternMinerProtocol because:
    - Validation is independent of mining
    - Can validate manually defined patterns
    - Different validation strategies possible
    - Follows Single Responsibility Principle

    Reuse Strategy:
    - Extend victor.core.workflow_validation.WorkflowValidator
    - Reuse existing validation infrastructure
    - Add pattern-specific validation rules
    """

    async def validate(
        self,
        pattern: CollaborationPattern,
        test_cases: Optional[List[TaskContext]] = None,
    ) -> PatternValidationResult:
        """Validate pattern structure and correctness.

        Args:
            pattern: Pattern to validate
            test_cases: Optional test cases for validation

        Returns:
            PatternValidationResult with is_valid flag and any errors
        """
        ...

    async def test_pattern(
        self,
        pattern: CollaborationPattern,
        test_tasks: List[TaskContext],
    ) -> Dict[str, Any]:
        """Test pattern against sample tasks.

        Args:
            pattern: Pattern to test
            test_tasks: List of test task contexts

        Returns:
            Dictionary with test results
        """
        ...

    async def estimate_success(
        self,
        pattern: CollaborationPattern,
        task_context: TaskContext,
    ) -> float:
        """Estimate success probability for pattern.

        Args:
            pattern: Pattern to estimate
            task_context: Task context

        Returns:
            Success probability (0-1)
        """
        ...


# =============================================================================
# Pattern Recommendation Protocol
# =============================================================================


@runtime_checkable
class PatternRecommenderProtocol(Protocol):
    """Pattern recommendation - separate concern.

    Recommends collaboration patterns for specific tasks.

    Single Responsibility: Recommendation ONLY (separate from mining/validation).

    Key Methods:
    - recommend: Recommend patterns for task context
    - rank_patterns: Rank patterns by suitability
    - explain_recommendation: Explain why pattern was recommended

    This protocol is SEPARATE from PatternMinerProtocol and PatternValidatorProtocol because:
    - Recommendation logic independent of mining
    - Can recommend from validated pattern catalog
    - Different recommendation strategies possible
    - Follows Single Responsibility Principle

    Reuse Strategy:
    - Extend victor.workflows.generation.templates.TemplateLibrary
    - Reuse existing template matching logic
    - Leverage existing pattern libraries
    """

    async def recommend(
        self,
        task_context: TaskContext,
        top_k: int = 5,
    ) -> List[PatternRecommendation]:
        """Recommend patterns for task context.

        Args:
            task_context: Task to find pattern for
            top_k: Number of recommendations to return

        Returns:
            List of pattern recommendations, sorted by score
        """
        ...

    async def rank_patterns(
        self,
        patterns: List[CollaborationPattern],
        task_context: TaskContext,
    ) -> List[PatternRecommendation]:
        """Rank patterns by suitability for task.

        Args:
            patterns: List of patterns to rank
            task_context: Task context

        Returns:
            List of pattern recommendations, ranked by score
        """
        ...

    async def explain_recommendation(
        self,
        recommendation: PatternRecommendation,
    ) -> str:
        """Explain why pattern was recommended.

        Args:
            recommendation: Pattern recommendation to explain

        Returns:
            Human-readable explanation
        """
        ...
