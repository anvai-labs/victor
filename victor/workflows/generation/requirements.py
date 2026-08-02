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

"""Requirement dataclasses for workflow generation.

This module defines the structured data models for extracted workflow requirements.
These models capture functional, structural, quality, and context requirements
from natural language descriptions.

Design Pattern: Dataclass Validation
- All dataclasses use Pydantic-style validation with dataclasses
- Required fields are marked as non-optional
- Optional fields have sensible defaults
- All models support to_dict() for serialization

Example:
    from victor.workflows.generation.requirements import WorkflowRequirements

    requirements = WorkflowRequirements(
        description="Analyze code and fix bugs",
        functional=FunctionalRequirements(...),
        structural=StructuralRequirements(...),
        quality=QualityRequirements(),
        context=ContextRequirements(vertical="coding"),
    )
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# =============================================================================
# Functional Requirements
# =============================================================================


@dataclass
class TaskRequirement:
    """Individual task requirement.

    Represents a single task within a workflow, including its type,
    dependencies, tools needed, and agent role if applicable.

    Attributes:
        id: Unique task identifier (e.g., "task_1", "analyze")
        description: What the task does (human-readable)
        task_type: Type of task (agent, compute, condition, transform, parallel)
        role: Agent role for agent tasks (researcher, executor, planner, reviewer)
        goal: Goal for the agent (if agent task)
        tools: List of tool names needed for this task
        dependencies: List of task IDs this task depends on
        tool_budget: Maximum tool calls allowed for this task

    Example:
        task = TaskRequirement(
            id="analyze",
            description="Analyze Python codebase for bugs",
            task_type="agent",
            role="researcher",
            goal="Find bugs using static analysis",
            tools=["code_search", "linter"],
            dependencies=[],
            tool_budget=20
        )
    """

    id: str
    description: str
    task_type: str  # agent, compute, condition, transform, parallel
    role: Optional[str] = None
    goal: Optional[str] = None
    tools: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    tool_budget: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "description": self.description,
            "task_type": self.task_type,
            "role": self.role,
            "goal": self.goal,
            "tools": self.tools,
            "dependencies": self.dependencies,
            "tool_budget": self.tool_budget,
        }


@dataclass
class InputRequirement:
    """Input requirement for workflow.

    Defines what data inputs the workflow needs and where they come from.

    Attributes:
        name: Input parameter name (e.g., "codebase_path", "query")
        type: Data type (file, text, code, api_response, dict)
        source: Where input comes from (user, file, api, task:{task_id})
        required: Whether input is required or optional
        validation: How to validate input (regex, schema, custom)

    Example:
        input_req = InputRequirement(
            name="query",
            type="text",
            source="user",
            required=True
        )
    """

    name: str
    type: str  # file, text, code, api_response, dict
    source: str  # user, file, api, task:{task_id}
    required: bool = True
    validation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "type": self.type,
            "source": self.source,
            "required": self.required,
            "validation": self.validation,
        }


@dataclass
class OutputRequirement:
    """Output requirement from workflow.

    Defines what the workflow produces and where it goes.

    Attributes:
        name: Output name (e.g., "report", "fixed_code")
        type: Output type (file, report, code_changes, data, console)
        destination: Where output goes (file, stdout, api, task:{task_id})
        format: Output format (markdown, json, code, plain_text)

    Example:
        output_req = OutputRequirement(
            name="bug_report",
            type="report",
            destination="file",
            format="markdown"
        )
    """

    name: str
    type: str  # file, report, code_changes, data, console
    destination: str  # file, stdout, api, task:{task_id}
    format: str  # markdown, json, code, plain_text

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "type": self.type,
            "destination": self.destination,
            "format": self.format,
        }


@dataclass
class FunctionalRequirements:
    """Core functional requirements for workflow tasks.

    Defines what the workflow should do, including tasks, tools,
    inputs, outputs, and success criteria.

    Attributes:
        tasks: List of tasks to perform (in order)
        tools: Tools needed for each task (task_id -> tool names)
        inputs: Data inputs required (files, APIs, user input)
        outputs: Expected outputs (reports, code changes, data)
        success_criteria: Conditions for successful completion

    Example:
        functional = FunctionalRequirements(
            tasks=[task1, task2, task3],
            tools={"task_1": ["code_search"], "task_2": ["bash"]},
            inputs=[InputRequirement(...)],
            outputs=[OutputRequirement(...)],
            success_criteria=["All tests pass", "No critical bugs"]
        )
    """

    tasks: List[TaskRequirement]
    tools: Dict[str, List[str]] = field(default_factory=lambda: {})
    inputs: List[InputRequirement] = field(default_factory=list)
    outputs: List[OutputRequirement] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tasks": [task.to_dict() for task in self.tasks],
            "tools": self.tools,
            "inputs": [inp.to_dict() for inp in self.inputs],
            "outputs": [out.to_dict() for out in self.outputs],
            "success_criteria": self.success_criteria,
        }


# =============================================================================
# Structural Requirements
# =============================================================================


@dataclass
class BranchRequirement:
    """Conditional branch requirement.

    Defines a conditional branch in the workflow execution.

    Attributes:
        condition_id: Unique branch identifier (e.g., "quality_check")
        condition: Description of branch condition (e.g., "quality_score > 0.8")
        true_branch: Task ID if condition is true
        false_branch: Task ID if condition is false (or "end")
        condition_type: Type of condition (quality_threshold, error_check, user_approval)

    Example:
        branch = BranchRequirement(
            condition_id="quality_check",
            condition="quality_score > 0.8",
            true_branch="create_report",
            false_branch="end",
            condition_type="quality_threshold"
        )
    """

    condition_id: str
    condition: str
    true_branch: str
    false_branch: str
    condition_type: str  # quality_threshold, error_check, user_approval, data_check

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "condition_id": self.condition_id,
            "condition": self.condition,
            "true_branch": self.true_branch,
            "false_branch": self.false_branch,
            "condition_type": self.condition_type,
        }


@dataclass
class LoopRequirement:
    """Loop/repetition requirement.

    Defines an iterative/repetitive pattern in the workflow.

    Attributes:
        loop_id: Unique loop identifier (e.g., "retry_loop")
        task_to_repeat: Which task to repeat (task ID)
        exit_condition: When to stop looping (e.g., "success or max_iterations")
        max_iterations: Maximum times to repeat (default: 3)

    Example:
        loop = LoopRequirement(
            loop_id="retry_fix",
            task_to_repeat="fix_bugs",
            exit_condition="all_tests_pass",
            max_iterations=3
        )
    """

    loop_id: str
    task_to_repeat: str
    exit_condition: str
    max_iterations: int = 3

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "loop_id": self.loop_id,
            "task_to_repeat": self.task_to_repeat,
            "exit_condition": self.exit_condition,
            "max_iterations": self.max_iterations,
        }


@dataclass
class StructuralRequirements:
    """Execution structure requirements.

    Defines how tasks are organized and executed.

    Attributes:
        execution_order: Overall pattern (sequential, parallel, mixed, conditional)
        dependencies: Explicit dependencies between tasks (task_id -> [dep IDs])
        branches: Conditional branching logic
        loops: Iterative/repetitive patterns
        joins: How parallel branches merge (parallel_task_id -> join_strategy)

    Example:
        structural = StructuralRequirements(
            execution_order="sequential",
            dependencies={"task_2": ["task_1"], "task_3": ["task_2"]},
            branches=[],
            loops=[]
        )
    """

    execution_order: str  # sequential, parallel, mixed, conditional
    dependencies: Dict[str, List[str]] = field(default_factory=dict)
    branches: List[BranchRequirement] = field(default_factory=list)
    loops: List[LoopRequirement] = field(default_factory=list)
    joins: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "execution_order": self.execution_order,
            "dependencies": self.dependencies,
            "branches": [branch.to_dict() for branch in self.branches],
            "loops": [loop.to_dict() for loop in self.loops],
            "joins": self.joins,
        }


# =============================================================================
# Quality Requirements
# =============================================================================


@dataclass
class TaskQualityRequirements:
    """Per-task quality requirements.

    Defines quality constraints for individual tasks.

    Attributes:
        task_id: Task this applies to
        timeout_seconds: Task-specific timeout
        tool_budget: Max tool calls for this task
        allowed_tools: Specific tools allowed (empty = all)
        quality_threshold: Task-specific quality threshold (0.0-1.0)

    Example:
        task_quality = TaskQualityRequirements(
            task_id="analyze",
            timeout_seconds=120,
            tool_budget=15,
            quality_threshold=0.8
        )
    """

    task_id: str
    timeout_seconds: Optional[int] = None
    tool_budget: Optional[int] = None
    allowed_tools: List[str] = field(default_factory=list)
    quality_threshold: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "timeout_seconds": self.timeout_seconds,
            "tool_budget": self.tool_budget,
            "allowed_tools": self.allowed_tools,
            "quality_threshold": self.quality_threshold,
        }


@dataclass
class QualityRequirements:
    """Quality and performance requirements.

    Defines constraints and performance targets for the workflow.

    Attributes:
        max_duration_seconds: Maximum workflow execution time
        max_cost_tier: Maximum tool cost tier (FREE, LOW, MEDIUM, HIGH)
        accuracy_threshold: Minimum accuracy/success rate (0.0-1.0)
        max_tool_calls: Maximum total tool calls across workflow
        max_tokens: Maximum LLM tokens to consume
        retry_policy: How to handle failures (retry, fail_fast, continue, fallback)
        task_requirements: Per-task quality requirements

    Example:
        quality = QualityRequirements(
            max_duration_seconds=300,
            max_cost_tier="MEDIUM",
            accuracy_threshold=0.8,
            retry_policy="retry"
        )
    """

    max_duration_seconds: Optional[int] = None
    max_cost_tier: str = "MEDIUM"
    accuracy_threshold: Optional[float] = None
    max_tool_calls: Optional[int] = None
    max_tokens: Optional[int] = None
    retry_policy: str = "retry"  # retry, fail_fast, continue, fallback
    task_requirements: List[TaskQualityRequirements] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "max_duration_seconds": self.max_duration_seconds,
            "max_cost_tier": self.max_cost_tier,
            "accuracy_threshold": self.accuracy_threshold,
            "max_tool_calls": self.max_tool_calls,
            "max_tokens": self.max_tokens,
            "retry_policy": self.retry_policy,
            "task_requirements": [tq.to_dict() for tq in self.task_requirements],
        }


# =============================================================================
# Context Requirements
# =============================================================================


@dataclass
class ProjectContext:
    """Project-specific context.

    Defines information about the project the workflow operates on.

    Attributes:
        repo_path: Path to codebase repository
        primary_language: Main programming language
        framework: Framework being used (e.g., "FastAPI", "React")
        testing_framework: Test framework (e.g., "pytest", "jest")
        build_system: Build tool (e.g., "make", "npm", "cargo")

    Example:
        project = ProjectContext(
            repo_path="/path/to/repo",
            primary_language="Python",
            testing_framework="pytest",
            build_system="make"
        )
    """

    repo_path: Optional[str] = None
    primary_language: Optional[str] = None
    framework: Optional[str] = None
    testing_framework: Optional[str] = None
    build_system: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "repo_path": self.repo_path,
            "primary_language": self.primary_language,
            "framework": self.framework,
            "testing_framework": self.testing_framework,
            "build_system": self.build_system,
        }


@dataclass
class ContextRequirements:
    """Context and environment requirements.

    Defines the domain and environment context for the workflow.

    Attributes:
        vertical: Domain vertical (coding, devops, research, rag, dataanalysis, benchmark)
        subdomain: Specific subdomain (e.g., "bug_fix", "deployment", "fact_checking")
        environment: Execution environment (local, cloud, sandbox)
        user_preferences: User-specific preferences (free-form dict)
        project_context: Project-specific information

    Example:
        context = ContextRequirements(
            vertical="coding",
            subdomain="bug_fix",
            environment="local",
            project_context=ProjectContext(primary_language="Python")
        )
    """

    vertical: str  # coding, devops, research, rag, dataanalysis, benchmark
    subdomain: Optional[str] = None
    environment: str = "local"  # local, cloud, sandbox
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    project_context: Optional[ProjectContext] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "vertical": self.vertical,
            "subdomain": self.subdomain,
            "environment": self.environment,
            "user_preferences": self.user_preferences,
            "project_context": (self.project_context.to_dict() if self.project_context else None),
        }


# =============================================================================
# Main Requirements Container
# =============================================================================


@dataclass
class ExtractionMetadata:
    """Metadata about the extraction process.

    Tracks how requirements were extracted and validated.

    Attributes:
        extraction_method: Method used (llm, rules, hybrid)
        model: LLM model used (if applicable)
        extraction_time: Time taken for extraction (seconds)
        ambiguity_count: Number of ambiguities detected
        resolution_strategy: How ambiguities were resolved
        confidence: Overall confidence score (0.0-1.0)

    Example:
        metadata = ExtractionMetadata(
            extraction_method="llm",
            model="claude-sonnet-4-5",
            extraction_time=2.5,
            confidence=0.9
        )
    """

    extraction_method: str  # llm, rules, hybrid
    model: Optional[str] = None
    extraction_time: float = 0.0
    ambiguity_count: int = 0
    resolution_strategy: str = "interactive"
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "extraction_method": self.extraction_method,
            "model": self.model,
            "extraction_time": self.extraction_time,
            "ambiguity_count": self.ambiguity_count,
            "resolution_strategy": self.resolution_strategy,
            "confidence": self.confidence,
        }


@dataclass
class WorkflowRequirements:
    """Complete workflow requirements extracted from natural language.

    This is the primary output of the requirement extraction system.
    It contains all information needed to generate a workflow graph.

    Attributes:
        description: Original natural language description
        functional: Functional requirements (tasks, tools, I/O)
        structural: Structural requirements (order, dependencies, branches)
        quality: Quality requirements (performance, constraints)
        context: Context requirements (domain, environment)
        confidence_scores: Confidence scores for each section (0.0-1.0)
        metadata: Extraction metadata

    Example:
        requirements = WorkflowRequirements(
            description="Analyze code, find bugs, fix them, run tests",
            functional=FunctionalRequirements(...),
            structural=StructuralRequirements(...),
            quality=QualityRequirements(),
            context=ContextRequirements(vertical="coding"),
            confidence_scores={"functional": 0.9, "structural": 0.85},
            metadata=ExtractionMetadata(...)
        )
    """

    description: str
    functional: FunctionalRequirements
    structural: StructuralRequirements
    quality: QualityRequirements
    context: ContextRequirements
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    metadata: Optional[ExtractionMetadata] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary representation suitable for JSON encoding.

        Example:
            req = WorkflowRequirements(...)
            json_str = json.dumps(req.to_dict(), indent=2)
        """
        return {
            "description": self.description,
            "functional": self.functional.to_dict(),
            "structural": self.structural.to_dict(),
            "quality": self.quality.to_dict(),
            "context": self.context.to_dict(),
            "confidence_scores": self.confidence_scores,
            "metadata": self.metadata.to_dict() if self.metadata else None,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string.

        Args:
            indent: JSON indentation level (default: 2)

        Returns:
            JSON string representation.

        Example:
            req = WorkflowRequirements(...)
            json_str = req.to_json()
        """
        return json.dumps(self.to_dict(), indent=indent)


# =============================================================================
# Validation and Ambiguity Types
# =============================================================================


@dataclass
class RequirementValidationError:
    """A validation error or warning for workflow requirements.

    Represents an issue found during requirement validation.

    Attributes:
        field: Field that failed validation (JSON path)
        message: Human-readable error message
        severity: Severity level (critical, error, warning, info)
        suggestion: Optional suggestion for fixing

    Example:
        error = RequirementValidationError(
            field="functional.tasks.0.role",
            message="Agent task missing role",
            severity="critical",
            suggestion="Specify agent role (researcher, executor, planner)"
        )
    """

    field: str
    message: str
    severity: str  # critical, error, warning, info
    suggestion: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "field": self.field,
            "message": self.message,
            "severity": self.severity,
            "suggestion": self.suggestion,
        }


@dataclass
class RequirementValidationResult:
    """Result of requirement validation.

    Contains the outcome of requirement validation with errors,
    warnings, and recommendations.

    Attributes:
        is_valid: Whether requirements are valid for generation
        errors: Critical issues that must be fixed
        warnings: Non-critical issues
        recommendations: Suggestions for improvement
        score: Overall quality score (0.0-1.0)

    Example:
        result = RequirementValidationResult(
            is_valid=True,
            errors=[],
            warnings=[warning],
            recommendations=["Add more specific success criteria"],
            score=0.85
        )
    """

    is_valid: bool
    errors: List[RequirementValidationError]
    warnings: List[RequirementValidationError]
    recommendations: List[str]
    score: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "is_valid": self.is_valid,
            "score": self.score,
            "errors": [e.to_dict() for e in self.errors],
            "warnings": [w.to_dict() for w in self.warnings],
            "recommendations": self.recommendations,
        }


@dataclass
class Ambiguity:
    """An ambiguity detected in requirements.

    Represents missing or unclear information in extracted requirements.

    Attributes:
        type: Type of ambiguity (missing, conflict, vague, infeasible)
        severity: Severity score (1-10, 10 = critical)
        message: Human-readable description
        suggestion: Suggested resolution
        field: JSON path to ambiguous field
        options: Multiple choice options (if applicable)

    Example:
        ambiguity = Ambiguity(
            type="missing_role",
            severity=7,
            message="Task 'task_1' has no agent role specified",
            suggestion="What role should perform this task?",
            field="functional.tasks.task_1.role",
            options=["researcher", "executor", "planner", "reviewer"]
        )
    """

    type: str
    severity: int
    message: str
    suggestion: str
    field: str
    options: Optional[List[str]] = None
