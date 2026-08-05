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

"""Unit tests for requirement extraction system."""

import pytest
from victor.workflows.generation.requirements import (
    TaskRequirement,
    FunctionalRequirements,
    StructuralRequirements,
    QualityRequirements,
    ContextRequirements,
    WorkflowRequirements,
    BranchRequirement,
    Ambiguity,
)
from victor.workflows.generation.rule_extractor import RuleBasedExtractor
from victor.workflows.generation.clarifier import AmbiguityDetector, QuestionGenerator
from victor.workflows.generation.validator import RequirementValidator


class TestTaskRequirement:
    """Test TaskRequirement dataclass."""

    def test_task_creation(self):
        """Test creating a task requirement."""
        task = TaskRequirement(
            id="task_1",
            description="Analyze code",
            task_type="agent",
            role="researcher",
            tools=["code_search"],
            dependencies=[],
        )

        assert task.id == "task_1"
        assert task.description == "Analyze code"
        assert task.task_type == "agent"
        assert task.role == "researcher"
        assert task.tools == ["code_search"]
        assert task.dependencies == []

    def test_task_to_dict(self):
        """Test converting task to dictionary."""
        task = TaskRequirement(
            id="task_1",
            description="Analyze code",
            task_type="agent",
        )

        task_dict = task.to_dict()

        assert task_dict["id"] == "task_1"
        assert task_dict["description"] == "Analyze code"
        assert task_dict["task_type"] == "agent"


class TestFunctionalRequirements:
    """Test FunctionalRequirements dataclass."""

    def test_functional_requirements(self):
        """Test creating functional requirements."""
        tasks = [
            TaskRequirement(id="task_1", description="Analyze", task_type="agent"),
            TaskRequirement(id="task_2", description="Fix", task_type="agent"),
        ]

        functional = FunctionalRequirements(
            tasks=tasks,
            tools={"task_1": ["code_search"]},
            success_criteria=["All tests pass"],
        )

        assert len(functional.tasks) == 2
        assert "task_1" in functional.tools
        assert len(functional.success_criteria) == 1


class TestRuleBasedExtractor:
    """Test rule-based requirement extractor."""

    def test_extract_sequential_workflow(self):
        """Test extracting a simple sequential workflow."""
        extractor = RuleBasedExtractor()

        requirements = extractor.extract("Analyze code then find bugs then fix them then run tests")

        assert (
            requirements.description == "Analyze code then find bugs then fix them then run tests"
        )
        assert len(requirements.functional.tasks) >= 1
        assert requirements.structural.execution_order == "sequential"

    def test_extract_conditional_workflow(self):
        """Test extracting a conditional workflow."""
        extractor = RuleBasedExtractor()

        requirements = extractor.extract("Run tests and if they pass, deploy to production")

        assert requirements.structural.execution_order == "conditional"
        # Should detect at least one branch
        # This is basic - LLM extraction would be more sophisticated

    def test_detect_vertical(self):
        """Test vertical detection."""
        extractor = RuleBasedExtractor()

        # Coding vertical
        req1 = extractor.extract("Fix the bug in the code")
        assert req1.context.vertical == "coding"

        # DevOps vertical
        req2 = extractor.extract("Deploy the application to production")
        assert req2.context.vertical == "devops"

        # Research vertical
        req3 = extractor.extract("Research AI trends and summarize findings")
        assert req3.context.vertical == "research"


class TestAmbiguityDetector:
    """Test ambiguity detection."""

    def test_detect_no_ambiguities(self):
        """Test with complete requirements."""
        detector = AmbiguityDetector()

        requirements = WorkflowRequirements(
            description="Complete workflow",
            functional=FunctionalRequirements(
                tasks=[
                    TaskRequirement(
                        id="task_1",
                        description="Analyze code thoroughly",
                        task_type="agent",
                        role="researcher",
                    )
                ],
                success_criteria=["Task completed"],
            ),
            structural=StructuralRequirements(execution_order="sequential"),
            quality=QualityRequirements(),
            context=ContextRequirements(vertical="coding"),
        )

        ambiguities = detector.detect(requirements)

        # Should have minimal ambiguities
        assert len(ambiguities) == 0

    def test_detect_missing_role(self):
        """Test detection of missing agent role."""
        detector = AmbiguityDetector()

        requirements = WorkflowRequirements(
            description="Incomplete workflow",
            functional=FunctionalRequirements(
                tasks=[
                    TaskRequirement(
                        id="task_1",
                        description="Analyze code",
                        task_type="agent",
                        role=None,  # Missing role
                    )
                ],
                success_criteria=[],
            ),
            structural=StructuralRequirements(execution_order="sequential"),
            quality=QualityRequirements(),
            context=ContextRequirements(vertical="coding"),
        )

        ambiguities = detector.detect(requirements)

        # Should detect missing role
        assert any(a.type == "missing_role" for a in ambiguities)

    def test_detect_missing_success_criteria(self):
        """Test detection of missing success criteria."""
        detector = AmbiguityDetector()

        requirements = WorkflowRequirements(
            description="Workflow without success criteria",
            functional=FunctionalRequirements(
                tasks=[
                    TaskRequirement(
                        id="task_1",
                        description="Do work",
                        task_type="agent",
                        role="executor",
                    )
                ],
                success_criteria=[],  # Missing success criteria
            ),
            structural=StructuralRequirements(execution_order="sequential"),
            quality=QualityRequirements(),
            context=ContextRequirements(vertical="coding"),
        )

        ambiguities = detector.detect(requirements)

        # Should detect missing success criteria
        assert any(a.type == "missing_success_criteria" for a in ambiguities)


class TestQuestionGenerator:
    """Test question generation."""

    def test_generate_missing_role_question(self):
        """Test generating question for missing role."""
        generator = QuestionGenerator()

        ambiguity = Ambiguity(
            type="missing_role",
            severity=7,
            message="Task 'task_1' missing role",
            suggestion="What role should perform this task?",
            field="functional.tasks.task_1.role",
            options=["researcher", "executor", "planner"],
        )

        question = generator.generate(ambiguity)

        assert question.options == ["researcher", "executor", "planner"]
        assert "role" in question.text.lower()

    def test_generate_vague_description_question(self):
        """Test generating question for vague description."""
        generator = QuestionGenerator()

        ambiguity = Ambiguity(
            type="vague_description",
            severity=4,
            message="Task 'task_1' has vague description",
            suggestion="Be more specific",
            field="functional.tasks.task_1.description",
        )

        question = generator.generate(ambiguity)

        assert "specific" in question.text.lower()


class TestRequirementValidator:
    """Test requirement validation."""

    def test_validate_complete_requirements(self):
        """Test validation of complete requirements."""
        validator = RequirementValidator()

        requirements = WorkflowRequirements(
            description="Complete workflow",
            functional=FunctionalRequirements(
                tasks=[
                    TaskRequirement(
                        id="task_1",
                        description="Analyze Python codebase thoroughly",
                        task_type="agent",
                        role="researcher",
                    )
                ],
                success_criteria=["Analysis complete"],
            ),
            structural=StructuralRequirements(execution_order="sequential"),
            quality=QualityRequirements(max_duration_seconds=300),
            context=ContextRequirements(vertical="coding"),
        )

        result = validator.validate(requirements)

        assert result.is_valid
        assert result.score > 0.8

    def test_validate_missing_task_description(self):
        """Test validation detects missing task description."""
        validator = RequirementValidator()

        requirements = WorkflowRequirements(
            description="Incomplete workflow",
            functional=FunctionalRequirements(
                tasks=[
                    TaskRequirement(
                        id="task_1",
                        description="",  # Empty description
                        task_type="agent",
                        role="researcher",
                    )
                ],
                success_criteria=[],
            ),
            structural=StructuralRequirements(execution_order="sequential"),
            quality=QualityRequirements(),
            context=ContextRequirements(vertical="coding"),
        )

        result = validator.validate(requirements)

        assert not result.is_valid
        assert any("missing description" in e.message.lower() for e in result.errors)

    def test_validate_invalid_task_type(self):
        """Test validation detects invalid task type."""
        validator = RequirementValidator()

        requirements = WorkflowRequirements(
            description="Invalid task type",
            functional=FunctionalRequirements(
                tasks=[
                    TaskRequirement(
                        id="task_1",
                        description="Do something",
                        task_type="invalid_type",  # Invalid
                        role="executor",
                    )
                ],
                success_criteria=[],
            ),
            structural=StructuralRequirements(execution_order="sequential"),
            quality=QualityRequirements(),
            context=ContextRequirements(vertical="coding"),
        )

        result = validator.validate(requirements)

        assert not result.is_valid
        assert any("invalid task type" in e.message.lower() for e in result.errors)

    def test_validate_agent_task_missing_role(self):
        """Test validation detects agent task without role."""
        validator = RequirementValidator()

        requirements = WorkflowRequirements(
            description="Agent task without role",
            functional=FunctionalRequirements(
                tasks=[
                    TaskRequirement(
                        id="task_1",
                        description="Analyze code",
                        task_type="agent",
                        role=None,  # Missing role for agent task
                    )
                ],
                success_criteria=[],
            ),
            structural=StructuralRequirements(execution_order="sequential"),
            quality=QualityRequirements(),
            context=ContextRequirements(vertical="coding"),
        )

        result = validator.validate(requirements)

        assert not result.is_valid
        assert any("missing role" in e.message.lower() for e in result.errors)

    def test_validate_timeout_too_short(self):
        """Test validation detects infeasible timeout."""
        validator = RequirementValidator()

        requirements = WorkflowRequirements(
            description="Too many tasks, too little time",
            functional=FunctionalRequirements(
                tasks=[
                    TaskRequirement(
                        id=f"task_{i}",
                        description=f"Task {i}",
                        task_type="agent",
                        role="executor",
                    )
                    for i in range(10)  # 10 tasks
                ],
                success_criteria=[],
            ),
            structural=StructuralRequirements(execution_order="sequential"),
            quality=QualityRequirements(max_duration_seconds=60),  # Only 60s for 10 tasks
            context=ContextRequirements(vertical="coding"),
        )

        result = validator.validate(requirements)

        # Should have error or warning about timeout
        assert any("timeout" in e.message.lower() for e in result.errors + result.warnings)


class TestWorkflowRequirements:
    """Test WorkflowRequirements dataclass."""

    def test_workflow_requirements_creation(self):
        """Test creating complete workflow requirements."""
        requirements = WorkflowRequirements(
            description="Test workflow",
            functional=FunctionalRequirements(
                tasks=[TaskRequirement(id="task_1", description="Analyze", task_type="agent")],
                success_criteria=["Done"],
            ),
            structural=StructuralRequirements(execution_order="sequential"),
            quality=QualityRequirements(),
            context=ContextRequirements(vertical="coding"),
        )

        assert requirements.description == "Test workflow"
        assert len(requirements.functional.tasks) == 1
        assert requirements.structural.execution_order == "sequential"
        assert requirements.context.vertical == "coding"

    def test_workflow_requirements_to_dict(self):
        """Test converting workflow requirements to dictionary."""
        requirements = WorkflowRequirements(
            description="Test workflow",
            functional=FunctionalRequirements(
                tasks=[TaskRequirement(id="task_1", description="Analyze", task_type="agent")],
                success_criteria=["Done"],
            ),
            structural=StructuralRequirements(execution_order="sequential"),
            quality=QualityRequirements(),
            context=ContextRequirements(vertical="coding"),
        )

        req_dict = requirements.to_dict()

        assert req_dict["description"] == "Test workflow"
        assert "functional" in req_dict
        assert "structural" in req_dict
        assert "quality" in req_dict
        assert "context" in req_dict

    def test_workflow_requirements_to_json(self):
        """Test converting workflow requirements to JSON."""
        requirements = WorkflowRequirements(
            description="Test workflow",
            functional=FunctionalRequirements(
                tasks=[TaskRequirement(id="task_1", description="Analyze", task_type="agent")],
                success_criteria=["Done"],
            ),
            structural=StructuralRequirements(execution_order="sequential"),
            quality=QualityRequirements(),
            context=ContextRequirements(vertical="coding"),
        )

        import json

        json_str = requirements.to_json()
        parsed = json.loads(json_str)

        assert parsed["description"] == "Test workflow"
        assert "functional" in parsed
