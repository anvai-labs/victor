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

"""Tests for StageBuilder class in victor/core/vertical_types.py."""

import pytest

from victor.core.vertical_types import StageBuilder, StageDefinition
from victor.tools.tool_names import ToolNames


class TestStageBuilder:
    """Test suite for StageBuilder class."""

    def test_builder_initialization(self):
        """Test that StageBuilder initializes correctly."""
        builder = StageBuilder()
        assert builder._stages == {}
        assert builder._last_stage_name is None

    def test_stage_fluent_api(self):
        """Test the fluent API for building stages."""
        builder = StageBuilder()
        stage = (
            builder.stage("TEST_STAGE")
            .description("Test stage description")
            .tools({ToolNames.READ, ToolNames.WRITE})
            .keywords(["test", "example"])
            .next_stages({"NEXT_STAGE"})
            .build()
        )

        assert isinstance(stage, StageDefinition)
        assert stage.name == "TEST_STAGE"
        assert stage.description == "Test stage description"
        assert stage.tools == {ToolNames.READ, ToolNames.WRITE}
        assert stage.keywords == ["test", "example"]
        assert stage.next_stages == {"NEXT_STAGE"}

    def test_standard_initial_stage(self):
        """Test standard_initial() factory method."""
        builder = StageBuilder()
        initial = builder.standard_initial()

        assert isinstance(initial, StageDefinition)
        assert initial.name == "INITIAL"
        assert initial.description == "Understanding the request"
        assert initial.tools == set()
        assert "what" in initial.keywords
        assert "how" in initial.keywords
        assert initial.next_stages == set()

    def test_standard_completion_stage(self):
        """Test standard_completion() factory method."""
        builder = StageBuilder()
        completion = builder.standard_completion()

        assert isinstance(completion, StageDefinition)
        assert completion.name == "COMPLETION"
        assert completion.description == "Task complete"
        assert completion.tools == set()
        assert "done" in completion.keywords
        assert "complete" in completion.keywords
        assert completion.next_stages == set()

    def test_add_prebuilt_stage(self):
        """Test add() method for pre-built StageDefinition instances."""
        builder = StageBuilder()
        stage = StageDefinition(
            name="CUSTOM",
            description="Custom stage",
            tools={ToolNames.READ},
            keywords=["custom"],
            next_stages={"NEXT"},
        )

        builder.add("CUSTOM", stage)
        assert "CUSTOM" in builder._stages
        assert builder._stages["CUSTOM"] == stage
        assert builder._last_stage_name == "CUSTOM"

    def test_workflow_builder(self):
        """Test workflow builder pattern."""
        builder = StageBuilder()
        stages = (
            builder.workflow("TestWorkflow")
            .initial()
            .add(
                "PLANNING",
                StageDefinition(
                    name="PLANNING",
                    description="Planning stage",
                    tools={ToolNames.READ},
                    keywords=["plan"],
                    next_stages={"EXECUTION"},
                ),
            )
            .add(
                "EXECUTION",
                StageDefinition(
                    name="EXECUTION",
                    description="Execution stage",
                    tools={ToolNames.WRITE},
                    keywords=["execute"],
                    next_stages={"COMPLETION"},
                ),
            )
            .completion()
            .build_workflow()
        )

        assert "INITIAL" in stages
        assert "PLANNING" in stages
        assert "EXECUTION" in stages
        assert "COMPLETION" in stages
        assert len(stages) == 4

    def test_workflow_resets_state(self):
        """Test that workflow() resets internal state."""
        builder = StageBuilder()

        # Build first workflow
        (builder.workflow("First").initial().completion().build_workflow())

        # Start second workflow
        stages = builder.workflow("Second").initial().completion().build_workflow()

        # Should only have stages from second workflow
        assert len(stages) == 2
        assert "INITIAL" in stages
        assert "COMPLETION" in stages

    def test_stage_builder_independence(self):
        """Test that multiple builders maintain independent state."""
        builder1 = StageBuilder()
        builder2 = StageBuilder()

        stage1 = builder1.stage("STAGE1").description("First").build()
        stage2 = builder2.stage("STAGE2").description("Second").build()

        assert stage1.name == "STAGE1"
        assert stage2.name == "STAGE2"
        assert stage1.description == "First"
        assert stage2.description == "Second"

    def test_build_workflow_returns_copy(self):
        """Test that build_workflow() returns a copy, not internal dict."""
        builder = StageBuilder()
        stages = builder.workflow("Test").initial().build_workflow()

        # Modify returned dict
        stages["MODIFIED"] = StageDefinition(
            name="MODIFIED",
            description="Modified stage",
            tools=set(),
            keywords=[],
            next_stages=set(),
        )

        # Internal state should not be affected
        assert "MODIFIED" not in builder._stages

    def test_minimal_stage_definition(self):
        """Test building a stage with only required fields."""
        builder = StageBuilder()
        stage = builder.stage("MINIMAL").build()

        assert stage.name == "MINIMAL"
        assert stage.description == ""
        assert stage.tools == set()
        assert stage.keywords == []
        assert stage.next_stages == set()
        assert stage.min_confidence == 0.5  # Default value

    def test_comprehensive_stage_with_all_fields(self):
        """Test building a comprehensive stage definition."""
        builder = StageBuilder()
        stage = (
            builder.stage("COMPREHENSIVE")
            .description("Full stage with all details")
            .tools({ToolNames.READ, ToolNames.WRITE, ToolNames.SHELL})
            .keywords(["read", "write", "shell"])
            .next_stages({"NEXT1", "NEXT2"})
            .build()
        )

        assert stage.name == "COMPREHENSIVE"
        assert len(stage.tools) == 3
        assert len(stage.keywords) == 3
        assert len(stage.next_stages) == 2


class TestStageBuilderIntegration:
    """Integration tests for StageBuilder with real workflows."""

    def test_research_workflow_example(self):
        """Test building a research-like workflow with StageBuilder."""
        builder = StageBuilder()

        stages = {
            "INITIAL": builder.standard_initial(),
            "SEARCHING": (
                builder.stage("SEARCHING")
                .description("Gathering sources")
                .tools({"web_search", "web_fetch"})
                .keywords(["search", "find"])
                .next_stages({"READING", "SEARCHING"})
                .build()
            ),
            "READING": (
                builder.stage("READING")
                .description("Reading sources")
                .tools({"read", "web_fetch"})
                .keywords(["read", "analyze"])
                .next_stages({"SYNTHESIZING"})
                .build()
            ),
            "COMPLETION": builder.standard_completion(),
        }

        assert len(stages) == 4
        assert all(isinstance(s, StageDefinition) for s in stages.values())
        # Verify transitions
        assert "READING" in stages["SEARCHING"].next_stages
        assert "SYNTHESIZING" in stages["READING"].next_stages

    def test_coding_workflow_example(self):
        """Test building a coding-like workflow with StageBuilder."""
        builder = StageBuilder()

        stages = (
            builder.workflow("Coding")
            .initial()
            .add(
                "PLANNING",
                (
                    builder.stage("PLANNING")
                    .description("Planning the implementation")
                    .tools({ToolNames.READ, ToolNames.GREP})
                    .keywords(["plan", "design"])
                    .next_stages({"READING", "EXECUTION"})
                    .build()
                ),
            )
            .add(
                "EXECUTION",
                (
                    builder.stage("EXECUTION")
                    .description("Implementing changes")
                    .tools({ToolNames.WRITE, ToolNames.EDIT})
                    .keywords(["implement", "write"])
                    .next_stages({"VERIFICATION", "COMPLETION"})
                    .build()
                ),
            )
            .completion()
            .build_workflow()
        )

        assert len(stages) == 4
        assert "PLANNING" in stages
        assert "EXECUTION" in stages
        # Verify INITIAL can transition to PLANNING
        assert stages["INITIAL"].tools == set()  # Should be customizable
