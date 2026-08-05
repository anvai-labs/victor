# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Agent definition and ensemble system.

Tests cover:
- AgentSpec creation and serialization
- AgentCapabilities and AgentConstraints
- Ensemble patterns (Pipeline, Parallel, Hierarchical)
- Preset agents
- YAML/JSON loading
"""

import pytest
import tempfile
from pathlib import Path

from victor.agent.specs import (
    AgentSpec,
    AgentCapabilities,
    AgentConstraints,
    ModelPreference,
    Pipeline,
    Parallel,
    Hierarchical,
    EnsembleType,
    researcher_agent,
    coder_agent,
    reviewer_agent,
    load_agents_from_dict,
    load_agents_from_yaml,
)
from victor.agent.specs.models import OutputFormat, DelegationPolicy
from victor.agent.specs.ensemble import ExecutionStatus, AgentResult, EnsembleResult
from victor.agent.specs.presets import get_preset_agent, list_preset_agents


class TestAgentCapabilities:
    """Tests for AgentCapabilities dataclass."""

    def test_create_capabilities(self):
        """Test creating capabilities."""
        caps = AgentCapabilities(
            tools={"read_file", "edit_file"},
            can_execute_code=True,
            can_modify_files=True,
        )

        assert "read" in caps.tools
        assert "edit" in caps.tools
        assert caps.can_execute_code is True
        assert caps.can_browse_web is False  # Default

    def test_canonicalizes_legacy_tool_names(self):
        """Legacy aliases should normalize to canonical internal tool names."""
        caps = AgentCapabilities(
            tools={
                "read_file",
                "write_file",
                "edit_file",
                "execute_bash",
                "list_directory",
            }
        )

        assert caps.tools == {"read", "write", "edit", "shell", "ls"}

    def test_allows_tool(self):
        """Test tool permission checking."""
        caps = AgentCapabilities(
            tools={"read_file", "write_file"},
            tool_patterns=["git_*"],
        )

        assert caps.allows_tool("read_file") is True
        assert caps.allows_tool("git_status") is True
        assert caps.allows_tool("git_commit") is True
        assert caps.allows_tool("delete_file") is False

    def test_allows_tool_accepts_legacy_or_canonical_names(self):
        """Capability checks should resolve aliases to the canonical tool surface."""
        caps = AgentCapabilities(tools={"read", "write", "shell"})

        assert caps.allows_tool("read") is True
        assert caps.allows_tool("read_file") is True
        assert caps.allows_tool("write_file") is True
        assert caps.allows_tool("execute_bash") is True

    def test_serialization(self):
        """Test capabilities to_dict/from_dict."""
        caps = AgentCapabilities(
            tools={"search", "read"},
            skills={"analysis"},
            can_browse_web=True,
        )

        data = caps.to_dict()
        restored = AgentCapabilities.from_dict(data)

        assert restored.tools == caps.tools
        assert restored.skills == caps.skills
        assert restored.can_browse_web == caps.can_browse_web


class TestAgentConstraints:
    """Tests for AgentConstraints dataclass."""

    def test_create_constraints(self):
        """Test creating constraints."""
        constraints = AgentConstraints(
            max_iterations=50,
            max_tool_calls=100,
            max_cost_usd=1.0,
            timeout_seconds=300.0,
        )

        assert constraints.max_iterations == 50
        assert constraints.max_cost_usd == 1.0
        assert constraints.timeout_seconds == 300.0

    def test_default_constraints(self):
        """Test default constraint values."""
        constraints = AgentConstraints()

        assert constraints.max_iterations == 50
        assert constraints.max_tool_calls == 100
        assert constraints.max_cost_usd is None  # No default limit

    def test_serialization(self):
        """Test constraints serialization."""
        constraints = AgentConstraints(
            max_iterations=30,
            required_tools={"read_file", "write_file"},
        )

        data = constraints.to_dict()
        restored = AgentConstraints.from_dict(data)

        assert restored.max_iterations == 30
        assert restored.required_tools == {"read", "write"}


class TestAgentSpec:
    """Tests for AgentSpec dataclass."""

    def test_create_agent_spec(self):
        """Test creating an agent specification."""
        agent = AgentSpec(
            name="my_agent",
            description="A test agent",
            capabilities=AgentCapabilities(tools={"read_file"}),
            model_preference=ModelPreference.REASONING,
        )

        assert agent.name == "my_agent"
        assert agent.description == "A test agent"
        assert agent.model_preference == ModelPreference.REASONING
        assert "read" in agent.capabilities.tools

    def test_agent_id_generation(self):
        """Test that agent ID is deterministic."""
        agent1 = AgentSpec(name="test", description="desc", version="1.0")
        agent2 = AgentSpec(name="test", description="different", version="1.0")
        agent3 = AgentSpec(name="test", description="desc", version="2.0")

        # Same name and version should have same ID
        assert agent1.id == agent2.id
        # Different version should have different ID
        assert agent1.id != agent3.id

    def test_with_capabilities(self):
        """Test creating new spec with added capabilities."""
        base = AgentSpec(
            name="base",
            description="Base agent",
            capabilities=AgentCapabilities(tools={"read_file"}),
        )

        extended = base.with_capabilities(
            tools={"write_file"},
            can_execute_code=True,
        )

        # Original unchanged
        assert "write_file" not in base.capabilities.tools

        # Extended has both
        assert "read" in extended.capabilities.tools
        assert "write" in extended.capabilities.tools
        assert extended.capabilities.can_execute_code is True

    def test_with_constraints(self):
        """Test creating new spec with updated constraints."""
        base = AgentSpec(
            name="base",
            description="Base agent",
            constraints=AgentConstraints(max_iterations=50),
        )

        limited = base.with_constraints(
            max_iterations=20,
            max_cost_usd=0.5,
        )

        # Original unchanged
        assert base.constraints.max_iterations == 50

        # New has updated constraints
        assert limited.constraints.max_iterations == 20
        assert limited.constraints.max_cost_usd == 0.5

    def test_serialization(self):
        """Test agent spec serialization."""
        agent = AgentSpec(
            name="serializable",
            description="Test serialization",
            capabilities=AgentCapabilities(tools={"tool1"}, can_browse_web=True),
            constraints=AgentConstraints(max_iterations=25),
            model_preference=ModelPreference.CODING,
            output_format=OutputFormat.CODE,
            tags={"test", "serialization"},
        )

        data = agent.to_dict()
        restored = AgentSpec.from_dict(data)

        assert restored.name == agent.name
        assert restored.description == agent.description
        assert restored.model_preference == agent.model_preference
        assert restored.output_format == agent.output_format
        assert "tool1" in restored.capabilities.tools
        assert restored.constraints.max_iterations == 25

    def test_from_dict_shorthand(self):
        """Test loading from dict with capability shorthand."""
        data = {
            "name": "shorthand",
            "description": "Using shorthand",
            "capabilities": ["tool1", "tool2", "tool3"],
            "model_preference": "fast",
        }

        agent = AgentSpec.from_dict(data)

        assert agent.name == "shorthand"
        assert "tool1" in agent.capabilities.tools
        assert "tool2" in agent.capabilities.tools
        assert agent.model_preference == ModelPreference.FAST


class TestPipeline:
    """Tests for Pipeline ensemble."""

    def test_create_pipeline(self):
        """Test creating a pipeline."""
        agents = [researcher_agent, coder_agent]
        pipeline = Pipeline(agents, name="test_pipeline")

        assert pipeline.name == "test_pipeline"
        assert pipeline.ensemble_type == EnsembleType.PIPELINE
        assert len(pipeline.agents) == 2

    @pytest.mark.asyncio
    async def test_execute_pipeline(self):
        """Test executing a pipeline."""
        agents = [researcher_agent, coder_agent]
        pipeline = Pipeline(agents)

        result = await pipeline.execute("Test task")

        assert result.success
        assert result.status == ExecutionStatus.COMPLETED
        assert len(result.agent_results) == 2
        assert all(r.success for r in result.agent_results)

    @pytest.mark.asyncio
    async def test_pipeline_serialization(self):
        """Test pipeline to_dict."""
        pipeline = Pipeline([researcher_agent, coder_agent])
        data = pipeline.to_dict()

        assert data["type"] == "pipeline"
        assert len(data["agents"]) == 2


class TestParallel:
    """Tests for Parallel ensemble."""

    def test_create_parallel(self):
        """Test creating a parallel ensemble."""
        agents = [researcher_agent, reviewer_agent]
        parallel = Parallel(agents, name="test_parallel")

        assert parallel.ensemble_type == EnsembleType.PARALLEL
        assert len(parallel.agents) == 2

    @pytest.mark.asyncio
    async def test_execute_parallel(self):
        """Test executing in parallel."""
        agents = [researcher_agent, reviewer_agent]
        parallel = Parallel(agents)

        result = await parallel.execute("Test task")

        assert result.success
        assert len(result.agent_results) == 2
        # All should complete (parallel execution)
        assert result.final_output is not None


class TestHierarchical:
    """Tests for Hierarchical ensemble."""

    def test_create_hierarchical(self):
        """Test creating a hierarchical ensemble."""
        hierarchical = Hierarchical(
            manager=researcher_agent,
            workers=[coder_agent, reviewer_agent],
        )

        assert hierarchical.ensemble_type == EnsembleType.HIERARCHICAL
        assert hierarchical.manager.name == "researcher"
        assert len(hierarchical.workers) == 2

    @pytest.mark.asyncio
    async def test_execute_hierarchical(self):
        """Test executing hierarchical."""
        hierarchical = Hierarchical(
            manager=researcher_agent,
            workers=[coder_agent],
        )

        result = await hierarchical.execute("Test task")

        assert result.success
        assert len(result.agent_results) >= 2  # Manager + workers


class TestPresets:
    """Tests for preset agents."""

    def test_researcher_preset(self):
        """Test researcher preset configuration."""
        assert researcher_agent.name == "researcher"
        assert researcher_agent.model_preference == ModelPreference.REASONING
        assert "web_search" in researcher_agent.capabilities.tools
        assert researcher_agent.capabilities.can_browse_web is True

    def test_coder_preset(self):
        """Test coder preset configuration."""
        assert coder_agent.name == "coder"
        assert coder_agent.model_preference == ModelPreference.CODING
        assert "edit" in coder_agent.capabilities.tools
        assert coder_agent.capabilities.can_modify_files is True

    def test_reviewer_preset(self):
        """Test reviewer preset configuration."""
        assert reviewer_agent.name == "reviewer"
        assert "review" in reviewer_agent.capabilities.skills
        assert reviewer_agent.capabilities.can_modify_files is False

    def test_get_preset_agent(self):
        """Test getting preset by name."""
        agent = get_preset_agent("researcher")
        assert agent.name == "researcher"

        with pytest.raises(ValueError):
            get_preset_agent("nonexistent")

    def test_list_preset_agents(self):
        """Test listing available presets."""
        presets = list_preset_agents()

        assert "researcher" in presets
        assert "coder" in presets
        assert "reviewer" in presets
        assert len(presets) >= 5


class TestAgentConfig:
    """Tests for agent configuration loading."""

    def test_load_from_dict_simple(self):
        """Test loading simple configuration."""
        data = {
            "agents": [
                {
                    "name": "custom",
                    "description": "Custom agent",
                    "capabilities": ["read_file", "write_file"],
                }
            ]
        }

        config = load_agents_from_dict(data)

        assert "custom" in config.agents
        assert "read" in config.agents["custom"].capabilities.tools
        assert "write" in config.agents["custom"].capabilities.tools

    def test_load_with_preset_extension(self):
        """Test extending preset agents."""
        data = {
            "agents": [
                {
                    "name": "custom_coder",
                    "extends": "coder",
                    "description": "Extended coder",
                    "capabilities": {
                        "tools": ["deploy_script"],
                    },
                }
            ]
        }

        config = load_agents_from_dict(data)

        custom = config.agents["custom_coder"]
        # Should have coder's tools plus new one
        assert "edit" in custom.capabilities.tools
        assert "deploy_script" in custom.capabilities.tools

    def test_load_with_ensemble(self):
        """Test loading with ensemble configuration."""
        data = {
            "agents": [
                {"name": "agent1", "description": "First"},
                {"name": "agent2", "description": "Second"},
            ],
            "ensemble": {
                "type": "pipeline",
                "agents": ["agent1", "agent2"],
            },
        }

        config = load_agents_from_dict(data)

        assert config.ensemble is not None
        assert config.ensemble.ensemble_type == EnsembleType.PIPELINE
        assert len(config.ensemble.agents) == 2

    def test_load_from_yaml(self):
        """Test loading from YAML file."""
        yaml_content = """
agents:
  - name: yaml_agent
    description: Agent from YAML
    capabilities:
      tools:
        - search
        - read
      can_browse_web: true
    model_preference: reasoning

ensemble:
  type: pipeline
  agents: [yaml_agent]
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            config = load_agents_from_yaml(temp_path)

            assert "yaml_agent" in config.agents
            agent = config.agents["yaml_agent"]
            assert "search" in agent.capabilities.tools
            assert agent.capabilities.can_browse_web is True
            assert config.ensemble is not None
        finally:
            Path(temp_path).unlink()


class TestAgentResult:
    """Tests for AgentResult dataclass."""

    def test_create_result(self):
        """Test creating an agent result."""
        result = AgentResult(
            agent_name="test",
            status=ExecutionStatus.COMPLETED,
            output="Task done",
            tokens_used=100,
            cost_usd=0.001,
        )

        assert result.success is True
        assert result.agent_name == "test"
        assert result.output == "Task done"

    def test_failed_result(self):
        """Test failed result."""
        result = AgentResult(
            agent_name="test",
            status=ExecutionStatus.FAILED,
            error="Something went wrong",
        )

        assert result.success is False
        assert result.error is not None


class TestEnsembleResult:
    """Tests for EnsembleResult dataclass."""

    def test_aggregate_stats(self):
        """Test stat aggregation."""
        result = EnsembleResult(
            ensemble_type=EnsembleType.PIPELINE,
            status=ExecutionStatus.COMPLETED,
            agent_results=[
                AgentResult(
                    agent_name="a1",
                    status=ExecutionStatus.COMPLETED,
                    tokens_used=100,
                    cost_usd=0.01,
                    duration_ms=1000,
                ),
                AgentResult(
                    agent_name="a2",
                    status=ExecutionStatus.COMPLETED,
                    tokens_used=200,
                    cost_usd=0.02,
                    duration_ms=2000,
                ),
            ],
        )

        result.aggregate_stats()

        assert result.total_tokens == 300
        assert result.total_cost_usd == pytest.approx(0.03)
        assert result.total_duration_ms == 2000  # Max of parallel tasks


class TestModelPreference:
    """Tests for ModelPreference enum."""

    def test_model_preferences(self):
        """Test all model preferences are valid."""
        assert ModelPreference.REASONING.value == "reasoning"
        assert ModelPreference.CODING.value == "coding"
        assert ModelPreference.FAST.value == "fast"
        assert ModelPreference.TOOL_USE.value == "tool_use"
        assert ModelPreference.LONG_CONTEXT.value == "long_context"
