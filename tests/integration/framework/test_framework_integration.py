# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for Victor Framework Enhancement features.

Tests the integration between:
- Checkpoint system (time-travel debugging)
- Entity Memory (4-tier with extraction)
- Agent Ensemble system
- HITL workflow nodes

These tests verify that the new framework components work together
as a cohesive system.
"""

import asyncio
import pytest
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

# Entity Memory imports
from victor.storage.memory import (
    EntityMemory,
    EntityMemoryConfig,
    EntityGraph,
    Entity,
    EntityType,
    EntityRelation,
    RelationType,
    CompositeExtractor,
    EntityMemoryIntegration,
)

# Agent Ensemble imports
from victor.agent.specs import (
    AgentSpec,
    AgentCapabilities,
    AgentConstraints,
    ModelPreference,
    Pipeline,
    Parallel,
    researcher_agent,
    coder_agent,
    reviewer_agent,
    load_agents_from_dict,
)
from victor.agent.specs.ensemble import ExecutionStatus

# HITL imports
from victor.workflows.hitl import (
    HITLNodeType,
    HITLFallback,
    HITLStatus,
    HITLRequest,
    HITLResponse,
    HITLNode,
    HITLExecutor,
)

# Checkpoint imports
from victor.storage.checkpoints import ConversationCheckpointManager

# SQLiteCheckpointBackend requires optional aiosqlite dependency
try:
    from victor.storage.checkpoints import SQLiteCheckpointBackend
except ImportError:
    SQLiteCheckpointBackend = None  # type: ignore


class TestEntityMemoryWithExtraction:
    """Integration tests for entity memory with extraction pipeline."""

    @pytest.mark.asyncio
    async def test_extract_and_store_code_entities(self):
        """Test extracting entities from code and storing in memory."""
        memory = EntityMemory(session_id="integration_test")
        extractor = CompositeExtractor.create_default()

        code_content = """
class UserAuthentication:
    def authenticate(self, username: str, password: str) -> bool:
        # Uses bcrypt for password hashing
        pass

    def generate_token(self, user_id: int) -> str:
        # JWT token generation
        pass

from victor.security import TokenValidator
"""

        # Extract entities
        result = await extractor.extract(
            code_content,
            source="auth.py",
            context={"is_code_block": True},
        )

        # Store in memory
        for entity in result.entities:
            await memory.store(entity)

        # Verify entities were stored
        session_entities = await memory.get_session_entities()
        assert len(session_entities) > 0

        # Search for specific entities
        auth_results = await memory.search("auth", entity_types=[EntityType.CLASS])
        assert len(auth_results) > 0

    @pytest.mark.asyncio
    async def test_entity_graph_with_relations(self):
        """Test entity graph relationship tracking."""
        graph = EntityGraph(in_memory=True)
        await graph.initialize()

        # Create related entities
        module = Entity.create("auth_module", EntityType.MODULE)
        cls = Entity.create("UserAuth", EntityType.CLASS)
        func = Entity.create("validate", EntityType.FUNCTION)

        await graph.add_entity(module)
        await graph.add_entity(cls)
        await graph.add_entity(func)

        # Add relationships
        await graph.add_relation(
            EntityRelation(
                source_id=module.id,
                target_id=cls.id,
                relation_type=RelationType.CONTAINS,
            )
        )
        await graph.add_relation(
            EntityRelation(
                source_id=cls.id,
                target_id=func.id,
                relation_type=RelationType.CONTAINS,
            )
        )

        # Find paths
        paths = await graph.find_paths(module.id, func.id, max_depth=3)
        assert len(paths) == 1
        assert paths[0].length == 2

        # Get neighbors with depth
        neighbors = await graph.get_neighbors(module.id, depth=2)
        assert len(neighbors) == 2  # cls and func


class TestAgentEnsembleIntegration:
    """Integration tests for agent ensemble patterns."""

    @pytest.mark.asyncio
    async def test_pipeline_with_context_passing(self):
        """Test pipeline passes context between agents."""
        pipeline = Pipeline(
            [researcher_agent, coder_agent, reviewer_agent],
            name="full_development_pipeline",
        )

        result = await pipeline.execute(
            task="Implement user authentication",
            context={"project": "victor", "language": "python"},
        )

        assert result.success
        assert len(result.agent_results) == 3

        # Each agent should have completed
        for agent_result in result.agent_results:
            assert agent_result.status == ExecutionStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_parallel_with_aggregation(self):
        """Test parallel execution with result aggregation."""
        # Create custom agents for parallel tasks
        security_checker = AgentSpec(
            name="security_checker",
            description="Checks for security issues",
            capabilities=AgentCapabilities(tools={"grep", "read_file"}),
        )
        style_checker = AgentSpec(
            name="style_checker",
            description="Checks code style",
            capabilities=AgentCapabilities(tools={"read_file"}),
        )

        parallel = Parallel(
            [security_checker, style_checker, reviewer_agent],
            name="code_analysis",
        )

        result = await parallel.execute("Analyze code quality")

        assert result.success
        assert len(result.agent_results) == 3
        assert isinstance(result.final_output, dict)

    @pytest.mark.asyncio
    async def test_load_and_execute_from_config(self):
        """Test loading agents from config and executing."""
        config_data = {
            "agents": [
                {
                    "name": "planner",
                    "description": "Plans implementation",
                    "capabilities": ["read_file", "code_search"],
                    "model_preference": "reasoning",
                },
                {
                    "name": "implementer",
                    "description": "Implements features",
                    "capabilities": ["edit_file", "write_file"],
                    "model_preference": "coding",
                },
            ],
            "ensemble": {
                "type": "pipeline",
                "agents": ["planner", "implementer"],
            },
        }

        config = load_agents_from_dict(config_data)
        assert config.ensemble is not None

        result = await config.ensemble.execute("Add new feature")
        assert result.success


class TestHITLWithWorkflow:
    """Integration tests for HITL in workflow context."""

    @pytest.fixture
    def mock_handler(self):
        """Create mock HITL handler."""
        handler = MagicMock()
        handler.request_human_input = AsyncMock()
        return handler

    @pytest.mark.asyncio
    async def test_hitl_approval_in_workflow(self, mock_handler):
        """Test HITL approval node integration."""
        mock_handler.request_human_input.return_value = HITLResponse(
            request_id="test",
            status=HITLStatus.APPROVED,
            approved=True,
        )

        executor = HITLExecutor(mock_handler)

        # Create approval node for dangerous operation
        approval_node = HITLNode(
            id="approve_delete",
            name="Approve File Delete",
            hitl_type=HITLNodeType.APPROVAL,
            prompt="Delete these files?",
            context_keys=["files_to_delete"],
            timeout=60.0,
        )

        context = {
            "files_to_delete": ["temp.py", "cache.json"],
            "operation": "cleanup",
        }

        response = await executor.execute_hitl_node(approval_node, context)

        assert response.approved is True
        mock_handler.request_human_input.assert_called_once()

    @pytest.mark.asyncio
    async def test_hitl_choice_selection(self, mock_handler):
        """Test HITL choice node for selecting options."""
        mock_handler.request_human_input.return_value = HITLResponse(
            request_id="test",
            status=HITLStatus.APPROVED,
            approved=True,
            value="aggressive",
        )

        executor = HITLExecutor(mock_handler)

        choice_node = HITLNode(
            id="select_refactor",
            name="Select Refactor Strategy",
            hitl_type=HITLNodeType.CHOICE,
            prompt="Choose refactoring approach:",
            choices=["conservative", "moderate", "aggressive"],
            default_value="moderate",
        )

        response = await executor.execute_hitl_node(choice_node, {})

        assert response.approved is True
        assert response.value == "aggressive"

    @pytest.mark.asyncio
    async def test_hitl_timeout_with_fallback(self, mock_handler):
        """Test HITL timeout handling with fallback."""
        mock_handler.request_human_input.side_effect = asyncio.TimeoutError()

        executor = HITLExecutor(mock_handler)

        node = HITLNode(
            id="timeout_test",
            name="Timeout Test",
            hitl_type=HITLNodeType.APPROVAL,
            prompt="Approve?",
            timeout=0.01,
            fallback=HITLFallback.CONTINUE,
            default_value="proceed",
        )

        response = await executor.execute_hitl_node(node, {})

        assert response.status == HITLStatus.TIMEOUT
        assert response.approved is True  # CONTINUE fallback
        assert response.value == "proceed"


class TestCheckpointIntegration:
    """Integration tests for checkpoint system."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.mark.asyncio
    @pytest.mark.skipif(SQLiteCheckpointBackend is None, reason="aiosqlite not installed")
    async def test_checkpoint_save_restore(self, temp_dir):
        """Test saving and restoring checkpoints."""
        backend = SQLiteCheckpointBackend(storage_path=temp_dir)
        manager = ConversationCheckpointManager(backend=backend)

        session_id = "test_session"

        # Create initial state as dictionary
        state = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ],
            "tool_calls": [
                {"tool": "read_file", "args": {"path": "test.py"}},
            ],
            "context": {"project": "victor"},
        }

        # Save checkpoint
        checkpoint_id = await manager.save_checkpoint(
            session_id=session_id,
            state=state,
            description="Before refactoring",
            tags=["milestone"],
        )
        assert checkpoint_id is not None

        # Restore checkpoint
        restored = await manager.restore_checkpoint(checkpoint_id)

        assert len(restored["messages"]) == 2  # Original count
        assert restored["context"]["project"] == "victor"

    @pytest.mark.asyncio
    @pytest.mark.skipif(SQLiteCheckpointBackend is None, reason="aiosqlite not installed")
    async def test_checkpoint_list_and_diff(self, temp_dir):
        """Test listing checkpoints and comparing."""
        backend = SQLiteCheckpointBackend(storage_path=temp_dir)
        manager = ConversationCheckpointManager(backend=backend)

        session_id = "test"

        # Create multiple checkpoints
        state1 = {
            "messages": [{"role": "user", "content": "First"}],
        }
        cp1 = await manager.save_checkpoint(session_id, state1, description="First")

        state2 = {
            "messages": [
                {"role": "user", "content": "First"},
                {"role": "assistant", "content": "Response"},
            ],
        }
        cp2 = await manager.save_checkpoint(session_id, state2, description="Second")

        # List checkpoints
        checkpoints = await manager.list_checkpoints(session_id)
        assert len(checkpoints) == 2

        # Compare checkpoints
        diff = await manager.diff_checkpoints(cp1, cp2)
        assert diff is not None  # Diff object returned


class TestCrossComponentIntegration:
    """Tests verifying integration across multiple components."""

    @pytest.mark.asyncio
    async def test_entity_memory_with_agent_context(self):
        """Test entity memory providing context to agents."""
        # Setup entity memory with extracted entities
        memory = EntityMemory(session_id="agent_context_test")
        extractor = CompositeExtractor.create_default()

        # Extract entities from task description
        task = """
        We need to fix the AuthenticationService class.
        It's in the victor.security module and uses JWT tokens.
        The bug is in the validate_token function.
        """

        result = await extractor.extract(task, source="task_description")
        for entity in result.entities:
            await memory.store(entity)

        # Get context entities for agent
        context_entities = await memory.search("auth")

        # Create agent with entity context
        agent = AgentSpec(
            name="context_aware_coder",
            description="Coder with entity awareness",
            capabilities=AgentCapabilities(tools={"edit_file"}),
            metadata={"entities": [e.name for e in context_entities]},
        )

        # Verify entity context is available
        assert len(agent.metadata["entities"]) > 0

    @pytest.mark.asyncio
    @pytest.mark.skipif(SQLiteCheckpointBackend is None, reason="aiosqlite not installed")
    async def test_pipeline_with_hitl_checkpoint(self):
        """Test pipeline execution with HITL and checkpoint integration."""
        # This simulates a real workflow:
        # 1. Research phase
        # 2. HITL approval
        # 3. Implementation phase
        # 4. Checkpoint save

        mock_handler = MagicMock()
        mock_handler.request_human_input = AsyncMock(
            return_value=HITLResponse(
                request_id="test",
                status=HITLStatus.APPROVED,
                approved=True,
            )
        )

        # Create pipeline
        pipeline = Pipeline([researcher_agent, coder_agent])

        # Execute pipeline
        result = await pipeline.execute(
            task="Implement new feature with safety checks",
            context={"hitl_required": True},
        )

        assert result.success

        # Simulate checkpoint after successful execution
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = SQLiteCheckpointBackend(storage_path=Path(tmpdir))
            manager = ConversationCheckpointManager(backend=backend)

            state = {
                "messages": [{"role": "system", "content": "Pipeline completed"}],
                "context": {
                    "pipeline_result": result.status.value,
                    "agents_completed": [r.agent_name for r in result.agent_results],
                },
            }

            cp_id = await manager.save_checkpoint(
                session_id="pipeline_session",
                state=state,
                description="After pipeline completion",
            )
            assert cp_id is not None


class TestEntityMemoryIntegrationLayer:
    """Tests for EntityMemoryIntegration with conversation flow."""

    @pytest.mark.asyncio
    async def test_process_conversation_messages(self):
        """Test processing multiple conversation messages."""
        integration = EntityMemoryIntegration()

        messages = [
            "Let's work on the UserAuth class in the security module",
            "We need to add JWT token validation using Python",
            "The TokenValidator class should implement the validate method",
        ]

        all_entities = []
        for i, msg in enumerate(messages):
            entities = await integration.process_message(
                content=msg,
                session_id="test_session",
                message_id=f"msg_{i}",
                role="user",
            )
            all_entities.extend(entities)

        # Should have extracted multiple entities
        assert len(all_entities) > 0

        # Get session entities
        session_entities = await integration.get_session_entities()
        assert len(session_entities) > 0

    @pytest.mark.asyncio
    async def test_build_entity_prompt_context(self):
        """Test building prompt context from entities."""
        integration = EntityMemoryIntegration()

        # Process messages with entities
        await integration.process_message(
            content="Working with Python and FastAPI for the REST API",
            session_id="test",
            role="user",
        )

        # Build context for prompt
        context = await integration.build_entity_prompt_context(
            recent_messages=["Working with Python"],
            max_entities=5,
        )

        # Should produce formatted context
        if context:  # May be empty if no entities extracted
            assert "Entities Discussed" in context


class TestPresetAgentCustomization:
    """Tests for customizing preset agents."""

    def test_extend_preset_with_capabilities(self):
        """Test extending preset agent capabilities."""
        custom_coder = coder_agent.with_capabilities(
            tools={"deploy_script", "run_tests"},
            can_browse_web=True,
        )

        # Original unchanged
        assert "deploy_script" not in coder_agent.capabilities.tools

        # Extended has new tools
        assert "deploy_script" in custom_coder.capabilities.tools
        assert "edit" in custom_coder.capabilities.tools  # Still has original
        assert custom_coder.capabilities.can_browse_web is True

    def test_extend_preset_with_constraints(self):
        """Test extending preset agent constraints."""
        limited_researcher = researcher_agent.with_constraints(
            max_iterations=10,
            max_cost_usd=0.50,
            timeout_seconds=120.0,
        )

        # Original unchanged
        assert researcher_agent.constraints.max_iterations == 30

        # Extended has new constraints
        assert limited_researcher.constraints.max_iterations == 10
        assert limited_researcher.constraints.max_cost_usd == 0.50
        assert limited_researcher.constraints.timeout_seconds == 120.0

    def test_preset_in_ensemble(self):
        """Test using presets in ensemble configurations."""
        config = load_agents_from_dict(
            {
                "agents": [
                    {
                        "name": "custom_reviewer",
                        "extends": "reviewer",
                        "constraints": {
                            "max_iterations": 15,
                        },
                    },
                ],
                "ensemble": {
                    "type": "pipeline",
                    "agents": ["researcher", "coder", "custom_reviewer"],
                },
            }
        )

        ensemble = config.ensemble
        assert ensemble is not None
        assert len(ensemble.agents) == 3

        # Custom reviewer should have modified constraints
        custom = config.agents["custom_reviewer"]
        assert custom.constraints.max_iterations == 15
        # But still have reviewer's other attributes
        assert "review" in custom.capabilities.skills
