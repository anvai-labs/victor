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

"""Unit tests for SubAgent protocols.

Tests the SubAgentContext protocol and SubAgentContextAdapter,
ensuring ISP and DIP compliance for subagent dependencies.
"""

import pytest
from typing import Any, Dict
from unittest.mock import MagicMock


class TestSubAgentContextProtocol:
    """Tests for the SubAgentContext protocol definition."""

    def test_protocol_is_runtime_checkable(self):
        """SubAgentContext should be runtime checkable."""
        from victor.agent.subagents.protocols import SubAgentContext

        # Create a minimal implementation
        class MinimalContext:
            @property
            def settings(self) -> Any:
                return {}

            @property
            def provider(self) -> Any:
                return MagicMock()

            @property
            def provider_name(self) -> str:
                return "test_provider"

            @property
            def model(self) -> str:
                return "test_model"

            @property
            def tool_registry(self) -> Any:
                return MagicMock()

            @property
            def temperature(self) -> float:
                return 0.7

            @property
            def vertical_context(self) -> Any:
                return None

        context = MinimalContext()
        assert isinstance(context, SubAgentContext)

    def test_protocol_rejects_incomplete_implementation(self):
        """SubAgentContext should reject incomplete implementations."""
        from victor.agent.subagents.protocols import SubAgentContext

        # Create an incomplete implementation (missing tool_registry)
        class IncompleteContext:
            @property
            def settings(self) -> Any:
                return {}

            @property
            def provider_name(self) -> str:
                return "test"

            @property
            def model(self) -> str:
                return "test"

            # Missing tool_registry

        context = IncompleteContext()
        # Runtime checkable will fail because tool_registry is missing
        assert not isinstance(context, SubAgentContext)

    def test_protocol_exported_in_subagents_module(self):
        """SubAgentContext should be exported from subagents module."""
        from victor.agent import subagents

        assert hasattr(subagents, "SubAgentContext")
        assert "SubAgentContext" in subagents.__all__

    def test_protocol_exported_in_protocols_module(self):
        """SubAgentContext should be importable from protocols module."""
        from victor.agent.subagents.protocols import SubAgentContext

        assert SubAgentContext is not None


class TestSubAgentContextAdapter:
    """Tests for SubAgentContextAdapter implementation."""

    @pytest.fixture
    def mock_orchestrator(self):
        """Create a mock orchestrator with required attributes."""
        orchestrator = MagicMock()
        orchestrator.settings = MagicMock(tool_budget=30, max_context_chars=100000)
        orchestrator.provider = MagicMock()
        orchestrator.provider_name = "anthropic"
        orchestrator.model = "claude-sonnet-4-20250514"
        # AgentOrchestrator uses both 'tools' and 'tool_registry' pointing to same object
        tool_reg = MagicMock()
        orchestrator.tools = tool_reg
        orchestrator.tool_registry = tool_reg
        orchestrator.temperature = 0.7
        return orchestrator

    def test_adapter_implements_protocol(self, mock_orchestrator):
        """SubAgentContextAdapter should implement SubAgentContext protocol."""
        from victor.agent.subagents.protocols import (
            SubAgentContext,
            SubAgentContextAdapter,
        )

        adapter = SubAgentContextAdapter(mock_orchestrator)
        assert isinstance(adapter, SubAgentContext)

    def test_adapter_delegates_settings(self, mock_orchestrator):
        """Adapter should delegate settings property to orchestrator."""
        from victor.agent.subagents.protocols import SubAgentContextAdapter

        adapter = SubAgentContextAdapter(mock_orchestrator)
        assert adapter.settings == mock_orchestrator.settings
        assert adapter.settings.tool_budget == 30

    def test_adapter_delegates_provider_name(self, mock_orchestrator):
        """Adapter should delegate provider_name property to orchestrator."""
        from victor.agent.subagents.protocols import SubAgentContextAdapter

        adapter = SubAgentContextAdapter(mock_orchestrator)
        assert adapter.provider_name == "anthropic"

    def test_adapter_delegates_model(self, mock_orchestrator):
        """Adapter should delegate model property to orchestrator."""
        from victor.agent.subagents.protocols import SubAgentContextAdapter

        adapter = SubAgentContextAdapter(mock_orchestrator)
        assert adapter.model == "claude-sonnet-4-20250514"

    def test_adapter_delegates_tool_registry(self, mock_orchestrator):
        """Adapter should delegate tool_registry property to orchestrator."""
        from victor.agent.subagents.protocols import SubAgentContextAdapter

        adapter = SubAgentContextAdapter(mock_orchestrator)
        assert adapter.tool_registry == mock_orchestrator.tool_registry

    def test_adapter_delegates_temperature(self, mock_orchestrator):
        """Adapter should delegate temperature property to orchestrator."""
        from victor.agent.subagents.protocols import SubAgentContextAdapter

        adapter = SubAgentContextAdapter(mock_orchestrator)
        assert adapter.temperature == 0.7

    def test_adapter_exported_in_subagents_module(self):
        """SubAgentContextAdapter should be exported from subagents module."""
        from victor.agent import subagents

        assert hasattr(subagents, "SubAgentContextAdapter")
        assert "SubAgentContextAdapter" in subagents.__all__


class TestSubAgentContextUsage:
    """Tests for SubAgentContext usage patterns."""

    def test_can_use_as_type_hint(self):
        """SubAgentContext can be used as a type hint."""
        from victor.agent.subagents.protocols import SubAgentContext

        def get_provider_info(context: SubAgentContext) -> str:
            """Function accepting SubAgentContext."""
            return f"{context.provider_name}/{context.model}"

        # Create conforming implementation
        class TestContext:
            @property
            def settings(self) -> Any:
                return {}

            @property
            def provider_name(self) -> str:
                return "openai"

            @property
            def model(self) -> str:
                return "gpt-4o"

            @property
            def tool_registry(self) -> Any:
                return None

            @property
            def temperature(self) -> float:
                return 0.5

        context = TestContext()
        result = get_provider_info(context)
        assert result == "openai/gpt-4o"

    def test_mock_substitution_for_testing(self):
        """SubAgentContext enables easy mock substitution."""
        from victor.agent.subagents.protocols import SubAgentContext

        # Create a mock that conforms to the protocol
        mock_context = MagicMock(spec=SubAgentContext)
        mock_context.settings = MagicMock(tool_budget=10)
        mock_context.provider_name = "test"
        mock_context.model = "test-model"
        mock_context.tool_registry = MagicMock()

        # Use in code expecting SubAgentContext
        assert mock_context.provider_name == "test"
        assert mock_context.settings.tool_budget == 10


class TestToolExecutorProtocol:
    """Tests for the enhanced ToolExecutorProtocol."""

    def test_protocol_is_runtime_checkable(self):
        """ToolExecutorProtocol should be runtime checkable."""
        from victor.agent.protocols import ToolExecutorProtocol

        # Create a minimal implementation
        class MinimalExecutor:
            def execute(
                self, tool_name: str, arguments: Dict[str, Any], context: Any = None
            ) -> Any:
                return {"result": "success"}

            async def aexecute(
                self, tool_name: str, arguments: Dict[str, Any], context: Any = None
            ) -> Any:
                return {"result": "success"}

            def validate_arguments(self, tool_name: str, arguments: Dict[str, Any]) -> bool:
                return True

        executor = MinimalExecutor()
        assert isinstance(executor, ToolExecutorProtocol)

    def test_protocol_rejects_incomplete_implementation(self):
        """ToolExecutorProtocol should reject incomplete implementations."""
        from victor.agent.protocols import ToolExecutorProtocol

        # Create an incomplete implementation (missing validate_arguments)
        class IncompleteExecutor:
            def execute(
                self, tool_name: str, arguments: Dict[str, Any], context: Any = None
            ) -> Any:
                return {}

            async def aexecute(
                self, tool_name: str, arguments: Dict[str, Any], context: Any = None
            ) -> Any:
                return {}

            # Missing validate_arguments

        executor = IncompleteExecutor()
        assert not isinstance(executor, ToolExecutorProtocol)

    def test_protocol_exported_in_all(self):
        """ToolExecutorProtocol should be exported in __all__."""
        from victor.agent import protocols

        assert "ToolExecutorProtocol" in protocols.__all__

    def test_can_use_as_type_hint(self):
        """ToolExecutorProtocol can be used as a type hint."""
        from victor.agent.protocols import ToolExecutorProtocol
        from typing import Any, Dict

        def run_tool_safely(executor: ToolExecutorProtocol, tool: str, args: Dict[str, Any]) -> Any:
            """Function accepting ToolExecutorProtocol."""
            if executor.validate_arguments(tool, args):
                return executor.execute(tool, args)
            raise ValueError(f"Invalid arguments for {tool}")

        # Create conforming implementation
        class TestExecutor:
            def execute(
                self, tool_name: str, arguments: Dict[str, Any], context: Any = None
            ) -> Any:
                return {"executed": tool_name}

            async def aexecute(
                self, tool_name: str, arguments: Dict[str, Any], context: Any = None
            ) -> Any:
                return {"executed": tool_name}

            def validate_arguments(self, tool_name: str, arguments: Dict[str, Any]) -> bool:
                return "required_arg" in arguments

        executor = TestExecutor()

        # Valid args
        result = run_tool_safely(executor, "test_tool", {"required_arg": "value"})
        assert result == {"executed": "test_tool"}

        # Invalid args
        with pytest.raises(ValueError, match="Invalid arguments"):
            run_tool_safely(executor, "test_tool", {"wrong_arg": "value"})


class TestToolExecutorProtocolImplementation:
    """Tests for concrete ToolExecutorProtocol implementations."""

    @pytest.fixture
    def mock_executor(self):
        """Create a mock executor implementing ToolExecutorProtocol."""

        class MockExecutor:
            def __init__(self):
                self.executed_tools = []
                self.valid_tools = {"read", "write", "search"}

            def execute(
                self, tool_name: str, arguments: Dict[str, Any], context: Any = None
            ) -> Any:
                self.executed_tools.append((tool_name, arguments))
                return {"tool": tool_name, "args": arguments, "sync": True}

            async def aexecute(
                self, tool_name: str, arguments: Dict[str, Any], context: Any = None
            ) -> Any:
                self.executed_tools.append((tool_name, arguments))
                return {"tool": tool_name, "args": arguments, "async": True}

            def validate_arguments(self, tool_name: str, arguments: Dict[str, Any]) -> bool:
                return tool_name in self.valid_tools

        return MockExecutor()

    def test_execute_returns_result(self, mock_executor):
        """Execute should return tool result."""
        result = mock_executor.execute("read", {"path": "/test"})

        assert result["tool"] == "read"
        assert result["args"] == {"path": "/test"}
        assert result["sync"] is True

    @pytest.mark.asyncio
    async def test_aexecute_returns_result(self, mock_executor):
        """Async execute should return tool result."""
        result = await mock_executor.aexecute("write", {"path": "/test", "content": "hello"})

        assert result["tool"] == "write"
        assert result["async"] is True

    def test_validate_arguments_valid_tool(self, mock_executor):
        """Validate should return True for valid tools."""
        assert mock_executor.validate_arguments("read", {"path": "/"}) is True
        assert mock_executor.validate_arguments("write", {"path": "/"}) is True
        assert mock_executor.validate_arguments("search", {}) is True

    def test_validate_arguments_invalid_tool(self, mock_executor):
        """Validate should return False for invalid tools."""
        assert mock_executor.validate_arguments("unknown_tool", {}) is False
        assert mock_executor.validate_arguments("delete_all", {}) is False

    def test_execute_with_context(self, mock_executor):
        """Execute should accept optional context parameter."""
        context = {"workspace": "/project", "session_id": "abc123"}
        result = mock_executor.execute("read", {"path": "/test"}, context=context)

        assert result is not None
        assert len(mock_executor.executed_tools) == 1

    def test_tracks_executed_tools(self, mock_executor):
        """Executor should track executed tools."""
        mock_executor.execute("read", {"path": "/a"})
        mock_executor.execute("write", {"path": "/b", "content": "x"})

        assert len(mock_executor.executed_tools) == 2
        assert mock_executor.executed_tools[0][0] == "read"
        assert mock_executor.executed_tools[1][0] == "write"


class TestProtocolIntegration:
    """Integration tests for protocol usage patterns."""

    def test_subagent_context_with_tool_executor(self):
        """Test using SubAgentContext with ToolExecutorProtocol together."""
        from victor.agent.subagents.protocols import SubAgentContext
        from victor.agent.protocols import ToolExecutorProtocol

        # Create conforming implementations
        class ContextImpl:
            @property
            def settings(self) -> Any:
                return {"tool_budget": 20}

            @property
            def provider(self) -> Any:
                return MagicMock()

            @property
            def provider_name(self) -> str:
                return "anthropic"

            @property
            def model(self) -> str:
                return "claude-sonnet-4-20250514"

            @property
            def tool_registry(self) -> Any:
                return MagicMock()

            @property
            def temperature(self) -> float:
                return 0.7

            @property
            def vertical_context(self) -> Any:
                return None

        class ExecutorImpl:
            def execute(
                self, tool_name: str, arguments: Dict[str, Any], context: Any = None
            ) -> Any:
                return {"success": True}

            async def aexecute(
                self, tool_name: str, arguments: Dict[str, Any], context: Any = None
            ) -> Any:
                return {"success": True}

            def validate_arguments(self, tool_name: str, arguments: Dict[str, Any]) -> bool:
                return True

        context = ContextImpl()
        executor = ExecutorImpl()

        assert isinstance(context, SubAgentContext)
        assert isinstance(executor, ToolExecutorProtocol)

        # Use together
        if executor.validate_arguments("test", {"arg": "value"}):
            result = executor.execute("test", {"arg": "value"})
            assert result["success"] is True

    def test_adapter_pattern_for_dependency_injection(self):
        """Test adapter pattern enables clean dependency injection."""
        from victor.agent.subagents.protocols import (
            SubAgentContext,
            SubAgentContextAdapter,
        )

        # Mock orchestrator
        orchestrator = MagicMock()
        orchestrator.settings = MagicMock(tool_budget=25)
        orchestrator.provider_name = "openai"
        orchestrator.model = "gpt-4o"
        orchestrator.tool_registry = MagicMock()
        orchestrator.temperature = 0.5

        # Adapt orchestrator to SubAgentContext
        context = SubAgentContextAdapter(orchestrator)

        # Function accepting SubAgentContext
        def configure_subagent(ctx: SubAgentContext) -> Dict[str, Any]:
            return {
                "provider": ctx.provider_name,
                "model": ctx.model,
                "budget": ctx.settings.tool_budget,
            }

        config = configure_subagent(context)

        assert config["provider"] == "openai"
        assert config["model"] == "gpt-4o"
        assert config["budget"] == 25
