"""Unit tests for the victor.framework module.

Tests the simplified "golden path" API layer.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# Test imports work
from victor.framework import (
    Agent,
    AgentConfig,
    AgentError,
    AgentExecutionEvent,
    BudgetExhaustedError,
    CancellationError,
    ChatSession,
    ConfigurationError,
    EventType,
    ProviderError,
    Stage,
    State,
    StateHooks,
    UnifiedAgentConfig,
    Task,
    TaskResult,
    FrameworkTaskType,
    ToolCategory,
    ToolError,
    Tools,
    ToolSet,
)

# Also test top-level imports
from victor import Agent as AgentTopLevel
from victor import ToolSet as ToolSetTopLevel


class TestToolSet:
    """Tests for ToolSet configuration."""

    def test_default_includes_core_and_git(self):
        """Default ToolSet should include core, filesystem, and git tools."""
        ts = ToolSet.default()
        tools = ts.get_tool_names()
        assert "read" in tools
        assert "write" in tools
        assert "shell" in tools
        assert "git" in tools or "git_status" in tools

    def test_minimal_includes_only_core(self):
        """Minimal ToolSet should include only core tools."""
        ts = ToolSet.minimal()
        tools = ts.get_tool_names()
        assert "read" in tools
        assert "write" in tools
        assert "edit" in tools
        # Should not include git
        assert "git" not in tools
        assert "git_status" not in tools

    def test_full_includes_all_categories(self):
        """Full ToolSet should include all categories."""
        ts = ToolSet.full()
        # All categories should be included
        assert len(ts.categories) == len(ToolCategory)

    def test_airgapped_excludes_network_tools(self):
        """Airgapped ToolSet should exclude network tools."""
        ts = ToolSet.airgapped()
        tools = ts.get_tool_names()
        assert "web_search" not in tools
        assert "web_fetch" not in tools
        assert "http_request" not in tools

    def test_from_categories(self):
        """ToolSet.from_categories should create set with specified categories."""
        ts = ToolSet.from_categories(["core", "git"])
        assert "core" in ts.categories
        assert "git" in ts.categories
        assert len(ts.categories) == 2

    def test_from_tools(self):
        """ToolSet.from_tools should create set with specific tools."""
        ts = ToolSet.from_tools(["read", "write", "custom_tool"])
        assert "read" in ts.tools
        assert "write" in ts.tools
        assert "custom_tool" in ts.tools

    def test_include_adds_tools(self):
        """include() should add tools to the set."""
        ts = ToolSet.minimal()
        ts2 = ts.include("docker")
        assert "docker" in ts2.tools
        # Original should be unchanged
        assert "docker" not in ts.tools

    def test_exclude_tools_removes_tools(self):
        """exclude_tools() should remove tools from the set."""
        ts = ToolSet.default()
        ts2 = ts.exclude_tools("shell")
        assert "shell" in ts2.exclude
        tools = ts2.get_tool_names()
        assert "shell" not in tools

    def test_contains_check(self):
        """__contains__ should check if tool is in the set."""
        ts = ToolSet.from_tools(["read", "write"])
        assert "read" in ts
        assert "unknown" not in ts


class TestConfigExports:
    """Tests for framework config exports."""

    def test_unified_agent_config_exported(self):
        """UnifiedAgentConfig should be available from victor.framework."""
        assert UnifiedAgentConfig.high_budget().tool_budget == 200


class TestEventType:
    """Tests for EventType enum."""

    def test_content_types(self):
        """Content event types should exist."""
        assert EventType.CONTENT.value == "content"
        assert EventType.THINKING.value == "thinking"

    def test_tool_types(self):
        """Tool event types should exist."""
        assert EventType.TOOL_CALL.value == "tool_call"
        assert EventType.TOOL_RESULT.value == "tool_result"
        assert EventType.TOOL_ERROR.value == "tool_error"

    def test_lifecycle_types(self):
        """Lifecycle event types should exist."""
        assert EventType.STREAM_START.value == "stream_start"
        assert EventType.STREAM_END.value == "stream_end"


class TestEvent:
    """Tests for Event dataclass."""

    def test_content_event(self):
        """Content event should have correct properties."""
        event = AgentExecutionEvent(type=EventType.CONTENT, content="Hello world")
        assert event.type == EventType.CONTENT
        assert event.content == "Hello world"
        assert event.is_content_event
        assert not event.is_tool_event
        assert not event.is_error_event

    def test_tool_call_event(self):
        """Tool call event should have correct properties."""
        event = AgentExecutionEvent(
            type=EventType.TOOL_CALL,
            tool_name="read",
            arguments={"path": "/tmp/test.txt"},
        )
        assert event.type == EventType.TOOL_CALL
        assert event.tool_name == "read"
        assert event.arguments == {"path": "/tmp/test.txt"}
        assert event.is_tool_event
        assert not event.is_content_event

    def test_error_event(self):
        """Error event should have correct properties."""
        event = AgentExecutionEvent(type=EventType.ERROR, error="Something went wrong")
        assert event.type == EventType.ERROR
        assert event.error == "Something went wrong"
        assert event.is_error_event

    def test_to_dict(self):
        """Event.to_dict() should return dictionary representation."""
        event = AgentExecutionEvent(type=EventType.CONTENT, content="Hello")
        d = event.to_dict()
        assert d["type"] == "content"
        assert d["content"] == "Hello"
        assert "timestamp" in d


class TestStage:
    """Tests for Stage enum."""

    def test_all_stages_exist(self):
        """All expected stages should exist."""
        assert Stage.INITIAL.value == "initial"
        assert Stage.PLANNING.value == "planning"
        assert Stage.READING.value == "reading"
        assert Stage.ANALYSIS.value == "analysis"
        assert Stage.EXECUTION.value == "execution"
        assert Stage.VERIFICATION.value == "verification"
        assert Stage.COMPLETION.value == "completion"

    def test_stage_count(self):
        """Should have exactly 7 stages."""
        assert len(Stage) == 7


class TestTaskResult:
    """Tests for TaskResult dataclass."""

    def test_basic_result(self):
        """Basic TaskResult should work."""
        result = TaskResult(content="Hello", success=True)
        assert result.content == "Hello"
        assert result.success
        assert result.error is None
        assert result.tool_calls == []

    def test_files_modified_extraction(self):
        """files_modified should extract paths from write/edit tool calls."""
        result = TaskResult(
            content="Done",
            tool_calls=[
                {"tool": "write", "arguments": {"path": "/tmp/a.txt"}, "success": True},
                {"tool": "edit", "arguments": {"path": "/tmp/b.txt"}, "success": True},
                {"tool": "read", "arguments": {"path": "/tmp/c.txt"}, "success": True},
            ],
        )
        modified = result.files_modified
        assert "/tmp/a.txt" in modified
        assert "/tmp/b.txt" in modified
        assert "/tmp/c.txt" not in modified  # read doesn't modify

    def test_files_read_extraction(self):
        """files_read should extract paths from read tool calls."""
        result = TaskResult(
            content="Done",
            tool_calls=[
                {"tool": "read", "arguments": {"path": "/tmp/a.txt"}, "success": True},
                {"tool": "ls", "arguments": {"path": "/tmp"}, "success": True},
            ],
        )
        read = result.files_read
        assert "/tmp/a.txt" in read
        assert "/tmp" in read

    def test_files_read_and_modified_extract_legacy_aliases(self):
        """Legacy core tool aliases should resolve to canonical read/write semantics."""
        result = TaskResult(
            content="Done",
            tool_calls=[
                {
                    "tool": "write_file",
                    "arguments": {"path": "/tmp/a.txt"},
                    "success": True,
                },
                {
                    "tool": "read_file",
                    "arguments": {"path": "/tmp/b.txt"},
                    "success": True,
                },
                {
                    "tool": "list_directory",
                    "arguments": {"path": "/tmp"},
                    "success": True,
                },
            ],
        )

        assert "/tmp/a.txt" in result.files_modified
        assert "/tmp/b.txt" in result.files_read
        assert "/tmp" in result.files_read

    def test_tool_count(self):
        """tool_count should return number of tool calls."""
        result = TaskResult(
            content="Done",
            tool_calls=[{"tool": "read"}, {"tool": "write"}, {"tool": "edit"}],
        )
        assert result.tool_count == 3


class TestTask:
    """Tests for Task dataclass."""

    def test_basic_task(self):
        """Basic Task should work."""
        task = Task(prompt="Do something")
        assert task.prompt == "Do something"
        assert task.type == FrameworkTaskType.CHAT

    def test_task_with_type(self):
        """Task with type should work."""
        task = Task(prompt="Edit file", type=FrameworkTaskType.EDIT, files=["test.py"])
        assert task.type == FrameworkTaskType.EDIT
        assert task.files == ["test.py"]


class TestAgentConfig:
    """Tests for AgentConfig."""

    def test_default_config(self):
        """Default config should have sensible values."""
        config = AgentConfig.default()
        assert config.tool_budget == 50
        assert config.max_iterations == 25
        # Default OFF (co-design review item 16): the read-only allowlist
        # gate makes it sound to enable, but it still changes execution
        # ordering/timing for a broad swath of tool calls, so it's opt-in.
        assert not config.enable_parallel_tools

    def test_minimal_config(self):
        """Minimal config should have reduced budgets."""
        config = AgentConfig.minimal()
        assert config.tool_budget < AgentConfig.default().tool_budget
        assert not config.enable_analytics

    def test_high_budget_config(self):
        """High budget config should have increased limits."""
        config = AgentConfig.high_budget()
        assert config.tool_budget == 200
        assert config.max_iterations == 100

    def test_to_settings_dict(self):
        """to_settings_dict should return Settings-compatible dict."""
        config = AgentConfig(tool_budget=100)
        d = config.to_settings_dict()
        assert d["tool_call_budget"] == 100


class TestErrors:
    """Tests for framework error types."""

    def test_agent_error(self):
        """AgentError should have message and recoverable flag."""
        err = AgentError("Something failed", recoverable=True)
        assert "Something failed" in str(err)
        assert err.recoverable

    def test_provider_error(self):
        """ProviderError should include provider name."""
        err = ProviderError("API error", provider="anthropic", status_code=500)
        assert "anthropic" in str(err)
        assert err.status_code == 500

    def test_tool_error(self):
        """ToolError should include tool name."""
        err = ToolError("File not found", tool_name="read")
        assert "read" in str(err)

    def test_budget_exhausted_error(self):
        """BudgetExhaustedError should include budget info."""
        err = BudgetExhaustedError(budget=50, used=50)
        assert "50" in str(err)
        assert not err.recoverable

    def test_configuration_error(self):
        """ConfigurationError should be non-recoverable."""
        err = ConfigurationError("Invalid config", invalid_fields=["model"])
        assert not err.recoverable
        assert "model" in str(err)


class TestState:
    """Tests for State wrapper."""

    def test_state_with_mock_orchestrator(self):
        """State should wrap orchestrator protocol methods.

        Phase 7.2: Updated to use protocol methods instead of direct attribute access.
        """
        from unittest.mock import MagicMock, PropertyMock
        from victor.agent.conversation.state_machine import ConversationStage

        mock_orchestrator = MagicMock()
        # Configure protocol methods
        mock_orchestrator.get_tool_calls_count.return_value = 5
        mock_orchestrator.get_tool_budget.return_value = 50
        mock_orchestrator.get_message_count.return_value = 2
        mock_orchestrator.get_stage.return_value = ConversationStage.INITIAL
        mock_orchestrator.get_observed_files.return_value = set()
        mock_orchestrator.get_modified_files.return_value = set()
        mock_orchestrator.get_iteration_count.return_value = 0
        mock_orchestrator.get_max_iterations.return_value = 25
        mock_orchestrator.is_streaming.return_value = False
        type(mock_orchestrator).current_provider = PropertyMock(return_value="anthropic")
        type(mock_orchestrator).current_model = PropertyMock(return_value="claude-sonnet")

        state = State(mock_orchestrator)
        assert state.tool_calls_used == 5
        assert state.tool_budget == 50
        assert state.tools_remaining == 45
        assert state.message_count == 2

    def test_state_to_dict(self):
        """State.to_dict() should return dictionary representation.

        Phase 7.2: Updated to use protocol methods instead of direct attribute access.
        """
        from unittest.mock import MagicMock, PropertyMock
        from victor.agent.conversation.state_machine import ConversationStage

        mock_orchestrator = MagicMock()
        # Configure protocol methods
        mock_orchestrator.get_tool_calls_count.return_value = 0
        mock_orchestrator.get_tool_budget.return_value = 50
        mock_orchestrator.get_message_count.return_value = 0
        mock_orchestrator.get_stage.return_value = ConversationStage.INITIAL
        mock_orchestrator.get_observed_files.return_value = set()
        mock_orchestrator.get_modified_files.return_value = set()
        mock_orchestrator.get_iteration_count.return_value = 0
        mock_orchestrator.get_max_iterations.return_value = 25
        mock_orchestrator.is_streaming.return_value = False
        type(mock_orchestrator).current_provider = PropertyMock(return_value="anthropic")
        type(mock_orchestrator).current_model = PropertyMock(return_value="claude-sonnet")

        state = State(mock_orchestrator)
        d = state.to_dict()
        assert "stage" in d
        assert "tool_calls_used" in d
        assert "provider" in d


class TestStateHooks:
    """Tests for StateHooks dataclass."""

    def test_state_hooks_creation(self):
        """StateHooks should be creatable with callbacks."""
        on_enter = MagicMock()
        on_exit = MagicMock()
        on_transition = MagicMock()

        hooks = StateHooks(
            on_enter=on_enter,
            on_exit=on_exit,
            on_transition=on_transition,
        )
        assert hooks.on_enter is on_enter
        assert hooks.on_exit is on_exit
        assert hooks.on_transition is on_transition


class TestTopLevelImports:
    """Tests that top-level imports work."""

    def test_agent_available_at_top_level(self):
        """Agent should be importable from victor."""
        assert AgentTopLevel is Agent

    def test_toolset_available_at_top_level(self):
        """ToolSet should be importable from victor."""
        assert ToolSetTopLevel is ToolSet


class TestEventFactoryFunctions:
    """Tests for event factory functions."""

    def test_content_event_factory(self):
        """content_event() should create CONTENT event."""
        from victor.framework.events import content_event

        event = content_event("Hello")
        assert event.type == EventType.CONTENT
        assert event.content == "Hello"

    def test_thinking_event_factory(self):
        """thinking_event() should create THINKING event."""
        from victor.framework.events import thinking_event

        event = thinking_event("Reasoning...")
        assert event.type == EventType.THINKING
        assert event.content == "Reasoning..."

    def test_tool_call_event_factory(self):
        """tool_call_event() should create TOOL_CALL event."""
        from victor.framework.events import tool_call_event

        event = tool_call_event("read", {"path": "/tmp"}, tool_id="abc123")
        assert event.type == EventType.TOOL_CALL
        assert event.tool_name == "read"
        assert event.tool_id == "abc123"
        assert event.arguments == {"path": "/tmp"}

    def test_error_event_factory(self):
        """error_event() should create ERROR event."""
        from victor.framework.events import error_event

        event = error_event("Something failed", recoverable=False)
        assert event.type == EventType.ERROR
        assert event.error == "Something failed"
        assert not event.recoverable
        assert not event.success
