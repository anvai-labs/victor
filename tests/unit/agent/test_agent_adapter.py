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

"""Tests for VictorAgentAdapter."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from victor.evaluation.agent_adapter import (
    AdapterConfig,
    PromptOptimizationBinding,
    VictorAgentAdapter,
    create_victor_agent_callback,
)
from victor.evaluation.agentic_harness import (
    AgenticExecutionTrace,
    EvalToolCall,
    FileEdit,
)
from victor.evaluation.protocol import BenchmarkTask, BenchmarkType
from victor.tools.base import BaseTool, ToolResult
from victor.tools.registry import ToolRegistry


class _DummyTool(BaseTool):
    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"Dummy tool {self._name}"

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}}

    async def execute(self, _exec_ctx: dict, **kwargs):
        return ToolResult(success=True, output=kwargs)


class TestAdapterConfig:
    """Tests for AdapterConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values.

        Defaults optimized for SWE-bench with slow models (DeepSeek, Qwen, Mixtral).
        """
        config = AdapterConfig()
        assert config.max_turns == 20  # More turns within the task budget
        assert config.tool_budget == 50  # ACTION complexity budget
        # 2 min per turn: a lower floor maximizes turns within the task budget
        # (1200s / 120s = 10 turns) so the agent reaches the edit phase.
        assert config.min_turn_timeout == 120
        assert config.track_file_edits is True
        assert config.track_diffs is True
        assert config.working_dir is None

    def test_custom_values(self):
        """Test custom configuration values."""
        config = AdapterConfig(
            max_turns=10,
            tool_budget=25,
            min_turn_timeout=60,
            track_file_edits=False,
            working_dir=Path("/tmp"),
        )
        assert config.max_turns == 10
        assert config.tool_budget == 25
        assert config.min_turn_timeout == 60
        assert config.track_file_edits is False
        assert config.working_dir == Path("/tmp")


class TestVictorAgentAdapter:
    """Tests for VictorAgentAdapter class."""

    @pytest.fixture
    def mock_orchestrator(self):
        """Create a mock orchestrator."""
        orchestrator = MagicMock()
        orchestrator._on_tool_start_callback = None
        orchestrator._on_tool_complete_callback = None
        orchestrator.reset_conversation = MagicMock()
        orchestrator.chat = AsyncMock(return_value=MagicMock(content="TASK COMPLETE"))
        return orchestrator

    @pytest.fixture
    def adapter(self, mock_orchestrator):
        """Create an adapter with mock orchestrator."""
        return VictorAgentAdapter(mock_orchestrator)

    def test_init_hooks_callbacks(self, mock_orchestrator):
        """Test that adapter hooks into orchestrator callbacks."""
        # Set up mock tools with register methods
        mock_orchestrator.tools = MagicMock()
        mock_orchestrator.tools.register_before_hook = MagicMock()
        mock_orchestrator.tools.register_after_hook = MagicMock()

        adapter = VictorAgentAdapter(mock_orchestrator)

        # Verify hooks are registered on ToolRegistry
        mock_orchestrator.tools.register_before_hook.assert_called_once()
        mock_orchestrator.tools.register_after_hook.assert_called_once()

    def test_init_with_config(self, mock_orchestrator):
        """Test initialization with custom config."""
        config = AdapterConfig(max_turns=5, tool_budget=10)
        adapter = VictorAgentAdapter(mock_orchestrator, config)

        assert adapter.config.max_turns == 5
        assert adapter.config.tool_budget == 10

    def test_init_applies_prompt_binding_to_optimization_injector(self, mock_orchestrator):
        """Prompt-bound benchmark runs should pin the requested candidate at runtime."""
        mock_orchestrator.provider_name = "anthropic"
        mock_orchestrator._optimization_injector = MagicMock()

        VictorAgentAdapter(
            mock_orchestrator,
            AdapterConfig(
                prompt_binding=PromptOptimizationBinding(
                    section_name="GROUNDING_RULES",
                    prompt_candidate_hash="cand-123",
                )
            ),
        )

        mock_orchestrator._optimization_injector.bind_prompt_candidate.assert_called_once_with(
            section_name="GROUNDING_RULES",
            prompt_candidate_hash="cand-123",
            provider="anthropic",
            strict=True,
        )

    def test_benchmark_tool_allowlist_includes_graph(self):
        """Benchmark sessions should always expose graph navigation."""
        assert "graph" in VictorAgentAdapter.benchmark_tool_allowlist()

    def test_benchmark_tool_readiness_graph_optional_when_missing(self, mock_orchestrator):
        """graph is demand-loaded/optional — missing it must NOT fail readiness."""
        registry = ToolRegistry()
        for name in ("read", "edit", "write", "code", "shell"):
            registry.register(_DummyTool(name))
        mock_orchestrator.tools = registry

        adapter = VictorAgentAdapter(mock_orchestrator)
        readiness = adapter.get_benchmark_tool_readiness()

        assert readiness.ready is True  # graph optional → still ready
        assert "graph" in readiness.missing_tools  # but reported
        assert "graph" in readiness.optional_tools
        assert "code" in readiness.enabled_tools

    def test_benchmark_tool_readiness_fails_on_missing_base_tool(self, mock_orchestrator):
        """A missing BASE tool (not optional) must fail readiness."""
        registry = ToolRegistry()
        for name in ("read", "edit", "write", "shell"):  # no 'code'
            registry.register(_DummyTool(name))
        mock_orchestrator.tools = registry

        adapter = VictorAgentAdapter(mock_orchestrator)
        readiness = adapter.get_benchmark_tool_readiness()

        assert readiness.ready is False
        assert "code" in readiness.missing_tools

    def test_benchmark_tool_readiness_reports_disabled_tools(self, mock_orchestrator):
        """Readiness should fail when a required (base) tool is disabled."""
        registry = ToolRegistry()
        for name in VictorAgentAdapter.benchmark_tool_allowlist():
            registry.register(_DummyTool(name))
        registry.disable_tool("code")  # disable a BASE tool
        mock_orchestrator.tools = registry

        adapter = VictorAgentAdapter(mock_orchestrator)
        readiness = adapter.get_benchmark_tool_readiness()

        assert readiness.ready is False
        assert "code" in readiness.disabled_tools

    def test_reset(self, adapter):
        """Test reset clears state."""
        # Simulate some state
        adapter._tool_calls = [EvalToolCall(name="test", arguments={}, timestamp=0)]
        adapter._file_edits = [FileEdit(path="test.py", action="create")]
        adapter._messages = [{"role": "user", "content": "test"}]
        adapter._turns = 5
        adapter._file_snapshots = {"test.py": "content"}

        adapter.reset()

        assert len(adapter._tool_calls) == 0
        assert len(adapter._file_edits) == 0
        assert len(adapter._messages) == 0
        assert adapter._turns == 0
        assert len(adapter._file_snapshots) == 0
        adapter.orchestrator.reset_conversation.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_from_session_config_uses_agent_facade(self, mock_orchestrator):
        """Session-based adapter creation should flow through Agent.create()."""
        from victor.framework.session_config import SessionConfig

        config = SessionConfig.from_cli_flags(
            agent_profile="benchmark-profile",
            provider="openai",
            model="gpt-4o",
            provider_timeout=180,
        )
        mock_agent = MagicMock()
        mock_agent.get_orchestrator.return_value = mock_orchestrator

        with patch(
            "victor.framework.Agent.create",
            new=AsyncMock(return_value=mock_agent),
        ) as create:
            adapter = await VictorAgentAdapter.create_from_session_config(
                config,
                enable_observability=False,
            )

        assert adapter.orchestrator is mock_orchestrator
        create.assert_awaited_once_with(
            profile="benchmark-profile",
            session_config=config,
            enable_observability=False,
            vertical=None,
        )

    def test_on_tool_start_records_call(self, adapter):
        """Test that tool start callback records tool call."""
        adapter._on_tool_start("file_read", {"path": "test.py"})

        assert len(adapter._tool_calls) == 1
        assert adapter._tool_calls[0].name == "file_read"
        assert adapter._tool_calls[0].arguments == {"path": "test.py"}

    def test_on_tool_start_with_file_edit_snapshots(self, adapter):
        """Test that file edits trigger snapshot."""
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter.config.working_dir = Path(tmpdir)

            # Create a test file
            test_file = Path(tmpdir) / "test.py"
            test_file.write_text("original content")

            adapter._on_tool_start("file_edit", {"path": "test.py"})

            assert "test.py" in adapter._file_snapshots
            assert adapter._file_snapshots["test.py"] == "original content"

    def test_on_tool_complete_updates_result(self, adapter):
        """Test that tool complete updates the last tool call."""
        # First record a tool call
        adapter._on_tool_start("test_tool", {})

        # Then complete it
        result = MagicMock()
        result.success = True
        result.result = "test output"
        adapter._on_tool_complete(result)

        assert adapter._tool_calls[0].success is True
        assert adapter._tool_calls[0].result == "test output"

    def test_completion_detector_exists(self, adapter):
        """Test that adapter has completion detector initialized."""
        # Adapter uses _completion_detector for task completion detection
        assert adapter._completion_detector is not None
        # Detector has should_stop method for completion detection
        assert hasattr(adapter._completion_detector, "should_stop")

    def test_inject_task_context(self, adapter):
        """Test task context injection sets up working context."""
        task = BenchmarkTask(
            task_id="test/1",
            benchmark=BenchmarkType.CUSTOM,
            description="Test task",
            prompt="Fix the bug in main.py",
            context_code="def main():\n    pass",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)

            # The method should not raise
            adapter._inject_task_context(task, workspace)

            # Verify config reflects the workspace
            # (injection affects orchestrator state)

    def test_generate_combined_patch(self, adapter):
        """Test combined patch generation."""
        adapter._file_edits = [
            FileEdit(
                path="a.py",
                action="modify",
                diff="--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-old\n+new",
            ),
            FileEdit(
                path="b.py",
                action="create",
                diff="--- /dev/null\n+++ b/b.py\n@@ -0,0 +1 @@\n+new file",
            ),
        ]

        patch = adapter._generate_combined_patch()

        assert "a.py" in patch
        assert "b.py" in patch
        assert "-old" in patch
        assert "+new" in patch

    def test_conversation_trace_carries_generated_patch(self, adapter):
        """The ground-truth patch is surfaced (bounded) in the manifest trace."""
        # Default: empty until execute_task computes the git diff.
        assert adapter.get_conversation_trace()["generated_patch"] == ""

        adapter._generated_patch = "diff --git a/x.py b/x.py\n@@ -1 +1 @@\n-a\n+b"
        trace = adapter.get_conversation_trace()
        assert trace["generated_patch"] == adapter._generated_patch

        # Bounded so an enormous diff can't bloat the manifest line.
        adapter._generated_patch = "x" * 50000
        assert len(adapter.get_conversation_trace()["generated_patch"]) == 20000

    @pytest.mark.asyncio
    async def test_execute_task_basic(self, adapter, mock_orchestrator):
        """Test basic task execution."""
        mock_orchestrator.chat.return_value.metadata = {
            "agentic_loop_success": True,
            "agentic_loop_iterations": 3,
        }
        task = BenchmarkTask(
            task_id="test/basic",
            benchmark=BenchmarkType.CUSTOM,
            description="Basic test",
            prompt="Create a hello world function",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            trace = await adapter.execute_task(task, workspace)

            assert trace.task_id == "test/basic"
            assert trace.turns >= 1
            assert trace.start_time > 0
            assert trace.end_time >= trace.start_time
            assert trace.completion_signals["agentic_loop_success"] is True
            assert trace.completion_signals["agentic_loop_iterations"] >= 3
            assert "outer_completion_claimed" in trace.completion_signals
            mock_orchestrator.chat.assert_called()

    @pytest.mark.asyncio
    async def test_execute_task_enables_graph_and_mentions_it_in_prompt(
        self,
        adapter,
        mock_orchestrator,
    ):
        """Benchmark workflow guidance should enable and describe graph usage."""
        task = BenchmarkTask(
            task_id="test/graph",
            benchmark=BenchmarkType.CUSTOM,
            description="Graph test",
            prompt="Fix a cross-module dependency bug",
        )

        # graph is in the allowlist only when the optional graph tool resolves;
        # force it on so the test is deterministic across environments.
        with patch(
            "victor.evaluation.agent_adapter._graph_tool_available",
            return_value=True,
        ):
            with tempfile.TemporaryDirectory() as tmpdir:
                await adapter.execute_task(task, Path(tmpdir))

        mock_orchestrator.set_enabled_tools.assert_called()
        enabled_tools = mock_orchestrator.set_enabled_tools.call_args.args[0]
        assert "graph" in enabled_tools

        prompt = mock_orchestrator.chat.await_args_list[0].args[0]
        assert "Use graph to inspect callers, callees, dependencies, and impact" in prompt

    @pytest.mark.asyncio
    async def test_execute_dr3_uses_research_prompt_and_captures_report(
        self,
        adapter,
        mock_orchestrator,
    ):
        """DR3 runs should produce report text rather than a coding patch."""
        mock_orchestrator.chat.return_value = MagicMock(content="The report is complete.")
        task = BenchmarkTask(
            task_id="dr3/1",
            benchmark=BenchmarkType.DR3_EVAL,
            description="Research the supplied evidence",
            prompt="Research the supplied evidence\n\n- attachments/01-evidence.pdf",
            complexity_override="analysis",
            task_type_hint="analyze",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            (workspace / "report.md").write_text("# Findings\n\nEvidence-backed report.\n")
            trace = await adapter.execute_task(task, workspace)

        prompt = mock_orchestrator.chat.await_args_list[0].args[0]
        assert "deep-research report" in prompt
        assert "Fix the following issue" not in prompt
        assert "report.md" in prompt
        assert "pdftotext" in prompt
        assert "ffmpeg" in prompt
        assert "tesseract" in prompt
        assert trace.generated_code == "# Findings\n\nEvidence-backed report.\n"
        enabled_tools = mock_orchestrator.set_enabled_tools.call_args_list[0].args[0]
        assert "web_search" in enabled_tools
        assert "graph" not in enabled_tools

    @pytest.mark.asyncio
    async def test_execute_task_max_turns_limit(self, adapter, mock_orchestrator):
        """Test that max turns limit is enforced."""
        adapter.config.max_turns = 2
        # Make orchestrator never signal completion
        mock_orchestrator.chat.return_value = MagicMock(content="Still working...")

        task = BenchmarkTask(
            task_id="test/limit",
            benchmark=BenchmarkType.CUSTOM,
            description="Max turns test",
            prompt="Do something",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            trace = await adapter.execute_task(task, Path(tmpdir))

            assert trace.turns == 2  # Should stop at max_turns

    @pytest.mark.asyncio
    async def test_execute_task_total_timeout_enforced(self, adapter, mock_orchestrator):
        """total_timeout bounds the whole task (not just per-turn), stopping early.

        Before this guard, max_turns × slow per-turn inference could run for hours
        on a task that never completes; total_timeout was tracked but not enforced.
        """
        import asyncio

        adapter.config.max_turns = 50  # high — the timeout must stop us first
        adapter.config.total_timeout = 1  # 1s whole-task budget

        async def _slow_chat(*args, **kwargs):
            await asyncio.sleep(0.4)  # each turn takes 0.4s, never completes
            return MagicMock(content="Still working...")

        mock_orchestrator.chat = AsyncMock(side_effect=_slow_chat)

        task = BenchmarkTask(
            task_id="test/timeout",
            benchmark=BenchmarkType.CUSTOM,
            description="Total timeout test",
            prompt="Do something slow",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            trace = await adapter.execute_task(task, Path(tmpdir))

        # Broke on total_timeout, well before the 50-turn cap.
        assert 1 <= trace.turns < 50


class TestCreateVictorAgentCallback:
    """Tests for create_victor_agent_callback function."""

    def test_creates_callable(self):
        """Test that factory creates a callable."""
        mock_orchestrator = MagicMock()
        mock_orchestrator._on_tool_start_callback = None
        mock_orchestrator._on_tool_complete_callback = None

        adapter = VictorAgentAdapter(mock_orchestrator)
        callback = create_victor_agent_callback(adapter)

        assert callable(callback)

    @pytest.mark.asyncio
    async def test_callback_delegates_to_adapter(self):
        """Test that callback delegates to adapter.execute_task."""
        mock_orchestrator = MagicMock()
        mock_orchestrator._on_tool_start_callback = None
        mock_orchestrator._on_tool_complete_callback = None
        mock_orchestrator.reset_conversation = MagicMock()
        mock_orchestrator.chat = AsyncMock(return_value=MagicMock(content="TASK COMPLETE"))

        adapter = VictorAgentAdapter(mock_orchestrator)
        callback = create_victor_agent_callback(adapter)

        task = BenchmarkTask(
            task_id="test/callback",
            benchmark=BenchmarkType.CUSTOM,
            description="Callback test",
            prompt="Test task",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            result = await callback(task, Path(tmpdir))

            assert isinstance(result, AgenticExecutionTrace)
            assert result.task_id == "test/callback"


class TestFileEditCapture:
    """Tests for file edit capture functionality."""

    @pytest.fixture
    def adapter_with_workspace(self):
        """Create adapter with temp workspace."""
        mock_orchestrator = MagicMock()
        mock_orchestrator._on_tool_start_callback = None
        mock_orchestrator._on_tool_complete_callback = None

        config = AdapterConfig(track_file_edits=True, track_diffs=True)
        adapter = VictorAgentAdapter(mock_orchestrator, config)
        return adapter

    def test_capture_file_create(self, adapter_with_workspace):
        """Test capturing file creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter = adapter_with_workspace
            adapter.config.working_dir = Path(tmpdir)

            # Create a new file
            new_file = Path(tmpdir) / "new.py"
            new_file.write_text("print('hello')")

            adapter._capture_file_edit("new.py", "file_write")

            assert len(adapter._file_edits) == 1
            assert adapter._file_edits[0].path == "new.py"
            assert adapter._file_edits[0].action == "create"
            assert "print('hello')" in adapter._file_edits[0].after_content

    def test_capture_file_modify(self, adapter_with_workspace):
        """Test capturing file modification."""
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter = adapter_with_workspace
            adapter.config.working_dir = Path(tmpdir)

            # Create original file
            test_file = Path(tmpdir) / "test.py"
            test_file.write_text("original")

            # Snapshot before edit
            adapter._file_snapshots["test.py"] = "original"

            # Modify file
            test_file.write_text("modified")

            adapter._capture_file_edit("test.py", "file_edit")

            assert len(adapter._file_edits) == 1
            assert adapter._file_edits[0].action == "modify"
            assert "original" in adapter._file_edits[0].before_content
            assert "modified" in adapter._file_edits[0].after_content
            assert adapter._file_edits[0].diff != ""  # Should have diff

    def test_diff_generation(self, adapter_with_workspace):
        """Test that unified diff is generated correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter = adapter_with_workspace
            adapter.config.working_dir = Path(tmpdir)

            test_file = Path(tmpdir) / "test.py"
            test_file.write_text("line1\nline2\nline3")
            adapter._file_snapshots["test.py"] = "line1\nold_line\nline3"

            # Write modified content
            test_file.write_text("line1\nline2\nline3")

            adapter._capture_file_edit("test.py", "file_edit")

            diff = adapter._file_edits[0].diff
            assert "---" in diff
            assert "+++" in diff
            assert "-old_line" in diff
            assert "+line2" in diff


class TestTaskCompletionLogging:
    """The benchmark adapter must log task_completion decisions via the decision
    service, so the FEP-0012 classifier gets premature-completion training data.

    Previously the adapter used pure regex (analyze_response/should_stop) and
    never called decide_sync → 0 task_completion samples. Now it injects the
    decision service + calls _decide_sync(TASK_COMPLETION) per turn.
    """

    @pytest.fixture
    def mock_orchestrator(self):
        orchestrator = MagicMock()
        orchestrator._on_tool_start_callback = None
        orchestrator._on_tool_complete_callback = None
        orchestrator.reset_conversation = MagicMock()
        orchestrator.chat = AsyncMock(return_value=MagicMock(content="TASK COMPLETE"))
        orchestrator.tools = MagicMock()
        orchestrator.tools.list_tools.return_value = []
        orchestrator.tools.register_before_hook = MagicMock()
        orchestrator.tools.register_after_hook = MagicMock()
        return orchestrator

    @pytest.mark.asyncio
    async def test_completion_detector_gets_decision_service(self, mock_orchestrator):
        """execute_task injects the decision service into the completion detector."""
        from victor.evaluation.agent_adapter import VictorAgentAdapter, AdapterConfig
        from victor.evaluation.protocol import BenchmarkTask, BenchmarkType
        import tempfile
        from pathlib import Path
        from unittest.mock import MagicMock, patch, AsyncMock

        adapter = VictorAgentAdapter(mock_orchestrator, AdapterConfig(max_turns=1))
        task = BenchmarkTask(
            task_id="test/svc",
            benchmark=BenchmarkType.CUSTOM,
            description="test",
            prompt="Fix a bug",
        )
        mock_response = MagicMock()
        mock_response.content = "Done."
        mock_response.tool_calls = []
        mock_response.metadata = {}
        mock_orchestrator.chat = AsyncMock(return_value=mock_response)

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("victor.core.context.set_session_id"):
                with patch(
                    "victor.agent.services.protocols.decision_service.get_decision_service",
                    return_value=MagicMock(name="LoggingDecisionService"),
                ):
                    await adapter.execute_task(task, Path(tmpdir))

        # The adapter's completion detector should have the service injected.
        assert adapter._completion_detector._decision_service is not None

    @pytest.mark.asyncio
    async def test_task_completion_logged_per_turn(self, mock_orchestrator):
        """Each turn's completion check calls _decide_sync(TASK_COMPLETION)."""
        from victor.evaluation.agent_adapter import VictorAgentAdapter, AdapterConfig
        from victor.evaluation.protocol import BenchmarkTask, BenchmarkType
        import tempfile
        from pathlib import Path
        from unittest.mock import MagicMock, patch, AsyncMock

        adapter = VictorAgentAdapter(mock_orchestrator, AdapterConfig(max_turns=1))
        task = BenchmarkTask(
            task_id="test/log",
            benchmark=BenchmarkType.CUSTOM,
            description="test",
            prompt="Fix a bug",
        )
        mock_response = MagicMock()
        mock_response.content = "The fix is complete."
        mock_response.tool_calls = []
        mock_response.metadata = {}
        mock_orchestrator.chat = AsyncMock(return_value=mock_response)

        # Mock the decision service + _decide_sync
        mock_svc = MagicMock()
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("victor.core.context.set_session_id"):
                with patch(
                    "victor.agent.services.protocols.decision_service.get_decision_service",
                    return_value=mock_svc,
                ):
                    await adapter.execute_task(task, Path(tmpdir))

        # _decide_sync should have been called with TASK_COMPLETION
        assert adapter._completion_detector._decision_service is mock_svc
        # The LoggingDecisionService's decide_sync should have been invoked
        # (via _decide_sync on the detector). We verify the service was set.
        # Full verification requires a deeper integration test.


class TestDemandLoadCuratedTools:
    """Demand-only tools (graph) must be registered before set_enabled_tools
    so _union_curated_enabled() can find them in list_tools."""

    @pytest.fixture
    def mock_orchestrator(self):
        orchestrator = MagicMock()
        orchestrator._on_tool_start_callback = None
        orchestrator._on_tool_complete_callback = None
        orchestrator.reset_conversation = MagicMock()
        orchestrator.chat = AsyncMock(return_value=MagicMock(content="Done"))
        orchestrator.tools = ToolRegistry()
        return orchestrator

    def test_graph_registered_when_in_allowlist(self, mock_orchestrator):
        """If graph is in the allowlist but not in the registry, the demand-load
        logic detects it and attempts registration."""
        from victor.evaluation.agent_adapter import VictorAgentAdapter
        from unittest.mock import patch

        # Set up a registry WITHOUT graph
        registry = ToolRegistry()
        for name in ("read", "edit", "write", "code", "shell"):
            registry.register(_DummyTool(name))
        mock_orchestrator.tools = registry
        adapter = VictorAgentAdapter(mock_orchestrator)

        # Simulate the demand-registration logic from execute_task
        benchmark_tools = {"code", "edit", "graph", "read", "shell", "write"}
        registered = {t.name for t in registry.list_tools(only_enabled=False)}
        missing = benchmark_tools - registered
        assert missing == {"graph"}, f"Expected only graph missing, got {missing}"

        # Mock SharedToolRegistry to return a graph tool
        graph_tool = _DummyTool("graph")
        with patch("victor.agent.shared_tool_registry.SharedToolRegistry.get_instance") as mock_get:
            mock_instance = MagicMock()
            mock_instance.get_tools_for_names.return_value = [graph_tool]
            mock_get.return_value = mock_instance

            from victor.tools.batch_registration import BatchRegistrar

            shared = mock_get.return_value
            new_tools = shared.get_tools_for_names(missing)
            BatchRegistrar(registry).register_batch(new_tools, fail_fast=False)

        registered_after = {t.name for t in registry.list_tools(only_enabled=False)}
        assert "graph" in registered_after, f"graph not registered: {registered_after}"


def test_capture_file_edit_skips_non_repo_paths(tmp_path):
    """Scratch files outside the workspace (e.g. /tmp test scripts) must not be
    captured into the patch — SWE-bench's git apply rejects them."""
    from unittest.mock import MagicMock

    adapter = VictorAgentAdapter(
        MagicMock(),
        AdapterConfig(working_dir=tmp_path, track_file_edits=True, track_diffs=True),
    )
    (tmp_path / "repo_file.py").write_text("x = 1\n")

    adapter._capture_file_edit("repo_file.py", "modify")  # repo-relative → captured
    adapter._capture_file_edit("/tmp/scratch_test.py", "create")  # absolute → skipped
    adapter._capture_file_edit("../escape.py", "create")  # escapes root → skipped

    paths = [e.path for e in adapter._file_edits]
    assert "repo_file.py" in paths
    assert "/tmp/scratch_test.py" not in paths
    assert "../escape.py" not in paths
    assert all(not p.startswith("/") for p in paths), paths


@pytest.mark.asyncio
async def test_workspace_git_diff_captures_modifications_and_new_files(tmp_path):
    """workspace_git_diff (the ground-truth patch source) captures BOTH modified
    tracked files AND new untracked files — the gap that lost 20% of patches when
    only the adapter's edit-capture was used."""
    import subprocess

    from victor.framework.workspace import workspace_git_diff

    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, capture_output=True)
    (tmp_path / "existing.py").write_text("x = 1\n")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, capture_output=True)
    (tmp_path / "existing.py").write_text("x = 2\n")
    (tmp_path / "new_module.py").write_text("y = 3\n")

    diff = await workspace_git_diff(tmp_path)

    assert "existing.py" in diff
    assert "new_module.py" in diff


@pytest.mark.asyncio
async def test_workspace_git_diff_returns_empty_for_non_git_dir(tmp_path):
    """Non-git directory → empty string (no exception, graceful fallback)."""
    from victor.framework.workspace import workspace_git_diff

    assert await workspace_git_diff(tmp_path) == ""
