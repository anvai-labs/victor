from unittest.mock import patch

import pytest

from victor.agent.orchestrator import AgentOrchestrator
from victor.config.settings import Settings
from victor.providers.base import BaseProvider
from victor.tools.base import BaseTool, ToolResult


class DummyProvider(BaseProvider):
    """Minimal provider stub for orchestrator unit tests."""

    def __init__(self):
        super().__init__(api_key=None)

    @property
    def name(self) -> str:  # pragma: no cover - trivial
        return "dummy"

    def supports_tools(self) -> bool:  # pragma: no cover - trivial
        return True

    def supports_streaming(self) -> bool:  # pragma: no cover - trivial
        return False

    async def chat(
        self, messages, *, model, temperature=0.7, max_tokens=4096, tools=None, **kwargs
    ):
        raise NotImplementedError

    async def stream(
        self, messages, *, model, temperature=0.7, max_tokens=4096, tools=None, **kwargs
    ):
        raise NotImplementedError

    async def close(self):  # pragma: no cover - trivial
        return None


class AlwaysFailTool(BaseTool):
    """Tool that always fails, capturing attempts."""

    name = "always_fail"
    description = "Always fails"
    parameters = {}

    def __init__(self):
        super().__init__()
        self.attempts = 0

    async def execute(self, _exec_ctx=None, **kwargs):
        self.attempts += 1
        return ToolResult(success=False, output=None, error="boom")


@pytest.mark.asyncio
async def test_repeated_failing_call_is_skipped_after_first_failure(monkeypatch, tmp_path):
    """Orchestrator should skip identical failing tool calls to break loops."""
    settings = Settings(
        analytics_enabled=False,
        analytics_log_file=str(tmp_path / "tmp_usage.log"),
        tool_cache_dir_override=str(tmp_path / "tmp_cache"),
    )
    settings.tool_cache_enabled = False
    provider = DummyProvider()

    # Minimal profile/model fields to construct orchestrator
    with patch("victor.core.bootstrap_services.bootstrap_new_services"):
        orch = AgentOrchestrator(
            settings=settings,
            provider=provider,
            model="dummy-model",
            temperature=0.0,
            max_tokens=10,
        )

    # Inject a failing tool into the orchestrator's existing tools registry
    failing_tool = AlwaysFailTool()
    orch.tools.register(failing_tool)
    # Update all components that reference the tools registry
    orch.tool_executor.tools = orch.tools
    orch._tool_pipeline.tools = orch.tools  # Pipeline's registry
    orch._tool_pipeline.executor.tools = orch.tools  # Pipeline's executor registry

    # Inject a mock _tool_service so is_tool_enabled can be configured.
    # bootstrap_new_services was patched out, so _tool_service is None on fresh orchestrators.
    # process_tool_results uses the real function to properly populate executed_tools.
    from unittest.mock import MagicMock
    from victor.agent.services.tool_service import process_tool_results_with_context

    mock_tool_svc = MagicMock()
    orch._tool_service = mock_tool_svc
    enabled_names = {"always_fail"} | {t.name for t in orch.tools.list_tools()}
    mock_tool_svc.is_tool_enabled = lambda name: name in enabled_names
    mock_tool_svc.process_tool_results = process_tool_results_with_context

    # Simulate a single tool call repeated twice
    tool_calls = [
        {"name": "always_fail", "arguments": {"foo": "bar"}},
        {"name": "always_fail", "arguments": {"foo": "bar"}},
    ]

    # Execute tool calls
    results = await orch.execute_tool_calls(tool_calls)

    # First call executed once (no retries for explicit ToolResult failures)
    # Second call skipped due to repeat signature (silent skip, no result added)
    # Note: ToolExecutor only retries on exceptions, not on explicit failures via ToolResult.success=False
    assert failing_tool.attempts == 1  # First call only, no retries for explicit failures
    assert orch.executed_tools.count("always_fail") == 1
    # Second call is silently skipped (no result added for repeated failures)
    assert len(results) == 1  # Only first call returns a result
    assert results[0]["success"] is False
