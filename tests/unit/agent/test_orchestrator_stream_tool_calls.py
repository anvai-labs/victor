from pathlib import Path
from types import SimpleNamespace
from typing import Any, AsyncIterator, Dict, List, Optional
from unittest.mock import patch

import pytest

from victor.agent.orchestrator import AgentOrchestrator
from victor.config.settings import Settings
from victor.providers.base import BaseProvider, Message, StreamChunk, ToolDefinition
from victor.tools.base import BaseTool


class DummyStreamProvider(BaseProvider):
    """Provider stub that yields a single tool call chunk."""

    def __init__(self):
        super().__init__(api_key=None)
        self.stream_calls = 0

    @property
    def name(self) -> str:  # pragma: no cover - trivial
        return "dummy-stream"

    def supports_tools(self) -> bool:  # pragma: no cover - trivial
        return True

    def supports_streaming(self) -> bool:  # pragma: no cover - trivial
        return True

    async def chat(self, *args: Any, **kwargs: Any):  # pragma: no cover - not used
        raise NotImplementedError

    async def stream(
        self,
        messages: List[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        self.stream_calls += 1
        if self.stream_calls == 1:
            yield StreamChunk(
                content="",
                tool_calls=[{"name": "dummy_tool", "arguments": {"echo": "hi"}}],
                stop_reason="tool_calls",
                is_final=True,
            )
        else:
            yield StreamChunk(
                content="done",
                tool_calls=None,
                stop_reason="stop",
                is_final=True,
            )

    async def close(self) -> None:  # pragma: no cover - trivial
        return None


class DummyTool(BaseTool):
    """Tool that records invocations."""

    name = "dummy_tool"
    description = "A dummy tool"
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {"echo": {"type": "string", "description": "Value to echo back"}},
    }

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[Dict[str, Any]] = []

    async def execute(self, _exec_ctx: Dict[str, Any] = None, **kwargs: Any):
        self.calls.append(kwargs)
        return SimpleNamespace(success=True, output="ok", error=None)


@pytest.mark.asyncio
async def test_orchestrator_executes_streamed_tool_call(monkeypatch):
    """End-to-end: streamed tool_call chunk should trigger tool execution once."""
    settings = Settings(
        analytics_enabled=False,
        analytics_log_file=str(Path("tmp_usage.log")),
        tool_cache_dir_override=str(Path("tmp_cache")),
    )
    settings.tool_cache_enabled = False
    provider = DummyStreamProvider()

    with patch("victor.core.bootstrap_services.bootstrap_new_services"):
        orch = AgentOrchestrator(
            settings=settings,
            provider=provider,
            model="dummy-model",
            temperature=0.0,
            max_tokens=32,
        )

    dummy_tool = DummyTool()
    orch.tools.register(dummy_tool)
    # Update all components that reference the tools registry
    orch.tool_executor.tools = orch.tools
    orch._tool_pipeline.tools = orch.tools  # Pipeline's registry
    orch._tool_pipeline.executor.tools = orch.tools  # Pipeline's executor registry

    # Inject a mock _tool_service so is_tool_enabled can be configured.
    # bootstrap_new_services was patched out, so _tool_service is None on fresh orchestrators.
    from unittest.mock import MagicMock

    mock_tool_svc = MagicMock()
    orch._tool_service = mock_tool_svc
    enabled_names = {"dummy_tool"} | {t.name for t in orch.tools.list_tools()}
    mock_tool_svc.is_tool_enabled = lambda name: name in enabled_names

    # Bypass heavy semantic selector; return just our dummy tool definition
    orch._select_tools = lambda *args, **kwargs: [
        ToolDefinition(
            name="dummy_tool",
            description="d",
            parameters={"type": "object", "properties": {}},
        )
    ]

    # Consume provider stream manually to avoid semantic selector overhead
    chunks = []
    async for chunk in provider.stream(messages=[], model="dummy-model"):
        chunks.append(chunk)
        if chunk.tool_calls:
            await orch.execute_tool_calls(chunk.tool_calls)

    assert dummy_tool.calls == [{"echo": "hi"}]
    assert orch.tool_calls_used == 1
    assert any(c.tool_calls for c in chunks)
