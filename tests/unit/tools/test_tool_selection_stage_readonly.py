# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from victor.agent.conversation.state_machine import (
    ConversationStage,
    ConversationStateMachine,
)
from victor.agent.tool_selection import ToolSelector
from victor.tools.base import (
    AccessMode,
    BaseTool,
    DangerLevel,
    ExecutionCategory,
    Priority,
    ToolMetadata,
    ToolResult,
)
from victor.tools.registry import ToolRegistry
from victor.tools.semantic_selector import SemanticToolSelector
from victor.tools import metadata_registry


@pytest.fixture(autouse=True)
def reset_global_registry():
    """Reset global metadata registry between tests to avoid cross-contamination."""
    metadata_registry._global_registry = None
    yield


@pytest.fixture(autouse=True)
def mock_non_build_mode():
    """Mock mode controller to return allow_all_tools=False for stage filtering tests.

    Stage filtering tests need the mode to be non-BUILD (i.e., allow_all_tools=False)
    so that stage-based tool filtering actually happens. Without this, the global
    mode controller would return is_build_mode=True and skip filtering.
    """
    mock_controller = MagicMock()
    mock_controller.current_mode.value = "PLAN"  # Non-BUILD mode
    mock_controller.config.allow_all_tools = False  # Enable stage filtering
    mock_controller.config.exploration_multiplier = 1.0
    mock_controller.config.sandbox_dir = None
    mock_controller.config.allowed_tools = set()
    mock_controller.config.disallowed_tools = set()

    with patch("victor.agent.mode_controller.get_mode_controller", return_value=mock_controller):
        yield mock_controller
    metadata_registry._global_registry = None


class StubTool(BaseTool):
    """Minimal tool with explicit metadata for deterministic selection."""

    def __init__(
        self,
        name: str,
        *,
        category: str = "core",
        access_mode: AccessMode = AccessMode.READONLY,
        execution_category: ExecutionCategory = ExecutionCategory.READ_ONLY,
        priority: Priority = Priority.CRITICAL,
        stages: list[str] | None = None,
        keywords: list[str] | None = None,
    ) -> None:
        self._name = name
        self._metadata = ToolMetadata(
            category=category,
            priority=priority,
            access_mode=access_mode,
            danger_level=None,  # default SAFE
            stages=stages or [],
            execution_category=execution_category,
        )
        # Attributes consumed by ToolMetadataEntry.from_tool
        self.category = category
        self.priority = priority
        self.access_mode = access_mode
        self.danger_level = DangerLevel.SAFE
        self.stages = stages or []
        self.execution_category = execution_category
        self.keywords = keywords or []

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"{self._name} description"

    @property
    def parameters(self):
        return {"type": "object", "properties": {}}

    @property
    def metadata(self) -> ToolMetadata:
        return self._metadata

    @property
    def is_critical(self) -> bool:
        return self._metadata.priority == Priority.CRITICAL

    async def execute(self, _exec_ctx, **kwargs):  # pragma: no cover - not exercised here
        return ToolResult(success=True, output=None)


def _make_registry(*tools: BaseTool) -> ToolRegistry:
    reg = ToolRegistry()
    for tool in tools:
        reg.register(tool)
    return reg


def test_analysis_stage_includes_only_readonly_core():
    """Analysis stage should inject read-only core tools and omit execute/write core tools."""
    readonly_core = StubTool(
        "read_ro",
        category="core",
        access_mode=AccessMode.READONLY,
        execution_category=ExecutionCategory.READ_ONLY,
    )
    execute_core = StubTool(
        "shell_exec",
        category="core",
        access_mode=AccessMode.EXECUTE,
        execution_category=ExecutionCategory.EXECUTE,
    )
    # Non-core write tool that would be keyword-matched without filtering
    write_tool = StubTool(
        "write_any",
        category="fs",
        access_mode=AccessMode.WRITE,
        execution_category=ExecutionCategory.WRITE,
        keywords=["analyze"],
        priority=Priority.MEDIUM,
    )
    metadata_registry.register_tool_metadata(readonly_core)
    metadata_registry.register_tool_metadata(execute_core)
    metadata_registry.register_tool_metadata(write_tool)
    assert "read_ro" in metadata_registry.get_core_readonly_tools()
    tools = _make_registry(readonly_core, execute_core, write_tool)

    state = ConversationStateMachine()
    state.state.stage = ConversationStage.ANALYSIS

    selector = ToolSelector(
        tools,
        conversation_state=state,
        semantic_selector=None,
        model="test-model",
        provider_name="test-provider",
    )

    selected = selector.select_keywords("analyze the repo structure")
    names = {t.name for t in selected}

    assert "read_ro" in names
    # Write tools are now kept but annotated with HITL prompt
    assert "write_any" in names
    # Verify HITL annotation on write tool
    write_tool = next(t for t in selected if t.name == "write_any")
    assert "[HITL]" in write_tool.description


def test_execution_stage_includes_full_core():
    """Execution stage should include the full critical core set (read-only + execute)."""
    readonly_core = StubTool(
        "read_ro",
        category="core",
        access_mode=AccessMode.READONLY,
        execution_category=ExecutionCategory.READ_ONLY,
    )
    execute_core = StubTool(
        "shell_exec",
        category="core",
        access_mode=AccessMode.EXECUTE,
        execution_category=ExecutionCategory.EXECUTE,
    )
    metadata_registry.register_tool_metadata(readonly_core)
    metadata_registry.register_tool_metadata(execute_core)
    assert "read_ro" in metadata_registry.get_core_readonly_tools()
    tools = _make_registry(readonly_core, execute_core)

    state = ConversationStateMachine()
    state.state.stage = ConversationStage.EXECUTION

    selector = ToolSelector(
        tools,
        conversation_state=state,
        semantic_selector=None,
        model="test-model",
        provider_name="test-provider",
    )

    selected = selector.select_keywords("apply changes")
    names = {t.name for t in selected}

    assert {"read_ro", "shell_exec"} <= names


def test_core_readonly_config_override(monkeypatch):
    """Configured core_readonly_tools should extend the curated list."""
    # pydantic-settings expects JSON list format for List[str] fields
    monkeypatch.setenv("CORE_READONLY_TOOLS", '["custom_ro", "extra_ro"]')
    metadata_registry._global_registry = None
    registry = metadata_registry.get_global_registry()
    core = registry.get_core_readonly_tools()
    assert "custom_ro" in core
    assert "extra_ro" in core


@pytest.mark.asyncio
async def test_semantic_selection_for_analysis_forces_readonly_core(monkeypatch, tmp_path):
    """Semantic selector should always inject core read-only tools for analysis queries."""
    # Avoid real embeddings by stubbing _get_embedding
    monkeypatch.setattr(
        SemanticToolSelector,
        "_get_embedding",
        AsyncMock(return_value=np.zeros(4)),
    )

    readonly_core = StubTool(
        "read_ro",
        category="core",
        access_mode=AccessMode.READONLY,
        execution_category=ExecutionCategory.READ_ONLY,
    )
    execute_core = StubTool(
        "shell_exec",
        category="core",
        access_mode=AccessMode.EXECUTE,
        execution_category=ExecutionCategory.EXECUTE,
    )
    metadata_registry.register_tool_metadata(readonly_core)
    metadata_registry.register_tool_metadata(execute_core)
    assert "read_ro" in metadata_registry.get_core_readonly_tools()
    tools = _make_registry(readonly_core, execute_core)

    selector = SemanticToolSelector(cache_embeddings=False, cache_dir=tmp_path)
    # Preload tool embeddings (uses stubbed _get_embedding)
    await selector.initialize_tool_embeddings(tools)

    selected = await selector.select_relevant_tools_with_context(
        user_message="please analyze this codebase",
        tools=tools,
        conversation_history=None,
        max_tools=5,
        similarity_threshold=0.0,  # accept all similarities
    )
    names = {t.name for t in selected}

    assert "read_ro" in names  # injected as mandatory
    assert "shell_exec" not in names  # not read-only, so not forced in analysis
