# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for LangChainAdapterTool and LangChainToolProjector."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from victor.tools.langchain_adapter_tool import (
    LangChainAdapterTool,
    LangChainToolProjector,
    _pydantic_to_json_schema,
)

# Try importing langchain_core for real tool tests
try:
    from langchain_core.tools import tool as lc_tool

    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False


# ── Mock-based tests (always run, no langchain dependency) ──


class FakeLCTool:
    """Lightweight mock matching LangChain BaseTool shape."""

    def __init__(self, name="search", description="Search the web", args_schema=None):
        self.name = name
        self.description = description
        self.args_schema = args_schema

    async def ainvoke(self, input_data, **kwargs):
        return f"result for {input_data}"


class TestPydanticToJsonSchema:
    def test_none_schema(self):
        schema = _pydantic_to_json_schema(None)
        assert schema == {"type": "object", "properties": {}}

    def test_dict_passthrough(self):
        d = {"type": "object", "properties": {"q": {"type": "string"}}}
        assert _pydantic_to_json_schema(d) is d

    def test_pydantic_v2_model(self):
        from pydantic import BaseModel

        class SearchInput(BaseModel):
            query: str
            limit: int = 10

        schema = _pydantic_to_json_schema(SearchInput)
        assert "properties" in schema
        assert "query" in schema["properties"]
        assert "limit" in schema["properties"]

    def test_unknown_type_returns_empty(self):
        schema = _pydantic_to_json_schema(42)
        assert schema == {"type": "object", "properties": {}}


class TestLangChainAdapterToolMock:
    def test_name_without_prefix(self):
        lc = FakeLCTool(name="search")
        adapter = LangChainAdapterTool(lc)
        assert adapter.name == "lgc_search"

    def test_name_with_prefix(self):
        lc = FakeLCTool(name="search")
        adapter = LangChainAdapterTool(lc, name_prefix="lc")
        assert adapter.name == "lc_search"

    def test_description_has_attribution(self):
        lc = FakeLCTool(description="Search the web")
        adapter = LangChainAdapterTool(lc)
        assert "Search the web" in adapter.description
        assert "LangChain" in adapter.description

    def test_langchain_tool_name_property(self):
        lc = FakeLCTool(name="search")
        adapter = LangChainAdapterTool(lc, name_prefix="lc")
        assert adapter.langchain_tool_name == "search"

    def test_parameters_from_none(self):
        lc = FakeLCTool(args_schema=None)
        adapter = LangChainAdapterTool(lc)
        assert adapter.parameters == {"type": "object", "properties": {}}

    @pytest.mark.asyncio
    async def test_execute_success(self):
        lc = FakeLCTool(name="search")
        adapter = LangChainAdapterTool(lc)
        result = await adapter.execute({}, query="python")
        assert result.success is True
        assert "result for" in str(result.output)
        assert result.metadata["source"] == "langchain"

    @pytest.mark.asyncio
    async def test_execute_failure(self):
        lc = FakeLCTool(name="broken")
        lc.ainvoke = AsyncMock(side_effect=RuntimeError("API down"))
        adapter = LangChainAdapterTool(lc)
        result = await adapter.execute({})
        assert result.success is False
        assert "API down" in result.error


class TestLangChainToolProjectorMock:
    def test_project_multiple_tools(self):
        tools = [FakeLCTool(name="search"), FakeLCTool(name="calc")]
        adapted = LangChainToolProjector.project(tools)
        assert len(adapted) == 2
        names = {t.name for t in adapted}
        assert names == {"lgc_search", "lgc_calc"}

    def test_project_with_prefix(self):
        tools = [FakeLCTool(name="search")]
        adapted = LangChainToolProjector.project(tools, prefix="lc")
        assert adapted[0].name == "lc_search"

    def test_project_collision_skip(self):
        tools = [FakeLCTool(name="search"), FakeLCTool(name="search")]
        adapted = LangChainToolProjector.project(tools, conflict_strategy="skip")
        assert len(adapted) == 1

    def test_project_collision_prefix_source(self):
        tools = [FakeLCTool(name="search"), FakeLCTool(name="search")]
        adapted = LangChainToolProjector.project(tools, conflict_strategy="prefix_source")
        assert len(adapted) == 2
        names = {t.name for t in adapted}
        assert len(names) == 2  # No collision

    def test_project_empty_list(self):
        assert LangChainToolProjector.project([]) == []


# ── Real LangChain tool tests (only run if langchain-core installed) ──


@pytest.mark.skipif(not HAS_LANGCHAIN, reason="langchain-core not installed")
class TestLangChainAdapterReal:
    def test_real_tool_name(self):
        @lc_tool
        def multiply(a: int, b: int) -> int:
            """Multiply two numbers."""
            return a * b

        adapter = LangChainAdapterTool(multiply)
        assert adapter.name == "lgc_multiply"
        assert "Multiply two numbers" in adapter.description

    def test_real_tool_schema(self):
        @lc_tool
        def multiply(a: int, b: int) -> int:
            """Multiply two numbers."""
            return a * b

        adapter = LangChainAdapterTool(multiply)
        params = adapter.parameters
        assert "properties" in params
        assert "a" in params["properties"]
        assert "b" in params["properties"]

    @pytest.mark.asyncio
    async def test_real_tool_execution(self):
        @lc_tool
        def multiply(a: int, b: int) -> int:
            """Multiply two numbers."""
            return a * b

        adapter = LangChainAdapterTool(multiply)
        result = await adapter.execute({}, a=3, b=7)
        assert result.success is True
        assert result.output == 21

    @pytest.mark.asyncio
    async def test_real_async_tool(self):
        @lc_tool
        async def async_add(x: int, y: int) -> int:
            """Add two numbers asynchronously."""
            return x + y

        adapter = LangChainAdapterTool(async_add)
        result = await adapter.execute({}, x=5, y=3)
        assert result.success is True
        assert result.output == 8
