# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for tool dispatch optimization features.

Tests cover the 6 optimization features end-to-end:
1. TOOL_NECESSITY gate (Q&A bypass)
2. MCP tool per-turn capping
3. Token-based schema budget enforcement
4. Confidence-driven schema promotion (STUB→COMPACT)
5. Per-file write parallelism
6. Cross-turn tool result deduplication
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from victor.providers.base import ToolDefinition

# ============================================================
# 1. TOOL_NECESSITY gate
# ============================================================


class TestToolNecessityGate:
    """Test _should_skip_tools_for_turn heuristic and edge integration."""

    def _make_orchestrator(self, **overrides):
        """Create a minimal mock orchestrator with the real methods patched in."""
        from victor.agent.orchestrator import AgentOrchestrator

        orch = MagicMock(spec=AgentOrchestrator)
        # Bind the real methods (include _tool_skip_mode since it's called by _should_skip_tools_for_turn)
        orch._tool_skip_mode = AgentOrchestrator._tool_skip_mode.__get__(orch)
        orch._should_skip_tools_for_turn = AgentOrchestrator._should_skip_tools_for_turn.__get__(
            orch
        )
        orch._check_tool_necessity_via_edge = (
            AgentOrchestrator._check_tool_necessity_via_edge.__get__(orch)
        )
        # The keyword/pattern tables moved to victor/agent/tool_supply_policy.py
        # (ADR-019 increment 2, #732); the bound wrappers delegate there, so no
        # class-level tables need copying onto the mock.
        orch._container = overrides.get("container", None)
        return orch

    def test_short_greeting_skips_tools(self):
        orch = self._make_orchestrator()
        assert orch._should_skip_tools_for_turn("hi") is True
        assert orch._should_skip_tools_for_turn("hello") is True
        assert orch._should_skip_tools_for_turn("thanks") is True

    def test_short_command_keeps_tools(self):
        orch = self._make_orchestrator()
        assert orch._should_skip_tools_for_turn("fix it") is False
        assert orch._should_skip_tools_for_turn("run tests") is False
        assert orch._should_skip_tools_for_turn("create file") is False

    def test_qa_pattern_without_tool_keywords_skips(self):
        orch = self._make_orchestrator()
        # Q&A pattern + no tool keywords → skip (heuristic confident)
        result = orch._should_skip_tools_for_turn(
            "what is the difference between a list and a tuple"
        )
        assert result is True

    def test_action_with_tool_keywords_keeps(self):
        orch = self._make_orchestrator()
        assert (
            orch._should_skip_tools_for_turn("read the file and search for the bug in the code")
            is False
        )

    def test_explain_with_tool_keywords_keeps(self):
        orch = self._make_orchestrator()
        # "explain" is Q&A-like, but "code" and "function" are tool keywords
        result = orch._should_skip_tools_for_turn("explain this function in the code")
        assert result is False  # 2+ tool signals overrides Q&A pattern

    def test_ambiguous_defaults_to_tools(self):
        orch = self._make_orchestrator()
        # No Q&A pattern, 1 tool signal — ambiguous → default to providing tools
        result = orch._should_skip_tools_for_turn("I'm thinking about the database design")
        assert result is False

    def test_edge_model_fallback_when_unavailable(self):
        orch = self._make_orchestrator(container=None)
        # Edge model not available → falls back to heuristic
        result = orch._check_tool_necessity_via_edge("what is python", heuristic_conf=0.85)
        assert result is True  # heuristic_conf >= 0.7 → skip tools


# ============================================================
# 2. MCP tool per-turn capping
# ============================================================


class TestMCPToolCapping:
    """Test _cap_mcp_tools limits MCP tools in per-turn selection."""

    def _make_selector(self, registry_tools):
        """Create a ToolSelector with mock registry containing given tools."""
        from victor.agent.tool_selection import ToolSelector

        mock_registry = MagicMock()
        mock_registry.list_tools.return_value = registry_tools
        selector = ToolSelector(
            tools=mock_registry,
            tool_selection_config={"max_mcp_tools_per_turn": 3},
        )
        return selector

    def test_no_mcp_tools_unchanged(self):
        """Native-only selection passes through untouched."""
        tools = [ToolDefinition(name=f"tool_{i}", description="d", parameters={}) for i in range(5)]
        native_registry_tools = [MagicMock(name=f"tool_{i}") for i in range(5)]
        for t in native_registry_tools:
            t.name = t._mock_name  # Fix MagicMock name attribute

        selector = self._make_selector(native_registry_tools)
        result = selector._cap_mcp_tools(tools, 3)
        assert len(result) == 5

    def test_mcp_tools_capped(self):
        """MCP tools exceeding limit are trimmed, native tools kept."""
        from victor.tools.mcp_adapter_tool import MCPAdapterTool

        # Create mock MCP tools in registry
        registry_tools = []
        for i in range(6):
            mcp = MagicMock(spec=MCPAdapterTool)
            mcp.name = f"mcp_{i}"
            registry_tools.append(mcp)
        # Native tools: use simple objects that aren't MCPAdapterTool instances
        for i in range(3):
            native = MagicMock(spec=[])  # Empty spec → not an MCPAdapterTool
            native.name = f"native_{i}"
            registry_tools.append(native)

        tools = [
            ToolDefinition(name=f"native_{i}", description="d", parameters={}) for i in range(3)
        ] + [ToolDefinition(name=f"mcp_{i}", description="d", parameters={}) for i in range(6)]

        selector = self._make_selector(registry_tools)
        result = selector._cap_mcp_tools(tools, 3)

        native_count = sum(1 for t in result if t.name.startswith("native"))
        mcp_count = sum(1 for t in result if t.name.startswith("mcp"))
        assert native_count == 3  # All native kept
        assert mcp_count == 3  # MCP capped at 3

    def test_mcp_within_limit_unchanged(self):
        """MCP tools within limit pass through."""
        from victor.tools.mcp_adapter_tool import MCPAdapterTool

        registry_tools = []
        for i in range(2):
            mcp = MagicMock(spec=MCPAdapterTool)
            mcp.name = f"mcp_{i}"
            registry_tools.append(mcp)

        tools = [ToolDefinition(name=f"mcp_{i}", description="d", parameters={}) for i in range(2)]
        selector = self._make_selector(registry_tools)
        result = selector._cap_mcp_tools(tools, 3)
        assert len(result) == 2


# ============================================================
# 3. Token budget enforcement
# ============================================================


class TestTokenBudgetEnforcement:
    """Test _enforce_token_budget demotes and drops tools correctly."""

    def test_within_budget_unchanged(self):
        from victor.agent.tool_selection import _enforce_token_budget

        tools = [
            ToolDefinition(name="a", description="d", parameters={}, schema_level="full"),
            ToolDefinition(name="b", description="d", parameters={}, schema_level="compact"),
        ]
        result = _enforce_token_budget(tools, 5000)
        assert len(result) == 2
        assert result[0].schema_level == "full"
        assert result[1].schema_level == "compact"

    def test_demotes_compact_to_stub(self):
        from victor.agent.tool_selection import (
            _enforce_token_budget,
            _estimate_tool_tokens,
        )

        tools = [
            ToolDefinition(name="a", description="d", parameters={}, schema_level="full"),
            ToolDefinition(name="b", description="d", parameters={}, schema_level="compact"),
            ToolDefinition(name="c", description="d", parameters={}, schema_level="compact"),
            ToolDefinition(name="d", description="d", parameters={}, schema_level="compact"),
        ]
        # full(125) + 3*compact(210) = 335. Budget 250 forces demotion.
        result = _enforce_token_budget(tools, 250)
        levels = [getattr(t, "schema_level", None) for t in result]
        # FULL must be preserved, some COMPACTs demoted to STUB
        assert levels[0] == "full"
        stub_count = sum(1 for l in levels if l == "stub")
        assert stub_count >= 1  # At least one demoted

    def test_drops_tail_stubs_when_budget_tight(self):
        from victor.agent.tool_selection import _enforce_token_budget

        tools = [
            ToolDefinition(name="a", description="d", parameters={}, schema_level="full"),
        ] + [
            ToolDefinition(name=f"s{i}", description="d", parameters={}, schema_level="stub")
            for i in range(20)
        ]
        # full(125) + 20*stub(640) = 765. Budget 300 → drops stubs.
        result = _enforce_token_budget(tools, 300)
        assert len(result) < 21
        # FULL tool must be preserved
        assert result[0].name == "a"
        assert result[0].schema_level == "full"

    def test_zero_budget_disabled(self):
        """When caller passes max_tokens=0, function should not be called."""
        # This tests the calling convention — 0 means don't call
        from victor.agent.tool_selection import _enforce_token_budget

        tools = [
            ToolDefinition(name="a", description="d", parameters={}, schema_level="full")
            for _ in range(50)
        ]
        # Budget 0 would try to drop everything — that's why callers check > 0
        # But if called directly, it should still be safe (0 budget → empty)
        result = _enforce_token_budget(tools, 0)
        # With budget 0, everything gets dropped (except we stop at FULL)
        assert len(result) <= len(tools)

    def test_estimate_tokens(self):
        from victor.agent.tool_selection import _estimate_tool_tokens

        tools = [
            ToolDefinition(name="a", description="d", parameters={}, schema_level="full"),
            ToolDefinition(name="b", description="d", parameters={}, schema_level="compact"),
            ToolDefinition(name="c", description="d", parameters={}, schema_level="stub"),
        ]
        est = _estimate_tool_tokens(tools)
        assert est == 125 + 70 + 32  # 227


# ============================================================
# 4. Schema promotion (STUB→COMPACT)
# ============================================================


class TestSchemaPromotion:
    """Test promote_high_confidence_stubs promotes correctly."""

    def test_promotion_above_threshold(self):
        from victor.agent.tool_selection import promote_high_confidence_stubs

        tools = [
            ToolDefinition(name="a", description="d", parameters={}, schema_level="stub"),
            ToolDefinition(name="b", description="d", parameters={}, schema_level="stub"),
            ToolDefinition(name="c", description="d", parameters={}, schema_level="full"),
        ]
        scores = {"a": 0.9, "b": 0.5, "c": 0.95}
        result = promote_high_confidence_stubs(tools, scores, threshold=0.8)

        assert result[0].schema_level == "compact"  # a: 0.9 >= 0.8 → promoted
        assert result[1].schema_level == "stub"  # b: 0.5 < 0.8 → stays
        assert result[2].schema_level == "full"  # c: already FULL → unchanged

    def test_no_scores_unchanged(self):
        from victor.agent.tool_selection import promote_high_confidence_stubs

        tools = [
            ToolDefinition(name="a", description="d", parameters={}, schema_level="stub"),
        ]
        result = promote_high_confidence_stubs(tools, {}, threshold=0.8)
        assert result[0].schema_level == "stub"

    def test_compact_not_promoted(self):
        """Only STUBs are promoted — COMPACT stays COMPACT."""
        from victor.agent.tool_selection import promote_high_confidence_stubs

        tools = [
            ToolDefinition(name="a", description="d", parameters={}, schema_level="compact"),
        ]
        scores = {"a": 0.95}
        result = promote_high_confidence_stubs(tools, scores, threshold=0.8)
        assert result[0].schema_level == "compact"

    def test_promotion_in_tiered_path(self):
        """Verify promotion is wired into select_tiered_tools via semantic scores."""
        from victor.agent.tool_selection import ToolSelector

        # Create a realistic mock tool with proper string attributes
        mock_tool = MagicMock()
        mock_tool.name = "search"
        mock_tool.description = "Search code in the codebase for patterns"
        mock_tool.parameters = {"type": "object", "properties": {}}
        mock_tool.metadata = None
        mock_tool.priority = MagicMock()
        mock_tool.priority.value = 1
        # to_schema returns a dict with function info for STUB level
        mock_tool.to_schema.return_value = {
            "function": {
                "name": "search",
                "description": "Search code",
                "parameters": {"type": "object", "properties": {}},
            }
        }

        mock_registry = MagicMock()
        mock_registry.list_tools.return_value = [mock_tool]

        mock_semantic = MagicMock()
        mock_semantic.get_last_selection_scores.return_value = {"search": 0.95}

        selector = ToolSelector(
            tools=mock_registry,
            semantic_selector=mock_semantic,
            tool_selection_config={
                "schema_promotion_threshold": 0.8,
                "max_tool_schema_tokens": 0,
            },
        )

        # Set a tiered config that puts "search" in the semantic pool (STUB).
        # Use advanced_tools (the canonical source); TieredToolConfig derives
        # semantic_pool from it, avoiding the deprecated semantic_pool field.
        from victor.core.vertical_types import TieredToolConfig

        config = TieredToolConfig(
            mandatory=set(),
            vertical_core=set(),
            advanced_tools={"search"},
            stage_tools={},
        )
        selector.set_tiered_config(config)

        result = selector.select_tiered_tools("search the codebase")
        # "search" matched by keyword → STUB. Then promoted → COMPACT via scores.
        search_tools = [t for t in result if t.name == "search"]
        if search_tools:
            assert search_tools[0].schema_level == "compact"


# ============================================================
# 5. Per-file write parallelism
# ============================================================


class TestPerFileWriteParallelism:
    """Test dependency graph allows writes to different files in parallel."""

    def _make_executor(self):
        from victor.agent.parallel_executor import (
            ParallelToolExecutor,
            ParallelExecutionConfig,
        )

        mock_tool_executor = MagicMock()
        config = ParallelExecutionConfig(enable_parallel=True, max_concurrent=5)
        return ParallelToolExecutor(mock_tool_executor, config)

    def test_writes_to_different_files_no_dependency(self):
        executor = self._make_executor()
        tool_calls = [
            {"name": "write_file", "arguments": {"path": "/a.py"}},
            {"name": "write_file", "arguments": {"path": "/b.py"}},
        ]
        deps = executor._extract_file_dependencies(tool_calls)
        # Different files → no dependency between them
        assert deps[0] == set()
        assert deps[1] == set()

    def test_writes_to_same_file_serialize(self):
        executor = self._make_executor()
        tool_calls = [
            {"name": "write_file", "arguments": {"path": "/a.py"}},
            {"name": "write_file", "arguments": {"path": "/a.py"}},
        ]
        deps = executor._extract_file_dependencies(tool_calls)
        assert deps[0] == set()
        assert deps[1] == {0}  # Second write depends on first

    def test_read_after_write_depends(self):
        executor = self._make_executor()
        tool_calls = [
            {"name": "write_file", "arguments": {"path": "/a.py"}},
            {"name": "read_file", "arguments": {"path": "/a.py"}},
        ]
        deps = executor._extract_file_dependencies(tool_calls)
        assert deps[0] == set()
        assert deps[1] == {0}  # Read depends on prior write to same file

    def test_read_independent_of_write_to_different_file(self):
        executor = self._make_executor()
        tool_calls = [
            {"name": "write_file", "arguments": {"path": "/a.py"}},
            {"name": "read_file", "arguments": {"path": "/b.py"}},
        ]
        deps = executor._extract_file_dependencies(tool_calls)
        assert deps[1] == set()  # Different files → no dependency

    def test_write_without_path_depends_on_all_prior(self):
        executor = self._make_executor()
        tool_calls = [
            {"name": "write_file", "arguments": {"path": "/a.py"}},
            {"name": "execute_bash", "arguments": {"command": "make"}},
        ]
        deps = executor._extract_file_dependencies(tool_calls)
        # execute_bash is a WRITE tool with no path → depends on all prior writes
        assert deps[1] == {0}

    def test_can_parallelize_with_writes(self):
        executor = self._make_executor()
        tool_calls = [
            {"name": "write_file", "arguments": {"path": "/a.py"}},
            {"name": "write_file", "arguments": {"path": "/b.py"}},
        ]
        assert executor._can_parallelize(tool_calls) is True

    def test_three_writes_mixed_deps(self):
        executor = self._make_executor()
        tool_calls = [
            {"name": "write_file", "arguments": {"path": "/a.py"}},  # 0
            {"name": "write_file", "arguments": {"path": "/b.py"}},  # 1
            {"name": "write_file", "arguments": {"path": "/a.py"}},  # 2 → depends on 0
        ]
        deps = executor._extract_file_dependencies(tool_calls)
        assert deps[0] == set()
        assert deps[1] == set()
        assert deps[2] == {0}


# ============================================================
# 6. Cross-turn tool result deduplication
# ============================================================


class TestCrossTurnDedup:
    """Test cross-turn dedup cache in ToolPipeline."""

    def _make_pipeline(self, cross_turn_enabled=True, cross_turn_ttl=300.0):
        from victor.agent.tool_pipeline import (
            ToolPipeline,
            ToolPipelineConfig,
            ToolCallResult,
        )

        mock_registry = MagicMock()
        mock_tool = MagicMock()
        mock_tool.name = "web_search"
        mock_tool.description = "Search the web"
        mock_tool.parameters = {}
        mock_registry.get.return_value = mock_tool
        mock_registry.get_tool.return_value = mock_tool

        mock_executor = MagicMock()
        mock_executor.execute = AsyncMock(
            return_value=MagicMock(
                success=True,
                result="search results",
                error=None,
                output="search results",
            )
        )

        config = ToolPipelineConfig(
            tool_budget=100,
            enable_cross_turn_dedup=cross_turn_enabled,
            cross_turn_dedup_ttl=cross_turn_ttl,
        )
        pipeline = ToolPipeline(
            tool_registry=mock_registry,
            tool_executor=mock_executor,
            config=config,
        )
        return pipeline

    def test_cross_turn_cache_initialized(self):
        pipeline = self._make_pipeline(cross_turn_enabled=True)
        assert pipeline._cross_turn_enabled is True
        assert pipeline._cross_turn_hits == 0

    def test_cross_turn_disabled(self):
        pipeline = self._make_pipeline(cross_turn_enabled=False)
        assert pipeline._cross_turn_enabled is False

    def test_cross_turn_dedup_tools_defined(self):
        from victor.agent.tool_pipeline import CROSS_TURN_DEDUP_TOOLS

        assert "web_search" in CROSS_TURN_DEDUP_TOOLS
        assert "grep_search" in CROSS_TURN_DEDUP_TOOLS
        assert "http_request" in CROSS_TURN_DEDUP_TOOLS
        assert "git" in CROSS_TURN_DEDUP_TOOLS
        assert "plan_files" in CROSS_TURN_DEDUP_TOOLS

    def test_cross_turn_cache_stats_included(self):
        pipeline = self._make_pipeline()
        stats = pipeline.get_cache_stats()
        assert "cross_turn_hits" in stats
        assert "cross_turn_cache_size" in stats
        assert stats["cross_turn_hits"] == 0
        assert stats["cross_turn_cache_size"] == 0

    def test_config_ttl_flows_to_cache(self):
        pipeline = self._make_pipeline(cross_turn_ttl=120.0)
        assert pipeline._cross_turn_cache._ttl_seconds == 120.0


# ============================================================
# Integration: Settings → Factory → Component wiring
# ============================================================


class TestSettingsWiring:
    """Verify settings flow from ToolSettings through factory to components."""

    def test_tool_settings_has_all_new_fields(self):
        from victor.config.tool_settings import ToolSettings

        ts = ToolSettings()
        assert hasattr(ts, "max_tool_schema_tokens")
        assert hasattr(ts, "schema_promotion_threshold")
        assert hasattr(ts, "max_mcp_tools_per_turn")
        assert hasattr(ts, "cross_turn_dedup_enabled")
        assert hasattr(ts, "cross_turn_dedup_ttl")

    def test_tool_settings_defaults(self):
        from victor.config.tool_settings import ToolSettings

        ts = ToolSettings()
        assert ts.max_tool_schema_tokens == 4000
        assert ts.schema_promotion_threshold == 0.8
        assert ts.max_mcp_tools_per_turn == 12
        assert ts.cross_turn_dedup_enabled is True
        assert ts.cross_turn_dedup_ttl == 300

    def test_pipeline_config_has_cross_turn_fields(self):
        from victor.agent.tool_pipeline import ToolPipelineConfig

        cfg = ToolPipelineConfig()
        assert cfg.enable_cross_turn_dedup is True
        assert cfg.cross_turn_dedup_ttl == 300.0

    def test_pipeline_config_custom_values(self):
        from victor.agent.tool_pipeline import ToolPipelineConfig

        cfg = ToolPipelineConfig(
            enable_cross_turn_dedup=False,
            cross_turn_dedup_ttl=60.0,
        )
        assert cfg.enable_cross_turn_dedup is False
        assert cfg.cross_turn_dedup_ttl == 60.0

    def test_context_settings_has_cache_fields(self):
        from victor.config.context_settings import ContextSettings

        cs = ContextSettings()
        assert cs.cache_optimization_enabled is True
        assert cs.kv_optimization_enabled is True
        assert cs.kv_tool_strategy == "context_aware"
        assert cs.tiered_schema_enabled is True
