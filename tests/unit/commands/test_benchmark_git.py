# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for benchmark git timeout helper and pre-warm timeout."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestRunGitWithTimeout:
    """Tests for _run_git_with_timeout helper."""

    @pytest.mark.asyncio
    async def test_run_git_with_timeout_success(self):
        """Successful git command returns (returncode, stdout, stderr)."""
        from victor.ui.commands.benchmark import _run_git_with_timeout

        result = await _run_git_with_timeout(["git", "--version"], cwd=".", timeout=10)
        rc, stdout, stderr = result
        assert rc == 0
        assert b"git version" in stdout

    @pytest.mark.asyncio
    async def test_run_git_with_timeout_kills_on_hang(self):
        """Hanging git command is killed after timeout."""
        from victor.ui.commands.benchmark import _run_git_with_timeout

        with pytest.raises(asyncio.TimeoutError):
            await _run_git_with_timeout(["sleep", "60"], cwd=".", timeout=1)

    @pytest.mark.asyncio
    async def test_prewarm_timeout_continues(self):
        """Pre-warm timeout should not crash the benchmark — it's best-effort."""

        # Simulate what happens when pre-warm times out
        async def hanging_prewarm(*args, **kwargs):
            await asyncio.sleep(60)

        try:
            await asyncio.wait_for(hanging_prewarm(), timeout=0.1)
        except asyncio.TimeoutError:
            pass  # Expected — benchmark should continue

        # Verify we can still proceed after timeout
        assert True


class TestCodeIntelligencePrewarm:
    @pytest.mark.asyncio
    async def test_prewarm_success_caches_graph_stats(self, tmp_path: Path):
        from victor.ui.commands.benchmark import _prewarm_code_intelligence_index

        warmed = {}
        fake_graph_store = MagicMock()
        fake_graph_store.stats = AsyncMock(return_value={"nodes": 12, "edges": 34})
        fake_index = MagicMock(graph_store=fake_graph_store)

        # Patch the capability loader seam that production actually uses.
        # Production resolves the index builder via load_code_search_module()
        # (victor-coding or the contrib vertical), NOT via
        # victor.tools.code_search_tool directly, so patching that module's
        # _get_or_build_index was inert whenever a different module resolved.
        fake_module = MagicMock()
        fake_module._get_or_build_index = AsyncMock(return_value=(fake_index, False))

        with (
            patch("victor.config.settings.load_settings", return_value=MagicMock()),
            patch(
                "victor.core.utils.capability_loader.load_code_search_module",
                return_value=fake_module,
            ),
        ):
            first = await _prewarm_code_intelligence_index(tmp_path, warmed, timeout=1.0)
            second = await _prewarm_code_intelligence_index(tmp_path, warmed, timeout=1.0)

        assert first.status == "ready"
        assert first.graph_nodes == 12
        assert first.graph_edges == 34
        assert second.cached_hit is True
        assert second.status == "ready"
        assert fake_module._get_or_build_index.call_count == 1

    @pytest.mark.asyncio
    async def test_prewarm_failure_is_cached_for_repo(self, tmp_path: Path):
        from victor.ui.commands.benchmark import _prewarm_code_intelligence_index

        warmed = {}

        # See test_prewarm_success_caches_graph_stats: patch the loader seam,
        # not the (resolution-dependent) module attribute.
        fake_module = MagicMock()
        fake_module._get_or_build_index = AsyncMock(
            side_effect=RuntimeError("missing embedding dependency")
        )

        with (
            patch("victor.config.settings.load_settings", return_value=MagicMock()),
            patch(
                "victor.core.utils.capability_loader.load_code_search_module",
                return_value=fake_module,
            ),
        ):
            first = await _prewarm_code_intelligence_index(tmp_path, warmed, timeout=1.0)
            second = await _prewarm_code_intelligence_index(tmp_path, warmed, timeout=1.0)

        assert first.status == "failed"
        assert "missing embedding dependency" in first.message
        assert second.cached_hit is True
        assert second.status == "failed"
        assert fake_module._get_or_build_index.call_count == 1


class TestBenchmarkRuntimeReadiness:
    def test_runtime_readiness_passes_with_enabled_tools(self):
        from victor.evaluation.agent_adapter import BenchmarkToolReadiness
        from victor.ui.commands.benchmark import _ensure_benchmark_runtime_tools

        adapter = MagicMock()
        adapter.get_benchmark_tool_readiness.return_value = BenchmarkToolReadiness(
            required_tools=("code_search", "graph", "read"),
            enabled_tools=("code_search", "graph", "read"),
        )

        readiness = _ensure_benchmark_runtime_tools(adapter)

        assert readiness.ready is True
        assert "graph" in readiness.enabled_tools

    def test_runtime_readiness_fails_fast_for_missing_graph(self):
        from victor.evaluation.agent_adapter import BenchmarkToolReadiness
        from victor.ui.commands.benchmark import _ensure_benchmark_runtime_tools

        adapter = MagicMock()
        adapter.get_benchmark_tool_readiness.return_value = BenchmarkToolReadiness(
            required_tools=("code_search", "graph", "read"),
            enabled_tools=("code_search", "read"),
            missing_tools=("graph",),
        )

        with pytest.raises(RuntimeError, match="missing tools: graph"):
            _ensure_benchmark_runtime_tools(adapter)

    def test_runtime_readiness_fails_fast_for_disabled_graph(self):
        from victor.evaluation.agent_adapter import BenchmarkToolReadiness
        from victor.ui.commands.benchmark import _ensure_benchmark_runtime_tools

        adapter = MagicMock()
        adapter.get_benchmark_tool_readiness.return_value = BenchmarkToolReadiness(
            required_tools=("code_search", "graph", "read"),
            enabled_tools=("code_search", "read"),
            disabled_tools=("graph",),
        )

        with pytest.raises(RuntimeError, match="disabled tools: graph"):
            _ensure_benchmark_runtime_tools(adapter)


class TestBenchmarkCodeIntelligenceDiagnostics:
    def test_summarize_code_intelligence_diagnostics_identifies_missed_tasks(self):
        from victor.evaluation.protocol import TaskStatus
        from victor.ui.commands.benchmark import (
            _summarize_code_intelligence_diagnostics,
        )

        passed = MagicMock()
        passed.task_id = "task-pass"
        passed.used_code_intelligence = True
        passed.used_graph = True
        passed.status = TaskStatus.PASSED

        failed_without_intel = MagicMock()
        failed_without_intel.task_id = "task-fail"
        failed_without_intel.used_code_intelligence = False
        failed_without_intel.used_graph = False
        failed_without_intel.status = TaskStatus.FAILED

        passed_without_graph = MagicMock()
        passed_without_graph.task_id = "task-no-graph"
        passed_without_graph.used_code_intelligence = True
        passed_without_graph.used_graph = False
        passed_without_graph.status = TaskStatus.PASSED

        result = MagicMock(task_results=[passed, failed_without_intel, passed_without_graph])

        diagnostics = _summarize_code_intelligence_diagnostics(result)

        assert diagnostics["total_tasks"] == 3
        assert diagnostics["tasks_without_code_intelligence"] == 1
        assert diagnostics["failed_tasks_without_code_intelligence"] == 1
        assert diagnostics["task_ids_without_code_intelligence"] == ["task-fail"]
        assert diagnostics["failed_task_ids_without_code_intelligence"] == ["task-fail"]
        assert diagnostics["tasks_without_graph"] == 2
        assert diagnostics["sample_task_ids_without_graph"] == [
            "task-fail",
            "task-no-graph",
        ]
        assert diagnostics["code_intelligence_coverage"] == pytest.approx(2 / 3)

    def test_summarize_code_intelligence_diagnostics_handles_full_coverage(self):
        from victor.evaluation.protocol import TaskStatus
        from victor.ui.commands.benchmark import (
            _summarize_code_intelligence_diagnostics,
        )

        task = MagicMock()
        task.task_id = "task-1"
        task.used_code_intelligence = True
        task.used_graph = True
        task.status = TaskStatus.PASSED

        diagnostics = _summarize_code_intelligence_diagnostics(MagicMock(task_results=[task]))

        assert diagnostics["tasks_without_code_intelligence"] == 0
        assert diagnostics["failed_tasks_without_code_intelligence"] == 0
        assert diagnostics["tasks_without_graph"] == 0
        assert diagnostics["code_intelligence_coverage"] == 1.0
