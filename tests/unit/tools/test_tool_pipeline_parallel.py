"""Hot-path test coverage for ToolPipeline parallel execution and caching (Item 6)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Synthesis checkpoint
# ---------------------------------------------------------------------------


class TestSynthesisCheckpoint:
    async def test_checkpoint_skipped_when_disabled(self):
        from victor.agent.tool_pipeline import ToolPipeline, ToolPipelineConfig

        registry = MagicMock()
        executor = MagicMock()
        config = ToolPipelineConfig(enable_synthesis_checkpoints=False)
        pipeline = ToolPipeline(registry, executor, config=config)
        assert pipeline._synthesis_checkpoint is None

    async def test_checkpoint_created_when_enabled(self):
        from victor.agent.tool_pipeline import ToolPipeline, ToolPipelineConfig

        registry = MagicMock()
        executor = MagicMock()
        config = ToolPipelineConfig(enable_synthesis_checkpoints=True)
        pipeline = ToolPipeline(registry, executor, config=config)
        assert pipeline._synthesis_checkpoint is not None


# ---------------------------------------------------------------------------
# Cross-turn deduplication
# ---------------------------------------------------------------------------


class TestCrossTurnDedup:
    async def test_cross_turn_dedup_enabled_by_config(self):
        from victor.agent.tool_pipeline import ToolPipeline, ToolPipelineConfig

        registry = MagicMock()
        executor = MagicMock()
        config = ToolPipelineConfig(enable_cross_turn_dedup=True, cross_turn_dedup_ttl=60.0)
        pipeline = ToolPipeline(registry, executor, config=config)
        assert pipeline._cross_turn_enabled is True

    async def test_cross_turn_dedup_disabled_by_config(self):
        from victor.agent.tool_pipeline import ToolPipeline, ToolPipelineConfig

        registry = MagicMock()
        executor = MagicMock()
        config = ToolPipelineConfig(enable_cross_turn_dedup=False)
        pipeline = ToolPipeline(registry, executor, config=config)
        assert pipeline._cross_turn_enabled is False
