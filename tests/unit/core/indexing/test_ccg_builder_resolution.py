"""CCG builder resolution knob (enhanced-builder boundary, 2026-10-24).

The victor-coding plugin registers a richer plugin-first CCG builder for
ALL languages via the CapabilityRegistry, shadowing the 81x native engine.
The knob makes the trade explicit: "auto" (plugin-first, historical),
"native" (the engine first — a measured performance/graph-shape trade),
"plugin"/"builtin" force a leg. Never a silent default flip: the plugin
graph is a DIFFERENT shape, not a superset.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from victor.core.indexing.ccg_builder import CodeContextGraphBuilder

FIXTURE = "victor/core/graph_rag/precise_providers.py"


def _builder(resolution: str) -> CodeContextGraphBuilder:
    return CodeContextGraphBuilder(language="python", builder_resolution=resolution)


def _fixture() -> Path | None:
    repo_root = Path(__file__).resolve().parents[4]
    fixture = repo_root / FIXTURE
    return fixture if fixture.exists() else None


def test_default_resolution_is_auto_plugin_first():
    builder = _builder("auto")
    assert builder._builder_resolution == "auto"
    assert builder._resolve_enhanced_builder("python") is not None


def test_invalid_resolution_falls_back_to_auto():
    assert _builder("")._builder_resolution == "auto"
    assert _builder("  NATIVE ")._builder_resolution == "native"


async def test_build_ccg_for_file_native_never_consults_the_plugin():
    """Under 'native', the plugin resolution is bypassed entirely and the
    native fast path serves the file (smaller measured graph shape)."""
    fixture = _fixture()
    if fixture is None:
        import pytest

        pytest.skip("fixture not present")

    builder = _builder("native")
    with patch.object(
        builder,
        "_resolve_enhanced_builder",
        side_effect=AssertionError("must not consult the plugin under 'native'"),
    ):
        nodes, edges = await builder.build_ccg_for_file(fixture, "python")
    assert len(nodes) == 754  # native shape (measured on this fixture)
    assert edges


async def test_build_ccg_for_file_auto_uses_the_plugin():
    fixture = _fixture()
    if fixture is None:
        import pytest

        pytest.skip("fixture not present")

    nodes, _ = await _builder("auto").build_ccg_for_file(fixture, "python")
    assert len(nodes) == 1161  # plugin shape (measured on this fixture)


def test_pipeline_config_reaches_thread_builders():
    """The indexing pipeline passes config.ccg_builder into every
    per-thread CodeContextGraphBuilder it constructs."""
    import inspect

    from victor.core.graph_rag import indexing

    source = inspect.getsource(indexing)
    assert 'builder_resolution=getattr(self.config, "ccg_builder", "auto")' in source
