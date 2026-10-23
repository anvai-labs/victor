"""Tier-3 precise-index drivability from the graph CLI (ADR-032 rev 1.8).

Pins the operator-handoff regressions end to end at the CLI construction
seam: settings.yaml-only enablement must reach GraphIndexConfig (the env-only
load_settings() snapshot used to drop it), the dump timeout must actually be
wired (the rust-analyzer 120s default silently killed real-workspace dumps),
and harvest health must be visible — index summary, the
``.victor/graph-index-summary.json`` sidecar that ``graph stats`` reads, and
``--list-partial`` — instead of dying in a log line.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest
from rich.console import Console

from victor.ui.commands import graph

_GRAPH_ENV_VARS = (
    "VICTOR_SEARCH__GRAPH__ENABLE_PRECISE_INDEX",
    "VICTOR_SEARCH__GRAPH__PRECISE_INDEX_TIMEOUT_SECONDS",
    "VICTOR_SEARCH__GRAPH__PRECISE_INDEX_TOOL",
    "VICTOR_SEARCH__GRAPH__ENABLE_SCOPE_BINDING",
)


@pytest.fixture(autouse=True)
def _clean_graph_env(monkeypatch):
    """Keep the developer's real environment out of the settings merge."""
    for var in _GRAPH_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def _settings_from_yaml(tmp_path, yaml_body):
    from victor.config.settings import Settings

    config_dir = tmp_path / "cfg"
    config_dir.mkdir(exist_ok=True)
    (config_dir / "settings.yaml").write_text(yaml_body)
    return Settings.from_sources(config_dir=config_dir)


async def _run_index(tmp_path, settings, stats, *, precise=None, precise_timeout=None):
    """Drive ``_index_async`` with the pipeline mocked; return the captured
    GraphIndexConfig and the rendered summary."""
    output = Console(record=True, width=200)
    with (
        patch.object(graph, "console", output),
        patch("victor.core.graph_rag.dependencies.require_graph_dependencies"),
        patch.object(graph, "create_graph_store", return_value=AsyncMock()),
        patch("victor.core.graph_rag.GraphIndexingPipeline") as pipeline,
        patch(
            "victor.core.graph_rag.indexing.run_indexing_with_lock",
            new_callable=AsyncMock,
            return_value=stats,
        ),
    ):
        ok = await graph._index_async(
            str(tmp_path),
            True,
            False,
            precise=precise,
            precise_timeout=precise_timeout,
            settings=settings,
        )
    return ok, pipeline.call_args[0][1], output.export_text()


async def test_yaml_only_settings_enable_precise_index(tmp_path):
    """Acceptance: search.graph.enable_precise_index: true in settings.yaml
    (no env var) must reach GraphIndexConfig through the CLI path."""
    from victor.core.graph_rag.indexing import GraphIndexStats

    settings = _settings_from_yaml(
        tmp_path,
        "search:\n"
        "  graph:\n"
        "    enable_precise_index: true\n"
        "    precise_index_timeout_seconds: 3600\n"
        "    precise_index_tool: scip-python\n",
    )

    ok, config, _ = await _run_index(tmp_path, settings, GraphIndexStats(files_processed=1))

    assert ok
    assert config.enable_precise_index is True
    assert config.precise_index_timeout_seconds == 3600
    assert config.precise_index_tool == "scip-python"


async def test_defaults_without_precise_configuration(tmp_path):
    """Negative: empty settings sources keep tier 3 off (opt-in default) and
    the timeout at 0 (per-provider defaults)."""
    from victor.core.graph_rag.indexing import GraphIndexStats

    settings = _settings_from_yaml(tmp_path, "")

    ok, config, _ = await _run_index(tmp_path, settings, GraphIndexStats(files_processed=1))

    assert ok
    assert config.enable_precise_index is False
    assert config.precise_index_timeout_seconds == 0
    assert config.precise_index_tool == "auto"


async def test_cli_flags_override_settings(tmp_path):
    """--precise/--no-precise and --precise-timeout win over settings.yaml."""
    from victor.core.graph_rag.indexing import GraphIndexStats

    enabled = _settings_from_yaml(
        tmp_path,
        "search:\n"
        "  graph:\n"
        "    enable_precise_index: true\n"
        "    precise_index_timeout_seconds: 3600\n",
    )
    disabled = _settings_from_yaml(tmp_path, "")

    _, opted_out, _ = await _run_index(
        tmp_path,
        enabled,
        GraphIndexStats(files_processed=1),
        precise=False,
        precise_timeout=99,
    )
    assert opted_out.enable_precise_index is False
    assert opted_out.precise_index_timeout_seconds == 99

    _, opted_in, _ = await _run_index(
        tmp_path, disabled, GraphIndexStats(files_processed=1), precise=True
    )
    assert opted_in.enable_precise_index is True


async def test_index_summary_sidecar_written_and_stats_surfaces_it(tmp_path):
    """Acceptance: per-provider outcomes and join accounting must be visible —
    written to the sidecar after the run and rendered by ``graph stats``."""
    from victor.core.graph_rag.indexing import GraphIndexStats

    settings = _settings_from_yaml(tmp_path, "")
    stats = GraphIndexStats(
        files_processed=3,
        precise_references_resolved=120,
        precise_index_unjoined=12,
        precise_index_unjoined_docs=2,
        precise_index_outcomes={
            "scip-python": "ok:120 edges (join 90.9%) [cached dump]",
            "rust-analyzer": "dump-failed",
        },
    )

    ok, _, rendered = await _run_index(tmp_path, settings, stats)
    assert ok

    sidecar = graph._index_summary_file(tmp_path)
    assert sidecar.is_file()
    payload = json.loads(sidecar.read_text())
    assert payload["precise_index"]["outcomes"]["rust-analyzer"] == "dump-failed"
    assert payload["precise_index"]["references_resolved"] == 120
    assert payload["precise_index"]["unjoined"] == 12

    # The index-run summary itself prints the same health inline.
    assert "Precise index (tier 3):" in rendered
    assert "rust-analyzer: dump-failed" in rendered

    stats_output = Console(record=True, width=200)
    store = AsyncMock()
    store.stats = AsyncMock(return_value={"nodes": 10, "edges": 20, "indexed_files": 3})
    with (
        patch.object(graph, "console", stats_output),
        patch.object(graph, "create_graph_store", return_value=store),
    ):
        assert await graph._stats_async(str(tmp_path))

    rendered_stats = stats_output.export_text()
    # The stats table renders provider/outcome as columns, not "key: value".
    assert "Precise index (last run)" in rendered_stats
    assert "enabled" in rendered_stats
    assert "scip-python" in rendered_stats
    assert "ok:120 edges (join 90.9%) [cached dump]" in rendered_stats
    assert "rust-analyzer" in rendered_stats
    assert "dump-failed" in rendered_stats
    assert "resolved=120" in rendered_stats
    assert "unjoined=12" in rendered_stats
    assert "out-of-corpus docs=2" in rendered_stats


async def test_stats_without_summary_sidecar_prints_no_precise_section(tmp_path):
    """Negative: no index-run sidecar (never indexed via CLI) must not render
    a precise section or crash."""
    stats_output = Console(record=True, width=200)
    store = AsyncMock()
    store.stats = AsyncMock(return_value={"nodes": 1, "edges": 2, "indexed_files": 1})
    with (
        patch.object(graph, "console", stats_output),
        patch.object(graph, "create_graph_store", return_value=store),
    ):
        assert await graph._stats_async(str(tmp_path))

    rendered = stats_output.export_text()
    assert "Graph Statistics:" in rendered
    assert "Precise" not in rendered


def test_index_summary_prints_precise_health_and_join_accounting():
    from victor.core.graph_rag.indexing import GraphIndexStats

    stats = GraphIndexStats(
        precise_references_resolved=900,
        precise_index_unjoined=52,
        precise_index_unjoined_docs=3,
        precise_index_outcomes={"rust-analyzer": "ok:952 edges (join 89.2%)"},
    )
    output = Console(record=True, width=200)
    with patch.object(graph, "console", output):
        graph._print_index_summary(stats)

    rendered = output.export_text()
    assert "Precise index (tier 3):" in rendered
    assert "rust-analyzer: ok:952 edges (join 89.2%)" in rendered
    assert "resolved=900, unjoined=52 (out-of-corpus docs=3)" in rendered


def test_index_summary_prints_skip_reason_without_outcomes():
    from victor.core.graph_rag.indexing import GraphIndexStats

    stats = GraphIndexStats(precise_index_skipped="toolchain-or-dump-failed")
    output = Console(record=True, width=200)
    with patch.object(graph, "console", output):
        graph._print_index_summary(stats)

    rendered = output.export_text()
    assert "skipped: toolchain-or-dump-failed" in rendered


def test_index_summary_omits_precise_section_when_unconfigured():
    from victor.core.graph_rag.indexing import GraphIndexStats

    output = Console(record=True, width=200)
    with patch.object(graph, "console", output):
        graph._print_index_summary(GraphIndexStats(files_processed=1))

    assert "Precise index" not in output.export_text()


def test_list_partial_prints_paths():
    from victor.core.graph_rag.indexing import GraphIndexStats

    stats = GraphIndexStats(files_partial=2, partial_files=["src/a.py", "src/b.rs"])
    output = Console(record=True, width=200)
    with patch.object(graph, "console", output):
        graph._print_index_summary(stats, list_partial=True)

    rendered = output.export_text()
    assert "Partial-parse files (2):" in rendered
    assert "src/a.py" in rendered
    assert "src/b.rs" in rendered

    empty_output = Console(record=True, width=200)
    with patch.object(graph, "console", empty_output):
        graph._print_index_summary(GraphIndexStats(), list_partial=True)
    assert "Partial-parse files: none recorded this pass" in empty_output.export_text()


def test_watch_refresh_manifest_records_precise_outcomes(tmp_path):
    """The watcher surface carries the same harvest health in its manifest."""
    from victor.core.graph_rag.indexing import GraphIndexStats

    stats = GraphIndexStats(
        files_processed=1,
        precise_index_outcomes={"scip-python": "ok:9 edges"},
        precise_index_skipped="",
    )
    manifest = graph._record_graph_watch_refresh(tmp_path, stats)

    payload = json.loads(manifest.read_text())
    last_refresh = payload["last_refresh"]
    assert last_refresh["precise_index_outcomes"] == {"scip-python": "ok:9 edges"}
    assert last_refresh["precise_index_skipped"] == ""


def test_watch_status_reports_precise_outcomes(tmp_path):
    graph._write_graph_watch_manifest(
        tmp_path,
        graph.GraphWatchDaemonState(pid_file=graph._default_graph_watch_pid_file(tmp_path)),
        last_refresh={
            "changed": 1,
            "deleted": 0,
            "unchanged": 2,
            "errors": 0,
            "duration_seconds": 1.5,
            "completed_at": 0.0,
            "precise_index_outcomes": {"rust-analyzer": "dump-failed"},
            "precise_index_skipped": "",
        },
    )
    output = Console(record=True, width=200)
    pid_file = graph._default_graph_watch_pid_file(tmp_path)
    with patch.object(graph, "console", output):
        graph.graph_watch_status(path=str(tmp_path), pid_file=pid_file)

    rendered = output.export_text()
    assert "Precise rust-analyzer" in rendered
    assert "dump-failed" in rendered
