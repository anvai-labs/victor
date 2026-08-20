# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "benchmark_graph_backends.py"
_SPEC = importlib.util.spec_from_file_location("benchmark_graph_backends", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
benchmark = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(benchmark)


def test_create_benchmark_store_pins_explicit_proxima_binary(tmp_path: Path) -> None:
    binary = tmp_path / "proximadb-server"
    store = MagicMock()

    with patch(
        "victor.storage.graph.proxima_store.ProximaGraphStore", return_value=store
    ) as proxima_store:
        result = benchmark._create_benchmark_store(
            "proxima", tmp_path / "project", server_binary=binary
        )

    assert result is store
    proxima_store.assert_called_once_with(
        project_path=tmp_path / "project", binary_path=str(binary)
    )


@pytest.mark.asyncio
async def test_time_reopen_times_readiness_and_validates_readback(tmp_path: Path) -> None:
    store = AsyncMock()
    store.get_all_nodes.return_value = [object(), object()]
    store.get_all_edges.return_value = [object()]

    with (
        patch.object(benchmark, "create_graph_store", return_value=store) as create,
        patch.object(benchmark.time, "perf_counter", side_effect=[10.0, 12.345]),
    ):
        result = await benchmark._time_reopen("proxima", tmp_path)

    create.assert_called_once_with("proxima", project_path=tmp_path)
    store.initialize.assert_awaited_once_with()
    store.close.assert_awaited_once_with()
    assert result == {"seconds": 2.35, "nodes": 2, "edges": 1}


@pytest.mark.asyncio
async def test_time_reopen_reports_startup_failure(tmp_path: Path) -> None:
    store = AsyncMock()
    store.initialize.side_effect = RuntimeError("replay failed")

    with patch.object(benchmark, "create_graph_store", return_value=store):
        result = await benchmark._time_reopen("proxima", tmp_path)

    assert result == {"error": "RuntimeError: replay failed"}
    store.close.assert_awaited_once_with()


def test_render_surfaces_reopen_measurement_and_failure() -> None:
    report = benchmark._render(
        [
            {
                "backend": "sqlite",
                "nodes": 2,
                "edges": 1,
                "index_seconds": 1.0,
                "reopen": {"seconds": 0.02, "nodes": 2, "edges": 1},
                "footprint": "1.0 KB",
                "footprint_bytes": 1024,
                "bytes_per_node": 512.0,
                "traversal": {"p50_ms": 0.1},
            },
            {
                "backend": "proxima",
                "nodes": 2,
                "edges": 1,
                "index_seconds": 2.0,
                "reopen": {"error": "TimeoutError: recovery stalled"},
                "footprint": "2.0 KB",
                "footprint_bytes": 2048,
                "bytes_per_node": 1024.0,
                "traversal": {"p50_ms": 0.2},
            },
        ],
        embeddings=False,
    )

    assert "reopen s" in report
    assert "0.02" in report
    assert "ERROR reopen failed: TimeoutError: recovery stalled" in report
