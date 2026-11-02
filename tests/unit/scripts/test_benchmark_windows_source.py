"""Acquisition benchmark rejects lost coverage and retains deterministic digests."""

import importlib.util
from pathlib import Path

import pytest


def load_benchmark():
    path = Path(__file__).resolve().parents[3] / "scripts/benchmark_windows_source.py"
    spec = importlib.util.spec_from_file_location("windows_source_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_acquisition_measurement_checks_coverage_and_retains_digest(tmp_path):
    benchmark = load_benchmark()
    (tmp_path / "a.py").write_bytes(b"source")
    first = benchmark.measure(tmp_path, 1, 6)
    second = benchmark.measure(tmp_path, 1, 6)
    assert first["sha256"] == second["sha256"]
    assert first["files"] == 1 and first["bytes"] == 6
    assert first["wall_seconds"] > 0 and first["bytes_per_second"] > 0
    with pytest.raises(ValueError, match="coverage changed"):
        benchmark.measure(tmp_path, 2, 6)
    with pytest.raises(ValueError, match="source bytes changed"):
        benchmark.measure(tmp_path, 1, 3)
