"""Benchmark data must stay bounded and cannot masquerade as measured detection."""

import importlib.util
from pathlib import Path

import pytest


def load():
    path = Path(__file__).resolve().parents[3] / "scripts/benchmark_graph_similarity_evaluation.py"
    spec = importlib.util.spec_from_file_location("similarity_protocol_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_protocol_benchmark_keeps_scope_and_implementation_evidence():
    result = load().benchmark(100, 1)
    assert result["pairs"] == 100
    assert not result["accuracy_benchmark"] and not result["detector_retrieval_benchmark"]
    assert result["median_wall_seconds"] >= 0
    assert all(len(value) == 64 for value in result["implementation"].values())


@pytest.mark.parametrize("pairs,runs", [(7, 1), (9, 1), (50_004, 1), (100, 0), (100, 21)])
def test_protocol_budget_is_validated(pairs, runs):
    with pytest.raises(ValueError):
        load().benchmark(pairs, runs)
