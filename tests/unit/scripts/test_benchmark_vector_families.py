"""Repeated vector observations retain answer parity and reject changed inputs."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def load():
    path = Path(__file__).resolve().parents[3] / "scripts/benchmark_vector_families.py"
    spec = importlib.util.spec_from_file_location("vector_family_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_repeated_control_keeps_discovery_answers_and_explains_each_component():
    result = load().benchmark(count=30, dimension=7, runs=2)
    assert not result["accuracy_benchmark"]
    assert result["locations"] == "synthetic"
    first, second = result["observations"]
    assert first["families"] > 0
    assert first["cohesion_components"] == first["families"]
    assert first["answer_sha256"] == second["answer_sha256"]


def test_fixture_replication_is_explicit_and_reproducible(tmp_path):
    path = tmp_path / "vectors.npy"
    np.save(path, np.eye(2, dtype=np.float32))
    result = load().benchmark(count=20, runs=1, fixture=path)
    assert result["vectors"] == "replicated_local_fixture"
    assert result["dimension"] == 2
    assert result["observations"][0]["matching_edges"] == 90
    assert len(result["fixture_sha256"]) == 64


def test_changed_fixture_cannot_be_accepted_as_one_observation(tmp_path, monkeypatch):
    module = load()
    path = tmp_path / "vectors.npy"
    np.save(path, np.eye(2, dtype=np.float32))
    original = module.semantic_clones.find_clone_families

    def change(*args, **kwargs):
        result = original(*args, **kwargs)
        np.save(path, np.ones((2, 2), dtype=np.float32))
        return result

    monkeypatch.setattr(module.semantic_clones, "find_clone_families", change)
    with pytest.raises(RuntimeError, match="fixture changed"):
        module.benchmark(count=20, runs=1, fixture=path)


@pytest.mark.parametrize(
    "options",
    [
        {"count": 1},
        {"count": 20001},
        {"dimension": 0},
        {"runs": 0},
        {"count": 20000, "dimension": 4096},
    ],
)
def test_invalid_or_excessive_workloads_are_rejected(options):
    with pytest.raises(ValueError):
        load().benchmark(**options)
