"""Cache observations preserve repeatable scores and reject stale measurements."""

import hashlib
import importlib.util
from pathlib import Path

import pytest


def setup_probe(tmp_path, monkeypatch):
    from victor.core.graph_rag.body_clones import LocalBodyEncoder
    from victor.core.graph_rag.structural_clones import StructuralReport, extract_units

    path = Path(__file__).resolve().parents[3] / "scripts/benchmark_body_cache.py"
    spec = importlib.util.spec_from_file_location("body_cache_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = tmp_path / "source"
    root.mkdir()
    file = root / "test.py"
    source = "def f(x):\n return x+1\n\ndef g(x):\n return x-1\n"
    file.write_text(source)
    report = StructuralReport(
        [],
        [],
        {"source_files": 1, "selected_regions": 1},
        units=extract_units("test.py", source, None, min_nodes=1, retain_source=True),
        source_manifest={"test.py": hashlib.sha256(source.encode()).hexdigest()},
    )
    monkeypatch.setattr(module, "scan", lambda *a, **k: report)

    class Encoder(LocalBodyEncoder):
        def windows(self, source):
            return [source], [(0, len(source))]

        def encode(self, texts):
            return [[1] * self.options.dimension for _ in texts]

    monkeypatch.setattr(
        module, "LocalBodyEncoder", lambda options: Encoder(options, producer_id="tests/probe-v1")
    )
    return module, root, file


def test_cold_and_warm_measurement_preserves_vector_and_score_hashes(tmp_path, monkeypatch):
    module, root, _ = setup_probe(tmp_path, monkeypatch)
    cache = tmp_path / "cache.sqlite"
    cold = module.benchmark(root, cache, state="cold")
    warm = module.benchmark(root, cache, state="warm")
    assert cold["stats"]["inferred_windows"] == 2
    assert warm["stats"]["cache_hits"] == 2
    assert warm["stats"]["inferred_windows"] == 0
    assert cold["vector_manifest_sha256"] == warm["vector_manifest_sha256"]
    assert cold["pair_scores_sha256"] == warm["pair_scores_sha256"]
    assert cold["pair_count"] == 1
    assert not cold["accuracy_benchmark"]
    with pytest.raises(ValueError, match="existence"):
        module.benchmark(root, cache, state="cold")


def test_measurement_rejects_changed_source_snapshot(tmp_path, monkeypatch):
    module, root, file = setup_probe(tmp_path, monkeypatch)
    original = module.analyze_bodies

    def change(*args, **kwargs):
        result = original(*args, **kwargs)
        file.write_text("# changed snapshot\n")
        return result

    monkeypatch.setattr(module, "analyze_bodies", change)
    with pytest.raises(RuntimeError, match="snapshot changed"):
        module.benchmark(root, tmp_path / "cache.sqlite", state="cold")


def test_cache_inside_snapshot_and_missing_warm_cache_rejected(tmp_path, monkeypatch):
    module, root, _ = setup_probe(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="outside"):
        module.benchmark(root, root / "cache.sqlite", state="cold")
    with pytest.raises(ValueError, match="existence"):
        module.benchmark(root, tmp_path / "missing.sqlite", state="warm")
