"""Failure probes must preserve explicit incomplete results and offline options."""

import importlib.util
from pathlib import Path


def test_failure_probe_constructs_once_and_preserves_each_error():
    path = Path(__file__).resolve().parents[3] / "scripts/benchmark_body_model_failure.py"
    spec = importlib.util.spec_from_file_location("body_failure_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []

    class InvalidModel:
        def get_sentence_embedding_dimension(self):
            return 384

    def factory(*args, **kwargs):
        calls.append(kwargs)
        return InvalidModel()

    result = module.benchmark(factory)
    assert result["model_constructions"] == 1
    assert len(result["errors"]) == result["functions"] == 2
    assert not result["coverage_complete"]
    assert not result["accuracy_benchmark"]
    assert calls[0]["local_files_only"] is True
    assert calls[0]["trust_remote_code"] is False
    assert len(result["implementation"]["body_clones.py"]) == 64
