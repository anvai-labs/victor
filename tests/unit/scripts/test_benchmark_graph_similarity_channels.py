"""Channel observations must retain the requested fingerprint configuration."""

import importlib.util
import json
from pathlib import Path


def test_token_fingerprint_observation_records_actual_option_and_engine_hashes(
    tmp_path, monkeypatch
):
    script = Path(__file__).resolve().parents[3] / "scripts/benchmark_graph_similarity_channels.py"
    spec = importlib.util.spec_from_file_location("source_channel_benchmark", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    received = []

    def analyze(*args, **kwargs):
        received.append(kwargs)
        return {"coverage_complete": True, "channels": {"tokens": {"fingerprints": []}}}

    output = tmp_path / "observation.json"
    monkeypatch.setattr(module, "analyze_source", analyze)
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            str(script),
            str(tmp_path),
            "--mode",
            "tokens",
            "--token-fingerprints",
            "--output",
            str(output),
        ],
    )
    module.main()
    result = json.loads(output.read_text())
    assert received[0]["token_options"].fingerprints is True
    assert result["invocation"]["token_fingerprints"] is True
    assert {"bounded_process.py", "source_inventory.py", "token_fingerprints.py"} <= set(
        result["implementation_hashes"]
    )
    assert not result["implementation_changed_during_run"]
