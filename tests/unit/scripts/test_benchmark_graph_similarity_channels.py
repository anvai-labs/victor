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
            "--scope",
            "src/query/",
            "--same-file",
            "--exclude-docs",
            "--token-min-lines",
            "3",
            "--token-memory-mib",
            "2048",
            "--token-max-file-bytes",
            "8388608",
            "--token-timeout",
            "120",
            "--token-exclude",
            "aliases/**",
            "--output",
            str(output),
        ],
    )
    module.main()
    result = json.loads(output.read_text())
    assert received[0]["token_options"].fingerprints is True
    assert received[0]["token_options"].min_lines == 3
    assert received[0]["token_options"].max_rss_bytes == 2 * 1024**3
    assert received[0]["token_options"].max_file_bytes == 8 * 1024**2
    assert received[0]["token_options"].timeout_seconds == 120
    assert received[0]["token_options"].exclude_patterns == ("aliases/**",)
    assert received[0]["scope"] == "src/query/"
    assert received[0]["include_same_file"] is True
    assert received[0]["include_docs"] is False
    assert result["invocation"]["token_fingerprints"] is True
    assert result["invocation"]["token_memory_mib"] == 2048
    assert result["invocation"]["token_max_file_bytes"] == 8388608
    assert result["invocation"]["token_timeout"] == 120
    assert result["invocation"]["token_exclude"] == ["aliases/**"]
    assert {
        "bounded_process.py",
        "source_inventory.py",
        "exclude_patterns.py",
        "token_fingerprints.py",
        "token_coverage.py",
        "copy_baseline.py",
        "copy_artifacts.py",
    } <= set(result["implementation_hashes"])
    assert not result["implementation_changed_during_run"]
