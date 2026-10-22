"""Tier-3 dump-artifact cache — key semantics, reuse, and safety.

The cache is derived state: deterministic per (provider, tool, corpus, schema),
so every test asserts the invalidation surface a real harvest relies on.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from victor.core.graph_rag.precise_cache import (
    atomic_dump_path,
    cache_paths,
    dump_cache_key,
    load_cached,
    slice_manifest_for_extensions,
    store,
)
from victor.core.graph_rag.precise_providers import (
    RustAnalyzerProvider,
    ScipPythonProvider,
    ScipTypescriptProvider,
    extensions_for_provider,
)


def test_key_is_order_independent_and_stable(tmp_path):
    tool = tmp_path / "scip-tool"
    tool.write_text("bin", encoding="utf-8")
    a = dump_cache_key("scip-python", str(tool), {"x.py": "1", "y.py": "2"})
    b = dump_cache_key("scip-python", str(tool), {"y.py": "2", "x.py": "1"})
    assert a == b
    assert dump_cache_key("scip-python", str(tool), {"x.py": "1"}) != a


def test_key_sensitivity(tmp_path):
    tool = tmp_path / "scip-tool"
    tool.write_text("bin", encoding="utf-8")
    base = dump_cache_key("scip-python", str(tool), {"x.py": "1"})
    # Content change, file add, file delete, provider, tool path, tool stat.
    assert dump_cache_key("scip-python", str(tool), {"x.py": "2"}) != base
    assert dump_cache_key("scip-python", str(tool), {"x.py": "1", "n.py": "9"}) != base
    assert dump_cache_key("scip-python", str(tool), {}) != base
    assert dump_cache_key("rust-analyzer", str(tool), {"x.py": "1"}) != base
    other = tmp_path / "other-tool"
    other.write_text("bin", encoding="utf-8")
    assert dump_cache_key("scip-python", str(other), {"x.py": "1"}) != base
    tool.write_text("bin2", encoding="utf-8")  # size change -> stat change
    assert dump_cache_key("scip-python", str(tool), {"x.py": "1"}) != base


def test_missing_tool_stats_to_zero_not_crash(tmp_path):
    assert dump_cache_key("p", str(tmp_path / "absent"), {}) is not None


def _seed_artifact(root: Path, provider: str, key: str, payload: bytes) -> Path:
    artifact, _ = cache_paths(root, provider)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(payload)
    store(root, provider, artifact, key)
    return artifact


def test_reuse_roundtrip_and_misses(tmp_path):
    tool = tmp_path / "t"
    tool.write_text("bin", encoding="utf-8")
    key = dump_cache_key("p", str(tool), {"x.py": "1"})
    artifact = _seed_artifact(tmp_path, "p", key, b"SCIPBYTES")
    assert load_cached(tmp_path, "p", key) == artifact
    # Different key -> miss.
    assert load_cached(tmp_path, "p", "other-key") is None
    # Truncated artifact (size guard) -> miss, never a bad join input.
    artifact.write_bytes(b"SCI")
    assert load_cached(tmp_path, "p", key) is None
    # Malformed sidecar -> miss, never an error.
    artifact.write_bytes(b"SCIPBYTES")
    sidecar = cache_paths(tmp_path, "p")[1]
    sidecar.write_text("{not json", encoding="utf-8")
    assert load_cached(tmp_path, "p", key) is None


def test_atomic_dump_path_flow(tmp_path):
    final, tmp = atomic_dump_path(tmp_path, "p")
    assert final.name == "p.scip"
    # PROCESS-UNIQUE tmp: concurrent harvests never interleave on one file
    # (a raced fixed name once persisted a truncated artifact that parsed
    # as a valid EMPTY index and harvested zero edges).
    assert tmp.name.startswith("p.scip.") and tmp.name.endswith(".tmp")
    assert str(os.getpid()) in tmp.name
    assert tmp != final
    tmp.write_bytes(b"dump")
    os.replace(tmp, final)
    store(tmp_path, "p", final, "k")
    sidecar = json.loads(cache_paths(tmp_path, "p")[1].read_text(encoding="utf-8"))
    assert sidecar == {"schema": "precise-dump-cache-v1", "key": "k", "artifact_bytes": 4}


def test_store_refuses_zero_byte_artifact(tmp_path):
    """A truncated protobuf parses as a VALID empty index — a zero-byte
    artifact must never enter the cache (it would harvest zero edges)."""
    artifact, _ = cache_paths(tmp_path, "p")
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b"")
    import pytest

    with pytest.raises(ValueError):
        store(tmp_path, "p", artifact, "k")


def test_store_sidecar_is_atomic_no_partial_json(tmp_path):
    artifact, _ = cache_paths(tmp_path, "p")
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b"x")
    store(tmp_path, "p", artifact, "k")
    assert not cache_paths(tmp_path, "p")[1].with_suffix(".json.tmp").exists()


def test_manifest_slice_by_language(tmp_path):
    manifest = {"a.py": "1", "b.ts": "2", "c.rs": "3", "d.tsx": "4", "e.md": "5"}
    assert slice_manifest_for_extensions(manifest, frozenset({".py"})) == {"a.py": "1"}
    assert slice_manifest_for_extensions(manifest, frozenset({".ts", ".tsx"})) == {
        "b.ts": "2",
        "d.tsx": "4",
    }
    assert slice_manifest_for_extensions(None, frozenset({".py"})) == {}
    assert slice_manifest_for_extensions(manifest, frozenset()) == {}


def test_provider_timeout_defaults_and_extensions():
    # Measured costs differ by ~40x; the ceilings must keep that ordering.
    assert ScipPythonProvider.default_timeout_seconds == 1200
    assert ScipTypescriptProvider.default_timeout_seconds == 600
    assert RustAnalyzerProvider.default_timeout_seconds == 120
    assert extensions_for_provider("scip-python") == frozenset({".py"})
    assert ".tsx" in extensions_for_provider("scip-typescript")
    assert extensions_for_provider("rust-analyzer") == frozenset({".rs"})
    assert extensions_for_provider("nope") == frozenset()


def test_config_sentinel_zero_means_provider_default():
    from victor.core.graph_rag.config import GraphIndexConfig

    assert GraphIndexConfig.precise_index_timeout_seconds == 0
    # Resolution rule as the harvest applies it.
    for config_timeout, provider, expected in (
        (0, ScipPythonProvider(), 1200),
        (0, RustAnalyzerProvider(), 120),
        (900, RustAnalyzerProvider(), 900),  # explicit override wins uniformly
    ):
        assert (config_timeout or provider.default_timeout_seconds) == expected
