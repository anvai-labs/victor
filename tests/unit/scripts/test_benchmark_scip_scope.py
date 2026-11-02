"""Scoped observations retain answer parity and reject changed inputs."""

import hashlib
import importlib.util
import os
from pathlib import Path

import pytest


def probe(tmp_path, monkeypatch):
    from victor.core.graph_rag.structural_clones import StructuralReport, extract_units

    script = Path(__file__).resolve().parents[3] / "scripts/benchmark_scip_scope.py"
    spec = importlib.util.spec_from_file_location("scip_scope_probe", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    file = tmp_path / "a.py"
    source = "def f(x):\n return x+1\n"
    file.write_text(source)
    index = tmp_path / "input.scip"
    index.write_bytes(b"immutable input")
    report = StructuralReport(
        [],
        [],
        {"source_files": 1, "selected_regions": 1},
        units=extract_units("a.py", source, None, min_nodes=1),
        source_manifest={"a.py": hashlib.sha256(source.encode()).hexdigest()},
    )
    monkeypatch.setattr(module, "scan", lambda *a, **kw: report)
    return module, index, file


def test_scoped_observation_preserves_answer_and_input_identity(tmp_path, monkeypatch):
    module, index, _ = probe(tmp_path, monkeypatch)
    first = module.benchmark(tmp_path, index, scope="a.py")
    second = module.benchmark(tmp_path, index, scope="a.py")
    for key in ("source_manifest", "unit_digest", "family_digest", "index", "implementation"):
        assert first[key] == second[key]
    assert first["units"] == 1
    assert not first["source_freshness_proof"]
    assert not first["accuracy_benchmark"]


@pytest.mark.parametrize("changed", ["source", "index"])
def test_changed_input_rejects_measurement(tmp_path, monkeypatch, changed):
    module, index, file = probe(tmp_path, monkeypatch)
    original = module.scan

    def change(*args, **kwargs):
        report = original(*args, **kwargs)
        (file if changed == "source" else index).write_text("changed")
        return report

    monkeypatch.setattr(module, "scan", change)
    with pytest.raises(RuntimeError, match="changed"):
        module.benchmark(tmp_path, index, scope="a.py")


@pytest.mark.parametrize("kind", ["symlink", "fifo"])
def test_artifact_swap_cannot_follow_a_link_or_block(tmp_path, monkeypatch, kind):
    module, index, _ = probe(tmp_path, monkeypatch)
    original = os.open
    replacement = tmp_path / "replacement"
    replacement.write_text("other bytes")
    swapped = []

    def swap(name, flags, *args, **kwargs):
        if name == index.name:
            swapped.append(True)
            index.unlink()
            if kind == "symlink":
                index.symlink_to(replacement)
            else:
                os.mkfifo(index)
        return original(name, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", swap)
    monkeypatch.setattr(os, "supports_dir_fd", os.supports_dir_fd | {swap})
    with pytest.raises((OSError, ValueError)):
        module.artifact(index)
    assert swapped == [True]


def test_growing_input_cannot_extend_captured_hash_budget(tmp_path, monkeypatch):
    module, index, _ = probe(tmp_path, monkeypatch)
    initial_size = index.stat().st_size
    original = os.read
    sizes = []

    def grow(descriptor, size):
        sizes.append(size)
        with index.open("ab") as stream:
            stream.write(b"x" * 100)
        return original(descriptor, size)

    monkeypatch.setattr(os, "read", grow)
    with pytest.raises(RuntimeError, match="changed"):
        module.artifact(index)
    assert sizes == [initial_size, 1]
