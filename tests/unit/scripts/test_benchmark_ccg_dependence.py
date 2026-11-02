"""Measurements must describe actual dispatch, bounded selection and stable inputs."""

import importlib.util
from pathlib import Path

import pytest


def load():
    path = Path(__file__).resolve().parents[3] / "scripts/benchmark_ccg_dependence.py"
    spec = importlib.util.spec_from_file_location("ccg_dependence_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def test_source_selection_and_python_dispatch_are_explicit(tmp_path):
    source = tmp_path / "code"
    source.mkdir()
    (source / "a.py").write_text("def f():\n return outer(inner(), sibling())\n")
    (source / "b.py").write_text("def g():\n return 0\n")
    result = await load().benchmark(tmp_path, backend="python", scope="code", runs=1, max_files=1)
    assert result["selection"]["truncated"]
    assert len(result["source_manifest"]) == 1
    assert result["program_semantics_verified"] is False
    row = result["runs"][0]["files"][0]
    assert row["backend"] == "python"
    assert row["nodes"] == row["unique_node_ids"] > 0


async def test_measurement_rejects_a_source_snapshot_changed_during_build(tmp_path, monkeypatch):
    source = tmp_path / "code"
    source.mkdir()
    file = source / "a.py"
    file.write_text("def f():\n return 0\n")
    module = load()
    original = module.PythonBuilder.build_ccg_for_file

    async def changed(*args, **kwargs):
        result = await original(*args, **kwargs)
        file.write_text("def f():\n return 1\n")
        return result

    monkeypatch.setattr(module.PythonBuilder, "build_ccg_for_file", changed)
    with pytest.raises(RuntimeError, match="snapshot changed"):
        await module.benchmark(tmp_path, backend="python", scope="code", runs=1)


@pytest.mark.parametrize(
    "options",
    [
        {"runs": 0},
        {"runs": 21},
        {"backend": "unknown"},
        {"max_files": 0},
        {"max_files": 5001},
        {"max_bytes": 0},
        {"scope": "../"},
        {"scope": "/etc"},
    ],
)
async def test_invalid_or_escaping_measurement_inputs_are_rejected(tmp_path, options):
    with pytest.raises(ValueError):
        await load().benchmark(tmp_path, **options)
