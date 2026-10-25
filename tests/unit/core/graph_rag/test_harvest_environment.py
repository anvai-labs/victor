"""Harvest environment constraint: import closure, env-file schema,
dump-based reflection, and the decline-never-break contract."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

from victor.core.graph_rag.harvest_environment import (
    build_python_environment,
    collect_import_roots,
    external_packages_from_index,
    reflect_harvest,
    repo_local_roots,
    select_distributions,
    write_environment_json,
)


@dataclass
class _FakeDist:
    name: str
    version: str = "1.0"
    requires: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)


class _FakeMetadata:
    def __init__(self, name):
        self.name = name

    def __getitem__(self, key):
        return self.name if key == "Name" else None


def _reader(catalog: dict[str, _FakeDist]):
    def read(name: str):
        key = name.strip().replace("_", "-").lower()
        if key not in catalog:
            raise LookupError(name)
        return catalog[key]

    return read


def _dist(catalog: dict, dist: _FakeDist) -> _FakeDist:
    dist.metadata = _FakeMetadata(dist.name)  # type: ignore[attr-defined]
    catalog[dist.name.replace("_", "-").lower()] = dist
    return dist


# --- import collection ------------------------------------------------------


def test_collect_import_roots_and_local_filter(tmp_path):
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "__init__.py").write_text("")
    (tmp_path / "app" / "main.py").write_text(
        "import os\n"
        "import numpy as np\n"
        "from pydantic import BaseModel\n"
        "from . import sibling\n"
        "from app import thing\n"
    )
    roots = collect_import_roots(tmp_path)
    assert roots == {"os", "numpy", "pydantic", "app"}
    assert repo_local_roots(tmp_path) == {"app"}
    # stdlib + local removed leaves exactly the third-party surface
    import sys

    assert roots - set(sys.stdlib_module_names) - repo_local_roots(tmp_path) == {
        "numpy",
        "pydantic",
    }


def test_collect_import_roots_skips_build_dirs(tmp_path):
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "x.py").write_text("import junksurface\n")
    assert collect_import_roots(tmp_path) == set()


def test_repo_local_roots_covers_nested_monorepo_packages(tmp_path):
    pkg = tmp_path / "victor-codegraph" / "victor_codegraph"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    vertical = tmp_path / "verticals" / "victor-coding" / "victor_coding"
    vertical.mkdir(parents=True)
    (vertical / "__init__.py").write_text("")
    assert {"victor_codegraph", "victor_coding"} <= repo_local_roots(tmp_path)


# --- distribution selection -------------------------------------------------


def test_select_distributions_closure_and_unmapped():
    catalog: dict[str, _FakeDist] = {}
    _dist(catalog, _FakeDist("pydantic", requires=["typing-extensions"]))
    _dist(catalog, _FakeDist("typing-extensions", version="4.9"))
    _dist(catalog, _FakeDist("Django", version="5.0"))

    selected, unmapped, provenance = select_distributions(
        {"pydantic", "not-installed-lib"},
        packages_distributions={"pydantic": ["pydantic"], "not-installed-lib": []},
        distribution_reader=_reader(catalog),
    )
    assert selected == {"pydantic", "typing-extensions"}
    assert unmapped == ["not-installed-lib"]
    assert provenance["pydantic"] == "import:pydantic"
    assert provenance["typing-extensions"] == "dep:pydantic"


def test_select_distributions_dedupes_case_variants():
    catalog: dict[str, _FakeDist] = {}
    _dist(catalog, _FakeDist("Jinja2", version="3.1"))
    selected, _, _ = select_distributions(
        {"jinja2"},
        packages_distributions={"jinja2": ["Jinja2", "jinja2"]},
        distribution_reader=_reader(catalog),
    )
    assert selected == {"jinja2"}


# --- env file schema + decline ----------------------------------------------


def test_write_environment_json_schema_and_roundtrip(tmp_path):
    catalog: dict[str, _FakeDist] = {}
    _dist(catalog, _FakeDist("pydantic", version="2.6", files=["pydantic/__init__.py"]))
    _dist(catalog, _FakeDist("Jinja2", version="3.1"))
    _dist(catalog, _FakeDist("jinja2", version="3.1"))
    path = write_environment_json(
        tmp_path / "env.json",
        {"pydantic", "Jinja2", "jinja2"},
        distribution_reader=_reader(catalog),
    )
    assert path is not None
    payload = json.loads(path.read_text())
    names = [item["name"].lower() for item in payload]
    assert sorted(names) == ["jinja2", "pydantic"]  # exactly one jinja entry
    pydantic_entry = next(item for item in payload if item["name"] == "pydantic")
    assert pydantic_entry["files"] == ["pydantic/__init__.py"]


def test_write_environment_json_declines_on_empty(tmp_path):
    assert write_environment_json(tmp_path / "env.json", set()) is None
    assert not (tmp_path / "env.json").exists()


def test_build_python_environment_end_to_end(tmp_path, monkeypatch):
    import sys as sys_module

    monkeypatch.setattr(
        "victor.core.graph_rag.harvest_environment.select_distributions",
        lambda roots, **kw: ({"pydantic"}, [], {"pydantic": "import:pydantic"}),
    )
    catalog: dict[str, _FakeDist] = {}
    _dist(catalog, _FakeDist("pydantic", version="2.6", files=["pydantic/__init__.py"]))
    monkeypatch.setattr(
        "importlib.metadata.distribution",
        _reader(catalog),
    )
    (tmp_path / "src.py").write_text("import pydantic\nimport os\n")

    path = build_python_environment(tmp_path, tmp_path / ".victor" / "env.json")
    assert path is not None
    payload = json.loads(path.read_text())
    assert payload == [{"name": "pydantic", "version": "2.6", "files": ["pydantic/__init__.py"]}]


def test_build_python_environment_declines_for_pure_stdlib(tmp_path):
    (tmp_path / "only.py").write_text("import os\nimport json\n")
    assert build_python_environment(tmp_path, tmp_path / "env.json") is None


# --- dump reflection ---------------------------------------------------------


def test_external_packages_from_index_and_reflection():
    def index(symbol_names: list[str], occurrences: int):
        symbols = [SimpleNamespace(symbol=name) for name in symbol_names]
        docs = [SimpleNamespace(occurrences=[0] * occurrences)]
        return SimpleNamespace(external_symbols=symbols, documents=docs)

    baseline = index(
        [
            "scip-python python pydantic 2.6 `pydantic`/BaseModel().",
            "scip-python python numpy 1.26 `numpy`/array().",
            "victor `victor`/core/thing().",
        ],
        occurrences=100,
    )
    constrained = index(
        [
            "scip-python python pydantic 2.6 `pydantic`/BaseModel().",
            "scip-python python fastapi 0.110 `fastapi`/APIRouter().",
        ],
        occurrences=80,
    )
    assert external_packages_from_index(baseline) == {"pydantic", "numpy"}
    reflection = reflect_harvest(baseline, constrained, {"pydantic", "fastapi"})
    assert reflection["missing_vs_baseline"] == ["numpy"]
    assert reflection["newly_resolved"] == ["fastapi"]
    assert reflection["baseline_occurrences"] == 100
    assert reflection["constrained_occurrences"] == 80
    assert reflection["selected_distributions"] == 2


def test_reflect_or_redump_covers_and_falls_back(monkeypatch, tmp_path):
    """The orchestration reflection: reader-stubbed both paths — clean
    coverage keeps the dump (no re-dump), a regression re-dumps once."""
    import asyncio
    from types import SimpleNamespace as NS

    from victor.core.graph_rag import indexing as indexing_module

    def stub_index(packages: list[str], occurrences: int):
        symbols = [NS(symbol=f"scip-python python {pkg} 1.0 `m`/x().") for pkg in packages]
        return NS(external_symbols=symbols, documents=[NS(occurrences=[0] * occurrences)])

    calls = {"redumps": 0}

    class _Provider:
        name = "scip-python"
        languages = frozenset({"python"})

        def dump(self, *args, **kwargs):
            calls["redumps"] += 1
            return tmp_path / "redump.scip"

    pipeline = indexing_module.GraphIndexingPipeline.__new__(indexing_module.GraphIndexingPipeline)

    async def run(baseline_packages, constrained_packages):
        import victor.core.graph_rag.precise_index as precise_module

        def fake_read(path):
            if "baseline" in str(path):
                return stub_index(baseline_packages, 10)
            return stub_index(constrained_packages, 10)

        monkeypatch.setattr(precise_module, "_read_index", fake_read)
        baseline_path = tmp_path / "baseline.scip"
        baseline_path.write_text("x", encoding="utf-8")
        return await pipeline._reflect_or_redump(
            _Provider(),
            "scip-python",
            tmp_path,
            baseline_path,
            tmp_path / "fresh.scip",
            dumped=tmp_path / "fresh.scip",
            timeout_seconds=1,
        )

    # Clean coverage: no missing packages -> original dump kept, no re-dump.
    result = asyncio.run(run(["pydantic"], ["pydantic", "numpy"]))
    assert calls["redumps"] == 0
    assert str(result) == str(tmp_path / "fresh.scip")

    # Regression: a baseline package is lost -> exactly ONE unconstrained
    # re-dump, and its result replaces the constrained dump.
    result = asyncio.run(run(["pydantic", "numpy"], ["pydantic"]))
    assert calls["redumps"] == 1
    assert str(result) == str(tmp_path / "redump.scip")


def test_reflect_or_redump_without_reader_keeps_dump(monkeypatch, tmp_path):
    import asyncio

    import victor.core.graph_rag.precise_index as precise_module
    from victor.core.graph_rag import indexing as indexing_module

    monkeypatch.setattr(precise_module, "_read_index", lambda path: None)
    pipeline = indexing_module.GraphIndexingPipeline.__new__(indexing_module.GraphIndexingPipeline)
    baseline = tmp_path / "baseline.scip"
    baseline.write_text("x", encoding="utf-8")
    dumped = tmp_path / "fresh.scip"

    class _Provider:
        name = "scip-python"
        languages = frozenset({"python"})

        def dump(self, *args, **kwargs):  # pragma: no cover - must not run
            raise AssertionError("no re-dump without a reader")

    result = asyncio.run(
        pipeline._reflect_or_redump(
            _Provider(), "t", tmp_path, baseline, dumped, dumped, timeout_seconds=1
        )
    )
    assert result is dumped
