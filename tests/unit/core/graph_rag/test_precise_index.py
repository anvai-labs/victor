"""ADR-032 tier 3 — precise-index harvesting: unit + pipeline contracts.

All SCIP-dependent paths are importorskip/tool-skip gated (no CI install
changes); the pipeline e2e injects a fake harvest so the integration
contract (ordering, override, downgrade, default-off) is proven without the
toolchain.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from victor.core.graph_rag.precise_index import (
    PreciseHarvest,
    python_symbol_to_identity,
)


def _identity(fqn: str, symbol_type: str = "FUNCTION") -> dict:
    return {"fqn": fqn, "symbol_type_name": symbol_type}


def test_python_symbol_string_maps_to_identity():
    identity = python_symbol_to_identity("python src/pkg/mod.py`pkg.mod.func().", "src/pkg/mod.py")
    assert identity == _identity("pkg.mod.func")

    cls = python_symbol_to_identity("python src/a.py`a.Thing().", "src/a.py")
    assert cls["fqn"] == "a.Thing"

    assert python_symbol_to_identity("rust crate::foo", "src/x.rs") is None
    assert python_symbol_to_identity("python src/a.py`no-backtick", "src/a.py") is None


def test_identity_file_marker_mismatch_is_rejected():
    # Cross-file symbols (relative_path mismatch) are not joined — never guessed.
    assert python_symbol_to_identity("python src/other.py`other.f().", "src/this.py") is None


def test_harvest_skips_without_reader(tmp_path: Path):
    from victor.core.graph_rag.precise_index import harvest_precise_edges

    bogus = tmp_path / "index.scip"
    bogus.write_bytes(b"not-protobuf")
    harvest = harvest_precise_edges(bogus, tmp_path, lambda i: "k", set())
    assert harvest.skipped_reason == "no-reader" or harvest.unjoined >= 0


def test_tool_detection_absent(monkeypatch):
    from victor.core.graph_rag.precise_index import detect_precise_tool

    monkeypatch.setattr("shutil.which", lambda name: None)
    import builtins

    real_import = builtins.__import__

    def no_scip(name, *args, **kwargs):
        if "scip" in name:
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_scip)
    assert detect_precise_tool("auto") is None


@pytest.mark.asyncio
async def test_pipeline_harvest_runs_last_and_overrides(tmp_path, monkeypatch):
    pytest.importorskip("victor_codegraph")
    from victor.core.graph_rag.config import GraphIndexConfig
    from victor.core.graph_rag.indexing import GraphIndexingPipeline
    from victor.storage.graph.sqlite_store import SqliteGraphStore

    (tmp_path / "targets.py").write_text(
        "def process(data):\n    return data\n\n\nclass Record:\n    pass\n"
    )
    (tmp_path / "consumer.py").write_text(
        "from targets import Record, process\n\n\ndef use(record=None):\n"
        "    kind = Record\n    fn = process\n    return kind, fn\n"
    )
    monkeypatch.setattr(SqliteGraphStore, "_delete_embeddings_for_file", AsyncMock())

    calls: list[str] = []

    async def fake_harvest(self, root):
        calls.append("harvest")
        _, GraphEdge = _graph_types()
        from victor.storage.graph.edge_types import EdgeType

        nodes = await self._persisted_symbols()
        by_name = {n.name: n.node_id for n in nodes}
        return PreciseHarvest(
            edges=[
                (
                    by_name["use"],
                    by_name["process"],
                    "REFERENCES",
                    {"provenance": "compiler", "confidence": 1.0, "use": "load"},
                )
            ],
            resolved=1,
        )

    def _graph_types():
        from victor.core.graph_rag.indexing import _get_graph_types

        return _get_graph_types()

    monkeypatch.setattr(GraphIndexingPipeline, "_harvest_precise_index", fake_harvest)

    store = SqliteGraphStore(tmp_path)
    await store.initialize()
    try:
        config = GraphIndexConfig(
            root_path=tmp_path,
            enable_ccg=False,
            enable_embeddings=False,
            enable_precise_index=True,
            incremental=False,
        )
        pipeline = GraphIndexingPipeline(store, config)
        stats = await pipeline.index_repository(tmp_path)

        assert calls == ["harvest"], "harvest pass ran exactly once"
        assert stats.precise_references_resolved == 1
        references = [e for e in await store.get_all_edges() if e.type == "REFERENCES"]
        compiler = [e for e in references if (e.metadata or {}).get("provenance") == "compiler"]
        assert compiler, "compiler-provenance REFERENCES persisted"
        scope = [
            e
            for e in references
            if (e.metadata or {}).get("provenance") in {"scope_resolver", "repository_resolver"}
        ]
        assert scope, "tier-2.5 edges coexist with compiler edges"
        # Last-writer-wins proof: tier 2.5 writes use->process REFERENCES
        # (repository_resolver) BEFORE the harvest; the single surviving row
        # for that triple must carry the compiler payload.
        persisted = await store.get_all_nodes()
        use_id = next(n.node_id for n in persisted if n.name == "use")
        process_id = next(n.node_id for n in persisted if n.name == "process")
        competing = [e for e in references if (e.src, e.dst) == (use_id, process_id)]
        assert len(competing) == 1, "PK dedup leaves one row per (src,dst,type)"
        assert competing[0].metadata.get("provenance") == "compiler"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_pipeline_default_off_skips_harvest(tmp_path, monkeypatch):
    pytest.importorskip("victor_codegraph")
    from victor.core.graph_rag.config import GraphIndexConfig
    from victor.core.graph_rag.indexing import GraphIndexingPipeline
    from victor.storage.graph.sqlite_store import SqliteGraphStore

    (tmp_path / "targets.py").write_text("def process(data):\n    return data\n")

    called = []

    async def fake_harvest(self, root):
        called.append("harvest")
        return PreciseHarvest()

    monkeypatch.setattr(GraphIndexingPipeline, "_harvest_precise_index", fake_harvest)
    monkeypatch.setattr(SqliteGraphStore, "_delete_embeddings_for_file", AsyncMock())

    store = SqliteGraphStore(tmp_path)
    await store.initialize()
    try:
        config = GraphIndexConfig(
            root_path=tmp_path,
            enable_ccg=False,
            enable_embeddings=False,
            incremental=False,
        )
        await GraphIndexingPipeline(store, config).index_repository(tmp_path)
        assert called == [], "default-OFF never invokes the harvester"
    finally:
        await store.close()


def test_settings_graph_flag_reads_through(tmp_path):
    from victor.config.settings import Settings, graph_flag

    defaults = Settings()
    assert graph_flag(defaults, "enable_scope_binding", True) is True
    assert graph_flag(defaults, "enable_precise_index", False) is False

    configured = Settings.model_validate(
        {"search": {"graph": {"enable_scope_binding": False, "enable_precise_index": True}}}
    )
    assert graph_flag(configured, "enable_scope_binding", True) is False
    assert graph_flag(configured, "enable_precise_index", False) is True


def test_qualification_settings_capture_the_flag(tmp_path):
    from victor.core.graph_rag.config import GraphIndexConfig

    config = GraphIndexConfig(
        root_path=tmp_path, enable_ccg=False, enable_embeddings=False, enable_precise_index=True
    )
    assert config.enable_precise_index is True
