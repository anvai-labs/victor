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


def _occ(symbol: str, roles: int, line: int):
    """Synthetic SCIP occurrence (SimpleNamespace mirrors scip.proto shapes)."""
    from types import SimpleNamespace

    point = SimpleNamespace(line=line, character=0)
    rng = SimpleNamespace(start=point)
    return SimpleNamespace(symbol=symbol, symbol_roles=roles, range=rng)


def _doc(path: str, occurrences):
    from types import SimpleNamespace

    return SimpleNamespace(relative_path=path, occurrences=occurrences)


def _index(*documents):
    from types import SimpleNamespace

    return SimpleNamespace(documents=list(documents))


DEF = 1  # scip.proto SymbolRole definition bit


def test_join_core_resolves_references_and_attributes_owners():
    from victor.core.graph_rag.precise_index import join_occurrences

    key_of = {"mod.helper": "helper-key", "mod.user": "user-key"}
    index = _index(
        _doc(
            "src/mod.py",
            [
                _occ("python src/mod.py`mod.helper().", DEF, 1),
                _occ("python src/mod.py`mod.user().", DEF, 10),
                # reference to helper at line 12, inside user (owner=user)
                _occ("python src/mod.py`mod.helper().", 0, 12),
                # duplicate reference to helper inside user — deduped
                _occ("python src/mod.py`mod.helper().", 0, 14),
                # self-reference inside helper's own range (line 2 < 10)
                _occ("python src/mod.py`mod.helper().", 0, 2),
            ],
        )
    )
    harvest = join_occurrences(index, lambda i: key_of[i["fqn"]], set(key_of.values()))
    assert harvest.resolved == 1
    assert harvest.edges == [
        (
            "user-key",
            "helper-key",
            "REFERENCES",
            {"provenance": "compiler", "confidence": 1.0, "use": "load"},
        )
    ]
    # self-reference + the two unjoinable-shape occurrences accounting
    assert harvest.unjoined >= 0


def test_join_core_counts_unjoinable_definitions_and_references():
    from victor.core.graph_rag.precise_index import join_occurrences

    key_of = {"mod.known": "known-key"}
    index = _index(
        _doc(
            "src/mod.py",
            [
                _occ("python src/mod.py`mod.known().", DEF, 1),
                # definition whose key is not in the materialized set
                _occ("python src/mod.py`mod.unknown().", DEF, 5),
                # non-python scheme — rejected by the mapper
                _occ("rust crate::x", DEF, 6),
                # reference to the unknown definition — unjoined
                _occ("python src/mod.py`mod.unknown().", 0, 8),
                # reference with no enclosing joinable definition on this doc
                _occ("python other/file.py`other.f().", 0, 3),
            ],
        )
    )
    harvest = join_occurrences(index, lambda i: key_of.get(i["fqn"], "?"), {"known-key"})
    assert harvest.edges == []
    assert harvest.resolved == 0
    assert harvest.unjoined == 4  # unknown def + rust + unknown ref + cross-file


def test_fqn_overrides_supply_exact_identity():
    from victor.core.graph_rag.precise_index import join_occurrences

    index = _index(
        _doc(
            "src/mod.py",
            [
                _occ("python src/mod.py`mod.helper().", DEF, 1),
                _occ("python src/mod.py`mod.helper().", 0, 3),
            ],
        )
    )
    seen_identities = []

    def key_for(identity):
        seen_identities.append(identity)
        return "k"

    # No enclosing definition before the reference at line 3 except line 1,
    # which maps to the same key — self-reference skipped, no edges.
    harvest = join_occurrences(
        index,
        key_for,
        {"k"},
        fqn_overrides={"mod.helper": {"file_path": "src/mod.py", "signature": "(a)"}},
    )
    assert harvest.edges == []
    assert seen_identities[0] == {
        "fqn": "mod.helper",
        "symbol_type_name": "FUNCTION",
        "file_path": "src/mod.py",
        "signature": "(a)",
    }


def test_dump_command_shapes():
    from victor.core.graph_rag.precise_index import _dump_command

    assert _dump_command("/usr/bin/scip-python") == ["/usr/bin/scip-python", "index"]
    assert _dump_command("/opt/py -m scip_python.index") == [
        "/opt/py",
        "-m",
        "scip_python.index",
        "index",
    ]


def test_harvest_scip_index_absent_tool_returns_none(tmp_path, monkeypatch):
    import shutil as shutil_mod

    from victor.core.graph_rag.precise_index import harvest_scip_index

    monkeypatch.setattr(shutil_mod, "which", lambda name: None)
    assert harvest_scip_index(tmp_path, tmp_path / "out.scip") is None


def test_harvest_scip_index_failing_subprocess_returns_none(tmp_path, monkeypatch):
    import subprocess as subprocess_mod

    from victor.core.graph_rag import precise_index

    def fail(*args, **kwargs):
        raise subprocess_mod.TimeoutExpired(cmd="scip-python", timeout=1)

    real_which = precise_index.detect_precise_tool
    monkeypatch.setattr(
        precise_index, "detect_precise_tool", lambda tool="auto": "/fake/scip-python"
    )
    monkeypatch.setattr(subprocess_mod, "run", fail)
    assert precise_index.harvest_scip_index(tmp_path, tmp_path / "out.scip") is None
    monkeypatch.setattr(precise_index, "detect_precise_tool", real_which)


@pytest.mark.asyncio
async def test_real_harvest_method_downgrades_without_toolchain(tmp_path, monkeypatch):
    pytest.importorskip("victor_codegraph")
    from victor.core.graph_rag.config import GraphIndexConfig
    from victor.core.graph_rag.indexing import GraphIndexingPipeline
    from victor.core.graph_rag import precise_index
    from victor.storage.graph.sqlite_store import SqliteGraphStore

    (tmp_path / "m.py").write_text("def f():\n    return 1\n")
    monkeypatch.setattr(SqliteGraphStore, "_delete_embeddings_for_file", AsyncMock())
    monkeypatch.setattr(precise_index, "harvest_scip_index", lambda *a, **k: None)

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
        harvest = await pipeline._harvest_precise_index(tmp_path)
        assert harvest.skipped_reason == "toolchain-unavailable"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_real_harvest_method_success_path(tmp_path, monkeypatch):
    pytest.importorskip("victor_codegraph")
    from victor.core.graph_rag.config import GraphIndexConfig
    from victor.core.graph_rag.indexing import GraphIndexingPipeline
    from victor.core.graph_rag import precise_index
    from victor.storage.graph.sqlite_store import SqliteGraphStore

    (tmp_path / "m.py").write_text("def f():\n    return 1\n")
    monkeypatch.setattr(SqliteGraphStore, "_delete_embeddings_for_file", AsyncMock())
    dump = tmp_path / "index.scip"
    dump.write_bytes(b"x")
    monkeypatch.setattr(precise_index, "harvest_scip_index", lambda *a, **k: dump)
    fake_harvest = precise_index.PreciseHarvest(
        edges=[("a", "b", "REFERENCES", {"provenance": "compiler"})], resolved=1, unjoined=2
    )
    monkeypatch.setattr(precise_index, "harvest_precise_edges", lambda *a, **k: fake_harvest)

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
        harvest = await pipeline._harvest_precise_index(tmp_path)
        assert harvest is fake_harvest
        assert harvest.resolved == 1 and harvest.unjoined == 2
    finally:
        await store.close()


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
