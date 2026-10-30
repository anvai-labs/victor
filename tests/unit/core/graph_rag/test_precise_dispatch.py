"""Registry/dispatch integration, with synthetic producer outputs and no compilers.

Exercise every registered provider in the real harvest loop, not just a Python
project with other provider classes patched but never selected.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from victor.core.graph_rag.indexing import GraphIndexingPipeline
from victor.core.graph_rag.precise_providers import DumpSkipped, all_providers
from victor.storage.graph.protocol import GraphNode

FILES = [
    "src/a.py",
    "ui/a.ts",
    "ui/b.js",
    "src/a.rs",
    "native/a.cpp",
    "api/A.java",
    "lib/a.rb",
    "cmd/a.go",
]
EXPECTED = {
    "scip-python",
    "scip-typescript",
    "rust-analyzer",
    "scip-clang",
    "scip-java",
    "scip-ruby",
    "scip-go",
}


def _pipeline():
    pipeline = object.__new__(GraphIndexingPipeline)
    pipeline.config = SimpleNamespace(precise_index_tool="auto", precise_index_timeout_seconds=0)
    pipeline._codegraph_repository = None
    pipeline._persisted_symbols = AsyncMock(
        return_value=[GraphNode(str(i), "function", "f", file) for i, file in enumerate(FILES)]
        + [GraphNode("py-g", "function", "g", "src/a.py")]
    )
    return pipeline


def _producers(
    monkeypatch,
    *,
    absent=None,
    skip=None,
    failure=None,
    raised=None,
    phase="dump",
    error_type=RuntimeError,
):
    from victor.core.graph_rag._scip import scip_pb2

    calls = []
    lookups = []

    def which(name):
        lookups.append(name)
        return None if name == absent else f"/installed/bin/{name}"

    monkeypatch.setattr("victor.core.graph_rag.precise_providers.shutil.which", which)
    monkeypatch.setattr(
        "victor.core.graph_rag.harvest_environment.build_python_environment", lambda *args: None
    )
    for provider in all_providers():

        def dump(self, tool, root, output_path, *, timeout_seconds, **kwargs):
            calls.append((self.name, tool, timeout_seconds))
            if self.name == raised and phase == "dump":
                raise error_type("synthetic producer failure")
            if self.name == skip:
                return DumpSkipped("synthetic-marker-absent")
            if self.name == failure:
                return None
            index = scip_pb2.Index()
            index.metadata.tool_info.name = self.name  # non-empty valid SCIP artifact
            if self.name == "scip-python":
                doc = index.documents.add(relative_path="src/a.py")
                for name, line, roles in (("f", 0, 1), ("g", 4, 1), ("f", 5, 0)):
                    occ = doc.occurrences.add(
                        symbol=f"scip-python python dispatch 1 `src.a`/{name}().",
                        symbol_roles=roles,
                    )
                    occ.range.extend([line, 0])
            output_path.write_bytes(index.SerializeToString())
            return output_path

        monkeypatch.setattr(type(provider), "dump", dump)
        if phase == "detect" and provider.name == raised:

            def detect(self, hint="auto"):
                raise error_type("synthetic detection failure")

            monkeypatch.setattr(type(provider), "detect", detect)
    if phase == "read":
        from victor.core.graph_rag import precise_index

        original = precise_index.harvest_precise_edges

        def read(path, *args, **kwargs):
            if path.name == raised + ".scip":
                raise error_type("synthetic reader failure")
            return original(path, *args, **kwargs)

        monkeypatch.setattr(precise_index, "harvest_precise_edges", read)
    return calls, lookups


async def test_all_registered_compilers_are_auto_selected_once_without_supplied_scip(
    tmp_path, monkeypatch
):
    pytest.importorskip("google.protobuf")
    calls, lookups = _producers(monkeypatch)
    result = await _pipeline()._harvest_precise_index(tmp_path)
    assert {name for name, _, _ in calls} == EXPECTED
    assert len(calls) == len(EXPECTED)  # TS and JS share a single provider invocation
    assert set(result.per_language) == EXPECTED
    assert result.per_language["scip-python"].startswith("ok:1 edges")
    assert all(value.startswith("ok:") for value in result.per_language.values())
    assert [(src, dst) for src, dst, _, _ in result.edges] == [("py-g", "0")]
    assert result.skipped_reason is None
    assert set(lookups) >= {
        "scip-python",
        "scip-typescript",
        "rust-analyzer",
        "scip-clang",
        "scip-java",
        "scip-ruby",
        "bundle",
        "scip-go",
    }
    assert all(timeout > 0 for _, _, timeout in calls)
    assert next(tool for name, tool, _ in calls if name == "scip-ruby") == "/installed/bin/bundle"


async def test_absent_skipped_and_failed_compilers_do_not_suppress_other_outcomes(
    tmp_path, monkeypatch
):
    pytest.importorskip("google.protobuf")
    calls, _ = _producers(
        monkeypatch, absent="scip-java", skip="scip-typescript", failure="scip-clang"
    )
    result = await _pipeline()._harvest_precise_index(tmp_path)
    assert set(result.per_language) == EXPECTED
    assert result.per_language["scip-java"] == "toolchain-absent"
    assert result.per_language["scip-typescript"] == "synthetic-marker-absent"
    assert result.per_language["scip-clang"] == "dump-failed"
    assert result.per_language["scip-go"].startswith("ok:")
    assert [(src, dst) for src, dst, _, _ in result.edges] == [("py-g", "0")]
    assert "scip-java" not in {name for name, _, _ in calls}
    assert result.skipped_reason is None


@pytest.mark.parametrize("phase", ["detect", "dump", "read"])
async def test_unexpected_provider_error_preserves_other_language_results(
    tmp_path, monkeypatch, phase
):
    pytest.importorskip("google.protobuf")
    calls, _ = _producers(monkeypatch, raised="scip-typescript", phase=phase)
    result = await _pipeline()._harvest_precise_index(tmp_path)
    assert result.per_language["scip-typescript"] == "error:RuntimeError"
    assert set(result.per_language) == EXPECTED
    assert result.per_language["scip-python"].startswith("ok:")
    assert result.per_language["scip-go"].startswith("ok:")
    assert result.skipped_reason is None
    assert [(src, dst) for src, dst, _, _ in result.edges] == [("py-g", "0")]
    assert "scip-go" in {name for name, _, _ in calls}


@pytest.mark.parametrize("phase", ["detect", "dump"])
async def test_cancellation_propagates_instead_of_becoming_provider_failure(
    tmp_path, monkeypatch, phase
):
    pytest.importorskip("google.protobuf")
    calls, _ = _producers(
        monkeypatch, raised="scip-typescript", phase=phase, error_type=asyncio.CancelledError
    )
    with pytest.raises(asyncio.CancelledError):
        await _pipeline()._harvest_precise_index(tmp_path)
    assert "scip-go" not in {name for name, _, _ in calls}
