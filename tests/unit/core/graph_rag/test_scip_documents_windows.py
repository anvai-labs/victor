"""Actual Windows immutable SCIP snapshots; qualified by the required OS matrix."""

import os

import pytest

pytestmark = pytest.mark.skipif(os.name != "nt", reason="actual Windows handle adapter required")


def write_index(tmp_path):
    from victor.core.graph_rag._scip import scip_pb2

    index = scip_pb2.Index()
    doc = index.documents.add(relative_path="src/a.rs", text="fn a() {}\n", position_encoding=1)
    doc.occurrences.add(range=[0, 3, 4], symbol="rust-analyzer cargo fixture 1 a().")
    doc.symbols.add(symbol="ignored metadata")
    path = tmp_path / "index.scip"
    path.write_bytes(index.SerializeToString())
    return path


def test_windows_snapshot_preserves_occurrences_without_definition_arena(tmp_path):
    from victor.core.graph_rag.scip_documents import ScipDocumentReader
    from victor.core.graph_rag.source_reader import project_source_reader

    path = write_index(tmp_path)
    with (
        project_source_reader(tmp_path) as root,
        ScipDocumentReader(path, source_root=root) as reader,
    ):
        doc = reader["src/a.rs"]
        assert doc.text == "fn a() {}\n"
        assert list(doc.occurrences[0].range) == [0, 3, 4]
        assert doc.occurrences[0].symbol.endswith("a().")
        assert not doc.symbols


def test_windows_artifact_is_an_immutable_captured_snapshot(tmp_path):
    from victor.core.graph_rag.scip_documents import ScipDocumentReader

    path = write_index(tmp_path)
    with ScipDocumentReader(path) as reader:
        captured = reader.stat.st_size
        path.write_bytes(b"replacement after snapshot")
        assert reader["src/a.rs"].text == "fn a() {}\n"
        assert reader.stat.st_size == captured
    with pytest.raises(ValueError, match="closed"):
        reader["src/a.rs"]


def test_windows_oversized_snapshot_is_rejected_before_protobuf_decoding(tmp_path):
    from victor.core.graph_rag.scip_documents import ScipDocumentReader

    path = write_index(tmp_path)
    with pytest.raises(ValueError, match="bounded regular"):
        ScipDocumentReader(path, max_bytes=1)
