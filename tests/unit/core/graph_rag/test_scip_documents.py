"""Bounded lazy SCIP inputs preserve consumed documents, not whole arenas."""

import os
from pathlib import Path
import tracemalloc

import pytest

try:
    from victor.core.graph_rag._scip import scip_pb2
except ImportError:
    if os.environ.get("VICTOR_REQUIRE_SCIP_READER_TESTS") == "1":
        raise
    pytest.skip("optional protobuf runtime unavailable", allow_module_level=True)

from victor.core.graph_rag.scip_documents import ScipDocumentReader
from victor.core.graph_rag.source_reader import project_source_reader
from victor.core.graph_rag.scip_documents import _encoded


def field(number, value):
    return _encoded(number * 8 + 2) + _encoded(len(value)) + value


def write_document(tmp_path, raw):
    path = tmp_path / "index.scip"
    path.write_bytes(field(2, raw))
    return path


def index_file(root: Path, count=3):
    index = scip_pb2.Index()
    for i in range(count):
        doc = index.documents.add(relative_path=f"src/{i}.rs", language="rust", position_encoding=1)
        doc.occurrences.add(range=[0, 3, 7], symbol=f"rust-analyzer cargo test 1 f{i}().")
        doc.symbols.add(symbol="ignored", documentation=["large definition metadata" * 1000])
    path = root / "index.scip"
    path.write_bytes(index.SerializeToString())
    return path, index


def test_lazy_reader_matches_consumed_fields_and_bounds_decoding(tmp_path):
    path, original = index_file(tmp_path)
    with ScipDocumentReader(path) as reader:
        assert sorted(reader) == ["src/0.rs", "src/1.rs", "src/2.rs"]
        assert reader.decoded_documents == 0
        selected = reader["src/1.rs"]
        assert selected.relative_path == original.documents[1].relative_path
        assert selected.position_encoding == original.documents[1].position_encoding
        assert [(list(o.range), o.symbol) for o in selected.occurrences] == [
            (list(o.range), o.symbol) for o in original.documents[1].occurrences
        ]
        assert not selected.symbols  # Unconsumed definition arenas are never materialized.
        assert reader.decoded_documents == 1
    with pytest.raises(ValueError, match="closed"):
        reader["src/0.rs"]


def test_selected_project_root_rejects_artifact_parent_symlinks(tmp_path):
    project, outside = tmp_path / "project", tmp_path / "outside"
    project.mkdir()
    outside.mkdir()
    path, _ = index_file(outside)
    (project / ".victor").symlink_to(outside, target_is_directory=True)
    with project_source_reader(project) as root:
        with pytest.raises(OSError):
            ScipDocumentReader(project / ".victor" / path.name, source_root=root)


@pytest.mark.parametrize("linked", [False, True])
def test_relative_scan_root_still_pins_default_compiler_artifacts(tmp_path, monkeypatch, linked):
    from victor.core.graph_rag.structural_clones import scan

    project, outside = tmp_path / "project", tmp_path / "outside"
    project.mkdir()
    (outside / "precise").mkdir(parents=True)
    source = "def f(x):\n return x+1\n"
    (project / "a.py").write_text(source)
    index = scip_pb2.Index()
    index.documents.add(relative_path="a.py", text=source)
    (outside / "precise" / "scip-python.scip").write_bytes(index.SerializeToString())
    if linked:
        (project / ".victor").symlink_to(outside, target_is_directory=True)
    else:
        target = project / ".victor" / "precise"
        target.mkdir(parents=True)
        (target / "scip-python.scip").write_bytes(index.SerializeToString())
    monkeypatch.chdir(project)
    report = scan(Path("."), min_nodes=1)
    assert report.stats["scip_unavailable_dumps"] == int(linked)
    assert report.stats["scip_files"] == int(not linked)
    assert len(report.units) == 1


@pytest.mark.parametrize("kind", ["symlink", "fifo", "directory", "oversized"])
def test_artifacts_fail_before_unbounded_or_special_reads(tmp_path, kind):
    path, _ = index_file(tmp_path)
    target = tmp_path / "input.scip"
    if kind == "symlink":
        target.symlink_to(path)
    elif kind == "fifo":
        os.mkfifo(target)
    elif kind == "directory":
        target.mkdir()
    else:
        with target.open("wb") as stream:
            stream.truncate(4096)
    with pytest.raises((OSError, ValueError)):
        ScipDocumentReader(target, max_bytes=1024)


def test_file_mutation_invalidates_later_document_reads(tmp_path):
    path, _ = index_file(tmp_path)
    with ScipDocumentReader(path) as reader:
        path.write_bytes(path.read_bytes() + b"\x08\x01")
        with pytest.raises(OSError, match="changed"):
            reader["src/0.rs"]


def test_document_and_occurrence_budgets_fail_without_partial_rows(tmp_path):
    path, _ = index_file(tmp_path)
    with ScipDocumentReader(path, max_document_bytes=8) as reader:
        with pytest.raises(ValueError, match="document"):
            reader["src/0.rs"]
    index = scip_pb2.Index()
    doc = index.documents.add(relative_path="src/0.rs")
    for _ in range(5):
        doc.occurrences.add(range=[0, 1, 2], symbol="local 0")
    path.write_bytes(index.SerializeToString())
    with ScipDocumentReader(path, max_occurrences=4) as reader:
        with pytest.raises(ValueError, match="occurrence"):
            reader["src/0.rs"]


def test_duplicate_paths_and_malformed_wire_are_rejected(tmp_path):
    index = scip_pb2.Index()
    index.documents.add(relative_path="same.rs")
    index.documents.add(relative_path="same.rs")
    path = tmp_path / "index.scip"
    for data in (index.SerializeToString(), b"\x12\xff", b"\x12\x7f\x0a\x01a", b"\0"):
        path.write_bytes(data)
        with pytest.raises(ValueError):
            ScipDocumentReader(path)


def test_optional_source_text_and_encoding_are_preserved(tmp_path):
    index = scip_pb2.Index()
    doc = index.documents.add(relative_path="a.py", text="λ = 1\r\n", position_encoding=2)
    doc.occurrences.add(range=[0, 0, 1], symbol="local 0")
    path = tmp_path / "index.scip"
    path.write_bytes(index.SerializeToString())
    with ScipDocumentReader(path) as reader:
        assert reader["a.py"].text == "λ = 1\r\n"
        assert reader["a.py"].position_encoding == 2


def test_path_budget_precedes_read_and_allocation(tmp_path):
    path = write_document(tmp_path, field(1, b"x" * (8 * 1024**2)))
    tracemalloc.start()
    try:
        before = tracemalloc.get_traced_memory()[0]
        with pytest.raises(ValueError, match="path"):
            ScipDocumentReader(path, max_document_bytes=128)
        peak = tracemalloc.get_traced_memory()[1] - before
    finally:
        tracemalloc.stop()
    assert peak < 1024**2


def test_full_reader_rejects_oversized_index_instead_of_accepting_empty(tmp_path):
    from victor.core.graph_rag.precise_index import _read_index
    from victor.core.graph_rag.scip_documents import MAX_INDEX_BYTES

    path = tmp_path / "large.scip"
    with path.open("wb") as stream:
        stream.truncate(MAX_INDEX_BYTES + 1)
    assert _read_index(path) is None


@pytest.mark.parametrize("value", [False, 0, -1, 1.5, 512 * 1024**2 + 1])
def test_invalid_budgets_fail_before_acquisition(tmp_path, value):
    with pytest.raises(ValueError, match="budget"):
        ScipDocumentReader(tmp_path / "absent.scip", max_bytes=value)


def test_late_path_skips_unconsumed_metadata_and_uses_final_identity(tmp_path):
    raw = field(3, b"x" * 100_000) + field(4, b"rust") + field(1, b"a.rs")
    path = write_document(tmp_path, raw)
    with ScipDocumentReader(path) as reader:
        assert reader["a.rs"].language == "rust"
    path = write_document(tmp_path, field(1, b"a.rs") + field(1, b"b.rs"))
    with ScipDocumentReader(path) as reader:
        with pytest.raises(ValueError, match="path changed"):
            reader["a.rs"]


@pytest.mark.parametrize(
    "occurrence",
    [
        field(1, b"\x01" * 5),
        field(2, b"s" * (64 * 1024 + 1)),
        field(2, b"\xff"),
        b"\x0a\x7f\x01",
        b"\x0b",  # Deprecated groups are outside this occurrence protocol.
    ],
)
def test_invalid_occurrence_projection_rejects_without_partial_rows(tmp_path, occurrence):
    path = write_document(tmp_path, field(1, b"a.rs") + field(2, occurrence))
    with ScipDocumentReader(path) as reader:
        with pytest.raises(ValueError):
            reader["a.rs"]
        assert reader.decoded_documents == 0


def test_unconsumed_occurrence_metadata_does_not_change_inputs(tmp_path):
    occurrence = scip_pb2.Occurrence(range=[1, 2, 3], symbol="local 0").SerializeToString()
    # Unused nested diagnostics need not be valid messages: they are never decoded.
    path = write_document(
        tmp_path, field(1, b"a.rs") + field(2, occurrence + field(10, b"invalid protobuf"))
    )
    with ScipDocumentReader(path) as reader:
        result = reader["a.rs"].occurrences[0]
        assert list(result.range) == [1, 2, 3]
        assert result.symbol == "local 0"


def test_failed_fdopen_closes_acquired_descriptor(tmp_path, monkeypatch):
    path, _ = index_file(tmp_path)
    original = os.fdopen
    captured = []

    def fail(descriptor, *args, **kwargs):
        captured.append(descriptor)
        raise OSError("injected fdopen failure")

    monkeypatch.setattr(os, "fdopen", fail)
    with pytest.raises(OSError, match="injected"):
        ScipDocumentReader(path)
    monkeypatch.setattr(os, "fdopen", original)
    assert len(captured) == 1
    with pytest.raises(OSError):
        os.fstat(captured[0])


def test_scan_distinguishes_text_binding_heuristics_and_rejected_inputs(tmp_path):
    from victor.core.graph_rag.structural_clones import scan

    source = "def f(x):\n return x+1\n"
    (tmp_path / "a.py").write_text(source)
    index = scip_pb2.Index()
    doc = index.documents.add(relative_path="a.py", text=source)
    path = tmp_path / "input.scip"
    path.write_bytes(index.SerializeToString())
    matched = scan(tmp_path, scip_path=path, min_nodes=1)
    assert matched.stats["scip_text_matched_files"] == 1
    assert matched.stats["scip_heuristic_files"] == 0
    doc.text = ""
    path.write_bytes(index.SerializeToString())
    heuristic = scan(tmp_path, scip_path=path, min_nodes=1)
    assert heuristic.stats["scip_text_matched_files"] == 0
    assert heuristic.stats["scip_heuristic_files"] == 1
    path.write_bytes(field(2, field(1, b"a.py") + field(2, field(1, b"\x01" * 5))))
    rejected = scan(tmp_path, scip_path=path, min_nodes=1)
    assert rejected.stats["scip_rejected_documents"] == 1
    assert rejected.stats["scip_files"] == 0
    assert len(rejected.units) == 1
    path.write_bytes(b"\0")
    unavailable = scan(tmp_path, scip_path=path, min_nodes=1)
    assert unavailable.stats["scip_unavailable_dumps"] == 1
    assert unavailable.stats["ast_only_files"] == 1
    assert unavailable.units[0].exact_hash == rejected.units[0].exact_hash
