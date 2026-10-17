from __future__ import annotations

from dataclasses import replace

import pytest

from victor_codegraph import apply_index_delta, diff_repository, parse_repo


def test_repository_delta_deletes_removed_and_replaced_symbols(tmp_path):
    a = tmp_path / "a.py"
    b = tmp_path / "b.py"
    a.write_text("def old_name():\n    return 1\n")
    b.write_text("def removed():\n    return 2\n")
    previous = parse_repo(tmp_path, repo_id="repo")

    a.write_text("def new_name():\n    return 1\n")
    b.unlink()
    current = parse_repo(tmp_path, repo_id="repo")
    delta = diff_repository(current, previous)

    old_function_ids = {s.id for s in previous.symbols if s.symbol_type.name == "FUNCTION"}
    assert old_function_ids <= set(delta.delete_symbol_ids)
    assert "new_name" in {s.simple_name for s in delta.upsert_symbols}

    rebuilt = apply_index_delta(previous, delta)
    assert {s.id for s in rebuilt.symbols} == {s.id for s in current.symbols}
    assert rebuilt.manifest == current.manifest
    assert delta.base_generation_id == previous.generation_id
    assert delta.target_generation_id == current.generation_id

    with pytest.raises(ValueError, match="base generation"):
        apply_index_delta(replace(previous, generation_id="stale"), delta)


def test_repository_delta_refreshes_relation_evidence_on_line_shift(tmp_path):
    path = tmp_path / "m.py"
    path.write_text("def a():\n    return b()\n\ndef b(): return 1\n")
    previous = parse_repo(tmp_path, repo_id="repo")
    path.write_text("\ndef a():\n    return b()\n\ndef b(): return 1\n")
    current = parse_repo(tmp_path, repo_id="repo")

    delta = diff_repository(current, previous)
    assert delta.upsert_relations
    rebuilt = apply_index_delta(previous, delta)
    assert rebuilt.relations[0].call_site.start_line == current.relations[0].call_site.start_line


def test_repository_quarantines_identity_collisions_and_delta_recovers(tmp_path):
    from victor_codegraph import CodeRelationType, CodeSymbolType, ParseStatus

    ambiguous = tmp_path / "duplicate.py"
    ambiguous.write_text("def helper(): return 1\n")
    (tmp_path / "caller.py").write_text(
        "from duplicate import helper\ndef caller(): return helper()\n"
    )
    previous = parse_repo(tmp_path, repo_id="repo")
    ambiguous.write_text(
        "def f(): return helper()\ndef f(): return helper()\ndef helper(): return 1\n"
    )
    current = parse_repo(tmp_path, repo_id="repo")
    raw = current.files["duplicate.py"]
    assert raw.status == ParseStatus.PARTIAL
    assert len(raw.symbols) == 3  # Keep original evidence, including both definitions.
    projected = [s for s in current.symbols if s.location.file_path == "duplicate.py"]
    assert len(projected) == 1
    marker = projected[0]
    assert marker.symbol_type == CodeSymbolType.FILE
    assert marker.metadata["parse_status"] == "partial"
    assert marker.metadata["projection_status"] == "file_marker_only"
    assert marker.metadata["omitted_symbol_count"] == 3
    assert marker.metadata["omitted_relation_count"] == len(raw.relations)
    assert [d["line"] for d in marker.metadata["diagnostics"]] == [1, 2]
    collisions = marker.metadata["identity_collisions"]
    assert len(collisions) == 1 and collisions[0]["symbol_id"] == raw.symbols[0].id
    assert collisions[0]["resolution"] == "ambiguous"
    assert [d["qualified_name"] for d in collisions[0]["declarations"]] == [
        "duplicate.py::f",
        "duplicate.py::f",
    ]
    assert [d["symbol_type"] for d in collisions[0]["declarations"]] == ["function", "function"]
    assert [d["source_span"]["start_line"] for d in collisions[0]["declarations"]] == [1, 2]
    from victor_codegraph import to_proxima_records

    marker.metadata["private_source_metadata"] = "must not be exported"
    records = to_proxima_records(current, repo_graph_id="repo")
    record = next(
        r
        for r in records
        if r.get("props", {}).get("file") == "duplicate.py"
        and r.get("props", {}).get("ast_kind") == "FILE"
    )
    for key in (
        "identity_collisions",
        "projection_status",
        "omitted_symbol_count",
        "omitted_relation_count",
    ):
        assert record["props"][key] == marker.metadata[key]
    assert "private_source_metadata" not in record["props"]
    assert not any(
        "/external/" not in r["edge"]["to_oid"]
        for r in records
        if "graph_edge" in r["labels"] and r["edge"]["edge_type"] == "CALLS"
    )
    assert len({s.id for s in current.symbols}) == len(current.symbols)
    assert not any(r.from_symbol_id == marker.id for r in current.relations)
    calls = [r for r in current.relations if r.relation_type == CodeRelationType.CALLS]
    assert len(calls) == 1 and calls[0].target_ref.name == "helper"
    assert any(
        r.relation_type == CodeRelationType.IMPORTS and r.to_symbol_id == marker.id
        for r in current.relations
    )
    delta = diff_repository(current, previous)
    assert delta.delete_symbol_ids and delta.delete_relation_keys
    rebuilt = apply_index_delta(previous, delta)
    assert {s.id for s in rebuilt.symbols} == {s.id for s in current.symbols}

    ambiguous.write_text("def helper(): return 2\n")
    repaired = parse_repo(tmp_path, repo_id="repo")
    rebuilt = apply_index_delta(current, diff_repository(repaired, current))
    assert {s.id for s in rebuilt.symbols} == {s.id for s in repaired.symbols}
    calls = [r for r in rebuilt.relations if r.relation_type == CodeRelationType.CALLS]
    assert len(calls) == 1 and calls[0].target_ref is None
    assert (
        next(s for s in repaired.symbols if s.id == marker.id).metadata["projection_status"]
        == "symbols"
    )


def test_selected_inventory_uses_existing_resolver_without_default_exclusions(
    tmp_path, monkeypatch
):
    from victor_codegraph import repository_index

    (tmp_path / "build").mkdir()
    (tmp_path / ".custom").mkdir()
    (tmp_path / "build/library.py").write_text("def target(): return 1\n")
    (tmp_path / ".custom/hidden.py").write_text("hidden = 1\n")
    (tmp_path / "main.py").write_text(
        "from build.library import target\ndef caller(): return target()\n"
    )
    (tmp_path / "excluded.py").write_text("def invalid(\n")
    reads = []
    original_read = repository_index.read_source_text

    def read(path):
        reads.append(path.relative_to(tmp_path).as_posix())
        return original_read(path)

    monkeypatch.setattr(repository_index, "read_source_text", read)
    selected = ["main.py", "build/library.py", ".custom/hidden.py"]
    result = parse_repo(tmp_path, selected_paths=iter(selected))
    assert set(result.files) == set(selected) == set(result.manifest)
    assert sorted(reads) == sorted(selected)
    ids = {s.simple_name: s.id for s in result.symbols}
    assert any(
        r.from_symbol_id == ids["caller"]
        and r.to_symbol_id == ids["target"]
        and r.relation_type.name == "CALLS"
        for r in result.relations
    )
    assert result == parse_repo(tmp_path, selected_paths=[*reversed(selected), "./main.py"])
    assert parse_repo(tmp_path, selected_paths=[]).files == {}


def test_explicit_default_inventory_preserves_default_graph_contract(tmp_path):
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested/inner.py").write_text("def inner(): return 1\n")
    (tmp_path / "z.py").write_text("last = 1\n")
    (tmp_path / "a.py").write_text("def target(): return 1\n")
    (tmp_path / "b.py").write_text("from a import target\ndef caller(): return target()\n")
    default = parse_repo(tmp_path, repo_id="repo", revision="test")
    explicit = parse_repo(tmp_path, repo_id="repo", revision="test", selected_paths=default.files)
    assert explicit == default
    assert parse_repo(tmp_path, languages=["rust"], selected_paths=default.files).files == {}


@pytest.mark.parametrize(
    "invalid", ["escape", "absolute_escape", "symlink_escape", "directory", "fifo", "scalar"]
)
def test_selected_inventory_rejects_invalid_paths_before_reading(tmp_path, monkeypatch, invalid):
    from victor_codegraph import repository_index

    root = tmp_path / "repo"
    root.mkdir()
    (root / "valid.py").write_text("x = 1\n")
    outside = tmp_path / "outside.py"
    outside.write_text("private = 1\n")
    (root / "link.py").symlink_to(outside)
    if invalid == "fifo":
        import os

        if not hasattr(os, "mkfifo"):
            pytest.skip("POSIX FIFO unavailable")
        os.mkfifo(root / "pipe.py")
    bad = {
        "escape": "../outside.py",
        "absolute_escape": outside,
        "symlink_escape": "link.py",
        "directory": ".",
        "fifo": "pipe.py",
        "scalar": "valid.py",
    }[invalid]

    def forbidden_read(path):
        pytest.fail(f"source read before inventory validation: {path}")

    monkeypatch.setattr(repository_index, "read_source_text", forbidden_read)
    with pytest.raises(ValueError, match="inventory"):
        parse_repo(root, selected_paths=bad if invalid == "scalar" else ["valid.py", bad])


def test_selected_inventory_deduplicates_internal_symlinks_and_reports_missing_by_omission(
    tmp_path,
):
    (tmp_path / "a.py").write_text("x = 1\n")
    (tmp_path / "alias.py").symlink_to(tmp_path / "a.py")
    result = parse_repo(tmp_path, selected_paths=["alias.py", "a.py", "missing.py"])
    assert list(result.manifest) == ["a.py"]
    assert list(result.files) == ["a.py"]
    assert result == parse_repo(tmp_path, selected_paths=["a.py"])


def test_selected_inventory_requires_existing_directory(tmp_path):
    with pytest.raises(ValueError, match="existing repository directory"):
        parse_repo(tmp_path / "missing", selected_paths=[])


def test_selected_unreadable_source_is_absent_from_manifest(tmp_path, monkeypatch):
    from pathlib import Path

    (tmp_path / "denied.py").write_text("x = 1\n")
    original_read = Path.read_bytes

    def deny(path):
        if path.name == "denied.py":
            raise PermissionError("source unavailable")
        return original_read(path)

    monkeypatch.setattr(Path, "read_bytes", deny)
    result = parse_repo(tmp_path, selected_paths=["denied.py"])
    assert result.files == {} and result.manifest == {}


def test_selected_inventory_propagates_type_validation_failure_before_reading(
    tmp_path, monkeypatch
):
    from pathlib import Path
    from victor_codegraph import repository_index

    (tmp_path / "a.py").write_text("x = 1\n")
    original_stat = Path.stat

    def deny(path, *args, **kwargs):
        if path.name == "a.py":
            raise PermissionError("cannot validate selected file type")
        return original_stat(path, *args, **kwargs)

    def forbidden_read(path):
        pytest.fail(f"source read before type validation: {path}")

    monkeypatch.setattr(Path, "stat", deny)
    monkeypatch.setattr(repository_index, "read_source_text", forbidden_read)
    with pytest.raises(PermissionError, match="cannot validate"):
        parse_repo(tmp_path, selected_paths=["a.py"])
