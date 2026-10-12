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
    assert [d["source_span"]["start_line"] for d in collisions[0]["declarations"]] == [1, 2]
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
