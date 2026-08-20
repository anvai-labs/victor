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
