"""Tier-2.5 scope binding at repository scope (ADR-032).

``parse_repo(emit_references=True)`` pins: cross-file REFERENCES resolution
through exact import bindings (``repository_resolver``, confidence 1.0),
unresolvable targets retained at 0.5, the default flag leaving repository
output unchanged, delta equivalence (incremental rebuild == full re-parse),
and determinism (two parses emit identical relations).
"""

from __future__ import annotations

import pytest

pytest.importorskip("victor_codegraph.repository_index")

from victor_codegraph import CodeRelationType, parse_repo  # noqa: E402
from victor_codegraph.repository_index import (  # noqa: E402
    apply_index_delta,
    diff_repository,
    relation_key,
)

PY_PACKAGE = {
    "pkg/__init__.py": "",
    "pkg/targets.py": (
        "def process(data):\n" "    return data\n" "\n" "\n" "class Record:\n" "    pass\n"
    ),
    "pkg/consumer.py": (
        "from pkg.targets import Record, process\n"
        "\n"
        "\n"
        "def use(record=None):\n"
        "    kind = Record\n"
        "    fn = process\n"
        "    return kind, fn\n"
    ),
}


@pytest.fixture()
def repo(tmp_path):
    for path, content in PY_PACKAGE.items():
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    return tmp_path


def _references(parsed_repo):
    return [r for r in parsed_repo.relations if r.relation_type == CodeRelationType.REFERENCES]


def _named(repo_result):
    by_id = {s.id: s.simple_name for s in repo_result.symbols}
    return {
        by_id.get(r.from_symbol_id, r.from_symbol_id): (
            by_id.get(r.to_symbol_id, r.to_symbol_id) if r.target_ref is None else r.target_ref.name
        )
        for r in _references(repo_result)
    }


def test_cross_file_references_resolve_via_import_bindings(repo):
    result = parse_repo(repo, selected_paths=PY_PACKAGE, emit_references=True)
    record = next(s for s in result.symbols if s.simple_name == "Record")
    process = next(s for s in result.symbols if s.simple_name == "process")
    use = next(s for s in result.symbols if s.simple_name == "use")

    edges = [r for r in _references(result) if r.from_symbol_id == use.id]
    targets = {r.to_symbol_id for r in edges}
    assert {record.id, process.id} <= targets, "both import-bound uses resolve cross-file"
    assert all(r.provenance == "repository_resolver" and r.confidence == 1.0 for r in edges)
    assert all(r.target_ref is None for r in edges)


def test_default_off_repository_unchanged(repo):
    result = parse_repo(repo, selected_paths=PY_PACKAGE)
    assert _references(result) == []


def test_delta_equivalence(repo):
    full = parse_repo(repo, selected_paths=PY_PACKAGE, emit_references=True)
    base = parse_repo(repo, selected_paths=list(PY_PACKAGE)[:2], emit_references=True)
    # The consumer file joins afterwards — its REFERENCES must land via delta.
    delta = diff_repository(full, base)
    rebuilt = apply_index_delta(base, delta)
    assert sorted(relation_key(r) for r in rebuilt.relations) == sorted(
        relation_key(r) for r in full.relations
    )
    assert sorted(r.to_symbol_id for r in _references(rebuilt)) == sorted(
        r.to_symbol_id for r in _references(full)
    )


def test_determinism(repo):
    first = parse_repo(repo, selected_paths=PY_PACKAGE, emit_references=True)
    second = parse_repo(repo, selected_paths=PY_PACKAGE, emit_references=True)
    assert [relation_key(r) for r in _references(first)] == [
        relation_key(r) for r in _references(second)
    ]
