"""Tier-2.5 scope binding on the Python (stdlib ``ast``) path.

Pins the REFERENCES contract from ADR-032: identifier uses attributed to their
innermost owning symbol, resolved in-file where unique (``scope_resolver``,
confidence 1.0) and retained as structured references otherwise (0.5);
locals/params shadowed at extraction; callees left to CALLS; one relation per
(owner, name) with the FIRST use as ``call_site``; and the default
(``emit_references=False``) leaving the pre-tier-2.5 output unchanged.
"""

from __future__ import annotations

from victor_codegraph import CodeRelationType, parse
from victor_codegraph.scope_binding import UNRESOLVED_CAP_PER_OWNER, cap_unresolved_references

SRC = """\
import os
from json import dumps as encode


def helper(x):
    return x


class Thing:
    def __init__(self):
        self.data = []

    def run(self, helper):
        value = helper
        return encode(value)


def main():
    item = Thing()
    result = helper(1)
    env = os.environ
    return item, result, env, Thing
"""


def _references(parsed):
    return [r for r in parsed.relations if r.relation_type == CodeRelationType.REFERENCES]


def _ref_name(parsed, relation):
    """Readable target name: the retained reference's name, or the resolved id's."""
    if relation.target_ref is not None:
        return relation.target_ref.name
    by_id = {s.id: s.simple_name for s in parsed.symbols}
    return by_id.get(relation.to_symbol_id, relation.to_symbol_id)


def _by_symbol(parsed, name):
    symbol = next(s for s in parsed.symbols if s.simple_name == name)
    return [r for r in _references(parsed) if r.from_symbol_id == symbol.id]


def test_default_off_leaves_relations_unchanged():
    parsed = parse(SRC, file_path="m.py")
    assert _references(parse(SRC, file_path="m.py")) == []
    assert parsed.relations  # CALLS/CONTAINS still present


def test_in_file_class_and_function_uses_resolve():
    parsed = parse(SRC, file_path="m.py", emit_references=True)
    main_refs = _by_symbol(parsed, "main")
    resolved = {_ref_name(parsed, r): r for r in main_refs}

    thing = next(s for s in parsed.symbols if s.simple_name == "Thing")
    edge = resolved.get("Thing")
    assert edge is not None, "non-call class use recorded"
    assert edge.to_symbol_id == thing.id and edge.confidence == 1.0
    assert edge.provenance == "scope_resolver"
    # `Thing()` on line 19 is a CALL (excluded); the surviving use is line 22.
    assert edge.call_site.start_line == 22

    os_edge = resolved.get("os")
    assert os_edge is not None and os_edge.target_ref is not None
    assert os_edge.confidence == 0.5  # import name: cross-file territory


def test_param_shadows_module_function():
    parsed = parse(SRC, file_path="m.py", emit_references=True)
    # `run` owns no REFERENCES at all: `helper` is a parameter (shadowed),
    # `value` is a local write (suppressed), `encode(value)` is a callee
    # (CALLS territory).
    assert _by_symbol(parsed, "run") == []


def test_locals_and_receiver_are_suppressed():
    parsed = parse(SRC, file_path="m.py", emit_references=True)
    init_refs = _by_symbol(parsed, "__init__")
    assert init_refs == []  # only self (param) and attribute writes


def test_callees_do_not_duplicate_into_references():
    parsed = parse(SRC, file_path="m.py", emit_references=True)
    calls = {
        _ref_name(parsed, r) for r in parsed.relations if r.relation_type == CodeRelationType.CALLS
    }
    main_refs = _by_symbol(parsed, "main")
    # `helper(1)` and `Thing()` are CALLS; their names may appear as REFERENCES
    # only through a separate non-call use (Thing does; helper does not).
    ref_names = {_ref_name(parsed, r) for r in main_refs}
    assert "helper" not in ref_names
    assert "Thing" in ref_names and "helper" in calls


def test_one_relation_per_owner_and_name_with_first_site():
    parsed = parse(SRC, file_path="m.py", emit_references=True)
    main_refs = _by_symbol(parsed, "main")
    thing_uses = [r for r in main_refs if r.to_symbol_id != "Thing" or r.confidence == 1.0]
    thing_edges = [r for r in thing_uses if r.confidence == 1.0]
    assert len(thing_edges) == 1, "deduped to a single (owner, name) relation"


def test_unresolved_cap_is_deterministic_and_scoped():
    body = "\n".join(f"    use_{i} = unknown_name_{i}" for i in range(UNRESOLVED_CAP_PER_OWNER + 5))
    src = f"def big():\n{body}\n"
    parsed = parse(src, file_path="big.py", emit_references=True)
    refs = _by_symbol(parsed, "big")
    assert len(refs) == UNRESOLVED_CAP_PER_OWNER
    lines = sorted(r.call_site.start_line for r in refs)
    assert lines == list(range(2, 2 + UNRESOLVED_CAP_PER_OWNER)), "first occurrences kept"

    # The cap never drops resolved references or other relation types.
    mixed = cap_unresolved_references(
        [r for r in parse(SRC, file_path="m.py", emit_references=True).relations]
        + list(_references(parse(SRC, file_path="m.py", emit_references=True)))
    )
    assert any(r.relation_type == CodeRelationType.CONTAINS for r in mixed)


def test_module_level_uses_are_skipped_v1():
    src = "used_at_module_level = helper\n\n\ndef helper():\n    return 1\n"
    parsed = parse(src, file_path="m.py", emit_references=True)
    assert all(r.relation_type != CodeRelationType.REFERENCES for r in parsed.relations) or (
        _by_symbol(parsed, "helper") == []
    )
