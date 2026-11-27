"""A collision makes its declarations ambiguous, not their whole file."""

import pytest

from victor_codegraph import CodeRelationType, CodeSymbolType, parse, parse_repo, quarantine_view

# Arrow writes one definition per configuration; both share a structural key.
CPP_READER = """#include <memory>
namespace arrow {
class Reader {
 public:
  Status Open(const std::string& path);
  Status Close();
};
#if defined(ARROW_WITH_IO)
Status Reader::Open(const std::string& path) { return Close(); }
#else
Status Reader::Open(const std::string& path) { return Status::OK(); }
#endif
Status Reader::Close() { return Status::OK(); }
}  // namespace arrow
"""

SHADOWED = """class C:
    def a(self):
        return 1

class C:
    def b(self):
        return 2

def top():
    return C()
"""


def _by_name(parsed, name):
    return next(s for s in parsed.symbols if s.simple_name == name)


def test_cpp_configuration_duplicates_leave_the_rest_of_the_file_intact(tmp_path):
    (tmp_path / "reader.cc").write_text(CPP_READER)
    repository = parse_repo(tmp_path)
    raw = repository.files["reader.cc"]
    opens = [s for s in raw.symbols if s.simple_name == "Open"]
    assert len(opens) == 2 and len({s.id for s in opens}) == 1

    marker, *kept = [s for s in repository.symbols if s.location.file_path == "reader.cc"]
    assert [(s.symbol_type, s.simple_name) for s in kept] == [
        (CodeSymbolType.CLASS, "Reader"),
        (CodeSymbolType.FUNCTION, "Close"),
    ]
    assert marker.metadata["projection_status"] == "symbols_quarantined"
    assert marker.metadata["omitted_symbol_count"] == 2
    assert marker.metadata["ambiguous_symbol_names"] == ["Open"]
    assert {
        d["source_span"]["start_line"]
        for d in marker.metadata["identity_collisions"][0]["declarations"]
    } == {9, 11}
    assert not any(opens[0].id in (r.from_symbol_id, r.to_symbol_id) for r in repository.relations)


def test_members_of_a_duplicated_container_are_left_out_too(tmp_path):
    (tmp_path / "shadowed.py").write_text(SHADOWED)
    repository = parse_repo(tmp_path)
    raw = repository.files["shadowed.py"]
    assert sorted(s.simple_name for s in raw.symbols) == ["C", "C", "a", "b", "top"]

    # Each copy of `C` shadows the other, so `a` and `b` have no unambiguous
    # parent; only `top` survives.
    marker, top = [s for s in repository.symbols if s.location.file_path == "shadowed.py"]
    assert top.simple_name == "top"
    assert marker.metadata["omitted_symbol_count"] == 4
    assert marker.metadata["ambiguous_symbol_names"] == ["C"]
    contains = [r for r in repository.relations if r.relation_type == CodeRelationType.CONTAINS]
    assert [(r.from_symbol_id, r.to_symbol_id) for r in contains] == [(marker.id, top.id)]
    # A reference to the duplicated name stays unresolved.
    (call,) = [r for r in repository.relations if r.relation_type == CodeRelationType.CALLS]
    assert call.from_symbol_id == top.id and call.target_ref is not None


def test_a_file_whose_every_declaration_collides_stays_a_marker(tmp_path):
    (tmp_path / "twice.py").write_text("def f(): return 1\ndef f(): return 2\n")
    repository = parse_repo(tmp_path)
    (marker,) = [s for s in repository.symbols if s.location.file_path == "twice.py"]
    assert marker.symbol_type == CodeSymbolType.FILE
    assert marker.metadata["projection_status"] == "file_marker_only"
    assert marker.metadata["omitted_symbol_count"] == 2


def test_a_healthy_parse_is_returned_as_itself():
    parsed = parse("def f(): return helper()\n\n\ndef helper(): return 1\n", file_path="m.py")
    view = quarantine_view(parsed)
    assert view.parsed is parsed
    assert not view.symbol_ids and not view.names and view.dropped_relations == 0


def test_a_locally_ambiguous_name_is_never_bound_through_an_import():
    """The repository resolver leaves such a reference unresolved, although the
    import binding alone would resolve it to a namesake in another file."""
    from dataclasses import replace

    from victor_codegraph.repository_index import _resolve_across_files

    a = parse("def caller(): return f()\n", file_path="a.py")
    b = parse("def f(): return 0\n", file_path="b.py")
    caller, namesake = a.symbols[0], b.symbols[0]
    (relation,) = [r for r in a.relations if r.relation_type == CodeRelationType.CALLS]
    assert relation.target_ref is not None and relation.target_ref.name == "f"
    files = {"a.py": replace(a, imports=["from b import f"]), "b.py": b}

    (bound,) = _resolve_across_files([caller, namesake], [relation], files)
    assert bound.target_ref is None and bound.to_symbol_id == namesake.id

    (kept,) = _resolve_across_files(
        [caller, namesake], [relation], files, None, None, {"a.py": frozenset({"f"})}
    )
    assert kept.target_ref is not None


def test_a_survivor_of_a_narrowed_path_is_not_a_unique_target(tmp_path):
    """`class f` survives two colliding `def f`: before quarantine an importer's
    call to `f` saw three declarations and stayed unresolved; it still does."""
    importer = "from x import f\n\n\ndef g():\n    return f()\n"
    for source in (
        "class f:\n    pass\n\n\ndef f():\n    return 1\n\n\ndef f():\n    return 2\n",
        "def f():\n    return 1\n\n\ndef f():\n    return 2\n\n\nclass f:\n    pass\n",
    ):
        (tmp_path / "x.py").write_text(source)
        (tmp_path / "y.py").write_text(importer)
        repository = parse_repo(tmp_path)
        (survivor,) = [
            s for s in repository.symbols if s.location.file_path == "x.py" and s.simple_name == "f"
        ]
        assert survivor.symbol_type == CodeSymbolType.CLASS
        (call,) = [r for r in repository.relations if r.relation_type == CodeRelationType.CALLS]
        assert call.target_ref is not None and call.to_symbol_id != survivor.id


def test_members_inside_a_colliding_overload_are_left_out_wherever_contains_points():
    """The parser attaches every member at scope path ``f`` to the FIRST ``f``;
    members inside the colliding ``f(int)`` bodies must not survive as members
    of the unrelated ``f(double)``."""
    parsed = parse(
        "void f(double y) { struct B { int x; }; }\n"
        "void f(int x) { struct A { int x; }; }\n"
        "void f(int x) { struct C { int x; }; }\n",
        language="cpp",
        file_path="a.cc",
    )
    view = quarantine_view(parsed)
    assert sorted(s.simple_name for s in view.parsed.symbols if s.symbol_type.name != "MODULE") == [
        "B",
        "f",
    ]
    (kept_f,) = [s for s in view.parsed.symbols if s.simple_name == "f"]
    assert kept_f.signature == "(double y)"
    contains = [
        (r.from_symbol_id, r.to_symbol_id)
        for r in view.parsed.relations
        if r.relation_type == CodeRelationType.CONTAINS
    ]
    b = next(s for s in view.parsed.symbols if s.simple_name == "B")
    assert contains == [(kept_f.id, b.id)]
    assert view.paths == frozenset({("f",), ("f", "A"), ("f", "C")})


def _write(root, files):
    for path, text in files.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text(text)


@pytest.mark.parametrize("healthy_sibling", ["", "\n\ndef ok():\n    return 0\n"])
def test_a_lookup_over_several_keys_is_not_narrowed_to_another_key(tmp_path, healthy_sibling):
    """`from pkg import mod` binds both `pkg/__init__.py::mod` and `pkg/mod.py`.
    With the package's `mod.f` ambiguous, `mod.f()` saw two candidates before
    quarantine; the one left at the other key is not unique. Without a healthy
    sibling the package file is reduced to its marker, and must still count."""
    _write(
        tmp_path,
        {
            "pkg/__init__.py": "class mod:\n    def f(self):\n        return 1\n\n\n"
            "class mod:\n    def f(self):\n        return 2\n" + healthy_sibling,
            "pkg/mod.py": "def f():\n    return 3\n",
            "user.py": "from pkg import mod\n\n\ndef g():\n    return mod.f()\n",
        },
    )
    repository = parse_repo(tmp_path)
    (call,) = [r for r in repository.relations if r.relation_type == CodeRelationType.CALLS]
    assert call.target_ref is not None and call.target_ref.text == "mod.f"


def test_a_use_of_a_quarantined_item_still_counts_as_consulted(tmp_path):
    """Rust: `f` names both the colliding `a::f` functions and the struct `b::f`.
    The binding to `a::f` must not vanish and leave the struct as the only target."""
    _write(
        tmp_path,
        {
            "Cargo.toml": '[package]\nname = "k"\nversion = "0.1.0"\nedition = "2021"\n',
            "src/lib.rs": "pub mod a;\npub mod b;\npub mod c;\n",
            "src/a.rs": "pub fn f() {}\npub fn f() {}\npub fn ok() {}\n",
            "src/b.rs": "pub struct f {}\n",
            "src/c.rs": "use crate::a::f;\nuse crate::b::f;\npub fn caller() { f(); }\n",
        },
    )
    repository = parse_repo(tmp_path)
    (caller,) = [s for s in repository.symbols if s.simple_name == "caller"]
    calls = [
        r
        for r in repository.relations
        if r.relation_type == CodeRelationType.CALLS and r.from_symbol_id == caller.id
    ]
    assert [r.target_ref.name for r in calls if r.target_ref is not None] == ["f"]
    assert all(r.target_ref is not None for r in calls)
