from __future__ import annotations

from victor_codegraph import parse, parse_path, to_proxima_records


def _by_name(parsed, name):
    return next(s for s in parsed.symbols if s.simple_name == name)


def test_parser_identity_survives_line_shift_and_keeps_legacy_alias():
    base = parse("def f(x):\n    return x\n", file_path="pkg/m.py")
    moved = parse("\n\n\ndef f(x):\n    return x\n", file_path="pkg/m.py")
    a = _by_name(base, "f")
    b = _by_name(moved, "f")

    assert a.id == b.id
    assert a.identity_version == b.identity_version == "v2"
    assert a.legacy_id != b.legacy_id


def test_parse_path_identity_is_portable_across_checkout_roots(tmp_path):
    roots = [tmp_path / "one", tmp_path / "two"]
    parsed = []
    for root in roots:
        path = root / "pkg" / "m.py"
        path.parent.mkdir(parents=True)
        path.write_text("def f():\n    return 1\n")
        parsed.append(parse_path(path, repo_root=root))

    assert parsed[0] is not None and parsed[1] is not None
    assert parsed[0].file_path == parsed[1].file_path == "pkg/m.py"
    assert parsed[0].symbols[0].id == parsed[1].symbols[0].id


def test_path_segments_cannot_alias_dotted_file_names():
    nested = parse("def f(): pass\n", file_path="a/b.py").symbols[0]
    dotted = parse("def f(): pass\n", file_path="a.b.py").symbols[0]
    assert nested.fully_qualified_name == dotted.fully_qualified_name
    assert nested.id != dotted.id


def test_cpp_overloads_have_distinct_stable_ids():
    parsed = parse(
        "int f(int x) { return x; }\ndouble f(double x) { return x; }\n",
        language="cpp",
        file_path="f.cpp",
    )
    overloads = [s for s in parsed.symbols if s.simple_name == "f"]
    assert len(overloads) == 2
    assert len({s.id for s in overloads}) == 2


def test_anonymous_function_expressions_do_not_collide():
    parsed = parse(
        "const a = function() { return 1 }; const b = function() { return 2 };",
        language="javascript",
        file_path="f.js",
    )
    assert all(s.simple_name != "<anonymous>" for s in parsed.symbols)
    assert len({s.id for s in parsed.symbols}) == len(parsed.symbols)


def test_python_decorator_change_updates_content_version_and_snippet():
    a = parse("@first\ndef f():\n    return 1\n", file_path="m.py")
    b = parse("@second\ndef f():\n    return 1\n", file_path="m.py")
    sa, sb = _by_name(a, "f"), _by_name(b, "f")
    ra = to_proxima_records(a, "repo")[0]
    rb = to_proxima_records(b, "repo")[0]

    assert sa.id == sb.id
    assert sa.source_code.startswith("@first")
    assert sb.source_code.startswith("@second")
    assert ra["props"]["content_version"] != rb["props"]["content_version"]
