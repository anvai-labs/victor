"""Shared grammar/SCIP path for structural analysis beyond Python."""

import pytest

from victor.core.graph_rag.structural_clones import scan


@pytest.mark.parametrize(
    "extension,source,left,right",
    [
        (
            "py",
            "def compute(x, y):\n    total = x - y\n    adjusted = total * 3\n    return adjusted + 4\n",
            "x - y",
            "y - x",
        ),
        (
            "rs",
            "fn compute(x:i32,y:i32)->i32 { let total=x-y; let adjusted=total*3; adjusted+4 }",
            "x-y",
            "y-x",
        ),
        (
            "go",
            "package p\nfunc compute(x,y int) int { total:=x-y; adjusted:=total*3; return adjusted+4 }",
            "x-y",
            "y-x",
        ),
    ],
)
def test_ordinary_ast_preserves_parameter_roles(tmp_path, extension, source, left, right):
    (tmp_path / f"first.{extension}").write_text(source)
    (tmp_path / f"second.{extension}").write_text(source.replace(left, right))
    report = scan(tmp_path, min_nodes=5, near_metric="subtree", near_threshold=1)
    assert report.families == []


def test_empty_supported_scope_is_not_a_clean_analysis(tmp_path):
    (tmp_path / "only.sql").write_text("SELECT 1")
    report = scan(tmp_path)
    assert report.stats["source_files"] == 0
    assert not report.coverage_complete


def test_grouped_go_parameters_can_be_consistently_renamed(tmp_path):
    source = (
        "package p\nfunc compute(x,y int) int { total:=x-y; adjusted:=total*3; return adjusted+4 }"
    )
    (tmp_path / "first.go").write_text(source)
    (tmp_path / "second.go").write_text(
        source.replace("x,y", "first,second").replace("x-y", "first-second")
    )
    report = scan(tmp_path, min_nodes=5, near_metric="subtree", near_threshold=1)
    assert any(f.tier == "exact" for f in report.families)


@pytest.mark.parametrize(
    "extension,source",
    [
        ("rs", "fn f(x: i32) -> i32 { let y = load(x); y + 1 }"),
        ("js", "function f(x) { let y = load(x); return y + 1; }"),
        ("ts", "function f(x: number): number { let y = load(x); return y + 1; }"),
        ("go", "package main\nfunc f(x int) int { y := load(x); return y + 1 }"),
        ("java", "class A { int f(int x) { int y = load(x); return y + 1; } }"),
        ("c", "int f(int x) { int y = load(x); return y + 1; }"),
        ("cpp", "int f(int x) { int y = load(x); return y + 1; }"),
        ("rb", "def f(x)\n y = load(x)\n y + 1\nend"),
    ],
)
def test_renamed_copies_with_real_grammars(tmp_path, extension, source):
    (tmp_path / f"a.{extension}").write_text(source)
    (tmp_path / f"b.{extension}").write_text(source.replace("x", "value").replace("y", "result"))
    report = scan(tmp_path, min_nodes=5, near_metric="subtree")
    assert not report.stats["unavailable_languages"], report.stats
    assert report.stats["functions"] == 2, report.stats
    assert any(f.tier == "exact" for f in report.families)
    assert all(u.subtree_hashes for f in report.families for u in f.members)


def test_rust_scip_separates_same_spelling_with_different_targets(tmp_path):
    scip_pb2 = pytest.importorskip("victor.core.graph_rag._scip.scip_pb2")

    source = "fn f(x: i32) -> i32 { let y = store.get(x); y + 1 }"
    index = scip_pb2.Index()
    for name, target in (("a", "Registry"), ("b", "Cache")):
        path = f"{name}.rs"
        (tmp_path / path).write_text(source)
        doc = index.documents.add(relative_path=path)
        start = source.index("get")
        doc.occurrences.add(
            range=[0, start, start + 3], symbol=f"rust-analyzer cargo pkg 1.0 {target}#get()."
        )
    directory = tmp_path / ".victor" / "precise"
    directory.mkdir(parents=True)
    (directory / "rust-analyzer.scip").write_bytes(index.SerializeToString())
    report = scan(tmp_path, min_nodes=5, near_metric="subtree")
    assert report.stats["scip_files"] == 2
    assert report.stats["resolved_symbols"] == 2
    assert not any(f.tier == "exact" for f in report.families)
    assert any(f.tier == "template" for f in report.families)


@pytest.mark.parametrize(
    "extension,provider,source",
    [
        ("js", "scip-typescript", "function f(x) { let y = load(x); return y + 1; }"),
        ("ts", "scip-typescript", "function f(x: number) { let y = load(x); return y + 1; }"),
        ("tsx", "scip-typescript", "function f(x: number) { let y = load(x); return y + 1; }"),
        ("go", "scip-go", "package main\nfunc f(x int) int { y := load(x); return y + 1 }"),
        ("java", "scip-java", "class A { int f(int x) { int y = load(x); return y + 1; } }"),
        ("c", "scip-clang", "int f(int x) { int y = load(x); return y + 1; }"),
        ("cpp", "scip-clang", "int f(int x) { int y = load(x); return y + 1; }"),
        ("rb", "scip-ruby", "def f(x)\n y = load(x)\n y + 1\nend"),
    ],
)
def test_other_language_providers_refine_real_syntax(tmp_path, extension, provider, source):
    """Same spelling + different compiler identities must not be exact matches."""
    scip_pb2 = pytest.importorskip("victor.core.graph_rag._scip.scip_pb2")
    index = scip_pb2.Index()
    prefix = source[: source.index("load")]
    row = prefix.count("\n")
    column = len(prefix.rsplit("\n", 1)[-1].encode("utf-8"))
    for name, target in (("a", "Registry"), ("b", "Cache"), ("c", "Registry")):
        path = f"{name}.{extension}"
        (tmp_path / path).write_text(source)
        doc = index.documents.add(relative_path=path, text=source, position_encoding=1)
        doc.occurrences.add(
            range=[row, column, column + 4], symbol=f"{provider} manager pkg 1 {target}#load()."
        )
    directory = tmp_path / ".victor" / "precise"
    directory.mkdir(parents=True)
    (directory / f"{provider}.scip").write_bytes(index.SerializeToString())
    report = scan(tmp_path, min_nodes=5, near_metric="subtree")
    assert report.stats["scip_dumps"] == [f"{provider}.scip"]
    assert report.stats["resolved_symbols"] == 3
    exact = [f for f in report.families if f.tier == "exact"]
    assert len(exact) == 1
    assert {u.file_path for u in exact[0].members} == {f"a.{extension}", f"c.{extension}"}
    assert report.coverage_complete


def test_missing_grammar_is_explicit_coverage_gap(tmp_path, monkeypatch):
    (tmp_path / "a.rs").write_text("fn f() { call(); }")
    monkeypatch.setattr(
        "victor.core.graph_rag.language_parsers.create_parser", lambda language: None
    )
    report = scan(tmp_path)
    assert report.stats["unavailable_languages"] == ["rust"]


def test_parse_error_is_explicit_coverage_gap(tmp_path):
    (tmp_path / "bad.rs").write_text("fn f( { broken")
    report = scan(tmp_path)
    assert report.stats["parse_error_files"] == 1


def test_python_parse_error_is_explicit_coverage_gap(tmp_path):
    (tmp_path / "bad.py").write_text("def f(\n")
    report = scan(tmp_path)
    assert report.stats["parse_error_files"] == 1
    assert not report.coverage_complete


def test_oversized_file_is_explicit_coverage_gap(tmp_path):
    (tmp_path / "large.rs").write_text("fn f() { call(); }")
    report = scan(tmp_path, max_file_bytes=1)
    assert report.stats["oversized_files"] == 1
    assert not report.coverage_complete


def test_nonexistent_project_cannot_report_clean_coverage(tmp_path):
    with pytest.raises(ValueError, match="not a directory"):
        scan(tmp_path / "absent")


def test_shingles_do_not_silently_group_different_languages(tmp_path):
    source = "int f(int x) { int y = load(x); return y + 1; }"
    (tmp_path / "a.c").write_text(source)
    (tmp_path / "b.cpp").write_text(source)
    assert scan(tmp_path, min_nodes=5, near_threshold=0.5).families == []


def test_real_parser_locations_above_line_256(tmp_path):
    source = "\n" * 400 + "fn f(x: i32) -> i32 { let y = load(x); y + 1 }"
    for name in ("a", "b"):
        (tmp_path / f"{name}.rs").write_text(source)
    report = scan(tmp_path, min_nodes=5, near_metric="subtree")
    assert report.stats["functions"] == 2
    assert {u.line for f in report.families for u in f.members} == {401}


def test_index_build_and_vendor_exclusions_are_reused(tmp_path):
    for rel in ("src/a.rs", "src/b.rs", "third_party/a.rs", "build/a.rs"):
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fn f(x: i32) -> i32 { let y = load(x); y + 1 }")
    report = scan(tmp_path, min_nodes=5)
    assert report.stats["functions"] == 2
    # The non-git inventory already omits build/; the shared filter removes vendor.
    assert report.stats["excluded_artifact_files"] >= 1


def test_scip_utf16_ranges_are_translated_to_utf8_columns(tmp_path):
    scip_pb2 = pytest.importorskip("victor.core.graph_rag._scip.scip_pb2")
    from victor.core.graph_rag.structural_clones import load_scip_occurrences, fresh_occurrence_map

    source = 'const s = "😀"; object.get(value);'
    start = len(source[: source.index("get")].encode("utf-16-le")) // 2
    index = scip_pb2.Index()
    doc = index.documents.add(relative_path="a.ts", position_encoding=2, text=source)
    doc.occurrences.add(
        range=[0, start, 0, start + 3], symbol="scip-typescript npm pkg 1 Store#get()."
    )
    path = tmp_path / "typescript.scip"
    path.write_bytes(index.SerializeToString())
    occurrences = load_scip_occurrences(path)["a.ts"]
    mapping = fresh_occurrence_map([source], occurrences)
    column = len(source[: source.index("get")].encode("utf-8"))
    assert mapping[(1, column)][0] == 3
    assert fresh_occurrence_map([source.replace("object", "other")], occurrences) is None


def test_scip_dump_older_than_source_falls_back_explicitly(tmp_path):
    import os

    scip_pb2 = pytest.importorskip("victor.core.graph_rag._scip.scip_pb2")

    source = "fn f(x: i32) -> i32 { let y = load(x); y + 1 }"
    for name in ("a", "b"):
        (tmp_path / f"{name}.rs").write_text(source)
    directory = tmp_path / ".victor" / "precise"
    directory.mkdir(parents=True)
    index = scip_pb2.Index()
    for name in ("a", "b"):
        index.documents.add(relative_path=f"{name}.rs")
    path = directory / "rust-analyzer.scip"
    path.write_bytes(index.SerializeToString())
    os.utime(path, (1, 1))
    report = scan(tmp_path, min_nodes=5)
    assert report.stats["stale_files"] == 2
    assert report.stats["scip_files"] == 0
