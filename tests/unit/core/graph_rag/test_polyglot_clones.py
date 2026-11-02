"""Automatic frontends, portable operation evidence and distinct FFI relationships."""

import hashlib
import json
import sqlite3

import pytest

from victor.core.graph_rag.structural_clones import report_to_dict, scan

PYTHON = """def calculate(x, y):
    total = x + y
    if total > 2:
        return total * 2
    return total - 1
"""

SOURCES = [
    ("rs", "fn calc(a:i32,b:i32)->i32 { let sum=a+b; if sum>2 { return sum*2; } sum-1 }"),
    ("js", "function calc(a,b) { let sum=a+b; if(sum>2) { return sum*2; } return sum-1; }"),
    (
        "ts",
        "function calc(a:number,b:number) { let sum=a+b; if(sum>2) { return sum*2; } return sum-1; }",
    ),
    (
        "tsx",
        "function calc(a:number,b:number) { let sum=a+b; if(sum>2) { return sum*2; } return sum-1; }",
    ),
    (
        "go",
        "package p\nfunc calc(a,b int) int { sum:=a+b; if sum>2 { return sum*2 }; return sum-1 }",
    ),
    (
        "java",
        "class A { int calc(int a,int b) { int sum=a+b; if(sum>2) { return sum*2; } return sum-1; } }",
    ),
    ("c", "int calc(int a,int b) { int sum=a+b; if(sum>2) { return sum*2; } return sum-1; }"),
    ("cpp", "int calc(int a,int b) { int sum=a+b; if(sum>2) { return sum*2; } return sum-1; }"),
    ("rb", "def calc(a,b)\n sum=a+b\n if sum>2\n return sum*2\n end\n sum-1\nend"),
]


def test_annotation_string_is_not_a_native_modifier(tmp_path):
    (tmp_path / "Native.java").write_text(
        'abstract class Native { @Note("native") abstract int run(int x); }'
    )
    (tmp_path / "native.c").write_text(
        "JNIEXPORT jint JNICALL Java_Native_run(JNIEnv* env,jobject obj,jint value){return value+1;}"
    )
    report = scan(tmp_path, cross_language=True, min_nodes=5)
    assert report.stats["ffi_jni_declarations"] == 0
    assert not report.ffi_links


@pytest.mark.parametrize("extension,source", SOURCES)
def test_portable_bodies_match_without_any_language_option(tmp_path, extension, source):
    (tmp_path / "reference.py").write_text(PYTHON)
    (tmp_path / f"native.{extension}").write_text(source)
    ordinary = scan(tmp_path, min_nodes=5, near_metric="subtree")
    assert not ordinary.families
    report = scan(tmp_path, min_nodes=5, cross_language=True, near_threshold=1)
    cross = [f for f in report.families if f.tier == "cross_language"]
    assert len(cross) == 1, report.stats
    assert cross[0].score == 1
    payload = report_to_dict(report, 10)
    assert payload["equivalence_verified"] is False
    assert payload["stats"]["portable_units"] == 2
    assert cross[0].files == [f"native.{extension}", "reference.py"]


@pytest.mark.parametrize(
    "source",
    [
        SOURCES[0][1].replace("a+b", "a-b"),
        SOURCES[0][1].replace("sum>2", "sum>3"),
        "fn calc(a:i32,b:i32)->i32 { b-a }",
    ],
)
def test_operators_constants_and_parameter_order_are_preserved(tmp_path, source):
    (tmp_path / "reference.py").write_text(PYTHON)
    (tmp_path / "native.rs").write_text(source)
    assert not any(
        f.tier == "cross_language"
        for f in scan(tmp_path, min_nodes=5, cross_language=True, near_threshold=1).families
    )


def test_parameter_positions_survive_subtree_alpha_renaming(tmp_path):
    (tmp_path / "reference.py").write_text(PYTHON.replace("x + y", "x - y"))
    (tmp_path / "native.rs").write_text(SOURCES[0][1].replace("a+b", "b-a"))
    report = scan(tmp_path, min_nodes=5, cross_language=True, near_threshold=1)
    assert report.stats["portable_units"] == 2
    assert not any(f.tier == "cross_language" for f in report.families)


def test_unsupported_effects_are_counted_and_never_dropped_from_a_matching_body(tmp_path):
    (tmp_path / "reference.py").write_text(
        PYTHON.replace("    total", "    await record()\n    total", 1).replace(
            "def ", "async def ", 1
        )
    )
    (tmp_path / "native.rs").write_text(SOURCES[0][1])
    report = scan(tmp_path, min_nodes=5, cross_language=True)
    assert report.stats["portable_unsupported_units"] == 1
    assert not report.stats["portable_coverage_complete"]
    assert not any(f.tier == "cross_language" for f in report.families)


def test_registered_scip_extensions_drive_parser_selection():
    from victor.core.graph_rag.language_parsers import SOURCE_LANGUAGES, SOURCE_PROVIDERS
    from victor.core.graph_rag.precise_providers import all_providers, extensions_for_provider

    for provider in all_providers():
        for extension in extensions_for_provider(provider.name):
            assert SOURCE_PROVIDERS[extension] == provider.name
            assert extension in SOURCE_LANGUAGES


def test_extension_detection_is_case_insensitive(tmp_path):
    (tmp_path / "reference.TS").write_text(SOURCES[2][1])
    (tmp_path / "native.RS").write_text(SOURCES[0][1])
    report = scan(tmp_path, min_nodes=5, cross_language=True, near_threshold=1)
    assert report.stats["portable_units"] == 2
    assert report.stats["cross_language_families"] == 1


def _ffi_graph(root, *, stale=False):
    root.mkdir(exist_ok=True)
    source = "def wrapper(x):\n    return native.calculate(x)\n"
    rust = "fn calculate(x:i32)->i32 { let value=x+2; value*3 }"
    (root / "wrapper.py").write_text(source)
    (root / "core.rs").write_text(rust)
    directory = root / ".victor"
    directory.mkdir()
    with sqlite3.connect(directory / "project.db") as db:
        db.executescript("""
          CREATE TABLE graph_node(node_id TEXT PRIMARY KEY, file TEXT, name TEXT,
            line INT, end_line INT, lang TEXT, type TEXT);
          CREATE TABLE graph_edge(src TEXT, dst TEXT, type TEXT, metadata TEXT);
          CREATE TABLE graph_file_mtime(file TEXT PRIMARY KEY, content_hash TEXT);
        """)
        db.executemany(
            "INSERT INTO graph_node VALUES(?,?,?,?,?,?,?)",
            [
                ("py", "wrapper.py", "wrapper", 1, 2, "python", "function"),
                ("rs", "core.rs", "calculate", 1, 1, "rust", "function"),
            ],
        )
        db.execute(
            "INSERT INTO graph_edge VALUES(?,?,?,?)",
            ("py", "rs", "REFERENCES", json.dumps({"kind": "FFI", "use": "calculate"})),
        )
        db.executemany(
            "INSERT INTO graph_file_mtime VALUES(?,?)",
            [
                ("wrapper.py", "stale" if stale else hashlib.sha256(source.encode()).hexdigest()),
                ("core.rs", hashlib.sha256(rust.encode()).hexdigest()),
            ],
        )


def test_indexed_ffi_call_is_a_relationship_not_a_duplicate(tmp_path):
    root = tmp_path / "project"
    _ffi_graph(root)
    payload = report_to_dict(scan(root, min_nodes=5, cross_language=True), 10)
    assert len(payload["ffi_links"]) == 1
    link = payload["ffi_links"][0]
    assert link["relationship"] == "calls_across_ffi"
    assert link["source"]["file"] == "wrapper.py"
    assert link["target"]["file"] == "core.rs"
    assert link["equivalence_verified"] is False
    assert not any(f["tier"] == "cross_language" for f in payload["families"])


def test_stale_ffi_index_is_not_attached_to_current_bodies(tmp_path):
    root = tmp_path / "project"
    _ffi_graph(root, stale=True)
    payload = report_to_dict(scan(root, min_nodes=5, cross_language=True), 10)
    assert payload["ffi_links"] == []
    assert payload["stats"]["ffi_stale_links"] == 1


def test_ffi_relationships_do_not_depend_on_the_clone_size_filter(tmp_path):
    root = tmp_path / "project"
    _ffi_graph(root)
    report = scan(root, min_nodes=100, cross_language=True)
    assert report.stats["functions"] == 0
    assert len(report.ffi_links) == 1
    assert report.ffi_links[0]["source"]["body_compared"] is False


def test_ffi_hashes_and_relationships_use_one_index_snapshot(tmp_path, monkeypatch):
    root = tmp_path / "project"
    _ffi_graph(root)
    database = root / ".victor" / "project.db"
    connect = sqlite3.connect
    with connect(database) as db:
        db.execute("PRAGMA journal_mode=WAL")
    changed = False

    def connection(*args, **kwargs):
        reader = connect(*args, **kwargs)

        def concurrent_index_update(sql):
            nonlocal changed
            if not changed and "FROM graph_edge e JOIN" in sql:
                changed = True
                with connect(database) as writer:
                    writer.execute("UPDATE graph_node SET name='new_generation' WHERE node_id='rs'")
                    writer.execute(
                        "UPDATE graph_file_mtime SET content_hash='new_generation' WHERE file='core.rs'"
                    )

        if kwargs.get("uri"):
            reader.set_trace_callback(concurrent_index_update)
        return reader

    monkeypatch.setattr(sqlite3, "connect", connection)
    report = scan(root, min_nodes=5, cross_language=True)
    assert changed, "the writer committed between the two index reads"
    assert report.ffi_links[0]["target"]["symbol"] == "calculate"
    with connect(database) as db:
        assert (
            db.execute("SELECT name FROM graph_node WHERE node_id='rs'").fetchone()[0]
            == "new_generation"
        )


@pytest.mark.parametrize("first,second", [("ts", "rs"), ("c", "cpp"), ("java", "cpp")])
def test_non_python_pairs_are_compared_automatically(tmp_path, first, second):
    for extension in (first, second):
        source = next(source for ext, source in SOURCES if ext == extension)
        (tmp_path / f"implementation.{extension}").write_text(source)
    families = scan(tmp_path, min_nodes=5, cross_language=True, near_threshold=1).families
    assert len([f for f in families if f.tier == "cross_language"]) == 1


@pytest.mark.parametrize("extension", ["c", "cpp"])
def test_jni_native_declaration_links_to_export_without_a_body_comparison(tmp_path, extension):
    (tmp_path / "Native.java").write_text(
        "package pkg; class Native_core { public native int sum(int x); }"
    )
    (tmp_path / f"native.{extension}").write_text(
        "JNIEXPORT jint JNICALL Java_pkg_Native_1core_sum(JNIEnv *env, jobject obj, jint x) { return x+1; }"
    )
    report = scan(tmp_path, min_nodes=5, cross_language=True)
    assert report.stats["parse_error_files"] == 0
    (link,) = report.ffi_links
    assert link["bridge"] == "JNI"
    assert link["source"]["symbol"] == "sum"
    assert link["source"]["has_body"] is False
    assert link["target"]["symbol"] == "Java_pkg_Native_1core_sum"
    assert link["runtime_verified"] is False
    assert link["abi_verified"] is False


def test_jni_overloads_use_parameter_descriptors(tmp_path):
    (tmp_path / "Native.java").write_text(
        "package pkg; class Native { native int sum(int x); native long sum(long x); }"
    )
    (tmp_path / "native.cpp").write_text(
        "JNIEXPORT jint JNICALL Java_pkg_Native_sum__I(JNIEnv *e, jobject o, jint x) { return x+1; }\n"
        "JNIEXPORT jlong JNICALL Java_pkg_Native_sum__J(JNIEnv *e, jobject o, jlong x) { return x+1; }\n"
    )
    report = scan(tmp_path, min_nodes=5, cross_language=True)
    assert {link["target"]["symbol"] for link in report.ffi_links} == {
        "Java_pkg_Native_sum__I",
        "Java_pkg_Native_sum__J",
    }


@pytest.mark.parametrize("extension", ["c", "cpp"])
@pytest.mark.parametrize(
    "java,native,result,status",
    [
        ("native int run(int x);", "JNIEnv *e, jobject o, jint x", "jint", "matching_declaration"),
        (
            "static native long run(long x);",
            "JNIEnv *e, jclass o, jlong x",
            "jlong",
            "matching_declaration",
        ),
        (
            "native int run(int x);",
            "JNIEnv *e, jobject o, jlong x",
            "jint",
            "conflicting_declaration",
        ),
        (
            "native int run(int x);",
            "JNIEnv *e, jobject o, jint x",
            "jlong",
            "conflicting_declaration",
        ),
        ("native int run(int x);", "JNIEnv *e, jobject o", "jint", "conflicting_declaration"),
        ("native int run(int x);", "JNIEnv *e, jobject o, int x", "jint", "unknown"),
        ("native int run(int x);", "JNIEnv *e, jobject o, const jint x", "jint", "unknown"),
        ("native int run(int x);", "JNIEnv *e, jobject o, jint *x", "jint", "unknown"),
        ("native int run(int x);", "JNIEnv *e, jobject o, ...", "jint", "unknown"),
        ('@Note("critical") native int run(int x);', "jint x", "jint", "unknown"),
        (
            "native void run(int a[], int[][] b);",
            "JNIEnv *e, jobject o, jintArray a, jobjectArray b",
            "void",
            "matching_declaration",
        ),
        (
            "native java.lang.String run();",
            "JNIEnv *e, jobject o",
            "jstring",
            "unknown",
        ),
        (
            "native java.lang.String run()[];",
            "JNIEnv *e, jobject o",
            "jobjectArray",
            "unknown",
        ),
        (
            "native int run(/*comment*/ Native this, int x);",
            "JNIEnv *e, jobject o, jint x",
            "jint",
            "matching_declaration",
        ),
        (
            "native int run(Unknown x, long y);",
            "JNIEnv *e, jobject o, jobject x, jint y",
            "jint",
            "conflicting_declaration",
        ),
    ],
)
def test_jni_declared_contracts_preserve_links_and_qualification(
    tmp_path, extension, java, native, result, status
):
    (tmp_path / "Native.java").write_text("class Native { " + java + " }")
    (tmp_path / f"native.{extension}").write_text(
        f"JNIEXPORT {result} JNICALL Java_Native_run({native}) {{ return 0; }}"
    )
    report = scan(tmp_path, min_nodes=5, cross_language=True)
    assert report.stats["parse_error_files"] == 0
    (link,) = report.ffi_links
    assert link["declared_contract"]["status"] == status
    assert link["abi_verified"] is False
    assert link["runtime_verified"] is False
    assert not report.families
    suffix = {
        "matching_declaration": "matching",
        "conflicting_declaration": "conflicting",
        "unknown": "unknown",
    }[status]
    assert report.stats[f"ffi_jni_{suffix}_declarations"] == 1


def test_jni_duplicate_exports_are_not_arbitrarily_resolved(tmp_path):
    (tmp_path / "Native.java").write_text("package pkg; class Native { native int sum(int x); }")
    source = (
        "JNIEXPORT jint JNICALL Java_pkg_Native_sum(JNIEnv *e, jobject o, jint x) { return x+1; }"
    )
    for name in ("first.c", "second.c"):
        (tmp_path / name).write_text(source)
    report = scan(tmp_path, min_nodes=5, cross_language=True)
    assert report.ffi_links == []
    assert report.stats["ffi_ambiguous_bindings"] == 1


@pytest.mark.parametrize("extension", ["c", "cpp"])
def test_jni_callback_parameter_does_not_match_primitive_value(tmp_path, extension):
    (tmp_path / "Native.java").write_text("class Native { native int run(int x); }")
    (tmp_path / f"native.{extension}").write_text(
        "JNIEXPORT jint JNICALL Java_Native_run(JNIEnv *e, jobject o, jint callback(jint)) { return 0; }"
    )
    report = scan(tmp_path, cross_language=True)
    assert report.stats["parse_error_files"] == 0
    assert report.ffi_links[0]["declared_contract"]["status"] == "unknown"


@pytest.mark.parametrize(
    "declaration",
    [
        "class Native { static class String {} native int run(String x); }",
        "class Native<String> { native int run(String x); }",
        "class Native { static class java { static class lang { static class String {} } } native int run(java.lang.String x); }",
    ],
)
def test_jni_reference_names_are_not_resolved_by_import_spelling(tmp_path, declaration):
    (tmp_path / "Native.java").write_text("import java.lang.String; " + declaration)
    (tmp_path / "native.c").write_text(
        "JNIEXPORT jint JNICALL Java_Native_run(JNIEnv *e, jobject o, jstring x) { return 0; }"
    )
    report = scan(tmp_path, cross_language=True)
    assert report.stats["parse_error_files"] == 0
    assert report.ffi_links[0]["declared_contract"]["status"] == "unknown"


def test_jni_reference_contract_can_use_fresh_compiler_type_identity():
    from victor.core.graph_rag.ffi_bindings import collect_binding_facts, jni_relationships
    from victor.core.graph_rag.language_parsers import create_parser, parser_input

    source = b"class Native { native int run(String x); }"
    column = source.index(b"String")
    occurrences = {(1, column): (6, "scip-java maven java.base 26 java/lang/String#")}
    tree = create_parser("java").parse(source)
    facts = collect_binding_facts("Native.java", source, tree.root_node, "java", occurrences)
    native = (
        b"JNIEXPORT jint JNICALL Java_Native_run(JNIEnv *e, jobject o, jstring x) { return 0; }"
    )
    native_tree = create_parser("c").parse(parser_input(native, "c"))
    facts += collect_binding_facts("native.c", native, native_tree.root_node, "c")
    links, stats = jni_relationships(facts)
    assert links[0]["declared_contract"]["status"] == "matching_declaration"
    assert stats["ffi_jni_matching_declarations"] == 1


@pytest.mark.parametrize(
    "java,owner",
    [
        (
            "class Outer { void make(){ class Local { native int sum(int x); } } }",
            "Outer_00024Local",
        ),
        ("class Outer { Object make(){return new Object(){native int sum(int x);};}}", "Outer"),
        ("enum Outer { VALUE { native int sum(int x); }; }", "Outer"),
    ],
)
def test_jni_local_and_anonymous_binary_owners_are_not_guessed(tmp_path, java, owner):
    (tmp_path / "Outer.java").write_text("package pkg; " + java)
    (tmp_path / "native.c").write_text(
        f"JNIEXPORT jint JNICALL Java_pkg_{owner}_sum(JNIEnv *e, jobject o, jint x) {{ return x; }}"
    )
    report = scan(tmp_path, cross_language=True)
    assert report.stats["parse_error_files"] == 0
    assert not report.ffi_links
    assert report.stats["ffi_jni_declarations"] == 1
    assert report.stats["ffi_unresolved_bindings"] == 1
    assert report.stats["ffi_jni_unresolved_owners"] == 1


def test_jni_direct_member_nested_class_keeps_known_owner(tmp_path):
    (tmp_path / "Outer.java").write_text(
        "package pkg; class Outer { static class Inner { native int sum(int x); } }"
    )
    (tmp_path / "native.c").write_text(
        "JNIEXPORT jint JNICALL Java_pkg_Outer_00024Inner_sum(JNIEnv *e, jobject o, jint x) { return x; }"
    )
    report = scan(tmp_path, cross_language=True)
    assert report.ffi_links[0]["declared_contract"]["status"] == "matching_declaration"
    assert report.stats["ffi_jni_unresolved_owners"] == 0


def test_jni_overloaded_short_export_prevents_a_runtime_binding_guess(tmp_path):
    (tmp_path / "Native.java").write_text(
        "package pkg; class Native { native int sum(int x); native long sum(long x); }"
    )
    (tmp_path / "native.cpp").write_text(
        "JNIEXPORT jint JNICALL Java_pkg_Native_sum(JNIEnv *e, jobject o, jint x) { return x; }\n"
        "JNIEXPORT jlong JNICALL Java_pkg_Native_sum__J(JNIEnv *e, jobject o, jlong x) { return x; }\n"
    )
    report = scan(tmp_path, min_nodes=5, cross_language=True)
    assert report.ffi_links == []
    assert report.stats["ffi_ambiguous_bindings"] == 2


def test_jni_static_function_is_not_an_export(tmp_path):
    (tmp_path / "Native.java").write_text("package pkg; class Native { native int sum(int x); }")
    (tmp_path / "native.c").write_text(
        "JNIEXPORT static jint JNICALL Java_pkg_Native_sum(JNIEnv *e, jobject o, jint x) { return x; }"
    )
    report = scan(tmp_path, min_nodes=5, cross_language=True)
    assert report.ffi_links == []
    assert report.stats["ffi_unresolved_bindings"] == 1


def test_jni_escapes_utf16_units_and_array_descriptors():
    from victor.core.graph_rag.ffi_bindings import jni_escape

    assert jni_escape("pkg/A_b") == "pkg_A_1b"
    assert jni_escape("[Ljava/lang/String;") == "_3Ljava_lang_String_2"
    assert jni_escape("😀") == "_0d83d_0de00"


def test_jni_macro_shim_preserves_offsets_comments_and_literals():
    from victor.core.graph_rag.language_parsers import parser_input, jni_exports, create_parser

    source = (
        b"/*\nJNIEXPORT jint JNICALL Java_fake_comment()\n*/\n"
        b'const char *message = R"tag(\nJNIEXPORT jint JNICALL Java_fake_string()\n)tag";\n'
        b'extern "C" JNIEXPORT jint JNICALL\nJava_pkg_Native_sum(JNIEnv *e, jobject o, jint x) { return x+1; }'
    )
    prepared = parser_input(source, "cpp")
    assert len(prepared) == len(source)
    assert prepared.count(b"\n") == source.count(b"\n")
    assert prepared[: source.index(b'extern "C"')] == source[: source.index(b'extern "C"')]
    assert jni_exports(source) == {("Java_pkg_Native_sum", source.index(b"Java_pkg_Native_sum"))}
    assert not create_parser("cpp").parse(prepared).root_node.has_error


def test_jni_array_overload_descriptor(tmp_path):
    (tmp_path / "Native.java").write_text("package pkg; class Native { native int sum(int[] x); }")
    (tmp_path / "native.c").write_text(
        "JNIEXPORT jint JNICALL Java_pkg_Native_sum___3I(JNIEnv *e, jobject o, jintArray x) { return 1; }"
    )
    links = scan(tmp_path, min_nodes=5, cross_language=True).ffi_links
    assert len(links) == 1
    assert links[0]["target"]["symbol"] == "Java_pkg_Native_sum___3I"


def test_jni_reference_overloads_use_scip_binary_type_identity(tmp_path):
    scip_pb2 = pytest.importorskip("victor.core.graph_rag._scip.scip_pb2")
    source = "package pkg; class Native { native int sum(String x); native int sum(Object x); }"
    (tmp_path / "Native.java").write_text(source)
    (tmp_path / "native.cpp").write_text(
        "JNIEXPORT jint JNICALL Java_pkg_Native_sum__Ljava_lang_String_2(JNIEnv *e, jobject o, jstring x) { return 0; }\n"
        "JNIEXPORT jint JNICALL Java_pkg_Native_sum__Ljava_lang_Object_2(JNIEnv *e, jobject o, jobject x) { return 0; }\n"
    )
    # Implicit reference names are not guessed: a package-local String could
    # shadow java.lang.String. Fresh compiler identities disambiguate them.
    without = scan(tmp_path, min_nodes=5, cross_language=True)
    assert without.ffi_links == []
    assert without.stats["ffi_unresolved_bindings"] == 2
    index = scip_pb2.Index()
    doc = index.documents.add(relative_path="Native.java", text=source, position_encoding=1)
    for name in ("String", "Object"):
        start = source.index(name)
        doc.occurrences.add(
            range=[0, start, start + len(name)], symbol=f"scip-java maven jdk 25 java/lang/{name}#"
        )
    directory = tmp_path / ".victor" / "precise"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "scip-java.scip").write_bytes(index.SerializeToString())
    with_compiler = scan(tmp_path, min_nodes=5, cross_language=True)
    assert len(with_compiler.ffi_links) == 2
    assert with_compiler.stats["ffi_unresolved_bindings"] == 0


@pytest.mark.parametrize(
    "filename,source,symbol",
    [
        (
            "Native.cpp",
            "JNIEXPORT jint JNICALL Java_pkg_Native_sum(JNIEnv *e, jobject o, jint x) { return x+1; }",
            "Java_pkg_Native_sum",
        ),
        ("View.tsx", "function View() { return <span>hello</span>; }", "View"),
        ("worker.rb", "def compute(x)\n x + 1\nend", "compute"),
    ],
)
def test_index_fallback_uses_the_same_frontend_inventory(tmp_path, filename, source, symbol):
    from unittest.mock import MagicMock
    from victor.core.graph_rag.config import GraphIndexConfig
    from victor.core.graph_rag.indexing import GraphIndexingPipeline
    from victor.core.graph_rag.language_parsers import create_parser, parser_input

    pipeline = GraphIndexingPipeline(
        MagicMock(), GraphIndexConfig(root_path=tmp_path, enable_ccg=False, enable_embeddings=False)
    )
    path = tmp_path / filename
    language = pipeline._detect_language(path)
    tree = create_parser(language).parse(parser_input(source.encode(), language))
    assert not tree.root_node.has_error
    nodes = pipeline._extract_definitions(tree.root_node, path, language, source)
    assert symbol in {node.name for node in nodes}
    assert all(node.type != "unknown" for node in nodes)
