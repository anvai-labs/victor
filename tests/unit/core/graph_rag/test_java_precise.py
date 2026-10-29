"""Java producer shapes captured from scip-java 0.13.2 on Iceberg.

SCIP messages here are synthetic boundary tests, not compiler qualification.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from victor.core.graph_rag.precise_index import join_occurrences
from victor.core.graph_rag.precise_providers import DumpSkipped, ScipJavaProvider
from victor.storage.graph.protocol import GraphNode

JV = ScipJavaProvider()
PREFIX = "scip-java maven . . org/apache/iceberg/Example#"


def test_overload_ordinals_are_names_not_signatures():
    for suffix in ("()", "(+1)", "(+27)"):
        assert JV.symbol_parts(PREFIX + "create" + suffix + ".")["path"] == "Example.create"
        assert JV.symbol_parts(PREFIX + "`<init>`" + suffix + ".")["path"] == "Example.constructor"
    for suffix in ("(1)", "(+0)", "(-1)", "(+01)", "(type)", "().[T]", ".field"):
        assert JV.symbol_parts(PREFIX + "create" + suffix + ".") is None


def _overloads(encoding):
    from victor.core.graph_rag._scip import scip_pb2

    index = scip_pb2.Index()
    doc = index.documents.add(relative_path="Example.java")
    for name, line in (("create()", 2), ("create(+1)", 6), ("run()", 10)):
        occ = doc.occurrences.add(symbol=PREFIX + name + ".", symbol_roles=1)
        if encoding == "packed":
            occ.range.extend([line, 0])
        elif encoding == "single":
            occ.single_line_range.line = line
        elif encoding == "multi":
            occ.multi_line_range.start_line = line
    for name in ("create()", "create(+1)"):
        occ = doc.occurrences.add(symbol=PREFIX + name + ".")
        occ.range.extend([11, 4])
    nodes = [
        GraphNode("example", "class", "Example", "Example.java", line=1, end_line=15),
        GraphNode(
            "zero", "method", "create", "Example.java", line=3, end_line=5, parent_id="example"
        ),
        GraphNode(
            "one", "method", "create", "Example.java", line=7, end_line=9, parent_id="example"
        ),
        GraphNode(
            "run", "method", "run", "Example.java", line=11, end_line=14, parent_id="example"
        ),
    ]
    return index, nodes


@pytest.mark.parametrize("encoding", ["packed", "single", "multi"])
def test_definition_ranges_join_overloads_without_guessing_ordinals(encoding):
    pytest.importorskip("google.protobuf")
    index, nodes = _overloads(encoding)
    result = join_occurrences(index, {"Example.java": nodes}, symbol_parts=JV.symbol_parts)
    assert {(a, b) for a, b, _, _ in result.edges} == {("run", "zero"), ("run", "one")}
    assert result.definitions_unjoined == 0


@pytest.mark.parametrize("fault", ["missing_bounds", "overlap", "absent_range", "outside_range"])
def test_ambiguous_or_unbounded_overload_stays_unjoined(fault):
    pytest.importorskip("google.protobuf")
    index, nodes = _overloads("packed")
    if fault == "missing_bounds":
        nodes[1].end_line = None
    elif fault == "overlap":
        nodes[1].end_line = 9
        nodes[2].line = 3
    elif fault == "absent_range":
        for occ in index.documents[0].occurrences[:2]:
            occ.ClearField("range")
    else:
        for occ in index.documents[0].occurrences[:2]:
            occ.range[0] = 50
    result = join_occurrences(index, {"Example.java": nodes}, symbol_parts=JV.symbol_parts)
    assert result.definitions_unjoined >= 1
    assert all(target != "one" for _, target, _, _ in result.edges)


def test_interface_members_with_identical_names_keep_container_scope():
    # Multiple interfaces in a file must not lose their names in parent traversal.
    def occ(symbol, line, roles):
        return SimpleNamespace(symbol=symbol, range=[line, 0], symbol_roles=roles)

    base = "scip-java maven . . demo/"
    doc = SimpleNamespace(
        relative_path="Example.java",
        symbols=[],
        occurrences=[
            occ(base + "Left#value().", 2, 1),
            occ(base + "Right#value().", 6, 1),
            occ(base + "Right#value().", 3, 0),
        ],
    )
    nodes = [
        GraphNode("left", "interface", "Left", "Example.java"),
        GraphNode("right", "interface", "Right", "Example.java"),
        GraphNode("lv", "method", "value", "Example.java", parent_id="left"),
        GraphNode("rv", "method", "value", "Example.java", parent_id="right"),
    ]
    result = join_occurrences(
        SimpleNamespace(documents=[doc]), {"Example.java": nodes}, symbol_parts=JV.symbol_parts
    )
    assert [(a, b) for a, b, _, _ in result.edges] == [("lv", "rv")]
    assert result.definitions_unjoined == 0


def test_actual_java_parser_projection_joins_explicit_constructor_overloads():
    pytest.importorskip("google.protobuf")
    pytest.importorskip("tree_sitter_language_pack")
    from victor.core.graph_rag._scip import scip_pb2
    from victor.core.graph_rag.codegraph_repository import project_file_symbols
    from victor_codegraph.treesitter_parser import parse_treesitter

    parsed = parse_treesitter(
        "class Example {\n Example() {}\n Example(int n) {}\n void run() { new Example(); }\n}",
        "Example.java",
        "java",
    )
    nodes = project_file_symbols(parsed)
    index = scip_pb2.Index()
    doc = index.documents.add(relative_path="Example.java")
    for name, line in (("`<init>`()", 1), ("`<init>`(+1)", 2), ("run()", 3)):
        occ = doc.occurrences.add(symbol=PREFIX + name + ".", symbol_roles=1)
        occ.single_line_range.line = line
    ref = doc.occurrences.add(symbol=PREFIX + "`<init>`().")
    ref.single_line_range.line = 3
    ref.single_line_range.start_character = 25
    result = join_occurrences(index, {"Example.java": nodes}, symbol_parts=JV.symbol_parts)
    assert result.definitions_unjoined == 0
    assert len(result.edges) == 1
    by_id = {node.node_id: node for node in nodes}
    owner, target, _, _ = result.edges[0]
    assert by_id[owner].name == "run"
    assert by_id[target].type == "constructor"
    assert by_id[target].signature == "()"


@pytest.mark.parametrize(
    "marker", ["MODULE.bazel", "WORKSPACE.bazel", "WORKSPACE", "gradlew", "pom.xml"]
)
def test_supported_build_markers_and_bazel_indexer_path(tmp_path, monkeypatch, marker):
    captured = []
    monkeypatch.setattr(
        "victor.core.graph_rag.precise_providers._run_dump",
        lambda cmd, **kwargs: captured.append(cmd),
    )
    (tmp_path / marker).touch()
    tool = str(tmp_path / "bin/scip-java")
    JV.dump(tool, tmp_path, tmp_path / "out.scip", timeout_seconds=1)
    assert captured[0][:2] == [tool, "index"]
    bazel = marker in JV._BAZEL_BUILD_MARKERS
    assert (f"--bazel-scip-java-binary={Path(tool).resolve()}" in captured[0]) == bazel


@pytest.mark.parametrize("other_marker", ["pom.xml", "gradlew", "settings.gradle.kts"])
def test_mixed_build_roots_keep_required_bazel_aspect_input(tmp_path, monkeypatch, other_marker):
    captured = []
    monkeypatch.setattr(
        "victor.core.graph_rag.precise_providers._run_dump",
        lambda cmd, **kwargs: captured.append(cmd),
    )
    (tmp_path / "MODULE.bazel").touch()
    (tmp_path / other_marker).touch()
    tool = str(tmp_path / "bin/scip-java")
    JV.dump(tool, tmp_path, tmp_path / "out.scip", timeout_seconds=1)
    assert f"--bazel-scip-java-binary={Path(tool).resolve()}" in captured[0]


def test_bazel_relative_binary_uses_subprocess_root_not_caller_directory(tmp_path, monkeypatch):
    captured = []
    root = tmp_path / "project"
    root.mkdir()
    (root / "MODULE.bazel").touch()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "victor.core.graph_rag.precise_providers._run_dump",
        lambda cmd, **kwargs: captured.append(cmd),
    )
    JV.dump("bin/scip-java", root, root / "out.scip", timeout_seconds=1)
    assert captured[0][0] == "bin/scip-java"
    assert f"--bazel-scip-java-binary={root / 'bin/scip-java'}" in captured[0]
    assert f"--bazel-scip-java-binary={tmp_path / 'bin/scip-java'}" not in captured[0]


def test_bazel_bare_binary_matches_path_lookup(tmp_path, monkeypatch):
    captured = []
    (tmp_path / "MODULE.bazel").touch()
    selected = tmp_path / "tools/scip-java"
    monkeypatch.setattr(
        "victor.core.graph_rag.precise_providers.shutil.which", lambda _: str(selected)
    )
    monkeypatch.setattr(
        "victor.core.graph_rag.precise_providers._run_dump",
        lambda cmd, **kwargs: captured.append(cmd),
    )
    JV.dump("scip-java", tmp_path, tmp_path / "out.scip", timeout_seconds=1)
    assert f"--bazel-scip-java-binary={selected}" in captured[0]


@pytest.mark.parametrize("marker", ["build.sbt", "MILL_VERSION", "build.xml"])
def test_unsupported_builds_skip_before_execution(tmp_path, monkeypatch, marker):
    def no_execution(*args, **kwargs):
        pytest.fail("unsupported build must not execute")

    monkeypatch.setattr("victor.core.graph_rag.precise_providers._run_dump", no_execution)
    (tmp_path / marker).touch()
    result = JV.dump("scip-java", tmp_path, tmp_path / "out.scip", timeout_seconds=1)
    assert isinstance(result, DumpSkipped)
    assert result.reason == "no-jvm-build"
