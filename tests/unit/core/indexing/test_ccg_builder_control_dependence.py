"""Builtin and native facade CDG share qualified, bounded graph semantics."""

from types import SimpleNamespace as N
from pathlib import Path

import pytest

from victor.core.indexing.ccg_builder import CodeContextGraphBuilder


def node(name, kind="expression"):
    return N(node_id=name, statement_type=kind, file="fixture.py", metadata={})


async def test_builder_excludes_join_from_branch_dependencies():
    builder = CodeContextGraphBuilder(builder_resolution="builtin")
    nodes = [node("branch", "condition"), node("left"), node("right"), node("join"), node("after")]
    edges = [
        N(src=a, dst=b)
        for a, b in [
            ("branch", "left"),
            ("branch", "right"),
            ("left", "join"),
            ("right", "join"),
            ("join", "after"),
        ]
    ]
    result = await builder._build_cdg(nodes, edges)
    assert {(e.src, e.dst) for e in result} == {("branch", "left"), ("branch", "right")}
    assert all(e.metadata["algorithm"] == "postdominance-exit-paths-v1" for e in result)
    assert all(e.metadata["program_semantics_verified"] is False for e in result)
    assert all(n.metadata["cfg_provenance"] == "source_heuristic" for n in nodes)


async def test_incomplete_or_invalid_cfg_has_observable_abstention():
    builder = CodeContextGraphBuilder(builder_resolution="builtin")
    for nodes, edges, reason in [
        ([node("loop", "loop")], [N(src="loop", dst="loop")], "non_exit_reaching_nodes"),
        ([node("a")], [N(src="a", dst="missing")], "invalid_cfg"),
        ([node("a")], [N(src="missing", dst="a")], "invalid_cfg"),
        ([node("a"), node("a")], [], "invalid_cfg"),
    ]:
        assert not await builder._build_cdg(nodes, edges)
        assert all(n.metadata["cdg_complete"] is False for n in nodes)
        assert all(n.metadata["cdg_reason"] == reason for n in nodes)


def test_native_facade_discards_legacy_reachability_records(monkeypatch):
    import victor.processing.native.ccg_extract as native

    class UnsafeLegacyCdg:
        def __iter__(self):
            raise AssertionError("Legacy reachability CDG must never be consumed")

    records = [
        N(
            node_id=name,
            node_type="if_statement" if name == "branch" else "expression_statement",
            start_line=i,
            end_line=i,
            scope_id="function",
            statement_type="condition" if name == "branch" else "expression",
            visibility="private",
            content_preview=name,
        )
        for i, name in enumerate(("branch", "left", "right", "join", "after"), 1)
    ]
    cfg = [
        N(src=a, dst=b, edge_type="CFG_SUCCESSOR", structural=False)
        for a, b in [
            ("branch", "left"),
            ("branch", "right"),
            ("left", "join"),
            ("right", "join"),
            ("join", "after"),
        ]
    ]
    result = N(statements=records, cfg_edges=cfg, cdg_edges=UnsafeLegacyCdg(), ddg_edges=[])
    monkeypatch.setattr(native, "build_ccg", lambda *a: result)
    nodes, edges = CodeContextGraphBuilder(builder_resolution="native")._build_native_ccg(
        Path("fixture.py"), "python", ""
    )
    cdg = [e for e in edges if str(getattr(e.type, "value", e.type)).startswith("CDG")]
    assert {(e.src, e.dst) for e in cdg} == {("branch", "left"), ("branch", "right")}
    assert all(n.metadata["cdg_complete"] for n in nodes)


@pytest.mark.parametrize("prefix", ["", "\n" * 300])
def test_source_statement_ids_distinguish_same_line_calls(prefix):
    from victor.core.graph_rag.language_parsers import create_parser

    parser = create_parser("python")
    if parser is None:
        pytest.skip("Python grammar unavailable")
    source = prefix + "def f():\n    return outer(inner(), sibling())\n"
    tree = parser.parse(source.encode())
    builder = CodeContextGraphBuilder()
    builder._extract_statement_nodes_native = lambda *args: None
    nodes = builder._extract_statement_nodes(
        tree.root_node, Path("calls.py"), source.split("\n"), source_code=source
    )
    calls = [n for n in nodes if n.statement_type == "call"]
    assert len(calls) == 3
    assert len({n.node_id for n in calls}) == 3
    assert all(n.line == prefix.count("\n") + 2 for n in calls)


def test_reference_coordinates_require_original_bytes_when_root_span_is_trimmed():
    from victor.core.graph_rag.language_parsers import create_parser

    parser = create_parser("python")
    if parser is None:
        pytest.skip("Python grammar unavailable")
    source = "\r\n" * 50 + "def f():\r\n    return outer(inner(), sibling())\r\n"
    tree = parser.parse(source.encode())
    builder = CodeContextGraphBuilder()
    with pytest.raises(ValueError, match="Full original source"):
        builder._extract_statement_nodes(tree.root_node, Path("crlf.py"), source.splitlines())
    builder._extract_statement_nodes_native = lambda *args: None
    nodes = builder._extract_statement_nodes(
        tree.root_node, Path("crlf.py"), source.splitlines(), source_code=source
    )
    assert [(n.line, n.end_line) for n in nodes] == [
        (51, 52),
        (52, 52),
        (52, 52),
        (52, 52),
        (52, 52),
    ]
    assert len({n.node_id for n in nodes}) == len(nodes)


def test_unretained_real_parser_input_is_not_reconstructed_from_split_lines():
    from victor.core.indexing.ccg_builder import _source_bytes

    with pytest.raises(ValueError, match="Full original source"):
        _source_bytes(N(text=None, start_byte=0), None)


async def test_file_builder_preserves_crlf_bytes_before_parsing(tmp_path):
    source = b"\r\ndef f():\r\n    return outer(inner(), sibling())\r\n"
    path = tmp_path / "crlf.py"
    path.write_bytes(source)
    builder = CodeContextGraphBuilder(builder_resolution="native")
    observed = []

    def reference_only(file, language, text):
        observed.append(text.encode("utf-8"))
        return None

    builder._build_native_ccg = reference_only
    builder._extract_statement_nodes_native = lambda *args: None
    nodes, _ = await builder.build_ccg_for_file(path, "python")
    assert observed == [source]
    assert len({n.node_id for n in nodes}) == len(nodes)
