# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from victor_coding.codebase.ccg_builder import PluginBackedCCGBuilder


def test_plugin_backed_ccg_builder_supports_registered_tree_sitter_languages() -> None:
    """Languages with real control flow opt in via
    ``LanguageCapabilities.supports_control_flow_graph``. Markup, schema,
    build, and pure-data formats correctly opt out — having a grammar
    isn't sufficient; CCG also requires actual control flow.
    """
    builder = PluginBackedCCGBuilder()

    for language in [
        # Core programming languages
        "python",
        "javascript",
        "typescript",
        "rust",
        "go",
        "java",
        "cpp",
        "c",
        "kotlin",
        "csharp",
        "ruby",
        "php",
        "swift",
        "scala",
        "bash",
        "lua",
        "elixir",
        "haskell",
        "r",
        # Post-TSA additions with real control flow
        "zig",
        "julia",
        "ocaml",
        "solidity",
        "perl",
        "objc",
        "groovy",
        # Hardware-description + shader languages (HDL process/always blocks
        # have if/case/for/while/loop; GLSL is straight C-like).
        "vhdl",
        "verilog",
        "glsl",
    ]:
        assert builder.supports_language(language), language

    # Markup / build / schema / pure-data formats: have grammars but no
    # control flow, so CCG correctly skips them. SQL is here too because
    # the installed tree-sitter-sql grammar is ANSI-SELECT focused and
    # produces ERROR nodes for PL/pgSQL BEGIN/IF/WHILE blocks — the
    # control flow exists in source but the grammar can't see it.
    for language in [
        "markdown",
        "xml",
        "html",
        "css",
        "json",
        "yaml",
        "toml",
        "ini",
        "hocon",
        "make",
        "cmake",
        "graphql",
        "hcl",
        "sql",
    ]:
        assert not builder.supports_language(language), language


@pytest.mark.asyncio
async def test_plugin_backed_ccg_builder_builds_rich_rust_graph(tmp_path: Path) -> None:
    pytest.importorskip("tree_sitter_rust")
    builder = PluginBackedCCGBuilder()
    source = """
pub fn schedule(items: Vec<i32>) -> i32 {
    let mut total = 0;
    for item in items {
        if item > 0 {
            total += item;
        }
    }
    println!("{}", total);
    return total;
}
"""
    file_path = tmp_path / "scheduler.rs"
    file_path.write_text(source, encoding="utf-8")

    nodes, edges = await builder.build_ccg_for_file(file_path, "rust")

    statement_types = {node.statement_type for node in nodes}
    edge_types = {getattr(edge.type, "value", edge.type) for edge in edges}
    metadata_sources = {node.metadata.get("source") for node in nodes}

    assert "function_def" in statement_types
    assert "loop" in statement_types
    assert "condition" in statement_types
    assert "assignment" in statement_types
    assert "call" in statement_types
    assert "return" in statement_types
    assert "CFG_SUCCESSOR" in edge_types or "CFG_TRUE" in edge_types
    assert "CDG" in edge_types or "CDG_LOOP" in edge_types
    assert "DDG_DEF_USE" in edge_types
    assert metadata_sources == {"victor-coding-language-plugin"}


def test_coding_plugin_registers_ccg_builder() -> None:
    from victor_coding.plugin import CodingPlugin

    class Context:
        def __init__(self) -> None:
            self.builders: list[tuple[str, Any]] = []

        def register_vertical(self, vertical: Any) -> None:
            self.vertical = vertical

        def register_ccg_builder(self, language: str, builder: Any) -> None:
            self.builders.append((language, builder))

    context = Context()
    CodingPlugin().register(context)  # type: ignore[arg-type]

    assert len(context.builders) == 1
    language, builder = context.builders[0]
    assert language == "all"
    assert isinstance(builder, PluginBackedCCGBuilder)
    assert builder.supports_language("rust")


def test_ccg_is_path_independent() -> None:
    """Identical content indexed from differently-named directories must
    produce identical graphs. Regression: CFG/DDG ordered records by the
    path-derived node_id, so same-line statements chained and resolved
    latest-definition differently per directory (measured on the boundary
    corpus: 3333 vs 3331 vs 3321 edges for byte-identical source)."""
    import asyncio

    source = (
        "import os\n"
        "VALUE = compute(os.getcwd())\n"
        "\n"
        "\n"
        "def compute(path: str) -> str:\n"
        "    prefix = path.strip()\n"
        "    result = wrap(prefix, len(prefix))\n"
        "    return result\n"
        "\n"
        "\n"
        "def wrap(text: str, width: int) -> str:\n"
        "    padded = text.ljust(width)\n"
        "    return padded\n"
    )
    builder = PluginBackedCCGBuilder()

    def build(directory: str) -> tuple[list[Any], list[Any]]:
        import tempfile

        target = Path(tempfile.mkdtemp()) / directory / "sample.py"
        target.parent.mkdir(parents=True)
        target.write_text(source, encoding="utf-8")
        return asyncio.run(builder.build_ccg_for_file(target))

    nodes_a, edges_a = build("a")
    nodes_b, edges_b = build("a_much_longer_directory_name")

    assert nodes_a and edges_a, "sample must yield statements and edges"

    def span_of(nodes: list[Any]) -> dict[str, tuple[Any, ...]]:
        return {n.node_id: (n.line, n.end_line, n.statement_type, n.name) for n in nodes}

    def node_signature(nodes: list[Any]) -> list[tuple[Any, ...]]:
        return sorted(
            (
                n.line,
                n.end_line,
                n.statement_type,
                n.name,
                tuple(sorted((n.metadata or {}).items())),
            )
            for n in nodes
        )

    def edge_signature(nodes: list[Any], edges: list[Any]) -> list[tuple[Any, ...]]:
        span = span_of(nodes)
        normalized = []
        for edge in edges:
            metadata = tuple(
                sorted(
                    (key, value) for key, value in (edge.metadata or {}).items() if key != "file"
                )
            )
            normalized.append((span.get(edge.src), span.get(edge.dst), str(edge.type), metadata))
        return sorted(normalized, key=repr)

    assert len(edges_a) == len(edges_b)
    assert node_signature(nodes_a) == node_signature(nodes_b)
    assert edge_signature(nodes_a, edges_a) == edge_signature(nodes_b, edges_b)
    # The same-line chain must follow source order, not hash order: the
    # assignment statement at line 2 precedes the call it wraps.
    spans = span_of(nodes_a)
    successors = {
        (spans[e.src], spans[e.dst]) for e in edges_a if str(e.type).endswith("CFG_SUCCESSOR")
    }
    same_line = {
        edge
        for edge in successors
        if edge[0] is not None and edge[1] is not None and edge[0][0] == edge[1][0] == 2
    }
    assert same_line, "same-line statements must be chained"


def _graph_signature(nodes, edges):
    return (
        sorted((n.node_id, n.name, tuple(sorted((n.metadata or {}).items()))) for n in nodes),
        sorted(
            (e.src, e.dst, str(e.type), tuple(sorted((e.metadata or {}).items()))) for e in edges
        ),
    )


@pytest.mark.parametrize("sample", ["sample.py", "sample.ts"], ids=["python", "typescript"])
def test_native_walk_flip_matches_python_walk(sample, tmp_path):
    """Option B phase 2: with the native extractor injected, the builder's
    graph must be IDENTICAL to its Python walk — nodes, metadata, edges,
    edge metadata. Skips cleanly on pure-Python installs (the flip is
    availability-gated); the injected-native path only runs where the
    extension serves the language."""
    import asyncio
    import shutil

    try:
        from victor.processing.native.plugin_ccg import (
            extract_plugin_statements,
            plugin_statements_available,
        )
    except ImportError:
        pytest.skip("native extension not built")

    language = {"sample.py": "python", "sample.ts": "typescript"}[sample]
    if not plugin_statements_available(language):
        pytest.skip(f"no native grammar for {language}")

    source = Path(__file__).parent / "goldens" / "samples" / sample
    target = tmp_path / sample
    shutil.copy(source, target)

    plain = PluginBackedCCGBuilder()
    flipped = PluginBackedCCGBuilder()
    flipped._native_extractor = extract_plugin_statements

    plain_result = asyncio.run(plain.build_ccg_for_file(target))
    flipped_result = asyncio.run(flipped.build_ccg_for_file(target))
    assert _graph_signature(*flipped_result) == _graph_signature(*plain_result)
    assert flipped_result[0], "the sample must yield statements"


def test_declining_extractor_falls_back_to_python_walk(tmp_path):
    """Extractor present but declining (None) — the Python walk serves."""
    import asyncio

    target = tmp_path / "fallback.py"
    target.write_text("def f(x):\n    return x + 1\n", encoding="utf-8")

    declining = PluginBackedCCGBuilder()
    declining._native_extractor = lambda files, tables, hints_by_file: None
    nodes, edges = asyncio.run(declining.build_ccg_for_file(target))

    plain = PluginBackedCCGBuilder()
    plain_nodes, plain_edges = asyncio.run(plain.build_ccg_for_file(target))
    assert _graph_signature(nodes, edges) == _graph_signature(plain_nodes, plain_edges)


def test_failing_extractor_falls_back_to_python_walk(tmp_path):
    """Extractor raising — bounded failure semantics: walk instead."""
    import asyncio

    target = tmp_path / "boom.py"
    target.write_text("VALUE = 1\n", encoding="utf-8")

    def _boom(files, tables, hints_by_file):
        raise RuntimeError("ffi blew up")

    failing = PluginBackedCCGBuilder()
    failing._native_extractor = _boom
    nodes, edges = asyncio.run(failing.build_ccg_for_file(target))
    assert nodes, "the Python walk must serve when the extractor fails"
