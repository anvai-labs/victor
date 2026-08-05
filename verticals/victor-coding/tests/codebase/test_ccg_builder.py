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
