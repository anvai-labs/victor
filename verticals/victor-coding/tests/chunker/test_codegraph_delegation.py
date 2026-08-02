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

"""Tests for the victor_codegraph delegation shims (TD-CG / ADR-014).

Verifies the ASTAwareChunker and TierAwareChunker tree-sitter paths delegate to
the shared ``victor_codegraph`` package when it is installed. The whole module is
skipped when ``victor_codegraph`` is absent — in that case the legacy in-repo
tree-sitter path is exercised by the other chunker tests instead.
"""

import tempfile
from pathlib import Path

import pytest

pytest.importorskip("victor_codegraph")
pytest.importorskip("victor_coding.codebase.embeddings.chunker")
pytest.importorskip("victor_coding.codebase.chunker")

from victor_coding.codebase.chunker import ChunkType, TierAwareChunker
from victor_coding.codebase.embeddings.chunker import ASTAwareChunker

JS_CODE = """function hello() {
    return 1;
}

class Greeter {
    greet() {
        return "hi";
    }
}
"""

GO_CODE = """package main

import "fmt"

func add(a int, b int) int {
    return a + b
}

type Calculator struct {
    total int
}

func (c *Calculator) Accumulate(v int) {
    c.total += v
}
"""


class TestASTAwareChunkerDelegation:
    """ASTAwareChunker.chunk_file delegates to victor_codegraph when installed."""

    def test_delegates_and_adapts_shape(self):
        # Explicit language isolates the delegation from the LanguageRegistry's
        # plugin discovery (the registry returns None for .js until discover_plugins
        # is called); the seam under test is the chunk() adaptation, not detection.
        chunker = ASTAwareChunker()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".js", delete=False) as f:
            f.write(JS_CODE)
            f.flush()
            chunks = chunker.chunk_file(Path(f.name), language="javascript")

        assert len(chunks) >= 1
        names = {c.symbol_name for c in chunks if c.symbol_name}
        assert {"hello", "Greeter", "greet"} <= names
        # Every adapted chunk carries the embeddings CodeChunk contract.
        for c in chunks:
            assert c.file_path is not None
            assert isinstance(c.chunk_type, str)  # lowercase kind, never the enum
            assert c.start_line >= 1
            assert c.end_line >= c.start_line

    def test_symbol_chunks_present_for_go(self):
        chunker = ASTAwareChunker()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".go", delete=False) as f:
            f.write(GO_CODE)
            f.flush()
            chunks = chunker.chunk_file(Path(f.name), language="go")

        names = {c.symbol_name for c in chunks if c.symbol_name}
        assert "add" in names  # victor_codegraph extracts the function symbol

    def test_legacy_fallback_when_victor_codegraph_absent(self, monkeypatch):
        # Simulate the package being unavailable: chunk_file must still return
        # chunks via the legacy line-fallback path (does not raise).
        from victor_coding.codebase.embeddings import chunker as mod

        monkeypatch.setattr(mod, "_victor_codegraph", None)
        chunker = ASTAwareChunker()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("def hello():\n    pass\n")
            f.flush()
            chunks = chunker.chunk_file(Path(f.name))

        assert len(chunks) >= 1


class TestTierAwareChunkerDelegation:
    """TierAwareChunker's tree-sitter fallback delegates to victor_codegraph."""

    def test_tree_sitter_path_uses_victor_codegraph_without_extractor(self):
        # No injected TreeSitterExtractor — delegation must still yield symbols.
        chunker = TierAwareChunker()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "calc.go"
            path.write_text(GO_CODE)
            chunks = chunker.chunk_file(path, "calc.go")

        assert len(chunks) >= 1
        assert any(c.metadata.get("source") == "victor_codegraph" for c in chunks)
        # Real symbol chunks, not only sliding-window METHOD_BODY windows.
        kinds = {c.chunk_type for c in chunks}
        assert kinds & {
            ChunkType.FILE_SUMMARY,
            ChunkType.CLASS_SUMMARY,
            ChunkType.METHOD_HEADER,
        }

    def test_python_stays_on_ast_path_not_delegated(self):
        # Scope decision: the ast/Python path (CodeChunker taxonomy) is NOT
        # delegated; Python must not be tagged source=victor_codegraph.
        chunker = TierAwareChunker()
        py = 'def foo():\n    """doc"""\n    return 1\n'
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "m.py"
            path.write_text(py)
            chunks = chunker.chunk_file(path, "m.py")

        assert len(chunks) >= 1
        assert all(c.metadata.get("source") != "victor_codegraph" for c in chunks)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
