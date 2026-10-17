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

"""Tests for codebase/indexer module."""

import pytest

pytest.importorskip("victor_coding.codebase.indexer")

import tempfile
from pathlib import Path

from victor_coding.codebase.indexer import Symbol, FileMetadata, CodebaseIndex


class TestSymbol:
    """Tests for Symbol model."""

    def test_symbol_creation(self):
        """Test creating a Symbol."""
        symbol = Symbol(
            name="my_function",
            type="function",
            file_path="/test/file.py",
            line_number=10,
        )
        assert symbol.name == "my_function"
        assert symbol.type == "function"
        assert symbol.file_path == "/test/file.py"
        assert symbol.line_number == 10

    def test_symbol_with_optional_fields(self):
        """Test Symbol with optional fields."""
        symbol = Symbol(
            name="MyClass",
            type="class",
            file_path="/test/file.py",
            line_number=20,
            docstring="A test class",
            signature="class MyClass(BaseClass)",
        )
        assert symbol.docstring == "A test class"
        assert symbol.signature == "class MyClass(BaseClass)"

    def test_symbol_references(self):
        """Test Symbol references."""
        symbol = Symbol(
            name="helper",
            type="function",
            file_path="/test/utils.py",
            line_number=5,
            references=["/test/main.py", "/test/app.py"],
        )
        assert len(symbol.references) == 2


class TestFileMetadata:
    """Tests for FileMetadata model."""

    def test_file_metadata_creation(self):
        """Test creating FileMetadata."""
        metadata = FileMetadata(
            path="/test/file.py",
            language="python",
            last_modified=1234567890.0,
            size=1024,
            lines=50,
        )
        assert metadata.path == "/test/file.py"
        assert metadata.language == "python"
        assert metadata.lines == 50

    def test_file_metadata_with_symbols(self):
        """Test FileMetadata with symbols."""
        symbol = Symbol(name="func", type="function", file_path="/test.py", line_number=1)
        metadata = FileMetadata(
            path="/test.py",
            language="python",
            symbols=[symbol],
            last_modified=0.0,
            size=100,
            lines=10,
        )
        assert len(metadata.symbols) == 1

    def test_file_metadata_with_imports(self):
        """Test FileMetadata with imports."""
        metadata = FileMetadata(
            path="/test.py",
            language="python",
            imports=["os", "sys", "pathlib"],
            last_modified=0.0,
            size=100,
            lines=10,
        )
        assert "os" in metadata.imports


class TestCodebaseIndex:
    """Tests for CodebaseIndex class."""

    def test_codebase_index_init(self):
        """Test CodebaseIndex initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index = CodebaseIndex(root_path=tmpdir)
            assert index.root == Path(tmpdir).resolve()
            assert len(index.ignore_patterns) > 0

    def test_codebase_index_custom_ignore_patterns(self):
        """Test CodebaseIndex with custom ignore patterns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            patterns = ["*.log", "temp/"]
            index = CodebaseIndex(root_path=tmpdir, ignore_patterns=patterns)
            assert index.ignore_patterns == patterns

    def test_codebase_index_default_ignore(self):
        """Test CodebaseIndex has default ignore patterns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index = CodebaseIndex(root_path=tmpdir)
            # Should have common patterns like venv, .git, etc.
            assert "venv/" in index.ignore_patterns or ".git/" in index.ignore_patterns


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", ["graph_clear", "vector_clear", "embedding_batch", "cancel", None]
)
async def test_full_rebuild_storage_failures_never_mark_index_ready(tmp_path, monkeypatch, failure):
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    (tmp_path / "app.py").write_text("def parse(): return 1\n")
    graph = SimpleNamespace(
        delete_by_repo=AsyncMock(), upsert_nodes=AsyncMock(), upsert_edges=AsyncMock()
    )
    index = CodebaseIndex(
        str(tmp_path),
        use_embeddings=False,
        enable_watcher=False,
        graph_store=graph,
        parallel_workers=1,
    )
    provider = SimpleNamespace(clear_index=AsyncMock(), index_documents=AsyncMock())
    index.use_embeddings = True
    index.embedding_provider = provider
    index._is_indexed = True
    index._is_stale = False
    index._last_indexed = 123

    async def parse(path, language):
        index.files["app.py"] = FileMetadata(
            path="app.py",
            language="python",
            last_modified=0,
            size=10,
            lines=1,
            symbols=[Symbol(name="parse", type="function", file_path="app.py", line_number=1)],
        )

    monkeypatch.setattr(index, "_index_tree_sitter_file", parse)
    monkeypatch.setattr(index, "_get_symbol_embedding_text", lambda symbol: "parse")
    error = asyncio.CancelledError() if failure == "cancel" else OSError("storage failed")
    if failure in {"graph_clear", "cancel"}:
        graph.delete_by_repo.side_effect = error
    elif failure == "vector_clear":
        provider.clear_index.side_effect = error
    elif failure == "embedding_batch":
        provider.index_documents.side_effect = error
    if failure is None:
        await index.index_codebase()
        provider.index_documents.assert_awaited_once()
        assert index._is_indexed is True
        assert index._is_stale is False
    else:
        with pytest.raises(type(error)):
            await index.index_codebase()
        assert index._is_indexed is False
        assert index._is_stale is True
        assert index._last_indexed == 123


@pytest.mark.parametrize("enabled", [True, False])
def test_requested_embeddings_cannot_silently_disable(tmp_path, monkeypatch, enabled):
    from unittest.mock import Mock
    from victor_coding.codebase.embeddings import EmbeddingRegistry

    create = Mock(side_effect=ImportError("missing embedding dependency"))
    monkeypatch.setattr(EmbeddingRegistry, "create", create)
    if enabled:
        with pytest.raises(ImportError, match="embedding"):
            CodebaseIndex(str(tmp_path), use_embeddings=True, enable_watcher=False)
        create.assert_called_once()
    else:
        index = CodebaseIndex(str(tmp_path), use_embeddings=False, enable_watcher=False)
        assert index.use_embeddings is False
        create.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["embedding", "vector_delete", "graph_delete", "cancel", None])
async def test_incremental_storage_failure_retains_incomplete_state(tmp_path, monkeypatch, failure):
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    source = tmp_path / "app.py"
    source.write_text("def parse(): return 1\n")
    graph = SimpleNamespace(delete_by_file=AsyncMock())
    provider = SimpleNamespace(index_document=AsyncMock(), delete_document=AsyncMock())
    index = CodebaseIndex(
        str(tmp_path), use_embeddings=False, enable_watcher=False, graph_store=graph
    )
    symbol = Symbol(name="parse", type="function", file_path="app.py", line_number=1)
    symbol_key = index.make_symbol_id("app.py", "parse")
    index.files["app.py"] = FileMetadata(
        path="app.py", language="python", last_modified=0, size=10, lines=1, symbols=[symbol]
    )
    index.symbols[symbol_key] = symbol
    index.symbol_index["app.py"] = [symbol_key]
    index.embedding_provider = provider
    index.use_embeddings = True
    index._is_indexed = True
    index._is_stale = False
    index._last_indexed = 123
    index._changed_files.add("app.py")
    monkeypatch.setattr(index, "_index_tree_sitter_file", AsyncMock())
    if failure in {"vector_delete", "graph_delete"}:
        source.unlink()
        method = provider.delete_document if failure == "vector_delete" else graph.delete_by_file
        method.side_effect = OSError("storage failed")
    elif failure == "embedding":
        provider.index_document.side_effect = OSError("storage failed")
    elif failure == "cancel":
        provider.index_document.side_effect = asyncio.CancelledError()

    if failure == "cancel":
        with pytest.raises(asyncio.CancelledError):
            await index.incremental_reindex()
    else:
        result = await index.incremental_reindex()
        assert bool(result["errors"]) == (failure is not None)
    if failure is None:
        assert index._is_indexed is True
        assert index._is_stale is False
        provider.index_document.assert_awaited_once()
    else:
        assert index._is_indexed is False
        assert index._is_stale is True
        assert index._last_indexed == 123
        assert "app.py" in index._changed_files
        if failure in {"vector_delete", "graph_delete"}:
            assert "app.py" in index.files
            assert symbol_key in index.symbols
            assert index.symbol_index["app.py"] == [symbol_key]
        # The next update must repair the partial derived state, not skip its mtime.
        monkeypatch.setattr(index, "index_codebase", AsyncMock())
        await index.incremental_reindex()
        index.index_codebase.assert_awaited_once()
