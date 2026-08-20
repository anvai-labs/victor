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

"""Tests for unified storage factory."""

import pytest
from pathlib import Path
from unittest.mock import patch

from victor.storage.unified.factory import create_symbol_store
from victor.storage.unified.sqlite_lancedb import SqliteLanceDBStore


class TestCreateSymbolStore:
    """Tests for create_symbol_store factory function."""

    def test_default_backend(self, tmp_path):
        """Test creating store with default backend."""
        store = create_symbol_store(repo_root=tmp_path)
        assert isinstance(store, SqliteLanceDBStore)
        assert store.repo_root == tmp_path.resolve()

    def test_sqlite_lancedb_backend(self, tmp_path):
        """Test creating store with explicit sqlite+lancedb backend."""
        store = create_symbol_store(
            repo_root=tmp_path,
            backend="sqlite+lancedb",
        )
        assert isinstance(store, SqliteLanceDBStore)

    def test_default_repo_root_is_cwd(self):
        """Test that default repo_root is current working directory."""
        with patch("victor.storage.unified.factory.Path") as mock_path:
            mock_path.cwd.return_value = Path("/test/dir")
            store = create_symbol_store(repo_root=None)
            assert store.repo_root == Path("/test/dir")

    def test_postgres_pgvector_not_implemented(self, tmp_path):
        """Test that postgres+pgvector raises ValueError (unknown backend)."""
        with pytest.raises((NotImplementedError, ValueError)):
            create_symbol_store(
                repo_root=tmp_path,
                backend="postgres+pgvector",
            )

    def test_lancedb_only_not_implemented(self, tmp_path):
        """Test that lancedb-only raises ValueError (unknown backend)."""
        with pytest.raises((NotImplementedError, ValueError)):
            create_symbol_store(
                repo_root=tmp_path,
                backend="lancedb",
            )

    def test_duckdb_lancedb_not_implemented(self, tmp_path):
        """Test that duckdb+lancedb raises ValueError (unknown backend)."""
        with pytest.raises((NotImplementedError, ValueError)):
            create_symbol_store(
                repo_root=tmp_path,
                backend="duckdb+lancedb",
            )

    def test_unknown_backend_raises_error(self, tmp_path):
        """Test that unknown backend raises ValueError."""
        with pytest.raises(ValueError, match="Unknown backend"):
            create_symbol_store(
                repo_root=tmp_path,
                backend="unknown",
            )

    def test_custom_persist_directory(self, tmp_path):
        """Test creating store with custom persist directory."""
        persist_dir = tmp_path / "custom_persist"
        store = create_symbol_store(
            repo_root=tmp_path,
            persist_directory=persist_dir,
        )
        assert store.persist_directory == persist_dir

    def test_custom_embedding_model(self, tmp_path):
        """Test creating store with custom embedding model."""
        store = create_symbol_store(
            repo_root=tmp_path,
            embedding_model_type="ollama",
            embedding_model_name="qwen3-embedding:8b",
        )
        assert store.embedding_model_type == "ollama"
        assert store.embedding_model_name == "qwen3-embedding:8b"
