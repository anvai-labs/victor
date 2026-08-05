# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

import pytest

from victor.storage.graph.registry import create_graph_store
from victor.storage.graph.sqlite_store import SqliteGraphStore
from victor.storage.graph.memory_store import MemoryGraphStore

try:
    from victor.storage.graph.duckdb_store import DuckDBGraphStore
except Exception:  # pragma: no cover - optional backend
    DuckDBGraphStore = None
try:
    import importlib.util

    DUCKDB_AVAILABLE = importlib.util.find_spec("duckdb") is not None
except Exception:  # pragma: no cover - defensive
    DUCKDB_AVAILABLE = False


def test_create_graph_store_sqlite(tmp_path: Path):
    db_path = tmp_path / "graph.db"
    store = create_graph_store("sqlite", db_path)
    assert isinstance(store, SqliteGraphStore)


def test_create_graph_store_invalid(tmp_path: Path):
    with pytest.raises(ValueError):
        create_graph_store("unknown", tmp_path / "db")


def test_create_graph_store_memory():
    store = create_graph_store("memory", Path(":memory:"))
    assert isinstance(store, MemoryGraphStore)


def test_create_graph_store_backend_alias_takes_priority(tmp_path: Path):
    db_path = tmp_path / "graph.db"
    store = create_graph_store("memory", db_path, backend="sqlite")
    assert isinstance(store, SqliteGraphStore)


@pytest.mark.skipif(DuckDBGraphStore is None or not DUCKDB_AVAILABLE, reason="duckdb not installed")
def test_create_graph_store_duckdb(tmp_path: Path):
    db_path = tmp_path / "graph.db"
    store = create_graph_store("duckdb", db_path)
    assert isinstance(store, DuckDBGraphStore)
