# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for victor.storage.proxima_runtime (oid correlation + helpers)."""

from __future__ import annotations

from pathlib import Path

from victor.storage.proxima_runtime import (
    ProximaEmbeddingMode,
    graph_id_for_repo,
    node_oid,
    repo_id_from_path,
    symbol_oid,
)


def test_node_oid_canonical_shape():
    sym = symbol_oid("src/auth.py", "login", "function")
    oid = node_oid("myrepo", sym)
    assert oid == f"graph/myrepo/node/{sym}"
    assert oid.startswith("graph/myrepo/node/")


def test_symbol_oid_is_stable_and_keyed_on_type_file_name():
    a = symbol_oid("src/a.py", "foo", "function")
    assert a == symbol_oid("src/a.py", "foo", "function")  # stable
    assert a != symbol_oid("src/a.py", "foo", "method")  # type matters
    assert a != symbol_oid("src/b.py", "foo", "function")  # file matters
    assert a != symbol_oid("src/a.py", "bar", "function")  # name matters


def test_repo_id_sanitized_from_path():
    assert repo_id_from_path(Path("/tmp/My-Repo.git")) == "my_repo_git"
    assert repo_id_from_path(None) == "repo"


def test_graph_id_for_repo():
    assert graph_id_for_repo("victor") == "victor_codegraph"


def test_embedding_mode_coerce():
    assert ProximaEmbeddingMode.coerce(None) is ProximaEmbeddingMode.MEMORY
    assert ProximaEmbeddingMode.coerce("cold") is ProximaEmbeddingMode.COLD
    assert ProximaEmbeddingMode.coerce("MEMORY") is ProximaEmbeddingMode.MEMORY
    assert ProximaEmbeddingMode.coerce(ProximaEmbeddingMode.COLD) is ProximaEmbeddingMode.COLD
    # Unknown values fall back to MEMORY rather than raising.
    assert ProximaEmbeddingMode.coerce("bogus") is ProximaEmbeddingMode.MEMORY


def test_registry_resolves_per_repo_marker(tmp_path):
    from victor.storage.graph.registry import resolve_graph_backend

    assert resolve_graph_backend(tmp_path) == "sqlite"  # default
    marker_dir = tmp_path / ".victor"
    marker_dir.mkdir(exist_ok=True)
    (marker_dir / "graph_backend").write_text("proxima\n", encoding="utf-8")
    assert resolve_graph_backend(tmp_path) == "proxima"
