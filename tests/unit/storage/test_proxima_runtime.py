# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for victor.storage.proxima_runtime (oid correlation + helpers)."""

from __future__ import annotations

from pathlib import Path

import victor.storage.proxima_runtime as proxima_runtime
from victor_codegraph import parse, to_proxima_records

from victor.storage.proxima_runtime import (
    ProximaEmbeddingMode,
    graph_id_for_repo,
    repo_id_from_path,
)


def test_proxima_runtime_does_not_define_symbol_identity():
    """Storage persists CodeGraph identity; it must never derive a competing key."""
    assert not hasattr(proxima_runtime, "symbol_oid")
    assert not hasattr(proxima_runtime, "node_oid")


def test_codegraph_adapter_owns_the_canonical_storage_oid():
    parsed = parse("def login():\n    return True\n", file_path="src/auth.py")
    symbol = parsed.symbols[0]
    records = to_proxima_records(parsed, repo_graph_id="myrepo")
    node_record = next(record for record in records if "code_symbol" in record["labels"])

    assert node_record["oid"] == f"graph/myrepo/node/{symbol.id}"


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
