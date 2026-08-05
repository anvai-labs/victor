# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for victor.storage.proxima_runtime (oid correlation + helpers)."""

from __future__ import annotations

from pathlib import Path

import pytest

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

    # The primary oid is the line-independent ADR-044/v2 symbol identity. The
    # pre-v2 line-coupled id remains only as the explicit mixed-read alias.
    assert node_record["oid"] == node_record["props"]["stable_oid"]
    assert node_record["oid"] == f"graph/myrepo/node/{symbol.id}"
    assert node_record["props"]["legacy_oid"] == f"graph/myrepo/node/{symbol.legacy_id}"
    assert node_record["props"]["legacy_oid"] != node_record["oid"]


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


class _FakeUdsDb:
    """Portless embedded instance: UDS sockets, no TCP port."""

    def __init__(self, socket_dir: Path) -> None:
        self.socket_dir = socket_dir
        self.rest_url = "http://localhost"  # host header only — carries no port

    @property
    def rest_socket_path(self) -> Path:
        return self.socket_dir / "proximadb.rest.sock"


class _FakeTcpDb:
    socket_dir = None
    rest_url = "http://localhost:15678"


def test_uds_kwargs_empty_for_a_tcp_instance():
    assert proxima_runtime._uds_kwargs(_FakeTcpDb()) == {}


def test_uds_kwargs_passes_the_socket_for_a_portless_instance(tmp_path):
    pytest.importorskip("proximadb_sdk")
    db = _FakeUdsDb(tmp_path)

    assert proxima_runtime._uds_kwargs(db) == {"uds_path": str(db.rest_socket_path)}


def test_uds_kwargs_degrades_when_the_sdk_lacks_uds_support(tmp_path, monkeypatch, caplog):
    """An older SDK must warn, not raise an unexpected-keyword TypeError."""
    sdk = pytest.importorskip("proximadb_sdk.unified_client")

    class _LegacyClient:
        def __init__(self, url=None, protocol=None):  # no uds_path parameter
            pass

    monkeypatch.setattr(sdk, "ProximaDBClient", _LegacyClient)

    with caplog.at_level("WARNING"):
        assert proxima_runtime._uds_kwargs(_FakeUdsDb(tmp_path)) == {}
    assert "uds_path" in caplog.text


def test_registry_resolves_per_repo_marker(tmp_path):
    from victor.storage.graph.registry import resolve_graph_backend

    assert resolve_graph_backend(tmp_path) == "sqlite"  # default
    marker_dir = tmp_path / ".victor"
    marker_dir.mkdir(exist_ok=True)
    (marker_dir / "graph_backend").write_text("proxima\n", encoding="utf-8")
    assert resolve_graph_backend(tmp_path) == "proxima"
