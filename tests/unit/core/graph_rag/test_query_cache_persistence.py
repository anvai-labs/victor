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

"""Persistent (project.db-backed) graph query cache.

Pins the PH4-005 completion: cache entries survive process restarts (each CLI
invocation is a fresh process), invalidation clears the persisted layer so a
re-index in one process invalidates entries a future process would load, and
persistence failures degrade to memory-only.
"""

from dataclasses import dataclass

import pytest

from victor.core.graph_rag.query_cache import GraphQueryCache, GraphQueryCacheConfig
from victor.core.graph_rag.retrieval import RetrievalResult
from victor.storage.graph.protocol import GraphNode


@dataclass
class MockConfig:
    seed_count: int = 5
    max_hops: int = 2
    top_k: int = 10
    edge_types: list | None = None


def _result(query: str) -> RetrievalResult:
    return RetrievalResult(
        nodes=[GraphNode(node_id="n1", type="function", name="auth", file="auth.py", line=10)],
        edges=[],
        subgraphs=[],
        query=query,
        seed_nodes=["n1"],
        scores={"n1": 1.0},
    )


@pytest.fixture
def repo(tmp_path):
    return str(tmp_path)


def test_cache_survives_process_restart(repo):
    """A fresh cache instance (new process) hits entries persisted by an old one."""
    config = MockConfig()
    cache1 = GraphQueryCache()
    cache1.put("find auth", config, _result("find auth"), repo)

    cache2 = GraphQueryCache()  # simulates a new CLI process
    cached = cache2.get("find auth", config, repo)

    assert cached is not None
    assert cached.query == "find auth"
    assert cached.nodes[0].name == "auth"
    assert cache2.get_stats()["persistent_hits"] == 1


def test_persistent_hit_promotes_to_l1(repo):
    config = MockConfig()
    GraphQueryCache().put("find auth", config, _result("find auth"), repo)

    cache = GraphQueryCache()
    assert cache.get("find auth", config, repo) is not None  # L2 hit + promote
    assert cache.get("find auth", config, repo) is not None  # L1 hit
    stats = cache.get_stats()
    assert stats["hits"] == 2
    assert stats["persistent_hits"] == 1  # only the first went to sqlite


def test_invalidate_repo_clears_persisted_entries(repo):
    config = MockConfig()
    cache1 = GraphQueryCache()
    cache1.put("find auth", config, _result("find auth"), repo)

    assert cache1.invalidate_repo(repo) >= 1

    cache2 = GraphQueryCache()
    assert cache2.get("find auth", config, repo) is None


def test_cross_process_invalidation(repo):
    """Invalidation in one process must drop rows a future process would read."""
    config = MockConfig()
    GraphQueryCache().put("find auth", config, _result("find auth"), repo)

    # A different process (e.g. `victor graph index`) invalidates after a refresh.
    GraphQueryCache().invalidate_repo(repo)

    assert GraphQueryCache().get("find auth", config, repo) is None


def test_invalidate_all_clears_opened_backends(repo):
    config = MockConfig()
    cache = GraphQueryCache()
    cache.put("find auth", config, _result("find auth"), repo)

    cache.invalidate_all()

    assert GraphQueryCache().get("find auth", config, repo) is None


def test_ttl_expiry_applies_to_persisted_entries(repo):
    config = MockConfig()
    GraphQueryCache().put("find auth", config, _result("find auth"), repo)

    expired = GraphQueryCache(GraphQueryCacheConfig(ttl_seconds=0))
    assert expired.get("find auth", config, repo) is None


def test_persist_disabled_writes_nothing(repo):
    config = MockConfig()
    off = GraphQueryCache(GraphQueryCacheConfig(persist=False))
    off.put("find auth", config, _result("find auth"), repo)

    # Same-instance hit still works (L1)...
    assert off.get("find auth", config, repo) is not None
    # ...but nothing was persisted for other processes.
    assert GraphQueryCache().get("find auth", config, repo) is None


def test_unwritable_repo_degrades_to_memory_only(repo):
    """A repo path we can't open a DB under degrades silently (no raise)."""
    config = MockConfig()
    cache = GraphQueryCache()
    bogus = "/proc/definitely-not-writable/repo"
    cache.put("find auth", config, _result("find auth"), bogus)
    assert cache.get("find auth", config, bogus) is not None  # L1 still works


def test_entries_scoped_per_repo(tmp_path):
    config = MockConfig()
    repo_a = str(tmp_path / "a")
    repo_b = str(tmp_path / "b")
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()

    GraphQueryCache().put("find auth", config, _result("find auth"), repo_a)

    fresh = GraphQueryCache()
    assert fresh.get("find auth", config, repo_a) is not None
    assert fresh.get("find auth", config, repo_b) is None
