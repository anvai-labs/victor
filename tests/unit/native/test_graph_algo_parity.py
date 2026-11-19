# Copyright 2026 Vijaykumar Singh <vijaykumar@anvaiops.com>
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

"""Native/Python graph-algorithm differential parity (FOU-C3, WS-2).

``tests/unit/native/test_graph_algo.py`` pins the PYTHON implementations'
properties only; nothing compared the Rust accelerators against them. These
tests run identical deterministic graphs through both sides
(``victor_native`` directly vs ``victor.native.python.graph_algo``).

Tolerance policy: PageRank/weighted-PageRank asserts within 1e-6, NEVER
bit-exact — the Rust power iteration sums in a hash-iteration order
(ahash), so the last bits vary per-process and across rebuilds (recorded
finding from the WS-2/detach-audit work; in-process deltas measured 0.0,
which is exactly why a bit-exact assertion would be a flake). Structural
results (components, cycles) compare canonically and exactly; betweenness
(Brandes) uses a tight 1e-9 tolerance.

Selected by the native-parity CI job (``pytest -m native_parity``).
"""

from __future__ import annotations

import pytest

from victor.native.python import graph_algo as py_graph
from victor.processing.native._base import _NATIVE_AVAILABLE, _native

pytestmark = pytest.mark.native_parity

_GRAPH_OPS = (
    "pagerank",
    "weighted_pagerank",
    "betweenness_centrality",
    "connected_components",
    "detect_cycles",
)

_requires_native = pytest.mark.skipif(
    not (_NATIVE_AVAILABLE and all(hasattr(_native, op) for op in _GRAPH_OPS)),
    reason="native wheel missing graph_algo ops — rebuild with maturin",
)

# Deterministic small graphs: cycle, star hub, DAG, disconnected pairs,
# isolated node, and the empty graph.
GRAPHS = [
    {"a": ["b"], "b": ["c"], "c": ["a"]},
    {"hub": [], "a": ["hub"], "b": ["hub"], "c": ["hub"]},
    {"a": ["b", "c"], "b": ["c"], "c": [], "d": []},
    {"a": ["b"], "b": ["c"], "c": [], "x": ["y"], "y": []},
    {"a": ["a"]},
    {"solo": []},
    {},
]

WEIGHTED_GRAPHS = [
    {"a": {"b": 10, "c": 1}, "b": {"c": 2}, "c": {}, "d": {}},
    {"a": {"b": 1}, "b": {"a": 1}, "c": {}},
    {"hub": {}, "a": {"hub": 5}, "b": {"hub": 1}},
    {},
]


def _max_diff(native: dict, fallback: dict) -> float:
    assert set(native) == set(fallback), "engines disagree on the node universe"
    return max((abs(native[k] - fallback[k]) for k in fallback), default=0.0)


class TestPageRankParity:
    @_requires_native
    @pytest.mark.parametrize("adjacency", GRAPHS, ids=lambda g: str(sorted(g))[:40])
    def test_pagerank_within_tolerance(self, adjacency):
        """1e-6 tolerance, deliberately NOT bit-exact (ahash iteration order —
        see module docstring)."""
        assert _max_diff(_native.pagerank(adjacency), py_graph.pagerank(adjacency)) <= 1e-6

    @_requires_native
    def test_ranking_order_agrees(self):
        adjacency = {"hub": [], "a": ["hub"], "b": ["hub"], "c": ["hub"]}
        native = _native.pagerank(adjacency)
        fallback = py_graph.pagerank(adjacency)
        assert max(native, key=native.get) == max(fallback, key=fallback.get) == "hub"

    @_requires_native
    def test_pagerank_mass_conserved_in_both(self):
        """Only for graphs without dangling nodes — this implementation does
        not redistribute dangling rank, so mass leaks on them (both engines
        equally, pinned by the tolerance test above)."""
        adjacency = {"a": ["b"], "b": ["c"], "c": ["a"]}
        for scores in (_native.pagerank(adjacency), py_graph.pagerank(adjacency)):
            assert sum(scores.values()) == pytest.approx(1.0, abs=1e-6)


class TestWeightedPageRankParity:
    @_requires_native
    @pytest.mark.parametrize("adjacency", WEIGHTED_GRAPHS, ids=lambda g: str(sorted(g))[:40])
    def test_weighted_pagerank_within_tolerance(self, adjacency):
        assert (
            _max_diff(_native.weighted_pagerank(adjacency), py_graph.weighted_pagerank(adjacency))
            <= 1e-6
        )

    @_requires_native
    def test_weight_dominance_agrees(self):
        adjacency = {"a": {"b": 10, "c": 1}, "b": {}, "c": {}}
        native = _native.weighted_pagerank(adjacency)
        fallback = py_graph.weighted_pagerank(adjacency)
        assert (native["b"] > native["c"]) == (fallback["b"] > fallback["c"])


class TestBetweennessParity:
    @_requires_native
    @pytest.mark.parametrize("adjacency", GRAPHS, ids=lambda g: str(sorted(g))[:40])
    def test_betweenness_matches(self, adjacency):
        """Brandes on both sides is deterministic — tight tolerance."""
        assert (
            _max_diff(
                _native.betweenness_centrality(adjacency),
                py_graph.betweenness_centrality(adjacency),
            )
            <= 1e-9
        )

    @_requires_native
    def test_chain_bridge_agrees(self):
        adjacency = {"a": ["b"], "b": ["c"], "c": []}
        native = _native.betweenness_centrality(adjacency)
        fallback = py_graph.betweenness_centrality(adjacency)
        assert native["b"] == pytest.approx(fallback["b"], abs=1e-9)
        assert native["b"] >= native["a"]
        assert fallback["b"] >= fallback["a"]


class TestStructuralParity:
    """Components and cycles are discrete — compared canonically, exactly."""

    @_requires_native
    @pytest.mark.parametrize("adjacency", GRAPHS, ids=lambda g: str(sorted(g))[:40])
    def test_connected_components_match(self, adjacency):
        native = sorted(sorted(c) for c in _native.connected_components(adjacency))
        fallback = sorted(sorted(c) for c in py_graph.connected_components(adjacency))
        assert native == fallback

    @_requires_native
    @pytest.mark.parametrize("adjacency", GRAPHS, ids=lambda g: str(sorted(g))[:40])
    def test_detect_cycles_match(self, adjacency):
        """Cycles compare as node-set multisets: rotation and traversal start
        are representation, not semantics."""
        native = sorted(frozenset(c) for c in _native.detect_cycles(adjacency))
        fallback = sorted(frozenset(c) for c in py_graph.detect_cycles(adjacency))
        assert native == fallback
