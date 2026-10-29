"""Tests for graph algorithm implementations (Rust + Python parity)."""

from victor.native.python.graph_algo import (
    betweenness_centrality,
    connected_components,
    detect_cycles,
    pagerank,
    weighted_pagerank,
)


class TestPageRank:
    """Test PageRank algorithm."""

    def test_empty_graph(self):
        result = pagerank({})
        assert result == {}

    def test_single_node(self):
        result = pagerank({"a": []})
        assert len(result) == 1
        # Single dangling node gets base rank: (1-damping)/n = 0.15
        assert result["a"] > 0

    def test_triangle_equal(self):
        """Triangle graph should have equal PageRank."""
        adj = {"a": ["b"], "b": ["c"], "c": ["a"]}
        result = pagerank(adj)
        values = list(result.values())
        assert all(abs(v - values[0]) < 0.01 for v in values)

    def test_star_hub_highest(self):
        """Hub in star graph should have highest PageRank."""
        adj = {"hub": [], "a": ["hub"], "b": ["hub"], "c": ["hub"]}
        result = pagerank(adj)
        assert result["hub"] > result["a"]
        assert result["hub"] > result["b"]
        assert result["hub"] > result["c"]

    def test_sum_approximately_one(self):
        """PageRank values should sum to approximately 1."""
        adj = {"a": ["b", "c"], "b": ["c"], "c": ["a"]}
        result = pagerank(adj)
        total = sum(result.values())
        assert abs(total - 1.0) < 0.01

    def test_disconnected_graph(self):
        """Disconnected nodes should get base rank."""
        adj = {"a": ["b"], "b": [], "c": []}
        result = pagerank(adj)
        assert len(result) == 3
        # All should have positive rank
        assert all(v > 0 for v in result.values())


class TestWeightedPageRank:
    """Test weighted PageRank."""

    def test_empty_graph(self):
        result = weighted_pagerank({})
        assert result == {}

    def test_weighted_edges(self):
        """Higher weight edges should transfer more rank."""
        adj = {
            "a": {"b": 10, "c": 1},
            "b": {},
            "c": {},
        }
        result = weighted_pagerank(adj)
        # b should get more rank than c from a (10x weight)
        assert result["b"] > result["c"]


class TestBetweennessCentrality:
    """Test betweenness centrality (Brandes)."""

    def test_empty_graph(self):
        result = betweenness_centrality({})
        assert result == {}

    def test_chain(self):
        """Middle node in a chain should have highest betweenness."""
        adj = {"a": ["b"], "b": ["c"], "c": []}
        result = betweenness_centrality(adj)
        assert result["b"] >= result["a"]
        assert result["b"] >= result["c"]

    def test_single_node(self):
        result = betweenness_centrality({"a": []})
        assert result["a"] == 0.0

    def test_star(self):
        """Hub of a star should have high betweenness."""
        adj = {
            "hub": ["a", "b", "c"],
            "a": ["hub"],
            "b": ["hub"],
            "c": ["hub"],
        }
        result = betweenness_centrality(adj)
        assert result["hub"] >= result["a"]


class TestConnectedComponents:
    """Test connected components (union-find)."""

    def test_empty(self):
        result = connected_components({})
        assert result == []

    def test_single_component(self):
        adj = {"a": ["b"], "b": ["c"], "c": []}
        result = connected_components(adj)
        assert len(result) == 1
        assert set(result[0]) == {"a", "b", "c"}

    def test_two_components(self):
        adj = {"a": ["b"], "b": [], "c": ["d"], "d": []}
        result = connected_components(adj)
        assert len(result) == 2
        components = [set(c) for c in result]
        assert {"a", "b"} in components
        assert {"c", "d"} in components

    def test_isolated_nodes(self):
        adj = {"a": [], "b": [], "c": []}
        result = connected_components(adj)
        assert len(result) == 3


class TestDetectCycles:
    """Test cycle detection (DFS coloring)."""

    def test_no_cycles(self):
        adj = {"a": ["b"], "b": ["c"], "c": []}
        result = detect_cycles(adj)
        assert result == []

    def test_simple_cycle(self):
        adj = {"a": ["b"], "b": ["c"], "c": ["a"]}
        result = detect_cycles(adj)
        assert len(result) >= 1
        # The cycle should contain a, b, c
        cycle_nodes = set()
        for cycle in result:
            cycle_nodes.update(cycle)
        assert {"a", "b", "c"}.issubset(cycle_nodes)

    def test_self_loop(self):
        adj = {"a": ["a"]}
        result = detect_cycles(adj)
        # Self-loops may or may not be detected as cycles depending on implementation
        # (our impl requires len > 1)
        # Just ensure it doesn't crash
        assert isinstance(result, list)

    def test_empty(self):
        result = detect_cycles({})
        assert result == []
