"""Pure graph algorithms shared by StateGraph and the vertical DependencyGraph."""

from __future__ import annotations

from victor.core.graph_algorithms import (
    build_adjacency_from_edges,
    detect_cycle,
    find_reachable,
    topological_sort,
)


class TestFindReachable:
    def test_bfs_reaches_transitive_targets_and_skips_sentinel(self):
        adjacency = {"a": ["b", "__end__"], "b": ["c"], "c": ["a"], "x": ["a"]}
        assert find_reachable("a", adjacency, sentinel="__end__") == {"a", "b", "c"}

    def test_isolated_start_reaches_only_itself(self):
        assert find_reachable("solo", {}) == {"solo"}

    def test_start_equal_to_sentinel_reaches_nothing(self):
        assert find_reachable("__end__", {"__end__": ["a"]}, sentinel="__end__") == set()


class TestDetectCycle:
    def test_acyclic_graph_returns_none(self):
        nodes = {"a", "b", "c"}
        assert detect_cycle(nodes, {"a": {"b"}, "b": {"c"}}) is None

    def test_cycle_is_returned_closed(self):
        nodes = {"a", "b", "c"}
        cycle = detect_cycle(nodes, {"a": {"b"}, "b": {"c"}, "c": {"a"}})
        assert cycle is not None
        assert cycle[0] == cycle[-1]
        assert set(cycle) == {"a", "b", "c"}

    def test_self_loop(self):
        cycle = detect_cycle({"a"}, {"a": {"a"}})
        assert cycle is not None and cycle[0] == cycle[-1] == "a" and set(cycle) == {"a"}

    def test_edges_to_unknown_nodes_are_ignored(self):
        assert detect_cycle({"a"}, {"a": {"ghost"}}) is None


class TestTopologicalSort:
    def test_dag_orders_dependencies_first(self):
        nodes = {"a", "b", "c", "d"}
        order, cycle = topological_sort(nodes, {"a": {"b", "c"}, "b": {"d"}, "c": {"d"}})
        assert cycle is None
        assert order.index("a") < order.index("b") < order.index("d")
        assert order.index("a") < order.index("c") < order.index("d")

    def test_cycle_reports_partial_order_and_the_cycle(self):
        nodes = {"root", "x", "y"}
        order, cycle = topological_sort(nodes, {"root": {"x"}, "x": {"y"}, "y": {"x"}})
        assert order == ["root"]
        assert cycle and set(cycle) == {"x", "y"}

    def test_unknown_dependency_targets_are_ignored(self):
        order, cycle = topological_sort({"a"}, {"a": {"outside"}})
        assert (order, cycle) == (["a"], None)


class TestBuildAdjacency:
    def test_string_edges_pass_through_and_others_are_dropped(self):
        assert build_adjacency_from_edges({"a": ["b", 42, "c"]}) == {"a": ["b", "c"]}

    def test_extractor_handles_single_and_conditional_targets(self):
        edges = {
            "a": [{"to": "b"}, {"branches": ["c", "d"]}, {"none": True}],
            "b": [],
        }

        def extract(edge):
            if "to" in edge:
                return edge["to"]
            if "branches" in edge:
                return edge["branches"]
            return None

        assert build_adjacency_from_edges(edges, extract) == {"a": ["b", "c", "d"], "b": []}
