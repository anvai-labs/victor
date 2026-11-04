"""Graph dependence must exclude joins and must not invent exits or partial answers."""

import pytest

from victor.core.graph_rag.control_dependence import DependenceLimits, control_dependence


def test_diamond_excludes_join_and_continuation():
    graph = {
        "branch": ("left", "right"),
        "left": ("join",),
        "right": ("join",),
        "join": ("after",),
        "after": (),
    }
    result = control_dependence(graph)
    assert result.complete
    assert result.dependencies == (("branch", "left"), ("branch", "right"))
    assert result.exit_basis == "inferred_sinks"


def test_loop_can_depend_on_its_own_condition():
    result = control_dependence({"loop": ("body", "exit"), "body": ("loop",), "exit": ()})
    assert result.dependencies == (("loop", "body"), ("loop", "loop"))


def test_multiple_exits_and_early_returns():
    result = control_dependence(
        {"branch": ("return", "work"), "work": ("exit",), "return": (), "exit": ()},
        exits={"return", "exit"},
    )
    assert result.complete
    assert set(result.dependencies) == {
        ("branch", "return"),
        ("branch", "work"),
        ("branch", "exit"),
    }
    assert result.exit_basis == "declared_exits"


@pytest.mark.parametrize(
    "graph",
    [
        {"loop": ("loop",)},
        {"branch": ("loop", "exit"), "loop": ("loop",), "exit": ()},
    ],
)
def test_non_exit_reaching_regions_abstain_as_a_whole(graph):
    result = control_dependence(graph)
    assert not result.complete
    assert not result.dependencies
    assert result.reason == "non_exit_reaching_nodes"


@pytest.mark.parametrize(
    "graph,exits",
    [
        ({"a": ("missing",)}, None),
        ({"a": ("b",), "b": ()}, {"a"}),
        ({"a": ()}, {"missing"}),
        ({"": ()}, None),
        ({"a": "a"}, None),
    ],
)
def test_invalid_explicit_graph_is_rejected(graph, exits):
    with pytest.raises(ValueError):
        control_dependence(graph, exits=exits)


@pytest.mark.parametrize(
    "limits",
    [
        DependenceLimits(max_nodes=1),
        DependenceLimits(max_edges=1),
        DependenceLimits(max_work=1),
        DependenceLimits(max_dependencies=1),
    ],
)
def test_budget_exhaustion_never_returns_partial_dependencies(limits):
    graph = {"branch": ("left", "right"), "left": ("exit",), "right": ("exit",), "exit": ()}
    result = control_dependence(graph, limits=limits)
    assert not result.complete
    assert result.reason == "budget_exceeded"
    assert not result.dependencies


def test_order_and_duplicate_edges_do_not_change_dependence():
    first = control_dependence({"a": ("b", "c"), "b": (), "c": ()})
    second = control_dependence({"c": (), "b": (), "a": ("c", "b", "b")})
    assert first.dependencies == second.dependencies


def test_empty_graph_is_complete_without_inventing_relationships():
    assert control_dependence({}).complete
    assert not control_dependence({}).dependencies


def test_exhaustive_small_graphs_against_vertex_deletion_oracle():
    # Independent characterization: Y postdominates X iff deleting Y removes
    # all X-to-exit paths. This oracle does not use iterative dataflow/bitsets.
    for terminal in ({3}, {2, 3}, {1, 2, 3}):
        active = [node for node in range(4) if node not in terminal]
        for mask in range(1 << (4 * len(active))):
            graph = {node: set() for node in range(4)}
            for index, node in enumerate(active):
                graph[node] = {child for child in range(4) if mask & (1 << (4 * index + child))}

            def reaches_exit(source, removed):
                pending = [source]
                seen = {removed}
                while pending:
                    node = pending.pop()
                    if node in seen:
                        continue
                    if node in terminal:
                        return True
                    seen.add(node)
                    pending.extend(graph[node])
                return False

            result = control_dependence(
                {
                    str(node): tuple(str(child) for child in children)
                    for node, children in graph.items()
                },
                exits={str(node) for node in terminal},
            )
            eligible = all(reaches_exit(node, -1) for node in graph)
            assert result.complete == eligible, (terminal, mask)
            if not eligible:
                assert not result.dependencies
                continue
            postdom = {
                node: {candidate for candidate in graph if not reaches_exit(node, candidate)}
                for node in graph
            }
            expected = {
                (str(node), str(candidate))
                for node in graph
                for candidate in graph
                if (candidate == node or candidate not in postdom[node])
                and any(candidate in postdom[child] for child in graph[node])
            }
            assert set(result.dependencies) == expected, (terminal, mask)
