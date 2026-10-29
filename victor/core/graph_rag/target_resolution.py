"""Deterministic selection of a node at a source location, shared by CLI and tools."""

from collections.abc import Iterable

from victor.storage.graph.protocol import GraphNode


def innermost_node(nodes: Iterable[GraphNode], line: int) -> str | None:
    """Return the unique smallest containing range, or refuse an ambiguous lookup.

    Nodes must already be filtered by the graph store for the requested file;
    the store owns relative/absolute path alias semantics.
    Invalid/missing ranges cannot identify a containing symbol. Without an end
    line, only the known declaration line is eligible. Equal-size best matches
    require an explicit node ID; store insertion order must not break the tie.
    """
    if type(line) is not int or line < 1:
        return None
    matches: dict[str, int] = {}
    for node in nodes:
        start = node.line
        end = node.end_line if node.end_line is not None else start
        if type(start) is int and type(end) is int and 1 <= start <= line <= end:
            matches[node.node_id] = end - start
    if not matches:
        return None
    smallest = min(matches.values())
    winners = [identifier for identifier, size in matches.items() if size == smallest]
    return winners[0] if len(winners) == 1 else None
