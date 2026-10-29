"""Deterministic selection of a node at a source location, shared by CLI and tools."""

from collections.abc import Iterable

from victor.storage.graph.protocol import GraphNode, GraphStoreProtocol


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


async def resolve_explicit_target(
    target: str, graph_store: GraphStoreProtocol
) -> tuple[bool, str | None]:
    """Resolve IDs/locations once; leave each caller's named-target policy intact.

    The boolean distinguishes a handled but unavailable/ambiguous location from
    a name. A failed location lookup must not fall through to unrelated names.
    """
    exact = await graph_store.get_node_by_id(target)
    if exact is not None:
        return True, exact.node_id
    if ":" in target:
        file_path, line_str = target.rsplit(":", 1)
        try:
            line = int(line_str)
        except ValueError:
            pass  # A qualified symbol name may itself contain a colon.
        else:
            nodes = await graph_store.get_nodes_by_file(file_path)
            return True, innermost_node(nodes, line)
    return False, None
