"""Pure Python fallback implementations of graph algorithms.

These mirror the Rust implementations in rust/src/graph_algo.rs.
"""

from __future__ import annotations

from collections import deque
from victor.core.graph_rag.semantic_clones import make_union_find


def pagerank(
    adjacency: dict[str, list[str]],
    damping: float = 0.85,
    iterations: int = 100,
    tolerance: float = 1e-6,
) -> dict[str, float]:
    """PageRank via power iteration."""
    all_nodes = list(adjacency.keys())
    n = len(all_nodes)
    if n == 0:
        return {}

    inv_n = 1.0 / n
    scores = dict.fromkeys(all_nodes, inv_n)

    for _ in range(iterations):
        new_scores = dict.fromkeys(all_nodes, (1.0 - damping) * inv_n)

        for src, neighbors in adjacency.items():
            if not neighbors:
                continue
            share = damping * scores.get(src, 0.0) / len(neighbors)
            for dst in neighbors:
                if dst in new_scores:
                    new_scores[dst] += share

        max_diff = max(abs(scores.get(n, 0) - new_scores.get(n, 0)) for n in all_nodes)
        scores = new_scores
        if max_diff < tolerance:
            break

    return scores


def weighted_pagerank(
    adjacency: dict[str, dict[str, int]],
    damping: float = 0.85,
    iterations: int = 100,
) -> dict[str, float]:
    """Weighted PageRank where adjacency values are dicts of {neighbor: weight}."""
    all_nodes = list(adjacency.keys())
    n = len(all_nodes)
    if n == 0:
        return {}

    inv_n = 1.0 / n
    scores = dict.fromkeys(all_nodes, inv_n)

    for _ in range(iterations):
        new_scores = dict.fromkeys(all_nodes, (1.0 - damping) * inv_n)

        for src, edges in adjacency.items():
            total_weight = sum(edges.values())
            if total_weight <= 0:
                continue
            src_score = scores.get(src, 0.0)
            for dst, weight in edges.items():
                if dst in new_scores:
                    new_scores[dst] += damping * src_score * weight / total_weight

        scores = new_scores

    return scores


def betweenness_centrality(
    adjacency: dict[str, list[str]],
    normalized: bool = True,
) -> dict[str, float]:
    """Betweenness centrality using Brandes algorithm."""
    all_nodes = list(adjacency.keys())
    n = len(all_nodes)
    cb = dict.fromkeys(all_nodes, 0.0)
    node_idx = {node: i for i, node in enumerate(all_nodes)}

    for s in all_nodes:
        stack: list[int] = []
        pred: list[list[int]] = [[] for _ in range(n)]
        sigma = [0] * n
        dist = [-1] * n
        s_idx = node_idx[s]
        sigma[s_idx] = 1
        dist[s_idx] = 0
        queue: deque[int] = deque([s_idx])

        while queue:
            v = queue.popleft()
            stack.append(v)
            for w_name in adjacency.get(all_nodes[v], []):
                w = node_idx.get(w_name)
                if w is None:
                    continue
                if dist[w] < 0:
                    queue.append(w)
                    dist[w] = dist[v] + 1
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    pred[w].append(v)

        delta = [0.0] * n
        while stack:
            w = stack.pop()
            for v in pred[w]:
                if sigma[w] > 0:
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
            if w != s_idx:
                cb[all_nodes[w]] += delta[w]

    if normalized and n > 2:
        norm = 1.0 / ((n - 1) * (n - 2))
        cb = {k: v * norm for k, v in cb.items()}

    return cb


def connected_components(adjacency: dict[str, list[str]]) -> list[list[str]]:
    """Find connected components using union-find."""
    all_nodes = set(adjacency.keys())
    for neighbors in adjacency.values():
        all_nodes.update(neighbors)
    all_nodes_list = sorted(all_nodes)
    node_idx = {n: i for i, n in enumerate(all_nodes_list)}
    n = len(all_nodes_list)

    find, union = make_union_find(n)

    for src, neighbors in adjacency.items():
        si = node_idx[src]
        for dst in neighbors:
            if dst in node_idx:
                union(si, node_idx[dst])

    components: dict[int, list[str]] = {}
    for i, node in enumerate(all_nodes_list):
        root = find(i)
        components.setdefault(root, []).append(node)

    return list(components.values())


def detect_cycles(adjacency: dict[str, list[str]]) -> list[list[str]]:
    """Detect cycles using DFS coloring."""
    all_nodes = list(adjacency.keys())
    node_idx = {n: i for i, n in enumerate(all_nodes)}
    n = len(all_nodes)
    color = [0] * n  # 0=white, 1=gray, 2=black
    cycles: list[list[str]] = []
    path: list[int] = []

    def dfs(v: int) -> None:
        color[v] = 1
        path.append(v)

        for w_name in adjacency.get(all_nodes[v], []):
            w = node_idx.get(w_name)
            if w is None:
                continue
            if color[w] == 1:
                pos = next((i for i, x in enumerate(path) if x == w), None)
                if pos is not None:
                    cycle = [all_nodes[i] for i in path[pos:]]
                    if len(cycle) > 1:
                        cycles.append(cycle)
            elif color[w] == 0:
                dfs(w)

        path.pop()
        color[v] = 2

    for i in range(n):
        if color[i] == 0:
            dfs(i)

    return cycles
