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

"""Shared graph algorithm utilities.

Provides reusable graph traversal algorithms used by both StateGraph
(victor.framework.graph) and DependencyGraph (victor.core.verticals.dependency_graph).

This module eliminates duplication of reachability analysis, cycle detection,
and topological sorting across the codebase.

Design Principles:
    - Pure functions (no side effects, no mutations)
    - Work with adjacency-list representations (Dict[str, Set[str]])
    - Protocol-agnostic: usable by any graph-like structure
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple


def find_reachable(
    start: str,
    adjacency: Dict[str, List[str]],
    sentinel: Optional[str] = None,
) -> Set[str]:
    """Find all nodes reachable from a start node via BFS.

    Args:
        start: Starting node ID.
        adjacency: Mapping of source node → list of target node IDs.
        sentinel: Optional sentinel value to exclude (e.g., ``"__end__"``).

    Returns:
        Set of reachable node IDs (excluding *start* duplicates and *sentinel*).
    """
    reachable: Set[str] = set()
    queue = deque([start])

    while queue:
        node_id = queue.popleft()
        if node_id in reachable or node_id == sentinel:
            continue
        reachable.add(node_id)
        for target in adjacency.get(node_id, []):
            if target not in reachable:
                queue.append(target)

    return reachable


def detect_cycle(
    nodes: Set[str],
    adjacency: Dict[str, Set[str]],
) -> Optional[List[str]]:
    """Detect a cycle in a directed graph using DFS.

    Args:
        nodes: Set of all node IDs.
        adjacency: Mapping of node → set of neighbours.

    Returns:
        A list of node IDs forming the cycle (with the first node repeated at
        the end), or ``None`` if the graph is acyclic.
    """
    WHITE, GRAY, BLACK = 0, 1, 2
    color: Dict[str, int] = dict.fromkeys(nodes, WHITE)
    parent: Dict[str, Optional[str]] = dict.fromkeys(nodes, None)

    def _dfs(node: str) -> Optional[List[str]]:
        color[node] = GRAY
        for neighbour in adjacency.get(node, set()):
            if neighbour not in color:
                continue
            if color[neighbour] == GRAY:
                # Reconstruct cycle
                cycle = [neighbour, node]
                current = node
                while parent[current] is not None and parent[current] != neighbour:
                    current = parent[current]  # type: ignore[assignment]
                    cycle.append(current)
                cycle.append(neighbour)
                cycle.reverse()
                return cycle
            if color[neighbour] == WHITE:
                parent[neighbour] = node
                result = _dfs(neighbour)
                if result is not None:
                    return result
        color[node] = BLACK
        return None

    for node in nodes:
        if color.get(node, WHITE) == WHITE:
            result = _dfs(node)
            if result is not None:
                return result
    return None


def topological_sort(
    nodes: Set[str],
    adjacency: Dict[str, Set[str]],
) -> Tuple[List[str], Optional[List[str]]]:
    """Kahn's algorithm for topological sorting.

    Args:
        nodes: Set of all node IDs.
        adjacency: Mapping of node → set of dependents (outgoing edges).

    Returns:
        Tuple of (sorted_order, cycle).
        *sorted_order* is the topological ordering if acyclic.
        *cycle* is ``None`` when the graph is a DAG, otherwise contains a
        detected cycle.
    """
    in_degree: Dict[str, int] = dict.fromkeys(nodes, 0)
    for node in nodes:
        for dep in adjacency.get(node, set()):
            if dep in in_degree:
                in_degree[dep] += 1

    queue = deque(n for n in nodes if in_degree[n] == 0)
    order: List[str] = []

    while queue:
        node = queue.popleft()
        order.append(node)
        for dep in adjacency.get(node, set()):
            if dep in in_degree:
                in_degree[dep] -= 1
                if in_degree[dep] == 0:
                    queue.append(dep)

    if len(order) != len(nodes):
        cycle = detect_cycle(nodes, adjacency)
        return order, cycle or []

    return order, None


def build_adjacency_from_edges(
    edges: Dict[str, List[Any]],
    target_extractor: Any = None,
) -> Dict[str, List[str]]:
    """Build a flat adjacency list from edge data.

    Handles both simple (str target) and conditional (dict target) edges.

    Args:
        edges: Mapping of source → list of edge objects.
        target_extractor: Optional callable(edge) → str|list[str] to extract
            targets. If None, treats edges as already being target strings.

    Returns:
        Mapping of source → list of target node IDs.
    """
    adjacency: Dict[str, List[str]] = {}

    for source, edge_list in edges.items():
        targets: List[str] = []
        for edge in edge_list:
            if target_extractor:
                result = target_extractor(edge)
                if isinstance(result, list):
                    targets.extend(result)
                elif isinstance(result, str):
                    targets.append(result)
            elif isinstance(edge, str):
                targets.append(edge)
        adjacency[source] = targets

    return adjacency
