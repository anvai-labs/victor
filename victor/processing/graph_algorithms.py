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

"""Graph algorithms for code analysis using NetworkX.

This module provides graph algorithms for analyzing code structure:
- Centrality measures (PageRank, betweenness, etc.)
- Community detection
- Path finding
- Subgraph extraction

Note: NetworkX is an optional dependency. These functions gracefully
degrade if NetworkX is not available.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Try to import NetworkX
try:
    import networkx as nx

    _NETWORKX_AVAILABLE = True
except ImportError:
    _NETWORKX_AVAILABLE = False
    nx = None  # type: ignore


@dataclass
class GraphMetrics:
    """Metrics computed for a code graph.

    Attributes:
        pagerank: PageRank scores for nodes
        betweenness: Betweenness centrality scores
        closeness: Closeness centrality scores
        in_degree: In-degree for each node
        out_degree: Out-degree for each node
        communities: Community assignments (if computed)
    """

    pagerank: Dict[str, float]
    betweenness: Dict[str, float]
    closeness: Dict[str, float]
    in_degree: Dict[str, int]
    out_degree: Dict[str, int]
    communities: Dict[str, int]

    def get_top_nodes(
        self,
        metric: str = "pagerank",
        n: int = 10,
    ) -> List[Tuple[str, float]]:
        """Get top n nodes by metric.

        Args:
            metric: Metric name (pagerank, betweenness, closeness)
            n: Number of nodes to return

        Returns:
            List of (node_id, score) tuples
        """
        scores = getattr(self, metric, {})
        sorted_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_nodes[:n]


def build_networkx_graph(
    nodes: List[Any],
    edges: List[Any],
) -> Optional[Any]:
    """Build a NetworkX DiGraph from nodes and edges.

    Args:
        nodes: List of GraphNode instances
        edges: List of GraphEdge instances

    Returns:
        NetworkX DiGraph or None if NetworkX unavailable
    """
    if not _NETWORKX_AVAILABLE:
        logger.warning("NetworkX not available, skipping graph construction")
        return None

    G = nx.DiGraph()

    # Add nodes
    for node in nodes:
        G.add_node(
            node.node_id,
            **{
                "name": node.name,
                "type": node.type,
                "file": node.file,
                "line": node.line,
            },
        )

    # Add edges
    for edge in edges:
        G.add_edge(
            edge.src,
            edge.dst,
            type=edge.type,
            weight=edge.weight or 1.0,
        )

    return G


def compute_centrality(
    graph: Any,
    weighted: bool = True,
) -> Dict[str, float]:
    """Compute PageRank centrality for nodes.

    Args:
        graph: NetworkX graph
        weighted: Whether to use edge weights

    Returns:
        Dict mapping node_id to centrality score
    """
    if not _NETWORKX_AVAILABLE or graph is None:
        return {}

    try:
        if weighted:
            # Use weight attribute if available
            centrality = nx.pagerank(graph, weight="weight")
        else:
            centrality = nx.pagerank(graph)
        return centrality
    except Exception as e:
        logger.warning(f"Error computing PageRank: {e}")
        return {}


def compute_betweenness(
    graph: Any,
    normalized: bool = True,
) -> Dict[str, float]:
    """Compute betweenness centrality for nodes.

    Args:
        graph: NetworkX graph
        normalized: Whether to normalize scores

    Returns:
        Dict mapping node_id to betweenness score
    """
    if not _NETWORKX_AVAILABLE or graph is None:
        return {}

    try:
        betweenness = nx.betweenness_centrality(graph, normalized=normalized)
        return betweenness
    except Exception as e:
        logger.warning(f"Error computing betweenness: {e}")
        return {}


def compute_closeness(
    graph: Any,
) -> Dict[str, float]:
    """Compute closeness centrality for nodes.

    Args:
        graph: NetworkX graph

    Returns:
        Dict mapping node_id to closeness score
    """
    if not _NETWORKX_AVAILABLE or graph is None:
        return {}

    try:
        closeness = nx.closeness_centrality(graph)
        return closeness
    except Exception as e:
        logger.warning(f"Error computing closeness: {e}")
        return {}


def detect_communities(
    graph: Any,
    resolution: float = 1.0,
) -> Dict[str, int]:
    """Detect communities using Louvain method.

    Args:
        graph: NetworkX graph
        resolution: Resolution parameter for community detection

    Returns:
        Dict mapping node_id to community ID
    """
    if not _NETWORKX_AVAILABLE or graph is None:
        return {}

    try:
        import networkx.algorithms.community as nx_community

        # Convert to undirected for community detection
        undirected = graph.to_undirected()

        # Use label propagation (faster than Louvain)
        communities = nx_community.label_propagation_communities(undirected)

        # Convert to dict mapping node_id -> community_id
        result: Dict[str, int] = {}
        for comm_id, community in enumerate(communities):
            for node in community:
                result[node] = comm_id

        return result

    except Exception as e:
        logger.warning(f"Error detecting communities: {e}")
        return {}


def find_shortest_path(
    graph: Any,
    source: str,
    target: str,
) -> List[str]:
    """Find shortest path between two nodes.

    Args:
        graph: NetworkX graph
        source: Source node ID
        target: Target node ID

    Returns:
        List of node IDs forming the path
    """
    if not _NETWORKX_AVAILABLE or graph is None:
        return []

    try:
        path = nx.shortest_path(graph, source, target)
        return path
    except nx.NetworkXNoPath:
        return []
    except Exception as e:
        logger.warning(f"Error finding path: {e}")
        return []


def extract_subgraph(
    graph: Any,
    center_node: str,
    radius: int = 2,
    max_nodes: int = 100,
) -> Optional[Any]:
    """Extract a subgraph around a center node.

    Args:
        graph: NetworkX graph
        center_node: Center node ID
        radius: Hop radius
        max_nodes: Maximum nodes in subgraph

    Returns:
        NetworkX subgraph or None
    """
    if not _NETWORKX_AVAILABLE or graph is None:
        return None

    try:
        # Get nodes within radius using ego_graph
        subgraph = nx.ego_graph(
            graph,
            center_node,
            radius=radius,
            undirected=False,
        )

        # Limit to max_nodes
        if len(subgraph.nodes) > max_nodes:
            # Keep center node and closest neighbors
            nodes_to_keep = {center_node}
            for neighbor in sorted(
                subgraph.neighbors(center_node),
                key=lambda n: subgraph.degree(n),
                reverse=True,
            )[: max_nodes - 1]:
                nodes_to_keep.add(neighbor)

            subgraph = subgraph.subgraph(nodes_to_keep)

        return subgraph

    except Exception as e:
        logger.warning(f"Error extracting subgraph: {e}")
        return None


def compute_all_metrics(
    nodes: List[Any],
    edges: List[Any],
) -> GraphMetrics:
    """Compute all graph metrics for a code graph.

    Args:
        nodes: List of GraphNode instances
        edges: List of GraphEdge instances

    Returns:
        GraphMetrics with all computed metrics
    """
    # Build graph
    graph = build_networkx_graph(nodes, edges)

    if graph is None:
        return GraphMetrics(
            pagerank={},
            betweenness={},
            closeness={},
            in_degree={n.node_id: 0 for n in nodes},
            out_degree={n.node_id: 0 for n in nodes},
            communities={},
        )

    # Compute metrics
    pagerank = compute_centrality(graph)
    betweenness = compute_betweenness(graph)
    closeness = compute_closeness(graph)
    communities = detect_communities(graph)

    # Degree metrics
    in_degree = {n: graph.in_degree(n) for n in graph.nodes()}
    out_degree = {n: graph.out_degree(n) for n in graph.nodes()}

    return GraphMetrics(
        pagerank=pagerank,
        betweenness=betweenness,
        closeness=closeness,
        in_degree=in_degree,
        out_degree=out_degree,
        communities=communities,
    )


__all__ = [
    "GraphMetrics",
    "build_networkx_graph",
    "compute_centrality",
    "compute_betweenness",
    "compute_closeness",
    "detect_communities",
    "find_shortest_path",
    "extract_subgraph",
    "compute_all_metrics",
]
