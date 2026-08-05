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

"""Warm-up utility for rebuilding tool reputation after fixes.

After significant improvements to a tool (like the graph tool's concurrent
indexing fixes), use this to run safe, known-good queries that will generate
positive outcomes and rebuild the tool's Q-value in the RL system.

The warm-up queries are designed to:
- Be highly likely to succeed (no complex edge cases)
- Cover common usage patterns (callers, callees, neighbors, stats)
- Work on small, stable codebases (victor itself)

Example:
    asyncio.run(warmup_graph_tool())
    # Runs 10-15 safe queries across different modes
    # Records successful outcomes to rebuild Q-value
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from victor.config.settings import load_settings
from victor.core.utils.capability_loader import load_graph_tool_module
from victor.tools.enums import GraphMode


def _load_graph_tool():
    """Load the graph tool from the victor-coding vertical (dynamic discovery).

    Core must not hard-depend on the external victor-coding package. The graph
    tool instance lives in the vertical; only the shared ``GraphMode`` enum was
    promoted into core (``victor.tools.enums``). Discovery routes through
    ``capability_loader`` so core holds no vertical string-literal import.
    """
    return load_graph_tool_module().graph


logger = logging.getLogger(__name__)


async def warmup_graph_tool(
    project_root: Optional[str] = None,
    warmup_queries: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Run warm-up queries to rebuild graph tool's reputation.

    Executes a series of safe, known-good graph queries that are highly
    likely to succeed, generating positive RL outcomes to rebuild the
    tool's Q-value.

    Args:
        project_root: Path to codebase (defaults to victor itself)
        warmup_queries: Custom list of queries to run (uses defaults if None)

    Returns:
        Dictionary with warm-up results including success/failure counts

    Example:
        >>> result = await warmup_graph_tool()
        >>> print(f"Warm-up complete: {result['success']}/{result['total']} successful")
        Warm-up complete: 15/15 successful
    """
    if project_root is None:
        # Use victor's own codebase for warm-up
        project_root = str(Path(__file__).parent.parent.parent)

    # Default warm-up queries designed for high success rate
    if warmup_queries is None:
        warmup_queries = _get_default_warmup_queries()

    results = {
        "total": len(warmup_queries),
        "success": 0,
        "failed": 0,
        "errors": [],
    }

    exec_ctx = {
        "settings": load_settings(),
    }

    # Graph tool instance lives in the victor-coding vertical; load it dynamically.
    graph = _load_graph_tool()

    logger.info(f"[warmup] Starting graph tool warm-up with {len(warmup_queries)} queries")

    for i, query in enumerate(warmup_queries, 1):
        try:
            logger.info(f"[warmup] Query {i}/{len(warmup_queries)}: {query.get('mode', 'unknown')}")

            # Execute graph query
            result = await graph(
                mode=GraphMode(query.get("mode", "stats")),
                path=project_root,
                node=query.get("node"),
                query=query.get("query"),
                depth=query.get("depth", 1),
                top_k=query.get("top_k", 5),
                _exec_ctx=exec_ctx,
            )

            # Check if query succeeded
            if result and isinstance(result, dict) and result.get("success", True):
                results["success"] += 1
                logger.debug(f"[warmup] ✓ Query {i} succeeded")
            else:
                results["failed"] += 1
                logger.warning(f"[warmup] ✗ Query {i} failed")

        except Exception as e:
            results["failed"] += 1
            error_msg = f"Query {i} ({query.get('mode')}): {str(e)}"
            results["errors"].append(error_msg)
            logger.error(f"[warmup] {error_msg}")

    logger.info(
        f"[warmup] Complete: {results['success']}/{results['total']} successful, "
        f"{results['failed']} failed"
    )

    return results


def _get_default_warmup_queries() -> List[Dict[str, Any]]:
    """Get default warm-up queries designed for high success rate.

    These queries are safe because they:
    - Use 'stats' mode (always works if graph is loaded)
    - Query well-known symbols in victor's codebase
    - Use shallow depths (avoid timeouts)
    - Don't require complex filters

    Returns:
        List of query dictionaries
    """
    return [
        # Stats mode (safest - always works if graph loaded)
        {"mode": "stats"},
        # Find mode with well-known symbols
        {"mode": "find", "query": "Agent", "top_k": 5},
        {"mode": "find", "query": "StateGraph", "top_k": 5},
        # Neighbors mode with core symbols
        {"mode": "neighbors", "node": "Agent", "depth": 1},
        {"mode": "neighbors", "node": "graph", "depth": 1},
        # PageRank (core analysis)
        {"mode": "pagerank", "top_k": 5},
        # Centrality
        {"mode": "centrality", "top_k": 5},
        # Stats again (reinforce success)
        {"mode": "stats"},
        # File deps (safe - no specific node needed)
        {"mode": "file_deps"},
        # Module-level analysis
        {"mode": "module_pagerank", "top_k": 5},
        # Call flow with simple node
        {"mode": "call_flow", "node": "run", "depth": 1},
        # Final stats
        {"mode": "stats"},
    ]


async def warmup_tool_generic(
    tool_name: str,
    tool_func: callable,
    queries: List[Dict[str, Any]],
    exec_ctx: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Generic warm-up for any tool.

    Args:
        tool_name: Name of the tool (for logging)
        tool_func: Async function to execute
        queries: List of query dictionaries to pass to tool_func
        exec_ctx: Execution context

    Returns:
        Results dictionary with success/failure counts
    """
    if exec_ctx is None:
        exec_ctx = {"settings": load_settings()}

    results = {
        "tool_name": tool_name,
        "total": len(queries),
        "success": 0,
        "failed": 0,
        "errors": [],
    }

    logger.info(f"[warmup] Starting {tool_name} warm-up with {len(queries)} queries")

    for i, query in enumerate(queries, 1):
        try:
            result = await tool_func(**query, _exec_ctx=exec_ctx)

            if result and isinstance(result, dict) and result.get("success", True):
                results["success"] += 1
            else:
                results["failed"] += 1

        except Exception as e:
            results["failed"] += 1
            results["errors"].append(f"Query {i}: {str(e)}")

    logger.info(
        f"[warmup] {tool_name} complete: {results['success']}/{results['total']} successful"
    )

    return results


# CLI command
async def main_async():
    """CLI entry point for warm-up."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Warm up graph tool to rebuild reputation after fixes"
    )
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="Path to codebase (defaults to victor itself)",
    )
    parser.add_argument(
        "--tool",
        type=str,
        default="graph",
        choices=["graph", "code_search"],
        help="Tool to warm up",
    )

    args = parser.parse_args()

    if args.tool == "graph":
        result = await warmup_graph_tool(project_root=args.path)
        print("\nGraph tool warm-up results:")
        print(f"  Successful: {result['success']}/{result['total']}")
        print(f"  Failed: {result['failed']}/{result['total']}")
        if result["errors"]:
            print("\nErrors:")
            for error in result["errors"]:
                print(f"  - {error}")
    else:
        print(f"Warm-up for {args.tool} not yet implemented")


def main():
    """Sync CLI entry point."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
