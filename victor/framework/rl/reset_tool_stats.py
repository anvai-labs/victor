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

"""Reset RL statistics for specific tools to give them a fresh start.

Use this when significant improvements have been made to a tool's reliability
or performance, and you want the RL system to relearn from scratch rather than
being biased by historical poor performance.

Example:
    # After fixing graph tool performance issues
    reset_tool_stats("graph")
    reset_tool_stats("code_search")
"""

from __future__ import annotations

import logging
from typing import List, Optional

from victor.core.database import get_database

logger = logging.getLogger(__name__)


def reset_tool_stats(tool_name: str, confirm: bool = False) -> bool:
    """Reset all RL statistics for a specific tool.

    This clears:
    - rl_tool_outcome: Historical success/failure records
    - rl_tool_q: Q-value and selection counts
    - rl_cache_tool: Estimated value and hit/miss ratios
    - rl_tool_task: Task-specific performance data

    WARNING: This action cannot be undone. The tool will start with a clean
    slate and must rebuild its reputation through successful executions.

    Args:
        tool_name: Name of the tool to reset (e.g., "graph", "code_search")
        confirm: Must be True to execute the reset (safety check)

    Returns:
        True if reset was successful, False otherwise

    Example:
        >>> reset_tool_stats("graph", confirm=True)
        Cleared 272 rl_tool_outcome records for 'graph'
        Cleared rl_tool_q record for 'graph'
        Tool 'graph' statistics reset successfully
    """
    if not confirm:
        logger.warning(
            f"[reset_tool_stats] DRY RUN: Would reset stats for '{tool_name}'. "
            "Set confirm=True to execute."
        )
        return False

    try:
        db = get_database()

        # Count records before deletion
        outcome_count = db.execute(
            "SELECT COUNT(*) FROM rl_tool_outcome WHERE tool_name = ?",
            (tool_name,),
        ).fetchone()[0]

        # Delete from rl_tool_outcome
        db.execute("DELETE FROM rl_tool_outcome WHERE tool_name = ?", (tool_name,))

        # Unified tables (live since the v0.7.0 migration) — the legacy deletes
        # above only touch frozen pre-migration rows.
        db.execute(
            "DELETE FROM rl_q_value WHERE learner_id = 'tool_selector' AND state_key = ?",
            (tool_name,),
        )
        db.execute(
            "DELETE FROM rl_task_stat WHERE learner_id = 'tool_selector' AND task_type = ?",
            (tool_name,),
        )
        db.execute(
            "DELETE FROM rl_transition WHERE learner_id = 'tool_selector' AND to_state = ?",
            (tool_name,),
        )

        if outcome_count > 0:
            logger.info(
                f"[reset_tool_stats] Cleared {outcome_count} rl_tool_outcome records for '{tool_name}'"
            )

        # Delete from rl_tool_q
        q_deleted = db.execute("DELETE FROM rl_tool_q WHERE tool_name = ?", (tool_name,)).rowcount

        if q_deleted > 0:
            logger.info(f"[reset_tool_stats] Cleared rl_tool_q record for '{tool_name}'")

        # Delete from rl_cache_tool
        cache_deleted = db.execute(
            "DELETE FROM rl_cache_tool WHERE tool_name = ?", (tool_name,)
        ).rowcount

        if cache_deleted > 0:
            logger.info(f"[reset_tool_stats] Cleared rl_cache_tool record for '{tool_name}'")

        # Delete from rl_tool_task
        task_deleted = db.execute(
            "DELETE FROM rl_tool_task WHERE tool_name = ?", (tool_name,)
        ).rowcount

        if task_deleted > 0:
            logger.info(
                f"[reset_tool_stats] Cleared {task_deleted} rl_tool_task records for '{tool_name}'"
            )

        logger.info(f"[reset_tool_stats] Tool '{tool_name}' statistics reset successfully")
        return True

    except Exception as e:
        logger.error(f"[reset_tool_stats] Failed to reset stats for '{tool_name}': {e}")
        return False


def reset_multiple_tools(tool_names: List[str], confirm: bool = False) -> dict[str, bool]:
    """Reset RL statistics for multiple tools.

    Args:
        tool_names: List of tool names to reset
        confirm: Must be True to execute the reset

    Returns:
        Dictionary mapping tool names to success status
    """
    results = {}
    for tool_name in tool_names:
        results[tool_name] = reset_tool_stats(tool_name, confirm=confirm)
    return results


def get_tool_stats_summary(tool_name: str) -> Optional[dict]:
    """Get summary of current RL statistics for a tool.

    Useful for reviewing a tool's reputation before deciding to reset.

    Args:
        tool_name: Name of the tool to query

    Returns:
        Dictionary with stats summary or None if tool not found
    """
    try:
        db = get_database()

        # Get rl_tool_q data
        q_row = db.execute(
            "SELECT q_value, selection_count, success_count FROM rl_tool_q WHERE tool_name = ?",
            (tool_name,),
        ).fetchone()

        # Get rl_tool_outcome summary
        outcome_stats = db.execute(
            "SELECT COUNT(*), AVG(success), AVG(quality_score) FROM rl_tool_outcome WHERE tool_name = ?",
            (tool_name,),
        ).fetchone()

        if not q_row and not outcome_stats[0]:
            return None

        result = {
            "tool_name": tool_name,
            "q_value": q_row[0] if q_row else None,
            "selection_count": q_row[1] if q_row else 0,
            "success_count": q_row[2] if q_row else 0,
            "outcome_count": outcome_stats[0] if outcome_stats else 0,
            "avg_success": (outcome_stats[1] if outcome_stats and outcome_stats[1] else None),
            "avg_quality": (outcome_stats[2] if outcome_stats and outcome_stats[2] else None),
        }

        if result["selection_count"] > 0:
            result["success_rate"] = result["success_count"] / result["selection_count"]
        else:
            result["success_rate"] = None

        return result

    except Exception as e:
        logger.error(f"[get_tool_stats_summary] Failed to get stats for '{tool_name}': {e}")
        return None


# CLI command for easy use
def main():
    """CLI entry point for resetting tool stats."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Reset RL statistics for tools to give them a fresh start"
    )
    parser.add_argument(
        "tool_name",
        nargs="+",
        help="Tool name(s) to reset (e.g., graph code_search)",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Actually execute the reset (default: dry run)",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Show current stats before resetting",
    )

    args = parser.parse_args()

    if args.summary:
        print("Current tool statistics:")
        print("=" * 80)
        for tool_name in args.tool_name:
            summary = get_tool_stats_summary(tool_name)
            if summary:
                print(f"\n{summary['tool_name']}:")
                print(
                    f"  Q-value: {summary['q_value']:.4f}"
                    if summary["q_value"]
                    else "  Q-value: N/A"
                )
                print(f"  Selections: {summary['selection_count']}")
                print(f"  Successes: {summary['success_count']}")
                print(f"  Outcomes: {summary['outcome_count']}")
                if summary["success_rate"] is not None:
                    print(f"  Success rate: {summary['success_rate']:.1%}")
                if summary["avg_success"] is not None:
                    print(f"  Avg success score: {summary['avg_success']:.3f}")
                if summary["avg_quality"] is not None:
                    print(f"  Avg quality score: {summary['avg_quality']:.3f}")
            else:
                print(f"\n{tool_name}: No statistics found")
        print()

    if args.confirm:
        print("Resetting tool statistics...")
        results = reset_multiple_tools(args.tool_name, confirm=True)
        for tool_name, success in results.items():
            status = "✓" if success else "✗"
            print(f"{status} {tool_name}")
    else:
        print("DRY RUN - use --confirm to actually reset")
        print("Use --summary to see current stats before resetting")


if __name__ == "__main__":
    main()
