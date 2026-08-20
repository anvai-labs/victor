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

"""Prompt history pruning to prevent unbounded database growth.

The agent_prompt_history table accumulates every prompt used during sessions.
This module provides automatic cleanup to keep only recent, high-value entries.

Usage:
    from victor.framework.rl.learners.prompt_history_pruner import prune_prompt_history

    # Keep last 1000 entries per provider
    prune_prompt_history(max_entries_per_provider=1000)

    # Keep entries from last 7 days
    prune_prompt_history(max_age_days=7)
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from victor.core.database import get_database

logger = logging.getLogger(__name__)


def prune_prompt_history(
    max_entries_per_provider: int = 1000,
    max_age_days: int = 30,
    dry_run: bool = False,
) -> dict[str, int]:
    """Prune agent_prompt_history table to prevent unbounded growth.

    Keeps only:
    - Most recent N entries per provider (default: 1000)
    - Entries from last N days (default: 30 days)

    Args:
        max_entries_per_provider: Max entries to keep per provider
        max_age_days: Max age in days to keep entries
        dry_run: If True, report what would be deleted without deleting

    Returns:
        Dictionary with pruning statistics

    Example:
        >>> stats = prune_prompt_history(max_entries_per_provider=1000, max_age_days=30)
        >>> print(f"Deleted {stats['deleted']} entries, kept {stats['kept']}")
        Deleted 57419 entries, kept 1000
    """
    db = get_database()
    conn = db.get_connection()

    # Count total before pruning
    total_before = conn.execute("SELECT COUNT(*) FROM agent_prompt_history").fetchone()[0]

    if dry_run:
        logger.info(
            f"[prune] DRY RUN: Would prune agent_prompt_history "
            f"(current: {total_before} entries)"
        )

    # Strategy 1: Delete old entries (by age)
    cutoff_date = (datetime.now() - timedelta(days=max_age_days)).isoformat()
    age_deleted = 0

    if not dry_run:
        age_deleted = conn.execute(
            "DELETE FROM agent_prompt_history WHERE timestamp < ?",
            (cutoff_date,),
        ).rowcount
        conn.commit()
        logger.info(f"[prune] Deleted {age_deleted} entries older than {max_age_days} days")
    else:
        # Dry run: count what would be deleted
        would_delete = conn.execute(
            "SELECT COUNT(*) FROM agent_prompt_history WHERE timestamp < ?",
            (cutoff_date,),
        ).fetchone()[0]
        age_deleted = would_delete
        logger.info(f"[prune] Would delete {would_delete} entries older than {max_age_days} days")

    # Strategy 2: Delete excess entries per provider (keep most recent N)
    providers = conn.execute("SELECT DISTINCT provider FROM agent_prompt_history").fetchall()

    per_provider_deleted = 0
    for (provider,) in providers:
        # Count entries for this provider
        count = conn.execute(
            "SELECT COUNT(*) FROM agent_prompt_history WHERE provider = ?",
            (provider,),
        ).fetchone()[0]

        if count <= max_entries_per_provider:
            continue

        # Delete excess, keeping most recent
        if not dry_run:
            deleted = conn.execute(
                """
                DELETE FROM agent_prompt_history
                WHERE provider = ?
                AND id NOT IN (
                    SELECT id FROM agent_prompt_history
                    WHERE provider = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                )
                """,
                (provider, provider, max_entries_per_provider),
            ).rowcount
            conn.commit()
            per_provider_deleted += deleted
            logger.info(
                f"[prune] Deleted {deleted}/{count} entries for provider '{provider}' "
                f"(kept {max_entries_per_provider})"
            )
        else:
            would_delete = count - max_entries_per_provider
            per_provider_deleted += would_delete
            logger.info(
                f"[prune] Would delete {would_delete}/{count} entries for provider '{provider}'"
            )

    if not dry_run:
        # Get underlying sqlite3 connection for commit
        conn = db.get_connection()
        conn.commit()

    # Count total after pruning
    total_after = conn.execute("SELECT COUNT(*) FROM agent_prompt_history").fetchone()[0]

    total_deleted = total_before - total_after

    logger.info(
        f"[prune] Pruned agent_prompt_history: "
        f"{total_deleted} deleted, {total_after} kept "
        f"({total_before} → {total_after})"
    )

    return {
        "total_before": total_before,
        "total_after": total_after,
        "deleted": total_deleted,
        "kept": total_after,
        "by_age": age_deleted,
        "by_provider_limit": per_provider_deleted,
    }


# CLI command for manual pruning
def main():
    """CLI entry point for prompt history pruning."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Prune agent_prompt_history to prevent database bloat"
    )
    parser.add_argument(
        "--max-entries",
        type=int,
        default=1000,
        help="Max entries to keep per provider (default: 1000)",
    )
    parser.add_argument(
        "--max-age-days",
        type=int,
        default=30,
        help="Max age in days to keep entries (default: 30)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be deleted without deleting",
    )

    args = parser.parse_args()

    print("Pruning agent_prompt_history...")
    stats = prune_prompt_history(
        max_entries_per_provider=args.max_entries,
        max_age_days=args.max_age_days,
        dry_run=args.dry_run,
    )

    print(f"\n{'=' * 60}")
    print("Results:")
    print(f"  Total before: {stats['total_before']:,}")
    print(f"  Total after:  {stats['total_after']:,}")
    print(f"  Deleted:      {stats['deleted']:,}")
    print(f"  Kept:         {stats['kept']:,}")
    print(f"  By age:       {stats['by_age']:,}")
    print(f"  By provider:  {stats['by_provider_limit']:,}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
