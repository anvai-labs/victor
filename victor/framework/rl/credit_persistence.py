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

"""
Persistence layer for credit assignment data.

Provides:
- SQLite storage for credit assignment results
- Query and aggregation capabilities
- Export and import functionality
- Historical analysis
"""

from __future__ import annotations

import json
from victor.core.json_utils import json_dumps
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import sqlite3
from contextlib import contextmanager

from victor.framework.rl import (
    CreditSignal,
    ActionMetadata,
    CreditMethodology,
    CreditGranularity,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Database Schema
# ============================================================================


CREDIT_ASSIGNMENT_SCHEMA = """
-- Credit assignment sessions
CREATE TABLE IF NOT EXISTS credit_sessions (
    session_id TEXT PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    methodology TEXT NOT NULL,
    granularity TEXT NOT NULL,
    trajectory_length INTEGER NOT NULL,
    total_reward REAL NOT NULL,
    success INTEGER NOT NULL,
    duration_seconds REAL,
    metadata TEXT,  -- JSON
    agent_count INTEGER,
    team_id TEXT
);

-- Individual credit signals
CREATE TABLE IF NOT EXISTS credit_signals (
    signal_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    action_id TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    team_id TEXT,
    raw_reward REAL NOT NULL,
    credit REAL NOT NULL,
    confidence REAL NOT NULL,
    methodology TEXT NOT NULL,
    granularity TEXT NOT NULL,
    timestamp REAL,
    tool_name TEXT,
    method_name TEXT,
    turn_index INTEGER,
    step_index INTEGER,
    attribution TEXT,  -- JSON
    FOREIGN KEY (session_id) REFERENCES credit_sessions(session_id)
);

-- Agent attribution summaries
CREATE TABLE IF NOT EXISTS agent_attribution (
    attribution_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    total_credit REAL NOT NULL,
    direct_credit REAL NOT NULL,
    received_credit REAL NOT NULL,
    contribution_count INTEGER NOT NULL,
    FOREIGN KEY (session_id) REFERENCES credit_sessions(session_id)
);

-- Indexes for efficient queries
CREATE INDEX IF NOT EXISTS idx_signals_session ON credit_signals(session_id);
CREATE INDEX IF NOT EXISTS idx_signals_agent ON credit_signals(agent_id);
CREATE INDEX IF NOT EXISTS idx_sessions_created ON credit_sessions(created_at);
CREATE INDEX IF NOT EXISTS idx_sessions_methodology ON credit_sessions(methodology);
CREATE INDEX IF NOT EXISTS idx_attribution_agent ON agent_attribution(agent_id);
"""


# ============================================================================
# Database Manager
# ============================================================================


class CreditAssignmentDB:
    """Manager for credit assignment database operations.

    Provides:
    - Session storage and retrieval
    - Signal storage and querying
    - Agent attribution tracking
    - Aggregation and analytics
    """

    def __init__(self, db_path: Union[str, Path] = ":memory:"):
        """Initialize database manager.

        Args:
            db_path: Path to SQLite database (default: in-memory)
        """
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None

    @contextmanager
    def _get_connection(self):
        """Get database connection with context manager."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def initialize(self) -> None:
        """Initialize database schema."""
        with self._get_connection() as conn:
            conn.executescript(CREDIT_ASSIGNMENT_SCHEMA)
            conn.commit()
        logger.info(f"Initialized credit assignment database: {self.db_path}")

    def save_session(
        self,
        session_id: str,
        methodology: CreditMethodology,
        granularity: CreditGranularity,
        signals: List[CreditSignal],
        success: bool,
        duration: float,
        metadata: Optional[Dict[str, Any]] = None,
        team_id: Optional[str] = None,
    ) -> None:
        """Save a credit assignment session to database.

        Args:
            session_id: Unique session identifier
            methodology: Credit assignment methodology used
            granularity: Credit granularity used
            signals: Credit signals from the session
            success: Whether execution succeeded
            duration: Execution duration in seconds
            metadata: Optional session metadata
            team_id: Optional team identifier
        """
        with self._get_connection() as conn:
            # Insert session
            conn.execute(
                """
                INSERT INTO credit_sessions (
                    session_id, methodology, granularity, trajectory_length,
                    total_reward, success, duration_seconds, metadata, agent_count, team_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    methodology.value,
                    granularity.value,
                    len(signals),
                    sum(s.credit for s in signals),
                    1 if success else 0,
                    duration,
                    json_dumps(metadata) if metadata else None,
                    len(set(s.metadata.agent_id for s in signals if s.metadata)),
                    team_id,
                ),
            )

            # Insert signals
            for signal in signals:
                signal_dict = signal.to_dict()
                conn.execute(
                    """
                    INSERT INTO credit_signals (
                        signal_id, session_id, action_id, agent_id, team_id,
                        raw_reward, credit, confidence, methodology, granularity,
                        timestamp, tool_name, method_name, turn_index, step_index, attribution
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        f"{session_id}_{signal.action_id}",
                        session_id,
                        signal.action_id,
                        signal_dict.get("metadata", {}).get("agent_id", "unknown"),
                        signal_dict.get("metadata", {}).get("team_id"),
                        signal.raw_reward,
                        signal.credit,
                        signal.confidence,
                        signal_dict["methodology"],
                        signal_dict["granularity"],
                        signal_dict.get("metadata", {}).get("timestamp"),
                        signal_dict.get("metadata", {}).get("tool_name"),
                        signal_dict.get("metadata", {}).get("method_name"),
                        signal_dict.get("metadata", {}).get("turn_index", 0),
                        signal_dict.get("metadata", {}).get("step_index", 0),
                        json_dumps(signal.attribution) if signal.attribution else None,
                    ),
                )

            # Compute and insert agent attributions
            agent_credits: Dict[str, Dict[str, float]] = {}
            for signal in signals:
                if signal.metadata:
                    agent = signal.metadata.agent_id
                    if agent not in agent_credits:
                        agent_credits[agent] = {
                            "direct": 0.0,
                            "received": 0.0,
                            "count": 0,
                        }

                    agent_credits[agent]["direct"] += signal.credit
                    agent_credits[agent]["count"] += 1

                    for contributor, amount in signal.attribution.items():
                        if contributor not in agent_credits[agent]:
                            agent_credits[agent][contributor] = 0.0
                        agent_credits[agent][contributor] += amount

            for agent, credits in agent_credits.items():
                received = sum(v for k, v in credits.items() if k != "direct" and k != "count")
                conn.execute(
                    """
                    INSERT INTO agent_attribution (
                        attribution_id, session_id, agent_id,
                        total_credit, direct_credit, received_credit, contribution_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        f"{session_id}_{agent}",
                        session_id,
                        agent,
                        credits["direct"] + received,
                        credits["direct"],
                        received,
                        credits["count"],
                    ),
                )

            conn.commit()
        logger.info(f"Saved credit session: {session_id}")

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a credit assignment session.

        Args:
            session_id: Session identifier

        Returns:
            Session data or None if not found
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM credit_sessions WHERE session_id = ?
                """,
                (session_id,),
            )
            row = cursor.fetchone()

            if row is None:
                return None

            return dict(row)

    def get_session_signals(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all credit signals for a session.

        Args:
            session_id: Session identifier

        Returns:
            List of signal data
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM credit_signals WHERE session_id = ?
                ORDER BY credit DESC
                """,
                (session_id,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_agent_history(
        self,
        agent_id: str,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get credit assignment history for an agent.

        Args:
            agent_id: Agent identifier
            limit: Maximum number of sessions to return

        Returns:
            List of session summaries
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT s.*, a.total_credit, a.direct_credit, a.received_credit
                FROM credit_sessions s
                JOIN agent_attribution a ON s.session_id = a.session_id
                WHERE a.agent_id = ?
                ORDER BY s.created_at DESC
                LIMIT ?
                """,
                (agent_id, limit),
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_team_attribution(
        self,
        team_id: str,
        limit: int = 50,
    ) -> Dict[str, Dict[str, float]]:
        """Get aggregated attribution for a team.

        Args:
            team_id: Team identifier
            limit: Maximum number of sessions to aggregate

        Returns:
            Dictionary mapping agent_id to credit totals
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT agent_id, SUM(total_credit) as total_credit
                FROM agent_attribution aa
                JOIN credit_sessions s ON aa.session_id = s.session_id
                WHERE s.team_id = ?
                GROUP BY agent_id
                ORDER BY total_credit DESC
                LIMIT ?
                """,
                (team_id, limit),
            )
            return {row["agent_id"]: row["total_credit"] for row in cursor.fetchall()}

    def get_methodology_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics by methodology.

        Returns:
            Dictionary mapping methodology to stats
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT
                    methodology,
                    COUNT(*) as session_count,
                    AVG(total_reward) as avg_reward,
                    AVG(duration_seconds) as avg_duration,
                    SUM(success) * 1.0 / COUNT(*) as success_rate
                FROM credit_sessions
                GROUP BY methodology
                """,
            )
            return {row["methodology"]: dict(row) for row in cursor.fetchall()}

    def get_top_agents(
        self,
        metric: str = "total_credit",
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get top-performing agents by credit.

        Args:
            metric: Metric to rank by (total_credit, direct_credit, received_credit)
            limit: Maximum number of agents to return

        Returns:
            List of agent summaries
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                f"""
                SELECT
                    agent_id,
                    SUM({metric}) as {metric},
                    COUNT(*) as session_count,
                    AVG(contribution_count) as avg_contributions
                FROM agent_attribution
                GROUP BY agent_id
                ORDER BY {metric} DESC
                LIMIT ?
                """,
                (limit,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def export_to_json(
        self,
        output_path: Union[str, Path],
        session_id: Optional[str] = None,
    ) -> None:
        """Export credit data to JSON file.

        Args:
            output_path: Path to save JSON
            session_id: Optional specific session to export
        """
        data = {
            "exported_at": datetime.now().isoformat(),
            "sessions": [],
            "signals": [],
            "attribution": [],
        }

        with self._get_connection() as conn:
            if session_id:
                cursor = conn.execute(
                    "SELECT * FROM credit_sessions WHERE session_id = ?",
                    (session_id,),
                )
            else:
                cursor = conn.execute("SELECT * FROM credit_sessions")

            for row in cursor.fetchall():
                data["sessions"].append(dict(row))

            # Get signals
            if session_id:
                cursor = conn.execute(
                    "SELECT * FROM credit_signals WHERE session_id = ?",
                    (session_id,),
                )
            else:
                cursor = conn.execute("SELECT * FROM credit_signals LIMIT 1000")

            for row in cursor.fetchall():
                data["signals"].append(dict(row))

        # Write to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Exported {len(data['sessions'])} sessions to {output_path}")

    def import_from_json(
        self,
        input_path: Union[str, Path],
    ) -> int:
        """Import credit data from JSON file.

        Args:
            input_path: Path to JSON file

        Returns:
            Number of sessions imported
        """
        with open(input_path) as f:
            data = json.load(f)

        imported = 0
        with self._get_connection() as conn:
            for session_data in data.get("sessions", []):
                # Check if session already exists
                cursor = conn.execute(
                    "SELECT session_id FROM credit_sessions WHERE session_id = ?",
                    (session_data["session_id"],),
                )
                if cursor.fetchone():
                    continue  # Skip existing

                # Insert session
                conn.execute(
                    """
                    INSERT INTO credit_sessions (
                        session_id, created_at, methodology, granularity,
                        trajectory_length, total_reward, success, duration_seconds,
                        metadata, agent_count, team_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_data["session_id"],
                        session_data["created_at"],
                        session_data["methodology"],
                        session_data["granularity"],
                        session_data["trajectory_length"],
                        session_data["total_reward"],
                        session_data["success"],
                        session_data.get("duration_seconds"),
                        session_data.get("metadata"),
                        session_data.get("agent_count"),
                        session_data.get("team_id"),
                    ),
                )
                imported += 1

            # Insert signals
            for signal_data in data.get("signals", []):
                conn.execute(
                    """
                    INSERT INTO credit_signals (
                        signal_id, session_id, action_id, agent_id, team_id,
                        raw_reward, credit, confidence, methodology, granularity,
                        timestamp, tool_name, method_name, turn_index, step_index, attribution
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        signal_data["signal_id"],
                        signal_data["session_id"],
                        signal_data["action_id"],
                        signal_data["agent_id"],
                        signal_data.get("team_id"),
                        signal_data["raw_reward"],
                        signal_data["credit"],
                        signal_data["confidence"],
                        signal_data["methodology"],
                        signal_data["granularity"],
                        signal_data.get("timestamp"),
                        signal_data.get("tool_name"),
                        signal_data.get("method_name"),
                        signal_data.get("turn_index", 0),
                        signal_data.get("step_index", 0),
                        signal_data.get("attribution"),
                    ),
                )

            conn.commit()

        logger.info(f"Imported {imported} sessions from {input_path}")
        return imported


# ============================================================================
# Convenience Functions
# ============================================================================


def get_default_db_path() -> Path:
    """Get default path for credit assignment database."""
    from victor.core.constants import VICTOR_HOME

    db_dir = VICTOR_HOME / "credit_assignment"
    db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir / "credit.db"


def get_persistent_db() -> CreditAssignmentDB:
    """Get persistent credit assignment database."""
    db_path = get_default_db_path()
    db = CreditAssignmentDB(db_path)

    # Initialize if needed
    if not db_path.exists() or db_path.stat().st_size == 0:
        db.initialize()

    return db


# ============================================================================
# Exports
# ============================================================================


__all__ = [
    "CreditAssignmentDB",
    "get_default_db_path",
    "get_persistent_db",
]
