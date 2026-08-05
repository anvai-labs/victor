# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
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

"""Unified ``db`` command tool — bash-style SQL/database surface.

Parses ``db connect|query|tables|describe|schema|disconnect`` and delegates to
the main-repo ``database(action=…)`` tool (SQL-write-guarded). Falls back to a
shell-driven ``sqlite3``/``psql`` query when needed. Advertised only in the
data-analysis vertical (see ``victor/config/vertical_tools.yaml``), so it costs
no base-schema tokens for coding/general sessions.

Example commands:
    db connect --type sqlite --database ./app.db
    db query "SELECT * FROM users LIMIT 5"
    db tables
    db describe users
    db schema
    db disconnect
"""

from __future__ import annotations

import argparse
import shlex
import sys
from typing import Any, Dict, Optional

from victor.tools.base import AccessMode, DangerLevel, ExecutionCategory, Priority
from victor.tools.decorators import tool
from victor.tools.unified.parser import split_command


class UnifiedDbParser(argparse.ArgumentParser):
    """Custom parser that raises instead of exiting on error."""

    def error(self, message):  # type: ignore[override]
        self.print_usage(sys.stderr)
        raise ValueError(f"Argument parsing error: {message}")


def create_db_parser() -> UnifiedDbParser:
    """Create the parser for the db tool."""
    parser = UnifiedDbParser(
        prog="db", description="Unified database operations.", exit_on_error=False
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="The operation to perform")

    connect = subparsers.add_parser("connect", help="Open a database connection")
    connect.add_argument("--type", default="sqlite", help="sqlite|postgres|mysql|sqlserver")
    connect.add_argument("--database", default=None, help="Database name/path")
    connect.add_argument("--host", default=None)
    connect.add_argument("--port", default=None, type=int)
    connect.add_argument("--username", default=None)
    connect.add_argument("--password", default=None)
    connect.add_argument("--id", dest="connection_id", default=None, help="Connection label")

    query = subparsers.add_parser("query", help="Run a SQL query (read-only by default)")
    query.add_argument("sql", help="SQL to execute")
    query.add_argument("--connection", default=None, help="Connection id to use")
    query.add_argument("--database", default=None, help="sqlite file for a stateless one-off query")
    query.add_argument("--limit", default=100, type=int)
    query.add_argument(
        "--write",
        action="store_true",
        help="Allow modification statements (INSERT/UPDATE/DELETE/DDL)",
    )

    tables = subparsers.add_parser("tables", help="List tables")
    tables.add_argument("--connection", default=None)

    describe = subparsers.add_parser("describe", help="Describe a table")
    describe.add_argument("table")
    describe.add_argument("--connection", default=None)

    subparsers.add_parser("schema", help="Full database schema").add_argument(
        "--connection", default=None
    )
    subparsers.add_parser("disconnect", help="Close a connection").add_argument(
        "--connection", default=None
    )

    return parser


def _format_result(result: Any) -> str:
    """Normalize a database() result (dict/str) into a display string."""
    if isinstance(result, dict):
        if result.get("success") is False:
            return f"### ❌ ERROR\n{result.get('error', 'database operation failed')}"
        for key in ("output", "rows", "tables", "schema", "result"):
            if key in result:
                return str(result[key])
        return str(result)
    return str(result)


@tool(
    name="db",
    category="database",
    access_mode=AccessMode.MIXED,
    danger_level=DangerLevel.MEDIUM,
    execution_category=ExecutionCategory.MIXED,
    priority=Priority.MEDIUM,
    keywords=["db", "sql", "database", "query", "postgres", "sqlite", "mysql", "table"],
    task_types=["action", "analysis"],
)
async def db_tool(cmd: str) -> str:
    """Database domain (bash-style): connect, query, tables, describe, schema, disconnect.
    SQL-write-guarded; delegates to the main-repo database tool. e.g. db query "SELECT 1" · db tables.
    """
    parser = create_db_parser()
    try:
        args_list = split_command(cmd)
        if args_list and args_list[0] == "db":
            args_list = args_list[1:]
        parsed = parser.parse_args(args_list)
    except ValueError as e:
        return f"### ❌ ERROR\n{e}"
    except Exception as e:
        return f"### ❌ ERROR\nUnexpected error parsing command: {e}"

    if not parsed.subcommand:
        return "### ❌ ERROR\nNo db subcommand given. Use: db connect|query|tables|describe|schema|disconnect"

    from victor.tools.database_tool import DatabaseConnection, database

    # Stateless ad-hoc query with no connection: go straight to a shell sqlite3
    # fallback (avoids the database tool's required connection_id for one-offs).
    if parsed.subcommand == "query" and not getattr(parsed, "connection", None):
        return await _shell_sql(parsed.sql, getattr(parsed, "database", None))

    action, kwargs = _map_kwargs(parsed, DatabaseConnection)
    try:
        result = await database(action=action, **kwargs)
    except Exception as e:
        return f"### ❌ ERROR\ndb {parsed.subcommand} failed: {e}"
    return _format_result(result)


def _map_kwargs(parsed: argparse.Namespace, conn_cls) -> tuple[str, Dict[str, Any]]:
    """Map parsed subcommand args to ``database(action=…)`` kwargs."""
    sub = parsed.subcommand
    cid = getattr(parsed, "connection", None) or getattr(parsed, "connection_id", None)
    if sub == "connect":
        conn = conn_cls(
            db_type=parsed.type,
            database=parsed.database,
            host=parsed.host,
            port=parsed.port,
            username=parsed.username,
            password=parsed.password,
        )
        return "connect", {"connection": conn, "connection_id": parsed.connection_id}
    if sub == "query":
        return "query", {
            "sql": parsed.sql,
            "connection_id": cid,
            "limit": parsed.limit,
            "allow_modifications": parsed.write,
        }
    if sub == "tables":
        return "tables", {"connection_id": cid}
    if sub == "describe":
        return "describe", {"table": parsed.table, "connection_id": cid}
    if sub == "schema":
        return "schema", {"connection_id": cid}
    if sub == "disconnect":
        return "disconnect", {"connection_id": cid}
    raise ValueError(f"Unknown db subcommand '{sub}'")


async def _shell_sql(sql: str, database: Optional[str] = None) -> str:
    """Fallback: run a read query via the production shell (sqlite3)."""
    from victor.tools.bash import shell

    target = database or ":memory:"
    result = await shell(cmd=f"sqlite3 {shlex.quote(target)} {shlex.quote(sql)}", readonly=True)
    if isinstance(result, dict):
        if result.get("success") is False:
            return f"### ❌ ERROR\n{result.get('stderr') or result.get('error', 'sql failed')}"
        return (result.get("stdout") or "").strip() or "Done."
    return str(result)


__all__ = ["db_tool", "create_db_parser"]
