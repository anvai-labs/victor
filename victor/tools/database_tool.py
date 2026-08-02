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

"""Unified database tool for SQL operations and schema inspection.

Consolidates all database operations into a single tool for better token efficiency.
Supports multiple database types:
- SQLite (built-in, no dependencies)
- PostgreSQL (requires psycopg2)
- MySQL (requires mysql-connector-python)
- SQL Server (requires pyodbc)

Features:
- Connect/disconnect to databases
- Execute SQL queries
- Inspect database schemas
- List tables
- Describe table structure
- Safe query execution with validation
"""

import sqlite3
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from victor.tools.base import AccessMode, DangerLevel, Priority
from victor.tools.decorators import tool
from victor.tools.formatters import format_database_results

if TYPE_CHECKING:
    from victor.tools.cache_manager import CacheNamespace

# Constants
_DEFAULT_ALLOW_MODIFICATIONS: bool = False
_DEFAULT_MAX_ROWS: int = 100

# Legacy session-level connection cache (use _get_connection_pool() for DI support)
_connections: Dict[str, Any] = {}


@dataclass
class DatabaseConnection:
    """Database connection configuration.

    Consolidates all connection parameters into a single object.
    Reduces parameter count from 6 to 1 for database operations.
    """

    db_type: str = "sqlite"
    database: Optional[str] = None
    host: Optional[str] = None
    port: Optional[int] = None
    username: Optional[str] = None
    password: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for internal use."""
        return {
            "db_type": self.db_type,
            "database": self.database,
            "host": self.host,
            "port": self.port,
            "username": self.username,
            "password": self.password,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatabaseConnection":
        """Create from dictionary for backward compatibility."""
        return cls(
            db_type=data.get("db_type", "sqlite"),
            database=data.get("database"),
            host=data.get("host"),
            port=data.get("port"),
            username=data.get("username"),
            password=data.get("password"),
        )


def _get_connection_pool(exec_ctx: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Get the connection pool, preferring DI-injected cache if available.

    Args:
        exec_ctx: Execution context that may contain cache_manager

    Returns:
        Connection pool dict-like object
    """
    # Try to get from execution context (DI pattern)
    if exec_ctx is not None:
        # Check for ToolExecutionContext with cache_manager
        from victor.tools.context import ToolExecutionContext

        if isinstance(exec_ctx, ToolExecutionContext):
            if exec_ctx.cache_manager is not None:
                return exec_ctx.connection_pool
        elif isinstance(exec_ctx, dict):
            # Legacy dict context - check for cache_manager
            cache_manager = exec_ctx.get("cache_manager")
            if cache_manager is not None:
                return cache_manager.connection_pool

    # Fallback to global cache for backward compatibility
    return _connections


# Dangerous SQL patterns that should be blocked
DANGEROUS_PATTERNS = [
    "DROP DATABASE",
    "DROP SCHEMA",
    "TRUNCATE",
    "DELETE FROM",
    "UPDATE",
    "INSERT INTO",
    "ALTER TABLE",
    "CREATE",
    "DROP TABLE",
]


# Helper functions for db-specific connections
async def _connect_sqlite(
    database: str, connection_pool: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Connect to SQLite database."""
    pool = connection_pool if connection_pool is not None else _connections
    try:
        conn = sqlite3.connect(database)
        conn.row_factory = sqlite3.Row  # Enable column names
        connection_id = f"sqlite_{id(conn)}"
        pool[connection_id] = conn

        return {
            "success": True,
            "connection_id": connection_id,
            "message": f"Connected to SQLite database: {database}",
        }
    except Exception as e:
        return {"success": False, "error": f"SQLite connection failed: {str(e)}"}


async def _connect_postgresql(
    kwargs: Dict[str, Any], connection_pool: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Connect to PostgreSQL database."""
    pool = connection_pool if connection_pool is not None else _connections
    try:
        import psycopg2

        conn = psycopg2.connect(
            host=kwargs.get("host", "localhost"),
            port=kwargs.get("port", 5432),
            database=kwargs.get("database"),
            user=kwargs.get("username"),
            password=kwargs.get("password"),
        )

        connection_id = f"postgresql_{id(conn)}"
        pool[connection_id] = conn

        return {
            "success": True,
            "connection_id": connection_id,
            "message": "Connected to PostgreSQL database",
        }
    except ImportError:
        return {
            "success": False,
            "error": "PostgreSQL support requires: pip install psycopg2-binary",
        }
    except Exception as e:
        return {"success": False, "error": f"PostgreSQL connection failed: {str(e)}"}


async def _connect_mysql(
    kwargs: Dict[str, Any], connection_pool: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Connect to MySQL database."""
    pool = connection_pool if connection_pool is not None else _connections
    try:
        import mysql.connector

        conn = mysql.connector.connect(
            host=kwargs.get("host", "localhost"),
            port=kwargs.get("port", 3306),
            database=kwargs.get("database"),
            user=kwargs.get("username"),
            password=kwargs.get("password"),
        )

        connection_id = f"mysql_{id(conn)}"
        pool[connection_id] = conn

        return {
            "success": True,
            "connection_id": connection_id,
            "message": "Connected to MySQL database",
        }
    except ImportError:
        return {
            "success": False,
            "error": "MySQL support requires: pip install mysql-connector-python",
        }
    except Exception as e:
        return {"success": False, "error": f"MySQL connection failed: {str(e)}"}


async def _connect_sqlserver(
    kwargs: Dict[str, Any], connection_pool: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Connect to SQL Server database."""
    pool = connection_pool if connection_pool is not None else _connections
    try:
        import pyodbc

        conn_string = (
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
            f"SERVER={kwargs.get('host', 'localhost')};"
            f"DATABASE={kwargs.get('database')};"
            f"UID={kwargs.get('username')};"
            f"PWD={kwargs.get('password')}"
        )

        conn = pyodbc.connect(conn_string)
        connection_id = f"sqlserver_{id(conn)}"
        pool[connection_id] = conn

        return {
            "success": True,
            "connection_id": connection_id,
            "message": "Connected to SQL Server database",
        }
    except ImportError:
        return {
            "success": False,
            "error": "SQL Server support requires: pip install pyodbc",
        }
    except Exception as e:
        return {"success": False, "error": f"SQL Server connection failed: {str(e)}"}


async def _do_connect(
    database: str,
    db_type: str,
    host: Optional[str],
    port: Optional[int],
    username: Optional[str],
    password: Optional[str],
    connection_pool: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Internal connect handler."""
    if db_type == "sqlite":
        return await _connect_sqlite(database, connection_pool)
    elif db_type == "postgresql":
        return await _connect_postgresql(
            {
                "database": database,
                "host": host,
                "port": port,
                "username": username,
                "password": password,
            },
            connection_pool,
        )
    elif db_type == "mysql":
        return await _connect_mysql(
            {
                "database": database,
                "host": host,
                "port": port,
                "username": username,
                "password": password,
            },
            connection_pool,
        )
    elif db_type == "sqlserver":
        return await _connect_sqlserver(
            {
                "database": database,
                "host": host,
                "username": username,
                "password": password,
            },
            connection_pool,
        )
    else:
        return {"success": False, "error": f"Unsupported database type: {db_type}"}


async def _do_query(
    connection_id: str,
    sql: str,
    limit: Optional[int] = None,
    allow_modifications: Optional[bool] = None,
    connection_pool: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Internal query handler."""
    pool = connection_pool if connection_pool is not None else _connections
    if not connection_id or connection_id not in pool:
        return {
            "success": False,
            "error": "Invalid or missing connection_id. Use action='connect' first.",
        }

    if not sql:
        return {"success": False, "error": "Missing required parameter: sql"}

    # Check for dangerous patterns
    allow_writes = (
        _DEFAULT_ALLOW_MODIFICATIONS if allow_modifications is None else allow_modifications
    )
    if not allow_writes:
        sql_upper = sql.upper()
        for pattern in DANGEROUS_PATTERNS:
            if pattern in sql_upper:
                return {
                    "success": False,
                    "error": f"Modification operations not allowed: {pattern}. "
                    "Modification operations are disabled by default for safety.",
                }

    try:
        conn = pool[connection_id]
        cursor = conn.cursor()
        cursor.execute(sql)

        # Check if query returns results
        if cursor.description:
            # SELECT query - fetch results
            columns = [desc[0] for desc in cursor.description]
            query_limit = limit if limit is not None else _DEFAULT_MAX_ROWS
            rows = cursor.fetchmany(query_limit)

            # Convert to list of dicts
            results = []
            for row in rows:
                if hasattr(row, "keys"):  # SQLite Row object
                    results.append(dict(row))
                else:  # Tuple
                    results.append(dict(zip(columns, row, strict=False)))

            return {
                "success": True,
                "columns": columns,
                "rows": results,
                "count": len(results),
                "limited": len(rows) == query_limit,
                "formatted_output": format_database_results(
                    {
                        "columns": columns,
                        "rows": results,
                        "count": len(results),
                    }
                ).content,
                "contains_markup": True,
            }
        else:
            # Non-SELECT query (INSERT, UPDATE, DELETE)
            conn.commit()
            return {
                "success": True,
                "rows_affected": cursor.rowcount,
                "message": f"Query executed successfully. Rows affected: {cursor.rowcount}",
            }

    except Exception as e:
        return {"success": False, "error": f"Query failed: {str(e)}"}


async def _do_tables(
    connection_id: str, connection_pool: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Internal list tables handler."""
    pool = connection_pool if connection_pool is not None else _connections
    if not connection_id or connection_id not in pool:
        return {"success": False, "error": "Invalid or missing connection_id"}

    try:
        conn = pool[connection_id]
        cursor = conn.cursor()

        if connection_id.startswith("sqlite"):
            sql = "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        elif connection_id.startswith("postgresql"):
            sql = "SELECT tablename FROM pg_tables WHERE schemaname='public' ORDER BY tablename"
        elif connection_id.startswith("mysql"):
            sql = "SHOW TABLES"
        elif connection_id.startswith("sqlserver"):
            sql = "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE='BASE TABLE' ORDER BY TABLE_NAME"
        else:
            return {"success": False, "error": "Unknown database type"}

        cursor.execute(sql)
        rows = cursor.fetchall()
        tables = [row[0] for row in rows]

        return {"success": True, "tables": tables, "count": len(tables)}

    except Exception as e:
        return {"success": False, "error": f"Failed to list tables: {str(e)}"}


async def _do_describe(
    connection_id: str, table: str, connection_pool: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Internal describe table handler."""
    pool = connection_pool if connection_pool is not None else _connections
    if not connection_id or connection_id not in pool:
        return {"success": False, "error": "Invalid or missing connection_id"}

    if not table:
        return {"success": False, "error": "Missing required parameter: table"}

    try:
        conn = pool[connection_id]
        cursor = conn.cursor()

        if connection_id.startswith("sqlite"):
            cursor.execute(f"PRAGMA table_info({table})")
            rows = cursor.fetchall()
            columns = [
                {
                    "name": row[1],
                    "type": row[2],
                    "nullable": not row[3],
                    "primary_key": bool(row[5]),
                }
                for row in rows
            ]

        elif connection_id.startswith("postgresql"):
            cursor.execute(f"""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_name = '{table}'
                ORDER BY ordinal_position
                """)
            rows = cursor.fetchall()
            columns = [
                {"name": row[0], "type": row[1], "nullable": row[2] == "YES"} for row in rows
            ]

        elif connection_id.startswith("mysql"):
            cursor.execute(f"DESCRIBE {table}")
            rows = cursor.fetchall()
            columns = [
                {
                    "name": row[0],
                    "type": row[1],
                    "nullable": row[2] == "YES",
                    "primary_key": row[3] == "PRI",
                }
                for row in rows
            ]

        else:
            return {
                "success": False,
                "error": "Describe not implemented for this database type",
            }

        return {
            "success": True,
            "table": table,
            "columns": columns,
            "count": len(columns),
        }

    except Exception as e:
        return {"success": False, "error": f"Failed to describe table: {str(e)}"}


async def _do_schema(
    connection_id: str, connection_pool: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Internal get schema handler."""
    pool = connection_pool if connection_pool is not None else _connections
    if not connection_id or connection_id not in pool:
        return {"success": False, "error": "Invalid or missing connection_id"}

    # Get list of tables first
    tables_result = await _do_tables(connection_id, pool)
    if not tables_result["success"]:
        return tables_result

    # Get schema for each table
    schema_info: Dict[str, List[Dict[str, Any]]] = {"tables": []}

    try:
        for table in tables_result["tables"]:
            describe_result = await _do_describe(connection_id, table, pool)
            if describe_result["success"]:
                schema_info["tables"].append({"name": table, "columns": describe_result["columns"]})

        return {"success": True, "tables": schema_info["tables"]}

    except Exception as e:
        return {"success": False, "error": f"Schema inspection failed: {str(e)}"}


async def _do_disconnect(
    connection_id: str, connection_pool: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Internal disconnect handler."""
    pool = connection_pool if connection_pool is not None else _connections
    if not connection_id or connection_id not in pool:
        return {"success": False, "error": "Invalid or missing connection_id"}

    try:
        conn = pool[connection_id]
        conn.close()
        del pool[connection_id]

        return {
            "success": True,
            "message": f"Disconnected from database: {connection_id}",
        }

    except Exception as e:
        return {"success": False, "error": f"Disconnect failed: {str(e)}"}


@tool(
    category="database",
    priority=Priority.MEDIUM,  # Task-specific database operations
    access_mode=AccessMode.MIXED,  # Reads and can modify database
    danger_level=DangerLevel.MEDIUM,  # Database modifications have impact
    mandatory_keywords=[
        "database",
        "sqlite",
        "sqlite3",
        "sql",
        "query",
        "tables",
        "schema",
        ".tables",
        "SELECT",
        "postgres",
        "mysql",
        "database file",
        ".db",
        ".sqlite",
    ],
    keywords=[
        "database",
        "sql",
        "query",
        "connect",
        "tables",
        "sqlite",
        "sqlite3",
        "postgres",
        "mysql",
        "schema",
        "db",
        "rows",
    ],
)
async def database(
    action: str,
    connection: Optional[DatabaseConnection] = None,
    connection_id: Optional[str] = None,
    sql: Optional[str] = None,
    table: Optional[str] = None,
    limit: Optional[int] = None,
    allow_modifications: bool = False,
    _exec_ctx: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Execute SQL queries and manage database schemas.

    For SQLite database files (.db, .sqlite), you can either:
    1. Use this tool: database(action="connect", connection=DatabaseConnection(db_type="sqlite", database="path/to/file.db"))
    2. Use shell: shell(cmd='sqlite3 file.db ".tables"') for quick inspection

    Actions:
    - connect: Connect to a database
    - query: Execute SQL queries
    - tables: List all tables
    - describe: Describe a table's structure
    - schema: Get complete database schema
    - disconnect: Close connection

    Args:
        action: Operation to perform - 'connect', 'query', 'tables', 'describe', 'schema', 'disconnect'.
        connection: DatabaseConnection object with db_type, database, host, port, username, password.
        connection_id: Connection ID from previous connect (required for query/tables/describe/schema/disconnect).
        sql: SQL query to execute (required for query action).
        table: Table name (required for describe action).
        limit: Maximum rows to return for query (default: 100).
        allow_modifications: Allow modification statements for query action (default: False).

    Returns:
        Dictionary containing:
        - success: Whether operation succeeded
        - For connect: connection_id, message
        - For query: columns, rows, count
        - For tables: tables list, count
        - For describe: table name, columns list
        - For schema: tables with their columns
        - error: Error message if failed

    Examples:
        # SQLite - Connect
        database(action="connect", connection=DatabaseConnection(db_type="sqlite", database="data.db"))

        # SQLite - List tables
        database(action="tables", connection_id="sqlite_12345")

        # SQLite - Query
        database(action="query", connection_id="sqlite_12345", sql="SELECT * FROM users LIMIT 10")

        # PostgreSQL - Connect
        database(action="connect", connection=DatabaseConnection(
            db_type="postgresql", database="mydb", host="localhost", username="user", password="pass"
        ))

        # Quick shell alternative for SQLite:
        # shell(cmd='sqlite3 file.db ".tables"')
        # shell(cmd='sqlite3 file.db "SELECT * FROM users LIMIT 10"')
    """
    action_lower = action.lower().strip()

    # Get connection pool using DI-aware accessor
    pool = _get_connection_pool(_exec_ctx)

    if action_lower == "connect":
        if connection is None:
            return {
                "success": False,
                "error": "Missing required parameter: connection (DatabaseConnection object)",
            }
        return await _do_connect(
            connection.database,
            connection.db_type,
            connection.host,
            connection.port,
            connection.username,
            connection.password,
            pool,
        )

    elif action_lower == "query":
        if not connection_id:
            return {
                "success": False,
                "error": "Missing required parameter: connection_id",
            }
        if not sql:
            return {"success": False, "error": "Missing required parameter: sql"}
        return await _do_query(connection_id, sql, limit, allow_modifications, pool)

    elif action_lower == "tables":
        if not connection_id:
            return {
                "success": False,
                "error": "Missing required parameter: connection_id",
            }
        return await _do_tables(connection_id, pool)

    elif action_lower == "describe":
        if not connection_id:
            return {
                "success": False,
                "error": "Missing required parameter: connection_id",
            }
        if not table:
            return {"success": False, "error": "Missing required parameter: table"}
        return await _do_describe(connection_id, table, pool)

    elif action_lower == "schema":
        if not connection_id:
            return {
                "success": False,
                "error": "Missing required parameter: connection_id",
            }
        return await _do_schema(connection_id, pool)

    elif action_lower == "disconnect":
        if not connection_id:
            return {
                "success": False,
                "error": "Missing required parameter: connection_id",
            }
        return await _do_disconnect(connection_id, pool)

    else:
        return {
            "success": False,
            "error": f"Unknown action: {action}. Valid actions: connect, query, tables, describe, schema, disconnect",
        }
