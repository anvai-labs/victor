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

"""Tests for unified database tool module."""

import pytest
import tempfile
import os

from victor.tools.database_tool import (
    database,
    DatabaseConnection,
    DANGEROUS_PATTERNS,
    _connections,
    _DEFAULT_ALLOW_MODIFICATIONS,
    _DEFAULT_MAX_ROWS,
)


class TestDatabaseConstants:
    """Tests for database tool constants."""

    def test_default_allow_modifications(self):
        """Test default allow_modifications constant."""
        assert _DEFAULT_ALLOW_MODIFICATIONS is False

    def test_default_max_rows(self):
        """Test default max_rows constant."""
        assert _DEFAULT_MAX_ROWS == 100


class TestDatabaseConnect:
    """Tests for database connect action."""

    @pytest.mark.asyncio
    async def test_connect_sqlite_memory(self):
        """Test SQLite in-memory connection."""
        conn = DatabaseConnection(database=":memory:")
        result = await database(action="connect", connection=conn)
        assert result["success"] is True
        assert "connection_id" in result
        assert "sqlite_" in result["connection_id"]
        # Cleanup
        if result["success"]:
            await database(action="disconnect", connection_id=result["connection_id"])

    @pytest.mark.asyncio
    async def test_connect_sqlite_file(self):
        """Test SQLite file connection."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            conn = DatabaseConnection(database=db_path, db_type="sqlite")
            result = await database(action="connect", connection=conn)
            assert result["success"] is True
            if result["success"]:
                await database(action="disconnect", connection_id=result["connection_id"])
        finally:
            os.unlink(db_path)

    @pytest.mark.asyncio
    async def test_connect_unsupported_type(self):
        """Test connection with unsupported database type."""
        conn = DatabaseConnection(database="test", db_type="mongodb")
        result = await database(action="connect", connection=conn)
        assert result["success"] is False
        assert "Unsupported database type" in result["error"]

    @pytest.mark.asyncio
    async def test_connect_missing_database(self):
        """Test connection with missing database parameter."""
        result = await database(action="connect", connection=None)
        assert result["success"] is False
        assert "Missing required parameter" in result["error"]


class TestDatabaseQuery:
    """Tests for database query action."""

    @pytest.fixture
    async def sqlite_conn(self):
        """Create SQLite connection for testing."""
        conn = DatabaseConnection(database=":memory:")
        result = await database(action="connect", connection=conn)
        conn_id = result["connection_id"]

        # Create test table
        _connections[conn_id].execute("""
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT
            )
        """)
        _connections[conn_id].execute(
            "INSERT INTO users (name, email) VALUES ('Alice', 'alice@test.com')"
        )
        _connections[conn_id].execute(
            "INSERT INTO users (name, email) VALUES ('Bob', 'bob@test.com')"
        )
        _connections[conn_id].commit()

        yield conn_id

        await database(action="disconnect", connection_id=conn_id)

    @pytest.mark.asyncio
    async def test_query_invalid_connection(self):
        """Test query with invalid connection."""
        result = await database(action="query", connection_id="invalid_conn", sql="SELECT 1")
        assert result["success"] is False
        assert "Invalid or missing connection_id" in result["error"]

    @pytest.mark.asyncio
    async def test_query_missing_sql(self, sqlite_conn):
        """Test query with missing SQL."""
        result = await database(action="query", connection_id=sqlite_conn)
        assert result["success"] is False
        assert "Missing required parameter" in result["error"]

    @pytest.mark.asyncio
    async def test_query_select(self, sqlite_conn):
        """Test SELECT query."""
        result = await database(
            action="query", connection_id=sqlite_conn, sql="SELECT * FROM users"
        )
        assert result["success"] is True
        assert result["count"] == 2
        assert "columns" in result
        assert "rows" in result
        assert "id" in result["columns"]
        assert "name" in result["columns"]

    @pytest.mark.asyncio
    async def test_query_select_with_limit(self, sqlite_conn):
        """Test SELECT query with limit."""
        result = await database(
            action="query",
            connection_id=sqlite_conn,
            sql="SELECT * FROM users",
            limit=1,
        )
        assert result["success"] is True
        assert result["count"] == 1

    @pytest.mark.asyncio
    async def test_query_dangerous_blocked(self, sqlite_conn):
        """Test dangerous queries are blocked."""
        result = await database(action="query", connection_id=sqlite_conn, sql="DROP TABLE users")
        assert result["success"] is False
        assert "Modification operations not allowed" in result["error"]

    @pytest.mark.asyncio
    async def test_query_insert_blocked(self, sqlite_conn):
        """Test INSERT is blocked by default."""
        result = await database(
            action="query",
            connection_id=sqlite_conn,
            sql="INSERT INTO users (name, email) VALUES ('Charlie', 'c@test.com')",
        )
        assert result["success"] is False
        assert "Modification operations not allowed" in result["error"]

    @pytest.mark.asyncio
    async def test_query_modifications_blocked_by_default(self, sqlite_conn):
        """Test modifications are blocked by default (secure setting)."""
        # Modifications are blocked by default for security
        # The _DEFAULT_ALLOW_MODIFICATIONS constant is False
        result = await database(
            action="query",
            connection_id=sqlite_conn,
            sql="INSERT INTO users (name, email) VALUES ('Charlie', 'c@test.com')",
        )
        # Modifications should be blocked
        assert result["success"] is False
        assert "Modification operations not allowed" in result["error"]

    @pytest.mark.asyncio
    async def test_query_invalid_sql(self, sqlite_conn):
        """Test invalid SQL."""
        result = await database(
            action="query", connection_id=sqlite_conn, sql="INVALID SQL STATEMENT"
        )
        assert result["success"] is False
        assert "Query failed" in result["error"]


class TestDatabaseTables:
    """Tests for database tables action."""

    @pytest.fixture
    async def sqlite_conn(self):
        """Create SQLite connection for testing."""
        conn = DatabaseConnection(database=":memory:")
        result = await database(action="connect", connection=conn)
        conn_id = result["connection_id"]

        # Create test tables
        _connections[conn_id].execute("CREATE TABLE users (id INTEGER PRIMARY KEY)")
        _connections[conn_id].execute("CREATE TABLE orders (id INTEGER PRIMARY KEY)")
        _connections[conn_id].commit()

        yield conn_id

        await database(action="disconnect", connection_id=conn_id)

    @pytest.mark.asyncio
    async def test_tables_invalid_connection(self):
        """Test listing tables with invalid connection."""
        result = await database(action="tables", connection_id="invalid_conn")
        assert result["success"] is False
        assert "Invalid or missing connection_id" in result["error"]

    @pytest.mark.asyncio
    async def test_tables_list(self, sqlite_conn):
        """Test listing tables."""
        result = await database(action="tables", connection_id=sqlite_conn)
        assert result["success"] is True
        assert "tables" in result
        assert "users" in result["tables"]
        assert "orders" in result["tables"]
        assert result["count"] == 2


class TestDatabaseDescribe:
    """Tests for database describe action."""

    @pytest.fixture
    async def sqlite_conn(self):
        """Create SQLite connection for testing."""
        conn = DatabaseConnection(database=":memory:")
        result = await database(action="connect", connection=conn)
        conn_id = result["connection_id"]

        _connections[conn_id].execute("""
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT,
                age INTEGER
            )
        """)
        _connections[conn_id].commit()

        yield conn_id

        await database(action="disconnect", connection_id=conn_id)

    @pytest.mark.asyncio
    async def test_describe_invalid_connection(self):
        """Test describe with invalid connection."""
        result = await database(action="describe", connection_id="invalid_conn", table="users")
        assert result["success"] is False
        assert "Invalid or missing connection_id" in result["error"]

    @pytest.mark.asyncio
    async def test_describe_missing_table(self, sqlite_conn):
        """Test describe with missing table name."""
        result = await database(action="describe", connection_id=sqlite_conn)
        assert result["success"] is False
        assert "Missing required parameter" in result["error"]

    @pytest.mark.asyncio
    async def test_describe_table(self, sqlite_conn):
        """Test describing a table."""
        result = await database(action="describe", connection_id=sqlite_conn, table="users")
        assert result["success"] is True
        assert result["table"] == "users"
        assert result["count"] == 4
        assert len(result["columns"]) == 4

        col_names = [c["name"] for c in result["columns"]]
        assert "id" in col_names
        assert "name" in col_names
        assert "email" in col_names
        assert "age" in col_names


class TestDatabaseSchema:
    """Tests for database schema action."""

    @pytest.fixture
    async def sqlite_conn(self):
        """Create SQLite connection for testing."""
        conn = DatabaseConnection(database=":memory:")
        result = await database(action="connect", connection=conn)
        conn_id = result["connection_id"]

        _connections[conn_id].execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
        _connections[conn_id].execute(
            "CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER)"
        )
        _connections[conn_id].commit()

        yield conn_id

        await database(action="disconnect", connection_id=conn_id)

    @pytest.mark.asyncio
    async def test_schema_invalid_connection(self):
        """Test schema with invalid connection."""
        result = await database(action="schema", connection_id="invalid_conn")
        assert result["success"] is False
        assert "Invalid or missing connection_id" in result["error"]

    @pytest.mark.asyncio
    async def test_schema_full(self, sqlite_conn):
        """Test getting full schema."""
        result = await database(action="schema", connection_id=sqlite_conn)
        assert result["success"] is True
        assert "tables" in result
        assert len(result["tables"]) == 2

        table_names = [t["name"] for t in result["tables"]]
        assert "users" in table_names
        assert "orders" in table_names

        for table in result["tables"]:
            assert "columns" in table


class TestDatabaseDisconnect:
    """Tests for database disconnect action."""

    @pytest.mark.asyncio
    async def test_disconnect_invalid_connection(self):
        """Test disconnect with invalid connection."""
        result = await database(action="disconnect", connection_id="invalid_conn")
        assert result["success"] is False
        assert "Invalid or missing connection_id" in result["error"]

    @pytest.mark.asyncio
    async def test_disconnect_success(self):
        """Test successful disconnect."""
        conn = DatabaseConnection(database=":memory:")
        conn_result = await database(action="connect", connection=conn)
        assert conn_result["success"] is True
        conn_id = conn_result["connection_id"]

        result = await database(action="disconnect", connection_id=conn_id)
        assert result["success"] is True
        assert "Disconnected" in result["message"]

        assert conn_id not in _connections


class TestDangerousPatterns:
    """Tests for dangerous SQL pattern detection."""

    def test_dangerous_patterns_list(self):
        """Test dangerous patterns list."""
        assert "DROP DATABASE" in DANGEROUS_PATTERNS
        assert "DROP TABLE" in DANGEROUS_PATTERNS
        assert "TRUNCATE" in DANGEROUS_PATTERNS
        assert "DELETE FROM" in DANGEROUS_PATTERNS
        assert "UPDATE" in DANGEROUS_PATTERNS
        assert "INSERT INTO" in DANGEROUS_PATTERNS


class TestDatabaseUnknownAction:
    """Tests for unknown action handling."""

    @pytest.mark.asyncio
    async def test_unknown_action(self):
        """Test unknown action returns error."""
        result = await database(action="invalid_action")
        assert result["success"] is False
        assert "Unknown action" in result["error"]
