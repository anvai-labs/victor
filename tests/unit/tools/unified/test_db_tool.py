"""Tests for the unified db tool facade (parser + kwargs mapping)."""

from victor.tools.database_tool import DatabaseConnection
from victor.tools.unified.db_tool import _map_kwargs, create_db_parser


class TestCreateDbParser:
    """The bash-style parser must surface every supported db_type."""

    def test_connect_parses_every_backend_type(self):
        parser = create_db_parser()
        for db_type in ("sqlite", "duckdb", "postgresql", "mysql", "sqlserver"):
            parsed = parser.parse_args(["connect", "--type", db_type, "--database", "x"])
            assert parsed.type == db_type

    def test_connect_parses_duckdb_type(self):
        parser = create_db_parser()
        parsed = parser.parse_args(["connect", "--type", "duckdb", "--database", ":memory:"])
        assert parsed.type == "duckdb"
        assert parsed.database == ":memory:"


class TestMapKwargs:
    """Parsed args map onto database(action=...) kwargs."""

    def test_connect_maps_type_to_db_type(self):
        parser = create_db_parser()
        parsed = parser.parse_args(["connect", "--type", "duckdb", "--database", ":memory:"])
        action, kwargs = _map_kwargs(parsed, DatabaseConnection)
        assert action == "connect"
        conn = kwargs["connection"]
        assert isinstance(conn, DatabaseConnection)
        assert conn.db_type == "duckdb"
        assert conn.database == ":memory:"

    def test_connect_defaults_to_sqlite(self):
        parser = create_db_parser()
        parsed = parser.parse_args(["connect", "--database", "app.db"])
        action, kwargs = _map_kwargs(parsed, DatabaseConnection)
        assert action == "connect"
        assert kwargs["connection"].db_type == "sqlite"
