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

"""Tests for LSP integration."""

import pytest

pytest.importorskip("victor_coding.lsp.config")
pytest.importorskip("victor_coding.lsp.client")
pytest.importorskip("victor_coding.lsp.manager")

from victor_coding.lsp.config import (
    LSPServerConfig,
    LANGUAGE_SERVERS,
    get_server_for_file,
    get_language_id,
)
from victor_coding.lsp.client import (
    Position,
    Range,
    Location,
    Diagnostic,
    CompletionItem,
    Hover,
)
from victor_coding.lsp.manager import (
    LSPConnectionPool,
    LSPStatus,
    get_lsp_manager,
    reset_lsp_manager,
)


class TestLSPServerConfig:
    """Tests for LSPServerConfig."""

    def test_create_config(self):
        """Test creating a server config."""
        config = LSPServerConfig(
            name="Test Server",
            language_id="test",
            file_extensions=[".test"],
            command=["test-server"],
        )

        assert config.name == "Test Server"
        assert config.language_id == "test"
        assert ".test" in config.file_extensions
        assert config.command == ["test-server"]

    def test_python_config_exists(self):
        """Test Python server config exists."""
        assert "python" in LANGUAGE_SERVERS
        config = LANGUAGE_SERVERS["python"]
        assert config.language_id == "python"
        assert ".py" in config.file_extensions

    def test_typescript_config_exists(self):
        """Test TypeScript server config exists."""
        assert "typescript" in LANGUAGE_SERVERS
        config = LANGUAGE_SERVERS["typescript"]
        assert ".ts" in config.file_extensions
        assert ".tsx" in config.file_extensions

    def test_rust_config_exists(self):
        """Test Rust server config exists."""
        assert "rust" in LANGUAGE_SERVERS
        config = LANGUAGE_SERVERS["rust"]
        assert config.language_id == "rust"
        assert ".rs" in config.file_extensions


class TestGetServerForFile:
    """Tests for get_server_for_file function."""

    def test_python_file(self):
        """Test getting server for Python file."""
        config = get_server_for_file("test.py")
        assert config is not None
        assert config.language_id == "python"

    def test_typescript_file(self):
        """Test getting server for TypeScript file."""
        config = get_server_for_file("test.ts")
        assert config is not None
        assert config.language_id == "typescript"

    def test_tsx_file(self):
        """Test getting server for TSX file."""
        config = get_server_for_file("component.tsx")
        assert config is not None
        assert config.language_id == "typescript"

    def test_rust_file(self):
        """Test getting server for Rust file."""
        config = get_server_for_file("main.rs")
        assert config is not None
        assert config.language_id == "rust"

    def test_go_file(self):
        """Test getting server for Go file."""
        config = get_server_for_file("main.go")
        assert config is not None
        assert config.language_id == "go"

    def test_unknown_file(self):
        """Test getting server for unknown file type."""
        config = get_server_for_file("unknown.xyz")
        assert config is None


class TestGetLanguageId:
    """Tests for get_language_id function."""

    def test_python_language_id(self):
        """Test getting language ID for Python."""
        lang_id = get_language_id("test.py")
        assert lang_id == "python"

    def test_typescript_language_id(self):
        """Test getting language ID for TypeScript."""
        lang_id = get_language_id("test.ts")
        assert lang_id == "typescript"

    def test_unknown_language_id(self):
        """Test getting language ID for unknown file."""
        lang_id = get_language_id("unknown.xyz")
        assert lang_id is None


class TestPosition:
    """Tests for Position dataclass."""

    def test_create_position(self):
        """Test creating a position."""
        pos = Position(line=10, character=5)
        assert pos.line == 10
        assert pos.character == 5

    def test_to_dict(self):
        """Test converting to dict."""
        pos = Position(line=10, character=5)
        d = pos.to_dict()
        assert d == {"line": 10, "character": 5}

    def test_from_dict(self):
        """Test creating from dict."""
        pos = Position.from_dict({"line": 10, "character": 5})
        assert pos.line == 10
        assert pos.character == 5


class TestRange:
    """Tests for Range dataclass."""

    def test_create_range(self):
        """Test creating a range."""
        start = Position(line=10, character=0)
        end = Position(line=10, character=15)
        r = Range(start=start, end=end)

        assert r.start.line == 10
        assert r.end.character == 15

    def test_to_dict(self):
        """Test converting to dict."""
        r = Range(
            start=Position(line=1, character=0),
            end=Position(line=1, character=10),
        )
        d = r.to_dict()
        assert d["start"]["line"] == 1
        assert d["end"]["character"] == 10

    def test_from_dict(self):
        """Test creating from dict."""
        r = Range.from_dict(
            {
                "start": {"line": 1, "character": 0},
                "end": {"line": 1, "character": 10},
            }
        )
        assert r.start.line == 1
        assert r.end.character == 10


class TestLocation:
    """Tests for Location dataclass."""

    def test_create_location(self):
        """Test creating a location."""
        loc = Location(
            uri="file:///test.py",
            range=Range(
                start=Position(line=0, character=0),
                end=Position(line=0, character=10),
            ),
        )

        assert loc.uri == "file:///test.py"
        assert loc.range.start.line == 0

    def test_from_dict(self):
        """Test creating from dict."""
        loc = Location.from_dict(
            {
                "uri": "file:///test.py",
                "range": {
                    "start": {"line": 0, "character": 0},
                    "end": {"line": 0, "character": 10},
                },
            }
        )
        assert loc.uri == "file:///test.py"


class TestDiagnostic:
    """Tests for Diagnostic dataclass."""

    def test_create_diagnostic(self):
        """Test creating a diagnostic."""
        diag = Diagnostic(
            range=Range(
                start=Position(line=5, character=0),
                end=Position(line=5, character=20),
            ),
            message="Undefined variable 'x'",
            severity=1,
            source="pyright",
        )

        assert "Undefined variable" in diag.message
        assert diag.severity == 1
        assert diag.source == "pyright"

    def test_from_dict(self):
        """Test creating from dict."""
        diag = Diagnostic.from_dict(
            {
                "range": {
                    "start": {"line": 5, "character": 0},
                    "end": {"line": 5, "character": 20},
                },
                "message": "Error message",
                "severity": 2,
                "source": "test",
            }
        )

        assert diag.message == "Error message"
        assert diag.severity == 2


class TestCompletionItem:
    """Tests for CompletionItem dataclass."""

    def test_create_completion(self):
        """Test creating a completion item."""
        item = CompletionItem(
            label="print",
            kind=3,  # Function
            detail="(value: object) -> None",
            documentation="Print to stdout",
        )

        assert item.label == "print"
        assert item.kind == 3
        assert item.detail is not None

    def test_from_dict(self):
        """Test creating from dict."""
        item = CompletionItem.from_dict(
            {
                "label": "test_function",
                "kind": 3,
                "detail": "def test_function()",
            }
        )

        assert item.label == "test_function"
        assert item.kind == 3

    def test_from_dict_with_markdown_doc(self):
        """Test creating from dict with markdown documentation."""
        item = CompletionItem.from_dict(
            {
                "label": "func",
                "kind": 3,
                "documentation": {
                    "kind": "markdown",
                    "value": "# Function\nDoes stuff",
                },
            }
        )

        assert item.documentation == "# Function\nDoes stuff"


class TestHover:
    """Tests for Hover dataclass."""

    def test_create_hover(self):
        """Test creating hover info."""
        hover = Hover(
            contents="def hello() -> str\n\nReturns a greeting",
        )

        assert "hello" in hover.contents
        assert hover.range is None

    def test_from_dict_simple(self):
        """Test creating from simple dict."""
        hover = Hover.from_dict(
            {
                "contents": "Simple hover text",
            }
        )

        assert hover.contents == "Simple hover text"

    def test_from_dict_with_markup(self):
        """Test creating from dict with markup content."""
        hover = Hover.from_dict(
            {
                "contents": {"kind": "markdown", "value": "**Bold text**"},
            }
        )

        assert hover.contents == "**Bold text**"


class TestLSPConnectionPool:
    """Tests for LSPConnectionPool."""

    @pytest.fixture
    def manager(self):
        """Create a fresh LSP manager."""
        reset_lsp_manager()
        return LSPConnectionPool(workspace_root="/tmp/test")

    def test_create_manager(self, manager):
        """Test creating a manager."""
        assert manager._workspace_root == "/tmp/test"
        assert "file://" in manager._root_uri

    def test_set_workspace_root(self, manager):
        """Test setting workspace root."""
        manager.set_workspace_root("/new/path")
        assert "/new/path" in manager._workspace_root

    def test_path_to_uri(self, manager):
        """Test path to URI conversion."""
        uri = manager._path_to_uri("/test/file.py")
        assert uri.startswith("file://")
        assert "file.py" in uri

    def test_uri_to_path(self, manager):
        """Test URI to path conversion."""
        path = manager._uri_to_path("file:///test/file.py")
        assert path == "/test/file.py"

    def test_get_available_servers(self, manager):
        """Test getting available servers."""
        servers = manager.get_available_servers()
        assert len(servers) > 0

        languages = [s["language"] for s in servers]
        assert "python" in languages
        assert "typescript" in languages

    def test_get_status_empty(self, manager):
        """Test getting status with no servers."""
        status = manager.get_status()
        assert len(status) == 0


class TestLSPStatus:
    """Tests for LSPStatus dataclass."""

    def test_create_status(self):
        """Test creating LSP status."""
        status = LSPStatus(
            language="python",
            server_name="Pyright",
            running=True,
            initialized=True,
            open_documents=5,
            capabilities=["completion", "hover", "definition"],
        )

        assert status.language == "python"
        assert status.server_name == "Pyright"
        assert status.running is True
        assert len(status.capabilities) == 3


class TestGlobalManager:
    """Tests for global manager functions."""

    def test_get_lsp_manager_singleton(self):
        """Test singleton behavior."""
        reset_lsp_manager()
        manager1 = get_lsp_manager()
        manager2 = get_lsp_manager()
        assert manager1 is manager2

    def test_reset_lsp_manager(self):
        """Test resetting manager."""
        reset_lsp_manager()
        manager1 = get_lsp_manager()
        reset_lsp_manager()
        manager2 = get_lsp_manager()
        assert manager1 is not manager2


class TestAllLanguageServers:
    """Tests to verify all configured language servers."""

    def test_all_servers_have_required_fields(self):
        """Test all servers have required configuration."""
        for lang, config in LANGUAGE_SERVERS.items():
            assert config.name, f"{lang} missing name"
            assert config.language_id, f"{lang} missing language_id"
            assert config.file_extensions, f"{lang} missing file_extensions"
            assert config.command, f"{lang} missing command"

    def test_all_servers_have_install_info(self):
        """Test all servers have installation info."""
        for lang, config in LANGUAGE_SERVERS.items():
            assert config.install_command, f"{lang} missing install_command"

    def test_common_languages_supported(self):
        """Test common programming languages are supported."""
        required_languages = [
            "python",
            "typescript",
            "rust",
            "go",
            "java",
            "c",
        ]

        for lang in required_languages:
            assert lang in LANGUAGE_SERVERS, f"Missing support for {lang}"

    def test_web_languages_supported(self):
        """Test web development languages are supported."""
        web_languages = ["html", "css", "json", "yaml"]

        for lang in web_languages:
            assert lang in LANGUAGE_SERVERS, f"Missing support for {lang}"
