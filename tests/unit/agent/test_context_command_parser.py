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

"""Tests for ContextCommandParser.

Covers @url, @file, @folder, @problems context commands.
"""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import tempfile
import os

from victor.context.command_parser import (
    ContextItem,
    ParsedCommand,
    ContextResolver,
    URLResolver,
    FileResolver,
    FolderResolver,
    ProblemsResolver,
    ContextCommandParser,
    create_default_parser,
    resolve_context_commands,
)


class TestParsedCommand:
    """Tests for ParsedCommand dataclass."""

    def test_basic_command(self):
        """Test basic parsed command creation."""
        cmd = ParsedCommand(
            command_type="file",
            argument="foo.py",
            original_text="@file:foo.py",
            start_pos=0,
            end_pos=12,
        )
        assert cmd.command_type == "file"
        assert cmd.argument == "foo.py"


class TestContextItem:
    """Tests for ContextItem dataclass."""

    def test_success_item(self):
        """Test successful context item."""
        item = ContextItem(
            source_type="file",
            source="/path/to/file.py",
            content="file content",
            success=True,
        )
        assert item.success is True
        assert item.error is None

    def test_error_item(self):
        """Test error context item."""
        item = ContextItem(
            source_type="url",
            source="https://example.com",
            content="",
            success=False,
            error="Connection failed",
        )
        assert item.success is False
        assert "Connection failed" in item.error


class TestContextCommandParser:
    """Tests for ContextCommandParser."""

    @pytest.fixture
    def parser(self):
        """Create a parser with default resolvers."""
        return ContextCommandParser()

    def test_parse_file_command(self, parser):
        """Test parsing @file command."""
        commands = parser.parse("Please read @file:foo.py and explain it")
        assert len(commands) == 1
        assert commands[0].command_type == "file"
        assert commands[0].argument == "foo.py"

    def test_parse_url_command(self, parser):
        """Test parsing @url command."""
        commands = parser.parse("Check @url:https://example.com for info")
        assert len(commands) == 1
        assert commands[0].command_type == "url"
        assert commands[0].argument == "https://example.com"

    def test_parse_folder_command(self, parser):
        """Test parsing @folder command."""
        commands = parser.parse("List @folder:src/components")
        assert len(commands) == 1
        assert commands[0].command_type == "folder"
        assert commands[0].argument == "src/components"

    def test_parse_problems_command(self, parser):
        """Test parsing @problems command (no argument)."""
        commands = parser.parse("Fix the @problems in my code")
        assert len(commands) == 1
        assert commands[0].command_type == "problems"
        assert commands[0].argument == ""

    def test_parse_multiple_commands(self, parser):
        """Test parsing multiple commands."""
        commands = parser.parse("Read @file:a.py and @file:b.py then check @problems")
        assert len(commands) == 3
        assert commands[0].argument == "a.py"
        assert commands[1].argument == "b.py"
        assert commands[2].command_type == "problems"

    def test_parse_case_insensitive(self, parser):
        """Test commands are case insensitive."""
        commands = parser.parse("@FILE:test.py @URL:example.com @FOLDER:src")
        assert len(commands) == 3
        assert all(cmd.command_type in ("file", "url", "folder") for cmd in commands)

    def test_parse_no_commands(self, parser):
        """Test parsing text with no commands."""
        commands = parser.parse("Just a regular message")
        assert len(commands) == 0

    def test_strip_commands(self, parser):
        """Test stripping commands from text."""
        text = "Please read @file:foo.py and explain it"
        clean = parser.strip_commands(text)
        assert "@file:foo.py" not in clean
        assert "Please read" in clean
        assert "explain it" in clean

    def test_strip_multiple_commands(self, parser):
        """Test stripping multiple commands."""
        text = "@file:a.py Check @url:example.com and @problems"
        clean = parser.strip_commands(text)
        assert "@file" not in clean
        assert "@url" not in clean
        assert "@problems" not in clean
        assert "Check" in clean


class TestURLResolver:
    """Tests for URLResolver."""

    @pytest.fixture
    def resolver(self):
        return URLResolver(timeout=5.0)

    @pytest.mark.asyncio
    async def test_invalid_scheme(self, resolver):
        """Test invalid URL scheme."""
        item = await resolver.resolve("ftp://example.com", {})
        assert item.success is False
        assert "Unsupported URL scheme" in item.error

    @pytest.mark.asyncio
    async def test_adds_https_scheme(self, resolver):
        """Test that https is added to URLs without scheme."""
        from unittest.mock import MagicMock

        with patch("aiohttp.ClientSession") as mock_session:
            # Create mock response
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.reason = "OK"
            mock_response.headers = {"Content-Type": "text/html"}
            mock_response.text = AsyncMock(return_value="<html>content</html>")

            # Track calls to get
            get_calls = []

            # Create an async context manager for session.get()
            class MockGetContextManager:
                async def __aenter__(self):
                    return mock_response

                async def __aexit__(self, *args):
                    pass

                def __init__(self, *args, **kwargs):
                    get_calls.append(args)

            # Create mock session instance
            mock_session_instance = MagicMock()
            mock_session_instance.get = MockGetContextManager
            mock_session_instance.__aenter__ = AsyncMock(return_value=mock_session_instance)
            mock_session_instance.__aexit__ = AsyncMock(return_value=None)

            mock_session.return_value = mock_session_instance

            item = await resolver.resolve("example.com", {})
            # Should have tried to fetch https://example.com
            assert len(get_calls) > 0
            # First arg should be the URL with https scheme
            assert "https://example.com" in get_calls[0]


class TestFileResolver:
    """Tests for FileResolver."""

    @pytest.fixture
    def resolver(self):
        return FileResolver(max_file_size=10000)

    @pytest.mark.asyncio
    async def test_file_not_found(self, resolver):
        """Test file not found error."""
        item = await resolver.resolve("/nonexistent/file.py", {"working_directory": "."})
        assert item.success is False
        assert "not found" in item.error.lower()

    @pytest.mark.asyncio
    async def test_read_file(self, resolver):
        """Test reading an existing file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("print('hello')")
            temp_path = f.name

        try:
            item = await resolver.resolve(temp_path, {"working_directory": "."})
            assert item.success is True
            assert "print('hello')" in item.content
            assert item.metadata["extension"] == ".py"
        finally:
            os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_relative_path(self, resolver):
        """Test relative path resolution."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a file
            file_path = Path(tmpdir) / "test.py"
            file_path.write_text("content")

            item = await resolver.resolve("test.py", {"working_directory": tmpdir})
            assert item.success is True
            assert item.content == "content"

    @pytest.mark.asyncio
    async def test_file_too_large(self):
        """Test file size limit."""
        resolver = FileResolver(max_file_size=10)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("x" * 100)  # 100 bytes, exceeds 10 byte limit
            temp_path = f.name

        try:
            item = await resolver.resolve(temp_path, {"working_directory": "."})
            assert item.success is False
            assert "too large" in item.error.lower()
        finally:
            os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_not_a_file(self, resolver):
        """Test error when path is a directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            item = await resolver.resolve(tmpdir, {"working_directory": "."})
            assert item.success is False
            assert "Not a file" in item.error


class TestFolderResolver:
    """Tests for FolderResolver."""

    @pytest.fixture
    def resolver(self):
        return FolderResolver(max_depth=2, max_files=50)

    @pytest.mark.asyncio
    async def test_folder_not_found(self, resolver):
        """Test folder not found error."""
        item = await resolver.resolve("/nonexistent/folder", {"working_directory": "."})
        assert item.success is False
        assert "not found" in item.error.lower()

    @pytest.mark.asyncio
    async def test_read_folder(self, resolver):
        """Test reading folder structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create some files and folders
            (Path(tmpdir) / "file1.py").write_text("content1")
            (Path(tmpdir) / "file2.py").write_text("content2")
            subdir = Path(tmpdir) / "subdir"
            subdir.mkdir()
            (subdir / "file3.py").write_text("content3")

            item = await resolver.resolve(tmpdir, {"working_directory": "."})
            assert item.success is True
            assert "file1.py" in item.content
            assert "file2.py" in item.content
            assert "subdir/" in item.content

    @pytest.mark.asyncio
    async def test_ignores_pycache(self, resolver):
        """Test that __pycache__ is ignored."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "file.py").write_text("content")
            pycache = Path(tmpdir) / "__pycache__"
            pycache.mkdir()
            (pycache / "file.pyc").write_text("bytecode")

            item = await resolver.resolve(tmpdir, {"working_directory": "."})
            assert item.success is True
            assert "__pycache__" not in item.content

    @pytest.mark.asyncio
    async def test_includes_readme_preview(self, resolver):
        """Test README preview is included."""
        with tempfile.TemporaryDirectory() as tmpdir:
            readme = Path(tmpdir) / "README.md"
            readme.write_text("# My Project\n\nThis is a test project.")

            item = await resolver.resolve(tmpdir, {"working_directory": "."})
            assert item.success is True
            assert "README.md" in item.content
            assert "My Project" in item.content

    @pytest.mark.asyncio
    async def test_not_a_folder(self, resolver):
        """Test error when path is a file."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = f.name

        try:
            item = await resolver.resolve(temp_path, {"working_directory": "."})
            assert item.success is False
            assert "Not a folder" in item.error
        finally:
            os.unlink(temp_path)


class TestProblemsResolver:
    """Tests for ProblemsResolver."""

    @pytest.fixture
    def resolver(self):
        return ProblemsResolver()

    @pytest.mark.asyncio
    async def test_no_problems(self, resolver):
        """Test when no problems exist."""
        item = await resolver.resolve("", {})
        assert item.success is True
        assert "No problems found" in item.content

    @pytest.mark.asyncio
    async def test_with_diagnostics(self, resolver):
        """Test with diagnostics."""
        context = {
            "diagnostics": [
                {
                    "severity": "error",
                    "file": "foo.py",
                    "line": 10,
                    "message": "Syntax error",
                },
                {
                    "severity": "warning",
                    "file": "bar.py",
                    "line": 5,
                    "message": "Unused variable",
                },
            ]
        }
        item = await resolver.resolve("", context)
        assert item.success is True
        assert "[ERROR]" in item.content
        assert "[WARNING]" in item.content
        assert "foo.py:10" in item.content
        assert item.metadata["count"] == 2

    @pytest.mark.asyncio
    async def test_with_workspace_errors(self, resolver):
        """Test with workspace errors."""
        context = {
            "workspace_errors": [
                "Build failed: missing dependency",
                "Test suite error",
            ]
        }
        item = await resolver.resolve("", context)
        assert item.success is True
        assert "Build failed" in item.content
        assert item.metadata["errors"] == 2


class TestResolveContextCommands:
    """Tests for the convenience function."""

    @pytest.mark.asyncio
    async def test_resolve_file_command(self):
        """Test resolving @file command."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.py"
            file_path.write_text("print('test')")

            text = f"Read @file:{file_path} please"
            clean_text, items = await resolve_context_commands(text, tmpdir)

            assert "@file" not in clean_text
            assert len(items) == 1
            assert items[0].success is True
            assert "print('test')" in items[0].content

    @pytest.mark.asyncio
    async def test_resolve_multiple_commands(self):
        """Test resolving multiple commands."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = Path(tmpdir) / "a.py"
            file1.write_text("content_a")
            file2 = Path(tmpdir) / "b.py"
            file2.write_text("content_b")

            text = f"Compare @file:{file1} with @file:{file2}"
            clean_text, items = await resolve_context_commands(text, tmpdir)

            assert len(items) == 2
            assert all(item.success for item in items)


class TestCreateDefaultParser:
    """Tests for factory function."""

    def test_create_default_parser(self):
        """Test creating parser with default resolvers."""
        parser = create_default_parser()
        assert "url" in parser._resolvers
        assert "file" in parser._resolvers
        assert "folder" in parser._resolvers
        assert "problems" in parser._resolvers

    def test_register_custom_resolver(self):
        """Test registering a custom resolver replaces default."""

        class CustomFileResolver(ContextResolver):
            @property
            def command_type(self):
                return "file"

            async def resolve(self, argument, context):
                return ContextItem(
                    source_type="file",
                    source=argument,
                    content=f"Custom file resolver: {argument}",
                )

        parser = ContextCommandParser()
        parser.register_resolver(CustomFileResolver())

        # The file resolver should be replaced with custom
        assert isinstance(parser._resolvers["file"], CustomFileResolver)

    @pytest.mark.asyncio
    async def test_custom_resolver_used(self):
        """Test that custom resolver is actually used."""

        class CustomFileResolver(ContextResolver):
            @property
            def command_type(self):
                return "file"

            async def resolve(self, argument, context):
                return ContextItem(
                    source_type="file",
                    source=argument,
                    content=f"Custom: {argument}",
                )

        parser = ContextCommandParser()
        parser.register_resolver(CustomFileResolver())

        _, items = await parser.resolve("@file:test.py", {})
        assert len(items) == 1
        assert items[0].content == "Custom: test.py"
